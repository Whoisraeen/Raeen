//! Kernel AIO engine: a real host-threadpool-backed asynchronous file I/O
//! backend for the `sceKernelAio*` HLE surface.
//!
//! ## Model
//!
//! A **submission** is one `sceKernelAioSubmit{Read,Write}Commands` call: a
//! batch of positional read/write requests that shares a single submit id.
//! `submit` enqueues the batch and returns immediately; a small pool of host
//! worker threads performs the file I/O through the SAME
//! [`VirtualFileSystem`](crate::filesystem::VirtualFileSystem) descriptor
//! table the synchronous `read`/`pread`/`pwrite` HLE path uses, so an fd the
//! guest opened synchronously is directly usable in an async request.
//!
//! **Guest memory is never touched from a worker thread.** A read request's
//! bytes land in a host staging buffer ([`AioCompletion::data`]); the HLE
//! layer copies them into the guest buffer — through its own `GuestMemory`
//! view, the same access layer the sync read path uses — when it drains
//! completions on a guest thread (wait / poll / cancel / delete). A write
//! request's bytes are captured from guest memory at submit time, on the
//! guest thread, for the same reason. This keeps the engine free of guest
//! address-space assumptions and makes deleting an in-flight submission safe:
//! the worker's result is simply dropped, never written anywhere.
//!
//! ## States (Orbis `SCE_KERNEL_AIO_STATE_*`, cross-checked against shadPS4
//! `src/core/libraries/kernel/aio.h`, GPL-2.0 — values re-derived, no code
//! ported)
//!
//! * `SUBMITTED (1)` — accepted, no worker has started it.
//! * `PROCESSING (2)` — at least one request of the batch is being performed.
//! * `COMPLETED (3)` — every request reached a terminal outcome via I/O.
//! * `ABORTED (4)` — the batch was cancelled before all requests started.
//!
//! `cancel` marks **not-yet-started** requests aborted; requests already on a
//! worker always run to completion (their staged data is drained normally).
//! Cancelling an already-completed submission is a no-op that reports
//! `COMPLETED`. `delete` removes the submission and retires its final state
//! into a bounded ring so a late `poll` of a deleted id still answers.

use crate::filesystem::VirtualFileSystem;
use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tracing::{debug, warn};

/// `SCE_KERNEL_AIO_STATE_SUBMITTED`.
pub const AIO_STATE_SUBMITTED: u32 = 1;
/// `SCE_KERNEL_AIO_STATE_PROCESSING`.
pub const AIO_STATE_PROCESSING: u32 = 2;
/// `SCE_KERNEL_AIO_STATE_COMPLETED`.
pub const AIO_STATE_COMPLETED: u32 = 3;
/// `SCE_KERNEL_AIO_STATE_ABORTED`.
pub const AIO_STATE_ABORTED: u32 = 4;

/// Worker threads in the pool. Two is enough to overlap I/O with guest
/// execution without competing with the GPU/present worker threads; the real
/// console's AIO scheduler is similarly narrow.
const WORKER_COUNT: usize = 2;

/// Defensive per-request byte cap (the HLE layer enforces its own
/// `MAX_HLE_BULK_BYTES`; this protects direct engine callers from a bogus
/// `nbyte` turning into a host allocation of that size).
const MAX_REQUEST_BYTES: u64 = 256 << 20;

/// Retired-state ring capacity: final states of deleted submissions kept for
/// late polls (a real kernel keeps the slot until id reuse).
const RETIRED_CAP: usize = 1024;

/// `returnValue` for a request aborted before it ran: the Orbis
/// `SCE_KERNEL_ERROR_ECANCELED` code (0x8002_0055) as a sign-extended i64,
/// matching how a guest compares the s64 `returnValue` against negative
/// SCE codes.
const RETURN_ECANCELED: i64 = 0x8002_0055_u32 as i32 as i64;
/// `returnValue` for a request whose descriptor was unknown/unusable
/// (`SCE_KERNEL_ERROR_EBADF`).
const RETURN_EBADF: i64 = 0x8002_0009_u32 as i32 as i64;
/// `returnValue` for a structurally invalid request
/// (`SCE_KERNEL_ERROR_EINVAL`).
const RETURN_EINVAL: i64 = 0x8002_0016_u32 as i32 as i64;
/// `returnValue` for a host I/O failure (`SCE_KERNEL_ERROR_EIO`).
const RETURN_EIO: i64 = 0x8002_0005_u32 as i32 as i64;

/// The file operation one AIO request performs.
#[derive(Debug)]
pub enum AioOp {
    /// Positional read of `nbyte` bytes at `offset` from `fd` into a host
    /// staging buffer (drained to the guest by the HLE layer).
    Read { fd: i32, offset: u64, nbyte: u64 },
    /// Positional write of `data` (captured from guest memory at submit
    /// time) at `offset` to `fd`.
    Write { fd: i32, offset: u64, data: Vec<u8> },
}

/// One request of a submission batch. `guest_buf` / `guest_result` are
/// opaque to the engine (never dereferenced here); they ride along so the
/// HLE layer knows where to deliver the completion.
#[derive(Debug)]
pub struct AioRequest {
    pub op: AioOp,
    /// Guest address of the request's data buffer (opaque).
    pub guest_buf: u64,
    /// Guest address of the request's `SceKernelAioResult` (opaque).
    pub guest_result: u64,
}

/// A terminal request outcome ready for delivery to the guest.
#[derive(Debug)]
pub struct AioCompletion {
    /// Index of the request within its submission batch.
    pub index: usize,
    /// The `returnValue` for the guest result struct: bytes transferred, or
    /// a negative (sign-extended) SCE error code.
    pub return_value: i64,
    /// Terminal request state: [`AIO_STATE_COMPLETED`] or
    /// [`AIO_STATE_ABORTED`].
    pub state: u32,
    /// The request's opaque guest buffer address.
    pub guest_buf: u64,
    /// The request's opaque guest result address.
    pub guest_result: u64,
    /// For a successful read: the staged bytes to copy into `guest_buf`.
    pub data: Option<Vec<u8>>,
}

/// Why a [`AioEngine::wait`] returned without a terminal state.
#[derive(Debug, PartialEq, Eq)]
pub enum AioWaitError {
    /// The timeout elapsed; carries the submission's state at that moment.
    TimedOut(u32),
    /// No live or retired submission has this id.
    Unknown,
}

#[derive(Debug)]
struct Slot {
    /// The pending operation; taken by the worker when it starts, or dropped
    /// when the slot is cancelled first.
    op: Option<AioOp>,
    guest_buf: u64,
    guest_result: u64,
    /// Terminal outcome `(return_value, state, staged read data)`.
    outcome: Option<(i64, u32, Option<Vec<u8>>)>,
    /// The outcome has been handed to the HLE layer already.
    drained: bool,
}

#[derive(Debug)]
struct Submission {
    state: u32,
    slots: Vec<Slot>,
    /// Slots without a terminal outcome yet.
    pending: usize,
    /// `cancel` touched this submission before it finished; the final state
    /// becomes [`AIO_STATE_ABORTED`] instead of COMPLETED.
    cancelled: bool,
}

#[derive(Default)]
struct Inner {
    submissions: HashMap<i32, Submission>,
    /// Final states of deleted submissions (bounded by `retired_order`).
    retired: HashMap<i32, u32>,
    retired_order: VecDeque<i32>,
    /// FIFO of `(submit id, slot index)` work items.
    queue: VecDeque<(i32, usize)>,
    next_id: i32,
    workers_spawned: bool,
    shutdown: bool,
}

struct Shared {
    fs: Arc<VirtualFileSystem>,
    inner: Mutex<Inner>,
    /// Wakes workers when work is queued (or shutdown is requested).
    work_cv: Condvar,
    /// Wakes waiters when any submission reaches a terminal state.
    done_cv: Condvar,
}

/// The process-scoped AIO engine. Cheap to construct: worker threads spawn
/// lazily on the first submit, so a kernel that never uses AIO owns no
/// threads. Dropping the engine shuts the pool down.
pub struct AioEngine {
    shared: Arc<Shared>,
}

impl AioEngine {
    /// Create an engine performing I/O through `fs` — the same descriptor
    /// table the synchronous file HLE path uses.
    pub fn new(fs: Arc<VirtualFileSystem>) -> Self {
        Self {
            shared: Arc::new(Shared {
                fs,
                inner: Mutex::new(Inner {
                    next_id: 1,
                    ..Inner::default()
                }),
                work_cv: Condvar::new(),
                done_cv: Condvar::new(),
            }),
        }
    }

    /// Submit a batch of requests under one new id (>= 1, never 0 — a zero
    /// id is `sceKernelAioCancelRequest`'s "no request" sentinel). Returns
    /// immediately; workers perform the I/O. An empty batch completes
    /// instantly.
    pub fn submit(&self, requests: Vec<AioRequest>) -> i32 {
        let mut inner = self.shared.inner.lock();
        let id = inner.next_id;
        // Skip 0 and negatives on wrap: ids are guest-visible s32 handles.
        inner.next_id = if inner.next_id == i32::MAX {
            1
        } else {
            inner.next_id + 1
        };
        let pending = requests.len();
        let slots = requests
            .into_iter()
            .map(|request| Slot {
                op: Some(request.op),
                guest_buf: request.guest_buf,
                guest_result: request.guest_result,
                outcome: None,
                drained: false,
            })
            .collect();
        inner.submissions.insert(
            id,
            Submission {
                state: if pending == 0 {
                    AIO_STATE_COMPLETED
                } else {
                    AIO_STATE_SUBMITTED
                },
                slots,
                pending,
                cancelled: false,
            },
        );
        for index in 0..pending {
            inner.queue.push_back((id, index));
        }
        if pending > 0 && !inner.workers_spawned {
            inner.workers_spawned = true;
            for worker in 0..WORKER_COUNT {
                let shared = Arc::clone(&self.shared);
                std::thread::Builder::new()
                    .name(format!("raeen-aio-{worker}"))
                    .spawn(move || worker_loop(&shared))
                    .expect("spawn AIO worker");
            }
        }
        self.shared.work_cv.notify_all();
        debug!(id, requests = pending, "AIO submit");
        id
    }

    /// Current state of a submission (never blocks). Retired (deleted)
    /// submissions answer with their final state; unknown ids answer `None`.
    pub fn poll(&self, id: i32) -> Option<u32> {
        let inner = self.shared.inner.lock();
        inner
            .submissions
            .get(&id)
            .map(|submission| submission.state)
            .or_else(|| inner.retired.get(&id).copied())
    }

    /// Block until the submission reaches a terminal state, or `timeout`
    /// elapses (`None` = wait indefinitely — callers that must remain
    /// interruptible should pass a slice and loop).
    pub fn wait(&self, id: i32, timeout: Option<std::time::Duration>) -> Result<u32, AioWaitError> {
        let deadline = timeout.map(|t| std::time::Instant::now() + t);
        let mut inner = self.shared.inner.lock();
        loop {
            let Some(submission) = inner.submissions.get(&id) else {
                return match inner.retired.get(&id) {
                    Some(&state) => Ok(state),
                    None => Err(AioWaitError::Unknown),
                };
            };
            let state = submission.state;
            if state == AIO_STATE_COMPLETED || state == AIO_STATE_ABORTED {
                return Ok(state);
            }
            match deadline {
                Some(deadline) => {
                    if self
                        .shared
                        .done_cv
                        .wait_until(&mut inner, deadline)
                        .timed_out()
                    {
                        let state = inner
                            .submissions
                            .get(&id)
                            .map(|submission| submission.state)
                            .unwrap_or(AIO_STATE_ABORTED);
                        if state == AIO_STATE_COMPLETED || state == AIO_STATE_ABORTED {
                            return Ok(state);
                        }
                        return Err(AioWaitError::TimedOut(state));
                    }
                }
                None => self.shared.done_cv.wait(&mut inner),
            }
        }
    }

    /// Cancel: mark every **not-yet-started** request aborted
    /// (`returnValue = ECANCELED`). Requests already running finish normally,
    /// but the submission's final state becomes [`AIO_STATE_ABORTED`].
    /// Cancelling a submission that already completed is a no-op reporting
    /// `COMPLETED`. Returns the submission state after the operation, or
    /// `None` for an unknown id (retired ids report their final state).
    pub fn cancel(&self, id: i32) -> Option<u32> {
        let mut inner = self.shared.inner.lock();
        let Some(submission) = inner.submissions.get_mut(&id) else {
            return inner.retired.get(&id).copied();
        };
        if submission.state == AIO_STATE_COMPLETED || submission.state == AIO_STATE_ABORTED {
            return Some(submission.state);
        }
        let mut any_aborted = false;
        for slot in &mut submission.slots {
            // A slot that still owns its op has not been started by a worker;
            // dropping the op both aborts it and makes the queued work item a
            // no-op when a worker eventually pops it.
            if slot.op.take().is_some() {
                slot.outcome = Some((RETURN_ECANCELED, AIO_STATE_ABORTED, None));
                submission.pending -= 1;
                any_aborted = true;
            }
        }
        // Only an actual abort poisons the final state. A cancel that found
        // every request already running could cancel nothing — those requests
        // finish normally and the submission completes (Orbis reports
        // PROCESSING for "could not cancel").
        if any_aborted {
            submission.cancelled = true;
        }
        if submission.pending == 0 {
            submission.state = AIO_STATE_ABORTED;
            self.shared.done_cv.notify_all();
        }
        debug!(id, state = submission.state, "AIO cancel");
        Some(submission.state)
    }

    /// Take the terminal, not-yet-drained request outcomes of a submission
    /// (each is handed out exactly once). The HLE layer delivers these to
    /// guest memory on a guest thread.
    pub fn drain_completions(&self, id: i32) -> Vec<AioCompletion> {
        let mut inner = self.shared.inner.lock();
        let Some(submission) = inner.submissions.get_mut(&id) else {
            return Vec::new();
        };
        let mut drained = Vec::new();
        for (index, slot) in submission.slots.iter_mut().enumerate() {
            if slot.drained {
                continue;
            }
            if let Some((return_value, state, data)) = slot.outcome.take() {
                slot.drained = true;
                // Keep a data-less copy so the outcome stays observable
                // (poll answers from submission.state, not slot outcomes,
                // so dropping is fine — nothing re-reads it).
                drained.push(AioCompletion {
                    index,
                    return_value,
                    state,
                    guest_buf: slot.guest_buf,
                    guest_result: slot.guest_result,
                    data,
                });
            }
        }
        drained
    }

    /// Delete a submission: cancel anything not yet started, remove it, and
    /// retire its final state (so a late poll of the deleted id still
    /// answers). Returns the final state plus the undrained terminal
    /// completions for the HLE layer to deliver, or `None` for an unknown
    /// id. Requests still running on a worker are detached — their outcome
    /// is dropped when the worker finds the submission gone, and no guest
    /// memory is ever written for them.
    pub fn delete(&self, id: i32) -> Option<(u32, Vec<AioCompletion>)> {
        // Abort whatever has not started; then take the submission out.
        self.cancel(id)?;
        let completions = self.drain_completions(id);
        let mut inner = self.shared.inner.lock();
        let submission = inner.submissions.remove(&id)?;
        // An in-flight submission's guest-observable fate is "aborted": its
        // remaining outcomes will never be delivered.
        let final_state = if submission.pending > 0 {
            AIO_STATE_ABORTED
        } else {
            submission.state
        };
        inner.retired.insert(id, final_state);
        inner.retired_order.push_back(id);
        if inner.retired_order.len() > RETIRED_CAP
            && let Some(evicted) = inner.retired_order.pop_front()
        {
            inner.retired.remove(&evicted);
        }
        // A waiter blocked on this id must wake and observe the retirement —
        // in-flight workers whose submission is now gone will never notify
        // for it.
        self.shared.done_cv.notify_all();
        debug!(id, final_state, "AIO delete");
        Some((final_state, completions))
    }
}

impl Drop for AioEngine {
    fn drop(&mut self) {
        let mut inner = self.shared.inner.lock();
        inner.shutdown = true;
        drop(inner);
        self.shared.work_cv.notify_all();
    }
}

/// Perform one request's file I/O through the shared VFS. Runs on a worker
/// thread; touches no guest memory.
fn perform(fs: &VirtualFileSystem, op: AioOp) -> (i64, u32, Option<Vec<u8>>) {
    match op {
        AioOp::Read { fd, offset, nbyte } => {
            if nbyte > MAX_REQUEST_BYTES {
                return (RETURN_EINVAL, AIO_STATE_ABORTED, None);
            }
            match fs.pread(fd, nbyte as usize, offset) {
                Ok(data) => (data.len() as i64, AIO_STATE_COMPLETED, Some(data)),
                Err(error) => {
                    warn!(fd, offset, nbyte, "AIO read failed: {error}");
                    (aio_io_error(&error), AIO_STATE_ABORTED, None)
                }
            }
        }
        AioOp::Write { fd, offset, data } => match fs.pwrite(fd, &data, offset) {
            Ok(written) => (written as i64, AIO_STATE_COMPLETED, None),
            Err(error) => {
                warn!(fd, offset, len = data.len(), "AIO write failed: {error}");
                (aio_io_error(&error), AIO_STATE_ABORTED, None)
            }
        },
    }
}

/// Map a host I/O error to the sign-extended SCE code a guest reads out of
/// `SceKernelAioResult.returnValue` (mirrors the sync `pread`/`pwrite` HLE
/// mapping: unknown/read-only fd → EBADF, bad arguments → EINVAL, else EIO).
fn aio_io_error(error: &std::io::Error) -> i64 {
    match error.kind() {
        std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied => RETURN_EBADF,
        std::io::ErrorKind::InvalidInput => RETURN_EINVAL,
        _ => RETURN_EIO,
    }
}

fn worker_loop(shared: &Shared) {
    loop {
        let (id, index, op) = {
            let mut inner = shared.inner.lock();
            loop {
                if inner.shutdown {
                    return;
                }
                if let Some((id, index)) = inner.queue.pop_front() {
                    let Some(submission) = inner.submissions.get_mut(&id) else {
                        continue; // deleted while queued
                    };
                    let Some(op) = submission.slots[index].op.take() else {
                        continue; // cancelled (or already handled) while queued
                    };
                    submission.state = AIO_STATE_PROCESSING;
                    break (id, index, op);
                }
                shared.work_cv.wait(&mut inner);
            }
        };

        let outcome = perform(&shared.fs, op);

        let mut inner = shared.inner.lock();
        if let Some(submission) = inner.submissions.get_mut(&id) {
            submission.slots[index].outcome = Some(outcome);
            submission.pending -= 1;
            if submission.pending == 0 {
                submission.state = if submission.cancelled {
                    AIO_STATE_ABORTED
                } else {
                    AIO_STATE_COMPLETED
                };
                shared.done_cv.notify_all();
            }
        }
        // Submission deleted mid-flight: outcome dropped, by design.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filesystem::open_flags::{O_CREAT, O_RDONLY, O_RDWR};
    use std::path::PathBuf;
    use std::time::Duration;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("raeen-aio-{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn engine_with_file(tag: &str, contents: &[u8]) -> (AioEngine, Arc<VirtualFileSystem>, i32) {
        let dir = temp_dir(tag);
        std::fs::write(dir.join("data.bin"), contents).unwrap();
        let fs = Arc::new(VirtualFileSystem::new());
        fs.set_game_directory(&dir);
        let engine = AioEngine::new(Arc::clone(&fs));
        let fd = fs.open("/app0/data.bin", O_RDONLY, 0).unwrap();
        (engine, fs, fd)
    }

    fn wait_terminal(engine: &AioEngine, id: i32) -> u32 {
        engine
            .wait(id, Some(Duration::from_secs(10)))
            .expect("submission reaches a terminal state")
    }

    #[test]
    fn submit_read_completes_and_stages_the_bytes() {
        let (engine, _fs, fd) = engine_with_file("read", b"hello, aio world");
        let id = engine.submit(vec![AioRequest {
            op: AioOp::Read {
                fd,
                offset: 7,
                nbyte: 3,
            },
            guest_buf: 0x1000,
            guest_result: 0x2000,
        }]);
        assert!(id >= 1);
        assert_eq!(wait_terminal(&engine, id), AIO_STATE_COMPLETED);

        let completions = engine.drain_completions(id);
        assert_eq!(completions.len(), 1);
        let completion = &completions[0];
        assert_eq!(completion.index, 0);
        assert_eq!(completion.state, AIO_STATE_COMPLETED);
        assert_eq!(completion.return_value, 3);
        assert_eq!(completion.guest_buf, 0x1000);
        assert_eq!(completion.guest_result, 0x2000);
        assert_eq!(completion.data.as_deref(), Some(&b"aio"[..]));

        // Drained exactly once.
        assert!(engine.drain_completions(id).is_empty());
    }

    #[test]
    fn submit_write_lands_in_the_descriptor_and_reads_back() {
        let dir = temp_dir("write");
        let fs = Arc::new(VirtualFileSystem::new());
        fs.set_game_directory(&dir);
        let engine = AioEngine::new(Arc::clone(&fs));
        let fd = fs.open("/app0/out.bin", O_RDWR | O_CREAT, 0o644).unwrap();

        let id = engine.submit(vec![AioRequest {
            op: AioOp::Write {
                fd,
                offset: 4,
                data: b"WXYZ".to_vec(),
            },
            guest_buf: 0,
            guest_result: 0x2000,
        }]);
        assert_eq!(wait_terminal(&engine, id), AIO_STATE_COMPLETED);
        let completions = engine.drain_completions(id);
        assert_eq!(completions[0].return_value, 4);
        assert!(completions[0].data.is_none());

        // The bytes are visible through the same descriptor table the sync
        // path uses (sparse gap zero-filled).
        let read_back = fs.pread(fd, 8, 0).unwrap();
        assert_eq!(&read_back, &[0, 0, 0, 0, b'W', b'X', b'Y', b'Z']);
    }

    #[test]
    fn multi_request_batch_completes_every_slot_under_one_id() {
        let (engine, _fs, fd) = engine_with_file("multi", b"0123456789");
        let requests = (0..4)
            .map(|i| AioRequest {
                op: AioOp::Read {
                    fd,
                    offset: i as u64 * 2,
                    nbyte: 2,
                },
                guest_buf: 0x1000 + i as u64 * 0x10,
                guest_result: 0x2000 + i as u64 * 0x10,
            })
            .collect();
        let id = engine.submit(requests);
        assert_eq!(wait_terminal(&engine, id), AIO_STATE_COMPLETED);
        let mut completions = engine.drain_completions(id);
        completions.sort_by_key(|completion| completion.index);
        assert_eq!(completions.len(), 4);
        for (i, completion) in completions.iter().enumerate() {
            assert_eq!(completion.index, i);
            assert_eq!(completion.return_value, 2);
            let expected = [b'0' + 2 * i as u8, b'1' + 2 * i as u8];
            assert_eq!(completion.data.as_deref(), Some(&expected[..]));
        }
    }

    #[test]
    fn poll_never_blocks_and_reports_the_lifecycle() {
        let (engine, _fs, fd) = engine_with_file("poll", b"abc");
        assert_eq!(engine.poll(1234), None);
        let id = engine.submit(vec![AioRequest {
            op: AioOp::Read {
                fd,
                offset: 0,
                nbyte: 3,
            },
            guest_buf: 0,
            guest_result: 0,
        }]);
        // Immediately after submit the state is SUBMITTED or later — never
        // an unknown/blocked answer.
        let early = engine.poll(id).expect("known id");
        assert!(
            (AIO_STATE_SUBMITTED..=AIO_STATE_ABORTED).contains(&early),
            "unexpected state {early}"
        );
        wait_terminal(&engine, id);
        assert_eq!(engine.poll(id), Some(AIO_STATE_COMPLETED));
    }

    #[test]
    fn wait_times_out_on_a_stalled_submission_and_unknown_ids_error() {
        let (engine, _fs, _fd) = engine_with_file("timeout", b"x");
        assert_eq!(
            engine.wait(555, Some(Duration::from_millis(10))),
            Err(AioWaitError::Unknown)
        );
        // A submission that never gets terminal: simulate by submitting an
        // empty engine id then... every real submission terminates, so test
        // the timeout path with a bad-fd read racing a long wait instead:
        // the observable contract is that a *pending* wait with a tiny
        // timeout returns TimedOut or the terminal state — never hangs.
        let id = engine.submit(vec![AioRequest {
            op: AioOp::Read {
                fd: 0x7FFF_0000,
                offset: 0,
                nbyte: 1,
            },
            guest_buf: 0,
            guest_result: 0,
        }]);
        match engine.wait(id, Some(Duration::from_millis(1))) {
            Ok(state) => assert_eq!(state, AIO_STATE_COMPLETED),
            Err(AioWaitError::TimedOut(state)) => {
                assert!((AIO_STATE_SUBMITTED..=AIO_STATE_PROCESSING).contains(&state));
                // It still terminates afterwards.
                assert_eq!(wait_terminal(&engine, id), AIO_STATE_COMPLETED);
            }
            Err(other) => panic!("unexpected wait error: {other:?}"),
        }
    }

    #[test]
    fn failed_request_reports_a_negative_sce_return_value() {
        let (engine, _fs, _fd) = engine_with_file("badfd", b"x");
        let id = engine.submit(vec![AioRequest {
            op: AioOp::Read {
                fd: 0x7FFF_0001,
                offset: 0,
                nbyte: 4,
            },
            guest_buf: 0,
            guest_result: 0,
        }]);
        // The batch completes (it was processed); the request itself aborted.
        assert_eq!(wait_terminal(&engine, id), AIO_STATE_COMPLETED);
        let completions = engine.drain_completions(id);
        assert_eq!(completions[0].state, AIO_STATE_ABORTED);
        assert_eq!(completions[0].return_value, RETURN_EBADF);
        assert!(completions[0].return_value < 0);
    }

    #[test]
    fn cancel_before_start_aborts_and_after_complete_is_a_noop() {
        let (engine, _fs, fd) = engine_with_file("cancel", b"abcdef");

        // Cancel-after-complete: no-op, stays COMPLETED.
        let done = engine.submit(vec![AioRequest {
            op: AioOp::Read {
                fd,
                offset: 0,
                nbyte: 2,
            },
            guest_buf: 0,
            guest_result: 0,
        }]);
        assert_eq!(wait_terminal(&engine, done), AIO_STATE_COMPLETED);
        assert_eq!(engine.cancel(done), Some(AIO_STATE_COMPLETED));
        assert_eq!(engine.poll(done), Some(AIO_STATE_COMPLETED));

        // Cancel-before-start: on a FRESH engine the first submit also has
        // to spawn the worker threads, so a cancel issued immediately after
        // submit wins the race in practice (thread spawn >> lock hand-off).
        // The loop with a new engine per attempt keeps the test sound even
        // if a scheduler quirk lets a worker win once.
        let mut aborted_seen = false;
        for attempt in 0..32 {
            let (fresh, _fs2, fd2) = engine_with_file(&format!("cancel-race-{attempt}"), b"abcdef");
            let id = fresh.submit(vec![AioRequest {
                op: AioOp::Read {
                    fd: fd2,
                    offset: 0,
                    nbyte: 2,
                },
                guest_buf: 0,
                guest_result: 0,
            }]);
            let state = fresh.cancel(id).expect("known id");
            if state == AIO_STATE_ABORTED {
                let completions = fresh.drain_completions(id);
                assert_eq!(completions.len(), 1);
                assert_eq!(completions[0].state, AIO_STATE_ABORTED);
                assert_eq!(completions[0].return_value, RETURN_ECANCELED);
                assert_eq!(fresh.poll(id), Some(AIO_STATE_ABORTED));
                aborted_seen = true;
                break;
            }
            // Raced: the worker already owned the request. It cannot be
            // cancelled, so it finishes normally.
            assert_eq!(wait_terminal(&fresh, id), AIO_STATE_COMPLETED);
        }
        assert!(
            aborted_seen,
            "cancel never beat worker spawn in 32 attempts — pool too eager?"
        );
        assert_eq!(engine.cancel(4242), None);
    }

    #[test]
    fn delete_retires_the_final_state_and_returns_undrained_completions() {
        let (engine, _fs, fd) = engine_with_file("delete", b"payload!");
        let id = engine.submit(vec![AioRequest {
            op: AioOp::Read {
                fd,
                offset: 0,
                nbyte: 8,
            },
            guest_buf: 0x1000,
            guest_result: 0x2000,
        }]);
        assert_eq!(wait_terminal(&engine, id), AIO_STATE_COMPLETED);
        let (state, completions) = engine.delete(id).expect("known id");
        assert_eq!(state, AIO_STATE_COMPLETED);
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].data.as_deref(), Some(&b"payload!"[..]));

        // Retired: poll still answers; wait reports the final state;
        // a second delete is unknown.
        assert_eq!(engine.poll(id), Some(AIO_STATE_COMPLETED));
        assert_eq!(engine.wait(id, None), Ok(AIO_STATE_COMPLETED));
        assert!(engine.delete(id).is_none());
    }

    #[test]
    fn ids_are_unique_positive_and_never_zero() {
        let (engine, _fs, fd) = engine_with_file("ids", b"abc");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..16 {
            let id = engine.submit(vec![AioRequest {
                op: AioOp::Read {
                    fd,
                    offset: 0,
                    nbyte: 1,
                },
                guest_buf: 0,
                guest_result: 0,
            }]);
            assert!(id >= 1);
            assert!(seen.insert(id), "id {id} reused");
        }
    }

    #[test]
    fn empty_batch_completes_instantly() {
        let (engine, _fs, _fd) = engine_with_file("empty", b"x");
        let id = engine.submit(Vec::new());
        assert_eq!(engine.poll(id), Some(AIO_STATE_COMPLETED));
        assert!(engine.drain_completions(id).is_empty());
    }
}
