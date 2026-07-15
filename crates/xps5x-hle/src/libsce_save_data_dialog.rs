//! HLE `libSceSaveDataDialog` — the save-data dialog lifecycle.
//!
//! A title initializes the dialog (`sceSaveDataDialogInitialize`), opens it
//! with a param block (`sceSaveDataDialogOpen`), then polls its status
//! (`sceSaveDataDialogUpdateStatus`/`GetStatus`) each frame until it reports
//! `FINISHED`, reads the result (`sceSaveDataDialogGetResult`), and terminates.
//! There is no host save dialog yet, so `Open` completes immediately —
//! status jumps straight to `FINISHED` with an OK result so a polling title
//! sees a finished dialog instead of spinning forever. Ported faithfully from
//! SharpEmu's `SaveDataDialogExports` (GPL-2.0); mirrors the immediate-finish
//! behavior of the already-ported [`crate::libsce_common_dialog`] MsgDialog.
//!
//! One dialog at a time (the real API's constraint), so the status + last
//! open-params are module-level atomics, matching the MsgDialog port.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

// `SceSaveDataDialogStatus`.
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
const STATUS_RUNNING: i32 = 2;
const STATUS_FINISHED: i32 = 3;

const OK: u64 = 0;
// libSceSaveDataDialog error codes.
const ERROR_NOT_INITIALIZED: u64 = 0x80B8_0003;
const ERROR_ALREADY_INITIALIZED: u64 = 0x80B8_0004;
const ERROR_NOT_FINISHED: u64 = 0x80B8_0005;
const ERROR_NOT_RUNNING: u64 = 0x80B8_000B;
const ERROR_ARG_NULL: u64 = 0x80B8_000D;
// Shared common-dialog memory-fault result.
const ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

/// `SceSaveDataDialogResult` size.
const RESULT_SIZE: usize = 0x48;
/// Offset of the `userData` field in the open param block.
const PARAM_USER_DATA_OFFSET: u64 = 0xC8;

static STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);
static LAST_MODE: AtomicI32 = AtomicI32::new(0);
static LAST_USER_DATA: AtomicU64 = AtomicU64::new(0);

/// Register the libSceSaveDataDialog functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogInitialize",
        hle_initialize,
    );
    registry.register("libSceSaveDataDialog", "sceSaveDataDialogOpen", hle_open);
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogGetStatus",
        hle_get_status,
    );
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogUpdateStatus",
        hle_get_status,
    );
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogIsReadyToDisplay",
        |_, _| 1,
    );
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogGetResult",
        hle_get_result,
    );
    registry.register("libSceSaveDataDialog", "sceSaveDataDialogClose", hle_close);
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogTerminate",
        hle_terminate,
    );
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogProgressBarInc",
        |_, _| OK,
    );
    registry.register(
        "libSceSaveDataDialog",
        "sceSaveDataDialogProgressBarSetValue",
        |_, _| OK,
    );
}

/// `sceSaveDataDialogInitialize()`: `NONE` → `INITIALIZED`; a second init
/// without a terminate is an error.
fn hle_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    if STATUS
        .compare_exchange(
            STATUS_NONE,
            STATUS_INITIALIZED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    {
        OK
    } else {
        ERROR_ALREADY_INITIALIZED
    }
}

/// `sceSaveDataDialogOpen(param)`: record the mode/userData and, since there
/// is no host dialog, complete immediately (`FINISHED`).
fn hle_open(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    if param == 0 {
        return ERROR_ARG_NULL;
    }
    let status = STATUS.load(Ordering::Acquire);
    if status != STATUS_INITIALIZED && status != STATUS_FINISHED {
        return ERROR_NOT_INITIALIZED;
    }

    let mut mode = [0u8; 4];
    LAST_MODE.store(
        if ctx.mem.read(param, &mut mode) {
            i32::from_le_bytes(mode)
        } else {
            0
        },
        Ordering::Relaxed,
    );
    let mut user_data = [0u8; 8];
    LAST_USER_DATA.store(
        if ctx.mem.read(param + PARAM_USER_DATA_OFFSET, &mut user_data) {
            u64::from_le_bytes(user_data)
        } else {
            0
        },
        Ordering::Relaxed,
    );

    STATUS.store(STATUS_FINISHED, Ordering::Release);
    OK
}

/// `sceSaveDataDialogGetStatus()` / `UpdateStatus()`: the current status.
fn hle_get_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.load(Ordering::Acquire) as u32 as u64
}

/// `sceSaveDataDialogGetResult(result)`: once `FINISHED`, write the 0x48-byte
/// result (mode @0x00, result/buttonId zeroed, userData @0x20).
fn hle_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let result_addr = args.first().copied().unwrap_or(0);
    if result_addr == 0 {
        return ERROR_ARG_NULL;
    }
    if STATUS.load(Ordering::Acquire) != STATUS_FINISHED {
        return ERROR_NOT_FINISHED;
    }

    let mut result = [0u8; RESULT_SIZE];
    result[0x00..0x04].copy_from_slice(&LAST_MODE.load(Ordering::Relaxed).to_le_bytes());
    result[0x20..0x28].copy_from_slice(&LAST_USER_DATA.load(Ordering::Relaxed).to_le_bytes());
    if !ctx.mem.write(result_addr, &result) {
        return ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceSaveDataDialogClose()`: `RUNNING` → `FINISHED`; closing a
/// non-running dialog is an error.
fn hle_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    if STATUS
        .compare_exchange(
            STATUS_RUNNING,
            STATUS_FINISHED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    {
        OK
    } else {
        ERROR_NOT_RUNNING
    }
}

/// `sceSaveDataDialogTerminate()`: back to `NONE`; terminating an
/// uninitialized dialog is an error.
fn hle_terminate(_ctx: &HleContext, _args: &[u64]) -> u64 {
    if STATUS.swap(STATUS_NONE, Ordering::AcqRel) == STATUS_NONE {
        return ERROR_NOT_INITIALIZED;
    }
    LAST_MODE.store(0, Ordering::Relaxed);
    LAST_USER_DATA.store(0, Ordering::Relaxed);
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;
    use std::sync::{Mutex, MutexGuard};

    /// The dialog is a process-wide singleton — one at a time is the real API's
    /// constraint, and the runtime's single-active-execution invariant (see
    /// `xps5x_runtime::dispatch::CALL_LOCK`) means one guest process per host
    /// process ever drives it. That is right for the HLE, but it leaves these
    /// tests sharing one dialog while `cargo test` runs them on parallel
    /// threads of a single process — so they must not interleave. Resetting at
    /// the top of each test is not enough: it cannot stop another test's
    /// `Initialize` from landing between this test's reset and its first call.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Take the dialog for the duration of a test, from a known-`NONE` state.
    /// The returned guard must be held (bind it, don't discard it) for the rest
    /// of the test. Poisoning is ignored so that one failing test reports one
    /// failure instead of cascading into the others.
    #[must_use]
    fn acquire_dialog() -> MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        STATUS.store(STATUS_NONE, Ordering::Relaxed);
        LAST_MODE.store(0, Ordering::Relaxed);
        LAST_USER_DATA.store(0, Ordering::Relaxed);
        guard
    }

    fn env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x200),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn init_open_result_lifecycle_finishes_immediately() {
        let _dialog = acquire_dialog();
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Double init is an error; single init succeeds.
        assert_eq!(hle_initialize(&ctx, &[]), OK);
        assert_eq!(hle_initialize(&ctx, &[]), ERROR_ALREADY_INITIALIZED);

        // Open with a NULL param is rejected; a real param finishes immediately.
        assert_eq!(hle_open(&ctx, &[0]), ERROR_ARG_NULL);
        let param = 0x40u64;
        assert!(crate::GuestMemory::write(&mem, param, &7i32.to_le_bytes())); // mode
        let ud: u64 = 0xCAFE_F00D_1234_5678;
        assert!(crate::GuestMemory::write(
            &mem,
            param + PARAM_USER_DATA_OFFSET,
            &ud.to_le_bytes()
        ));
        assert_eq!(hle_open(&ctx, &[param]), OK);
        assert_eq!(hle_get_status(&ctx, &[]) as i32, STATUS_FINISHED);

        // Result reflects the recorded mode + userData.
        let result_addr = 0x100u64;
        assert_eq!(hle_get_result(&ctx, &[result_addr]), OK);
        let mut mode = [0u8; 4];
        assert!(crate::GuestMemory::read(&mem, result_addr, &mut mode));
        assert_eq!(i32::from_le_bytes(mode), 7);
        let mut got_ud = [0u8; 8];
        assert!(crate::GuestMemory::read(
            &mem,
            result_addr + 0x20,
            &mut got_ud
        ));
        assert_eq!(u64::from_le_bytes(got_ud), ud);

        // Terminate returns to NONE; a second terminate errors.
        assert_eq!(hle_terminate(&ctx, &[]), OK);
        assert_eq!(hle_get_status(&ctx, &[]) as i32, STATUS_NONE);
        assert_eq!(hle_terminate(&ctx, &[]), ERROR_NOT_INITIALIZED);
    }

    #[test]
    fn errors_when_out_of_order() {
        let _dialog = acquire_dialog();
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Open before init → NotInitialized.
        assert_eq!(hle_open(&ctx, &[0x40]), ERROR_NOT_INITIALIZED);
        // GetResult before finished → NotFinished.
        assert_eq!(hle_initialize(&ctx, &[]), OK);
        assert_eq!(hle_get_result(&ctx, &[0x100]), ERROR_NOT_FINISHED);
        // GetResult with NULL → ArgNull.
        assert_eq!(hle_get_result(&ctx, &[0]), ERROR_ARG_NULL);
        // Close when not running → NotRunning.
        assert_eq!(hle_close(&ctx, &[]), ERROR_NOT_RUNNING);
    }
}
