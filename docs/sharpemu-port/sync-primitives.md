# Guest synchronization primitives — SharpEmu port (2026-07-28)

Two of Raeen's weakest guest synchronization primitives replaced with real ones.
Engine titles (UE5, GTA V) spin on exactly these.

Sources studied (read-only, `reference/sharpemu`, GPL-2.0-or-later):

| SharpEmu | Locus |
|---|---|
| PR #422 `09bd4f0` — `sceKernelSyncOnAddress` wait/wake | `src/SharpEmu.Libs/Kernel/KernelSyncOnAddressCompatExports.cs` |
| PR #439 `73e8821` — hand off mutex ownership to the head waiter on unlock | `src/SharpEmu.Libs/Kernel/KernelPthreadCompatExports.cs` (`PthreadMutexUnlockCore`, `TryGrantMutexWaiterLocked`, `EnqueueMutexWaiterLocked`) |

Both were read at the **live tip** rather than as isolated commits, per the revert
trap (`6db095e` reverted work that `db4339f`/#650 later restored).

---

## 1. The unified waiter primitive

Before this batch the tree had exactly one true per-waiter park/unpark queue —
`PthreadCondWaiter` / `PthreadCond` in `crates/raeen-kernel/src/lib.rs` — and two
places that needed one and did not have it.

It was split into two reusable pieces, both in `crates/raeen-kernel/src/lib.rs`:

* **`GuestWaiter`** (was `PthreadCondWaiter`) — one parked guest thread: a
  private `signaled` bool under its own `parking_lot::Mutex`, plus a `changed`
  condvar. The private bit is what closes the **unpark race**: a waiter registers
  under whichever queue lock owns it, drops that lock, then parks. A wake landing
  in that window sets `signaled` under the waiter's *own* lock, so
  `wait_for_signal` returns immediately instead of parking on a wake that will
  never come again.
* **`GuestWaitQueue`** (was `PthreadCond`'s body) — the FIFO:
  `enqueue_waiter` / `cancel_waiter` / `signal_one` / `signal_many` /
  `signal_thread` / `broadcast` / `waiter_count`. Wake selection completes
  *while holding the queue lock*, which is what makes `cancel_waiter` returning
  `false` mean "a waker already took me, and my wake bit is ready to observe".

Three users now:

| User | Queue container | Why |
|---|---|---|
| `PthreadCond` | owns a `GuestWaitQueue`, delegates every method | Public API and the `monotonic` clock flag are byte-for-byte unchanged, so `crates/raeen-hle/src/pthread_cond.rs` needed no edits at all. |
| `SyncAddressTable` | `DashMap<u64, Arc<GuestWaitQueue>>` | One queue per watched guest address. |
| `PthreadMutex` | a plain `VecDeque<Arc<GuestWaiter>>` **inside the mutex state**, not a `GuestWaitQueue` | See the justification below. |

### Why the mutex reuses the *waiter* but not the *queue container*

`GuestWaitQueue` carries its own lock. The mutex handoff must transfer `owner`
and dequeue the head as **one atomic step** under the mutex's existing state
lock; a second lock nested inside would be sound (the ordering state → queue is
consistent on every path, and nothing takes queue → state) but it buys nothing
and makes the atomicity argument harder to read. So the mutex holds the deque
directly under `parking_lot::Mutex<PthreadMutex>` — the same single-lock shape
SharpEmu uses (`lock (state.SyncRoot)` around both) — while still reusing
`GuestWaiter` for the actual parking. The genuinely reusable part (the race-free
per-waiter wake bit) is shared; the container is not.

### `sys_futex`

`crates/raeen-kernel/src/syscalls/thread.rs` `sys_futex` was three `debug!` lines
and `Ok(0)`. It now **shares `OrbisKernel::sync_addresses`** with the HLE entry
points, so a guest that reaches the same watched word through the syscall and
through libkernel parks in one queue rather than two. It deliberately does *not*
do the value compare: that needs `GuestMemory` access this crate does not own, and
skipping it only costs an unnecessary park (released by the bounded slice) — it
never loses a wake. The syscall path is not the launch hot path today (launches go
through HLE trampolines), but a split parking lot would be a silent lost-wake bug
the day it is.

---

## 2. `sceKernelSyncOnAddressWait` / `Wake` — the parking lot

### What was there

`crates/raeen-hle/src/libkernel.rs` (~2968–2992 before this change):

* `Wait` did `std::thread::sleep(10ms)` and returned `SCE_OK`. It **never read or
  compared the watched word**. `_ctx` was unused — there was no per-address state
  of any kind.
* `Wake` logged and returned `SCE_OK`.

A guest that treats a 0 return as "the value changed", without re-checking, will
livelock. The registration comment admitted "Raeen has no true parking lot here".

### What it is now

`SyncAddressTable` (`crates/raeen-kernel/src/lib.rs`), reached as
`ctx.kernel.sync_addresses`, plus `sync_on_address_wait` in
`crates/raeen-hle/src/libkernel.rs`. Newly registered:
`sceKernelSyncOnAddressWait32` and `sceKernelSyncOnAddressWait64` (names already
present in `crates/raeen-firmware/src/dynlib/nid_names.txt`:
`0769fc683a2b487e`, `3d94218a22d7445b`; `Wait` = `1dce02691e8904bd`,
`Wake` = `ab6cbfc032155990`).

**Enqueue-before-read is the whole correctness argument.** The waiter joins the
address's FIFO *first*, then the watched word is read:

* a waker that writes the word **after** our read necessarily finds us already
  queued, and wakes us;
* a waker that wrote **before** our read is observed by the compare, and we
  return `EAGAIN` without parking.

There is no window where a wake is lost. This is why Raeen needs none of
SharpEmu's per-address wake **generation counter** — that counter exists
precisely to approximate this ordering, and it has the same failure mode Raeen
already removed from `PthreadCond` (a generation bump is observable by every
waiter, so wake-one silently becomes broadcast).

Behavior:

| Case | Result |
|---|---|
| `addr == 0` | `SCE_KERNEL_ERROR_EINVAL` |
| `*addr != expected` (sized variants) | `SCE_KERNEL_ERROR_EAGAIN` (`0x8002000B`), no park |
| `addr` unreadable (sized variants) | `SCE_KERNEL_ERROR_EFAULT`, logged |
| woken by `Wake` | `SCE_OK` |
| decoded deadline elapsed | `SCE_KERNEL_ERROR_ETIMEDOUT` (`0x8002003C`) |
| deadline elapsed but a wake already took us | `SCE_OK` — the wake wins the race |
| no decodable deadline, 100 ms elapsed | `SCE_OK` as a permitted spurious wakeup |
| process terminating | `SCE_OK`, waiter cancelled |

`Wake(addr, count)`: `count` of 1 is wake-one in FIFO order; 0 or an implausibly
large value is wake-all (SharpEmu's reading of the same argument). Waking an
address nobody is parked on is success, not an error — a wake that beats its wait
is the ordinary uncontended case, and the waiter's compare-on-entry catches it.

### Honest gaps

* **The generic unsized `sceKernelSyncOnAddressWait` does no value compare.** Its
  argument layout past the address is not recovered; SharpEmu records the same gap
  verbatim ("that exact value is not recovered here"). Guessing which register
  holds the expected word risks returning a spurious `EAGAIN` forever, so that
  variant parks without a compare and relies on the wake plus the 100 ms
  self-heal. The `*32`/`*64` spellings name their own width, and the
  `(addr, expected, timeout)` triple is the standard futex shape, so those do the
  real compare. If a real trace pins the generic layout, that variant should
  switch to the compare path — it is a one-argument change to
  `sync_on_address_wait`.
* **The timeout argument is interpreted, not known.** 0 is the futex `NULL`
  spelling (no deadline). A value in `1..=60_000_000` becomes a relative-µs
  deadline. Anything larger is refused as a deadline rather than gambled on,
  because reading a guest pointer as microseconds would either hang or (if it were
  a pointer to a struct) manufacture instant timeouts. A mis-decode therefore
  degrades to "no deadline", which the self-heal bounds — never to a hang.
* Every park is bounded. A wait can never hang the guest even if the wake arrives
  through a path Raeen has not resolved.

---

## 3. pthread mutex unlock — direct handoff

### What was there

`crates/raeen-hle/src/pthread_sync.rs` `hle_mutex_unlock` set `state.owner = 0`,
**dropped the guard**, then called `shared.unlocked.notify_one()`. Despite the
comment claiming to "hand the lock to" a waiter, an arriving thread could barge in
and take ownership before the woken waiter reacquired. Wake order was
`parking_lot`'s, not FIFO. Waiters parked on a shared per-mutex condvar — parking,
but **no queue** — so starvation was bounded only by the 10 ms re-check slice in
`lock_core`.

### What it is now

`PthreadMutex::try_grant_head` (`crates/raeen-kernel/src/lib.rs`) transfers
ownership **while the state lock is held** and wakes exactly the granted waiter:

```
if self.owner != 0 { return None }
let waiter = self.waiters.pop_front()?;
self.owner = waiter.thread;
self.recursion = 1;
waiter.wake();
Some(waiter.thread)
```

`hle_mutex_unlock`'s final release calls it with the guard still held. The woken
thread observes `owner == self` and returns `OK` immediately — there is nothing to
reacquire and nothing to race. `PthreadMutexShared::unlocked` (the shared condvar)
is **gone**: `notify_one` on a shared condvar wakes an arbitrary parked thread,
which is precisely the barging the handoff exists to stop.

`lock_core` was restructured into two phases:

1. **Under the state lock** — self-relock matrix (unchanged), uncontended
   acquire, `try_only`/termination/deadline refusals, then `enqueue_waiter`
   *before* the lock is dropped, followed by an unconditional `try_grant_head`
   (which either grants us the mutex or nudges a free-but-queued mutex to its
   real head).
2. **Parked on our own `GuestWaiter`** in 10 ms slices — so termination and
   deadlines are still observed if no unlock ever arrives — with the >3 s
   stuck-holder diagnostic intact (it now also reports the queue depth).

### Preserved behaviors — the type matrix and the rest

Everything the old implementation did is still done, in the same order:

| Behavior | Where | Status |
|---|---|---|
| `owner == current` checked **before** the free-mutex check | `lock_core` phase 1 | unchanged — self-relock never queues, so recursion is never double-counted by a grant |
| `MUTEX_RECURSIVE` re-lock by count | phase 1 match arm | verbatim |
| `MUTEX_NORMAL` / `MUTEX_ADAPTIVE` deliberately-lenient self-relock (counts instead of `EDEADLK`) | phase 1 match arm | verbatim |
| `MUTEX_ERRORCHECK` self-relock → `EDEADLK`, `EBUSY` under `try_only` | phase 1 match arm (`_` arm) | verbatim |
| `EPERM` on non-owner unlock | `hle_mutex_unlock` | unchanged; a rejected unlock grants nothing |
| lenient `recursion <= 0` (`OK` for NORMAL/ADAPTIVE, `EINVAL` otherwise) | `hle_mutex_unlock` | unchanged, and evaluated before the owner check as before |
| nested unlock does not release | `hle_mutex_unlock` | handoff only runs at `recursion == 0` |
| >3 s stuck-holder warning, named holder | `lock_core` phase 2 | preserved, plus `queued` |
| owner-death recovery (`release_locks_owned_by`, `release_mutexes_owned_by`) | `crates/raeen-kernel/src/lib.rs` | now **grants the head** instead of `notify_all` |
| SCE vs POSIX error coding (`posix_to_sce`) | unchanged | unchanged |

Three new invariants come with the queue:

* **Anti-barging.** A free mutex with a queued waiter is refused by the acquire
  fast path (`has_waiters()`), so arrivals queue instead of jumping ahead.
  Consequence: `Trylock` reports `EBUSY` in that window. That is always a legal
  trylock answer and callers already loop on it — including Minecraft's
  `_Mtx_trylock`, which maps SCE `EBUSY` to `_Thrd_busy`.
* **A timeout that races a grant loses.** `cancel_waiter` returning `false` means
  the handoff already granted us the mutex, so the caller returns `OK`, not
  `ETIMEDOUT` — reporting a timeout there would leak ownership the guest never
  learns it holds. `abandon_wait` in `pthread_sync.rs` is the single place this is
  decided.
* **One queue entry per thread.** `enqueue_waiter` prunes a thread's earlier
  entry. A stale entry (most often a `cond_timedwait` timeout whose re-acquire
  handoff was lost) would clog the FIFO head, and the handoff would then grant the
  mutex to a thread nobody is parked on — wedging it permanently. SharpEmu hit
  exactly this deadlocking Hades.

Free-with-waiters is the one state that must never persist, since anti-barging
makes it unacquirable. Every path that clears `owner` therefore calls
`try_grant_head` while still holding the lock: normal unlock, `abandon_wait` after
a cancel, and both owner-death recovery functions.

---

## Tests

All deterministic — **no host threads, no sleeps**. The model is
`signal_wakes_only_the_oldest_waiter`: drive the queue API directly and assert the
per-waiter wake bit plus `waiter_count()`.

| Module | Tests | Covers |
|---|---:|---|
| `raeen-kernel` `sync_address_table_tests` | 4 | address scoping, wake-one FIFO vs wake-all, wake with no waiters (and that it materializes no queue), cancel-loses-to-wake |
| `raeen-kernel` `pthread_mutex_handoff_tests` | 6 | ownership moves to the head + only it wakes, FIFO across successive releases, `has_waiters` anti-barging gate, granted waiter cannot cancel, cancel removes so the survivor is granted, stale-entry pruning |
| `raeen-hle` `libkernel::tests` | 8 | all four exports registered, `EINVAL` on null, `EAGAIN` without parking on mismatch, width-correct 32/64 compare (incl. masking the high half of `expected`), `EFAULT` on unmapped, `Wake` FIFO + address scoping via the entry point, wake with no waiter, timeout decoding |
| `raeen-hle` `pthread_sync::tests` | 6 | unlock hands ownership + wakes only the head, recursive hands off only at the last level, lenient NORMAL self-relock unwinds first (+ `recursion <= 0`), non-owner unlock is `EPERM` and grants nothing, `Trylock` does not barge, owner-death hands to the head |

Counts (measured in this worktree):

| Package | Before | After |
|---|---:|---:|
| `raeen-hle` lib | 525 | **539** |
| `raeen-kernel` lib | 53 | **63** |
| `raeen-runtime` lib / `execute` | 77 / 56 | **77 / 56** (unchanged) |

`cargo fmt --all` clean. `cargo clippy -p raeen-hle -p raeen-kernel --all-targets
-- -D warnings` clean. `cargo clippy --workspace --all-targets --exclude
raeen-gpu -- -D warnings` clean.

### Pre-existing failures, not caused by this batch

* `raeen-gpu` fails the workspace clippy gate on `clippy::identity_op` at
  `crates/raeen-gpu/src/vulkan/compute.rs:2200` (`(1 << 16) | 0`). That crate has
  **zero diff** from `HEAD` here; the line came in with `1a078d2`.
* `libsce_video_out::tests::consecutive_vblank_waits_land_one_period_apart` is a
  wall-clock test with a ±3 ms / +10 ms window and flakes under host load. It
  passed 5/5 in isolation after this change.

---

## Follow-ups

1. Pin the generic `sceKernelSyncOnAddressWait` argument layout from a real trace,
   then enable its value compare.
2. `scePthreadCreate` / a real second guest context (M1 wall #5) is still the gate
   on any of this being exercised by more than one guest thread at a time. The
   primitives are now correct *for* multiple contexts; the contexts do not exist
   yet. **This batch closes no milestone.**
3. `sys_futex` cannot compare values from `raeen-kernel`. If the syscall path ever
   becomes hot, it needs guest-memory access plumbed in — or the compare hoisted
   to its caller.
