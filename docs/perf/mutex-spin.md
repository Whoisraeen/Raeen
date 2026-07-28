# Spin-then-park for guest pthread mutexes

**Status:** implemented, awaiting live retail A/B (soak harness).
**Baseline:** `36a9b18` (main HEAD did not compile at branch time — parallel
in-flight `raeen-gpu` change).

## Measured problem (artifacts/soak/soak-1785266485239, 2026-07-28)

Minecraft's 7 "Streaming Pool" workers + MAIN convoy on one guest mutex — the
title's libc.prx allocator lock (`owner_acquired_at=libc.prx+0x5ec9`). MAIN
alone: 138,305 `scePthreadMutexLock` HLE calls in ~15 s, ~5.3 s inside them
(`RAEEN_TIME_HLE` per-thread sinks). Worst frozen window 24.7 s; in-world FPS
1–2 during streaming.

Root cause of the cost: the FIFO direct-handoff mutex (correct, fair,
anti-barging) parked on a host primitive on **every** contended lock and did a
host wakeup on every unlock. Malloc-class critical sections are sub-microsecond
at very high frequency; a host park/unpark round trip is microseconds to tens
of microseconds, thousands of times per second, serialized through one lock.
Real adaptive mutexes (glibc `PTHREAD_MUTEX_ADAPTIVE_NP`) spin briefly before
parking for exactly this workload.

## Design (all existing semantics preserved)

- The waiter still **enqueues in FIFO order immediately** under the state
  lock; unlock still grants ownership to the queue head under the state lock
  (`PthreadMutex::try_grant_head`). No barging, no order change, no
  timeout/cancellation change, no recursion/type-matrix change.
- **After** enqueueing, the waiter spins on its own grant flag for a bounded
  budget before falling back to the existing 10 ms-sliced park loop.
- `GuestWaiter` (raeen-kernel) gained a lock-free `AtomicBool` mirror of its
  mutex-protected `signaled` bool. `wake()` Release-stores the atomic **first**,
  then sets the bool under the waiter lock and notifies. New
  `GuestWaiter::spin_for_signal(budget)` Acquire-loads the atomic with a
  pause/yield ladder (first 1024 iterations pure `spin_loop`, then every 64th
  iteration `yield_now`).
- **No lost wake at the spin→park transition:** the atomic is a fast-path hint
  only. The park path (`wait_for_signal`) re-checks the mutex-protected bool
  under its lock before sleeping, and the waker sets that bool under the same
  lock before notifying — the standard check-flag-then-park pattern, unchanged.
  Regression test:
  `raeen-kernel guest_waiter_spin_tests::grant_landing_at_the_spin_to_park_transition_is_not_lost`
  (zero-timeout park proves the flag check, not a notify, observes the grant).
- The same spin phase was applied to `sceKernelSyncOnAddressWait*` (futex),
  which shares `GuestWaiter` naturally through `GuestWaitQueue`. The
  compare-before-park futex contract is unchanged (waiter is already enqueued
  before the compare, exactly as before).
- `scePthreadCondWait` was deliberately **not** touched (same primitive, but a
  cond wait is expected to be long; keep the diff focused).

## Knob

- `RAEEN_MUTEX_SPIN=<iterations>` — pre-park spin budget, read once per
  process (`raeen_kernel::guest_waiter_spin_budget()`).
  - default: `2000` (`DEFAULT_GUEST_WAITER_SPIN`) ≈ tens of µs on a Zen core,
    comparable to one park/unpark round trip, so spin never costs meaningfully
    more than the park it replaces.
  - `0` — disables spinning entirely, restoring the previous exact
    park-per-contention behavior (A/B baseline).
  - invalid/absent — default.

## Extra uncontended-path win

`lock_core` no longer executes `Instant::now()` (a QPC syscall on Windows) on
the uncontended fast path — the wait-start timestamp is now taken only after
the spin phase fails. Additionally, the `InFlightWait` diagnostic
(`format!` + DashMap insert) is now skipped for contentions resolved within
the spin budget; previously every contended lock paid it.

## Files

- `crates/raeen-kernel/src/lib.rs` — `GuestWaiter::signaled_fast`,
  `spin_for_signal`, `guest_waiter_spin_budget`, tests.
- `crates/raeen-hle/src/pthread_sync.rs` — spin phase in `lock_core`,
  fast-path `Instant::now()` removal, multithreaded stress test.
- `crates/raeen-hle/src/libkernel.rs` — spin phase in `sync_on_address_wait`.

## Verification

- `cargo test -p raeen-kernel` — 63 + 2 green (4 new spin tests).
- `cargo test -p raeen-hle` — 568 green (1 new stress test: 4 threads × 2000
  lock/unlock over a non-atomic counter proves mutual exclusion; passes with
  default, `RAEEN_MUTEX_SPIN=0`, and `RAEEN_MUTEX_SPIN=25`).
- `cargo test -p raeen-runtime` — 151 green (downstream consumer).
- `cargo clippy -p raeen-kernel -p raeen-hle --all-targets -- -D warnings`
  clean; `cargo fmt --all -- --check` clean.
- Smoke (not asserted in CI): the stress test finishes in ~0.00 s with the
  default spin vs ~0.07 s with `RAEEN_MUTEX_SPIN=0` on the dev box.

## Live A/B (run on the main session)

Same soak scenario (in-world streaming), two runs:

```powershell
# A: spin enabled (default budget 2000)
.\target\release\raeen.exe   # with RAEEN_TIME_HLE sinks as in soak-1785266485239

# B: baseline behavior
$env:RAEEN_MUTEX_SPIN = "0"; .\target\release\raeen.exe
```

Compare: MAIN's time inside `scePthreadMutexLock` per 15 s window, worst
frozen-window length, and in-world FPS during streaming. Optional sweep:
`RAEEN_MUTEX_SPIN` in {500, 2000, 8000}.
