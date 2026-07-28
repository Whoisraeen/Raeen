# VEH dispatch hardening — three defects

Date: 2026-07-28. Scope: `crates/raeen-runtime/src/dispatch.rs`,
`crates/raeen-runtime/src/thread.rs`, `crates/raeen-kernel/src/lib.rs`.

Motivating symptom: ASTRO.BOT ends its host process with exit code
`0xC0000409` (`STATUS_STACK_BUFFER_OVERRUN`). That code is what
`__fastfail(FAST_FAIL_FATAL_APP_EXIT)` raises, and the MSVC CRT's `abort()`
routes through `__fastfail` — so it is the ordinary Windows presentation of a
**Rust panic/abort inside a `nounwind` boundary**, which a vectored exception
handler is. A crash of that shape produces no Raeen fault report at all,
because every reporting path in the runtime runs *after* the VEH returns.

The three defects below were found by audit, not by reproducing the crash;
each is a real bug independent of ASTRO. Defect 1 is the only one that can
produce `0xC0000409`, and the reasoning is at the end.

---

## Defect 1 — `RefCell` borrowed across a faultable guest write, inside the VEH

`ActiveContext::callback_frames` was `RefCell<Vec<GuestCallbackFrame>>`. Every
other field of `ActiveContext` is a `Cell`, deliberately, because a borrow held
across the register-only context restore the genuine-fault path performs would
never be released.

The unwind sites did this:

```rust
for frame in ctx.callback_frames.borrow_mut().drain(..).rev() {
    if let Some(completion) = frame.completion {
        let _ = mem.atomic_store_u32(completion.address, completion.failure_u32);
    }
}
```

`completion.address` is a **guest-controlled** address (it comes from the HLE
handler that requested the callback, which took it from guest arguments). If
that store faults, Windows re-enters the VEH synchronously **on the same
thread**, with the `borrow_mut()` guard still alive on the abandoned stack
frame — and the re-entered handler reaches the same code. The inner
`borrow_mut()` then panics with `BorrowMutError`.

### Fix

`callback_frames` is now `Cell<Vec<GuestCallbackFrame>>`, with three helpers
that all take the vector **out of the cell before any guest memory is
touched**:

- `push_callback_frame` — take / push / set
- `pop_callback_frame` — take / pop / set
- `take_callback_frames_innermost_first` — takes the whole vector and reverses
  it, so the caller iterates owned frames with the cell already empty

Applied at all four sites (illegal-instruction guest fault ~L2266, callback
return ~L2360, generic guest fault ~L2591, callback push ~L3008) plus the new
host-fault path. A re-entrant handler now finds an empty vector instead of a
live borrow: worst case one completion word is not rolled back, versus a
process-killing panic. `Cell::take` leaves `Vec::new()`, which neither
allocates nor frees, so the VEH stays allocation-free on these paths.

### Related hazard, NOT fixed (recorded)

`GuestArena::atomic_store_u32` / `read` / `write` take the arena's
`std::sync::Mutex` state lock, and `write` holds it across the raw
`copy_nonoverlapping`. A fault raised *under* that lock re-enters the VEH,
which calls back into `mem.read`/`atomic_store_u32` → the same non-reentrant
mutex → **self-deadlock** (a hang, not a panic). This is pre-existing (the
generic fault arm already calls `mem.read` unconditionally) and out of scope
here; fixing it needs either a try-lock path for VEH-side accesses or a
lock-free committed-range probe. It is a plausible cause of an ASTRO *hang*,
not of `0xC0000409`.

---

## Defect 2 — the access-violation arm had no RIP classification

The illegal-instruction arm gates on `fault_addr >= GUEST_ARENA_BASE` before
claiming a fault as the guest's. The access-violation arm did not: **any** AV
delivered on a thread with an armed `ActiveContext` was recorded as
`RuntimeError::Faulted { addr: <rip> }` and long-jumped back into `run`. A null
dereference inside a Rust HLE handler therefore surfaced as
`guest fault at 0x7ff…` — an emulator bug wearing a title bug's clothes.

### The rules chosen

New pure function `classify_access_violation(kind, rip_is_guest_memory,
rip_is_host_owned, rip_is_stub, access_is_stub) -> AvOrigin`, evaluated **after**
the FS-base re-arm and demand-commit fast paths (both of which fix rather than
report, and must keep their first refusal) and before the unresolved-stub
reporting:

1. **`rip` or the access address resolves to an unresolved-import stub →
   `UnresolvedStub`.** Existing behaviour, untouched: both the CALLED shape
   (rip *is* the stub) and the READ shape (a relocation slot still pointing at
   one) must keep naming the missing NID, including the non-strict
   resume-with-`rax=0` inventory path. This is the hot path; it is checked first
   so a host verdict can never shadow it.
2. **`rip` reads back through `GuestMemory` → `Guest`.** The common case, and
   the one that must not move at all.
3. **The access type is `Execute` → `Guest`, always.** An instruction *fetch*
   failure is the signature of a guest wild jump (`jmp rax` into `argc`, a call
   through a null function pointer, a `ret` into data), and by definition its
   rip is not mapped executable. Host code cannot produce this shape, because
   it is running. `tests/execute.rs` observes `argc`/`argv[0][0]` exactly this
   way (`Faulted { addr: 1 }`, `{ addr: 0x2F }`), so this rule is load-bearing.
4. **A read/write whose `rip` the VMA map attributes to the host process →
   `Host`.** The instruction was fetched and executed, so its page *is* mapped
   executable, and the map says the page is the host's.
5. **Everything else → `Guest`.** Host attribution is a *positive* claim only:
   an address outside `[VMM_MIN, VMM_MAX)`, or an access type Windows reports as
   neither read/write/execute (the `#GP` shape, `ExceptionInformation[1] == -1`),
   stays on the pre-existing path.

"Host-owned" is answered by the address-space map, not by an address range:
new `GuestAllocator::address_is_host_owned` (default `false` = *unknown*, so
every test double keeps the old classification) is implemented by `GuestArena`
as `VmaType::Foreign` at that address — the kind whose whole purpose is "the
host process's DLLs, heap, thread stacks, and every range we never asked for".
A range check could not do this job: the guest legitimately maps memory far
*below* the arena base (ASTRO's libc mspace at 12 GiB), and the host image
loads far *above* it (measured `0x7ff6_…` > the 16 TiB arena base).

`RuntimeError::HostFaulted { rip, access, kind, hle }` is the new outcome.
Recovery is identical to a guest fault (longjmp to `run`'s recovery point, so
the Shell survives and the run reports) — only the label and the log differ.
`run` fills in `hle` from `active_hle` and logs `tracing::error!` "HOST FAULT:
… this is a Raeen bug, not a guest fault", because the VEH must not allocate.

Emulator-generated code (the executable trampoline slots/bridges at
`0x4000_0000_0000` and the TLS re-arm stub) is `Foreign` in the map and so also
classifies as `Host` — which is correct: a fault inside our own generated stub
is our bug. One residual ambiguity is documented on the variant rather than
hidden: guest code that jumps into an *executable* host page and then faults
also lands in `Host`. The reported rip plus the call trace separate the two by
inspection, and the previous behaviour (silently "guest fault") was strictly
worse.

### Tests

`dispatch.rs` unit tests pin the table (5 tests: guest read/write/execute,
host read/write, fetch-failure-is-guest even over host-owned space, both stub
shapes, unattributed/unknown-kind stays guest). `tests/execute.rs` adds
`an_access_violation_inside_an_hle_handler_is_reported_as_a_host_fault`, an
end-to-end run where a registered HLE handler dereferences a bad pointer and
the run must return `HostFaulted` naming `libtest::sceTestHostFault` — it fails
loudly with "laundered as a guest fault" if the classification regresses.

---

## Defect 3 — abandoned-lock recovery was incomplete

`thread.rs` called `release_locks_owned_by(handle)` **only when
`result.is_err()`**, on the theory that "a clean Returned/Exited already
unlocked what it held". That is false in both directions:

- `Ok(Returned)` is also how **`scePthreadExit`** ends a worker. A thread that
  exits from inside a critical section — the ordinary shape of a C++ worker that
  throws, catches, and exits — never reaches its unlock, and left the mutex
  owned forever. Every later `scePthreadMutexLock` on it blocks permanently
  (mutexes truly block now), which is the "stuck >3s — deadlock" cascade.
- `Ok(Exited)` is cooperative process termination: the VEH abandons the guest
  stack at the next safe point, as abruptly as a fault.

### Fix

Release now runs unconditionally on every worker exit path; the log line names
which path (`returned/pthread-exit`, `process-exit`, `faulted`). It is
idempotent and cheap: a thread holding nothing scans the maps, frees nothing,
and logs nothing (`LockReleaseSummary::any()` is false).

`LockReleaseSummary` gains `cond_waiters`, and `release_locks_owned_by` now
also drops the dying thread's **condition-variable wait entries**
(`PthreadCond::remove_waiters_of`). This is not a lock, but it is the same leak
with the same consequence: a waiter is dequeued *by the signaler*, so a dead
thread's entry stays queued, the next `scePthreadCondSignal` pops it, wakes
nobody, and reports success — one lost wakeup per abandoned waiter, and the
live waiter behind it never runs. Entries are **discarded, never signalled**;
signalling a dead waiter is precisely the bug.

### Semaphore ownership: not trackable today

Answering the question directly: **the kernel does not track semaphore
ownership, and I did not invent it.**

- `Semaphore { count, max_count }` (kernel `sceKernelCreateSema` family) —
  a count and a ceiling. No owner field, no per-thread record of who took a
  unit.
- `PosixSem { count, posted }` (`sem_*`) — same.
- `EventFlag { bits, attributes }` — global bits; no notion of a thread owning
  a bit.

Nothing here can be attributed to a dying thread, and inventing an owner would
be *wrong* for the dominant producer/consumer use, where the waiter is never
the party expected to post back. What it would take, recorded rather than
guessed at:

1. A per-`(thread, semaphore)` ledger of successful acquisitions, incremented
   in `sceKernelWaitSema`/`sem_wait` and decremented in the signal paths,
   mirroring the existing `pthread_rwlock_read_holds` map (which exists for the
   identical reason — a shared count cannot say *which* thread holds it).
2. A policy decision on release semantics, because "give back the dead
   thread's units" is only defensible for the mutex-shaped
   `initial == max == 1` usage; for a real counting semaphore it would
   fabricate units the producer never posted.
3. For event flags, a per-thread record of bits set, which has no meaningful
   inverse at all (bits are shared state, not a resource held).

Recommendation: do (1) only if a measured deadlock is traced to a
semaphore-as-mutex, and gate the release on `max_count == 1` so a genuine
counting semaphore is never fabricated into.

### Tests

- `raeen-kernel`:
  `release_locks_owned_by_drops_the_dead_threads_cond_waiters_without_signalling_them`
  — aliased cond (object address + handle) counted once, live waiter left
  queued and *unsignalled*, the next `signal_one` now reaching it, and
  idempotence on a second call.
- `raeen-runtime`:
  `a_worker_exiting_via_pthread_exit_releases_the_mutex_it_still_held` — a guest
  worker locks a mutex, calls `scePthreadExit`, is joined by the main thread,
  and the mutex must be unowned afterwards. Deterministic (a real join, no
  sleeps). Verified falsifiable: with the old `if result.is_err()` guard
  restored it fails with `owner: 2`.

---

## Could Defect 1 or 3 produce ASTRO's `0xC0000409`?

**Defect 1: yes — it is the only one of the three that can.** The chain is
mechanical: a nested guest callback with a completion word (the shape
`pthread_once`-style APIs and any HLE that hands a synchronization word to a
callback produce) + a guest fault while frames are live + a completion address
the guest has since corrupted or that names a page the arena refuses →
`atomic_store_u32` faults under the live `borrow_mut()` → re-entrant VEH →
`BorrowMutError` panic → panic across an `extern "system"` boundary → `abort`
→ `__fastfail` → `0xC0000409`, with no Raeen report, which matches the observed
"process just dies with 409 and no fault line". ASTRO is also the title that
faults *inside* callbacks often enough for the preconditions to be routine.
Nothing here proves it was the cause — the crash was not reproduced under a
debugger — but it is the one defect whose failure mode *is* this exit code.

**Defect 3: no.** Its failure mode is a hang: waiters parked forever on a
mutex or robbed of a wakeup. That is the measured "stuck >3s — deadlock"
signature, not a `__fastfail`. It cannot terminate the process at all.

**Defect 2: no, but it hid the evidence.** Laundering a host fault into
`Faulted { addr: <host rip> }` cannot crash anything — it *recovers*. Its cost
is that if ASTRO's host process was dying from an emulator-side bug in an HLE
handler, the report pointed at the guest, and every investigation started in
the wrong crate. There is also a latent danger the classification now removes
from the path: for a host-rip fault the FS-re-arm arm would have tried to stage
a return trampoline at `[host rsp - 16]` through `GuestMemory::write` — refused
today only because the host stack is `Foreign` and the arena declines it.

### Not addressed here

- The arena state-mutex re-entrancy hang described under Defect 1.
- `hle_call_time` (keyed `(thread, String)`) and other per-thread diagnostic
  maps are not pruned on thread death — an unbounded diagnostic leak on a title
  that churns threads, harmless to correctness.
- A guest thread's *own* fault report still says nothing about which host
  thread died; only the guest handle is logged.
