# Raeen — M1-E Guest Threads (real `scePthreadCreate`) Design

**Date:** 2026-07-14
**Status:** Draft (design) — **emulator-reviewer audit complete (§6); NOT yet safe to implement as originally written.** Address C1–C3 + I1–I4 in §6, follow the §6 implementation order, per-slice reviewer sign-off.
**Scope of this spec:** Turning `scePthreadCreate`/`scePthreadJoin`/`scePthreadExit` from HLE no-ops into a **real second guest execution context** — a genuinely concurrent guest thread that runs its entry function on the shared `GuestArena`. This is M1 wall **E** (the last synthetic-vs-real execution gap) and the enabling substrate for the reference-port ledger's `todo` rows: SharpEmu/Kyty `pthread`, `Threads`, `Fiber`, and `AMPR` become real ports only once a second context can run.

---

## 1. Goal & Context

### 1.1 What is proven today

`crates/raeen-runtime` runs a **single** guest execution end-to-end: `execute_process` maps the module into a `GuestArena`, installs a main TCB (`fs:0` self-ptr + `fs:0x28` canary), lays a crt0 stack, and enters `_start` via `enter_guest_at_start`. HLE calls are serviced by a process-wide Vectored Exception Handler (VEH) that traps `HLE_TRAMPOLINE_BASE`-range faults. M1 (commit `38f603a`) proved this against real compiler-emitted code (crt0 + TLS + canary + printf/write).

### 1.2 The core invariant that blocks concurrency

The execution core is deliberately **single-active-execution** (`dispatch.rs`):

- **`CALL_LOCK`** (`dispatch.rs:99`) — a global `Mutex<()>` held for the entire `execute_process`/`execute_linked` pipeline. Guarantees "only one OS thread is ever *inside* a guarded guest call at a time."
- **`ACTIVE_CONTEXT`** (`dispatch.rs:194`) — a single global `AtomicPtr<ActiveContext>` the VEH reads **without a lock of its own**, relying on the single-active invariant for soundness. It carries the trampoline table + `HleContext` (kernel/mem/alloc) the VEH needs to service a trap.
- The VEH is process-wide; today it can assume the one faulting thread is the one active context.

A real second pthread means **two OS threads running guest code concurrently**, both faulting into the same VEH. The single global `ACTIVE_CONTEXT` and the whole-pipeline `CALL_LOCK` are the exact things that make that unsound today. This is the crux: M1-E is not a stub — it is surgery on the runtime's most safety-critical unsafe core.

### 1.3 Non-goals for this spec

- A real preemptive scheduler / priorities / affinity. Host OS threads do the scheduling; we honor create/join/exit/detach semantics, not PS5 scheduler fidelity.
- `scePthreadCancel` async cancellation, per-thread signal masks, thread-priority inheritance on mutexes. Deferred.
- Fiber `Run`/`Switch` (cooperative) — a follow-up once threads land; different mechanism (context swap on one OS thread), same TCB/stack machinery.

---

## 2. Design

### 2.1 Per-thread context (the central change)

Replace the single global `ACTIVE_CONTEXT` with **per-OS-thread** context so each host thread's VEH lookup finds *its own* guest thread's trampolines + `HleContext`:

- Make `ACTIVE_CONTEXT` a `thread_local!` (each OS thread running guest code registers its own `ActiveContext` for the duration of its guarded call; the VEH reads the *current thread's* TLS slot).
- The VEH stays process-wide (`AddVectoredExceptionHandler` is per-process) but dispatches using the faulting thread's TLS context. A fault on a thread with no active context → `EXCEPTION_CONTINUE_SEARCH` (unchanged fall-through).
- `fs` base is already per-OS-thread (a hardware register set via `WRFSBASE`); each guest thread sets its own `fs` to its own TCB, so TLS/`fs:0x28` canary are naturally per-thread with no extra work beyond installing a TCB per thread.

### 2.2 `CALL_LOCK` evolution

`CALL_LOCK` currently serializes *all* execution. For concurrency it must NOT serialize the guest-run phase. Options (to be decided in the plan, with emulator-reviewer):

- **(A) Drop the whole-pipeline lock; make shared mutable runtime state explicitly `Sync`.** The arena is shared read/exec + guest-managed writes; `OrbisKernel` is already `DashMap`/atomics. The `TrampolineGuard` / trampoline table are set up once per module and read-only during the run. Preferred if the audit shows no remaining single-writer assumption.
- **(B) Keep a lock only around setup/teardown** (trampoline reservation, context register/deregister), releasing it for the guest-run phase. Safer incremental step.

The plan must enumerate every `static`/global the VEH path touches and prove each is either per-thread or genuinely `Sync`.

### 2.3 Thread lifecycle (`raeen-runtime` + `raeen-kernel` + `raeen-hle`)

- **`spawn_guest_thread(entry, arg, stack_size, tls_template) -> GuestThreadHandle`** (new, `raeen-runtime`): allocate a fresh guest stack region + a per-thread TCB (reuse `GuestArena::setup_main_tcb`'s logic, generalized to allocate an *additional* TCB/stack from the guest heap rather than the main one), spawn a host `std::thread` that: sets its `fs` base to the new TCB, registers its thread-local `ActiveContext`, enters the guest at `entry(arg)` via a variant of `enter_guest_at_start` that passes one arg in `rdi` and provides a return trampoline (so the entry can `ret` into a controlled thread-exit path instead of faulting), captures the return value, deregisters, and stores the result in the handle.
- **`OrbisKernel` thread table** (extend the existing thread manager): map guest pthread handle → `GuestThreadHandle` (join handle + result slot + detach flag).
- **HLE (`pthread_thread.rs`)**: `scePthreadCreate` writes a new handle, calls `spawn_guest_thread`; `scePthreadJoin` joins the host thread and writes the retval; `scePthreadExit` unwinds the current guest thread to its return trampoline; `scePthreadDetach` marks detached (no join needed). The existing `pthread_sync`/`pthread_attr`/`pthread_tls` ports already provide the mutex/attr/TLS surface these threads will use.

### 2.4 Entry/exit ABI for a thread

Unlike `_start` (reads argc off the stack, exits via HLE `exit`), a pthread entry is `void* entry(void* arg)`: `arg` in `rdi`, returns a `void*` in `rax`. So the thread needs a distinct entry path: push a **return trampoline** address as the return address, `mov rdi, arg`, `jmp entry`; when the entry `ret`s, control lands on the return trampoline (an `HLE_TRAMPOLINE_BASE`-range address) which the VEH recognizes as "thread returned" and captures `rax` as the retval — symmetric with how `exit` is trapped, but per-thread.

---

## 3. Acceptance (M1-E done)

A **compiler-built** (rustc, the same fixture recipe as the M1 test) homebrew that:

1. Calls `scePthreadCreate` with an entry that writes an observable value (e.g. a magic into a guest-heap cell, or a distinct printf line) — proving a *second* context actually ran.
2. `scePthreadJoin`s it and reads back the retval.
3. In-tree acceptance test asserts the worker's side effect + the joined retval, byte-exact — the same falsification-verified discipline as the M1 test (`launcher.rs::compiler_built_homebrew_runs_through_shell_and_prints`).

The worker must run on a genuinely separate OS thread with its own TCB (assert the `fs:0x28` canary and a `#[thread_local]` in the worker resolve independently of the main thread's).

---

## 4. Risks / review gates

- **Unsafe soundness** — per-thread `ACTIVE_CONTEXT` + concurrent VEH is the delicate part; requires `emulator-reviewer` sign-off on the `Send`/`Sync` erasure (`dispatch::run` currently `transmute`s `&dyn GuestMemory` to `'static` under the single-active invariant — that reasoning must be re-derived for the multi-thread case).
- **Arena aliasing** — two threads writing the shared arena is the guest's own responsibility (real hardware behavior), but the *host* wrappers (`GuestMemory::read/write`) must stay data-race-free at the Rust level (they operate on `*mut u8` into a shared `VirtualAlloc` region — need to confirm this is sound under concurrent access, likely via raw-pointer ops with no `&mut` aliasing).
- **Windows-only** — same `#[cfg(target_os = "windows")]` boundary; the POSIX backend is unaffected (still `MapFailed`).
- **Scope creep** — resist building a real scheduler. Minimum: create/join/exit/detach with real execution. Everything else defers.

---

## 5. Relationship to the reference-port `/goal`

M1-E is the substrate that unblocks the ledger's remaining non-graphics `todo`/`wip` rows: SharpEmu `pthread`/`Fiber`/`AMPR` and Kyty `Threads` can only be *real* ports (not init-only stubs — the flagged "no-op instead of real execution" anti-pattern) once a second guest context runs. It does not touch graphics (owned separately). Finishing M1-E + graphics + the fixture-gated loader is what moves both reference trees toward `fully_ported`.

---

## 6. Soundness audit findings (emulator-reviewer, 2026-07-14)

A pre-implementation adversarial soundness audit against the actual runtime
internals found the §1–§4 design **directionally correct but not safe to
implement as written**. The per-thread `ACTIVE_CONTEXT` / `recovery_ctx` /
`RtlCaptureContext` reasoning is confirmed sound (each `run()` frame already owns
its survive-restore `Cell`s; making the global a `thread_local!` is the right
move, and `OrbisKernel` is already `Sync`). But these must be fixed first:

### Critical (block any code)

- **C1 — `HOST_RSP_SLOT` is a single process-wide slot** (`stack.rs:27-37`), used by
  both `call_on_guest_stack` and `enter_guest_at_start` via a RIP-relative
  `mov [rip+slot], rsp` / `mov rsp, [rip+slot]`. Its `unsafe impl Sync` is
  justified *only* by `CALL_LOCK` serialization. Two concurrent guest threads
  clobber each other's saved **host** RSP → host-stack corruption on an otherwise
  "successful" call. The original spec missed this because it lives in the *entry*
  path, not the VEH path. Must become per-thread (a `thread_local!` slot or an
  `enter`-provided per-thread pointer) **while preserving** the "no GP register
  survives the guest `call`" property that `guest_clobbering_r15_does_not_corrupt_host_rsp`
  (`execute.rs:1139`) guards.
- **C2 — the `'static` transmute needs shared *ownership*, not just `Sync`.**
  `dispatch.rs:269-281` erases `&dyn GuestMemory`/`GuestAllocator` to `'static`
  on the single-active + `CALL_LOCK` argument. A worker `std::thread` borrows the
  arena/kernel/hle/trampolines owned by `execute_process`'s stack frame; a
  **detached** worker outlives that frame → **use-after-free** of `GuestArena`
  (`VirtualFree`), `OrbisKernel`, and `TrampolineGuard` on parent unwind. `Sync`
  fixes aliasing, not lifetime. Shared runtime state must move to `Arc<…>`
  (arena / kernel / hle / trampoline table / guard) with workers holding owning
  clones — join-before-teardown can't cover detached threads, so `Arc` is
  effectively mandatory.
- **C3 — process-exit teardown with live detached workers is unhandled.** When the
  main thread `exit`s, `execute_process` returns and removes the VEH + unmaps the
  trampoline guard region; a still-running detached worker's next import call
  faults into a removed VEH → real crash. Must force-terminate live workers before
  teardown (and/or keep the VEH process-lifetime, see I2).

### Important

- **I1 — arena `read`/`write` SAFETY comments become false** (`arena.rs:388-415`
  assert "no concurrent writer under single-active"). Concurrent host-side
  `copy_nonoverlapping` on shared bytes is a data race (UB at the Rust abstract-
  machine level). Rewrite the comments to the honest weaker justification (host
  wrappers inherit the guest's own data-race responsibility; non-atomic, may tear)
  and record it as a deliberate deviation, not "sound". `alloc`/`free` (Mutex-guarded) are fine.
- **I2 — register the VEH once per process** (not per `run()`), removed only at
  final teardown — avoids N duplicate handlers under N workers and resolves C3's
  "handler removed while worker running".
- **I3 — Windows FS-base persistence across preemption is unverified and
  load-bearing.** `WRFSBASE` survival across context switches was only validated
  for a non-preempted main run. Real workers *will* be preempted; if the scheduler
  doesn't save/restore the user-set FS-base MSR, TLS + `fs:0x28` canary silently
  break after any switch (a heisenbug). **Spike this first** (set fsbase, force a
  switch under contention, re-read `RDFSBASE`) before relying on TLS in workers.
- **I4 — thread-local read inside the VEH:** use the `thread_local! { static X = const { Cell::new(null) } }`
  form (no lazy `Once`, no `Drop`) so a first-touch inside the handler (an
  unrelated AV on a thread that never ran guest code) is a cheap non-allocating
  read; and write the thread-local ctx in `run()` *before* entering guest code so
  the VEH is only ever a reader on faulting guest threads.

### Revised implementation order (each slice keeps the named regression tests green)

1. `ACTIVE_CONTEXT` → `thread_local!` (const-init null), `CALL_LOCK` unchanged, still single-threaded.
2. `HOST_RSP_SLOT` → per-thread (C1); hold `guest_clobbering_r15_does_not_corrupt_host_rsp` green.
3. VEH → once-per-process (I2); shared state → `Arc` (C2); keep `CALL_LOCK` for the whole run.
4. Run the FS-base-across-preemption spike (I3) — gate before any worker TLS reliance.
5. Add `spawn_guest_thread` but spawn-then-immediately-join (no real concurrency yet) to exercise the `arg`-in-`rdi` entry + return-trampoline retval capture in isolation.
6. Only then release `CALL_LOCK` around the guest-run phase (keep it for setup/teardown — **Option B**; the fixed-base `GuestArena`@`GUEST_ARENA_BASE` / `TrampolineGuard`@`HLE_TRAMPOLINE_BASE` singletons *require* serialized reservation, also for the parallel test harness, so Option A "drop the lock" is not reachable). Add the C3 detached-thread teardown story before enabling `scePthreadDetach`.

**Regression guards to hold green** (`crates/raeen-runtime/tests/execute.rs`):
`guest_clobbering_r15_does_not_corrupt_host_rsp` (1139),
`host_fsbase_is_restored_after_execute_linked_returns` (1369),
`host_fsbase_is_restored_after_a_recovered_genuine_fault` (1420),
`execute_process_restores_host_fsbase_after_an_exit_longjmp` (1704),
`genuine_wild_fault_recovers_as_faulted_then_process_keeps_running` (198),
`start_stub_wild_fault_still_recovers_as_faulted_through_execute_process` (1664),
`tls_variable_read_through_linker_computed_tpoff64_round_trips_tdata` (1772),
`stack_chk_guard_canary_at_fs_0x28_is_nonzero_with_terminator_byte` (1888),
`printf_with_guest_format_string_lands_in_the_kernel_console` (1953), plus all
`arena.rs` fixed-base serialization tests.

### Acceptance addendum (supersedes §3's minimum)

The §3 test is an honest gate **only if** it also (a) forces a preemption in the
worker (exercises I3 — e.g. contended yield/sleep before the canary check) and
(b) uses a genuinely **detached**-thread variant (exercises C2/C3). Without both,
it is a synthetic best-case that passes over the exact traps above.

---

## 7. Decisions after audit (implementation)

### I3 spike RESULT — FS base does NOT survive Windows preemption (fixed)

The I3 spike **failed** and exposed a **live bug**, not just an M1-E blocker.
Measured (`tls::tests::fsbase_does_not_survive_preemption_on_windows`): a user
`WRFSBASE` value reads back correctly immediately and across `yield_now`, but is
`0` after a `sleep` **and** after a pure user-mode busy-wait (no syscall). So a
bare timer-interrupt context switch clears it — Windows restores the FS base
from its own notion (0 for native x64 threads). `CR4.FSGSBASE` only makes the
instruction legal; it does not make the value survive scheduling. The original
RT2c-b spike only tested the VEH/`RtlCaptureContext` round trip, never
preemption.

**Consequence (was already breaking the single-threaded path):** a raw
`WRFSBASE` guest TCB is valid only until the next quantum (~15 ms); the guest's
next `fs:`-relative access (TLS or the `fs:0x28` canary) then reads a near-null
address → outside the guard region → VEH calls it a genuine wild fault → title
reports `Faulted`. M1's test passes only because that guest finishes in µs.

**Fix (landed):** trap-and-re-arm in `veh_callback`. The fault *is* the
notification Windows won't give us: an out-of-region AV whose thread has
`tls_active` and whose current FS base ≠ the guest TCB is re-armed with
`WRFSBASE` and the instruction retried (`EXCEPTION_CONTINUE_EXECUTION`).
Terminating: the retry runs with the correct base, so a genuine fault falls
through on the next pass. `ActiveContext` gained `guest_fsbase`. Guarded by
`execute.rs::guest_tls_survives_preemption_via_fsbase_rearm` (falsified: with
the re-arm disabled it faults at `GUEST_ARENA_BASE + 15`, the exact `fs:[0]`
instruction offset).

### C1 decision — DELETE `HOST_RSP_SLOT` (return-trampoline recovery)

Adversarial judge panel (5 independent designs × 3 refutation lenses →
synthesis) chose the `open` approach with **0 fatal attacks**; 3 of 5 proposers
independently converged on it. Do **not** make the slot per-thread — **delete
it**. Recover the host RSP wholesale from `run`'s `recovery_ctx` snapshot via the
return-trampoline longjmp `dispatch.rs` already performs for `exit`/RT1a.

- **stack.rs:** replace `call_on_guest_stack` + `enter_guest_at_start` with one
  diverging `enter_guest(entry, guest_rsp, args) -> !` (`mov rsp; jmp entry`,
  `options(noreturn)`). No save slot, no `unsafe impl Sync`, no restore line,
  nothing to race. Function/process/thread mode differ only in what the *caller*
  puts on the guest stack.
- **trampoline.rs:** reserve one more guarded slot; `return_tramp()` = index
  `count+1` (past the `UnresolvedTrampoline` sentinel at `count`, so it can't
  shadow that diagnostic — the one serious attack found). `logical_len` grows to
  `count*8 + 16`.
- **dispatch.rs:** `ActiveContext` gains `return_tramp: u64`, `returned:
  Cell<bool>`, `retval: Cell<u64>`; `veh_callback` gets an arm for `fault_addr
  == ctx.return_tramp` (capture `rax`, longjmp like exit/RT1a); `run`'s
  `call_guest` becomes `-> !` and the tail reads `retval` from ctx. Precedence:
  **exited → error → returned** (preserves `UnresolvedTrampoline` outranking a
  return).
- **Property:** the *restore* is STRENGTHENED (whole register file recovered, not
  just RSP — the r15 test passes for a stronger reason). The *reachability* is
  narrowly weakened (the guest `ret` must fault, needing OS frame room at guest
  RSP) but that dependency already exists for every HLE call / exit / RT1a fault;
  on the hot path RSP = `stack_top` with GiB of room. Optional closure: a 64 KiB
  `PAGE_NOACCESS` red-zone below `GUEST_ARENA_BASE`.
- **Grafts:** `#[repr(C, align(16))]` wrapper for `recovery_ctx` (VERIFY
  `align_of::<CONTEXT>()` first — `RtlCaptureContext` uses `movaps`, `#GP` on
  misalignment; now load-bearing as the sole recovery path); a `recovery_armed`
  gate so a fault in the `AddVEH → RtlCaptureContext` window can't longjmp to a
  half-built context.
- **Spikes before relying:** `align_of::<CONTEXT>()`; `noreturn` + SysV `in(reg)`
  operands compile on 1.97; return-tramp *fetch* AV delivers to the VEH like the
  existing `call [import_slot]` fetch fault. **I2 (register VEH once per process)
  becomes MANDATORY at step 3/6** — routing clean returns through the VEH means
  it must be live for the whole run, and N concurrent workers must not stack N
  handlers.
- **New tests:** `guest_return_recovers_host_context_through_trampoline` (clobber
  full callee-saved set + stack, `ret` a sentinel, then a second run on the same
  thread succeeds); `concurrent_guest_returns_recover_independent_host_contexts`
  (N threads, thread-distinct values, no cross-corruption) — gated at step 6.
