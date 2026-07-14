# XPS5X — M1-E Guest Threads (real `scePthreadCreate`) Design

**Date:** 2026-07-14
**Status:** Draft (design), pending implementation plan + emulator-reviewer sign-off
**Scope of this spec:** Turning `scePthreadCreate`/`scePthreadJoin`/`scePthreadExit` from HLE no-ops into a **real second guest execution context** — a genuinely concurrent guest thread that runs its entry function on the shared `GuestArena`. This is M1 wall **E** (the last synthetic-vs-real execution gap) and the enabling substrate for the reference-port ledger's `todo` rows: SharpEmu/Kyty `pthread`, `Threads`, `Fiber`, and `AMPR` become real ports only once a second context can run.

---

## 1. Goal & Context

### 1.1 What is proven today

`crates/xps5x-runtime` runs a **single** guest execution end-to-end: `execute_process` maps the module into a `GuestArena`, installs a main TCB (`fs:0` self-ptr + `fs:0x28` canary), lays a crt0 stack, and enters `_start` via `enter_guest_at_start`. HLE calls are serviced by a process-wide Vectored Exception Handler (VEH) that traps `HLE_TRAMPOLINE_BASE`-range faults. M1 (commit `38f603a`) proved this against real compiler-emitted code (crt0 + TLS + canary + printf/write).

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

### 2.3 Thread lifecycle (`xps5x-runtime` + `xps5x-kernel` + `xps5x-hle`)

- **`spawn_guest_thread(entry, arg, stack_size, tls_template) -> GuestThreadHandle`** (new, `xps5x-runtime`): allocate a fresh guest stack region + a per-thread TCB (reuse `GuestArena::setup_main_tcb`'s logic, generalized to allocate an *additional* TCB/stack from the guest heap rather than the main one), spawn a host `std::thread` that: sets its `fs` base to the new TCB, registers its thread-local `ActiveContext`, enters the guest at `entry(arg)` via a variant of `enter_guest_at_start` that passes one arg in `rdi` and provides a return trampoline (so the entry can `ret` into a controlled thread-exit path instead of faulting), captures the return value, deregisters, and stores the result in the handle.
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
