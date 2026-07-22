# Raeen crt0 / Process Environment (Wall #1) — Design Spec

**Date:** 2026-07-13
**Status:** Design (pending implementation)
**Scope:** Let the runtime enter a real ELF executable at its `_start`/crt0 entry: lay out the initial **process stack** (`argc`/`argv`/`envp`/`auxv`) the System V x86-64 process-startup ABI requires, enter the guest as `_start` (not a bare function), and make **`exit()`** actually end guest execution and hand the exit code back to the runtime. This is "wall #1" from the homebrew gap analysis — the first thing a real compiler-produced binary hits (its very first instruction reads `[rsp]` as `argc`).
**Builds on:** RT2c guest stack (`call_on_guest_stack`, the arena stack region) and the VEH + `RtlCaptureContext` fault-recovery machinery in `dispatch.rs`.

---

## 1. The problem

`execute_linked` calls the module entry as `extern "sysv64" fn(u64×6) -> u64` — a **function call**: args in registers, a return value expected. A real ELF executable's entry is `_start`, which instead:

- Reads the **process stack** the kernel/loader set up: at entry `rsp` points at `argc`, followed by `argv[0..argc]`, a NULL, `envp[..]`, a NULL, then the **auxiliary vector** (`(type,value)` pairs) terminated by `AT_NULL`; `rsp` is 16-byte aligned at the `_start` instruction.
- Sets up the C runtime and calls `main`, then calls **`exit(main_ret)`** — `_start` **never returns** to its caller.

So a real binary faults immediately: it reads `[rsp]` expecting `argc` and gets whatever the register-args call left there. And even with a stack, without `exit()` terminating execution the runtime never regains control.

## 2. Process stack layout

A new `crates/raeen-runtime/src/process.rs` builds the initial stack in the arena's stack region (top-down), returning the `rsp` value `_start` must see:

```
(low addresses)                                        (high = stack top)
 ┌───────── rsp (16-aligned, _start sees this) ─────────┐
 argc │ argv[0] … argv[argc-1] │ NULL │ envp[0] … │ NULL │
 auxv: (AT_PAGESZ, 0x4000) (AT_NULL, 0) │ … padding … │ arg/env strings │
```

- `build_process_stack(stack_top, argv: &[&str], envp: &[&str], mem: &dyn GuestMemory) -> Result<u64, RuntimeError>`: write the argv/envp **strings** near the top first (NUL-terminated), record each string's guest address, then write the pointer arrays + `argc` + auxv below them, all through `mem` (bounds-checked). Compute the final `rsp` so that after the pointer table is placed, `rsp % 16 == 0` (the `_start` ABI — the pointer table's base must land 16-aligned). Everything lives inside the committed stack region; overflow → `MapFailed`.
- **Minimal auxv:** `AT_PAGESZ (6) = 0x4000`, `AT_NULL (0) = 0`. (Real binaries also read `AT_PHDR`/`AT_PHNUM`/`AT_ENTRY`/`AT_RANDOM`; start minimal, extend when a real binary demonstrably needs one — `AT_RANDOM` in particular may be needed for glibc-style stack canary init, noted as a follow-up.)
- `argv[0]` defaults to the module name (e.g. `/app/eboot.bin`); `envp` empty for now.

## 3. Entry mode: `_start` vs. function

`execute_linked` stays the low-level **function-mode** primitive (register args, returns RAX) — the synthetic tests and simple stubs use it unchanged. A new **process-mode** entry point runs `_start`:

```rust
pub enum RunOutcome { Returned(u64), Exited(u64) }   // run() now yields which happened

pub fn execute_process(
    module: &LinkedModule, hle: &HleRegistry, kernel: &OrbisKernel,
    argv: &[&str], envp: &[&str],
) -> Result<RunOutcome, RuntimeError>;
```

- `execute_process` builds the arena (as `execute_linked` does), sets up TLS, builds the process stack (§2), and invokes the guest with **`rsp` = the process-stack pointer** and control transferred to `module.entry` such that the guest sees `rsp` pointing at `argc` (no extra pushed return address — a `jmp`-style entry, not a `call`, since `_start` reads `[rsp]=argc`, and never returns).
- Because `_start` never returns, the ONLY way execution ends cleanly is `exit()` (§4). A `_start` that "returns" (pops `argc` as a return address) is a malformed program; it will fault and be caught by RT1a recovery as `Faulted` — acceptable.
- `call_on_guest_stack` (RT2c-a) gains a sibling `enter_guest_at` (or a mode flag) that sets `rsp` to the given process-stack pointer and enters at `entry` **without** pushing a return address, still saving/restoring the host `rsp` via the RIP-relative static so the host survives (the guest is expected to leave via the exit-longjmp, but a faulting/returning guest must still not corrupt the host). The host-RSP save/restore discipline from RT2c-a is preserved exactly.

## 4. `exit()` termination (the unwind)

`_start` ends the program with `exit`/`exit_group`/`_exit(code)`. These are HLE functions reached through the trampoline VEH. Terminating means unwinding from deep in guest code back to `run` — which is exactly what the RT1a `RtlCaptureContext` longjmp already does for faults. Reuse it:

- The runtime recognizes a small set of **terminating functions** by `(library, function)`: `libc::exit`, `libc::exit_group`, `libc::_exit` (and treat `libkernel::sceKernelExit` likewise if present).
- In `veh_callback`, when a resolved trampoline call targets a terminating function: read the exit code from `context.Rdi` (SysV arg 0), store it + an "exited" flag in `ActiveContext` (memory, reached via `ACTIVE_CONTEXT` — same returns-twice-safe discipline as `error`/`resumed`), then **overwrite the delivered context with `*recovery_ctx` and `EXCEPTION_CONTINUE_EXECUTION`** — i.e. the same longjmp the fault path uses, unwinding to `run`'s recovery point instead of servicing-and-resuming.
- `run` distinguishes the arrivals: on the resumed arrival it checks the exited flag first → returns `RunOutcome::Exited(code)`; a genuine fault still → `Err(Faulted)`; a normal function-mode return → `RunOutcome::Returned(rax)`.
- `run`'s signature becomes `-> Result<RunOutcome, RuntimeError>`; `execute_linked` maps `Returned(v) => Ok(v)`, `Exited(v) => Ok(v)` (function-mode callers just see the value; a function-mode stub that calls `exit` is unusual but harmless). `execute_process` returns the `RunOutcome` as-is.

The terminating-function recognition is the only new branch in `veh_callback`; the fault path and the normal HLE-servicing path are otherwise unchanged.

## 5. Shell wiring

`FirmwareLauncher::load` (which loads real `eboot.bin`s) switches from `execute_linked(entry, &[])` to `execute_process(module, hle, kernel, &["/app/eboot.bin"], &[])`, and maps the outcome to `SessionOutcome`: `Exited(code)` → a new `SessionOutcome::Exited { code, resolved, unresolved }` shown honestly ("Program exited with code N"); `Returned` / `Faulted` / unresolved as today. Synthetic-stub tests keep using `execute_linked`.

## 6. Milestones

- **W1a — process stack + `_start` entry.** `build_process_stack` + `enter_guest_at`; `execute_process` runs a `_start`-style stub that reads `argc`/`argv[0]` off the stack and returns/loops. **Acceptance:** a hand-assembled `_start` stub that reads `argc` from `[rsp]` (and `argv[0]`'s first byte) and yields it proves the stack layout is ABI-correct and the guest sees it.
- **W1b — `exit()` termination.** The exit-family longjmp. **Acceptance:** a `_start` stub that calls `exit(0x2A)` via its HLE trampoline makes `execute_process` return `RunOutcome::Exited(0x2A)`; the process survives; a normal fault still returns `Faulted`; existing `execute_linked` tests unchanged.
- **W1c — shell wiring.** `FirmwareLauncher` uses `execute_process`; `SessionOutcome::Exited`. End-to-end: the homebrew-shaped module (now via `_start` + `exit`) reports an honest exit.
- **Future:** richer auxv (`AT_PHDR`/`AT_ENTRY`/`AT_RANDOM`), real `envp`, `atexit`, multi-threaded exit semantics.

## 7. Verification

- W1a: `_start` reads `argc`/`argv[0]` off the process stack → observed correct (Windows).
- W1b: `exit(0x2A)` → `Exited(0x2A)`; fault → `Faulted`; the exit-longjmp restores host state (RSP, fsbase) via the existing shared continuation.
- W1c: `FirmwareLauncher` end-to-end reports `Exited`.
- Guardrail: `#![forbid(unsafe_op_in_unsafe_fn)]`; every asm/unsafe has a `SAFETY:` note; the new `enter_guest_at` preserves RT2c-a's host-RSP save/restore and RT2c-b's fsbase restore; clippy clean; no panics on guest input; no keys/firmware.

## 8. Global constraints

- Rust 2024, ≥1.85, GPL-2.0-only. No new external deps. Windows-first (`process`/`stack`/`dispatch` stay `#[cfg(target_os="windows")]`); `execute_linked`'s public signature unchanged; `execute_process` is the new process-mode entry; non-Windows stub returns `MapFailed`.
- The exit-longjmp reuses the existing `RtlCaptureContext`/`recovery_ctx` mechanism — it must not perturb the fault path or the trampoline-servicing path; the exited flag/code live in `ActiveContext` (returns-twice-safe), never a local carried across `RtlCaptureContext`.
- Clean-room/trust boundary unchanged: only LM1-pipeline images run; no keys, no firmware, no circumvention.
