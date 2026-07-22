# Raeen Execution Runtime — Design Spec

**Date:** 2026-07-13
**Status:** Design (pending plan)
**Scope:** The subsystem that takes a **linked module** (`raeen_firmware::LinkedModule`) and actually **executes** its code, dispatching each HLE import call to the `HleRegistry`. This is the step from "a homebrew `.sprx` *links*" (LM1) to "its code *runs*."
**Builds on:** LM1 firmware spine (`load_module` → `LinkedModule { image, base, hle_trampolines, unresolved }`), `raeen-hle::HleRegistry`, and the SM3 `FirmwareLauncher` seam in the shell.

---

## 1. Goal & non-goals

**Goal (the runtime track):** run PS5 userland module code natively and service its library calls through Raeen's HLE. The PS5 CPU is x86-64 (Zen 2) and the host (Windows) is x86-64, so module code executes **natively** — no CPU interpreter or JIT (per `implementation_plan.md` and the firmware-spine design). The runtime's job is the *environment*: guest memory, the ABI boundary, and trapping the HLE trampoline addresses the linker planted so a guest call to an imported symbol becomes a call into a Rust HLE function.

**Non-goals (this spec):**
- Full Orbis process/thread model, real TLS/`fsbase` setup, signals, `syscall`-instruction emulation at scale — RT0 handles only what a trivial module exercises; broader coverage is later RT milestones.
- Executing encrypted retail modules (needs keys the user supplies; orthogonal to this mechanism).
- GNM→Vulkan / audio / input execution paths (those HLE functions already exist as stubs; the runtime just routes to them).

## 2. The core mechanism (why it's tractable)

Because guest and host share the ISA, the runtime does not context-switch a virtual CPU. It **calls the guest function directly** as a foreign function pointer and lets it run on the host thread, with a **Vectored Exception Handler (VEH)** armed to service HLE trampoline calls:

```
load_module → LinkedModule.image (import slots hold HLE_TRAMPOLINE_BASE + n*8)
      ↓ map image into guest memory at `base`
map trampoline region [HLE_TRAMPOLINE_BASE .. ) as PAGE_NOACCESS (a guard)
arm VEH
      ↓ call (base + entry_off) as `extern "sysv64" fn(...) -> u64`
guest code runs natively …
      ↓ guest `call [import_slot]` → jumps to a trampoline addr in the guard region
ACCESS_VIOLATION → VEH:
   • map faulting addr → (library, function) via LinkedModule.hle_trampolines
   • read SysV arg regs from the CONTEXT (RDI, RSI, RDX, RCX, R8, R9)
   • call HleRegistry.call(library, function, &args) → result
   • set CONTEXT.Rax = result;  pop return addr: Rip = *Rsp; Rsp += 8
   • EXCEPTION_CONTINUE_EXECUTION   (emulates the call returning)
      ↓ guest continues, eventually returns to the runtime
runtime returns the guest function's RAX
```

This is standard emulator practice (trap-and-emulate), needs no code generation, and keeps the boundary tiny: the only host↔guest contract is the SysV AMD64 calling convention.

## 3. ABI boundary

- **Guest convention:** SysV AMD64 (Orbis/FreeBSD-derived). Integer args in RDI, RSI, RDX, RCX, R8, R9; return in RAX.
- **HLE signature:** existing `HleFunction = fn(&[u64]) -> u64`. The VEH marshals the six SysV integer arg registers into a `&[u64; 6]` slice and writes the `u64` result to RAX. RT0 supports integer/pointer args only (no float/stack-spilled args yet — a documented limitation, extended when a real module needs it).
- The guest function Raeen calls at entry is declared `extern "sysv64"` so Rust passes RT0's test arguments in the convention the guest expects.

## 4. Components (`raeen-runtime`, new crate)

```
crates/raeen-runtime/
└── src/
    ├── lib.rs          # execute_linked(...) public entry
    ├── mem.rs          # guest memory mapping (VirtualAlloc) + protections + Drop cleanup
    ├── trampoline.rs    # the trampoline guard region + faulting-addr → (lib,fn) resolution
    └── dispatch.rs      # the VEH: CONTEXT marshalling + HleRegistry dispatch
```

- **`mem.rs`:** map `LinkedModule.image` at `base` into host memory. RT0: a single `PAGE_EXECUTE_READWRITE` region for simplicity (documented as a deliberate RT0 shortcut — real per-segment W^X protections are a later milestone). Owns the allocation; frees on `Drop`.
- **`trampoline.rs`:** reserve the `HLE_TRAMPOLINE_BASE` region as `PAGE_NOACCESS` (or a distinct sentinel range) so any call through an import slot faults deterministically; map a faulting address back to its `HleTrampoline { library, function }` by `(addr - HLE_TRAMPOLINE_BASE)/8` indexed into the module's trampoline table.
- **`dispatch.rs`:** the `AddVectoredExceptionHandler` callback. On an `EXCEPTION_ACCESS_VIOLATION` whose address is in the trampoline region, marshal + dispatch + resume as in §2. Any other exception → `EXCEPTION_CONTINUE_SEARCH` (don't swallow real crashes). Thread-safety: the handler needs the active module's trampoline table + `HleRegistry`; RT0 keeps a single active `ExecutionContext` in a guarded thread-local/`OnceLock` set up before the call and cleared after.
- **`lib.rs`:** `execute_linked(module, hle, entry_offset, args) -> Result<u64, RuntimeError>`.

## 5. Public API

```rust
pub fn execute_linked(
    module: &raeen_firmware::LinkedModule,
    hle: &raeen_hle::HleRegistry,
    entry_offset: u64,       // offset into module.image of the function to call
    args: &[u64],            // up to 6 integer/pointer args (SysV)
) -> Result<u64, RuntimeError>;
```
`RuntimeError`: `MapFailed`, `UnresolvedTrampoline(u64)` (a call hit a trampoline with no HLE mapping — surfaced, not silently ignored), `Faulted { addr }` (a genuine guest fault outside the trampoline region), `TooManyArgs`.

## 6. Safety & trust boundary

Running guest module code natively means executing code the user supplied. This is inherent to any native-ISA emulator and is bounded to modules the user chose to load (homebrew they built, or — later — modules they decrypted from hardware they own). RT0 documents this explicitly. The RWX shortcut and the VEH are the two `unsafe` surfaces; both are isolated in `mem.rs`/`dispatch.rs` with SAFETY comments, and the runtime never executes anything it did not map from a `LinkedModule` produced by the vetted LM1 pipeline. No keys, no firmware, no circumvention — unchanged from the firmware spine's boundary.

## 7. Milestones

- **RT0 — trap-and-dispatch a single call.** `execute_linked` runs a hand-assembled synthetic linked module whose entry function calls one HLE import (an HLE function registered to return a known sentinel) and returns its result; the VEH services the trampoline call. Acceptance: the runtime returns the sentinel, proving native execution + HLE dispatch + ABI marshalling compose. Also: a call to an unmapped trampoline → `UnresolvedTrampoline`; a genuine fault → `Faulted`, not a hang or silent pass. Windows-first.
- **RT1 — module init + multiple imports + a real homebrew.** Run a module's `module_start`, resolve several imports across HLE libraries, support pointer args into guest memory; wire `FirmwareLauncher` (SM3) to actually `execute_linked` a loaded module and report real execution state.
- **RT2+ — thread/TLS/`fsbase`, stack setup, `syscall` handling, W^X segment protections, cross-platform (the VEH abstraction gets a POSIX `sigaction`/`SIGSEGV` backend).**

## 8. Verification

- **RT0 automated (`cargo test`, Windows):** a test assembles minimal x86-64 bytes for `entry: call [rip+slot]; ret` with the slot pre-filled by the LM1 linker to a trampoline for a test-registered HLE function returning `0xC0DE`; `execute_linked` returns `0xC0DE`. Plus: unmapped-trampoline → `UnresolvedTrampoline`; deliberate wild write → `Faulted`.
- The synthetic module is built through the **real** LM1 `link_module` so the runtime consumes exactly what the spine produces (no bespoke slot values).
- **Guardrail:** the crate is `#![forbid(unsafe_op_in_unsafe_fn)]`-clean where practical; every `unsafe` block carries a SAFETY note; clippy clean.

## 9. Global constraints

- Rust edition 2024, rust-version ≥ 1.85, GPL-2.0-only. New crate `raeen-runtime` added to the workspace; depends on `raeen-firmware`, `raeen-hle`, `raeen-core`, and `windows`/`windows-sys` for VEH + `VirtualAlloc` (a new dependency — the first OS-API crate; scoped to this crate).
- RT0 is Windows-first (the user's platform); the exception-handling seam is written so a POSIX backend slots in at RT2 without touching callers.
- Clean-room boundary unchanged: no keys, no firmware, no circumvention; the runtime executes only modules produced by the LM1 pipeline from inputs the user supplied.
