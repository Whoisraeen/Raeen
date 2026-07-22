# Raeen Guest Stack + TLS (RT2c) — Design Spec

**Date:** 2026-07-13
**Status:** Design (pending plan)
**Scope:** Run the guest on its **own stack** (in the arena's stack region) via an RSP switch around the native call, and set up **thread-local storage** (`fsbase` → a guest TCB) so guest code that uses TLS or a stack-protector canary works. This is the step from "guest code runs on the host stack with no TLS" to "guest code runs in a proper thread environment."
**Builds on:** RT2 `GuestArena` (the stack region `[base+0x8000_0000, base+0xA000_0000)` is already reserved+committed), the RT0/RT1 VEH (`dispatch::run`, trap-and-emulate, `RtlCaptureContext` fault recovery).

---

## 1. The problem

Today `dispatch::run` calls the guest as `entry(args)` directly, so the guest executes on the **host thread's stack** and with **no `fsbase`**:

- **Stack:** the guest shares the host Rust stack. It works for the small stubs run so far, but a real function expects a large stack it owns at *guest* addresses, and mixing guest pushes with host frames is fragile. We already reserved a 512 MiB guest stack region; the guest should run on it.
- **TLS:** Orbis (FreeBSD-derived) userland accesses thread-local storage through the **FS segment base** (`mov rax, fs:[off]`) — the TCB pointer sits at `fs:[0]`, and compiler-emitted stack-protector code reads a canary at a fixed FS offset. With `fsbase` unset, any such access reads garbage or faults, so most non-trivial compiled C code can't run. On **Windows x86-64** the FS base is not set through a documented API (Windows uses GS for the TEB and leaves FS free), which makes this the hard part of RT2c.

## 2. Mechanism A — guest stack (RSP switch)

Before calling the guest, switch RSP to the top of the arena stack region; after it returns, restore the host RSP. A small `core::arch::asm!` trampoline (`call_on_guest_stack`) does this:

```
save host rsp  →  rsp = guest_stack_top (16-aligned)  →  call entry  →  restore host rsp  →  return rax
```

- **Alignment:** SysV requires `rsp % 16 == 0` *before* a `call` (so the callee sees `rsp ≡ 8 mod 16` after the pushed return address). `guest_stack_top` is 16-aligned; the asm `call` pushes the return address, satisfying the callee ABI. (This is a *function* call, not `_start`; argc/argv/envp/auxv crt0 setup is a later milestone.)
- **Saving host RSP across the guest call:** store it in a location that survives an arbitrary guest call. A callee-saved register (`r15`) is the standard idiom (a well-behaved SysV callee preserves it), but to be robust against an ABI-violating guest, **save host RSP to a fixed memory slot reachable without RSP** — e.g. a field in `ActiveContext` (already reached via the `ACTIVE_CONTEXT` static), written before the switch and read after. The asm marks the SysV caller-saved set clobbered via `clobber_abi("sysv64")`.
- **Guest-stack layout:** RSP starts at `base + STACK_OFFSET + STACK_SIZE` (top; the stack grows down toward `base + STACK_OFFSET`). Optionally push a sentinel/"fake return address" pointing at a guarded page so a guest that `ret`s past its entry faults deterministically rather than running wild — RT2c can start with a return into a small guarded trap and treat that as normal completion.

**Compatibility with the VEH + RT1a fault recovery (critical):** `RtlCaptureContext(&recovery_ctx)` is taken in `run` **before** the switch, so `recovery_ctx` holds the *host* RSP. On a genuine guest fault (on the guest stack now), `veh_callback` copies `recovery_ctx` over the fault context and resumes — restoring the host RSP and landing back at `run`'s recovery point, abandoning the guest stack frames entirely. This already does the right thing regardless of which stack the guest was on; the `resumed` `Cell` guard still routes the second arrival past the (asm) guest call. The HLE trampoline path is likewise unaffected: `veh_callback` reads the return address from `[Rsp]` (now the guest stack, valid committed arena memory) exactly as before.

## 3. Mechanism B — TLS / `fsbase`

Allocate a **TCB** (thread control block) plus the module's TLS block in the arena, and point `fsbase` at it so `fs:[0]` reads the TCB self-pointer and canary/TLS offsets resolve.

- **TLS image:** a real module carries a `PT_TLS` segment (init image + `memsz`/`align`). RT2c-b sets up a single main-thread TLS block: allocate `tls_memsz` (arena heap/mmap), copy the init image, zero the `.tbss` tail. The TCB is a small block whose `self` pointer at `fs:[0]` points to itself (the FreeBSD/Orbis "variant II" TLS layout places the TCB just above the TLS block, with `fs:[0]` = TCB address). Exact TCB layout follows the Orbis/FreeBSD variant-II convention (TLS block below the TCB; `fs:[0]` self-pointer; the stack canary at its ABI offset).
- **Setting `fsbase` on Windows — the hard part.** Primary approach: the **`WRFSBASE` instruction** (FSGSBASE ISA extension). Windows 10 1709+ enables `CR4.FSGSBASE`, permitting user-mode `RD/WRFSBASE`; Windows saves/restores the FS base across context switches, and 64-bit Windows itself does not use FS, so writing it is safe (unlike GS = TEB). `run` executes `WRFSBASE guest_tcb` immediately before the guest call and restores the prior FS base after.
- **Spike-gated.** Because `WRFSBASE`'s reliability *within our VEH/exception model* is not certain (does the FS base survive exception dispatch and the `RtlCaptureContext` recovery?), RT2c-b begins with a **spike**: confirm on this machine that (a) `CPUID` reports FSGSBASE, (b) user-mode `WRFSBASE`/`RDFSBASE` are permitted (don't `#UD`), and (c) a value written with `WRFSBASE` is readable by a guest `mov rax, fs:[0]` executed through `execute_linked`, and survives a trampoline HLE call. The spike's result decides the design:
  - **If `WRFSBASE` works:** implement the TCB + `WRFSBASE` set/restore in `run`.
  - **If it doesn't:** document TLS as unsupported for now (honest limitation) and defer to a fallback milestone (candidates: trap-and-emulate FS-prefixed accesses via the VEH, or rewriting/`fs`-prefix patching — both larger). Do **not** ship a fragile half-working `fsbase`.

**Status: spike ran clean (Recommendation A), and RT2c-b is now implemented.** All four spike questions came back positive on this machine (CPUID reports FSGSBASE; user-mode `RDFSBASE`/`WRFSBASE` round-trip with no `#UD`; a guest `fs:`-prefixed load reflects the `WRFSBASE`'d value; the FS base survives the VEH + `RtlCaptureContext` fault-recovery round trip, 20/20 clean runs across debug/release — the x64 `CONTEXT` structure has no FS-base field, so nothing in that mechanism has a slot to reset it through). `crates/raeen-runtime/src/tls.rs` implements `fsgsbase_available()` (cached `CPUID.(7,0):EBX[0]` check) and `read_fsbase`/`write_fsbase` via raw instruction bytes (`.byte` — not the `rdfsbase`/`wrfsbase` mnemonics, to avoid a crate-wide `-C target-feature=+fsgsbase` rustflag); `GuestArena::setup_main_tcb` carves a minimal TCB (self-pointer at offset 0) from the heap allocator; `dispatch::run` takes an `Option<u64>` `tcb` parameter and threads the original FS base through `ActiveContext::orig_fsbase` (a `Cell<u64>`, not a plain local — required by the spike's "returns twice" finding, since a local carried across `RtlCaptureContext` is not reliably restored on the fault-recovery arrival), setting `WRFSBASE` immediately before the guest call and restoring it in the shared continuation on both the normal-return and RT1a-fault-recovery paths. Full per-module `PT_TLS` init-image loading is **not** implemented — `setup_main_tcb` only provides the self-pointer plus headroom for small TLS-offset probes — and remains a follow-up (RT2c+).

## 4. API / component changes

- `crates/raeen-runtime/src/stack.rs` (new): `call_on_guest_stack(entry, args, guest_rsp) -> u64` — the asm RSP-switch trampoline, isolated with a thorough `SAFETY:` note; and the guest-stack-top computation from the arena.
- `crates/raeen-runtime/src/tls.rs` (new, RT2c-b): TCB/TLS-block setup in the arena + the `WRFSBASE` set/restore helpers (guarded by an FSGSBASE availability check).
- `crates/raeen-runtime/src/dispatch.rs`: `ActiveContext` gains a `host_rsp` slot (for the robust memory-based RSP save); `run` computes the guest stack top, optionally sets `fsbase`, and calls `call_on_guest_stack` instead of `entry(...)` directly — the `resumed`/recovery structure is otherwise unchanged.
- `crates/raeen-runtime/src/arena.rs`: expose the stack region bounds (`stack_top()`), and (RT2c-b) a TLS allocation helper if the heap allocator isn't reused.
- `execute_linked`'s public signature is unchanged.

## 5. Milestones

- **RT2c-a — Guest stack (RSP switch).** `call_on_guest_stack` runs the guest on the arena stack region. **Acceptance:** (1) during a trampoline HLE call, the observed guest `RSP` is inside `[base+STACK_OFFSET, base+STACK_OFFSET+STACK_SIZE)` (a test HLE function records `RSP` — reachable via the VEH context — and the test asserts the range); (2) a guest function that genuinely uses the stack (e.g. recursion or a large local array written then read back) returns the correct result; (3) the RT1a genuine-fault recovery still returns `Faulted` and the process survives; (4) all existing execute tests pass.
- **RT2c-b — TLS / `fsbase`. Done.** Spike (§3) confirmed `WRFSBASE` viable; TCB + `WRFSBASE` set/restore is implemented in `dispatch::run`/`tls.rs`/`arena.rs::setup_main_tcb`. **Acceptance (met):** a guest function that reads `fs:[0]` returns the TCB pointer we installed (`guest_fs_zero_load_reads_the_installed_tcb`), and a value stored at a TLS offset round-trips (`guest_fs_offset_round_trip_writes_and_reads_back`) — both run through `execute_linked` on Windows. Additional tests confirm the host FS base is restored after both an ordinary return and a recovered RT1a fault. Per-module `PT_TLS` init-image loading remains a follow-up (RT2c+).
- **RT2c+ (future):** per-thread stacks/TLS for guest threads (`scePthreadCreate` executing real code), full `_start`/crt0 (argc/argv/auxv), guard-page stack overflow detection, POSIX backend.

> **Note (RT2c-a review):** the guest stack region currently abuts the mmap region directly (`stack_top == base + MMAP_OFFSET`), with **no guard page** between them. A guest stack overflow therefore silently grows into the adjacent guest mmap region instead of faulting — contained within guest memory (no host impact), but it means the RT1a fault-recovery net does not catch guest stack overflow. Adding a `PAGE_NOACCESS` guard page below the stack region (so overflow faults and is recovered as `Faulted`) is a small follow-up folded into the guard-page item above.

## 6. Verification

- **RT2c-a (`cargo test`, Windows):** the RSP-in-stack-region assertion via a test HLE function; a recursion/large-local guest stub returning a known value; the existing fault-recovery test unchanged; existing execute tests green.
- **RT2c-b:** the spike test (FSGSBASE cpuid + `WRFSBASE`/`RDFSBASE` round-trip + guest `fs:[0]` read through `execute_linked`); then the TLS round-trip if implemented.
- **Guardrail:** `#![forbid(unsafe_op_in_unsafe_fn)]`; every `unsafe`/asm block carries a `SAFETY:` note stating the ABI and register/stack contract; clippy clean; no keys/firmware; no panics on guest input.

## 7. Global constraints

- Rust edition 2024, rust-version ≥ 1.85, GPL-2.0-only. **No new external dependencies** (`core::arch::asm!` and `windows-sys` cover it).
- Windows-first; `stack`/`tls`/`dispatch` stay `#[cfg(target_os = "windows")]`; `execute_linked`'s public signature unchanged and platform-independent (non-Windows stub still returns `MapFailed`).
- The asm RSP switch is the most delicate `unsafe` in the project — it must be robust against a guest that does not perfectly honor the SysV ABI (save host RSP to memory, not only a callee-saved register), and must not break the existing VEH trampoline path or RT1a recovery. Review it adversarially.
- Clean-room/trust boundary unchanged: only LM1-pipeline images run; no keys, no firmware, no circumvention.
- TLS honesty: ship a real `fsbase` only if the spike proves it works in our model; otherwise document the limitation rather than a fragile approximation.
