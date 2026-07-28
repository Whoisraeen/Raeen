# Two measured game-loading blockers

Date: 2026-07-28. Branch: `port-loading-blockers` (based on
`integration/sharpemu-sweep` @ `75c456e`).

Baseline: `artifacts/compat/post-sweep-validated-20260728.json`, live 9-title
run at build `2e4bdca`.

Scope: `crates/raeen-runtime/src/{dispatch.rs,lib.rs,thread.rs}`,
`crates/raeen-hle/src/{exception.rs,lib.rs,libkernel.rs,libc.rs}`,
`crates/raeen-kernel/src/{lib.rs,memory/mod.rs}`.

---

## (A) A Plague Tale Requiem — `0xC0000094`, silent host death

Measured: stage `crashed` at 40.8 s, exit `-1073741676` =
`0xC0000094` = `STATUS_INTEGER_DIVIDE_BY_ZERO`, 0 flips, 0 unresolved NIDs, and
**no ERROR line logged before it**.

### The premise was wrong, and that is the finding

The task hypothesis was that *our Rust host code* divided by a guest-supplied
zero. **That cannot produce this exit code.** rustc emits an explicit zero check
before every integer `div`/`idiv` — in debug *and* release, independent of
`-C overflow-checks`, because unchecked division is LLVM UB. A host-side Rust
divide-by-zero therefore surfaces as

```
thread '...' panicked at ...: attempt to divide by zero
```

a Rust panic with a message and a backtrace — not a Windows SEH integer-divide
exception. Verified empirically on this machine with `rustc -O -C panic=abort`
over a `black_box(0)` divisor: panic message, no `0xC0000094`.

Two facts in this tree corroborate that the `#DE` came from **guest-native code**
(or a C/C++ dependency), not from safe Rust:

1. **No unchecked division anywhere in the workspace.** No `unchecked_div`,
   `unchecked_rem`, or `core::intrinsics` division. Every `asm!`/`global_asm!`
   site in the tree is FSGSBASE / stack / trampoline
   (`raeen-runtime/src/tls.rs`, `stack.rs`, `dispatch.rs`,
   `raeen-gui/src/launcher.rs`) — none contains a `div`.
2. **The VEH did not handle `EXCEPTION_INT_DIVIDE_BY_ZERO` at all.** It
   dispatched on `EXCEPTION_ACCESS_VIOLATION`, `EXCEPTION_ILLEGAL_INSTRUCTION`
   and `EXCEPTION_BREAKPOINT`; everything else took
   `EXCEPTION_CONTINUE_SEARCH`. A search for `INT_DIVIDE_BY_ZERO` /
   `0xC0000094` across the whole tree returned zero hits. With no other handler
   installed, a guest `div` with a zero divisor propagated unhandled and killed
   the process — **exactly** the observed signature, including "no ERROR line
   before it". The crash was invisible *by construction*.

### The audit (honest negative)

Every non-comment integer `/`, `%`, `div_ceil`, `next_multiple_of`, `align_*`
site in `raeen-gpu/src`, `kyty-graphics/src`, `raeen-hle/src`, and
`raeen-kernel/src` was classified. Float division cannot trap and was excluded.

| Crate | Div/mod lines examined | Non-literal divisor | Unguarded & guest-derived |
|---|---|---|---|
| `raeen-gpu/src` | 150 | 20 | **0** |
| `kyty-graphics/src` | 69 | 2 | **0** |
| `raeen-hle/src` | ~190 | 8 | **0** |
| `raeen-kernel/src` | 63 | 0 | **0** |

Notable guest-derived divisors, all already guarded:

- `agc_exec.rs` `compose_presentable_to_scanout` — `desc.width`/`desc.height`
  come straight from the guest VideoOut descriptor and are the divisors at
  `:1764`/`:1768`. **Explicitly** refused at `:1716-1717`
  (`|| desc.width == 0 || desc.height == 0`) before either divide.
- `libsce_audio_out.rs` `decode_to_stereo` — guest channel count × format;
  doubly guarded (`.max(1)` plus `if frame_bytes == 0 { return }`).
- `libkernel.rs` `hle_available_direct_memory_size` / `hle_map_direct_memory` /
  `hle_reserve_virtual_range` — guest `alignment` normalized through
  `.max(PS5_PAGE_SIZE)` or an explicit power-of-two check before use.
- `libsce_libc_internal.rs` `sceLibcMspaceMemalign` — guest alignment gated by
  `is_power_of_two()`, which rejects 0.
- `texture/tiling.rs` block dimensions — `block_dimensions` returns
  `(1 << w, 1 << h)`, structurally non-zero; the mip-chain placement path
  already has the explicit `if block_width == 0 || block_height == 0` refusal.
- `raeen-gpu` texture/format block extents — `format_block_extent` is a
  `const fn` whose fallback arm is `1`.

So: **no divide-by-zero site was found in host code, and the exit code proves
none could have been the cause.**

### What was implemented instead

**`RuntimeError::IntegerDivideFault { rip, cause, origin, hle }`** plus a VEH
arm for `EXCEPTION_INT_DIVIDE_BY_ZERO` (`0xC0000094`) **and**
`EXCEPTION_INT_OVERFLOW` (`0xC0000095`) — the same `idiv` instruction, the same
`#DE`, the same silent-death signature (`INT_MIN / -1`).

The arm mirrors the existing illegal-instruction recovery exactly: record the
error, roll callback completions back innermost-first (out of the `Cell` before
any guest write, per `ActiveContext::callback_frames`), capture the full
register `FaultSnapshot`, and recover through `run`'s `RtlCaptureContext`
snapshot. Resuming the faulting instruction is not an option — it would re-fault
forever.

`origin` is decided by `rip >= GUEST_ARENA_BASE`, the same test the
illegal-instruction arm uses. The two verdicts print different guidance,
because they want opposite investigations:

- **Guest** — "very often that zero came FROM US: an HLE stub returning 0 (or
  leaving an out-parameter untouched) for a grain size, sample rate, stride,
  element size, or frequency the title then divides by. The recent-HLE-call
  trace below is the place to look." The `div`'s own instruction bytes are
  dumped, which encode the divisor register, and the register file names its
  value.
- **Host** — "this cannot come from safe Rust integer division, so it is a C/C++
  dependency or inline assembly."

`log_call_trace` now runs for this variant, so the report carries the in-flight
HLE call, its six arguments, and the register dump.

**One hardening item from the audit was applied**: `raeen-kernel`'s
`memory::align_up(value, alignment)` uses the mask form
`(value + alignment - 1) & !(alignment - 1)`, which underflows for
`alignment == 0` and silently corrupts for a non-power-of-two. All four callers
pass the `PS5_PAGE_SIZE` constant today, so it cannot fire; a
`debug_assert!(alignment.is_power_of_two())` now makes it fail loudly in tests
the day a *guest-supplied* alignment is threaded through.

### Status

The crash cause itself is **not** fixed — it was never in our code. What changed
is that the next occurrence reports `rip`, cause, guest-vs-host, the faulting
instruction bytes, the full register file, and the HLE call in flight, and the
run survives to keep producing evidence instead of the process vanishing. Re-run
A Plague Tale Requiem to collect that report; the guidance line points at the
zero-returning stub.

---

## (B) Subnautica Below Zero — `sceKernelRaiseException` now really delivers

Measured first blocker:

```
WARN raeen_hle::libkernel: sceKernelRaiseException: guest handler is registered
but asynchronous delivery is not implemented; acknowledging
target_thread=0x1 signum=30 handler=<ADDR>
```

`timed_out` at 180 s, **1.4 s of CPU**, 0 flips. `signum=30` is FreeBSD/Orbis
`SIGUSR1`, and `target_thread=0x1` is the main thread. That combination is a
managed runtime's stop-the-world collector suspending a thread. Acknowledging
without delivering leaves the collector waiting forever for a suspension that
never happens — which is exactly what 1.4 s of CPU across 180 s of wall clock
looks like.

### Reference semantics

- **shadPS4** (GPL-2.0, `core/libraries/kernel/threads/exception.cpp`):
  `sceKernelRaiseException` accepts only `POSIX_SIGUSR1`; the handler ABI is
  `void handler(int signum, ucontext_t*)`; on Windows it delivers via a special
  user APC (`NtQueueApcThreadEx`) that fills an Orbis `Ucontext` from the
  target thread's `PCONTEXT`. Its `Ucontext`/`Mcontext` in `exception.h` are the
  layout authority used below.
- **SharpEmu** (GPL-2.0, current working tree — `6db095e` reverted and
  `db4339f` restored the surrounding work): `DirectExecutionBackend
  ::TryRaiseGuestException` **queues** the raise and lets the target's own
  executor consume it "at its next HLE boundary, where the original guest thread
  is safely paused", because "running its signal handler concurrently on a new
  managed thread corrupts the worker's control state". Its own comment records
  that Unity begins its next stop-the-world cycle inside that window.

Re-implemented in Rust; no code copied from either.

### The model: raise queues, the target thread delivers

A guest signal handler must run **on the target thread's own stack**, with that
thread's TLS and its guest frames below it. The raising thread cannot run it.

So `sceKernelRaiseException` records a `raeen_kernel::PendingException
{ signum, handler, raised_by }` against the target thread and returns `SCE_OK`.
Every HLE dispatch is a **safe point** — the guest is stopped at a known
instruction boundary with its full register file captured, and
`raeen-runtime`'s `call_guest` can synchronously re-enter guest code there.
`exception::deliver_pending` runs at the end of `HleRegistry::call`, the single
chokepoint both dispatch paths share, so the target picks up its own signal at
its next import. For a self-raise the safe point is the same call that raised
it, so delivery is immediate.

*After* the handler body, not before: the import has completed and holds no
HLE-internal lock, so a guest signal handler that blocks (a collector parking
until resume — the point of the signal) cannot wedge the kernel state that call
was using.

New state on `OrbisKernel`:

| Field | Purpose |
|---|---|
| `pending_exceptions: DashMap<u64, PendingException>` | one slot per target thread, newest wins |
| `pending_exception_count: AtomicUsize` | the per-call fast path — `DashMap::is_empty` locks every shard, which is unacceptable on the dispatch path |
| `exception_delivery_active: DashMap<u64, ()>` | re-entrancy guard: the handler's own imports are safe points too |
| `exception_contexts: DashMap<u64, u64>` | per-thread guest `ucontext_t` scratch, allocated once |

Newest-wins rather than a queue: a raise is a level, not an event stream, and an
unbounded backlog would let a collector that raises each cycle accumulate
deliveries it no longer wants. `handler` is latched at raise time so a
concurrent `sceKernelRemoveExceptionHandler` cannot turn a queued raise into a
jump through a stale slot.

Thread exit calls `discard_pending_exception`, which both drops the undelivered
signal (warning once, since the raiser's wait will not be satisfied) and keeps
the pending set empty — a stale entry would leave `has_pending_exceptions()`
true forever and turn every later HLE call's one-atomic fast path into a map
lookup.

### The machine context

`HleContext` gained `caller_gprs: Option<GuestGpRegs>` — the interrupted guest
thread's **complete** integer register file plus `rflags` and the FS base,
filled from the trap `CONTEXT` on the VEH path. The argument slice only carries
the six SysV argument registers; a handler that receives a *machine context*
needs the callee-saved set too, above all `rbp`, which a managed collector
unwinds the suspended thread through. The direct leaf gateway supplies `None`
(no `CONTEXT` exists there; it arrives by a plain `call`).

`ucontext_t` layout, cross-checked against shadPS4's `Ucontext`/`Mcontext` and
pinned by `ucontext_layout_matches_the_freebsd_amd64_abi`:

```
sizeof(ucontext_t)   = 0x500
uc_mcontext          @ 0x040   (uc_sigmask[16] + 0x30 private bytes)
sizeof(mcontext_t)   = 0x480
  mc_rdi/rsi/rdx/rcx/r8/r9/rax/rbx/rbp/r10/r11/r12/r13/r14/r15
                     @ +0x08 .. +0x78   (FreeBSD order, NOT SysV order)
  mc_rip             @ +0x0A0
  mc_rflags          @ +0x0B0
  mc_rsp             @ +0x0B8
  mc_len             @ +0x0C8   = 0x480
  mc_fpformat        @ +0x0D0   = _MC_FPFMT_NODEV   (0x10000)
  mc_ownedfp         @ +0x0D8   = _MC_FPOWNED_NONE  (0x20000)
  mc_fpstate[104]    @ +0x100 .. +0x440
  mc_fsbase/gsbase   @ +0x440 / +0x448
```

The buffer is zero-filled first, so a handler reading an unmodelled field sees a
defined zero rather than allocator residue. `mc_rsp` is `caller_rsp + 8` — the
interrupted RSP, one slot above the `call`-pushed return address.

### What delivery does NOT do (named, not hidden)

- **Timing.** Delivery is at the target's next HLE call, not pre-emptive. A
  thread in pure guest compute with no imports is not interrupted.
- **Direct-gateway-only threads.** `trampoline::direct_dispatchable` imports
  (`scePthreadMutexLock`, `scePthreadCondWait`, `sceKernelWaitSema`, …) reach
  their handlers by a plain `call` on a private host stack and **cannot** re-enter
  guest code — `call_guest` refuses loudly there by design. A thread whose only
  imports are on that list never reaches a delivering safe point. The raise is
  **requeued, not dropped**, and after 64 consecutive deferrals a single `warn`
  names the condition and the `RAEEN_DISABLE_DIRECT_HLE=1` escape hatch, which
  routes every import through the VEH path where delivery always works. This is
  the most likely residual blocker if Subnautica still stalls.
- **`mc_fpstate`.** Zeroed, with `mc_fpformat`/`mc_ownedfp` set to the ABI's
  "no FP state here" values, so a handler that checks them reads no garbage —
  but a handler that needs the XMM file will not find it.
- **Segment selectors, `mc_trapno`, `mc_err`, `mc_addr`.** Zero. There was no
  trap, so there is no honest value.
- **Resuming *from* the ucontext.** The handler returns normally to us. A handler
  that modifies the context and expects the thread to resume from it is not
  supported, and nothing signals that refusal.
- `sceKernelRaiseException` still accepts any `signum` in `0..128` rather than
  only `SIGUSR1` as shadPS4 does. Install/remove keep the tighter
  Orbis-allowed set (1, 4, 8, 10, 11, 30), so an unsupported signal has no
  handler and the raise is a no-op anyway.

### Status

Delivery is real and proven end-to-end by a hand-assembled guest fixture whose
handler validates the context it was handed. Subnautica has **not** been re-run;
this is a mechanism claim, not a boot claim.

---

## Tests

All deterministic — no threads started for timing, no sleeps.

| Suite | Before | After | Added |
|---|---|---|---|
| `raeen-hle` lib | 553 | **563** | 10 (`exception::tests`) |
| `raeen-kernel` lib | 65 | **65** | 0 |
| `raeen-runtime` lib | 83 | **83** | 0 |
| `raeen-runtime` `tests/execute.rs` | 57 + 1 ignored | **60** + 1 ignored | 3 |

(`raeen-kernel` 65 and `raeen-runtime` lib 83 are the counts at branch base
`75c456e`, which is ahead of the `2e4bdca` build the baseline JSON was measured
at — hence 65/83 rather than the handoff's 64/82. Neither moved here.)

New `raeen-runtime` acceptance tests:

- `guest_divide_by_zero_is_classified_instead_of_killing_the_process` — a guest
  `xor ecx,ecx; xor edx,edx; mov eax,1; div ecx` must come back as
  `Err(IntegerDivideFault { rip: <the div>, cause: ByZero, origin: Guest })`,
  and the test process must survive to run more guest code afterwards. Before
  the VEH arm this test would have killed the test binary with `0xC0000094`.
- `guest_idiv_quotient_overflow_is_classified_as_a_divide_fault` — the
  `INT_MIN / -1` arm.
- `raise_exception_runs_the_installed_guest_handler_with_a_real_ucontext` — a
  guest installs a handler for signal 30 and raises it at itself; the handler is
  **real guest code** that starts from its `signum` argument and then validates
  the context it was handed (`or eax,0x100` only if
  `uctx->uc_mcontext.mc_len == 0x480`, `or eax,0x200` only if `mc_rip != 0`).
  The single returned value therefore proves the handler ran, got the right
  signal, received a genuine `ucontext_t`, and that the context describes the
  interrupted guest instruction. Also asserts the delivery counter advanced by
  exactly one and that neither the pending map nor the delivering set leaked.

`raeen-hle` `exception::tests` covers delivery at the safe point with a full
register file, per-thread targeting (thread 2 must not consume thread 1's
signal), the re-entrancy refusal, requeue-instead-of-drop on an undeliverable
dispatch path, the named null-handler refusal, the zero-pending fast path,
dead-thread discard re-arming the fast path, newest-wins replacement, the
`ucontext_t` layout constants, and the Orbis-allowed signal set.

`cargo fmt --all` clean; `cargo clippy -p raeen-hle -p raeen-kernel
-p raeen-runtime --all-targets -- -D warnings` clean.
