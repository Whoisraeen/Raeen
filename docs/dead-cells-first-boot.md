# Dead Cells (PPSA15552) — first boot

Adding a second title to the compat registry exposed two defects that had been
latent behind a single-title sample. Neither is Dead Cells-specific; both were
bugs that Minecraft happened not to trip.

Evidence: `artifacts/compat/deadcells-first.json`, raw log
`artifacts/compat/raw/baseline-1785284515961/PPSA15552-d31b55cde5bd.stdout.log`,
crash report `logs/crashes/PPSA15552_20260729-002158Z.report.md`.
Run of 2026-07-29, v01.005.000. Died 2.2 s in.

---

## 1. `install_null_free_guard` planted a stub with half an instruction in it

### Symptom

```
WARN  null_free_guard: installed sceLibcMspaceFree at 0x1000029637d0 (stub=0x100002a95b80)
INFO  Registered module: id=2, name='libc.prx', base=0x100002954000, size=0x141b80
ERROR guest fault at 0x100002a95b97 (read 0x0) — 35 HLE call(s) recorded before the fault
WARN  fault module: rip 0x100002a95b97 is in NO loaded module
INFO  bytes at rip: 02 00 10 00 00 ff e0 31 c0 c3
```

### Root cause — (b), not (a)

The log's fault-site dump prints the whole stub verbatim, which settles it:

```
0x100002a95b80:  48 85 ff        test rdi, rdi
0x100002a95b83:  74 19           jz   .null (-> +0x1e)
0x100002a95b85:  55              push rbp        \
0x100002a95b86:  48 89 e5        mov  rbp, rsp    |
0x100002a95b89:  41 57           push r15         | the 13 bytes copied
0x100002a95b8b:  41 56           push r14         | from the real prologue
0x100002a95b8d:  41 54           push r12         |
0x100002a95b8f:  53              push rbx         |
0x100002a95b90:  48 89           <TRUNCATED>     /   <-- 2 of a 3-byte insn
0x100002a95b92:  48 b8 dd 37 96 02 00 10 00 00    movabs rax, 0x1000029637dd
0x100002a95b9c:  ff e0                            jmp rax
0x100002a95b9e:  31 c0 c3                         .null: xor eax,eax; ret
```

Dead Cells' `libc.prx` build of `sceLibcMspaceFree` opens with
`push rbp; mov rbp,rsp; push r15; push r14; push r12; push rbx; mov rbx,rdi`.
That is **11 bytes of whole instructions followed by a 3-byte `48 89 fb`** — the
next boundary is 14, not 13. The caller hardcoded `prologue_len = 13`, so the
copy sliced `mov rbx,rdi` in half and the decoder desynchronized:

| addr | bytes | decodes as |
|------|-------|------------|
| +0x10 | `48 89 48 b8` | `mov [rax-0x48], rcx` — a stray store (rax still held the stub address from the entry patch) |
| +0x14 | `dd 37` | `fnsave [rdi]` — 108 bytes of FPU state dumped into the mspace |
| +0x16 | `96` | `xchg esi, eax` — zero-extends, leaving **rax = 0** |
| +0x17 | `02 00` | `add al, [rax]` — **read of 0x0** |

The fault snapshot corroborates every step: `rax=0x0`, `rsi=0x2a95b80` (the low
32 bits of the stub address, swapped out by the `xchg`), `rdi=0x100002a682d8`
(the live mspace the `fnsave` scribbled on).

So `stub_addr` was **correct**. `GUEST_ARENA_BASE + image.len()` is right because
the *composed* image is what is mapped at `GUEST_ARENA_BASE`; libc.prx lives
inside it at `+0x2954000`, and the guest demonstrably entered the stub at `+0`
and ran the first two instructions correctly. The base-address theory (b's first
alternative) is disproved; the truncated-prologue theory is proved.

Hypothesis (a) is also real but is a **separate, non-fatal** defect: because
libc.prx is the last module placed, `base + size = 0x100002954000 + 0x141b80 =
0x100002a95b80` is *exactly* the stub, so the stub sits one byte past every
registered extent. That is why the report said "in NO loaded module". It did not
cause the crash — the memory is mapped and executable — but it made the crash
report point nowhere.

### Why Minecraft never hit it

Minecraft's build has one extra `push r13`, so its boundaries are
1, 4, 6, 8, 10, 12, **13**, 16. 13 is a genuine instruction boundary there. The
hardcoded constant was correct for the only title anyone had measured.

### Fix

`crates/raeen-runtime/src/native_trap.rs`

* The relocated prologue length is now **measured, not assumed**.
  `relocatable_prologue_len` walks whole instructions from the target until at
  least `PATCH_LEN` (12) bytes are covered. Dead Cells -> 14, Minecraft -> 12.
* The decoder is a **whitelist**, not a general length decoder: `endbr64`,
  `push`/`pop r64`, register-direct (`mod == 3`) ALU/`mov`, group-1
  `imm8`/`imm32` on a register, `mov r, imm`. Every accepted form is
  register-only, so nothing rip-relative and no relative branch can be copied.
  Anything else is refused.
* **A refused guard is not installed and says so loudly.** An uninstalled guard
  risks a null free later; a mis-copied prologue is a guaranteed crash in
  seconds. Refusal leaves the image byte-for-byte untouched.
* `install` (the `RAEEN_TRAP_MSPACE` diagnostic detour) now *validates* its
  caller-supplied length against the same decoder and refuses if it is not a
  boundary. Its two Minecraft call sites pass 13, which is a real boundary there.
* `install_null_free_guard` returns a `GuardStub { image_offset, len }`, and
  `cover_guard_stub` folds it into the owning module's `unwind.image_size` when
  it is contiguous with the module's end (fix for (a)). When it is not
  contiguous the extent is left alone and a warning names the consequence.

**Guard status for dependency modules: still installed.** Nothing about being a
dependency makes it unsafe — the stub address is composed-image absolute and was
always correct. The refusal path triggers on *prologue shape*, not on module
identity, so any future libc whose prologue is not safely relocatable gets a loud
refusal instead of a planted crash.

---

## 2. The crash report dropped its most valuable section

`logs/crashes/PPSA15552_20260729-002158Z.report.md`:

```
## Recent HLE calls (most recent first)

<none recorded>
```

…for a fault whose own log line reads "35 HLE call(s) recorded before the
fault", and where the dispatch layer distilled leads from that same ring.

### Root cause

There are **two** rings and they were never connected:

| ring | owner | populated |
|------|-------|-----------|
| `dispatch::CallTrace` | the faulting thread's `ActiveContext` | always on, 4096 deep — this is the one with 35 entries |
| `OrbisKernel::recent_hle_calls` | process-wide `DashMap` | **only** under `RAEEN_TRACE_EINVAL`, and only on the direct-gateway path |

`raeen-gui`'s report reads the second. `CallTrace` dies with the run, so on any
normal launch the report found an empty map and rendered `<none recorded>`. The
renderer was never at fault.

This matters beyond debugging: `.github/ISSUE_TEMPLATE/game-report.yml` asks
users to paste this report and calls that section the most useful thing they can
provide.

### Fix

`crates/raeen-runtime/src/dispatch.rs` — `log_call_trace` now publishes the last
`CRASH_REPORT_RING_LEN` (24) entries of the authoritative `CallTrace` into
`kernel.recent_hle_calls` for the faulting thread, oldest-first (the deque
convention the report's renderer reverses), **keeping each call's return value**:
a `-> 0x0` a few calls before a null dereference is the whole lead. It replaces
rather than appends, since the always-on ring supersedes the opt-in one.

`crates/raeen-gui/src/crash_report.rs` — the kernel-to-report mapping moved out
of `main.rs` into `recent_hle_for_report`, so it is unit-testable.

---

## Tests

| Test | Crate | Covers |
|------|-------|--------|
| `dead_cells_prologue_boundary_is_fourteen_not_thirteen` | raeen-runtime (lib) | the exact bytes from the crash, boundary is 14 |
| `minecraft_prologue_boundary_is_still_valid` | raeen-runtime (lib) | no regression for the title that plays |
| `endbr64_and_stack_adjust_prologues_decode` | raeen-runtime (lib) | modern CET/frame-setup prologues |
| `non_relocatable_prologues_are_refused` | raeen-runtime (lib) | rip-relative `lea`, `jmp rel32`, memory operand, short buffer |
| `guard_stub_relocates_whole_instructions_and_returns_its_extent` | raeen-runtime (lib) | full stub layout on the real Dead Cells bytes |
| `refused_guard_leaves_the_image_untouched` | raeen-runtime (lib) | refusal is byte-for-byte inert |
| `out_of_range_target_is_refused` | raeen-runtime (lib) | bounds |
| `a_contiguous_guard_stub_is_folded_into_its_module` | raeen-runtime (lib) | the measured Dead Cells geometry, defect (a) |
| `a_detached_guard_stub_does_not_stretch_the_module` | raeen-runtime (lib) | never stretch over unrelated bytes |
| `a_fault_publishes_its_hle_call_ring_for_the_crash_report` | raeen-runtime (execute) | a real fault fills `recent_hle_calls` — verified to fail without the fix |
| `report_names_the_calls_the_ring_recorded` | raeen-gui | a populated ring reaches the rendered markdown |

All deterministic; no sleeps, no GPU.

Counts after: raeen-runtime 98 lib + 62 execute (1 ignored), raeen-gui 204,
raeen-hle 571, raeen-kernel 63, raeen-firmware 127. `cargo clippy --workspace
--all-targets -- -D warnings` and `cargo fmt --all -- --check` clean.

---

## Not addressed

Dead Cells has not been re-run against this build, so **no claim is made that it
boots further**. What is proven is that the 2.2 s crash had a single mechanical
cause, that cause is fixed and covered by a test using the exact faulting bytes,
and the next failure — whatever it is — will come with a crash report that
actually names the calls leading up to it. The title also reports 21 unresolved
imports (`libSceNgs2` among them); those are untouched here.
