# RE: ASTRO.BOT guest fault at module+0xe03f1a (NULL-base deref) — 2026-07-22/23

**Status:** diagnosis (no behavior change). Evidence run:
`scratch/astro-voicelist6-20260723.out.log` (all three faults reproduced in one
run, plus a later main-thread fault at the OLD site +0x33f335).
Ground truth: `scratch/astro-apr-fix-20260722.out.log`.
Ring-at-DEBUG run: `scratch/astro-ring-debug-20260723.out.log`.
NGS2 create-family trace run: `scratch/astro-ngs2-trace-20260723.out.log`.

Two clearly-marked TEMP-DIAG additions were left in the working tree (no
commit); they are diagnostics only and should be removed or kept deliberately:

- `crates/raeen-hle/src/libsce_media.rs` — env-gated (`RAEEN_TRACE_NGS2`) arg
  dump in `hle_ngs2_create_out2`.
- `crates/raeen-runtime/src/dispatch.rs` — prints r12–r15 from the already
  captured `FaultSnapshot` (they were captured but never reported), plus an
  env-gated (`RAEEN_DUMP_VOICE_LIST`) SAL voice-list walk at fault time.

---

## 1. Executive summary

The fault at +0xe03f1a is **not** a missing HLE return value and **not** an
NGS2 initialization gap. It is a **secondary fault**: the title's SAL audio
layer walked its voice list and loaded a **poison value `0xAAAAAAAC`** (the
title allocator's fresh-allocation fill) as the 6th list node, because that
node's link struct at `0x1000055bc8` was allocated and linked into the list
but its "next voice" field (`+0x10`) was **never written**. The half-linked
state **persists** (identical 12 s later at the second fault), so the producer
never finished — this is not a benign race window.

Raeen's permissive identity-mapped guest arena makes low addresses readable
as zero, so the title's wild deref of the poison pointer (`[0xAAAAAAAC+0xe8]`,
`[0xAAAAAAAC+0x120]`) returned 0 instead of faulting at the true site; the
visible fault surfaced one instruction later as a NULL-base deref at
`[0+0x10]`. On real hardware the same corruption faults at the poison deref.

All three new worker faults are the same family: title worker threads running
0x1100xxxx-command-dispatcher handlers iterate object registries (SAL voice
list; a stride-0x60 object array; the named-property table near
`PriHeroAnimation.cpp` / "pri_hero") that each contain one half-built entry,
seconds after `LevelDocument Loaded: ui_pause_next [pause_menu]`. The
pause-menu async load published its consumers before its registrations
finished — the same *family* as the 2026-07-22 APR completion bug, and the
main thread later died at the OLD fault site +0x33f335 (read
`0xFFFFFFFFFFFFFFFF`), so that family is not fully closed.

## 2. Fault 1 — module+0xe03f1a (t43)

### 2.1 Site and source file

The two string args of the variadic log call at +0xe03f0e (target +0xdfdc80,
variadic prologue = title printf-like logger) resolve to:

- module+0x84e20f0: `sal_ngs2.c`
- module+0x80e15c5: `NGS2 : Failed to set output (%d)` (esi=0x470 = line 1136)

So the faulting function is in the title's **SAL (Sony Audio Library) NGS2
glue** — statically linked into eboot.

### 2.2 The loop (aligned disassembly, verified against fault-site bytes)

```
+e03ec0  mov dword [rbx+8E8h], 1          ; rbx = big stack-local context (rbx==rsp at fault)
+e03eca  mov rax, [rdi+38h]               ; rdi = [0E7F6020h] = global SAL manager (0x315f99b00)
+e03ece  mov r14, [rax+10h]               ; r14 = first voice node
+e03ed2  test r14, r14
+e03ed5  jne +0xe03f27                    ; non-empty → enter loop
+e03ed7  mov edx, 1 ; jmp +0xe040ad       ; empty → done

loop_head:
+e03f13  mov rax, [r14+0E8h]              ; voice+0xe8 = link struct
+e03f1a  mov r14, [rax+10h]               ; *** FAULT *** next voice = [link+0x10]
+e03f1e  test r14, r14
+e03f21  je  +0xe040a0                    ; next==0 → done
+e03f27  cmp byte [r14+120h], 0
+e03f2f  je  loop_head                    ; flag clear → skip
         ; ... process voice: reads +0x118/+0x110/+0xb8/+0x114/+0x127/+0x58,
         ;     [r14+r13*8+0xf8], +0x108 — builds an output-routing command
+e03f66  call +0xe19770                   ; command interpreter (jump table for
                                         ;   cmd ids 1..6, 0x1000000x, 0x2000000x)
+e03f6b  test eax, eax
+e03f6d  js  +0xe03ef0                    ; error → log "NGS2 : Failed to set output (%d)"
+e03f75  cmp rax(counter), 1Fh
+e03f79  ja  loop_head                    ; advance
```

Object model: voice structs (~0x158 stride) hold at **+0xe8 a pointer to a
separately heap-allocated link struct**; `[link+0x10]` = next voice;
`[voice+0x120]` = active flag. List head: `[[[0xE7F6020]]+0x38]+0x10`.

### 2.3 Register truth (TEMP-DIAG, this run)

```
r14 = 0xaaaaaaac   r12 = 0x200   r13 = 0x0   r15 = 0x40   rax = 0   rbx = rsp
```

`0xAAAAAAAC` is the smoking gun: the title's own allocator poison-fills fresh
(or freed) blocks with 0xAA (no 0xAA fill exists anywhere in Raeen — verified
by grep; the only 0xAA constants in-tree are PM4 test patterns). The faulting
`mov r14,[rax+0x10]` never completed, so this r14 is the CURRENT node — the
walker had already loaded the poison value as "next voice" from
`[0x1000055bc8+0x10]`.

Mechanics of the secondary fault: `[0xAAAAAAAC+0xe8]` must have returned 0
(rax=0) — only possible because Raeen's guest arena lets the low 4 GB read as
zeros. Real PS5: unmapped → fault at the poison deref itself.

### 2.4 The list state at fault time (TEMP-DIAG `RAEEN_DUMP_VOICE_LIST`)

```
mgr  = [0xe7f6020]  = 0x315f99b00
cont = [mgr+0x38]   = 0x1000001568
node[0]=0x1000001488 [+0xe8]=0x10000016c0 [+0x120]=1
node[1]=0x10000015e0 [+0xe8]=0x1000001818 [+0x120]=1
node[2]=0x1000001738 [+0xe8]=0x1000001970 [+0x120]=1
node[3]=0x1000001890 [+0xe8]=0x1000001ac8 [+0x120]=1
node[4]=0x10000019e8 [+0xe8]=0x1000055bc8 [+0x120]=1   <-- link in a DIFFERENT heap region
node[5]=0xaaaaaaac                                    <-- [0x1000055bc8+0x10] = POISON
```

Nodes 0–3 are the statically created voices (contiguous heap, stride 0x158).
Node 4's link struct lives at 0x1000055bc8, a different allocation region —
the dynamically added (pause-menu) voice. Its link's "next voice" field was
never written: it still holds the allocator's 0xAA fill. **The identical dump
at fault 2, 12 s later, proves the half-linked node is permanent** — the
producer did not merely lose a race; it never completed the insert.

### 2.5 HLE ring evidence (t43)

4096-entry ring (DEBUG run): 3205 libkernel + 864 libSceAudioOut2 + 27
libScePosix calls. **Zero libSceNgs2 / libSceAjm calls.** No HLE call returned
an Orbis error; no pointer-returning call handed back 0 (distilled leads
empty). rdx at fault = 0x1110110 — a 0x1100xxxx command id: t43 was executing
a dispatcher command handler (audio output-routing rebuild), consistent with
the loop reading `[rbx+0x8e8]` as an output index.

## 3. The NGS2 hypothesis — tested and REFUTED

`sal_ngs2.c` + "Failed to set output" made a stubbed/mis-ABI'd NGS2 HLE the
prime suspect. Measured instead:

- **The title never calls any registered NGS2 create** in 75 s of boot+run
  (`RAEEN_TRACE_NGS2`: zero hits on `sceNgs2SystemCreateWithAllocator` /
  `sceNgs2RackCreateWithAllocator` / `sceNgs2RackGetVoiceHandle`).
- **Zero sceNgs2\*/sceAjm\* calls in any of the three 4096-entry rings** —
  NGS2 is not being driven per-frame at all; audio runs through AudioOut2.
- No libSceNgs2 NID is in the 176 missing-NID link list; no unresolved import
  was ever called (`RAEEN_RESUME_ON_MISSING` never fired).
- Separately noted for the record: Raeen's `hle_ngs2_create_out2` writes the
  out-handle to args[2] (rdx) for **all three** creates, but SharpEmu's
  `Ngs2RackCreateWithAllocator` takes the out pointer in **r8 (args[4])**
  (`reference/sharpemu/.../Ngs2Exports.cs:143`). Real ABI bug — but COLD for
  this title (never called). Fix when NGS2 actually gets driven.

## 4. Faults 2 and 3 — same family

### Fault 2 — module+0xe47a43 (t44), `cmp byte [r15+0x29],0`, read at 0x29

```
+e47a00 mov rdi,[rbx+18h]        ; context -> owner
+e47a07 mov eax,[rdi+98h]        ; count
+e47a0d cmp r13,rax / jae exit
+e47a16 mov r14,[rdi+90h]        ; array base   (r14=0x10000730c0 at fault)
+e47a1d lea r15,[r13+r13*2] / shl r15,5        ; entry stride 0x60
+e47a26 cmp dword [r14+r15+48h],2 / jne +0xe47a3e
+e47a33 lea rdi,[r14+r15] / xor esi,esi / call +0xe49330
+e47a3e mov r15,[r14+r15+28h]    ; entry+0x28 = sub-object
+e47a43 cmp byte [r15+29h],0     ; *** FAULT *** entry+0x28 was 0
```

Outer loop snapshots object pointers into a stack array via +0xdf3540
(lock-free `lock cmpxchg` gather). Chain `+0xf4082c <- +0xdc2a7e <- +0x10e91
<- +0xdfb602 <- +0xded2d9`: +0xded2d9 sits in the known 0x1100xxxx dispatcher
territory (query-status handler 0xDF02EC → 0xDED4E0, completion flag
[[0xE7F5BE0]+0x288]) — a dispatcher handler iterating a registry array whose
entries have a NULL sub-object at +0x28: half-initialized entries, same shape
as fault 1. The voice-list dump at this fault (above) shows the SAL list
still identically corrupted.

### Fault 3 — libc.prx+0x356ba (t45; ground truth), strcpy byte-loop, wild read 0xfaab60664

Caller +0xe54d3f / +0xe53889 iterates a packed name table:

```
+e53839 movsxd rax,[rbx]                ; entry offset record
+e5384a movzx eax, word [rbx+rax]
+e53850 imul rdx, rax
+e53854 movsxd rdi,[rdx+rcx+0Ch]
+e53867 lea r13,[rdi+rax+0Ch]           ; r13 = name ptr
+e5386c test rdi,rdi
+e53873 lea rax,[80D62E2h]              ; default/empty name
+e5387d cmove r13,rax                   ; offset==0 → default
+e53881 mov r8,r13                      ; → strcpy src
```

The guard only handles offset==0; an entry whose offset is non-zero but wild
yields the unterminated runaway strcpy. String region identified:
`D:\asobi\6.0\source\app\PlayRoomB\Game\Room\PreInstall\PriHeroAnimation.cpp`
with property names ("pri_hero", Hero::RestoreArgs, Combo, …) — the title's
named-property/parameter registry. Same family: a registry entry half-built.

In this diagnosis run the third fault instead surfaced on the **main thread at
the OLD site +0x33f335** (read 0xFFFFFFFFFFFFFFFF — 0xFF poison), followed by
`RESULT: guest fault` termination — direct evidence that the 2026-07-22 APR
fix did not fully close the completion-ordering family.

## 5. Timeline correlation (both runs)

1. `LevelDocument Loaded: ui_pause_next [pause_menu]` on a load worker
   (t72 this run, +5 s before fault 1).
2. t43/t44/t45 trip over half-built registry entries within ~5–15 s.
3. All three faulting threads run 0x1100xxxx dispatcher command handlers.

The pause-menu load registers new objects (a voice in the SAL list, entries in
the stride-0x60 array, name-table rows) and then signals completion through
the dispatcher's completion-flag protocol. Consumers ran while at least one
registration was incomplete — and the registration never completed afterwards.

## 6. Root cause

**Proven:**
- The faulting object is not a valid-but-NULL-field voice; the "next voice"
  pointer itself is the title allocator's 0xAA poison, read from link struct
  0x1000055bc8 whose +0x10 was never initialized. (Direct guest-memory dump.)
- The corruption is permanent, not transient (identical list state 12 s
  apart).
- The NULL-base deref at +0xe03f1a is secondary: Raeen's permissive arena
  reads low addresses as zero, deferring the true fault (poison deref) by one
  instruction. (Register mechanics: r14=0xAAAAAAAC with fault access 0x10.)
- No NGS2/AJM HLE call is involved at all; "no HLE error" is accurate — the
  emulator did not hand the title a bad value on these threads.

**Strong hypothesis (emulator-side):** the pause-menu async-load/registration
path publishes "ready" to the 0x1100xxxx dispatcher before its object
registrations are complete — the same completion-ordering family as the
2026-07-22 APR bug (whose main-thread +0x33f335 fault recurred in this very
run). Alternatively the registering worker stalled or died mid-insert (the
chronic `scePthreadMutexLock stuck >3s — owner=21, mutex=0x300944e00` warnings
fire in every run, including before fault 1 in ground truth); a producer
wedged mid-insert leaves exactly this permanent half-linked node.

**Cannot rule out honestly:** a genuine title race that hardware also has but
never loses (e.g., producer/consumer handler affinity that our scheduling
breaks). No evidence for it, but it was not disproven.

## 7. Proposed fixes (in priority order)

1. **Surface the true fault site (diagnostics fidelity)** —
   `crates/raeen-runtime` GuestArena: stop letting low guest addresses (at
   least the first 4 GB) read as zero; map them no-access. This turns the
   confusing NULL-base deref at +0xe03f1a into the real "wild read
   0xAAAAAB94/0xFAAB60664" at the first bad load and names poison pointers
   immediately. Cheap, high diagnostic value; not a behavior fix.
2. **Re-open the APR/async completion path** — audit that the pause-menu
   (`ui_pause_next`) load signals subsystem completion
   ([[0xE7F5BE0]+0x288] protocol) only after ALL object registrations are
   fully written. The surviving +0x33f335 main-thread fault says the
   2026-07-22 fix is incomplete. Files: the APR completion path touched on
   2026-07-22 (raeen-hle APR handlers) + the dispatcher completion-flag
   consumer.
3. **Investigate the chronic mutex hold** — `guest_thread=21` holds
   0x300944e00 >3 s repeatedly (`raeen_hle::pthread_sync` warnings) in every
   run. If that thread is the registry producer wedged mid-insert, fixing its
   stall fixes the family. Next diagnostic: dump thread 21's entry point and
   the mutex's wait chain when the >3 s warning fires.
4. **Cold but real ABI bug** — `libsce_media.rs::hle_ngs2_create_out2`:
   `sceNgs2RackCreateWithAllocator`'s out-handle pointer is in **r8
   (args[4])**, not rdx; split the handler per-function. Only when NGS2 is
   actually exercised.

## 8. Not determined (open threads)

- Which title code allocates/fills link struct 0x1000055bc8 and why it
  stopped before writing +0x10 (needs a write watchpoint on the link node, or
  tracing the voice-registration handler in the 0x1100xxxx dispatcher).
- Whether the 0xAAAAAAAC fill marks *fresh* or *freed* memory in this title
  (fresh-alloc fill and free-fill are both 0xAA in some allocators); the
  "different heap region" observation slightly favors fresh.
- The exact identity of guest_thread 21 and whether it is the producer.
- Why the run-to-run reproduction rate is ~50% (boot-time scheduling
  variance); several 75–110 s runs never reached +46 s-equivalent progress.
