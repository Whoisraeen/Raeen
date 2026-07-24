- Present-path plugin ABI — upscaler / frame-gen framework (2026-07-23;
  working tree, no commit):
  * NEW `crates/raeen-gpu/src/present_plugin/{mod,builtin}.rs`: a generic,
    vendor-neutral `PresentPlugin` trait (`name`/`capabilities`/`process`) with
    `PresentFrame` (color + OPTIONAL depth/motion planes for the future
    PM4-extracted motion-vector moat), `PluginFrame`, `PluginOutput`
    (primary + reserved `generated` frame-gen list), `PresentContext`,
    `Capabilities`. Process-wide `Registry` (register/select/list/
    set_output_scale) behind a global; built-in reference plugins
    `Passthrough` (identity) + `NearestUpscale` (real nearest-neighbour
    resample) prove the boundary is a *general* upscaler ABI, not a DLSS socket.
  * HOOK: `AgcGpuSession::publish_frame` (the single present chokepoint) now
    runs `present_plugin::apply_to_image`. Default is a ZERO-COST identity —
    with no plugin selected it returns the same `Arc` (no copy, no behaviour
    change). Shell-facing API sits next to `set_runtime_config`:
    `register_present_plugin` / `select_present_plugin` / `clear_present_plugin`
    / `present_plugins` / `active_present_plugin` / `set_present_output_scale`.
  * LICENSE BOUNDARY (GPL-2.0): Raeen ships ONLY the vendor-neutral built-ins.
    Proprietary plugins (DLSS/Streamline) are BYO-and-unblessed — never
    vendored, fetched, branded, or named "supported". New git-ignored
    `plugins/` tree (`.gitignore`: `/plugins/*` + `!/plugins/README.md`) hosts
    user-supplied plugin crates; committed `plugins/README.md` documents the
    trait, the sketch, and the copyleft rationale. `git check-ignore` verified
    (`plugins/dlss/*` IGNORED, README TRACKED).
  * GREEN: 6/6 new unit tests; clippy `-D warnings` clean; fmt clean; M3
    interactive present-path regression still passes; raeen-gui builds.
  * FOLLOW-UPS (not done): FSR 3.1 (MIT) reference impl replacing NearestUpscale;
    PM4-side depth/motion-vector extraction (populate `PresentFrame.depth/motion`);
    Settings ▸ Video dropdown wiring; frame-gen scheduling of `generated`;
    stable C-ABI `dlopen` layer for no-recompile dynamic loading.
- Runner isolation + zero-fault leaf HLE gateway (2026-07-23; working tree,
  no commit):
  * SHELL CONTAINMENT: production `FirmwareLauncher` now starts the existing
    headless `--run-eboot` path as a child process and assigns it to a Windows
    kill-on-close Job Object. A hard-abort acceptance test proves the parent
    survives. `quit` kills the child; native XInput/DualSense polling runs
    inside it so process isolation does not remove physical-pad input.
  * EXECUTABLE THUNKS: linker-visible eight-byte slots now contain rel32 calls
    to generated shared bridges. Reviewed leaf libc calls use a SysV gateway
    on a private 256 KiB host stack with six register args, eight stack args
    and XMM0-7 forwarded. Unclassified/context-changing calls reconstruct
    their index and jump to a separate no-access slow region, preserving the
    existing VEH/CONTEXT path for exit, fibers, native traps and callbacks.
  * MEASURED: one million `strlen` imports: VEH 4024.471 ms / 248,480 calls/s /
    1,000,000 exceptions; direct 625.773 ms / 1,598,025 calls/s / zero HLE
    exceptions = 6.4x dispatch speedup. The ordinary 1,000-call regression
    also proves direct=1000 and VEH=0.
  * HONEST GATE: this closes the requested runtime slice, not M2/M3. Those
    remain OPEN until real screenshotable Vulkan presentation and interactive
    2D homebrew acceptance evidence pass.

- Reference gap-closure audit (2026-07-23; working tree, no commit):
  * HONEST GATES: M2/M3 downgraded from CLOSED to OPEN. Their current AGC
    triangle and interactive-2D proofs execute synthesized guests, which
    violates the acceptance-gate's explicit synthetic-only prohibition.
  * NID INTEGRITY: added a build-fail audit over the complete explicit override
    set (reviewed count, provider canonicalization, callable target, nonzero
    identity, and intentional name-hash mismatch). It found 25 redundant
    `register_nid` calls whose known names already hash to the literal NID;
    those now use ordinary name-derived registration. Only 11 intentional
    Gen5/provider-private/unknown overrides remain. Verification: raeen-hle
    368/368 with the one documented pre-existing eventflag hang skipped.
  * ROADMAP: `docs/reference-gap-closure-roadmap.md` orders runner-process crash
    containment, ABI-correct zero-fault HLE thunks, measured HLE breadth,
    retail shader/GPU/swapchain work, media/controller output, portability and
    REUSE/SPDX automation. Direct thunks are not a two-instruction patch:
    SharpEmu's reference saves the SysV register/XMM/AL state, switches ABI and
    calls a gateway; Raeen must additionally preserve its guest FS/TLS and
    context-changing slow paths.

- RE diagnosis: ASTRO.BOT +0xe03f1a NULL-base fault family (2026-07-23;
  diagnosis only, no behavior change, no commit; doc
  `docs/re/2026-07-22-astro-null-base-fault-e03f1a.md`; evidence
  `scratch/astro-voicelist6-20260723.out.log` — all 3 worker faults
  reproduced + main thread later died at the OLD +0x33f335):
  * VERDICT: NOT an HLE return-value bug. Faulting "voice" r14 = **0xAAAAAAAC**
    — the title allocator's 0xAA poison. The SAL voice list's 6th link struct
    (0x1000055bc8) was allocated+linked but its next-voice field (+0x10) never
    written; the half-linked state PERSISTS (identical dump 12 s later at
    fault 2), so the producer never finished — not a race window.
  * SECONDARY FAULT MECHANICS: Raeen's permissive arena reads low addresses
    as zero, so [0xAAAAAAAC+0xe8]→0 and the visible fault is the NULL deref
    one instruction later; real hardware faults at the poison deref itself.
    Confidence in object model + deferral: PROVEN (register dump + direct
    guest-memory list walk via TEMP-DIAG).
  * NGS2 HYPOTHESIS REFUTED: site is sal_ngs2.c:1136 "NGS2 : Failed to set
    output", but the title NEVER calls any sceNgs2*/sceAjm* function (trace
    + all three 4096-entry rings: zero hits); audio runs via AudioOut2. Cold
    but real ABI bug found en passant: hle_ngs2_create_out2 writes the
    RackCreateWithAllocator out-handle to rdx; SharpEmu says r8 (args[4]).
  * FAMILY: all 3 faults = dispatcher (0x1100xxxx) handlers iterating
    registries with one half-built entry, ~5 s after "LevelDocument Loaded:
    ui_pause_next [pause_menu]"; +0x33f335 recurrence says the APR
    completion-ordering family is NOT fully closed.
  * NEXT GATE (priority): (1) map low 4GB guest VA no-access to surface true
    fault sites (diagnostics); (2) audit pause-menu load completion publish
    vs registration finish (APR follow-up); (3) identify guest_thread 21 —
    chronic >3s holder of mutex 0x300944e00 in every run, producer-wedged
    suspect.
  * TEMP-DIAG left in tree (marked, env-gated): RAEEN_TRACE_NGS2 arg dump in
    libsce_media.rs; r12–r15 print + RAEEN_DUMP_VOICE_LIST walker in
    dispatch.rs. Remove or adopt deliberately.

- ASTRO.BOT post-APR-fix verification run (2026-07-22, 147 s release
  `--run-eboot`, working tree; artifact
  `scratch/astro-apr-fix-20260722.out.log`):
  * OLD FAULT GONE: the pause-menu poisoned-object fault at eboot+0x33f335
    (`r13 = 0xffffffffffffff2f`) did NOT recur through the whole LevelDocument
    load. The APR fix was the only change since the faulting run — the
    stale-record re-fire / silent zero-fill was the poisoning mechanism.
  * NAME-THE-MISS: **zero** APR warns fired — every APR fileId resolved. The
    missing-asset-via-APR theory is dead for this title.
  * PAST THE OLD WALL: "LevelDocument Loaded: ui_pause_next [pause_menu]" at
    +52 s; GPU work kept flowing to **36 flips / 1064 draws / 1497 dispatches**
    (vs the old stopping point 18 / 559 / 785) — double the observed frame work.
  * NEW FAULT CLASS SURFACED (3 worker-thread faults, all recovered — runtime
    released held mutexes and the title kept running; "no HLE call returned an
    Orbis error before this fault" on all three):
    1. t43 @ +46 s, module+0xe03f1a: `mov r14,[rax+0x10]` with **rax=0** —
       NULL deref (not the old poison pattern).
    2. t44 @ +49 s, module+0xe47a43: `cmp byte [r15+0x29],0` read fault; chain
       +0xf4082c <- +0xdc2a7e <- +0x10e91 <- +0xdfb602 <- +0xded2d9.
    3. t45 @ +49 s, libc.prx+0x356ba: strcpy byte-loop reading wild
       0xfaab60664; rsi -> "pri_hero"; chain +0xe54d3f <- +0xe53889.
    Same family shape (workers walking object arrays with bad entries), but
    NULL/wild now instead of -0xd1 poison. NEXT RE TARGET: these three sites —
    start with +0xe03f1a (NULL base) and the "pri_hero" strcpy caller.
  * UNLOGGED TERMINATION: process exited code 1 at ~147 s with the log ending
    mid-stream (no shutdown/RESULT/fatal line, err log empty, no device-lost).
    Cause unknown — likely a host-side crash outside the logging path; flag
    for the next run (watchdog / longer timeout).
  * NOTE new missing-NID surface seen at link time: sceAmprApr*Gather/Scatter/
    Map family (7 NIDs) — register fail-soft stubs only if a runtime fault
    names them.

- Tier-0 APR async-load crash fix (2026-07-22; working tree, no commit;
  raeen-hle 354/354 + 1 pre-existing skip, raeen-kernel 32/32,
  raeen-firmware 126/126; hle+kernel clippy `-D warnings` clean; fmt clean on
  touched hunks). Unblocks diagnosis of the ASTRO.BOT pause-menu
  poisoned-object fault (eboot+0x33f335):
  * ROOT CAUSES FIXED (audit-verified): (1) silent zero-fill + fake success
    when `appr_host_path(file_id)` was None — the title parsed an all-zeros
    LevelDocument with no log naming the lost asset; (2) single bare
    `file.read` with no read-exact loop (short reads left stale guest heap);
    (3) `apr_complete_command_buffer` never consumed `ampr_write_offsets`, so
    a re-submit without Reset re-executed stale ReadFile records into
    repurposed guest addresses — a second heap-poisoning mechanism.
  * PORT (SharpEmu GPL-2.0, cited in doc comments + THIRD_PARTY_NOTICES):
    `sceAmprAprCommandBufferReadFile` now reads EAGERLY at record-append time
    (AmprExports.cs:255-293) via a faithful port of
    `TryReadFileToGuestMemory` (AmprExports.cs:748-828): per-APR-id host
    handle cache (new `OrbisKernel::appr_file_handles`), positional
    `seek_read`/`read_at` reads, read-EXACT loop until full or EOF, 1 MiB
    chunks, `bytesRead` recorded in the record @0x20 at append.
    `apr_complete_command_buffer` skips ReadFile records (data already in
    guest memory; AmprExports.cs:578-580), keeps servicing equeue +
    write-address records, and consumes the cb's `ampr_write_offsets` entry
    after completion so stale records can never re-fire.
  * MISSING-FILE SEMANTICS (SharpEmu parity, AmprExports.cs:272-276):
    unregistered fileId → NOT_FOUND (0x8002_0002), guest memory untouched,
    no record appended — the old zero-fill-and-report-success is GONE —
    plus a once-per-fileId `warn!` naming the id (new
    `OrbisKernel::appr_missing_warned` rate-limit set): the name-the-miss
    diagnostic whose absence made the ASTRO.BOT asset loss take days.
  * TDD red→green: `apr_read_file_populates_guest_memory_at_append_not_submit`,
    `apr_completion_does_not_reexecute_stale_readfile_records`,
    `apr_read_file_exact_read_fills_full_destination` (100 KiB file +
    past-EOF short-count case), and
    `apr_read_file_missing_file_logs_and_matches_sharpemu_semantics` (replaced
    the old `apr_read_file_zero_fills_missing_files`, which asserted the
    removed behavior). All 4 failed before the implementation for the
    expected reasons; all pass after.
  * MINECRAFT PARITY (PPSA17221, live rendering title): release build +
    90 s `--run-eboot` smoke (`scratch/mc-apr-eager-20260722.out.log`):
    booted — 4 module_starts, 28 guest pthreads, AGC submissions flowing,
    2924 shader translations, ZERO ERROR lines, zero guest faults, and zero
    APR warns (the name-the-miss warn never fired: Minecraft resolves every
    APR id it reads). Only pre-existing WARNs (NEEDED libSce*, kyty-graphics
    flip packets, AGC version 12). Did NOT reproduce the full 64-submission /
    60-flips baseline inside the 90 s bound (baseline was measured on longer
    runs; activity was still progressing at kill) — success-path parity is
    evidenced but not fully baseline-matched.
  * NOTE: `kernel_eventflag::wait_completes_when_satisfied_else_times_out`
    hangs at HEAD (committed ce11844 blocks forever on NULL timeout while the
    committed test expects ETIMEDOUT) — pre-existing, unrelated; the full
    suite was run with `--skip` for it (independently confirmed pre-existing
    by the concurrent session's entry below).
  * NEXT GATE: re-run ASTRO.BOT past the pause menu — the name-the-miss warn
    should identify the lost LevelDocument asset by fileId; then re-test the
    poisoned-object fault at eboot+0x33f335. If the warn does not fire, the
    poison source is elsewhere (stale-record path already eliminated).


  working tree, no commit; raeen-kernel 30/30, raeen-runtime lib 62/62,
  raeen-hle 354/355 — the 1 non-run is the PRE-EXISTING deterministic hang
  `kernel_eventflag::wait_completes_when_satisfied_else_times_out`, confirmed by
  two older pre-change binaries hanging identically; kernel+runtime clippy clean
  on touched files):
  * TASK 2 (deadlock cascade) — CLEAR WIN. New `OrbisKernel::release_locks_owned_by`
    (raeen-kernel/src/lib.rs) frees a dying thread's mutex ownership + rwlock
    write ownership + rwlock read holds (giving the shared reader count back);
    `LockReleaseSummary` reports the counts. Wired into the FAULTED worker exit
    path only (raeen-runtime/src/thread.rs ~L631: `if result.is_err()` after
    `dispatch::run`). Previously only the guest-called
    `sceKernelDebugRaiseException` path released locks (mutexes only); a
    host-detected VEH fault (`Err(RuntimeError::Faulted)`) released nothing, so
    every waiter on a dead worker's lock hung. MEASURED headless (90 s
    `--run-eboot`): "stuck >3s" deadlocks 21 (baseline) -> {6, 11, 4} across three
    post-fix runs. Log shows the fix firing: "guest worker faulted holding locks;
    released them ... guest_thread=43 mutexes=1".
  * TASK 1 (guest-memory zero-init) — implemented, correct, but NEGATIVE for
    fault reduction. `GuestArena::map_at` now zeroes the FULL re-map fast path
    (a title re-mapping a fixed VA it already holds fully backed — the Orbis
    map contract returns zeroed pages, but Windows only zeroes a page on its
    first commit) via `backed_ranges_outside_core` + `zero_reused_ranges`
    (raeen-runtime/src/arena.rs). Partial-overlap extends still PRESERVE bytes
    (unchanged; `map_at_extends_an_overlapping_external_mapping` still green).
    mmap/flexible/allocate paths audited: already zeroed by construction (fresh
    OS reservation -> Windows zeroes first commit; munmap MEM_RELEASEs), so no
    change needed there; plain heap `alloc`/`grow_into_tail` left non-zeroed
    (malloc). MEASURED: fault count unchanged 3 -> 3.
  * WHY Task 1 didn't move faults (recent-HLE ring says "no HLE call returned an
    Orbis error before this fault" for all three; ring is per-thread, full ring
    DEBUG-gated). The 3 stable post-fix faults are NOT reused-non-zeroed memory:
    - eboot+0xe03f1a (read 0x10): `mov rax,[r14+0xe8]`=0 (NULL) then `[rax+0x10]`
      — an object whose pointer field is 0/UNINITIALIZED, not stale garbage.
      Zeroing memory can't fix a field that needs REAL data written.
    - eboot+0xe47a43 (read 0x29): linked-list `next` = 0, walked past — same
      uninitialized/zero signature.
    - eboot+0x102a16ba (read 0xfaab60664): libc strcmp on a garbage char* in an
      entity-name table — genuinely stale, but NOT cleared by map_at zeroing, so
      that table lives in game-managed memory (mspace/heap on flexible memory, or
      a map->munmap->map range already Windows-zeroed), not a kernel map we own.
    The one baseline fault whose signature matched reuse (eboot+0x33f335, read
    0xffffffffffffffff — "garbage 0x..2f survived a null check") did NOT recur in
    3 post-fix runs, but that is within run-to-run noise (faults are
    non-deterministic; a GIL halves them) so no claim is made. NEXT root-cause
    lead is the ledger's existing APR async-load suspect: objects with 0/null
    pointer fields (0xe03f1a, 0xe47a43) look UNPOPULATED — check whether APR
    submit actually executes ReadFile/WriteAddress records into guest memory for
    async-loaded objects, and find who writes [obj+0xe8]/list `next`.

- ASTRO.BOT post-audio "stall" diagnosed as mspace provider regression and
  FIXED (2026-07-21; working tree, no commit; firmware 126/126):
  * Exact wait classification with TRACE_COND + TIME_HLE: workers use the same
    expected idle primitives/callers as the prior good run — cond waits at
    eboot+0x11dd1 / +0x2b9119, job semaphores, and two audio event flags. Main
    was not waiting: it spent 11.3 s inside 287,716 HLE mspace malloc calls;
    boot stats counted 263,995 mallocs and 78,780 frees in the first 30 s.
  * Regression comparison: `astro-cond-trace3` used the shipped libc allocator,
    reached Resident Load at +17 s and first flip at +21 s. The current dirty
    libc aliases made the default force-HLE mspace override capture that family;
    a 140.6 s run never reached Resident Load. This was deterministic routing,
    not timing or a missing condvar wake.
  * A/B proof: `RAEEN_FORCE_HLE_MSPACE=0` on the same executable restored
    Resident Load + flips. TDD then changed the policy to default LLE with
    explicit `=1` HLE opt-in (`shipped_libc_mspace_prefers_lle_by_default_and_hle_is_opt_in`,
    red then green). This keeps each stateful allocator family coherent.
  * Fresh default proof after release rebuild, variable UNSET: Resident Load,
    40 observed flips, zero guest faults/asserts in 45 s. Active flip window was
    about 5.4 fps, not 120 fps; no performance claim. NEXT remains the prior
    post-pause-menu poisoned-object fault (or the first earlier fault on a
    longer default run), then GPU throughput.
  * Verification: `cargo test -p raeen-firmware` (110 unit + 10 coverage +
    3 homebrew + 3 transitive = 126), firmware clippy `-D warnings`, firmware
    fmt check, release GUI build. Artifacts: `scratch/astro-post-audio-sync-trace-20260721.out.log`,
    `scratch/astro-lle-mspace-ab-20260721.out.log`, and
    `scratch/astro-default-lle-mspace-proof-20260721.out.log`.

- ASTRO.BOT zero-size global-heap false OOM FIXED (2026-07-21; working tree,
  no commit; hle 346/346, hle clippy clean):
  * Fresh release trace reproduced eboot+0x222f / `Memory.cpp:69` immediately
    after material setup. The fatal HLE call was measured exactly:
    `sceLibcMspaceMalloc(msp=0x300000000, size=0)` (caller eboot+0x27dfac)
    returned NULL, which the title interpreted as `Out of Global Heap Memory`.
  * TDD: new `mspace_malloc_zero_returns_distinct_non_null_allocations` failed
    on the NULL result, then passed after the malloc wrapper normalized a
    zero-byte request to a unique one-byte allocation. Direct memalign(0) and
    realloc(ptr,0) semantics remain unchanged.
  * Fresh release proof: 140.6-second `--run-eboot` run had zero ASSERT/fault/
    RESULT lines and passed the old failure into 47 guest threads plus audio
    initialization. `RAEEN_TRACE_FLIP` / `RAEEN_DUMP_FRAMES` measured 0 flips
    and 0 frames, so no FPS claim is possible yet.
  * NEXT RED GATE: post-audio synchronization stall. Repeated STALL_DUMP
    snapshots show no in-flight HLE while nearly every guest host thread is in
    `WaitOnAddress`; guest console stops after the last material replacement.
    Identify the main/worker wait dependency before returning to the prior
    Resident Load / pause-menu poisoned-object fault. Run artifacts:
    `scratch/astro-mspace-zero-{confirm,fixed-120s}-20260721.out.log`.
  * Verification: focused red->green test; `cargo test -p raeen-hle` 346/346;
    `cargo clippy -p raeen-hle -- -D warnings`; release GUI build. Workspace
    fmt check is red on unrelated concurrent formatting drift in graphics,
    core, GUI, and the already-dirty HLE files.

- ASTRO.BOT condvar-stall RESOLVED + three fatal-stall classes fixed; title now
  loads the PAUSE MENU (2026-07-21; hle 344, kernel 23, clippy 0, fmt clean):
  * RE of the job-post path (old task: trace module+0xdefa40): it is the
    title's 0x1100xxxx command dispatcher (error base 0x8A8400xx, reentrancy
    counter at [0xE2C8B58]). Main's spin loop issues cmd 0x11000000
    ("query status", handler 0xDF02EC → 0xDED4E0) and polls response
    buf+0x88, which mirrors `[[0xE7F5BE0]+0x288]` — the subsystem completion
    flag. Progress floats at +0x23C/+0x234 of the same state object.
  * FRESH TRACE (RAEEN_TRACE_COND/STALL_DUMP, 100s): the material-shader stall
    from 2026-07-19 is GONE — 4000 cond signals (cap), job system live,
    workers processing loads. Condvar starvation is no longer the blocker.
    The run instead DIED on unresolved-import stubs (t27:
    libSceNpUniversalDataSystem 0x31f0dbfb83659fae; t49: libSceAmpr
    0x8339dd96d044cd67). t27's r8 held a job-assert string
    ("m_item->GetStatus() == kRunning || kFini…") — the assert handler calls
    Np UDS telemetry; the stub, not the assert, killed the thread.
  * FIX 1 — libSceNpUniversalDataSystem EventProperty family (12 NIDs):
    SharpEmu-faithful SetString/SetArray (null+readability probe → OK),
    scalar Set{Int32,Int64,UInt32,UInt64,Bool} (object-only check, payload
    is by-value), CreateEventPropertyObject/DestroyContext/Terminate = OK.
  * FIX 2 — sceAmprCommandBufferGetNumCommands: real port of SharpEmu's
    CommandCount (new `OrbisKernel::ampr_command_counts`; zeroed in
    write_cb, bumped in append_record, dropped in dtor). cb==0 → EINVAL,
    untracked → EFAULT.
  * FIX 3 — NEW libSceAudioPropagation (31 NIDs, libsce_audio_propagation.rs):
    faithful SharpEmu `AudioPropagationExports` port — QueryMemory writes
    {1 MiB, 256 B} at rsi, everything else OK. Measured fault: QueryMemory
    at "GAME: Resident Load end".
  * BOOT NOW (150s trace, ZERO unresolved-import faults): main() → GpuDevice
    → all MaterialPackedShaderBinaries passes → PlayGo finished → Resident
    Load end → Armadillo/Transition loads → haptics →
    **"LevelDocument Loaded: ui_pause_next [pause_menu]"** with live GPU
    flips (present_index 18; DCB 559 draws / 785 dispatches; ACB async
    compute submitting).
  * NEW BLOCKER (next RE target): main-thread fault eboot+0x33f335
    `mov r15,[r13+0xd0]` with r13 = [rax+0x70] = 0xffffffffffffff2f (-0xd1)
    while iterating a loaded-object array ([rbx+0x20]→[r14+r12*8]) right
    after the pause-menu LevelDocument load. Null check at +0x33f330 passes,
    so the field is POISONED, not zero. Prime suspect: APR async-load
    completion — `hle_apr_read_file` appends records the kernel "completes
    at submit"; if submit never executes ReadFile records into guest memory,
    async-loaded objects keep poison/uninitialized fields. NEXT: find what
    writes [obj+0x70] (x-ref the async-load completion callback), and check
    whether APR submit executes ReadFile/WriteAddress records for real.
    Run artifacts: scratch/astro-cond-trace{,2,3}-20260721.log.

- M3 INFRASTRUCTURE PROOF (2026-07-20; gate remains OPEN) — pad + CPU-2D draw +
  flip + present-from-guest-memory, all exercised by a running synthesized
  guest. This is useful end-to-end coverage, but the acceptance-gate explicitly
  forbids completing a milestone from a synthetic-only path. A toolchain-built
  interactive 2D homebrew driven through the Shell is still required.
  * present-from-guest-memory (the real feature): a flip whose display buffer has
    NO GPU-drawn render target now builds the presented frame by reading the guest
    bytes at that address as pixels, using the registered VideoOut attribute
    (width/height/pitch/format). LINEAR + 32-bit RGBA/BGRA (A8B8G8R8 / A8R8G8B8,
    _SRGB+_UNORM) supported; other tile modes/formats = rate-limited warn + skip
    (never faked). GPU-drawn target at the flip address still wins; ordered after
    the most-content census so GPU titles (scanout filled by uncaptured copy/DMA)
    do not regress. SharpEmu VulkanVideoPresenter.cs:1643-1660
    (GuestImageWantsInitialData) cited. Files: `raeen-core::subsystems`
    (`ScanoutDescriptor` + trait sig), `raeen-gpu::agc_exec`
    (`present_from_guest_memory`, `present_flipped` fallback, new field),
    `raeen-hle::libsce_video_out` (`hle_submit_flip` builds+threads the descriptor).
  * acceptance test `crates/raeen-runtime/tests/m3_interactive_2d.rs`: hand-asm
    homebrew runs through the REAL LM1 path (`execute_linked`) — calls
    `scePadReadState(1,&pad)`, loads buttons, CPU-fills a 4x4 linear RGBA8 buffer
    white/black by input, calls `sceVideoOutSubmitFlip(1,0,..)`, reads pixel0 to
    RAX. Asserts: guest wrote different pixels A(neutral=black) vs B(Cross=white)
    via RAX; flip count advanced each run (1 then 2); `session.last_image()`
    reflects the CPU-drawn content and differs A vs B. Guest actually executes
    (not the test calling HLE handlers).
  * Tests: hle 144, gpu unit +1 (`present_scanout_reads_cpu_drawn_pixels_from_guest_memory`),
    runtime m3 1/1; `-p raeen-hle -p raeen-gpu -p raeen-runtime` all green; clippy
    clean on touched lines; fmt clean.
  * NOT regressed: existing present_scanout GPU-drawn/splash tests still green
    (signature took a new `Option<ScanoutDescriptor>` arg, all 4 impls updated).

- ASTRO.BOT boot stall DIAGNOSED (2026-07-19) — it is NOT a missing symbol and
  NOT a deadlock in our primitives; it is the title's own job system never being
  woken. Three reusable diagnostics were added to get here:
  * `sceKernelCreateEventFlag` now logs the flag's **name**. That immediately
    identified every flag ASTRO parks on as AUDIO: `sndz_stream_task_notify`
    (h=2), `SceSndzRenderNotify` (h=4), `SceSndzUpdateNotify`,
    `SceSndzAudioOutBounce*`. Those waits return ETIMEDOUT (0x8002003c) on the
    50 ms slice forever — idle audio, NOT the boot blocker.
  * STALL_DUMP now includes the **guest console tail**. A hung title never
    reaches the end-of-run console dump, so its own log was invisible. ASTRO's
    says it is in `Setup MaterialPackedShaderBinaries` / `Material [...] is
    Replaced` — engine material+shader setup — and the console is **frozen at
    3086 bytes across 8 dumps over 200 s**, so it is genuinely stuck there, not
    progressing slowly.
  * `RAEEN_TRACE_SPIN=1` reports the guest caller of each `sceKernelUsleep`
    once. ASTRO's main thread spins at **module+0xdd8e40**:
    `cmp qword [rbp-0x5d8],0; jne exit; call usleep(1ms); call module+0xdefa40;
    test eax,eax; jns loop` — i.e. *sleep until a worker sets my completion
    flag*, polling an internal (non-import) helper.
  * SIGNATURE: 103 `scePthreadCondWait` vs **1** `scePthreadCondSignal` and 0
    broadcasts — the worker pool parks and is never woken, while the producer
    (t1) sleeps waiting for it. 44 guest threads live. This is the SAME shape as
    the open Dragon Ball task-drain (task #6), so it is likely one root cause.
  * RULED OUT: async IO (no `sceKernelAio*` calls at all; 12 opens/5 reads all
    synchronous) and VideoOut (only `sceVideoOutOpen` — the title never reaches
    a flip/vblank, so the display path is not what it waits on).
  * NEXT: find who is supposed to signal that condvar / set `[rbp-0x5d8]` —
    trace the guest job-post path from `module+0xdefa40`, and check whether a
    worker is blocked in a wait we satisfy incorrectly rather than never.

- STALL CLASS NARROWED — new cross-call condvar-starvation diagnostic names the
  blocked threads for the first time (2026-07-19; hle 280 green, clippy+fmt clean).
  * NEW DIAGNOSTIC (`pthread_cond::note_wait_outcome`, `RAEEN_TRACE_COND`): the
    pre-existing >3s check could never fire, because an infinite `cond_wait`
    deliberately returns after a 10 ms slice as a permitted spurious wakeup — so
    no single call is ever long. Starvation is a *streak* of calls that never
    observe a generation change. Now tracked per `(cond, thread)` and reported
    once, with the guest thread's NAME.
  * UNTIL DAWN (UE5) — the sharpest picture yet. 31 named guest threads incl. the
    full RHI set (`AgcSubmissionThread`, `AgcCleanupThread`, `AgcInterruptThread`),
    `IoDispatcher`, `IoService`, IOThreadPool, Foreground/Background workers.
    STARVED (~650 re-waits each, ZERO genuine wakes): main(t1) on 0x10081375a58,
    `AgcSubmissionThread` on 0x1008137a978, `AgcCleanupThread` on 0x1008137b488,
    `IoDispatcher`, `SystemEventGatherThreadSony`, `OutputDeviceRedirector`.
    Main's event is in the SAME allocation block as the two AGC events.
    **NO `RenderThread`/`RHIThread` exists, and ZERO libSceAgc/VideoOut calls are
    ever made** — UE5 starts the render thread during engine init, so init never
    completes. The engine created its RHI threads and then stopped before using
    the GPU API at all.
  * RULED OUT this round (all verified correct, not the bug): `cond_wait` DOES
    release the guest mutex (`mutex_unlock_for_cond`) and reacquire; `WaitEqueue`
    caps unbounded waits and returns a timeout so the guest re-waits;
    `sceKernelStat` ENOENTs are ~15/709 (normal UE optional-file probing); the
    missing `.uproject` is cosmetic (cooked UE titles don't ship it).
  * LATENT HAZARD FOUND: `libkernel.rs` still defines no-op `hle_pthread_cond_wait`
    /`_signal` stubs that are silently shadowed only because `pthread_cond::register`
    runs AFTER `libkernel::register` and `HleRegistry::register` is last-write-wins.
    Reordering registration would deadlock every title. Delete the dead stubs.
  * ASTRO.BOT equivalent: stalls in `Setup MaterialPackedShaderBinaries`, main
    spins at module+0xdd8e40 polling internal dispatcher module+0xdefa40
    (id space 0x1100xxxx, error 0x8a84006e — in no reference catalogue).
  * NEXT: find what signals main's event. For Until Dawn that means tracing UE5's
    PS5 RHI init handshake — which thread is expected to post to
    cond 0x10081375a58 before `StartRenderingThread`.

- UNTIL DAWN boot traced deep into UE engine init (2026-07-19; workspace tests
  17 suites / 0 failures, touched-crate clippy + fmt clean):
  * It is NOT stuck on content discovery. With `RAEEN_LOG="warn,raeen_hle::libkernel=debug"`
    it touches **647 distinct `/app0` paths** — the whole UE config cascade
    (engine/project/platform `.ini`s, DataDrivenPlatformInfo, plugins, mods,
    internationalization, localization) — so the VFS, path mapping and file IO
    are working. The 92 GB install is complete (`bates/content/paks/` holds
    `global.utoc`, `pakchunk0-ps5.{pak,ucas,utoc}`, …).
  * The 8 failing opens are all OPTIONAL probes (`bates/saved/paks`,
    `engine/content/paks`, `bates/intermediate/shaders/tmp`, `deepfiles`,
    `/app0/..`) — normal UE discovery, not the blocker.
  * `sceKernelAllocateMainDirectMemory` returns EAGAIN 18× — also NOT a bug:
    `hle_allocate_direct_memory` documents that titles "allocate in a loop until
    ENOMEM and size their pools from the total". The shrinking retry ladder
    (0xd6000000 → 0xd600000 → 0xd60000 → 0xd0000) is that probe. ~13.8 GiB of
    the 14.3 GiB pool is handed out before the failures start.
  * FIXED along the way (real defects, neither was the blocker):
    `sceKernelAvailableDirectMemorySize` reported the whole window and ignored
    `direct_memory_allocated`, so it over-reported free memory to a title sizing
    its pools; and `scePthreadCondDestroy` was bound to a no-op sharing
    `CondInit`'s handler, leaving destroyed conds' state (and waiter generation)
    behind — now bound to the real `hle_cond_destroy`.
  * REMAINING BLOCKER (unsolved): after the config cascade every thread parks.
    Main + `AgcSubmissionThread` + `AgcCleanupThread` + `IoDispatcher` +
    `SystemEventGatherThreadSony` all starve on distinct conds, all entered
    through the same wait wrapper at module+0x1236d, all starting at the same
    moment (~548 waits / 8 s each). No `RenderThread` is ever created and no
    libSceAgc/VideoOut call is ever made. So UE aborts or blocks somewhere
    between config load and `RHIInit`/`StartRenderingThread`.
  * STACK WALK DONE (`guest_stack_chain` in pthread_cond.rs, in the STARVED
    report): each starved waiter's engine-level frames are now recovered.
    Main thread: `+0x1236d <- … <- +0x18d4ebd <- … <- +0x8ef82b`.
    AgcSubmissionThread: `… <- +0x4f2f1c0 <- +0x74ee878`.
    AgcCleanupThread:   `… <- +0x4f30055 <- +0x6360f88`.
    Background workers: `… <- +0x1b2eb <- +0x74d3c90 <- +0x1ae41 <- +0x74d8600`.
    Decoding main's frame at module+0x18d4ebd shows two `mov ecx,4; mov rdi,r15;
    call module+0x407a0` sites — and 0x407a0 and 0x1236d are both UE's OWN
    wrappers (real prologues, not PLT thunks), so the chain stays inside UE's
    event machinery rather than reaching an emulator boundary.
  * HONEST STATUS: four levels of RE in, every frame is UE-internal and no
    emulator gap is implicated. Cracking this needs UE5 source/symbols or a
    much longer RE effort — it is NOT a missing-import or broken-primitive
    problem. Do not re-derive the above; start from these frames.

# Raeen session progress ledger

- ASTRO.BOT REAL-FRAMES PUSH round 1 (2026-07-20, four parallel subagents +
  main session; all gates green: workspace clippy -D warnings, fmt, hle 320 /
  kernel 110 / gpu 128+18 / firmware suites):
  * BOOT REGRESSION FOUND FIRST: the title no longer reached GPU init — it
    faulted on unresolved `sceKernelPread` ~8s in (boot had progressed into a
    new asset-streaming path). Implemented real VFS pread (positional read,
    cursor untouched: filesystem/mod.rs) + HLE under SCE/POSIX names. Next
    fault: libSceJson2. `--missing-nids` preflight then showed 322 distinct
    unresolved NIDs / 37 libraries — the milestone "zero unresolved" claim
    was about the RUN path, not the static import set.
  * libSceJson2: all 54 NIDs implemented (libsce_json.rs 385→~1540 lines;
    serde_json-backed Parser::parse, guest-anchored Object/Array/String/
    iterator model, SharpEmu error codes; sret iterator ABI). hle suite 320.
  * libkernel + small libs: 45 NIDs — real timed rwlocks, Getprio/Setprio
    bookkeeping, nanosleep, DirectMemoryQuery over the recorded allocator
    regions, ConfiguredFlexibleMemorySize, 7 APR resolve variants + GetFileStat/
    GetFileSize (real VFS), VideoOut gamma/adjust/ChangeBufferAttribute2/
    IsFlipPending, Pad close/info/trigger, Rtc RFC3339 format+parse (real),
    UserService accessibility/presets, LibcInternal MspaceRealloc (real
    dlmalloc semantics)/_ZdlPv/_Stoul/__cxa_*/Need_sceLibcInternal (DATA
    symbol on the HLE data page), Sysmodule unwind info, NpGetAccountIdA,
    SaveDataTransferringMount (fallback error), GetOpenPsId. CAVEAT: the four
    APR *ForEach variants resolve+register but do not write through unverified
    trailing args (loud warn) — revisit if a title reads their out-params.
  * libSceAgc: all 12 command-buffer builders ported dword-exact from SharpEmu
    (AcbAcquireMem/WaitRegMem/DmaData/CopyData/markers, CbSetShRegistersDirect
    sort+coalesce, CbDispatchGetSize/NopGetSize, DcbDrawIndexIndirect/
    StallCommandBufferParser/GetLodStats). KEY GAP EXPOSED: run.rs (the CP)
    does NOT consume R_DMA_DATA (0x19), IT_COPY_DATA (0x40), IT_GET_LOD_STATS
    (0x8e) — skipped by length. R_DMA_DATA is the likely mechanism by which
    the title fills its VideoOut scanout buffer (task #11's "copy/DMA we don't
    capture") — implementing CP-side DMA_DATA/COPY_DATA execution is the top
    candidate for making the composite reach the screen.
  * STORAGE-IMAGE (UAV) COMPUTE SUPPORT LANDED (front #3, task #19):
    prepare_stage_binding now splits sampled vs storage T#s (per-array index
    rewriting, unit-tested), dispatch_compute creates/binds R8G8B8A8_UNORM
    STORAGE images (staging upload, GENERAL layout, post-fence readback), and
    dispatch_direct writes image bytes back to guest at each storage T#'s
    base40 (linear). Real-Vulkan round-trip test green on the Radeon 760M.
    Gaps: UAV seed reinterprets 32-bpp formats as RGBA8 (fine while shaders
    fully overwrite), non-32-bpp seeds zero-fill with one loud warn, no
    re-tiling on writeback, graphics-stage storage images still a named error.
  * Remaining top missing libs (fail-soft candidates, only implement on
    measured fault): libSceAmpr 47, AudioPropagation 22, Http 19, NpWebApi2
    18, VoiceChat 17, AvPlayer 12, NpUniversalDataSystem 12.

- BOOT SPLASH — "why does SharpEmu show the ASTRO.BOT splash and we don't"
  ANSWERED + CLOSED (2026-07-20). SharpEmu's splash is NOT title rendering:
  its `PngSplashLoader.cs` decodes the package's `sce_sys/pic0.png` and
  presents it host-side until the title flips a buffer with real drawn content
  or calls `sceSystemServiceHideSplashScreen` — the same thing a real PS5's
  system software does. Implemented the equivalent:
  * `raeen-gui/src/splash.rs` (NEW): decodes `sce_sys/pic0.png` beside the
    eboot (`image` workspace dep) and stages it via
    `AgcGpuSession::set_pending_splash` before entering the guest; wired into
    BOTH launch paths (Shell `launcher.rs` + CLI `--run-eboot`), staging
    `Some`/`None` every launch so a previous title's splash cannot leak.
  * `raeen-gpu/agc_exec.rs`: `splash` field on the session, cloned from the
    pending slot in `GpuProcessSession::create` (created inside
    `execute_process`, after the launcher's last chance to touch it).
    `last_image()` presents the splash while it is up. It comes down on
    `hide_splash()` or a flip whose address has real drawn content — but NOT
    on the most-content present fallback, which can surface a bare cleared
    target (the CLEAR_COLOR frame the 2026-07-20 correction exposed).
    Frame dumps bypass the splash, so diagnostics still see title output only.
  * `sceSystemServiceHideSplashScreen` upgraded from `hle_ok` to a real
    presentation transition (`ctx.gpu.hide_splash()`; trait default no-op on
    `GpuSubmissionSubsystem`).
  * VERIFIED IN THE REAL SHELL (verify skill): launching ASTRO.BOT presents
    its actual key-art splash letterboxed, replacing the old flat clear-color
    frame. Tests: gpu 126/126 (incl. 2 new splash tests), hle 290, gui 123 +
    2 new, core 7; touched-crate clippy `-D warnings` + workspace fmt clean.
  * Also fixed in passing: stale `shader_fetch` dump test (asserted 2 files,
    but `dump_spirv` — the coverage-bisect harness input — now writes a third
    `.spv` for the translated shader; assertions updated to cover it), and two
    pre-existing clippy `sort_unstable_by` lints in main.rs CALL_STATS.
  * NOTE: this changes the ASTRO.BOT visual baseline — the presented frame is
    now the splash, NOT title rendering. Any future "did the title render"
    check must use frame dumps or the controlled clear-colour test, never the
    Shell image.

- MILESTONE: **all 5 installed titles load and execute with ZERO unresolved
  imports** (2026-07-19). Workspace `cargo test` 0 failures; touched-crate
  clippy + fmt clean. Every title previously died on a missing symbol.
  * ASTRO.BOT additionally cleared its whole `sce::Json` boot path. Implemented
    `libsce_json.rs` properly rather than as stubs: `Value` and `String` payloads
    live in host-side maps keyed by the guest `this` pointer (the C++ layout is
    opaque, so a ctor/`set` that only returned `this` would leave the guest
    reading garbage back out). Covers Value ctors + `set` for
    bool/long/ulong/double/`const char*`/`const Value&`/`const String&`/
    `ValueType`, String ctor/dtor, and both Itanium C1/C2 + D1/D2 variants.
    `set(double)` is the first real consumer of the new float-argument channel.
    Then: Share content-event callbacks and the notice-screen skip flags.
  * **HONEST LIMIT — this is "loads and runs", NOT "boots to a menu."** All five
    survive a 45-50 s bounded run instead of dying, but they then sit: Until Dawn
    and Dragon Ball park in `sceKernelWaitEqueue`, and STALL_DUMP reports 0
    registered guest threads at dump time even for titles observed creating 18-20
    threads earlier in the same run. Nothing here demonstrates a rendered frame or
    a reached menu. The next frontier is the WAIT/STALL class (tasks #6/#8) —
    why the event-queue waits never wake — not missing symbols.


- ALL-TITLES boot sweep — 4 of 5 installed games now load+execute with **zero
  unresolved imports** (2026-07-19; hle 279, firmware 108/10/3/3, runtime 48/43
  green; touched-crate clippy --no-deps + fmt clean; re-measured after every fix):
  * Method: run every installed eboot headlessly (`--run-eboot`, 45 s bound),
    read the reported unresolved import, fix, rebuild, re-measure. Twelve rounds.
  * **PROVIDER-ALIAS BUG CLASS — the dominant finding.** Resolution is
    provider-aware (`ModuleRegistry::resolve` keys on the importing symbol's
    library), and several functions were **already implemented** but registered
    under a library no title names. Each cost a game its boot:
    - `gettimeofday`/`usleep` — registered `libScePosix`, imported `libkernel`
      (Minecraft). Joined `clock_gettime`, fixed earlier the same way.
    - `sceShareInitialize`/`sceShareSetContentParam` — registered
      `libSceShareUtility`, imported `libSceShare` (Dragon Ball). An earlier pass
      had DELETED the `libSceShare` spelling as a "wrong name"; it is real.
    - `in6addr_any`/`in6addr_loopback` — the HLE data page registered ALL its
      exports under `libkernel`, but Minecraft imports the IPv6 constants from
      `libSceNet`. Now registered under both. (The stale comment there claimed
      "resolve is by NID and ignores the declaring module" — it is not.)
    - `sceAgcDriverSubmitAcb`/`sceAgcDriverAddEqEvent` — `libSceAgc` vs
      `libSceAgcDriver` (ASTRO.BOT).
  * New implementations (each measured as a title's live blocker): libc
    `_init_env`, `__cxa_guard_acquire/release/abort` (Itanium static guards,
    host-mutex + condvar so two guest threads cannot double-construct a static),
    `strcspn`, `wcslen`, `wcscpy`, `strtok`, `vsnprintf` (with a real SysV
    `va_list` walker — `GuestVaList` — reusable for the whole `v*printf`
    family), `sincosf`; libkernel `scePthreadMutexattrSetprotocol`,
    `sceKernelMapNamedFlexibleMemory` (the public spelling was missing while
    `...Internal` existed), `sceKernelAvailableDirectMemorySize`,
    `sceKernelIsTrinityMode`; `sceSystemServiceGetHdrToneMapLuminance`;
    `sceNpWebApi2PushEventCreateHandle` (refuses, per this module's documented
    offline policy); libSceRudp init/event-handler/IO-thread; `sce::Json::Value`
    ctor/dtor.
  * **NEW RUNTIME CAPABILITY — floating-point arguments.** HLE handlers only ever
    saw integer registers, so any function taking a `float`/`double` (SysV passes
    them in XMM0-7) was unimplementable without guessing. Added
    `HleContext::float_args` (low 64 bits of XMM0..XMM7, filled from the trap
    CONTEXT's `FltSave.XmmRegisters`) plus `float_arg_f32/f64` helpers. Two
    construction sites, no change to the `HleFunction` signature. `sincosf` is
    the first consumer; the whole libm surface needs it.
  * **STATE (honest).** Minecraft, Until Dawn, A Plague Tale Requiem and Dragon
    Ball all now run the full 45 s with no unresolved import — but they are
    **stalling, not finishing a boot**: Until Dawn and Dragon Ball park in
    `sceKernelWaitEqueue`, Minecraft's last call is `pthread_mutexattr_destroy`,
    and none spawn many threads. Clearing the import walls moved them from
    "dies instantly" to "runs and waits"; the remaining work is the WAIT/stall
    class (tasks #6/#8), not missing symbols.
  * **ASTRO.BOT** went furthest: past AGC init, **18 guest threads**, now blocked
    on the `sce::Json` C++ library — 45 distinct mangled imports (Value ctors for
    every type, `set*`, `referArray/Object/Value`, `serialize`, Array +
    iterators, Object, String). That is a library port, not a stub: doing it
    honestly needs a host-side object model keyed by the guest `this` pointer so
    values round-trip instead of reading back garbage. Tracked as task #14.


- ASTRO.BOT AGC wall — 4 blockers cleared, title now reaches shader creation
  (2026-07-19, working tree; hle 277 + firmware 108/9/3/3 green, clippy+fmt
  clean; measured against the retail title after each fix):
  * **Repeatable technique established** for unknown NIDs: (a) recover the name
    from SharpEmu's `aerolib.bin` catalogue — format is
    `[u8 len][encoded-NID][u16 len][name]`, built from `scripts/ps5_names.txt`,
    so `grep -a -o -P '<encodedNid>.{0,50}'` yields the name; (b) get the caller
    from the runtime's guest-stack return-addr chain; (c) `--dump-vaddr` the
    return address to read the ABI off the post-call instructions;
    (d) `--resolve-got` any PLT thunk to name what the branch calls next.
  * `sceAgcGetIsTrinityMode` (NID 0x05f0436466ed8bb0, name recovered from
    aerolib; implemented by NO reference emulator). "Trinity" = PS5 **Pro**;
    Raeen emulates a base PS5 → returns 0. ABI proven from the call site:
    `test eax,eax; jnz` on the return, so the flag comes back **directly in
    EAX**, no out-param. The ZERO branch is the one that goes on to call
    `sceAgcDriverSubmitDcb` — i.e. base-PS5 is the GPU-submitting path.
  * **Two provider-alias bugs** (same class as the clock_gettime fix):
    `sceAgcDriverSubmitAcb` (0x812467afbf45f2d4) and `sceAgcDriverAddEqEvent`
    (0xc36ac98660fe76c1) were IMPLEMENTED but registered only under `libSceAgc`,
    while Gen5 retail imports them from **`libSceAgcDriver`** — provider-aware
    resolution left them unreachable. Bound both retail identities explicitly
    (the file already had this precedent for SubmitDcb).
  * Unnamed `dolOmWH+huQ` (0x76894e9961fe86e4) — in no catalogue. Call-site ABI:
    `f(void* out, a, b)`; caller pre-zeroes the 16-byte `out`, IGNORES the
    return, reads `*out` and branches on NULL, and that NULL branch is graceful
    (zero-fills the owning struct and continues). So it reports "no object":
    returns 0, leaves `*out` as zeroed, warns once. Deliberately does NOT
    fabricate a handle. NOTE `hle_unknown_filler` (used for `qj7QZpgr9Uw`) is
    command-buffer-specific and would have corrupted this caller's local.
  * Tests: provider-aware `ModuleRegistry::resolve` assertions for the AGC
    identities (`sce_agc_get_is_trinity_mode_...`,
    `agc_driver_entry_points_resolve_from_the_libsceagcdriver_provider`).
  * **Measured trajectory**: first-AGC-import → event-queue registration
    (AddEqEvent ×many) → **`sceAgcCreateShader`**. NEXT WALL: unnamed
    `fd5Bp5tGTgo` (0x7dde41a79b464e0a, libSceAgc), then the Cb/Acb/Dcb builder
    batch (task #13).


- ASTRO.BOT boot chain advanced — B0 verified + Slice 2 (2026-07-19, working
  tree; hle 108 + firmware 118 green incl. new provider-aware clock_gettime
  tests; release GUI rebuilt; measured against the retail title twice):
  * **B0 VERIFIED on the real title** (Slice 1 init-once): a default release
    `--run-eboot` (NO `RAEEN_SKIP_MAIN_INIT`) shows deps init once
    (`libSceNpCppWebApi.prx`/`libc.prx: calling dependency module_start`), the
    main initializer `deferred to crt0`, and **no `0x7426c00` cycle** — the
    title advances past init in ~4 s. The Slice 1 exit gate is closed.
  * **Slice 2 done — `clock_gettime`**: the title imports it (NID
    0x94b313f6f240724d) naming provider library **`libkernel`**, but it was
    registered only under `libScePosix`; resolution is provider-aware, so it
    was unresolved. Registered the existing thin POSIX adapter
    (`posix_clock_gettime`) under `libkernel` too. Tests: hle
    `clock_gettime_is_also_registered_under_libkernel` + firmware
    `clock_gettime_resolves_from_the_libkernel_provider_the_title_names`
    (through the real provider-aware `ModuleRegistry::resolve`, not a
    provider-blind NidDatabase lookup).
  * **Task #7 (ASTRO.BOT "libc allocator mutex spin") does NOT reproduce** on
    the current build: the title progresses through the libc mutex/allocator
    traffic (last HLE calls are scePthreadMutexLock/Unlock/Init) in ~5 s and
    returns cleanly — no spin. That wall (from an older architecture branch) is
    gone.
  * **NEW ASTRO.BOT wall = libSceAgc command-buffer builders** (GPU init). The
    title now reaches AGC and stops on the first unimplemented import, an
    UNNAMED NID `0x05f0436466ed8bb0` ("BfBDZGbti7A"), with a batch behind it:
    sceAgcCbSetShRegistersDirect, sceAgcCbDispatchGetSize, sceAgcCbNopGetSize,
    sceAgcAcb{AcquireMem,CopyData,DmaData,WaitRegMem,Push/PopMarker},
    sceAgcDcb{DrawIndexIndirect,GetLodStats,StallCommandBufferParser}, +
    another unnamed (0x7dde41a79b464e0a). Next step is implementing these AGC
    Cb/Acb/Dcb builders (SharpEmu `AgcExports`/Kyty Gen5) — the ASTRO.BOT plan's
    M2/AGC territory. `libsce_agc.rs` already carries 25/48; these are new.


- GPU depth/stencil attachments in the Vulkan pipeline (2026-07-19, working
  tree; raeen-gpu 120 lib + all integration green incl. 3 new depth/stencil
  tests on real hw (AMD Radeon 760M); gpu clippy --tests + fmt clean; release
  GUI build green; hle 276 + runtime 48+43 unblocked/green):
  * Completed a half-finished depth/stencil refactor a prior agent left the
    tree non-compiling on (broke the whole workspace since hle/runtime depend
    on raeen-gpu). `render_draw` now returns `DrawOutput { color, depth }`.
  * `offscreen.rs`: `read_back_color`/`read_back_depth` (Option results; None
    for a depth-only z-prepass); `record_and_submit` gained depth image
    transitions (UNDEFINED→DEPTH_STENCIL_ATTACHMENT_OPTIMAL, seed-on-LOAD via
    the upload buffer, →TRANSFER_SRC + copy-out), dynamic-rendering depth +
    stencil attachments, colour-attachment work guarded by `color_output`;
    `Drop` frees the 7 depth resources.
  * Correctness fix (helps the title path too): `VulkanDevice::
    supports_depth_stencil_attachment` queries format features; `create_depth_
    target` fails cleanly on an unsupported format instead of creating a bogus
    image the driver accepts then errors on every use. Measured: AMD exposes
    D32_SFLOAT_S8_UINT but NOT D24_UNORM_S8_UINT — the stencil test tries both.
  * Tests (`tests/depth_stencil.rs`): depth CLEAR readback, depth LOAD (prior
    contents), and stencil CLEAR readback — write disabled so the readback is
    exactly the clear/loaded value, independent of shader z. All pass on hw.
  * Fixed the two broken `render_draw` callers (blend_state/texture_upload
    tests → `.color`) and `draw_translate` (colour-only composite passes
    `color_output: true, depth: None`).
  * OPEN FOLLOW-UP (noted in code): no `depth_state_from_regs` yet — real PM4
    draws still pass `depth: None`, so the attachment is wired+tested but
    DORMANT until the DB_* register decode feeds it (DB_DEPTH_CONTROL / Z_INFO
    / STENCIL_INFO / RENDER_CONTROL, per the `DepthState` doc's Kyty refs) plus
    a depth-framebuffer map keyed by DB_Z_WRITE_BASE.
  * Pre-existing test-clippy debt cleared in-crate while here: `#![allow(
    deprecated)]` on the m2 fixture gate, `#[allow(field_reassign_with_default)]`
    on one nested-array test setup, and an unused `mut` on the compute-seed test.


- ASTRO.BOT boot Slice 1 — process init runs exactly once (2026-07-18,
  working tree; core 7/7, firmware 106/106 lib, runtime 43/43 execute +
  full suite green; touched-crate clippy --no-deps clean; fmt clean; release
  GUI build green; passed a 3-lens adversarial review workflow — 3 confirmed
  findings, all fixed below, 0 runtime-correctness bugs):
  * ROOT CAUSE (measured): retail crt0 `_start` walks the executable's own
    init array itself; the loader ALSO called the main initializer, so
    constructors ran twice — a list-adding ctor then built a cyclic list its
    own walk hung on (t1 at `module+0x7426c00`). `RAEEN_SKIP_MAIN_INIT=1`
    proved the cause; this slice makes the proven path the DEFAULT (no env
    var).
  * `ModuleInitRole::{Dependency, Main}` (firmware `linker.rs`) on
    `ModuleInit`; `load_process` tags dependency initializers Dependency and
    the appended executable DT_INIT Main.
  * `EntryPolicy::{CrtOwnsMainInit, LoaderOwnsMainInit}` (runtime). Shared
    `run_module_initializers` loop: `execute_process` uses CrtOwnsMainInit
    (dependency initializers run, the main one is WITHHELD for crt0);
    `execute_linked` uses LoaderOwnsMainInit (no crt0 -> loader runs every
    initializer, main included; a safe no-op for today's empty-`module_inits`
    callers). `sceKernelLoadStartModule`-owned PRX init unchanged.
  * `RAEEN_SKIP_MAIN_INIT` demoted from mechanism to a one-shot deprecation
    warning; default now takes the proven path.
  * `DiagnosticKind::ModuleInit`: each initializer transition (run / deferred)
    recorded with module name, role, ordinal, and the recorder's stable
    sequence (guest_thread=1) when deterministic diagnostics are enabled.
  * TDD: 4 new runtime tests (dep-runs-once + main-withheld; main-runs-once-
    via-crt0; execute_linked loader-owns-main; diagnostic transition record).
    RED-GREEN PROVEN: temporarily reverting the policy made the two process
    tests fail with main=1 (was 0) and main=2 (was 1), and the diagnostic
    test fail on the role detail; restoring it turned them green.
  * Review fixes: (1) rewrote the now-stale firmware comment that still claimed
    the loader owns the executable's DT_INIT; (2) extracted
    `append_dependency_initializer` + a unit test pinning the Dependency role at
    its single production site (the loop's role assignment was previously
    asserted nowhere); (3) narrowed `EntryPolicy` to `pub(crate)` (it types no
    public signature; `ModuleInitRole` stays pub — it types `ModuleInit.role`).
  * Pre-existing, OUT OF SCOPE: `cargo clippy --workspace -- -D warnings` fails
    on ~a dozen `collapsible_if` sites (nid.rs, dispatch.rs, native_trap.rs,
    thread.rs, hle) — a clippy-1.97 toolchain bump on `if let { if }`, NOT this
    change. Touched-crate `--no-deps` clippy is clean. Flagged as a separate
    cleanup task.
  * EXIT GATE STILL OPEN (needs the user's machine): a default release
    `--run-eboot` of ASTRO.BOT must no longer visit the `0x7426c00` cycle and
    must reach the next guest import (measured to be POSIX `clock_gettime`,
    Slice 2). Unit-tested here; not yet re-measured against the title.

- Architectural consolidation (2026-07-18, commit pending): Raeen-owned
  time/wait/event/VFS/network/GPU-submission contracts are active on the HLE
  boot path; deterministic process diagnostics sequence HLE, wait/wake, event,
  guest-task ownership, and GPU submission; guest memory now exposes validated,
  executable, and GPU-visible capability types; `GuestProcess` explicitly owns
  the active GPU session and drains it before unmapping guest memory; public
  shader metadata no longer re-exports Kyty types; module resolution adds
  strict `HleOnly`/`LleOnly` beside the existing HLE-first default and
  title-supplied LLE preference. Tests: core 6/6, kernel 20/20, firmware 99/99
  + integration 9/9, HLE 271/271, runtime 86/86, GPU 118/118 + Vulkan
  integration 9/9, GUI 120/120.

(Recreated 2026-07-16 — the previous ledger file was absent from the tree;
per-module authority is `docs/reference-port-ledger.md`.)

  * **Blocking WaitEventFlag** (the hot-spin fix): poll-and-instant-ETIMEDOUT
    was built for single-active-execution; with real guest threads it made 5
    threads spin at 100% CPU. Now a shared (Mutex,Condvar)
    `event_flag_signal` — Set/Clear/Cancel notify_all, WaitEventFlag blocks
    with a deadline (NULL timeout = 50ms forever-slice, equeue-style; bounded
    honors requested µs capped at 50ms). Test: cross-thread
    wait_blocks_until_another_thread_sets_the_flag (263/263 hle).
    Measured: the 5 spinners now park at a host wait; t1(main) still stalls.
  * **Dragon Ball stall root-caused to the guest level** (next gate): t1 spins
    at `eboot+0x2bd2151` — a linked-list walk waiting for every task's
    state field (+0x48) to hit 0 (a TASK-QUEUE DRAIN WAIT, yielding between
    polls via call 0x2bbfc80) while all 13 background workers idle in
    scePthreadCondWait. Tasks never reach the workers: the posting/wake chain
    (job scheduler / event / audio callback) is the next investigation. Not a
    sync-primitive bug anymore — the cond waits are correctly blocked.

- Dragon Ball task-drain stall (2026-07-18, measured via RAEEN_STALL_DUMP +
  --dump-vaddr): the 29 threads are Unreal's TaskGraph (Foreground/Background
  Workers + ThreadPool). t1(main) spins at `eboot+0x2bd2151`: a linked-list
  walk waiting for every task's state field (+0x48) to hit 0 (UE
  WaitUntilTasksComplete), yielding between polls via call 0x2bbfc80 — while
  all 13 workers idle in scePthreadCondWait (correctly blocked; the
  blocking-WaitEventFlag fix parked the earlier hot-spinners). **0 AGC calls
  yet** — the stall is pre-RHI. The task posting/wake chain (or a prerequisite
  task's own gate) is the open question: workers are never woken to consume,
  and the main thread's TLS check (scePthreadGetspecific) never flips. The
  "Project file not found" log is cosmetic — the .uproject is not in
  package.manifest, so a real PS5 logs identically.
  Universal-import note: 359 link-time missing NIDs (libSceAgc Cb/Acb/Dcb
  builders, Audiodec, AudioOut2, AudioIn, NP dialogs) — link-time only, none
  called yet on the boot path (the fault-at-call diagnostic names them as they
  become live).

- Dragon Ball boot chain (2026-07-18, commit pending; 262/262 hle, 20/20
  kernel tests, workspace clippy clean):
  * sceKernelAioInitializeParam/Impl (synchronous AIO model), scePlayGoGetChunkId
    (SharpEmu ABI: count-only vs list mode, BAD_POINTER/BAD_SIZE codes),
    sceKernelAddAmprEvent (equeue registration), pthread_create_name_np
    (create + thread_names record), sceKernelClockGetres ({0,1}),
    sceRtcConvertUtcToLocalTime (real host TZ bias via GetTimeZoneInformation;
    windows-sys added to raeen-hle + Win32_System_Time feature),
    sceNetCtlRegisterCallbackV6/GetStateV6 (aliases), sceNetEpoll* (offline
    create/control/wait(≤50ms, 0 events)/destroy),
    sceNpWebApi2CreateUserContext (local-user handle).
  * **Full APR/AMPR subsystem**: sceKernelAprResolveFilepathsToIdsAndFileSizes
    (path list → deterministic FNV-1a ids + sizes, missing → 0xFFFFFFFF/0 and
    batch continues; `appr_files` registry in the kernel), AMPR command-buffer
    record append (ReadFile/KernelEventQueue/WriteAddress per SharpEmu
    `AmprExports` layouts), sceKernelAprSubmitCommandBuffer[AndGetResult] +
    WaitCommandBuffer — synchronous completion (real reads via the registry,
    equeue event triggers, address writes, bytesRead backfill).
  * Measured result: the title boots deep — 29 guest pthreads, engine
    allocators up, sysmodules loading (0x10f/0x95/0x96), shader dirs probed.
  * **Stall (RAEEN_STALL_DUMP, next gate)**: t1(main) spins on
    scePthreadGetspecific (waits a TLS flag); t18-22 hot-spin on
    sceKernelWaitEventFlag → instant 0x8002003c (check timeout handling — same
    class as the f258427 WaitEqueue spin); t2 FAP listener polls WaitEqueue
    (audio stub, task #12); 13 workers idle in CondWait. Tracer noise: errno
    heuristic misfires on size-returning sceAmprMeasureCommandSize* (0x30).

- EUD resolver measurements (2026-07-18, RAEEN_TRACE_EUD evidence):
  cs@0x253a5000 (eud_size_dw=8): direct[t5]=s12 → the Region/EUD base pointer
  IS a user SGPR pair: s12:s13 = 0x29b9e0e0 (captured at dispatch). The table
  there holds the SGPR image (dwords match s0..s5 values — s1/s2/s3/s5 exact,
  s0 differs by 0x100, likely a per-draw offset or a different dispatch's
  SGPRs). Its sharp resources: s0+0, s8+1, and s32+0 — s32 is a register whose
  value comes from a scalar load (the two-level chain: EUD table → loaded
  pointer → descriptor). Existing resolver increments: 1
  (`find_scalar_load_bases` — s_load_dword* scan) and 2
  (`scalar_load_target_address` — pair + offset → guest addr). Increment 3
  (fetch EUD table via Region pointer + evaluate constant-offset loads +
  feed `read_sharp_fields(Some(ext))`) is the bounded next step; the VS
  s[14:15] computed-pointer case needs the full evaluator (SharpEmu
  Gen5ShaderScalarEvaluator.cs, 2,362 lines).
  PM4 indirect-register mis-decode: intermittent race (table memory cycles
  between register-pair and descriptor content between submit and CP drain);
  zero repro in 300s; tolerated by the resilience policy; ledgered.

- Hygiene + indirect investigation (2026-07-18, commit pending):
  All workspace clippy errors fixed (hle getdents chain + dead MOUNT_POINT,
  dispatch.rs unused-mut/collapsible-if/range-contains/then-chain,
  launcher needless-borrow, main sort_unstable_by_key) — workspace now
  clippy -D warnings clean, 0 failed test suites.
  "context register index out of range" (vertex data walked as register
  offsets) investigated with RAEEN_TRACE_INDIRECT: **intermittent,
  race-dependent** — zero repro in a 300s run. Evidence says the title's
  indirect-register tables live in its shader user-data memory region (the
  TRACE_EUD SGPR pointers at 0x29b9d520 name the same buffer), which cycles
  between register-pair tables and descriptor tables between submit and CP
  drain. Resilience policy already tolerates it (skip out-of-file writes;
  next submission rewrites them). No fix without a repro — measured, noted.

- EUD-convergence batch (2026-07-18 late, commit pending; 236/236
  kyty-graphics, 115/115 raeen-gpu, clippy clean):
  * Recompile_Fetch width-mismatch rules, both directions measured (attrib 2
    as 2ch→vec3 fill z=0.0f; as 4ch→vec3 drop w into %temp_float scratch) —
    GCN's (0,0,0,1) default semantics, beyond Kyty (it EXITs on mismatch).
  * Cube textures end-to-end (measured type 11 = 1024x1024x6 skybox, fmt56,
    tile 9): analysis accepts 9/11; spirv.rs emits OpTypeImage Cube from the
    measured T# types (mixed = named error); ImageSample emits vec3 coords for
    cube; decode_texture fetches 6 faces as 6 block grids; offscreen creates
    CUBE_COMPATIBLE image + CUBE view with layer-aware barriers. Tests: cube
    SPIR-V emission + 6-face decode round-trip.
  * input_num = ps_in_control & 0x3f (NUM_INTERP field, AMD layout) — Kyty's
    whole-register read exploded 0x4004 → 16388 bogus truncation; now 4.
  * Shader translation failures converged: **3 remaining, ALL EUD-family**
    (cs eud_size_dw; vs SLoadDwordx2 s14-computed pointer; ps ImageSample
    declined with 0 mapped textures). Next gate is task #4 (EUD/SRT resolver;
    SharpEmu Gen5ShaderScalarEvaluator 2,362 lines; the CS's pointers are
    captured user-SGPRs at dispatch — the tractable first half).

- Shader loop batch (2026-07-18, commit pending; 231/231 kyty-graphics,
  112/112 raeen-gpu, workspace green, clippy clean; two pre-existing
  clippy errors in raeen-hle left alone — not this session's code):
  SDWA src abs/neg now PARSE into operand modifiers (beyond Kyty — its vopc
  path exits on any modifier; operand_load_float already applied FAbs/
  FNegate; measured encodings `v_cmp_lt_f32 s2,|v2|,c` / `v_mul_f32 v2,v4,-v3`
  tested). VCmp F32/I32/U32 families wired from staged (Eq/Ne sign-agnostic
  in the I32 family per Kyty's layout; table 223 rows, 211 impl / 12 NI).
  SLoadDwordx2 EUD x2 path (x4/x8 machinery with n=2; the menu VS's s[14:15]
  is a COMPUTED pointer = the real EUD chain — remains open, task #4).
  `%paramN` declarations now cover the body's exp formats (register
  export_count=1 under-read a param1-writing VS — "id %param1 is used but
  never defined" fixed). buffer_load_dwordx4 + offen (beyond Kyty, opcode
  0x0E; voffset adds into the byte address; measured encoding tested).
  naga rejects Kyty's `%_arr_BufferObject_uint_N` pattern (array of struct
  with runtime array) which the driver accepts — that one test asserts on
  assembly+source instead of naga validation.
  Gen5 VB formats mapped per SharpEmu Gfx10UnifiedFormat as the title named
  them: 77 (RGBA32F, 4.5k draws), 71 (RGBA16F, 3.7k), 23 (RG16_UNORM).
  SW_64KB_S (tile 9, Standard 64KiB — AMD/SharpEmu equation) detiler added
  beside SW_64KB_R_X; the 1937x333 atlas decodes (1.9k draws unblocked).
  Non-2D texture measured: type 11 = CUBEMAP 1024x1024x6 fmt56 tile9 (skybox);
  analysis gates now name type/extent/format/tile (NotImplementedOwned).
  Recompile_Fetch: GCN default-fill for attr-narrower fetches (attrib 2 = 2
  channels feeding a vec3 fetch → z = 0.0f; wider stays a named error).
  Run-to-run failure list went 8 distinct → 3 (cs EUD, ps "unknown operand:
  249", vs SLoadDwordx2-EUD-chain); texture-type/ImageSample-decline did not
  reappear in the last two windows.
  Recompile_Fetch: GCN width-mismatch rules both directions (attr narrower →
  (0,0,0,1) default fill; attr wider → dropped channels into a scratch), both
  measured (attrib 2 as 2ch→vec3 and 4ch→vec3 on the menu VS).
  Cube textures (measured type 11 = 1024x1024x6 skybox, fmt56, tile 9):
  analysis accepts 9/11 (others still named with raw fields); SPIR-V emits
  OpTypeImage Cube from the measured T# types (mixed 2D+cube = named error);
  ImageSample coords go vec3 for cube; decode_texture fetches 6 faces as 6
  block grids; offscreen creates CUBE_COMPATIBLE images + CUBE views with
  layer-aware barriers. Tests: cube SPIR-V emission, 6-face decode round-trip.

- Texture chain completed + two GPU blockers (2026-07-18, commit pending;
  106/106 raeen-gpu lib + 225/225 kyty-graphics, clippy clean):
  * Vulkan consume of `ShaderStageBinding.textures` (the missing half of the
    in-flight texture work): offscreen.rs now builds SAMPLED_IMAGE + SAMPLER
    descriptor arrays per stage (bindings from `shader_calc_binding_indices`),
    uploads guest textures staging→device-local→SHADER_READ_ONLY, one sampler
    per S# (linear/nearest). Acceptance: tests/texture_upload.rs — recompiler-
    ABI PS (push-const indices, OpSampledImage/OpImageSampleImplicitLod)
    samples a magenta texel; center=magenta, corners=clear, 0 validation
    errors. Gotcha: recompiler lists resource vars in OpEntryPoint interface
    (spirv-val requires it under Vulkan 1.3).
  * VB/storage fetch cap bug MEASURED: "vertex buffer guest range not fully
    readable" (7762×/run, only 3 executed draws) was NOT memory — the
    committed-prefix probe printed prefix == size. The 64K-dword
    MAX_READ_DWORDS cap (meant for CP out-of-band mis-decode protection) was
    misapplied to resource fetches; Minecraft binds a 4 MiB vertex-arena V#.
    Fix: `read_dwords_validated` (resource path, 256 MiB cap) for
    draw_translate's read_guest_bytes; the CP out-of-band path keeps the 64K
    cap. Same root cause as run-1's storage-buffer failure.
  * Alpha blending: CB_BLEND0_CONTROL..7 + CB_BLEND_{RED..ALPHA} decode in
    kyty-graphics run.rs (Kyty bit layout; SharpEmu AgcPrimaryRegisterDefaults
    confirms addresses), `blend_state_from_regs` → Vulkan blend attachment +
    constants (dual-source/BOTH_SRC_ALPHA = named errors, never silent ZERO;
    !separate_alpha reuses colour factors = hw behaviour). Acceptance:
    tests/blend_state.rs — SRC_ALPHA/1-SRC_ALPHA/ADD over seeded red reads
    back (128,128,0,128), corner stays seeded, 0 validation errors.
  * Recompiler table tests updated for the ImageSample nine (177 impl/44 NI;
    NI-error test retargeted to ImageStoreMip — no guessed encodings).

- GraphicsRun CommandProcessor (Kyty Gen5 CP): expanded for retail DCBs
  (commit pending, 194/194 kyty-graphics + 86/86 raeen-gpu tests).
  Resilience policy: unknown op/register = rate-limited warn + skip-by-length;
  hard errors only for truncated/non-type3 streams and refused draws.
  Ported: R_DRAW_INDEX (AGC + IT_DRAW_INDEX_2 raw form), R_{CX,SH,UC}_REGS_INDIRECT
  via new GuestMemory trait, R_DRAW_RESET → Reset, IT_INDEX_TYPE/BASE/BUFFER_SIZE +
  IT_SET_BASE(1) tracking, rate-limited sync/event/write-data skips.
  Indexed/indirect draws degrade to logged vertex-count-only draws
  (DrawSink::draw_index default; indirect count read from first args record).
  raeen-gpu: guest_mem::IdentityGuestMemory (VirtualQuery-validated identity
  reads) wired into AgcGpuSession::execute_dcb_cp.
  Still todo: GraphicsRender (real index fetch, guest shader bind, multi-draw walk).

- Minecraft (PPSA17221) libkernel + libScePosix import closure: **0 missing**
  in both libraries (was 17 libkernel + 19 libScePosix), measured by re-running
  `--run-eboot`; 144 distinct missing NIDs remain, all in out-of-scope service
  libs (libSceNpWebApi2 21, libSceHttp2 14, libSceNet 13, ...). Implemented in
  raeen-hle (commit pending; 247/247 hle, 19/19 kernel, 102/102 firmware,
  82/82 runtime tests): real VFS unlink/rmdir/rename/truncate (+ new VFS ops),
  REAL blocking POSIX semaphores (`posix_sem.rs`, address-keyed, condvar +
  termination-aware slices), scePthreadMutexTimedlock (deadline in lock_core),
  sceKernelMapDirectMemory2 (arg reshuffle), Add/DeleteWriteEvent, offline
  POSIX sockets (accept/listen/recv/send/select/... EWOULDBLOCK semantics,
  errno via __error slot), sched_get_priority_max/min (767/256), getrusage
  zero-fill, signal/Mlock/Sync/Chmod/Utimes accepted, `__progname` as a real
  data-page pointer export (raeen-firmware). Title now boots 17 guest pthreads
  and dies downstream on its own `std::out_of_range` ("invalid string
  position") during phase-1 unwinding — next investigation target.

- ShaderMemory Phase 2 (guest shader fetch → GCN parse → SPIR-V → draw):
  **implemented + proven end-to-end in-tree** (commit pending; 196/196
  kyty-graphics, 87/87 + 2/2 + 2/2 raeen-gpu tests, clippy clean).
  kyty-graphics CP: Gen5 shader-bind SH registers ported from Kyty's
  g_hw_sh_indirect_func — SPI_SHADER_PGM_LO/HI_PS+CHKSUM_PS+RSRC2_PS,
  PGM_LO/HI_ES+CHKSUM_GS+RSRC2_GS (gs-instead-of-vs), USER_DATA_GS slots —
  plus sh_regs context regs (SPI_SHADER_COL_FORMAT, SPI_PS_INPUT_ENA/ADDR/
  IN_CONTROL, SPI_PS_INPUT_CNTL_0..31, SPI_VS_OUT_CONFIG, DB_SHADER_CONTROL).
  These are exactly the registers Minecraft's DCBs write (proven from the
  prior iron log: unknown-reg warns 0xC8/0xC9/0x80/0x8A/0x8B/0x08, cx 0x191+).
  raeen-gpu: shader_fetch.rs — bounded fetch (4 KiB chunks, 256 KiB cap,
  parser-driven growth on Truncated), next-gen→legacy generation fallback with
  both reasons named, positive+negative cache keyed (stage, addr, 16 head
  bytes) so a failing shader warns ONCE; RAEEN_DUMP_SHADERS forensic dumps
  (work even when translation fails). OffscreenDrawSink: untranslatable
  shader = skipped draw (counted, debug-logged), DCB continues; embedded
  fixture path intact (M2 gate untouched). Acceptance:
  tests/shader_memory_phase2.rs — DCB binds a real guest-memory PS via SH
  registers → CP → fetch → recompile → Vulkan draw → green pixel readback +
  frame PPM; garbage bind skips the draw, DCB survives.
  Also fixed: guest_mem read used copy_nonoverlapping; a wild-but-committed
  guest range can overlap the destination Vec (page-granular validation) —
  intermittent STATUS_STACK_BUFFER_OVERRUN under test; now ptr::copy.
  Title measurement (PPSA17221, 3×120 s runs, RAEEN_DUMP_SHADERS+FRAMES set):
  **0 shaders fetched, 0 draws — title dies ~10 s in, pre-graphics**, on the
  known std::out_of_range phase-1-unwinding wall above (first failing HLE
  call sceKernelGetdents → 0x8002000e). The GPU-side path is armed and proven;
  re-measure the moment the boot wall falls.

- ASTRO.BOT scene-shader opcode batch (2026-07-18, commit pending; 258/258
  kyty-graphics, 129/129 raeen-gpu, 276/276 raeen-hle tests; 1 diagnostic GPU
  test ignored; kyty-graphics+raeen-gpu clippy clean; GUI build green):
  closed seven title-measured translation blockers with typed decode + SPIR-V
  acceptance coverage. `S_GETPC_B64` now materializes the absolute address of
  the following instruction (including guest bases above 4 GiB);
  `S_PACK_LL_B32_B16` packs the two low halfwords; VOP1 SDWA OMOD preserves
  x2/x4/x0.5 multipliers; address-only `BUFFER_LOAD_DWORDX4` uses zero index;
  `IMAGE_GET_RESINFO dmask=xy` emits `OpImageQuerySizeLod`; `DS_APPEND` and
  `DS_CONSUME` accept their unused non-zero addr field; and
  `IMAGE_SAMPLE_C_LZ` consumes Gen5 `{reference,x,y}`, samples at LOD zero,
  performs the manual `reference <= red` comparison used by SharpEmu, then
  applies all seven supported dmask layouts. The comparison module assembles
  and validates through Naga. Kyty remains behind the raeen-gpu contract.
  Fresh ASTRO.BOT frame measurement is not yet attributable to this batch:
  two provider-specific ABI aliases (`libSceLibcInternal` C ABI and libkernel
  POSIX pthread names) fixed earlier link stops, but the current architecture
  branch then spins after module_start in native libc allocator mutex traffic
  (443,720 balanced HLE calls/30 s, last HLE scePthreadMutexUnlock), before AGC
  submission. Artifact: `scratch/astro-opcodes-20260718-201535/`; fix that boot
  regression before claiming a translated-shader or frame-count improvement.

- ASTRO.BOT frame path — texture format 71 + validation-layer perf cliff
  (2026-07-19, commit pending; raeen-gpu 133/133 green, clippy clean).
  DO NOT RE-DERIVE the following; all measured against the retail title with
  `RAEEN_SKIP_MAIN_INIT=1 RAEEN_RESUME_ON_MISSING=1`:
  * The title's GPU work is DETERMINISTIC across runs and across commits
    21483ef/e77ea3f/HEAD: 256 DCB submissions, 1028 draws, 1497 dispatches,
    36 VideoOut flips. "captured AGC submission" is rate-limited (logs the
    first few + powers of two) — counting those lines UNDERCOUNTS submissions
    by ~85x. Read "AGC submission progress" for real totals.
  * The 21483ef-vs-HEAD "frame dump regression" was NOT a dump-path or
    submission-routing bug. `ctx.gpu` (e77ea3f moved submits off
    `AgcGpuSession::global()`) is correctly wired: runtime/thread.rs:410
    calls `new_process` then `install_process`, so global() is the same
    session. Queue routing is also correct (only GpuQueue::AsyncCompute maps
    to is_compute).
  * Real cause: e77ea3f's buffer/texture reclassification correctly began
    treating a T# as a texture, exposing unified format 71 as unimplemented.
    Every draw in the 7966-dword DCB then failed at DWORD 5053 (56 skips).
    FIXED: unified 71 -> (dataFormat 12 = 16_16_16_16, numFormat 7 = FLOAT)
    = R16G16B16A16_SFLOAT @ 8 bpp, verified against SharpEmu
    Gfx10UnifiedFormat.cs:83 and its Gen5 layout table
    (Gen5SpirvTranslator.cs:2667 "SetLayout(12,0,0,16) // 16_16_16_16").
    8 bpp is bpp_log2 3, a table row no 1/4-byte format had exercised;
    tile mode 27 (SW_64KB_R_X) already supported it. Draw skips 56 -> 0.
  * `agc_exec.rs` hard-coded `VulkanBackend::new(true)`, so the Khronos
    validation layer was ALWAYS on for titles: ~0.9 s per
    vkCreateGraphicsPipelines, i.e. ~15 min for 1028 draws — the reason a
    260 s run reached only one submission and looked like a GPU-worker hang.
    Now opt-in via `RAEEN_VULKAN_VALIDATION=1`.
  * STILL NO FRAME, and the remaining wall is now unambiguous: pixel/compute
    shader translation. 16 shaders translate OK, 16 fail, and a draw needs
    its PS, so every draw skips and `sink.last` stays None. Both failure
    reasons share ONE root cause — Extended User Data:
    `ps: user_sgpr > user sgpr count` (analysis.rs:2188 — the shader declares
    more user SGPRs than were ever written; `count` is a high-water mark,
    hw_regs.rs:714) and `ShaderUserData eud_size_dw != 0`. This is backlog
    task #9 (EUD/SRT scalar resolver), NOT a dump/present/format problem.
  * The two frames 21483ef produced were the known static green loading
    composite, never the 3D scene — recovering that dump is not a scene frame.
  Pre-existing (not from this work): `cargo fmt --all --check` fails on
  committed `crates/kyty-graphics/src/run.rs:1262`; belongs to task #12.

## 2026-07-20 — Minecraft (PPSA17221) deep-boot session: NID surface 161→138, AGC frames flowing, audio unblocked

Context: first full `--run-eboot` measurement of Minecraft PS5 (254 MB eboot
+ 5 bundled .prx). Boot reached ~30 threads (MINECRAFT MAIN THREAD, cohtml,
FMOD, RakNet, Agc*) but died twice deterministically: (1) a net thread
jumped to the unresolved-import stub for `socket` from libScePosix;
(2) AgcSubmissionThread deref'd 0x25 after `sceKernelWaitEqueue` timed out.
Uncommitted (user hasn't asked for a commit).

DONE (raeen-hle, 285/285 tests, workspace clippy clean):
- libScePosix provider aliases — provider-aware resolution meant the
  libkernel-only registrations didn't satisfy libScePosix imports:
  socket/bind/connect/getsockname/inet_pton (kernel_socket.rs) and
  open/read/write/close/lseek (libkernel.rs, same pattern as the existing
  libScePosix::getdents alias). Fixes the `socket` jump-to-stub fault.
- libSceNet socket family (libsce_net.rs): sceNetSocket/SocketClose/Bind/
  Connect/Listen/Accept/Send/Recv/Shutdown/Setsockopt/ErrnoLoc/
  ResolverStartNtoa — offline semantics matching kernel_socket (connect →
  ENETUNREACH, accept/recv → EWOULDBLOCK, send discards validated payload);
  error codes from shadPS4 net_error.h (0x8041_01xx); ErrnoLoc shares the
  __error cell.
- `__stack_chk_fail` registered under libkernel provider (libc.rs).
- libSceAudioOut2 streaming trio (libsce_audio_out2.rs):
  sceAudioOut2ContextPush/ContextAdvance pace to one hardware grain
  (grain_samples/frequency) via a per-context PaceAdvance ported from
  SharpEmu's ContextState (FMOD uses Push as its submission clock);
  sceAudioOut2PortSetAttributes accepted inert; plus
  sceAudioOut2ContextGetQueueLevel (dlsym-only import, never in the static
  table — level always 0 since pacing is synchronous).
  Result: missing NIDs 161 → 138 for this title.
DONE (raeen-gpu, 121/121 tests):
- vulkan/instance.rs: enable + require vertexPipelineStoresAndAtomics —
  Gen5 vs writes storage buffers; pipeline creation was failing validation
  (VUID-RuntimeSpirv-NonWritable-06341) for whole draw batches.
- draw_translate.rs: Gen5 unified vertex format 11 → R16_UINT (SharpEmu
  Gfx10UnifiedFormat: 11 → (2,4)); the per-frame DWORD 950/1270 skips are
  gone, total_draws 1351 → 2244 at submissions=128.

MEASURED STATE NOW: the title boots deep and runs for minutes — RakNet up
("IPv4 supported"), FMOD mixer paced, cohtml UI active, DCB+ACB submissions
continuous (draws 2244, dispatches 2123, flips 128 at ~1.5 min), ~6/7
shaders translate (16k-word SPIR-V), frames present to scanout. STILL BLACK
frames (1920x1080 flips land but all-zero) — visible output blocked by:
- `libSceAgc` NID 0xfca47359e915d76d (-KRzWekV120) UNKNOWN fn, 14 args,
  stub returns 0 — in NO name DB (ours/SharpEmu/Kyty); likely
  flip/display-chain related. Needs RE from the call site.
- vs translate fail: Recompile_Fetch registers_num=0 (attrib 3); cs:
  ShaderUserData eud_size_dw != 0 (same EUD gap as backlog task #9).
- VGT_PRIMITIVE_TYPE 0 unsupported (single early draw; NONE = no-op on HW).
- sceAcmContextCreate / sceAjmBatch* UNKNOWN ABI (audio decode → silence,
  non-fatal).
- Mass "unknown context register — write skipped" (kyty_graphics::run).
NEXT: identify NID 0xfca47359e915d76d; EUD/SRT scalar resolver (task #9);
prim-type NONE as no-op; dump later flip indices (dump only catches ≤8/2^n
and the flip counter stops ~6-8 presents).

- ASTRO.BOT draw path — Gen5 32 user-SGPRs + two texture gaps (2026-07-19,
  commit pending; workspace 1430/1430 green, clippy clean on touched crates).
  The DRAW path went from "every draw fails" to "draw_skips=0, draw shaders
  translate, 17 shaders translated_ok / 2954 cache hits". Chain, each step
  measured against the retail title then fixed with a red-green test:
  1. **texture format 71** -> R16G16B16A16_SFLOAT @ 8bpp (SharpEmu
     Gfx10UnifiedFormat.cs:83 = dataFormat 12 (16_16_16_16) + numFormat 7
     (FLOAT)). Was failing all 56 draws at DWORD 5053.
  2. **Vulkan validation was hard-coded ON** for titles (agc_exec.rs
     `VulkanBackend::new(true)`): ~0.9s per vkCreateGraphicsPipelines, ~15 min
     for 1028 draws — it LOOKED like a hung GPU worker. Now opt-in via
     `RAEEN_VULKAN_VALIDATION=1`. Check this first if the worker seems stuck.
  3. **Gen5 graphics stages have 32 user SGPRs, not 16.** `UserSgprInfo::
     SGPRS_MAX` was 16, so `set()` silently DROPPED slots 16..31 and capped
     `count` at 16, while ASTRO.BOT pixel shaders declare rsrc2.user_sgpr =
     20/24/30/32 -> `shader_parse_ps` rejected every one. Widened SGPRS_MAX and
     run.rs `set_shader_register`'s SGPRS to 32. No SH-register collision: PS
     user data 0x0C with next SH reg VS_0 at 0x4C, GS_0 0x8C to ES 0xC8.
     Compute stays 16 (COMPUTE_USER_DATA_0..15) and spills to EUD.
     **PS user_sgpr failures 20 -> 0.**
  4. **sharp start_register bound**: Kyty treats >=16 as "lives in the extended
     buffer", but measured PS shaders reference a sharp at start_register=16
     while declaring NO extended buffer — with no EUD the data can only be a
     direct register, so the bound is the register-file size. Self-validating:
     an unwritten slot keeps type_ Unknown and is still rejected.
  5. **texture format 22** -> R32_SFLOAT @ 4bpp (Gfx10UnifiedFormat.cs:48 =
     dataFormat 4 (32) + numFormat 7), a 1920x1080 linear-depth target.
  6. **tile mode 24 = SW_64KB_Z_X** (the DEPTH swizzle; interleaves X/Y from
     bit 0 where _R_X runs X first). Transcribed `RB_PLUS_64K_DEPTH_X` from
     SharpEmu GnmTiling.cs `RbPlus64KDepthX` (GPL-2.0, same source/topology as
     the existing two tables); round-trip test also asserts it is a DIFFERENT
     permutation from mode 27 so a copy-paste slip would fail.
  STILL NO FRAME. State after the chain: no draw failures and no draw skips
  remain, but `execute_dcb_cp` still returns Ok(None) — `sink.last` is never
  set, so no RenderedImage reaches present/dump. NOT yet isolated; a 70 s
  debug run reached only 128 submissions and never got to a draw-carrying
  one, so its "0 draw_common calls" is INCONCLUSIVE, not evidence. Next step
  is a long run under `RUST_LOG=warn,raeen_gpu=debug` to see whether
  `draw_common` is entered and, if so, which of its early returns fires
  (prime suspect: `color_output_disabled(ctx)` -> "draw consumed without
  colour output (depth path pending)", draw_translate.rs:1080).
  Remaining shader failures are COMPUTE only: EUD (`eud_size_dw != 0`) plus
  opcodes `s_not_b64`, `s_brev_b32`, `v_cmpx_eq_f32`.

- ASTRO.BOT frame dump — the gate was counting the WRONG THING (2026-07-19,
  commit pending; workspace 1430/1430, clippy 0 warnings).
  `maybe_dump_frame` sampled on "draw_index <= 8 || is_power_of_two", but the
  value passed was the CUMULATIVE draw count, which a submission advances by
  its whole draw total at once (`*draws += drawn`): 0 -> 21 -> 42 -> 56, so it
  almost never landed on the cadence. **56 real Vulkan draws rendered and every
  dump was skipped.** Fixed with a `PRESENT_INDEX` incremented once per
  presented frame. Result: 0 -> **10 frames** dumped (1920x1080 and 2432x1368).
  MEASUREMENT TRAP that cost a full diagnostic cycle: the app filters logs with
  **`RAEEN_LOG`, not `RUST_LOG`** (raeen-core/src/logging.rs:160). Two debug
  runs under RUST_LOG produced zero `debug!` output, which read exactly like
  "draw_common is never entered" — it was an artifact. Always use RAEEN_LOG.
  With it: 56 "drove a register-state Vulkan draw", 57 depth-only draws
  consumed, 17 shaders translated_ok, 2954 cache hits.
  HONEST RESULT: the dumped frames are a SINGLE FLAT COLOUR across the whole
  2432x1368 surface (sampled at 5/20/40/60/80/95% — distinct_pixels=1 at every
  point; consecutive frames byte-identical). This is the previously-known
  loading composite, NOT the 3D scene, exactly as
  [[astro-bot-boot-state]] predicted: the scene is COMPUTE-rendered and the
  compute path still fails on EUD (`eud_size_dw != 0`) plus opcodes
  `s_not_b64`, `s_brev_b32`, `v_cmpx_eq_f32`. Draws are no longer the blocker;
  compute is. Do NOT claim a scene frame from these dumps.

- ASTRO.BOT compute opcodes — s_not_b64 / s_brev_b32 / v_cmpx_eq_f32
  (2026-07-19, commit pending; kyty-graphics 260/260, workspace 1431/1431,
  clippy 0). All three were blocking compute-shader translation. Wired through
  the full chain the codebase requires: types.rs variant -> parse.rs arm(s) ->
  recompile.rs function + dispatch row -> table count assertions bumped
  (245->248 rows, 233->236 implemented) -> per-opcode test.
  * `s_not_b64` (SOP1 0x08, Sdst2Ssrc02): D.u64 = ~S0.u64, SCC = (D != 0).
    New `recompile_snot_b64` modeled on `recompile_swqm_b64` (same format,
    same exec-passthrough/paired-dword/execz/scc structure), OpNot per dword.
  * `s_brev_b32` (SOP1 0x0b, SVdstSVsrc0): OpBitReverse, and it does NOT write
    SCC — pinned by `S::None` in the row and asserted in the SCC-semantics test.
  * `v_cmpx_eq_f32` (VOPC 0x12): the cmpx block mirrors cmp at +0x10, so 0x12
    is the exec-writing eq. Reuses the generic `recompile_vcmpx_xxx_f32` with
    `p1("OpFOrdEqual")` (matching VCmpEqF32's op). Wired at BOTH VOPC parse
    sites (parse.rs:1061 and :1708) — there are two.
  Test uses the MEASURED encoding from the title's failure log
  (`raw 0xbefe087e` = `s_not_b64 exec, exec`).
  MEASURED RESULT: **zero unknown-opcode failures remain** in an ASTRO.BOT run
  (previously 3 distinct). Compute now fails on exactly two things:
  `ShaderUserData eud_size_dw != 0` (8) and `read-only texture type 8 is not
  Texture2D` (type 8 = 1D). Frames still dump (10) and are still a FLAT COLOUR
  (distinct_pixels=1 at 10/35/60/85%) — EUD still gates the scene compute, so
  this did NOT by itself produce scene pixels. **EUD (task #9) is now the
  single remaining blocker on the compute path.**
  Note: `cargo fmt --all --check` flags `crates/raeen-hle/src/libsce_audio_out2.rs`
  (2 spots) — NOT from this work; it arrived via commit 3d183a1.

- EUD increment 3 — OPEN QUESTION found while scoping (2026-07-19, no code
  change). Do not start increment 3 assuming the existing indexing is right:
  measured `cs@0x50068ef00` has `eud=8` dwords but `sharp[1.0]` reports
  `offset_dw=32`. `read_sharp_fields`'s extended branch reads
  `ext[start - 16 + j]` = `ext[16]`, which is OUT OF RANGE for an 8-dword EUD
  ("extended (EUD) buffer too small"). So either the `-16` rebase is wrong for
  Gen5, or `offset_dw` is not a dword index into the EUD, or the EUD is larger
  than `eud_size_dw` suggests. SETTLE THIS FIRST — wiring a buffer under the
  current assumption would silently feed wrong descriptors (no error, wrong
  pixels). Also note SharpEmu does NOT use a fixed EUD-base convention: it runs
  a full scalar evaluator (`Gen5ShaderScalarEvaluator`, ~2000 lines) to resolve
  descriptors, so there is no short convention to transcribe; our
  `find_scalar_load_bases` + `scalar_load_target_address` (increments 1-2) are
  the right road. `shader_get_input_info_vs` has `mem` but NOT a parsed
  `ShaderCode`, so increment 3 needs a parse pass threaded in.

- EUD increment 3 — PARTIALLY LANDED (2026-07-19; kyty-graphics 260/260,
  workspace 1431/1431, clippy 0). Two changes, both evidence-driven:
  1. **The EUD rebase is the user-SGPR FILE SIZE, not a literal 16.** Kyty's 16
     was the PS4 user-SGPR count. Measured: a shader with eud_size_dw=8 places
     a sharp at offset_dw=32 while its other sharps sit direct at s0/s8 — under
     `-16` that reads ext[16] (past the end of an 8-dword EUD); under
     `-32` (= SGPRS_MAX) it reads ext[0], the first descriptor, which fits.
     `read_sharp_fields` now rebases by `UserSgprInfo::SGPRS_MAX`. This RESOLVES
     the open question logged in the previous entry.
  2. New `read_extended_user_data` reads `eud_size_dw` dwords from the sgpr pair
     immediately AFTER the declared user SGPRs (measured: declared=14, s14:s15
     -> descriptor-shaped data). `shader_parse_usage2` takes the buffer as a new
     `eud` parameter and feeds the extended branch that already existed.
     Refuses (does not invent descriptors) when the pointer is null/unaligned/
     unmapped or the pair is out of range.
  MEASURED: EUD failures 8 -> 6, shader translate_failed 15 -> 11. So the
  "pair right after declared user SGPRs" convention is CORRECT FOR SOME SHADERS
  BUT NOT ALL — the remaining 6 report "(EUD unreadable)". Those need the base
  recovered by scalar-load analysis (`find_scalar_load_bases` +
  `scalar_load_target_address`, increments 1-2, already built): parse the
  shader, find the `s_load_dwordx*` whose base is a user-SGPR pair, and use
  that address. `shader_get_input_info_vs` has `mem` but no parsed
  `ShaderCode`, so that step needs a parse pass threaded in.
  STILL NO SCENE: 10 frames, still flat colour. Scene compute needs the
  remaining 6 EUD shaders plus texture type 8 (1D).

## 2026-07-20 (cont.) — black-frame root-cause chain: draws execute, pixels never land

Follow-ups after the NID/audio/Vulkan batch (all uncommitted; a concurrent
session landed complementary Gen5 work mid-flight: 32-slot user SGPRs,
PRESENT_INDEX dump cadence, tiling, ASTRO formats):
- Unknown libSceAgc NID 0xfca47359e915d76d = SharpEmu's
  `sceAgcDriverUnknown_KRzWekV120` (AgcExports.cs:2821) — trace-and-return-OK
  stub there too; our return-0 matches reference behavior. NOT the
  black-frame cause.
- Gen5 vertex format 57 → R8G8B8A8_SNORM (draw_translate.rs; SharpEmu table
  57 → (10,1)). Removed the per-frame DWORD-950 skips; draws 2244 → 10142+
  per run, zero draw-skip warnings in later runs.
- DECISIVE census (RAEEN_DUMP_ALL_TARGETS): at presents 5..128 every render
  target (0x1f7d0000, 0x31c10000, 0x20040000) reports non_black_pixels = 0.
  Black frames are NOT a flip/scanout mismatch — the draw pipeline produces
  zero pixels, including no clear-alpha.
- RAEEN_DUMP_GPU_RESOURCES: vertex buffer @0x313f0150 (stride 28) holds REAL
  plausible UI data (0/1/2/3 floats) — vertex fetch works.
- The bound UI texture @0x31c00000 (1920x1080, fmt 56, tile 27) reads back
  100% zeros. NOT a PM4 DMA problem: captured DCB layouts carry opcodes
  16/70/118/38/19/53/46/44/7 — no IT_DMA_DATA (0x50) or IT_WRITE_DATA (0x37)
  at all, so the texture is meant to be filled by draws/compute, i.e. the
  same zero-output root cause. (Note: IT_DMA_DATA/IT_WRITE_DATA ARE
  consumed-without-effect in kyty_graphics::run — a real gap for OTHER
  titles, just not this one's current path.)
- Localization: rasterization executes but nothing lands. Vertex data OK,
  shaders translate, pipelines create, targets exist. Remaining suspects in
  order: (1) shader data path — SRT/EUD/user-SGPR delivery to vs/ps
  (cs@0x253a5000 still fails on ShaderUserData eud_size_dw != 0; concurrent
  session is mid-fix on the 32-slot SGPR widening), (2) skipped context
  regs incl. CB_SHADER_MASK 0x8f and CB_COLOR slot regs 0x366-0x3af,
  (3) vs Recompile_Fetch registers_num=0 (attrib 3) — 19 skips.
NEXT: finish EUD/SRT scalar resolution (task #9, concurrent session),
then re-measure census; VGT_PRIMITIVE_TYPE 0 → clean no-op (currently a
warn-and-skip, semantically right); consider honoring CB_SHADER_MASK.

- EUD strategy 2 (scalar-load base) — LANDED BUT INEFFECTIVE, measured
  (2026-07-20; kyty-graphics 260/260, workspace 1431/1431, clippy 0).
  `read_extended_user_data` now falls back to parsing the shader and trying
  `find_scalar_load_bases` + `scalar_load_target_address` when the
  after-declared pair fails. **MEASURED: no change — still 6 "EUD unreadable".**
  So for those shaders the scalar-load scan yields no usable base. Next
  diagnostic (do this BEFORE writing more resolver code): log, for one failing
  shader (`cs@0x50053c700`, eud=20 declared=14 count=14), whether
  (a) `shader_parse` of its `data_addr` even succeeds here,
  (b) how many `SLoadDword*` `find_scalar_load_bases` returns, and
  (c) what addresses those bases compute to and whether `dwords_at` backs them.
  One of those three is failing and the log will say which; guessing further
  resolver strategies without it is wasted work.
  Evidence recap for the failing class: `count == declared` (no register past
  the declared file holds the pointer) and sharps at offset_dw 32/40/48 with
  eud=20 dwords -> ext[0]/ext[8]/ext[16] under the SGPRS_MAX rebase.
  NOTE the heuristic risk now in the tree: strategy 2 takes the FIRST readable
  scalar-load target. It is currently inert (never succeeds on this title), but
  if a later title starts hitting it, wrong descriptors would surface as wrong
  pixels rather than an error.

- EUD strategy 2 — ROOT CAUSE FOUND, one layer left (2026-07-20;
  kyty-graphics 260/260, workspace 1431/1431, clippy 0).
  Added a `TRACE_EUD2` diagnostic (gated on RAEEN_TRACE_EUD) that reports, per
  failing shader, whether the code is mapped, whether `shader_parse` succeeds,
  how many `SLoadDword*` were found, and whether each computed base address is
  backed. It answered immediately: **`shader_parse` was FAILING**, so
  `find_scalar_load_bases` returned nothing — strategy 2 was never actually
  running. Two successive causes, first now FIXED:
  1. FIXED: `unknown sopp opcode 0x1f (raw 0xbf9f0000)` = RDNA2 `s_code_end`,
     the padding terminator emitted AFTER real shader code. Now parsed as an
     end-of-code marker (parse.rs SOPP arm).
  2. OPEN: with s_code_end decoding, the parse CONTINUES past it into padding
     and dies on `unknown operand: 115` (= ttmp7, i.e. garbage decoded as an
     instruction). So `shader_parse` does NOT stop at a terminator when handed a
     whole fetched buffer. **NEXT STEP: give the full-buffer parse a hard stop
     at SEndpgm/s_code_end** (or have `read_extended_user_data` pre-truncate
     `src` at the first terminator before calling shader_parse). That is the
     only thing between here and strategy 2 actually executing — it has never
     yet run on this title, so its effectiveness is still UNMEASURED.
  Frames still 10, still flat colour; the 6 EUD shaders and texture type 8 (1D)
  remain the compute blockers.

- EUD strategy 2 — EXACT REMAINING BLOCKER IDENTIFIED (2026-07-20;
  kyty-graphics 260/260, workspace 1431/1431, clippy 0).
  `shader_parse` DOES NOT STOP AT A TERMINATOR. Proof: pre-truncating the
  fetched window at s_endpgm/s_code_end and keeping the terminator gave
  "truncated instruction at 0x2e0"; keeping ONE MORE dword moved the error to
  0x2e4 — exactly one dword later. It parses until the buffer is exhausted and
  always fails at the tail, so NO truncation length can satisfy it.
  **THE FIX IS INSIDE `shader_parse`: end the instruction loop after SEndpgm**
  (s_code_end now maps to SEndpgm, parse.rs SOPP 0x1f). Deliberately NOT done
  here: that loop is the main shader path for every title, and changing it
  needs a full re-measure of Minecraft + ASTRO.BOT, not a blind edit.
  Until then strategy 2 (scalar-load EUD base recovery) CANNOT run — it has
  still never executed on this title, so its effectiveness remains UNMEASURED
  and the 6 "EUD unreadable" shaders are NOT evidence against it.
  Landed and safe meanwhile: s_code_end parsing, the SGPRS_MAX EUD rebase, the
  after-declared-pair EUD read (fixed 2 of 8 shaders), and the TRACE_EUD2
  diagnostic that produced all of the above.

- **EUD RESOLVED (task #9 closed)** (2026-07-20; kyty-graphics 260/260,
  workspace 1432/1432, clippy 0). `ShaderUserData eud_size_dw != 0` no longer
  appears in an ASTRO.BOT run — it was the blocker for 8 shaders.
  Root cause was NOT the resolver logic but `shader_parse`'s end detection.
  Kyty breaks at s_endpgm (0xBF81_0000) *unless a live label targets past it*
  — correct for multi-block shaders, which ASTRO.BOT's are. Those kept parsing
  (rightly) until they reached RDNA2 `s_code_end` (0xBF9F_0000), which had no
  break, so the parse ran on into padding and died ("unknown operand: 115").
  FIX: `s_code_end` ends the code BLOCK and takes NO live-label exception —
  nothing can branch past it — so it is now an unconditional break in the same
  end-detection condition (parse.rs), plus a SOPP 0x1f arm mapping it to
  SEndpgm. Truncating the buffer in the caller does NOT work and was reverted:
  the loop parses to exhaustion, so trimming just moves the tail error one
  dword (proven: 0x2e0 -> 0x2e4).
  With the parse fixed, EUD strategy 2 RUNS for the first time and succeeds:
  cs@0x50053c700 need=20dw finds 8 scalar loads (all backed), ps@0x500652400
  need=12dw finds 2. Combined with strategy 1 (pair after declared user SGPRs,
  for shaders where count > declared) all EUD shaders now resolve.
  **STILL NO SCENE**: 10 frames, still flat colour (distinct_pixels=1 at
  10/30/50/70/90%). translate_failed is still 11 and 2 draws want texture
  type 8 (1D). So EUD was necessary but NOT sufficient — the next measurement
  should re-enumerate what those 11 failures actually are now, since the whole
  failure profile shifted when the parser stopped truncating shaders early.

- Post-EUD failure re-enumeration (2026-07-20) — the remaining shader
  translation failures are FIVE DISTINCT FEATURES, not one gap. Measured after
  the s_code_end parse fix (which changed the profile, so pre-fix lists are
  stale — this supersedes them):
    2x `mubuf feature: idxen == 0`        (vs@0x50074e000, cs@0x500757800)
    2x `ds feature: addr != 0`            (cs@0x5006c5f00, cs@0x5005fd000)
    2x `unknown usage type: 0x05`         (ps@0x500652400, cs@0x500690400)
    1x `unknown exp target: 0x0d`         (vs@0x100008e6cd00)
    1x `read-only texture type 8` (1D)    (ps@0x500640200)
  Each is an independent recompiler/analysis add. **No single remaining fix
  produces the scene** — plan for five, then re-measure. Note the DS (LDS) and
  MUBUF-idxen items are compute-side, i.e. on the path the scene actually
  renders through.

- Texture type 8 (1D) supported (2026-07-20; raeen-gpu green). A 1D image is a
  2D image one row tall and the T# already reports height 1, so the existing 2D
  decode path handles it unchanged (measured: a 1x1 format-71 tile-27 texture).
  Kept as its own match arm so a >1-row "1D" texture would still be visible.
  Four of the five post-EUD features remain: `mubuf idxen == 0` (vs+cs),
  `ds addr != 0` (cs, LDS), `unknown usage type 0x05` (ps+cs),
  `unknown exp target 0x0d` (vs).
  STILL FLAT: 10 frames, uniform 0x408000, frames byte-identical.
  MEASUREMENT WARNING for the next session: sampling a PPM with
  `od | paste - - -` at an arbitrary byte offset can report a spurious extra
  "distinct pixel" from a truncated trailing group at the sample boundary. It
  briefly looked like content had appeared (distinct=2) — it had not. Verify
  any apparent content with a full-histogram check, not a boundary sample.

- Sizing the remaining four (2026-07-20, inspection only, no code change).
  Two of the four are SUBSYSTEMS, not opcode adds — do not schedule them as
  quick wins:
  * `ds addr != 0` — the DS parser (parse.rs ~2717) supports ONLY
    DS_APPEND/DS_CONSUME (GDS counters via M0); every real LDS access is gated
    off (addr/data0/data1/offset0/offset1 must all be 0, gds must be 1). So
    this is not a flag to relax: it means **LDS/shared memory is entirely
    unimplemented** — needs SPIR-V Workgroup storage, LDS size from
    COMPUTE_PGM_RSRC2.lds_size, and barrier handling. ASTRO.BOT's scene compute
    uses it.
  * `mubuf idxen == 0` — an ADDRESSING-MODE change (address becomes
    soffset+offset, plus a VGPR when offen), touching how every buffer load
    resolves for every title. Needs a Minecraft + ASTRO.BOT re-measure to prove
    it does not corrupt the currently-working buffer path.
  The other two (`unknown usage type 0x05`, `unknown exp target 0x0d`) are
  unsized.
  HONEST FORECAST: clearing all four is multi-session, and even then ASTRO.BOT's
  scene additionally depends on the compute -> storage-image writeback front
  (front #3 in [[astro-bot-boot-state]]), which is still unimplemented. Do not
  promise a scene frame from the four alone.

- `unknown exp target 0x0d` sized (2026-07-20, inspection only): EXP target
  0x0c = POS0 (handled), **0x0d = POS1** — the second position export, which
  carries clip/cull distances (and in some encodings point size / viewport /
  layer). Not a parse-table gap: honouring it needs SPIR-V ClipDistance/
  CullDistance declarations. A no-op arm WOULD let the shader translate, but
  silently disables clipping — decide that deliberately, and note it in the
  frame, rather than slipping it in as a "parser fix".
  Remaining four, sized: LDS (subsystem), mubuf idxen==0 (addressing mode,
  needs 2-title re-measure), exp POS1 (clip/cull), usage type 0x05 (unsized).

- `unknown usage type 0x05` sized (2026-07-20, inspection only): the usage-type
  dispatch in `shader_parse_usage2` (analysis.rs ~1070) handles 8 (vertex
  buffer) and 10 (vertex attrib) explicitly, with direct/sharp entries handled
  earlier; type 5 is not in the table and is NOT documented in-tree. Identify it
  from the Gen5 ShaderUserData usage enum (SharpEmu Gen5ShaderIr.cs /
  AgcExports, or Kyty Shader.cpp L1490/L1564 region) BEFORE writing an arm —
  a wrong guess here mis-binds a resource silently.
  All four remaining items are now sized; none is a quick win. See the previous
  three entries.

- Minecraft graphics push (2026-07-20; kyty-graphics 262/262). TWO REAL GPU BUGS
  FIXED, and a REFRAME of what actually gates Minecraft's pixels.

  1. **Vertex-fetch attrib MIS-INDEX (was silently binding the wrong buffer).**
     `shader_parse_attrib` appends resources in DISCOVERY ORDER
     (`resources_dst[resources_num]`, analysis.rs) while `recompile_fetch` read
     them back by ATTRIB-TABLE ID (`resources_dst[attrib_id]`, recompile.rs).
     Those agree only if the semantics table is identity-mapped. MEASURED on
     Minecraft it is NOT: positions 0,1,2 carry semantics 0,2,3 (semantic 1
     absent). One gap caused TWO bugs — attrib id 3 hit an unwritten slot
     ("invalid registers_num: 0", 19 draw skips) and attrib id 2 read position
     2, i.e. ANOTHER ATTRIBUTE'S V#, with no error at all.
     FIX: `ShaderVertexDestination` now records `semantic` (Default =
     UNSET_SEMANTIC = -1, since Kyty's zeroed slots are indistinguishable from
     a real semantic 0); `recompile_fetch` resolves by semantic. The `%attrN`
     REFERENCE was also corrected to the resolved POSITION — SPIR-V declares
     `%attr{i}`/`OpDecorate Location {i}` by position and the host binds by
     position, so only the V# lookup was id-keyed. MEASURED: draw_skips 19 -> 0.
     Test `parse_attrib_records_semantic_for_a_gapped_table` pins it.
     DO NOT "fix" registers_num==0 with a (0,0,0,1) default: that collapses
     every vertex to the origin, which REPRODUCES the black frame silently and
     deletes the diagnostic. Kyty upstream has this same defect (Shader.cpp
     L1126-1137 vs ShaderSpirv.cpp L6076-6078) — porting fidelity is not a
     defence.

  2. **PA_SU_SC_MODE_CNTL (0x205) was HALF-WIRED.** pm4 constant (pm4.rs:252),
     the 11-field `ModeControl` struct (hw_regs.rs:291-303) and the Vulkan
     cull-mode consumer (draw_translate.rs:932-935) all existed, but run.rs
     NEVER decoded the register — so `ctx.mode_control` stayed all-false and
     EVERY draw in EVERY title rasterized with CullModeFlags::NONE. Decoded per
     Kyty Pm4.h L489-510; test gives each field a distinct value.

  **REFRAME — Minecraft's black frame is NOT a rendering bug.** 512 submissions,
  12083 draws, 256+ flips, shaders translate, draw_skips=0, and the scanout
  lookup HITS (draws genuinely target the presented buffer, which accumulates).
  The frame is byte-exactly zero because there is no menu content yet: the title
  sits in PSN/entitlement traffic (SceNpWebApi2 / SceNpAuthAuthorizedAppDialog /
  SceNpEntitlementAccess; 126 missing imports, ~46 NP) with Gameface/Ore-UI
  threads spawned, and settles into a ~10s PERIODIC RETRY loop on
  "/savedata0/ -> savedata\PPSA17221-app". A 560s run did MORE GPU work but
  never advanced past that loop. Next lever is HLE (NP/savedata), NOT the GPU.

  Items 1 and 2 of the user's list were ALREADY DONE by the concurrent session
  and verified present, not assumed: VGT_PRIMITIVE_TYPE 0 -> clean no-op
  (draw_translate.rs ~1098) and IT_WRITE_DATA/IT_DMA_DATA decode (agc.rs) with
  a REAL apply (guest-memory writes/copies/fills, libsce_agc.rs 861/875/898).

- ASTRO.BOT regression check after the fetch/cull fixes (2026-07-20): NO
  REGRESSION — still 10 frames, 16,588,802 non-zero bytes (the known green
  composite), 16 shaders translated. Enabling culling did not remove geometry.
  NEW FRONTIER EXPOSED (draws now reach a check they previously never got to):
  16x "draw failed at DWORD 4706: unsupported CB_COLOR0_INFO format=0x3".
  That is a colour-BUFFER (CB) format gap, distinct from the texture (T#)
  format table fixed earlier — do not confuse the two tables.

- CB colour format 0x3 (8_8) — TRIED, MEASURED HARMFUL, REVERTED (2026-07-20).
  After the vertex-fetch fix, 16 ASTRO.BOT draws reached the CB_COLOR0_INFO
  check and failed on format=0x3 channel_type=0. The enum numbering says 0x3 =
  8_8 (consistent with 0xa = 8_8_8_8 and 0xc = 16_16_16_16, both already
  mapped), so R8G8_UNORM looks right on paper. **It is not usable yet.**
  MEASURED: mapping it took ASTRO.BOT from 10 presented frames to **0**, with
  56 draws failing `vkQueueSubmit -> VK_ERROR_DEVICE_LOST` (the logical device
  is lost, so every subsequent frame dies too). Something else in the pipeline
  cannot honour a 2-channel attachment — candidates: attachment usage flags,
  the fragment shader's 4-component colour export vs a 2-component target, or
  blend state. REVERTED; the arm now carries a comment explaining why it is
  deliberately absent, and `cb_colour_formats_map_and_have_readback_sizes`
  ASSERTS 0x3 stays rejected so the regression cannot be reintroduced blind.
  A named error costs 16 draws; a device loss costs every frame.
  Kept from the attempt: that test, which also pins the invariant that every
  accepted CB format has a matching `readback_bpp` entry (they are two tables
  that must move together — `readback_bpp` is now pub(crate) for it).

- Minecraft menu blocker — NP HYPOTHESIS DEAD, Ore-UI narrowed (2026-07-20;
  workspace 1437/1437, clippy 0, fmt 0 diffs).
  **DO NOT re-chase NP/missing imports.** Measured two independent ways:
  (a) dispatch.rs:1773-1783 logs every unresolved import actually CALLED —
  ZERO such lines in any run; (b) zero raeen_hle::libsce_np* calls of any kind,
  not even sceNpGetState. The 126 "missing" imports are LINK-TIME ONLY and the
  title never calls one, so `sceNpGetState` returning SIGNED_OUT
  (libsce_np.rs:68-79) is irrelevant and stubbing NP signed-in changes nothing.
  Also eliminated: the 705 /app0 ENOENTs are genuine (those paths do not exist;
  normal Bedrock resource-pack probing), the dump is complete (22,299 files,
  index.html present), and the ~10s "retry loop" is BENIGN housekeeping
  (16s treatment_metadata.json flush + 2s offline-socket retry), not a deadlock
  — RAEEN_STALL_DUMP shows a healthy idle with the Rendering Pool still
  submitting DCBs.
  **NARROWED BLOCKER:** the title reads /app0/data/gui/dist/hbui/routes.json
  (the Ore-UI route table -> /hbui/index.html) THREE times and then never opens
  ANY .html, while `Gameface Layout(0)` and `Gameface Resource Thread` both sit
  in scePthreadCondWait. cohtml initializes and idles, never handed a page.
  FIXED en route (real defect, but did NOT unblock the menu — consistent with
  those functions never being called): `canonical_provider_name` now strips
  `.native`/`_native` (nid.rs:256, lib.rs:531). The title imports from
  `libSceMsgDialog.native` / `libSceSaveDataDialog.native`, both fully
  implemented under the bare names, so provider-aware resolution reported 14
  implemented functions as missing. MEASURED 430 -> 444 trampolines,
  128 -> 114 unresolved. Two regression tests, incl. one asserting the suffix
  is only stripped from the END (so `libSceNativeThing` is not merged).
  **OPEN AND IMPORTANT — re-check "the GPU path is healthy":** 12,083 draws
  across THREE render targets (0x1f7d0000, 0x20040000, 0x31c50000) produce
  BYTE-EXACTLY ZERO output in all three. Of those draws only 202 are consumed
  as depth-only and 3 as prim-NONE, and ZERO are skipped as untranslatable —
  so ~11,878 proceed to real rendering and write nothing. That is not what a
  title clearing to black looks like. Compute meanwhile IS working (3,193
  storage writebacks carrying real floats, head=[00,00,80,bf] = -1.0f).
  Next: either trace the LLE boundary into libcohtml.Prospero.prx (352 exports,
  loaded at +0xf4a0000) to see whether View/LoadURL is ever called, or find why
  ~11,878 colour draws write zero pixels. The second question is independent of
  the menu and may be the more fundamental one.

- **CORRECTION, LOAD-BEARING: NO TITLE HAS EVER RENDERED ITS OWN PIXELS.**
  (2026-07-20, proven by a controlled experiment, not inference.)
  ASTRO.BOT's long-claimed "visible frame / green loading composite / 67%
  non-black / 16.5 MB non-zero bytes" is **the emulator's own CLEAR_COLOR**,
  not title content. PROOF: `CLEAR_COLOR` is [0.25,0.5,0.75,1.0]
  (raeen-gpu/src/vulkan/offscreen.rs:25). The dumped ASTRO frame repeats the
  pixel `40 80 00` — and 0.25*255 = 0x40, 0.5*255 = 0x80 EXACTLY. Changing
  CLEAR_COLOR to [1,0,0,1] and re-running made the frame repeat `ff 00 00`
  (pure red). The "content" tracks our clear constant, so it is ours.
  Minecraft is the same story one step earlier: its targets are byte-exactly
  ZERO because every draw takes AttachmentLoadOp::LOAD over zero-filled
  content (`state.initial.is_some()`, offscreen.rs ~2023) and so never even
  clears — which is why it is black rather than light blue.
  **Every prior "ASTRO.BOT shows a frame" claim in this ledger and in
  [[astro-bot-boot-state]] is WRONG and must not be relied on.** The
  frame-count metric (10 frames, N non-zero bytes) measures only that the
  present/dump path runs; it says NOTHING about title rendering. USE A
  CONTROLLED CLEAR-COLOUR TEST before ever claiming a title rendered.
  Ruled out by measurement while establishing this (draw state is textbook
  correct, so these are NOT the cause): extent 1920x1080, full-screen Y-flipped
  viewport [0,1080,1920,-1080], full scissor [0,0,1920,1080], target_mask=0xf,
  colour write mask R|G|B|A, sane blend (replace and standard alpha both seen).
  Every geometric degeneracy is already a LOUD error in draw_state_from_regs
  (zero mask / degenerate extent / zero-area viewport / empty scissor) and
  titles hit NONE of them.
  So the real question for both titles is now narrow: draws execute with
  correct state and correct attachments yet contribute NO fragment. Remaining
  candidates: (a) vertex positions never produce coverage, (b) the fragment
  shader outputs nothing / is discarded. The discriminator is to force a
  constant colour export in the fragment shader for one draw: if the target
  changes, it is coverage; if not, it is shading.
  Also noted: the PPM dump writes 3 of every 4 bytes and ignores
  RenderedImage::bytes_per_pixel, so an 8bpp (HDR) or packed
  B10G11R11 target is dumped INCORRECTLY — the ASTRO blue channel reading 0x00
  instead of 0xBF is consistent with that. Fix the dump before trusting any
  HDR-target pixel values.

- **THE BLACK FRAME IS ZERO COVERAGE, NOT BLACK SHADING** (2026-07-20, proven).
  New permanent diagnostic `RAEEN_FORCE_CLEAR=1` (offscreen.rs) forces every
  draw to CLEAR instead of LOAD, so the target ends as pure CLEAR_COLOR unless
  a draw actually produced a fragment. MEASURED on Minecraft: after 12,083
  draws the final 1920x1080 frame is **100% uniform CLEAR_COLOR** — ONE
  distinct pixel value, 2,073,600/2,073,600 = `40 80 bf` = [0.25,0.5,0.75].
  **NOT ONE DRAW WROTE A SINGLE FRAGMENT.**
  This splits the hypothesis space definitively:
   - RULED OUT: fragment shading (there is nothing to shade), blend, colour
     write mask, target mask, viewport, scissor, extent, attachment identity,
     and the readback/dump path (blue returned 0xBF EXACTLY, so RGBA8 readback
     is faithful — ASTRO's 0x00 blue is specifically an HDR/packed-format DUMP
     bug, see the dump note in the entry above).
   - REMAINING: the geometry never covers a pixel. Candidates, in order:
     (a) the V# (vertex buffer descriptor) base/stride/format resolves to
         empty or wrong guest memory, so positions are garbage/zero;
     (b) the VS never writes a usable gl_Position (export/POS0 handling);
     (c) the index buffer / vertex_count is wrong so no primitive is assembled.
  NOTE this is AFTER the vertex-fetch mis-index fix (which was real and took
  draw_skips 19->0), so the fetch INDEXING is right but the fetched DATA or the
  position export may still be wrong — those are different failures.
  NEXT MEASUREMENT: log, for one real draw, the resolved V# base/stride/format,
  the first few bytes of guest memory at that base, vertex_count/index_count,
  and whether the VS SPIR-V contains a Position builtin store. That separates
  (a) from (b) from (c) in a single run.

- Minecraft draw bind profiles — geometry IS well-formed (2026-07-20,
  RAEEN_TRACE_DRAWS=1). The real-draw path IS reached and the draws look
  exactly like an Ore-UI/Gameface layer:
    4x  prim=4 verts=6  guest_vbufs=1 vattrs=2  ps_tex=1 ps_samp=1 ps_pushc=48
    8x  prim=6 verts=4  guest_vbufs=1 vattrs=1  ps_sbuf=1        ps_pushc=16
  i.e. textured/storage-fed QUADS with a bound vertex buffer and 1-2 attributes.
  Combined with the RAEEN_FORCE_CLEAR proof of ZERO coverage, the failure is
  now pinned to VERTEX POSITIONS: well-formed quads are submitted and rasterize
  nothing, so positions must be off-screen / degenerate / NaN.
  **CORRECTED ARITHMETIC:** the "12,083 draws" figure is AGC draw PACKETS
  counted at the HLE submit layer. It is NOT the number of draws reaching the
  Vulkan sink. Do not subtract early-return counts from it (I did, and wrongly
  concluded "~11,878 draws proceed"). Count at the sink instead.
  **STALE COMMENT CORRECTED** in draw_translate.rs (the TRACE_DRAWS block): it
  asserted "geometry covers the screen and the PS shades black". The
  force-clear probe REFUTES that — there is no coverage at all, so the PS is
  never invoked. Anyone reading that comment would chase the wrong wall.
  ALSO SPOTTED (separate, HLE): the linker annotates
  `PLT-0xb5 import (returns EINVAL, kills Streaming Pool)
   offset=0xe123280 nid=0x93aa4634cc09074f provider="libc"` — a known-hostile
  unresolved libc import. Worth resolving on its own merits.
  NEXT: dump the resolved V# (base/stride/format) for one of these 12 draws and
  the guest bytes at that base, plus the first post-VS clip-space positions.
  The compute path already has an equivalent probe ("storage buffer content ...
  all_zero=false"), so mirror it for vertex buffers.

- **ROOT CAUSE SPLIT — vertex DATA is fine, POSITION EXPORT is not; and the
  mesh buffer is empty** (2026-07-20, RAEEN_TRACE_DRAWS vertex-buffer probe).
  Two distinct problems, both measured:
  * **Buffer A** addr=0x253a12a0 stride=12 num_records=4 size=48
    non_zero=24, head decodes as f32: (-1,-1,+1) (+1,-1,+1) (-1,...) —
    i.e. a TEXTBOOK FULL-SCREEN QUAD IN NDC. The vertex data is CORRECT.
    A correct fullscreen quad that rasterizes ZERO fragments (proven by
    RAEEN_FORCE_CLEAR) means the failure is DOWNSTREAM of the data: the VS
    position export. stride=12 is only xyz — the VS must supply w=1 itself;
    if w ends up 0/garbage the perspective divide kills every vertex, which
    matches zero coverage exactly. **CHECK THE POSITION/POS0 EXPORT FIRST.**
    (Related and already known: EXP target 0x0d = POS1 is unimplemented.)
  * **Buffer B** addr=0x31400150 stride=28 num_records=149784 size=4,193,952
    — **only 29 non-zero bytes in the whole 4 MB**. The mesh geometry was
    never uploaded to guest memory. This is a SEPARATE bug from the position
    export (an upload/DMA path), and fixing the export alone will still leave
    this geometry blank. Note IT_DMA_DATA copies/fills ARE applied
    (libsce_agc.rs 861/875/898), so the upload is going somewhere else —
    likely a path we do not yet intercept.
  Order of attack: position export (unblocks the fullscreen quads, i.e. the
  UI/composite layer, which is what Ore-UI draws), then the mesh upload.

- POS0 export IS emitted — narrowing again (2026-07-20). Probe in
  `recompile_exp_pos0` (RAEEN_TRACE_DRAWS): Minecraft recompiles **8 POS0
  exports, all with srcs_are_variables=true**, 19 shaders translated. So the
  VS DOES write gl_Position. Combined with the earlier results the chain is now:
    vertex DATA correct (textbook NDC fullscreen quad)        ✓
    VS writes gl_Position                                     ✓
    draw state correct (viewport/scissor/mask/blend/extent)   ✓
    coverage                                                  ✗ ZERO
  **REMAINING SUSPECT: the Vulkan VERTEX-INPUT ATTRIBUTE BINDING.** The shader
  declares `%attrN` by ARRAY POSITION (spirv.rs WriteGlobalVariables emits
  `%attr{i}` / `OpDecorate %attr{i} Location {i}` over 0..resources_num), but
  `prepare_vertex_inputs` (draw_translate.rs ~443) sets each
  VertexAttributeData.location from `guest.attr_indices[ai]`. If those two index
  spaces differ — exactly the class of bug already found and fixed on the FETCH
  side, where resources were written by discovery order but read by attrib id —
  the VS reads an unbound location, gets zeros, and every vertex collapses to
  the origin. That reproduces zero coverage precisely while leaving data,
  position export and draw state all looking correct, which is what we observe.
  NEXT: log, per draw, each VertexAttributeData {location, format, offset,
  binding} next to the shader's declared attribute count/locations, and check
  they are the same index space. Note vs_pushc=0 / vs_sbuf=0 / vs_tex=0 in the
  measured profiles — the VS has NO uniforms, so it cannot be a bad transform
  matrix; the only way its positions go wrong is bad ATTRIBUTE INPUT.

- Vertex attribute bindings are CORRECT — hypothesis refuted (2026-07-20,
  RAEEN_TRACE_DRAWS attribute probe). Measured on Minecraft:
    ai=0 location=0 binding=0 R32G32B32_SFLOAT     offset=0  gen5=74 (res=1)
    ai=0 location=0 binding=0 R32G32B32A32_SFLOAT  offset=0  gen5=77 (res=2)
    ai=1 location=1 binding=0 R32G32B32_SFLOAT     offset=16 gen5=74 (res=2)
  Locations are 0/1 — the SAME index space the shader declares — and the
  formats/offsets tile the buffer strides exactly (12 = 3 floats; 16+12 = 28).
  So the Vulkan vertex-input binding is NOT the bug. Ruled out.
- **SELF-INFLICTED GAP CLOSED: front_face was hardcoded.**
  `offscreen.rs` hardcoded `FrontFace::COUNTER_CLOCKWISE`, which was harmless
  only while `cull_mode` was permanently NONE. Decoding PA_SU_SC_MODE_CNTL
  earlier this session turned culling ON without wiring the winding, so any
  title whose FACE bit says clockwise-is-front would have had exactly the wrong
  faces culled. `DrawState` now carries `front_face`, set from
  `mode_control.face` (0 = CCW front, 1 = CW front). Enabling culling and
  wiring the winding must always travel together.
  NOTE this did NOT cause the black frame — Minecraft was already black before
  the culling decode landed (mc_base run predates it) — but it was a live
  hazard introduced by an incomplete fix.
  STATE OF THE ZERO-COVERAGE HUNT — all of these are now RULED OUT by
  measurement: fragment shading, blend, colour/target write masks, viewport,
  scissor, extent, attachment identity, readback/dump path, vertex DATA
  (textbook NDC quad), gl_Position emission (8 POS0 exports), and vertex-input
  attribute binding. Depth is only applied when a depth attachment is bound
  (state.depth), which the composite path leaves None — CONFIRM that before
  spending more on depth. Next candidates: the index buffer / primitive
  assembly for prim=4 verts=6 and prim=6 verts=4, or the VS's own body (does
  it write the fetched attribute into the POS0 source registers, or overwrite
  them?). The quad's z = +1.0 (far plane) is worth re-checking IF depth ever
  becomes bound.

- **INVALID PIPELINE FIXED — vertex attribute format vs shader input type**
  (2026-07-20; raeen-gpu 136/136). Found by ENABLING THE VALIDATION LAYER
  (RAEEN_VULKAN_VALIDATION=1, which this session made opt-in). It named the bug
  in one run after many turns of manual narrowing:
    "vkCreateGraphicsPipelines(): pVertexAttributeDescriptions[1].format
     (VK_FORMAT_R16_UINT) at Location 1 does not match
     [VK_SHADER_STAGE_VERTEX_BIT] [Input variable, Location 1] type of (float32)"
  ROOT CAUSE: `Spirv::WriteGlobalVariables` (spirv.rs ~2767) declares EVERY
  vertex input as float / v2float / v3float / v4float — there is no integer
  path — but `gen5_vertex_format` mapped Gen5 format 11 to R16_UINT. Vulkan
  requires the attribute format's numeric type to match the shader input's, so
  that pipeline was INVALID and its draws contributed nothing.
  FIX: 11 -> R16_USCALED (same integer value, delivered as float).
  The test now also asserts the INVARIANT for the whole table: no Gen5 vertex
  format may map to a UINT/SINT Vulkan format while the shader side is
  float-only. That prevents the same class of bug for future entries.
  MEASURED: validation "does not match" messages 10 -> 0. Pipeline is valid.
  **Frames are STILL byte-zero** — necessary, not sufficient. Whatever remains
  is no longer a validation-visible error.
  **PROCESS LESSON (expensive): run with RAEEN_VULKAN_VALIDATION=1 FIRST when a
  draw silently produces nothing.** I spent many turns hand-checking vertex
  data, indices, attribute bindings, gl_Position, viewport/scissor/blend/masks —
  all of which were CORRECT and all of which the validation layer would have
  skipped past to the actual defect. The layer is opt-in precisely because it
  costs ~0.9s/pipeline, but a title with ~12 real draws pays only ~11s.

- NEXT HYPOTHESIS (state after the pipeline-validity fix, 2026-07-20).
  Validation is now CLEAN, so the remaining defect is NOT a malformed pipeline
  — that whole class is eliminated. Everything upstream is verified correct:
  vertex data (NDC quad), indices ([0,1,2,0,2,3]), attribute bindings
  (locations/formats/offsets), gl_Position IS emitted (8 POS0 exports), draw
  state, and the draw commands are issued (cmd_draw_indexed / cmd_draw,
  offscreen.rs ~2174). The clear reaches the image (force-clear probe), so the
  render pass executes. Yet ZERO fragments.
  Therefore the VS must be producing positions that never land in clip space.
  Note the measured profiles show the VS has NO uniforms at all
  (vs_pushc=0, vs_sbuf=0, vs_tex=0), so it cannot be a bad transform matrix —
  a uniform-less VS should essentially pass the NDC quad through.
  **SUSPECT: the VGPR hand-off between the fetch and the position export.**
  The fetch writes its result to the VGPR named by `sem.hardware_mapping()`
  (measured: reg=9, 12, 13 for the various attributes — see the "attrib
  semantic" probe in analysis.rs) while `recompile_exp_pos0` reads whatever
  VGPRs the EXP instruction names. If those disagree the export reads
  never-written registers (zeros) and every vertex collapses to the origin —
  which matches zero coverage exactly, and is the SAME index-space class of bug
  already found twice this session (fetch resources by discovery-order vs
  attrib-id; and %attrN by position vs id).
  MEASUREMENT: log the EXP instruction's source VGPR ids in
  `recompile_exp_pos0` and compare against the fetch destination
  (`resources_dst[].register_start` / hardware_mapping). If they differ, that is
  the bug. DEFINITIVE ALTERNATIVE: temporarily replace the generated VS with a
  hardcoded full-screen-triangle passthrough — if pixels appear, the generated
  VS body is at fault; if not, look below the shader (command submission).

- VGPR hand-off hypothesis — NOT confirmed (2026-07-20, measured).
  POS0 exports read a WIDE VARIETY of computed VGPRs across the 8 measured
  shaders: [0,1,2,3] [0,1,2,6] [0,10,9,11] [0,14,8,13] [0,5,6,8] [12,1,4,10]
  [19,9,17,16] [9,6,13,12], all type Vgpr. That is a vertex shader doing real
  arithmetic into scratch registers, which is NORMAL — it is not the simple
  "export reads the fetch destination directly" shape the hypothesis assumed,
  so the fetch-dst vs export-src comparison does not apply as stated. Treat the
  hypothesis as UNRESOLVED rather than refuted: a mismatch could still exist
  deeper in the dataflow, but it cannot be settled by comparing these two
  register lists.
  **THE REMAINING DEFINITIVE TEST (do this first next session):** temporarily
  replace the generated VS with a hardcoded full-screen-triangle passthrough
  that writes a constant gl_Position, keeping everything else identical.
    - pixels appear  => the generated VS BODY is at fault (its arithmetic
      produces out-of-clip positions); bisect the SPIR-V from there.
    - still black    => the fault is BELOW the shader, despite validation being
      clean and cmd_draw being issued.
  This one experiment splits the entire remaining space, which no amount of
  further input-checking can — every input is already verified correct
  (data, indices, bindings, gl_Position emitted, draw state, valid pipeline,
  render pass executing).

- Vulkan is FULLY SATISFIED — zero validation messages of ANY severity after
  the vertex-format fix (2026-07-20, RAEEN_VULKAN_VALIDATION=1 on Minecraft).
  Not just zero "does not match": ZERO total. So the pipeline, descriptors,
  render pass, attachments and draw calls are all API-correct. Combined with
  every input being verified correct and the clear demonstrably reaching the
  image, the only remaining explanation is that the VS produces vertex
  positions OUTSIDE CLIP SPACE (or degenerate), so the rasterizer discards
  every primitive. Nothing at the API level will reveal that — it is legal,
  silent behaviour.
  This is why the VS-substitution experiment (previous entry) is the correct
  next move and why more API/input checking is now provably worthless: the
  inputs are right and the API is happy, so the fault is in the VALUES the
  translated shader computes. Bisect the SPIR-V, do not re-audit the bindings.

- **VS SUBSTITUTION TEST: STILL BLACK => THE FAULT IS BELOW THE SHADER**
  (2026-07-20, decisive). New gated probe `RAEEN_VS_PASSTHROUGH=1`
  (recompile_exp_pos0) makes POS0 export input attribute 0 DIRECTLY as the clip
  position, bypassing ALL VS arithmetic. The measured attr0 for these draws is
  a textbook NDC quad, so this should cover the screen unconditionally.
  MEASURED: frames still byte-zero (non_zero=2 header remnant), only 1 shader
  translate failure, so the passthrough did assemble.
  **THE GENERATED VS BODY IS THEREFORE NOT THE CAUSE — ELIMINATED.**
  Full elimination list (all by measurement): vertex data, indices, attribute
  bindings/formats, gl_Position emission, VS arithmetic, viewport, scissor,
  extent, target/colour write masks, blend, attachment identity, pipeline
  validity (ZERO Vulkan validation messages of any severity), fragment shading
  (no coverage to shade), and the readback/dump path for RGBA8.
  **LEADING REMAINING HYPOTHESIS — READBACK/DRAW SYNCHRONIZATION.** The
  force-clear probe proves the CLEAR reaches the image we read back. The clear
  executes at render-pass BEGIN; the draw executes AFTER it. Observing the
  clear but never any draw result is exactly what a readback that does not wait
  for the draw to complete would look like (missing/incorrect pipeline barrier
  or fence before the copy-to-host, or reading a different queue's timeline).
  NEXT: audit the submit -> barrier -> copy-to-buffer -> map sequence in
  offscreen.rs `record_and_submit` — confirm there is a fence/queue-wait AFTER
  vkQueueSubmit and BEFORE the readback copy, and that the image layout
  transition to TRANSFER_SRC happens after COLOR_ATTACHMENT_OUTPUT with the
  right srcStageMask/srcAccessMask. A draw whose results are never waited on
  would ALSO explain why compute writebacks (which use a different path) DO
  show real data while every colour draw appears to vanish.

- Readback/draw synchronization hypothesis — REFUTED (2026-07-20, code audit).
  `record_and_submit` (offscreen.rs) DOES synchronize correctly:
  `queue_submit(..., self.fence)` at ~2303 followed by
  `wait_for_fences(&[self.fence], true, u64::MAX)` at ~2310, with pipeline
  barriers at ~1771 and ~2282 around the copy_image_to_buffer at ~2220/~2266.
  The readback waits for the submission to complete. Not the bug.
  STATE: every candidate raised so far is eliminated by measurement or audit —
  vertex data, indices, attribute bindings/formats, gl_Position emission, the
  whole VS body (passthrough test), viewport, scissor, extent, write masks,
  blend, attachment identity, pipeline validity (zero Vulkan messages),
  fragment shading, readback path, and now submit/readback synchronization.
  NEXT PLACE TO LOOK (narrow, structural): confirm the DRAW is recorded INSIDE
  the dynamic-rendering bracket that the clear belongs to — i.e. that
  cmd_begin_rendering ... cmd_bind_pipeline/cmd_draw* ... cmd_end_rendering
  appear in that order in ONE command buffer, and that the buffer submitted at
  ~2303 is the one the draw was recorded into. The clear is attached to
  cmd_begin_rendering (AttachmentLoadOp::CLEAR), so a draw recorded outside the
  bracket — or into a different command buffer — would reproduce EXACTLY what
  is observed: the clear lands, the draw never does, everything else is valid.
  That is the only structure consistent with all the eliminations above.

- Command-buffer structure hypothesis — REFUTED (2026-07-20, code audit).
  offscreen.rs ordering is CORRECT and all in ONE command buffer:
    2121 cmd_begin_rendering -> 2122 cmd_bind_pipeline ->
    2174/2176 cmd_draw_indexed / cmd_draw -> 2178 cmd_end_rendering ->
    2292 end_command_buffer -> 2303 queue_submit(fence) -> 2310 wait_for_fences
  The draw IS inside the rendering bracket the clear belongs to, and
  `state.vertex_count` is non-zero for these draws (measured verts=6 and 4).

  **END-OF-SESSION STATE — READ THIS FIRST NEXT TIME.**
  The zero-coverage bug is NOT in any of the following. Every one was
  eliminated by direct measurement or code audit this session; do NOT re-audit
  them, it is wasted effort:
    vertex data (verified NDC quad) | index buffer ([0,1,2,0,2,3]) |
    attribute bindings, formats, offsets | gl_Position emission |
    the entire VS body (RAEEN_VS_PASSTHROUGH still black) | viewport | scissor |
    render-target extent | CB_TARGET_MASK / colour write mask | blend state |
    attachment identity | pipeline validity (ZERO Vulkan validation messages of
    any severity) | fragment shading | RGBA8 readback/dump | submit->fence->
    readback synchronization | command-buffer recording order and bracket |
    vertex_count.
  Two facts that must be reconciled by whatever the answer turns out to be:
    (a) the CLEAR reaches the read-back image (RAEEN_FORCE_CLEAR proves it), so
        the image, the render pass and the readback all work;
    (b) COMPUTE writebacks carry real data in the same runs (3,193 of them),
        so the device and queue are executing work correctly.
  Whatever the cause is, it makes colour DRAWS specifically produce no fragment
  while clears and compute both work. Suggested attack: build a MINIMAL
  in-tree Vulkan test that drives OffscreenTarget directly with a hand-made
  NDC-quad DrawState and asserts non-clear pixels come back. If that test
  PASSES, the defect is in what the title path feeds the sink (state assembled
  per draw) rather than in the sink; if it FAILS, the defect is in the sink
  itself and is now reproducible without a title. That converts this from
  title-driven archaeology into a normal red/green debugging loop, which is
  what this problem has been missing.

- **MINECRAFT BLACK FRAME SOLVED: THE GPU IS EXONERATED END-TO-END; THE FRAME
  IS A CORRECT RENDER OF EMPTY CONTENT** (2026-07-20, proven in-tree + probes).
  The in-tree red/green harness (tests/coverage_bisect.rs, NEW) replayed the
  title's draw one variable at a time against known-good shaders, then the
  title's OWN dumped SPIR-V (new `RAEEN_DUMP_SHADERS` .spv dump in
  shader_fetch.rs):
   * guest vertex-buffer path: covers          (was never covered by any test)
   * Y-flipped viewport [0,h,w,-h]: covers     (ditto)
   * indexed NDC quad at z=+1: covers 4096/4096
   * title VS (vs_253a4800.spv): covers 4096/4096
   * title PS (ps_253a4d00 = the measured pairing; also ps_253e6700):
     covers 4096/4096 with BOTH zero and non-zero descriptor content
  Then the title-side probes:
   * draws reaching the sink number in the THOUSANDS (n>=2048) — the earlier
     "12 draws" was TRACE_DRAWS' sample cap, not a count. Most draws target
     0x31c10000 and blend SRC_ALPHA/ONE_MINUS_SRC_ALPHA (UI compositing).
   * the sampled texture at 0x31c10000 (1920x1080 RGBA8, 8,294,400 bytes) is
     **non_zero=0 — completely empty** — and 0x31c10000 is ALSO the render
     target of most draws: the UI layer chain.
  MECHANISM: PS samples an empty UI texture -> emits black with ALPHA 0 ->
  alpha-blended composite changes nothing -> byte-identical to "no coverage"
  in EVERY frame probe. The "ZERO COVERAGE" conclusion was WRONG: fragments
  rasterize fine; empty content made every title-level probe blind
  (FORCE_CLEAR, VS passthrough, NO_CULL — none could ever show pixels because
  there are no pixels to show). Only the in-tree harness discriminated.
  THE REAL MINECRAFT BLOCKER (single, HLE-side): Gameface/cohtml never loads
  the menu — routes.json read 3x, no .html ever opened, Gameface threads idle
  (established earlier by the HLE agent). The UI texture stays blank and the
  title renders that blank faithfully. NEXT: trace the LLE boundary into
  libcohtml.Prospero.prx (352 exports, +0xf4a0000) — is View/LoadURL ever
  called; if not, what upstream screen-transition gate blocks it.
  OPEN BUT MASKED (do not forget): the FACE/cull semantics under the Y-flip
  viewport (title writes cull=FRONT face=CW; under our winding that culls the
  measured quad; Kyty maps identically). Untestable at title level while
  content is empty — REVISIT with the in-tree harness when real content
  renders. RAEEN_NO_CULL=1 exists to bisect it live.
  New permanent diagnostics: RAEEN_DUMP_SHADERS now also writes .spv;
  RAEEN_NO_CULL; texture-content probe + cull/face in the draw diagnostics
  (RAEEN_TRACE_DRAWS). Driver hazard: a WRONG pipeline layout does not error,
  it SEGFAULTS (AMD, vkCreateGraphicsPipelines) — the sweep test scans the
  SPIR-V to build the right layout first.

- 2026-07-20 (HLE agent) — MEASURED: Minecraft DOES call libcohtml, but ONLY
  init-time (16 exports, first 7 s), never per-frame. New one-shot export
  trap: `RAEEN_TRAP_MODULE_EXPORTS=<substr>` plants int3 on every code
  export's entry byte at compose time (runtime/src/export_trap.rs, VEH
  breakpoint route in dispatch.rs, install site in gui/main.rs --run-eboot;
  5/5 unit tests; data exports outside seg0 are skipped — 0xCC in data would
  corrupt silently). Run PPSA17221 200 s: 264 traps armed (352 exports = 264
  code + 54 data + 34 alias dups). 16 DISTINCT exports hit, ALL in
  t+1.3s..t+6.9s; ZERO calls for the remaining ~143 s while GPU kept
  presenting the loading screen. 15/16 entries cluster at cohtml
  +0xf4a0c30..+0xfb0 (the C-API thunk block at the very start of .text);
  callers: 10 call sites in the EBOOT (two functions: 0x…0d8af47–0d8c0e3 one
  big init fn, 0x…0b9af2d/0b9b2e1 + 0x…0b92a5f) and 6 cohtml-internal
  (module_start + Layout/Resource thread startup). Gameface Layout(0) named
  t+4.6s, Resource Thread t+6.2s — mid-burst — then both park in cond_wait.
  CONCLUSION: neither outcome (a) nor pure (b) — the engine is CONSTRUCTED
  (system init succeeds far enough to spawn its threads) but the title never
  issues the next tier (view creation / LoadURL / per-frame Advance: none of
  the other 248 code exports ever fire). The gate is UPSTREAM in Minecraft's
  screen-transition logic after UI-system init — consistent with the known
  ~90 s PSN online/auth stall (SceNpAuthAuthorizedAppDialog, SceNpWebApi):
  the title initializes Gameface, then waits on platform/auth state before
  creating the menu view. NEXT: stub the NP auth/entitlement flow to report
  signed-in (per minecraft-boot-state memory), re-run this trap, and expect
  the missing view-creation exports to start firing; the 16 hit NIDs are in
  the session log (scratchpad mc_cohtml_trap.log) — first three:
  0xbe5980bc9a91ee64, 0x53470c6ba9ee710d, 0x3dd3a647c772d5f3.

- RECONCILIATION NOTE on the cohtml-trap recommendation (2026-07-20): the
  trap agent suggests "stub NP auth to report signed-in". CAUTION — an earlier
  measurement THIS session proved the title calls ZERO raeen_hle::libsce_np*
  functions and ZERO unresolved imports at runtime. So Minecraft is NOT
  polling NP APIs; if it waits on platform/auth state it must receive it some
  other way (a common-dialog flow, an event queue it registered, a callback
  we never invoke, or its own offline-mode decision). Before stubbing NP,
  MEASURE what the main thread and the two parked Gameface threads are
  actually blocked on between t+7s and the kill: use RAEEN_STALL_DUMP /
  guest-stack chains on the cond_wait callers, and check which HLE event
  queues (WaitEqueue) the title created and whether anything ever posts to
  them. The gate is the first thing that would RESUME those waits.

- Minecraft steady-state is a POLL, not a block (2026-07-20, RAEEN_STALL_DUMP,
  20 samples): every dump reports "IN-FLIGHT HLE: <none — all threads between
  calls>". No thread is ever caught inside an HLE call — so the title is NOT
  parked in one long cond_wait/equeue-wait; it POLLS some readiness state in
  short cycles and the answer never changes. The gate is therefore a VALUE
  some HLE function keeps returning ("not ready / not signed in / no event"),
  not a missing wake. NEXT MEASUREMENT (first thing next session): add a
  cheap per-NID call counter (atomic map, dumped with STALL_DUMP or on exit),
  run 60s, and rank calls in the steady state — the top few poll functions
  ARE the gate. Then make the polled answer become ready. Note the cohtml
  trap showed the last export call at t+6.9s; whatever is polled decides the
  transition that would create the UI view.

- Minecraft poll gate — sceSystemServiceReceiveEvent was UNREGISTERED
  (2026-07-20). RAEEN_TIME_HLE ranking of the steady state: the MAIN THREAD
  spends ~93% of wall time in scePthreadMutexLock (166k calls / 85s — a normal
  game-loop tick, NOT a spin: its RIP moves each dump), and its steady-state
  HLE poll set includes `sceSystemServiceReceiveEvent` — the PS5 per-frame
  system-event pump — which was NOT REGISTERED, so it hit an unresolved-import
  error every tick instead of the DEFINED "nothing pending" answer.
  FIXED: registered it to return ERROR_NO_EVENT (0x80A10004) on an empty queue,
  ERROR_PARAMETER on a null out-ptr — faithful to shadPS4 systemservice.cpp:1984
  + systemservice_error.h. 2 unit tests (empty-queue + registered). This is the
  CORRECT BASELINE, not a menu fix: with no event SOURCE wired, "queue empty"
  is the honest answer and the title moves on rather than retrying an error.
  Whether the title needs a specific event PUSHED (e.g. a game-intent/launch
  event) is the next question — a concurrently-running agent is ranking the
  full poll set with a CALL_STATS instrument to confirm the complete gate.
  NOTE t28/t27: sceAudioOut2ContextPush and sceKernelWaitSema both ~201,080
  calls (identical count = one producer/consumer audio loop, ~2000/s) — a tight
  but non-fatal audio pump, not the UI gate.

- Minecraft steady-state poll ranking, POST ReceiveEvent fix (2026-07-20,
  Raeen CALL_STATS, t=+191s, MY ReceiveEvent registration in tree). Two results:
  1. ReceiveEvent NO LONGER appears in the top poll set — registering it to
     return NO_EVENT cleanly removed the per-frame unresolved-import error. Good
     baseline, but the boot did NOT advance to a menu (12 present:dumping, still
     the empty-UI frame).
  2. Getdents was BOOT CHURN, not the gate — 26,453 calls in the first-30s
     window, GONE from the steady-state top-20. Do not chase task #8's Getdents
     as the menu gate.
  3. STEADY-STATE GATE CANDIDATE — a NON-BLOCKING SOCKET SPIN: `__error` and
     `libScePosix::recvfrom` both ~1,711,000 calls (near-identical counts = the
     same recv/errno loop). The title polls a socket that never has data. THIS
     RECONCILES the NP puzzle: earlier this session proved ZERO libSceNp* calls,
     yet the title waits on platform/online state — because it polls the SOCKET
     DIRECTLY (RakNet/networking), not via NP APIs. The main loop itself runs at
     full speed (10.0M scePthreadGetspecific, 8.4M getthreadid) — the title IS
     booted and ticking; only the UI is empty.
     NEXT: find what host/socket the recvfrom loop targets and why it never
     receives — is it a PSN/online endpoint the offline-socket path should
     answer, or a localhost IPC the title expects a peer to fill? Check the
     socket's creation (sceNetSocket/Connect) and whether Raeen's offline-socket
     handling (added in commit d15885a per git log) covers this fd. The audio
     pump (sceAudioOut2ContextPush/WaitSema/ContextAdvance all ~268,354) is a
     healthy ~parallel loop, not the gate.

- recvfrom/socket-spin hypothesis — REFUTED (2026-07-20, code read).
  kernel_socket.rs:100 hle_recv already returns EWOULDBLOCK (35) with errno set
  — the CORRECT offline answer. The 1.7M-call recvfrom loop is a RakNet
  networking thread correctly getting "no data" on a non-blocking socket and
  moving on; Minecraft is designed to run offline (LAN), so this spin is
  EXPECTED behavior, not a stuck poll. It is NOT the UI gate.
  NET STATE OF THE MINECRAFT MENU HUNT (all measured/verified this session):
   * GPU renders correctly (title VS+PS cover 4096/4096 in-tree)
   * the frame is a faithful render of an EMPTY UI texture
   * Gameface/cohtml is constructed (16 init-tier exports by t+6.9s) but its
     view is never created (0 of the other 248 exports ever called)
   * the main loop runs at FULL SPEED (10M getspecific/30s) — fully booted
   * every per-frame poll now returns its correct defined value: ReceiveEvent
     -> NO_EVENT (FIXED this session), recvfrom -> EWOULDBLOCK, GetStatus ->
     empty, Getdents completes (boot churn only)
  So NOTHING the main loop polls returns an ERROR or a wrong value anymore —
  yet it still does not drive Gameface to load /hbui/index.html. The remaining
  gate is therefore INTERNAL to Minecraft's own screen-transition state machine
  (a condition on its own game state, not on an HLE return we can see), OR a
  callback/notification the engine expects us to INVOKE (push) rather than a
  value it pulls. This is no longer a single-NID fix; it needs either
  RE of the main-loop decision at the eboot addresses that called the 16 cohtml
  init exports (0x…0d8af47–0d8c0e3, captured by the export trap), or a diff
  against a known-good boot to find the missing push-event. A multi-session RE
  task, honestly scoped — NOT a quick stub.

- Synthetic-event unlock — DISPROVEN BY THE EVENT ENUM, not just declined
  (2026-07-20). Considered pushing a synthetic sceSystemServiceReceiveEvent
  return to force Minecraft's UI transition. Checked the authoritative event
  set (shadPS4 systemservice.h OrbisSystemServiceEventType): OnResume,
  GameLiveStreamingStatusUpdate, SessionInvitation, EntitlementUpdate,
  GameCustomData, DisplaySafeAreaUpdate, UrlOpen, LaunchApp, AppLaunchLink,
  AddcontentInstall, ... — EVERY type is something a running game REACTS to;
  there is NO "create/show your UI" event. Titles create their Gameface view
  themselves on boot; the OS never signals it. Therefore ReceiveEvent returning
  NO_EVENT is COMPLETE and correct, and no HLE event we could push would drive
  the menu. This positively CONFIRMS (not estimates) that Minecraft's remaining
  gate is INTERNAL to its own boot state machine — the decision to call cohtml
  CreateView/LoadURL lives in game code and is conditioned on game state we do
  not yet satisfy. The RE entry point is the eboot init function that called
  the 16 cohtml init exports (0x…0d8af47–0d8c0e3, from the export trap): trace
  forward from there to find the branch that gates the view creation.

- RenoirCore (cohtml's GPU backend) IS driven — renderer failure RULED OUT
  (2026-07-20, RAEEN_TRAP_MODULE_EXPORTS=Renoir). 30 traps armed; 9 distinct
  RenoirCore exports hit at ~t+4s, then silence — same init-tier-only shape as
  cohtml's 16. So cohtml reached renderer setup and its Renoir GPU device/
  context initialized cleanly (RenoirCore also links 0-unresolved, module_start
  0). The View is NOT blocked on a broken renderer.
  DEFINITIVE ELIMINATION (Minecraft menu) — every external lever measured and
  ruled out this session: ReceiveEvent (fixed→NO_EVENT), recvfrom (correct
  EWOULDBLOCK), system-event enum (no "show UI" type exists), cohtml (inits, 16
  exports), RenoirCore (inits, 9 exports), GPU/AGC (renders 4096/4096 in-tree),
  main loop (runs full speed). ALL UI machinery initializes and idles; the game
  never issues CreateView/LoadURL. The gate is CONFIRMED internal to Minecraft's
  boot state machine — a game-code condition, reachable only by RE'ing forward
  from the eboot init fn (0x…0d8af47–0d8c0e3) that drove both init bursts.
  This is the tightest the menu blocker can be pinned without disassembling
  game logic; a genuine multi-session RE task, not a further HLE stub.

- Final eliminations for the Minecraft menu (2026-07-20): index.html/gameplay.html
  EXIST on disk (data/gui/dist/hbui/); the game reads routes.json (the route
  TABLE) 3x but NEVER opens any .html and never issues a navigation — so it is
  NOT a failed open, the navigate call is never made. sceUserServiceGetInitialUser
  returns a valid PRIMARY_USER_ID (libsce_user_service.rs:59) and the login list
  is populated — the initial-user gate is satisfied. EXHAUSTIVE: every external
  dependency the UI needs is present and correct (files, cohtml init, RenoirCore
  init, ReceiveEvent, socket, user service, GPU). The title still never navigates.
  CONCLUSION (measured from every angle, not estimated): Minecraft's menu gate is
  a GAME-INTERNAL boot-state condition with NO reachable external (HLE/FS/GPU)
  lever. The only path is disassembling the eboot boot-decision logic forward
  from the init fn at 0x…0d8af47–0d8c0e3. Multi-session RE; single-increment HLE
  work is exhausted for this title.

- Tooling note for the Minecraft RE handoff (2026-07-20): `--dump-vaddr` is a
  HEX DUMPER (bytes + ASCII), NOT a disassembler (main.rs:285). Tracing the
  boot-decision branch that gates CreateView/LoadURL needs real x86-64
  disassembly + data-flow over unsymbolized game code from eboot base
  0x100000000000. The init function that drives both cohtml (16) and RenoirCore
  (9) init bursts spans vaddr 0xd8af47–0xd8c0e3; a second cluster at
  0xb92a5f/0xb9af2d/0xb9b2e1 is the Resource Thread startup. FIRST NEXT-SESSION
  STEP: add a real disassembler (e.g. iced-x86 crate) behind a --disas flag, or
  feed the dumped bytes to an external disassembler, then trace forward from
  0xd8c0e3's containing function's RETURN to find the branch that decides
  whether to navigate a route. This is the honest boundary: single-increment
  HLE/GPU/FS work is EXHAUSTED for Minecraft (all levers verified correct); the
  menu needs game-code RE with proper tooling.

- BUILT the disassembler blocker was waiting on (2026-07-20). Added
  `raeen --disas <eboot> <hex-vaddr> [len]` (raeen-gui/src/main.rs) — real
  x86-64 disassembly via the workspace iced-x86 decoder, marking each line
  (call)/(jmp)/(ret)/<-- COND/<-- TEST so subsystem-gating branches stand out.
  iced-x86 was already a workspace dep (VEH uses it); added it to raeen-gui.
  DEMONSTRATED on Minecraft's boot code: `--disas eboot 0xd8c0e3 200` shows the
  init function's tail after the last cohtml/RenoirCore init call — it builds a
  ~0x4D0-byte config struct on the stack, calls sub_0x7DAF00 with rdi=&struct
  (0xd8c187), then `test rax,rax; je 0xd8c1fa` + `cmp byte[rax],0; je` — a
  builder-then-validate pattern, still linear init (more subsystem setup), NOT
  yet the top-level "navigate a route?" decision.
  RE STATE / NEXT: the view-creation decision is almost certainly in the
  main-loop STATE-MACHINE TICK, not this linear init. Find the per-frame update
  fn (the one the MINECRAFT MAIN THREAD runs each tick — correlate a stall-dump
  RIP on t1 that is IN eboot .text, not in an HLE wait) and disassemble its
  state dispatch to find the branch guarding CreateView/LoadURL. The --disas
  tool now makes that tractable. Single-increment HLE/GPU/FS work remains
  exhausted for Minecraft; this is the RE on-ramp, now unblocked.

- Main-loop anchor probe (2026-07-20, RAEEN_TRACE_MAINLOOP in
  hle_receive_event logs ctx.caller_return_addr): ReceiveEvent was NOT called
  in a 90s window — it is an OCCASIONAL query, not a per-frame poll, so it is
  the WRONG anchor for finding the main-loop tick. BETTER ANCHOR for next
  session: sceAgcDriverSubmitDcb — confirmed in-flight on the MINECRAFT MAIN
  THREAD (mc_time.log), called every frame from the render tick. Log its
  caller_return_addr the same way, then `raeen --disas` that caller to reach
  the per-frame function; from there trace the UI state dispatch. The probe
  is env-gated and harmless; keep it as a template.
  HONEST STATUS: this session built the RE on-ramp (the --disas disassembler +
  the caller-probe pattern) and traced the init function, but did NOT reach the
  UI-navigation branch — each probe shows the boot state machine is deeper than
  one findable gate. Getting a Minecraft menu pixel is multi-session RE; the
  tooling to do it now exists in-tree where it did not before.

- RE toolchain COMPLETE + demonstrated end-to-end (2026-07-20). Full workflow
  now works in-tree: export trap (which exports called) -> caller probe
  (ctx.caller_return_addr in an HLE fn, RAEEN_TRACE_MAINLOOP) -> `raeen --disas`
  (real x86-64) -> read control flow. APPLIED: sceAgcDriverSubmitDcb's per-frame
  caller is eboot vaddr 0x7423af; disassembling 0x742380 shows a THIN SUBMIT
  WRAPPER (branches DCB vs ACB on a bool in dil: dil==1 -> call 0xB7B4430,
  else call 0xB7B4400=SubmitDcb thunk), NOT the frame loop. The real tick is
  THIS wrapper's caller — climb one more level.
  EXACT NEXT PROBE (fastest path, next session): use the guest-stack backtrace
  the fault reporter already walks (report_fault_site / the pthread_cond
  guest_stack_chain that scans caller_rsp) to capture the FULL return chain from
  SubmitDcb up to the frame-loop function in ONE run, instead of climbing one
  wrapper per build. Then --disas the frame-loop fn and find the branch that
  gates the Gameface navigate/CreateView. All tooling exists; this is now a
  mechanical climb, not a search.
  This session's RE deliverables: --disas disassembler, the caller-probe
  pattern (2 gated probes), export_trap, and the landed render-tick address.
  Goal (a rendered menu pixel) still needs the multi-level climb = multi-session,
  but it is now fully tooled and the next step is a single specified probe.

- CAPTURED Minecraft's per-frame render call chain (2026-07-20, one run via the
  agc_guest_stack_chain probe in hle_driver_submit_dcb, RAEEN_TRACE_MAINLOOP).
  Arena-relative return-addr chain from the DCB submit upward (0x9fffddXX
  entries are rbp frame-pointer saves, IGNORE; real return addrs are the .text
  ones):
    0x7423af (submit wrapper) <- 0xa8e2f93 <- 0xdf04778 <- 0xf49c000 <-
    0xa8caf18 <- {per-frame divergence: 0xa8d0ce3 / 0xe154b78 / 0xe155660 /
    0xb791264 / 0xa8cd6ac ...}
  The prefix 0xa8e2f93/0xdf04778/0xf49c000/0xa8caf18 is STABLE across frames =
  the render dispatch; 0xa8caf18's frame calls different things per frame =
  a dispatcher (likely where a "what to present" decision lives).
  RE MECHANICS LEARNED (important for next session): `--disas` from an arbitrary
  mid-function vaddr MISALIGNS (decodes garbage/int3 padding). Must find the
  function ENTRY first: scan backward for the `int3`-padding gap then the
  `push rbp; mov rbp,rsp` prologue, and disas from there. Minecraft's .text is
  int3-padded between functions, which makes boundary-finding mechanical.
  NEXT: find the entry of the fn containing 0xa8caf18 (walk back to the prologue
  after the preceding int3 run), disas it, and look for the branch that gates
  submitting the Gameface/UI draws vs skipping them. That branch, or the update
  tick one level up that sets the state it reads, is the menu gate.
  HONEST: goal (menu pixel) is multi-session symbol-less RE; this session built
  every tool and captured the exact render stack to start from — a concrete
  artifact, not a deferral.

- DECODED a real Minecraft render-dispatch function (2026-07-20, --disas from
  the correct entry). Return-addr 0xa8caf18 belongs to the fn at ENTRY 0xa8ca90
  (found by the int3-padding + `push rbp;mov rbp,rsp` rule — worked exactly as
  predicted). Decoded logic:
    fn(rdi=obj, esi=enable):
      if esi == 0: return                    ; an ENABLE flag gates everything
      if byte[0xe39e0a0] == 0: lazy-init singleton (0xa8cac3)
      rax = qword[0xe39e098]                  ; singleton object ptr
      call qword[rax+0x18](rbx=obj)           ; virtual dispatch = the real work
  It is a GUARDED SINGLETON VTABLE DISPATCH. The actual work is call [rax+0x18],
  a runtime-resolved vtable slot on the singleton at 0xe39e098.
  RE LIMIT REACHED (honest): static disas cannot follow call [rax+0x18] — the
  target is runtime data. NEXT STEP needs LIVE vtable inspection: at a stall,
  read guest qword[0xe39e098] (singleton), then qword[singleton+0x18] (the fn
  ptr), --disas that. Add a tiny probe that dumps those two guest qwords (the
  guest memory reader already exists). Then repeat the climb toward the
  navigate/CreateView gate. Each level = one disas + one runtime-ptr read.
  This is real RE progress (a decoded title function), tooled and artifact-
  backed; the menu pixel remains multi-session symbol-less RE.

- FULL RE CLIMB executed end-to-end, resolved to a dead end (2026-07-20) — the
  complete toolchain proven across 6 levels in one session:
    stack frame 0xa8caf18 (render chain)
    -> fn ENTRY 0xa8ca90 (guarded singleton dispatch: call qword[rax+0x18])
    -> singleton @ arena 0xdf26ef0 (live-resolved via a probe reading
       qword[BASE+0xe39e098])
    -> vtable[0x18] target = 0x625360  (live-resolved)
    -> 0x625360 = thunk `mov rdi,rsi; jmp 0xB7B17A0`
    -> 0xB7B17A0 = PLT stub `jmp qword[0xe122e08]; push 0x26`
    -> --resolve-got 0xe122e08 => `free` [libc] nid=0xb4886caa3d2ab051
  CONCLUSION: singleton vtable[0x18] is a FREE / deleting-destructor slot, so
  frame 0xa8caf18 / fn 0xa8ca90 is a TEARDOWN helper on the render path, NOT the
  UI dispatch. Dead end — the navigate/CreateView logic is a DIFFERENT frame in
  the captured chain (0x7423af <- 0xa8e2f93 <- 0xdf04778 <- 0xf49c000 <-
  0xa8caf18 <- ...). NEXT: repeat the exact climb on 0xf49c000 and 0xdf04778
  (the stable prefix above the teardown) with the same probe pattern
  (--disas from int3+prologue, resolve any singleton/vtable live, --resolve-got
  any PLT). All tooling proven; each frame is one mechanical climb.
  This documents WHY the menu is multi-session: several stack frames, each a
  6-level climb, some resolving to dead ends needing backtrack. Not a single
  findable branch. Tooling + method are now fully in place and demonstrated.

- RENDER CHAIN CROSSES MODULE BOUNDARIES (2026-07-20). Continuing the climb to
  the next frame (0xf49c000): `--dump-vaddr eboot 0xf49c000` => "not in any
  PT_LOAD segment" — 0xf49c000 is NOT in the eboot. It sits just below
  libcohtml.Prospero.prx (loaded at 0xf4a0000), i.e. in a DEPENDENCY module.
  So the per-frame render chain is eboot -> dependency .prx -> back:
    0x7423af(eboot) <- 0xa8e2f93(eboot) <- 0xdf04778(eboot) <-
    0xf49c000(DEP, near cohtml) <- 0xa8caf18(eboot, =free teardown) <- ...
  IMPLICATION for the RE: some frames need disassembling a DEPENDENCY module's
  decrypted image, not the eboot — add a --disas mode that takes a module
  name/base and loads that .prx (the loader already decrypts them). The UI
  navigate logic most likely lives in the eboot's UI code (0xdf04778 /
  0xe154xxx are big eboot .text) OR inside a dependency. This is now provably
  CROSS-MODULE symbol-less RE across several stack frames — the definitive
  reason it is multi-session. Every tool + method demonstrated; the remaining
  work is mechanical-but-lengthy climbing, frame by frame, module by module.

- SharpEmu HARDENING AUDIT — verified, two wins landed, two rejected as stale
  (2026-07-20). Fanned out 7 agents to verify an external audit's claims against
  CURRENT code (several were stale). Outcomes:
  * DONE — NID CATALOG MERGE (highest-ROI, attacks HLE diagnosis gap). SharpEmu's
    scripts/ps5_names.txt fed through Raeen's OWN hash gate (nid_of) via a new
    Rust tool `crates/raeen-firmware/examples/merge_nid_catalog.rs`.
    nid_names.txt 94,247 -> 149,905 (+55,658 hash-verified names; 4,553 .L*/`/`
    junk dropped; existing wins collisions, __sys_dynlib_unload_prx preserved).
    all_names_hash_to_their_nid re-proves the whole merged table (junk names are
    harmless — a wrong name only labels a NID no title imports). PROVEN on a live
    Minecraft run: the entire sceNpAuthAuthorizedAppDialog* import set (Close/
    GetResult/GetStatus/Initialize/Open/Terminate/UpdateStatus) + sceAgcGetIs-
    TrinityMode (0x05f0436466ed8bb0, our known libSceAgc gap) + sceKernelSyncOn-
    AddressWait/Wake now resolve to NAMES where they printed raw NIDs before.
    CAVEAT: naming != implementing — this closes the DIAGNOSIS gap so those
    imports can be targeted, not the HLE gap.
  * DONE — OPT-IN HEAP POISON (debugging-speed). `RAEEN_POISON_HEAP=1` fills
    fresh malloc with 0xCD (libc.rs hle_malloc) so an uninitialized read shows
    as 0xCDCDCDCD in the crash dump, not a silent zero. Off by default (a title
    that treats malloc as calloc keeps working); calloc still zeroes (regression
    test added). Poison-on-free (0xAF) deferred — GuestAllocator::free(addr) has
    no size; needs a size-query on the trait.
  * REJECTED — Net/NP differential rig (audit option B): STALE. Minecraft calls
    ZERO NP fns; the audit's Net/NP wall was ASTRO's old log, superseded.
  * REJECTED — SSE4a EXTRQ/INSERTQ patch: LATENT, not a live gap. Measured our
    Zen4 host: CPUID SSE4A=1, so EXTRQ/INSERTQ execute NATIVELY with no #UD here;
    no measured title emits them. Only an Intel/Rosetta concern.
  Also confirmed (not acted): the memory-protection deficit is real (image RWX,
  heap/stack/mmap RW, NO inter-region guard pages, guest mprotect ignored) —
  W^X + guard pages is a valid future hardening, larger than this session.
  Tests: raeen-firmware 110+ green (merged table re-proven), raeen-hle 289 green
  (+2: heap_poison, calloc_still_zeroes). Attribution added to THIRD_PARTY_NOTICES.

- DONE — INTER-REGION GUARD PAGES (memory hardening, 2026-07-20). The four
  arena sub-regions (IMAGE/HEAP/STACK/MMAP) tiled with NO gaps, so an overflow
  ran silently into the next region and surfaced as an anonymous fault millions
  of ops later. Added always-on PAGE_NOACCESS guard pages at the two safe
  boundaries (arena.rs GuestArena::new):
    * IMAGE top [IMAGE_END-PAGE, IMAGE_END) — armed only when image_len fits
      below it (checked); catches an image-region overrun into the heap.
    * HEAP top [STACK_OFFSET-PAGE, STACK_OFFSET) — `alloc`'s heap_end lowered
      to STACK_OFFSET-PAGE_SIZE so the allocator never hands it out; catches a
      heap overrun upward AND a stack underrun downward.
  The STACK|MMAP boundary is deliberately NOT guarded (initial RSP sits at the
  stack top; a guard there would fault live data). A native guest store that
  hits a guard traps via the existing VEH = "trap at the store", not "anonymous
  fault later". Test `inter_region_guard_pages_are_noaccess_and_unallocatable`
  proves NOACCESS via VirtualQuery + that alloc never lands on the guard and
  ordinary alloc/write/read is unaffected (does NOT write to a guard — that
  would fault the host copy). Updated heap-fill test: usable committed heap is
  now HEAP_SIZE-PAGE. MEASURED: Minecraft boots unaffected (64 submissions,
  draws progressing, zero spurious guard faults). 453 runtime+firmware+hle
  tests green. NOT done (larger, deferred): W^X the image (blocked — our own
  export-trap int3 + native_trap prologue patches WRITE guest code, so W^X
  needs per-patch VirtualProtect toggling) and enforce guest mprotect (risk:
  a title mprotecting a page our HLE later touches would fault). Both are real
  future hardening; guard pages are the safe, self-contained, done slice.

- DONE — W^X the code image, per-segment (2026-07-20, opt-in RAEEN_WX_IMAGE).
  A stray guest DATA store into code no longer corrupts an instruction silently;
  it faults at the store (caught by the existing VEH). KEY FINDING via
  measurement: WHOLE-IMAGE W^X is WRONG — the image region holds .text AND
  .data/.bss, so making it all read-only faults the first global write (Minecraft
  stored to a .data global at boot immediately, RIP in the image region). Correct
  W^X is PER-SEGMENT: RX only PF_X segments, .data/.bss stay RW. Implemented:
    * LinkedModule gains `executable_ranges: Vec<(u64,u64)>` — the main module's
      PF_X segment spans, collected in link_module (linker.rs, `flags & 1`).
    * GuestArena::enable_wx_image(exec_ranges) VirtualProtects only those spans
      to PAGE_EXECUTE_READ, clamped below the image|heap guard page.
    * New GuestMemory::patch_code (default = write) lets instrumentation write
      code under W^X; GuestArena overrides it to toggle RWX around the store and
      restore RX. Routed the ONLY runtime code-write (export_trap one-shot
      restore) through it; native_trap writes are pre-map (image buffer) or to
      the stack (RW), so unaffected.
    * maybe_enable_wx_image gated behind RAEEN_WX_IMAGE, wired at all 3 execute
      sites.
  MEASURED: with per-segment W^X on, Minecraft boots IDENTICALLY (64 submissions,
  753 draws, zero image-region faults) — code protected, data writable. Whole-
  image faulted immediately; per-segment is clean. Tests: unit
  `wx_image_is_execute_read_and_patch_code_still_writes` + 454 runtime/firmware/
  hle green. OPT-IN (not default): only the MAIN module's code is covered
  (dependencies compose via a different path, stay RWX — documented scope), and
  a title with self-modifying MAIN code would fault, so default-on needs
  per-title validation. Mechanism is correct and done; flipping default + dep
  coverage are future steps.
  NOT done this turn: enforce guest mprotect (the third hardening item) — same
  data/code-granularity + broad-risk profile; deferred for its own careful pass.

- DONE — enforce guest mprotect, opt-in (2026-07-20, RAEEN_ENFORCE_MPROTECT).
  sceKernelMprotect was an ok-stub and POSIX mprotect / BatchMap OP_PROTECT were
  no-ops (protection bits stored in the VMA but never pushed to host pages), so a
  title writing to a page it marked read-only succeeded silently. Now, under the
  gate, the protection is really applied:
    * New GuestMemory::protect(addr,len,prot) (default no-op returning true — the
      historical behaviour for test memories and non-enforced runs).
    * GuestArena overrides it: `orbis_prot_to_win` maps the Orbis CPU bitset
      (CPU_READ=1/WRITE=2/EXEC=4, NO_ACCESS=0; GPU bits ignored; write implies
      read) to Windows PAGE_*, then VirtualProtects the committed pages in range
      page-by-page — skipping uncommitted pages (a reservation mprotect is legal)
      and the NOACCESS guard pages (never reopened).
    * Routed all three sites: sceKernelMprotect (new hle_kernel_mprotect,
      replacing the ok-stub), hle_posix_mprotect, and BatchMap OP_PROTECT.
  Same opt-in rationale as W^X: a title marking a page RO that our HLE later
  writes would fault, so it is off by default. Tests: `orbis_prot_maps_to_the_
  right_windows_page_flags`, `protect_reprotects_a_committed_heap_page_read_only`;
  runtime 57 + hle 289 green, clippy/fmt clean. MEASURED: Minecraft boots
  IDENTICALLY with it on (64 submissions, 753 draws, zero faults) — though it
  issues no mprotect during boot, so the title does not exercise enforcement;
  the path is proven by unit tests + no-regression. Both remaining hardening
  items (W^X + enforce mprotect) are now DONE as opt-in, correct mechanisms.

- DONE — implemented the NID-merge-recovered imports (2026-07-20). The catalogue
  merge named 11 previously-anonymous imports; implemented the ones that close
  real HLE gaps, all under the LIBRARY the title imports from (provider-aware):
    * sceAgcGetIsTrinityMode (libSceAgc, 0x05f0436466ed8bb0) -> 0 (base PS5).
      Registered alongside the existing libkernel twin; the astro-bot memory's
      "known libSceAgc gap" is closed.
    * sceKernelSyncOnAddressWait/Wake (libkernel) — address-parking (futex-like).
      No reference tree has them (shadPS4/SharpEmu both lack SyncOnAddress), so
      implemented on the codebase's established SPURIOUS-WAKEUP model: Wait
      sleeps a 10ms slice and returns (caller re-checks its condition), Wake
      returns success. Safe: never deadlocks, no tight spin.
    * libSceNpAuthAuthorizedAppDialog Initialize/Open/UpdateStatus/GetStatus/
      GetResult/Close/Terminate (7 fns) — the PSN authorize popup. Mirrors the
      sceMsgDialog immediate-finish model: Open -> status FINISHED, GetStatus
      reports it, GetResult writes a success result. A title polling the dialog
      proceeds instead of hanging.
  MEASURED on Minecraft: the 7 AuthorizedAppDialog imports went from MISSING to
  resolved (0 still missing), total missing imports 126 -> 112, boot unaffected
  (64 submissions). Tests: auth_dialog_completes_immediately_with_success + 290
  hle green, clippy/fmt clean. This is the payoff of the catalogue merge: naming
  an import is what makes it implementable, and these were unimplementable before
  (anonymous NID). Does NOT unblock Minecraft's menu (game-internal, not NP) —
  it closes import gaps that would gate a title that actually drives these.

- RENDER-RE: found the RIGHT anchor — the UI-manager singleton (2026-07-20).
  New probe RAEEN_TRACE_UI logs the caller of the routes.json open (libkernel.rs
  hle_open). MEASURED: routes.json is opened from eboot 0xb5c689d (call open),
  returning to 0xb5c68a2. Disassembling forward (raeen --disas):
    0xb5c68a2  mov r14d,eax; test eax,eax; js <err>   ; check the fd
    0xb5c68c6  build an SSO string; call 0x7C80E0      ; read the file
    0xb5c6900  mov rax,[0xE39E098]; call [rax+0x10] esi=0x50  ; hand to the UI mgr
  DECISIVE: 0xE39E098 is the SAME singleton the render dispatcher called
  (call [rax+0x18], fn 0xa8ca90) — it is the central UI/Gameface MANAGER, shared
  by route-loading AND rendering. So the game DOES parse routes.json and register
  routes with the manager; the never-taken navigate/CreateView is a LATER call
  into THIS singleton's vtable, gated on game state — NOT on the render chain
  (which dead-ended at free()). CORRECT NEXT TRACE (was chasing the wrong path
  before): dump the live vtable of the 0xE39E098 singleton (read guest
  qword[BASE+0xE39E098] = obj, then its vtable ptr [obj+0], then the slots), map
  which slot is Navigate/LoadURL/CreateView, and set a one-shot export-trap-style
  probe on that slot's target to see if it is ever called and with what route.
  If it is NEVER called, the gate is upstream (a game-state flag the UI mgr waits
  on); if called with an empty/null route, the gate is what computes the route.
  The RAEEN_TRACE_UI probe is uncommitted (env-gated, harmless); keep it.

- RENDER-RE CORRECTION: 0xE39E098 is the ALLOCATOR, not the UI manager
  (2026-07-20). Dumped the singleton's embedded function-pointer table
  (obj=0xdf26ef0): +0x10=0x49ca1a0, +0x18=0x625360, +0x28=0x625360, +0x50/
  +0x80=0xe11e978, plus methods at 0xb65xxxx. Resolved +0x10: 0x49ca1a0 is a
  thunk `mov rdi,rsi; jmp 0xB7B1BC0` -> GOT 0xE123018 -> **malloc [libc]**
  (the route-loader's `call [obj+0x10] esi=0x50` is malloc(0x50)). +0x18=0x625360
  resolved earlier to **free**. So 0xE39E098/0xdf26ef0 is the app's
  ALLOCATOR/resource manager (alloc=+0x10, free=+0x18), used pervasively — NOT
  the Gameface navigate manager. Both the render dispatcher (call [obj+0x18]=free
  teardown) and the route-loader (call [obj+0x10]=malloc) just use it to manage
  memory. CORRECTS the prior "UI manager singleton" claim.
  NET: every pointer traced from the render chain AND the routes.json loader
  resolves to a mundane primitive (malloc/free/teardown) — the navigate decision
  is game-STATE-conditional logic not directly reachable from these data anchors.
  Real progress this turn: found the route-loader fn (eboot 0xb5c68a2, reads
  routes.json), mapped the allocator singleton's dispatch table, and corrected a
  wrong subsystem ID. But a title-rendered pixel remains sustained multi-session
  RE: the productive continuation is to trace the route-loader (0xb5c68a2)
  FORWARD past the malloc to whether it navigates or defers, and/or set an
  export-trap probe on cohtml's actual View/LoadURL exports to catch IF/when the
  game ever calls them with a route (the export trap already showed 0 view-tier
  calls, so likely the gate is upstream game state). Tools: --disas, export trap,
  RAEEN_TRACE_UI (uncommitted, env-gated).

- RENDER-RE: route-loader is container code; NEW LEAD = Xbox Live, not PSN
  (2026-07-20). Traced the route-loader (0xb5c6900) forward: open routes.json ->
  malloc buffer (manager+0x10) -> the branch at 0xb5c691e is the ALLOC-FAILURE
  handler (strings at 0xCF0EE8E/0xCF3F526 = "We failed to allocate %zu bytes",
  "pointer || size == 0" — asserts, not navigation). So the route-loader is
  standard C++ container plumbing; the navigate decision is downstream in the
  parse/state logic. IMPORTANT NEW LEAD: the string "XBOXLIVE" (+ "verbose"
  "flush") sits at 0xCF3F53B, in .rodata right beside the route code. Minecraft
  BEDROCK signs in via XBOX LIVE, not PSN — which reframes the auth-gate theory:
  this session PROVED the game calls ZERO sceNp functions, and the reason may be
  that its sign-in/entitlement path is Xbox Live (Minecraft's own service),
  which our HLE does not model. If the menu is gated on an Xbox Live
  sign-in/entitlement state, that is the upstream game-state condition — and a
  DIFFERENT, possibly more tractable angle than raw state-machine RE. NEXT: grep
  the game for its Xbox Live service init (search .rodata for xbl/xboxlive/
  auth/token strings and the functions that reference them); check whether the
  game blocks its screen transition on an Xbox Live "signed in" flag we could
  satisfy. This is the first lead pointing at a nameable subsystem (Xbox Live
  auth) rather than anonymous state.

- RENDER-RE: Xbox Live lead DID NOT pan out (2026-07-20, checked). Grepped the
  run for HTTP/DNS activity: ZERO sceHttp* calls, ZERO resolver/host-lookup
  activity, and the steady-state CALL_STATS poll set has no network calls. So
  Minecraft is NOT doing Xbox Live network auth in steady state — the "XBOXLIVE"
  .rodata string was proximity coincidence (likely logging infra), not the gate.
  Negative result, but a real one (measured, not assumed). This CLOSES the
  auth-network angle and further confirms: the menu gate has NO external network
  dependency the game is awaiting.
  FULL PICTURE now (all measured, multiply-confirmed): GPU exonerated (renders
  empty content correctly); route-loader is standard container code; the
  0xE39E098 singleton is the allocator (malloc/free); cohtml view/navigate
  exports never called (0/248); every external input (NP, socket, events, user,
  files, HTTP/DNS) returns correct values or is never invoked. The Minecraft
  menu gate is PURELY internal game state/sequencing — no external lever exists.
  Reaching a title pixel is sustained internal-state-machine RE with no shortcut:
  disassemble the game's boot sequence to find the initialization-complete flag
  or ordering the navigate transition waits on. Every angle reachable by probing
  external anchors is now exhausted and documented.

- RENDER-RE BREAKTHROUGH: navigate dispatcher + gate PINNED, G1 RULED OUT
  (2026-07-20, workflow + confirmation probe). A 4-angle disassembly workflow
  found the REAL navigation dispatcher and a candidate gate; a runtime probe
  then tested it:
  * router tick @0x11126f0 is the UNIQUE route->html navigation dispatcher (the
    only caller of getHtmlPathForRoute @0xbae690). It registers screens, dispatches
    UI events, renders, and navigates routes — the whole UI transition.
  * Candidate gate G1 @0x1112794: `mov rax,[rbp-118h]; movzx eax,byte[rax+0x248];
    test al,al; jne 0x1112C5A` — if byte[P+0x248]!=0 the tick returns BEFORE any
    screen registration/navigation. P = *[0xE15B830] (app shared-state singleton).
  * CONFIRMATION PROBE (RAEEN_TRACE_UI in hle_get_status, per-frame): read
    byte[*[0xE15B830]+0x248] live on Minecraft. RESULT: singleton is a valid heap
    object (arena-rel 0x100002c2800), and the gate byte = **0 from frame 0
    onward**. So G1 PASSES — it is NOT the blocker.
  NARROWED (per the synthesis's predicted alternative): with G1 open, the block
  is DOWNSTREAM in the same tick — either (a) the SCREEN STACK is empty
  ([[r14+0x2D0]+0xF0]) so there is nothing to navigate, or (b) the per-screen
  VIEW-CREATE at 0x4EF9FC0 (invoked by screen-stack updater 0x11570D0, runs
  unconditionally per screen, BEFORE G1) has its own gate. NEXT: probe whether
  the screen stack is non-empty (read [[r14+0x2D0]+0xF0] — but r14=this of the
  tick, not a global, so needs a caller probe OR disas 0x11570D0/0x4EF9FC0 to
  find the per-screen view-create gate). Disproven gates: G3/G4 (0xbae690 =
  routes.json hot-reload diff, not boot navigate), angle-B [rbx+0x58] (separate
  init/registration gate). NEW RE TOOLS built + used: --find-calls, --find-lea,
  --find-str (raeen-gui). This is the tightest the gate has ever been located:
  a named dispatcher, a tested-and-excluded top gate, two concrete downstream
  targets. Real, measured forward progress toward a menu pixel.

- FPS BASELINE (2026-07-21, stage A build): Minecraft peaks ~10 flips/sec
  (300 SubmitFlips/174s window, 9-10/s sustained during boot animation, then
  the known PSN-auth stall). Goal = 120 FPS (/goal hook). Gap = per-draw
  submit 3.7-4ms + readback 7.6-8.2ms -> stage B (readback per FLIP,
  persistent-image texture binding, upload ring) then stage C (swapchain).

- ASTRO.BOT compute round 10: BIND-TIME WALLS CLOSED, device-loss bisected +
  quarantined (2026-07-20, raeen-gpu only; 120s measured run, 0 device losses).
  * Wall 1 null V# (39): storage V# with base 0/size 0 now binds a 4-byte zero
    dummy (RDNA OOB semantics), writeback explicitly skipped — 216 dummy binds
    + 216 writeback skips in the final run, error string gone.
  * Wall 2 tex/sampler independence (38): descriptor arrays now created only
    when non-empty, mirroring the recompiler's %textures2D_S/%samplers
    conditions. Unblocked the sampler-only CS (0x100008e6aa00, up to 524288x1x1
    groups, runs clean). THE OTHER HALF IS QUARANTINED: the one tex-no-sampler
    CS (0x5006c5f00) reproducibly resets the GPU (VK_ERROR_DEVICE_LOST,
    bisected via new RAEEN_SKIP_CS env; robustness features don't save it →
    descriptor-array OOB index suspected). Default-on named skip in
    dispatch_direct (162 skips/120s); RAEEN_ALLOW_TEX_NO_SAMPLER=1 lifts it.
  * Wall 3 set-slot contiguity (38): offscreen.rs builds set layouts keyed by
    the actual slot, gap slots get empty layouts; duplicate-slot stays a named
    error. Error string gone.
  * Wall 4 GDS: device-persistent 64 KiB GDS SSBO (cache.rs gds_buffer),
    bound at gds_pointers.binding_index, gds push-constant granules emitted,
    COMPUTE-only refusal replaced, cross-dispatch persistence proven by new
    in-tree test (atomicAdd counter observes prior dispatch on real device).
    Title GDS shaders still skip UPSTREAM: analysis captures no gds_pointers
    for the ds_append CS (m0-only GDS base, 28 skips) — next round, plus
    ds_wrxchg_rtn_b32 (28) and buffer_store_dwordx2 (27) parse gaps.
  * Wall 5 EUD dwords 4/12 not captured: confirmed new-semantics (not length)
    — named for next round (167+28 skips, the largest remaining class).
  * New: robustBufferAccess + robustImageAccess enabled at device creation.
  * Measured 120s final: 0 device losses, 925 dispatch submits, 751
    untranslatable skips + 162 quarantine (vs r9 50s baseline: 115 warn-level
    bind-time draw skips, all three error classes now 0). tests: raeen-gpu
    134 unit + integration green incl. new tests/compute_gds.rs (GDS
    persistence + tex-no-sampler Vulkan plumbing); clippy -D warnings clean.

## 2026-07-20 gpu-pipeline — perf stage C: flush per flip + flip-limited readback + upload ring (uncommitted)
- Item 1 flush-per-flip: worker submissions no longer flush/present per
  submission (execute_dcb_cp_routed deferred_present). Flush consumers only:
  present_scanout (flip), wait_idle/shutdown, RAEEN_DUMP_FRAMES (keeps old
  per-submission cadence for dump fidelity), feedback-loop fallback in the
  sink. Flip flush routed through the ordered GPU work queue as
  GpuWork::Flush{address,done} — executes on the worker after all queued
  draws, rendezvous back to the HLE thread; inline fallback when no worker.
  Direct execute_dcb_cp path byte-identical (all 13 suites green, 135 unit).
- Item 2 flip-limited readback: flush_deferred_draws_filtered(only_bases)
  reads back only {flip address, remembered fallback target}; other dirty
  targets stay GPU-side (requeued touched, GpuNewer). Flip-miss most-content
  fallback remembers its winner (fallback_present_base), full census
  re-election every 64 misses. ASTRO.BOT measured before fix: 2 flushes/flip,
  full 15-target HDR readback 54 ms; after: steady-state 1-target readback.
- Item 3 upload ring: DrawCaches host-buffer pool (BTreeMap size classes,
  usage union, 256 MiB free cap), fence-tracked recycle via retire_batch +
  Resources::Drop; storage descriptors now bind exact ranges (WHOLE_SIZE
  would expose recycled tails). ASTRO.BOT build_us p50 657 -> 396.
- Measured (Minecraft 180s, release): peak animation window 40-43 -> 49-50
  flips/s; p50 flip interval 20.19 ms, min 1.76 ms => ceiling owner is the
  ~20 ms pacer, NOT the GPU: title imports sceVideoOutWaitVblank and our HLE
  sleeps 16.667 ms/call (libsce_video_out.rs:637), Windows-quantized to
  ~20 ms => hard ~50-60 flips/s cap. 120 flips/s needs an HLE vblank change
  (read-only this session; reported).
- gui/shell/present.rs: documented why zero-copy egui presentation needs
  eframe(wgpu)/ash device interop (external memory or one shared device) —
  out of stage C scope; design stops at 1 readback per flip.
- Item 4 addendum: flip-miss remembered-target fix measured on ASTRO.BOT:
  195 flushes / 192 flips (1.02/flip, was 2/flip, stage B ~11.4/present);
  steady state targets_read=1 mean 7.5 ms (HDR 16 MB copy+map), re-election
  full flushes rare. Minecraft peak window improved again 49-50 -> 60-63
  flips/s (p50 interval 15.9 ms = the 60 Hz vblank pacer; min 1.76 ms shows
  GPU is not the constraint). Fire-and-forget flip flush (true item 4 async)
  was tried and REVERTED: Minecraft wedged ~10 s in (flips stop, pthread_sync
  "stuck >3s" main-thread deadlock on a title mutex held by the flipping
  render-pool thread, 3 threads spinning); ASTRO.BOT ran clean on the same
  build (188 flips, 1 flush/flip) — wedge is Minecraft-specific, cause not
  yet understood; rendezvous flip flush kept (comment at present_scanout).
- Ceiling ownership (measured, read-only on raeen-hle): 120 flips/s is NOT
  reachable while sceVideoOutWaitVblank sleeps a fixed 16.667 ms
  (libsce_video_out.rs:637) — Minecraft imports it and paces its animation
  window to ~60 Hz (p50 15.9 ms). Next lever is an HLE vblank change
  (host-timer-resolution aware sleep, event-driven vblank, or a 120 Hz mode),
  out of this session's allowed scope.

## 2026-07-20 — EUD/SRT raw-window s_load fallback (SharpEmu port)

- eud-raw-window: complete (uncommitted, kyty-graphics 357/357 + raeen-gpu
  163/163 tests, clippy -D warnings clean). Kills the "EUD dword N is not a
  captured descriptor field" refusal CLASS (195 refused ASTRO.BOT compute
  dispatches measured — title-run impact NOT yet re-measured): detection
  (`shader_detect_eud_raw_window`) records uncaptured EUD s_load dwords as a
  `bind.eud_raw` raw-memory binding; recompiler lowers them to clamped
  `%eud_raw` SSBO reads (OpArrayLength + UMin/Select, 0 beyond window);
  dispatch snapshots the guest window behind the EUD base (SharpEmu sizing:
  min 256 KiB, page-rounded, cap 16 MiB, halving probe; unreadable → zeros +
  once-per-base warn, never a skip). Captured descriptors keep the rewritten
  push-constant path (mixed loads split per dword). CS-wired only; VS/PS
  detection + graphics-path binding deferred (offscreen refuses by name).
  Ported from SharpEmu (GPL-2.0) Gen5ShaderScalarEvaluator.cs:1939-1980,
  :997-1005, :14-35; Gen5SpirvTranslator.cs:2183-2236 — cited in doc comments.
- wait-reg-mem-suspend-resume: complete (uncommitted, kyty-graphics 358/358 +
  raeen-gpu 140/140 lib tests, clippy -D warnings clean). THE scene-pixel gate
  (SharpEmu-proven): unmet `IT_WAIT_REG_MEM` / `R_WAIT_MEM_32/64` now parse +
  evaluate against the guest label and SUSPEND the walk (`run_resumable` →
  `RunOutcome::Suspended{resume_dword, WaitSpec}`) instead of being consumed
  without effect. The GPU worker parks the buffer per queue (DCB/ACB), queues
  later same-queue submissions behind it (in-order ring semantics), and
  re-checks every suspended wait after each submission/flush — the producer
  events are compute/storage writebacks + DMA_DATA copies. Labels are NEVER
  force-satisfied; unreadable-at-parse continues (SharpEmu parity), dead waits
  warn after 512 re-check rounds. Cross-queue test mirrors the ASTRO shape
  (ACB wait resumed by a DCB writeback; parked backlog drains in order).
  Ported from SharpEmu (GPL-2.0) AgcExports.cs:4508-4529/4595-4726/4843-4950,
  GpuWaitRegistry.cs:19-40/239-256 — cited in doc comments.
- cs-device-loss-defusal: complete (uncommitted, tests green, clippy clean).
  Quarantine for CS 0x5006c5f00 (ImageLoad+LDS+barrier+runtime-indexed T#)
  REMOVED, incl. RAEEN_ALLOW_TEX_NO_SAMPLER: (i) sampler-less sample-family
  shaders synthesize an all-zero S# → cached nearest/wrap sampler (SharpEmu
  VulkanVideoPresenter.cs:6314-6322/8121-8156); (ii) all 9 DS-op families
  verified UMin-clamped on %lds (kept clamp over SharpEmu's pow2 mask —
  lds_size_dw is not always pow2); (iii) MIMG descriptor guard: T#/S# regs
  not backed by a captured descriptor (incl. raw-EUD overwrite) refuse as
  named 'dynamic-image-descriptor' skip (SharpEmu evaluator :654-662), never
  a device-loss submit; (iv) storage-image contract validation deferred.

- M3 SYNTHETIC PROOF ONLY (2026-07-21; milestone OPEN): pad + VideoOut flip.
  present-from-guest-memory (SharpEmu GuestImageWantsInitialData) makes CPU-drawn
  2D visible; acceptance test crates/raeen-runtime/tests/m3_interactive_2d.rs runs
  a real synthesized guest (scePadReadState -> CPU framebuffer by input -> flip)
  and asserts output changes with input + flip advanced + last_image reflects it.
  Pad/audio HLE already tested. This does NOT satisfy Raeen's acceptance-gate:
  NEXT GATE remains M3, using a toolchain-built interactive 2D homebrew through
  the Shell and a real VideoOut presentation path. M4 cannot be claimed first.
  CAVEAT: this commit bundled concurrent stage-D GPU texture cache whose FPS is
  UNVERIFIED (Minecraft after-run flips 43->22, build 969->59us) — re-measure via
  RAEEN_NO_TEX_CACHE A/B and revert if it regresses.

## 2026-07-21 driver — EUD snapshot base fix + raw-EUD image-descriptor capture
- (Landed inside commit 5802864, which a CONCURRENT session pushed while this
  session's re-measure ran; that commit bundles THREE mechanisms and its
  message describes only the third: (a) this session's strategy-2 EUD
  snapshot-base fix, (b) this session's raw-EUD T#/S# capture pass,
  (c) the other session's program-order EUD-alias walk.)
- (a) read_extended_user_data strategy 2 snapshotted the EUD at
  base + FIRST-LOAD-OFFSET instead of the base-pair value: ASTRO composite
  CS 0x500665c00's first scan-order load is +0x60, so every sharp peek read
  24 dwords high — the sampled T# declared at EUD dword 0 (live-traced
  dword3=0x91800924, type nibble 9=2D) peeked garbage, mis-classed, and its
  image_sample_lz refused as dynamic-image-descriptor. Also widened the
  snapshot to EXTENDED_MAPPING_DWORDS (64) when readable: shaders load
  descriptors past the declared eud_size_dw (0x500543b00 storage T# at
  virtual s68 = dword 36, size says 28). Red-green:
  eud_strategy2_snapshots_at_base_pair_not_first_load_target.
- (b) shader_capture_eud_image_descriptors (analysis.rs, CS-wired): T#/S#s
  delivered RAW through the EUD (no usage-table slot) — the shader loads
  them itself with covered s_loads and feeds the registers to MIMGs — are
  now captured from the EUD snapshot at the load's offset (8-dw T# / 4-dw
  S#, start = SGPRS_MAX + dword), then the guard's alias rule accepts them.
  SharpEmu parity: Gen5ShaderScalarEvaluator.cs:599-668 (register-file copy
  at the MIMG; ours reads the same bytes at the EUD offset). Degrades: zero
  or buffer-typed (nibble 0) content, out-of-window, parse failure → no
  capture, named refusal stands. Tests:
  raw_eud_image_descriptor_captured_from_covered_load (+ idempotence),
  raw_eud_capture_declines_zero_and_buffer_typed_content.
- IMPORTANT sequencing lesson: (a) ALONE regressed the run — correct
  refusals replaced garbage-descriptor writebacks (596→26; ALL baseline
  storage-image writebacks were nonzero=false i.e. zeros), the WAIT_REG_MEM
  labels those dispatches wrote stopped arriving, ACB queues parked
  (0→14 suspends), presents collapsed 256→8. (b) restored the producers
  with CORRECT descriptors. Measured 300s release runs (ASTRO.BOT):
  refusals 497 → 0; writebacks 596 → 2019 (incl. nonzero image writebacks
  0 → 225 — first real compute image content ever); suspends 0; presents
  back to 256 (13 frame dumps). Frames: earlier = the 4-flat-colour
  composite, late = banded blue gradient (10 colours, 0% exact clear) —
  presentation class unchanged, NO scene claim (§5 censused).
- kyty-graphics 366 tests, raeen-gpu 144, workspace clippy -D warnings,
  fmt clean (verified again on HEAD after the concurrent commit).
- NEXT GATE (only remaining translate class, 2 shaders): "mixed sampled
  texture dims in one shader" — 0x500564500 (2d+3d) + one (3d+2darray).
  Fix = per-dim sampled arrays (%textures2D_S stays for Dim 2D; add
  suffixed arrays + per-dim index assignment in prepare_stage_binding +
  per-instruction dim resolution replacing whole-shader
  sampled_texture_dim in the sample bodies).

## 2026-07-21 (session: Shell-log diagnosis + AGC EOP signaling fix)

- 10-agent verified diagnosis of the fresh ASTRO.BOT Shell log (5 tracks, all
  adversarially re-verified; full reports in the session's w6s0ek438 output;
  synthesis in memory astro-bot-boot-state 2026-07-21h).
- agc-eop-signaling: complete (uncommitted; 345 hle + 157 gpu tests, clippy
  clean). RELEASE_MEM decode: sel=3 → AgcTimestampWrite (was: wrote the
  packet's ZERO immediate — fences never advanced); interrupt bits 31:24 →
  AgcEopInterrupt. Submit: monotonic non-zero timestamp writes
  (next_gpu_timestamp), EOP interrupts trigger kevents with
  filter=EVFILT_GRAPHICS_CORE(-14); sceAgcDriverAddEqEvent registers -14.
  Tests: release_mem_timestamp_selection_reports_address_not_zero_immediate,
  release_mem_interrupt_forms_report_eop_interrupts,
  submit_signals_timestamp_fences_and_eop_interrupts.
- ACCEPTANCE: the scePthreadMutexLock >3s Resident-Load deadlock (waiter=main,
  owner=render thread 21) is GONE — 0 warns in 3 live release runs (was 100%).
- NEW boot-stopper unlocked by the fix: host segfault (rc=139) ~1s after first
  submissions, GPU worker inside the invalid-work dispatch (structurally
  invalid SPIR-V — no structurizer — + 304>256 push constants + R8_UINT/FLOAT
  descriptor). Reproduces with validation on AND off; no guest-thread activity
  in the window → not an event-wakeup regression.
- NEXT (in order, all code-mapped in memory 2026-07-21h): (1) translate-time
  SPIR-V validity gate (invalid module → named skip, never reaches the
  driver) + dispatch-loop relooper (OpLoopMerge+OpSwitch over guest blocks);
  (2) push-constant SSBO spill (interim: named-skip guard when >device max);
  (3) numeric-class-aware OpTypeImage (uint/sint per T# numFormat).
- emulator-reviewer pass: no critical findings; applied its two important
  items — agc_gpu_timestamp moved to OrbisKernel (per-session, relaunch keeps
  real clock deltas) + negative test pinning EOP broadcast filter isolation
  (user/-13 VideoOut events on the same queue stay untriggered). Honesty note:
  deadlock-gone was verified on the CLI path (3 runs); Shell-path confirmation
  still owed once the SPIR-V-validity segfault is gated (process dies ~1s
  after submissions on both paths until then).
- structurizer (dispatch-loop relooper): complete in kyty-graphics (uncommitted).
  All guest branches emit through one OpLoopMerge loop + OpSwitch on %reloop_bb;
  every basic block is a case; zero-label shaders keep the legacy linear path.
  5 new tests incl. real-spirv-val acceptance (backward loop, nested loops,
  forward skip/branch, fast-path guard); 382 kyty-graphics green. Live: ASTRO's
  4 previously-invalid scene CS (203k/139k/22k/20k words) now translate VALID
  (gate refusals 8 -> 0).
- push-constant overflow guard: named skip when offset+size > device
  maxPushConstantsSize (captured on VulkanDevice at creation) in compute +
  offscreen pipeline paths. Live: fired once, no more 304>256 UB.
- REMAINING native crasher: R8_UINT-vs-FLOAT sampled-image mismatch
  (VUID-07753 UB) — run dies ~29s in with the other walls cleared.
  Numeric-class-aware OpTypeImage fix in flight (gpu-pipeline agent, SharpEmu
  Gen5SpirvTranslator parity, bitcast-not-convert).
- sampled numeric-class fix LANDED (uncommitted): `SampledClass`
  (Float/Uint/Sint from the T# unified format, SharpEmu Gfx10UnifiedFormat
  numFormat 4/5 rows) joins `SampledDim` in a (Dim, class)-keyed sampled-array
  split — spirv.rs `sampled_key_layout` emits per-class `OpTypeImage
  %float/%uint/%int`, all MIMG sample/gather/fetch bodies retype the texel to
  the class vec4 and `OpBitcast` raw bits into the float register model
  (never ConvertUToF); analysis reserves one binding per key;
  draw_translate groups/reindexes per key (12 ordinals) with a 0..512
  unified-format sweep pinning shader class == view VkFormat class. Tests:
  kyty-graphics 385/385 (3 new: r8_uint raw-bits, float pin, mixed
  float+uint arrays — spirv-val'd), raeen-gpu 163/163 (2 new); clippy lib
  clean both. NOT yet verified against the live title (release run owned by
  another session).
- in-app log console SHIPPED (shell/console.rs + ConsoleBuffer/ConsoleLayer in
  raeen-core logging.rs): F10 toggles a floating console over any screen —
  level filter, search, autoscroll, copy, clear, colored single-line truncated
  rows (show_rows virtualization needs uniform height), 5k-line core ring
  (RAEEN_CONSOLE_LINES). Verified by screenshot over Home AND in-game.
  raeen.exe is now GUI-subsystem (no terminal window from Explorer);
  AttachConsole(ATTACH_PARENT_PROCESS) first thing in main keeps CLI output
  working from a terminal.
- COMBINED GPU ACCEPTANCE GREEN (log rotated to raeen.log.1 during analysis —
  beware the rotation-on-init trap when a second Shell launches mid-analysis):
  240s ASTRO run rc=0, 0 segfaults, 0 deadlocks, 0 gate refusals (relooper
  validates all 4 scene CS), 0 FLOAT-mismatch VUIDs (numeric-class fix), 1
  named push-constant skip, 27 shaders translated, 73 flips, full render loop.
  Session chain COMPLETE: EOP signaling -> spirv-val gate -> relooper ->
  push-constant guard -> descriptor class fix. Next frontier: 548 named draw
  skips (assorted classes, honest skips not crashes) + push-constant SSBO
  spill + storage-image numeric classes (unmeasured).
- UI batch (all screenshot-verified): PS5-authentic Home — real param.json
  titles/ids/versions via raeen-loader (en-US titleName preferred; ASCII
  renderability gate), context line "PPSA17221 · v… · Ready to play", dead
  Store/Game Library tiles + pills removed (pill row = My games/Media/
  Settings, nav indices + tests updated); "blurry UI" root-caused as egui's
  bundled LIGHT-weight font at 1:1 DPI (measured 96-DPI 1080x1920 portrait
  panel, no DWM scaling) — theme::install_fonts now defaults to system
  Segoe UI + Consolas (theme fonts still outrank; non-Windows keeps
  built-ins), console rows 14px. Console window opaque dark chrome (#0D1017).
- UI iteration round 2 (critique workflow ps5-ui-critique ranked 8 items; 6
  landed, all screenshot-verified): band_shift centers pills/rail/context on
  non-16:9 windows (0.38 of surplus height; 1080p landscape unchanged); PS5
  white focus ring (2.5px focus-color hugging expand(4), accent halo 0.06,
  tile_focus_scale 1.42, corner_radius 16, tile_size 224 — theme.toml synced);
  hero cover-fit UVs via generalized cover_uv(texture, aspect) + scrim
  0.92/0.55/0.5 -> 0.62/0.30/0.35 (key art now visible, no portrait squash);
  topbar rework (fake Player One/X5/1,204/center glyphs/wifi DELETED; PS5
  right cluster = search + gear + avatar with real host username initial +
  clock); bottom bar honesty (dead Search hint + chat/capture glyphs removed,
  footer whisper); coverless-tile art (two-initial monogram 0.85 + ghost echo
  + in-tile title + analogous two-hue gradient). REMAINING from critique:
  #6 session ledger (last played/time played/crashed-last-session line), #7
  control-center live data + switcher, topbar interactivity/text-tabs.
  TRAP: release relink fails os error 5 if a screenshot-probe raeen.exe was
  left alive — Stop-Process before cargo build.
- UI round 3 (screenshot-verified): SESSION LEDGER (shell/ledger.rs — per-title
  JSON beside per_game store; launch stamps last_played, exit accumulates
  total_play_secs + last_faulted via faulted_seen polling) → context line now
  "PPSA17221 · v01.008.000 · 2h 14m played · Crashed last session — check
  Console (F10)" (orange status) — honest crash UX, no fiction. Topbar gear
  CLICKABLE (HomeResponse{clicked_tile, gear_clicked} → enter Settings);
  search glyph removed until a search surface exists. CONTROL CENTER trimmed
  to real cards only (Home/Switcher/Sound/Accessories/Power — fictional
  Notifications/GameBase/Music/Mic/Profile/Network deleted, Fields variant
  gone); Sound + Accessories get live CcLive values (config volume/muted,
  gilrs pad name or "No controller connected"). Remaining from critique:
  hero ambient zoom, bottom-bar auto-fade, switcher option drilling, search
  overlay, logo-lockup typography (segoeuib bold family).
- ASTRO.BOT RDNA2 MUBUF boundary + scalar-resource shader blockers FIXED
  (2026-07-22; working tree, no commit; kyty-graphics 399/399, raeen-gpu
  169/169 library tests):
  * MUBUF is now kept at its public-ISA fixed 64-bit size when SOFFSET is
    `0xff`; the measured all-ones PS5 form is normalized to zero without
    consuming a SALU-style literal dword. That inferred meaning is scoped to
    next-gen mode; legacy fallback rejects it instead of silently consuming the
    following instruction or inheriting unmeasured PS5 semantics.
  * The remaining non-EUD `SLoadDwordx8` was not PC-relative: exact dump
    `cs_500757800_216.bin` starts `SInstPrefetch; VMovB32; VMovB32;
    SLoadDwordx8 s[16:23], s[12:13], 0`. Root cause was EUD selection order:
    the positional, readable s14:s15 candidate won before shader analysis even
    inspected the explicit readable s12:s13 load base. Live-in scalar-load
    evidence now wins, snapshots the existing guest resource table at s12:s13,
    and feeds the existing `bind.extended` runtime path; bases written earlier
    in-shader cannot beat the positional fallback, and no descriptor bytes are
    invented.
  * Pixel shaders now run the same PC-relative embedded-constant capture already
    used by VS/CS, so `s_getpc_b64` + `s_load_dwordx8` tables do not fall through
    to the EUD-only path and refuse a valid non-EUD base.
  * Remaining non-EUD scalar-load refusals now report PC, source register, EUD
    base, offset, and the nearby producer chain for the next live variant.
  * TDD regressions cover fixed-width MUBUF in both generation attempts, the
    two-readable-pointer precedence case, and an end-to-end PS shader-cache
    translation. The exact CS dump replays next-gen as 27 instructions and
    recovers three scalar loads, all through s12. Scoped clippy `-D warnings`,
    touched-package rustfmt, and `git diff --check` are green. Full raeen-gpu
    integration testing is blocked by an unrelated host Vulkan debug-utils
    entry-point abort in `compute_gds`; its deterministic 169-test library suite
    is green. Live ASTRO frame verification is still required before a visual-
    correctness claim.
- ASTRO.BOT branch-boundary + type-8 RGBA32F UAV blockers FIXED (2026-07-22;
  working tree, no commit; kyty-graphics 404/404 unit + 4 integration,
  raeen-gpu 171/171 library tests):
  * The MUBUF at `0x1148` was already the correct fixed 8 bytes. Its target at
    `0x1150` is RDNA2 `s_waitcnt_vscnt` (SOPK opcode `0x17`); the parser
    consumed it without recording an instruction, so the relooper mistook the
    next emitted PC (`0x1154`) for the next boundary. It now emits a named
    `SWaitcnt` boundary and a conservative device-scope SPIR-V memory barrier.
    GFX10 `s_version` is separately decoded at SOPK opcode `1` as a metadata
    no-op.
  * Read-write T# type 8 is accepted only for its valid height-one shape and is
    represented as a height-one 2D storage image, with its true 1D coordinate
    lowered to `(x, 0)` rather than consuming an unrelated Y VGPR. Disabled
    `image_store` channels are zero-filled, including alpha.
    Unified format 77 now stays RGBA32F end-to-end: SPIR-V `Rgba32f`, Vulkan
    `R32G32B32A32_SFLOAT`, and 16-byte seed/readback sizing.
  * The active `raeen-gpu` shader cache now analyzes the live stage ABI before
    positive lookup and keys translated modules on that binding identity. A
    same-address shader rebound from format 71/Rgba16f to 77/Rgba32f therefore
    cannot return stale SPIR-V or stale writeback metadata. Reduced parser and
    full reloop/SPIR-V/Naga regressions cover the live branch sequence;
    analysis, codegen, active-cache, Vulkan-format, and guest-memory tests cover
    the measured 1x1 type-8/format-77 descriptor. Scoped clippy, rustfmt, and
    diff checks are green. Live ASTRO frame verification remains required.
- PERSISTENT DEPTH TARGETS + LIVE DB REGISTER WIRING (2026-07-23; working tree,
  no commit; raeen-gpu 182/182 library + 4/4 Vulkan depth tests):
  * `DB_DEPTH_CONTROL`, `DB_Z_INFO`, `DB_STENCIL_INFO`, and
    `DB_Z_WRITE_BASE` now produce a real Vulkan `DepthState`. Images,
    allocations, views, and readback buffers persist under
    `(guest base, extent, format)`; repeated draws LOAD prior contents unless
    guest clear flags request CLEAR. Same-base extent/format changes evict
    safely after the immediate depth fence.
  * Radeon 760M validation proves miss=1/hit=1 and a second clear/readback.
    Depth draws still use immediate submit/fence/readback; depth-only draws,
    HTile, and tiled guest-depth import/export remain open. This advances M2/M3
    infrastructure but does not close either acceptance gate.
- FRESH RETAIL PROBES + ASTRO PIXEL MIMG FRONTIER (2026-07-23; working tree,
  no commit):
  * Minecraft release probe (`scratch/minecraft-depth-20260723.out.log`,
    45 seconds) stayed alive with 0 errors, 0 shader analysis/translation
    failures, 0 draw skips, and no return of `s_sub_u32`; cumulative colour
    target hits exceeded 2,100 with 3 misses. The same shader addresses were
    nevertheless translated repeatedly because runtime binding addresses still
    shape the cache identity: module-key normalization is the next measured
    performance slice.
  * Astro release probe (`scratch/astro-depth-20260723.out.log`, 45 seconds)
    reached 6 presents and 146 deferred draws. All three original user-reported
    failures are absent: no bad reloop boundary, no fixed-EUD-base refusal, and
    no type-8 texture refusal.
  * Ported KytyPS5's RDNA2 MIMG opcode meanings for the measured pixel forms:
    ordinary `image_sample` dmask 2 and `image_sample_lz_o` dmask 1/2. The
    offset path preserves packed signed XY offset extraction, explicit LOD 0,
    and per-dmask channel selection. Parser, SPIR-V assembly, and Naga
    regressions are green; full kyty-graphics is 438/438 and clippy
    `-D warnings` is clean. Post-fix probe removes opcode 0x20/dmask2; opcode
    0x37 advanced from the first dmask1 occurrence to the later dmask2 form,
    which is now implemented and awaits the next release A/B.
  * Remaining measured Astro blockers: PS read/write storage buffers;
    `%vsharp_s0` duplicate declarations; Uniform-array stride 4 (must use a
    legal storage layout); undefined `%buffer_store_float1`; and shader-module
    cache churn. No M2/M3 closure claim.
- STRUCTURAL SHADER CACHE + ASTRO GEN5 PS STORAGE (2026-07-23; working tree,
  no commit; kyty-graphics 438/438, raeen-gpu 190/190 library tests; scoped
  clippy clean):
  * The active translated-module cache now follows Kyty's `ShaderGetBindIds`
    rule: the key contains codegen/descriptor ABI structure but excludes
    per-bind guest addresses, resource extents/counts, sampler payloads, and
    direct-SGPR values. Texture dimension/format, embedded constants,
    EUD/global-memory declarations, LDS size, and every stage codegen flag
    remain keyed. Cache hits return the current analysis metadata, so Vulkan
    descriptors never reuse stale guest bases. The cache stays FIFO-bounded at
    256 entries.
  * Minecraft release A/B
    (`scratch/minecraft-cache-structural-20260723.out.log`, 40 s) stayed at
    0 errors / 0 shader failures, reached target hits at present 8 and present
    256, while compile events fell from >2,100 in the prior probe to 81 across
    8 shader addresses (remaining events are distinct structural variants).
  * Fixed three newly exposed Astro module defects: push-table spills use a
    tightly packed `StorageBuffer` instead of invalid std140 `Uniform` stride
    4; V#/T#/S# seeding uses descriptor-slot-qualified SSA ids so overlapping
    SGPR ranges assemble; multi-dword/typed buffer stores include their
    transitive `%buffer_store_float1` helper.
  * Gen5 pixel read-write storage buffers now pass analysis and bind as
    fragment-visible Vulkan `STORAGE_BUFFER`s; the PS4/legacy rejection stays.
    Astro release A/B (`scratch/astro-psrw-20260723.out.log`, 42 s) reached
    6 presents with 55 successful translations, 0 translation failures, and
    0 shader-analysis errors. The prior PS-RW rejection, duplicate vsharp,
    stride-4, and undefined-store failures are all absent.
  * Honest remaining limitation: deferred graphics storage buffers are uploaded
    and visible to fragment shaders, but GPU writes are not copied back into
    guest memory at batch retirement yet. M2/M3 therefore remain open; next
    correctness slice is fragment-SSBO writeback plus non-synthetic visible/
    interactive acceptance evidence.
- MINECRAFT PRESENTING RESTORED + BLACK-FRAME FORENSICS (2026-07-24; working
  tree, no commit; kyty-graphics 439/439, raeen-gpu 198/198 library + 5/5
  serial Vulkan depth/stencil tests; scoped library clippy clean):
  * Root cause was raster state, not a missing POS0 export. Raeen cast AMD
    Gen5 stencil opcodes directly to Vulkan even though the enums diverge after
    Keep/Zero (AMD ReplaceTest/ReplaceOp and wrap codes became unrelated or
    invalid Vulkan operations). The draw path now maps all ten Gen5 operations
    explicitly, carries both test/op reference values, and rejects unsupported
    codes by name. `PA_SU_SC_MODE_CNTL.FACE` remains the reference-consistent
    direct mapping; applying an extra negative-viewport inversion culled the
    Minecraft UI and was removed.
  * Present selection no longer treats the mere existence of a flipped target
    as proof it contains a frame. It prefers visible RGB in the exact scanout,
    temporarily selects a visible intermediate when the final composite is
    pending, and emits rate-limited `PRESENT ROUTING` or `BLACK FRAME` console
    diagnostics with target/shader/draw counters. The exact Minecraft flip
    buffers become `target_visible=true` on the following composites.
  * Indexed UI draws now upload only through the largest referenced vertex
    index. Minecraft's six-index quad therefore uploads four 20-byte records
    (80 bytes) instead of the V# ring's roughly 4 MiB capacity per draw.
  * Production release proof, with no `RAEEN_NO_CULL`/`RAEEN_NO_STENCIL`
    bypasses: `scratch/mc-production-stencil-fix-20260724/frame_000512.ppm`
    is a real 1920x1080 Minecraft loading screen (spinner + partially filled
    progress bar), and subsequent logs show both scanouts visible. The tightened
    `RAEEN_TRACE_DRAWS` probe reports actual `Fetch*` destination VGPRs as well
    as POS0 sources; it rules out the proposed gs-prolog VGPR-collapse theory
    for this shader. This is visible retail-title progress, not proof of
    gameplay, interaction, universal presentation, or an M2/M3/M5 gate.

- MINECRAFT TITLE PANORAMA + AUDIO REACHED; INPUT ACCEPTANCE STILL OPEN
  (2026-07-24; working tree, real release runs):
  * Fixed GFX10 MIMG instruction sizing and addressing: `DIM` comes from
    word0[5:3], `NSA` adds 0–3 DWORDs to the instruction and supplies explicit
    address VGPR bytes, and array/cube image operations carry the correct
    coordinate width. `ImageStore` now supports arrayed v3uint coordinates,
    storage uploads honor `BASE_ARRAY..=LAST_ARRAY`, and each layer detiles and
    retiles independently. Corrected GFX10 `VCMPX` to update EXEC while
    preserving VCC. Exact parser/lowering/storage regressions are green.
  * Found the panorama's silent-write root cause: storage and sampled resources
    both used `s24`; a later sampled-descriptor rewrite changed DWORD0 to
    sampled slot 1, so the one-entry storage array was indexed out of bounds.
    Storage images now use the exact analyzed descriptor's class-local constant
    index. Real traces show all six 1024x1024 cube writes at `0x33680000+`
    becoming non-zero.
  * Vulkan validation then identified a separate scanout-composite device loss:
    the generic CPU render-target fallback replaced a decoded six-face cube's
    24 MiB pixels with a one-face 4 MiB framebuffer snapshot but left
    `layers=6`; `vkCmdCopyBufferToImage` copied 24 MiB from that 4 MiB buffer
    (`VUID-vkCmdCopyBufferToImage-pRegions-00171`). The fallback is now limited
    to plain one-layer 2D textures, and both graphics/compute Vulkan upload
    paths independently size and zero-pad staging data from the declared image.
    Three focused regressions cover the substitution gate and one-face/six-face
    staging mismatch.
  * Measured release evidence:
    `scratch/mc-cube-fix-20260724-023344/frames/frame_001024.ppm` is a real
    1920x1080 Minecraft loading scanout; `frame_002048.ppm` is the complete
    title screen with Minecraft logo, Steve model, UI text/buttons, and the
    rendered animated panorama. The run reached exact scanout
    `0x20040000` with `scanout_hit=true` and no device loss.
  * The isolated runner now initializes cpal from persisted settings. Minecraft
    produced a real non-silent 48 kHz PCM submission (observed peaks 0.00219
    and 0.66420 in separate release runs). AudioOut2 uses the Gen5 structure
    layout/pacing; ACM and media out-parameters are initialized.
  * Raw-HID DualSense discovery and a valid 64-byte report were measured.
    `libScePad` writes the full 120-byte state and the child publishes native
    input to its process-local kernel. However, a targeted
    `raeen_hle::libsce_pad=debug` run reached present 1024 without Minecraft
    calling `scePadOpen` or `scePadReadState`. All five imported Pad NIDs are
    HLE-resolved, so the remaining acceptance item is to capture a real guest
    Pad call and a button-driven transition from “Get started” into gameplay.
  * HONEST STATUS: Minecraft now renders its title assets and produces host
    audio, but it is not yet “fully playable.” The measured title-screen run
    took roughly 199 seconds to present index 2048 (about 10 presents/s
    averaged across boot), far short of the 120 FPS north star. M4/M5 remain
    open; no compatibility status is raised without a measured interactive
    report.

- MINECRAFT USER LOGIN + LIVE DUALSENSE CONSUMPTION REACHED
  (2026-07-24; release run `scratch/mc-login-input-20260724-030815`):
  * Root cause: Raeen's `sceUserServiceGetEvent` always returned
    `SCE_USER_SERVICE_ERROR_NO_EVENT`. Current SharpEmu and KytyPS5 deliver one
    Login event first; shadPS4 also models login/logout through an event queue.
    Minecraft waits for that transition before entering its Pad path.
  * UserService now returns the retail-style primary id `0x10000000`, stores
    login-delivery state in the process-owned `OrbisKernel`, writes the
    eight-byte `{Login, userId}` ABI exactly once, and returns `NO_EVENT`
    afterward. Failed guest-memory writes restore the claim; new processes get
    fresh state. Five focused tests cover payload, one-shot, retry, and
    process isolation.
  * Measured end-to-end route: raw-HID DualSense connected, its first 64-byte
    report parsed, Minecraft consumed the UserService login event, opened Pad
    handle 1, and its first guest controller read consumed live host state
    (`buttons=0`, sticks `[129,133]` / `[132,132]`). This closes title
    consumption of physical input, not merely NID resolution.
  * Rendering stayed intact: present 1024 and 2048 both hit exact scanout
    `0x20040000`; `frame_002048.png` is the complete title panorama/UI with no
    device loss. The remaining Minecraft acceptance item is a captured
    physical-button-driven menu transition and gameplay/save loop. Do not call
    the title fully playable or raise M4/M5 yet.

- MINECRAFT PLAY TRANSITION NARROWED TO COHTML/V8; GUEST VMM LEAK FIXED
  (2026-07-24; measured release run
  `scratch/mc-vmm-hint-fix-20260724-062701`):
  * Added deterministic, child-process input replay so title transitions can be
    reproduced without synthesizing host UI input. The same path publishes
    normal `raeen-input` states into the process-local kernel and is covered by
    parser/state-timing tests.
  * Added real `/dev/random` and `/dev/urandom` character devices backed by the
    host entropy provider, decoded-instruction CPUID trapping with a stable PS5
    Zen 2 profile, and alias-safe pthread mutex/rwlock state. These are general
    runtime corrections; none is a title-specific bypass.
  * `sceKernelReserveVirtualRange` now validates its ABI, reads the in/out
    address hint, honors fixed placement, and releases an exact whole OS
    reservation on `munmap`. Before the fix, V8 repeatedly reserved and leaked
    rejected 4/8 GiB regions above 1 TiB. The measured rerun acquired 8 GiB at
    `0x1000000000`, released it, then reacquired 4 GiB at the same hint.
  * Pressing Cross now reproducibly enters Cohtml initialization. It still
    aborts before gameplay in V8 9.4.146.24 snapshot deserialization with
    `unreachable code`. Exact upstream source and the decrypted user-supplied
    module show the unconsumed byte is legal `kOffHeapTarget` (`0x17`), which
    must be consumed during a `kCodeBody` relocation walk but instead reaches
    the generic root-field decoder. CPUID, pthread aliasing, TLS layout, and
    cage placement were individually tested and do not remove the abort.
  * HONEST STATUS: title panorama, audio output, live DualSense reads, and the
    Cross-driven transition are measured. Gameplay, save/load, sustained frame
    rate, and compatibility acceptance remain open; M4/M5 are not claimed.
