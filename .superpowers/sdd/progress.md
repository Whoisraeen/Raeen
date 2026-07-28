- SYNCHRONOUS GUEST CALLBACKS — checklist item 7 (2026-07-27; worktree agent,
  commit 6ff4132, branch worktree-agent-a8a0f93f4ac611946; raeen-runtime 77
  lib + 56 execute
  (+6 new, 1 pre-existing ignored bench) + 1 m3 green; raeen-hle 500/500
  (+3 new); raeen-firmware 125/125; fmt clean; clippy --all-targets
  -D warnings green on raeen-runtime + raeen-hle):
  * MECHANISM: `GuestCallScheduler::call_guest(entry, [u64; 6]) ->
    Result<u64, GuestCallError>` (raeen-hle lib.rs, default = Unsupported for
    test doubles) implemented on `ActiveContext` (dispatch.rs). An HLE
    handler calls back INTO guest code mid-call and receives its RAX on the
    current guest thread — the synchronous complement to the existing
    deferred `request()` tail-call.
  * STACK STRATEGY: on the VEH path the handler already runs ON the guest
    stack (Windows dispatches vectored exceptions on the faulting thread's
    stack), so the callback is a plain native `extern "sysv64"` call —
    frames land below the trapped guest frame, alignment kept by the
    compiler, and the FS-rearm staging writes at [rsp-16] stay inside
    arena-backed memory. No second stack region. All recovery machinery
    (armed recovery ctx, TLS/FS re-arm, commit-on-demand, terminating arms)
    stays live, so nested HLE imports from the callback trap normally as
    nested VEH dispatches (a pattern the runtime already relied on).
  * NESTING: bounded only by guest stack space; depth 2 (HLE → callback →
    HLE → callback) pinned by `nested_call_guest_composes_to_depth_two`.
    The interrupted handler's `active_hle` attribution and its deferred
    `pending_guest_call` are saved/restored around the callback so nested
    dispatches can't mis-attribute a fault or steal a deferred request.
  * UNWIND COMPOSITION: callback faults genuinely → run returns
    Err(Faulted) and the requesting handler NEVER resumes (longjmp to the
    recovery ctx; pinned with a resumed-flag static). `request_exit` under
    the callback (`__stack_chk_fail`, d818df9 lineage) → clean thread
    unwind with the fatal code, handler provably never resumes. Terminating
    functions and cooperative process exit compose identically (same
    longjmp seam).
  * DISPATCH PATHS: VEH = full support. Direct leaf gateway = REFUSED
    LOUDLY (GuestCallError::Unsupported + error log): the generated bridge
    re-bases RSP to the fixed private host-stack top on every entry
    (trampoline.rs `mov rsp,[r11+8]`), so a nested direct import called
    from a callback hosted there would clobber the outer gateway's live
    frames. `direct_dispatchable` deliberately lists only never-re-enter
    imports; the refusal is pinned by a test overriding libc::strlen with a
    call_guest probe (returns Unsupported, dispatch itself unharmed). New
    `ActiveContext::direct_gateway_active` save/restore around the
    gateway's hle.call detects the path.
  * CONSUMER RETIRED: `qsort` (libc.rs) — the ledger 2026-07-25 "needs a
    synchronous guest callback" skip. Real in-place heapsort over guest
    memory: every comparator call receives GENUINE pointers into the live
    array (max ABI fidelity, no scratch allocs), swaps via bounds-checked
    reads/writes, O(n log n), abortable mid-flight with honest logging
    (refusal precedes the first swap; null comparator / overflow extents
    untouched). End-to-end acceptance: guest fixture sorts a 6-elem u64
    array through a REAL guest comparator that counts its own calls, guest
    code verifies ascending order, returns the counter (≥ n-1 asserted).
    Plus 3 hle unit tests (host-comparator double sorts through the trait;
    Unsupported leaves array untouched; degenerate inputs are no-ops).
  * NOT DONE (follow-ups now unblocked): atexit chain (needs a hook in the
    VEH terminating-function arm BEFORE the exit longjmp — the exit HLE
    handler never runs on the VEH path, it is intercepted by
    TERMINATING_FUNCTIONS), module init/fini callbacks, VideoOut/GPU event
    callbacks. Float-returning callbacks unsupported through this interface
    (documented on the trait).

- GTA V ACB PHASE B — COMPUTE-QUEUE EXECUTION (2026-07-27; gpu-pipeline agent
  worktree branch; kyty-graphics 480/480 (475 lib + 5 integration, +3 new),
  raeen-hle 486/486 (+5 new; one wall-clock vblank flake passed on isolated
  rerun), raeen-gpu 269 lib green (+1 new), raeen-kernel 45/45, fmt clean,
  clippy on touched crates: only the pre-existing analysis.rs MSRV lint +
  raeen-gpu phase-1 dead-code warnings — none from this batch):
  * REALITY CHECK vs the Phase A handoff plan: submitted ACBs were ALREADY
    executing — `hle_driver_submit_acb` → `submit_command_buffer` →
    `GpuQueue::AsyncCompute` → a dedicated compute `CommandProcessor` with
    WAIT_MEM32/64 suspend/resume, parked in-order backlogs, and the
    bidirectional produced-label latch (all tested). What was actually
    missing, now ported from KytyPS5 (MIT lineage, attribution updated):
  * (1) ACB DESCRIPTOR INDIRECTION (`submit_acb`, agc.cpp L3928-3946): the
    submitted buffer may be 5 DWORDs `[addr_lo, addr_hi, size, flags=0,
    magic 0x5533ccaa]` pointing at the real stream. Previously that
    descriptor failed PM4 decode → the whole compute submission dropped as
    SCE_ERROR_INVALID_ARGUMENT. Unwrapped for SubmitAcb AND SubmitMultiAcbs.
  * (2) GRAPHICS→COMPUTE ORDERING (`flush_pending_graphics_segment_before_acb`
    + segment tracker, agc.cpp L214-264/L3691-3839): graphics PM4 the title
    builds AFTER its last DCB submit (tracked via `alloc_command_dwords`,
    the `AllocateDW` mirror; state on `OrbisKernel::agc_pending_graphics_
    segment`) is flushed as a DCB before every ACB submit — truncated to the
    last RELEASE_MEM whose label the ACB awaits, trimmed to whole packets.
    Without this an ACB waiting on a not-yet-submitted producer parks until
    the dead-wait net fires (glitch path).
  * (3) COMPUTE PACKET ARMS (kyty-graphics run.rs): IT_DISPATCH_INDIRECT now
    EXECUTES (both KytyPS5 forms: base+offset via the new indirect-dispatch
    base, and absolute-address; missing base/memory = once-warned skip);
    IT_SET_BASE routes select-1 by the header shader-type bit (draw vs
    dispatch args) per CpOpSetBase. DISPATCH_DIRECT/ACQUIRE_MEM/RELEASE_MEM/
    WAIT/COND_EXEC verified already queue-indexed — not re-ported.
  * TESTS: descriptor unwrap → real stream on AsyncCompute (+ raw-stream
    negative), builder-built RELEASE_MEM flushed as DCB *before* the waiting
    ACB (order asserted on a recording GPU), wait-match truncation (unawaited
    producer NOT flushed), non-contiguous allocation ignored, ACB RELEASE_MEM
    resumes a suspended graphics wait (reverse of the existing cross-queue
    test), dispatch-indirect base/absolute/degrade trio.
  * NEXT: checklist 2D — live GTA V re-measure. Look for: the UD2 assert at
    module+0xae36 moving or clearing; "ACB submission carries dispatches" +
    "flushing pending graphics segment before ACB" in the log; whether
    descriptor-form ACBs appear (debug line "ACB submission descriptor
    unwrapped"); remaining unresolved NIDs should be the ~46 libSceAmpr
    (agent C) + _Ctype.

- ACTIONABLE CRASH REPORTS IN THE SHELL — checklist item 12 (2026-07-27;
  shell-ui agent, worktree branch worktree-agent-ad777e259801c210c;
  raeen-gui 195/195 [+10 over the 185 baseline], raeen-kernel 43/43,
  raeen-hle 481/481, fmt clean; clippy clean on the new/touched code — the
  remaining `-D warnings` failures are the on-record kyty-graphics MSRV
  lint (item 13, in progress elsewhere) and two raeen-gpu dead-code/
  type-complexity lints from the GPU-resident-present commit ea6efd0):
  * CORE (`crates/raeen-gui/src/crash_report.rs`, NEW): pure, headless-
    tested report assembly — `CrashReport::render` produces one
    self-contained markdown file `logs/crashes/<title-id>_<UTC>.report.md`
    with title id+version (param.json via `title_meta_for`, folder-name
    fallback), session duration, fault one-liner (stable `- Fault: ` anchor
    the list view parses back), fault site (module+offset+16 RIP bytes via
    the pure `locate_fault`, shared with `main.rs::report_fault_site`),
    last-10 HLE calls per guest thread, unresolved-NID inventory, GPU
    counters, host sysinfo block, paired `.dmp`/log paths. UTC stamps are a
    dependency-free `civil_from_days` (tested against epoch + the
    billennium second). Listing/pairing helpers: `list_reports` (newest
    first), `newest_dump_since` / `newest_report_since`.
  * WRITERS: the runner child (`--run-eboot`) writes the rich report when
    `execute_process` returns Err — the only process still holding the
    kernel rings, the composed image, and the GPU session
    (`main.rs::write_runner_crash_report`). The Shell
    (`launcher.rs::run_isolated_child`) writes a fallback ONLY when a
    minidump landed without a report (hard crash — abort path), so
    launch-stage failures (missing eboot) do not spam junk reports; the
    fault overlay message now names the report path either way
    (`crash_report::ensure_report_for_crashed_runner`).
  * KERNEL: `OrbisKernel::unresolved_nid_inventory()` (NEW, raeen-kernel) —
    the sorted formatted lines; `log_unresolved_nid_inventory` now consumes
    it so log and report can never drift. Inventory format pinned by test.
  * SHELL VIEW (Settings ▸ System): recent reports newest-first (cap 8,
    `MAX_CRASH_REPORT_ROWS`), one allocated row-card each (hit rect ==
    painted rect, no bottom_up anywhere) — Confirm opens the report file;
    trailing action rows "Open Crash Reports Folder" (opener) and "Copy
    Newest Report to Clipboard" (`ctx.copy_text`), toasts on every action
    (TopLeft anchor as configured). Row counts are live:
    `settings_row_counts` gained a crash_report_count param
    (SYSTEM = 2 fixed + N reports + 2 actions, tested); list refreshes on
    every Settings open (`enter_settings`).
  * Also fixed in passing: two pre-existing clippy-1.97 lints in
    `shell/sounds.rs` (doc blockquote, needless range loop);
    `crashdump.rs` now sources its dump dir from
    `crash_report::REPORTS_DIR` so dumps and reports cannot diverge.
  * NOT live-verified from the worktree (parallel-build isolation): the
    Shell drive/screenshot pass is listed for the main session.
- TIER-B OFFLINE-SEMANTICS SERVICE-LIB BATCH (2026-07-27; agent worktree
  branch, raeen-hle 463/463, raeen-firmware 125/125, fmt clean, clippy clean
  on raeen-hle itself — the only `-D warnings` failure is the pre-existing
  kyty-graphics `is_multiple_of` MSRV lint already on record):
  * SCOPE: the measured non-AGC/non-Ampr unresolved surface from
    `artifacts/compat/nid-coverage.json` + the GTA V blocker doc — ~134
    functions across 30 libraries, every one with deliberate documented
    offline semantics per the 2026-07-25 no-blanket-stubs policy. AGC (83) and
    Ampr (46) remain Tier C; `_Ctype` (libc data object) remains out of scope.
  * NpWebApi2 (+17): no user context can exist (CreateUserContext already
    refuses), so request/context-keyed calls return the matching shadPS4-
    cross-checked `*_NOT_FOUND` codes and push-channel creation refuses
    NOT_SIGNED_IN. Nothing fabricates a PSN response.
  * NpManager (+10) + NpAuth (+3): real request registry re-derived from
    shadPS4's model — create returns a live id, checks complete it with
    SIGNED_OUT (async returns OK, `sceNpPollAsync` then reports the offline
    result immediately), abort/delete/not-found/INVALID_ID rules pinned by
    tests. Premium events register and stay silent.
  * libSceVoice (NEW module, 15): init OK, ports are real tracked handles,
    output reads report 0 bytes (silence), writes accepted and dropped,
    `GetPortInfo` writes the shadPS4-layout struct with nonzero frame_size.
    Voice error codes are undocumented — generic kernel codes used and marked.
  * AvPlayer (+10 in libsce_media.rs): `sceAvPlayerInitEx` now hands out a
    REAL handle with honest lifecycle — AddSource/Start/Stop/Pause/Resume/
    JumpToTime accepted, zero streams, IsActive stays false, GetVideoDataEx
    no-frame => immediate EOS, so video waits terminate promptly. Decode is
    explicitly future work (`register_incomplete`). Legacy `sceAvPlayerInit`
    keeps its measured null-handle behavior.
  * Dialogs: ImeDialog (NEW, 6 — completes instantly as USER_CANCELED, its
    own None/Running/Finished enum, panel size nominal), WebBrowserDialog
    (NEW, 6) and NpCommerceDialog (NEW, 7 — completes instantly as canceled,
    never grants a purchase), PlayerInvitation/Selection dialog leftovers,
    SystemService player dialog (param-init OK, Launch refuses PARAMETER).
  * libsce_online_misc.rs (NEW, grouped): Remoteplay (DISCONNECT),
    SharePlay/GameLiveStreaming handshakes, ContentDelete, ContentSearch
    (empty library -> not-found; layouts undocumented), NpUtility bandwidth
    test (SIGNED_OUT), NpGameIntent (no pending intent), VideoRecordingP
    (disabled recorder refuses up front, stop/close OK, status 0; the
    measured anonymous NID 0x8904ba0d4b4bc9b1 is register_nid-bound —
    audited explicit-NID count 11 -> 12).
  * Real implementations where cheap: `sceHttpUriEscape` (bounded RFC 3986
    percent-encoder + size-query mode), `sceRtcGetTime_t` (datetime -> Unix
    seconds via the existing tick math), `sceSaveDataSaveIconByPath` aliased
    onto the real icon writer.
  * Tail sweep: NpEntitlementAccess consume flow (+5, refuses SIGNED_OUT,
    never fabricates a transaction id), Share Terminate/Permit/Prohibit (+3),
    NetCtl GetResult/UnregisterCallback (+2) + `sceNetGetSockInfo`
    (acknowledge-only, register_incomplete), NpUniversalDataSystem (+2),
    NpTrophy2 GetGameInfo (NOT_FOUND, same rationale as GetTrophyInfo),
    ContentExport FromFile[WithThumbnail] (+2), Coredump Unregister/
    WriteUserData (+2), AudioOut SetMixLevelPadSpk (+1).
  * HONEST STATUS: this closes the measured *resolution* gap for the
    online/social/service tier — it is import evidence, not behavior proof.
    GTA V's blocker remains the 83-NID AGC/ACB surface (Tier C, M2/M5 work).

- GTA V LOAD + AUDIOOUT2 BLOCKER SLICE (2026-07-27; working tree at
  `b9c2daf+dirty`, no commit; pre-fix live trace `logs/raeen.log.1`):
  * FIX: `read`/`pread` no longer silently clamp a valid guest request to
    16 MiB. Transfers up to the existing 256 MiB HLE safety ceiling are
    range-validated once and streamed through 1 MiB staging chunks. Live GTA
    proof: `/app0/rpf.cache` now returns its complete `0x1844404` bytes
    instead of `0x1000000`; the false missing packaged-shader assertion
    disappeared. A `16 MiB + 0x23456` regression covers sequential and
    positional reads.
  * NEW MEASURED WALL: GTA advanced to AudioOut2 init, then asserted at module
    `+0x26320c8` after 4,096 recorded HLE calls. Its context allocation was
    only `0x10000`, then `sceAudioOut2GetSpeakerArrayMemorySize`
    (`0x1b560e2832585f66`) returned fail-soft zero immediately before the
    assertion. `sceRemoteplayInitialize` was the only other called unresolved
    NID. No flips were observed in this short init run.
  * FIX: KytyPS5's live Gen5 sizing is now used:
    `0x10000 + queue_depth * 0x590` (`0x11640` at reset defaults). GTA's full
    four-import speaker-array family is registered: bounded memory sizing,
    32-slot create/destroy, and zero-initialized ambisonics coefficients with
    the reference W-channel normalization. Attribution updated in
    `docs/reference-port-ledger.md` and `THIRD_PARTY_NOTICES.md`; no C++ copied.
  * GTA-ONLY AGC COVERAGE: sanitized static coverage over the installed GTA V
    build began at 672 HLE / 468 LLE / 289 unresolved imports, with only
    134/241 render-path imports resolved. The KytyPS5-cross-checked packet-size
    family (+19), direct Cx/Sh/Uc register writers (+3), and DCB/ACB
    conditional-execution writers (+2) move that to 696 HLE / 468 LLE / 265
    unresolved and 158/241 render-path imports resolved (83 remain, all
    `libSceAgc`). This is import/implementation evidence, not a rendered-frame
    claim.
  * FIX: direct register writers unpack the by-value
    `{u32 offset, u32 value}` SysV argument and emit standard 3-DWORD
    SET_CONTEXT/SET_SH/SET_UCONFIG packets already consumed by
    `kyty-graphics`. Conditional execution emits KytyPS5's exact 5-DWORD
    packet; the command processor now reads the 32-bit guest label, skips the
    guarded DWORD range only when it is zero, and fails open with a
    rate-limited diagnostic when memory is unavailable.
  * VERIFY (AGC slice): `kyty-graphics` 465/465 and `raeen-hle` 443/443.
    Targeted registration/emission/skip tests were observed red before the
    implementations and green afterward. `cargo clippy` could not start
    because Windows Application Control blocked `cargo-clippy` (OS error
    4551), not because of a reported Rust lint.
  * VERIFY: `raeen-hle` 439/439; focused AudioOut2 14/14; touched-crate clippy
    (`--no-deps -D warnings`) green; isolated release
    `target/codex-save-fix/release/raeen.exe` built successfully. Full
    dependency clippy remains red only on a pre-existing dirty-tree
    `kyty-graphics/src/shader/analysis.rs` lint
    (`user_sgpr.count.max(0)`), intentionally not overwritten here.
  * HONESTY / BLOCKER: the post-AudioOut2 GTA rerun is NOT measured. Windows
    Enterprise Application Control began rejecting the freshly rebuilt
    unsigned executable (Code Integrity events 3033/3077, policy
    `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`), including an approved elevated
    launch. Do not claim GTA passed `+0x26320c8` until the user runs/signs the
    release and captures a new report. GTA—not Minecraft—is the active title.

- 4.4 BLANK POST-MENU PAGE — COHTML EXPORT TRAP (2026-07-26; working tree at
  `b9c2daf+dirty`, no commit; runs `run-1785112776063` / `run-1785113013865`):
  * METHOD: `RAEEN_TRAP_MODULE_EXPORTS=cohtml` armed 264 one-shot int3 traps
    on libcohtml.Prospero.prx exports (54 data / 34 duplicate skipped);
    scripted Cross at t=45 s. Run 1 silently had NO input thread — compat
    `--run-eboot` only spawns the scripted-input thread when
    `RAEEN_RUNNER_CHILD=1` is in the environment (main.rs:953); run 2 added
    it and the press is confirmed applied (`buttons=0x00004000` at
    elapsed_ms=45002, released 45303).
  * RESULT: exactly 16 export hits, ALL between t=2.6 s and t=5.3 s (module
    init), ZERO for the remaining 145 s — before AND after the confirmed
    press. All 16 NIDs are anonymous in the 185k dictionary (Gameface
    proprietary). Conclusion: eboot→cohtml EXPORT crossings are init-only;
    menu rendering and press navigation do not cross that boundary (Gameface
    is driven via vtables/created objects, not exports). The trap boundary
    was the wrong level for the navigation question; the signal is negative
    but clean. Next probe: `libRenoirCore.PS5.prx` (64 exports, Minecraft's
    Ore-UI middleware) with the same recipe, then the cohtml View URL dump.
  * RECIPE (proven): `RAEEN_RUNNER_CHILD=1 RAEEN_TRAP_MODULE_EXPORTS=<sub>
    RAEEN_INPUT_SCRIPT='0:neutral;45000:cross;45300:neutral' cargo xtask
    compat run --registry artifacts/compat/registry-mc-only.json --timeout
    150`. Analysis: `scratch/analyze_export_trap.py <raw stdout log>`.

- 4.4 RENOIR TRAP — SECOND CLEAN NEGATIVE + SYNTHESIS (2026-07-26; run
  `run-1785113208206`, same proven recipe with `RAEEN_TRAP_MODULE_EXPORTS=renoir`):
  * RESULT: 9 export hits, ALL at t=2.85–3.18 s (init), zero for the
    remaining 147 s across a confirmed press. Both UI libraries are therefore
    EXPORT-silent after init: Minecraft drives Gameface/Renoir entirely
    through C++ vtable interfaces on objects created during startup. Export
    traps are structurally blind to the navigation path — this probe class
    is exhausted for 4.4.
  * SYNTHESIZED 4.4 STATE (all measured, this + prior sessions): input
    reaches the guest correctly; the page navigates and paints its
    background but no content; NO new asset VFS reads follow the press;
    neither UI library receives any post-init export call. Prior RE already
    built V8-snapshot diagnostics into the trap path (`export_trap.rs`
    ADDR-TRAP Cohtml parser/code/relocation handlers, and the pinned
    `reference/v8-9.4` tree) — consistent with the page's JS bundle failing
    to load or execute inside the View while the chrome paints. The two
    leads that remain, in order: (1) dump the cohtml View's loaded URL +
    ready state from guest memory (needs the View object address — Ghidra on
    the eboot's cohtml-vtable call sites, Phase 4.1's toolchain); (2) trace
    the V8 script load for the post-press route with the existing ADDR-TRAP
    deserializer probes. Both are explicitly multi-session RE; do not
    substitute further HLE guesses.

- 4.4 V8 DESERIALIZER — MECHANISM NAILED FROM UPSTREAM SOURCE (2026-07-26;
  source walk of `reference/v8-9.4` at HEAD `6301cc0db1cf` = exactly
  9.4.146.24, the version inside libcohtml.Prospero.prx):
  * ABORT SITE: `src/snapshot/deserializer.cc:953-956` — `case
    kInternalReference: case kOffHeapTarget: UNREACHABLE()` inside
    `ReadSingleBytecodeData`. These bytes are legal ONLY during RelocInfo
    iteration; the abort is the generic root decoder meeting a reloc byte.
  * WHY THE WALK CONSUMES NOTHING: the relocation walk
    (`deserializer.cc:1063-1068`) iterates the Code object's relocation
    ByteArray, whose length is whatever the `kRelocationInfoOffset` header
    slot resolved to (`deserializer.cc:1043-1044`). The prior probe
    (`scratch/mc-reloc-header-20260724-072245`) measured
    `relocation_length_smi = 0` with nonzero data beyond it — the slot
    resolved to the WRONG OBJECT (or a zeroed region), not to a corrupted
    right object. There is NO flag/version/CPU gate that can skip the walk;
    zero entries means a wrong object reference.
  * WRONG-OBJECT CANDIDATES (from the source): (a) a `kBackref` index that
    lands on the wrong `back_refs_` entry (release builds have NO bounds
    check, `deserializer.cc:548-558`); (b) an unresolved
    `kRegisterPendingForwardRef` leaving `Smi::uninitialized_deserialization_
    value()` in the slot (`deserializer.cc:973-1002`); (c) a root-constants
    reference resolving to `empty_byte_array`. All three are deterministic
    functions of the stream + build config — so the divergence is most
    likely IN THE STREAM BYTES the guest reads or in a build-config
    constant (pointer compression / SnapshotSpace encoding / field offsets)
    that differs between the snapshot producer and the running build while
    the version-STRING check still passes (`snapshot.cc:626-643` checks
    only the version string + checksum; there is NO CPU-feature bitmask in
    this version).
  * KEY QUESTION OPEN: where does the second (post-press) deserialization's
    stream come from — the module's embedded blob (then bytes are
    deterministic and the failure is config/state) or a runtime-read file
    (then VFS content is suspect)? The first (menu) isolate deserializes
    FINE from the same module, which points at second-isolate state or a
    second, different stream.
  * NEXT PROBE RUNNING: `RAEEN_TRAP_ADDR=103d7a04` (the exact abort site,
    image-relative) + scripted Cross. Hit => the broken deserializer path
    is still reached post-press and the blank page is downstream of it;
    no hit => the current blank page never reaches V8 deserialization and
    the failure moved earlier.

- 4.4 V8 ABORT SITE NEVER REACHED NOW (2026-07-26; run `run-1785127248677`):
  `RAEEN_TRAP_ADDR=103d7a04` (the 2026-07-24 `deserializer.cc:956` abort
  site) armed 1 trap; ZERO hits in 150 s across a confirmed Cross press
  (applied at elapsed_ms=45003). The deserializer failure the 07-24 RE
  cornered is no longer on the post-press path — the blank page either
  never reaches V8 deserialization (failure moved earlier) or
  deserialization now succeeds and the page is blank downstream (dead
  JS/content). Discriminator running: traps at three return addresses
  inside the 07-24 deserialization frame chain (`103d6a40`, `103ea2b6`,
  `103e566a`) — a boot-time hit is the positive control (menu isolate),
  a post-press hit means the second deserialization runs.

- 4.4 PATH TRAPS SILENT + FRAME A/B RUNNING (2026-07-26; runs
  `run-1785127248677`, `run-1785127460504`):
  * The abort site (`103d7a04`) AND three frame-chain return addresses
    (`103d6a40`, `103ea2b6`, `103e566a`) from the 07-24 deserialization
    backtrace all armed cleanly and fired ZERO times in 150 s — including
    at BOOT, where the menu's V8 isolate must initialize. So those
    addresses belong to a cached-script deserialization path that neither
    the boot menu nor the current post-press flow executes; the 07-24
    failure shape is gone entirely, and nothing V8-flavoured runs
    post-press now. Behaviour CHANGE between 07-24 and today is
    unattributed (Phase 0/1 HLE breadth, link cache, pthread cond FIFO —
    no bisect yet; do not claim a fix, the page is still blank).
  * RUNNING: `RAEEN_DUMP_FRAMES` + scripted Cross at 45 s to answer the
    more basic question visually — does the page even CHANGE at the press
    in the current build (navigation happens, renders empty) or is it
    pixel-identical pre/post press (navigation never starts)?

- 4.4 RESOLVED — MINECRAFT MAIN MENU RENDERS POST-PRESS (2026-07-27; run
  `run-1785129*`, frame evidence `scratch/phase4-frames-20260727/`):
  * VISUAL PROOF: `RAEEN_DUMP_FRAMES` + scripted Cross at elapsed 45.003 s
    (applied 04:48:38): frame_002048 (04:48:08, pre-press) is the black
    loading phase; frame_004096 (04:48:52, 14 s post-press) is the
    COMPLETE main menu — Play/Settings/Store/Dressing Room buttons,
    "Sign in to PlayStation Network", animated panorama, Steve, "X Select"
    prompt, version v1.21.43; frame_008192 (04:50:00) confirms it is
    stable with the panorama still animating.
  * The blank post-menu page documented 2026-07-24 ("navigates and
    renders empty") is GONE in the current tree. Attribution: the fix
    arrived with the user's in-progress dirty-tree graphics work (the
    `analysis.rs` EUD zero-extended tail-pointer recovery / draw_translate
    changes — the same work that took Minecraft 8192 -> 12192 flips in
    the re-verify sweep). The 07-24 V8 deserializer abort is likewise no
    longer reached (probes above) — consistent with the page's JS now
    executing far enough to build the menu DOM.
  * CONSEQUENCE: scripted in-world measurement is UNBLOCKED. Next: extend
    the input script past the main menu (Play -> world creation) with
    frame-verified timing, then the Phase 2 "Minecraft >= 60 FPS
    in-world" gate is measurable for the first time.

- MC PLAY SCREEN + WORLD-LIST RENDER (2026-07-27; run `run-1785128*`,
  frames `scratch/phase4-world-20260727/`):
  * Scripted Cross at 45 s (Get started -> main menu, renders fully) and
    75 s opened the PLAY screen: Worlds/Friends/Servers tabs, Create New,
    Realms offline notice (expected without PSN), and a live world list —
    TWO "My World" survival saves (07/27/26 0.44 MB, 07/25/26 0.01 MB;
    savedata read path works, and a world was already created on 07/27 by
    an earlier session). A third Cross at 95 s left the screen unchanged
    (frame_008192) — no default-focus activation on this screen.
  * IN-PROGRESS EDIT COLLISION (resolved): the user's in-flight
    `libkernel.rs` chunked-read work (GTA V `rpf.cache` 25 MB read)
    removed `READ_MAX_BYTES` but left one use in `hle_getdents`; completed
    as `.min(MAX_HLE_BULK_BYTES)` per the new convention. raeen-hle
    437/437 green. `compat/COMPATIBILITY.md` republished from the
    re-verify (9 rows; generator stamps HEAD `b9c2daf5501d`, tree was
    dirty — read them as `b9c2daf5501d+dirty`).
  * RUNNING: load the 07/27 world (Cross 45 s, Cross 75 s, ls_down 100 s,
    Cross 108 s) with frame dumps — first in-world FPS measurement attempt.

- MC WORLD LOAD FAILS WITH "Launch failed" DIALOG (2026-07-27; runs
  `run-1785128995368` + trace run):
  * Navigation is SOLVED and reproducible: Cross 45 s (main menu), Cross
    75 s (Play screen), ls_down x2 (Create New -> Realms notice -> first
    world row), Cross 112 s selects "My World" (07/27, 0.44 MB). Focus
    order measured one probe at a time: 1 down = Realms notice, 2 = first
    world row.
  * The game mounts the world (`VFS: /savedata0/ -> savedata\PPSA17221-
    app\BedrockWorldHNj1yfs13Kw@P1`) then shows "Launch failed — There was
    a problem loading this world" ~seconds later, with ZERO error/warn
    lines from the VFS/savedata path at default log levels. This is the
    next measured blocker for the in-world FPS gate — a world-LOAD
    failure, not navigation, not rendering. Note: Minecraft worlds are
    LevelDB stores (db/*.ldb, MANIFEST, LOG, LOCK) — suspect VFS
    directory/rename semantics around LevelDB open before anything else.
  * DIAGNOSTIC RUNNING: `RAEEN_TRACE_FILE_IO_AFTER_MS=105000` (per-call
    file tracing from just before the press) to name the exact failing
    open/stat/rename.

- MC WORLD-DIR FORENSICS + TRACE-RUN FLAKE (2026-07-27):
  * BOTH existing "My World" saves are CREATION HUSKS, not loadable
    worlds: `BedrockWorldHNj1yfs13Kw@P1` is an EMPTY directory (no
    level.dat); `BedrockWorldfD72jnHh4GA@P1` has level.dat + a skeleton
    LevelDB (CURRENT + MANIFEST-000010 only, no .ldb data, no LOG/LOCK).
    `BedrockLevelInfoCache` holds an entry per husk. So the "Launch
    failed" dialog on the first world is CORRECT guest behaviour for a
    husk, not (yet) a proven emulator bug — and the husks themselves are
    the fingerprint of an earlier world-CREATION attempt that failed
    part-way (suspect the savedata write path during creation, unproven).
  * The `RAEEN_TRACE_FILE_IO_AFTER_MS=105000` run went LOG-SILENT at
    ~75.5 s (process alive to the 150 s kill; guest submissions, GPU
    worker telemetry, and the input thread all stopped together) — a
    whole-guest stall at menu-load time, ONE occurrence, not reproduced
    in the three adjacent runs with the same press script. Recorded as a
    flake to watch, not a diagnosis.
  * RUNNING: Create New activation probe with dense interval dumps
    (`RAEEN_DUMP_FRAME_AFTER_MS=70000 RAEEN_DUMP_FRAME_INTERVAL=300`) to
    see what the Cross on "Create New" actually does.

- MC PLAY-SCREEN FOCUS MODEL (2026-07-27; frame-verified probes):
  * Initial focus on the Play screen is the Worlds TAB (green fill), NOT
    "Create New" — a blind Cross there is a no-op (tab already active).
  * Measured so far: down x1 -> Realms offline notice; down x2 -> first
    world row (activation proven — "Launch failed" on the husk);
    up x1 from Realms -> FRIENDS tab focus (spatial focus navigation
    skips "Create New" in both directions probed). "Create New" has not
    been reached by direction pad input yet; a full 4-down/4-up chain
    probe with 250-present dumps is running to map the vertical order
    definitively (is Create New in the chain at all?).
  * All screens render correctly throughout (tabs, dialogs, lists) —
    this is purely an input-navigation problem for scripted measurement,
    not a rendering gap.

- MC IN-WORLD: WORLD LOADS, PLAYER CONNECTS+SPAWNS, SLOW STREAMING
  (2026-07-27; run `run-1785130*`, frames `scratch/phase4-loadreal2-20260727/`):
  * REPRODUCIBLE LAUNCH SCRIPT: Cross 45 s (menu), Cross 75 s (Play),
    ls_down x2 (100 s, 104 s), Cross 112 s — loads "My World"
    (`BedrockWorldfD72jnHh4GA@P1`; the empty husk world was moved to
    `scratch/husk-backup/` — reversible, no data existed in it).
  * MEASURED PROGRESSION: world mounts, guest logs "Opening level
    'minecraftWorlds/fD72jnHh4GA@/db'" (LevelDB skeleton opened), MC_SERVER
    thread starts, DBStorage snapshot create/release OK, "Player
    connected." at T+192 s, fresh level.dat (2899 B) + levelname.txt
    WRITTEN to disk (savedata write path works end-to-end), "Player
    Spawned" at T+~3 min. Loading screen renders throughout with a
    slowly-advancing progress bar.
  * OPEN BLOCKER — STREAMING MUTEX CONVOY: title mutex `0x100aa691098`
    (ty=3) is contended by all 7 Streaming Pool threads; the "stuck >3s"
    warns name a ROTATING owner (3 -> 2 -> 4 -> 8 -> 2), i.e. NOT a hard
    deadlock — a convoy of long hold times while chunk work is serialized.
    Streaming is real but glacial (spawn at ~3 min post-connect; loading
    bar still moving at kill time). Unproven whether this is just slow
    worldgen+LevelDB on the VFS, an HLE wait-primitive quantization
    inflating each hold, or the skeleton LevelDB forcing full regen.
  * MINOR VFS ODDITY: `unlink('/savedata0/level.dat')` and
    `/savedata1/fD72jnHh4GA@/level.dat` failed "not found" while
    level.dat existed; the write succeeded anyway seconds later. Two
    different savedata mount spellings are in play (world-root vs
    world-id-relative) — worth a look, non-blocking.
  * FLAKE WATCH: one earlier run (`run-1785129597039`) went log-silent at
    ~75 s for no diagnosed reason; not reproduced since.
  * NEXT: long 600 s run to determine whether streaming converges
    in-world; if it does, the Phase 2 in-world FPS gate is finally
    measurable. If it convoys forever, the next probe is where each
    hold's time goes (chunk decode vs LevelDB read vs wait primitives).

- MC IN-WORLD CRASH ROOT CAUSE — DIRECT-MEMORY BUDGET EXHAUSTION
  (2026-07-27; run `run-1785131*`, crashed at 466.3 s):
  * DISTILLED REPORT NAMED IT CLEANLY: during in-world streaming the
    title's `sceKernelAllocateMainDirectMemory` hit `0x8002000B`
    (EAGAIN — the 13.5 GiB `PS5_DIRECT_MEMORY_SIZE` budget, enforced at
    `libkernel.rs:2371-2395`) and later returned 0x0 x2; Streaming
    Pool(1) then faulted writing a direct VA in a libc SSE copy loop,
    and a second thread executed the guest's deliberate
    `mov dword [0], 0xDEADC0DE` poison abort. The crash report's
    "HLE calls that returned an ERROR before this fault" section worked
    exactly as designed.
  * THE QUESTION: on real hardware Minecraft streams within budget, so
    either the title releases through a path that does not decrement
    `direct_memory_allocated`, or Raeen over-charges. Code audit found
    ONE genuine accounting leak: the physAddrOut-write-failure path
    (`libkernel.rs:2419-2423`) removes the mapping and returns HLE_ERROR
    WITHOUT refunding the budget (the mmap-failure path right above does
    `fetch_sub`) — small, not the main leak. Map paths do not (and
    should not) charge the budget; `AvailableDirectMemorySize` reports
    honestly. MEASUREMENT RUNNING: `RAEEN_TRACE_DIRECT_MEMORY=1` 600 s
    run to get the per-call allocate/release ledger with sizes.

- MC DIRECT-MEMORY LEDGER — THE BUDGET IS NOT THE BUG (2026-07-27;
  `RAEEN_TRACE_DIRECT_MEMORY=1` run `run-1785132354613`, survived 600 s):
  * MEASURED LEDGER: 3902 allocates = 25.90 GiB, 1115 releases = 12.68
    GiB, net live set 13.22 GiB vs budget 13.5 GiB. ALL allocations are
    memoryType=12; the bulk is 785 x 32 MiB blocks (24.5 GiB gross).
    Both release paths (plain + CheckedRelease) decrement correctly —
    the kernel accounting has no big leak (the small physAddrOut rollback
    leak was fixed separately with a regression test).
  * KEY FACT: this run NEVER hit ENOMEM (survived to the 600 s kill);
    the crash run crossed 13.5 GiB by ~a hair. The title's live set
    converges to ~13.2 GiB — it sizes itself to available memory and
    rides the edge on purpose. A budget bump would only paper over the
    real mechanism; the budget correctly models hardware.
  * MECHANISM HYPOTHESIS (unproven, consistent with all evidence): the
    streaming mutex convoy delays chunk CONSUMPTION and therefore the
    title's frees, while worldgen keeps allocating at full rate —
    allocations outpace frees until a transient ENOMEM, which the guest
    handles with a deliberate `0xDEADC0DE` poison abort. On hardware the
    same code streams fast enough that frees keep up. If true, the fix
    is the convoy/slow-streaming, not memory accounting.
  * NEXT LEADS (ranked): (1) where does each convoy hold spend its time —
    LevelDB reads through the VFS (suspect open/read amplification on
    small chunk reads), decompression, or worldgen compute;
    (2) the skeleton-LevelDB factor — a husk-derived world forces full
    regen churn; a world created FRESH through the working creation flow
    (once Create New is reached) may stream far less.

- PHASE 2.1 GATE GREEN — ASYNC FLIP NOW DEFAULT (2026-07-26; working tree at

  * Also observed pre-crash: the streaming mutex convoy persists and
    worsens (present cadence collapsed to ~5 fps near the end) — likely
    the same memory pressure from the title's side.










  `b9c2daf+dirty`, no commit):
  * THREE-RUN NO-WEDGE EVIDENCE (release builds, max-fps profile =
    `RAEEN_ASYNC_FLIP=1`): `run-1785110215494` Minecraft 12192 flips,
    `run-1785111755329` 10432 flips, `run-1785111946064` 9920 flips — all
    timed_out cleanly at 180 s with flips still flowing in the final
    sub-second window, zero "stuck >3s" warnings, zero blockers. (Runs 2/3
    had an idle Shell GUI resident; their ~15 % lower flip counts vs run 1
    are unattributed between that and the 2.3/2.4 changes — recorded as
    noise, not a regression claim.)
  * MEASURED EFFECT: per-flip worker drain 16–24 µs (ASTRO's synchronous
    path measured ~2.3 s in the Phase 0 diagnosis); fence wait 0.386–1.649 ms
    against a 17.5–19.3 ms frame (~2–9 %, NO LONGER DOMINANT — worker
    occupancy is submit_pct 52–54 / flush_pct 18–19 / idle ~28, so PM4
    translation/recording is now the binding constraint, not the fence);
    readback 0.294–0.596 ms; srgb_encode 0 µs on Minecraft's 8-bit presents.
  * DEFAULT FLIPPED: `async_flip_enabled(None)` now returns true;
    `RAEEN_ASYNC_FLIP=0/false/no/off` is the documented A/B opt-out. The
    2026-07-20 wedge was never root-caused to a named mutex; the bound
    (cap-2 FlipSemaphore + flip-time scanout snapshots + panic-safe permit
    release) is the mitigation the gate measured. If ANY future wedge shows
    a "stuck >3s" warn, its mutex key + owner name are the diagnosis path —
    do not revert the default without that record. AGENTS.md and the xtask
    profile comment updated to match.
  * 2.2 SCOPING DECISION (command-buffer ring): DEFERRED with evidence. The
    ring's Phase 2 purpose was killing the dominant fence stall; after 2.1
    the fence wait is 2–9 % of frame time and submission-side translation
    dominates. Building N=8 fenced segments now would be unmeasured-value
    work; revisit when the timing HUD shows fence_wait dominant again.
  * VERIFICATION: raeen-gpu 262 lib tests green (incl. the rewritten
    default-on test), clippy `-D warnings` green, fmt clean.
  * PRIOR-PROBE HONESTY NOTE: the 2026-07-26 05:08 "create-world" probe
    (build `5bcf76c353b8`, observed_fps 52.5) dumped BLACK frames
    (frame_000008/000032 verified) — it is NOT a credible in-world FPS
    measurement. The "Minecraft ≥60 FPS in-world" gate item remains
    unmeasured and is coupled to the Phase 4.4 blank post-menu page.
  * RECON #9 MEASURED NEGATIVE: the `MIP_VIEW_BASE_LEVEL_IGNORED` tripwire
    fired ZERO times across the full 9-binary re-verify sweep — no tracked
    title selects a non-zero mip view base, so the GFX10 mip-tail addressing
    port (SharpEmu 6ee445f) is NOT justified by measurement. The counter
    stays as the tripwire; `docs/reference-recon-roadmap.md` #9 updated from
    "OPEN" to "MEASURED IRRELEVANT for the corpus". Same discipline as #11
    (MRT) — implement when a title trips it, not before.

- PHASE 1 RE-VERIFY + PHASE 2 START (2026-07-26; working tree at
  `b9c2daf+dirty`, no commit; measurements on isolated release builds):
  * PHASE 1 GATE RE-VERIFIED on the current tree (`run-1785110215494`,
    `artifacts/compat/phase1-reverify-b9c2daf-dirty.json`, exe SHA-256
    `453db9d562885b325e3d0e7357b776cf2416ced1e80773f2894dc42536065eb1`):
    6/8 distinct titles >30 s, zero unimplemented-import deaths, and every
    first-blocker signature matches the ledger (UE5 `read 0xa`/execute fault,
    GTA V fault after exactly 2733 HLE calls, Avatar `s_load_dwordx16`,
    Subnautica async-exception acknowledgement). The dirty tree's in-progress
    graphics work IMPROVED three titles: Minecraft 8192 -> 12192 flips (no
    wedge), ASTRO.BOT 0 -> 96 flips, Avatar shader errors 8565 -> 1777 with
    132 -> 137 flips. A Plague Tale still dies with no in-log crash record
    (the known host-side death, Phase 4.2) at 29.3 s vs 33.5 s — same crash
    class, marginally earlier; watch it.
  * NOTE — max-fps compat profile already opts into `RAEEN_ASYNC_FLIP=1`
    (xtask/src/main.rs), so the 12192-flip Minecraft run above is ALSO gated
    async-flip no-wedge run #1 of the 3-run Phase 2.1 gate.
  * 2.4 sRGB PRESENT COST: replaced the per-pixel `powf` HDR->sRGB encode with
    exact 64 KiB LUTs keyed by binary16 bit pattern (a new test locks all
    65536 patterns to the scalar reference, NaN/Inf included), and added an
    Arc-identity encode cache (Weak-validated, bounded to 8 entries) so a
    title flipping without redrawing no longer re-encodes 8.3 Mpx per flip.
    raeen-gpu 262 lib tests green; clippy `-D warnings` green; fmt clean.
    Pipeline-cache serialization (recon #28) and the cheaper guest-mem
    present pass / sticky flip-miss fallback (recon #27) were verified
    ALREADY LANDED — do not redo.
  * 2.3 VBLANK PACING: `sceVideoOutWaitVblank` no longer yield-spins up to a
    full ~15.6 ms Windows timer tick per wait. Epoch-anchored absolute
    schedule kept; the bulk wait now uses a per-thread high-resolution
    waitable timer (CreateWaitableTimerExW HIGH_RESOLUTION, concept from
    shadPS4 AccurateSleep — reimplemented, no code copied) with only the
    final 1 ms spinning; falls back to coarse-sleep when no hi-res timer is
    available. New test pins two consecutive waits one period apart.
    raeen-hle 436 lib tests green; clippy `-D warnings` green; fmt clean.

  * FETCH/PULL: fast-forwarded SharpEmu `21f964a -> 0535783` (8 commits),
    shadPS4 `8161049 -> d976c33` (7), and Mesa `3e2d851 -> 780727e` (12).
    Kyty, KytyPS5, OpenOrbis, ps4libdoc, and ps5-payload-sdk were already
    current. V8 remained on its intentional detached pin. `ghidra-orbis` has
    one local edit, so only its origin metadata was fetched; no merge was
    attempted over that work.
  * AUDIT NEGATIVES: Mesa's delta contains zero `src/amd` changes. SharpEmu's
    new host-cached guest buffer, detile cache identity/resource pooling, and
    sparse large-region reserve work are already present in Raeen's Vulkan
    pools/`TextureKey` and sparse guest arena, so no duplicate ports landed.
    shadPS4's PS4 minimum-LOD and buffer-sharp changes were not copied into the
    PS5 descriptor path without measured semantic evidence.
  * BUG FOUND from shadPS4 `26f4270`: Raeen's pthread condition state used one
    global generation. `signal` incremented it and notified one host waiter,
    but every other waiter observed the changed generation on its next 10 ms
    bounded polling wake—turning signal-one into a delayed broadcast.
  * FIX: original Rust FIFO/per-waiter wake objects in `raeen-kernel`; wait
    registration remains atomic with guest-mutex release, signal/timeout races
    resolve under one queue lock, broadcast drains the queue, and
    `scePthreadCondSignalto` selects the requested guest thread instead of
    approximating arbitrary signal-one. The two-waiter regression failed first
    against the generation model, then passed with the FIFO.
  * VERIFIED: `raeen-kernel` 41 + 2 integration tests, `raeen-hle` 433 tests,
    `raeen-runtime` 77 + 46 execution + interactive-2D tests all green; clippy
    `-D warnings` green for kernel/HLE/runtime; isolated release build green.
  * MINECRAFT A/B: `max-fps`, release, 180.395 s, **11,392 flips / 68.3
    observed FPS / 1.170 GB peak**, zero shader/GPU/audio errors, no blocker or
    wedge (`scratch/reference-refresh-minecraft-max-fps.json`,
    `run-1785076269737`). Immediate pre-slice comparison was 10,368 flips /
    56.7 FPS / 1.468 GB peak. This proves no regression and records an observed
    +9.9% flip count; run variance means the FIFO is not claimed as the sole
    cause.

- PHASE 1 LOAD-PATH CONTINUATION — measured profiling replaced the proposed
  relocation parallelism with two higher-ROI fixes (2026-07-26; working tree at
  `ef2e7dc15665+dirty`, no commit; Phase 1 was already green and this does not
  start Phase 2):
  * RELOCATION HYPOTHESIS REJECTED: with `RAEEN_NO_LINK_CACHE=1` and the new
    env-gated `RAEEN_TIME_LINK=1` telemetry, ASTRO applied **560,259
    relocations in 3.584 ms** (~156 million/s). A second sample was 2.628 ms.
    Adding Rayon would target under 0.1% of the measured 6.762 s cold load,
    add scheduling/ordering risk around shared import state, and cannot produce
    a material launch-time win, so no parallel relocation code/dependency was
    added.
  * ACTUAL WALL: `ModuleIndex::build` spent **5,177.928 ms** recursively
    enumerating the package-root `data/` tree to find only the same two
    modules. Direct corpus checks found **zero `.prx`/`.sprx` files** in both
    Minecraft's and ASTRO's top-level `data/` trees; enumerating them alone
    cost about 9.9 s and 18.8 s respectively from PowerShell.
  * FIX: module discovery now prunes package-root `data/` only, while retaining
    `sce_module`, `Media/Modules`, `Media/Plugins`, and nested engine paths.
    `module_index_prunes_bulk_data_but_keeps_known_module_directories` first
    failed on the old traversal, then passed with the bounded rule.
  * COLD A/B (same ASTRO executable, isolated release build, link cache
    disabled): module indexing fell **5,177.9 ms -> 41.1 ms** (99.2%); total
    process load fell **6,762 ms -> 1,529 ms** (77.4%). Both runs found the
    same two modules and reached the same runtime stage with four flips in the
    30 s probe. Reports: `scratch/phase1-relocation-profile/
    serial-phase-profile.json` (`run-1785073643413`) and
    `data-prune-profile.json` (`run-1785074007187`).
  * SYSCALL/CPUID PREFILTER FINISHED: the pre-existing SIMD opcode prefilter
    still decoded the 251 MB main executable twice—once for `syscall`, once for
    `cpuid`. Both traps now share one iced-x86 instruction-boundary pass.
    `syscall_and_cpuid_share_one_executable_decode_pass` proves real opcodes are
    patched and matching bytes inside immediates remain untouched. The measured
    main-image patch stage fell from an inferred **727.7 ms -> 498.3 ms**
    (31.5%); whole-load time varied between runs, so no second whole-load claim
    is made. Report: `one-pass-profile.json` (`run-1785074291213`).
  * MINECRAFT A/B GREEN, compared on the exact recorded `max-fps` profile:
    baseline `artifacts/compat/phase1-final.json` was **8,192 flips / 180.363 s
    / 1.880 GB peak**; this build produced **10,368 flips / 180.342 s / 56.7
    observed FPS / 1.468 GB peak**, with zero shader/GPU/audio errors
    (`scratch/phase1-relocation-profile/minecraft-max-fps-final.json`,
    `run-1785074928163`). That is 26.6% more flips and 21.9% lower peak working
    set for this measured run, not a universal performance claim. The default
    paced `compatibility` profile separately produced 5,952 flips and 21.2
    observed FPS; it is recorded but not compared to the `max-fps` baseline.
  * VERIFIED: `cargo test -p raeen-firmware` (125 library + 17 integration,
    all green), `cargo test -p raeen-runtime` (77 unit + 46 execution + the
    interactive-2D acceptance test; one manual benchmark ignored), firmware
    clippy with `-D warnings`, isolated release `raeen-gui` build, fmt check,
    and `git diff --check` all exit 0.

- CLEAN CLONE + CI UNBROKEN — `plugins/upscale` moved in-tree to
  `crates/raeen-upscale` (2026-07-26; working tree, no commit; fmt clean,
  `cargo clippy --workspace -- -D warnings` exit 0 / zero warnings):
  * THE BUG, measured with a control rather than reasoned: `c0f6303`
    (2026-07-23 21:16) added
    `raeen-upscale = { path = "../../plugins/upscale", optional = true }` to
    `crates/raeen-gui/Cargo.toml`, but `.gitignore:82` is `/plugins/*`, so
    `plugins/upscale/` was NEVER IN THE REPOSITORY. Cargo resolves path
    dependencies at manifest-load time regardless of `optional` and regardless
    of `[workspace] exclude`, so a fresh checkout dies at resolution:
    `failed to get raeen-upscale as a dependency ... failed to read
    plugins/upscale/Cargo.toml`.
  * IMPACT: `ci.yml` runs `actions/checkout@v4` then fmt/clippy/test over the
    workspace, so **every CI run failed at resolution for 15 commits (~3 days)**
    and produced zero signal, and **no clean clone could build at all**. All
    "green" claims in that window were local-only, valid only on a machine with
    an untracked `plugins/upscale/`. Found by accident while building a baseline
    worktree for an unrelated regression check.
  * WHY `exclude` DID NOT SAVE IT (the trap, now documented in `Cargo.toml`):
    `exclude` only stops a directory being a workspace *member*. A path
    *dependency* from a member still has to resolve. Comment added at the
    `exclude` line telling future work not to reintroduce a path dep under
    `plugins/`.
  * PROOF (control + treatment in the same fresh worktree, `plugins/` holding
    only `README.md`): unfixed HEAD `cargo metadata` → **exit 101** with the
    resolution error; with the fix → **exit 0**. Note `cargo metadata
    --no-deps` does NOT reproduce it (it skips dependency resolution) — an
    earlier probe using `--no-deps` gave a false pass and was corrected.
  * FIX: `plugins/upscale` → `crates/raeen-upscale`, added to `[workspace]
    members`, `raeen-gui` path → `../raeen-upscale`, crate switched to
    `version/edition/license.workspace` (edition 2021 → 2024, compiles clean),
    its own `Cargo.lock`/`target/` dropped. `plugins/` is now reserved for
    user-supplied plugin BINARIES loaded through the C ABI, which is what
    `plugins/README.md` always said it was for.
  * APPROACH CHANGED AFTER READING THE CODE: the plan was to convert this crate
    to a C-ABI loaded plugin. Inspection showed it is pure Rust, GPL-2.0-only,
    with NO proprietary dependencies — the `fsr`/`dlss`/`xess` backends only
    *probe* for a vendor runtime by filename (`nvngx_dlss.dll`, …) and report
    unavailable; nothing links or loads one. Converting would have been a
    586-line rewrite AND would have taken working GPL-clean spatial upscalers
    away from users. Moving it in-tree is both smaller and better.
  * LATENT BREAKAGE SURFACED: because the crate was workspace-EXCLUDED, it had
    never been covered by `clippy --workspace -- -D warnings` or
    `test --workspace`. Including it immediately produced **4 clippy errors**
    (`default()` on a unit struct, a collapsible `if` → let-chain, two
    `identity_op` `* 1` in a test) — i.e. adding it would have broken CI a
    second way. All fixed; its **7 tests now run in CI for the first time**.
  * VERIFIED: `cargo fmt --all -- --check` exit 0; `cargo clippy --workspace --
    -D warnings` exit 0, zero warnings; `cargo build -p raeen-gui --features
    upscale-plugins` succeeds (the opt-in path still works).
  * MY OWN FLAKY TEST, CAUGHT BY THE NEWLY-WORKING GATE AND FIXED: the first
    `cargo test --workspace` after the resolution fix exited **101** — not on the
    known GPU test, but on `present_plugin_dylib::a_real_plugin_survives_many_
    frames_without_leaking_or_faulting`:
    `LoadLibraryExW Os { code: 4551, "An Application Control policy has blocked
    this file." }`. Windows Application Control refused a freshly-written
    unsigned DLL in `%TEMP%`; two of the three tests loaded first and the third
    was blocked, so it is reputation-based and NONDETERMINISTIC on identical
    code — the same defect class this session criticized in
    `guest_memory_pixel_shader_draws_green`, introduced by me hours earlier.
    Three changes: (1) compile the example **once** for the suite behind a
    `OnceLock` instead of three times — each distinct fresh unsigned binary is
    an independent chance to be blocked; (2) write it under
    `<target>/<profile>/raeen-plugin-dylib-test` instead of `%TEMP%`, which
    Application Control scrutinizes far harder than build output (and it now
    cleans with `cargo clean`); (3) new `policy_blocked()` treats an
    executable-policy refusal as a loud `SKIP:` rather than a failure, matching
    OS code 4551 in the `Debug` rendering so the check is locale-independent.
    Stable across 5 consecutive runs and ~2.5x faster (one rustc, not three).
    HONEST LIMIT: on a host with strict Application Control these 3 tests SKIP
    instead of verifying — visible in output, not silent.
  * EXIT-CODE MASKING, AGAIN, VIA A NEW ROUTE: the background-task notification
    for that run reported "exit code 0" because the shell pipeline's last
    command was a `grep`. Cargo's real 101 was only visible because the script
    wrote `TEST_EXIT=$?` into the log. Same trap the ledger already documents
    for `| tail`, arriving through task-runner summaries.
  * FINAL GATE STATE (this build): `cargo test --workspace` = **1285 passed,
    22 suites, 1 suite FAILED** — the sole failure is the pre-existing,
    environment-dependent `guest_memory_pixel_shader_draws_green` (proven this
    session to fail identically at clean HEAD, see its own entry). fmt exit 0;
    `clippy --workspace -- -D warnings` exit 0, zero warnings. So CI moves from
    "cannot resolve, zero signal on every gate" to "compiles, lints clean, 1285
    tests green, one known-flaky GPU test red."
  * NOTE: a separate 15 GiB scratch `CARGO_TARGET_DIR` used earlier in the
    session developed corrupted fingerprints (cargo reported deps fresh while
    passing `--extern` paths rustc could not resolve, failing `raeen-gpu` with
    `can't find crate for raeen_core`). Unrelated to this change — the same
    build succeeds in the default `target/`. Dir deleted.

- PRESENT-PLUGIN C ABI + RUNTIME LOADING — BYO plugins no longer require
  rebuilding Raeen (2026-07-25; working tree, no commit; raeen-gpu 248/248
  incl. 12 new, clippy `-D warnings` clean on raeen-gpu + raeen-gui, fmt clean):
  * CLOSES the follow-up recorded in the 2026-07-23 plugin-ABI entry ("stable
    C-ABI `dlopen` layer for no-recompile dynamic loading"). Until now
    `PresentPlugin` was a Rust trait, so a plugin had to be COMPILED INTO
    `raeen-gpu` — exactly the linking arrangement `plugins/README.md` forbids
    for proprietary code. A plugin is now a separate user-supplied binary
    loaded at runtime; the distributed artifact links none of it.
  * NEW `crates/raeen-gpu/src/present_plugin/cabi.rs`: `#[repr(C)]`
    `RaeenPluginV1` vtable (create/destroy/name/capabilities/process/
    release_output), ABI version 1, single exported entry `raeen_plugin_v1`.
    `DynamicPlugin` adapts a loaded vtable to the in-tree trait;
    `load_from_path` / `scan_dir` / `load_and_register_dir` do the
    `LoadLibrary`/`dlopen` work. `libloading` 0.8 added (workspace + raeen-gpu).
  * OWNERSHIP RULE: the plugin allocates output pixels and frees them via
    `release_output`, which Raeen calls whenever `process` returned success —
    INCLUDING when Raeen then rejects the output. Neither side frees the
    other's allocation, so the two may use different allocators/CRTs/languages.
    Pinned by `output_is_released_back_to_the_plugin_after_every_frame` (8
    frames, unfreed count must return to 0) and by the rejection test.
  * NAME-BOUNDED BY DESIGN: `name` writes into a caller-supplied 128-byte
    buffer and returns the count, rather than returning `const char *` — a
    returned pointer would force Raeen to scan for a NUL a buggy plugin might
    never place.
  * FINDING (test caught a real limit in my own validation, not a code bug):
    `copy_frame` verifies `pixels_len == width*height*bpp`, but BOTH are the
    plugin's own claims. A plugin that under-allocates and reports a
    *consistent* pair (4-byte buffer described as 2x2x4) is UNDETECTABLE — the
    first version of the lying-plugin test did exactly that and produced a real
    16-byte read off a 4-byte allocation (heap garbage in the assert output).
    No portable mechanism can ask a foreign allocator a block's true size, so
    the test was changed to model the DETECTABLE lie (length disagreeing with
    dimensions, which is refused) and the residual risk is documented
    prominently on `copy_frame` and in `plugins/README.md`. This is the same
    exposure every C plugin ABI carries and is why loading is user-gated.
  * VALIDATION that IS enforced (each a refusal + named warning, never a silent
    no-op): null vtable, ABI mismatch (checked BEFORE any other vtable call —
    on a version mismatch the struct read may not be the struct written), null
    `create`, empty/over-long/non-UTF-8 name (instance destroyed on the refusal
    path, not leaked), null pixels, zero or >16384 edge, bytes-per-pixel not 4
    or 8, length disagreeing with dimensions, >1 GiB buffer, >8 generated
    frames. `copy_frame_rejects_every_malformed_descriptor` covers 6 shapes.
  * WIRED (not dead code): `AgcGpuSession::load_present_plugins_from(dir)` +
    a startup scan of `plugins/` in `raeen-gui/src/main.rs`, ordered BEFORE
    `apply_present_plugin` so a persisted selection naming an out-of-tree
    plugin resolves. The Settings ▸ Video dropdown already existed and now
    lists loaded plugins. Missing directory = empty, not an error.
  * LICENSE BOUNDARY re-verified: `git check-ignore` confirms
    `plugins/my-upscaler.dll` and `plugins/dlss/shim.dll` IGNORED,
    `plugins/README.md` TRACKED. README rewritten with the full C header, the
    enforced rules, a working Rust `cdylib` example, and install steps.
  * END-TO-END PROOF (closes the earlier "no real `.dll` round-tripped" gap):
    NEW `docs/examples/present-plugin-example.rs` — a complete, dependency-free
    nearest-neighbour upscaler, single-file so a bare `rustc` builds it. NEW
    `crates/raeen-gpu/tests/present_plugin_dylib.rs` (3 tests) compiles THAT
    EXACT FILE into a real cdylib with `rustc` (not `cargo` — a nested cargo
    contends for the target-dir lock and can deadlock), then loads it through
    the same `scan_dir` the Shell uses. Asserts the 2x upscale is a correct 2D
    nearest map (all four source texels land in the right blocks, so a row/
    column smear would fail), that a declined frame returns the source
    pixel-for-pixel, and that 250 consecutive frames neither leak nor fault.
    The shipped example is therefore the verified artifact.
  * SHELL-LEVEL PROOF (measured, not reasoned): built the example into
    `plugins/` and ran the debug Shell 12 s. Log:
    `registered out-of-tree present plugin plugin=example-nearest
    source=plugins\raeen_example_plugin.dll capabilities=Capabilities {
    upscale: true, ... }` then `loaded user-supplied present plugins count=1
    plugins=["example-nearest"]`. rustc's `.exp`/`.lib`/`.pdb` siblings are
    correctly ignored by the extension filter.
  * FINDING — REPO DOES NOT BUILD FROM A CLEAN CLONE (pre-existing, unrelated
    to this work, found while building a baseline worktree):
    `crates/raeen-gui/Cargo.toml:38` declares
    `raeen-upscale = { path = "../../plugins/upscale", optional = true }`, but
    `plugins/*` is gitignored, so `plugins/upscale/` is not in the repo. Cargo
    resolves path dependencies at manifest-load time even when optional, so a
    fresh `git clone` fails workspace resolution with "failed to read
    plugins/upscale/Cargo.toml" — `cargo build` cannot run at all. The new C
    ABI removes the reason to have a plugin as a workspace member; migrating
    `raeen-upscale` to a loaded binary (or tracking it) would fix this.
  * FINDING — `shader_memory_phase2::guest_memory_pixel_shader_draws_green` is
    ENVIRONMENT-DEPENDENT, not a regression. Sequence measured this session:
    passed at clean HEAD in a fresh worktree; then failed at clean HEAD (3
    runs) AND at clean HEAD + only this session's changes (3 runs), with the
    only variable being elapsed time and a `raeen.exe` having been run in
    between (PID 13960 still resident at the last measurement). Since clean
    HEAD fails identically, this work is exonerated. An earlier note in this
    session attributing the failure to the uncommitted `RAEEN_ASYNC_FLIP`
    work in `agc_exec.rs` was WRONG and is retracted — that hypothesis was
    formed before the clean-HEAD re-measurement. Real open question: why a
    resident `raeen.exe` (or whatever else changed) flips a Vulkan offscreen
    shader test that does not skip on device acquisition.
  * NOT DONE / HONEST LIMITS: (b) frames are
    still CPU pixel buffers, so real GPU upscalers (DLSS/FSR3/XeSS) cannot use
    this yet — a `VkImage`-handle ABI v2 is gated on the GPU-side present path.
    (c) `depth`/`motion` remain NULL (PM4 extraction not started). (d)
    `generated` frames are validated and copied but never scheduled for
    display. (e) No Minecraft A/B was run — the present path is unchanged when
    no plugin is selected (identity `Arc` return), but that is reasoning, not
    a measurement.

- BATTLE-READY WORKFLOW PHASE 1 GREEN — HLE BREADTH + LOAD PATH
  (2026-07-25; working tree at `01f7b613911a+dirty`, no commit; Phase 2 may
  now start):
  * INVENTORY + FAMILY BATCH: the Phase 0 nine-binary harvest found seven
    actually-called missing imports in Avatar/Subnautica. All seven are
    registered by family with bounded guest-memory-aware behavior:
    PlayGo/Dialog, VideoOut latency control, standard DualSense class
    information, pthread stack attributes, and kernel exception registration/
    acknowledgement. Unsupported asynchronous exception delivery, host dialog
    UI/PlayGo metadata, and latency-controller semantics stay explicitly
    incomplete.
  * HONEST COVERAGE: `HleRegistry::register_incomplete` plus schema-v2
    `cargo xtask nids coverage` emits a deterministic
    `registered_but_not_implemented` table. The sanitized
    `artifacts/compat/phase1-nid-coverage.json` covers all nine binaries and
    reports 40 incomplete registrations. The hash-gated name hunt scanned
    48,660,122 candidates without admitting guesses; eight measured imports
    remain anonymous. Required AGC sizing and AJM silence-path audits remain
    classified as incomplete where they do not implement real driver/codec
    behavior.
  * DIRECT VFS READS: `VfsSubsystem`, `VirtualFileSystem`, HLE guest memory,
    and `GuestArena` now support validated caller-buffer reads. Retail reads no
    longer clone handles, allocate/copy a staging `Vec`, or hold a global write
    lock. Invalid guest output ranges fail before host I/O, so `read(EFAULT)`
    no longer consumes the shared file cursor; `pread` remains positional.
  * LINK/LOAD CACHE: syscall/CPUID patching prefilters raw opcodes; relocation
    linking resolves each provider/NID/table-marker once per imported symbol.
    A version/source/HLE/input/dependency-keyed persistent cache stores the
    decrypted+linked+patched image as raw bytes plus bounded JSON metadata under
    a gitignored machine-local directory. Cache restore refreshes per-process
    HLE data and replays every module export, including modules without unwind
    records. Corrupt or structurally invalid entries fail open to a cold load.
    The cache may contain decrypted user-owned executable bytes and must never
    be committed or published.
  * MEASURED EXIT GATE: isolated release SHA-256
    `EA891F1C76FA7B27E6ED899755EA5DE1564061373675F013E8F82AF5899D875F`
    produced
    `artifacts/compat/phase1-final.json` / `run-1785034944986`. Seven of eight
    distinct titles exceeded 30 seconds (seven of nine registered binaries;
    the second Dragon Ball image exits early), with zero
    unimplemented-import-death or called-unresolved markers. Minecraft ran
    180.4 s and retained exactly 8192 flips, matching Phase 0. ASTRO's fresh
    cold process load was 7612 ms versus the workflow's ~27 s baseline
    (71.8% reduction). Avatar improved from the Phase 0 57.8 s shader crash to
    a 180.6 s timeout with 132 flips, but still reports 8565 unsupported-shader
    errors; this is not a gameplay claim.
  * WARM-CACHE CAVEAT: a controlled ASTRO relaunch did hit the persistent cache
    and restored its 272,886,072-byte image/two dependencies, but measured
    6495 ms. That is only a modest improvement over the 7612 ms cold path, not
    the workflow's aspirational tens-of-milliseconds result. The formal Phase 1
    gate is green; eliminating large-image read/hash/materialization cost
    remains a measured optimization, not hidden success.
  * VERIFICATION: raeen-core 10/10; raeen-kernel 40/40 plus two homebrew tests;
    raeen-hle 428/428; raeen-firmware 123 library + 11 NID + 3 homebrew + 3
    transitive tests; raeen-runtime 75/75 library + 46/46 native execution
    (one manual benchmark ignored) + M3 fixture; scoped clippy across the
    touched core/kernel/HLE/firmware/runtime crates is clean with
    `-D warnings`; formatting is clean. Phase 1 is compatibility/load evidence,
    not a new M2-M5 compatibility claim.

- BATTLE-READY WORKFLOW PHASE 0 GREEN (2026-07-25; working tree at
  `01f7b613911a+dirty`, no commit; Phase 1 NOT STARTED):
  * TASK 0.1: rewrote the local `AGENTS.md` from current commercial-title
    reality. It now identifies M4-class compatibility/diagnostics rather than
    obsolete M1 bring-up, points at `.agents/skills` + `.codex/agents`, records
    the measured baseline and enforces isolated Cargo targets. The file is
    intentionally workspace-local under the repository's existing
    case-insensitive `Agents.md` ignore rule.
  * TASK 0.2: callable unresolved NIDs now fail soft by default, log structured
    NID/resolved-name/import-library/calling-module data once per process key,
    return `RAX=0`, repair the guest call stack and continue.
    `RAEEN_STRICT_NIDS=1` retains hard failure; unresolved data imports remain
    hard failures because no honest object value can be synthesized. The
    process kernel owns the deduplicated counted inventory and logs its sorted
    summary on orderly teardown.
  * RETAIL BUG FOUND BY THE EXIT GATE: the executable HLE bridge stored
    `DirectThreadState*` at guest `FS:[0x7f0]`. Windows clears guest FS on
    preemption, so retail initializers faulted inside the bridge at
    `0x4000...` before fail-soft dispatch. A controlled Subnautica A/B was
    1.9 s with direct thunks versus a clean 45.1 s timeout with them disabled.
    The bridge now uses the x64 TEB's application-owned, preemption-stable
    `GS:[0x28]` slot, preserves/restores its prior value, and has generated-byte
    plus TEB round-trip regressions. The fixed direct path also timed out
    cleanly at 45.1 s.
  * TASK 0.3: verified the 4096-call crash ring is emitted as one DEBUG event;
    default INFO reports lead with one distilled ERROR plus bounded WARN
    context. No per-call ring spam appeared in the measured logs.
  * TASK 0.4: added always-on per-frame microsecond timing for worker queue
    drain, Vulkan fence wait, readback and sRGB encode; IPC v5 transports it to
    the Shell, where egui conversion/texture upload is measured. Native GUI
    acceptance visibly showed
    `drain 0.0  fence 0.4  read 0.6  sRGB 0.0  UI 0.8 ms` on a real Minecraft
    loading frame (`scratch/phase0-minecraft-hud.png`).
  * TASK 0.5: verified the existing fail-closed VFS resolution covers drive
    letters, parent traversal, malformed input and symlink/junction escape,
    while legitimate contained create paths remain green (raeen-kernel 39/39).
  * TASK 0.6 / CLEAN ROOM: shallow sparse Mesa reference acquired at
    `3e2d8517b897026377267c09975db83525d2fc95` under gitignored
    `reference/mesa`; AddrLib per-file MIT headers checked and Mesa state/notice
    recorded. Ghidra 12.1.2 + JDK 21 installed externally under
    `C:\Users\whoisraeen\Tools`; the GPL-3.0 GhidraOrbis PRX/SELF loader is
    installed externally only and no incompatible code entered Raeen.
  * PHASE 0 EXIT EVIDENCE: release `cargo xtask compat run --tier all
    --timeout 180` produced sanitized
    `artifacts/compat/phase0-final.json` / `run-1785019657785` for all eight
    named titles (nine binaries because both Dragon Ball variants are
    registered). Minecraft = 180.4 s / 8192 flips / 1737 MiB / no blocker
    (prior baseline 2048 flips, so no presentation regression); Subnautica =
    180.1 s / four unique called NIDs each logged once / no blocker; Astro =
    180.2 s; GTA V = 180.1 s. Honest negative results: Until Dawn still exits
    at 4.7 s, one Dragon Ball variant has the UE5 read-`0xa` worker fault while
    the process survives 180.1 s, its decrypted duplicate exits at 4.4 s,
    A Plague Tale crashes at 34.0 s, and Avatar reaches 102 flips then crashes
    at 57.8 s on unsupported `s_load_dwordx16`.
  * VERIFICATION: raeen-gpu 233/233, raeen-kernel 39/39,
    raeen-runtime 74/74 library + 46/46 native execution (one manual benchmark
    ignored) + M3 fixture,
    raeen-gui 160/160; scoped clippy for kernel/runtime/GPU/GUI is clean with
    `-D warnings`. Phase 0 is instrumentation/containment evidence, not a new
    M2-M5 compatibility claim.

- MINECRAFT WORLD-ENTRY PERFORMANCE + DEAD-WORKER FIXES (2026-07-25; no
  commit; HLE 392/392, GPU 233/233, scoped clippy `-D warnings` + fmt clean,
  release Shell built):
  * Persistent sampled-texture caching is default-on again after a real
    correctness A/B: Minecraft's panorama, menu text/icons and world thumbnail
    remained correct in the GUI. Worker-frame time fell from ~100-115 ms to
    ~22-29 ms and the visible Worlds screen measured 38-42 FPS. The cache key
    now includes tile layout, preventing identical guest bytes decoded under
    incompatible linear/SW_64KB layouts; `RAEEN_NO_TEX_CACHE=1` remains the
    diagnostic escape hatch.
  * Offline RakNet `recv`/`recvfrom` now backs off 1 ms on EWOULDBLOCK instead
    of hot-spinning. Measured replay: 17,883,821 calls -> 54,703 over the same
    interval (~178x fewer), with normalized child CPU about one-third lower.
  * Infinite pthread condition waits no longer manufacture a guest-visible
    spurious wake every 10 ms. They remain in the host condvar until a real
    generation change/deadline/process termination, so every idle Minecraft
    worker no longer traps through mutex+predicate+wait at 100 Hz. A focused
    cross-thread wake test pins the behavior.
  * ROOT CAUSE OF THE POST-WORLD 0-FPS STALL FOUND: async workers faulted on
    three unresolved imports, then the main thread waited forever on their
    completion condition. Added process-scoped offline `libSceNpAuth`
    Create/GetAuthorizationCodeV3/Delete, offline
    `libSceNpAuthAuthorizedApp::sceNpAuthGetAuthorizedAppCode`, and aliased
    `sceSaveDataTransferringMountPs4` to the existing honest no-foreign-save
    result. Clean-room behavior cross-checked against GPL-2.0 shadPS4; no
    credentials are fabricated.
  * MEASURED EFFECT: before the fix, workers faulted at 37-39 s and the title
    stopped flipping after saved-world Cross. After the fix, the same
    deterministic replay had zero unimplemented-import/worker faults through
    the world selection and advanced to present/submission 4096 immediately
    afterward (111,623 draws / 74,524 dispatches recorded), so the 0-FPS
    dead-worker stall is removed.
  * HONEST STATUS: active world-loading/render work is now running, but a
    screenshot proving recognizable gameplay and the launch->move->save loop
    are still OPEN. Do not call this full gameplay or stable 60 FPS yet.

- NP STATE CALLBACK NOW DELIVERED (real fix, insufficient for the blank page)
  + recvfrom exonerated (2026-07-24; raeen-hle 390/390, raeen-kernel 36/36,
  clippy `-D warnings` clean):
  * REAL BUG FIXED: `sceNpRegisterStateCallback{,A,ForToolkit}` recorded nothing
    and `sceNpCheckCallback` was `hle_ok` — so a title that registers an NP
    state callback and pumps `sceNpCheckCallback` waiting for the initial
    account-state event waited forever. Now the registration is recorded
    (`OrbisKernel::np_state_callbacks`, new `NpStateCallbackRegistration`) and
    the first pump delivers the current state (SIGNED_OUT, consistent with
    `sceNpGetState`) to the guest callback via the deferred guest-call channel,
    once, with the correct per-form ABI: legacy = `(userId, state, npId*,
    userdata)`, A/toolkit = `(userId, state, userdata)`. shadPS4
    `np_manager.cpp DispatchPendingNpStateCallbacks` cited. 2 focused tests
    (recording GuestCallScheduler) pin the delivery + the per-form arg layout.
  * MEASURED EFFECT: the guest reacted — Minecraft registers via
    `sceNpRegisterStateCallbackA` (the 3-arg form, so the form-tracking is
    load-bearing) and `recvfrom` dropped 15.2M -> 9.7M steady. But the
    post-"Get started" page STILL renders blank (frame_004096 captured). So the
    NP state event was a genuine missing handshake but NOT the page's gate.
  * recvfrom RULED OUT as the gate: it climbs into the millions during the
    30-40 s window while the WORKING menu is on screen and BEFORE the press, so
    it is RakNet (two "RakThread"s) idle-polling a UDP socket that never
    receives — background noise, not the blank-page cause. The earlier
    "15M recvfrom starves the UI" theory is now fully retired (the GIL is
    off-by-default AND recvfrom is unrelated).
  * HONEST STATUS: the blank post-menu page remains OPEN and is the documented
    multi-session-RE blocker (progress.md ~2196 "Getting a Minecraft menu pixel
    is multi-session RE"). cohtml renders the menu fine, so the engine works;
    the post-press view loads no content. Next real lead is NOT another HLE
    guess — it is to find what the Ore-UI page is waiting on (a view-creation
    call that never comes, an asset/route it can't load, or a JS-side gate),
    e.g. via `RAEEN_TRAP_MODULE_EXPORTS=cohtml` diffing which cohtml exports
    fire before vs after the press, or dumping the cohtml View's loaded URL.

- TEXTURE CACHE DEFAULTED OFF — Minecraft renders textured out of the box
  (2026-07-24; raeen-gpu 229/229, clippy `-D warnings` clean, fmt clean):
  * `diagnostics.rs`: `no_tex_cache: !on("RAEEN_TEX_CACHE")` — the persistent
    sampled-texture cache is now opt-IN. Verified by frame capture: with NO env
    vars set, Minecraft's panorama renders fully textured
    (scratchpad `default_2048.png`); previously it needed
    `RAEEN_NO_TEX_CACHE=1`.
  * HONEST: this is a MITIGATION, not a root cause. The cache is measurably
    guilty (cache ON: flat untextured, 5/12 sampled probes all-zero; cache OFF:
    fully textured, 0/12) but the MECHANISM is still unknown, and the two
    obvious theories are DISPROVED — the sample hash is whole-range (exact) for
    the failing 1 KiB texture, and `sampled_render_target` already runs BEFORE
    `decode_texture`, so the cache cannot short-circuit a live render-target
    bind. Recorded in the task with the remaining avenues. Restoring the perf
    win (build 969 -> 59 us) requires a capture with `RAEEN_TEX_CACHE=1` that is
    pixel-identical to the default.
  * CORRECTION to the earlier entry below: the claim that 15M `recvfrom` calls
    starve the UI via "the global CALL_LOCK + guest-GIL" is WEAKER than stated.
    The guest GIL is behind `RAEEN_SINGLE_THREAD_GUEST` (dispatch.rs:321-379)
    and is OFF by default, and `CALL_LOCK` is taken by `execute_linked` for the
    whole pipeline rather than per HLE call — so HLE calls are not globally
    serialized in a normal run. The 15M-call spin is real and worth bounding,
    but it is NOT established as the cause of the blank UI page.

- MINECRAFT RENDERING: texture cache proven guilty; panorama fixed; blank
  post-press page isolated as a SEPARATE bug (2026-07-24; frame-capture
  evidence, no commit):
  * PANORAMA FIXED (user-confirmed). Cause was the `image_get_resinfo` non-2D
    refusal ported this session: the panorama is a type-11 cube lowered to
    2DArray, and resinfo refused anything but plain 2D — which failed the WHOLE
    shader recompile and dropped those draws, not just the query.
  * TEXTURE CACHE IS A REAL BUG — PROVEN WITH PIXELS. `RAEEN_NO_TEX_CACHE=1` +
    `RAEEN_DUMP_FRAMES` + scripted cross: the dumped menu frame renders
    PERFECTLY — full textured panorama, logo, Steve's skin, green "Get started",
    version text. With the cache ON the same scene is flat untextured geometry.
    Probes: cache ON = 5/12 sampled textures all-zero (base=0x23140000);
    cache OFF = 0/12. This ANSWERS the open question
    `rendering-blockers-and-port-plan-2026-07-22.md` demanded be settled before
    further perf work ("the stage-D texture cache is UNVERIFIED — run the
    `RAEEN_NO_TEX_CACHE` A/B first; if it regresses, revert it"): it regresses.
    Do NOT just delete it (real perf win, build 969 -> 59 us) — fix the
    invalidation. Open sub-question recorded in the task: whether the guest
    bytes legitimately stay zero because the content is produced by a GPU pass
    we never read back (cache innocent, missing writeback) versus the sparse
    sample hash missing the change.
  * MRT RULED OUT — a new `note_active_color_slots` diagnostic
    (`draw_translate.rs`) reports any draw binding CB_COLOR slots above 0. It
    fired ZERO times across a full run, so Minecraft renders one target at a
    time and the large block of skipped CB_COLOR1-7 context registers is a red
    herring. This killed a ~large MRT implementation before it was written.
  * BLANK POST-PRESS PAGE IS SEPARATE AND STILL OPEN. `frame_004096` captured
    after the press WITH the cache disabled is still a flat light-grey page —
    so the cache is not its cause. No guest console output and no new asset
    VFS reads follow the press (only savedata writes), so the Ore-UI/cohtml
    page navigates and renders empty. Next lead: why the HTML page paints its
    background but no content.
  * TOOLING: frame dumps are PPM (`RAEEN_DUMP_FRAMES=<dir>`, power-of-two
    draw indices, 960x540 here); a PowerShell PPM->PNG converter lives in the
    session scratchpad for viewing them.

- MINECRAFT BUTTON-PRESS CRASH FIXED — `sceKernelVirtualQuery` under-reported
  direct mappings (2026-07-24; working tree, no commit; raeen-hle 388/388,
  raeen-gpu 227/227):
  * SYMPTOM: pressing X at the "Get started" menu did nothing and killed the
    title. Reported as an input bug; it was not one.
  * INPUT IS FINE — PROVEN. Reproduced deterministically with
    `RAEEN_RUNNER_CHILD=1 RAEEN_INPUT_SCRIPT="0:neutral;50000:cross;50400:neutral"`
    + `RAEEN_TRACE_PAD=1`: the runner applied `buttons=0x00004000`
    (SCE_PAD_BUTTON_CROSS) at t=50.003 s and the guest read `0x00004000`
    through `scePadReadState` **8 ms later**. Host -> IPC -> kernel -> guest is
    complete and correct. (NOTE: scripted input only runs when
    `RAEEN_RUNNER_CHILD` is set — a headless `--run-eboot` without it silently
    has no input thread at all.)
  * WHAT ACTUALLY HAPPENED: 65 ms after the guest read the press, Minecraft's
    embedded V8 (in `libcohtml.Prospero.prx`, the Gameface UI module; message
    written via `libc.prx`) printed `# Fatal error in , line 0 / # unreachable
    code` and the process exited **code 3**. Empty file + line 0 = a
    release-build V8 `UNREACHABLE()`. This was ALSO the cause of the earlier
    "pressed X and it crashed" report whose log ended mid-stream with no fault
    line and no Windows Error Reporting event.
  * ROOT CAUSE (found via the `RAEEN_TRACE_EINVAL`-gated HLE ring): the last
    calls before the fatal were `AllocateDirectMemory -> MapDirectMemory ->
    **VirtualQuery**` — V8 maps direct memory and immediately queries the range
    back to verify it. `hle_virtual_query` filled only start/end/protection/
    is_committed of the 72-byte `SceKernelVirtualQueryInfo`, leaving `offset`
    (0x10), `memory_type` (0x1C) and every kind bit (0x20) ZERO. So memory the
    guest had just mapped as DIRECT, type 12, phys 0x5bc10000 read back as
    "anonymous, type 0, offset 0". V8's invariant check failed -> UNREACHABLE.
    Every allocate/map SUCCEEDED; nothing was failing, the read-back just
    disagreed with the write.
  * FIX: `MemoryRegion` gained `kind: MappingKind` (Anonymous/Direct/Flexible/
    Stack/Pooled, with `query_flag_bit()` matching shadPS4's bit order),
    `direct_offset` and `direct_memory_type`; new
    `VirtualMemoryManager::record_mapping_of_kind` +
    `direct_allocation_type(phys)`. `hle_allocate_direct_memory` now records the
    allocation's `type` against its physical range and
    `hle_map_direct_memory` records the mapping as Direct carrying the physical
    offset and that type. `hle_virtual_query` writes offset at 0x10,
    memory_type at 0x1C, and ORs in the kind bit at 0x20. Layout pinned against
    shadPS4 `core/libraries/kernel/memory.h` `OrbisVirtualQueryInfo`.
  * MEASURED AFTER: same scripted run, two X presses — **zero** "unreachable
    code", process ran the full 80 s (exit 124 timeout, not 3), guest read every
    transition `0 -> 0x4000 -> 0 -> 0x4000 -> 0`, and GPU work kept flowing
    ACROSS the presses (draws 15,210 -> 45,458; flips 2,048 -> 4,096).
  * HONEST LIMIT: the title survives the press and keeps rendering, but no new
    guest console output followed it, so "advances to the next screen" is NOT
    demonstrated — only "no longer dies". The missing panorama is a SEPARATE,
    still-open bug (ruled out as a texture-format/tile-mode refusal: zero
    unsupported-swizzle diagnostics, zero format refusals, zero draw skips).

- SHARPEMU GPU/SHADER REFRESH 21f964a → 26c5029 (2026-07-24; working tree, no
  commit; raeen-gpu lib 227/227, kyty-graphics 450+1+4, clippy `-D warnings`
  clean on both by project convention, fmt clean):
  * METHOD: both GPU commits compared MECHANICALLY, not by inspection — the
    swizzle tables were decoded to `(xmask, ymask)` pairs on both sides (4
    tables × 5 bpp rows × 16 bits = 320 entries, **0 mismatches**) and both
    detile algorithms were re-simulated on the 8 vectors from SharpEmu's own
    `GnmTilingDetileTests.cs` (**8/8 byte-identical**). This is what turned two
    assumed gaps into "already covered" and found the real ones elsewhere.
  * THREE STALE CLAIMS CORRECTED. (a) `rendering-blockers-and-port-plan`'s
    "`tiling.rs` is CPU-only, 2 modes" — it has **4** (5/9/24/27) at 5 element
    sizes, equations bit-identical to SharpEmu. (b) Bpp coverage is **equal**,
    not broader (my own earlier claim): SharpEmu's tables always had 5 rows;
    "4bpp" in the commit title is its GPU kernel's scope. (c) the port ledger's
    "flat-shader render path not exercised end-to-end / renderer priority #1" —
    the FLAT decode landed in `c0f6303` and the 3D depth transport is covered.
  * INTEGRATED (#592): row-parallel CPU detile (rayon, ≥512×512-element
    threshold, mirroring `GnmTiling.cs:533-539`) — every output row is an
    independent destination slice, proven by a round-trip test that runs the
    parallel branch and the serial branch on the same surface; a
    non-power-of-two element-size guard (`bpp_log2_is_supported`) — callers
    derive bpp with `trailing_zeros`, so a 3-byte element was silently read as
    1 byte and a 32-byte element indexed **one past** the last table row, a
    latent panic, both now named refusals; and a rate-limited
    `(tile_mode, format)` warning on the refusal path so the modes titles
    actually bind become a MEASUREMENT (the refusal drops the draw and shows up
    only as the BLACK FRAME warning, naming nothing).
  * INTEGRATED (#587): `image_get_resinfo` no longer refuses non-2D. It
    hard-refused anything but plain 2D, and that failed the **whole shader
    recompile** — dropping the draw, not just the query — and caught **2D-array
    as well as 3D**, i.e. every cube T# (types 11/13 lower to 2DArray, which is
    exactly Minecraft's panorama). `OpImageQuerySizeLod`'s result width is fixed
    by the image's dim, so the query type now follows the descriptor (`%v3int`
    for 3D/2DArray, `%v2int` for 2D); only x/y are stored either way. Added
    `%v3int` to the type preamble. Test asserts all three descriptor types
    recompile, emit the right query type, and pass naga validation.
  * DELIBERATELY NOT PORTED, with reasons: the GPU compute detile pass (well
    specified, Raeen's deferred-batch machinery fits it, ~700-1000 lines — but
    it targets *texture-upload* CPU cost, which is NOT in Tier 4's measured
    top-8; per-flip readback and per-submit fence waits are); SharpEmu's rule
    that MIMG `DIM` **overrides** the descriptor (Raeen already decodes DIM and
    uses it to gate the third address VGPR while taking `Dim` from the T#
    nibble — descriptor-wins is defensible and tested, DIM-wins is unverified).
  * QUEUED, EACH GATED ON A MEASUREMENT: block-table modes 1/4/8 (SharpEmu's own
    comment concedes it is a *model*, not a transcribed AddrLib PATINFO table —
    port only if the new diagnostic shows a title using them); mip-chain base
    placement (Raeen always reads `t.base40()`, so any `last_level > 0` texture
    samples from the wrong offset); `VkImageType` from the type nibble rather
    than `depth > 1` (a type-10 T# with `depth()==0` gets SPIR-V `Dim3D` but a
    2D image — the mismatch class blamed for an ASTRO.BOT device loss); tiled 3D
    upload/UAV detile (today an honest named refusal, not a silent under-read);
    FLAT D16 opcodes 0x19/0x1b/0x20-0x25.
  * ADJACENT GAP FOUND, BIGGER THAN EITHER COMMIT: `texture_vk_format`
    (`draw_translate.rs:909-985`) has **no block-compressed format arms at all**
    (no BC1/BC5/BC7). Most retail PS5 textures are BC. Not a SharpEmu port — a
    Raeen gap deserving its own task.
  * DOCS: `THIRD_PARTY_NOTICES.md` (both commits attributed, GPL-2.0→GPL-2.0),
    `docs/reference-port-ledger.md` (new 2026-07-24 refresh section),
    `compat/reference-state.json` SharpEmu baseline 6db095e → 26c5029,
    `rendering-blockers-and-port-plan-2026-07-22.md` stale claim corrected
    in place.
  * NOT DONE: Minecraft A/B still outstanding and now covers these GPU changes
    too (the resinfo path affects 2DArray = its panorama). `#605`
    `sceAudioOutOutputs` not reviewed (out of scope for a GPU/shader pass).

- SUBNAUTICA UNBLOCKED: nested-directory `.prx` discovery + dependency-first
  module init (2026-07-24; working tree, no commit; raeen-hle 386/386,
  raeen-firmware 119+11+3+3, raeen-runtime 68+45+1, raeen-kernel 36+2 — 674
  tests, 0 failures):
  * MEASURED 8-TITLE BASELINE FIRST (build df38544, `cargo xtask compat run
    --tier all --timeout 180`, saved to scratchpad `baseline-df38544.json`).
    Minecraft 2048 flips / ASTRO.BOT 0 flips timeout / Until Dawn + Dragon Ball
    guest fault `read 0xa` / Avatar `stage=rendering` 1 flip then
    `libScePlayGoDialog` unimplemented import / GTA V fault after 2733 HLE calls
    / A Plague Tale host crash `0xC0000094` STATUS_INTEGER_DIVIDE_BY_ZERO /
    Subnautica exit at 1 s.
  * ROOT CAUSE 1 (search path, NOT a missing loader): Raeen already had the
    whole file-backed load -> link -> register -> LoadStartModule chain. It
    searched `<app>/` and `<app>/sce_module/` only (`DEPENDENCY_SUBDIRS`), but
    Unity ships modules in `Media/Modules` + `Media/Plugins`. So Subnautica's
    `Il2CppUserAssemblies.prx` — the ENTIRE game's C# compiled to native, 67 MB
    / 260 exports — was never placed and `sceKernelLoadStartModule` returned a
    code-less pseudo-handle. NEW `ModuleIndex` (bounded depth-4 recursive walk,
    prunes `sce_sys`/`savedata`/`streamingassets`, no symlink following,
    canonical case/extension-insensitive lookup) backs both the `DT_NEEDED`
    search and the plugin pre-placement scan. Explicit app-root -> `sce_module/`
    precedence preserved; the index only ADDS reach. `Media/Modules` classified
    eager, everything else lazy — byte-for-byte SharpEmu's `StartAtBoot` split
    (`SharpEmuRuntime.cs:636-645`). All 8 of Subnautica's `.prx` now load.
  * ROOT CAUSE 2 (GENERAL, affects every title shipping `.prx`): module
    initializers ran in breadth-first load order, i.e. the eboot's `DT_NEEDED`
    declaration order. Subnautica's order was Il2Cpp -> PS5Util -> **libc**,
    but Il2Cpp NEEDs both — so IL2CPP's `module_start` called into a libc whose
    own `module_start` had not run and died on a null function pointer
    (`guest fault at 0x0 (execute 0x0)`, libc.prx frames on the stack). NEW
    `topological_init_order` (post-order DFS over the NEEDED graph) makes
    dependencies initialize before dependents; order is now libc -> PS5Util ->
    Il2Cpp. Cycles are broken at the back-edge with a warning, never dropping a
    module; unshipped/HLE-covered NEEDEDs impose no constraint, so a graph with
    no real edges keeps the old order (no reordering risk for working titles).
  * ALSO: image-budget guard (recursive scan can pull in far more modules — an
    over-budget composition now names itself with per-module sizes instead of
    surfacing as an opaque arena `MapFailed`); `GUEST_IMAGE_REGION_BYTES`
    mirrored from `arena::IMAGE_SIZE` and pinned by a cross-crate test, since
    firmware cannot depend on runtime.
  * HLE: `sceUserServiceGetAgeLevel` (18) + the remaining accessibility getter
    family — ChatTranscription / PressAndHoldDelay / ZoomEnabled /
    ZoomFollowFocus (all 0), SharpEmu `UserServiceExports.cs:227-273` values.
    Registered as a FAMILY rather than one-per-fault: each miss otherwise costs
    a whole measure/build/re-run cycle. NID pinned in firmware
    (`0xc28369bbee3944b9`, cross-checked against shadPS4 `aerolib.inl`).
  * PRINTF FLOATS (real gap, was flooding A Plague Tale's log): `%f %F %e %E
    %g %G` implemented with C spelling (two-digit signed exponent, `%g`
    trailing-zero trimming, `nan`/`inf`/`INF`). `format_c` now takes a SECOND,
    independent iterator for floats, because SysV passes variadic floats in XMM
    and integers in GP — consuming a float from the GP sequence would
    desynchronize every later integer conversion. `printf`/`snprintf` feed it
    from `ctx.float_args` (XMM0-7); `vsnprintf` gained the `va_list`
    `fp_offset` cursor (`GuestVaListFloats`) the old code ignored. The module
    header's claim that "the dispatcher does not capture XMM" was STALE —
    `float_args` had been added earlier.
  * SUBNAUTICA TRAJECTORY (measured, each a separate release run): 1 s exit ->
    all modules load + Il2Cpp `module_start` faults null -> init order fixed,
    Il2Cpp init SUCCEEDS and the title prints its own Unity launcher banner
    (311 bytes of guest console: "Argument Count = 1 / LAUNCHER CONTROL ...") ->
    `sceUserServiceGetAgeLevel` -> `GetAccessibilityChatTranscription` ->
    now `scePadDeviceClassGetExtendedInformation`. Real game code is running.
  * NOT DONE / NEXT: (a) re-run the full 8-title sweep on this build — the
    init-order fix is general and may move other titles, but that is UNMEASURED
    and no claim is made; (b) Minecraft A/B is REQUIRED before trusting the
    printf change (the user was playing it, so it was not run); (c) the shared
    UE5 `read 0xa` fault (Until Dawn + Dragon Ball, identical signature, both
    after `sceKernelWaitEqueue` ETIMEDOUT) is the top remaining multi-title
    blocker; (d) A Plague Tale's host divide-by-zero is OUR bug, not the
    title's.
  * BUILD GOTCHA (cost several cycles): a running `raeen.exe` — including the
    user's own Shell — makes `cargo build` fail with `failed to remove file
    target\release\raeen.exe / Access is denied`, AND `cargo build ... | tail`
    reports tail's exit code (0), so the failure is invisible and the next run
    silently measures a STALE binary. Redirect to a file and capture `$?`.
    `tasklist` name-filtering missed a holder that a module scan
    (`Get-Process | %{ $_.Modules }`) found. Session workaround: a separate
    `CARGO_TARGET_DIR` (`../Raeen-target-dev`) so development never touches the
    binary the user is running.

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
- MINECRAFT ISOLATED-RUNNER FRAME PRESENTATION RESTORED
  (2026-07-24; working tree, release Shell run):
  * ROOT CAUSE: runner isolation moved retail execution and
    `AgcGpuSession::last_image` into a child process, but the Shell continued
    polling its own process-local GPU session. Offscreen dumps could render
    correctly while the real client area had no path to receive those pixels.
  * FIX: new `raeen-gpu::frame_ipc` creates a unique pagefile-backed Windows
    mapping per launch. The child publishes complete RGBA8 frames through a
    sequence-locked latest-frame slot; the Shell copies only when the sequence
    is stable before/after the copy and otherwise keeps its cached last complete
    frame. The child remains crash-isolated under the Job Object.
  * REGRESSION COVERAGE: shared mapping round-trip; malformed frame cannot
    replace the last complete frame; local splash epoch cannot alias the first
    remote epoch. `raeen-gpu` bridge tests 2/2, `raeen-gui` 157/157, focused
    presenter tests 7/7, clippy `-D warnings`, fmt, and release build are green.
    The full GPU integration run's parallel Vulkan validation setup aborted in
    `coverage_bisect`; that binary passes 7/7 when rerun serially.
  * MEASURED UI EVIDENCE:
    `scratch/minecraft-frame-ipc-single-writer.png` shows Minecraft's own
    loading spinner/progress bar in the Shell; the final run logged exactly one
    child connection, and
    `scratch/minecraft-frame-ipc-single-writer-title.png` shows the complete
    Minecraft logo, animated-panorama frame, Steve model, version text and
    "Get started" menu. This closes the black client-area transport regression,
    not Minecraft gameplay or M4/M5.

- MINECRAFT PANORAMA CUBEMAP LOWERING + COMPUTE WRITEBACK FIXED
  (2026-07-24; working tree, release Shell/profile runs):
  * ROOT CAUSE (rendering): RDNA's `V_CUBE*` sequence already converts a
    direction to `(s, t, face)` before `image_sample`. Raeen bound guest
    texture type 11 as a Vulkan cube, so Vulkan interpreted those values as a
    direction a second time and produced the measured radial panorama smear.
    Type 11 now lowers to a six-layer `2DArray` in both SPIR-V and Vulkan.
  * LIVE EVIDENCE: `scratch/minecraft-panorama-2darray-live.png` shows the
    complete sharp panorama, logo, Steve, buttons, and version text through the
    isolated Shell client. The shared presenter reported a measured 4 FPS at
    that menu; correctness is fixed, but the 60 FPS target remains open.
  * ROOT CAUSE (compute correctness/performance): the host discarded
    `ShaderStorageUsage` and read every compute V# back into guest memory,
    including `ReadOnly` and `Constant` inputs. Writeback now preserves the
    translated usage table and returns only `ReadWrite` buffers.
  * MEASURED PROFILE: before the usage fix the first 35 seconds reached about
    672 flips; the same release/profile run after it reached about 1,376 flips.
    The remaining slow phase is still compute submission (about 21-23 ms per
    dispatch, 94-96% worker busy), not texture decode. Command/resource
    lifetime batching is the next measured performance wall.
  * VERIFICATION: `kyty-graphics` 445/445 tests and the complete
    `raeen-gpu` unit + Vulkan integration suite are green; release Shell build
    is green. This does not claim Minecraft gameplay or M4/M5.

- MINECRAFT FOUR-PATH RELIABILITY SLICE (2026-07-24; working tree, two release
  Shell runs):
  * FINAL COMPOSITE: when a title's undecoded final copy leaves a visible
    intermediate and an empty linear RGBA/BGRA VideoOut buffer, the presenter
    now writes that intermediate into the real guest-visible scanout. The copy
    scales the configured internal resolution (measured 1440x810 at 0.75x) to
    the registered 1920x1080 display and preserves pitch/channel order. The
    measured rerun copied every subsequent frame into alternating
    `0x1f7d0000`/`0x20040000`; the prior `PRESENT ROUTING` warning stopped once
    content began. Exact-copy and scaled-copy regressions are covered.
  * PERSISTENT CACHES: translated SPIR-V is content/ABI-addressed under
    `shader_cache/spirv-v1`; malformed entries are validated and discarded.
    The Vulkan driver cache is keyed by vendor/device/driver and checkpointed
    after sparse pipeline generations so force-terminated child runners still
    retain it. On the measured warm run all five boot shaders were disk hits
    (`translated_ok=0`, `disk_hits=5`) and a 22,184-byte AMD pipeline blob was
    present.
  * SHARED PRESENT/INPUT: the Windows child/Shell mapping is protocol v3 with
    two 136-MiB RGBA8 slots, so the next child copy no longer overwrites the
    slot the Shell normally reads. Stable-sequence validation still rejects a
    lapped/torn copy. The same bidirectional mapping continued to report
    `runner applied controller state ... source="shell-ipc"`.
  * GLOBAL RESOURCES: an interrupted Bedrock runner left an exact zero-byte
    `resource_init_lock`; the new runner safely removes only that empty session
    marker before mounting save data. The live rerun logged one recovered
    marker; packaged resources and nonempty files are never altered.
  * VERIFICATION: `raeen-gpu` 216/216, `raeen-gui` focused suite plus new
    resource-lock test, GPU clippy `-D warnings`, fmt, and release build are
    green. Live evidence is `scratch/minecraft-scaled-scanout-live.png`.
    Presentation was measured at 4 FPS (later 2 FPS during loading), so the
    60-FPS/menu/gameplay acceptance remains open and M4/M5 are not claimed.

## 2026-07-25 — NID coverage tooling + dictionary fill

* NEW `raeen_firmware::inspect_module`: static half of `load_module` (SELF
  passthrough -> sprx parse -> dynlib decode), so diagnostics and the loader
  share one code path. `load_module` now calls it. raeen-firmware 119/119,
  xtask 5/5, clippy+fmt green.
* NEW `cargo xtask nids coverage [--eboot PATH] [--full]`: per-game NID
  coverage over eboot + on-disk NEEDED .prx chain, classified with the
  linker's own rules (HLE via live HleRegistry, LLE via shipped modules,
  else unresolved), render-path libs broken out. Local report:
  `artifacts/compat/nid-coverage.json` (gitignored).
* FIX (tool + doc): LLE export keying — PS5 `PT_SCE_MODULE_PARAM` carries
  NID-encoded strings, no ASCII name; module identity is the FILE name
  (matches `load_process`). Corrected stale `SprxModule::name` doc.
* MEASURED (9 images, 8 titles): union unresolved 2,779 -> 735 once LLE
  keyed right (e.g. Minecraft 556 -> 92). 7/8 titles: render-path imports
  100% symbol-resolved (registration, NOT working behavior). GTA V is the
  render-path holdout: 107 libSceAgc unresolved (all but 7 dictionary-named).
  Union anonymous: 9 NIDs total.
* DICTIONARY: ps5-payload-sdk headers (GPL-3.0, identifiers only, hash gate)
  +35,181 verified names; idc/ps4libdoc measured +0 (fully redundant);
  `hunt_nid_names` token attack (58.9M candidates) recovered
  `sceSaveDataPrepareForTransferring`; 8 NIDs remain anonymous (6 sceAgc*,
  1 sceAudioIn*, 1 sceVideoRecording*) — register_nid route. Catalog now
  185,088 entries; provenance in nid_names.rs + THIRD_PARTY_NOTICES.md;
  refs logged in docs/reference-port-ledger.md.
* NEXT: GTA V's 107 named AGC functions are the measured M2/M5 work queue;
  cross-game HLE gap is service libs (NpWebApi2, Http2, AvPlayer, Ampr).

## 2026-07-25 (cont.) — Missing-import resolution, tier A

* POLICY: no blanket stubs. Tiers: A = real host-backed impls; B = honest
  offline semantics (later); C = milestone work (GTA 107 AGC) / impossible
  (8 anonymous NIDs). Stubbing 735 to zero would poison the blocker signal.
* libc batch (delegated): 64/140 implemented in raeen-hle/src/libc.rs — CRT
  internals, stdio+format/scan engines, strto, time/locale, C++ ABI throws,
  mutex/syslock/atomics. Skips with reasons: 33 libm blocked on missing
  XMM0 float-return channel; 19 data objects; 5 unwinder; qsort (needs sync
  guest callback); 17 layout-risky Dinkumware internals.
* libkernel/posix batch (delegated): 16/23 — pwrite/ftruncate/rename/stat/
  QueryMemoryProtection/CheckedReleaseDirectMemory/IsStack/SetGPO/
  schedparam/solosched/getstack/mutexattr_setprotocol/sleep (+minimal
  raeen-kernel VFS pwrite/ftruncate). Skips: 5 AIO (no infra — do not fake),
  __progname/__stack_chk_guard (already resolved via firmware HLE data page).
* FIX coverage blind spot: `hle_data_page_export_names()` (raeen-firmware)
  exposes data-page exports; `nids coverage` models them; consistency test
  `hle_data_page_resolves_every_listed_export`.
* FIX 8 pre-existing clippy 1.97 lints (gate green again): stray duplicated
  #[test] + hex grouping (libkernel), >= y+1 (audio_out2), deprecated DCB
  fixture allow (libsce_agc), PI-approx test value (fmt), redundant must_use
  (save_data_dialog), complex type (user_service), manual is_multiple_of
  (hle lib.rs).
* MEASURED: union unresolved 735 -> 650; HLE resolved up in every title
  (Plague Tale 357->422). IN FLIGHT: XMM0 float-return channel + ~36 libm
  handlers (agent-0 resumed).
* raeen-hle 420, raeen-kernel 38, raeen-firmware 120, xtask 5 tests green;
  clippy+fmt green on touched crates.

## 2026-07-25/26 (cont.) — five-item program: fail-soft, UE5, swapchain, ordered side effects, doc

* ITEM 5 (doc): rendering-blockers-and-port-plan-2026-07-22.md corrected, not
  deleted — Tier 0 marked DONE (APR port 07-23), Tier 2 depth/BC claims
  CORRECTED-stale (depth_state_from_regs + BC1-7 arms exist), order-of-attack
  updated, currency notice added. Regression rules preserved (load-bearing).
* ITEM 1 (fail-soft): already in-tree from a prior uncommitted session —
  called-but-unresolved NIDs return 0 + resume by default with per-NID
  inventory (record_unresolved_nid_call); RAEEN_STRICT_NIDS=1 restores
  hard-fail; data reads stay hard failures. Activated on release rebuild.
* XMM0 float-return channel landed (agent): register_float marker + both
  dispatch paths (VEH apply_hle_result writes FltSave xmm0; direct gateway
  routes to a float bridge with movq xmm0,rax); 33 libm handlers registered
  with real implementations. hle 427->430, runtime 74+46, all gates green.
* ITEM 4 (ordered GPU side effects): design pass done (explore) — found two
  unreported bugs: the two DMA forms had OPPOSITE ordering (standard eager,
  AGC worker-only), timestamps double-written from two clocks. Steps 0+1
  landed: RAEEN_DEFER_GPU_SIDE_EFFECTS gate (default OFF) + fail-open under
  gate + dual-policy tests (SIDEFX_ENV_LOCK serializes). Step 2 landed
  (agent): standard IT_DMA_DATA executes in-stream (cp_op_it_dma_data in
  kyty-graphics run.rs, mirrors AGC arm) + eager duplicates gated;
  kyty-graphics 456, hle 430 green. Steps 3-5 (clock unify, events/EOP,
  flip-pending) staged; step 3 needs A/B (ASTRO timestamp-fence regression
  territory). NOTE: ~91 pre-existing clippy-1.97 lints in kyty-graphics
  (mostly recompile.rs) red at HEAD — separate cleanup pass owed.
* ITEM 2 (UE5) root-cause progress: Until Dawn = NOT a read-0xa; deterministic
  __stack_chk_fail after Open/Fstat/Getdents on /app0/deepfiles (empty dir
  returning 0x200 TWICE = overflow smell), then our hle_stack_chk_fail
  WRONGLY RETURNS 0 -> guest walks into UD2. Dragon Ball = crash-only (zero
  unresolved-NID calls): worker threads dereference a count (rax=2 -> 0x20)
  as a list pointer at module+0x241c820 (99-entry task-gather loop) after
  WaitEqueue timeouts. Two fixes delegated (stack_chk_fail non-return +
  getdents/fstat layout audit vs shadPS4; DB fault-region disassembly).
* ITEM 3 (swapchain): design pass in flight (explore) — present flow seams,
  WSI requirements, SharpEmu MAILBOX model, MC default-OFF policy.
* BASELINE: repeated environmental failures (other session shares
  target\debug\xtask.exe lock; cygwin fork exhaustion; pipe exit-code
  masking). Workaround: scratch/run-baseline-parts.py chunked per-game driver
  invoking the prebuilt binary directly, merging into latest.json on full
  success. Other session's runs left intact (never kill theirs).

## 2026-07-27 — Shell UI/UX: full pointer+pad+keyboard reach, Plugins UI
* shell-input: complete (uncommitted; raeen-gui 166/166, raeen-gpu
  present_plugin 22/22 green). Left-stick menu nav (StickNav hysteresis
  0.6/0.4, unit-tested), DualSense Options button (gilrs Start) -> per-game
  overlay, WASD/Space/Backspace keyboard alternates, F11 fullscreen toggle.
* shell-pointer: complete. Nav pills clickable (HomeResponse.clicked_pill),
  Control Center cards/options clickable + backdrop click dismisses
  (control_center::CcClick), pointing-hand hover cursors on all interactive
  rects (tiles, gear, pills, settings, per-game, CC). Verified on-device via
  PostMessage drive + screenshots (portrait 1080x1920 display: CC row sits
  at y~1830 — capture the window bottom, not the top 1080 rows).
* BUG FIXED (pre-existing): tick_animations closed the Control Center when
  drilling into a card's option list — cc_open target now covers
  NavMode::ControlCenterOption too (mod.rs).
* settings-plugins: complete. New Settings ▸ Plugins section (SECTION_*
  consts introduced; Plugins=6, System=7, Advanced=8) listing every present
  plugin with capability labels, source (built-in vs dll name), Active
  marker; Confirm/click toggles active + applies live; Rescan Plugins Folder
  (re-runs load_present_plugins_from) and Open Plugins Folder rows; load
  refusals surfaced in-UI. raeen-gpu: PresentPlugin::source_path,
  PluginInfo/list_info, per-scan load_failures store + AgcGpuSession
  wrappers. Verified live incl. real BYO raeen_example_plugin.dll.
* KNOWN-RED (pre-existing, not this session): raeen-gpu
  tests/shader_memory_phase2 guest_memory_pixel_shader_draws_green fails at
  HEAD+earlier-uncommitted shader_fetch/draw_translate edits (verified by
  stashing this session's gpu changes — still fails). Spawned follow-up task.
  Also: Application Control policy blocks cargo-clippy + some fresh test
  exes (kyty-graphics gcn_to_spirv) — environmental, os error 4551.

## 2026-07-27 (later) — GPU test unblocked; Game Folders UX; settings-backend audit
* shader_memory_phase2 FIXED: not a code regression — persistent shader cache
  (default ON, `shader_cache/` under test CWD) served the fixture PS from a
  prior run's disk entry, so `translated_ok` never incremented. Tests now
  disable the persistent cache (hermetic; passes repeatedly); stale
  crates/raeen-gpu/shader_cache removed. NOTE: user separately started the
  spawned fix-task chip — that worktree session is now redundant.
* game-folders UX: complete (raeen-gui 169/169). Settings ▸ Game Folders adds
  "Browse & Add Folder…" (rfd native picker, new workspace dep rfd 0.15) and
  "Rescan Games"; any folder add/remove auto-rescans. Shell::rescan_library
  swaps library/meta/ledgers/covers/key-art/nav rail in place
  (NavState::set_games_rail, tested); verified live (rescanned count=9).
* settings-backend audit: Advanced dumps + Frame Limit now REALLY apply on
  next launch — launcher::stage_runner_env stages explicit set/remove env for
  the isolated runner child from the per-game *effective* config (dev env
  vars recorded at startup still win; unit-tested). Previously env was bridged
  once at Shell startup only, and Off could never unset. Honest-hint pass:
  VSync = restart; Spatial Audio + DualSense Features labeled reserved (no
  consumer exists — raeen_audio mixer is stereo-only, InputManager never
  constructed); encrypted-SELF fault message no longer points at Settings ▸
  Key Provider (key_provider_path is stored but unconsumed; file KeyProvider
  is future work).

## 2026-07-27 (later still) — PS5-style Settings redesign; wallpapers; sound packs
* settings-redesign: complete (raeen-gui 173/173 + audio/core green). Settings
  rebuilt PS5-style: icon sidebar (9 new/reused painter glyphs — Monitor,
  Folder, Key, Palette, Puzzle, Wrench added to icons.rs), row cards with
  rounded focus fill + accent bar, right-aligned values, top sheen gradient,
  per-section hints. CLICK-ALIGNMENT FIXED STRUCTURALLY: every row/sidebar
  entry is an allocated egui widget so hit rect == painted rect (the old
  fixed-28px overlay grid drifted from the painted pitch). Verified live.
* wallpapers: complete. `wallpapers/` (user-supplied images) + Settings ▸
  Theme ▸ Wallpaper cycles them; overrides theme background, applies live,
  persists as general.wallpaper. settings::available_wallpapers tested.
* ui-sound-packs: complete. `sounds/<pack>/` with move/confirm/back/launch
  .wav (hound-decoded, mono→stereo, 5s cap); raeen-audio gains a UI ring
  mixed additively over guest PCM in the cpal callback (clamped, respects
  volume/enable, never touches guest submit diagnostics). Settings ▸ Audio ▸
  UI Sound Pack cycles with audible preview; cues wired to nav moves,
  confirm/back, launch, and all pointer paths. shell/sounds.rs tested
  (decode, widen, listing).
* new workspace crates: rfd 0.15 (folder picker), hound 3.5 (wav), opener
  0.7 (plugins-folder open — replaced manual explorer spawn). Candidates
  noted for later: notify (auto-rescan watcher).
* NOTE: user has parallel Codex sessions building in this repo — watch the
  target/ lock (cargo-parallel-build-deadlock memory) and expect screenshots
  to be occluded by their windows.

## 2026-07-27 (eve) — Crate adoption sweep: 17 crates wired, all verified
* Goal: wire every recommended crate for real. 662 tests green across
  gui/gpu/loader/firmware/audio/core. All integrations live:
  - memmap2: raeen_loader::mapped::MappedFile (map-with-read-fallback,
    tested) — eboot launch read + NEEDED dep loads (PUP already mapped).
  - rayon: parallel cover/key-art decode in the Shell (was already live in
    texture detiling).
  - bytemuck: SPIR-V disk-cache serialize/deserialize (pod_collect/cast_slice,
    LE-identical to old per-word loop).
  - rustc-hash: FxHashMap across the NID linker maps (~719k-reloc hot path).
  - walkdir: find_eboot now finds eboots nested to depth 4 (-app dir wins,
    else shallowest).
  - egui-notify: Shell toasts (save-failure TODO paid, rescan results,
    plugin rescan, update staged). GOTCHA: default TopRight anchor rendered
    OFF-SCREEN because the user's window (1920 wide, windowed) exceeds the
    1080-wide portrait display — anchored TopLeft. Verified visually.
  - notify: recursive watcher on the game folders, 800ms debounce after the
    last event -> auto-rescan + toast. Verified end-to-end (drop/remove a
    game dir on disk -> library + toast update, no input).
  - sysinfo: host CPU/cores/RAM/OS logged at startup (verified in log).
  - puffin + puffin_http: RAEEN_PROFILE=1 -> scopes on + viewer server :8585
    default port; new_frame in RaeenApp; scopes on execute_dcb_cp /
    publish_frame / shell_update. (puffin_egui skipped: no egui 0.31 build —
    puffin_http+puffin_viewer is the supported pairing.)
  - renderdoc: RAEEN_RENDERDOC_CAPTURE=N brackets N DCB executions in
    Start/EndFrameCapture (headless offscreen needs explicit brackets).
  - gpu-allocator: Allocator now lives in VulkanDevice (created at init,
    dropped before vkDestroyDevice); GDS arena is the first sub-allocated
    consumer (managed/raw fallback split). Remaining 7 raw allocate_memory
    sites are pooled by hand and migrate per-site later — full conversion is
    its own refactor.
  - crash-handler + minidumper: Shell hosts a minidump server (unix-socket
    in temp, thread for process lifetime); runner child attaches
    crash-handler via RAEEN_CRASH_SOCKET and requests dumps out-of-process
    into logs/crashes/*.dmp. VEH coexistence: HLE traps are handled
    first-chance and never reach the last-chance handler.
  - rubato: UI sound clips sinc-resampled to 48k once at load
    (output_delay-compensated, padded flush; tested), replacing per-play
    linear resampling for clips.
  - proptest: nav state machine bounded under arbitrary input sequences;
    loader parse_elf/parse_pkg_header never panic on arbitrary bytes.
  - insta: PM4 fixture DCB encodings pinned (m2 + cp scissor halves,
    snapshots checked in).
  - criterion: benches/nid_link.rs (nid_of/encode/resolve vs 4k-name
    corpus); compiles (cargo bench runnable; AppControl policy may block
    fresh bench exes — known env issue).
  - cargo-fuzz: fuzz/ scaffolding (parse_elf, parse_pkg_header targets +
    README), workspace-excluded, nightly tool-gated.
* NOTE: user's window is windowed 1920x1080 on a 1080x1920 portrait display
  — right ~840px of the Shell render off-screen. Worth surfacing to the
  user; also why Settings looked clipped in screenshots.

## 2026-07-27 (night) — Present-path spike: native wgpu upload + plugin ABI v2
* USER-PRIORITIZED graphics spike (explicit). 481 gpu+gui tests green.
* shell-present: complete. GameFrameView now uploads guest frames into a
  native wgpu texture (Rgba8UnormSrgb, registered via egui-wgpu
  register_native_texture, one queue.write_texture per published frame) —
  the per-frame egui::ColorImage::from_rgba_unmultiplied conversion copy +
  egui delta copy are GONE. ColorImage path retained as non-wgpu/non-RGBA8
  fallback. RenderState plumbed RaeenApp -> Shell::update -> overlay.
  MEASURED live (Minecraft menu, epoch 8192): egui_upload_us 369-671us
  (was: same upload PLUS a full-frame per-pixel CPU conversion). Visual
  correctness verified on the real Minecraft menu (colors exact).
* plugin-abi-v2: complete (raeen-gpu present_plugin::cabi; 24 ABI tests).
  `raeen_plugin_v2` entry (authoritative when exported; v1 fully supported),
  RaeenHostContextV2 (host_flags + opaque Vulkan dispatch context, zeroed
  until RAEEN_HOST_GPU_FRAMES), frame `kind` = CPU now / VULKAN later,
  RAEEN_CAP_GPU_FRAMES (Settings shows "GPU"), reserved GPU output
  (produced_kind/produced_image; claiming GPU output on a CPU host = named
  refusal). DynamicPlugin drives both vtables; loader negotiates v2-first.
  A v2 plugin written today needs NO changes when GPU frames arrive.
  LEGAL BOUNDARY unchanged and restated in plugins/README.md: vendor-neutral
  socket, proprietary implementations (DLSS/XeSS) are user-supplied
  out-of-tree binaries, never shipped/fetched/named as supported; FSR-class
  MIT code fills the same slot in-tree.
* plan: docs/superpowers/plans/2026-07-27-gpu-resident-present.md — phased
  path to full GPU residency: (1) VK_EXT_external_memory_host GPU-copy
  straight into the IPC slot, (2) cross-process image sharing
  (external_memory_win32 -> wgpu-hal import), (3) set RAEEN_HOST_GPU_FRAMES
  + GPU plugin passes, (4) PM4 motion-vector/depth extraction (the moat).
  Phases 3-4 gated on M4/M5 titles rendering — no queue jumping.

## 2026-07-27 (late) — Sync starvation fixed; Minecraft reaches IN-WORLD
* SPIN->PARK (commit eef31c1, fmt ba3c9ce): every contended guest mutex/rwlock
  wait was a `yield_now()` loop burning a full host core per blocked thread.
  New `PthreadMutexShared`/`PthreadRwlockShared` pair state with a host
  condvar; waiters park (bounded 10ms re-check), unlock + owner-death
  recovery notify. `pthread_once` -> 200us backoff. VideoOut's <=1ms terminal
  vblank spin deliberately LEFT (edge accuracy, not starvation).
  MEASURED: in-game CPU all-cores-busy -> 0.42 of 12 cores.
* MILESTONE EVIDENCE (Minecraft, user-owned retail PS5 title):
  - Interactive menu, input proven: D-pad moved focus Play->Store->Play,
    Cross opened the Worlds screen (screenshots in session scratchpad).
  - Save-data host map WORKS: VFS `/savedata0/ -> savedata\PPS<id>-app\
    BedrockWorld<...>`; the persisted world lists AND loads.
  - Actionable logs proven by outcome: the deadlock warning named the exact
    mutex/owner/waiter that led directly to the fix above.
  - IN-WORLD REACHED: rendered 3D world with HUD (hearts, hotbar, Emote
    prompt, trees/water) — was previously stalled at 4 FPS.
* REMAINING BLOCKER (top priority): a genuine deadlock, NOT spin starvation.
  In-world the MINECRAFT main thread sits >3s on mutex 0x1019a1d48c0 /
  0x1019a1d32e0 with frames frozen and only 0.42 cores busy. Needs the
  holder's HLE call path instrumented (RAEEN_TRACE_HLE + owner name).
* PM4 register triage: the "143x unknown context register" is 143 DISTINCT
  registers, one first-sighting log each (already deduped) — mostly per-MRT
  colour-buffer sub-registers (DCC/CMASK/FMASK/CLEAR_WORD, MRT1-7 blocks).
  Skipping compression metadata is safe/intentional; MRT1-7 + fast-clear are
  real but NOT Minecraft blockers. Quiet debt, not an active bug.
* CI: `.github/workflows/ci.yml` ALREADY runs fmt + clippy -D warnings +
  workspace tests on windows-latest. Caught a rustfmt violation locally that
  the AppControl-blocked environment would have missed.
* NOTE: the parallel Codex session committed most of this session's earlier
  work (37d0449 plugin ABI v2/GPU frames, d87765a crash reporting+sysinfo,
  b3e7277 sound packs+memmap, 6a13734 savedata/video sync).

## 2026-07-27 (final) — M4 MET; direct-sgprs fix takes Minecraft 4 -> 56 FPS
* SHADER FIX (commit d21e727): `ps: direct sgprs` was a STAGE ASYMMETRY, not
  missing functionality. VS never rejected direct user SGPRs; CS already
  exempted `next_gen`; PS rejected unconditionally — while the consuming
  push-constant path (shader_calc_binding_indices -> push_constant_size) is
  stage-agnostic. One guard + a test (next_gen allowed / legacy still
  rejected). 477 kyty-graphics tests green.
  MEASURED IN-GAME: Minecraft in-world **56 FPS** (baseline 4 FPS) with real
  block textures instead of flat colour. The two fixes compound: parking
  (eef31c1) freed the cores, direct-sgprs (d21e727) stopped skipping the
  draws.
* M4: **MET**. Acceptance artifact written: `docs/m4-acceptance-minecraft.md`
  (reproduce steps, the five observations, and the "logs are actionable"
  clause demonstrated BY OUTCOME — both blockers this session were located
  from log lines alone). Honest limitations recorded: not a perf claim, no
  soak test, rendering recognisable-not-correct, synthetic pad input, and
  explicitly NOT an M5 claim.
* Earlier in-world hang did NOT reproduce post-fix; task reframed from
  "fix deadlock" to "re-verify over a longer session before investing".

## 2026-07-27 (M5 evaluation, worktree) — M5 MET on the recorded evidence
* M5: **MET**. Acceptance artifact: `docs/m5-acceptance-minecraft.md`. No new
  runtime evidence was produced — the record evaluates the committed
  2026-07-27 iron run (M4 record + ledger + d21e727) against the M5 clauses:
  named 3D title (Minecraft Bedrock PS5), recognizable frames (in-world
  textured 3D terrain + HUD, 56 FPS), shader MVP for that title (Gen5
  analysis -> SPIR-V -> Vulkan; decisive fix d21e727 "ps: direct sgprs").
* Same-title reasoning recorded in the doc: the gate names no title and gates
  capability, not a per-title quota; GTA V (docs/gta5-blocker-analysis) cited
  as a supporting-but-not-gate-bearing second 3D data point.
* In-tree, retail-free evidence re-run in this worktree:
  `cargo test -p kyty-graphics` 477/477 green, including
  input_info_ps_direct_sgprs_allowed_on_next_gen_rejected_on_legacy and the
  full_chain_{vs,ps}_gcn_bytes_to_validated_spirv SPIR-V chain tests.
* Known issues documented per gate: recognizable-not-correct (skipped PM4
  context regs, MRT1-7/fast-clear unimplemented), MVP scoped to Minecraft's
  shaders, no soak/perf/playability claim, synthetic pad input, evidence not
  re-run live for this record.
## 2026-07-27 (shell) — Perf HUD overlay + F12/Create screenshot capture
* perf-hud: complete. Settings ▸ Advanced ▸ "Performance HUD (F3)" (row 8,
  named const ADVANCED_ROW_PERF_HUD) + F3 hotkey following the F11 pattern
  (both route through apply_setting_adjust, config bit stays in sync;
  persisted as general.perf_hud, serde-default off for old configs).
  GameFrameView gains FrameTimeStats: a 120-sample rolling window of
  per-frame ms derived from published-frame epochs (rebaselines on the
  REMOTE_EPOCH_BIT source flip and on counter resets; frames published
  between Shell repaints share elapsed time evenly). paint_perf_hud is
  painter+explicit-rects in the top-right corner slot (semi-transparent,
  supersedes — never stacks with — the plain FPS badge): epoch-derived FPS,
  avg/worst frame ms, flip-count FPS, upload ms, drain/fence/read/sRGB ms.
  No puffin dependency — everything reads the always-available PresentTiming
  counters. 5 FrameTimeStats tests.
* screenshot: complete. F12 anywhere + the pad Create button's rising edge
  in-session (Create IS the PS5 screenshot button; the press is still
  forwarded to the guest). Dumps the currently published guest frame (same
  sources/priority as GameFrameView::paint: remote IPC frame, else local
  session last_image) to screenshots/<sanitized-title-id>_<UTC
  YYYYMMDD-HHMMSS-mmm>.png via the existing image workspace dep (png
  feature already on). Refuses non-RGBA8/truncated frames with a reasoned
  error; no session or no published frame -> info toast (never captures the
  Shell UI itself). Success/failure toasts via the TopLeft-anchored
  egui-notify. screenshots/ gitignored (frames of user-owned titles).
  4 screenshot tests (UTC filename vs known epochs, hostile-id
  sanitization, PNG decode round-trip, refusals) + 1 config persistence
  test.
* Tests: raeen-gui 184 (was 174), raeen-core 11, raeen-gpu suite green,
  cargo fmt clean. clippy not run (AppControl, os error 4551) — cargo check
  clean. NOT live-verified (worktree; display occupied) — needs a
  main-session verify pass: F3 toggle + HUD legibility over a running
  title, F12/Create toast + PNG on disk, Settings row focus/click.
## 2026-07-27 (worktree agent) — ITEM 2 (UE5/Until Dawn): stack_chk_fail non-return + getdents/fstat layout fix

* FIX 1 — hle_stack_chk_fail never returns (raeen-hle/src/libc.rs): reports
  thread+name, guest ra, recent HLE calls, stack code-addr chain (shared
  helper guest_stack_code_addrs factored into lib.rs, reused by
  DebugRaiseException), releases dying thread's mutexes, then
  request_exit(STACK_CHK_FAIL_EXIT_CODE=0xa002_0006; pub const) with
  request_process_exit escalation fallback — dispatcher restores recovery
  ctx at the HLE boundary so control NEVER returns to the smashed frame.
  Tests: recording-scheduler unit test (libc.rs) + runtime acceptance
  stack_chk_fail_unwinds_the_guest_instead_of_returning (poison tail after
  the call; old stub surfaced POISON, i.e. the walk-into-UD2 that masked
  Until Dawn's real cause).
* FIX 2 — getdents packed-dirent rewrite (raeen-kernel filesystem):
  AUDITED vs shadPS4 (reference/shadps4 NormalDirectory + file_system.cpp).
  BUG (root cause of the 0x200-twice overflow smell): we returned ONE
  512-byte record per call with d_reclen=512; sizeof(Orbis dirent)=264, so
  any guest copying a record by d_reclen overflows a stack dirent by 248
  bytes -> canary smash. Real layout: packed records d_reclen=align4(8+
  namlen+1), records never cross 512-byte blocks, last record per block
  absorbs slack, whole blocks per call, 0 at EOF. Empty dir = ONE 0x200
  block (. + .. packed, reclen 12+500) then 0 — shadPS4 DOES prepend dots
  (IterateDirectory), our dots kept (also added to "/" root listing).
  Dir cursor is now the byte offset in position (rewinddir = lseek 0 works;
  SEEK_END = 512-aligned listing size, matching shadPS4 lseek/fstat).
* FIX 2b — hle_fstat directory support (raeen-hle/libkernel.rs): dir fds
  reported ORBIS_MODE_REGULAR size 0 blksize 512 (never S_IFDIR). Now
  S_IFDIR 0x41ff, st_size=packed listing size, st_blocks=8,
  st_blksize=0x8000 (shadPS4 NormalDirectory::fstat values). New
  VirtualFileSystem::is_directory(fd). CONFIRMED-CORRECT in audit: stat
  struct offsets (mode@8 size@72 blocks@80 blksize@88), path-based
  hle_stat dir values (65536/128/65536 = shadPS4 posix_stat), nbytes<512
  EINVAL, nonzero d_fileno, basep semantics (byte offset before call).
* Tests: raeen-kernel 43+2 (3 new: empty-dir repro, packed-block audit,
  multi-block boundary), raeen-hle 445 (fstat-dir + stack_chk unit; getdents
  test rewritten to walk packed records), raeen-runtime 77+1+47+1 (new
  acceptance), raeen-firmware 125+11+3+3 — all green; fmt green; clippy
  green on touched crates (kyty-graphics MSRV lint pre-existing, untouched).
## 2026-07-27 (worktree agent) — GTA V AGC wall Phase A: GetSize family + ACB builders
* libsce_agc Phase A batch: all 83 measured-missing libSceAgc NIDs
  (docs/gta5-blocker-analysis-2026-07-27.md) registered — 63 real
  (KytyPS5 agc.cpp ports, this-file writer-pinned GetSizes, mesa-cited
  architectural PM4 sizes), 11 honest-error (register_incomplete, loud log,
  never a guessed packet: Dcb/AcbAtomicMem, Dcb/AcbMemSemaphore, CbCondWrite,
  DcbSetIndexIndirectArgs, GetDefaultCxStateFlat, SetNop,
  GetGsOversubscription, SetAmmSemaphoreMemory, GetSemaphoreLabel).
* CbBranch UPGRADED from a wrong Jump-alias to the faithful KytyPS5 14-DWORD
  conditional chain (w1KFAHVqpaU binds GraphicsCbBranch upstream); paired
  BranchPatchSetCompareAddress distinguishes the 14dw chain from the 4dw jump.
* MEASURED: cargo xtask nids coverage re-run vs live registry — PPSA04264
  render-path unresolved 83 -> 0 (graphics block: 60/60 resolved); total
  unresolved 265 -> 88, all online/dialog/input (stub tier, doc step 3).
* Tests: raeen-hle 459/459 green (12 new Phase A tests incl. registry
  resolution sweep), kyty-graphics 477/477 green, fmt clean; clippy blocked
  only by pre-existing kyty-graphics is_multiple_of MSRV lint (untouched).
* Phase B (NOT started, next wall): PM4 compute-queue execution — ACBs reach
  hle_driver_submit_acb but only the graphics-queue CP path executes; then
  re-run GTA V to move the UD2 assert.

## 2026-07-27 (worktree agent) — checklist item 10: `cargo xtask baseline` (run + diff)
* baseline-run: complete in-tree; LIVE VALIDATION PENDING (needs real installed
  games — do not delete scratch/run-baseline-parts.py until one live
  `baseline run` matches its behavior). Native port of the python driver:
  per-game child process via the PREBUILT target/release/raeen.exe (xtask
  never builds the gui — that is the shared-target cargo deadlock the script
  existed to avoid; missing binary = clear error, binary older than HEAD
  commit = error unless --allow-stale), retry per game (--attempts, default
  2), per-game part JSON + merged.json under
  artifacts/compat/baseline-parts/<run-id>/, merge into
  artifacts/compat/latest.json ONLY on full success (partial run exits
  nonzero and never touches latest.json). Flags: --registry --output --exe
  --timeout(180) --tier --profile(max-fps) --attempts --parts-dir
  --allow-stale.
* baseline-diff: `cargo xtask baseline diff <old.json> [new.json=latest.json]
  [--strict]` — per-title stage changes (documented rank heuristic), exit-code
  changes, flip/fps deltas, unresolved-NID count deltas + newly-missing /
  newly-resolved NID lists (capped at 20 shown, exact counts), only-old/
  only-new titles, cross-machine warning; --strict exits nonzero on any
  regression (the AGC/stub-churn tripwire).
* Schema: Evidence gains OPTIONAL `unresolved_nids` (Option<Vec<{library,
  nid, function?}>>), harvested from the runtime's first-occurrence
  "UNRESOLVED NID CALLED" tracing lines (ANSI-stripped token scan; nid_0x…
  describe-fallbacks stored as anonymous). None = run predates harvesting
  (old reports round-trip byte-identically, schema_version stays 1;
  round-trip tested); Some(empty) = measured zero — diff refuses NID deltas
  when either side is None instead of faking "all resolved". `compat run`
  now harvests too (shared run_one); stage classification extracted to a
  tested pure fn (timeout > exit-failure > flips precedence preserved).
* Tests: xtask 25/25 green (18 new: classification 3, NID parsing 3, schema
  round-trip 2, merge/publish gating 3, staleness 1, diff 5, render cap 1);
  CLI smoke-tested headlessly (diff on synthetic reports incl. pre-harvest
  old side; run's missing-exe and stale-exe errors). fmt green; clippy
  `-p xtask --no-deps -- -D warnings` green (full `-p xtask` blocked only by
  the pre-existing kyty-graphics is_multiple_of MSRV lint, item 13; clippy
  itself NOT AppControl-blocked in this worktree).
* FOLLOW-UP for main checkout (worktree isolation blocked the edit):
  scratch/run-baseline-parts.py still needs its SUPERSEDED header pointing
  at `cargo xtask baseline run`; keep the script until a live run validates
  the port.
## 2026-07-27 (worktree agent) — checklist 13: kyty-graphics + raeen-gpu clippy debt paid
* kyty-graphics: 87 clippy-1.97 lint instances -> 0 under
  `--all-targets -- -D warnings`. Breakdown: 82 field_reassign_with_default
  (ALL in #[cfg(test)] fixtures of recompile.rs + analysis.rs — covered by
  two module-scoped #[allow]s with justification: fixtures assign direct,
  nested, and indexed fields, which a struct-literal rewrite cannot express);
  4 clone_on_copy fixed (push(sample/min.clone()) -> push(sample/min),
  ShaderInstruction is Copy); 1 incompatible_msrv fixed in
  examples/shader_probe.rs (is_multiple_of -> % 4). The lib-side
  analysis.rs:3195 MSRV lint was fixed on main (d030259) and merged in —
  not re-touched here.
* raeen-gpu: 8 lint instances -> 0. Fixed mechanically: never_loop in
  tests/external_memory_host.rs (first-device probe -> into_iter().next()),
  assertions_on_constants in frame_ipc.rs (-> const block, now a
  compile-time header-overflow guard), type_complexity in
  vulkan/instance.rs (PickedDevice type alias). Allowed with justification:
  3 dead_code (SLOT_ALIGNMENT const, imported_host_pointer_alignment
  field + accessor — all deliberately plumbed by ea6efd0 for GPU-resident
  present phase 1, consumer lands in phase 1 wiring) and 2 deprecated
  (pm4_snapshot pins the deprecated build_m2_draw_dcb bytes on purpose).
* Tests: kyty-graphics 477/477 (baseline preserved), raeen-gpu 308/308,
  fmt clean. Clippy ran locally (no AppControl block this session).
* Workspace gate NOT yet green — remaining debt is outside this task's
  two-crate scope: raeen-gui (type_complexity launcher.rs:61,
  doc_lazy_continuation sounds.rs:113, needless_range_loop sounds.rs:199 —
  these 3 fail the exact CI invocation `clippy --workspace -- -D warnings`)
  and raeen-input (too_many_arguments hid.rs:398, lib test target only).
## 2026-07-27 (worktree agent) — abort/exit noreturn fix + GTA V Ampr Tier-C batch (46 NIDs)
* abort/exit no longer return into (noreturn) call sites — the residual walk-
  into-garbage hazard d818df9 left open. hle_abort: actionable report
  (thread+name, guest RA, recent-HLE ring, stack code-addr chain, EOWNERDEAD
  mutex release — shared report_fatal_thread_diagnostics factored out of the
  stack-chk handler) then request_exit(ABORT_EXIT_CODE=0xa002_0106; SIGABRT
  low byte, bit 8 distinguishes deliberate abort from a canary smash), with
  request_process_exit escalation fallback. hle_exit(status):
  request_process_exit(status) + request_exit(status) — the guest's OWN
  status, orderly not fatal; defense-in-depth behind the VEH
  TERMINATING_FUNCTIONS intercept (which still handles libc/
  libSceLibcInternal exit before dispatch). _Assert comment corrected (its
  call sites have well-formed code after the call; still returns).
* Runtime acceptance: abort_unwinds_the_guest_instead_of_returning +
  exit_unwinds_the_guest_with_its_status_instead_of_returning (poison tail
  mov eax,0xBAD after the call must NOT execute; results ABORT_EXIT_CODE /
  0x2A). Unit tests pin request_exit/request_process_exit recording.
* libSceAmpr batch: all 46 measured-missing NIDs registered in
  libsce_ampr.rs from KytyPS5 src/libs/libAmpr.cpp semantics (behavioral
  port; THIRD_PARTY_NOTICES + reference-port-ledger updated).
  - 37 REAL: nop/marker family as inert SELF-SIZING records (type 4,
    [type][total_size], walker skips; KytyPS5 appends zeroed no-ops the
    same way) with KytyPS5 arg bounds (Nop 1..=16 dwords, NopWithData <=15,
    marker size = align4(hdr+strlen+1)); GetType (host-tracked flag word:
    0x10000 GS-valid / 0x20000 map-active) + GetBufferBaseAddress;
    WriteAddress_04_00 -> existing type-3 completion record;
    ReadFileGather/Scatter/GatherScatter + ResetGatherScatterState with
    host-tracked stream continuation (OrbisKernel::ampr_gather_scatter —
    file id sticks, dest/offset continue past each read) and the eager-read
    model — REAL data movement through guest memory, never faked; every
    MeasureCommandSize* returns exactly its paired writer's advance (the
    invariant a title sizing buffers by measure calls observes; sizes are
    Raeen's records, deliberately not console packet sizes).
  - 9 register_incomplete honest-parity: WaitOnAddress/WaitOnCounter (waits
    dropped — synchronous completion), WriteCounter (counters unmodeled),
    WriteAddressFromCounter/Pair/TimeCounter (complete by writing 0 —
    KytyPS5 does exactly this), MapBegin/MapDirectBegin/MapEnd (16KiB-
    granular validation + EPERM window state machine, no actual mapping).
* apr_complete_command_buffer walks the new type-4 skip records; corrupt
  total_size fails loudly (EINVAL). Ctor/reset/dtor/ResetGatherScatterState
  clear the new host state (reset keeps flags — KytyPS5 parity).
* MEASURED: local xtask nids coverage re-run — libSceAmpr unresolved
  46 -> 0; the 9 degraded entries appear in registered_but_not_implemented.
* Tests: raeen-hle 492 green (481 baseline + 2 abort/exit + 9 Ampr),
  raeen-runtime 77+49(+2 new)+1 green, raeen-kernel 43+2 green; fmt green;
  clippy clean on raeen-hle/raeen-kernel (mod-4/page-align checks written as
  masks to dodge the manual_is_multiple_of vs MSRV-1.85 trap); workspace
  clippy still blocked only by the pre-existing kyty-graphics
  is_multiple_of MSRV lint (checklist item 13, another agent's wave).
* NEXT: item 2D re-measure GTA V once 2B (compute-queue execution) merges;
  watch for titles depending on real AMPR counter/time values (currently 0).

## 2026-07-27 (hle-stubber agent, worktree) — ITEM 8: real kernel AIO infrastructure

* CLOSES the 2026-07-25 "Skips: 5 AIO (no infra — do not fake)" gap. The 5
  measured NIDs (phase1-nid-coverage.json: Until Dawn + Dragon Ball Sparking
  Zero, identical set): sceKernelAioSubmitReadCommands (0x1e05fbf80391239f),
  SubmitWriteCommands (0x5d0f02f32f9d7be1), WaitRequest (0x28e17fa096d056f7),
  PollRequests (0xa3b3b8cf78f02b3a), DeleteRequest (0xe5380c13a018b72e) —
  all REAL, plus 7 sibling spellings sharing the machinery
  (SubmitRead/WriteCommandsMultiple, WaitRequests, PollRequest,
  CancelRequest[s], DeleteRequests). InitializeParam/Impl were already real
  (2026-07-18) and stay in libkernel.rs.
* ENGINE (`crates/raeen-kernel/src/aio.rs`, new; `OrbisKernel.aio` field):
  host-threadpool AIO with Orbis submit/poll/wait/cancel/delete semantics.
  - 2 worker threads, spawned lazily on first submit (a kernel that never
    uses AIO owns no threads); shutdown on engine drop.
  - I/O runs through the SAME Arc<VirtualFileSystem> descriptor table as the
    sync read/pread/pwrite path — a synchronously-opened fd works in an
    async request, and async writes are visible to sync preads.
  - States SUBMITTED(1)/PROCESSING(2)/COMPLETED(3)/ABORTED(4); ids s32 >= 1
    (0 reserved as cancel's "no request" sentinel), wrap-safe.
  - cancel aborts only not-yet-started slots (returnValue ECANCELED,
    sign-extended); in-flight requests finish normally and the batch
    completes — matching Orbis "PROCESSING = could not cancel". Cancel
    after complete is a no-op. delete retires the final state into a
    1024-entry ring so a late poll of a deleted id still answers; delete
    notifies waiters (found + fixed a lost-wakeup on delete-while-waiting).
* GUEST-MEMORY DISCIPLINE: workers never touch guest memory. Write payloads
  are captured from the guest buffer at submit time on the guest thread;
  read completions are staged host-side and copied into the guest buffer +
  SceKernelAioResult through ctx.mem (the same GuestMemory layer the sync
  path uses) when the guest drains via wait/poll/cancel/delete. Result
  structs read SUBMITTED until the API first reports the terminal state.
* LAYOUTS re-derived from the public C declarations, cross-checked against
  shadPS4 src/core/libraries/kernel/aio.h (GPL-2.0): RWRequest 0x28
  (offset@0 s64, nbyte@8 s64, buf@0x10, result@0x18, fd@0x20 s32); Result
  0x10 (returnValue@0 s64, state@8 u32). No code ported (shadPS4's AIO is
  synchronous-inline; ours is genuinely async) — no THIRD_PARTY_NOTICES
  change needed.
* HLE surface: `crates/raeen-hle/src/kernel_aio.rs` (new module, registered
  in HleRegistry::new after kernel_equeue). EFAULT/EINVAL/ESRCH/ETIMEDOUT
  per null-pointer/bad-size/unknown-id/timeout; infinite waits sliced at
  50ms re-checking process_is_terminating so an AIO wait can never outlive
  its guest process; batch walks capped at 128 (SCE_KERNEL_AIO_MAX_REQUESTS
  class bound).
* Tests: raeen-kernel 53 (+10: submit/poll/wait/cancel/delete lifecycles,
  multi-request batch, cancel-race with deterministic fresh-engine spawn
  window, retire ring, unique nonzero ids, negative-SCE returnValue) + 2
  integration; raeen-hle 507 (+10: struct-layout round-trips, write-persists
  -through-shared-fd-table, poll-array, timeout, delete-before-wait still
  delivers staged read, cancel(0) sentinel, validation, registry). One
  UNRELATED pre-existing flake observed once under full parallel load:
  libsce_video_out consecutive_vblank_waits_land_one_period_apart (13.0ms <
  16.6ms period; passes solo and on rerun — host timer granularity).
  raeen-runtime 77+1+49+1 green. cargo fmt --all clean; clippy
  --all-targets -D warnings clean on raeen-kernel + raeen-hle.
* NEXT: re-run xtask nids coverage against installed titles to confirm the
  5 measured AIO imports flip resolved (needs local game installs); Until
  Dawn live re-test (checklist item 4) now has one fewer missing surface.
## 2026-07-27 — Ordered GPU side effects, steps 3-5 (gpu-pipeline agent, worktree)

* ITEM 5 steps 3-5 CODE-COMPLETE, both new gates default OFF (zero behavior
  change with no env vars — Minecraft M4/M5 state protected).
* STEP 3 (unified timestamp clock, `RAEEN_UNIFIED_GPU_CLOCK=1`): the two
  disagreeing RELEASE_MEM fence clocks (HLE eager = kernel session ns clock;
  worker cp_op_release_mem = process-local 1,2,3 counter — same address
  double-written from different domains) now share ONE authority under the
  gate: `raeen_gpu::gpu_clock::next_unified_gpu_timestamp()` (monotonic ns,
  strictly increasing, one AtomicU64). HLE `next_gpu_timestamp` delegates
  under the gate (kernel session clock untouched, tested); the session CPs
  get `CommandProcessor::set_timestamp_source` whose installed source
  DECLINES per call when the gate is off (legacy counter bit-identical).
  DO NOT default ON without live A/B — ASTRO timestamp-fence regression
  territory; the flip stays with the main session.
* STEPS 4-5 (events/EOP/flips in-stream, under RAEEN_DEFER_GPU_SIDE_EFFECTS):
  kyty-graphics CP now RECORDS completion side effects in PM4 stream order
  (`SideEffect`: EventWrite from IT_EVENT_WRITE — no longer consumed without
  effect; EopInterrupt from the AGC RELEASE_MEM interrupt byte incl.
  interrupt-only packets; Flip from R_FLIP — no longer consumed), drained via
  `take_side_effects` (survives reset; a suspended walk records NOTHING past
  its unmet wait — that is the step-5 ordering proof). raeen-gpu publishes
  them to the process-global `ordered_side_effects` queue IFF the defer gate
  is on (gate off would double-deliver the eager duplicates; publish site =
  both CP run sites in execute_dcb_cp_authorized). raeen-hle gained
  `apply_ordered_gpu_side_effects` (shared signal helpers with the eager
  path, so gate flips change WHEN not WHAT) draining at the observation
  points: submit_command_buffer entry, sceKernelWaitEqueue poll loop,
  VideoOut GetFlipStatus/IsFlipPending/GetVblankStatus/WaitVblank. Eager
  event/EOP/flip application at submit is now inside `if !defer`.
  `defer_gpu_side_effects` has ONE reader
  (raeen_gpu::ordered_side_effects; HLE delegates). SIDEFX_ENV_LOCK moved to
  the raeen-hle crate root (kernel_equeue + video_out tests share it — the
  queue is process-global, tests that touch it must serialize).
* Tests: kyty-graphics 479 lib (+4: stream-order record, eager-decoder
  parity negatives, flip-behind-unmet-wait, timestamp-source override) +1+4
  = 484 total; raeen-gpu 273 lib (+2 gpu_clock, +2 ordered_side_effects) +
  new tests/ordered_side_effects.rs session-level dual-policy test (state-
  only DCB through the real CP, no Vulkan needed); raeen-hle 502 lib (+5:
  eager-default events/EOP/flips, defer-gate defer+drain, unified-clock
  same-clock interleave, WaitEqueue drains worker events, GetFlipStatus
  drains worker flips); raeen-runtime 77+49+1+1 green; all raeen-gpu
  integration suites green; fmt green; clippy -D warnings green on
  kyty-graphics + raeen-gpu + raeen-hle (all targets).
* NEXT (main session, live A/B checklist): (1) baseline run gates OFF;
  (2) RAEEN_DEFER_GPU_SIDE_EFFECTS=1 on Minecraft — watch cross-queue waits,
  flip cadence, equeue-driven frame loops; (3) RAEEN_UNIFIED_GPU_CLOCK=1
  alone on ASTRO.BOT — watch the render-thread timestamp-fence park;
  (4) both gates together; only then discuss defaults.
## 2026-07-28 — Item 19: DualSense rumble passthrough (worktree agent)
* checklist item 19 (rumble half) CODE DONE — guest vibration now reaches the
  physical controller end-to-end. Haptics/adaptive triggers/lightbar remain
  open (scePadSetTriggerEffect is still validate-and-ack).
* Guest → host: `scePadSetVibration` implemented for real in
  raeen-hle/libsce_pad.rs (was a log-and-OK stub). 2-byte
  `ScePadVibrationParam { largeMotor, smallMotor }` + error semantics
  (invalid handle / NULL / unreadable param) cross-checked against shadPS4
  pad.cpp and SharpEmu PadExports (NID yFVnOdGxvZY). Lands on
  `OrbisKernel::set_pad_rumble` — the sequence bumps on EVERY call (even
  unchanged values) because a repeated call is the title's keep-alive
  refresh; testable without hardware by asserting the kernel channel.
* Transport: new `raeen_input::rumble` module — `RumbleState`,
  encode/decode of the one-u64 rumble word (seq<<16|large<<8|small, seq 0 =
  never set), and `RumbleRouter` (pure, Duration-injected state machine:
  settings gate, write dedup, immediate silence on session end, 5 s
  no-refresh safety auto-stop). Child → Shell crossing is a single AtomicU64
  at frame-IPC header offset 104 (reverse direction of the pad-input
  channel; deliberately NO protocol VERSION bump — the field lives in
  previously-zeroed padding, so mismatched peers degrade to "no rumble",
  not "no video"). Runner child forwards the word from its input thread;
  bridgeless `--run-eboot` runs drive their own NativeGamepads through the
  same router rules.
* Host output path chosen: direct DualSense HID output reports + direct
  XInputSetState — NOT gilrs ff (gilrs's Windows backend is XInput-only, so
  ff would cover Xbox pads at best and never the raw-HID DualSense the Shell
  reads natively; zero new deps either way). hid.rs (SharpEmu
  WindowsDualSenseReader port, extended): pure `build_output_report` — USB
  id 0x02/48 B; BT id 0x31/78 B with rolling seq nibble + CRC-32 (0xA2
  seed) without which the pad drops the frame; valid_flag0=0x03
  (compatible-vibration + haptics-select), lightbar/LED bytes untouched.
  Dedicated `dualsense-hid-writer` thread on a second handle to the same
  device path (never serializes against the blocking reader); transport
  detected from the first parsed input report; reconnect generation counter
  re-applies live rumble to a re-plugged pad. xinput.rs: active-slot atomic
  + `XInputSetState` (255→65535 exact 257× expansion). NativeGamepads::
  set_rumble fans out to both sinks (each no-ops when absent).
* Shell: `tick_rumble` each frame after push_pad_state — source = frame-IPC
  word (isolated runner) else in-process session kernel; RumbleRouter
  applies Settings ▸ Controllers ▸ DualSense Features live (ON routes, OFF
  drops), silences on quit/crash/session-None, dedupes hardware writes.
  Settings toggle UN-RESERVED (2026-07-27 audit debt): hint rewritten to
  the honest behavior; toggle already existed at (2,0) and persists via
  config.toml (new round-trip + old-config-defaults-ON test).
* Real-hardware note: shadPS4 passes SDL_RumbleGamepad duration -1
  (persist until changed) — i.e. real firmware persists an output report
  indefinitely. The 5 s auto-stop is deliberately stricter (no stuck motors
  after a guest hang/kill); every guest SetVibration call refreshes it.
* Tests (all green): raeen-input 27 (18 baseline + 6 rumble router/wire +
  3 HID output report/CRC), raeen-hle +2 (SetVibration channel + error
  paths), raeen-gpu +1 (IPC rumble word round-trip), raeen-core +1
  (dualsense_features persistence), raeen-kernel/raeen-gui suites green.
  fmt green; clippy clean on touched crates.
* LIVE VERIFY pending (needs the user's controller): DualSense USB, then
  BT, then an Xbox pad in a rumbling title; toggle OFF mid-rumble (stops);
  PS-hold quit mid-rumble (stops). See checklist item 19.
## 2026-07-28 (worktree agent) — checklist item 16: MRT1-7 + fast-clear register support
* CB REGISTER DECODE COMPLETE (kyty-graphics): every CB_COLOR{0-7}
  sub-register now decodes into a named RenderTarget field — VIEW, ATTRIB,
  DCC_CONTROL, CMASK(+SLICE), FMASK(+SLICE), CLEAR_WORD0/1, DCC_BASE (all
  stride-15 per-slot families), the Gen5 BASE/CMASK/FMASK/DCC_BASE_EXT
  high-byte blocks (0x390-0x3AF, stride 1), and the full CB_COLOR0_INFO
  field set (FMASK_COMPRESSION_DISABLE / COMPRESS_1FRAG_ONLY /
  CMASK_ADDR_TYPE). Offsets/fields verified against Kyty Pm4.h L601-719 +
  GraphicsRun.cpp L3522-3700; new hw_regs structs mirror Kyty
  HardwareContext.h L13-152. Compression metadata (DCC/CMASK/FMASK) is
  decoded but deliberately NOT emulated — one process-wide INFO note
  (note_compression_metadata_ignored) replaces the per-register "unknown
  context register" warnings for the whole CB block. CLEAR_WORD/VIEW are
  live feature state, not metadata.
* MRT1-7 ATTACH FOR REAL (raeen-gpu): DrawState.mrt carries slots 1-7 with
  per-slot format (CB_COLOR{n}_INFO), write mask (CB_TARGET_MASK nibble),
  and blend (CB_BLEND{n}_CONTROL via blend_state_for_slot). The offscreen
  pipeline declares N colour attachments + N blend states (pipeline key
  extended), begin_rendering attaches every extra, each extra seeds from
  the framebuffer map (LOAD) or CLEARs, and every attachment reads back to
  its own guest base. independentBlend is now enabled at device creation
  when supported; without it the pipeline degrades to identical attachment
  states (warn-once) per VUID-...-pAttachments-00605 — measured live: the
  very first 2-MRT iron run tripped that VUID. MRT draws are immediate-only
  (the deferred batch cannot file multi-target readbacks); a slot with an
  extent mismatch / unmapped format / untranslatable blend is dropped with
  a named warn; the old "attaches only slot 0 — dropped" warn is retired.
* FAST CLEAR (shadPS4 FilterDraw port, attributed in THIRD_PARTY_NOTICES):
  CB_COLOR_CONTROL.MODE 2 (EliminateFastClear) is consumed and applied as a
  real direct clear — the packed CLEAR_WORD0/1 splatted over the target's
  framebuffer entry (raw target-format bytes; no per-format unpack needed)
  + eviction of stale persistent images + PM4-ordered pre-flush; modes
  3/5/6 (resolve / FMASK / DCC decompress) are once-logged named skips.
  FCE with fast clear unarmed is a quiet no-op (shadPS4 parity).
* HONEST LIMIT (named in checklist): guest pixel shaders exporting mrt1+
  still fail translation — the recompiler handles MRT0 exports only
  (parse.rs exp target 0x00; recompile stores %outColor only). Real-title
  MRT output needs the exp mrt1-7 shader extension (declare %outColor1..7 +
  target-parameterized export handlers); the pipeline/attachment side is
  ready for it. That is the follow-up work item.
* Tests: kyty-graphics 484 (479 lib + 5 integration; was 480 — +4: pm4
  stride/offset pins, per-slot decode sweep over slots 0/3/7, BASE_EXT
  high-byte assembly, INFO compression flags). raeen-gpu lib 275 (was 269
  — +6: MRT draw-state translation, masked/mismatched slot refusal,
  fast-clear splat + refusal, session-level FCE end-to-end, CP fixture DCB
  with two bound targets -> one extra attachment). NEW
  tests/mrt_targets.rs: 2 iron Vulkan tests green on the real device —
  dual-output FS writes both attachments (readback pixel-exact, filed per
  guest base), per-attachment write mask R|A + LOAD seed verified.
  raeen-hle 496+1 green (one timing-flaky vblank test passed on isolated
  rerun; untouched by this diff). clippy -p kyty-graphics -p raeen-gpu
  --all-targets -D warnings green; cargo fmt --all --check green.
