# XPS5X session progress ledger

(Recreated 2026-07-16 — the previous ledger file was absent from the tree;
per-module authority is `docs/reference-port-ledger.md`.)

- Texture chain completed + two GPU blockers (2026-07-18, commit pending;
  106/106 xps5x-gpu lib + 225/225 kyty-graphics, clippy clean):
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
  (commit pending, 194/194 kyty-graphics + 86/86 xps5x-gpu tests).
  Resilience policy: unknown op/register = rate-limited warn + skip-by-length;
  hard errors only for truncated/non-type3 streams and refused draws.
  Ported: R_DRAW_INDEX (AGC + IT_DRAW_INDEX_2 raw form), R_{CX,SH,UC}_REGS_INDIRECT
  via new GuestMemory trait, R_DRAW_RESET → Reset, IT_INDEX_TYPE/BASE/BUFFER_SIZE +
  IT_SET_BASE(1) tracking, rate-limited sync/event/write-data skips.
  Indexed/indirect draws degrade to logged vertex-count-only draws
  (DrawSink::draw_index default; indirect count read from first args record).
  xps5x-gpu: guest_mem::IdentityGuestMemory (VirtualQuery-validated identity
  reads) wired into AgcGpuSession::execute_dcb_cp.
  Still todo: GraphicsRender (real index fetch, guest shader bind, multi-draw walk).

- Minecraft (PPSA17221) libkernel + libScePosix import closure: **0 missing**
  in both libraries (was 17 libkernel + 19 libScePosix), measured by re-running
  `--run-eboot`; 144 distinct missing NIDs remain, all in out-of-scope service
  libs (libSceNpWebApi2 21, libSceHttp2 14, libSceNet 13, ...). Implemented in
  xps5x-hle (commit pending; 247/247 hle, 19/19 kernel, 102/102 firmware,
  82/82 runtime tests): real VFS unlink/rmdir/rename/truncate (+ new VFS ops),
  REAL blocking POSIX semaphores (`posix_sem.rs`, address-keyed, condvar +
  termination-aware slices), scePthreadMutexTimedlock (deadline in lock_core),
  sceKernelMapDirectMemory2 (arg reshuffle), Add/DeleteWriteEvent, offline
  POSIX sockets (accept/listen/recv/send/select/... EWOULDBLOCK semantics,
  errno via __error slot), sched_get_priority_max/min (767/256), getrusage
  zero-fill, signal/Mlock/Sync/Chmod/Utimes accepted, `__progname` as a real
  data-page pointer export (xps5x-firmware). Title now boots 17 guest pthreads
  and dies downstream on its own `std::out_of_range` ("invalid string
  position") during phase-1 unwinding — next investigation target.

- ShaderMemory Phase 2 (guest shader fetch → GCN parse → SPIR-V → draw):
  **implemented + proven end-to-end in-tree** (commit pending; 196/196
  kyty-graphics, 87/87 + 2/2 + 2/2 xps5x-gpu tests, clippy clean).
  kyty-graphics CP: Gen5 shader-bind SH registers ported from Kyty's
  g_hw_sh_indirect_func — SPI_SHADER_PGM_LO/HI_PS+CHKSUM_PS+RSRC2_PS,
  PGM_LO/HI_ES+CHKSUM_GS+RSRC2_GS (gs-instead-of-vs), USER_DATA_GS slots —
  plus sh_regs context regs (SPI_SHADER_COL_FORMAT, SPI_PS_INPUT_ENA/ADDR/
  IN_CONTROL, SPI_PS_INPUT_CNTL_0..31, SPI_VS_OUT_CONFIG, DB_SHADER_CONTROL).
  These are exactly the registers Minecraft's DCBs write (proven from the
  prior iron log: unknown-reg warns 0xC8/0xC9/0x80/0x8A/0x8B/0x08, cx 0x191+).
  xps5x-gpu: shader_fetch.rs — bounded fetch (4 KiB chunks, 256 KiB cap,
  parser-driven growth on Truncated), next-gen→legacy generation fallback with
  both reasons named, positive+negative cache keyed (stage, addr, 16 head
  bytes) so a failing shader warns ONCE; XPS5X_DUMP_SHADERS forensic dumps
  (work even when translation fails). OffscreenDrawSink: untranslatable
  shader = skipped draw (counted, debug-logged), DCB continues; embedded
  fixture path intact (M2 gate untouched). Acceptance:
  tests/shader_memory_phase2.rs — DCB binds a real guest-memory PS via SH
  registers → CP → fetch → recompile → Vulkan draw → green pixel readback +
  frame PPM; garbage bind skips the draw, DCB survives.
  Also fixed: guest_mem read used copy_nonoverlapping; a wild-but-committed
  guest range can overlap the destination Vec (page-granular validation) —
  intermittent STATUS_STACK_BUFFER_OVERRUN under test; now ptr::copy.
  Title measurement (PPSA17221, 3×120 s runs, XPS5X_DUMP_SHADERS+FRAMES set):
  **0 shaders fetched, 0 draws — title dies ~10 s in, pre-graphics**, on the
  known std::out_of_range phase-1-unwinding wall above (first failing HLE
  call sceKernelGetdents → 0x8002000e). The GPU-side path is armed and proven;
  re-measure the moment the boot wall falls.
