# HLE out-buffer audit — eliminating the guest stack-canary smash class

**Date:** 2026-07-28
**Scope:** `crates/raeen-hle/src/**` out-parameter writes, plus the POSIX
`-1`/`errno` return convention per export.
**Motivating failures:** GTA V dies in `__stack_chk_fail` on guest thread 31;
Until Dawn dies the same way at ~6.7 s.
**Reference:** SharpEmu (GPL-2.0-or-later) out-buffer fix series, validated
against the live reference tip, not against a single commit — SharpEmu's
`6db095e` reverted `e13cb28` and `db4339f` (#650) later restored it, so a lone
commit is not evidence of current behavior.

---

## 1. The bug class

An HLE `*GetInfo` / `*GetState` / `*Query*` export receives a pointer to a
caller **local** far more often than to a heap object:

```c
SceFooInfo info;                  /* 0x20 bytes of stack frame        */
sceFooGetInfo(handle, &info);     /* HLE writes 0x40 -> frame smashed */
```

The compiler places the stack-protector canary immediately above that local.
Every byte the HLE writes past the real ABI struct size lands on an adjacent
local, a saved register, the canary, or the return address. The guest then dies
in `__stack_chk_fail`, or takes a wild jump some milliseconds later — with a
stack trace that points at the **victim**, never at the HLE function that did
the damage. That indirection is why this class survives so long: the crash is
never reported where the bug is.

The same class covers wrong **field width**: an `int *out_level` slot is 4
bytes, so storing a `u64` there clobbers the 4 bytes of whatever local the
compiler packed next to it.

### Rules

1. Write **exactly** the ABI struct size — never rounded up, never "generous".
2. Write **exactly** the ABI field width.
3. Never derive a write length from a guest register/argument that is not itself
   the ABI's declared buffer length.
4. Bulk-initialize (zero-fill) only objects that are **not** caller locals.
5. The same out parameter can legitimately have two shapes; pick by size, not
   by convenience.
6. Do not write "reserved" / secondary out slots — those bytes are usually
   adjacent caller locals.
7. Do not page-align or round up a size the guest may turn around and use as an
   `alloca`/VLA length.

Verified in the reference tip: `AudioOut2Exports.cs` carries the comment that
clearing 0x80 bytes "overwrote the caller's stack canary immediately following
the 0x40-byte parameter block" — the same shape as the two AudioOut2 bugs Raeen
fixed in `7c220d8`.

---

## 2. The guard: `crates/raeen-hle/src/out_buffer.rs`

SharpEmu detects a stack out-pointer with an address-range heuristic
(`0x00006FFFAC1FF000..0x00006FFFAC200000`, its own import stack window). Raeen
does not need a heuristic — it can answer the question **exactly**.

### How stack residency is determined

Two sources, in priority order:

1. **Registered thread stack bounds** —
   `raeen_kernel::OrbisKernel::guest_thread_stacks: DashMap<u64, (u64, u64)>`
   holds `[base, top)` per guest thread id:
   * `raeen_runtime::register_main_thread_stack` publishes the arena's stack
     region (`GuestArena::stack_region()`) for the main thread (id `1`) in both
     `execute_linked` and `execute_process`.
   * `raeen_runtime::thread` publishes each `scePthreadCreate` worker's freshly
     allocated stack as the worker starts, and **removes the registration
     before the stack memory is freed** — a stale entry would make a recycled
     heap allocation look like a caller frame.

   The key is the same id `GuestThreadScheduler::current_thread()` reports, so
   the lookup is exact. This matters most for secondary threads: their stacks
   are ordinary arena allocations, so nothing about the *address* distinguishes
   a worker's stack from a heap object.

2. **Bounded window above `caller_rsp`** — fallback when nothing is registered
   (unit tests, direct HLE calls, host embeddings). The caller's locals live
   *above* the callee-entry RSP, so the window is
   `[caller_rsp, caller_rsp + CALLER_FRAME_WINDOW)` with
   `CALLER_FRAME_WINDOW = 64 KiB`: larger than any single ordinary frame, far
   smaller than a thread stack. The fallback deliberately errs toward
   `NonStack`, because a false `Stack` verdict would truncate a legitimate
   heap-object initialization, while a false `NonStack` verdict only loses the
   extra diagnostic — the width and size clamps apply unconditionally.

`caller_rsp == 0` and no registration yields `OutRegion::Unknown`, which the
write helpers treat as `NonStack` (permissive) but report distinctly.

### API (inherent methods on `HleContext`)

| Call | Contract |
|------|----------|
| `caller_stack_region() -> Option<(u64, u64)>` | The calling thread's `[base, top)`, from either source above |
| `classify_out(ptr, len) -> OutRegion` | `Stack` / `NonStack` / `Unknown`; half-open overlap test, so a write that *runs into* the stack is a frame write |
| `write_out_struct(export, ptr, abi_size, payload)` | Writes at most `abi_size`. An oversized payload is clamped, counted, and logged **once per export** at `warn` |
| `zero_out_object(export, ptr, abi_size, minimal)` | Rule 4: bulk-clears `abi_size` only off-stack; on a caller frame clears just `minimal` bytes (`0` = skip) |
| `write_out_bounded(export, ptr, declared, data)` | For a caller-declared length (`getsockopt`'s in/out `optlen`): a polluted length can only ever shrink the write |
| `write_out_u8/u16/u32/i32/u64/i64(ptr, value)` | Exact declared width. `write_out_u32` taking a `u32` makes the narrowing visible at the call site instead of silently storing 8 bytes into a 4-byte slot |

`clamped_write_count()` / `clamped_write_count_for(export)` expose the clamp
counters, so a crash report can name the offending export and a unit test can
assert that a bad write was caught. 10 unit tests cover classification (both
sources, priority, boundaries), clamping with a poisoned canary past the ABI
struct, stack-vs-heap bulk init, exact scalar widths, null refusal, and bounded
writes.

**Honest limitation:** the guard cannot invent an ABI size. It enforces the size
a call site declares. A call site that declares the *wrong* size is still wrong
— which is why the audit below matters, and why every unverifiable size is
labelled as such rather than silently blessed.

---

## 3. Audit table

`function | site | declared size | written size | verdict`

Files audited: `libkernel.rs`, `libc.rs`, `libsce_agc.rs`,
`libsce_save_data.rs`, `libsce_rtc.rs`, `libsce_media.rs`, `libsce_pad.rs`,
`libsce_video_out.rs`, `libsce_np*.rs`, `libsce_ampr.rs`,
`libsce_system_service.rs`, `libsce_playgo.rs`, `libsce_font.rs`,
`libsce_net.rs`, `kernel_socket.rs`, `kernel_aio.rs`, `kernel_equeue.rs`,
`libsce_fiber.rs`, `libsce_voice.rs`, `libsce_json.rs`, `libsce_http.rs`,
`libsce_libc_internal.rs`, `pthread_*.rs`, `libsce_posix.rs`.
(`libsce_audio_out2.rs` was fixed in `7c220d8`; `libsce_user_service.rs` was
spot-checked correct.)

### Fixed

| Function | Site | Declared (ABI) | Was written | Verdict |
|---|---|---|---|---|
| `sceRtcFormatRFC3339` | `libsce_rtc.rs` `hle_format_rfc3339` | 32 B (`SCE_RTC_STRING_BUFSIZE`) | up to **36 B** | **FIXED** — `timeZoneMinutes` is a guest register formatted with `{:02}` (a *minimum* width): `i32::MAX` renders a `+35791394:07` suffix. Now rejects an offset outside ±14 h (`ERR_INVALID_ARG`) and a 5-digit year (`ERR_INVALID_VALUE`), and stores through `write_out_struct` with `abi_size = 32`. Test poisons the 8 bytes past the buffer and asserts they survive. |
| `sceKernelAioInitializeParam` | `libkernel.rs` `hle_aio_initialize_param` | unknown; `size` is a guest register | `size` B, up to **64 KiB** | **FIXED** — the only site in the file that turned a caller-supplied size into a *write length* rather than a gate. Now `zero_out_object(..., size, 0)`: full clear off-stack, nothing on a caller frame (zeroing a block the caller filled in is not an ABI guarantee). |
| `localeconv` | `libc.rs` `hle_localeconv` | `struct lconv` = 94 B of fields (10 `char*` + **14** `char`), padded to 96 | 96 B with the char block treated as **8** fields ending at 88, and `"."` written **at offset 88** | **FIXED** — offset 88 is `int_p_cs_precedes`: a guest read it back as `'.'` (46) instead of `CHAR_MAX`, and a guest writing any `int_*` field corrupted the `decimal_point` string the struct still pointed at. Strings moved past the struct; all 14 char fields set to `CHAR_MAX`. |
| `sceAgcDriverQueryResourceRegistrationUserMemoryRequirements` | `libsce_agc.rs` `hle_driver_query_resource_memory` | counts are `uint32_t` (the sibling `InitResourceRegistration` stores `ownerCount` into a `u32` and rejects wider) | counts taken as full 64-bit registers | **FIXED** — a stale register high half inflated `required`, the size the title feeds to its allocator (rule 7). Now rejects a count above `u32::MAX` and leaves the out slot untouched. |
| `sceKernelAprSubmitCommandBufferAndGetResult` | `libkernel.rs` | unverified: sibling `outSubmissionId` is `u32`, an SCE completion code is `int`, the SharpEmu port stores 8 | 8 B | **FIXED (shape-by-residency, rule 5)** — `zero_out_object(..., 8, 4)`: 8 zero bytes off-stack (behavior unchanged), 4 on a caller frame. The value is zero either way, so this cannot regress the off-stack case. |
| `_sceFiberInitializeImpl` | `libsce_fiber.rs` | caller's own `size_context` | unconditional 8-byte guard stamp | **FIXED** — `size_context` was read and stored but never validated, so a small/zero context with a non-null pointer put the stamp past the caller's buffer (a fiber context is usually a stack local). Now bounded by the declared size, with a `warn` when it does not fit. shadPS4 rejects sub-minimum contexts for the same reason. |
| `sceNpEntitlementAccessInitialize` | `libsce_np_entitlement.rs` | 0x20 assumed, no in-tree evidence | 0x20 B bulk clear of a **caller-owned input** block | **FIXED** — `zero_out_object(..., 0x20, 0)`: clears off-stack, skips on a caller frame. |
| `sceAmprCommandBufferConstructor` / `…Reset` (aux slots) | `libsce_ampr.rs` `write_cb` | `sizeof(SceAmprCommandBuffer)` unverified in-tree | 0x00..0x28 (40 B), of which **0x18/0x20 are never read back** | **FIXED (rule 6)** — the three load-bearing fields (`self`/`data`/`size`) are still always written; the two speculative aux slots go through `zero_out_object(..., 16, 0)`. Titles declare the buffer as a stack local and this runs once per command buffer per frame, so if the real struct ends at 0x18/0x20 those stores were hitting the frame every frame. |

### POSIX convention fixes (SharpEmu #461 + #567 class)

| Export | Site | Was | Verdict |
|---|---|---|---|
| `libScePosix::fstat` | `libkernel.rs` `hle_fstat`, registered raw | `-9` raw on a bad fd, `0x8002_000E` on a fault — **two conventions in one function**, and neither is POSIX | **FIXED** — `hle_fstat` now returns this module's internal `-errno` uniformly; `libScePosix::fstat` (and a new `libkernel::fstat`, matching how `read`/`write` are registered under both providers) goes through `file_result_posix` → `-1` + `errno`. A guest's `if (fstat(...) == -1)` could not previously fire. |
| `libkernel::sceKernelFstat` | same | `-9` instead of `SCE_KERNEL_ERROR_EBADF` | **FIXED** — `file_result_sce` → `0x8002_0009`. |
| `libScePosix::mprotect` | `libkernel.rs` `hle_posix_mprotect` | `-22` raw, `errno` never set | **FIXED** — `file_result_posix(FILE_EINVAL)` → `-1` + `errno = EINVAL`. |
| `gettimeofday`, `clock_gettime`, `usleep` (both providers), and every `-1` from `sce_to_posix` | `libsce_posix.rs` | `-1` with `errno` **never set** — the doc justified this with "Raeen has no guest errno yet" | **FIXED** — Raeen does have one (`libkernel::set_guest_errno` over the per-thread `__error()` slot). `sce_to_posix` now takes `ctx` and sets `errno` from an SCE code's low 16 bits, from a bare internal `-errno`, or `EINVAL` for anything unrecognisable: a wrong-but-defined errno still beats a stale one. |

### Verified OK (no change)

Widths and sizes checked against field offsets, the backing Rust types, or the
reference tip. Not exhaustive prose — the notable ones:

| Function | Why OK |
|---|---|
| `sceVideoOutGetOutputStatus` | 0x30 struct, zero-filled first, `refreshRate` stored as `u64` at 0x08 — **byte-identical** to the reference tip (`VideoOutExports.cs` writes `WriteUInt64LittleEndian(status[0x08..0x10], port.RefreshRate)`), and entirely inside the struct |
| `sceVideoOutInitializeOutputOptions` | 0x40 == reference tip's `VideoOutOutputOptionsSize` (verified in the live tree, not a lone commit) |
| `sceVideoOutGetFlipStatus` / `GetResolutionStatus` | 64 / 16 B match the public `SceVideoOutFlipStatus` / `…ResolutionStatus` layouts |
| `getsockopt` | Length is the ABI's own in/out `*optlen`, clamped to 128 |
| `select` | `set_bytes = howmany(nfds, NFDBITS) * sizeof(fd_mask)`, capped at the real 128-byte `fd_set` — BSD's own copy-out length, not a rounding |
| `sceKernelVirtualQuery` (72) / `GetModuleInfoForUnwind` (304) / `DirectMemoryQuery` (0x14) | Caller size used as a **gate**, then an exact constant written |
| `write_orbis_stat` / `hle_fstat` payload | 120 == FreeBSD-11 `struct stat` |
| `gettimeofday` / `clock_gettime` / `clock_getres` / `nanosleep` rem (16), `getrusage` (144), `sigprocmask` oset (16), `uuid_create` (16) | Exact `sizeof` |
| `struct tm` family (`write_tm`/`read_tm`, `gmtime_s`, `mktime`) | 9 × `int` = 36 B; **no** 8-byte-per-field store anywhere. `gmtime`/`localtime`'s 56-byte form is its own allocation, where `long tm_gmtoff` / `char *tm_zone` genuinely are 8 B |
| All `SceRtcDateTime` writers | Exactly u16×6 + u32 = 16 B |
| `scePadReadState` (120), `scePadGetControllerInformation` (0x1C) | Field offsets tile the struct exactly |
| `sceSaveData*` family (`GetMountInfo` 48, Mount result 0x40 with `mountStatus` as `u32` at 0x1c, `DirNameSearch` counts as `u32`) | Exact |
| `sceKernelAioResult` (16), `SceKernelEvent` (32), equeue counts as `u32` | Exact; delivered count bounded by the caller's `num` |
| `pthread_attr` / `pthread_thread` / `pthread_sync` getters | `detach_state`/`prio`/`policy` are `i32` → 4 B; stack size/guard/address are `size_t`/`void**` → 8 B; mutex/rwlock handles are pointer-sized |
| `sceKernelGetCompiledSdkVersion`, `LoadStartModule` `pRes`, `BatchMap` `numEntriesOut`, `QueryMemoryProtection` `protOut`, `set_guest_errno`, `playgo` `outEntries` | 4 B into genuine `int*`/`uint32_t*` slots |
| `libSceLibcInternal` `mspace_malloc_stats` / `heap_get_trace_info` | Skips the caller-filled header; validates the caller's declared `size == 32` first |
| `scePthreadGetaffinity` | 8 B — `SceKernelCpumask` is the 64-bit type in both shadPS4 and Kyty |

### Open / unverifiable — deliberately NOT changed

Each of these is a size or width that **no in-tree evidence establishes**.
Changing them on a guess risks regressing the proven Minecraft M4/M5 run, so
they are recorded rather than "fixed":

| Function | Concern | Why not changed |
|---|---|---|
| `sceAjmBatchInitialize` (`libsce_media.rs`) | Writes a fixed 40 B descriptor; the publicly modelled shape is 3 slots (24 B). Two call sites depend on the 40-B assumption (`try_append_batch_job` writes `info + 24`) | On Minecraft's proven codec path. Highest-value item to confirm against a real header; a wrong shrink breaks a working title |
| `write_decode_stream_result` (`libsce_media.rs`) | Always writes the maximum 32-B sideband; the real length follows the job's sideband flags, which the handler never sees | Same proven path; needs the job payload parsed before the length can be derived honestly |
| `clear_ajm_batch_error` | Clears 24 B of what is probably a 32-B struct (under-write, so no smash) | Leaves `job_ra` stale; harmless to the caller's frame |
| `scePadDeviceClassGetExtendedInformation` | Blanket `[0u8; 0x20]` with zero field writes; the union tail is 8 B in some revisions (total 0x10) | Would be a 16-byte frame smash at pad init if 0x10 is right — flagged as the cheapest high-value verification remaining |
| `sceAgcCreateInterpolantMapping` | Writes a fixed 32 slots (256 B) regardless of `output_count` | 32 `SPI_PS_INPUT_CNTL` registers is the hardware's fixed count, so 32 slots is most likely correct; on the proven shader path |
| `sceAgcGetDataPacketPayloadRange` | 16 B `{base, size}`; if `size` is `uint32_t` the real struct is 12 B | Reference-consistent; unverified |
| `sceAgcCreatePrimState` | 24 B into `ucRegisters` (3 entries); `hle_update_prim_state` reads `uc+20`, so the 3-entry layout is internally consistent | Self-consistent; confirm `SceAgcPrimState` once |
| `sceNpTrophy*Icon` `size` outs | 8 B `size_t*` per the KytyPS5 prototype | Unverified |
| `sceNpUniversalDataSystemCreateHandle` | Falls back to writing through `args[1]`, which is caller-clobbered garbage if the real export takes one argument | Latent (only when `out0` is 0/unwritable); mirrors the hazard `libkernel`'s `getdents`/`basep` comment already documents |
| `sce::Json::String::c_str` | Writes 8 B into the caller's object through a slot that is not an out parameter | Deliberate hedge for titles that inlined `data()`; not an overrun |
| AGC `*GetSize` under-reports (`CbSetShRegisterRangeDirectGetSize` returns DWORDs where siblings return bytes; `DcbDrawIndexAuto`/`DrawIndex`/`SetIndexBuffer` report fewer bytes than their emitters write) | Under-reservation, not an over-write | `alloc_command_dwords` bound-checks and *fails* rather than writing past the reservation, so this cannot smash memory. Real bug, different class — the visible symptom is a frame that stops emitting partway |
| `sceVideoOutGetVblankStatus` (32 of 40 B), `libSceVoice` port info (28 of 32 B), `sceAppContentGetMountPoint` (7 of 16 B) | Under-writes | Leave stale bytes for the guest to read; no frame damage |

### Missing POSIX exports (noted, not added)

No registration under any spelling: `lstat`/`sceKernelLstat`, `readv`,
`writev`, `preadv`, `pwritev`, POSIX `fsync`, POSIX `mkdir`, POSIX `truncate`,
POSIX `mmap`, `dup`/`dup2`/`access`/`openat`/`fstatat`, and the underscore
forms `_lseek`/`_fstat`/`_stat`. Provider asymmetries remain:
`unlink`/`rmdir` have no `libScePosix` registration, `getdents` no `libkernel`
one, `getdirentries` no `libScePosix` one.

`open`'s `EACCES` mapping is **correct** (`ErrorKind::PermissionDenied` →
`FILE_EACCES` → `-1` + `errno = 13`) but is not reachable for the obvious case:
in `raeen-kernel`'s VFS a *writable* open of an existing file goes through
`std::fs::read`, which succeeds on a read-only host file, and the descriptor is
then marked writable. So `open(path, O_WRONLY)` on a read-only host file returns
a valid fd and the failure surfaces later at flush-on-close, far from the call
the guest can attribute it to. Left as a recorded VFS gap.

---

## 4. What this does and does not claim

* The guard **prevents** the class where a call site declares the right ABI size
  and a payload/bulk-clear exceeds it, and it prevents any bulk initialization
  of a caller frame.
* Eight concrete over-writes and four convention defects are fixed with tests.
* This is **not** a claim that GTA V or Until Dawn now boot further. Neither was
  re-run as part of this work; the fixes are justified by ABI evidence and unit
  tests, not by a measured title run. The next step for those titles is to run
  them and read `clamped_write_count()` plus the once-per-export `warn` lines,
  which now name any remaining offender directly instead of leaving a
  `__stack_chk_fail` with no attribution.
