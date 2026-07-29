# Blasphemous II (PPSA13580) — the blockers after the dlsym fix

**Date:** 2026-07-28 · **Branch:** `fix-blasphemous-blockers` · **Title:** Blasphemous II,
PPSA13580, Unity/IL2CPP, user-owned retail copy.

Context: `sceKernelDlsym(handle=0, ...)` was fixed and confirmed on hardware, but the
frame path still reported `reached=nothing` — the title never opens VideoOut, so it is
blocked earlier. This document records the three blockers investigated next, what was
actually measured, and what was deliberately **not** done.

Everything below is static measurement (`cargo xtask nids coverage` against the
installed eboot) plus in-tree tests. No retail title was executed for this work.

---

## Measured effect (static, whole-title import table)

`cargo xtask nids coverage --eboot <PPSA13580-app>/eboot.bin --full`, same command
before and after:

| | before | after |
|---|---|---|
| unique imports | 3470 | 3470 |
| resolved via HLE | 825 | **852** |
| resolved via LLE (title's own modules) | 2496 | 2496 |
| **unresolved** | **149** | **122** |

Per-provider deltas (only providers that changed):

| provider | before | after |
|---|---|---|
| `libSceAudioIn` | 5 | **0** |
| `libSceVrSetupDialog` | 6 | **0** |
| `libSceErrorDialog` | 4 | **0** |
| `libSceMsgDialog.native` | 3 | **0** |
| `libScePosix` | 9 | **2** |
| `libkernel` | 2 | **0** |

27 imports that resolved to nothing now resolve. This is a *link* claim, not a boot
claim — see "Not verified" at the end.

---

## 1. `.native` suffix defeated module → HLE-library matching

### The rule, and where it comes from

`.native` is a **spelling of the same library**, and the module identity is the bare
name. Established from **KytyPS5** (`reference/kytyps5`, MIT © InoriRus / Nmzik) — the
only reference with a PS5 `.native` model at all. Its `LIB_VERSION(library, lv, module,
…)` macro (`src/libs/libs.h:24`) declares a library and the module it belongs to
separately, and every `.native` library there names the **bare** module:

* `src/libs/libDialog.cpp:87` — `LIB_VERSION("SaveDataDialog.native", 1, "SaveDataDialog", 1, 1)`
* `src/libs/libDialog.cpp:108` — `LIB_VERSION("MsgDialog.native", 1, "MsgDialog", 1, 1)`
* `src/libs/dialog.cpp:497` — `LIB_NAME("MsgDialog.native", "MsgDialog")`

Both spellings there are also backed by the **same** C++ functions
(`Dialog::SaveDataDialog::*` serves `"SaveDataDialog"` and `"SaveDataDialog.native"`
alike; the `.native` set is a *superset* of NIDs, not a different implementation).

shadPS4 has no `.native` spelling anywhere — it is a PS4 emulator and the suffix is a
PS5-era thing — so it could not settle the question either way.

The rule was deliberately **not** generalized. Kyty aliases `AudioOut2 → AudioOut` and
`LibcInternalExt → LibcInternal` per-library because those are genuinely different
libraries sharing a module. `.native` is the only pure spelling variant, so it is the
only suffix collapsed.

### What was actually broken

The rule already existed — in **one** of the two places that needed it. The crate had
three near-copies of "canonicalize a module name" and they had drifted:

| site | stripped `.prx`/`.sprx` | stripped `.native` |
|---|---|---|
| `dynlib/nid.rs::canonical_provider_name` (NID → HLE lookup) | yes | **yes** |
| `registry.rs::canonical_module_name` (module policy, LLE export table, loader visit-set) | yes | **no** |
| `lib.rs` NEEDED-coverage diagnostic | yes (raw stem) | **no** |

Consequences:

* A `.native` import reached HLE but looked up **module policy and LLE exports under a
  different key** than the loader registered them with. Any title shipping a bare-named
  module while importing the `.native` provider silently missed real exports.
* All four measured warnings were **false**: `libSceAjm` (in `libsce_media.rs`),
  `libSceAvPlayer` (same), `libSceMsgDialog` (`libsce_common_dialog.rs`), and
  `libSceSaveDataDialog` (`libsce_save_data_dialog.rs`) are all implemented in-tree.
  The loader was comparing a `DT_NEEDED` *file name* against bare HLE *library* names.

### Fix

`registry::canonical_module_name` is now the single source of the rule (`.prx`/`.sprx`,
then `.native`/`_native`), `dynlib::nid::canonical_provider_name` delegates to it, and
the loader's NEEDED diagnostic canonicalizes both sides through it. `raeen-hle` keeps a
private copy (it cannot depend on `raeen-firmware`) with a lockstep note.

### Aliased vs. declined

**Aliased (all `.native` variants, by the rule above):** the collapse is a property of
the suffix, so every library gets it — including the five Blasphemous II names
(`libSceAjm.native`, `libSceAvPlayer.native`, `libSceMsgDialog.native`,
`libSceSaveData.native`, `libSceSaveDataDialog.native`).

**Declined:** no per-library exception was carved out, because nothing in either
reference gives a `.native` variant a distinct ABI. The risk of a blanket collapse would
be a title naming *both* spellings of one library in the same `DT_NEEDED` list, where
the loader's visit-set would then load only one. Checked across every measured title
(Blasphemous II, Minecraft, Until Dawn, DBSZ, Subnautica, A Plague Tale, Avatar,
ASTRO.BOT, GTA V): **no collisions** — no title names both spellings of any library. If
one ever does, the visit-set is the place that needs the exception, not the name rule.

Also declined: aliasing other suffix families (`2`, `Ext`, `Hq`). Those are real
separate libraries in Kyty's model and must stay separate.

---

## 2. `sceKernelMkdir('././')` failed as "path is not mounted"

Two independent defects hid behind one log line.

### 2a. Path normalization

VFS mount matching is a literal string prefix compare, so a spelling the guest is
entitled to use but that does not *look* like `/<mount>/...` matched nothing. `'././'`
is a legal spelling of the current directory and is neither absolute nor
mount-prefixed.

`VirtualFileSystem::resolve_path` now normalizes first
(`raeen-kernel/src/filesystem/mod.rs::normalize_guest_path`):

1. `.` and empty components drop — which also collapses the `//` doubled slashes
   shadPS4 corrects for the same reason (`reference/shadps4`,
   `src/core/file_sys/fs.cpp:46`).
2. A **relative** path is anchored at `/app0`, the app root. Orbis titles run with their
   app root as the working directory (KytyPS5 loads the executable as `/app0/eboot.bin`
   and mounts the app directory at `/app0` and `/hostapp` — `src/main.cpp:138`,
   `src/emulator.cpp:201`). This is the *sandboxed* form of KytyPS5's behavior: its
   `MountPoints::GetRealFilename` (`src/kernel/fileSystem.cpp:226-245`) returns an
   unmatched guest path **verbatim**, which on the host resolves against the emulator's
   own working directory. That is the escape `combine_within_mount` exists to close and
   was deliberately not copied.
3. Trailing slashes disappear with the empty components.

`..` is deliberately **left in place**. Resolving it before mount matching would change
which mount a path matches; it belongs to `combine_within_mount`, which pops it with a
clamp at the mount root. The existing sandbox guards are untouched and re-asserted:
`../escape.bin`, `./../../escape.bin` and `../../../escape.bin` all clamp *into* the app
root, `./C:/Windows/...` is still refused as drive-qualified, `/dev/./null` still
resolves to nothing (unbacked mount), and `open("")` still fails. Every normalization
only ever *shortens* the path, so lexical containment by construction is unchanged.

### 2b. `libScePosix::mkdir` was not registered at all

Measured separately, and the more serious half: of the 9 unresolved `libScePosix`
imports, seven are the filesystem family — `mkdir`, `rmdir`, `unlink`, `chmod`,
`fchmod`, `utimes`, `futimes`. `sceKernelMkdir` worked the whole time; `unlink` and
`rmdir` were registered under `libkernel` only. A NID hashes the function name and
resolution is provider-aware, so the title's `libScePosix::mkdir` call hit a null stub
while the Sony spelling right next to it worked.

All seven now register under both `libkernel` and `libScePosix`, plus `sceKernelFchmod`
and `sceKernelSleep` (the Sony spelling of `sleep`, a distinct NID, also unresolved).
`mkdir` and the metadata calls were refactored into `-errno` cores so the POSIX
spellings return `-1` + `errno` and the `sce*` spellings return
`SCE_KERNEL_ERROR_*` — the crate's existing convention.

Remaining unresolved `libScePosix`: `sendmsg`, `recvmsg`. Those are socket
message-vector semantics, not a naming gap, and were left alone.

### `/devlog` — decided: stays ENOENT, no mount

`/devlog` is a **development-console** mount. It does not exist on a retail unit either,
so a retail title that opens `/devlog/app/debug.log` gets ENOENT on real hardware and
continues — which is exactly what it does here. Mounting it would invent a device the
guest cannot distinguish from a real devkit, and the title would start believing its log
writes land somewhere readable.

What *was* wrong is that the miss logged at `WARN`, reading as a missing mount, and that
misdirection is what put `/devlog` on this task list at all. The message is now `debug`
with an explicit "retail hardware does not have this either" note, gated on a one-entry
`DEVKIT_ONLY_ROOTS` list. `/hostapp` looks similar but is deliberately **not** in that
list — KytyPS5 (`src/emulator.cpp:202`) and shadPS4 (`src/core/file_sys/fs.cpp:104`)
both treat it as a second name for the app root, so an unresolved `/hostapp` open is a
genuine missing mount and must keep warning.

---

## 3. Genuinely missing libraries — added by measured import, not by name

The brief listed six candidates. `cargo xtask nids coverage` decides which are worth
implementing: a library can be `DT_NEEDED` and still be imported from **zero** times.

| library | unresolved imports | action |
|---|---|---|
| `libSceVrSetupDialog` | 6 | **added** (`libsce_dialog_misc.rs`) |
| `libSceAudioIn` | 5 | **added** (`libsce_audio_in.rs`) |
| `libSceErrorDialog` | 4 | **added** (`libsce_dialog_misc.rs`) |
| `libSceMsgDialog` progress-bar trio | 3 | **added** (existing `libsce_common_dialog.rs`) |
| `libSceAudio3d` | **0** | declined |
| `libSceRazorCpu` | **0** | declined |
| `ulobjmgr` | **0** | declined |

The three declined are `DT_NEEDED` (or needed by `libSceJobManager.prx`) but contribute
**no** unresolved import to this title — and since the crate had zero registrations for
any of them, zero unresolved means zero imports, not "already covered". Writing
`libSceAudio3d` or a `libSceRazorCpu` profiler no-op would produce registrations no
measured title calls. They stay unimplemented and the NEEDED warning stays honest.

### Semantics chosen (Tier-B policy, `libsce_online_misc.rs` pattern)

* **`libSceAudioIn`** — a *real* port that captures *real silence*. `Open`/`AsyncOpen`
  allocate a port and return a real positive handle (shadPS4's
  `(type << 16) | port_id | 0x30000000` encoding, so a title inspecting the handle sees
  a plausible one); `Input` **zero-fills** the guest buffer in the port's own PCM layout,
  returns the sample count, and paces one grain period so a capture thread neither hangs
  nor spins (the rule `libsce_audio_out.rs` already uses, and the simulated block
  KytyPS5 applies in `Audio::AudioInInput`, `src/libs/audio.cpp:564-582`);
  `GetSilentState` reports `DEVICE_NONE`, which is exactly how shadPS4 answers with no
  microphone (`src/core/libraries/audio/audioin.cpp:250-252`). No fabricated audio, and
  the "no device" fact travels through the API's own channel instead of being left for
  the title to infer from silent buffers. Unsupported formats and zero rates return the
  real error codes rather than a handle.
* **`libSceErrorDialog`** — completes immediately (no host popup exists, and a status
  that never reaches `FINISHED` parks the title's poll loop forever). The error code the
  title passed is logged at `warn!`: a title opening this dialog is reporting a
  user-visible problem, which is precisely the line the next debugging session needs.
* **`libSceVrSetupDialog`** — completes immediately with a **canceled**-shaped outcome,
  not a success one: there is no VR device, and claiming a headset was configured would
  send a title into a VR mode with nothing behind it. `GetResult` is
  `register_incomplete` — the `SceVrSetupDialogResult` layout is undocumented, so no
  guessed struct is written into caller memory (the same rule
  `libsce_signin_dialog.rs` follows).
* **`sceMsgDialogProgressBar{Inc,SetMsg,SetValue}`** — acknowledge. Nothing is on screen
  to advance and none returns data, so there is nothing to fabricate; refusing would
  make a title treat the operation the bar reports on as failed. KytyPS5 registers the
  same three against its shared `Dialog::MsgDialog` implementation
  (`src/libs/libDialog.cpp:114-116`).

---

## 4. `argv[0]` was the raw host path (side finding — fixed, no boot-outcome claim)

Observed on hardware from the title's own launcher banner:

```
Arg 0 = E:\PS5\PPSA13580-app\eboot.bin
```

The isolated runner (`crates/raeen-gui/src/main.rs`) passed `&[path.as_str()]` — the
**host** path it was invoked with — and an empty `envp`. That is a string no guest API
can open: the title's content is mounted at `/app0`, so a title that opens or parses
`argv[0]` looks up a path that does not exist in its own address space. It also leaks the
host filesystem layout into guest memory for no reason.

The Shell's in-process launcher already passed `/app0/eboot.bin`; only the runner (the
path every real launch takes) did not — so the two launch paths disagreed about the
process environment a title is entered with.

### Fix

`raeen_kernel::filesystem::guest_argv0(host_eboot)` builds `argv[0]` as the eboot's
basename under `GUEST_WORKING_DIRECTORY` (`/app0/eboot.bin`), which is the spelling the
app mount resolves back to that same file — both launch paths already bind that mount to
the eboot's own directory before entering the guest (`launcher.rs:696`, `main.rs:1331`),
so `argv[0]` names a file the guest can actually open. Both now call it, and both pass
`GUEST_ENVP` (`LD_LIBRARY_PATH=/app0/sce_module`) instead of an empty environment — an
Orbis process has a real environment, and the PS5 SDK's own rtld reads exactly that
variable out of it (`reference/ps5-payload-sdk`, `crt/rtld.c:223`).

References: shadPS4 builds `argv[0]` the same way (`src/emulator.cpp:285`,
`"/app0/" + eboot_name`); KytyPS5 passes a bare `"KytyEmu"`
(`src/loader/runtimeLinker.cpp:1359`) — guest-visible, but not openable.

`build_process_stack` now also logs a `WARN` for any `argv`/`envp` entry carrying host
path syntax, so a future call site that regresses this says so instead of silently
handing the guest a host string.

### Tests

* `raeen-runtime`, `tests/execute.rs`:
  `process_argv_and_envp_carry_guest_paths_never_the_host_path` — a `_start` stub reads
  `argv[0]` and `envp[0]` off its own process stack and hands each to the real HLE
  `puts`, so the assertions are made against the bytes the **guest dereferenced**, from
  an `argv[0]` built by the production helper out of the exact host path above. No drive
  letter, no backslash, and no fragment of `E:\PS5\PPSA13580-app\…` reaches guest memory.
* `raeen-kernel`, `filesystem`: `guest_argv0` maps Windows, UNC and POSIX host paths to
  the app-mount spelling; `GUEST_ENVP`'s paths are pinned under
  `GUEST_WORKING_DIRECTORY`; `looks_like_host_path` covers drive letters and
  backslashes without flagging legitimate guest paths.

**Explicitly not a stage claim.** This is a correctness fix. Nothing measured suggests
Blasphemous II — or any other title — boots differently because of it; the title read the
wrong string and carried on. Its value is that a title which *does* act on `argv[0]` no
longer gets a lie.

## Not verified

* **This does not claim the title boots.** 27 imports link that did not; whether any of
  them was *the* blocker is unmeasured. The frame path still has to be re-checked on
  hardware.
* 122 imports remain unresolved, concentrated in `libSceHttp2` (19),
  `libSceNpUniversalDataSystem` (15), `libSceNpEntitlementAccess` (11),
  `libSceNpSessionSignaling` (10) and `libSceAvPlayer` (9). Those are online/PSN and
  media-playback surfaces, not obvious boot-path blockers.
* `libSceAvPlayer`'s 9 remaining unresolved functions are a real gap in an implemented
  library (media playback), untouched here.

## Confirm on hardware

```powershell
cargo run --release -p raeen-gui -- --run-eboot "E:\PS5\Blasphemous II\PPSA13580-app\eboot.bin" 2>&1 |
  Tee-Object -FilePath blasphemous-after-native-fix.log
```

Then check, in that log:

1. No `no HLE library named 'libSce*.native'` warnings remain.
2. No `sceKernelMkdir('././') failed: path is not mounted`.
3. `/devlog/app/debug.log` no longer produces a `WARN`.
4. Whether the frame path advances past `reached=nothing`.
