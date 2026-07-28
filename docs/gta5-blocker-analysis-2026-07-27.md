# GTA V (PPSA04264 v01.005.000) — where it actually stops

Measured 2026-07-27 against a user-owned retail install. Same host as the M4
record: Ryzen 5 7640HS, Radeon 760M, Windows 11, Vulkan 1.3.

**Short version: it launches and presents, then trips its own assertion
because ~83 `libSceAgc` entry points it needs do not exist yet.** This is a
scope statement, not a bug report — the gap is breadth of AGC coverage.

## What already works

- Library entry, cover art, and metadata resolve (`PPSA04264`,
  `v01.005.000`, "Ready to play").
- The 66 MB eboot loads: **0x62CBA00-byte image (~103 MB), 3 dependencies,
  612 HLE trampolines linked.**
- The guest runs far enough to create **hundreds of event flags**
  (`sceKernelCreateEventFlag` → handles into `0x2a7`+), load system modules
  (`sceSysmoduleLoadModule → OK`), map save data
  (`VFS: /savedata0/ -> savedata\PPSA04264-app`), and drive AGC submissions.
- Frames reach the Shell: the session overlay shows a presented (cleared
  blue) framebuffer, so the present path is intact end to end.

## Where it stops

1. **Guest assertion trap.** Fault at `rip=0x10000000ae36` (module +0xae36),
   classified `execute`. The bytes at RIP are `0F 0B` — **`UD2`**, preceded by
   a `call`. That is the title's own assert/panic-then-trap idiom, i.e. GTA V
   decided its own state was invalid. `no HLE call returned an Orbis error
   before this fault`, so nothing we implemented reported failure — something
   we returned was accepted and then judged wrong.
2. **271 distinct missing NIDs** (311 unresolved relocations: 259 in the main
   module + 52 across dependencies).
3. **A recursive-mutex deadlock** after the fault
   (`mutex=0x1000045755c8 waiter=4 owner=20 ty=2`) — likely a consequence of
   a thread dying inside a lock, not an independent cause.

## The real wall: AGC breadth

Missing NIDs by library (top):

| Count | Library | Note |
|-------|---------|------|
| **83** | `libSceAgc` | **the blocker** — GPU command building |
| 46 | `libSceAmpr` | async compute / prefetch |
| 17 | `libSceNpWebApi2` | online; stubbable |
| 15 | `libSceVoice` | audio chat; stubbable |
| 10 each | `libSceNpManager`, `libSceAvPlayer` | online / video |
| ~1–9 | 30 further libraries | mostly online, dialogs, streaming |

The non-AGC ~188 are overwhelmingly **online/social/dialog** surfaces that a
single-player boot can stub. The 83 AGC ones cannot be faked: they build the
actual GPU command stream.

Their shape is informative — they are dominated by the **ACB (async compute
buffer)** and **`*GetSize`** families:

```
sceAgcAcbAtomicMem / AtomicMemGetSize / AtomicGdsGetSize
sceAgcAcbJump / JumpGetSize / Rewind / RewindGetSize
sceAgcAcbDispatchIndirectGetSize / EventWriteGetSize / CopyDataGetSize
sceAgcAcbMemSemaphore / PrimeUtcl2 / WaitOnAddressGetSize
sceAgcAcbSetFlip / SetMarker / SetWorkloadComplete / SetWorkloadsActive
sceAgcAcbWaitUntilSafeForRendering
sceAgcAsyncCondExecPatchSet* / sceAgcAsyncRewindPatchSet* / sceAgcBranchPatchSet*
sceAgcCbBranchGetSize / CbCondWrite / CbSetShRegistersDirectGetSize ...
sceAgcQueueEndOfPipeActionPatchData        (observed CALLED, unresolved)
sceAgcGetDefaultCxStateFlat                (observed missing)
```

Two structural facts follow:

- **GTA V drives async compute queues (ACB), not just the graphics DCB.**
  Minecraft only needed the DCB path. This is a different, larger surface.
- **The `*GetSize` pattern is a size-query/emit pair**: the title asks how
  many bytes a packet needs, reserves, then emits. A missing `GetSize` makes
  the title reserve a wrong (often zero) size — which is exactly the kind of
  invariant violation that ends in `UD2`. These are *mechanical* to implement
  (return the packet's dword count) and should be done as a family, not
  one-off.

## Honest scope

"Fully playable GTA V" is not a session of work. Realistically:

1. Implement the AGC `*GetSize` family (mechanical, high leverage).
2. Implement the ACB command-buffer family and wire it to the existing PM4
   command processor (which already handles the DCB).
3. Stub the ~188 online/dialog NIDs so they stop being unresolved.
4. Only then does the assert move, and the *next* wall becomes visible.

Steps 1–3 are tractable and well-defined. What comes after them is unknown
until measured — GTA V is one of the heaviest titles on the platform, and no
amount of planning substitutes for re-running after each step.

**Do not treat this document as a promise of playability.** It is a map of
the first wall, produced so the next attempt starts from evidence instead of
from a launch.

---

## Re-measure 2026-07-28 (post ACB/Ampr/Tier-B wave, build 9c7cc30)

The 83-NID AGC gap, 46-NID Ampr gap, and ~188 online/dialog NIDs are all
resolved — **zero unresolved NIDs** in this run. The early UD2 assert is gone.

- **Stage: `timed_out`** — the title now survives the full 180 s window
  (was: early self-assert). **4 flip events** presented; ~25 s CPU over 183 s
  wall; 1.06 GB peak working set.
- **New first blocker:** `__stack_chk_fail` on thread 31 — a real guest stack
  canary smash, now caught and reported by the noreturn handler (d818df9)
  instead of walking into UD2. Only that thread terminates; the process keeps
  running. Blocker signature `e86038e1…` (see `artifacts/compat/latest.json`).
- **Next work:** find what smashes the canary (same family as the Until Dawn
  canary, which also persists post-dirent-fix — possibly a shared libc/kernel
  struct-size mismatch on another syscall surface, hunted the same way the
  getdents one was).

Scope statement, not a playability promise — but the wall moved: from "needs
131 GPU entry points" to "one canary-smash bug plus shader breadth."
