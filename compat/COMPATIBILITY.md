# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `81310510cc6c` | Rendering | 120.5s | 4880 MiB | 64 | 56 | 2026-07-31T07:07:23.295199Z ERROR ThreadId(01) raeen_hle::pthread_sync: scePthreadMutexLock waiting >30s with one owner — probable deadlock mutex=<ADDR> waiter=1 waiter_name=<unnamed> owner=21 owner_name=<unnamed> owner_acquired_at=module+0x660 owner_acquire_stack=module+<ADDR> <- module+<ADDR> <- module+0x10bc owner_site=<ADDR> owner_wait=<guest code> ty=2 recursion=1 queued=1 owner_changes=0 owner_stable_ms=30008 |
| Minecraft | `81310510cc6c` | TimedOut | 180.5s | 1205 MiB | 7008 | 0 | 2026-07-31T07:09:52.851414Z ERROR ThreadId(01) raeen_hle::pthread_sync: scePthreadMutexLock waiting >30s with one owner — probable deadlock mutex=<ADDR> waiter=1 waiter_name=MINECRAFT MAIN THREAD owner=12 owner_name=Rendering Pool(1) owner_acquired_at=libc.prx+0x5ec9 owner_acquire_stack=module+<ADDR> <- module+<ADDR> <- module+<ADDR> <- module+<ADDR> <- module+<ADDR> <- libc.prx+0x5eb9 <- module+<ADDR> <- libc.prx+0x5201 owner_site=<ADDR> owner_wait=<guest code> ty=3 recursion=1 queued=1 owner_c |
| Until Dawn | `81310510cc6c` | TimedOut | 180.2s | 1507 MiB | 0 | 0 | none observed |
| Subnautica Below Zero | `81310510cc6c` | TimedOut | 185.6s | 2232 MiB | 4504 | 453 | 2026-07-31T07:15:31.100935Z ERROR ThreadId(23) raeen_runtime::dispatch: guest fault at <ADDR> (read 0x2b) — 4096 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
