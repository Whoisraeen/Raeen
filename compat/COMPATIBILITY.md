# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| Minecraft | `43546e18644c` | TimedOut | 20.3s | 1570 MiB | 0 | 0 | none observed |
| Until Dawn | `43546e18644c` | TimedOut | 20.2s | 1797 MiB | 0 | 0 | 2026-07-23T21:41:24.831901Z ERROR ThreadId(04) raeen_runtime::dispatch: guest fault at <ADDR> (read 0xa) — 27 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| Dragon Ball Sparking Zero | `43546e18644c` | TimedOut | 20.2s | 1031 MiB | 0 | 0 | 2026-07-23T21:41:44.759782Z ERROR ThreadId(04) raeen_runtime::dispatch: guest fault at <ADDR> (read 0xa) — 25 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| Dragon Ball Sparking Zero | `43546e18644c` | Exited | 5.7s | 1031 MiB | 0 | 0 | none observed |
| Subnautica Below Zero | `43546e18644c` | Exited | 0.6s | 184 MiB | 0 | 0 | 2026-07-23T21:42:04.391595Z WARN ThreadId(01) raeen_hle::libkernel: sceKernelLoadStartModule: '/app0/Media/Modules/PS5Util.prx' registered as an HLE-backed pseudo-module — its imports resolve via NID against the HLE registry; file-backed .prx loading is not implemented |
| A Plague Tale Requiem | `43546e18644c` | TimedOut | 20.1s | 345 MiB | 0 | 0 | none observed |
| Avatar Frontiers of Pandora | `43546e18644c` | Exited | 1.5s | 1153 MiB | 0 | 0 | 2026-07-23T21:42:26.037332Z ERROR ThreadId(01) raeen_runtime::dispatch: guest called unimplemented import nid <ADDR> from Some("libkernel") (stub <ADDR>, rip <ADDR>) — 21 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| ASTRO.BOT | `43546e18644c` | TimedOut | 20.2s | 1587 MiB | 0 | 0 | none observed |
| Grand Theft Auto V | `43546e18644c` | TimedOut | 20.1s | 480 MiB | 0 | 0 | 2026-07-23T21:42:49.595220Z ERROR ThreadId(22) raeen_runtime::dispatch: guest fault at <ADDR> (read <ADDR>) — 2733 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
