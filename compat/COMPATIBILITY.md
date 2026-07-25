# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `df38544b1546` | TimedOut | 180.2s | 1487 MiB | 0 | 0 | none observed |
| Minecraft | `df38544b1546` | TimedOut | 180.3s | 1767 MiB | 2048 | 0 | none observed |
| Until Dawn | `df38544b1546` | Exited | 4.3s | 997 MiB | 0 | 0 | 2026-07-25T07:06:29.688657Z ERROR ThreadId(01) raeen_runtime::dispatch: guest fault at <ADDR> (execute <ADDR>) — 4096 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| Subnautica Below Zero | `df38544b1546` | Exited | 1.9s | 405 MiB | 0 | 0 | 2026-07-25T07:06:31.702552Z ERROR ThreadId(01) raeen_runtime::dispatch: guest called unimplemented import nid <ADDR> from Some("libScePad") (stub <ADDR>, rip <ADDR>) — 3518 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
