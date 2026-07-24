# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `f8ca653d55be` | TimedOut | 180.2s | 1523 MiB | 0 | 0 | none observed |
| Minecraft | `f8ca653d55be` | TimedOut | 181.1s | 2188 MiB | 512 | 0 | none observed |
| Until Dawn | `f8ca653d55be` | TimedOut | 180.6s | 1230 MiB | 0 | 0 | 2026-07-24T07:07:38.906049Z ERROR ThreadId(06) raeen_runtime::dispatch: guest fault at <ADDR> (read 0xa) — 27 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| Subnautica Below Zero | `f8ca653d55be` | Exited | 1.0s | 203 MiB | 0 | 0 | 2026-07-24T07:10:32.678517Z WARN ThreadId(01) raeen_hle::libkernel: sceKernelLoadStartModule: '/app0/Media/Modules/PS5Util.prx' registered as an HLE-backed pseudo-module — its imports resolve via NID against the HLE registry; file-backed .prx loading is not implemented |
