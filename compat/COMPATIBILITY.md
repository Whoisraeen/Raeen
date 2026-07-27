# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `b9c2daf5501d` | TimedOut | 180.4s | 2670 MiB | 96 | 83 | 2026-07-27T07:02:44.511964Z ERROR ThreadId(62) kyty_graphics::shader::parse: unknown exp target: 0x09 at addr <ADDR> (en=0x0 done=1 compr=0 vm=1) (hash0 = <ADDR>, crc32 = <ADDR>) |
| Minecraft | `b9c2daf5501d` | TimedOut | 180.4s | 1794 MiB | 13536 | 0 | none observed |
| Until Dawn | `b9c2daf5501d` | Exited | 4.0s | 998 MiB | 0 | 0 | 2026-07-27T07:08:38.643154Z ERROR ThreadId(01) raeen_runtime::dispatch: guest fault at <ADDR> (execute <ADDR>) — 4096 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| Subnautica Below Zero | `b9c2daf5501d` | TimedOut | 180.0s | 406 MiB | 0 | 0 | 2026-07-27T07:08:40.812277Z WARN ThreadId(05) raeen_hle::libkernel: sceKernelRaiseException: guest handler is registered but asynchronous delivery is not implemented; acknowledging target_thread=0x1 signum=30 handler=<ADDR> |
