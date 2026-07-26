# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `5bcf76c353b8` | TimedOut | 182.9s | 3056 MiB | 73 | 17 | 2026-07-26T07:06:31.620073Z ERROR ThreadId(62) kyty_graphics::shader::parse: unknown exp target: 0x09 at addr <ADDR> (en=0x0 done=1 compr=0 vm=1) (hash0 = <ADDR>, crc32 = <ADDR>) |
| Minecraft | `5bcf76c353b8` | TimedOut | 180.8s | 1400 MiB | 7872 | 0 | none observed |
| Until Dawn | `5bcf76c353b8` | Exited | 6.0s | 998 MiB | 0 | 0 | 2026-07-26T07:11:53.663150Z ERROR ThreadId(01) raeen_runtime::dispatch: guest fault at <ADDR> (execute <ADDR>) — 4096 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
| Subnautica Below Zero | `5bcf76c353b8` | TimedOut | 180.1s | 406 MiB | 0 | 0 | 2026-07-26T07:11:56.188914Z WARN ThreadId(05) raeen_hle::libkernel: sceKernelRaiseException: guest handler is registered but asynchronous delivery is not implemented; acknowledging target_thread=0x1 signum=30 handler=<ADDR> |
