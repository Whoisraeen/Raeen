# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `472275f2ce37` | Crashed | 39.5s | 1685 MiB | 4 | 20 | 2026-07-28T07:07:00.393432Z ERROR ThreadId(62) kyty_graphics::shader::spirv: storage_texture_dim_format: not supported: mixed storage image dims/formats in one shader ((Three, "Rgba16f") vs (Two, Rgba16f)) |
| Minecraft | `472275f2ce37` | TimedOut | 180.3s | 1358 MiB | 11232 | 0 | none observed |
| Until Dawn | `472275f2ce37` | Exited | 5.7s | 579 MiB | 0 | 0 | 2026-07-28T07:10:35.136984Z ERROR ThreadId(01) raeen_hle::libc: __stack_chk_fail: guest stack canary smashed on thread 1 ('<unnamed>'), guest ra=<ADDR> — terminating the calling guest thread (exit code <ADDR>); the frame that called this is the one that overflowed |
| Subnautica Below Zero | `472275f2ce37` | TimedOut | 180.1s | 411 MiB | 0 | 0 | 2026-07-28T07:10:38.278774Z WARN ThreadId(05) raeen_hle::libkernel: sceKernelRaiseException: guest handler is registered but asynchronous delivery is not implemented; acknowledging target_thread=0x1 signum=30 handler=<ADDR> |
