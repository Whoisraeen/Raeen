# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| ASTRO.BOT | `e2bdfacca1a1` | Crashed | 64.2s | 4265 MiB | 18 | 20 | 2026-07-30T07:06:41.242572Z ERROR ThreadId(64) kyty_graphics::shader::analysis: shader analysis: not implemented: too many textures |
| Minecraft | `e2bdfacca1a1` | TimedOut | 181.7s | 1606 MiB | 1056 | 0 | none observed |
| Until Dawn | `e2bdfacca1a1` | TimedOut | 182.5s | 1528 MiB | 0 | 0 | none observed |
| Subnautica Below Zero | `e2bdfacca1a1` | TimedOut | 184.2s | 2010 MiB | 2394 | 224 | 2026-07-30T07:14:15.502656Z ERROR ThreadId(24) raeen_runtime::dispatch: guest fault at <ADDR> (read 0x2b) — 4096 HLE call(s) recorded before the fault; distilled leads follow at WARN, the full oldest-first ring once at DEBUG |
