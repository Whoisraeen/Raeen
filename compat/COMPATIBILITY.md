# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| Minecraft | `36a9b18ccbb1` | TimedOut | 180.3s | 1749 MiB | 13184 | 0 | none observed |
| Until Dawn | `36a9b18ccbb1` | TimedOut | 180.2s | 1527 MiB | 0 | 0 | none observed |
| Dragon Ball Sparking Zero | `36a9b18ccbb1` | TimedOut | 180.1s | 1071 MiB | 0 | 0 | none observed |
| Dragon Ball Sparking Zero | `36a9b18ccbb1` | Exited | 5.8s | 1071 MiB | 0 | 0 | none observed |
| Subnautica Below Zero | `36a9b18ccbb1` | TimedOut | 180.1s | 411 MiB | 0 | 0 | none observed |
| A Plague Tale Requiem | `36a9b18ccbb1` | Exited | 30.2s | 354 MiB | 0 | 0 | 2026-07-28T19:04:09.048400Z ERROR ThreadId(01) raeen_runtime::dispatch: INTEGER DIVIDE FAULT: guest code divided by a value that was zero. Very often that zero came FROM US: an HLE stub returning 0 (or leaving an out-parameter untouched) for a grain size, sample rate, stride, element size, or frequency the title then divides by. The recent-HLE-call trace below is the place to look. rip=<ADDR> cause=divide by zero origin=guest hle="<none>" |
| Avatar Frontiers of Pandora | `36a9b18ccbb1` | TimedOut | 180.4s | 2380 MiB | 192 | 2398 | 2026-07-28T19:04:15.919430Z ERROR ThreadId(63) kyty_graphics::shader::spirv: can't recompile (no table entry for BufferLoadFormatXyzw/Vdata4VaddrSvSoffsIdxen): BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen ] v[0:3], v5, s[12:15], 0, idxen |
| ASTRO.BOT | `36a9b18ccbb1` | TimedOut | 180.7s | 5648 MiB | 128 | 126 | 2026-07-28T19:07:19.310265Z ERROR ThreadId(62) kyty_graphics::shader::parse: not implemented smem feature: offset != 0 with register soffset on an s_buffer_load (V# base) at addr <ADDR> |
| Grand Theft Auto V | `36a9b18ccbb1` | TimedOut | 183.2s | 2142 MiB | 192 | 98 | 2026-07-28T19:12:35.333039Z ERROR ThreadId(35) kyty_graphics::shader::spirv: can't recompile: SBufferLoadDwordx8 [Sdst8SvSoffset ] s[24:31], s[20:23], 0 |
