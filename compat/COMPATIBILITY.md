# Measured compatibility

Generated only from Raeen's sanitized compatibility-result schema. A result is evidence for this build and machine class, not a universal compatibility claim.

| Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |
|---|---:|---|---:|---:|---:|---:|---|
| Minecraft | `1acd114e6f6b` | TimedOut | 180.2s | 1642 MiB | 13440 | 0 | none observed |
| Until Dawn | `1acd114e6f6b` | TimedOut | 180.2s | 1507 MiB | 0 | 0 | none observed |
| Dragon Ball Sparking Zero | `1acd114e6f6b` | TimedOut | 180.1s | 975 MiB | 0 | 0 | none observed |
| Dragon Ball Sparking Zero | `1acd114e6f6b` | Exited | 4.9s | 580 MiB | 0 | 0 | none observed |
| Subnautica Below Zero | `1acd114e6f6b` | TimedOut | 180.1s | 339 MiB | 0 | 0 | none observed |
| A Plague Tale Requiem | `1acd114e6f6b` | Exited | 30.8s | 216 MiB | 0 | 0 | 2026-07-29T00:06:45.840792Z ERROR ThreadId(01) raeen_runtime::dispatch: INTEGER DIVIDE FAULT: guest code divided by a value that was zero. Very often that zero came FROM US: an HLE stub returning 0 (or leaving an out-parameter untouched) for a grain size, sample rate, stride, element size, or frequency the title then divides by. The recent-HLE-call trace below is the place to look. rip=<ADDR> cause=divide by zero origin=guest hle="<none>" |
| Avatar Frontiers of Pandora | `1acd114e6f6b` | TimedOut | 180.3s | 2356 MiB | 256 | 1899 | 2026-07-29T00:06:51.588500Z ERROR ThreadId(63) kyty_graphics::shader::spirv: can't recompile: BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen ] v[0:3], v5, s[12:15], 0, idxen |
| ASTRO.BOT | `1acd114e6f6b` | Rendering | 66.8s | 4757 MiB | 36 | 45 | 2026-07-29T00:09:55.166614Z ERROR ThreadId(62) kyty_graphics::shader::parse: unknown sop2 opcode: 0x30 at addr <ADDR> (hash0 = <ADDR>, crc32 = <ADDR>) |
| Grand Theft Auto V | `1acd114e6f6b` | TimedOut | 180.4s | 2119 MiB | 256 | 1671 | 2026-07-29T00:12:02.999904Z ERROR ThreadId(35) kyty_graphics::shader::spirv: Recompile_SBufferLoadDword_SdstSvSoffset: not supported: no storage buffer bound for the V# and no resolved capture: SBufferLoadDword [SdstSvSoffset] x1 dwords, V#=s[12:15], soffset=none, imm=0x90, pc=0x88 |
