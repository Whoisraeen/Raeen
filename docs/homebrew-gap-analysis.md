# Homebrew gap analysis

## What was just proven

`crates/xps5x-gui/src/launcher.rs` (`firmware_launcher_tests::executes_module`) now has an
end-to-end acceptance test for a hand-built, homebrew-*shaped* module that goes through the
Shell's real load path — `FirmwareLauncher::launch` → `load` → `xps5x_firmware::load_module`
(SELF passthrough → `.sprx` parse → `PT_SCE_DYNLIBDATA` decode → NID link) →
`xps5x_runtime::execute_linked` — both by loading a temp file directly and by having the real
`scan_dir` discover it as `Games/<name>/eboot.bin` first. The module's entry does real work
through three real, NID-resolved HLE imports (`malloc`, `memset`, and libkernel's
`sceKernelMapFlexibleMemory` — the last resolved but not called): it calls `malloc(0x40)`,
`memset(ptr, 0xAB, 0x40)`, reads byte 0 back, and returns it. Observed outcome:
`SessionOutcome::Ran { returned: 0xab, resolved: 3, unresolved: 0 }` on both paths.

That is a genuinely meaningful proof — the whole SELF→ELF→dynlib→NID-link→native-execution→HLE
dispatch→guest-heap-memory pipeline is real, not simulated. But the module itself is still
*hand-assembled to fit exactly what this pipeline supports*. This document is the honest
accounting of what a **real, compiler-produced PS4/PS5 homebrew binary** (e.g. from an
OpenOrbis-style toolchain) would additionally need, that this synthetic fixture sidesteps by
construction.

## Entry / crt0

The synthetic module's "entry" is a handful of instructions called directly as a bare
`sysv64 fn(u64,u64,u64,u64,u64,u64) -> u64` (`xps5x-runtime/src/dispatch.rs`, `run`) with up to
six integer arguments and nothing else set up. A real homebrew binary's ELF entry point is
`_start`/crt0, and it unconditionally assumes:

- **A process stack shaped like a Linux/BSD-style ELF stack**: at the initial `entry` call, `rsp`
  must point at `argc`, followed by `argv[0..argc]` (each a pointer), a `NULL`, `envp[0..]`
  pointers, a `NULL`, then the auxiliary vector (`Elf64_auxv_t` array — `AT_PHDR`, `AT_PHENT`,
  `AT_PHNUM`, `AT_ENTRY`, `AT_BASE`, `AT_RANDOM`, etc.), terminated by `AT_NULL`. crt0 reads all
  of this directly off the stack before calling `main`.
- **A call into `__libc_start_main`** (or the PS4/PS5 SDK's equivalent init sequence), which does
  C runtime setup (TLS init, atexit table, stdio init) *before* ever calling the homebrew's own
  `main`.
- Some homebrew toolchains additionally expect `sceKernelGetProcParam`'s process-parameter block
  to be populated with real SDK-version / entry metadata (currently a stub — see HLE breadth
  below) before `_start` gets far.

Right now `execute_linked` calls the module's `e_entry` directly with a caller-supplied `args`
slice (0–6 raw integers) — there is no stack layout at all beyond the dedicated guest stack region
`GuestArena` provides (`STACK_OFFSET`/`STACK_SIZE`), and nothing writes argc/argv/envp/auxv onto
it. A real `_start` reading `[rsp]` as `argc` would read garbage (or zero, since the stack is
freshly committed/zeroed) and immediately misbehave.

## Relocations

`link_module` (`crates/xps5x-firmware/src/dynlib/linker.rs`) currently handles exactly four
`R_X86_64_*` types:

| Type | Value | Handling |
|---|---|---|
| `R_X86_64_64` | 1 | symbol value/trampoline + addend |
| `R_X86_64_GLOB_DAT` | 6 | symbol value/trampoline |
| `R_X86_64_JUMP_SLOT` | 7 | symbol value/trampoline |
| `R_X86_64_RELATIVE` | 8 | `base + addend` |

Any other type hits the `other => Err(FirmwareError::UnsupportedRelocation(other))` arm — a hard
link failure, not a soft "unresolved" like an unknown NID. A real compiler-produced `.prx`/`self`
built with a PIC/PIE toolchain (which OpenOrbis-style PS4/PS5 SDKs are) will commonly also emit:

- **`R_X86_64_DTPMOD64` (16) / `R_X86_64_DTPOFF64` (17) / `R_X86_64_TPOFF64` (18)** — TLS model
  relocations for any `__thread`/`thread_local` variable (including compiler-inserted TLS like the
  stack-protector guard, if the toolchain doesn't route it through `__stack_chk_guard`/`fs:`
  directly). **Currently unhandled — would hard-fail linking.**
- **`R_X86_64_IRELATIVE` (37)** — used by ifunc-resolved libc functions (memcpy/memset dispatch by
  CPU features is a common one). **Unhandled.**
- **`R_X86_64_COPY` (5)** — copy relocations for data symbols pulled from a shared object into the
  executable's BSS. Less likely in a `.prx` (which is itself the shared object), but possible
  depending on how the toolchain links the main module. **Unhandled.**
- **`R_X86_64_PC32`/`R_X86_64_32`/`R_X86_64_32S`** — non-PIC-style relocations; unlikely from a
  modern PIE toolchain but possible from an old-style build. **Unhandled.**

Practically: the *first* time a real homebrew's `.prx` has a single TLS variable or an ifunc-based
libc call, `link_module` returns `Err(UnsupportedRelocation(..))` and the whole load aborts before
a single instruction runs.

## HLE breadth

`crates/xps5x-hle` currently registers, across `libc`, `libkernel`, `libSceSysmodule`, and several
`libSce*` device stubs (gnm_driver, audio_out, net, save_data, pad, video_out):

- **Real, working memory semantics**: `malloc`/`calloc`/`realloc`/`free`/`memalign`/
  `posix_memalign`, `memcpy`/`memset`/`memmove`, `strlen`/`strcmp`/`strcpy`/`strncpy`, and
  `libkernel`'s `sceKernelAllocateDirectMemory`/`sceKernelMapFlexibleMemory`/`sceKernelMapDirectMemory`/
  `sceKernelMunmap`/`sceKernelMmap` — these actually touch `GuestArena`-backed memory, not stubs.
- **Present but stubbed/placeholder** (logged, return a plausible constant, do no real work):
  - `printf`/`snprintf` — "no formatting performed" (see `libc.rs` doc comments); a homebrew that
    prints anything to observe its own behavior gets silence, not text.
  - `puts` — "cannot read guest memory" placeholder in some paths.
  - `scePthreadCreate`/`scePthreadJoin`/`scePthreadExit`/mutex/cond functions — registered, but do
    not actually spawn a second guest execution context (there is exactly one call-and-return
    native thread per `execute_linked` call; no multi-threading model yet). Any homebrew that
    starts a worker thread (extremely common — audio thread, render thread, job system) will get a
    no-op instead of real concurrency.
  - `sceKernelGetProcParam` — returns a stub value, not a populated process-parameter block SDK
    version info readers expect.
  - `__stack_chk_fail` — registered as a no-op-ish stub, i.e. a triggered stack-protector failure
    is not treated as fatal the way a real libc's would be.
- **Not present at all** (would resolve as `Unresolved`, i.e. a wild-address call trampoline that
  faults if actually invoked):
  - `sceKernelLoadStartModule` and friends — no support for a module loading and starting *another*
    module/library at runtime. Any homebrew split into a main executable + separate `.prx`
    dependencies (a very common shape, since PS4/PS5 SDKs ship most functionality as separate
    system `.prx`s) cannot pull those in.
  - `write`/`read`/file-descriptor I/O syscalls, `sceKernelOpen`/`sceKernelClose`/etc. — no
    filesystem-facing syscall surface.
  - Any real graphics (`libSceGnmDriver` is a stub module — see its file), audio, input, or
    networking beyond the placeholder registrations already listed.

## TLS

`xps5x-runtime/src/tls.rs` + `arena.rs`'s `setup_main_tcb` gives the guest a **minimal** TCB: a
`fs:[0]` self-pointer (so `mov rax, fs:[0]` round-trips, proven by
`guest_fs_zero_load_reads_the_installed_tcb`) and general `fs:`-relative offset addressing works
(proven by `guest_fs_offset_round_trip_writes_and_reads_back`). That is exactly enough for a
hand-written stub to prove FSGSBASE works. A real compiler's TLS/stack-protector expectations go
further:

- **A real `PT_TLS` segment image**: the compiler emits a `.tdata`/`.tbss` template segment; a
  real loader copies `.tdata` into each thread's TCB-adjacent memory and zeros `.tbss`, then
  resolves TLS-relocated symbols (`DTPMOD64`/`DTPOFF64`/`TPOFF64`, see Relocations above) against
  it. `load_module`/`link_module` do not parse or lay out `PT_TLS` at all today.
  - **`__stack_chk_guard` at the ABI-mandated offset**: the System V x86-64 ABI (as extended by
    glibc/PS4 libc) keeps the stack canary at a specific `fs:`-relative offset (glibc: `fs:0x28`).
    Compiler-generated function prologues/epilogues read/compare that exact offset unconditionally
    whenever stack-protector is enabled (the toolchain default for most homebrew SDKs). The current
    TCB has no canary installed at any offset — a real homebrew's very first stack-protected
    function return would read uninitialized/zeroed memory at `fs:0x28`, and either randomly trip
    `__stack_chk_fail` or (if it reads zero and the reference value is also zero) silently work by
    accident. Either way it is not a real, ABI-correct canary.
- **Per-thread TCBs**: `setup_main_tcb` sets up exactly one TCB for the one call-and-return native
  execution. Any real multi-threaded homebrew (see HLE breadth: `scePthreadCreate`) would need a
  fresh, correctly-linked TCB per thread — not modeled.

## Module format realities

- **Encrypted retail SELF**: `NoKeysProvider` is used throughout by design (Shell never handles
  keys — spec §2). A real retail `eboot.bin`/`.sprx` is AES-encrypted and needs actual per-console
  or per-title key material this project deliberately never has. Out of scope, permanently, for
  this codebase's clean-room boundary — not a "todo," a hard wall.
- **`.prx`/library dependencies**: real homebrew is rarely a single self-contained module. It
  typically declares `NEEDED`-style dependencies on system `.prx`s (`libSceLibcInternal`,
  `libSceFios2`, etc.) that must be loaded and linked in first via `sceKernelLoadStartModule` (see
  HLE breadth) — the `ModuleRegistry` here supports multiple modules' exports being registered
  (`register_module_exports`), but nothing drives *discovering and loading* a dependency chain from
  one module's declared imports; the Shell only ever loads the one `eboot.bin` path it's given.
- **Fat/multi-arch ELF (fatelf) and non-trivial `PT_SCE_*` segments**: `xps5x-firmware`'s `.sprx`
  parser handles `PT_LOAD`, `PT_SCE_DYNLIBDATA`, and `PT_DYNAMIC` — real PS4/PS5 SELFs carry
  additional segments (`PT_SCE_PROCPARAM`, `PT_SCE_MODULE_PARAM`, `PT_SCE_RELRO`'s more nuanced
  handling — `SprxModule` has a `relro` field but its consumption elsewhere in the pipeline was not
  audited as part of this task) that a compiler toolchain will populate and that a genuinely
  complete loader needs to at least recognize (ignore-with-a-log at minimum, parse at best) rather
  than silently drop.

## Prioritized shortlist: the first wall a real homebrew hits

Ranked by "how early in loading/execution this bites, for an ordinary compiler-produced binary":

1. **Missing stack/argc-argv-envp-auxv layout for `_start`/crt0.** This is the very first
   instruction of a real binary's execution reading real data — every ELF `_start` immediately
   reads `[rsp]` as `argc`. Without this, execution is DOA for anything beyond a bare, hand-written
   entry that never expects a process environment. Highest priority: nothing else matters if
   `_start` faults on its first stack read.
2. **`R_X86_64_DTPMOD64`/`DTPOFF64`/`TPOFF64` (TLS relocations) hard-failing the link.** Any
   `__thread` variable, or compiler-inserted TLS the toolchain doesn't route through
   `__stack_chk_guard`/`fs:` some other way, aborts `link_module` outright — this is a hard error,
   not a soft unresolved-import, so it's a total block rather than a degraded run.
3. **No `sceKernelLoadStartModule`.** Almost every real PS4/PS5 homebrew is built against system
   `.prx` libraries loaded at runtime, not statically bundled — without this, a homebrew that calls
   even one exported function from `libSceLibcInternal`/`libSceFios2`/etc. cannot resolve it.
4. **No real threading (`scePthreadCreate` is a no-op).** Homebrew that spins up even one worker
   thread (audio callback, render thread) silently gets nothing instead of a thread — a subtle
   correctness gap rather than a crash, which is arguably worse (it may appear to "run" while doing
   the wrong thing).
5. **No `__stack_chk_guard` canary at the ABI offset (`fs:0x28`) + no `PT_TLS` image.** Any
   stack-protector-enabled compile (the common default) either trips a spurious
   `__stack_chk_fail` or silently "works" by coincidence — either way, not the real ABI contract,
   and the first `__thread` variable read/write has nowhere correct to land.
