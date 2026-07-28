//! # Raeen — PlayStation 5 Emulator
//!
//! Main entry point for the Raeen desktop application.
//! Initializes the emulator subsystems and launches the GUI.
// GUI-subsystem build: launching from Explorer/a shortcut opens no terminal
// window (logs live in the in-app console [F10] and logs/raeen.log). CLI
// invocations still print — `main` reattaches the parent console first thing.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod app;
mod compat;
mod crash_report;
mod crashdump;
mod launcher;
mod library;
mod shell;
mod splash;
mod theme;
mod updater;

use std::path::Path;
use tracing::info;

/// Say what the guest was executing when it faulted: which module the faulting
/// `rip` lands in, its offset within that module, and the bytes there.
///
/// A bare `rip` names nothing. In a 250 MB stripped C++ binary one address looks
/// like any other, and the temptation is to reason from whatever HLE call
/// happened to be last — which is adjacency, not causation, and has already cost
/// this project one wrong diagnosis. The module and offset make a fault
/// comparable across runs and greppable against a disassembly; the bytes make it
/// decodable on the spot.
///
/// Reads the composed process image the loader already holds, so it cannot
/// perturb the guest or the fault path.
fn report_fault_site(
    linked: &raeen_firmware::LinkedModule,
    deps: &[raeen_firmware::LoadedDependency],
    addr: u64,
) {
    // Dependencies are composed above the eboot at known offsets, so the last
    // one at or below the rip owns it; below them all, it is the eboot's —
    // resolved by the same pure locator the crash report uses.
    match crash_report::locate_fault(
        &linked.image,
        &dep_offset_pairs(deps),
        raeen_runtime::GUEST_ARENA_BASE,
        addr,
    ) {
        crash_report::FaultLocation::BelowImage => {
            info!("        rip is below the guest image — not guest code");
        }
        crash_report::FaultLocation::PastImage { image_len } => {
            info!("        rip is past the loaded image ({image_len:#x} byte(s)) — not guest code");
        }
        crash_report::FaultLocation::Site(site) => {
            info!("        module: {} at +{:#x}", site.module, site.offset);
            info!("        bytes at rip: {:02x?}", site.rip_bytes);
        }
    }
}

/// `(name, image offset)` pairs for the loaded dependencies — the shape the
/// pure fault locator in [`crash_report`] takes.
fn dep_offset_pairs(deps: &[raeen_firmware::LoadedDependency]) -> Vec<(String, u64)> {
    deps.iter()
        .map(|d| (d.name.clone(), d.image_offset))
        .collect()
}

/// Assemble and write the actionable crash report for a `--run-eboot` session
/// that ended in a runtime error. Runs in the guest-executing process, which
/// is the only place the kernel's call rings, the composed image, and the GPU
/// session are all still alive. Never fatal — a report that cannot be written
/// is a warning, not a second failure.
fn write_runner_crash_report(
    eboot: &Path,
    error: &raeen_runtime::RuntimeError,
    linked: &raeen_firmware::LinkedModule,
    deps: &[raeen_firmware::LoadedDependency],
    kernel: &raeen_kernel::OrbisKernel,
) {
    let (fault, fault_site) = match error {
        raeen_runtime::RuntimeError::Faulted { addr, access, kind } => (
            format!("Guest fault at {addr:#x} ({kind} of {access:#x})"),
            match crash_report::locate_fault(
                &linked.image,
                &dep_offset_pairs(deps),
                raeen_runtime::GUEST_ARENA_BASE,
                *addr,
            ) {
                crash_report::FaultLocation::Site(site) => Some(site),
                _ => None,
            },
        ),
        raeen_runtime::RuntimeError::UnimplementedImport { nid, library, .. } => (
            format!(
                "Unimplemented import: {} ({}) — nid {nid:#018x}",
                raeen_firmware::dynlib::nid_names::describe(*nid),
                library.as_deref().unwrap_or("<unknown library>")
            ),
            None,
        ),
        other => (format!("Runtime error: {other:?}"), None),
    };

    // Most-recent-first call ring per guest thread, labeled by thread name.
    let mut recent_hle: Vec<(String, Vec<String>)> = kernel
        .recent_hle_calls
        .iter()
        .map(|entry| {
            let tid = *entry.key();
            let name = kernel
                .thread_names
                .get(&tid)
                .map_or_else(String::new, |n| n.clone());
            let label = if name.is_empty() {
                format!("t{tid}")
            } else {
                format!("t{tid} ({name})")
            };
            let calls: Vec<String> = entry
                .value()
                .lock()
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect();
            (label, calls)
        })
        .collect();
    recent_hle.sort();

    let gpu = raeen_gpu::AgcGpuSession::global();
    let shader = gpu.shader_stats();
    let gpu_summary = format!(
        "draws={} presented_frames={} shaders: fetched={} translated_ok={} failed={} \
         skipped_draws={}",
        gpu.draw_count(),
        gpu.present_epoch(),
        shader.distinct_fetched,
        shader.translated_ok,
        shader.translate_failed,
        gpu.shader_skip_count()
    );

    let (title_id, title, version) = crash_report::title_meta_for(eboot);
    let report = crash_report::CrashReport {
        title_id,
        title,
        version,
        session_duration: Some(kernel.uptime()),
        fault,
        fault_site,
        recent_hle,
        unresolved_nids: kernel.unresolved_nid_inventory(),
        gpu_summary: Some(gpu_summary),
        host: Some(crash_report::HostInfo::collect()),
        dump_path: None,
        log_path: Some(std::path::PathBuf::from("logs/raeen.log")),
    };
    match report.write_now(Path::new(crash_report::REPORTS_DIR)) {
        Ok(path) => info!("crash report written: {}", path.display()),
        Err(error) => tracing::warn!(%error, "crash report could not be written"),
    }
}

/// Reattach the parent process's console so CLI invocations (`--run-eboot`,
/// `--firmware-info`, dev `cargo run`) still print to the terminal they were
/// launched from despite the GUI subsystem. Launched from Explorer there is
/// no parent console — the call fails and output goes to the file log and
/// the in-app console instead, which is the point.
#[cfg(windows)]
fn attach_parent_console() {
    use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole};
    // SAFETY: plain Win32 call with no pointers; failure is the normal
    // Explorer-launch case and needs no handling.
    unsafe {
        let _ = AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(windows)]
    attach_parent_console();

    // FIRST, before anything else in the process allocates: claim the guest
    // title fixed-VA window. Retail titles map direct memory at literal
    // addresses (ASTRO.BOT: its libc mspace at 0x3_0000_0000) and write to that
    // address regardless of the call's result. Measured 2026-07-21: launched
    // from the Shell the map failed and the title faulted at libc.prx+0x103c6,
    // while the SAME build booted fine via `--run-eboot` — because the GUI
    // process had seconds of eframe/egui/wgpu/Vulkan allocations squatting the
    // window before launch, and the CLI process was clean there by luck.
    // Reserving here — before logging, before eframe, before Vulkan — makes the
    // window deterministically the guest's in BOTH paths (this `main` is also
    // the CLI entry). The report is logged below once logging exists.
    let title_va = raeen_runtime::reserve_title_va_window();

    // Initialize logging to BOTH stderr and `logs/raeen.log`. `_log` must stay
    // alive for the whole process — dropping it shuts down the background
    // writer thread and loses buffered events (see `LogGuard`). Binding it here
    // in `main` is what makes the log file complete on exit.
    //
    // Falls back to stderr-only if the log directory can't be created (e.g. a
    // read-only working directory) — never a reason to refuse to boot.
    let _log = match raeen_core::logging::init_with_file(
        "info",
        std::path::Path::new(raeen_core::logging::DEFAULT_LOG_DIR),
    ) {
        Ok(guard) => guard,
        Err(e) => {
            let guard = raeen_core::logging::init("info");
            tracing::warn!("file logging unavailable ({e}); continuing with stderr only");
            guard
        }
    };

    // Host facts (CPU model, cores, RAM, OS build) at the top of every log —
    // the difference between a diagnosable user report and guesswork.
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_cpu_all();
        sys.refresh_memory();
        info!(
            cpu = sys.cpus().first().map(|c| c.brand().trim()).unwrap_or("unknown"),
            cores = sys.cpus().len(),
            ram_gb = format_args!("{:.1}", sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)),
            os = %sysinfo::System::long_os_version().unwrap_or_else(|| "unknown".into()),
            "host system"
        );
    }

    // Opt-in frame profiler: `RAEEN_PROFILE=1` turns puffin scopes on and
    // serves them on the default port (34567) for `puffin_viewer` to attach.
    // Off (the default) the scope macros compile to a cheap branch.
    let _puffin_server = if std::env::var_os("RAEEN_PROFILE").is_some() {
        puffin::set_scopes_on(true);
        match puffin_http::Server::new(&format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT)) {
            Ok(server) => {
                info!(
                    port = puffin_http::DEFAULT_PORT,
                    "puffin profiler serving — connect with puffin_viewer"
                );
                Some(server)
            }
            Err(e) => {
                tracing::warn!("puffin profiler server failed to start ({e})");
                None
            }
        }
    } else {
        None
    };

    // Now that logging exists, say what the startup reservation achieved. A
    // squatter here means something allocated before `main` (a static
    // initializer, the loader) — each one is address space a guest fixed map
    // can no longer be served at, so it is worth a warning per region.
    info!(
        window = format_args!("{:#x}..{:#x}", title_va.window_start, title_va.window_end),
        blocks = title_va.reserved_blocks,
        reserved_bytes = format_args!("{:#x}", title_va.reserved_bytes),
        "guest title-VA window claimed at startup"
    );
    for squatter in &title_va.squatters {
        tracing::warn!(
            region = %squatter,
            "host allocation was already inside the guest title-VA window before main()"
        );
    }

    // Diagnostic: `raeen --firmware-info <PUP>` inspects a firmware package
    // and exits without launching the GUI. It never decrypts anything.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--firmware-info") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--firmware-info requires a path to a PUP file"))?;
        let firmware = raeen_firmware::Firmware::open(path)?;
        print!("{}", raeen_firmware::summarize(&firmware));
        return Ok(());
    }

    // Diagnostic: `raeen --load-sprx <sprx>` runs the LM1 homebrew pipeline
    // (SELF decrypt-or-passthrough -> .sprx parse -> dynlibdata decode ->
    // NID link against HLE) over a file and prints a summary, then exits
    // without launching the GUI. Uses `NoKeysProvider` throughout — it never
    // decrypts anything without a user-supplied key.
    if let Some(pos) = args.iter().position(|a| a == "--load-sprx") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--load-sprx requires a path to a .sprx/SELF file"))?;
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        if let Some(output) = std::env::var_os("RAEEN_DUMP_DECRYPTED_ELF") {
            std::fs::write(&output, &decrypted.elf)?;
            println!(
                "decrypted ELF: {} byte(s) -> {}",
                decrypted.elf.len(),
                std::path::Path::new(&output).display()
            );
        }
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let dyn_tags = match &module.dynamic {
            Some(d) => raeen_firmware::dynlib::parse_sce_dynamic(d)?,
            None => Vec::new(),
        };
        // Two dynamic models: the PT_SCE_DYNLIBDATA blob (homebrew/.sprx) or
        // standard vaddr-based tags with no such segment (real PS5 titles).
        let standard = raeen_firmware::dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
        let dynlib_data = match &standard {
            Some((image, tags)) => raeen_firmware::dynlib::parse_dynlibdata(image, tags)?,
            None => raeen_firmware::dynlib::parse_dynlibdata(
                module.dynlib_data.as_deref().unwrap_or(&[]),
                &dyn_tags,
            )?,
        };
        if std::env::var_os("RAEEN_DUMP_TLS_RELOCS").is_some() {
            use std::collections::HashMap;

            let modules: HashMap<u16, &str> = dynlib_data
                .import_modules
                .iter()
                .map(|(index, name)| (*index, name.as_str()))
                .collect();
            let libraries: HashMap<u16, &str> = dynlib_data
                .import_libs
                .iter()
                .map(|(index, name)| (*index, name.as_str()))
                .collect();
            let mut count = 0usize;
            for relocation in &dynlib_data.relocations {
                let r_type = relocation.info as u32;
                if !matches!(r_type, 16..=18) {
                    continue;
                }
                count += 1;
                let symbol_index = (relocation.info >> 32) as usize;
                let symbol = dynlib_data.symbols.get(symbol_index);
                let provider = dynlib_data
                    .symbol_providers
                    .get(symbol_index)
                    .and_then(|provider| *provider);
                let module_name = provider
                    .and_then(|provider| modules.get(&provider.module_index).copied())
                    .unwrap_or("<local>");
                let library_name = provider
                    .and_then(|provider| libraries.get(&provider.library_index).copied())
                    .unwrap_or("<local>");
                println!(
                    "tls relocation off={:#x} type={} sym={} addend={:#x} value={:#x} \
                     import={} nid={:#018x} provider={module_name}::{library_name}",
                    relocation.offset,
                    r_type,
                    symbol_index,
                    relocation.addend,
                    symbol.map_or(0, |symbol| symbol.value),
                    symbol.is_some_and(|symbol| symbol.is_import),
                    symbol.map_or(0, |symbol| symbol.nid),
                );
            }
            println!("TLS relocations: {count}");
        }
        if let Ok(value) = std::env::var("RAEEN_RELOC_OFFSET") {
            let value = value.trim_start_matches("0x");
            if let Ok(offset) = u64::from_str_radix(value, 16) {
                for relocation in dynlib_data
                    .relocations
                    .iter()
                    .filter(|relocation| relocation.offset == offset)
                {
                    let symbol_index = (relocation.info >> 32) as usize;
                    let r_type = relocation.info as u32;
                    match dynlib_data.symbols.get(symbol_index) {
                        Some(symbol) => println!(
                            "relocation {offset:#x}: type={r_type} sym={symbol_index} \
                             nid={:#018x} ({}) value={:#x} import={} {}",
                            symbol.nid,
                            raeen_firmware::dynlib::nid::encode_nid(symbol.nid),
                            symbol.value,
                            symbol.is_import,
                            raeen_firmware::dynlib::nid_names::describe(symbol.nid),
                        ),
                        None => println!(
                            "relocation {offset:#x}: type={r_type} invalid sym={symbol_index}"
                        ),
                    }
                }
            }
        }
        let hle = std::sync::Arc::new(raeen_hle::HleRegistry::new());
        let db = raeen_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        let mut registry = raeen_firmware::ModuleRegistry::new(db);
        registry.register_module_exports(&module.name, &dynlib_data.exports);
        let linked = raeen_firmware::link_module(&module, &dynlib_data, &registry, &hle, 0)?;
        println!("module: {}", module.name);
        println!(
            "imports: {}  exports: {}",
            dynlib_data.imports.len(),
            dynlib_data.exports.len()
        );
        // The export *table* and the symbol table are different things, and the
        // gap between them is where Minecraft's boot dies: a symbol this module
        // defines but never publishes as an export is invisible to dlsym. If
        // `defined` here materially exceeds `exports`, the export parse is
        // dropping symbols the guest can legitimately ask for.
        let defined = dynlib_data.symbols.iter().filter(|s| !s.is_import).count();
        println!(
            "dynsym: {} symbol(s) — {} defined, {} imported",
            dynlib_data.symbols.len(),
            defined,
            dynlib_data.symbols.len() - defined
        );
        // What a module exports is the whole contract a dependent links against,
        // and "41 exports" says nothing about whether the three symbols a title
        // actually wants are among them. Minecraft's boot dies on exactly that
        // gap: MediaDecoders parses 41 exports and none is CreateMP3Decoder.
        for export in &dynlib_data.exports {
            println!(
                "  export nid={:#018x} ({}) vaddr={:#x}  {}",
                export.nid,
                raeen_firmware::dynlib::nid::encode_nid(export.nid),
                export.value,
                raeen_firmware::dynlib::nid_names::describe(export.nid)
            );
        }
        println!(
            "resolved HLE trampolines: {}  unresolved: {}",
            linked.hle_trampolines.len(),
            linked.unresolved.len()
        );

        // Turn "N unresolved" into an actionable list. A NID is a one-way hash,
        // so an unresolved one can't be turned back into a function name — but
        // each import symbol carries a library_index, and DT_SCE_IMPORT_LIB_1
        // maps that to a real library name.
        //
        // Two things this must NOT do, both of which produced a badly wrong
        // headline before:
        //  * index library_index into the *needed-module* table (renames every
        //    library: "libc" became "libfmod"), and
        //  * report the raw relocation count as if it were a count of missing
        //    functions. It is one entry PER RELOCATION, and on a real C++ title
        //    a handful of RTTI symbols generate ~99% of them. The JUMP_SLOT
        //    subtotal is the number that actually sizes the HLE work.
        if !linked.unresolved.is_empty() {
            use std::collections::HashMap;
            const R_X86_64_64: u32 = 1;
            const R_X86_64_GLOB_DAT: u32 = 6;
            const R_X86_64_JUMP_SLOT: u32 = 7;

            let lib_names: HashMap<u16, &str> = dynlib_data
                .import_libs
                .iter()
                .map(|(id, n)| (*id, n.as_str()))
                .collect();
            let nid_to_lib: HashMap<u64, u16> = dynlib_data
                .imports
                .iter()
                .map(|s| (s.nid, s.library_index))
                .collect();

            let mut by_type: HashMap<u32, usize> = HashMap::new();
            // (relocations, distinct called NIDs) per library.
            let mut per_lib: HashMap<&str, usize> = HashMap::new();
            let mut called_per_lib: HashMap<&str, std::collections::HashSet<u64>> = HashMap::new();
            let mut unknown = 0usize;
            for u in &linked.unresolved {
                *by_type.entry(u.r_type).or_default() += 1;
                match nid_to_lib.get(&u.nid).and_then(|i| lib_names.get(i)) {
                    Some(name) => {
                        *per_lib.entry(name).or_default() += 1;
                        if u.r_type == R_X86_64_JUMP_SLOT {
                            called_per_lib.entry(name).or_default().insert(u.nid);
                        }
                    }
                    None => unknown += 1,
                }
            }

            println!("\nunresolved relocations by type:");
            let mut types: Vec<_> = by_type.into_iter().collect();
            types.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            for (t, n) in &types {
                let what = match *t {
                    R_X86_64_JUMP_SLOT => "JUMP_SLOT  - a function the guest CALLS",
                    R_X86_64_64 => "R_X86_64_64 - a data pointer slot (RTTI/vtable)",
                    R_X86_64_GLOB_DAT => "GLOB_DAT   - a data pointer slot",
                    _ => "other",
                };
                println!("  {n:>6}  {what}");
            }

            let called: usize = linked
                .unresolved
                .iter()
                .filter(|u| u.r_type == R_X86_64_JUMP_SLOT)
                .count();
            println!(
                "\nunresolved imports by library (relocations, then distinct CALLED functions).\n\
                 The second column is the real work: {called} function(s) are actually called."
            );
            let mut ranked: Vec<_> = per_lib.into_iter().collect();
            ranked.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            println!("  {:>8}  {:>8}  library", "relocs", "called");
            for (lib, n) in ranked.iter().take(20) {
                let c = called_per_lib.get(lib).map_or(0, |s| s.len());
                println!("  {n:>8}  {c:>8}  {lib}");
            }
            if unknown > 0 {
                println!("  {unknown:>8}  {:>8}  <library unknown>", "?");
            }
        }
        return Ok(());
    }

    // Diagnostic: `raeen --dump-vaddr <eboot.bin> <hex-vaddr> [<len>]` prints
    // hex + ASCII at a module-relative virtual address, straight from the
    // parsed segments without executing anything.
    //
    // This exists because a fault report can only print what was in registers
    // at the crash; the *static* strings an assert site references (its
    // expression text, source-file path) live at RIP-relative addresses the
    // report names but cannot read once the process is gone. This turns those
    // offsets into the message the game was trying to log.
    if let Some(pos) = args.iter().position(|a| a == "--dump-vaddr") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--dump-vaddr requires a path to an eboot.bin"))?;
        let vaddr = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--dump-vaddr requires a hex vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad vaddr {s:?}: {e}"))
            })?;
        let len = match args.get(pos + 3) {
            Some(s) => s.parse::<usize>()?,
            None => 256,
        };
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let segment = module
            .segments
            .iter()
            .find(|s| vaddr >= s.vaddr && vaddr < s.vaddr + s.mem_size)
            .ok_or_else(|| anyhow::anyhow!("vaddr {vaddr:#x} is not in any PT_LOAD segment"))?;
        let start = (vaddr - segment.vaddr) as usize;
        let file_backed = segment.data.len().saturating_sub(start);
        if file_backed == 0 {
            println!(
                "vaddr {vaddr:#x} is in BSS (zero-initialized) of segment {:#x}",
                segment.vaddr
            );
            return Ok(());
        }
        let slice = &segment.data[start..start + len.min(file_backed)];
        for (i, chunk) in slice.chunks(16).enumerate() {
            let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
            let ascii: String = chunk
                .iter()
                .map(|&b| {
                    if (0x20..0x7f).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            println!(
                "{:#014x}  {:<47}  {ascii}",
                vaddr + (i * 16) as u64,
                hex.join(" ")
            );
        }
        return Ok(());
    }

    // Diagnostic: `raeen --disas <eboot.bin> <hex-vaddr> [<len-decimal>]`
    // disassembles x86-64 from a module vaddr — the missing piece for RE'ing
    // guest boot-decision logic (e.g. Minecraft's never-taken CreateView
    // branch). Reuses the workspace iced-x86 decoder the VEH already links.
    if let Some(pos) = args.iter().position(|a| a == "--disas") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--disas requires a path to an eboot.bin"))?;
        let vaddr = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--disas requires a hex vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad vaddr {s:?}: {e}"))
            })?;
        let len = match args.get(pos + 3) {
            Some(s) => s.parse::<usize>()?,
            None => 256,
        };
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let segment = module
            .segments
            .iter()
            .find(|s| vaddr >= s.vaddr && vaddr < s.vaddr + s.mem_size)
            .ok_or_else(|| anyhow::anyhow!("vaddr {vaddr:#x} is not in any PT_LOAD segment"))?;
        let start = (vaddr - segment.vaddr) as usize;
        let file_backed = segment.data.len().saturating_sub(start);
        if file_backed == 0 {
            println!("vaddr {vaddr:#x} is in BSS");
            return Ok(());
        }
        let slice = &segment.data[start..start + len.min(file_backed)];
        use iced_x86::Formatter as _;
        let mut decoder =
            iced_x86::Decoder::with_ip(64, slice, vaddr, iced_x86::DecoderOptions::NONE);
        let mut formatter = iced_x86::IntelFormatter::new();
        let mut out = String::new();
        let mut inst = iced_x86::Instruction::default();
        while decoder.can_decode() {
            decoder.decode_out(&mut inst);
            out.clear();
            formatter.format(&inst, &mut out);
            // Flag control-flow so a branch that gates a subsystem stands out.
            // Keyed off the formatted mnemonic to avoid the `code_asm`/flow
            // feature; conditional jumps start with "j" but not "jmp".
            let m = out.split_whitespace().next().unwrap_or("");
            let mark = if m == "call" {
                " (call)"
            } else if m == "ret" {
                " (ret)"
            } else if m == "jmp" {
                " (jmp)"
            } else if m.starts_with('j') {
                " <-- COND"
            } else if m == "test" || m == "cmp" {
                " <-- TEST"
            } else {
                ""
            };
            println!("{:#014x}  {out}{mark}", inst.ip());
        }
        return Ok(());
    }

    // Diagnostic: `raeen --find-calls <eboot.bin> <hex-target-vaddr>` scans every
    // executable segment for a call whose target is the given vaddr — the core
    // "who calls X" RE operation. Finds direct `call rel32` (e8) and
    // `call [rip+disp32]` (ff 15) through a GOT slot holding the target. Prints
    // each call-site vaddr, so the guarding branch upstream can be disassembled.
    if let Some(pos) = args.iter().position(|a| a == "--find-calls") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--find-calls requires an eboot.bin path"))?;
        let target = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--find-calls requires a hex target vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad target {s:?}: {e}"))
            })?;
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let mut hits = 0usize;
        for seg in &module.segments {
            // Executable segments only (PF_X = bit 0).
            if seg.flags & 1 == 0 {
                continue;
            }
            let data = &seg.data;
            let base = seg.vaddr;
            // Direct `call rel32`: e8 <rel32>. Target = insn_end + rel32.
            for i in 0..data.len().saturating_sub(5) {
                if data[i] == 0xe8 {
                    let rel =
                        i32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
                    let site = base + i as u64;
                    let dest = (site + 5).wrapping_add(rel as i64 as u64);
                    if dest == target {
                        println!("{site:#014x}  call {target:#x} (direct)");
                        hits += 1;
                    }
                }
            }
        }
        eprintln!("{hits} direct call site(s) to {target:#x}");
        return Ok(());
    }

    // Diagnostic: `raeen --find-lea <eboot.bin> <hex-target-vaddr>` scans every
    // executable segment for `lea reg, [rip+disp32]` whose target is the given
    // vaddr — "who references this data/string". Pairs with --find-calls: locate
    // a .rodata string (e.g. "coui://", "index.html") then find the code that
    // loads its address to build a navigate URL.
    if let Some(pos) = args.iter().position(|a| a == "--find-lea") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--find-lea requires an eboot.bin path"))?;
        let target = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--find-lea requires a hex target vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad target {s:?}: {e}"))
            })?;
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let mut hits = 0usize;
        for seg in &module.segments {
            if seg.flags & 1 == 0 {
                continue;
            }
            let data = &seg.data;
            let base = seg.vaddr;
            // REX.W lea rip-relative: 48 8d <modrm> disp32, modrm in {05,0d,15,
            // 1d,25,2d,35,3d} (mod=00, rm=101). insn length = 7. Target =
            // (site+7) + disp32.
            for i in 0..data.len().saturating_sub(7) {
                if data[i] == 0x48 && data[i + 1] == 0x8d && (data[i + 2] & 0xc7) == 0x05 {
                    let disp =
                        i32::from_le_bytes([data[i + 3], data[i + 4], data[i + 5], data[i + 6]]);
                    let site = base + i as u64;
                    let dest = (site + 7).wrapping_add(disp as i64 as u64);
                    if dest == target {
                        let reg = (data[i + 2] >> 3) & 7;
                        println!("{site:#014x}  lea r{reg}, [{target:#x}]");
                        hits += 1;
                    }
                }
            }
        }
        eprintln!("{hits} lea reference(s) to {target:#x}");
        return Ok(());
    }

    // Diagnostic: `raeen --find-str <eboot.bin> <needle>` searches the decrypted
    // image segments for an ASCII substring and prints each match's vaddr — the
    // string can then be fed to --find-lea to locate the code that references
    // it. (Raw `strings` fails: the eboot is an encrypted SELF.)
    if let Some(pos) = args.iter().position(|a| a == "--find-str") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--find-str requires an eboot.bin path"))?;
        let needle = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--find-str requires a search string"))?
            .as_bytes();
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let mut hits = 0usize;
        for seg in &module.segments {
            let data = &seg.data;
            let mut i = 0usize;
            while i + needle.len() <= data.len() {
                if &data[i..i + needle.len()] == needle {
                    let vaddr = seg.vaddr + i as u64;
                    // Show a little context so the exact string is visible.
                    let end = (i + needle.len() + 24).min(data.len());
                    let ctx: String = data[i..end]
                        .iter()
                        .map(|&b| {
                            if (0x20..0x7f).contains(&b) {
                                b as char
                            } else {
                                '.'
                            }
                        })
                        .collect();
                    println!("{vaddr:#014x}  {ctx:?}");
                    hits += 1;
                    if hits >= 64 {
                        break;
                    }
                }
                i += 1;
            }
        }
        eprintln!("{hits} match(es) for {:?}", String::from_utf8_lossy(needle));
        return Ok(());
    }

    // Diagnostic: `raeen --resolve-got <eboot.bin> <hex-got-vaddr>...` maps a
    // PLT/GOT slot vaddr (the `jmp qword [rip+disp]` target of an import thunk)
    // back to the import symbol it binds — NID, recovered name, and library.
    // A JMPREL relocation's `offset` *is* the GOT slot vaddr, so a call site's
    // `ff 25 <disp>` thunk can be turned into "which libc function is this".
    if let Some(pos) = args.iter().position(|a| a == "--resolve-got") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--resolve-got requires a path to an eboot.bin"))?;
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let dyn_tags = match &module.dynamic {
            Some(d) => raeen_firmware::dynlib::parse_sce_dynamic(d)?,
            None => Vec::new(),
        };
        // Real PS5 titles use the standard dynamic model (vaddr-addressed
        // tables), not the `PT_SCE_DYNLIBDATA` blob — mirror `load_module`.
        let standard = raeen_firmware::dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
        let dynlib = match &standard {
            Some((image, tags)) => raeen_firmware::dynlib::parse_dynlibdata(image, tags)?,
            None => raeen_firmware::dynlib::parse_dynlibdata(
                module.dynlib_data.as_deref().unwrap_or(&[]),
                &dyn_tags,
            )?,
        };
        for gs in &args[pos + 2..] {
            let Ok(target) = u64::from_str_radix(gs.trim_start_matches("0x"), 16) else {
                continue;
            };
            match dynlib.relocations.iter().find(|r| r.offset == target) {
                Some(r) => {
                    let symidx = (r.info >> 32) as usize;
                    let nid = dynlib.symbols.get(symidx).map_or(0, |s| s.nid);
                    let lib = dynlib
                        .imports
                        .iter()
                        .find(|i| i.nid == nid)
                        .and_then(|i| {
                            dynlib
                                .import_libs
                                .iter()
                                .find(|(idx, _)| *idx == i.library_index)
                        })
                        .map_or("?", |(_, n)| n.as_str());
                    println!(
                        "GOT {target:#x} -> sym#{symidx} nid={nid:#018x} {} [{lib}] rtype={}",
                        raeen_firmware::dynlib::nid_names::describe(nid),
                        r.info & 0xffff_ffff
                    );
                }
                None => {
                    println!("GOT {target:#x} -> no exact relocation; nearest by offset:");
                    let mut near: Vec<&raeen_firmware::dynlib::SceRela> =
                        dynlib.relocations.iter().collect();
                    near.sort_by_key(|r| r.offset.abs_diff(target));
                    for r in near.iter().take(4) {
                        let symidx = (r.info >> 32) as usize;
                        let nid = dynlib.symbols.get(symidx).map_or(0, |s| s.nid);
                        println!(
                            "    off={:#x} (Δ{:#x}) sym#{symidx} {} rtype={}",
                            r.offset,
                            r.offset.abs_diff(target),
                            raeen_firmware::dynlib::nid_names::describe(nid),
                            r.info & 0xffff_ffff
                        );
                    }
                }
            }
        }
        println!(
            "[{} relocations total; offset range {:#x}..={:#x}]",
            dynlib.relocations.len(),
            dynlib
                .relocations
                .iter()
                .map(|r| r.offset)
                .min()
                .unwrap_or(0),
            dynlib
                .relocations
                .iter()
                .map(|r| r.offset)
                .max()
                .unwrap_or(0),
        );
        return Ok(());
    }

    // Diagnostic: `raeen --missing-nids <eboot.bin>` loads a title exactly as
    // `--run-eboot` does but stops before executing, and prints every DISTINCT
    // import nothing resolves — encoded NID, raw NID, and library — grouped by
    // library and ranked.
    //
    // This exists because a NID is a one-way hash: "352 unresolved" is a number
    // nobody can act on, while `n88vx3C5nW8 (libScePosix)` can be brute-forced
    // back to `gettimeofday` and implemented. It is the input to sizing the
    // remaining HLE work at all.
    if let Some(pos) = args.iter().position(|a| a == "--missing-nids") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--missing-nids requires a path to an eboot.bin"))?;
        let bytes = std::fs::read(path)?;
        let hle = std::sync::Arc::new(raeen_hle::HleRegistry::new());
        let db = raeen_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        let mut registry = raeen_firmware::ModuleRegistry::new(db);
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let process = raeen_firmware::load_process(
            &bytes,
            dir,
            &raeen_firmware::NoKeysProvider,
            &mut registry,
            &hle,
            raeen_runtime::GUEST_ARENA_BASE,
        )?;

        use std::collections::BTreeMap;
        let mut by_lib: BTreeMap<&str, Vec<&raeen_firmware::UnresolvedStub>> = BTreeMap::new();
        for s in &process.linked.unresolved_stubs {
            by_lib
                .entry(s.library.as_deref().unwrap_or("<unknown library>"))
                .or_default()
                .push(s);
        }
        let mut ranked: Vec<_> = by_lib.into_iter().collect();
        ranked.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));

        println!(
            "# {} distinct unresolved import NID(s) across {} librar(ies)",
            process.linked.unresolved_stubs.len(),
            ranked.len()
        );
        // `name` is hash-verified when known (see dynlib::nid_names), else it
        // repeats the encoded NID — an anonymous import nothing can name yet.
        println!("# encoded_nid  nid  library  name");
        for (lib, stubs) in ranked {
            let named = stubs
                .iter()
                .filter(|s| raeen_firmware::dynlib::nid_names::name_of(s.nid).is_some())
                .count();
            println!("\n## {lib}  ({} missing, {named} named)", stubs.len());
            let mut rows: Vec<String> = stubs
                .iter()
                .map(|s| {
                    format!(
                        "{}  {:#018x}  {lib}  {}",
                        raeen_firmware::dynlib::nid::encode_nid(s.nid),
                        s.nid,
                        raeen_firmware::dynlib::nid_names::describe(s.nid),
                    )
                })
                .collect();
            rows.sort();
            for r in rows {
                println!("{r}");
            }
        }
        return Ok(());
    }

    // Diagnostic: `raeen --imports <eboot|sprx> [library-substring]` prints the
    // module's FULL import table grouped by importing library — resolved and
    // unresolved alike — unlike `--missing-nids`, which shows only what the HLE
    // registry lacks. The optional filter is a case-insensitive substring match
    // on the library name (e.g. `agc`, `videoout`). This is what answers "which
    // graphics API surface does this title actually call?" for a title whose
    // AGC imports are already implemented and therefore invisible to
    // --missing-nids.
    if let Some(pos) = args.iter().position(|a| a == "--imports") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--imports requires a path to an eboot/.sprx"))?;
        let filter = args.get(pos + 2).map(|s| s.to_ascii_lowercase());
        let bytes = std::fs::read(path)?;
        let decrypted = raeen_firmware::decrypt_self(&bytes, &raeen_firmware::NoKeysProvider)?;
        let module = raeen_firmware::parse_sprx(&decrypted.elf)?;
        let dyn_tags = match &module.dynamic {
            Some(d) => raeen_firmware::dynlib::parse_sce_dynamic(d)?,
            None => Vec::new(),
        };
        let standard = raeen_firmware::dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
        let dynlib_data = match &standard {
            Some((image, tags)) => raeen_firmware::dynlib::parse_dynlibdata(image, tags)?,
            None => raeen_firmware::dynlib::parse_dynlibdata(
                module.dynlib_data.as_deref().unwrap_or(&[]),
                &dyn_tags,
            )?,
        };
        let hle = std::sync::Arc::new(raeen_hle::HleRegistry::new());
        let db = raeen_firmware::dynlib::nid::NidDatabase::from_hle(&hle);

        use std::collections::BTreeMap;
        let lib_names: std::collections::HashMap<u16, &str> = dynlib_data
            .import_libs
            .iter()
            .map(|(i, n)| (*i, n.as_str()))
            .collect();
        let mut by_lib: BTreeMap<&str, Vec<&raeen_firmware::dynlib::SymbolRef>> = BTreeMap::new();
        for import in &dynlib_data.imports {
            let lib = lib_names
                .get(&import.library_index)
                .copied()
                .unwrap_or("<unknown library>");
            if let Some(f) = &filter
                && !lib.to_ascii_lowercase().contains(f.as_str())
            {
                continue;
            }
            by_lib.entry(lib).or_default().push(import);
        }
        println!(
            "# {}: {} import(s) across {} librar(ies){}",
            module.name,
            by_lib.values().map(Vec::len).sum::<usize>(),
            by_lib.len(),
            filter
                .as_deref()
                .map(|f| format!(" (filter: {f})"))
                .unwrap_or_default(),
        );
        for (lib, imports) in by_lib {
            let hle_count = imports
                .iter()
                .filter(|s| db.resolve_for_provider(lib, s.nid).is_some())
                .count();
            println!(
                "\n## {lib}  ({} imports, {hle_count} HLE-resolved)",
                imports.len()
            );
            let mut rows: Vec<String> = imports
                .iter()
                .map(|s| {
                    let status = if db.resolve_for_provider(lib, s.nid).is_some() {
                        "HLE "
                    } else if db.resolve(s.nid).is_some() {
                        "HLE(other-lib) "
                    } else {
                        "MISSING "
                    };
                    format!(
                        "{status}{}  {:#018x}  {}",
                        raeen_firmware::dynlib::nid::encode_nid(s.nid),
                        s.nid,
                        raeen_firmware::dynlib::nid_names::describe(s.nid),
                    )
                })
                .collect();
            rows.sort();
            for r in rows {
                println!("{r}");
            }
        }
        return Ok(());
    }

    // Diagnostic: `raeen --run-eboot <eboot.bin>` drives the **real** launch
    // path headlessly — exactly what the Shell does (`load_module` at
    // `GUEST_ARENA_BASE`, then `execute_process` into the module's `_start` on a
    // genuine argc/argv/envp/auxv stack) — and reports the outcome. Same
    // `NoKeysProvider`: it never decrypts anything without a user-supplied key.
    // This exists so a real title's execution can be observed (and its logs
    // read) without standing up the GUI.
    if let Some(pos) = args.iter().position(|a| a == "--run-eboot") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--run-eboot requires a path to an eboot.bin"))?;
        // The Shell executes retail titles in this child process. Process-local
        // subsystem state configured by the Shell does not cross that boundary:
        // without bringing up cpal here, every sceAudioOut/AudioOut2 submission
        // was silently dropped because the host ring had never been created.
        // Load the same persisted settings as the Shell so mute and master
        // volume remain authoritative in both launch modes.
        // Crash reporting: when launched by the Shell, connect back to its
        // minidump server so a fatal fault in this guest-executing process
        // produces a dump under logs/crashes/ instead of a silent death.
        if let Ok(socket) = std::env::var("RAEEN_CRASH_SOCKET") {
            crashdump::attach_client(&socket);
        }
        let runner_config =
            raeen_core::config::EmulatorConfig::load(std::path::Path::new("config.toml"))?;
        raeen_gpu::AgcGpuSession::set_runtime_config(
            runner_config.graphics.validation_layers,
            runner_config.graphics.resolution_scale,
            runner_config.graphics.gpu_device_index,
            runner_config.graphics.shader_cache,
            runner_config.paths.shader_cache_dir.clone(),
        );
        raeen_audio::output::set_volume(runner_config.audio.volume);
        raeen_audio::output::set_enabled(runner_config.audio.enabled);
        raeen_audio::output::init();
        info!(
            enabled = runner_config.audio.enabled,
            volume = runner_config.audio.volume,
            "runner host audio configured"
        );
        let bytes = std::fs::read(path)?;
        let hle = std::sync::Arc::new(raeen_hle::HleRegistry::new());
        let db = raeen_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        let mut registry = raeen_firmware::ModuleRegistry::new(db);
        let kernel = std::sync::Arc::new(raeen_kernel::OrbisKernel::new());
        // The isolated runner cannot borrow the Shell's in-process kernel.
        // Read the Shell's merged native/gilrs/keyboard state from the
        // bidirectional frame mapping. This avoids opening a second raw-HID
        // reader in the child, which can consume DualSense reports before the
        // Shell sees them. A child-native reader remains the fallback for
        // direct runner launches without the Shell bridge.
        if std::env::var_os("RAEEN_RUNNER_CHILD").is_some() {
            let input_kernel = std::sync::Arc::clone(&kernel);
            let rumble_enabled = runner_config.input.dualsense_features;
            let input_script = std::env::var("RAEEN_INPUT_SCRIPT")
                .ok()
                .map(|spec| raeen_input::InputScript::parse(&spec))
                .transpose()
                .map_err(|error| anyhow::anyhow!("invalid RAEEN_INPUT_SCRIPT: {error}"))?;
            std::thread::Builder::new()
                .name("raeen-runner-input".to_string())
                .spawn(move || {
                    let shared_input = raeen_gpu::frame_ipc::FrameIpcInputReader::open_from_env();
                    let pads = shared_input
                        .is_none()
                        .then(raeen_input::NativeGamepads::start);
                    let started = std::time::Instant::now();
                    if let Some(script) = input_script.as_ref() {
                        tracing::info!(
                            events = script.len(),
                            "runner enabled deterministic controller input replay"
                        );
                    }
                    let mut last_pad_state = None;
                    // Guest → host rumble return path. With the Shell bridge
                    // the child only forwards the encoded word — the Shell
                    // owns hardware delivery, its Settings gate, and the
                    // safety auto-stop. Direct (bridgeless) runs drive the
                    // child's own native pads through the same router rules.
                    let mut last_rumble_motors: Option<(u8, u8)> = None;
                    let mut rumble_router = raeen_input::rumble::RumbleRouter::new();
                    loop {
                        let scripted = input_script
                            .as_ref()
                            .and_then(|script| script.state_at(started.elapsed()));
                        let is_scripted = scripted.is_some();
                        let encoded = if let Some(state) = scripted {
                            state.to_orbis_pad_data()
                        } else if let Some(state) =
                            shared_input.as_ref().and_then(|input| input.latest())
                        {
                            state
                        } else {
                            pads.as_ref()
                                .and_then(raeen_input::NativeGamepads::poll)
                                .unwrap_or_default()
                                .to_orbis_pad_data()
                        };
                        let buttons =
                            u32::from_le_bytes(encoded[0..4].try_into().expect("pad prefix"));
                        if last_pad_state.as_ref() != Some(&encoded) {
                            tracing::info!(
                                elapsed_ms = started.elapsed().as_millis(),
                                buttons = format_args!("{buttons:#010x}"),
                                left_x = encoded[4],
                                left_y = encoded[5],
                                right_x = encoded[6],
                                right_y = encoded[7],
                                l2 = encoded[8],
                                r2 = encoded[9],
                                source = if is_scripted {
                                    "script"
                                } else if shared_input.is_some() {
                                    "shell-ipc"
                                } else {
                                    "child-native"
                                },
                                "runner applied controller state"
                            );
                            last_pad_state = Some(encoded);
                        }
                        input_kernel.set_pad_state(encoded);
                        // Rumble return path (see the declaration above).
                        let (rumble_seq, large, small) = input_kernel.pad_rumble();
                        if let Some(input) = shared_input.as_ref() {
                            // Log motor-value changes only — the sequence
                            // bumps on every guest keep-alive call.
                            if rumble_seq != 0 && last_rumble_motors != Some((large, small)) {
                                tracing::info!(
                                    large,
                                    small,
                                    "guest vibration forwarded to the Shell"
                                );
                                last_rumble_motors = Some((large, small));
                            }
                            input.publish_rumble_word(raeen_input::rumble::encode_word(
                                rumble_seq,
                                raeen_input::rumble::RumbleState::new(large, small),
                            ));
                        } else if let Some(pads) = pads.as_ref() {
                            let source = (rumble_seq != 0).then(|| {
                                (
                                    rumble_seq,
                                    raeen_input::rumble::RumbleState::new(large, small),
                                )
                            });
                            if let Some(command) =
                                rumble_router.update(started.elapsed(), source, rumble_enabled)
                            {
                                tracing::info!(
                                    large = command.large,
                                    small = command.small,
                                    "guest vibration routed to controller (direct runner)"
                                );
                                pads.set_rumble(command.large, command.small);
                            }
                        }
                        std::thread::sleep(std::time::Duration::from_millis(4));
                    }
                })?;
        }
        // Load as a whole process: the eboot plus every DT_NEEDED .prx that
        // ships beside it (M1-D). A real title's imports are overwhelmingly
        // satisfied by those bundled libraries, not by HLE.
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        // `/app0` is the directory containing the selected title's eboot,
        // for every title and every host layout. Never leave the VFS on its
        // placeholder `games/current` mount during a real launch.
        kernel.filesystem.set_game_directory(dir);
        let title_dir = dir.file_name().unwrap_or_default();
        let writable_root = std::env::temp_dir().join("raeen").join(title_dir);
        let temp_dir = writable_root.join("temp");
        let download_dir = writable_root.join("download");
        let savedata_dir = std::path::Path::new("savedata").join(title_dir);
        for writable_dir in [&temp_dir, &download_dir, &savedata_dir] {
            std::fs::create_dir_all(writable_dir)?;
        }
        // Bedrock writes this zero-byte marker while global resource packs are
        // being initialized. A killed isolated runner cannot execute the
        // title's normal unlink, so the next boot interprets the orphan as a
        // prior resource crash and raises "Global Resources Reset" even though
        // every packaged asset is present. There is no payload to preserve:
        // recover only this exact empty session marker, only while no prior
        // title process is alive (we are inside the newly-created runner).
        let recovered_locks = recover_stale_resource_init_locks(&savedata_dir)?;
        if recovered_locks > 0 {
            info!(
                recovered_locks,
                root = %savedata_dir.display(),
                "recovered stale global-resource initialization marker(s)"
            );
        }
        kernel.filesystem.set_temp_directory(&temp_dir);
        kernel.filesystem.set_download_directory(&download_dir);
        kernel.filesystem.set_savedata_directory(&savedata_dir);
        let process = raeen_firmware::load_process(
            &bytes,
            dir,
            &raeen_firmware::NoKeysProvider,
            &mut registry,
            &hle,
            raeen_runtime::GUEST_ARENA_BASE,
        )?;
        for d in &process.dependencies {
            info!(
                "  dep {} at +{:#x}: {} exports, {} unresolved",
                d.name, d.image_offset, d.exports, d.unresolved
            );
        }
        // Diagnostic: RAEEN_TRAP_CXA_THROW patches the eboot's STATICALLY-LINKED
        // __cxa_throw (which is not an import, so it can't be redirected by the
        // linker) so the C++ exception a title's worker threads throw gets
        // NAMED. The function is found by its relocation-free prologue
        // fingerprint (matched against libc.prx's copy); each hit's entry is
        // overwritten with `movabs rax, <trampoline>; jmp rax` into an appended
        // libc::__cxa_throw HLE trampoline. __cxa_throw is noreturn, so the
        // trap only reads tinfo and exits the thread — the original prologue is
        // never needed. Gated so normal runs are untouched.
        let mut linked = process.linked;
        if std::env::var_os("RAEEN_TRAP_CXA_THROW").is_some() {
            // Prologue of __cxa_throw up to its first relative call — the
            // relocation-free bytes are a reliable fingerprint (see libc.prx
            // +0x18a30): push rbp; mov rbp,rsp; push r15..rbx; push rax;
            // mov r14,rdx; mov r15,rsi; mov r12,rdi.
            const PROLOGUE: &[u8] = &[
                0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50,
                0x49, 0x89, 0xd6, 0x49, 0x89, 0xf7, 0x49, 0x89, 0xfc,
            ];
            let tramp_addr = raeen_firmware::dynlib::linker::HLE_TRAMPOLINE_BASE
                + (linked.hle_trampolines.len() as u64) * 8;
            linked.hle_trampolines.push(raeen_firmware::HleTrampoline {
                library: "libc".to_string(),
                function: "__cxa_throw".to_string(),
                addr: tramp_addr,
            });
            let mut patch = Vec::with_capacity(12);
            patch.extend_from_slice(&[0x48, 0xb8]); // movabs rax, imm64
            patch.extend_from_slice(&tramp_addr.to_le_bytes());
            patch.extend_from_slice(&[0xff, 0xe0]); // jmp rax
            let mut patched = 0usize;
            let mut search = 0usize;
            while let Some(rel) = linked.image[search..]
                .windows(PROLOGUE.len())
                .position(|w| w == PROLOGUE)
            {
                let at = search + rel;
                linked.image[at..at + patch.len()].copy_from_slice(&patch);
                info!(
                    "__cxa_throw trap: patched eboot __cxa_throw at guest {:#x} -> trampoline {:#x}",
                    raeen_runtime::GUEST_ARENA_BASE + at as u64,
                    tramp_addr
                );
                patched += 1;
                search = at + patch.len();
            }
            info!(
                "__cxa_throw trap: patched {patched} internal copies, trampoline at {tramp_addr:#x}"
            );
        }
        // Diagnostic: RAEEN_TRAP_MSPACE installs a log-and-continue detour on the
        // title's native libc `sceLibcMspaceCreate` impl (`libc.prx+0xbe50`) so
        // its args + return value are logged — it returns null over valid memory
        // under our native execution while succeeding interpreted in SharpEmu.
        // The impl's prologue is `push rbp; mov rbp,rsp; push r15/r14/r13/r12/rbx`
        // = 13 relocation-free bytes, safe to relocate into the continuation stub.
        if std::env::var_os("RAEEN_TRAP_MSPACE").is_some() {
            if let Some(libc) = process
                .dependencies
                .iter()
                .find(|d| d.name.contains("libc"))
            {
                // (libc.prx offset, whole-prologue bytes, label). Create impl at
                // 0xbe50 and the sceLibcMspaceFree wrapper at 0xf600 — both start
                // with relocation-free register pushes + a reg-reg mov.
                for (off, plen, name) in [
                    (0xbe50u64, 13usize, "MspaceCreate"),
                    (0xf600, 13, "MspaceFree"),
                ] {
                    raeen_runtime::native_trap::install(
                        &mut linked.image,
                        &mut linked.hle_trampolines,
                        libc.image_offset + off,
                        plen,
                        name,
                    );
                }
            } else {
                info!("RAEEN_TRAP_MSPACE: no libc dependency found to trap");
            }
        } else {
            // Permanent, always-on fix (not gated behind the diagnostic detour):
            // resolve the title's native `sceLibcMspaceFree` by NID from the
            // linked libc module and install the null-mspace-free guard on it, so
            // the title's `sceLibcMspaceFree(0, ptr)` returns 0 instead of the
            // retail impl faulting on the null. Title-agnostic and zero-overhead —
            // the common (non-null) free path runs the real function directly.
            const SCE_LIBC_MSPACE_FREE_NID: u64 = 0x5656_bf67_e797_971a;
            if let Some(target) = linked
                .unwind_modules
                .iter()
                .find(|m| m.name.contains("libc"))
                .and_then(|m| {
                    m.exports
                        .iter()
                        .find(|e| e.nid == SCE_LIBC_MSPACE_FREE_NID)
                        .map(|e| m.image_offset + e.value)
                })
            {
                raeen_runtime::native_trap::install_null_free_guard(
                    &mut linked.image,
                    target,
                    13,
                    "sceLibcMspaceFree",
                );
            }
        }
        // Diagnostic: RAEEN_TRAP_MODULE_EXPORTS=<substring> plants a one-shot
        // `int3` on the entry byte of EVERY export of each matching loaded
        // module. The first call to each export logs module + NID + caller and
        // restores the byte permanently — near-zero overhead after the hit.
        // Answers "is this module ever actually entered, and from where"
        // (e.g. `=cohtml` to see whether the title drives its HTML engine).
        if let Ok(filter) = std::env::var("RAEEN_TRAP_MODULE_EXPORTS") {
            struct TrapTarget {
                name: String,
                image_offset: u64,
                exports: Vec<(u64, u64)>,
                exec_range: Option<(u64, u64)>,
            }
            let targets: Vec<TrapTarget> = linked
                .unwind_modules
                .iter()
                .filter(|m| raeen_runtime::export_trap::module_matches(&filter, &m.name))
                .map(|m| TrapTarget {
                    name: m.name.clone(),
                    image_offset: m.image_offset,
                    exports: m.exports.iter().map(|e| (e.nid, e.value)).collect(),
                    // First PT_LOAD = the text segment; exports outside it are
                    // data and must not be patched (see export_trap docs).
                    exec_range: (m.unwind.seg0_size > 0).then(|| {
                        (
                            m.unwind.seg0_vaddr,
                            m.unwind.seg0_vaddr + m.unwind.seg0_size,
                        )
                    }),
                })
                .collect();
            if targets.is_empty() {
                tracing::warn!(
                    "RAEEN_TRAP_MODULE_EXPORTS={filter}: no loaded module matches — nothing armed"
                );
            }
            for t in targets {
                raeen_runtime::export_trap::install_module_exports(
                    &mut linked.image,
                    raeen_runtime::GUEST_ARENA_BASE,
                    &t.name,
                    t.image_offset,
                    &t.exports,
                    t.exec_range,
                );
            }
        }
        // Diagnostic: RAEEN_TRAP_ADDR=<hex>[,<hex>...] plants a one-shot int3 at
        // each eboot-relative address — an RE probe for "does this code ever
        // execute", logging the caller when hit. Used to confirm whether a code
        // path (e.g. Minecraft's per-screen view-create loop) ever runs.
        if let Ok(list) = std::env::var("RAEEN_TRAP_ADDR") {
            let addrs: Vec<u64> = list
                .split(',')
                .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .collect();
            raeen_runtime::export_trap::install_addr_traps(
                &mut linked.image,
                raeen_runtime::GUEST_ARENA_BASE,
                &addrs,
            );
        }
        // `RAEEN_REPEAT_TRAP_ADDR` is the repeatable form for code paths that
        // execute more than once. The runtime accepts only instructions it can
        // emulate exactly without changing flags or memory (currently
        // `mov r32, imm32`), and skips every other site loudly.
        if let Ok(list) = std::env::var("RAEEN_REPEAT_TRAP_ADDR") {
            let addrs: Vec<u64> = list
                .split(',')
                .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .collect();
            raeen_runtime::export_trap::install_repeating_addr_traps(
                &mut linked.image,
                raeen_runtime::GUEST_ARENA_BASE,
                &addrs,
            );
        }
        let linked = std::sync::Arc::new(linked);
        info!(
            "loaded: entry={:#x} image={:#x} byte(s) resolved={} unresolved={}",
            linked.entry,
            linked.image.len(),
            linked.hle_trampolines.len(),
            linked.unresolved.len()
        );
        // The module's thread-local template. `tdata` is the part that must be
        // *copied* into every thread's block; the rest is `.tbss` and is zero.
        // A title whose thread-locals silently read as zero is the shape of a
        // whole class of null-dereference bugs, so say what the template is.
        match &linked.tls {
            Some(tls) => info!(
                "  PT_TLS: vaddr={:#x} tdata={:#x} memsz={:#x} align={:#x} init={:02x?}",
                tls.vaddr,
                tls.data.len(),
                tls.mem_size,
                tls.align,
                &tls.data[..tls.data.len().min(32)]
            ),
            None => info!("  PT_TLS: none"),
        }
        // Stall diagnosis: with RAEEN_STALL_DUMP set (and RAEEN_TRACE_EINVAL to
        // populate the ring), periodically log every guest thread's most recent
        // HLE calls. A thread blocked at a boot gate shows its last call frozen
        // or a tight spin — naming exactly what the game is waiting on.
        if std::env::var_os("RAEEN_STALL_DUMP").is_some() {
            let kmon = std::sync::Arc::clone(&kernel);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(6));
                    let mut lines: Vec<String> = kmon
                        .recent_hle_calls
                        .iter()
                        .map(|entry| {
                            let tid = *entry.key();
                            let name = kmon
                                .thread_names
                                .get(&tid)
                                .map_or_else(String::new, |n| n.clone());
                            let ring = entry.value().lock();
                            let recent: Vec<String> = ring.iter().rev().take(5).cloned().collect();
                            format!("t{tid}({name}): {}", recent.join(" <- "))
                        })
                        .collect();
                    lines.sort();
                    // The call ring goes blank when a thread spins in GUEST code
                    // (it calls nothing). RIP is the only thing that still says
                    // where it is — resolve it against the loaded modules so it
                    // reads as module+offset, which `--dump-vaddr` can decode.
                    let mut rips: Vec<String> = raeen_runtime::sample_guest_rips(&kmon)
                        .into_iter()
                        .map(|(id, rip)| {
                            let site = kmon.unwind_module_for_addr(rip).map_or_else(
                                || format!("{rip:#x}"),
                                |m| format!("{}+{:#x}", m.name, rip - m.start),
                            );
                            format!("t{id}@{site}")
                        })
                        .collect();
                    rips.sort();
                    // With RAEEN_TIME_HLE, name where each thread's wall-clock
                    // actually went. A thread whose top entry accounts for most
                    // of the run is parked in that one call — that is the thing
                    // to fix, and it is invisible in the call ring above.
                    // Per THREAD, not globally: every idle worker parks ~the
                    // whole run in scePthreadCondWait, so a global top-N is 12
                    // rows of "idle" and buries the one thread that matters.
                    // Report each thread's own biggest sink plus its total, so a
                    // busy thread (many short calls) is distinguishable at a
                    // glance from a parked one (all its time in a single wait).
                    let mut per_thread: std::collections::HashMap<u64, (u128, u128, u64, String)> =
                        std::collections::HashMap::new();
                    for e in kmon.hle_call_time.iter() {
                        let ((tid, func), (calls, micros)) = (e.key().clone(), *e.value());
                        let slot = per_thread.entry(tid).or_insert((0, 0, 0, String::new()));
                        slot.0 += micros; // total across all calls
                        if micros > slot.1 {
                            *slot = (slot.0, micros, calls, func);
                        }
                    }
                    let mut spent: Vec<(u128, String)> = per_thread
                        .into_iter()
                        .map(|(tid, (total, top_us, calls, func))| {
                            let name = kmon
                                .thread_names
                                .get(&tid)
                                .map_or_else(String::new, |n| n.clone());
                            (
                                total,
                                format!(
                                    "t{tid}({name}) total {:.1}s | top {func}: {:.1}s over {calls}",
                                    total as f64 / 1e6,
                                    top_us as f64 / 1e6
                                ),
                            )
                        })
                        .collect();
                    spent.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
                    let top: Vec<String> = spent.into_iter().map(|(_, s)| s).collect();
                    // The HLE call each thread is CURRENTLY inside (empty = not in
                    // one → blocked in guest code or runtime infra). Names exactly
                    // which blocking call a frozen thread never returns from.
                    let mut inflight: Vec<String> = kmon
                        .in_flight_hle
                        .iter()
                        .map(|e| format!("t{}={}", e.key(), e.value()))
                        .collect();
                    inflight.sort();
                    // Shallow host backtrace per thread (module+offset), so a
                    // thread parked in a host wait OUTSIDE any HLE call is shown
                    // with the call chain through our code that reached it.
                    let mut bt: Vec<String> = raeen_runtime::sample_host_backtraces(&kmon)
                        .into_iter()
                        .map(|(id, chain)| format!("t{id}: {chain}"))
                        .collect();
                    bt.sort();
                    // The title's OWN log output (its `write`s to fd 1/2) is the
                    // single most informative thing during a stall — it says what
                    // the game thinks it is doing. It is otherwise only printed
                    // when the run ends normally, which a hung title never does,
                    // so surface the tail here.
                    let console = kmon.console.contents();
                    let console_tail = if console.is_empty() {
                        "<empty>".to_owned()
                    } else {
                        let tail: String = console
                            .lines()
                            .rev()
                            .take(25)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("({} bytes, last 25 lines)\n{tail}", console.len())
                    };
                    info!(
                        "STALL_DUMP ({} threads):\n{}\nIN-FLIGHT HLE: {}\nHOST BACKTRACES:\n{}\nRIPs: {}{}\nGUEST CONSOLE: {}",
                        lines.len(),
                        lines.join("\n"),
                        if inflight.is_empty() {
                            "<none — all threads between calls>".to_owned()
                        } else {
                            inflight.join("  ")
                        },
                        bt.join("\n"),
                        rips.join(" "),
                        if top.is_empty() {
                            String::new()
                        } else {
                            format!("\nTIME IN HLE (top):\n{}", top.join("\n"))
                        },
                        console_tail
                    );
                }
            });
        }
        // Poll-gate diagnosis: with RAEEN_CALL_STATS set, the dispatch path
        // counts every HLE call per function, split into a boot window (first
        // 30 s) and steady state. Dump the top of each ranking periodically —
        // a title that polls a "not ready" value in short cycles puts the
        // polled function at the top of the STEADY window (timing/allocator
        // noise like clock_gettime ranks high too; the status queries below
        // them are the gate). Periodic, not at-exit: diagnosis runs are
        // usually killed hard (timeout -s KILL), so an exit hook never fires.
        if std::env::var_os("RAEEN_CALL_STATS").is_some() {
            let kmon = std::sync::Arc::clone(&kernel);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(10));
                    let mut boot: Vec<(u64, String)> = Vec::new();
                    let mut steady: Vec<(u64, String)> = Vec::new();
                    for e in kmon.hle_call_counts.iter() {
                        let (b, s) = e.value();
                        let b = b.load(std::sync::atomic::Ordering::Relaxed);
                        let s = s.load(std::sync::atomic::Ordering::Relaxed);
                        if b > 0 {
                            boot.push((b, e.key().clone()));
                        }
                        if s > 0 {
                            steady.push((s, e.key().clone()));
                        }
                    }
                    boot.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
                    steady.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
                    // Full list, not a top-N: the poll being hunted can be a
                    // per-frame call (40 Hz x 80 s ~ 3200) that a top-40 cut
                    // hides below allocator/timing noise, and the decisive
                    // signal is often a function present in STEADY but absent
                    // from BOOT (it started with a screen transition).
                    let render = |v: &[(u64, String)]| {
                        v.iter()
                            .map(|(n, f)| format!("  {n:>9}  {f}"))
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    info!(
                        "CALL_STATS t=+{:.0}s\nBOOT WINDOW (first 30s, {} distinct) top 40:\n{}\nSTEADY STATE (after 30s, {} distinct) top 40:\n{}",
                        kmon.uptime().as_secs_f64(),
                        boot.len(),
                        render(&boot),
                        steady.len(),
                        render(&steady),
                    );
                }
            });
        }
        // Same boot-splash staging as the Shell launcher, so `--run-eboot`
        // and the Shell remain one launch path observably.
        splash::stage_boot_splash(std::path::Path::new(&path));

        info!("entering guest _start via execute_process ...");
        let outcome = raeen_runtime::execute_process_shared(
            std::sync::Arc::clone(&linked),
            std::sync::Arc::clone(&hle),
            std::sync::Arc::clone(&kernel),
            &[path.as_str()],
            &[],
        );
        match &outcome {
            Ok(o) => info!("RESULT: {o:?}"),
            // The whole point of the per-NID unresolved stub: say WHICH import
            // the guest wanted. Report it as a worklist item, not an address.
            Err(raeen_runtime::RuntimeError::UnimplementedImport {
                nid,
                library,
                stub_addr,
                rip,
            }) => {
                let library = library.as_deref().unwrap_or("<unknown library>");
                info!(
                    "RESULT: guest needs an UNIMPLEMENTED import — {} \
                     — nid {nid:#018x} (encoded {}) from library '{library}'",
                    raeen_firmware::dynlib::nid_names::describe(*nid),
                    raeen_firmware::dynlib::nid::encode_nid(*nid)
                );
                info!("        guest rip {rip:#x}; its stub is {stub_addr:#x}");
                info!("        implement it, or supply the module that exports it, and re-run");
            }
            // A fault reports where the guest *was*, which on its own names
            // nothing: 250 MB into a stripped C++ binary, one address looks like
            // any other. The bytes there are the difference between a lead and a
            // dead end — they say whether the guest was loading a vtable,
            // calling through a pointer, or reading a thread-local. The image is
            // right here in the loader, so this costs nothing and never touches
            // the fault path itself.
            Err(raeen_runtime::RuntimeError::Faulted { addr, access, kind }) => {
                info!("RESULT: guest fault at {addr:#x} ({kind} {access:#x})");
                report_fault_site(&linked, &process.dependencies, *addr);
            }
            Err(e) => info!("RESULT: {e:?}"),
        }
        // The actionable crash report: everything above (fault site, call
        // rings, unresolved inventory, GPU counters) folded into ONE file
        // under logs/crashes/, next to any minidump. Written here — in the
        // guest-executing process — because only this process still holds
        // the kernel and the composed image.
        if let Err(error) = &outcome {
            write_runner_crash_report(
                Path::new(path.as_str()),
                error,
                &linked,
                &process.dependencies,
                &kernel,
            );
        }
        let console = kernel.console.contents();
        if console.is_empty() {
            info!("guest console: <empty>");
        } else {
            info!("guest console ({} byte(s)):\n{console}", console.len());
        }
        if std::env::var_os("RAEEN_RUNNER_CHILD").is_some()
            && let Err(error) = outcome
        {
            return Err(anyhow::anyhow!("isolated guest run failed: {error}"));
        }
        return Ok(());
    }

    info!("╔══════════════════════════════════════════════╗");
    info!(
        "║          Raeen — PS5 Emulator v{}        ║",
        raeen_core::VERSION
    );
    info!("║        Cross-Platform Compatibility Layer     ║");
    info!("╚══════════════════════════════════════════════╝");

    // Load configuration.
    let config_path = std::path::Path::new("config.toml");
    let config = raeen_core::config::EmulatorConfig::load(config_path)?;
    info!("Configuration loaded from {}", config_path.display());

    // Apply the persisted logging settings now that config is loaded (logging
    // was initialized earlier at the default level, before config existed); the
    // Shell changes these live afterwards via `logging::set_level`.
    raeen_core::logging::set_level(if config.debug.logging {
        config.debug.log_level.as_str()
    } else {
        "off"
    });

    // Mirror the persisted GPU settings into the GPU crate: Validation Layers
    // (applied when the Vulkan backend is first created), Resolution Scale
    // (applied to each guest draw), and GPU Device (physical-device selection).
    // All take effect from this launch onward.
    raeen_gpu::AgcGpuSession::set_runtime_config(
        config.graphics.validation_layers,
        config.graphics.resolution_scale,
        config.graphics.gpu_device_index,
        config.graphics.shader_cache,
        config.paths.shader_cache_dir.clone(),
    );
    // Register the BYO upscaler plugin's backends (DLSS/FSR/XeSS + spatial), if
    // this build opted into it, BEFORE applying the saved selection so a
    // persisted choice like "fsr" resolves. No-op unless `upscale-plugins` is on.
    #[cfg(feature = "upscale-plugins")]
    raeen_upscale::register_all();
    // Load user-supplied, out-of-tree present plugins from the git-ignored
    // `plugins/` tree via the stable C ABI. Raeen ships none of these and
    // fetches none: a plugin is a separate binary the user placed there, loaded
    // at runtime so nothing proprietary is ever linked into this artifact (see
    // `plugins/README.md`). Runs BEFORE the persisted selection is applied so a
    // saved choice naming an out-of-tree plugin resolves.
    //
    // SAFETY: loading executes native code from `plugins/`, which is a
    // user-controlled directory alongside the executable — the documented,
    // opt-in trust boundary for BYO plugins. Refusals are logged, not fatal.
    let plugins_dir = std::path::Path::new("plugins");
    let loaded = unsafe { raeen_gpu::AgcGpuSession::load_present_plugins_from(plugins_dir) };
    if !loaded.is_empty() {
        tracing::info!(
            count = loaded.len(),
            plugins = ?loaded,
            "loaded user-supplied present plugins"
        );
    }
    // Apply the persisted present-plugin (upscaler / frame gen) selection so a
    // saved choice is live from startup, not only after the user re-touches it.
    shell::apply_present_plugin(&config.graphics);

    // Bring up host audio output and apply the persisted Audio settings (Master
    // Volume / Audio Enabled). The guest's sceAudioOutOutput feeds this sink;
    // the Shell also updates it live when the user changes those settings.
    raeen_audio::output::set_volume(config.audio.volume);
    raeen_audio::output::set_enabled(config.audio.enabled);
    raeen_audio::output::init();

    // Record which RAEEN_* bridge variables the developer set manually BEFORE
    // the bridge below writes any of them — those manual overrides win over the
    // Settings toggles both here and in every per-launch runner environment
    // (`launcher::stage_runner_env`).
    launcher::record_dev_env_overrides();

    // Bridge the persisted Advanced diagnostics to the environment variables the
    // GPU/runtime read, so those settings actually take effect. A manually-set
    // env var always wins (a dev CLI override is never clobbered).
    let bridge_flag = |name: &str, on: bool| {
        if on && std::env::var_os(name).is_none() {
            // SAFETY: called in `main` before any thread is spawned (before
            // eframe and the guest session), so no other thread is reading the
            // environment concurrently — the edition-2024 unsafety is satisfied.
            unsafe { std::env::set_var(name, "1") };
        }
    };
    bridge_flag("RAEEN_DUMP_SHADERS", config.debug.dump_shaders);
    bridge_flag("RAEEN_DUMP_GPU_RESOURCES", config.debug.dump_gpu_commands);
    bridge_flag("RAEEN_TRACE_HLE", config.debug.trace_syscalls);
    bridge_flag("RAEEN_DUMP_FRAMES", config.debug.dump_frames);
    bridge_flag("RAEEN_CALL_STATS", config.debug.call_stats);
    bridge_flag("RAEEN_STALL_DUMP", config.debug.stall_dump);
    if std::env::var_os("RAEEN_VBLANK_HZ").is_none() {
        // SAFETY: as above — single-threaded startup, no concurrent env access.
        unsafe {
            std::env::set_var("RAEEN_VBLANK_HZ", config.graphics.frame_limit.to_string());
        }
    }

    // Initialize the kernel.
    let _kernel = raeen_kernel::OrbisKernel::new();
    info!("Orbis kernel HLE initialized");

    // Initialize the HLE registry.
    let _hle = raeen_hle::HleRegistry::new();
    info!("HLE library registry initialized");

    // Launch the GUI.
    info!("Launching Raeen GUI...");

    // The Shell is a full-screen, PS5-style console experience by default
    // (spec §7): borderless fullscreen, sized by the OS to the active
    // monitor. Forcing an inner_size alongside fullscreen used to strand a
    // 1920x1080 window in the corner of larger displays, so the configured
    // window size only applies when `general.fullscreen = false` opts into
    // a normal desktop window.
    let viewport = egui::ViewportBuilder::default()
        .with_title("Raeen")
        .with_min_inner_size([800.0, 600.0]);
    let viewport = if config.general.fullscreen {
        viewport.with_fullscreen(true).with_decorations(false)
    } else {
        viewport.with_inner_size([
            config.general.window_width as f32,
            config.general.window_height as f32,
        ])
    };
    let native_options = eframe::NativeOptions {
        viewport,
        // Honor the Video ▸ VSync setting (present mode). Applied at launch.
        vsync: config.general.vsync,
        ..Default::default()
    };

    eframe::run_native(
        "Raeen",
        native_options,
        Box::new(|cc| {
            // Set dark theme.
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app::RaeenApp::new(
                &cc.egui_ctx,
                config,
                config_path.to_path_buf(),
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;

    info!("Raeen shutting down");
    Ok(())
}

fn recover_stale_resource_init_locks(root: &std::path::Path) -> std::io::Result<usize> {
    let mut directories = vec![root.to_path_buf()];
    if root.exists() {
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                directories.push(entry.path());
            }
        }
    }
    let mut recovered = 0;
    for directory in directories {
        let marker = directory.join("resource_init_lock");
        match std::fs::metadata(&marker) {
            Ok(metadata) if metadata.is_file() && metadata.len() == 0 => {
                std::fs::remove_file(&marker)?;
                recovered += 1;
            }
            Ok(_) => {
                tracing::warn!(
                    path = %marker.display(),
                    "resource initialization marker contains data; preserving it"
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(recovered)
}

#[cfg(test)]
mod resource_recovery_tests {
    use super::recover_stale_resource_init_locks;

    #[test]
    fn empty_session_marker_is_removed_but_nonempty_data_is_preserved() {
        let root =
            std::env::temp_dir().join(format!("raeen-resource-lock-test-{}", std::process::id()));
        let slot = root.join("BedrockUserSettingsStorage");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&slot).expect("slot");
        let stale = slot.join("resource_init_lock");
        std::fs::write(&stale, []).expect("empty marker");
        assert_eq!(recover_stale_resource_init_locks(&root).unwrap(), 1);
        assert!(!stale.exists());

        std::fs::write(&stale, b"payload").expect("nonempty marker");
        assert_eq!(recover_stale_resource_init_locks(&root).unwrap(), 0);
        assert_eq!(std::fs::read(&stale).unwrap(), b"payload");
        let _ = std::fs::remove_dir_all(root);
    }
}
