//! # XPS5X — PlayStation 5 Emulator
//!
//! Main entry point for the XPS5X desktop application.
//! Initializes the emulator subsystems and launches the GUI.

mod app;
mod launcher;
mod library;
mod shell;
mod theme;
mod updater;

use tracing::info;

fn main() -> anyhow::Result<()> {
    // Initialize logging to BOTH stderr and `logs/xps5x.log`. `_log` must stay
    // alive for the whole process — dropping it shuts down the background
    // writer thread and loses buffered events (see `LogGuard`). Binding it here
    // in `main` is what makes the log file complete on exit.
    //
    // Falls back to stderr-only if the log directory can't be created (e.g. a
    // read-only working directory) — never a reason to refuse to boot.
    let _log = match xps5x_core::logging::init_with_file(
        "info",
        std::path::Path::new(xps5x_core::logging::DEFAULT_LOG_DIR),
    ) {
        Ok(guard) => guard,
        Err(e) => {
            let guard = xps5x_core::logging::init("info");
            tracing::warn!("file logging unavailable ({e}); continuing with stderr only");
            guard
        }
    };

    // Diagnostic: `xps5x --firmware-info <PUP>` inspects a firmware package
    // and exits without launching the GUI. It never decrypts anything.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--firmware-info") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--firmware-info requires a path to a PUP file"))?;
        let firmware = xps5x_firmware::Firmware::open(path)?;
        print!("{}", xps5x_firmware::summarize(&firmware));
        return Ok(());
    }

    // Diagnostic: `xps5x --load-sprx <sprx>` runs the LM1 homebrew pipeline
    // (SELF decrypt-or-passthrough -> .sprx parse -> dynlibdata decode ->
    // NID link against HLE) over a file and prints a summary, then exits
    // without launching the GUI. Uses `NoKeysProvider` throughout — it never
    // decrypts anything without a user-supplied key.
    if let Some(pos) = args.iter().position(|a| a == "--load-sprx") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--load-sprx requires a path to a .sprx/SELF file"))?;
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let dyn_tags = match &module.dynamic {
            Some(d) => xps5x_firmware::dynlib::parse_sce_dynamic(d)?,
            None => Vec::new(),
        };
        // Two dynamic models: the PT_SCE_DYNLIBDATA blob (homebrew/.sprx) or
        // standard vaddr-based tags with no such segment (real PS5 titles).
        let standard = xps5x_firmware::dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
        let dynlib_data = match &standard {
            Some((image, tags)) => xps5x_firmware::dynlib::parse_dynlibdata(image, tags)?,
            None => xps5x_firmware::dynlib::parse_dynlibdata(
                module.dynlib_data.as_deref().unwrap_or(&[]),
                &dyn_tags,
            )?,
        };
        let hle = xps5x_hle::HleRegistry::new();
        let db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle_names(hle.registered_names());
        let mut registry = xps5x_firmware::ModuleRegistry::new(db);
        registry.register_module_exports(&module.name, &dynlib_data.exports);
        let linked = xps5x_firmware::link_module(&module, &dynlib_data, &registry, &hle, 0)?;
        println!("module: {}", module.name);
        println!(
            "imports: {}  exports: {}",
            dynlib_data.imports.len(),
            dynlib_data.exports.len()
        );
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

    // Diagnostic: `xps5x --missing-nids <eboot.bin>` loads a title exactly as
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
        let hle = xps5x_hle::HleRegistry::new();
        let db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle_names(hle.registered_names());
        let mut registry = xps5x_firmware::ModuleRegistry::new(db);
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let process = xps5x_firmware::load_process(
            &bytes,
            dir,
            &xps5x_firmware::NoKeysProvider,
            &mut registry,
            &hle,
            xps5x_runtime::GUEST_ARENA_BASE,
        )?;

        use std::collections::BTreeMap;
        let mut by_lib: BTreeMap<&str, Vec<&xps5x_firmware::UnresolvedStub>> = BTreeMap::new();
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
        println!("# encoded_nid  nid  library");
        for (lib, stubs) in ranked {
            println!("\n## {lib}  ({} missing)", stubs.len());
            let mut rows: Vec<String> = stubs
                .iter()
                .map(|s| {
                    format!(
                        "{}  {:#018x}  {lib}",
                        xps5x_firmware::dynlib::nid::encode_nid(s.nid),
                        s.nid
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

    // Diagnostic: `xps5x --run-eboot <eboot.bin>` drives the **real** launch
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
        let bytes = std::fs::read(path)?;
        let hle = xps5x_hle::HleRegistry::new();
        let db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle_names(hle.registered_names());
        let mut registry = xps5x_firmware::ModuleRegistry::new(db);
        let kernel = xps5x_kernel::OrbisKernel::new();
        // Load as a whole process: the eboot plus every DT_NEEDED .prx that
        // ships beside it (M1-D). A real title's imports are overwhelmingly
        // satisfied by those bundled libraries, not by HLE.
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let process = xps5x_firmware::load_process(
            &bytes,
            dir,
            &xps5x_firmware::NoKeysProvider,
            &mut registry,
            &hle,
            xps5x_runtime::GUEST_ARENA_BASE,
        )?;
        for d in &process.dependencies {
            info!(
                "  dep {} at +{:#x}: {} exports, {} unresolved",
                d.name, d.image_offset, d.exports, d.unresolved
            );
        }
        let linked = process.linked;
        info!(
            "loaded: entry={:#x} image={:#x} byte(s) resolved={} unresolved={}",
            linked.entry,
            linked.image.len(),
            linked.hle_trampolines.len(),
            linked.unresolved.len()
        );
        info!("entering guest _start via execute_process ...");
        let outcome = xps5x_runtime::execute_process(&linked, &hle, &kernel, &[path.as_str()], &[]);
        match &outcome {
            Ok(o) => info!("RESULT: {o:?}"),
            // The whole point of the per-NID unresolved stub: say WHICH import
            // the guest wanted. Report it as a worklist item, not an address.
            Err(xps5x_runtime::RuntimeError::UnimplementedImport { nid, addr }) => {
                let stub = linked.unresolved_stubs.iter().find(|s| s.nid == *nid);
                let library = stub
                    .and_then(|s| s.library.as_deref())
                    .unwrap_or("<unknown library>");
                // `addr` is the faulting instruction's Rip — where the guest
                // was, NOT the stub. Naming it "stub" was wrong and confusing.
                info!(
                    "RESULT: guest needs an UNIMPLEMENTED import — nid {nid:#018x} \
                     (encoded {}) from library '{library}'",
                    xps5x_firmware::dynlib::nid::encode_nid(*nid)
                );
                info!(
                    "        guest rip {addr:#x}{}",
                    match stub {
                        Some(s) => format!("; its stub is {:#x}", s.addr),
                        None => String::new(),
                    }
                );
                info!("        implement it, or supply the module that exports it, and re-run");
            }
            Err(e) => info!("RESULT: {e:?}"),
        }
        let console = kernel.console.contents();
        if console.is_empty() {
            info!("guest console: <empty>");
        } else {
            info!("guest console ({} byte(s)):\n{console}", console.len());
        }
        return Ok(());
    }

    info!("╔══════════════════════════════════════════════╗");
    info!(
        "║          XPS5X — PS5 Emulator v{}        ║",
        xps5x_core::VERSION
    );
    info!("║        Cross-Platform Compatibility Layer     ║");
    info!("╚══════════════════════════════════════════════╝");

    // Load configuration.
    let config_path = std::path::Path::new("config.toml");
    let config = xps5x_core::config::EmulatorConfig::load(config_path)?;
    info!("Configuration loaded from {}", config_path.display());

    // Initialize the kernel.
    let _kernel = xps5x_kernel::OrbisKernel::new();
    info!("Orbis kernel HLE initialized");

    // Initialize the HLE registry.
    let _hle = xps5x_hle::HleRegistry::new();
    info!("HLE library registry initialized");

    // Launch the GUI.
    info!("Launching XPS5X GUI...");

    // The Shell is a full-screen, PS5-style console experience by default
    // (spec §7): borderless fullscreen, sized by the OS to the active
    // monitor. Forcing an inner_size alongside fullscreen used to strand a
    // 1920x1080 window in the corner of larger displays, so the configured
    // window size only applies when `general.fullscreen = false` opts into
    // a normal desktop window.
    let viewport = egui::ViewportBuilder::default()
        .with_title("XPS5X")
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
        ..Default::default()
    };

    eframe::run_native(
        "XPS5X",
        native_options,
        Box::new(|cc| {
            // Set dark theme.
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app::XPS5XApp::new(
                &cc.egui_ctx,
                config,
                config_path.to_path_buf(),
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;

    info!("XPS5X shutting down");
    Ok(())
}
