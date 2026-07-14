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
    // Initialize logging.
    xps5x_core::logging::init("info");

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
        let dynlib_data = xps5x_firmware::dynlib::parse_dynlibdata(
            module.dynlib_data.as_deref().unwrap_or(&[]),
            &dyn_tags,
        )?;
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
