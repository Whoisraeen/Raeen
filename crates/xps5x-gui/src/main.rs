//! # XPS5X — PlayStation 5 Emulator
//!
//! Main entry point for the XPS5X desktop application.
//! Initializes the emulator subsystems and launches the GUI.

mod app;
mod launcher;
mod library;
mod shell;
mod theme;

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

    info!("╔══════════════════════════════════════════════╗");
    info!("║          XPS5X — PS5 Emulator v{}        ║", xps5x_core::VERSION);
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

    // The Shell is a full-screen, PS5-style console experience — borderless
    // and maximized rather than a resizable desktop window (spec §7).
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("XPS5X")
            .with_fullscreen(true)
            .with_decorations(false)
            .with_inner_size([config.general.window_width as f32, config.general.window_height as f32])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "XPS5X",
        native_options,
        Box::new(|cc| {
            // Set dark theme.
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app::XPS5XApp::new(config)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;

    info!("XPS5X shutting down");
    Ok(())
}
