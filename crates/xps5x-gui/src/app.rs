//! XPS5X application state and GUI rendering.

use egui::{self, Color32, RichText, Vec2};
use xps5x_core::config::EmulatorConfig;

/// Main application state.
pub struct XPS5XApp {
    /// Emulator configuration.
    config: EmulatorConfig,
    /// Currently selected tab.
    current_tab: Tab,
    /// Game list entries.
    games: Vec<GameEntry>,
    /// Status message.
    status: String,
    /// Whether emulation is running.
    running: bool,
}

/// Navigation tabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Games,
    Settings,
    Debug,
    About,
}

/// A game in the library.
#[derive(Debug, Clone)]
struct GameEntry {
    title: String,
    title_id: String,
    path: String,
    size: String,
}

impl XPS5XApp {
    pub fn new(config: EmulatorConfig) -> Self {
        Self {
            config,
            current_tab: Tab::Games,
            games: vec![
                GameEntry {
                    title: "No games found".to_string(),
                    title_id: "—".to_string(),
                    path: "Add games to the 'games' directory".to_string(),
                    size: "—".to_string(),
                },
            ],
            status: "Ready — No game loaded".to_string(),
            running: false,
        }
    }
}

impl eframe::App for XPS5XApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ─── Top menu bar ──────────────────────────────────
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Game...").clicked() {
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Emulation", |ui| {
                    if ui.button("Start").clicked() {
                        self.running = true;
                        self.status = "Emulation starting...".to_string();
                        ui.close_menu();
                    }
                    if ui.button("Stop").clicked() {
                        self.running = false;
                        self.status = "Emulation stopped".to_string();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Reset").clicked() {
                        self.status = "Emulation reset".to_string();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About XPS5X").clicked() {
                        self.current_tab = Tab::About;
                        ui.close_menu();
                    }
                });
            });
        });

        // ─── Bottom status bar ─────────────────────────────
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(28.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let status_color = if self.running {
                        Color32::from_rgb(100, 220, 100)
                    } else {
                        Color32::from_rgb(180, 180, 180)
                    };
                    ui.label(
                        RichText::new(format!("● {}", self.status))
                            .color(status_color)
                            .size(12.0),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("XPS5X v{}", xps5x_core::VERSION))
                                .color(Color32::from_rgb(120, 120, 120))
                                .size(11.0),
                        );
                    });
                });
            });

        // ─── Left navigation panel ─────────────────────────
        egui::SidePanel::left("nav_panel")
            .exact_width(160.0)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);
                    ui.heading(
                        RichText::new("XPS5X")
                            .color(Color32::from_rgb(0, 120, 255))
                            .size(24.0),
                    );
                    ui.label(
                        RichText::new("PS5 Emulator")
                            .color(Color32::from_rgb(150, 150, 150))
                            .size(11.0),
                    );
                });
                ui.add_space(16.0);
                ui.separator();
                ui.add_space(8.0);

                let tabs = [
                    (Tab::Games, "🎮  Games"),
                    (Tab::Settings, "⚙  Settings"),
                    (Tab::Debug, "🔧  Debug"),
                    (Tab::About, "ℹ  About"),
                ];

                for (tab, label) in tabs {
                    let selected = self.current_tab == tab;
                    let text = if selected {
                        RichText::new(label).color(Color32::from_rgb(0, 120, 255)).strong()
                    } else {
                        RichText::new(label).color(Color32::from_rgb(200, 200, 200))
                    };

                    if ui
                        .add(egui::Button::new(text).min_size(Vec2::new(140.0, 32.0)).frame(selected))
                        .clicked()
                    {
                        self.current_tab = tab;
                    }
                }
            });

        // ─── Central content area ──────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_tab {
                Tab::Games => self.render_games_tab(ui),
                Tab::Settings => self.render_settings_tab(ui),
                Tab::Debug => self.render_debug_tab(ui),
                Tab::About => self.render_about_tab(ui),
            }
        });
    }
}

impl XPS5XApp {
    fn render_games_tab(&self, ui: &mut egui::Ui) {
        ui.heading("Game Library");
        ui.separator();
        ui.add_space(8.0);

        // Game list table.
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("game_list")
                .striped(true)
                .min_col_width(100.0)
                .show(ui, |ui| {
                    ui.strong("Title");
                    ui.strong("Title ID");
                    ui.strong("Path");
                    ui.strong("Size");
                    ui.end_row();

                    for game in &self.games {
                        ui.label(&game.title);
                        ui.label(&game.title_id);
                        ui.label(&game.path);
                        ui.label(&game.size);
                        ui.end_row();
                    }
                });
        });
    }

    fn render_settings_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.separator();
        ui.add_space(8.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Graphics settings.
            ui.collapsing("🖥  Graphics", |ui| {
                ui.horizontal(|ui| {
                    ui.label("GPU Backend:");
                    ui.label(format!("{:?}", self.config.graphics.backend));
                });
                ui.add(
                    egui::Slider::new(&mut self.config.graphics.resolution_scale, 0.5..=4.0)
                        .text("Resolution Scale")
                        .suffix("x"),
                );
                ui.checkbox(&mut self.config.graphics.shader_cache, "Enable Shader Cache");
                ui.checkbox(&mut self.config.graphics.validation_layers, "Vulkan Validation Layers");
            });

            // Audio settings.
            ui.collapsing("🔊  Audio", |ui| {
                ui.checkbox(&mut self.config.audio.enabled, "Enable Audio");
                ui.add(
                    egui::Slider::new(&mut self.config.audio.volume, 0.0..=1.0)
                        .text("Master Volume"),
                );
                ui.checkbox(&mut self.config.audio.spatial_audio, "3D Spatial Audio (Tempest)");
            });

            // Input settings.
            ui.collapsing("🎮  Input", |ui| {
                ui.checkbox(&mut self.config.input.dualsense_features, "DualSense Features (Haptics + Adaptive Triggers)");
                ui.add(
                    egui::Slider::new(&mut self.config.input.deadzone, 0.0..=0.5)
                        .text("Stick Deadzone"),
                );
            });

            // Debug settings.
            ui.collapsing("🔧  Debug", |ui| {
                ui.checkbox(&mut self.config.debug.trace_syscalls, "Trace Syscalls");
                ui.checkbox(&mut self.config.debug.dump_gpu_commands, "Dump GPU Commands");
                ui.checkbox(&mut self.config.debug.dump_shaders, "Dump Shaders");
            });
        });
    }

    fn render_debug_tab(&self, ui: &mut egui::Ui) {
        ui.heading("Debug Tools");
        ui.separator();
        ui.add_space(8.0);

        ui.label("Debug tools will be available when a game is running:");
        ui.add_space(8.0);

        egui::Grid::new("debug_tools").show(ui, |ui| {
            ui.label("📋 GPU Command Viewer");
            ui.label("View decoded PM4 packets in real-time");
            ui.end_row();

            ui.label("🔍 Shader Disassembly");
            ui.label("Inspect recompiled SPIR-V shaders");
            ui.end_row();

            ui.label("💾 Memory Inspector");
            ui.label("Browse emulated PS5 memory map");
            ui.end_row();

            ui.label("📊 Performance Metrics");
            ui.label("FPS, GPU utilization, syscall frequency");
            ui.end_row();
        });
    }

    fn render_about_tab(&self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(40.0);
            ui.heading(
                RichText::new("XPS5X")
                    .color(Color32::from_rgb(0, 120, 255))
                    .size(48.0),
            );
            ui.add_space(8.0);
            ui.label(
                RichText::new("PlayStation 5 Emulator / Compatibility Layer")
                    .size(16.0)
                    .color(Color32::from_rgb(180, 180, 180)),
            );
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("Version {}", xps5x_core::VERSION))
                    .size(13.0)
                    .color(Color32::from_rgb(120, 120, 120)),
            );
            ui.add_space(24.0);
            ui.separator();
            ui.add_space(16.0);

            ui.label("Architecture:");
            ui.add_space(4.0);
            let features = [
                "• Native x86-64 execution (no CPU interpretation)",
                "• GNM → Vulkan GPU command translation",
                "• RDNA2 ISA → SPIR-V shader recompilation",
                "• Orbis OS (FreeBSD) syscall HLE",
                "• Tempest 3D Audio emulation",
                "• DualSense haptics & adaptive trigger support",
            ];
            for feature in features {
                ui.label(feature);
            }

            ui.add_space(24.0);
            ui.label(
                RichText::new("Licensed under GNU General Public License v2.0")
                    .size(11.0)
                    .color(Color32::from_rgb(100, 100, 100)),
            );
        });
    }
}
