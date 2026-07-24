//! In-app log console (SharpEmu-style).
//!
//! A floating, resizable window that mirrors the tracing log live — level
//! colors, substring search, minimum-level filter, autoscroll, copy and
//! clear — so diagnosing a title never requires the terminal or opening
//! `logs/raeen.log`. Fed by `raeen_core::logging::console()`, the bounded
//! ring every subscriber event is mirrored into; the file log is unaffected
//! by anything done here (Clear only empties the ring).

use eframe::egui;
use raeen_core::logging::{ConsoleLine, console};

/// Client-side line cap. The core ring holds the recent past; the pane keeps
/// its own copy so filtering/search never contend with the log writers.
const MAX_LINES: usize = 20_000;

/// Minimum-level filter, ordered coarsest-first for the combo box.
const LEVELS: [(&str, tracing::Level); 5] = [
    ("Error", tracing::Level::ERROR),
    ("Warn", tracing::Level::WARN),
    ("Info", tracing::Level::INFO),
    ("Debug", tracing::Level::DEBUG),
    ("Trace", tracing::Level::TRACE),
];

pub struct ConsolePane {
    pub open: bool,
    search: String,
    autoscroll: bool,
    /// Index into [`LEVELS`]: show events at this level or more severe.
    min_level: usize,
    lines: Vec<ConsoleLine>,
    last_seq: Option<u64>,
}

impl Default for ConsolePane {
    fn default() -> Self {
        Self {
            open: false,
            search: String::new(),
            autoscroll: true,
            min_level: 2, // Info
            lines: Vec::new(),
            last_seq: None,
        }
    }
}

impl ConsolePane {
    /// Pull new lines and draw the window when open. Cheap when closed (the
    /// ring is still drained so reopening never replays a burst).
    pub fn ui(&mut self, ctx: &egui::Context) {
        if let Some(seq) = console().read_since(self.last_seq, &mut self.lines) {
            self.last_seq = Some(seq);
        }
        if self.lines.len() > MAX_LINES {
            let excess = self.lines.len() - MAX_LINES;
            self.lines.drain(..excess);
        }
        if !self.open {
            return;
        }

        // Own chrome, always dark and fully opaque (SharpEmu console
        // palette): the Shell theme may style egui windows light or
        // translucent, and a log console over bright game art becomes
        // low-contrast mush ("blurry") if it inherits that.
        let frame = egui::Frame::window(&ctx.style())
            .fill(egui::Color32::from_rgb(13, 16, 23))
            .stroke(egui::Stroke::new(
                1.0_f32,
                egui::Color32::from_rgb(35, 43, 58),
            ));
        egui::Window::new("raeen_console")
            .title_bar(false)
            .frame(frame)
            .default_size([920.0, 460.0])
            .min_size([480.0, 220.0])
            .resizable(true)
            .show(ctx, |ui| self.contents(ui));
    }

    fn contents(&mut self, ui: &mut egui::Ui) {
        // Scoped dark widget style (checkbox/combo/search/window text) so the
        // console is self-consistent under any Shell theme.
        *ui.visuals_mut() = egui::Visuals::dark();
        ui.visuals_mut().override_text_color = Some(egui::Color32::from_rgb(208, 214, 222));
        // Log rows at 14px: the 12px default is below Consolas' comfortable
        // floor at 1:1 DPI (reads as blur, not smallness).
        ui.style_mut()
            .text_styles
            .insert(egui::TextStyle::Monospace, egui::FontId::monospace(14.0));

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Console")
                    .strong()
                    .color(egui::Color32::from_rgb(230, 235, 240)),
            );
            ui.separator();
            ui.label("Level:");
            egui::ComboBox::from_id_salt("console_level")
                .selected_text(LEVELS[self.min_level].0)
                .show_ui(ui, |ui| {
                    for (index, (name, _)) in LEVELS.iter().enumerate() {
                        ui.selectable_value(&mut self.min_level, index, *name);
                    }
                });
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Search…")
                    .desired_width(240.0),
            );
            ui.checkbox(&mut self.autoscroll, "Autoscroll");
            if ui.button("Copy").clicked() {
                let text: String = self
                    .filtered()
                    .map(format_line)
                    .collect::<Vec<_>>()
                    .join("\n");
                ui.ctx().copy_text(text);
            }
            if ui.button("Clear").clicked() {
                console().clear();
                self.lines.clear();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("✕").on_hover_text("Close (F10)").clicked() {
                    self.open = false;
                }
                ui.weak(format!("{} lines", self.lines.len()));
            });
        });
        ui.separator();

        let visible: Vec<usize> = {
            let max_level = LEVELS[self.min_level].1;
            let needle = self.search.to_ascii_lowercase();
            self.lines
                .iter()
                .enumerate()
                .filter(|(_, line)| {
                    line.level <= max_level
                        && (needle.is_empty()
                            || line.message.to_ascii_lowercase().contains(&needle)
                            || line.target.to_ascii_lowercase().contains(&needle))
                })
                .map(|(index, _)| index)
                .collect()
        };

        let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(self.autoscroll)
            .show_rows(ui, row_height, visible.len(), |ui, range| {
                for &index in &visible[range] {
                    let line = &self.lines[index];
                    // Single-line rows (SharpEmu NoWrap parity): `show_rows`
                    // assumes uniform row height, so a wrapped multi-line
                    // label would corrupt the scroll math. Long messages
                    // truncate; Copy yields the full text.
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.label(
                            egui::RichText::new(format!("{:9.3}", line.elapsed_ms as f64 / 1000.0))
                                .monospace()
                                .color(egui::Color32::from_rgb(96, 106, 120)),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:5}", line.level.as_str()))
                                .monospace()
                                .color(level_color(line.level)),
                        );
                        ui.label(
                            egui::RichText::new(short_target(&line.target))
                                .monospace()
                                .color(egui::Color32::from_rgb(120, 132, 148)),
                        );
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&line.message)
                                    .monospace()
                                    .color(message_color(line.level)),
                            )
                            .truncate(),
                        );
                    });
                }
            });
    }

    fn filtered(&self) -> impl Iterator<Item = &ConsoleLine> {
        let max_level = LEVELS[self.min_level].1;
        let needle = self.search.to_ascii_lowercase();
        self.lines.iter().filter(move |line| {
            line.level <= max_level
                && (needle.is_empty()
                    || line.message.to_ascii_lowercase().contains(&needle)
                    || line.target.to_ascii_lowercase().contains(&needle))
        })
    }
}

/// Last module-path segment: `raeen_hle::libsce_agc` → `libsce_agc`.
fn short_target(target: &str) -> &str {
    target.rsplit("::").next().unwrap_or(target)
}

fn format_line(line: &ConsoleLine) -> String {
    format!(
        "{:9.3} {:5} {}: {}",
        line.elapsed_ms as f64 / 1000.0,
        line.level.as_str(),
        line.target,
        line.message
    )
}

fn level_color(level: tracing::Level) -> egui::Color32 {
    match level {
        tracing::Level::ERROR => egui::Color32::from_rgb(255, 85, 85),
        tracing::Level::WARN => egui::Color32::from_rgb(255, 184, 108),
        tracing::Level::INFO => egui::Color32::from_rgb(80, 250, 123),
        tracing::Level::DEBUG => egui::Color32::from_rgb(139, 233, 253),
        tracing::Level::TRACE => egui::Color32::from_rgb(98, 114, 164),
    }
}

fn message_color(level: tracing::Level) -> egui::Color32 {
    match level {
        tracing::Level::ERROR => egui::Color32::from_rgb(255, 121, 121),
        tracing::Level::WARN => egui::Color32::from_rgb(245, 200, 140),
        _ => egui::Color32::from_rgb(208, 214, 222),
    }
}
