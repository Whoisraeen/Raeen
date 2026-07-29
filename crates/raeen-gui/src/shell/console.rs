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

/// Which view the pane is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Tab {
    /// The structured "where is this title, and why did it stop" view.
    #[default]
    Status,
    /// The raw tracing tail.
    Log,
}

pub struct ConsolePane {
    pub open: bool,
    tab: Tab,
    search: String,
    autoscroll: bool,
    /// Index into [`LEVELS`]: show events at this level or more severe.
    min_level: usize,
    lines: Vec<ConsoleLine>,
    last_seq: Option<u64>,
    /// Distinct guest blocker lines already mirrored into the Shell's own
    /// tracing ring, so F10 ▸ Log carries them once and not once per second.
    mirrored: std::collections::HashSet<String>,
}

impl Default for ConsolePane {
    fn default() -> Self {
        Self {
            open: false,
            // Status first: a user opening this pane after a title misbehaves
            // wants the answer, not 5,000 lines to read. The Log tab is one
            // click away and unchanged.
            tab: Tab::Status,
            search: String::new(),
            autoscroll: true,
            min_level: 2, // Info
            lines: Vec::new(),
            last_seq: None,
            mirrored: std::collections::HashSet::new(),
        }
    }
}

impl ConsolePane {
    /// Pull new lines and draw the window when open. Cheap when closed (the
    /// ring is still drained so reopening never replays a burst).
    pub fn ui(&mut self, ctx: &egui::Context) {
        // Mirror the guest child's blockers into the Shell's own ring before
        // draining it, so F10 ▸ Log at the default level carries them. Without
        // this the Log tab is structurally blind to the guest: the ring is a
        // process-local tracing layer and the guest runs in a child spawned
        // with inherited, unpiped stdio.
        self.mirror_guest_blockers();
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
            ui.selectable_value(&mut self.tab, Tab::Status, "Status");
            ui.selectable_value(&mut self.tab, Tab::Log, "Log");
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

        if self.tab == Tab::Status {
            self.status_contents(ui);
            return;
        }

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

    /// The structured view: where the running title got to, and what refused.
    ///
    /// Every value here comes from the guest child across the frame-IPC page —
    /// the Shell cannot compute any of it. When there is no bridge, or the
    /// child has not published yet, this says so rather than rendering zeroes
    /// that would read as measurements.
    fn status_contents(&mut self, ui: &mut egui::Ui) {
        let dim = egui::Color32::from_rgb(140, 150, 164);
        let strong = egui::Color32::from_rgb(230, 235, 240);

        let Some(status) = raeen_gpu::frame_ipc::latest_remote_status() else {
            ui.add_space(8.0);
            ui.colored_label(dim, "No title is running in an isolated runner.");
            ui.colored_label(
                dim,
                "Launch a game and reopen this tab to see how far it gets. The Log tab \
                 shows the Shell's own events either way.",
            );
            return;
        };
        if !status.published() {
            ui.add_space(8.0);
            ui.colored_label(dim, "A title is running but has not reported yet.");
            ui.colored_label(dim, "The first report lands about a second after launch.");
            return;
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(4.0);

                // Headline: the two labels that answer "where did it stop".
                let stage = status
                    .stage
                    .and_then(|index| raeen_core::frame_path::Stage::ALL.get(index as usize))
                    .map_or("nothing", |s| s.label());
                let phase = status
                    .phase
                    .and_then(|index| raeen_core::frame_path::Phase::ALL.get(index as usize))
                    .map_or("nothing", |p| p.label());
                ui.horizontal(|ui| {
                    ui.colored_label(dim, "Load phase reached");
                    ui.label(
                        egui::RichText::new(phase)
                            .monospace()
                            .strong()
                            .color(strong),
                    );
                    ui.separator();
                    ui.colored_label(dim, "Frame path reached");
                    ui.label(
                        egui::RichText::new(stage)
                            .monospace()
                            .strong()
                            .color(strong),
                    );
                });

                // CPU vs wall: the measurement that separates a title parked on
                // a primitive that never fires from one spinning a core. Those
                // are different bugs and the difference is not visible any
                // other way.
                if status.wall_ms > 0 {
                    let ratio = status.cpu_ms as f64 / status.wall_ms as f64;
                    let (shape, color) = if ratio >= 0.8 {
                        (
                            "at least one thread spinning",
                            egui::Color32::from_rgb(255, 184, 108),
                        )
                    } else if ratio <= 0.1 {
                        ("parked in a wait", egui::Color32::from_rgb(139, 233, 253))
                    } else {
                        ("partially active", dim)
                    };
                    ui.horizontal(|ui| {
                        ui.colored_label(dim, "CPU");
                        ui.colored_label(
                            color,
                            format!(
                                "{:.1} s over {:.1} s wall ({:.0}% of one core) — {shape}",
                                status.cpu_ms as f64 / 1000.0,
                                status.wall_ms as f64 / 1000.0,
                                ratio * 100.0
                            ),
                        );
                    });
                }

                ui.horizontal(|ui| {
                    ui.colored_label(dim, "Blockers");
                    ui.colored_label(
                        strong,
                        format!(
                            "{} distinct, {} occurrence(s)",
                            status.distinct_blockers, status.total_events
                        ),
                    );
                    if status.dropped_distinct > 0 {
                        // A truncated table that does not say so reads as a
                        // complete one.
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 184, 108),
                            format!("+{} dropped at cap", status.dropped_distinct),
                        );
                    }
                });

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                if ui.button("Copy status").clicked() {
                    ui.ctx().copy_text(status.digest.clone());
                }
                ui.add_space(4.0);

                // The digest verbatim: the frame-path summary line, then the
                // ranked blocker lines, exactly as the crash report renders
                // them, so what a user pastes here matches the file on disk.
                for line in status.digest.lines() {
                    let color = if line.starts_with("frame path:") {
                        dim
                    } else {
                        egui::Color32::from_rgb(208, 214, 222)
                    };
                    ui.label(egui::RichText::new(line).monospace().color(color));
                }
            });
    }

    /// Copy any guest blocker the Shell has not seen into its own tracing ring.
    ///
    /// Bounded twice over: the child's table caps at a few hundred distinct
    /// entries, and `mirrored` makes each one cross exactly once no matter how
    /// often the child republishes.
    fn mirror_guest_blockers(&mut self) {
        let Some(status) = raeen_gpu::frame_ipc::latest_remote_status() else {
            return;
        };
        if !status.published() {
            return;
        }
        for line in status.digest.lines() {
            // The frame-path summary is not a blocker, and the truncation note
            // is about the channel rather than about the guest.
            if line.starts_with("frame path:") || line.starts_with('…') || line.is_empty() {
                continue;
            }
            if self.mirrored.insert(line.to_string()) {
                tracing::warn!(target: "raeen::guest", "blocker {line}");
            }
        }
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
