---
name: verify
description: Build, launch, drive, and screenshot the XPS5X shell (native eframe GUI) to verify UI changes on Windows
---

# Verifying XPS5X shell changes

Surface: native eframe window (`xps5x.exe`), fullscreen-borderless, sized from
`config.toml` (default 1920x1080, top-left of the primary display).

1. Build: `cargo build -p xps5x-gui` → `target/debug/xps5x.exe`.
2. Launch (PowerShell): `Start-Process` the exe with the repo root as the
   working directory — it scans `Games/`, loads `themes/`, and writes a
   default `config.toml` if missing (don't commit that file). Wait ~6s
   (startup + ~2.1s boot animation).
3. Focus guard before any keystrokes: `WScript.Shell.AppActivate($proc.Id)`,
   then verify the app owns the foreground via Win32 `GetForegroundWindow` +
   `GetWindowThreadProcessId`; abort if not, so keys can't land in another app.
4. Drive with `[System.Windows.Forms.SendKeys]::SendWait`:
   `{RIGHT}`/`{LEFT}` rail nav, `{ENTER}` launch, `c` control center, `{TAB}`
   Games/Media tab, `{ESC}` back. Sleep 200–800ms after each so animations
   settle (exponential easers, speed 6–12).
5. Capture: `System.Drawing.Bitmap` + `Graphics.CopyFromScreen` over
   `0,0,1920,1080`, save PNG to the scratchpad, then Read it. Crop a strip
   with `Graphics.DrawImage` when you need pixel-accurate measurements.
6. `Stop-Process` the app when done.

Gotchas:
- `Add-Type` Win32 wrapper types do NOT persist between PowerShell tool
  invocations — redefine them each call, with a fresh class name (re-adding
  an existing name errors).
- With `Games/` empty the sample library shows. Launching a sample game shows
  the honest fault overlay ("No module file at Games/…") — expected; Esc
  returns to Home.
- Never lay out bottom-anchored Home content with nested egui `bottom_up`
  scopes: rows land ~120px too low, overflow the window, and the overflow
  grows the parent `Ui`'s rects, which then mis-anchors the Control Center
  drawn after Home. Use painter + explicit rects (see
  `crates/xps5x-gui/src/shell/home.rs::draw_context_block`).
