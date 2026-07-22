---
name: verify
description: >
  Build, launch, drive, and screenshot the XPS5X egui Shell on Windows
  (PostMessage + GetClientRect). Use after Shell/UI/launcher/theme/fullscreen
  changes or when the user asks to verify the GUI.
---

# Verifying XPS5X shell changes

Surface: native eframe window (`raeen.exe`), fullscreen-borderless, sized from
`config.toml` (default 1920x1080, top-left of the primary display).

1. Build: `cargo build -p xps5x-gui` → `target/debug/raeen.exe`.
2. Launch (PowerShell): `Start-Process` the exe with the repo root as the
   working directory — it scans `Games/`, loads `themes/`, and writes a
   default `config.toml` if missing (don't commit that file). Wait ~6s
   (startup + ~2.1s boot animation).
3. Drive by posting key messages straight to the app's window handle — no
   focus needed, and keys can never land in another app. (SendKeys +
   AppActivate loses whenever the user is actively typing: Windows blocks
   focus stealing.) `PostMessage(hwnd, WM_KEYDOWN/WM_KEYUP, vk, lParam)`
   with the scancode from `MapVirtualKey` in lParam bits 16–23 and the
   extended-key bit (24) set for arrows; hwnd from `$proc.MainWindowHandle`
   after `$proc.Refresh()`. Keys: RIGHT/LEFT rail nav, RETURN launch, `C`
   control center, TAB Games/Media tab, ESC back. Sleep 200–800ms after
   each so animations settle (exponential easers, speed 6–12).
4. Capture: `System.Drawing.Bitmap` + `Graphics.CopyFromScreen` over the
   primary screen bounds, save PNG to the scratchpad, then Read it. Crop a
   strip with `Graphics.DrawImage` when you need pixel-accurate
   measurements. Window geometry claims (size/decorations) need no
   screenshot at all: `GetClientRect` + `GetWindowLong(GWL_STYLE) &
   WS_CAPTION` measure the window even when it's occluded.
5. `Stop-Process` the app when done.

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
