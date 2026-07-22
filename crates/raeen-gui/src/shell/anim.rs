//! Time-based tween helpers.
//!
//! egui is immediate-mode, so the mockup's crossfades, eased rail slides,
//! and focus scaling are driven by explicit lerps evaluated once per frame
//! from `stable_dt`, kept out of widget code (spec §7).

/// A single scalar that eases toward a target value over time.
///
/// Uses exponential smoothing (`value += (target - value) * (1 - e^-speed*dt)`),
/// which reads as a soft ease-out and never overshoots.
#[derive(Debug, Clone, Copy)]
pub struct Animated {
    pub value: f32,
    target: f32,
    /// Higher = snappier.
    speed: f32,
}

impl Animated {
    pub fn new(value: f32) -> Self {
        Self {
            value,
            target: value,
            speed: 10.0,
        }
    }

    pub fn with_speed(value: f32, speed: f32) -> Self {
        Self {
            value,
            target: value,
            speed,
        }
    }

    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Reserved for callers that need to check the in-flight target (e.g. a
    /// future screen deciding whether to re-trigger a transition).
    #[allow(dead_code)]
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Advance the animation by `dt` seconds. Returns `true` while still
    /// converging (callers use this to decide whether to request a repaint).
    pub fn tick(&mut self, dt: f32) -> bool {
        let delta = self.target - self.value;
        if delta.abs() < 0.0005 {
            self.value = self.target;
            return false;
        }
        let t = 1.0 - (-self.speed * dt).exp();
        self.value += delta * t;
        true
    }

    /// Alternative to checking `tick`'s return value, for callers that poll
    /// state outside the tick loop.
    #[allow(dead_code)]
    pub fn is_animating(&self) -> bool {
        (self.value - self.target).abs() >= 0.0005
    }
}

/// Linearly interpolate between two values. `lerp_color` (below) is the
/// color counterpart used by the Home/Control Center renderers.
#[allow(dead_code)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Linearly interpolate between two colors, channel-wise.
pub fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    egui::Color32::from_rgba_unmultiplied(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
        (a.a() as f32 + (b.a() as f32 - a.a() as f32) * t) as u8,
    )
}

/// Cubic ease-out, matching the mockup's `cubic-bezier(0.22, 0.61, 0.36, 1)`
/// closely enough for our purposes.
pub fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animated_converges_to_target() {
        let mut a = Animated::new(0.0);
        a.set_target(100.0);
        for _ in 0..500 {
            if !a.tick(1.0 / 60.0) {
                break;
            }
        }
        assert!((a.value - 100.0).abs() < 0.01);
        assert!(!a.is_animating());
    }

    #[test]
    fn animated_snaps_when_close_enough() {
        let mut a = Animated::new(9.9997);
        a.set_target(10.0);
        assert!(!a.tick(1.0 / 60.0) || (a.value - 10.0).abs() < 0.001);
    }

    #[test]
    fn lerp_at_bounds() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn ease_out_cubic_at_bounds() {
        assert_eq!(ease_out_cubic(0.0), 0.0);
        assert!((ease_out_cubic(1.0) - 1.0).abs() < 1e-6);
    }
}
