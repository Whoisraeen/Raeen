//! Which frame kind a plugin receives is decided by what it declares.
//!
//! `Capabilities::gpu_frames` used to be inert on the ABI-v3 path: the host
//! gated every v3 step on `vulkan_requirements_v3()` alone, so a plugin that
//! declared itself CPU-only (`gpu_frames: false`) but happened to implement the
//! v3 methods was handed borrowed `VkImage`s anyway, and had its Vulkan
//! requirements forced onto the device at creation time. The declaration is now
//! load-bearing and both halves are required.
//!
//! Pinned here, without needing a Vulkan device (the registry decides routing
//! before any GPU work is recorded):
//!
//! 1. A GPU-capable plugin (`gpu_frames` + v3 requirements) is routed to the
//!    GPU path — the host asks for its requirements and offers it a GPU frame.
//! 2. A CPU-only plugin that nonetheless implements the v3 methods is routed to
//!    the CPU path: no requirements requested, no `process_gpu_v3` call, and it
//!    still receives correct CPU pixels through `process`.
//! 3. Declaring `gpu_frames` without v3 requirements is not enough either — a
//!    plugin Raeen cannot describe a device for must not be handed one.
//!
//! One `#[test]`: the plugin registry is process-global, so tests that select
//! an active plugin cannot run concurrently in one binary.

use raeen_gpu::present_plugin::{
    self, Capabilities, PluginOutput, PresentContext, PresentFrame, PresentPlugin, cabi_v3,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Counts how many times each ABI path was taken, so the test asserts on what
/// the host actually did rather than on what it claims it would do.
#[derive(Default)]
struct Calls {
    requirements: AtomicU32,
    process_gpu_v3: AtomicU32,
    process_cpu: AtomicU32,
}

/// A plugin whose declared capabilities and v3 support are set per-instance, so
/// one implementation covers all three routing cases.
struct Probe {
    name: &'static str,
    capabilities: Capabilities,
    /// Whether this plugin states Vulkan requirements at all.
    offers_v3: bool,
    calls: Arc<Calls>,
}

impl PresentPlugin for Probe {
    fn name(&self) -> &str {
        self.name
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn vulkan_requirements_v3(&self) -> Option<cabi_v3::RaeenVulkanRequirementsV3> {
        if !self.offers_v3 {
            return None;
        }
        self.calls.requirements.fetch_add(1, Ordering::Relaxed);
        let mut requirements = cabi_v3::RaeenVulkanRequirementsV3::empty();
        requirements.required_queue_flags = cabi_v3::RAEEN_V3_QUEUE_GRAPHICS;
        Some(requirements)
    }

    fn process_gpu_v3(
        &mut self,
        _frame: &cabi_v3::RaeenPresentFrameV3,
        _output: &mut cabi_v3::RaeenPluginOutputV3,
    ) -> i32 {
        self.calls.process_gpu_v3.fetch_add(1, Ordering::Relaxed);
        cabi_v3::RAEEN_V3_DECLINED
    }

    /// Inverts every colour byte, so "the CPU path ran and produced correct
    /// pixels" is checkable rather than merely counted.
    fn process(&mut self, frame: &PresentFrame<'_>, _context: &PresentContext) -> PluginOutput {
        self.calls.process_cpu.fetch_add(1, Ordering::Relaxed);
        let mut out = PluginOutput::identity(frame);
        for byte in &mut out.primary.pixels {
            *byte = !*byte;
        }
        out
    }
}

/// Does the host currently consider the active plugin GPU-resident? Observed
/// through the one public v3 surface: the host only asks for a plugin's Vulkan
/// requirements when it intends to give it GPU frames.
fn host_would_request_requirements() -> bool {
    present_plugin::active_vulkan_requirements_v3().is_some()
}

#[test]
fn gpu_frames_decides_the_path_and_a_cpu_only_plugin_still_gets_pixels() {
    // (1) GPU-capable: declares gpu_frames AND states requirements.
    let gpu_calls = Arc::new(Calls::default());
    present_plugin::register(Box::new(Probe {
        name: "probe-gpu",
        capabilities: Capabilities {
            upscale: true,
            gpu_frames: true,
            ..Capabilities::default()
        },
        offers_v3: true,
        calls: Arc::clone(&gpu_calls),
    }));

    // (2) CPU-only, but implements every v3 method. The trap case.
    let cpu_calls = Arc::new(Calls::default());
    present_plugin::register(Box::new(Probe {
        name: "probe-cpu-with-v3-methods",
        capabilities: Capabilities {
            upscale: true,
            gpu_frames: false,
            ..Capabilities::default()
        },
        offers_v3: true,
        calls: Arc::clone(&cpu_calls),
    }));

    // (3) Claims gpu_frames but describes no device.
    let bare_calls = Arc::new(Calls::default());
    present_plugin::register(Box::new(Probe {
        name: "probe-gpu-without-requirements",
        capabilities: Capabilities {
            gpu_frames: true,
            ..Capabilities::default()
        },
        offers_v3: false,
        calls: Arc::clone(&bare_calls),
    }));

    // --- (1) the GPU-capable plugin is routed to the GPU path -------------
    assert!(present_plugin::select("probe-gpu"), "probe-gpu registered");
    assert!(
        host_would_request_requirements(),
        "a plugin declaring gpu_frames with v3 requirements must be routed to \
         the GPU path"
    );

    // --- (2) the CPU-only plugin is routed to the CPU path ----------------
    assert!(present_plugin::select("probe-cpu-with-v3-methods"));
    assert_eq!(
        cpu_calls.requirements.load(Ordering::Relaxed),
        0,
        "sanity: nothing has asked this plugin for requirements yet"
    );
    assert!(
        !host_would_request_requirements(),
        "a plugin that declares gpu_frames: false must NOT have its Vulkan \
         requirements forced onto the device, even though it implements the v3 \
         methods"
    );
    assert_eq!(
        cpu_calls.requirements.load(Ordering::Relaxed),
        0,
        "the host must not even call vulkan_requirements_v3 on a CPU-only plugin"
    );

    // A GPU frame offered to it must be declined by the HOST, before the
    // plugin's own process_gpu_v3 can run.
    // SAFETY: both are `#[repr(C)]` PODs of integers, floats and raw handles,
    // for which all-zero is a valid (and here, deliberately "absent") bit
    // pattern. The host must decline before the frame is ever inspected, which
    // is exactly what the assertions below check.
    let frame: cabi_v3::RaeenPresentFrameV3 = unsafe { std::mem::zeroed() };
    let mut output: cabi_v3::RaeenPluginOutputV3 = unsafe { std::mem::zeroed() };
    assert_eq!(
        present_plugin::process_active_gpu_v3(&frame, &mut output),
        cabi_v3::RAEEN_V3_DECLINED,
        "the host must decline on a CPU-only plugin's behalf"
    );
    assert_eq!(
        cpu_calls.process_gpu_v3.load(Ordering::Relaxed),
        0,
        "a CPU-only plugin must never see process_gpu_v3"
    );

    // And it still gets correct CPU pixels — declining the GPU path must not
    // cost it the frame.
    let source = Arc::new(raeen_gpu::RenderedImage {
        width: 2,
        height: 1,
        pixels: vec![0x00, 0x11, 0x22, 0xff, 0x33, 0x44, 0x55, 0xff],
        bytes_per_pixel: 4,
    });
    let processed = present_plugin::apply_to_image_for_tests(Arc::clone(&source));
    assert_eq!(
        cpu_calls.process_cpu.load(Ordering::Relaxed),
        1,
        "the CPU-only plugin must receive the frame through `process`"
    );
    assert_eq!(
        processed.pixels,
        source.pixels.iter().map(|b| !b).collect::<Vec<u8>>(),
        "the CPU path must deliver the plugin's real output, not the source"
    );
    assert_eq!(
        (processed.width, processed.height, processed.bytes_per_pixel),
        (2, 1, 4),
        "the CPU output keeps the frame's geometry"
    );

    // --- (3) gpu_frames alone is not enough ------------------------------
    assert!(present_plugin::select("probe-gpu-without-requirements"));
    assert!(
        !host_would_request_requirements(),
        "a plugin Raeen cannot describe a device for must not be routed to the \
         GPU path just because it claims gpu_frames"
    );
    assert_eq!(
        present_plugin::process_active_gpu_v3(&frame, &mut output),
        cabi_v3::RAEEN_V3_DECLINED
    );
    assert_eq!(bare_calls.process_gpu_v3.load(Ordering::Relaxed), 0);

    // --- no plugin selected is still the zero-cost identity --------------
    present_plugin::select_none();
    assert!(!host_would_request_requirements());
    let untouched = present_plugin::apply_to_image_for_tests(Arc::clone(&source));
    assert!(
        Arc::ptr_eq(&untouched, &source),
        "with no plugin selected the frame must pass through as the SAME Arc"
    );
}
