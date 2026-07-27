//! `VK_EXT_external_memory_host` capability probe — phase 1 of the
//! GPU-resident present plan
//! (`docs/superpowers/plans/2026-07-27-gpu-resident-present.md`).
//!
//! Phase 1 replaces "copy the finished image into a staging buffer, memcpy it
//! into a `Vec`, memcpy that into the frame-IPC slot" with a single
//! `vkCmdCopyImageToBuffer` **straight into the IPC slot**, by importing the
//! shared mapping as `VkDeviceMemory`. That is only possible where the driver
//! exposes this extension, and only correct if the slot pointer honours
//! `minImportedHostPointerAlignment`.
//!
//! This test pins the two facts the implementation depends on, so a driver or
//! device change that would silently disable the fast path fails loudly here
//! instead of quietly costing two full-frame copies per frame.
//!
//! It is a **capability** test, not a behaviour test: on a machine without the
//! extension it reports and passes, because the present path is required to
//! fall back to the buffered copy rather than break.

use ash::vk;

/// Probe the extension + alignment directly, independent of Raeen's device
/// setup, so a failure here is unambiguously about the driver.
fn probe() -> Option<(String, bool, vk::DeviceSize)> {
    // SAFETY: loads the platform Vulkan loader; a missing loader is reported
    // as an error, not UB.
    let entry = unsafe { ash::Entry::load() }.ok()?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let create = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: `create` is a live local; the instance is destroyed below.
    let instance = unsafe { entry.create_instance(&create, None) }.ok()?;

    // SAFETY: `instance` is live for the whole block.
    let result = unsafe {
        let mut found = None;
        // Only the first enumerated device is probed (same one Raeen picks by
        // default), so take it directly instead of a loop that never loops.
        if let Some(pd) = instance
            .enumerate_physical_devices()
            .ok()?
            .into_iter()
            .next()
        {
            let props = instance.get_physical_device_properties(pd);
            let name = props
                .device_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let exts = instance
                .enumerate_device_extension_properties(pd)
                .unwrap_or_default();
            let has = exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(c"VK_EXT_external_memory_host"));
            let mut align = 0;
            if has {
                let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
                instance.get_physical_device_properties2(pd, &mut p2);
                align = host.min_imported_host_pointer_alignment;
            }
            found = Some((name, has, align));
        }
        found
    };

    // SAFETY: no child objects were created from this instance.
    unsafe { instance.destroy_instance(None) };
    result
}

#[test]
fn external_memory_host_capability_is_usable_or_absent() {
    let Some((name, available, alignment)) = probe() else {
        eprintln!("external_memory_host: SKIP — no Vulkan loader/device");
        return;
    };

    if !available {
        // Not a failure: the present path must fall back to the buffered copy.
        eprintln!(
            "external_memory_host: {name} lacks VK_EXT_external_memory_host — \
             GPU-resident present phase 1 stays disabled on this device"
        );
        return;
    }

    eprintln!("external_memory_host: {name} supports it (alignment {alignment} B)");

    // An import alignment must be a non-zero power of two, or the "align the
    // slot pointer" arithmetic the present path does is meaningless.
    assert!(alignment > 0, "a supported import alignment cannot be zero");
    assert!(
        alignment.is_power_of_two(),
        "import alignment {alignment} must be a power of two to align slot offsets against"
    );
    // Sanity bound: an alignment larger than a slot would make per-slot import
    // impossible. Real drivers report a page (4096 measured on Radeon 760M).
    assert!(
        alignment <= 64 * 1024,
        "import alignment {alignment} B is implausibly large for per-slot import"
    );
}
