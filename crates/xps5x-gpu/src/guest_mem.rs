//! Process-authorized guest memory for the PM4 command processor.
//!
//! Several Gen5 packets carry guest pointers out-of-band — `R_*_REGS_INDIRECT`
//! register lists and indirect-draw argument buffers (see
//! `kyty_graphics::run`). Kyty dereferences those pointers directly because
//! its command processor runs inside the guest's address space; XPS5X's
//! `GuestArena` is **identity-mapped**, so the same is true here: a guest
//! virtual address *is* a host address in this process.
//!
//! A committed Windows page is not necessarily guest-owned. Every access is
//! therefore routed through the active process's [`GpuGuestMemory`] authority;
//! the GPU crate never probes or dereferences arbitrary host pages itself.

use kyty_graphics::run::GuestMemory;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

/// Address-space authority supplied by the owning guest process.
pub trait GpuGuestMemory: Send + Sync {
    fn validate_gpu_range(&self, addr: u64, len: u64, write: bool) -> bool;
    fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool;
    fn write_gpu(&self, addr: u64, data: &[u8]) -> bool;
}

pub(crate) struct DenyGpuMemory;

impl GpuGuestMemory for DenyGpuMemory {
    fn validate_gpu_range(&self, _addr: u64, _len: u64, _write: bool) -> bool {
        false
    }

    fn read_gpu(&self, _addr: u64, _out: &mut [u8]) -> bool {
        false
    }

    fn write_gpu(&self, _addr: u64, _data: &[u8]) -> bool {
        false
    }
}

thread_local! {
    static ACTIVE_MEMORY: RefCell<Option<Arc<dyn GpuGuestMemory>>> = RefCell::new(None);
    static ACTIVE_BUDGET: Cell<u64> = const { Cell::new(0) };
}

const MAX_SUBMISSION_GUEST_BYTES: u64 = 256 << 20;

struct ActiveMemoryGuard {
    memory: Option<Arc<dyn GpuGuestMemory>>,
    budget: u64,
}

impl Drop for ActiveMemoryGuard {
    fn drop(&mut self) {
        ACTIVE_MEMORY.with(|active| {
            *active.borrow_mut() = self.memory.take();
        });
        ACTIVE_BUDGET.with(|budget| budget.set(self.budget));
    }
}

pub(crate) fn with_guest_memory<T>(memory: &Arc<dyn GpuGuestMemory>, f: impl FnOnce() -> T) -> T {
    let previous = ACTIVE_MEMORY.with(|active| active.borrow_mut().replace(Arc::clone(memory)));
    let previous_budget = ACTIVE_BUDGET.with(|budget| budget.replace(MAX_SUBMISSION_GUEST_BYTES));
    let _guard = ActiveMemoryGuard {
        memory: previous,
        budget: previous_budget,
    };
    f()
}

fn with_active_memory<T>(f: impl FnOnce(&dyn GpuGuestMemory) -> T) -> Option<T> {
    ACTIVE_MEMORY.with(|active| active.borrow().as_deref().map(f))
}

fn charge_guest_bytes(bytes: u64) -> bool {
    ACTIVE_BUDGET.with(|budget| {
        let remaining = budget.get();
        if bytes > remaining {
            return false;
        }
        budget.set(remaining - bytes);
        true
    })
}

/// Upper bound on a single out-of-band read. The largest legitimate consumer
/// is an indirect register list (`UC_NUM` pairs = 32 Ki dwords); anything
/// bigger is a mis-decode.
const MAX_READ_DWORDS: u32 = 0x1_0000;

/// [`GuestMemory`] over the identity-mapped guest arena.
///
/// Reads are refused (returning `None`, which the command processor turns
/// into a rate-limited warn + packet skip) when the address is null,
/// unaligned, oversized, or any page in the range is not committed readable
/// memory in this process.
pub(crate) struct IdentityGuestMemory;

impl GuestMemory for IdentityGuestMemory {
    fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
        read_dwords_checked(addr, count)
    }

    /// DMA payload read: dword-granular under the RESOURCE cap (a DMA fill of
    /// a 1080p scanout buffer is ~8 MiB — far beyond the pointer-read cap but
    /// a legitimate resource-sized transfer).
    fn read_bytes(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        if len == 0 || !len.is_multiple_of(4) {
            return None;
        }
        let count = u32::try_from(len / 4).ok()?;
        let dwords = read_dwords_validated(addr, count)?;
        Some(dwords.iter().flat_map(|w| w.to_le_bytes()).collect())
    }

    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        write_bytes_checked(addr, bytes)
    }
}

/// Upper bound on a single *resource* fetch (vertex/index/storage buffers,
/// textures): 256 MiB. Refuses wraparound garbage while covering any real
/// title resource — measured: Minecraft binds a 4 MiB vertex arena V#.
pub(crate) const MAX_RESOURCE_READ_DWORDS: u32 = 0x0400_0000;

/// Committed-readable prefix of a guest range, for naming a failed fetch
/// precisely (wild base ≈ 0; lazy tail ≈ a page-aligned interior cut).
pub(crate) fn readable_prefix(addr: u64, size: u64) -> u64 {
    with_active_memory(|memory| {
        if memory.validate_gpu_range(addr, size, false) {
            size
        } else {
            0
        }
    })
    .unwrap_or(0)
}

/// VirtualQuery-validated read of guest dwords (identity map). `None` when the
/// range is null/unaligned/oversized or not fully committed-readable. Shared
/// by [`IdentityGuestMemory`] and the shader fetch layer.
pub(crate) fn read_dwords_checked(addr: u64, count: u32) -> Option<Vec<u32>> {
    if count == 0 || count > MAX_READ_DWORDS {
        return None;
    }
    read_dwords_validated(addr, count)
}

/// The same validated read without the out-of-band cap, for guest *resources*
/// (vertex/storage buffers, textures) whose legitimate size runs to megabytes.
/// `MAX_READ_DWORDS` exists only to refuse mis-decoded command-processor
/// pointer reads; a V# declaring a 4 MiB vertex arena is not a mis-decode
/// (measured on Minecraft's menu draws).
pub(crate) fn read_dwords_validated(addr: u64, count: u32) -> Option<Vec<u32>> {
    if count == 0 || count > MAX_RESOURCE_READ_DWORDS || addr == 0 || !addr.is_multiple_of(4) {
        return None;
    }
    let bytes = count as usize * 4;
    if !charge_guest_bytes(bytes as u64) {
        return None;
    }
    let mut out = Vec::<u32>::new();
    out.try_reserve_exact(count as usize).ok()?;
    out.resize(count as usize, 0);
    // SAFETY: `out` owns `count * 4` initialized bytes and u32 has no invalid
    // bit patterns. The slice is used only as the destination of a bounded
    // process-authorized copy.
    let raw = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes) };
    let accepted = with_active_memory(|memory| {
        memory.validate_gpu_range(addr, bytes as u64, false) && memory.read_gpu(addr, raw)
    })?;
    if !accepted {
        return None;
    }
    Some(out)
}

/// Process-authorized write into GPU-visible guest memory. Used for Vulkan
/// compute storage-buffer writeback after the queue fence signals.
pub(crate) fn write_bytes_checked(addr: u64, bytes: &[u8]) -> bool {
    if addr == 0 || bytes.is_empty() || bytes.len() > MAX_RESOURCE_READ_DWORDS as usize * 4 {
        return false;
    }
    if addr.checked_add(bytes.len() as u64).is_none() {
        return false;
    }
    if !charge_guest_bytes(bytes.len() as u64) {
        return false;
    }
    with_active_memory(|memory| {
        memory.validate_gpu_range(addr, bytes.len() as u64, true) && memory.write_gpu(addr, bytes)
    })
    .unwrap_or(false)
}

/// Install a bounded host-allocation authority for unit tests that model
/// identity-mapped guest buffers with live Vec/array storage. Production code
/// cannot call this and never regains the old arbitrary-host-page probe.
#[cfg(test)]
pub(crate) fn with_test_ranges<T>(ranges: &[(u64, usize)], f: impl FnOnce() -> T) -> T {
    struct TestRanges(Vec<(u64, u64)>);

    impl GpuGuestMemory for TestRanges {
        fn validate_gpu_range(&self, addr: u64, len: u64, _write: bool) -> bool {
            self.0.iter().any(|&(start, size)| {
                addr >= start
                    && addr
                        .checked_add(len)
                        .is_some_and(|end| end <= start.saturating_add(size))
            })
        }

        fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
            if !self.validate_gpu_range(addr, out.len() as u64, false) {
                return false;
            }
            // SAFETY: test callers describe live allocations and the validated
            // access is wholly contained in one of those allocations for the
            // synchronous closure lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len())
            };
            true
        }

        fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
            if !self.validate_gpu_range(addr, data.len() as u64, true) {
                return false;
            }
            // SAFETY: same bounded live-allocation proof as `read_gpu`.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
            true
        }
    }

    let memory: Arc<dyn GpuGuestMemory> = Arc::new(TestRanges(
        ranges
            .iter()
            .map(|&(start, size)| (start, size as u64))
            .collect(),
    ));
    with_guest_memory(&memory, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct HostRange {
        start: u64,
        len: u64,
    }

    impl GpuGuestMemory for HostRange {
        fn validate_gpu_range(&self, addr: u64, len: u64, _write: bool) -> bool {
            addr >= self.start
                && addr
                    .checked_add(len)
                    .is_some_and(|end| end <= self.start + self.len)
        }

        fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
            if !self.validate_gpu_range(addr, out.len() as u64, false) {
                return false;
            }
            // SAFETY: this test authority is created from a live Vec and the
            // validated range stays within that Vec for the closure lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len())
            };
            true
        }

        fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
            if !self.validate_gpu_range(addr, data.len() as u64, true) {
                return false;
            }
            // SAFETY: same bounded test allocation argument as `read_gpu`.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
            true
        }
    }

    fn memory_for(data: &mut [u32]) -> Arc<dyn GpuGuestMemory> {
        Arc::new(HostRange {
            start: data.as_mut_ptr() as u64,
            len: std::mem::size_of_val(data) as u64,
        })
    }

    #[test]
    fn reads_committed_host_memory_identity_mapped() {
        let mut data: Vec<u32> = vec![0xAABB_CCDD, 1, 2, 3];
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        let got = with_guest_memory(&memory, || IdentityGuestMemory.read_dwords(addr, 4));
        assert_eq!(got, Some(data.clone()));
    }

    #[test]
    fn refuses_null_unaligned_zero_and_oversized() {
        let mut data: Vec<u32> = vec![1, 2];
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        with_guest_memory(&memory, || {
            assert_eq!(IdentityGuestMemory.read_dwords(0, 1), None, "null");
            assert_eq!(
                IdentityGuestMemory.read_dwords(addr + 2, 1),
                None,
                "unaligned"
            );
            assert_eq!(IdentityGuestMemory.read_dwords(addr, 0), None, "zero len");
            assert_eq!(
                IdentityGuestMemory.read_dwords(addr, MAX_READ_DWORDS + 1),
                None,
                "over the cap"
            );
        });
    }

    #[test]
    fn writes_committed_host_memory_identity_mapped() {
        let mut data = vec![0u32; 4];
        let replacement = [1u32, 2, 3, 4];
        let bytes: Vec<u8> = replacement
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let addr = data.as_mut_ptr() as u64;
        let memory = memory_for(&mut data);
        assert!(with_guest_memory(&memory, || write_bytes_checked(
            addr, &bytes
        )));
        assert_eq!(data, replacement);
    }

    #[test]
    fn refuses_memory_without_process_authority() {
        let data = [1u32, 2, 3, 4];
        assert_eq!(
            IdentityGuestMemory.read_dwords(data.as_ptr() as u64, 4),
            None,
            "committed host memory outside a GuestProcess must be refused"
        );
    }

    /// A range that starts committed but runs into an uncommitted region must
    /// be refused as a whole.
    #[test]
    fn refuses_range_that_leaves_process_authority() {
        let mut data = vec![1u32, 2];
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        with_guest_memory(&memory, || {
            assert_eq!(IdentityGuestMemory.read_dwords(addr, 4), None);
            assert!(IdentityGuestMemory.read_dwords(addr, 2).is_some());
        });
    }
}
