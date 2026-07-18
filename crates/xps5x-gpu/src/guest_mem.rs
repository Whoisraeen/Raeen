//! Identity-mapped guest memory for the PM4 command processor.
//!
//! Several Gen5 packets carry guest pointers out-of-band — `R_*_REGS_INDIRECT`
//! register lists and indirect-draw argument buffers (see
//! `kyty_graphics::run`). Kyty dereferences those pointers directly because
//! its command processor runs inside the guest's address space; XPS5X's
//! `GuestArena` is **identity-mapped**, so the same is true here: a guest
//! virtual address *is* a host address in this process.
//!
//! [`IdentityGuestMemory`] therefore implements
//! [`kyty_graphics::run::GuestMemory`] with a plain read — but only after
//! validating the whole range with `VirtualQuery`, because a corrupt DCB (or a
//! title poking addresses we mis-decoded) must degrade to a skipped packet,
//! not a host access violation.

use kyty_graphics::run::GuestMemory;

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
pub struct IdentityGuestMemory;

impl GuestMemory for IdentityGuestMemory {
    fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
        read_dwords_checked(addr, count)
    }
}

/// Upper bound on a single *resource* fetch (vertex/index/storage buffers,
/// textures): 256 MiB. Refuses wraparound garbage while covering any real
/// title resource — measured: Minecraft binds a 4 MiB vertex arena V#.
pub(crate) const MAX_RESOURCE_READ_DWORDS: u32 = 0x0400_0000;

/// Pages readable by guest-side fetches: everything a committed, non-guard
/// read may touch.
#[cfg(windows)]
const READABLE_PAGES: u32 = {
    use windows_sys::Win32::System::Memory::{
        PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_READONLY,
        PAGE_READWRITE, PAGE_WRITECOPY,
    };
    PAGE_READONLY
        | PAGE_READWRITE
        | PAGE_WRITECOPY
        | PAGE_EXECUTE_READ
        | PAGE_EXECUTE_READWRITE
        | PAGE_EXECUTE_WRITECOPY
};

/// Length of the committed-readable prefix of `[addr, addr+size)`: 0 when the
/// very first page is already bad. Shared by the validating read (which needs
/// prefix == size) and by error paths that must say *where* a guest range
/// stops being readable — the difference between a wild base and a lazy tail.
#[cfg(windows)]
fn committed_prefix_len(addr: u64, size: u64) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, VirtualQuery};

    let Some(end) = addr.checked_add(size) else {
        return 0;
    };
    let mut cursor = addr;
    while cursor < end {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: VirtualQuery inspects the address space only; it never
        // dereferences `cursor`, so any value is safe to pass.
        let got = unsafe {
            VirtualQuery(
                cursor as *const _,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if got == 0
            || info.State != MEM_COMMIT
            || (info.Protect & READABLE_PAGES) == 0
            || (info.Protect & PAGE_GUARD) != 0
        {
            break;
        }
        let region_end = info.BaseAddress as u64 + info.RegionSize as u64;
        if region_end <= cursor {
            break; // defensive: no forward progress means bad info
        }
        cursor = region_end;
    }
    // The containing region usually runs past `size`; cap at the query span.
    (cursor - addr).min(size)
}

/// Committed-readable prefix of a guest range, for naming a failed fetch
/// precisely (wild base ≈ 0; lazy tail ≈ a page-aligned interior cut).
#[cfg(windows)]
pub(crate) fn readable_prefix(addr: u64, size: u64) -> u64 {
    committed_prefix_len(addr, size)
}

#[cfg(not(windows))]
pub(crate) fn readable_prefix(_addr: u64, _size: u64) -> u64 {
    0
}

/// VirtualQuery-validated read of guest dwords (identity map). `None` when the
/// range is null/unaligned/oversized or not fully committed-readable. Shared
/// by [`IdentityGuestMemory`] and the shader fetch layer.
#[cfg(windows)]
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
#[cfg(windows)]
pub(crate) fn read_dwords_validated(addr: u64, count: u32) -> Option<Vec<u32>> {
    if count == 0 || count > MAX_RESOURCE_READ_DWORDS || addr == 0 || !addr.is_multiple_of(4) {
        return None;
    }
    let bytes = count as usize * 4;
    if committed_prefix_len(addr, bytes as u64) != bytes as u64 {
        return None;
    }

    let mut out = vec![0u32; count as usize];
    // SAFETY: [addr, addr+bytes) was just verified committed + readable in
    // this process (the guest arena is identity-mapped, so the guest pointer
    // is a host pointer). The read is unsynchronized with the guest — a
    // concurrent guest write can tear a value, which for register/args data
    // yields a stale or mixed *value*, never UB-through-invalid-memory; a
    // concurrent unmap is the same TOCTOU the whole identity-map runtime
    // accepts. `ptr::copy` (memmove) rather than `copy_nonoverlapping`: a
    // wild-but-committed guest range validated only at page granularity can
    // land arbitrarily in this process — including over the freshly allocated
    // `out` — and overlap must degrade to garbage *values*, not UB.
    unsafe {
        std::ptr::copy(addr as *const u8, out.as_mut_ptr().cast::<u8>(), bytes);
    }
    Some(out)
}

#[cfg(not(windows))]
pub(crate) fn read_dwords_checked(_addr: u64, _count: u32) -> Option<Vec<u32>> {
    // The identity-mapped runtime is Windows-only today (see CLAUDE.md);
    // keep the cfg honest rather than pretend to validate with libc tricks.
    None
}

#[cfg(not(windows))]
pub(crate) fn read_dwords_validated(_addr: u64, _count: u32) -> Option<Vec<u32>> {
    None
}

/// VirtualQuery-validated write into identity-mapped guest memory. Used for
/// Vulkan compute storage-buffer writeback after the queue fence signals.
#[cfg(windows)]
pub(crate) fn write_bytes_checked(addr: u64, bytes: &[u8]) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY,
        PAGE_GUARD, PAGE_READWRITE, PAGE_WRITECOPY, VirtualQuery,
    };

    if addr == 0 || bytes.is_empty() || bytes.len() > MAX_RESOURCE_READ_DWORDS as usize * 4 {
        return false;
    }
    let Some(end) = addr.checked_add(bytes.len() as u64) else {
        return false;
    };
    const WRITABLE: u32 =
        PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let mut cursor = addr;
    while cursor < end {
        // SAFETY: zero is a valid initial state for this output-only Win32 struct.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: VirtualQuery inspects the address space and does not
        // dereference the queried address.
        let got = unsafe {
            VirtualQuery(
                cursor as *const _,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if got == 0
            || info.State != MEM_COMMIT
            || (info.Protect & WRITABLE) == 0
            || (info.Protect & PAGE_GUARD) != 0
        {
            return false;
        }
        let region_end = info.BaseAddress as u64 + info.RegionSize as u64;
        if region_end <= cursor {
            return false;
        }
        cursor = region_end;
    }

    // SAFETY: the complete destination range was just verified committed and
    // writable. `ptr::copy` tolerates accidental overlap with the source.
    unsafe { std::ptr::copy(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
    true
}

#[cfg(not(windows))]
pub(crate) fn write_bytes_checked(_addr: u64, _bytes: &[u8]) -> bool {
    false
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn reads_committed_host_memory_identity_mapped() {
        let data: Vec<u32> = vec![0xAABB_CCDD, 1, 2, 3];
        let addr = data.as_ptr() as u64;
        let got = IdentityGuestMemory.read_dwords(addr, 4);
        assert_eq!(got, Some(data.clone()));
    }

    #[test]
    fn refuses_null_unaligned_zero_and_oversized() {
        let data: Vec<u32> = vec![1, 2];
        let addr = data.as_ptr() as u64;
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
    }

    #[test]
    fn writes_committed_host_memory_identity_mapped() {
        let mut data = vec![0u32; 4];
        let replacement = [1u32, 2, 3, 4];
        let bytes: Vec<u8> = replacement
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        assert!(write_bytes_checked(data.as_mut_ptr() as u64, &bytes));
        assert_eq!(data, replacement);
    }

    #[test]
    fn refuses_reserved_but_uncommitted_memory() {
        use windows_sys::Win32::System::Memory::{
            MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
        };
        // SAFETY: reserving (not committing) fresh pages touches no existing
        // mapping; released before the test ends.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null(), "VirtualAlloc(MEM_RESERVE) failed");
        let addr = base as u64;
        assert_eq!(
            IdentityGuestMemory.read_dwords(addr, 4),
            None,
            "reserved-but-uncommitted pages must be refused, not faulted on"
        );
        // SAFETY: releasing the reservation made above.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
    }

    /// A range that starts committed but runs into an uncommitted region must
    /// be refused as a whole.
    #[test]
    fn refuses_range_that_leaves_committed_memory() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        // Reserve two pages, commit only the first.
        // SAFETY: fresh reservation; released below.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: committing the first page of our own reservation.
        let commit = unsafe { VirtualAlloc(base, 0x1000, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!commit.is_null());
        let addr = base as u64 + 0x1000 - 8;
        assert_eq!(
            IdentityGuestMemory.read_dwords(addr, 4),
            None,
            "a read straddling into an uncommitted page must be refused"
        );
        assert!(
            IdentityGuestMemory.read_dwords(addr, 2).is_some(),
            "the committed prefix alone is readable"
        );
        // SAFETY: releasing the reservation made above.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
    }
}
