//! Guest image mapping: copies a [`xps5x_firmware::LinkedModule`]'s `image`
//! into a host allocation the guest can execute from, and frees it on
//! `Drop`.
//!
//! # RT0 W^X shortcut
//!
//! Real per-segment W^X (write XOR execute) protections, matching each
//! `PT_LOAD` segment's own flags, are a later milestone (design doc §4/§7).
//! RT0 maps the entire image as a single `PAGE_EXECUTE_READWRITE` region for
//! simplicity — this is a deliberate, documented shortcut, not an
//! oversight. It is bounded by the same trust boundary as the rest of the
//! runtime (design doc §6): only images the LM1 pipeline produced are ever
//! mapped and executed here.

use core::ffi::c_void;
use core::ptr::NonNull;

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree,
};

use crate::RuntimeError;

/// x86-64 Windows' page size. Fixed at 4 KiB for every supported release;
/// hardcoded rather than queried via `GetSystemInfo` to avoid an extra API
/// surface for RT0.
const PAGE_SIZE: u64 = 4096;

/// A `LinkedModule.image` copied into a `PAGE_EXECUTE_READWRITE` host
/// allocation. Owns the allocation; frees it via `VirtualFree` on `Drop`.
pub(crate) struct MappedImage {
    ptr: NonNull<u8>,
    /// The real (unpadded) `LinkedModule.image` length, for entry-offset
    /// bounds checks. The underlying allocation may be larger (page-rounded).
    image_len: usize,
    /// The actual (page-rounded) committed allocation length — the full
    /// range of host bytes backing this mapping, and therefore the bound
    /// [`GuestMemory`] read/write checks against (guest offsets into the
    /// zero-padding past `image_len` are still valid, committed memory).
    alloc_len: usize,
}

impl MappedImage {
    /// Allocate a `PAGE_EXECUTE_READWRITE` region sized to (page-rounded)
    /// `image.len()`, copy `image` into it, and return the mapping.
    pub(crate) fn map(image: &[u8]) -> Result<Self, RuntimeError> {
        let alloc_len = ((image.len() as u64).max(1)).div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let alloc_len = alloc_len as usize;

        // SAFETY: a null `lpAddress` lets the OS place the allocation
        // anywhere in the address space; `alloc_len` is a valid, nonzero,
        // page-rounded size, and `MEM_COMMIT | MEM_RESERVE` with
        // `PAGE_EXECUTE_READWRITE` is a well-formed `VirtualAlloc` request.
        // The returned pointer, if non-null, is exclusively owned by the
        // `MappedImage` constructed below and freed exactly once in `Drop`.
        let raw = unsafe {
            VirtualAlloc(
                core::ptr::null(),
                alloc_len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            )
        };
        let ptr = NonNull::new(raw as *mut u8).ok_or(RuntimeError::MapFailed)?;

        // SAFETY: `ptr` was just allocated above with `alloc_len >=
        // image.len()` writable, exclusively-owned bytes; `image` is a
        // disjoint, immutably-borrowed source slice of exactly `image.len()`
        // bytes. Non-overlapping copy of `image.len()` bytes stays within
        // both.
        unsafe {
            core::ptr::copy_nonoverlapping(image.as_ptr(), ptr.as_ptr(), image.len());
        }

        Ok(Self {
            ptr,
            image_len: image.len(),
            alloc_len,
        })
    }

    /// The host address of guest offset `entry_offset` into the mapped
    /// image, bounds-checked against the real (unpadded) image length.
    pub(crate) fn entry_ptr(&self, entry_offset: u64) -> Result<*const u8, RuntimeError> {
        let offset = usize::try_from(entry_offset).map_err(|_| RuntimeError::MapFailed)?;
        if offset >= self.image_len {
            return Err(RuntimeError::MapFailed);
        }
        // SAFETY: `offset < self.image_len`, and the allocation backing
        // `self.ptr` is at least `self.image_len` bytes (see `map`), so the
        // resulting pointer stays within the allocation.
        Ok(unsafe { self.ptr.as_ptr().add(offset) })
    }
}

/// [`crate::dispatch`]'s VEH gives every HLE call a [`xps5x_hle::GuestMemory`]
/// view of the currently-executing guest's address space. RT0 images are
/// laid out with guest vaddr `0` at the start of the mapping (design doc
/// §2), so "guest vaddr `V`" and "`mapped_base + V`" are the same offset —
/// translation is just pointer arithmetic, bounds-checked against
/// `alloc_len` (the real, committed allocation size, not just the unpadded
/// `image_len`) so an HLE function handed a wild guest pointer gets `false`
/// rather than an OOB host read/write.
impl xps5x_hle::GuestMemory for MappedImage {
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else { return false };
        let Some(end) = addr.checked_add(out.len()) else { return false };
        if end > self.alloc_len {
            return false;
        }
        // SAFETY: `addr + out.len() <= self.alloc_len`, and `self.ptr` is a
        // committed, readable allocation of at least `self.alloc_len` bytes
        // (see `map`) that this `MappedImage` exclusively owns for its
        // lifetime — no other live reference can be writing these same
        // bytes concurrently under RT0's single-active-execution invariant
        // (design doc §4/§6/§9, `dispatch::CALL_LOCK`).
        unsafe {
            core::ptr::copy_nonoverlapping(self.ptr.as_ptr().add(addr), out.as_mut_ptr(), out.len());
        }
        true
    }

    fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else { return false };
        let Some(end) = addr.checked_add(data.len()) else { return false };
        if end > self.alloc_len {
            return false;
        }
        // SAFETY: same bounds argument as `read` above; the allocation is
        // `PAGE_EXECUTE_READWRITE` (see `map`), so it is writable, and the
        // single-active-execution invariant means no concurrent access to
        // these bytes exists while this call runs.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr().add(addr), data.len());
        }
        true
    }
}

impl Drop for MappedImage {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is the exact pointer returned by `VirtualAlloc`
        // in `map` and is freed here exactly once, with `dwSize = 0` as
        // `MEM_RELEASE` requires.
        unsafe {
            VirtualFree(self.ptr.as_ptr() as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
