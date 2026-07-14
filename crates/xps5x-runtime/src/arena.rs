//! [`GuestArena`]: the fixed-base, identity-mapped guest address space
//! (design doc §2/§3/§5). Reserves one contiguous 4 GiB host region at
//! [`crate::GUEST_ARENA_BASE`], commits its four fixed sub-regions (image,
//! heap, stack, mmap) with the layout's protections, copies the module image
//! in, and frees the whole reservation on `Drop`.
//!
//! Guest address `A` *is* host address `A` here (identity mapping): a guest
//! pointer returned by `alloc`/`mmap`, or baked into the image by the
//! linker's relocations, is directly dereferenceable by the native CPU — no
//! translation layer, unlike the now-retired `mem::MappedImage` this module
//! replaces.
//!
//! # RT2 status
//!
//! Wired into [`crate::execute_linked`] since RT2 Task 3: every guest
//! execution builds a real `GuestArena` and passes it as both the
//! `GuestMemory` and `GuestAllocator` view.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;

use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE,
    PAGE_NOACCESS, PAGE_READWRITE,
};

use xps5x_hle::{GuestAllocator, GuestMemory};

use crate::{RuntimeError, GUEST_ARENA_BASE};

/// x86-64 Windows' page size (see `mem.rs`/`trampoline.rs`'s equivalent
/// constants); also the default alignment `mmap` bumps to.
const PAGE_SIZE: u64 = 4096;

/// Image region: `[base + IMAGE_OFFSET, base + IMAGE_OFFSET + IMAGE_SIZE)`,
/// committed `PAGE_EXECUTE_READWRITE` (design doc §3, the documented RT
/// shortcut — see `mem.rs`'s equivalent note).
const IMAGE_OFFSET: u64 = 0x0;
const IMAGE_SIZE: u64 = 0x4000_0000; // 1 GiB

/// Heap region: `[base + HEAP_OFFSET, base + STACK_OFFSET)`, committed
/// `PAGE_READWRITE`. Backs `GuestAllocator::alloc`/`free`/`realloc`.
const HEAP_OFFSET: u64 = 0x4000_0000;
const HEAP_SIZE: u64 = 0x4000_0000; // 1 GiB

/// Stack region: `[base + STACK_OFFSET, base + MMAP_OFFSET)`, committed
/// `PAGE_READWRITE`. Reserved and committed this task but otherwise unused —
/// the RSP switch onto it is a later milestone (design doc §7).
const STACK_OFFSET: u64 = 0x8000_0000;
const STACK_SIZE: u64 = 0x2000_0000; // 512 MiB

/// Mmap region: `[base + MMAP_OFFSET, base + ARENA_SPAN)`, committed
/// `PAGE_READWRITE`. Backs `GuestAllocator::mmap`/`munmap`.
const MMAP_OFFSET: u64 = 0xA000_0000;
const MMAP_SIZE: u64 = 0x6000_0000; // 1.5 GiB

/// Total reserved span: `IMAGE_SIZE + HEAP_SIZE + STACK_SIZE + MMAP_SIZE`,
/// i.e. the four sub-regions exactly tile `[base, base + ARENA_SPAN)` with
/// no gaps (design doc §3's layout table).
const ARENA_SPAN: u64 = 0x1_0000_0000; // 4 GiB

/// Size of the main-thread TCB [`GuestArena::setup_main_tcb`] carves from
/// the heap region (design doc §3, RT2c-b/M1-B). Large enough for the
/// self-pointer at offset 0, the `__stack_chk_guard` canary at the
/// ABI-mandated `fs:0x28`, and headroom for small TLS-offset probes. The
/// module's `PT_TLS` init image is laid out immediately *below* the TCB
/// (variant-II x86-64 TLS), sized separately from its template.
const TCB_SIZE: u64 = 0x800; // 2 KiB

/// The `fs:`-relative offset of the stack-protector canary
/// (`__stack_chk_guard`) in the x86-64 ABI as extended by glibc/Orbis libc:
/// compiler-generated prologues/epilogues read `fs:0x28` unconditionally
/// whenever stack-protector is enabled (M1-B, wall #2).
const CANARY_TCB_OFFSET: usize = 0x28;

/// A per-process stack-protector canary value: derived from
/// [`std::collections::hash_map::RandomState`]'s per-process random keys
/// (no new dependency), with the low byte forced to zero (glibc's
/// "terminator canary" convention — a NUL so string functions can't leak
/// it) and guaranteed nonzero (the m1-homebrew anti-pattern: a zero canary
/// would let stack-protected code "work" by coincidence against a zeroed
/// TCB rather than proving a real install).
fn stack_canary() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(0x5A_FE_57_AC_C4_AA_2D_00);
    let masked = hasher.finish() & !0xFF;
    if masked == 0 {
        0x100
    } else {
        masked
    }
}

/// Round `align` up to a power of two no smaller than 16 (the minimum
/// alignment `GuestAllocator` methods honor, design doc §5), returning
/// `None` — rather than panicking — if `align` is too large for
/// `next_power_of_two` to compute without overflow.
fn normalize_align(align: u64) -> Option<u64> {
    let align = align.max(16);
    if align.is_power_of_two() {
        return Some(align);
    }
    // `next_power_of_two` panics on overflow for inputs above roughly
    // `1 << 63`; no legitimate caller needs an alignment anywhere near that,
    // so reject rather than risk it.
    if align > (1u64 << 62) {
        return None;
    }
    Some(align.next_power_of_two())
}

/// Round `value` up to the next multiple of `align` (`align` must already be
/// a power of two — every caller here passes the result of
/// [`normalize_align`]), returning `None` on overflow rather than panicking.
fn align_up(value: u64, align: u64) -> Option<u64> {
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

/// Commit `len` bytes at `addr` — a sub-range of the `[base, base +
/// ARENA_SPAN)` region `new` already reserved via `VirtualAlloc(MEM_RESERVE)`
/// — with protection `protect`. On failure, releases the *entire* `base`
/// reservation (undoing any sub-regions already committed) so `new`'s caller
/// gets a clean `Err` with nothing leaked.
fn commit_region(base: u64, addr: u64, len: u64, protect: u32) -> Result<(), RuntimeError> {
    // SAFETY: `addr` lies within `[base, base + ARENA_SPAN)`, a range already
    // reserved (not yet committed) by the `MEM_RESERVE` call in `new`, so
    // committing a sub-range of it is well-formed; `len` is a fixed,
    // page-aligned layout constant and `protect` is one of the fixed
    // `PAGE_*` constants passed by this module's own callers.
    let raw = unsafe { VirtualAlloc(addr as *const c_void, len as usize, MEM_COMMIT, protect) };
    if raw.is_null() {
        // SAFETY: `base` is the exact address the outer `MEM_RESERVE` call in
        // `new` returned. Releasing it with `dwSize = 0` (`MEM_RELEASE`)
        // frees the whole reservation — including any sub-regions already
        // committed above — exactly once, undoing this partially
        // constructed arena.
        unsafe {
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
        return Err(RuntimeError::MapFailed);
    }
    Ok(())
}

/// The heap/mmap allocator's interior-mutable state, guarded by
/// [`GuestArena`]'s `Mutex`. `heap_sizes` maps a live (not-yet-freed)
/// allocation's address to its committed block size, so `free`/`realloc` can
/// recover how much to release/copy without the caller repeating it.
struct AllocState {
    /// Next never-yet-used heap address; only ever moves forward.
    heap_bump: u64,
    /// Blocks released by `free`, available for first-fit reuse: `(addr,
    /// size)`.
    heap_free: Vec<(u64, u64)>,
    /// Live allocations' sizes, keyed by address.
    heap_sizes: HashMap<u64, u64>,
    /// Next never-yet-used mmap address; only ever moves forward. RT2a has
    /// no mmap free list — `munmap` is best-effort (design doc §5/§7).
    mmap_bump: u64,
}

impl AllocState {
    fn new(base: u64) -> Self {
        Self {
            heap_bump: base + HEAP_OFFSET,
            heap_free: Vec::new(),
            heap_sizes: HashMap::new(),
            mmap_bump: base + MMAP_OFFSET,
        }
    }
}

/// The fixed-base, identity-mapped guest address space (design doc §2/§5).
/// Owns the entire `[GUEST_ARENA_BASE, GUEST_ARENA_BASE + ARENA_SPAN)` host
/// reservation; frees it via `VirtualFree` on `Drop`.
///
/// Only one `GuestArena` may exist at a time (fixed base + the runtime's
/// single-active-execution invariant, `dispatch::CALL_LOCK` — design doc
/// §2/§9): a second concurrent reservation attempt at the same fixed address
/// would fail with `MapFailed`. Every test in this module that constructs a
/// real `GuestArena` therefore holds `crate::dispatch::call_lock()` for its
/// duration.
pub(crate) struct GuestArena {
    /// Always [`GUEST_ARENA_BASE`] — stored rather than referenced directly
    /// so `GuestMemory`/`GuestAllocator` read this struct's own state, not a
    /// free-standing global.
    base: u64,
    /// The real (unpadded) `LinkedModule.image` length passed to `new`, for
    /// `entry_ptr`'s bounds check. The committed image *region* is always
    /// the full `IMAGE_SIZE`, page-aligned and larger than this.
    image_len: u64,
    state: Mutex<AllocState>,
}

impl GuestArena {
    /// Reserve the fixed-base 4 GiB arena, commit its four sub-regions with
    /// their layout protections, and copy `image` into the image region.
    /// `image.len() > IMAGE_SIZE` fails with `MapFailed` before anything is
    /// reserved.
    pub(crate) fn new(image: &[u8]) -> Result<Self, RuntimeError> {
        // `image.len()` (a `usize`) never truncates when widened to `u64` on
        // any platform this crate targets (Windows x86-64, where
        // `usize == u64`).
        let image_len = image.len() as u64;
        if image_len > IMAGE_SIZE {
            return Err(RuntimeError::MapFailed);
        }

        let base = GUEST_ARENA_BASE;

        // SAFETY: `base` is a fixed, far-more-than-page-aligned high
        // sentinel address (design doc §3), passed as an explicit non-null
        // `lpAddress`. `VirtualAlloc` with an explicit address either
        // reserves exactly that range or fails (returns null) — it never
        // silently relocates elsewhere. `MEM_RESERVE` (no commit yet) with
        // `PAGE_NOACCESS` is a well-formed request; the reservation is
        // released exactly once, either by `commit_region`'s cleanup path
        // below or by `Drop`.
        let raw = unsafe {
            VirtualAlloc(
                base as *const c_void,
                ARENA_SPAN as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
            )
        };
        if raw.is_null() {
            return Err(RuntimeError::MapFailed);
        }
        // `VirtualAlloc` with an explicit non-null `lpAddress` reserves exactly
        // that (allocation-granularity-aligned) range or returns null — it
        // never relocates — so `raw == base` always holds here. Guard it as a
        // hard error rather than only a `debug_assert` anyway: if that
        // invariant were ever violated, the rest of this function (and every
        // identity read/write) uses the `base` *constant*, so a mismatched
        // `raw` would leak the real reservation (Drop frees `base`) and commit
        // into unreserved space. Free `raw` and fail cleanly instead.
        if raw as u64 != base {
            // SAFETY: `raw` is the exact non-null pointer `VirtualAlloc` just
            // returned; releasing it once with `dwSize = 0` (`MEM_RELEASE`) is
            // well-formed and undoes this reservation.
            unsafe {
                VirtualFree(raw, 0, MEM_RELEASE);
            }
            return Err(RuntimeError::MapFailed);
        }

        commit_region(
            base,
            base + IMAGE_OFFSET,
            IMAGE_SIZE,
            PAGE_EXECUTE_READWRITE,
        )?;
        commit_region(base, base + HEAP_OFFSET, HEAP_SIZE, PAGE_READWRITE)?;
        commit_region(base, base + STACK_OFFSET, STACK_SIZE, PAGE_READWRITE)?;
        commit_region(base, base + MMAP_OFFSET, MMAP_SIZE, PAGE_READWRITE)?;

        // SAFETY: the image region `[base, base + IMAGE_SIZE)` is now
        // committed, writable memory (the `commit_region` call just above),
        // and `IMAGE_SIZE >= image_len == image.len()` (checked at the top
        // of this function); `image` is a disjoint, immutably-borrowed
        // source slice of exactly `image.len()` bytes. Non-overlapping copy
        // of `image.len()` bytes stays within both.
        unsafe {
            core::ptr::copy_nonoverlapping(image.as_ptr(), base as *mut u8, image.len());
        }

        Ok(Self {
            base,
            image_len,
            state: Mutex::new(AllocState::new(base)),
        })
    }

    /// The host address of guest offset `entry_offset` into the mapped
    /// image, bounds-checked against the real (unpadded) image length.
    pub(crate) fn entry_ptr(&self, entry_offset: u64) -> Result<*const u8, RuntimeError> {
        if entry_offset >= self.image_len {
            return Err(RuntimeError::MapFailed);
        }
        // `self.base + entry_offset` cannot overflow: `entry_offset <
        // self.image_len <= IMAGE_SIZE`, and `self.base` (`GUEST_ARENA_BASE`)
        // is far below `u64::MAX - IMAGE_SIZE`. No dereference happens here,
        // so no `unsafe` is needed to build the pointer value itself.
        Ok((self.base + entry_offset) as *const u8)
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, AllocState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The host address of the top (highest address) of the guest stack
    /// region — `[base + STACK_OFFSET, base + STACK_OFFSET + STACK_SIZE)`
    /// (design doc §2/§7, RT2c-a). This is the initial RSP a guest call
    /// should start with; the stack grows down from here toward `base +
    /// STACK_OFFSET`. Always 16-byte aligned: `base` (`GUEST_ARENA_BASE`),
    /// `STACK_OFFSET`, and `STACK_SIZE` are all far-more-than-16-aligned
    /// constants, so their sum is too — [`crate::stack::call_on_guest_stack`]
    /// relies on this alignment.
    pub(crate) fn stack_top(&self) -> u64 {
        self.base + STACK_OFFSET + STACK_SIZE
    }

    /// Set up the main-thread TCB and, if the module has a `PT_TLS`
    /// template, its static TLS block (design doc §3, RT2c-b/M1-B):
    ///
    /// - Carves `tls.block_size() + TCB_SIZE` bytes from the heap allocator
    ///   (variant-II x86-64 layout: the TLS block sits immediately *below*
    ///   the TCB, so a `TPOFF64`-relocated `fs:[-off]` access lands in it).
    ///   `TlsTemplate::block_size` is the same size the LM1 linker computed
    ///   `TPOFF64` offsets against — the two must agree exactly.
    /// - Copies the `.tdata` init image into the block's start; `.tbss` and
    ///   padding stay zero.
    /// - Writes the TCB self-pointer at `fs:[0]` (the FreeBSD/Orbis
    ///   "variant II" convention) and a real, nonzero `__stack_chk_guard`
    ///   canary at the ABI-mandated `fs:0x28` ([`CANARY_TCB_OFFSET`]).
    ///
    /// Returns the TCB's guest address (the FS base to install), or `None`
    /// if the heap allocation or the write-back fails.
    pub(crate) fn setup_main_tcb(&self, tls: Option<&xps5x_firmware::TlsTemplate>) -> Option<u64> {
        let tls_block = tls.map(|t| t.block_size()).unwrap_or(0);
        let align = tls.map(|t| t.align.max(16)).unwrap_or(16);
        let total = tls_block.checked_add(TCB_SIZE)?;
        let base = self.alloc(total, align)?;
        let tcb = base.checked_add(tls_block)?;

        let mut block = vec![0u8; total as usize];
        if let Some(t) = tls {
            // `.tdata` at the block's start; a template whose `data`
            // somehow exceeds `block_size()` (malformed: filesz > memsz)
            // is truncated rather than trusted.
            let n = t.data.len().min(tls_block as usize);
            block[..n].copy_from_slice(&t.data[..n]);
        }
        let tcb_off = tls_block as usize;
        block[tcb_off..tcb_off + 8].copy_from_slice(&tcb.to_le_bytes());
        block[tcb_off + CANARY_TCB_OFFSET..tcb_off + CANARY_TCB_OFFSET + 8]
            .copy_from_slice(&stack_canary().to_le_bytes());

        if !self.write(base, &block) {
            return None;
        }
        Some(tcb)
    }
}

impl Drop for GuestArena {
    fn drop(&mut self) {
        // SAFETY: `self.base` is the exact address `VirtualAlloc(MEM_RESERVE)`
        // returned in `new`, released here exactly once with `dwSize = 0` as
        // `MEM_RELEASE` requires. Releasing the reservation also releases
        // every committed sub-region within it.
        unsafe {
            VirtualFree(self.base as *mut c_void, 0, MEM_RELEASE);
        }
    }
}

/// Identity `GuestMemory`: guest address `A` is host address `A` (design doc
/// §2). `read`/`write` bounds-check `[guest_addr, guest_addr + len)` against
/// the whole committed arena span (via `checked_add`, so an overflowing
/// request returns `false` rather than wrapping) before ever touching host
/// memory — an HLE function handed a wild guest pointer gets `false`, never
/// an OOB host access or a panic.
impl GuestMemory for GuestArena {
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
        // `out.len()` (a `usize`) never truncates when widened to `u64` on
        // this crate's target (Windows x86-64, `usize == u64`).
        let len = out.len() as u64;
        if guest_addr < self.base {
            return false;
        }
        let Some(end) = guest_addr.checked_add(len) else {
            return false;
        };
        if end > self.base + ARENA_SPAN {
            return false;
        }
        // SAFETY: `[guest_addr, guest_addr + out.len())` lies within
        // `[self.base, self.base + ARENA_SPAN)`, which `new` committed in
        // full as readable memory (the image/heap/stack/mmap sub-regions
        // exactly tile the span). Guest address == host address here
        // (identity mapping), so `guest_addr as *const u8` is a valid,
        // readable host pointer for `out.len()` bytes. No other live
        // reference writes these bytes concurrently under the
        // single-active-execution invariant (design doc §9,
        // `dispatch::CALL_LOCK`).
        unsafe {
            core::ptr::copy_nonoverlapping(guest_addr as *const u8, out.as_mut_ptr(), out.len());
        }
        true
    }

    fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
        let len = data.len() as u64;
        if guest_addr < self.base {
            return false;
        }
        let Some(end) = guest_addr.checked_add(len) else {
            return false;
        };
        if end > self.base + ARENA_SPAN {
            return false;
        }
        // SAFETY: same bounds argument as `read` above. Every sub-region is
        // committed `PAGE_READWRITE` or `PAGE_EXECUTE_READWRITE` (see
        // `new`), so it is writable, and the single-active-execution
        // invariant means no concurrent access to these bytes exists while
        // this call runs.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), guest_addr as *mut u8, data.len());
        }
        true
    }
}

/// Heap bump/free-list allocator + mmap bump allocator, both scoped to their
/// fixed sub-regions of the arena (design doc §5).
impl GuestAllocator for GuestArena {
    fn alloc(&self, size: u64, align: u64) -> Option<u64> {
        let align = normalize_align(align)?;
        let size = align_up(size.max(1), align)?;
        let mut state = self.lock_state();

        // First-fit reuse: any freed block big enough, whose address
        // already satisfies the requested alignment.
        if let Some(idx) = state
            .heap_free
            .iter()
            .position(|&(addr, blk_size)| blk_size >= size && addr % align == 0)
        {
            let (addr, blk_size) = state.heap_free.remove(idx);
            state.heap_sizes.insert(addr, blk_size);
            return Some(addr);
        }

        // No reusable block: bump within the heap region.
        let addr = align_up(state.heap_bump, align)?;
        let end = addr.checked_add(size)?;
        if end > self.base + STACK_OFFSET {
            // Heap region is [base + HEAP_OFFSET, base + STACK_OFFSET).
            return None;
        }
        state.heap_bump = end;
        state.heap_sizes.insert(addr, size);
        Some(addr)
    }

    fn free(&self, addr: u64) {
        let mut state = self.lock_state();
        if let Some(size) = state.heap_sizes.remove(&addr) {
            state.heap_free.push((addr, size));
        }
        // An unrecognized `addr` is simply ignored (trait contract).
    }

    fn realloc(&self, addr: u64, new_size: u64) -> Option<u64> {
        let old_size = {
            let state = self.lock_state();
            state.heap_sizes.get(&addr).copied()
        };

        let new_addr = self.alloc(new_size, 16)?;

        if let Some(old_size) = old_size {
            let copy_len = old_size.min(new_size) as usize;
            if copy_len > 0 {
                // SAFETY: `addr` is a live allocation recorded in
                // `heap_sizes` (just confirmed above) — i.e. a previous
                // `alloc`/`realloc` return value, which is always inside the
                // committed heap region with at least `old_size` bytes
                // available; `new_addr` was just returned by `self.alloc`
                // above with at least `new_size >= copy_len` bytes available,
                // also inside the committed heap region. `copy_len =
                // min(old_size, new_size)` fits within both, and the two
                // blocks are distinct, non-overlapping allocations (a fresh
                // block is never handed out while `addr`'s old block is
                // still live, since `free(addr)` below hasn't run yet but
                // `alloc` above never reuses a still-`heap_sizes`-tracked
                // address).
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        addr as *const u8,
                        new_addr as *mut u8,
                        copy_len,
                    );
                }
            }
        }
        // If `addr` wasn't a recognized live allocation, there is nothing
        // known to copy — `new_addr` is returned as a fresh, empty block,
        // matching `free`'s "unrecognized address is ignored" contract.

        self.free(addr);
        Some(new_addr)
    }

    fn mmap(&self, length: u64, align: u64) -> Option<u64> {
        let align = normalize_align(align.max(PAGE_SIZE))?;
        let length = align_up(length.max(1), align)?;
        let mut state = self.lock_state();

        let addr = align_up(state.mmap_bump, align)?;
        let end = addr.checked_add(length)?;
        if end > self.base + ARENA_SPAN {
            return None;
        }
        state.mmap_bump = end;
        Some(addr)
    }

    fn munmap(&self, _addr: u64, _length: u64) {
        // Best-effort no-op reclaim for RT2a (design doc §5/§7): the mmap
        // region has no free list yet, so there is nothing meaningful to
        // record here without risking a double-free-shaped bug for a
        // feature (`munmap` reuse) nothing depends on this task.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) `new` succeeds and `entry_ptr` bounds-checks against the real
    /// image length.
    #[test]
    fn new_succeeds_and_entry_ptr_bounds_checks() {
        let _lock = crate::dispatch::call_lock();
        let image = vec![0x90u8; 16];
        let arena = GuestArena::new(&image).expect("fixed-base reservation should succeed");

        let ptr0 = arena.entry_ptr(0).expect("offset 0 is within the image");
        assert_eq!(ptr0 as u64, GUEST_ARENA_BASE);

        assert!(
            arena.entry_ptr(image.len() as u64).is_err(),
            "offset == image_len must be rejected"
        );
    }

    /// (b) `alloc` returns a 16-aligned address inside the heap region;
    /// two allocations are distinct and don't overlap.
    #[test]
    fn alloc_returns_aligned_distinct_nonoverlapping_addresses_in_heap_region() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let a = arena.alloc(64, 16).expect("heap alloc should succeed");
        let b = arena.alloc(64, 16).expect("heap alloc should succeed");

        assert_eq!(a % 16, 0);
        assert_eq!(b % 16, 0);
        assert_ne!(a, b);

        let heap_start = GUEST_ARENA_BASE + HEAP_OFFSET;
        let heap_end = GUEST_ARENA_BASE + STACK_OFFSET;
        assert!(a >= heap_start && a + 64 <= heap_end);
        assert!(b >= heap_start && b + 64 <= heap_end);
        assert!(a + 64 <= b || b + 64 <= a, "allocations must not overlap");
    }

    /// (c) `free` then `alloc` of the same size reuses the freed block.
    #[test]
    fn free_then_alloc_same_size_reuses_freed_block() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let a = arena.alloc(128, 16).expect("heap alloc should succeed");
        arena.free(a);
        let b = arena.alloc(128, 16).expect("heap alloc should succeed");

        assert_eq!(
            a, b,
            "freeing then re-allocating the same size should reuse the freed block"
        );
    }

    /// (d) `mmap` returns a page-aligned address inside the mmap region.
    #[test]
    fn mmap_returns_page_aligned_address_in_mmap_region() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let addr = arena.mmap(0x2000, PAGE_SIZE).expect("mmap should succeed");
        assert_eq!(addr % PAGE_SIZE, 0);

        let mmap_start = GUEST_ARENA_BASE + MMAP_OFFSET;
        let arena_end = GUEST_ARENA_BASE + ARENA_SPAN;
        assert!(addr >= mmap_start && addr + 0x2000 <= arena_end);
    }

    /// (e) `GuestMemory` write-then-read roundtrip inside an alloc'd block
    /// returns the written bytes.
    #[test]
    fn guest_memory_write_then_read_roundtrips_inside_alloc_block() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let addr = arena.alloc(32, 16).expect("heap alloc should succeed");
        let pattern = [0xABu8; 32];
        assert!(arena.write(addr, &pattern));

        let mut out = [0u8; 32];
        assert!(arena.read(addr, &mut out));
        assert_eq!(out, pattern);
    }

    /// (f) `read`/`write` of wild addresses — outright out-of-arena, exactly
    /// one past the end, and an overflowing length — all return `false`
    /// without panicking.
    #[test]
    fn read_write_of_wild_addresses_return_false_without_panicking() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let buf = [0u8; 4];
        let mut mut_buf = [0u8; 4];

        assert!(!arena.read(0xDEAD_0000, &mut mut_buf));
        assert!(!arena.write(0xDEAD_0000, &buf));

        let one_past_end = GUEST_ARENA_BASE + ARENA_SPAN;
        assert!(!arena.read(one_past_end, &mut mut_buf));
        assert!(!arena.write(one_past_end, &buf));

        let overflowing_addr = u64::MAX - 2;
        assert!(!arena.read(overflowing_addr, &mut mut_buf));
        assert!(!arena.write(overflowing_addr, &buf));
    }

    /// (h) `stack_top` is 16-aligned and lands exactly at the end of the
    /// stack region (design doc §2/§7's alignment requirement for
    /// `call_on_guest_stack`).
    #[test]
    fn stack_top_is_aligned_and_bounds_the_stack_region() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let top = arena.stack_top();
        assert_eq!(top % 16, 0, "stack_top must be 16-byte aligned");
        assert_eq!(top, GUEST_ARENA_BASE + STACK_OFFSET + STACK_SIZE);
        assert!(top > GUEST_ARENA_BASE + STACK_OFFSET);
    }

    /// (g) construct, drop, construct again: the fixed base is reusable —
    /// no leftover reservation or leak.
    #[test]
    fn arena_is_reusable_across_construct_drop_cycles() {
        let _lock = crate::dispatch::call_lock();
        {
            let _arena = GuestArena::new(&[]).expect("first reservation should succeed");
        }
        let _arena2 =
            GuestArena::new(&[]).expect("second reservation, after Drop, should also succeed");
    }
}
