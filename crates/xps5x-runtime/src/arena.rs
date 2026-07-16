//! [`GuestArena`]: the fixed-base, identity-mapped guest address space
//! (design doc §2/§3/§5). Reserves one contiguous [`RESERVED_SPAN`] (2 TiB)
//! host range at [`crate::GUEST_ARENA_BASE`], commits its four fixed
//! sub-regions (image, heap, stack, mmap) in its first [`ARENA_SPAN`] (4 GiB),
//! leaves the rest as a sparse reservation tail committed on demand, copies the
//! module image in, and frees the whole range on `Drop`.
//!
//! Guest address `A` *is* host address `A` here (identity mapping): a guest
//! pointer returned by `alloc`/`mmap`, or baked into the image by the
//! linker's relocations, is directly dereferenceable by the native CPU — no
//! translation layer, unlike the now-retired `mem::MappedImage` this module
//! replaces.
//!
//! # The map is the authority
//!
//! Placement and bounds are [`crate::vmm::VmaMap`]'s, not this module's. The
//! arena owns *host* memory — what is reserved, what is committed — and the map
//! owns the *address space*: what lives at an address, where there is room, and
//! whether the guest may touch it. Every method here is that division of labour:
//! ask the map where to put something, ask Windows to back it, tell the map what
//! happened.
//!
//! It reads as a lot of ceremony for an allocator, and the measurements are why.
//! The three monotonic bumps this replaced could not free, so Until Dawn's
//! opening 512 GiB `sceKernelReserveVirtualRange` permanently spent the tail and
//! every later allocation failed down to 64 KiB; `munmap` was a no-op, so
//! nothing a title returned ever came back; and the parallel `Vec`s that tracked
//! reservations and committed ranges were scanned linearly on every single guest
//! read and write. One interval map answers all of it.
//!
//! # RT2 status
//!
//! Wired into [`crate::execute_linked`] since RT2 Task 3: every guest
//! execution builds a real `GuestArena` and passes it as both the
//! `GuestMemory` and `GuestAllocator` view.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS, PAGE_READWRITE,
    VirtualAlloc, VirtualFree,
};

use xps5x_hle::{GuestAllocator, GuestMemory};

use crate::vmm::{Vma, VmaMap, VmaType, prot};
use crate::{GUEST_ARENA_BASE, RuntimeError};

/// x86-64 Windows' page size (see `mem.rs`/`trampoline.rs`'s equivalent
/// constants); also the default alignment `mmap` rounds to.
const PAGE_SIZE: u64 = 4096;

/// Lowest address the [`VmaMap`] answers for. Page zero is deliberately left
/// outside it, so a null dereference stays an unanswerable fault rather than an
/// address the map has an opinion about.
const VMM_MIN: u64 = PAGE_SIZE;

/// One past the highest address the map answers for: Windows' user-mode ceiling
/// on x86-64 (`[0, 0x7FFF_FFFF_FFFF]`, 128 TiB).
///
/// The map spans the *whole* user address space rather than just the arena
/// because two things legitimately live outside it, and both must be
/// answerable. `reserve` lets the OS place huge reservations wherever it likes
/// (Until Dawn opens with 512 GiB), and `map_at` must honour an address the
/// guest picked for itself (ASTRO.BOT demands its libc mspace at
/// `0x3_0000_0000` — 12 GiB, far below the 16 TiB arena base — then writes to
/// that literal address). Everything we do not own is [`VmaType::Foreign`].
const VMM_MAX: u64 = 0x8000_0000_0000;

/// Diagnostic names for the ranges the arena hands out. Titles read the `name`
/// of a mapping back through the Named* map calls, and a name also keeps two
/// unrelated pools from coalescing into one another in the map.
const HEAP_NAME: &str = "heap";
const MMAP_NAME: &str = "mmap";
const RESERVE_NAME: &str = "reserve";

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

/// Sparse address-space tail reserved after the 4 GiB committed core. Games
/// commonly reserve several multi-gigabyte heaps up front and map only small
/// pieces later. The tail consumes host virtual addresses, not RAM.
///
/// Sized from a **measured** retail request, not a guess: Until Dawn opens with
/// `sceKernelReserveVirtualRange(len = 0x80_0000_0000)` — 512 GiB in a single
/// call — and treats failure as fatal (it goes straight to its crash reporter).
/// A 68 GiB tail could not answer that. 2 TiB leaves ~1.5 TiB of headroom after
/// such a reservation for [`GuestAllocator::alloc`]'s and `mmap`'s sparse
/// growth, which share `reserve_bump` with it.
///
/// This costs address space, not memory: `new` only `MEM_RESERVE`s the span
/// (pages are committed on demand), the arena base is 16 TiB, and Windows gives
/// a user process 128 TiB — verified by reserving 512 GiB/1 TiB/2 TiB at
/// `GUEST_ARENA_BASE` before choosing this value.
const RESERVED_SPAN: u64 = 0x200_0000_0000; // 2 TiB: 4 GiB core + sparse tail

/// Pages backed lazily by [`GuestArena::commit_on_demand`] since process start.
static DEMAND_COMMITS: AtomicU64 = AtomicU64::new(0);

/// How many reserved pages have been demand-committed. Monotonic; never reset.
/// Zero means no title has touched an unbacked reservation — the pre-existing
/// behaviour — so this doubles as the regression signal for that path.
// Diagnostic counter for the lazy-commit path; read by tooling, not the crate.
#[allow(dead_code)]
pub fn demand_commit_count() -> u64 {
    DEMAND_COMMITS.load(Ordering::Relaxed)
}

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
    if masked == 0 { 0x100 } else { masked }
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

/// The allocator's interior-mutable state, guarded by [`GuestArena`]'s `Mutex`.
///
/// # Why there is one map and not three bumps
///
/// This used to be `heap_bump`/`mmap_bump`/`reserve_bump` plus three parallel
/// `Vec`s (`heap_free`, `sparse_mappings`, `reservations`). The bumps had no
/// free path, so nothing ever came back: measured on retail titles, Until Dawn
/// opens with a single 512 GiB `sceKernelReserveVirtualRange`, a few of those
/// consumed the whole 2 TiB tail, and every later allocation then failed — down
/// to 64 KiB — after which the title died in its own crash reporter. Dragon Ball
/// fails identically.
///
/// [`VmaMap`] is that free path, and being one structure rather than four it is
/// also the single answer to "what is at this address": `range_is_committed`
/// went from a linear scan of an ever-growing `Vec` to one `BTreeMap` lookup.
struct AllocState {
    /// The guest address space. Every address belongs to exactly one
    /// [`Vma`], so this alone answers all three questions the bumps and their
    /// side tables used to answer separately — and sometimes inconsistently:
    /// what is here, where is there room, and may the guest touch this.
    vmm: VmaMap,
    /// Live heap allocations' sizes, keyed by address, so `free`/`realloc` can
    /// recover how much to release or copy without the caller repeating it.
    /// The map knows the *ranges*, but adjacent allocations coalesce within it,
    /// so it cannot recover where one `alloc`'s block ended — only this can.
    heap_sizes: HashMap<u64, u64>,
    /// Base addresses of whole blocks this arena took from the OS outside its
    /// own reservation (`GuestAllocator::reserve`, and `map_at` for a guest
    /// address beyond the arena). Each must be `MEM_RELEASE`d on `Drop`; only
    /// the block base is a legal argument, which is why the base is kept rather
    /// than the aligned address handed to the guest.
    os_reservations: Vec<u64>,
    /// Avoid repeating the heap-growth notice for every allocation after the
    /// fixed 1 GiB fast path fills.
    sparse_heap_announced: bool,
    /// Avoid repeating the demand-commit notice for every committed page.
    demand_commit_announced: bool,
}

impl AllocState {
    fn new(base: u64) -> Self {
        let mut vmm = VmaMap::new(VMM_MIN, VMM_MAX);
        let arena_end = base + RESERVED_SPAN;

        // Everything outside our own reservation is the host process's. Marking
        // it explicitly is what lets the map span the whole address space
        // without `search_free` ever offering the guest a host DLL.
        vmm.map_range(
            VMM_MIN,
            base - VMM_MIN,
            VmaType::Foreign,
            prot::NO_ACCESS,
            None,
            "host",
            false,
        );
        vmm.map_range(
            arena_end,
            VMM_MAX - arena_end,
            VmaType::Foreign,
            prot::NO_ACCESS,
            None,
            "host",
            false,
        );

        // The image and the stack are described as the furniture they are,
        // because HLE reads and writes both through `GuestMemory` and so both
        // must stay host-backed: the linker's relocated pointers and `.rodata`
        // live in the image, and `process::build_process_stack` writes
        // argc/argv/envp/auxv onto the stack. Neither is `is_guest_releasable`,
        // so no `munmap` can pull them out from under the running program.
        vmm.map_range(
            base + IMAGE_OFFSET,
            IMAGE_SIZE,
            VmaType::Code,
            prot::CPU_READ_WRITE | prot::CPU_EXEC,
            None,
            "image",
            true,
        );
        vmm.map_range(
            base + STACK_OFFSET,
            STACK_SIZE,
            VmaType::Stack,
            prot::CPU_READ_WRITE,
            None,
            "stack",
            true,
        );

        // The heap and mmap regions stay `Free`, and so does the sparse tail.
        // Free here means "unclaimed, and ours to place in" — the difference
        // between the two is only whether the host pages are committed yet
        // (`new` commits the core; the tail is committed as it is handed out),
        // which is `grow_into_tail`'s business rather than the map's.
        Self {
            vmm,
            heap_sizes: HashMap::new(),
            os_reservations: Vec::new(),
            sparse_heap_announced: false,
            demand_commit_announced: false,
        }
    }
}

/// Whether every VMA covering `[start, end)` satisfies `pred`, i.e. whether the
/// range is uniformly one thing. `false` if any part of it falls outside the
/// map, which is what makes a range straddling the map's edge a refusal rather
/// than a half-honoured request.
fn range_all(vmm: &VmaMap, start: u64, end: u64, pred: impl Fn(&Vma) -> bool) -> bool {
    if end <= start {
        return false;
    }
    let mut cursor = start;
    while cursor < end {
        let Some(vma) = vmm.find(cursor) else {
            return false;
        };
        if !pred(vma) {
            return false;
        }
        cursor = vma.end();
    }
    true
}

/// The fixed-base, identity-mapped guest address space (design doc §2/§5).
/// Owns `[GUEST_ARENA_BASE, GUEST_ARENA_BASE + RESERVED_SPAN)`; only the
/// leading [`ARENA_SPAN`] is committed host memory. Frees the whole range via
/// `VirtualFree` on `Drop`.
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
    /// Reserve the fixed-base 68 GiB range, commit its 4 GiB core with the
    /// four sub-region protections, and copy `image` into the image region.
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
                RESERVED_SPAN as usize,
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

    /// Whether an absolute guest address lies in the loaded executable image.
    pub(crate) fn is_executable_address(&self, address: u64) -> bool {
        address >= self.base && address < self.base.saturating_add(self.image_len)
    }

    /// Back `addr` with memory if it lies in a range the guest reserved but
    /// that was never committed. Returns whether the caller should retry the
    /// faulting access.
    ///
    /// **Why this exists (demand paging).** `sceKernelReserveVirtualRange` hands
    /// out address space, not memory, and real titles reserve far more than
    /// they touch — Until Dawn opens with a **512 GiB** reservation. Committing
    /// a reservation eagerly is therefore impossible (it would demand 512 GiB of
    /// RAM), yet the same titles then *use* their reservation directly: Until
    /// Dawn indexes a free-page bitmap 33.6 MB into a 64 MiB reservation
    /// (`mov rdi, [rdx + rcx*8]` at `eboot+0x84352`, rdx = the reservation base)
    /// having explicitly mapped only 16 KiB of it. Dragon Ball — also Unreal
    /// Engine 5 — faults on the identical address, so this is one engine
    /// pattern, not one title's quirk.
    ///
    /// So the reservation must be lazily backed: reserve the span, commit each
    /// page the first time it is touched. This is what a real VMM does, and it
    /// is the "lazy commit" RT2's design doc §7 deferred.
    ///
    /// Called only from the VEH, on the faulting thread, for an address that is
    /// otherwise about to be reported as a genuine fault — so a `false` return
    /// costs nothing and preserves the existing diagnostics exactly.
    pub fn commit_on_demand(&self, addr: u64) -> bool {
        // The map is the authority, and it is the only safe one: since `reserve`
        // takes its span from the OS, a reservation can live anywhere in the
        // address space, not just the arena's sparse tail. A range check against
        // the arena would reject exactly the reservations this exists to back.
        let mut state = self.lock_state();
        let page = addr & !(PAGE_SIZE - 1);

        match state.vmm.find(page).map(|vma| vma.kind) {
            // An untouched reservation page: back it, and the retry lands on
            // memory.
            Some(VmaType::Reserved) => {}
            // Already backed, which here can only mean another thread raced us
            // into this same page. Let its retry proceed rather than
            // double-commit.
            Some(VmaType::ReservedBacked) => return true,
            // Everything else is not a reservation, and declining leaves the
            // access to be reported as the fault it is: the committed core (a
            // fault there is real and unfixable — the pages already exist),
            // unclaimed free space, or the host process's own address space.
            _ => return false,
        }

        // SAFETY: `[page, page + PAGE_SIZE)` is page aligned and lies inside a
        // `Reserved` VMA. Only `reserve` creates one, and only ever over a range
        // it has just `MEM_RESERVE`d from the OS and recorded in
        // `os_reservations` — so the reservation is real and outlives this
        // arena's guest run. `MEM_COMMIT` over an already-committed page is a
        // documented no-op, so losing a race here is benign either way.
        let raw = unsafe {
            VirtualAlloc(
                page as *const c_void,
                PAGE_SIZE as usize,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if raw.is_null() || raw as u64 != page {
            return false;
        }

        // Only this page becomes backed. Its neighbours stay `Reserved`, which
        // is what keeps a 512 GiB reservation costing address space instead of
        // 512 GiB of RAM. Contiguous backed pages coalesce, so a title walking
        // its reservation linearly does not shred the map into slivers.
        state.vmm.map_range(
            page,
            PAGE_SIZE,
            VmaType::ReservedBacked,
            prot::CPU_READ_WRITE,
            None,
            RESERVE_NAME,
            false,
        );
        if !state.demand_commit_announced {
            tracing::info!(
                address = page,
                "guest touched a reserved-but-uncommitted range; backing it on demand"
            );
            state.demand_commit_announced = true;
        }
        DEMAND_COMMITS.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Whether every byte of `[start, end)` has host memory behind it.
    ///
    /// The map is authoritative, which is what lets `read`/`write` drop their
    /// old `< base` reject: an address below the arena is perfectly legal now
    /// that `reserve` and `map_at` honour OS-placed and guest-chosen addresses.
    /// What is *not* legal is anything the guest never obtained — free space,
    /// an untouched reservation, or the host's own memory — and those are
    /// exactly the kinds that are not host-backed, so a wild pointer is still
    /// refused.
    fn range_is_committed(&self, start: u64, end: u64) -> bool {
        let state = self.lock_state();
        if end == start {
            // A zero-length access touches nothing, but still has to name a
            // real address to be a legal one.
            return state
                .vmm
                .find(start)
                .is_some_and(|vma| vma.kind.is_host_backed());
        }
        range_all(&state.vmm, start, end, |vma| vma.kind.is_host_backed())
    }

    /// Commit `size` bytes in the arena's sparse reservation tail and claim them
    /// as `name`. Returns the address **and the length actually mapped**, which
    /// is `size` rounded up to a page: the caller must record that rather than
    /// its own request, or a later `free` would return less than was taken and
    /// strand the remainder for the arena's lifetime.
    ///
    /// The tail's address space is already `MEM_RESERVE`d by `new`, so this only
    /// has to commit — and only the pages actually handed out, which is what
    /// lets a 2 TiB tail cost no RAM until it is used. Both `alloc` and `mmap`
    /// grow here once their fixed regions fill; `mmap` lacking this fallback
    /// was an asymmetry rather than a deliberate ceiling, and ASTRO.BOT's
    /// opening 1.94 GiB `sceKernelAllocateDirectMemory` fell through the gap.
    fn grow_into_tail(
        &self,
        state: &mut AllocState,
        size: u64,
        align: u64,
        name: &str,
    ) -> Option<(u64, u64)> {
        let committed_len = align_up(size, PAGE_SIZE)?;
        let addr = state
            .vmm
            .search_free(self.base + ARENA_SPAN, committed_len, align)?;
        // `search_free` cannot stray past the tail — `Foreign` space is not free
        // and it will not place there — but the tail's own end is a real limit,
        // and a request that does not fit must fail rather than run past it.
        if addr.checked_add(committed_len)? > self.base + RESERVED_SPAN {
            return None;
        }

        // SAFETY: `[addr, addr + committed_len)` is page aligned and lies inside
        // this arena's own sparse `MEM_RESERVE` from `new` — `search_free`
        // returned it from the tail, and the map hands no range out twice, so it
        // cannot overlap a live mapping. The reservation outlives the mapping.
        // `MEM_COMMIT` over pages a previous `munmap` returned to the pool
        // without decommitting is a documented no-op.
        let raw = unsafe {
            VirtualAlloc(
                addr as *const c_void,
                committed_len as usize,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if raw.is_null() || raw as u64 != addr {
            return None;
        }
        state.vmm.map_range(
            addr,
            committed_len,
            VmaType::Flexible,
            prot::CPU_READ_WRITE,
            None,
            name,
            false,
        );
        Some((addr, committed_len))
    }

    /// Give a just-reserved OS block straight back, and fail the request that
    /// asked for it. Always returns `None`, so callers can `return
    /// self.release_unusable_block(block)` at the point they give up.
    fn release_unusable_block(&self, block: u64) -> Option<u64> {
        // SAFETY: `block` is the exact non-null base a `VirtualAlloc(MEM_RESERVE)`
        // has just returned, and every caller reaches here before recording it in
        // `os_reservations` — so `Drop` will not release it too, and this single
        // `MEM_RELEASE` with `dwSize = 0` cannot double-free. Releasing it here is
        // the only chance to: nothing else knows the block exists.
        unsafe {
            VirtualFree(block as *mut c_void, 0, MEM_RELEASE);
        }
        None
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

    /// Allocate an independent variant-II TLS block and TCB for a guest
    /// worker. The layout is identical to the main thread; only ownership and
    /// FS-base installation differ.
    pub(crate) fn setup_thread_tcb(
        &self,
        tls: Option<&xps5x_firmware::TlsTemplate>,
    ) -> Option<u64> {
        self.setup_main_tcb(tls)
    }
}

impl Drop for GuestArena {
    fn drop(&mut self) {
        // Blocks taken from the OS outside this arena's own reservation are not
        // covered by releasing `base`, so they must go back individually or a
        // title's reservations would outlive every run that made them.
        let blocks = std::mem::take(&mut self.lock_state().os_reservations);
        for block in blocks {
            // SAFETY: each `block` is the exact base a `VirtualAlloc(MEM_RESERVE)`
            // returned (the block base, never the aligned address derived from
            // it), released once with `dwSize = 0` as `MEM_RELEASE` requires.
            // `os_reservations` was drained above, so no later drop repeats it.
            unsafe {
                VirtualFree(block as *mut c_void, 0, MEM_RELEASE);
            }
        }
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
/// §2). `read`/`write` check `[guest_addr, guest_addr + len)` against the map
/// (via `checked_add`, so an overflowing request returns `false` rather than
/// wrapping) before ever touching host memory — an HLE function handed a wild
/// guest pointer gets `false`, never an OOB host access or a panic.
///
/// The bound is "every byte is host-backed", not "inside the arena": the arena
/// is no longer the only place the guest's memory lives, since `reserve` lets
/// the OS place a range anywhere and `map_at` honours an address the guest chose
/// for itself. `range_is_committed` is what makes that safe — it admits only the
/// committed core, ranges the guest actually mapped, and reservation pages
/// demand-commit has backed.
impl GuestMemory for GuestArena {
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
        // `out.len()` (a `usize`) never truncates when widened to `u64` on
        // this crate's target (Windows x86-64, `usize == u64`).
        let len = out.len() as u64;
        let Some(end) = guest_addr.checked_add(len) else {
            return false;
        };
        // No `< self.base` reject: a guest address below the arena is legal now
        // that `reserve`/`map_at` honor OS-placed and guest-chosen addresses.
        // `range_is_committed` is the real gate.
        if !self.range_is_committed(guest_addr, end) {
            return false;
        }
        // SAFETY: every byte of `[guest_addr, guest_addr + out.len())` is backed
        // by committed host memory — `range_is_committed` just walked the map
        // and found each VMA covering the range `is_host_backed`, which only the
        // core's own sub-regions and ranges this arena has itself committed ever
        // are. Guest address == host address here (identity mapping), so
        // `guest_addr as *const u8` is a valid, readable host pointer for
        // `out.len()` bytes. No other live reference writes these bytes through
        // Rust. Native guest threads can concurrently touch the same
        // VirtualAlloc pages, just as real guest CPUs can race; these host-side
        // copies are non-atomic and may tear. Guest synchronization is
        // responsible for conflicting accesses.
        unsafe {
            core::ptr::copy_nonoverlapping(guest_addr as *const u8, out.as_mut_ptr(), out.len());
        }
        true
    }

    fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
        let len = data.len() as u64;
        let Some(end) = guest_addr.checked_add(len) else {
            return false;
        };
        // Same bounds argument as `read`: `range_is_committed` is authoritative.
        if !self.range_is_committed(guest_addr, end) {
            return false;
        }
        // SAFETY: same bounds argument as `read` above. Every sub-region is
        // committed `PAGE_READWRITE` or `PAGE_EXECUTE_READWRITE` (see
        // `new`), so it is writable. As with `read`, native guest concurrency
        // is deliberately inherited: unsynchronized conflicting access may
        // tear and is the guest program's responsibility.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), guest_addr as *mut u8, data.len());
        }
        true
    }

    fn atomic_load_u32(&self, guest_addr: u64) -> Option<u32> {
        let end = guest_addr.checked_add(4)?;
        if !self.range_is_committed(guest_addr, end)
            || guest_addr % core::mem::align_of::<std::sync::atomic::AtomicU32>() as u64 != 0
        {
            return None;
        }
        // SAFETY: the checked address is 4-byte aligned and lies in committed
        // guest memory for the lifetime of this arena. Guest synchronization
        // words are accessed atomically through this API by HLE.
        Some(unsafe {
            (&*(guest_addr as *const std::sync::atomic::AtomicU32))
                .load(std::sync::atomic::Ordering::SeqCst)
        })
    }

    fn atomic_compare_exchange_u32(&self, guest_addr: u64, current: u32, new: u32) -> Option<u32> {
        let end = guest_addr.checked_add(4)?;
        if !self.range_is_committed(guest_addr, end)
            || guest_addr % core::mem::align_of::<std::sync::atomic::AtomicU32>() as u64 != 0
        {
            return None;
        }
        // SAFETY: same address/alignment/lifetime proof as atomic_load_u32.
        let atomic = unsafe { &*(guest_addr as *const std::sync::atomic::AtomicU32) };
        Some(
            atomic
                .compare_exchange(
                    current,
                    new,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .unwrap_or_else(|observed| observed),
        )
    }

    fn atomic_store_u32(&self, guest_addr: u64, value: u32) -> bool {
        let end = match guest_addr.checked_add(4) {
            Some(end) => end,
            None => return false,
        };
        if !self.range_is_committed(guest_addr, end)
            || guest_addr % core::mem::align_of::<std::sync::atomic::AtomicU32>() as u64 != 0
        {
            return false;
        }
        // SAFETY: same address/alignment/lifetime proof as atomic_load_u32.
        unsafe {
            (&*(guest_addr as *const std::sync::atomic::AtomicU32))
                .store(value, std::sync::atomic::Ordering::SeqCst);
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

        // The committed heap region. `search_free` walks free ranges in address
        // order, so a block a previous `free` returned is reused before the
        // region grows. That is the reuse the old first-fit `Vec` provided, but
        // without its linear scan, and without its "only if the freed block's
        // own address happens to satisfy this alignment" restriction, which
        // silently skipped blocks that were perfectly usable a few bytes in.
        let heap_end = self.base + STACK_OFFSET;
        if let Some(addr) = state.vmm.search_free(self.base + HEAP_OFFSET, size, align)
            && addr.checked_add(size).is_some_and(|end| end <= heap_end)
        {
            state.vmm.map_range(
                addr,
                size,
                VmaType::Flexible,
                prot::CPU_READ_WRITE,
                None,
                HEAP_NAME,
                false,
            );
            state.heap_sizes.insert(addr, size);
            return Some(addr);
        }

        // Large games legitimately consume more than the fixed 1 GiB heap.
        // Growing on demand beats imposing an emulator-specific memory ceiling.
        // Record the length that was mapped, not the length that was asked for:
        // `free` releases whatever `heap_sizes` says, and the two differ by the
        // page rounding.
        let (addr, mapped) =
            self.grow_into_tail(&mut state, size, align.max(PAGE_SIZE), HEAP_NAME)?;
        state.heap_sizes.insert(addr, mapped);
        if !state.sparse_heap_announced {
            tracing::info!(
                address = addr,
                "guest heap exceeded 1 GiB; growing on demand in sparse address space"
            );
            state.sparse_heap_announced = true;
        }
        Some(addr)
    }

    fn free(&self, addr: u64) {
        let mut state = self.lock_state();
        if let Some(size) = state.heap_sizes.remove(&addr) {
            // Back to the free pool, coalescing with any free neighbours. The
            // host pages stay committed — for a core block they were committed
            // at construction, and for a tail block `grow_into_tail`'s
            // `MEM_COMMIT` is a no-op if the range is handed out again. What
            // matters is that the *address space* returns, which under the bump
            // it never did.
            state.vmm.unmap_range(addr, size);
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

        // Fast path: the pre-committed 1.5 GiB mmap region. The bound check is
        // load-bearing, not defensive — the mmap region and the sparse tail are
        // contiguous free space, so `search_free` will happily answer with a
        // tail address once the region fills, and the tail's pages are reserved
        // rather than committed. Below that boundary the memory already exists;
        // above it, `grow_into_tail` has to commit it first.
        let arena_end = self.base + ARENA_SPAN;
        if let Some(addr) = state
            .vmm
            .search_free(self.base + MMAP_OFFSET, length, align)
            && addr.checked_add(length).is_some_and(|end| end <= arena_end)
        {
            state.vmm.map_range(
                addr,
                length,
                VmaType::Flexible,
                prot::CPU_READ_WRITE,
                None,
                MMAP_NAME,
                false,
            );
            return Some(addr);
        }

        // A retail title's first request can exceed the committed region
        // outright: ASTRO.BOT opens with `sceKernelAllocateDirectMemory(len =
        // 0x7980_0000)` — 1.94 GiB — which routes here. Capping at `ARENA_SPAN`
        // failed that call, and the title then built its heap over a garbage
        // base and died.
        self.grow_into_tail(&mut state, length, align.max(PAGE_SIZE), MMAP_NAME)
            .map(|(addr, _)| addr)
    }

    fn reserve(&self, length: u64, align: u64) -> Option<u64> {
        let align = normalize_align(align.max(PAGE_SIZE))?;
        let length = align_up(length.max(1), align)?;

        // Ask the OS for this span instead of carving it out of the arena's
        // sparse tail.
        //
        // Titles reserve enormously and never release — Until Dawn opens with a
        // single 512 GiB request — while `reserve`, `mmap`'s sparse growth and
        // `alloc`'s all drew from one monotonic `reserve_bump` that has no free
        // path. A few such reservations consumed the whole tail, after which
        // every later allocation failed all the way down to 64 KiB (measured on
        // Until Dawn and Dragon Ball: a cascade of `sceKernelAllocateDirectMemory:
        // arena mmap failed`, ending in the title's own crash reporter — the
        // long-blamed "write to 0x10" fault was that reporter, not the bug).
        //
        // Reservations are address space, so the OS is the right allocator for
        // them: it has the whole 128 TiB user range to place them in and can
        // reuse a released one, which a bump never could. The identity map is
        // untouched — guest VA is still host VA, wherever it lands, and the
        // guest reads the address back out of the call.
        //
        // Over-reserve by `align` and align within the block: `VirtualAlloc`'s
        // own granularity is 64 KiB, which cannot satisfy a larger request.
        let span = length.checked_add(align)?;
        // SAFETY: a null `lpAddress` asks the OS to choose any free range.
        // `MEM_RESERVE` with `PAGE_NOACCESS` takes address space without
        // committing memory, so a 512 GiB reservation costs no RAM. The block
        // is recorded in `os_reservations` and released in `Drop`.
        let raw =
            unsafe { VirtualAlloc(core::ptr::null(), span as usize, MEM_RESERVE, PAGE_NOACCESS) };
        if raw.is_null() {
            return None;
        }
        let block = raw as u64;

        let Some(addr) = align_up(block, align) else {
            return self.release_unusable_block(block);
        };
        let Some(end) = addr.checked_add(length) else {
            return self.release_unusable_block(block);
        };
        // A range the map cannot describe is a range we cannot serve: every
        // later read, write and demand-commit asks the map, so an unrecorded
        // reservation would hand the guest an address that then behaves as if it
        // did not exist. Windows keeps user allocations inside `VMM_MAX`, so
        // this is a guard against the impossible rather than an expected path —
        // but a silent unusable address is exactly the failure mode that cost
        // this project a week, so it fails loudly instead.
        if addr < VMM_MIN || end > VMM_MAX {
            tracing::error!(
                block,
                addr,
                length,
                "OS placed a reservation outside the mapped address space"
            );
            return self.release_unusable_block(block);
        }

        let mut state = self.lock_state();
        state.os_reservations.push(block);
        // Teach the map where the OS put it. Nothing else knows: the range lands
        // in what was `Foreign` space, and only this record makes a later touch
        // demand-committable (`commit_on_demand`) rather than a fault.
        state.vmm.map_range(
            addr,
            length,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            RESERVE_NAME,
            false,
        );
        Some(addr)
    }

    fn commit_on_demand(&self, addr: u64) -> bool {
        GuestArena::commit_on_demand(self, addr)
    }

    fn map_at(&self, addr: u64, length: u64, align: u64) -> Option<u64> {
        let align = normalize_align(align.max(PAGE_SIZE))?;
        if addr % align != 0 {
            return None;
        }
        let length = align_up(length.max(1), PAGE_SIZE)?;
        let end = addr.checked_add(length)?;
        if addr < VMM_MIN || end > VMM_MAX {
            return None;
        }

        let mut state = self.lock_state();

        // Already backed end to end: the memory exists, so hand the address
        // straight back.
        if range_all(&state.vmm, addr, end, |vma| vma.kind.is_host_backed()) {
            return Some(addr);
        }

        // Inside the committed core the host pages already exist; the range only
        // has to be claimed, so a later `search_free` cannot hand it out twice.
        if addr >= self.base && end <= self.base + ARENA_SPAN {
            if !range_all(&state.vmm, addr, end, |vma| {
                vma.kind.is_free() || vma.kind.is_host_backed()
            }) {
                return None;
            }
            state.vmm.map_range(
                addr,
                length,
                VmaType::Flexible,
                prot::CPU_READ_WRITE,
                None,
                MMAP_NAME,
                false,
            );
            return Some(addr);
        }

        // Serving an out-of-arena address at all is the point of this method: a
        // title picks its own VA and then uses that literal value — ASTRO.BOT
        // puts its libc mspace at 0x300000000 and faults on the first write when
        // the address it asked for does not exist. Guest pointers round-trip
        // through guest memory, so the guest must receive the address it asked
        // for; a fixed base cannot answer that.
        //
        // Where it lives decides how it can be backed. Inside the sparse tail,
        // or inside a range the guest already reserved, the address space is
        // ours and only needs committing — and `MEM_RESERVE` over an existing
        // reservation fails with ERROR_INVALID_ADDRESS, so it must *not* be
        // re-reserved. Outside everything, nothing owns it yet and it needs
        // both. A range that is not uniformly one of the two is refused rather
        // than half-honoured: the halves need different flags.
        let in_tail = addr >= self.base + ARENA_SPAN && end <= self.base + RESERVED_SPAN;
        let already_owned = (in_tail && range_all(&state.vmm, addr, end, |vma| vma.kind.is_free()))
            || range_all(&state.vmm, addr, end, |vma| {
                matches!(vma.kind, VmaType::Reserved | VmaType::ReservedBacked)
            });
        let unowned = range_all(&state.vmm, addr, end, |vma| vma.kind == VmaType::Foreign);
        if !already_owned && !unowned {
            return None;
        }
        let flags = if already_owned {
            MEM_COMMIT
        } else {
            MEM_RESERVE | MEM_COMMIT
        };

        // SAFETY: `[addr, end)` is page aligned and the map has just shown it to
        // be uniformly either memory this arena already reserved (its sparse
        // tail, or a prior `reserve` — committed in place) or wholly unowned
        // (reserved and committed together). `VirtualAlloc` at an explicit
        // address succeeds there or returns null; it never silently relocates,
        // and the `raw != addr` check below rejects a relocation regardless.
        // Only a newly reserved block is recorded for release, so `Drop` cannot
        // double-release a block `reserve` already owns.
        let raw = unsafe {
            VirtualAlloc(
                addr as *const c_void,
                length as usize,
                flags,
                PAGE_READWRITE,
            )
        };
        if raw.is_null() || raw as u64 != addr {
            return None;
        }
        if !already_owned {
            state.os_reservations.push(addr);
        }
        state.vmm.map_range(
            addr,
            length,
            VmaType::Flexible,
            prot::CPU_READ_WRITE,
            None,
            MMAP_NAME,
            false,
        );
        Some(addr)
    }

    fn munmap(&self, addr: u64, length: u64) {
        let Some(length) = align_up(length.max(1), PAGE_SIZE) else {
            return;
        };
        let Some(end) = addr.checked_add(length) else {
            return;
        };
        let mut state = self.lock_state();

        // Only ranges the guest itself obtained go back to the pool. This is the
        // guard that keeps `munmap` from declaring the guest's own image, its
        // stack, or the host process's address space free for the next
        // `search_free` to hand out — an unrecognized range is ignored, per the
        // trait contract, rather than acted on.
        if !range_all(&state.vmm, addr, end, |vma| vma.kind.is_guest_releasable()) {
            return;
        }
        state.heap_sizes.remove(&addr);
        // The host pages are left committed: for a core range they were
        // committed at construction, and for a tail range the `MEM_COMMIT` that
        // hands it out again is a documented no-op. Returning the address space
        // is the part that was missing — under the bump, `munmap` was a no-op
        // and nothing ever came back.
        state.vmm.unmap_range(addr, length);
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

    #[test]
    fn heap_grows_into_sparse_tail_after_committed_fast_path_fills() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let core = arena
            .alloc(HEAP_SIZE, 16)
            .expect("the full committed heap fast path");
        assert_eq!(core, GUEST_ARENA_BASE + HEAP_OFFSET);

        let grown = arena
            .alloc(0x2000, 16)
            .expect("heap must grow into sparse address space");
        assert!(grown >= GUEST_ARENA_BASE + ARENA_SPAN);
        assert!(arena.write(grown + 0x1fff, &[0xAB]));
        let mut byte = [0u8; 1];
        assert!(arena.read(grown + 0x1fff, &mut byte));
        assert_eq!(byte, [0xAB]);

        let reservation = arena
            .reserve(0x4000, PAGE_SIZE)
            .expect("reservation after sparse heap growth");
        // Non-overlap is the property that matters; the address ordering that
        // used to imply it no longer holds, because a reservation now comes from
        // the OS and may land anywhere.
        assert!(
            reservation + 0x4000 <= grown || reservation >= grown + 0x2000,
            "reservation {reservation:#x} overlaps the grown heap block {grown:#x}"
        );
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

    /// ASTRO.BOT's opening `sceKernelAllocateDirectMemory(0x7980_0000)` — 1.94
    /// GiB — routes to `mmap`, whose committed region is only 1.5 GiB. Before
    /// `mmap` grew a sparse-tail fallback this failed outright on the title's
    /// FIRST allocation; the game then built its heap over a garbage base.
    /// The returned memory must be genuinely usable, not merely a live address.
    #[test]
    fn mmap_request_larger_than_the_committed_region_grows_into_the_sparse_tail() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let astro_bot_request = 0x7980_0000u64; // measured from the retail title

        assert!(
            astro_bot_request > MMAP_SIZE,
            "this test is only meaningful while the request exceeds the committed \
             mmap region ({MMAP_SIZE:#x}); it is the whole point of the fallback"
        );

        let addr = arena
            .mmap(astro_bot_request, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("a 1.94 GiB mmap must succeed via the sparse tail");
        assert!(
            addr >= GUEST_ARENA_BASE + ARENA_SPAN,
            "oversized mmap must land in the sparse tail, got {addr:#x}"
        );

        // Committed, not just reserved: read/write both ends of the mapping.
        let mut byte = [0u8; 1];
        assert!(arena.write(addr, &[0xAB]));
        assert!(arena.read(addr, &mut byte));
        assert_eq!(byte, [0xAB]);
        let last = addr + astro_bot_request - 1;
        assert!(
            arena.write(last, &[0xCD]),
            "tail page must be committed too"
        );
        assert!(arena.read(last, &mut byte));
        assert_eq!(byte, [0xCD]);
    }

    /// Until Dawn opens with `sceKernelReserveVirtualRange(0x80_0000_0000)` —
    /// 512 GiB in one call — and treats failure as fatal.
    ///
    /// The address is deliberately NOT asserted: reservations come from the OS
    /// now, so the guest is told where its range landed and no fixed location is
    /// promised. What must hold is that the request succeeds, costs address
    /// space rather than RAM, and leaves the allocator able to keep working.
    #[test]
    fn a_single_five_hundred_twelve_gib_reservation_succeeds_and_stays_unbacked() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let until_dawn_request = 0x80_0000_0000u64; // measured from the retail title

        let addr = arena
            .reserve(until_dawn_request, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("a 512 GiB reservation must succeed");
        assert_ne!(addr, 0);

        // Reserved only — untouched pages must still be unreadable, or this
        // would be a 512 GiB commit rather than a 512 GiB reservation.
        let mut byte = [0u8; 1];
        assert!(!arena.read(addr, &mut byte));

        let after = arena
            .alloc(0x1000, 16)
            .expect("heap must still grow after a 512 GiB reservation");
        assert!(after != 0);
    }

    /// The measured Until Dawn / Dragon Ball failure, reduced to its mechanism.
    ///
    /// `reserve`, `mmap` and `alloc` all drew from one monotonic `reserve_bump`
    /// with no free path, so a few 512 GiB reservations consumed the whole
    /// 2 TiB tail and every later allocation failed — measured down to 64 KiB
    /// (`sceKernelAllocateDirectMemory: arena mmap failed`), after which the
    /// title died in its own crash reporter. Four such reservations is exactly
    /// what the old code could not survive.
    #[test]
    fn repeated_huge_reservations_do_not_starve_later_allocations() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let half_tib = 0x80_0000_0000u64;

        let mut reserved = Vec::new();
        for i in 0..4 {
            let addr = arena
                .reserve(half_tib, xps5x_core::PS5_PAGE_SIZE as u64)
                .unwrap_or_else(|| panic!("reservation {i} of 512 GiB must succeed"));
            reserved.push((addr, addr + half_tib));
        }

        // Reservations must not overlap each other. The OS guarantees this by
        // never handing out a range it already gave us; assert it rather than
        // trust it, since a silent overlap would corrupt one of the two.
        for (i, &(a_start, a_end)) in reserved.iter().enumerate() {
            for &(b_start, b_end) in &reserved[i + 1..] {
                assert!(
                    a_end <= b_start || b_end <= a_start,
                    "reservations {a_start:#x}..{a_end:#x} and {b_start:#x}..{b_end:#x} overlap"
                );
            }
        }

        // The sizes the real titles requested after their reservations, ending
        // with the 64 KiB that used to fail.
        for len in [0x4000_0000u64, 0x2000_0000, 0x20c000, 0x10000] {
            let addr = arena
                .mmap(len, xps5x_core::PS5_PAGE_SIZE as u64)
                .unwrap_or_else(|| panic!("mmap of {len:#x} must survive huge reservations"));
            assert!(arena.write(addr, &[0xAB]), "mapped memory must be usable");
        }
    }

    /// Demand paging: a reserved page carries no memory until touched, and
    /// `commit_on_demand` backs exactly the touched page. This is what lets a
    /// title reserve 512 GiB and then use a bitmap 33.6 MB into it (Until Dawn,
    /// Dragon Ball) without the emulator committing 512 GiB of RAM.
    #[test]
    fn touching_a_reserved_page_commits_it_on_demand() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let len = 0x400_0000u64; // 64 MiB, Until Dawn's second reservation
        let base = arena
            .reserve(len, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("reservation");

        // Reserved != usable: nothing is backed yet.
        let deep = base + 0x200_1860; // ~33.6 MB in, Until Dawn's bitmap offset
        let mut byte = [0u8; 1];
        assert!(
            !arena.read(deep, &mut byte),
            "reservation must start unbacked"
        );

        // The fault handler's question. It must say yes, and only once per page.
        assert!(arena.commit_on_demand(deep), "a touch inside a reservation");
        assert!(
            arena.write(deep, &[0xAB]),
            "page must be usable after commit"
        );
        assert!(arena.read(deep, &mut byte));
        assert_eq!(byte, [0xAB]);

        // Idempotent: a second touch of a now-backed page still says "retry".
        assert!(arena.commit_on_demand(deep));

        // Committing one page must not silently back its neighbours — that
        // would turn a 512 GiB reservation into a 512 GiB commit.
        assert!(!arena.read(deep + PAGE_SIZE, &mut byte));
    }

    /// The safety boundary: demand commit answers ONLY for reserved ranges. A
    /// wild pointer must stay a fault, or every guest bug becomes silent memory.
    #[test]
    fn commit_on_demand_declines_addresses_outside_any_reservation() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        // In the sparse tail but never reserved by anyone.
        assert!(!arena.commit_on_demand(GUEST_ARENA_BASE + ARENA_SPAN + 0x5000_0000));
        // Inside the committed core: a fault there is real (already mapped).
        assert!(!arena.commit_on_demand(GUEST_ARENA_BASE + 0x1000));
        // Past the reservation entirely, and a wild low address.
        assert!(!arena.commit_on_demand(GUEST_ARENA_BASE + RESERVED_SPAN + 0x1000));
        assert!(!arena.commit_on_demand(0x1000));

        // A reservation must not bless addresses past its own end.
        let base = arena
            .reserve(0x1_0000, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("reservation");
        assert!(arena.commit_on_demand(base));
        assert!(!arena.commit_on_demand(base + 0x1_0000));
    }

    #[test]
    fn two_eight_gib_reservations_coexist_without_mapping_them() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let eight_gib = 0x2_0000_0000;
        let first = arena
            .reserve(eight_gib, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("first sparse reservation");
        let second = arena
            .reserve(eight_gib, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("second sparse reservation");

        // Addresses come from the OS now, so only non-overlap is promised —
        // not adjacency, and not a location inside the arena.
        assert!(
            first + eight_gib <= second || second + eight_gib <= first,
            "reservations {first:#x} and {second:#x} overlap"
        );
        let mut byte = [0u8; 1];
        assert!(!arena.read(first, &mut byte));
        assert!(!arena.write(first, &[1]));

        let mapped = arena
            .map_at(first, 0x2_0000, xps5x_core::PS5_PAGE_SIZE as u64)
            .expect("commit inside prior sparse reservation");
        assert_eq!(mapped, first);
        assert!(arena.write(first + 0x10, &[0xAB]));
        assert!(arena.read(first + 0x10, &mut byte));
        assert_eq!(byte, [0xAB]);
        assert!(!arena.write(first + 0x2_0000, &[1]));
    }

    /// `munmap` must return the range to the free pool. Until the VMA map
    /// arrived there was no free path at all — `munmap` was a documented no-op
    /// and `mmap` only ever bumped forward, so a title that mapped and unmapped
    /// in a loop (every streaming asset system does) walked the address space
    /// until it ran out, with every byte it had returned still spent.
    #[test]
    fn munmap_returns_the_range_so_a_later_mmap_can_reuse_it() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let len = 0x10_0000u64;

        let first = arena.mmap(len, PAGE_SIZE).expect("first mmap");
        arena.munmap(first, len);
        let second = arena.mmap(len, PAGE_SIZE).expect("second mmap");

        assert_eq!(
            first, second,
            "an unmapped range must be handed out again, not leaked"
        );
    }

    /// Reuse must be stable, not just possible once: every cycle has to land
    /// back on the same address. A bump drifts forward by `len` each time, so
    /// this pins that the range genuinely returns to the free pool rather than
    /// the allocator merely having somewhere else to go.
    #[test]
    fn repeated_map_unmap_cycles_reuse_the_same_range() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let len = MMAP_SIZE / 2;

        let first = arena.mmap(len, PAGE_SIZE).expect("first mmap");
        assert!(arena.write(first, &[0xAB]), "mapped memory must be usable");
        arena.munmap(first, len);

        for i in 1..8 {
            let addr = arena
                .mmap(len, PAGE_SIZE)
                .unwrap_or_else(|| panic!("mmap cycle {i} must succeed"));
            assert_eq!(
                addr, first,
                "cycle {i} drifted instead of reusing the range"
            );
            assert!(arena.write(addr, &[0xAB]), "mapped memory must be usable");
            arena.munmap(addr, len);
        }
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
