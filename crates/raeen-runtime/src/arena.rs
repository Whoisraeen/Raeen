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
    MEM_COMMIT, MEM_DECOMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEM_RELEASE,
    MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
    VirtualQuery,
};

use raeen_hle::{GuestAccess, GuestAddress, GuestAllocator, GuestMemory, GuestRange};

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

/// PS5 kernel-selected mappings first use the system-managed virtual-address
/// window. Native title libc validates mspace storage against these bounds, so
/// returning memory from the emulator's 16 TiB image arena is not ABI-valid.
const SYSTEM_MANAGED_MIN: u64 = 0x0040_0000; // 4 MiB
const SYSTEM_MANAGED_LIMIT: u64 = 0x08_0000_0000; // one past 32 GiB

/// If the system-managed window cannot satisfy a mapping, use the PS5 user
/// window. Keep the upper limit below native libc's low-user-range guard.
const USER_MAPPING_MIN: u64 = 0x10_0000_0000; // 64 GiB
const USER_MAPPING_LIMIT: u64 = 0x00FB_FFC0_0000;

/// Title reservations (`sceKernelReserveVirtualRange`) are placed ABOVE the two
/// kernel-mapping windows, not inside them.
///
/// Measured on Until Dawn / Dragon Ball: both open with a 512 GiB reservation.
/// Placed by a low-first scan it landed inside `USER_MAPPING` and shredded it —
/// the window still had hundreds of GiB free, but its largest *contiguous* block
/// was 510 MiB, so every later 1 GiB `sceKernelAllocateDirectMemory` had nowhere
/// to go and the title cascaded down to 128 KiB requests and gave up.
///
/// A reservation is address space the title picks up and uses directly; unlike
/// libc's mspace storage it is not validated against the mapping windows, so it
/// does not need to live in them. Keeping the two apart means one title's huge
/// reservation can no longer starve the kernel of placeable mapping space.
/// `[RESERVE_MIN, GUEST_ARENA_BASE)` is ~15 TiB — dozens of such reservations.
const RESERVE_MIN: u64 = 0x0100_0000_0000; // 1 TiB — above USER_MAPPING_LIMIT

/// Explicit Windows reservations must start on a 64 KiB boundary.
const WINDOWS_ALLOCATION_GRANULARITY: u64 = 0x1_0000;

/// The guest fixed-VA window retail titles map direct memory into by literal
/// address, defended for the guest by a process-startup reservation.
///
/// A guest-requested `sceKernelMapNamedDirectMemory` address is not advisory:
/// ASTRO.BOT's libc lays its mspace at `0x3_0000_0000` and writes to that
/// literal address regardless of the call's result. Measured 2026-07-21
/// (logs/raeen.txt): launched from the SHELL the map failed —
/// `cannot map len=0x79800000 at requested=0x300000000` — and the title
/// faulted at libc.prx+0x103c6 writing `0x300000020`, while the SAME build
/// booted the SAME title from the CLI (`--run-eboot`) repeatedly. The
/// difference is who else lives in the process: the GUI had been running
/// eframe/egui/wgpu/Vulkan for seconds before launch and host allocations had
/// landed inside the window, so the fixed-address `VirtualAlloc` collided;
/// the CLI process was clean there by luck, not by design.
///
/// [`reserve_title_va_window`] closes that race: called as the first statement
/// of `main`, it `MEM_RESERVE`s every free hole in the window so no later host
/// allocation can squat there, and [`GuestAllocator::map_at`] serves guest
/// fixed maps out of those process-lifetime reservations with `MEM_COMMIT`
/// alone. The window spans the full direct-memory size a title can map
/// (`sceKernelGetDirectMemorySize` = `PS5_DIRECT_MEMORY_SIZE` = 0x3_5800_0000
/// in `raeen-hle`) from the measured mspace base, so any fixed map of any
/// direct-memory slice based at `0x3_0000_0000` fits.
const TITLE_VA_WINDOW_MIN: u64 = 0x3_0000_0000; // 12 GiB — ASTRO.BOT's mspace base
const TITLE_VA_WINDOW_LIMIT: u64 = TITLE_VA_WINDOW_MIN + 0x3_5800_0000;

/// The process-lifetime claim on [`TITLE_VA_WINDOW_MIN`]`..`[`TITLE_VA_WINDOW_LIMIT`]:
/// each entry is one whole `MEM_RESERVE` block covering one free hole
/// (disjoint, address-ordered). Never released — the window must stay claimed
/// for as long as the process can launch a title. Committed pages inside the
/// blocks are per-arena and are decommitted by `GuestArena::drop`.
struct TitleVaWindow {
    blocks: Vec<(u64, u64)>,
    report: crate::TitleVaWindowReport,
}

static TITLE_VA_WINDOW: std::sync::OnceLock<TitleVaWindow> = std::sync::OnceLock::new();

/// Claim every free hole of the title fixed-VA window for the guest. Idempotent
/// and cheap after the first call. Returns what was claimed and what was
/// already lost so the caller can log it once logging exists.
pub fn reserve_title_va_window() -> &'static crate::TitleVaWindowReport {
    &TITLE_VA_WINDOW.get_or_init(claim_title_va_window).report
}

/// The blocks claimed at startup, or empty if [`reserve_title_va_window`] has
/// not run (CLI paths that never claimed the window keep their pre-existing
/// direct `MEM_RESERVE` behaviour).
fn title_va_blocks() -> &'static [(u64, u64)] {
    TITLE_VA_WINDOW
        .get()
        .map(|w| w.blocks.as_slice())
        .unwrap_or(&[])
}

/// The startup block containing `addr`, if any.
fn title_va_block_containing(addr: u64) -> Option<(u64, u64)> {
    title_va_blocks()
        .iter()
        .copied()
        .find(|&(base, len)| addr >= base && addr < base + len)
}

/// The nearest startup-block edge (a block's start or end) strictly above
/// `addr`, or `u64::MAX` when none. `map_at`'s segment walk clamps to this so
/// no single `VirtualAlloc` action ever straddles the boundary between address
/// space we already reserved (commit-only) and address space we did not.
fn next_title_va_edge(addr: u64) -> u64 {
    let mut edge = u64::MAX;
    for &(base, len) in title_va_blocks() {
        for e in [base, base + len] {
            if e > addr && e < edge {
                edge = e;
            }
        }
    }
    edge
}

fn claim_title_va_window() -> TitleVaWindow {
    let mut blocks = Vec::new();
    let mut squatters = Vec::new();
    let mut reserved_bytes = 0u64;
    let mut cursor = TITLE_VA_WINDOW_MIN;
    while cursor < TITLE_VA_WINDOW_LIMIT {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `VirtualQuery` only inspects the region containing `cursor`;
        // `info` is a valid, correctly sized out-buffer.
        let queried = unsafe {
            VirtualQuery(
                cursor as *const c_void,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if queried == 0 {
            break;
        }
        let region_base = info.BaseAddress as u64;
        let Some(region_end) = region_base.checked_add(info.RegionSize as u64) else {
            break;
        };
        let usable_end = region_end.min(TITLE_VA_WINDOW_LIMIT);
        if info.State == MEM_FREE {
            if let Some(candidate) =
                align_up(cursor.max(region_base), WINDOWS_ALLOCATION_GRANULARITY)
                && candidate < usable_end
            {
                let len = usable_end - candidate;
                // SAFETY: Windows just reported `[candidate, usable_end)` free
                // and `candidate` has allocation-granularity alignment; an
                // explicit `MEM_RESERVE` either lands there or returns null.
                // The block is process-lifetime by design (never released), so
                // no ownership record beyond `TITLE_VA_WINDOW` is needed.
                let raw = unsafe {
                    VirtualAlloc(
                        candidate as *const c_void,
                        len as usize,
                        MEM_RESERVE,
                        PAGE_NOACCESS,
                    )
                };
                if !raw.is_null() && raw as u64 == candidate {
                    blocks.push((candidate, len));
                    reserved_bytes += len;
                } else if !raw.is_null() {
                    // Defensive only: explicit VirtualAlloc does not relocate.
                    // SAFETY: `raw` is a fresh, unpublished reservation.
                    unsafe {
                        VirtualFree(raw, 0, MEM_RELEASE);
                    }
                }
            }
        } else {
            squatters.push(describe_host_region(
                cursor.max(region_base),
                usable_end,
                &info,
            ));
        }
        let next = region_end.max(cursor.saturating_add(1));
        if next <= cursor {
            break;
        }
        cursor = next;
    }
    TitleVaWindow {
        report: crate::TitleVaWindowReport {
            window_start: TITLE_VA_WINDOW_MIN,
            window_end: TITLE_VA_WINDOW_LIMIT,
            reserved_blocks: blocks.len(),
            reserved_bytes,
            squatters,
        },
        blocks,
    }
}

/// One host region as a human-readable line: range, state, backing type, and
/// the owning module when the loader knows it (DLL/EXE images). Private
/// allocations cannot be attributed to their allocator, but state+type alone
/// already separates "a DLL landed here" from "the heap/driver grabbed it".
fn describe_host_region(start: u64, end: u64, info: &MEMORY_BASIC_INFORMATION) -> String {
    let state = match info.State {
        MEM_FREE => "free",
        MEM_COMMIT => "committed",
        MEM_RESERVE => "reserved",
        _ => "unknown-state",
    };
    let kind = if info.State == MEM_FREE {
        ""
    } else {
        match info.Type {
            MEM_IMAGE => " image",
            MEM_MAPPED => " mapped",
            MEM_PRIVATE => " private",
            _ => "",
        }
    };
    let owner = if info.State == MEM_FREE {
        String::new()
    } else {
        crate::thread::host_module_for_addr(info.AllocationBase as u64)
            .map(|module| format!(" owner={module}"))
            .unwrap_or_default()
    };
    format!("{start:#x}..{end:#x} {state}{kind}{owner}")
}

/// One-shot diagnostic for a failed guest fixed-address map: survey the
/// requested range with `VirtualQuery` and log every region with its state,
/// backing type, and owner where diagnosable. This is the log line that makes
/// the Shell-vs-CLI collision class self-explaining — a bare "cannot map"
/// cannot distinguish a host DLL squatting the address from commit exhaustion
/// or a guest double-map.
fn diagnose_fixed_map_failure(addr: u64, end: u64, failed_at: u64) {
    tracing::warn!(
        requested = format_args!("{addr:#x}..{end:#x}"),
        failed_at = format_args!("{failed_at:#x}"),
        title_window_reserved = !title_va_blocks().is_empty(),
        "guest fixed-address map failed; host regions in the requested range:"
    );
    let mut cursor = addr;
    while cursor < end {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `VirtualQuery` only inspects the region containing `cursor`;
        // `info` is a valid, correctly sized out-buffer.
        let queried = unsafe {
            VirtualQuery(
                cursor as *const c_void,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if queried == 0 {
            break;
        }
        let region_base = info.BaseAddress as u64;
        let Some(region_end) = region_base.checked_add(info.RegionSize as u64) else {
            break;
        };
        tracing::warn!(
            "  {}",
            describe_host_region(cursor.max(region_base), region_end.min(end), &info)
        );
        let next = region_end.max(cursor.saturating_add(1));
        if next <= cursor {
            break;
        }
        cursor = next;
    }
}

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

/// Test-only fault injection for [`GuestArena::map_at`]'s commit path. There is
/// no way to exhaust the real Windows commit limit in a unit test, so a test
/// arms this counter to make the next N up-front commits behave as refused,
/// exercising the demand-commit fallback that keeps a fixed-address
/// direct-memory map alive (the ASTRO.BOT crash in logs/raeen.txt).
#[cfg(test)]
static FORCE_MAP_AT_COMMIT_REFUSALS: AtomicU64 = AtomicU64::new(0);

/// Consume one armed refusal, if any. Compiles to a constant `false` outside
/// tests, so the production commit path is untouched and branch-free.
#[cfg(test)]
fn take_forced_commit_refusal() -> bool {
    FORCE_MAP_AT_COMMIT_REFUSALS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
        .is_ok()
}

#[cfg(not(test))]
#[inline(always)]
fn take_forced_commit_refusal() -> bool {
    false
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

/// Reserve an uncommitted host range at the first available address in the
/// guest's deterministic low-VA mapping window.
fn reserve_guest_address_space(start: u64, limit: u64, length: u64, align: u64) -> Option<u64> {
    let placement_align = align.max(WINDOWS_ALLOCATION_GRANULARITY);
    let mut cursor = align_up(start, placement_align)?;

    while cursor < limit {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `VirtualQuery` only inspects the region containing `cursor`;
        // `info` is a valid, correctly sized out-buffer.
        let queried = unsafe {
            VirtualQuery(
                cursor as *const c_void,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if queried == 0 {
            return None;
        }

        let region_base = info.BaseAddress as u64;
        let region_end = region_base.checked_add(info.RegionSize as u64)?;
        if info.State == MEM_FREE {
            let candidate = align_up(cursor.max(region_base), placement_align)?;
            let end = candidate.checked_add(length)?;
            if end <= region_end && end <= limit {
                // SAFETY: Windows just reported `[candidate, end)` free;
                // `candidate` has allocation-granularity alignment. An
                // explicit reservation either succeeds there or returns null.
                let raw = unsafe {
                    VirtualAlloc(
                        candidate as *const c_void,
                        length as usize,
                        MEM_RESERVE,
                        PAGE_NOACCESS,
                    )
                };
                if !raw.is_null() {
                    if raw as u64 == candidate {
                        return Some(candidate);
                    }
                    // Defensive only: explicit VirtualAlloc does not relocate.
                    unsafe {
                        VirtualFree(raw, 0, MEM_RELEASE);
                    }
                }
            }
        }

        let next = align_up(region_end.max(cursor.saturating_add(1)), placement_align)?;
        if next <= cursor {
            return None;
        }
        cursor = next;
    }
    None
}

/// Diagnostic: walk `[start, limit)` and report `(largest_free_block, total_free)`.
///
/// Read-only (`VirtualQuery` only). Explains a console-VA placement failure —
/// exhaustion (little total free) reads very differently from fragmentation
/// (lots free, no single block big enough) or a placement bug (plenty of both).
fn survey_free_space(start: u64, limit: u64) -> (u64, u64) {
    let mut cursor = start;
    let mut largest = 0u64;
    let mut total = 0u64;
    while cursor < limit {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `VirtualQuery` only inspects the region containing `cursor`;
        // `info` is a valid, correctly sized out-buffer.
        let queried = unsafe {
            VirtualQuery(
                cursor as *const c_void,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if queried == 0 {
            break;
        }
        let region_base = info.BaseAddress as u64;
        let Some(region_end) = region_base.checked_add(info.RegionSize as u64) else {
            break;
        };
        if info.State == MEM_FREE {
            let usable_start = cursor.max(region_base);
            let usable_end = region_end.min(limit);
            if usable_end > usable_start {
                let size = usable_end - usable_start;
                total = total.saturating_add(size);
                largest = largest.max(size);
            }
        }
        let next = region_end.max(cursor.saturating_add(1));
        if next <= cursor {
            break;
        }
        cursor = next;
    }
    (largest, total)
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
    /// Mappings created as independent Windows reservations outside the fixed
    /// arena, keyed by their exact `MEM_RELEASE` base. Usually fully committed;
    /// `map_at`'s demand-commit fallback may enter a reserved-but-lazily-backed
    /// one, which `munmap`/`Drop` release identically (`MEM_RELEASE` frees a
    /// reservation whatever its commit state).
    external_mappings: HashMap<u64, u64>,
    /// Ranges this arena committed (or served demand-committed) inside the
    /// process-lifetime title-VA window blocks ([`TITLE_VA_WINDOW`]). The
    /// blocks themselves are never released, so `Drop` must `MEM_DECOMMIT`
    /// these explicitly — otherwise a relaunch in the same Shell process would
    /// read the previous title's bytes where fresh zero pages belong, and the
    /// RAM would stay charged for the life of the process.
    window_commits: Vec<(u64, u64)>,
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
            external_mappings: HashMap::new(),
            window_commits: Vec::new(),
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
    /// When set, the code image is `PAGE_EXECUTE_READ` (W^X): a stray data
    /// write into code faults at the store instead of silently corrupting an
    /// instruction. Instrumentation code writes go through [`patch_code`], which
    /// transiently lifts the bar. `false` keeps the permissive RWX default.
    wx_image: std::sync::atomic::AtomicBool,
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

        // Opt-in bring-up aid: make one image page RX so the *first writer*
        // to code faults at its real call site instead of silently corrupting
        // instructions and failing much later. The normal mapping remains the
        // current RWX compatibility shortcut until LinkedModule carries every
        // PT_LOAD permission into this layer.
        if let Ok(value) = std::env::var("RAEEN_DIAGNOSTIC_RX_PAGE")
            && let Ok(address) = u64::from_str_radix(value.trim_start_matches("0x"), 16)
        {
            let page = address & !(PAGE_SIZE - 1);
            let image_end = base + image_len;
            if page >= base && page < image_end {
                let mut old_protect = 0u32;
                // SAFETY: `page` is page-aligned and lies in the committed
                // image range owned by this arena. The protection is restored
                // implicitly when the whole reservation is released on Drop.
                let ok = unsafe {
                    VirtualProtect(
                        page as *const c_void,
                        PAGE_SIZE as usize,
                        PAGE_EXECUTE_READ,
                        &mut old_protect,
                    )
                } != 0;
                if ok {
                    tracing::info!(
                        "diagnostic RX page armed at {page:#x} (old protection {old_protect:#x})"
                    );
                } else {
                    tracing::warn!("failed to arm diagnostic RX page at {page:#x}");
                }
            }
        }

        // Inter-region guard pages. The four sub-regions tile with no gaps, so
        // an overflow off one runs silently into the next — a heap block into
        // the stack, an image overrun into the heap — and surfaces as an
        // anonymous fault millions of ops later. A `PAGE_NOACCESS` page at a
        // boundary turns that into a trap AT THE STORE. Two boundaries are
        // guarded, both safe by construction:
        //   * IMAGE top `[IMAGE_END - PAGE, IMAGE_END)` — only when the loaded
        //     image ends at least a page below it (checked), so no real code or
        //     data lives there.
        //   * HEAP top `[STACK_OFFSET - PAGE, STACK_OFFSET)` — `alloc` caps at
        //     `heap_end = STACK_OFFSET - PAGE_SIZE`, so the allocator never
        //     hands this page out; a heap overrun upward or a stack underrun
        //     downward both land on it.
        // The STACK|MMAP boundary is deliberately NOT guarded: the initial RSP
        // sits at the top of the stack region, so a guard there would fault
        // live data. Drop releases the whole reservation, restoring protection.
        let install_guard = |page: u64, what: &str| {
            let mut old = 0u32;
            // SAFETY: `page` is page-aligned and within the committed 4 GiB core
            // this arena just mapped; NOACCESS on one owned page is well-formed.
            let ok = unsafe {
                VirtualProtect(
                    page as *const c_void,
                    PAGE_SIZE as usize,
                    PAGE_NOACCESS,
                    &mut old,
                )
            } != 0;
            if ok {
                tracing::debug!("guard page armed at {page:#x} ({what})");
            } else {
                tracing::warn!("failed to arm {what} guard page at {page:#x}");
            }
        };
        if image_len <= IMAGE_SIZE - PAGE_SIZE {
            install_guard(base + IMAGE_SIZE - PAGE_SIZE, "image|heap");
        }
        install_guard(base + STACK_OFFSET - PAGE_SIZE, "heap|stack");

        Ok(Self {
            base,
            image_len,
            state: Mutex::new(AllocState::new(base)),
            wx_image: std::sync::atomic::AtomicBool::new(false),
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
        // `Reserved` VMA. Every producer of one places it over a range that is
        // really `MEM_RESERVE`d and outlives this arena's guest run: `reserve`
        // and `map_console_va` over an OS reservation recorded in
        // `os_reservations`; `map_at`'s demand-commit fallback over either a
        // newly-reserved Foreign range (also pushed to `os_reservations`) or the
        // arena master reservation (the free sparse tail, released in `Drop`).
        // So the reservation is real either way. `MEM_COMMIT` over an
        // already-committed page is a documented no-op, so losing a race here is
        // benign too.
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

    /// Back every `Reserved` (demand-commit) page in `[addr, addr + len)` with
    /// host memory, exactly as [`GuestArena::commit_on_demand`] does for one VEH
    /// fault — but in a single pass under one lock, for the GPU read path.
    ///
    /// The GPU reads guest memory host-side (`read_gpu`), so it never traps the
    /// guest VEH that lazily backs a reservation on CPU access. On real hardware
    /// mapped direct memory is always GPU-readable, so our lazy commit must be
    /// transparent to the GPU rather than a refusal: a title that maps a texture
    /// pool and has the GPU read it before any CPU write (streamed/DMA-filled
    /// assets, or a fresh target) would otherwise see every such draw skipped
    /// (measured on ASTRO.BOT: 47 "texture guest range … readable prefix 0x0"
    /// draw skips per frame on the Shell path). A page that is neither `Reserved`
    /// nor already host-backed (free space, foreign, a wild pointer) leaves the
    /// range not fully backed, so the caller still refuses that read. Under host
    /// memory pressure a `MEM_COMMIT` here can fail — the range is then reported
    /// not-backed and the draw degrades (skipped), never a fake.
    fn commit_range_on_demand(&self, addr: u64, len: u64) -> bool {
        let Some(end) = addr.checked_add(len) else {
            return false;
        };
        if end == addr {
            return self.range_is_committed(addr, end);
        }
        // Cap the range this backs so a mis-decoded GPU read of a wild/huge
        // range cannot spin this loop over billions of pages. A real title
        // resource fits well under this; anything larger is refused (read
        // skipped). 256 MiB matches the resource read cap in
        // `raeen_gpu::guest_mem`.
        const MAX_COMMIT_RANGE: u64 = 0x1000_0000;
        if len > MAX_COMMIT_RANGE {
            return false;
        }
        // Back each `Reserved` page via the per-page `commit_on_demand`, which
        // acquires and RELEASES the state lock per page so a guest thread
        // faulting into its own demand-commit can interleave rather than block on
        // one long lock hold. `commit_on_demand` is a no-op for already-backed and
        // non-reservation pages, so the final `range_is_committed` decides
        // visibility (free/foreign pages leave the range not fully backed and the
        // read is still refused).
        let mut page = addr & !(PAGE_SIZE - 1);
        while page < end {
            self.commit_on_demand(page);
            page = match page.checked_add(PAGE_SIZE) {
                Some(next) => next,
                None => break,
            };
        }
        self.range_is_committed(addr, end)
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
        Self::range_is_committed_locked(&state, start, end)
    }

    /// Locked form used by host copies and atomics. Keeping `state` borrowed
    /// across the actual pointer access prevents a concurrent `munmap` from
    /// releasing an independent Windows reservation after validation but
    /// before the copy/load/store.
    fn range_is_committed_locked(state: &AllocState, start: u64, end: u64) -> bool {
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
    /// The host-backed sub-ranges of `[addr, end)` that lie OUTSIDE the
    /// committed 4 GiB core — the pages a fixed-address [`GuestAllocator::map_at`]
    /// may hand back holding a previous mapping's bytes.
    ///
    /// Orbis `sceKernelMapDirectMemory`/`sceKernelMapFlexibleMemory` return
    /// ZEROED pages. Windows only zeroes a page on its FIRST commit, so when a
    /// title reuses one of its own fixed VAs (ASTRO.BOT maps successive levels'
    /// asset pools into the same `0x3_0000_0000` direct-memory window) the pages
    /// are already committed and Windows leaves the old level's bytes in place. A
    /// freshly-constructed object over those bytes then reads an unset pointer
    /// field as a stale non-null value, survives its null check, and faults —
    /// the measured level-transition worker faults. Only pages that were
    /// host-backed BEFORE the call are stale; fresh `MEM_RESERVE|MEM_COMMIT` and
    /// first-time `MEM_COMMIT` pages are already zeroed by Windows and are left
    /// out, so a large first-time fixed map pays no redundant memset.
    ///
    /// The committed core is excluded: it backs `alloc` (malloc, which the guest
    /// zeroes itself) and carries the inter-region guard pages, which must never
    /// be written.
    fn backed_ranges_outside_core(
        state: &AllocState,
        addr: u64,
        end: u64,
        core_start: u64,
        core_end: u64,
    ) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        let mut cursor = addr;
        while cursor < end {
            let Some(vma) = state.vmm.find(cursor) else {
                break;
            };
            let seg_end = vma.end().min(end);
            if seg_end <= cursor {
                break;
            }
            if vma.kind.is_host_backed() {
                // Clip the core out of `[cursor, seg_end)`: the part below the
                // core and the part above it are both eligible.
                let below_end = seg_end.min(core_start);
                if cursor < below_end {
                    ranges.push((cursor, below_end - cursor));
                }
                let above_start = cursor.max(core_end);
                if above_start < seg_end {
                    ranges.push((above_start, seg_end - above_start));
                }
            }
            cursor = seg_end;
        }
        ranges
    }

    /// Zero the reuse ranges captured by [`backed_ranges_outside_core`], in
    /// place. The caller must still hold the state lock — that serialises this
    /// against `munmap`'s `MEM_RELEASE`, so a range host-backed at capture is
    /// still backed here.
    fn zero_reused_ranges(&self, ranges: &[(u64, u64)]) {
        for &(start, len) in ranges {
            if len == 0 {
                continue;
            }
            // SAFETY: `[start, start + len)` was host-backed (committed
            // read/write) in the VMA map at capture and outside the core (so no
            // guard page), and the state lock is held across the whole `map_at`,
            // so no concurrent `munmap` can have released it. The arena is
            // identity-mapped, so the guest address is the host address.
            unsafe {
                std::ptr::write_bytes(start as *mut u8, 0, len as usize);
            }
        }
    }

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

    /// Reserve and commit one kernel-selected mapping in the PS5 virtual
    /// address windows. The fixed image arena intentionally lives far above
    /// console VA space; exposing that address to native title libc breaks its
    /// allocator range checks.
    fn map_console_va(
        &self,
        state: &mut AllocState,
        length: u64,
        align: u64,
        name: &str,
    ) -> Option<u64> {
        let addr =
            reserve_guest_address_space(SYSTEM_MANAGED_MIN, SYSTEM_MANAGED_LIMIT, length, align)
                .or_else(|| {
                    reserve_guest_address_space(USER_MAPPING_MIN, USER_MAPPING_LIMIT, length, align)
                });
        let Some(addr) = addr else {
            // Both console-VA windows refused. Measured on Until Dawn / Dragon
            // Ball: a cascade of failures down to 128 KiB. Say WHY — how much
            // room each window actually has — instead of a bare "mmap failed",
            // which cannot distinguish exhaustion from a placement bug.
            let (sm_largest, sm_free) = survey_free_space(SYSTEM_MANAGED_MIN, SYSTEM_MANAGED_LIMIT);
            let (um_largest, um_free) = survey_free_space(USER_MAPPING_MIN, USER_MAPPING_LIMIT);
            tracing::warn!(
                want = format_args!("{length:#x}"),
                align = format_args!("{align:#x}"),
                system_managed_largest_free = format_args!("{sm_largest:#x}"),
                system_managed_total_free = format_args!("{sm_free:#x}"),
                user_mapping_largest_free = format_args!("{um_largest:#x}"),
                user_mapping_total_free = format_args!("{um_free:#x}"),
                "console-VA placement failed — no free block in either window"
            );
            return None;
        };

        let end = addr.checked_add(length)?;
        if !range_all(&state.vmm, addr, end, |vma| vma.kind == VmaType::Foreign) {
            tracing::warn!(
                addr = format_args!("{addr:#x}"),
                len = format_args!("{length:#x}"),
                "console-VA rejected: range is not Foreign in the VMM map"
            );
            // SAFETY: `addr` is the exact reservation base returned above and
            // has not been published or recorded yet.
            unsafe {
                VirtualFree(addr as *mut c_void, 0, MEM_RELEASE);
            }
            return None;
        }

        // SAFETY: `[addr, end)` is the page-aligned reservation just obtained
        // from Windows. Committing it in place cannot relocate it.
        let raw = unsafe {
            VirtualAlloc(
                addr as *const c_void,
                length as usize,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if raw.is_null() {
            // Committing the WHOLE length up front charges it against the
            // process commit limit (RAM + pagefile). A title asking for GiB of
            // direct memory exhausts that (measured: Until Dawn / Dragon Ball
            // open with a 1 GiB `sceKernelAllocateDirectMemory`, Windows answers
            // ERROR_COMMITMENT_LIMIT "the paging file is too small"), and every
            // later request then failed too — a cascade all the way down to
            // 128 KiB, because each one tried to commit up front as well.
            //
            // Failing is the wrong answer: the address space IS reserved (the
            // `reserve_guest_address_space` above took it with MEM_RESERVE), and
            // titles touch far less than they allocate. So degrade to the same
            // demand-commit the arena already runs for `sceKernelReserveVirtualRange`
            // — record the span as `Reserved` and let `commit_on_demand` back
            // each page from the fault handler as the guest actually reaches it.
            // The guest cannot tell the difference; only untouched pages stay
            // free, which is the whole point.
            tracing::warn!(
                addr = format_args!("{addr:#x}"),
                len = format_args!("{length:#x}"),
                last_os_error = %std::io::Error::last_os_error(),
                "console-VA: MEM_COMMIT refused; serving the range demand-committed"
            );
            state.os_reservations.push(addr);
            state.vmm.map_range(
                addr,
                length,
                VmaType::Reserved,
                prot::NO_ACCESS,
                None,
                name,
                false,
            );
            return Some(addr);
        }
        if raw as u64 != addr {
            // Defensive only: explicit VirtualAlloc does not relocate.
            // SAFETY: same exact, unpublished reservation-base proof as above.
            unsafe {
                VirtualFree(addr as *mut c_void, 0, MEM_RELEASE);
            }
            return None;
        }

        state.os_reservations.push(addr);
        state.external_mappings.insert(addr, length);
        state.vmm.map_range(
            addr,
            length,
            VmaType::Flexible,
            prot::CPU_READ_WRITE,
            None,
            name,
            false,
        );
        Some(addr)
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

    /// Set up the main-thread TCB and, if any module in the process has a
    /// `PT_TLS` template, the **combined** static TLS area (design doc §3,
    /// RT2c-b/M1-B):
    ///
    /// - Carves `static_tls_total(layout) + TCB_SIZE` bytes from the heap
    ///   allocator (variant-II x86-64 layout: every module's TLS block sits
    ///   *below* the TCB, at the `tp_offset` the linker computed its
    ///   `TPOFF64` offsets against — the two must agree exactly).
    /// - Copies each module's `.tdata` init image into its block; `.tbss`
    ///   and padding stay zero. Skipping a module's image is not a lesser
    ///   mode: libc.prx keeps errno/locale/strtok state in 0x188 bytes of
    ///   initialized TLS, and a zeroed copy is undefined behavior the guest
    ///   has no way to see coming.
    /// - Writes the TCB self-pointer at `fs:[0]` (the FreeBSD/Orbis
    ///   "variant II" convention) and a real, nonzero `__stack_chk_guard`
    ///   canary at the ABI-mandated `fs:0x28` ([`CANARY_TCB_OFFSET`]).
    ///
    /// Returns the TCB's guest address (the FS base to install), or `None`
    /// if the heap allocation or the write-back fails.
    pub(crate) fn setup_main_tcb(
        &self,
        tls_layout: &[raeen_firmware::StaticTlsModule],
    ) -> Option<u64> {
        let area = raeen_firmware::static_tls_total(tls_layout);
        let align = tls_layout
            .iter()
            .map(|m| m.template.align.max(16))
            .max()
            .unwrap_or(16);
        let total = area.checked_add(TCB_SIZE)?;
        let base = self.alloc(total, align)?;
        let tcb = base.checked_add(area)?;

        let mut block = vec![0u8; total as usize];
        for module in tls_layout {
            // The module's block starts `tp_offset` below the TCB. Its
            // `.tdata` goes at the block's start; a template whose `data`
            // somehow exceeds its slot (malformed: filesz > memsz, or an
            // offset outside the area) is truncated rather than trusted.
            let Some(start) = (area as usize).checked_sub(module.tp_offset as usize) else {
                continue;
            };
            let n = module
                .template
                .data
                .len()
                .min((area as usize).saturating_sub(start));
            block[start..start + n].copy_from_slice(&module.template.data[..n]);
        }
        let tcb_off = area as usize;
        block[tcb_off..tcb_off + 8].copy_from_slice(&tcb.to_le_bytes());
        block[tcb_off + CANARY_TCB_OFFSET..tcb_off + CANARY_TCB_OFFSET + 8]
            .copy_from_slice(&stack_canary().to_le_bytes());

        if !self.write(base, &block) {
            return None;
        }
        Some(tcb)
    }

    /// Allocate an independent variant-II TLS area and TCB for a guest
    /// worker. The layout is identical to the main thread; only ownership and
    /// FS-base installation differ.
    pub(crate) fn setup_thread_tcb(
        &self,
        tls_layout: &[raeen_firmware::StaticTlsModule],
    ) -> Option<u64> {
        self.setup_main_tcb(tls_layout)
    }
}

impl Drop for GuestArena {
    fn drop(&mut self) {
        // Pages this arena committed inside the process-lifetime title-VA
        // window blocks: the blocks are never released, so this memory is not
        // covered by any `MEM_RELEASE` below. Decommit it so a relaunch in the
        // same Shell process starts from fresh zero pages (not the previous
        // title's bytes) and the RAM goes back to the OS.
        let window_commits = std::mem::take(&mut self.lock_state().window_commits);
        for (addr, len) in window_commits {
            // SAFETY: each range was recorded by `map_at` inside a
            // `TITLE_VA_WINDOW` block, which lives for the whole process, so
            // the reservation is still valid; `MEM_DECOMMIT` of a partially
            // committed range inside one reservation is well-formed and leaves
            // the reservation itself intact.
            unsafe {
                VirtualFree(addr as *mut c_void, len as usize, MEM_DECOMMIT);
            }
        }
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
/// Enable W^X on a freshly-built arena's image when `RAEEN_WX_IMAGE` is set.
/// Call right after `GuestArena::new`, once the loader has composed the image
/// (export-trap `int3`, `native_trap` prologues, and relocations are all baked
/// into the buffer `new` copied). `exec_ranges` are the image-relative
/// `(offset, len)` spans of the main module's PF_X (executable) segments — only
/// those are made read-only; `.data`/`.bss` and dependency segments stay RWX so
/// ordinary global writes never fault. Off by default — the image stays RWX.
pub(crate) fn maybe_enable_wx_image(arena: &GuestArena, exec_ranges: &[(u64, u64)]) {
    if std::env::var_os("RAEEN_WX_IMAGE").is_some() {
        arena.enable_wx_image(exec_ranges);
    }
}

impl GuestArena {
    /// Enable W^X on the code image: flip `[base, image guard)` from RWX to
    /// `PAGE_EXECUTE_READ`. Call after the loader has finished writing and
    /// relocating the image. A stray guest DATA store into code then faults at
    /// the store (caught by the VEH) instead of silently corrupting an
    /// instruction; legitimate instrumentation patches keep working through
    /// [`GuestMemory::patch_code`], which lifts the bar transiently. Idempotent.
    /// The image|heap guard page stays `PAGE_NOACCESS` — protection is applied
    /// only below it.
    pub(crate) fn enable_wx_image(&self, exec_ranges: &[(u64, u64)]) -> bool {
        use std::sync::atomic::Ordering;
        // Only PF_X (executable) segments become read-only. The image region
        // holds the whole module — .text AND .data/.bss — so making it ALL
        // read-only would fault the first global-variable write (measured:
        // Minecraft stores to a .data global immediately). RX-ing just the code
        // spans keeps data writable while still trapping a stray store into code.
        let mut any = false;
        for &(offset, len) in exec_ranges {
            // Clamp to the committed image region, never touching the
            // image|heap guard page at the very top.
            let start = self.base + offset;
            let cap = self.base + IMAGE_SIZE - PAGE_SIZE;
            let end = start.saturating_add(len).min(cap);
            if end <= start {
                continue;
            }
            let page_start = start & !(PAGE_SIZE - 1);
            let span = (end - page_start) as usize;
            let mut old = 0u32;
            // SAFETY: `[page_start, end)` lies in the committed image region
            // this arena owns; Drop releases the reservation, restoring it.
            let ok = unsafe {
                VirtualProtect(
                    page_start as *const c_void,
                    span,
                    PAGE_EXECUTE_READ,
                    &mut old,
                )
            } != 0;
            if ok {
                any = true;
                tracing::debug!(
                    "W^X: {page_start:#x}..{:#x} -> execute+read",
                    page_start + span as u64
                );
            } else {
                tracing::warn!("W^X: failed to protect code range {page_start:#x} (+{span:#x})");
            }
        }
        if any {
            self.wx_image.store(true, Ordering::SeqCst);
            tracing::info!(
                "W^X image enabled: {} executable range(s) are now execute+read",
                exec_ranges.len()
            );
        }
        any
    }

    /// Store `data` into the code image, honouring W^X. When it is off the
    /// image is RWX and this is a plain [`GuestMemory::write`]. When it is on,
    /// the covered pages are toggled `PAGE_EXECUTE_READWRITE` for the store and
    /// restored to `PAGE_EXECUTE_READ` — so an export-trap `int3`, a
    /// `native_trap` prologue, or a one-shot restore still lands while a guest
    /// self-modifying store keeps faulting.
    fn patch_code_wx(&self, guest_addr: u64, data: &[u8]) -> bool {
        use std::sync::atomic::Ordering;
        if !self.wx_image.load(Ordering::SeqCst) {
            return self.write(guest_addr, data);
        }
        if data.is_empty() {
            return true;
        }
        let first_page = guest_addr & !(PAGE_SIZE - 1);
        let Some(last_byte) = guest_addr.checked_add(data.len() as u64 - 1) else {
            return false;
        };
        let last_page = last_byte & !(PAGE_SIZE - 1);
        let span = (last_page - first_page + PAGE_SIZE) as usize;
        let mut old = 0u32;
        // SAFETY: the patched pages lie in the committed image region.
        let unlocked = unsafe {
            VirtualProtect(
                first_page as *const c_void,
                span,
                PAGE_EXECUTE_READWRITE,
                &mut old,
            )
        } != 0;
        if !unlocked {
            return false;
        }
        let wrote = self.write(guest_addr, data);
        let mut discard = 0u32;
        // SAFETY: same pages, restoring the W^X protection.
        unsafe {
            VirtualProtect(
                first_page as *const c_void,
                span,
                PAGE_EXECUTE_READ,
                &mut discard,
            );
        }
        wrote
    }

    /// `range_is_committed`, with the same lazy-commit semantics a native
    /// guest STORE gets: a range inside a demand-commit reservation that no
    /// instruction has touched yet is NOT a wild pointer — a guest store
    /// there would fault into `commit_on_demand` and succeed. An HLE call
    /// writing there must behave identically (measured: Minecraft hands
    /// `getdents` a dirent buffer carved from a big lazy reservation; the
    /// old strict check turned that into EFAULT, which the title escalated
    /// to a fatal `std::out_of_range`). Commits page-by-page and re-checks;
    /// pages outside any reservation still fail, so wild pointers stay
    /// rejected. Bounded: refuses ranges over 16 MiB so a wild length
    /// cannot turn into a commit storm.
    ///
    /// Write-side only, deliberately: HLE READS of untouched reservations
    /// stay strict, keeping "a 512 GiB reservation is not a 512 GiB commit"
    /// observable (see the reservation tests) — an HLE read has no data to
    /// deliver from a page nothing ever wrote.
    fn ensure_range_backed_for_write(&self, start: u64, end: u64) -> bool {
        if self.range_is_committed(start, end) {
            return true;
        }
        const MAX_DEMAND_COMMIT_SPAN: u64 = 16 << 20;
        if end.saturating_sub(start) > MAX_DEMAND_COMMIT_SPAN {
            return false;
        }
        let mut page = start & !0xFFF;
        while page < end {
            // Failures are fine (pages outside any reservation); the
            // re-check below is authoritative.
            self.commit_on_demand(page);
            page = page.saturating_add(0x1000);
        }
        self.range_is_committed(start, end)
    }
}

/// Whether guest `mprotect` is actually applied to host pages. Opt-in
/// (`RAEEN_ENFORCE_MPROTECT`) for the same reason W^X is: a title that marks a
/// page read-only which our HLE later writes would then fault. Read once.
fn enforce_mprotect() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RAEEN_ENFORCE_MPROTECT").is_some())
}

/// Orbis CPU-protection bitset -> Windows `PAGE_*`. GPU bits are ignored for the
/// CPU mapping; `CPU_WRITE` implies read on Windows (no write-only page).
fn orbis_prot_to_win(prot: u32) -> u32 {
    let read = prot & prot::CPU_READ != 0;
    let write = prot & prot::CPU_WRITE != 0;
    let exec = prot & prot::CPU_EXEC != 0;
    match (exec, write, read) {
        (false, false, false) => PAGE_NOACCESS,
        (false, false, true) => PAGE_READONLY,
        (false, true, _) => PAGE_READWRITE,
        (true, false, false) => PAGE_EXECUTE,
        (true, false, true) => PAGE_EXECUTE_READ,
        (true, true, _) => PAGE_EXECUTE_READWRITE,
    }
}

impl GuestMemory for GuestArena {
    fn patch_code(&self, guest_addr: u64, data: &[u8]) -> bool {
        self.patch_code_wx(guest_addr, data)
    }

    fn protect(&self, addr: u64, len: u64, prot: u32) -> bool {
        if !enforce_mprotect() || len == 0 {
            // Historical no-op: record nothing, change nothing, succeed.
            return true;
        }
        let win = orbis_prot_to_win(prot);
        let start = addr & !(PAGE_SIZE - 1);
        let Some(end) = addr
            .checked_add(len)
            .map(|e| (e + PAGE_SIZE - 1) & !(PAGE_SIZE - 1))
        else {
            return false;
        };
        // Re-protect page by page so a partially-committed range (some pages
        // still reserved) protects what it can without failing the whole call,
        // and so a guard page in the range is left NOACCESS rather than reopened.
        let mut page = start;
        let mut any = false;
        while page < end {
            let committed = {
                let state = self.lock_state();
                Self::range_is_committed_locked(&state, page, page + PAGE_SIZE)
            };
            let is_guard = page == self.base + IMAGE_SIZE - PAGE_SIZE
                || page == self.base + STACK_OFFSET - PAGE_SIZE;
            if committed && !is_guard {
                let mut old = 0u32;
                // SAFETY: `page` is one committed page this arena owns.
                let ok = unsafe {
                    VirtualProtect(page as *const c_void, PAGE_SIZE as usize, win, &mut old)
                } != 0;
                any |= ok;
            }
            page += PAGE_SIZE;
        }
        // Even a fully-uncommitted range is a legal mprotect of a reservation;
        // report success so the guest is not told its own call failed.
        let _ = any;
        true
    }

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
        let state = self.lock_state();
        if !Self::range_is_committed_locked(&state, guest_addr, end) {
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
        // responsible for conflicting accesses. The address-space lock stays
        // held through the copy, so `munmap` cannot release an external
        // reservation underneath this host pointer.
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
        // Same bounds argument as `read`, except a write demand-commits
        // reservation pages exactly as a native guest store would.
        if !self.ensure_range_backed_for_write(guest_addr, end) {
            return false;
        }
        // Demand-commit above may take the state lock repeatedly. Reacquire it
        // once and revalidate before dereferencing; a mapping may have been
        // removed between the final commit/check and this point.
        let state = self.lock_state();
        if !Self::range_is_committed_locked(&state, guest_addr, end) {
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

    fn validate_range(&self, range: GuestRange, _access: GuestAccess) -> bool {
        let Some(end) = range.end() else {
            return false;
        };
        self.range_is_committed(range.start().raw(), end)
    }

    fn is_executable_range(&self, range: GuestRange) -> bool {
        let Some(end) = range.end() else {
            return false;
        };
        !range.is_empty()
            && range.start().raw() >= self.base
            && end <= self.base.saturating_add(self.image_len)
    }

    fn is_gpu_visible_range(&self, range: GuestRange) -> bool {
        let Some(end) = range.end() else {
            return false;
        };
        let start = range.start().raw();
        // Fast path: already fully host-backed (the common case once a texture
        // has been committed, and for the committed core).
        if self.range_is_committed(start, end) {
            return true;
        }
        // A demand-commit (`Reserved`) range the GPU is about to read: back it
        // now, exactly as the guest VEH would on a CPU access. On real hardware
        // mapped direct memory is always GPU-readable — our lazy commit must be
        // transparent to the GPU, not a refusal.
        self.commit_range_on_demand(start, end.saturating_sub(start))
    }

    fn atomic_load_u32(&self, guest_addr: u64) -> Option<u32> {
        let end = guest_addr.checked_add(4)?;
        let state = self.lock_state();
        if !Self::range_is_committed_locked(&state, guest_addr, end)
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
        let state = self.lock_state();
        if !Self::range_is_committed_locked(&state, guest_addr, end)
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
        let state = self.lock_state();
        if !Self::range_is_committed_locked(&state, guest_addr, end)
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

impl raeen_gpu::GpuGuestMemory for GuestArena {
    fn validate_gpu_range(&self, addr: u64, len: u64, _write: bool) -> bool {
        GuestRange::new(GuestAddress::new(addr), len)
            .and_then(|range| raeen_hle::GpuVisibleGuestRange::validate(self, range))
            .is_some()
    }

    fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
        GuestMemory::read(self, addr, out)
    }

    fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
        GuestMemory::write(self, addr, data)
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
        // One page below STACK_OFFSET is the heap|stack guard (see `new`), so an
        // in-region allocation must end at or before it — never on it.
        let heap_end = self.base + STACK_OFFSET - PAGE_SIZE;
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

        // Kernel-chosen mappings are guest ABI, not an internal allocator
        // detail. The image/heap arena is at 16 TiB so it cannot collide with
        // title-selected VAs, but native PS5 libc rejects an mspace whose base
        // comes from that range.
        self.map_console_va(&mut state, length, align, MMAP_NAME)
    }

    fn reserve(&self, length: u64, align: u64) -> Option<u64> {
        self.reserve_with_hint(0, length, align, false)
    }

    fn reserve_with_hint(&self, hint: u64, length: u64, align: u64, fixed: bool) -> Option<u64> {
        let align = normalize_align(align.max(PAGE_SIZE))?;
        let length = align_up(length.max(1), align)?;
        if fixed && (hint == 0 || hint % align != 0) {
            return None;
        }

        // Ask the OS for this span instead of carving it out of the arena's
        // sparse tail. A non-zero address is a real placement hint in the
        // Orbis ABI; MAP_FIXED strengthens it to an exact address. V8 uses the
        // hint to acquire its 4 GiB-aligned pointer-compression cage. Ignoring
        // it made the guest allocate and reject multiple 4/8 GiB reservations
        // above 1 TiB before it happened to straddle a suitable boundary.
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
        // Scan on the requested alignment before reserving. Windows requires a
        // 64 KiB allocation-granularity base; `reserve_guest_address_space`
        // raises smaller alignments to that granularity.
        let start = if hint == 0 {
            RESERVE_MIN
        } else {
            align_up(hint, align)?
        };
        let limit = if fixed {
            start.checked_add(length)?
        } else {
            self.base
        };
        // `MEM_RESERVE` with `PAGE_NOACCESS` takes address space without
        // committing memory, so a 512 GiB reservation costs no RAM. The block
        // is recorded in `os_reservations` and released on exact unmap or Drop.
        let block = reserve_guest_address_space(start, limit, length, align).or_else(|| {
            (!fixed && hint != 0)
                .then(|| reserve_guest_address_space(RESERVE_MIN, self.base, length, align))
                .flatten()
        })?;

        let addr = block;
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
        // An exact whole-range `munmap` can now release this OS reservation
        // immediately instead of merely relabeling the VMA while Windows keeps
        // the address space occupied until process teardown.
        state.external_mappings.insert(block, length);
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

        // Already backed end to end: a title re-mapping a fixed VA it already
        // holds fully backed — ASTRO.BOT re-maps successive levels' asset pools
        // into the same 0x3_0000_0000 direct-memory window without unmapping
        // between them. The memory exists, so hand the address straight back —
        // but first clear the previous level's bytes, because Orbis
        // `sceKernelMapDirectMemory`/`MapFlexibleMemory` return ZEROED pages and
        // Windows only zeroes a page on its first commit. Left uncleared, a
        // freshly-constructed object reads an unset pointer field as a stale
        // non-null value, survives its null check, and faults (the measured
        // level-transition worker faults). Only this FULL re-map is zeroed;
        // partial-overlap extends (below) intentionally preserve the backed
        // overlap so a growing mapping keeps its data.
        if range_all(&state.vmm, addr, end, |vma| vma.kind.is_host_backed()) {
            let reuse = Self::backed_ranges_outside_core(
                &state,
                addr,
                end,
                self.base,
                self.base + ARENA_SPAN,
            );
            self.zero_reused_ranges(&reuse);
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
        // Each VMA segment decides how it can be backed. Inside the sparse tail,
        // or inside a range the guest already reserved, the address space is
        // ours and only needs committing — and `MEM_RESERVE` over an existing
        // reservation fails with ERROR_INVALID_ADDRESS, so it must *not* be
        // re-reserved. Outside everything, nothing owns it yet and it needs
        // both. Mixed ranges are handled segment-by-segment so an overlap can
        // preserve existing pages while extending into an unowned tail.
        let mut actions = Vec::new();
        // Segments served out of the process-lifetime title-VA window blocks:
        // recorded so `Drop` can decommit them (the blocks are never released,
        // so releasing `base`/`os_reservations` does not return this memory).
        let mut window_segments = Vec::new();
        let mut cursor = addr;
        while cursor < end {
            let vma = state.vmm.find(cursor)?;
            // Never let one segment straddle a title-VA-window block edge:
            // inside a block the address space is already ours (commit-only),
            // outside it is not, and one VirtualAlloc action cannot mix the two.
            let segment_end = vma.end().min(end).min(next_title_va_edge(cursor));
            let segment_len = segment_end.checked_sub(cursor)?;
            // Startup claimed this address space for exactly this call: the
            // map still calls it `Foreign` (nothing guest-visible lives there
            // yet) or `Free` (a previous fixed map here was munmapped), but the
            // host pages only need committing, and a fixed-address MEM_RESERVE
            // would fail against our own block.
            let in_title_window = title_va_block_containing(cursor)
                .is_some_and(|(base, len)| segment_end <= base + len);
            if vma.kind.is_host_backed() {
                // Preserve already-backed overlap byte-for-byte.
            } else if matches!(vma.kind, VmaType::Reserved)
                || (in_title_window && (vma.kind.is_free() || vma.kind == VmaType::Foreign))
                || (vma.kind.is_free()
                    && cursor >= self.base + ARENA_SPAN
                    && segment_end <= self.base + RESERVED_SPAN)
            {
                actions.push((cursor, segment_len, MEM_COMMIT, false));
                if in_title_window {
                    window_segments.push((cursor, segment_len));
                }
            } else if vma.kind == VmaType::Foreign {
                actions.push((cursor, segment_len, MEM_RESERVE | MEM_COMMIT, true));
            } else {
                diagnose_fixed_map_failure(addr, end, cursor);
                return None;
            }
            cursor = segment_end;
        }

        // Back every uncovered segment before publishing the combined VMA.
        // Newly reserved blocks are recorded only after every action succeeds,
        // so `Drop` cannot double-release partial work.
        let mut newly_reserved = Vec::new();
        // Segments whose up-front commit the host refused (commit limit): kept
        // reserved and served demand-committed instead of failing. Relabeled
        // `Reserved` after the combined map so `commit_on_demand` backs them.
        let mut demand_committed = Vec::new();
        for &(segment_addr, segment_len, flags, owns_reservation) in &actions {
            let raw = if take_forced_commit_refusal() {
                // Test-only: behave as if the host refused this up-front commit,
                // driving the demand-commit fallback below deterministically.
                std::ptr::null_mut()
            } else {
                // SAFETY: the VMA walk proved this segment either belongs to one
                // of our reservations (`MEM_COMMIT`) or is wholly unowned
                // (`MEM_RESERVE | MEM_COMMIT`). The exact-address check rejects
                // any allocation-granularity relocation.
                unsafe {
                    VirtualAlloc(
                        segment_addr as *const c_void,
                        segment_len as usize,
                        flags,
                        PAGE_READWRITE,
                    )
                }
            };
            if !raw.is_null() && raw as u64 == segment_addr {
                if owns_reservation {
                    newly_reserved.push((segment_addr, segment_len));
                }
                continue;
            }

            // The up-front commit was refused. Committing the whole segment
            // charges it against the process commit limit (RAM + pagefile), and
            // a title mapping GiB of direct memory at a fixed address exhausts it
            // exactly as the mmap path does — ASTRO.BOT's 1.94 GiB libc mspace at
            // 0x300000000 faults here the moment the machine is under commit
            // pressure. Failing is the wrong answer when only *commit*, not the
            // address space, is unavailable: degrade to the same demand-commit
            // that `map_console_va` and `sceKernelReserveVirtualRange` already
            // run. Reserve the address space (when we do not own it yet), record
            // the span `Reserved`, and let `commit_on_demand` back each page from
            // the fault handler as the guest reaches it. Caveat (as for every
            // demand-committed range): only guest CPU access and HLE *writes*
            // back a page; an HLE/GPU host-side *read* of a never-written page
            // reads unbacked. Strictly better than the old hard failure — the
            // whole mapping was `0xffffffff` then — and direct memory is
            // overwhelmingly written before it is read.
            if owns_reservation {
                // A combined RESERVE|COMMIT never half-succeeds, but a stray
                // misplacement must still be released before we retry.
                if !raw.is_null() {
                    // SAFETY: this iteration's own fresh, unpublished mapping.
                    unsafe {
                        VirtualFree(raw, 0, MEM_RELEASE);
                    }
                }
                // MEM_RESERVE charges address space, not commit, so it clears the
                // commit limit that just refused the combined call. If the
                // address itself is unavailable it fails and nothing can back it.
                // SAFETY: `segment_addr`/`segment_len` are page aligned and, by
                // the VMA walk, name wholly unowned space.
                let reserved = unsafe {
                    VirtualAlloc(
                        segment_addr as *const c_void,
                        segment_len as usize,
                        MEM_RESERVE,
                        PAGE_READWRITE,
                    )
                };
                if reserved.is_null() || reserved as u64 != segment_addr {
                    if !reserved.is_null() {
                        // SAFETY: fresh, unpublished reservation from just above.
                        unsafe {
                            VirtualFree(reserved, 0, MEM_RELEASE);
                        }
                    }
                    for &(base, _) in &newly_reserved {
                        // SAFETY: each entry is a complete reservation created by
                        // an earlier iteration and not yet stored in arena state.
                        unsafe {
                            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
                        }
                    }
                    for &(w_addr, w_len) in &window_segments {
                        // SAFETY: each entry lies inside a process-lifetime
                        // title-VA-window block; decommitting undoes any pages
                        // an earlier iteration committed there (a no-op on the
                        // still-uncommitted ones) without touching the block.
                        unsafe {
                            VirtualFree(w_addr as *mut c_void, w_len as usize, MEM_DECOMMIT);
                        }
                    }
                    // Say WHO owns the address: this is the Shell-vs-CLI
                    // divergence class (a host allocation squatting a guest
                    // fixed address) making itself legible in the log.
                    diagnose_fixed_map_failure(addr, end, segment_addr);
                    return None;
                }
                newly_reserved.push((segment_addr, segment_len));
            }
            // A segment we already own (`!owns_reservation`) needs no OS call:
            // leaving its existing reservation uncommitted is precisely the
            // demand-commit state the fault handler expects.
            demand_committed.push((segment_addr, segment_len));
        }
        for (base, size) in newly_reserved {
            state.os_reservations.push(base);
            state.external_mappings.insert(base, size);
        }
        // Window segments are not `os_reservations` (the block is process-
        // lifetime, not this arena's to release); they are remembered so `Drop`
        // can decommit their pages instead.
        state.window_commits.extend(window_segments);
        state.vmm.map_range(
            addr,
            length,
            VmaType::Flexible,
            prot::CPU_READ_WRITE,
            None,
            MMAP_NAME,
            false,
        );
        // Relabel any demand-committed segment `Reserved` so `commit_on_demand`
        // recognises it and backs it per-page; the combined map above already
        // published the committed and already-backed segments as host memory.
        if !demand_committed.is_empty() {
            for &(segment_addr, segment_len) in &demand_committed {
                state.vmm.map_range(
                    segment_addr,
                    segment_len,
                    VmaType::Reserved,
                    prot::NO_ACCESS,
                    None,
                    MMAP_NAME,
                    false,
                );
            }
            tracing::warn!(
                addr = format_args!("{addr:#x}"),
                len = format_args!("{length:#x}"),
                segments = demand_committed.len(),
                "map_at: commit refused; serving the range demand-committed"
            );
        }
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

        // Independent low-VA mappings are whole Windows reservations. Release
        // exact whole mappings eagerly; Windows does not permit `MEM_RELEASE`
        // of only a sub-range.
        if state.external_mappings.get(&addr).copied() == Some(length) {
            state.external_mappings.remove(&addr);
            if let Some(index) = state.os_reservations.iter().position(|&base| base == addr) {
                state.os_reservations.swap_remove(index);
            }
            // SAFETY: `addr` is the exact reservation base and was removed from
            // both ownership tables above, preventing a repeated release.
            unsafe {
                VirtualFree(addr as *mut c_void, 0, MEM_RELEASE);
            }
            state.vmm.map_range(
                addr,
                length,
                VmaType::Foreign,
                prot::NO_ACCESS,
                None,
                "host",
                false,
            );
            return;
        }

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

        // The committed heap's usable size is one page short of HEAP_SIZE: the
        // top page is the heap|stack guard (see `new`). Allocating exactly that
        // usable span fills the committed fast path at HEAP_OFFSET.
        let core = arena
            .alloc(HEAP_SIZE - PAGE_SIZE, 16)
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

    /// Kernel-selected mappings must use the console's low virtual-address
    /// windows. Minecraft passes an 8 MiB direct-memory mapping straight to
    /// native `sceLibcMspaceCreate`; the old 16 TiB arena address fails libc's
    /// range validation and turns into a null mspace.
    #[test]
    fn mmap_returns_page_aligned_address_accepted_by_native_libc() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");

        let addr = arena.mmap(0x2000, PAGE_SIZE).expect("mmap should succeed");
        assert_eq!(addr % PAGE_SIZE, 0);
        assert!(
            (SYSTEM_MANAGED_MIN..SYSTEM_MANAGED_LIMIT).contains(&addr)
                || (USER_MAPPING_MIN..USER_MAPPING_LIMIT).contains(&addr),
            "kernel-selected mapping {addr:#x} is outside PS5 VA windows"
        );
        assert!(
            addr + 0x2000 <= USER_MAPPING_LIMIT,
            "native libc rejects the mapping end"
        );
    }

    /// ASTRO.BOT's opening `sceKernelAllocateDirectMemory(0x7980_0000)` — 1.94
    /// GiB — routes to `mmap`. Before `mmap` could obtain a full console-VA
    /// mapping this failed outright on the title's FIRST allocation; the game
    /// then built its heap over a garbage base.
    /// The returned memory must be genuinely usable, not merely a live address.
    #[test]
    fn mmap_handles_astro_bots_opening_direct_memory_request() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let astro_bot_request = 0x7980_0000u64; // measured from the retail title

        let addr = arena
            .mmap(astro_bot_request, raeen_core::PS5_PAGE_SIZE as u64)
            .expect("a 1.94 GiB console-VA mmap must succeed");
        assert!(
            addr >= SYSTEM_MANAGED_MIN && addr + astro_bot_request <= USER_MAPPING_LIMIT,
            "oversized mmap must stay in console VA space, got {addr:#x}"
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
            .reserve(until_dawn_request, raeen_core::PS5_PAGE_SIZE as u64)
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

    /// Kernel-chosen virtual mappings are part of the guest ABI: libc records
    /// the returned address and uses it to choose its allocator layout. They
    /// therefore cannot depend on whichever hole Windows ASLR happens to hand
    /// `VirtualAlloc(NULL, ...)` in this emulator process — reservations must
    /// come from a fixed base and advance deterministically.
    ///
    /// That determinism is what this asserts. The *base* is [`RESERVE_MIN`], not
    /// the console mapping base it used to be: a reservation placed low lands
    /// inside the two ABI-validated mapping windows and shreds them. Measured on
    /// Until Dawn / Dragon Ball, whose opening 512 GiB reservation left
    /// `USER_MAPPING` with hundreds of GiB free but a largest contiguous block of
    /// 510 MiB, so every later 1 GiB `sceKernelAllocateDirectMemory` failed to
    /// place and the title cascaded down to 128 KiB and gave up (15 failures ->
    /// 0 with the split). Mappings must live in those windows; reservations —
    /// address space the title reads back and uses directly — need not, so they
    /// are the ones that move.
    #[test]
    fn kernel_chosen_reservations_are_deterministic_and_outside_the_mapping_windows() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let len = 0x0200_0000u64;

        let first = arena
            .reserve(len, raeen_core::PS5_PAGE_SIZE as u64)
            .expect("first reservation");
        let second = arena
            .reserve(len, raeen_core::PS5_PAGE_SIZE as u64)
            .expect("second reservation");

        // Deterministic: a fixed base, then contiguous — never an ASLR hole.
        assert_eq!(first, RESERVE_MIN);
        assert_eq!(second, first + len);
        // And clear of both mapping windows, which is the point of the split.
        assert!(first >= USER_MAPPING_LIMIT);
    }

    #[test]
    fn hinted_reservation_honors_the_guest_window_and_exact_unmap_releases_it() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let hint = 0x0010_0000_0000u64; // 64 GiB, V8's measured cage hint.
        let len = 0x0200_0000u64;

        let first = arena
            .reserve_with_hint(hint, len, raeen_core::PS5_PAGE_SIZE as u64, false)
            .expect("hinted reservation");
        assert_eq!(first, hint, "a free hint must be used, not discarded");

        arena.munmap(first, len);
        let second = arena
            .reserve_with_hint(hint, len, raeen_core::PS5_PAGE_SIZE as u64, true)
            .expect("the exact range must be reusable after whole unmap");
        assert_eq!(second, hint);
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
                .reserve(half_tib, raeen_core::PS5_PAGE_SIZE as u64)
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
                .mmap(len, raeen_core::PS5_PAGE_SIZE as u64)
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
            .reserve(len, raeen_core::PS5_PAGE_SIZE as u64)
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
            .reserve(0x1_0000, raeen_core::PS5_PAGE_SIZE as u64)
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
            .reserve(eight_gib, raeen_core::PS5_PAGE_SIZE as u64)
            .expect("first sparse reservation");
        let second = arena
            .reserve(eight_gib, raeen_core::PS5_PAGE_SIZE as u64)
            .expect("second sparse reservation");

        // Addresses come from the OS now, so only non-overlap is promised —
        // not adjacency, and not a location inside the arena.
        assert!(
            first + eight_gib <= second || second + eight_gib <= first,
            "reservations {first:#x} and {second:#x} overlap"
        );
        let mut byte = [0u8; 1];
        // An HLE READ of an untouched reservation still refuses — there is
        // no data a never-written page could deliver, and this is what keeps
        // "reserved" observably different from "committed".
        assert!(!arena.read(first, &mut byte));
        // An HLE WRITE demand-commits the touched page exactly as a native
        // guest store would (the measured getdents-into-lazy-malloc case) —
        // one page, not the reservation.
        assert!(arena.write(first, &[1]));
        assert!(arena.read(first, &mut byte));
        assert_eq!(byte, [1]);
        // A page nothing wrote is still unreadable: the write above committed
        // its own page only.
        assert!(!arena.read(first + 0x1000, &mut byte));

        let mapped = arena
            .map_at(first, 0x2_0000, raeen_core::PS5_PAGE_SIZE as u64)
            .expect("commit inside prior sparse reservation");
        assert_eq!(mapped, first);
        assert!(arena.write(first + 0x10, &[0xAB]));
        assert!(arena.read(first + 0x10, &mut byte));
        assert_eq!(byte, [0xAB]);
        // Past the mapped run but inside the reservation: a write
        // demand-commits (native-store parity). Outside every reservation:
        // still a hard reject — wild pointers don't get pages.
        assert!(arena.write(first + 0x2_0000, &[1]));
        assert!(!arena.write(0x1000, &[1]));
    }

    /// Fixed direct-memory mappings may overlap an existing mapping and extend
    /// into the adjacent unowned range. Minecraft grows one resource window
    /// this way: the second 1.5 MiB map starts 512 KiB into the first and adds
    /// another 512 KiB at the end. Rejecting the mixed backed/foreign range
    /// sends its allocator into an unbounded retry loop.
    #[test]
    fn map_at_extends_an_overlapping_external_mapping() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let base = 0x2062_f0000;
        let len = 0x18_0000;

        assert_eq!(arena.map_at(base, len, PAGE_SIZE), Some(base));
        assert!(arena.write(base + 0x10, &[0xAB]));

        let overlapping = base + 0x8_0000;
        assert_eq!(
            arena.map_at(overlapping, len, PAGE_SIZE),
            Some(overlapping),
            "a fixed map must keep the backed overlap and reserve its new tail"
        );

        let mut byte = [0u8; 1];
        assert!(arena.read(base + 0x10, &mut byte));
        assert_eq!(
            byte,
            [0xAB],
            "overlap extension must preserve existing bytes"
        );
        assert!(arena.write(base + 0x1f_ffff, &[0xCD]));
        assert!(arena.read(base + 0x1f_ffff, &mut byte));
        assert_eq!(byte, [0xCD], "the newly extended tail must be backed");
    }

    /// Orbis `sceKernelMapDirectMemory`/`MapFlexibleMemory` return ZEROED pages.
    /// A title re-mapping a fixed VA it already holds fully backed (ASTRO.BOT
    /// reuses its 0x3_0000_0000 direct-memory window across level transitions,
    /// without unmapping between them) hits `map_at`'s fully-backed fast path,
    /// where Windows will NOT re-zero the already-committed pages. Left stale, a
    /// freshly-constructed object reads an unset pointer field as a leftover
    /// non-null value, passes its null check, and faults — the measured
    /// level-transition worker faults. The full re-map must come back zeroed.
    #[test]
    fn map_at_full_remap_zeroes_stale_bytes_for_orbis_contract() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let base = 0x4_0000_0000u64; // a fixed VA outside the arena core
        let len = 0x4_0000u64; // 256 KiB

        // First map: fresh reserve+commit, Windows-zeroed.
        assert_eq!(arena.map_at(base, len, PAGE_SIZE), Some(base));

        // The previous "level" scribbles a stale non-null pointer into the pool,
        // exactly the kind of leftover field the fault walks.
        let stale = 0xffff_ffff_ffff_ff2fu64;
        assert!(arena.write(base + 0x10, &stale.to_le_bytes()));
        assert!(arena.write(base + len - 8, &stale.to_le_bytes()));

        // Second map at the SAME VA (no unmap between): the new level's fresh
        // Orbis map must hand back zeroed memory, not the stale bytes.
        assert_eq!(
            arena.map_at(base, len, PAGE_SIZE),
            Some(base),
            "re-mapping a fully-backed fixed VA must still succeed"
        );

        let mut buf = [0u8; 8];
        assert!(arena.read(base + 0x10, &mut buf));
        assert_eq!(
            u64::from_le_bytes(buf),
            0,
            "the re-mapped range must be zeroed (Orbis map contract)"
        );
        assert!(arena.read(base + len - 8, &mut buf));
        assert_eq!(
            u64::from_le_bytes(buf),
            0,
            "the whole re-mapped range, not just the first page, must be zeroed"
        );
    }

    /// ASTRO.BOT's opening move is a fixed-address direct-memory map: its libc
    /// mspace at 0x300000000, 1.94 GiB. Committing that whole span up front
    /// charges the host commit limit (RAM + pagefile), and under pressure the
    /// commit is refused — the guest then wrote to 0x300000020 and faulted
    /// (measured, logs/raeen.txt). `map_at` must serve the request either way:
    /// commit up front when the machine has the headroom, or reserve and
    /// demand-commit when it does not. Both outcomes leave the address the guest
    /// asked for backed on first touch, so the exact faulting offset round-trips.
    #[test]
    fn map_at_serves_astro_bots_fixed_1_94_gib_direct_memory() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let base = 0x3_0000_0000u64; // ASTRO.BOT's libc mspace address
        let len = 0x7980_0000u64; // 1.94 GiB, the measured opening request

        assert_eq!(
            arena.map_at(base, len, PAGE_SIZE),
            Some(base),
            "a fixed direct-memory map must never fail on commit pressure alone"
        );

        // The exact byte the guest faulted on in the log.
        let mut byte = [0u8; 1];
        assert!(
            arena.write(base + 0x20, &[0xAB]),
            "base+0x20 must be usable"
        );
        assert!(arena.read(base + 0x20, &mut byte));
        assert_eq!(byte, [0xAB]);

        // Deep inside the mapping stays reachable too — a lazily backed span
        // must back any page the guest reaches, not only the first.
        assert!(arena.write(base + len - 0x10, &[0xCD]));
        assert!(arena.read(base + len - 0x10, &mut byte));
        assert_eq!(byte, [0xCD]);
    }

    /// The demand-commit fallback itself — forced deterministically, since a
    /// unit test cannot exhaust the real Windows commit limit. On a refused
    /// up-front commit `map_at` must still return the requested address, leave
    /// it reserved-but-unbacked, back a page on first write (the faulting offset
    /// from logs/raeen.txt), keep neighbours unbacked, and release the
    /// reservation exactly once so a later map at the same address succeeds.
    #[test]
    fn map_at_demand_commits_when_the_up_front_commit_is_refused() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let base = 0x3_0000_0000u64; // ASTRO.BOT's fixed direct-memory VA
        let len = 0x20_0000u64; // 2 MiB — a handful of pages

        // Arm one refusal: the range is a single Foreign segment, so its one
        // up-front RESERVE|COMMIT is forced to fail, taking the reserve-only +
        // demand-commit path.
        FORCE_MAP_AT_COMMIT_REFUSALS.store(1, Ordering::Relaxed);
        assert_eq!(
            arena.map_at(base, len, PAGE_SIZE),
            Some(base),
            "a refused commit must degrade to demand-commit, not fail"
        );
        assert_eq!(
            FORCE_MAP_AT_COMMIT_REFUSALS.load(Ordering::Relaxed),
            0,
            "exactly one refusal consumed"
        );

        // Reserved-but-unbacked: an untouched page reads back nothing — the
        // observable difference from the eager-commit path.
        let mut byte = [0u8; 1];
        assert!(!arena.read(base, &mut byte), "range must start unbacked");

        // A write demand-commits its own page and round-trips.
        assert!(arena.write(base + 0x20, &[0xAB]));
        assert!(arena.read(base + 0x20, &mut byte));
        assert_eq!(byte, [0xAB]);
        // A page nothing wrote stays unbacked: the reservation costs address
        // space, not RAM.
        assert!(!arena.read(base + PAGE_SIZE, &mut byte));

        // The reservation releases cleanly — a later (unforced) map at the same
        // address proves no Windows reservation leaked.
        arena.munmap(base, len);
        assert_eq!(arena.map_at(base, len, PAGE_SIZE), Some(base));
    }

    /// The GPU read path backs a `Reserved` (demand-commit) range the guest
    /// never touched — real HW keeps mapped direct memory always GPU-readable,
    /// so our lazy commit must be transparent to the GPU. Regression guard for
    /// the Shell-path "texture guest range … readable prefix 0x0" draw skips.
    #[test]
    fn gpu_read_demand_commits_a_reserved_range_the_guest_never_touched() {
        use raeen_gpu::GpuGuestMemory;
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let base = 0x3_0000_0000u64;
        let len = 0x20_0000u64; // 2 MiB, several pages

        // Force the demand-commit (reserve-only) path, as under host pressure.
        FORCE_MAP_AT_COMMIT_REFUSALS.store(1, Ordering::Relaxed);
        assert_eq!(arena.map_at(base, len, PAGE_SIZE), Some(base));

        // The guest never touched it: the host-side CPU read still sees it
        // unbacked (only the guest VEH lazily backs on a CPU access).
        let mut byte = [0u8; 1];
        assert!(
            !arena.read(base, &mut byte),
            "range starts unbacked to the CPU read path"
        );

        // The GPU read path, in contrast, backs the whole range on demand and
        // reports it visible — the draw that reads this texture proceeds.
        assert!(
            GpuGuestMemory::validate_gpu_range(&arena, base, len, false),
            "GPU read must back a Reserved range on demand, not refuse it"
        );

        // Having been backed for the GPU, the same pages now read back on the
        // CPU path too (they are ReservedBacked, i.e. host-backed).
        assert!(
            arena.read(base + PAGE_SIZE, &mut byte),
            "a page the GPU-visible commit backed is now host-backed"
        );
        assert_eq!(byte, [0u8], "freshly committed pages read back zero");

        arena.munmap(base, len);
    }

    /// The Shell-vs-CLI wall, regression-proofed. Measured 2026-07-21
    /// (logs/raeen.txt): launched from the Shell, ASTRO.BOT's fixed-address
    /// `sceKernelMapNamedDirectMemory` at 0x300000000 failed and the title
    /// faulted, while the SAME build booted the SAME title from the CLI. In the
    /// GUI process, host allocations (eframe/egui/wgpu/Vulkan, their DLLs) had
    /// seconds to land inside the window before launch; the CLI process was
    /// clean there by accident.
    ///
    /// The guarantee under test is the ORDER the Shell now enforces — reserve
    /// the window first (first statement of `main`), let the host allocate
    /// whatever it likes afterwards, then serve the guest's fixed map:
    /// reserve → squat → map must succeed, and the window itself must refuse
    /// the squat.
    #[test]
    fn title_va_window_reservation_defends_fixed_maps_from_host_squatters() {
        let _lock = crate::dispatch::call_lock();

        // 1. Reserve — exactly what `main` runs before anything else.
        // Idempotent and process-global, like the real thing.
        let report = reserve_title_va_window();
        assert!(
            report.reserved_blocks > 0,
            "the test process should have free space in the window: {report:?}"
        );

        let len = 0x20_0000u64; // 2 MiB — a small direct-memory slice
        // ASTRO.BOT's measured mspace base when the reservation covers it;
        // otherwise the first claimed block that fits (a hostile test-process
        // layout must not turn this into a false failure).
        let target = title_va_block_containing(TITLE_VA_WINDOW_MIN)
            .filter(|&(base, block_len)| base + block_len >= TITLE_VA_WINDOW_MIN + len)
            .map(|_| TITLE_VA_WINDOW_MIN)
            .or_else(|| {
                title_va_blocks()
                    .iter()
                    .find(|&&(_, block_len)| block_len >= len)
                    .map(|&(base, _)| base)
            })
            .expect("at least one claimed block should fit a 2 MiB map");

        // 2. Squat — the host allocation the GUI would have made. OS-chosen,
        // because after the reservation the window is simply not on offer.
        // SAFETY: plain anonymous reservation+commit at an OS-chosen address,
        // released at the end of the test.
        let squatter = unsafe {
            VirtualAlloc(
                core::ptr::null(),
                0x10_0000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!squatter.is_null(), "host allocation should succeed");
        let squatter_addr = squatter as u64;
        assert!(
            !(TITLE_VA_WINDOW_MIN..TITLE_VA_WINDOW_LIMIT).contains(&squatter_addr),
            "a post-reservation host allocation must not land in the window \
             (got {squatter_addr:#x})"
        );
        // An explicit grab AT the defended address must be refused outright.
        // SAFETY: a fixed-address MEM_RESERVE attempt; it must fail, and if it
        // somehow succeeded the release below would undo it.
        let steal = unsafe {
            VirtualAlloc(
                target as *const c_void,
                WINDOWS_ALLOCATION_GRANULARITY as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
            )
        };
        if !steal.is_null() {
            // SAFETY: fresh reservation from just above.
            unsafe {
                VirtualFree(steal, 0, MEM_RELEASE);
            }
        }
        assert!(
            steal.is_null(),
            "the startup reservation must already own {target:#x}"
        );

        // 3. The guest's fixed map succeeds and the memory is real.
        {
            let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
            assert_eq!(
                arena.map_at(target, len, PAGE_SIZE),
                Some(target),
                "a fixed map inside the defended window must succeed even after \
                 host allocations"
            );
            let mut byte = [0u8; 1];
            assert!(arena.write(target + 0x20, &[0xAB]));
            assert!(arena.read(target + 0x20, &mut byte));
            assert_eq!(byte, [0xAB]);

            // Relaunch-within-one-run pattern: unmap, then map the same
            // address again — the range is now `Free` inside our block and
            // must still be commit-only servable.
            arena.munmap(target, len);
            assert_eq!(
                arena.map_at(target, len, PAGE_SIZE),
                Some(target),
                "remap after munmap must succeed inside the window"
            );
        }

        // 4. Arena drop decommitted the window pages but kept the block
        // claimed: a second arena (the next launch) maps the same address.
        {
            let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
            assert_eq!(
                arena.map_at(target, len, PAGE_SIZE),
                Some(target),
                "the window must survive an arena teardown (next launch)"
            );
        }

        // SAFETY: exact base of the squatter reservation made above.
        unsafe {
            VirtualFree(squatter, 0, MEM_RELEASE);
        }
    }

    /// Host-side HLE/GPU copies must not retain an unpinned raw pointer after
    /// validation. An external mapping is a whole Windows reservation and can
    /// be released by another guest thread; the arena lock must serialize the
    /// release with the complete copy, then make every later access fail
    /// cleanly instead of touching freed host memory.
    #[test]
    fn concurrent_external_unmap_cannot_race_guest_memory_copy() {
        let _lock = crate::dispatch::call_lock();
        let arena = std::sync::Arc::new(
            GuestArena::new(&[]).expect("fixed-base reservation should succeed"),
        );
        let base = 0x2068_00000;
        let len = 0x10_0000;
        assert_eq!(arena.map_at(base, len, PAGE_SIZE), Some(base));
        assert!(arena.write(base, &[0xAB; 64]));

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let reader_arena = std::sync::Arc::clone(&arena);
        let reader_barrier = std::sync::Arc::clone(&barrier);
        let reader = std::thread::spawn(move || {
            reader_barrier.wait();
            let mut observed_mapping = false;
            let mut out = [0u8; 64];
            for _ in 0..10_000 {
                if reader_arena.read(base, &mut out) {
                    observed_mapping = true;
                    assert_eq!(out, [0xAB; 64]);
                }
            }
            observed_mapping
        });

        barrier.wait();
        arena.munmap(base, len);
        let _ = reader.join().expect("reader thread must not fault");
        let mut out = [0u8; 64];
        assert!(
            !arena.read(base, &mut out),
            "a released external mapping must fail validation"
        );
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

    /// Inter-region guard pages are `PAGE_NOACCESS`, and the heap allocator
    /// never hands out the heap|stack guard while ordinary allocation keeps
    /// working. A native guest store that overruns into a guard traps (caught
    /// by the VEH); this test proves the guards exist and don't disturb normal
    /// use — it does NOT write to a guard (that would fault the host copy).
    #[test]
    fn inter_region_guard_pages_are_noaccess_and_unallocatable() {
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("fixed-base reservation should succeed");
        let base = GUEST_ARENA_BASE;

        let protect_of = |addr: u64| -> u32 {
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: read-only query of the region containing `addr`, which
            // lies in this arena's committed core.
            let n = unsafe {
                VirtualQuery(
                    addr as *const c_void,
                    &mut info,
                    std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            assert_ne!(n, 0, "VirtualQuery failed at {addr:#x}");
            info.Protect
        };

        let image_guard = base + IMAGE_SIZE - PAGE_SIZE;
        let heap_guard = base + STACK_OFFSET - PAGE_SIZE;
        assert_eq!(protect_of(image_guard), PAGE_NOACCESS, "image|heap guard");
        assert_eq!(protect_of(heap_guard), PAGE_NOACCESS, "heap|stack guard");

        // The page just below each guard is still normal committed memory.
        assert_ne!(
            protect_of(heap_guard - PAGE_SIZE),
            PAGE_NOACCESS,
            "the page below the heap guard stays usable"
        );

        // Ordinary allocation is unaffected, and never lands on the guard.
        for _ in 0..64 {
            let a = arena.alloc(0x1000, 16).expect("heap alloc");
            assert!(
                a < heap_guard || a >= base + STACK_OFFSET,
                "alloc {a:#x} must not fall on the heap|stack guard page"
            );
            assert!(arena.write(a, &[0xAB; 16]));
            let mut out = [0u8; 16];
            assert!(arena.read(a, &mut out));
            assert_eq!(out, [0xAB; 16]);
        }
    }

    /// W^X flips the image to execute+read, and instrumentation patches still
    /// land through `patch_code` (which toggles the bar), while ordinary data
    /// writes elsewhere are unaffected. A guest DATA store into the RX image
    /// would fault the host copy, so this does not attempt one — the VEH proves
    /// that path at runtime.
    #[test]
    fn wx_image_is_execute_read_and_patch_code_still_writes() {
        let _lock = crate::dispatch::call_lock();
        // A one-page image so there is real committed code to protect.
        let arena = GuestArena::new(&[0x90u8; 0x40]).expect("reservation should succeed");
        let base = GUEST_ARENA_BASE;

        let protect_of = |addr: u64| -> u32 {
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            let n = unsafe {
                VirtualQuery(
                    addr as *const c_void,
                    &mut info,
                    std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            assert_ne!(n, 0);
            info.Protect
        };

        // Before: image is RWX.
        assert_eq!(protect_of(base), PAGE_EXECUTE_READWRITE, "image starts RWX");

        // The whole one-page image is the (only) executable segment here.
        assert!(arena.enable_wx_image(&[(0, 0x40)]), "enable W^X");
        assert_eq!(
            protect_of(base),
            PAGE_EXECUTE_READ,
            "the code page is execute+read under W^X"
        );

        // A code patch still lands (patch_code toggles RWX around the store)
        // and leaves the page RX afterwards.
        assert!(arena.patch_code(base + 0x8, &[0xCC]));
        let mut byte = [0u8; 1];
        assert!(arena.read(base + 0x8, &mut byte));
        assert_eq!(byte, [0xCC], "patch_code wrote through the W^X bar");
        assert_eq!(
            protect_of(base),
            PAGE_EXECUTE_READ,
            "protection restored to RX after the patch"
        );

        // Ordinary heap writes (outside the image) are unaffected by W^X.
        let a = arena.alloc(16, 16).expect("heap alloc");
        assert!(arena.write(a, &[0xAB; 16]));
    }

    /// The Orbis-protection → Windows `PAGE_*` translation covers every
    /// meaningful CPU-bit combination (GPU bits ignored; write implies read).
    #[test]
    fn orbis_prot_maps_to_the_right_windows_page_flags() {
        use crate::vmm::prot;
        assert_eq!(orbis_prot_to_win(prot::NO_ACCESS), PAGE_NOACCESS);
        assert_eq!(orbis_prot_to_win(prot::CPU_READ), PAGE_READONLY);
        assert_eq!(orbis_prot_to_win(prot::CPU_READ_WRITE), PAGE_READWRITE);
        assert_eq!(orbis_prot_to_win(prot::CPU_WRITE), PAGE_READWRITE);
        assert_eq!(orbis_prot_to_win(prot::CPU_EXEC), PAGE_EXECUTE);
        assert_eq!(
            orbis_prot_to_win(prot::CPU_READ | prot::CPU_EXEC),
            PAGE_EXECUTE_READ
        );
        assert_eq!(
            orbis_prot_to_win(prot::CPU_READ_WRITE | prot::CPU_EXEC),
            PAGE_EXECUTE_READWRITE
        );
        // GPU bits do not affect the CPU mapping.
        assert_eq!(
            orbis_prot_to_win(prot::CPU_READ | prot::GPU_READ_WRITE),
            PAGE_READONLY
        );
    }

    /// `protect` re-protects a committed page when enforcement is on. The
    /// default no-op path is covered by the trait default; here we drive the
    /// arena override directly so the env gate does not have to be toggled.
    #[test]
    fn protect_reprotects_a_committed_heap_page_read_only() {
        use crate::vmm::prot;
        let _lock = crate::dispatch::call_lock();
        let arena = GuestArena::new(&[]).expect("reservation should succeed");
        let a = arena
            .alloc(PAGE_SIZE, PAGE_SIZE)
            .expect("page-aligned alloc");

        // Force the RO protection regardless of the env gate by calling the
        // Windows path the enforced `protect` uses.
        let mut old = 0u32;
        let ok = unsafe {
            VirtualProtect(
                a as *const c_void,
                PAGE_SIZE as usize,
                PAGE_READONLY,
                &mut old,
            )
        } != 0;
        assert!(ok, "VirtualProtect to RO");

        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe {
            VirtualQuery(
                a as *const c_void,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        assert_ne!(n, 0);
        assert_eq!(info.Protect, PAGE_READONLY, "page is now read-only");
        // Sanity: the translation the enforced path would have used agrees.
        assert_eq!(orbis_prot_to_win(prot::CPU_READ), PAGE_READONLY);
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
