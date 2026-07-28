//! Out-pointer write guard: exact-ABI-width stores into guest out parameters,
//! plus a real stack-residency predicate so an oversized HLE write can never
//! smash the calling guest frame or its `__stack_chk_guard` canary.
//!
//! # The bug class this exists to kill
//!
//! An HLE `*GetInfo` / `*GetState` / `*Query*` export receives a pointer to a
//! caller **local** far more often than to a heap object — a title writes
//!
//! ```c
//! SceFooInfo info;                  // 0x20 bytes of stack frame
//! sceFooGetInfo(handle, &info);     // HLE writes 0x40 -> smashes the frame
//! ```
//!
//! and the compiler placed the stack-protector canary immediately above that
//! local. Every extra byte the HLE writes past the real ABI struct size lands
//! on an adjacent local, a saved register, the canary, or the return address.
//! The guest then dies in `__stack_chk_fail` (GTA V, thread 31) or takes a
//! wild jump some milliseconds later (Until Dawn, ~6.7 s) — with a stack trace
//! that points at the *victim*, never at the HLE function that did the damage.
//!
//! The same class covers writing a field at the wrong width: an
//! `int *out_level` slot is 4 bytes, so storing a `u64` there clobbers the 4
//! bytes of whatever local the compiler packed next to it.
//!
//! # Rules (derived from SharpEmu's out-buffer fix series, GPL-2.0)
//!
//! 1. Write **exactly** the ABI struct size — never rounded up, never
//!    "generous".
//! 2. Write **exactly** the ABI field width.
//! 3. Never derive a write length from a guest register/argument that is not
//!    itself the ABI's declared buffer length.
//! 4. Bulk-initialize (zero-fill) only objects that are **not** caller locals.
//! 5. The same out parameter can legitimately have two shapes; pick by size,
//!    not by convenience.
//! 6. Do not write "reserved" / secondary out slots — those bytes are usually
//!    adjacent caller locals.
//! 7. Do not page-align or round up a size the guest may turn around and use
//!    as an `alloca`/VLA length.
//!
//! # How stack residency is determined
//!
//! Raeen does **not** need an address-range heuristic. Two sources, in order:
//!
//! * **Registered thread stack bounds** —
//!   [`raeen_kernel::OrbisKernel::guest_thread_stacks`] holds
//!   `[base, top)` for every live guest thread: the runtime registers the
//!   arena's stack region for the main thread and each `scePthreadCreate`
//!   worker's freshly allocated stack. Keyed by the same guest thread id
//!   [`crate::GuestThreadScheduler::current_thread`] reports, so the lookup is
//!   exact — no guessing, and a secondary thread's stack (which comes out of
//!   the arena heap, indistinguishable from a heap object by address alone) is
//!   classified correctly.
//! * **Bounded window above `caller_rsp`** — fallback when the runtime
//!   registered nothing (unit tests, direct HLE calls, a host embedding). The
//!   caller's locals live *above* the callee-entry RSP, so the window is
//!   `[caller_rsp, caller_rsp + `[`CALLER_FRAME_WINDOW`]`)`. Anything outside
//!   it is reported as non-stack: the fallback deliberately errs toward
//!   `NonStack`, because a false `Stack` verdict would truncate a legitimate
//!   heap-object initialization, while a false `NonStack` verdict only loses
//!   the extra diagnostic — the width/size clamp still applies unconditionally.

use crate::HleContext;
use dashmap::DashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// How far above the callee-entry RSP the calling frame is assumed to reach
/// when the runtime registered no stack bounds for this thread. 64 KiB is
/// larger than any single ordinary frame (the SysV ABI's own guard-page
/// probing threshold is one page) yet far smaller than a thread stack, so the
/// window catches caller locals without swallowing unrelated allocations.
pub const CALLER_FRAME_WINDOW: u64 = 64 * 1024;

/// Where an out pointer lives, relative to the calling guest thread's stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutRegion {
    /// Inside the calling guest thread's stack: a caller local. Bulk
    /// initialization here is forbidden; only the exact ABI form may be
    /// written.
    Stack,
    /// Outside every stack Raeen knows about — heap, mmap, or image. Safe to
    /// bulk-initialize up to the ABI size.
    NonStack,
    /// No stack information at all: `caller_rsp == 0` and no registered bounds
    /// (unit tests, host-side direct calls). Treated as [`OutRegion::NonStack`]
    /// by the write helpers, but reported distinctly so a caller can be strict.
    Unknown,
}

impl OutRegion {
    /// Whether bulk (whole-object) initialization is permitted here.
    #[must_use]
    pub fn allows_bulk_init(self) -> bool {
        !matches!(self, OutRegion::Stack)
    }
}

/// Per-export count of writes this module clamped or refused. Keyed by the
/// caller-supplied export name, so a test can assert that a bad write was
/// caught and a log reader can see which NID is at fault.
static CLAMPED: LazyLock<DashMap<String, u64>> = LazyLock::new(DashMap::new);

/// Total clamped/refused writes across all exports.
static CLAMPED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// How many oversized out-writes the guard has clamped or refused so far.
/// Surfaced in crash reports: a nonzero value means an HLE export tried to
/// write past an ABI struct and the guard stopped a frame smash.
#[must_use]
pub fn clamped_write_count() -> u64 {
    CLAMPED_TOTAL.load(Ordering::Relaxed)
}

/// How many oversized out-writes the guard clamped for one export name.
#[must_use]
pub fn clamped_write_count_for(export: &str) -> u64 {
    CLAMPED.get(export).map_or(0, |entry| *entry)
}

/// Record an oversized write and log it **once per export** at `warn`, so a
/// title that calls the offender every frame does not drown the log.
fn report_clamp(export: &str, region: OutRegion, ptr: u64, attempted: usize, abi_size: usize) {
    CLAMPED_TOTAL.fetch_add(1, Ordering::Relaxed);
    let first = {
        let mut entry = CLAMPED.entry(export.to_string()).or_insert(0);
        *entry += 1;
        *entry == 1
    };
    if first {
        tracing::warn!(
            export,
            out_ptr = format_args!("{ptr:#x}"),
            attempted_bytes = attempted,
            abi_bytes = abi_size,
            ?region,
            "HLE out-buffer write exceeds the ABI struct size; clamped to the \
             ABI size. On a stack out-pointer the discarded bytes would have \
             smashed the caller's frame or its __stack_chk_guard canary"
        );
    }
}

impl HleContext<'_> {
    /// The calling guest thread's stack as `[base, top)`, or `None` when
    /// neither the runtime's registration nor `caller_rsp` says anything.
    ///
    /// See the module docs for the two sources and why the fallback window is
    /// shaped the way it is.
    #[must_use]
    pub fn caller_stack_region(&self) -> Option<(u64, u64)> {
        let thread = self.guest_threads.current_thread();
        if let Some(bounds) = self.kernel.guest_thread_stacks.get(&thread) {
            let (base, top) = *bounds;
            if base < top {
                return Some((base, top));
            }
        }
        if self.caller_rsp == 0 {
            return None;
        }
        Some((
            self.caller_rsp,
            self.caller_rsp.saturating_add(CALLER_FRAME_WINDOW),
        ))
    }

    /// Classify a prospective out-parameter write of `len` bytes at `ptr`.
    #[must_use]
    pub fn classify_out(&self, ptr: u64, len: u64) -> OutRegion {
        let Some((base, top)) = self.caller_stack_region() else {
            return OutRegion::Unknown;
        };
        let end = ptr.saturating_add(len.max(1));
        // Half-open overlap test: any byte of the write inside the stack makes
        // the whole write a frame write.
        if ptr < top && end > base {
            OutRegion::Stack
        } else {
            OutRegion::NonStack
        }
    }

    /// Write an out struct of **exactly** `abi_size` bytes.
    ///
    /// `payload` shorter than `abi_size` writes only what it carries (a
    /// partial field init is a correctness question for the caller, never a
    /// memory-safety one). `payload` longer than `abi_size` is the bug this
    /// module exists for: the write is clamped to `abi_size`, counted, and
    /// logged once with `export`.
    ///
    /// Returns whether the guest memory write succeeded.
    pub fn write_out_struct(
        &self,
        export: &str,
        ptr: u64,
        abi_size: usize,
        payload: &[u8],
    ) -> bool {
        if ptr == 0 {
            return false;
        }
        let len = if payload.len() > abi_size {
            let region = self.classify_out(ptr, payload.len() as u64);
            report_clamp(export, region, ptr, payload.len(), abi_size);
            abi_size
        } else {
            payload.len()
        };
        if len == 0 {
            return true;
        }
        self.mem.write(ptr, &payload[..len])
    }

    /// Zero-initialize a whole out object of `abi_size` bytes — but only when
    /// the pointer is **not** a caller local.
    ///
    /// This is SharpEmu rule 4. A heap object may be bulk-cleared so the guest
    /// never reads uninitialized padding; a stack object may not, because
    /// "the whole object" as the HLE believes it is routinely larger than the
    /// local the caller actually reserved. On a stack pointer only the first
    /// `minimal` bytes (the fields the export genuinely defines) are cleared.
    ///
    /// Returns whether the guest memory write succeeded.
    pub fn zero_out_object(&self, export: &str, ptr: u64, abi_size: usize, minimal: usize) -> bool {
        if ptr == 0 {
            return false;
        }
        let minimal = minimal.min(abi_size);
        let region = self.classify_out(ptr, abi_size as u64);
        let len = if region.allows_bulk_init() {
            abi_size
        } else {
            if abi_size > minimal {
                report_clamp(export, region, ptr, abi_size, minimal);
            }
            minimal
        };
        if len == 0 {
            return true;
        }
        self.mem.write(ptr, &vec![0u8; len])
    }

    /// Write a caller-declared-length out buffer: `declared` comes from the
    /// ABI's own length parameter (e.g. `getsockopt`'s in/out `optlen`), and
    /// `data` is clamped to it. Use this — never a bare `mem.write` — whenever
    /// the length is not a compile-time ABI constant, so a polluted or hostile
    /// length can only ever shrink the write.
    pub fn write_out_bounded(&self, export: &str, ptr: u64, declared: usize, data: &[u8]) -> bool {
        self.write_out_struct(export, ptr, declared, data)
    }

    /// Write an 8-bit out parameter (`uint8_t *` / `bool *`).
    pub fn write_out_u8(&self, ptr: u64, value: u8) -> bool {
        ptr != 0 && self.mem.write(ptr, &[value])
    }

    /// Write a 16-bit out parameter (`uint16_t *`).
    pub fn write_out_u16(&self, ptr: u64, value: u16) -> bool {
        ptr != 0 && self.mem.write(ptr, &value.to_le_bytes())
    }

    /// Write a **32-bit** out parameter (`int *` / `uint32_t *`).
    ///
    /// The whole point of this helper's existence: taking a `u32` makes the
    /// declared width part of the type, so a `u64` value must be narrowed at
    /// the call site (visible in review) instead of silently storing 8 bytes
    /// into a 4-byte slot and clobbering the next local.
    pub fn write_out_u32(&self, ptr: u64, value: u32) -> bool {
        ptr != 0 && self.mem.write(ptr, &value.to_le_bytes())
    }

    /// Write a signed 32-bit out parameter (`int *`).
    pub fn write_out_i32(&self, ptr: u64, value: i32) -> bool {
        self.write_out_u32(ptr, value as u32)
    }

    /// Write a 64-bit out parameter (`uint64_t *` / `size_t *` / a pointer).
    pub fn write_out_u64(&self, ptr: u64, value: u64) -> bool {
        ptr != 0 && self.mem.write(ptr, &value.to_le_bytes())
    }

    /// Write a signed 64-bit out parameter (`int64_t *` / `off_t *`).
    pub fn write_out_i64(&self, ptr: u64, value: i64) -> bool {
        self.write_out_u64(ptr, value as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, HleContext, TestAllocator, TestMemory, test_ctx};

    /// A `TestMemory` plus the surrounding scaffolding an `HleContext` needs.
    struct Fixture {
        kernel: raeen_kernel::OrbisKernel,
        mem: TestMemory,
        alloc: TestAllocator,
    }

    impl Fixture {
        fn new(size: usize) -> Self {
            Self {
                kernel: raeen_kernel::OrbisKernel::new(),
                mem: TestMemory::new(size),
                alloc: TestAllocator::new(0x1_0000),
            }
        }

        fn ctx(&self) -> HleContext<'_> {
            test_ctx(&self.kernel, &self.mem, &self.alloc)
        }

        /// A context whose caller RSP is `rsp` — the fallback-window source.
        fn ctx_at_rsp(&self, rsp: u64) -> HleContext<'_> {
            HleContext {
                caller_rsp: rsp,
                ..self.ctx()
            }
        }

        /// Register `[base, top)` as guest thread 1's stack (thread 1 is what
        /// the test scheduler double reports as current).
        fn register_stack(&self, base: u64, top: u64) {
            self.kernel.guest_thread_stacks.insert(1, (base, top));
        }
    }

    #[test]
    fn no_stack_information_is_unknown_not_stack() {
        let f = Fixture::new(0x1000);
        let ctx = f.ctx();
        assert_eq!(ctx.caller_stack_region(), None);
        assert_eq!(ctx.classify_out(0x400, 8), OutRegion::Unknown);
        assert!(
            ctx.classify_out(0x400, 8).allows_bulk_init(),
            "unknown must stay permissive so tests and direct calls behave"
        );
    }

    #[test]
    fn registered_thread_bounds_classify_exactly() {
        let f = Fixture::new(0x1000);
        f.register_stack(0x800, 0xC00);
        let ctx = f.ctx();
        assert_eq!(ctx.caller_stack_region(), Some((0x800, 0xC00)));
        assert_eq!(ctx.classify_out(0x900, 0x10), OutRegion::Stack);
        assert_eq!(ctx.classify_out(0x100, 0x10), OutRegion::NonStack);
        // A write starting below the stack but running into it is a frame
        // write: the overlap test is half-open on both ends.
        assert_eq!(ctx.classify_out(0x7F8, 0x10), OutRegion::Stack);
        // Exactly one byte past the top is outside.
        assert_eq!(ctx.classify_out(0xC00, 8), OutRegion::NonStack);
    }

    #[test]
    fn registered_bounds_win_over_the_caller_rsp_window() {
        let f = Fixture::new(0x1000);
        f.register_stack(0x800, 0xC00);
        // `caller_rsp` sits in the heap here (a nonsense value); the
        // registration must be authoritative, so a heap pointer near it is
        // still NonStack.
        let ctx = f.ctx_at_rsp(0x100);
        assert_eq!(ctx.caller_stack_region(), Some((0x800, 0xC00)));
        assert_eq!(ctx.classify_out(0x108, 8), OutRegion::NonStack);
    }

    #[test]
    fn caller_rsp_window_covers_the_frame_above_it_only() {
        let f = Fixture::new(0x1000);
        let ctx = f.ctx_at_rsp(0x8000_0000);
        assert_eq!(
            ctx.caller_stack_region(),
            Some((0x8000_0000, 0x8000_0000 + CALLER_FRAME_WINDOW))
        );
        // A local just above the pushed return address.
        assert_eq!(ctx.classify_out(0x8000_0020, 0x20), OutRegion::Stack);
        // Below the callee-entry RSP is not the caller's frame.
        assert_eq!(ctx.classify_out(0x7FFF_FF00, 8), OutRegion::NonStack);
        // Beyond the window the fallback deliberately errs toward NonStack.
        assert_eq!(
            ctx.classify_out(0x8000_0000 + CALLER_FRAME_WINDOW + 8, 8),
            OutRegion::NonStack
        );
    }

    #[test]
    fn write_out_struct_clamps_an_oversized_payload_and_counts_it() {
        let f = Fixture::new(0x1000);
        f.register_stack(0x800, 0xC00);
        let ctx = f.ctx();
        // Poison the bytes above the 0x20-byte "local" so a clamp failure is
        // visible as a smashed canary.
        const CANARY: [u8; 8] = [0xCA; 8];
        assert!(f.mem.write(0x820, &CANARY));

        let before = clamped_write_count_for("test::Oversized");
        // The export believes the struct is 0x40 bytes; the ABI says 0x20.
        let payload = [0xAAu8; 0x40];
        assert!(ctx.write_out_struct("test::Oversized", 0x800, 0x20, &payload));
        assert_eq!(
            clamped_write_count_for("test::Oversized"),
            before + 1,
            "the clamp must be counted so a crash report can name the export"
        );

        let mut seen = [0u8; 0x28];
        assert!(f.mem.read(0x800, &mut seen));
        assert!(
            seen[..0x20].iter().all(|&b| b == 0xAA),
            "the ABI-sized prefix must be written in full"
        );
        assert_eq!(
            &seen[0x20..0x28],
            &CANARY,
            "bytes past the ABI struct size must be untouched — this is the \
             canary that GTA V dies on"
        );
    }

    #[test]
    fn write_out_struct_writes_an_exact_payload_untouched() {
        let f = Fixture::new(0x1000);
        let ctx = f.ctx();
        let before = clamped_write_count();
        let payload = [0x5Au8; 0x20];
        assert!(ctx.write_out_struct("test::Exact", 0x100, 0x20, &payload));
        let mut seen = [0u8; 0x20];
        assert!(f.mem.read(0x100, &mut seen));
        assert_eq!(seen, payload);
        assert_eq!(
            clamped_write_count(),
            before,
            "an exact-size write must not be reported"
        );
    }

    #[test]
    fn zero_out_object_bulk_initializes_only_off_stack() {
        let f = Fixture::new(0x1000);
        f.register_stack(0x800, 0xC00);
        let ctx = f.ctx();

        // Heap target: the full object may be cleared.
        assert!(f.mem.write(0x100, &[0xFFu8; 0x40]));
        assert!(ctx.zero_out_object("test::Bulk", 0x100, 0x40, 0x10));
        let mut heap = [0xAAu8; 0x40];
        assert!(f.mem.read(0x100, &mut heap));
        assert!(heap.iter().all(|&b| b == 0), "heap object clears in full");

        // Stack target: only the minimal form is cleared.
        assert!(f.mem.write(0x900, &[0xFFu8; 0x40]));
        assert!(ctx.zero_out_object("test::Bulk", 0x900, 0x40, 0x10));
        let mut stack = [0u8; 0x40];
        assert!(f.mem.read(0x900, &mut stack));
        assert!(
            stack[..0x10].iter().all(|&b| b == 0),
            "the minimal form still clears"
        );
        assert!(
            stack[0x10..].iter().all(|&b| b == 0xFF),
            "a caller local must keep every byte past the minimal form"
        );
    }

    #[test]
    fn scalar_helpers_write_exactly_their_declared_width() {
        let f = Fixture::new(0x1000);
        let ctx = f.ctx();
        assert!(f.mem.write(0x200, &[0xFFu8; 16]));

        assert!(ctx.write_out_u32(0x200, 0x1234_5678));
        let mut seen = [0u8; 8];
        assert!(f.mem.read(0x200, &mut seen));
        assert_eq!(&seen[..4], &0x1234_5678u32.to_le_bytes());
        assert_eq!(
            &seen[4..8],
            &[0xFF; 4],
            "a 32-bit out slot must not consume the next 4 bytes — the \
             AudioOut2 queue-level bug"
        );

        assert!(ctx.write_out_u16(0x208, 0xBEEF));
        assert!(ctx.write_out_u8(0x20A, 0x42));
        let mut tail = [0u8; 4];
        assert!(f.mem.read(0x208, &mut tail));
        assert_eq!(tail, [0xEF, 0xBE, 0x42, 0xFF]);
    }

    #[test]
    fn null_out_pointers_are_refused_not_written() {
        let f = Fixture::new(0x1000);
        let ctx = f.ctx();
        assert!(!ctx.write_out_u32(0, 1));
        assert!(!ctx.write_out_u64(0, 1));
        assert!(!ctx.write_out_struct("test::Null", 0, 8, &[0u8; 8]));
        assert!(!ctx.zero_out_object("test::Null", 0, 8, 8));
    }

    #[test]
    fn write_out_bounded_clamps_to_the_caller_declared_length() {
        let f = Fixture::new(0x1000);
        let ctx = f.ctx();
        assert!(f.mem.write(0x300, &[0xFFu8; 16]));
        // The ABI declared 4 bytes; the handler produced 8. Only 4 land.
        assert!(ctx.write_out_bounded("test::Bounded", 0x300, 4, &[0x11u8; 8]));
        let mut seen = [0u8; 8];
        assert!(f.mem.read(0x300, &mut seen));
        assert_eq!(&seen[..4], &[0x11; 4]);
        assert_eq!(&seen[4..], &[0xFF; 4]);
    }
}
