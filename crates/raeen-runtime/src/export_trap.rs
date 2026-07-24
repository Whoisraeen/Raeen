//! One-shot **module-export call trap** (diagnostic, env-gated).
//!
//! Answers "does the title ever CALL into this module's exports, and from
//! where?" with near-zero overhead. Gated by `RAEEN_TRAP_MODULE_EXPORTS=<sub>`
//! (case-insensitive substring of the module name, e.g. `cohtml`):
//!
//! 1. At compose time — BEFORE the image is mapped — [`install_module_exports`]
//!    overwrites the **entry byte** of every export of each matching module
//!    with `int3` (`0xCC`), recording `addr -> (module, NID, original byte)`.
//!    Exports outside the module's first `PT_LOAD` (its text segment) are
//!    skipped defensively: they are data exports, and a `0xCC` written into
//!    data would never fault and never be restored.
//! 2. The first call lands on the `int3`; the VEH routes the
//!    `EXCEPTION_BREAKPOINT` here ([`take_hit`]), which logs ONCE at WARN —
//!    module, NID, export address, and the caller's return address at `[rsp]`
//!    — then restores the original byte **permanently** and resumes at the
//!    same RIP. Every later call runs the untouched original at native speed.
//!
//! Unlike [`crate::native_trap`] (a log-every-call detour through relocated
//! prologue stubs), this is a one-shot presence probe: it never needs the
//! prologue to be position-independent and costs one fault per export ever.
//!
//! Deliberately free of `windows_sys`: pure bookkeeping over [`GuestMemory`],
//! so the mechanics unit-test on any host. The VEH glue lives in `dispatch`.

use std::collections::HashMap;
use std::sync::Mutex;

use raeen_hle::GuestMemory;
use tracing::{debug, warn};

/// The trapping instruction planted on each export's entry byte.
pub const INT3: u8 = 0xCC;

struct ExportTrap {
    module: String,
    nid: u64,
    orig: u8,
    /// One-shot: set on the first hit (byte restored). A concurrent second
    /// fault on the same address resumes silently — see [`take_hit`].
    hit: bool,
}

#[derive(Default)]
struct Registry {
    traps: HashMap<u64, ExportTrap>, // absolute guest addr of the export entry
    hits: u64,
}

static REG: Mutex<Option<Registry>> = Mutex::new(None);

/// Register snapshot supplied by the VEH for arbitrary-address diagnostics.
///
/// Export traps normally need only `rsp` to recover the caller. Keeping the
/// additional values in one plain struct makes opt-in reverse-engineering
/// probes useful without growing [`take_hit`] by one parameter per register.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrapRegisters {
    pub rsp: u64,
    pub rbp: u64,
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub r8: u64,
    pub r9: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

/// Case-insensitive substring match of the env filter against a module name.
#[must_use]
pub fn module_matches(filter: &str, module_name: &str) -> bool {
    !filter.trim().is_empty()
        && module_name
            .to_ascii_lowercase()
            .contains(&filter.trim().to_ascii_lowercase())
}

/// Plant a one-shot `int3` on the entry byte of every export of one module.
///
/// Call BEFORE the composed `image` is mapped (like the `__cxa_throw` trap).
/// `exports` are `(nid, module-relative vaddr)` pairs; `exec_range` is the
/// module-relative `[start, end)` of its first `PT_LOAD` (text) segment —
/// exports outside it are **data** exports and are skipped (a `0xCC` in data
/// would corrupt it silently and never restore). Pass `None` to skip that
/// filter when the range is unknown.
///
/// Returns how many traps were installed (also logged at WARN, so a run's log
/// proves the mechanism armed — "no hits" is only meaningful alongside it).
pub fn install_module_exports(
    image: &mut [u8],
    base: u64,
    module_name: &str,
    module_image_offset: u64,
    exports: &[(u64, u64)],
    exec_range: Option<(u64, u64)>,
) -> usize {
    let mut reg = REG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let r = reg.get_or_insert_with(Registry::default);
    let mut installed = 0usize;
    let mut skipped_data = 0usize;
    let mut skipped_range = 0usize;
    let mut skipped_dup = 0usize;
    for &(nid, value) in exports {
        if let Some((start, end)) = exec_range
            && !(start..end).contains(&value)
        {
            skipped_data += 1;
            debug!(
                "export_trap: {module_name} nid={nid:#018x} at +{value:#x} is outside the text \
                 segment [{start:#x},{end:#x}) — data export, skipping"
            );
            continue;
        }
        let Some(off) = module_image_offset
            .checked_add(value)
            .and_then(|o| usize::try_from(o).ok())
            .filter(|&o| o < image.len())
        else {
            skipped_range += 1;
            continue;
        };
        let addr = base.wrapping_add(module_image_offset).wrapping_add(value);
        if r.traps.contains_key(&addr) {
            // Two NIDs aliasing one entry (common for C/posix name pairs):
            // the first registration owns the byte.
            skipped_dup += 1;
            continue;
        }
        let orig = image[off];
        if orig == INT3 {
            // Already int3 (padding or a prior pass) — nothing to learn and
            // restoring "int3" would be meaningless. Leave it alone.
            skipped_dup += 1;
            continue;
        }
        image[off] = INT3;
        r.traps.insert(
            addr,
            ExportTrap {
                module: module_name.to_string(),
                nid,
                orig,
                hit: false,
            },
        );
        installed += 1;
    }
    warn!(
        "export_trap: {module_name} armed {installed} one-shot export trap(s) \
         (skipped {skipped_data} data, {skipped_range} out-of-image, {skipped_dup} duplicate) \
         at base +{module_image_offset:#x}"
    );
    installed
}

/// Arm a one-shot `int3` at each arbitrary eboot-relative code address (RE
/// probe: "does this instruction ever execute"). Reuses the export-trap
/// registry and VEH; the synthetic "nid" is the address itself and the module
/// is `addr-trap`, so a hit logs the address + caller like any export trap.
/// `addrs` are module-relative (0-based); `base` is the guest arena base.
pub fn install_addr_traps(image: &mut [u8], base: u64, addrs: &[u64]) -> usize {
    let mut reg = REG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let r = reg.get_or_insert_with(Registry::default);
    let mut installed = 0usize;
    for &value in addrs {
        let Some(off) = usize::try_from(value).ok().filter(|&o| o < image.len()) else {
            continue;
        };
        let addr = base.wrapping_add(value);
        if r.traps.contains_key(&addr) {
            continue;
        }
        let orig = image[off];
        if orig == INT3 {
            continue;
        }
        image[off] = INT3;
        r.traps.insert(
            addr,
            ExportTrap {
                module: "addr-trap".to_string(),
                nid: value,
                orig,
                hit: false,
            },
        );
        installed += 1;
    }
    warn!("export_trap: armed {installed} arbitrary-address one-shot trap(s)");
    installed
}

/// Service a breakpoint at `fault_addr` if it is one of our traps.
///
/// Returns `true` if the address is (or was) a registered export trap. The
/// caller must resume at `fault_addr`, where the original byte is back in
/// place. Returns `false` for unrelated breakpoints, or if the byte
/// could not be restored (resuming would fault forever, so the caller should
/// pass the exception on and let the run die loudly).
///
/// First hit per export: logs module + NID + export address + the caller's
/// return address read from `[rsp]`, restores the original byte permanently.
/// A concurrent second fault (another thread already fetched the `int3`
/// before the restore) finds `hit == true` and resumes silently.
#[must_use]
pub fn take_hit(fault_addr: u64, mem: &dyn GuestMemory, regs: &TrapRegisters) -> bool {
    let mut reg = REG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(r) = reg.as_mut() else {
        return false;
    };
    let hits_so_far = r.hits;
    let Some(trap) = r.traps.get_mut(&fault_addr) else {
        return false;
    };
    if trap.hit {
        // Lost the race with the restoring thread — the original byte is
        // already back; just re-execute it.
        return true;
    }
    // `patch_code`, not `write`: this restores a byte in the CODE image, which
    // is read-only under W^X. `patch_code` lifts the write bar transiently; it
    // is a plain write when W^X is off.
    if !mem.patch_code(fault_addr, &[trap.orig]) {
        warn!(
            "export_trap: {} nid={:#018x} hit at {fault_addr:#x} but the original byte could \
             not be restored — passing the breakpoint on",
            trap.module, trap.nid
        );
        return false;
    }
    trap.hit = true;
    let mut bytes = [0u8; 8];
    let caller = if mem.read(regs.rsp, &mut bytes) {
        u64::from_le_bytes(bytes)
    } else {
        0
    };
    let n = hits_so_far + 1;
    r.hits = n;
    warn!(
        "EXPORT-TRAP hit #{n}: {} nid={:#018x} entry={fault_addr:#x} caller={caller:#x} \
         (byte restored, further calls run native)",
        trap.module, trap.nid
    );
    if trap.module == "addr-trap" && std::env::var_os("RAEEN_TRACE_STACK_GUARD").is_some() {
        let read_u64 = |address| {
            let mut bytes = [0u8; 8];
            mem.read(address, &mut bytes)
                .then(|| u64::from_le_bytes(bytes))
        };
        warn!(
            "ADDR-TRAP stack guard: rbp={:#x} r13={:#x} \
             saved[rbp-0x30]={:#x?} live[r13]={:#x?}",
            regs.rbp,
            regs.r13,
            regs.rbp.checked_sub(0x30).and_then(read_u64),
            read_u64(regs.r13),
        );
    }
    if trap.module == "addr-trap" && std::env::var_os("RAEEN_TRACE_ADDR_REGS").is_some() {
        warn!(
            "ADDR-TRAP registers: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} \
             rsi={:#x} rdi={:#x} r8={:#x} r9={:#x} r12={:#x} r13={:#x} \
             r14={:#x} r15={:#x} rbp={:#x} rsp={:#x}",
            regs.rax,
            regs.rbx,
            regs.rcx,
            regs.rdx,
            regs.rsi,
            regs.rdi,
            regs.r8,
            regs.r9,
            regs.r12,
            regs.r13,
            regs.r14,
            regs.r15,
            regs.rbp,
            regs.rsp,
        );
    }
    if trap.module == "addr-trap" && std::env::var_os("RAEEN_TRACE_COHTML_PARSER").is_some() {
        let read_u64 = |address| {
            let mut bytes = [0u8; 8];
            mem.read(address, &mut bytes)
                .then(|| u64::from_le_bytes(bytes))
        };
        let read_u32 = |address| {
            let mut bytes = [0u8; 4];
            mem.read(address, &mut bytes)
                .then(|| u32::from_le_bytes(bytes))
        };
        let stream = read_u64(regs.r14.wrapping_add(0x30));
        let cursor = read_u32(regs.r14.wrapping_add(0x3c));
        let around = stream.zip(cursor).and_then(|(stream, cursor)| {
            let start = stream.wrapping_add(u64::from(cursor).saturating_sub(16));
            let mut bytes = [0u8; 48];
            mem.read(start, &mut bytes).then(|| {
                bytes
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
        });
        warn!(
            "ADDR-TRAP Cohtml parser: object={:#x} stream={stream:#x?} \
             cursor={cursor:#x?} bytes[cursor-16..cursor+32]={}",
            regs.r14,
            around.as_deref().unwrap_or("<unreadable>"),
        );
    }
    if trap.module == "addr-trap" && std::env::var_os("RAEEN_TRACE_ADDR_MEMORY").is_some() {
        let dump = |label: &str, center: u64| {
            let start = center.saturating_sub(0x100);
            let mut bytes = [0u8; 0x200];
            if !mem.read(start, &mut bytes) {
                return format!("{label}: unreadable around {center:#x}");
            }
            let rows = bytes
                .chunks(16)
                .enumerate()
                .map(|(index, row)| {
                    let address = start + (index as u64 * 16);
                    let hex = row
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!("{address:#x}: {hex}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("{label} around {center:#x}:\n{rows}")
        };
        warn!(
            "ADDR-TRAP memory:\n{}\n{}",
            dump("field cursor (rbx)", regs.rbx),
            dump("deserializer (r14)", regs.r14),
        );
    }
    if trap.module == "addr-trap" && std::env::var_os("RAEEN_TRACE_ADDR_FRAMES").is_some() {
        let read_u64 = |address| {
            let mut bytes = [0u8; 8];
            mem.read(address, &mut bytes)
                .then(|| u64::from_le_bytes(bytes))
        };
        let mut frame = regs.rbp;
        let mut frames = Vec::new();
        for depth in 0..12 {
            let Some(previous) = read_u64(frame) else {
                break;
            };
            let Some(return_address) = read_u64(frame.wrapping_add(8)) else {
                break;
            };
            let saved = (1..=6)
                .map(|slot| read_u64(frame.wrapping_sub(slot * 8)).unwrap_or(0))
                .map(|value| format!("{value:#x}"))
                .collect::<Vec<_>>()
                .join(",");
            frames.push(format!(
                "#{depth} rbp={frame:#x} ret={return_address:#x} saved[-8..-48]=[{saved}]"
            ));
            if previous <= frame || previous - frame > 0x10_0000 {
                break;
            }
            frame = previous;
        }
        warn!("ADDR-TRAP frame chain:\n{}", frames.join("\n"));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat test memory pretending to be guest space at `base`.
    struct TestMem {
        base: u64,
        bytes: Mutex<Vec<u8>>,
    }

    impl GuestMemory for TestMem {
        fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
            let b = self.bytes.lock().unwrap();
            let Some(off) = guest_addr
                .checked_sub(self.base)
                .and_then(|o| usize::try_from(o).ok())
            else {
                return false;
            };
            let Some(src) = b.get(off..off + out.len()) else {
                return false;
            };
            out.copy_from_slice(src);
            true
        }

        fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
            let mut b = self.bytes.lock().unwrap();
            let Some(off) = guest_addr
                .checked_sub(self.base)
                .and_then(|o| usize::try_from(o).ok())
            else {
                return false;
            };
            let Some(dst) = b.get_mut(off..off + data.len()) else {
                return false;
            };
            dst.copy_from_slice(data);
            true
        }
    }

    // The registry is process-global; each test claims a DISTINCT fake base
    // so parallel tests never collide on trap addresses.

    #[test]
    fn install_patches_entry_bytes_and_skips_data_exports() {
        let base = 0x7000_0000_0000;
        let mut image = vec![0u8; 0x100];
        image[0x10] = 0x55; // push rbp — a code export
        image[0x20] = 0x41; // another code export
        image[0x80] = 0x2a; // a data export (outside "text" [0, 0x40))
        let exports = [(0xAAAA, 0x10u64), (0xBBBB, 0x20), (0xCCCC, 0x80)];
        let n = install_module_exports(&mut image, base, "libtest_a", 0, &exports, Some((0, 0x40)));
        assert_eq!(n, 2, "only the two text-segment exports are armed");
        assert_eq!(image[0x10], INT3);
        assert_eq!(image[0x20], INT3);
        assert_eq!(image[0x80], 0x2a, "data export byte untouched");
    }

    #[test]
    fn one_shot_hit_restores_byte_and_reports_caller() {
        let base = 0x7100_0000_0000;
        let mut image = vec![0u8; 0x100];
        image[0x10] = 0x55;
        // A fake guest stack slot holding the caller's return address.
        let rsp_off = 0x40u64;
        image[0x40..0x48].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
        let n = install_module_exports(
            &mut image,
            base,
            "libtest_b",
            0,
            &[(0x1234, 0x10)],
            Some((0, 0x30)),
        );
        assert_eq!(n, 1);
        let mem = TestMem {
            base,
            bytes: Mutex::new(image),
        };

        // Unrelated address: not ours.
        let regs = TrapRegisters {
            rsp: base + rsp_off,
            ..TrapRegisters::default()
        };
        assert!(!take_hit(base + 0x11, &mem, &regs));

        // First hit: handled, byte restored.
        assert!(take_hit(base + 0x10, &mem, &regs));
        let mut b = [0u8; 1];
        assert!(mem.read(base + 0x10, &mut b));
        assert_eq!(b[0], 0x55, "original entry byte permanently restored");

        // Second hit (the concurrent-race path): still handled, byte intact.
        assert!(take_hit(base + 0x10, &mem, &regs));
        assert!(mem.read(base + 0x10, &mut b));
        assert_eq!(b[0], 0x55);
    }

    #[test]
    fn install_skips_out_of_image_and_duplicate_addresses() {
        let base = 0x7200_0000_0000;
        let mut image = vec![0u8; 0x40];
        image[0x8] = 0x53;
        // 0x9999 is far outside the image; 0x8 twice = one duplicate.
        let exports = [(0x1, 0x8u64), (0x2, 0x8), (0x3, 0x9999)];
        let n = install_module_exports(&mut image, base, "libtest_c", 0, &exports, None);
        assert_eq!(n, 1);
        assert_eq!(image[0x8], INT3);
    }

    #[test]
    fn module_matches_is_case_insensitive_substring() {
        assert!(module_matches("cohtml", "libcohtml.Prospero.prx"));
        assert!(module_matches("COHTML", "libcohtml.Prospero.prx"));
        assert!(module_matches(" cohtml ", "libcohtml.Prospero.prx"));
        assert!(!module_matches("cohtml", "libfmod.prx"));
        assert!(!module_matches("", "libcohtml.Prospero.prx"));
        assert!(!module_matches("   ", "libcohtml.Prospero.prx"));
    }

    #[test]
    fn unrestorable_byte_is_not_claimed() {
        let base = 0x7300_0000_0000;
        let mut image = vec![0u8; 0x20];
        image[0x4] = 0x55;
        let n = install_module_exports(&mut image, base, "libtest_d", 0, &[(0x7, 0x4)], None);
        assert_eq!(n, 1);
        // Memory that refuses every write: restore fails, hit not claimed.
        struct NoWrite;
        impl GuestMemory for NoWrite {
            fn read(&self, _a: u64, _o: &mut [u8]) -> bool {
                false
            }
            fn write(&self, _a: u64, _d: &[u8]) -> bool {
                false
            }
        }
        assert!(!take_hit(base + 0x4, &NoWrite, &TrapRegisters::default()));
        // And it stays armed: a later fault with working memory succeeds.
        let mem = TestMem {
            base,
            bytes: Mutex::new(image),
        };
        assert!(take_hit(base + 0x4, &mem, &TrapRegisters::default()));
        let mut b = [0u8; 1];
        assert!(mem.read(base + 0x4, &mut b));
        assert_eq!(b[0], 0x55);
    }
}
