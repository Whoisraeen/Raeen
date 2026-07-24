//! Module-export and arbitrary-address traps (diagnostic, env-gated).
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
//! prologue stubs), export and ordinary address traps are one-shot presence
//! probes: they never need the prologue to be position-independent and cost one
//! fault per site ever. `RAEEN_REPEAT_TRAP_ADDR` is the deliberately narrow
//! exception: supported side-effect-free instructions are emulated while their
//! breakpoint remains armed.
//!
//! Deliberately free of `windows_sys`: pure bookkeeping over [`GuestMemory`],
//! so the mechanics unit-test on any host. The VEH glue lives in `dispatch`.

use std::collections::HashMap;
use std::sync::Mutex;

use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};
use raeen_hle::GuestMemory;
use tracing::{debug, warn};

/// The trapping instruction planted on each export's entry byte.
pub const INT3: u8 = 0xCC;

struct ExportTrap {
    module: String,
    nid: u64,
    orig: u8,
    repeat_action: Option<RepeatAction>,
    /// One-shot: set on the first hit (byte restored). A concurrent second
    /// fault on the same address resumes silently — see [`take_hit`].
    hit: bool,
    /// Per-site count. Repeatable probes can suppress their early diagnostics
    /// with `RAEEN_REPEAT_TRAP_LOG_AFTER` while still emulating every hit.
    hit_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RepeatAction {
    Mov32 {
        register: TrapRegister,
        value: u32,
        len: u8,
    },
}

/// Register write required when a repeatable diagnostic trap emulates the
/// side-effect-free instruction it replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapRegister {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

/// How the VEH should resume after servicing one of our breakpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrapHit {
    pub resume_rip: u64,
    pub register_write: Option<(TrapRegister, u64)>,
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
                repeat_action: None,
                hit: false,
                hit_count: 0,
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
    install_addr_traps_inner(image, base, addrs, false)
}

/// Arm repeatable arbitrary-address probes.
///
/// Repeating a software breakpoint normally requires single-stepping the
/// restored instruction, which conflicts with Raeen's FS-base rearm
/// trampoline. Instead, this diagnostic accepts only instructions that can be
/// emulated exactly without memory or flag side effects. It currently supports
/// `mov r32, imm32`, leaves the `int3` armed, and resumes after the original
/// instruction. Unsupported addresses are skipped loudly.
pub fn install_repeating_addr_traps(image: &mut [u8], base: u64, addrs: &[u64]) -> usize {
    install_addr_traps_inner(image, base, addrs, true)
}

fn install_addr_traps_inner(image: &mut [u8], base: u64, addrs: &[u64], repeating: bool) -> usize {
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
        let repeat_action = if repeating {
            let Some(action) = decode_repeat_action(&image[off..], addr) else {
                warn!(
                    "export_trap: repeatable address {addr:#x} is not a supported \
                     side-effect-free instruction; skipping"
                );
                continue;
            };
            Some(action)
        } else {
            None
        };
        image[off] = INT3;
        r.traps.insert(
            addr,
            ExportTrap {
                module: "addr-trap".to_string(),
                nid: value,
                orig,
                repeat_action,
                hit: false,
                hit_count: 0,
            },
        );
        installed += 1;
    }
    let kind = if repeating { "repeatable" } else { "one-shot" };
    warn!("export_trap: armed {installed} arbitrary-address {kind} trap(s)");
    installed
}

fn decode_repeat_action(bytes: &[u8], ip: u64) -> Option<RepeatAction> {
    let mut decoder = Decoder::with_ip(
        64,
        bytes.get(..15.min(bytes.len()))?,
        ip,
        DecoderOptions::NONE,
    );
    let instruction = decoder.decode();
    if instruction.is_invalid()
        || instruction.mnemonic() != Mnemonic::Mov
        || instruction.op0_kind() != OpKind::Register
        || instruction.op1_kind() != OpKind::Immediate32
    {
        return None;
    }
    let register = match instruction.op0_register() {
        Register::EAX => TrapRegister::Rax,
        Register::ECX => TrapRegister::Rcx,
        Register::EDX => TrapRegister::Rdx,
        Register::EBX => TrapRegister::Rbx,
        Register::ESP => TrapRegister::Rsp,
        Register::EBP => TrapRegister::Rbp,
        Register::ESI => TrapRegister::Rsi,
        Register::EDI => TrapRegister::Rdi,
        Register::R8D => TrapRegister::R8,
        Register::R9D => TrapRegister::R9,
        Register::R10D => TrapRegister::R10,
        Register::R11D => TrapRegister::R11,
        Register::R12D => TrapRegister::R12,
        Register::R13D => TrapRegister::R13,
        Register::R14D => TrapRegister::R14,
        Register::R15D => TrapRegister::R15,
        _ => return None,
    };
    Some(RepeatAction::Mov32 {
        register,
        value: instruction.immediate32(),
        len: instruction.len() as u8,
    })
}

/// Service a breakpoint at `fault_addr` if it is one of our traps.
///
/// Returns a resume disposition if the address is (or was) a registered trap.
/// One-shot callers resume at `fault_addr`, where the original byte is back in
/// place. Returns `None` for unrelated breakpoints, or if the byte
/// could not be restored (resuming would fault forever, so the caller should
/// pass the exception on and let the run die loudly).
///
/// First hit per export: logs module + NID + export address + the caller's
/// return address read from `[rsp]`, restores the original byte permanently.
/// A concurrent second fault (another thread already fetched the `int3`
/// before the restore) finds `hit == true` and resumes silently.
#[must_use]
pub fn take_hit(fault_addr: u64, mem: &dyn GuestMemory, regs: &TrapRegisters) -> Option<TrapHit> {
    let mut reg = REG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let r = reg.as_mut()?;
    let hits_so_far = r.hits;
    let trap = r.traps.get_mut(&fault_addr)?;
    if trap.hit {
        // Lost the race with the restoring thread — the original byte is
        // already back; just re-execute it.
        return Some(TrapHit {
            resume_rip: fault_addr,
            register_write: None,
        });
    }
    let disposition = if let Some(RepeatAction::Mov32 {
        register,
        value,
        len,
    }) = trap.repeat_action
    {
        TrapHit {
            resume_rip: fault_addr.wrapping_add(u64::from(len)),
            // A 32-bit x86 register write zero-extends into the corresponding
            // 64-bit register.
            register_write: Some((register, u64::from(value))),
        }
    } else {
        // `patch_code`, not `write`: this restores a byte in the CODE image, which
        // is read-only under W^X. `patch_code` lifts the write bar transiently; it
        // is a plain write when W^X is off.
        if !mem.patch_code(fault_addr, &[trap.orig]) {
            warn!(
                "export_trap: {} nid={:#018x} hit at {fault_addr:#x} but the original byte could \
             not be restored — passing the breakpoint on",
                trap.module, trap.nid
            );
            return None;
        }
        trap.hit = true;
        TrapHit {
            resume_rip: fault_addr,
            register_write: None,
        }
    };
    trap.hit_count = trap.hit_count.saturating_add(1);
    let repeat_log_after = std::env::var("RAEEN_REPEAT_TRAP_LOG_AFTER")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1)
        .max(1);
    let verbose = trap.repeat_action.is_none() || trap.hit_count >= repeat_log_after;
    if !verbose {
        r.hits = r.hits.saturating_add(1);
        return Some(disposition);
    }
    let mut bytes = [0u8; 8];
    let caller = if mem.read(regs.rsp, &mut bytes) {
        u64::from_le_bytes(bytes)
    } else {
        0
    };
    let n = hits_so_far + 1;
    r.hits = n;
    if trap.repeat_action.is_some() {
        warn!(
            "EXPORT-TRAP hit #{n}: {} nid={:#018x} entry={fault_addr:#x} caller={caller:#x} \
             (instruction emulated, repeatable trap remains armed)",
            trap.module, trap.nid
        );
    } else {
        warn!(
            "EXPORT-TRAP hit #{n}: {} nid={:#018x} entry={fault_addr:#x} caller={caller:#x} \
             (byte restored, further calls run native)",
            trap.module, trap.nid
        );
    }
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
        if trap.repeat_action.is_some() {
            let code = regs.rsi;
            let compression_base = code & 0xffff_ffff_0000_0000;
            let relocation_compressed = read_u32(code.wrapping_add(3));
            let relocation = relocation_compressed
                .map(u64::from)
                .map(|value| compression_base | value);
            let relocation_length_smi =
                relocation.and_then(|address| read_u32(address.wrapping_add(3)));
            let relocation_length = relocation_length_smi.map(|value| value >> 1);
            let hex = |address: u64, length: usize| {
                let mut bytes = vec![0u8; length];
                mem.read(address, &mut bytes).then(|| {
                    bytes
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            };
            let code_bytes = hex(code.saturating_sub(1), 0x90);
            let relocation_object =
                relocation.and_then(|address| hex(address.saturating_sub(1), 0x40));
            let relocation_bytes =
                relocation
                    .zip(relocation_length)
                    .and_then(|(address, length)| {
                        usize::try_from(length)
                            .ok()
                            .map(|length| length.min(0x100))
                            .and_then(|length| hex(address.wrapping_add(7), length))
                    });
            warn!(
                "ADDR-TRAP Cohtml code: tagged={code:#x} compression_base={compression_base:#x} \
                 relocation_compressed={relocation_compressed:#x?} relocation={relocation:#x?} \
                 relocation_length_smi={relocation_length_smi:#x?} \
                 relocation_length={relocation_length:#x?}\n\
                 code[-1..+0x8f]={}\nrelocation_object[-1..+0x3f]={}\n\
                 relocation_data={}",
                code_bytes.as_deref().unwrap_or("<unreadable>"),
                relocation_object.as_deref().unwrap_or("<unreadable>"),
                relocation_bytes.as_deref().unwrap_or("<unreadable>"),
            );
        }
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
    Some(disposition)
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
        assert!(take_hit(base + 0x11, &mem, &regs).is_none());

        // First hit: handled, byte restored.
        assert!(take_hit(base + 0x10, &mem, &regs).is_some());
        let mut b = [0u8; 1];
        assert!(mem.read(base + 0x10, &mut b));
        assert_eq!(b[0], 0x55, "original entry byte permanently restored");

        // Second hit (the concurrent-race path): still handled, byte intact.
        assert!(take_hit(base + 0x10, &mem, &regs).is_some());
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
    fn repeatable_mov_immediate_is_emulated_and_remains_armed() {
        let base = 0x7250_0000_0000;
        let mut image = vec![0u8; 0x40];
        // mov edx, 0x1f3e
        image[0x10..0x15].copy_from_slice(&[0xba, 0x3e, 0x1f, 0, 0]);
        assert_eq!(install_repeating_addr_traps(&mut image, base, &[0x10]), 1);
        assert_eq!(image[0x10], INT3);
        let mem = TestMem {
            base,
            bytes: Mutex::new(image),
        };

        for _ in 0..2 {
            let hit = take_hit(base + 0x10, &mem, &TrapRegisters::default())
                .expect("repeatable trap handled");
            assert_eq!(hit.resume_rip, base + 0x15);
            assert_eq!(hit.register_write, Some((TrapRegister::Rdx, 0x1f3e)));
            let mut byte = [0u8; 1];
            assert!(mem.read(base + 0x10, &mut byte));
            assert_eq!(byte[0], INT3, "repeatable breakpoint remains armed");
        }
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
        assert!(take_hit(base + 0x4, &NoWrite, &TrapRegisters::default()).is_none());
        // And it stays armed: a later fault with working memory succeeds.
        let mem = TestMem {
            base,
            bytes: Mutex::new(image),
        };
        assert!(take_hit(base + 0x4, &mem, &TrapRegisters::default()).is_some());
        let mut b = [0u8; 1];
        assert!(mem.read(base + 0x4, &mut b));
        assert_eq!(b[0], 0x55);
    }
}
