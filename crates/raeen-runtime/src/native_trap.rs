//! Native-function **log-and-continue detour** (diagnostic, env-gated).
//!
//! Some guest functions behave differently under our native execution than they
//! do interpreted in other emulators (e.g. the title's `libc.prx`
//! `sceLibcMspaceCreate` impl returns null over valid memory). To see *why*
//! without stubbing the function out, this installs a detour:
//!
//! 1. The function's first whole instructions (>= 12 bytes) are copied into a
//!    relocatable **stub** (the copied bytes must be position-independent — no
//!    rip-relative operands or relative calls; a plain register-only prologue is)
//!    followed by `movabs rax, <func+N>; jmp rax`.
//! 2. Those bytes in the image are overwritten with `movabs rax, <entry_tramp>;
//!    jmp rax` (NOP-padded). Calling the function now faults into `entry_tramp`.
//! 3. [`handle`] on the entry trampoline: reads the caller return address at
//!    `[rsp]`, saves it, rewrites `[rsp]` to a **return trampoline**, logs the
//!    args, and resumes at the stub — which runs the real prologue and jumps back
//!    into the function body. When the function returns it lands in the return
//!    trampoline, where [`handle`] logs `rax` and resumes the real caller.
//!
//! Only touches a run when `install` is called (behind an env gate in the CLI),
//! so normal launches are byte-for-byte unaffected.

use std::collections::HashMap;
use std::sync::Mutex;

use raeen_firmware::HleTrampoline;
use raeen_firmware::dynlib::linker::LinkedUnwindModule;
use raeen_hle::GuestMemory;
use tracing::warn;
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

use crate::GUEST_ARENA_BASE;

const HLE_TRAMPOLINE_BASE: u64 = raeen_firmware::dynlib::linker::HLE_TRAMPOLINE_BASE;

struct EntryTrap {
    name: String,
    /// Guest address of the relocated-prologue stub that continues the function.
    stub: u64,
    /// Trampoline the function's `ret` is redirected to.
    ret_tramp: u64,
}

#[derive(Default)]
struct Registry {
    entries: HashMap<u64, EntryTrap>, // entry-trampoline addr -> continuation
    returns: HashMap<u64, String>,    // return-trampoline addr -> name
    saved: HashMap<u64, Vec<u64>>,    // guest tid -> stack of real caller returns
    /// Every `sceLibcMspaceCreate(base,size) -> handle` seen, newest last.
    /// Used to resolve a null-mspace `Free` back to the mspace that owns the
    /// pointer (see [`handle`]). Newest-first search finds the innermost region
    /// when mspaces nest (a sub-mspace carved out of a parent's memory).
    mspaces: Vec<(u64, u64, u64)>, // (base, end, handle)
    /// tid -> (base,size) captured at a `Create` ENTER, paired with the handle
    /// at its RETURN.
    pending_create: HashMap<u64, (u64, u64)>,
}

static REG: Mutex<Option<Registry>> = Mutex::new(None);

/// The library name every native-trap trampoline is registered under so the VEH
/// dispatch can route it here instead of the HLE registry.
pub const TRAP_LIBRARY: &str = "__native_trap";

/// Bytes the entry patch needs: `movabs rax, imm64` (10) + `jmp rax` (2). Every
/// detour must therefore relocate **at least** this many bytes of prologue.
pub const PATCH_LEN: usize = 12;

/// Where [`install_null_free_guard`] appended its stub, so the caller can keep
/// the owning module's registered extent covering it.
///
/// Without this the stub lands one byte past the module it patches: the fault
/// reporter then says "rip is in NO loaded module" for an address that is very
/// much ours, which is exactly the wrong thing to tell a bug reporter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardStub {
    /// Composed-image offset of the stub's first byte.
    pub image_offset: u64,
    /// Stub length in bytes.
    pub len: u64,
}

/// Length of the whole instruction at `bytes[0]`, or `None` unless it is one
/// this detour is *allowed* to relocate.
///
/// A copied prologue byte-for-byte executes at a different address, so every
/// instruction in it must be position-independent: no rip-relative memory
/// operand, no `rel8`/`rel32` branch. Rather than decode all of x86-64 and try
/// to prove each form safe, this whitelists the handful of forms a compiler
/// actually emits in a function prologue — all of them register-only, none with
/// a memory operand or a displacement — and refuses everything else.
///
/// Refusing is the safe direction: an uninstalled guard risks a null free later,
/// while a mis-copied prologue is a *guaranteed* crash (see the module test
/// `dead_cells_prologue_boundary_is_fourteen_not_thirteen`).
fn relocatable_insn_len(bytes: &[u8]) -> Option<usize> {
    // endbr64 — the first instruction of every CET-enabled function.
    if bytes.starts_with(&[0xf3, 0x0f, 0x1e, 0xfa]) {
        return Some(4);
    }
    // At most one REX prefix. A legacy prefix (operand-size, segment, lock,
    // rep) means this is not the plain register traffic we accept.
    let rex = matches!(bytes.first(), Some(0x40..=0x4f));
    let mut i = usize::from(rex);
    let rex_w = rex && bytes[0] & 0x08 != 0;
    let op = *bytes.get(i)?;
    i += 1;
    // `mod == 3` = register direct: no memory operand, so nothing rip-relative
    // and no displacement to fix up.
    let reg_direct = |i: usize| bytes.get(i).is_some_and(|modrm| modrm >> 6 == 0b11);
    let len = match op {
        // push/pop r64 — the canonical prologue, operand encoded in the opcode.
        0x50..=0x5f => i,
        0x90 => i, // nop
        // add/or/and/sub/xor/cmp/test/mov, r/m <- r and r <- r/m.
        0x01 | 0x09 | 0x21 | 0x29 | 0x31 | 0x39 | 0x85 | 0x89 | 0x8b => {
            reg_direct(i).then_some(i + 1)?
        }
        // Group 1 with imm8 / imm32 on a register (`sub rsp, 0x28`).
        0x83 => reg_direct(i).then_some(i + 2)?,
        0x81 => reg_direct(i).then_some(i + 5)?,
        // mov r/m, imm32 on a register.
        0xc7 => reg_direct(i).then_some(i + 5)?,
        // mov r32, imm32 / movabs r64, imm64.
        0xb8..=0xbf => i + if rex_w { 8 } else { 4 },
        _ => return None,
    };
    (len <= bytes.len()).then_some(len)
}

/// The first whole-instruction boundary at or after `min_len` in `bytes`, if
/// every instruction up to it is safe to relocate (see [`relocatable_insn_len`]).
///
/// This is the fix for the defect that killed Dead Cells in 2.2 s: the caller
/// used to hardcode a byte count. Whether that count lands on an instruction
/// boundary depends on the *title's own* `libc.prx` build, and Dead Cells'
/// differs from Minecraft's — 13 bytes cut a 3-byte `mov rbx,rdi` in half, so
/// the copied tail desynchronized the decoder straight into the stub's own
/// `movabs` immediate.
fn relocatable_prologue_len(bytes: &[u8], min_len: usize) -> Option<usize> {
    let mut total = 0usize;
    while total < min_len {
        total += relocatable_insn_len(bytes.get(total..)?)?;
    }
    Some(total)
}

/// Install a detour on the function at composed-image offset `target_off`,
/// copying `prologue_len` (>= 12) bytes of whole instructions. Appends the stub
/// to `image` and two trampolines to `trampolines`. Call BEFORE the image is
/// mapped (like the `__cxa_throw` trap). No-op on out-of-range input.
pub fn install(
    image: &mut Vec<u8>,
    trampolines: &mut Vec<HleTrampoline>,
    target_off: u64,
    prologue_len: usize,
    name: &str,
) {
    let off = target_off as usize;
    if prologue_len < PATCH_LEN || off + prologue_len > image.len() {
        warn!("native_trap: refusing to install {name} at {target_off:#x} (out of range)");
        return;
    }
    // The caller names a byte count; it is only correct if it happens to land on
    // a whole-instruction boundary of relocatable instructions in THIS title's
    // build. Verify rather than trust — copying a truncated instruction plants a
    // stub that decodes into garbage.
    if relocatable_prologue_len(&image[off..], prologue_len) != Some(prologue_len) {
        warn!(
            "native_trap: refusing to install {name} at {target_off:#x} — {prologue_len} bytes is \
             not a relocatable whole-instruction boundary in this build (prologue {:02x?})",
            &image[off..(off + prologue_len).min(image.len())]
        );
        return;
    }
    let target_abs = GUEST_ARENA_BASE + target_off;
    let cont = target_abs + prologue_len as u64;

    // The stub: original prologue bytes, then jump back past the patch.
    let stub_off = image.len() as u64;
    let stub_addr = GUEST_ARENA_BASE + stub_off;
    let mut stub = image[off..off + prologue_len].to_vec();
    stub.extend_from_slice(&[0x48, 0xb8]); // movabs rax, imm64
    stub.extend_from_slice(&cont.to_le_bytes());
    stub.extend_from_slice(&[0xff, 0xe0]); // jmp rax
    image.extend_from_slice(&stub);

    let entry_tramp = HLE_TRAMPOLINE_BASE + (trampolines.len() as u64) * 8;
    trampolines.push(HleTrampoline {
        library: TRAP_LIBRARY.to_string(),
        function: format!("{name}_entry"),
        addr: entry_tramp,
    });
    let ret_tramp = HLE_TRAMPOLINE_BASE + (trampolines.len() as u64) * 8;
    trampolines.push(HleTrampoline {
        library: TRAP_LIBRARY.to_string(),
        function: format!("{name}_ret"),
        addr: ret_tramp,
    });

    // Overwrite the prologue with `movabs rax, entry_tramp; jmp rax`, NOP-padded.
    let mut patch = vec![0x48, 0xb8];
    patch.extend_from_slice(&entry_tramp.to_le_bytes());
    patch.extend_from_slice(&[0xff, 0xe0]);
    patch.resize(prologue_len, 0x90);
    image[off..off + prologue_len].copy_from_slice(&patch);

    let mut reg = REG.lock().unwrap_or_else(|p| p.into_inner());
    let r = reg.get_or_insert_with(Registry::default);
    r.entries.insert(
        entry_tramp,
        EntryTrap {
            name: name.to_string(),
            stub: stub_addr,
            ret_tramp,
        },
    );
    r.returns.insert(ret_tramp, name.to_string());
    warn!(
        "native_trap: installed {name} at {target_abs:#x} (entry={entry_tramp:#x} ret={ret_tramp:#x} stub={stub_addr:#x})"
    );
}

/// Install a **permanent null-mspace-free guard** on the function at
/// composed-image offset `target_off`. Returns where the stub was appended, or
/// `None` if the guard was refused.
///
/// The relocated prologue length is **measured, not assumed**: whole
/// instructions are walked from `target_off` until at least [`PATCH_LEN`] bytes
/// are covered, and the guard is refused outright if any of them is not
/// position-independent. Hardcoding the count is what broke Dead Cells — see
/// [`relocatable_prologue_len`]. Refusing is deliberately preferred over
/// guessing: a missing guard risks a null free later, a bad stub crashes now.
///
/// The returned [`GuardStub`] must be folded into the owning module's registered
/// extent by the caller (`raeen-gui`), otherwise a fault inside the stub reports
/// as belonging to no module at all.
///
/// Unlike the diagnostic [`install`] detour this adds NO trampoline and takes
/// NO VEH trap on any path — it is pure guest code. The retail libc
/// `sceLibcMspaceFree` dereferences its mspace argument; the title passes
/// `mspace = 0` on scoped-pool cleanup (`0` = "the mspace that owns this ptr" on
/// real HW, which our native libc has no default for), faulting on the null.
/// The patched entry becomes `test rdi,rdi; jnz <real>; xor eax,eax; ret`, so:
///   * a valid mspace (`rdi != 0` — 52k of 52.5k calls) flows straight into the
///     real function at native speed, and
///   * the rare null free returns 0 without dereferencing, leaking that one
///     scoped temp (harmless; the buffer's pool is freed wholesale later).
///
/// This is the load-time, title-agnostic replacement for running the whole boot
/// under the `RAEEN_TRAP_MSPACE` detour — resolve `sceLibcMspaceFree` by NID and
/// point this at it.
pub fn install_null_free_guard(
    image: &mut Vec<u8>,
    target_off: u64,
    name: &str,
) -> Option<GuardStub> {
    let off = usize::try_from(target_off).ok()?;
    if off >= image.len() {
        warn!("null_free_guard: refusing {name} at {target_off:#x} (out of range)");
        return None;
    }
    // Measure the prologue instead of assuming it. Refusing here leaves the
    // retail `sceLibcMspaceFree(0, ptr)` able to fault later — strictly better
    // than planting a stub whose copied tail is half an instruction.
    let Some(prologue_len) = relocatable_prologue_len(&image[off..], PATCH_LEN) else {
        let window = &image[off..(off + 24).min(image.len())];
        warn!(
            "null_free_guard: REFUSING {name} at {target_off:#x} — its prologue is not safely \
             relocatable ({PATCH_LEN}+ bytes of position-independent whole instructions required, \
             got {window:02x?}). The guard is NOT installed: a null-mspace free in this title will \
             fault in the retail libc instead of returning 0."
        );
        return None;
    };
    let target_abs = GUEST_ARENA_BASE + target_off;
    let cont = target_abs + prologue_len as u64;
    let orig = image[off..off + prologue_len].to_vec();

    // Guard stub, appended to the image:
    //   test rdi,rdi; jz .null; <orig prologue>; movabs rax,cont; jmp rax
    //   .null: xor eax,eax; ret
    let stub_off = image.len() as u64;
    let stub_addr = GUEST_ARENA_BASE + stub_off;
    let mut stub = vec![0x48, 0x85, 0xff]; // test rdi, rdi
    // jz over: orig prologue + `movabs rax,imm64` (10) + `jmp rax` (2).
    let jz_rel = u8::try_from(orig.len() + PATCH_LEN).ok()?;
    stub.extend_from_slice(&[0x74, jz_rel]); // jz .null
    stub.extend_from_slice(&orig); // the real prologue
    stub.extend_from_slice(&[0x48, 0xb8]); // movabs rax, imm64
    stub.extend_from_slice(&cont.to_le_bytes());
    stub.extend_from_slice(&[0xff, 0xe0]); // jmp rax
    stub.extend_from_slice(&[0x31, 0xc0, 0xc3]); // .null: xor eax,eax; ret
    let stub_len = stub.len() as u64;
    image.extend_from_slice(&stub);

    // Patch the real prologue with `movabs rax, stub; jmp rax`, NOP-padded.
    let mut patch = vec![0x48, 0xb8];
    patch.extend_from_slice(&stub_addr.to_le_bytes());
    patch.extend_from_slice(&[0xff, 0xe0]);
    patch.resize(prologue_len, 0x90);
    image[off..off + prologue_len].copy_from_slice(&patch);
    warn!(
        "null_free_guard: installed {name} at {target_abs:#x} (stub={stub_addr:#x}+{stub_len:#x}, \
         relocated {prologue_len}-byte prologue)"
    );
    Some(GuardStub {
        image_offset: stub_off,
        len: stub_len,
    })
}

/// Fold a [`GuardStub`] into `module`'s registered extent, so a fault inside the
/// stub is attributed to the module whose function it guards. Returns `false`
/// (leaving `module` untouched) if the stub is not contiguous with its end.
///
/// The stub is appended at the end of the *composed image*. When the module it
/// patches is the last one placed — which is what happens with `libc.prx` — that
/// is exactly one byte past the module, and nothing covers it: both the kernel
/// module table and the unwind table are built from `unwind.image_size`. The
/// measured symptom was a fault report reading "rip is in NO loaded module" for
/// an address Raeen itself planted, which is the least actionable thing a crash
/// report can say.
pub fn cover_guard_stub(module: &mut LinkedUnwindModule, stub: GuardStub) -> bool {
    let module_end = module
        .image_offset
        .wrapping_add(module.unwind.image_vaddr)
        .wrapping_add(module.unwind.image_size);
    if stub.image_offset != module_end {
        warn!(
            "null_free_guard: guard stub at image+{:#x} is NOT contiguous with {} (ends at \
             image+{module_end:#x}) — a fault inside the stub will report as belonging to no \
             loaded module",
            stub.image_offset, module.name,
        );
        return false;
    }
    module.unwind.image_size += stub.len;
    true
}

/// Handle a native-trap trampoline hit from the VEH dispatch. `context.Rip` is
/// the trampoline address. Returns `true` if it was one of ours (the caller then
/// resumes with `EXCEPTION_CONTINUE_EXECUTION`).
#[must_use]
pub fn handle(context: &mut CONTEXT, mem: &dyn GuestMemory, tid: u64) -> bool {
    let mut reg = REG.lock().unwrap_or_else(|p| p.into_inner());
    let Some(r) = reg.as_mut() else {
        return false;
    };
    let rip = context.Rip;

    // Entry: divert the return to `ret_tramp`, log args, resume at the stub.
    if let Some((name, stub, ret_tramp)) = r
        .entries
        .get(&rip)
        .map(|e| (e.name.clone(), e.stub, e.ret_tramp))
    {
        let mut bytes = [0u8; 8];
        let caller_ret = if mem.read(context.Rsp, &mut bytes) {
            u64::from_le_bytes(bytes)
        } else {
            0
        };

        // The retail libc `sceLibcMspaceFree` dereferences the mspace it is
        // handed (reads `mspace+0x370` deep inside). ASTRO.BOT's
        // `fontMemoryCreateByMalloc` (and other scoped-pool paths) frees a
        // pointer back to its pool with `mspace = 0` (`xor edi,edi; call
        // MspaceFree`) — on real HW `0` resolves to "the mspace that owns this
        // pointer", but our native impl has no default and faults on the null.
        // Resolve the owning mspace from the pointer using the Create-region
        // map (newest-first = innermost when pools nest) and free it properly.
        // The 52k valid frees (real handle in rdi) fall straight through.
        if context.Rdi == 0 && name.contains("Free") {
            let ptr = context.Rsi;
            let owner = r
                .mspaces
                .iter()
                .rev()
                .find(|(base, end, _)| ptr >= *base && ptr < *end)
                .map(|(_, _, h)| *h);
            if let Some(handle) = owner {
                context.Rdi = handle;
                warn!(
                    "NATIVE-TRAP {name} null-mspace: resolved ptr={ptr:#x} -> owner mspace {handle:#x}, freeing natively"
                );
                // fall through to the normal native-continue below
            } else {
                context.Rsp = context.Rsp.wrapping_add(8);
                context.Rip = caller_ret;
                context.Rax = 0;
                warn!(
                    "NATIVE-TRAP {name} SKIP null-mspace free (ptr={ptr:#x}, no owning region) -> resume {caller_ret:#x} (leaks)"
                );
                return true;
            }
        }

        // Capture a Create's (base,size) so the RETURN can record the region.
        if name.contains("Create") {
            r.pending_create.insert(tid, (context.Rsi, context.Rdx));
        }

        let _ = mem.write(context.Rsp, &ret_tramp.to_le_bytes());
        r.saved.entry(tid).or_default().push(caller_ret);
        warn!(
            "NATIVE-TRAP {name} ENTER rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x} r8={:#x} r9={:#x} (caller {caller_ret:#x})",
            context.Rdi, context.Rsi, context.Rdx, context.Rcx, context.R8, context.R9
        );
        context.Rip = stub;
        return true;
    }

    // Return: log the result and resume the real caller.
    if let Some(name) = r.returns.get(&rip).cloned() {
        let resume = r
            .saved
            .get_mut(&tid)
            .and_then(std::vec::Vec::pop)
            .unwrap_or(0);
        // Record a completed Create so later null-mspace frees can resolve the
        // owning region. `rax` is the handle; base/size were saved at ENTER.
        if name.contains("Create")
            && let Some((base, size)) = r.pending_create.remove(&tid)
            && context.Rax != 0
            && size != 0
        {
            r.mspaces.push((base, base.wrapping_add(size), context.Rax));
        }
        warn!(
            "NATIVE-TRAP {name} RETURN rax={:#x} rdx={:#x} -> resume {resume:#x}",
            context.Rax, context.Rdx
        );
        if resume != 0 {
            context.Rip = resume;
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dead Cells' (PPSA15552) `libc.prx` `sceLibcMspaceFree` prologue, read
    /// back verbatim from the guard stub in the crash log of the run this fix
    /// came from:
    ///
    /// ```text
    /// 55           push rbp
    /// 48 89 e5     mov  rbp, rsp
    /// 41 57        push r15
    /// 41 56        push r14
    /// 41 54        push r12
    /// 53           push rbx
    /// 48 89 fb     mov  rbx, rdi   <- 3 bytes; byte 13 lands INSIDE it
    /// ```
    const DEAD_CELLS_PROLOGUE: &[u8] = &[
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xfb,
    ];

    /// Minecraft's (PPSA17221) build of the same function: one extra `push r13`,
    /// so its instruction boundaries differ from Dead Cells'. This is the title
    /// the guard must not regress.
    const MINECRAFT_PROLOGUE: &[u8] = &[
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x89,
        0xfb,
    ];

    fn image_with(prologue: &[u8], at: usize) -> Vec<u8> {
        let mut image = vec![0xccu8; 0x200];
        image[at..at + prologue.len()].copy_from_slice(prologue);
        image
    }

    /// The whole defect in one assertion: 13 is not an instruction boundary in
    /// this build, so a hardcoded 13 truncates `mov rbx,rdi` and the copied tail
    /// desynchronizes the decoder into the stub's own `movabs` immediate.
    #[test]
    fn dead_cells_prologue_boundary_is_fourteen_not_thirteen() {
        assert_eq!(
            relocatable_prologue_len(DEAD_CELLS_PROLOGUE, PATCH_LEN),
            Some(14)
        );
        // 13 is not reachable as a boundary: asking for at least 13 yields 14.
        assert_eq!(relocatable_prologue_len(DEAD_CELLS_PROLOGUE, 13), Some(14));
    }

    /// The regression guard for the one title that plays. Minecraft's prologue
    /// reaches 12 exactly on a boundary, so the guard copies 12 whole bytes —
    /// a boundary, and enough for the 12-byte patch.
    #[test]
    fn minecraft_prologue_boundary_is_still_valid() {
        assert_eq!(
            relocatable_prologue_len(MINECRAFT_PROLOGUE, PATCH_LEN),
            Some(12)
        );
        // ...and 13, the value the old code hardcoded, is a real boundary here —
        // which is exactly why the bug hid until a second title showed up.
        assert_eq!(relocatable_prologue_len(MINECRAFT_PROLOGUE, 13), Some(13));
    }

    #[test]
    fn endbr64_and_stack_adjust_prologues_decode() {
        // endbr64; push rbp; mov rbp,rsp; sub rsp,0x30; push rbx
        let bytes = [
            0xf3, 0x0f, 0x1e, 0xfa, 0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x30, 0x53,
        ];
        assert_eq!(relocatable_prologue_len(&bytes, PATCH_LEN), Some(12));
    }

    /// Anything with a memory operand, a relative branch, or an unrecognized
    /// opcode must be refused rather than guessed at.
    #[test]
    fn non_relocatable_prologues_are_refused() {
        // lea rdi, [rip+0x1234] — the classic position-DEPENDENT prologue byte.
        let riprel = [
            0x55, 0x48, 0x89, 0xe5, 0x48, 0x8d, 0x3d, 0x34, 0x12, 0x00, 0x00, 0x53, 0x53,
        ];
        assert_eq!(relocatable_prologue_len(&riprel, PATCH_LEN), None);
        // A `jmp rel32` inside the window would land at the wrong target.
        let branch = [
            0x55, 0x48, 0x89, 0xe5, 0xe9, 0x00, 0x10, 0x00, 0x00, 0x53, 0x53, 0x53, 0x53,
        ];
        assert_eq!(relocatable_prologue_len(&branch, PATCH_LEN), None);
        // `mov rbx, [rdi]` has a memory operand (mod != 3).
        let load = [
            0x55, 0x48, 0x89, 0xe5, 0x48, 0x8b, 0x1f, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53,
        ];
        assert_eq!(relocatable_prologue_len(&load, PATCH_LEN), None);
        // Running off the end of the buffer is a refusal, not a panic.
        assert_eq!(relocatable_prologue_len(&[0x55, 0x53], PATCH_LEN), None);
    }

    /// End-to-end on the real Dead Cells bytes: the guard installs, the stub is
    /// appended intact, and — the property the crash proves was violated — the
    /// copied prologue is whole, so no offset inside the stub is reachable
    /// mid-immediate.
    #[test]
    fn guard_stub_relocates_whole_instructions_and_returns_its_extent() {
        const AT: usize = 0x40;
        let mut image = image_with(DEAD_CELLS_PROLOGUE, AT);
        let before = image.len();
        let stub = install_null_free_guard(&mut image, AT as u64, "sceLibcMspaceFree")
            .expect("Dead Cells' prologue is relocatable at a 14-byte boundary");

        assert_eq!(stub.image_offset, before as u64);
        assert_eq!(stub.image_offset + stub.len, image.len() as u64);

        let s = stub.image_offset as usize;
        let body = &image[s..s + stub.len as usize];
        // test rdi,rdi ; jz .null
        assert_eq!(&body[..3], &[0x48, 0x85, 0xff]);
        assert_eq!(body[3], 0x74);
        // The WHOLE 14-byte prologue was copied — including `48 89 fb`, the
        // instruction the old 13-byte copy sliced in half.
        assert_eq!(&body[5..5 + 14], DEAD_CELLS_PROLOGUE);
        // movabs rax, <target + 14> ; jmp rax ; xor eax,eax ; ret
        assert_eq!(&body[19..21], &[0x48, 0xb8]);
        let cont = u64::from_le_bytes(body[21..29].try_into().unwrap());
        assert_eq!(cont, GUEST_ARENA_BASE + AT as u64 + 14);
        assert_eq!(&body[29..31], &[0xff, 0xe0]);
        assert_eq!(&body[31..34], &[0x31, 0xc0, 0xc3]);

        // The jz must land exactly on `.null`, not inside the immediate.
        let null_target = 5 + usize::from(body[4]);
        assert_eq!(null_target, 31, "jz must reach `xor eax,eax; ret`");

        // The patched entry jumps to the stub.
        assert_eq!(&image[AT..AT + 2], &[0x48, 0xb8]);
        let entry_target = u64::from_le_bytes(image[AT + 2..AT + 10].try_into().unwrap());
        assert_eq!(entry_target, GUEST_ARENA_BASE + stub.image_offset);
        assert_eq!(&image[AT + 10..AT + 12], &[0xff, 0xe0]);
        // Padding runs to the measured boundary, not to a hardcoded count.
        assert!(image[AT + 12..AT + 14].iter().all(|&b| b == 0x90));
    }

    /// An unsafe prologue must leave the image byte-for-byte untouched. A
    /// refused guard risks a null free later; a bad stub crashes immediately.
    #[test]
    fn refused_guard_leaves_the_image_untouched() {
        const AT: usize = 0x40;
        // push rbp; mov rbp,rsp; lea rdi,[rip+...] — not relocatable.
        let prologue = [
            0x55, 0x48, 0x89, 0xe5, 0x48, 0x8d, 0x3d, 0x34, 0x12, 0x00, 0x00, 0x53, 0x53,
        ];
        let mut image = image_with(&prologue, AT);
        let original = image.clone();
        assert_eq!(
            install_null_free_guard(&mut image, AT as u64, "sceLibcMspaceFree"),
            None
        );
        assert_eq!(image, original, "a refused guard must not touch the image");
    }

    fn unwind_module(name: &str, image_offset: u64, image_size: u64) -> LinkedUnwindModule {
        LinkedUnwindModule {
            name: name.to_string(),
            image_offset,
            unwind: raeen_firmware::sprx::UnwindInfo {
                eh_frame_hdr_vaddr: 0,
                eh_frame_vaddr: 0,
                eh_frame_size: 0,
                seg0_vaddr: 0,
                seg0_size: image_size,
                image_vaddr: 0,
                image_size,
            },
            exports: Vec::new(),
            init_vaddr: None,
        }
    }

    /// The measured Dead Cells geometry: libc.prx is last in the composed image,
    /// so the guard stub begins exactly at its end — `base 0x100002954000 + size
    /// 0x141b80 == stub 0x100002a95b80`, which is why the fault reporter found
    /// the RIP in no module at all. Covering it makes the stub part of libc.
    #[test]
    fn a_contiguous_guard_stub_is_folded_into_its_module() {
        let mut libc = unwind_module("libc.prx", 0x295_4000, 0x14_1b80);
        let stub = GuardStub {
            image_offset: 0x295_4000 + 0x14_1b80,
            len: 34,
        };
        assert!(cover_guard_stub(&mut libc, stub));
        assert_eq!(libc.unwind.image_size, 0x14_1b80 + 34);
        // The stub's last byte is now inside the module's registered range.
        let end = libc.image_offset + libc.unwind.image_vaddr + libc.unwind.image_size;
        assert!(stub.image_offset + stub.len <= end);
    }

    /// A module that is NOT last must not have its extent stretched over
    /// unrelated bytes just to swallow the stub.
    #[test]
    fn a_detached_guard_stub_does_not_stretch_the_module() {
        let mut libc = unwind_module("libc.prx", 0x1000, 0x2000);
        let before = libc.unwind.image_size;
        let stub = GuardStub {
            image_offset: 0x9000,
            len: 34,
        };
        assert!(!cover_guard_stub(&mut libc, stub));
        assert_eq!(libc.unwind.image_size, before);
    }

    #[test]
    fn out_of_range_target_is_refused() {
        let mut image = vec![0u8; 0x20];
        let original = image.clone();
        assert_eq!(install_null_free_guard(&mut image, 0x100, "x"), None);
        assert_eq!(image, original);
    }
}
