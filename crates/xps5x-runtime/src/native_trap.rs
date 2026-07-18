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

use tracing::warn;
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;
use xps5x_firmware::HleTrampoline;
use xps5x_hle::GuestMemory;

use crate::GUEST_ARENA_BASE;

const HLE_TRAMPOLINE_BASE: u64 = xps5x_firmware::dynlib::linker::HLE_TRAMPOLINE_BASE;

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
    if prologue_len < 12 || off + prologue_len > image.len() {
        warn!("native_trap: refusing to install {name} at {target_off:#x} (out of range)");
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
        if name.contains("Create") {
            if let Some((base, size)) = r.pending_create.remove(&tid) {
                if context.Rax != 0 && size != 0 {
                    r.mspaces.push((base, base.wrapping_add(size), context.Rax));
                }
            }
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
