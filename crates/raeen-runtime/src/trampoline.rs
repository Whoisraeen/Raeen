//! Executable HLE import slots plus the legacy fault-dispatch guard.
//!
//! Linker-visible slots remain eight bytes wide at
//! [`HLE_TRAMPOLINE_BASE`]. Each contains a relative call to one of two nearby
//! bridges. Reviewed context-preserving imports use the zero-fault direct
//! bridge; imports that need a captured machine context reconstruct their slot
//! index and jump into a separate `PAGE_NOACCESS` range so the existing VEH
//! path retains full control-transfer semantics.

use core::ffi::c_void;

use raeen_firmware::{HLE_TRAMPOLINE_BASE, HleTrampoline};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS,
    PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::RuntimeError;

const PAGE_SIZE: u64 = 4096;
const SLOT_SIZE: u64 = 8;
const SLOT_CODE_LEN: u64 = 5;
const TLS_REARM_TRAMPOLINE_BASE: u64 = HLE_TRAMPOLINE_BASE - 0x1_0000;

/// Per-index no-access targets for the compatibility VEH path.
pub(crate) const HLE_SLOW_TRAMPOLINE_BASE: u64 = HLE_TRAMPOLINE_BASE + 0x1000_0000;

// wrfsbase r11; pop r11; ret
const TLS_REARM_CODE: [u8; 8] = [0xF3, 0x49, 0x0F, 0xAE, 0xD3, 0x41, 0x5B, 0xC3];

pub(crate) struct TrampolineGuard {
    code_base: u64,
    slow_base: u64,
    len: u64,
    return_trampoline: u64,
    tls_rearm_trampoline: u64,
}

impl TrampolineGuard {
    /// `returns_float` marks trampolines whose HLE result travels in guest
    /// XMM0 (SysV float/double return) — the runtime forwards
    /// [`raeen_hle::HleRegistry::returns_float`]. Direct-dispatchable float
    /// imports then use the float bridge twin below, so the direct gateway
    /// and the VEH path deliver the result register-identically.
    pub(crate) fn reserve(
        trampolines: &[HleTrampoline],
        returns_float: impl Fn(&HleTrampoline) -> bool,
    ) -> Result<Self, RuntimeError> {
        let count = trampolines.len();
        let logical_len = count as u64 * SLOT_SIZE + 16;
        let slow_len = logical_len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let return_trampoline = HLE_SLOW_TRAMPOLINE_BASE + (count as u64 + 1) * SLOT_SIZE;

        let slow = unsafe {
            VirtualAlloc(
                HLE_SLOW_TRAMPOLINE_BASE as *const c_void,
                slow_len as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
            )
        };
        if slow.is_null() {
            return Err(RuntimeError::MapFailed);
        }

        // Leave room after the compact slots for the generated bridges.
        let code_len = (logical_len + 512).div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let code = unsafe {
            VirtualAlloc(
                HLE_TRAMPOLINE_BASE as *const c_void,
                code_len as usize,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if code.is_null() {
            unsafe { VirtualFree(slow, 0, MEM_RELEASE) };
            return Err(RuntimeError::MapFailed);
        }

        let slow_code = slow_bridge_code();
        let direct_code = direct_bridge_code();
        let float_code = direct_bridge_code_float();
        let slow_bridge = HLE_TRAMPOLINE_BASE + logical_len;
        // Space the bridges by their MEASURED lengths (rounded up), never a
        // bare constant: the slow bridge is 36 bytes but the direct/float
        // bridges are well over 64, and a tighter spacing silently splices
        // one bridge into another (that regression is exactly why the
        // lengths are asserted below).
        let direct_bridge = slow_bridge + (slow_code.len() as u64).div_ceil(64) * 64;
        let float_bridge = direct_bridge + (direct_code.len() as u64).div_ceil(64) * 64;
        debug_assert!(
            float_bridge + float_code.len() as u64 <= HLE_TRAMPOLINE_BASE + logical_len + 512,
            "generated bridges must fit the reserved slack after the slots"
        );
        unsafe {
            core::ptr::copy_nonoverlapping(
                slow_code.as_ptr(),
                slow_bridge as *mut u8,
                slow_code.len(),
            );
            core::ptr::copy_nonoverlapping(
                direct_code.as_ptr(),
                direct_bridge as *mut u8,
                direct_code.len(),
            );
            core::ptr::copy_nonoverlapping(
                float_code.as_ptr(),
                float_bridge as *mut u8,
                float_code.len(),
            );
        }
        for (index, trampoline) in trampolines.iter().enumerate() {
            let slot = HLE_TRAMPOLINE_BASE + index as u64 * SLOT_SIZE;
            write_call_slot(
                slot,
                if direct_dispatchable(trampoline) {
                    if returns_float(trampoline) {
                        float_bridge
                    } else {
                        direct_bridge
                    }
                } else {
                    slow_bridge
                },
            );
        }
        // Index `count` is the invalid-trampoline diagnostic sentinel.
        write_call_slot(HLE_TRAMPOLINE_BASE + count as u64 * SLOT_SIZE, slow_bridge);

        let mut old_protect = 0;
        if unsafe { VirtualProtect(code, code_len as usize, PAGE_EXECUTE_READ, &mut old_protect) }
            == 0
        {
            unsafe {
                VirtualFree(code, 0, MEM_RELEASE);
                VirtualFree(slow, 0, MEM_RELEASE);
            }
            return Err(RuntimeError::MapFailed);
        }
        unsafe {
            FlushInstructionCache(GetCurrentProcess(), code, code_len as usize);
        }

        let rearm = unsafe {
            VirtualAlloc(
                TLS_REARM_TRAMPOLINE_BASE as *const c_void,
                PAGE_SIZE as usize,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            )
        };
        if rearm.is_null() {
            unsafe {
                VirtualFree(code, 0, MEM_RELEASE);
                VirtualFree(slow, 0, MEM_RELEASE);
            }
            return Err(RuntimeError::MapFailed);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                TLS_REARM_CODE.as_ptr(),
                rearm.cast::<u8>(),
                TLS_REARM_CODE.len(),
            );
        }

        Ok(Self {
            code_base: HLE_TRAMPOLINE_BASE,
            slow_base: HLE_SLOW_TRAMPOLINE_BASE,
            len: slow_len,
            return_trampoline,
            tls_rearm_trampoline: TLS_REARM_TRAMPOLINE_BASE,
        })
    }

    pub(crate) fn base(&self) -> u64 {
        self.slow_base
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn return_trampoline(&self) -> u64 {
        self.return_trampoline
    }

    pub(crate) fn tls_rearm_trampoline(&self) -> u64 {
        self.tls_rearm_trampoline
    }
}

impl Drop for TrampolineGuard {
    fn drop(&mut self) {
        unsafe {
            VirtualFree(self.code_base as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(self.slow_base as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(self.tls_rearm_trampoline as *mut c_void, 0, MEM_RELEASE);
        }
    }
}

pub(crate) fn resolve(fault_addr: u64, trampolines: &[HleTrampoline]) -> Option<&HleTrampoline> {
    let offset = fault_addr.checked_sub(HLE_SLOW_TRAMPOLINE_BASE)?;
    let idx = usize::try_from(offset / SLOT_SIZE).ok()?;
    trampolines.get(idx)
}

/// Imports proven not to request a nested guest callback or replace the live
/// guest machine context may run through the executable gateway.  This list is
/// deliberately measured rather than broad: Minecraft's 231-second call-stat
/// capture put these functions at the top of the VEH path (tens of millions of
/// calls), while every entry is an ordinary return-value/memory operation.
///
/// Blocking synchronization and socket calls are safe here: the gateway has
/// already switched to the per-thread host stack and `direct_hle_gateway`
/// releases the diagnostic guest GIL around the handler.  Exit, fibers,
/// `pthread_once`, module initializers, and fatal-exception handlers remain on
/// VEH because they can replace the context or schedule a guest callback.
fn direct_dispatchable(trampoline: &HleTrampoline) -> bool {
    if std::env::var_os("RAEEN_DISABLE_DIRECT_HLE").is_some() {
        return false;
    }
    matches!(
        trampoline.function.as_str(),
        // libc leaves.
        "strlen"
            | "strcmp"
            | "strncmp"
            | "memcmp"
            | "bcmp"
            // Thread identity/TLS and errno.
            | "scePthreadGetspecific"
            | "pthread_getspecific"
            | "scePthreadSetspecific"
            | "pthread_setspecific"
            | "scePthreadGetthreadid"
            | "scePthreadSelf"
            | "pthread_self"
            | "__error"
            | "__errno_location"
            | "__tls_get_addr"
            // The measured synchronization hot path. These calls may block,
            // but never transfer execution to guest code.
            | "scePthreadMutexLock"
            | "scePthreadMutexTrylock"
            | "scePthreadMutexUnlock"
            | "pthread_mutex_lock"
            | "pthread_mutex_trylock"
            | "pthread_mutex_unlock"
            | "scePthreadRwlockRdlock"
            | "scePthreadRwlockWrlock"
            | "scePthreadRwlockTryrdlock"
            | "scePthreadRwlockTrywrlock"
            | "scePthreadRwlockUnlock"
            | "pthread_rwlock_rdlock"
            | "pthread_rwlock_wrlock"
            | "pthread_rwlock_tryrdlock"
            | "pthread_rwlock_trywrlock"
            | "pthread_rwlock_unlock"
            | "scePthreadCondWait"
            | "scePthreadCondTimedwait"
            | "scePthreadCondSignal"
            | "scePthreadCondBroadcast"
            | "pthread_cond_wait"
            | "pthread_cond_timedwait"
            | "pthread_cond_signal"
            | "pthread_cond_broadcast"
            // Clock/status polling and non-blocking network polling.
            | "gettimeofday"
            | "clock_gettime"
            | "sceKernelGetProcessTime"
            | "sceKernelGetProcessTimeCounter"
            | "sceKernelGetProcessTimeCounterFrequency"
            | "recvfrom"
            // Semaphore/audio loops visible in the same title capture.
            | "sceKernelWaitSema"
            | "sceKernelSignalSema"
            | "sceAudioOut2ContextPush"
            | "sceAudioOut2ContextAdvance"
            | "sceAudioOut2PortSetAttributes"
            | "sceAjmBatchInitialize"
            // AGC packet emitters: guest-memory writes only; actual GPU work
            // remains asynchronous in the command submission subsystem.
            | "sceAgcSetCxRegIndirectPatchAddRegisters"
            | "sceAgcGetDataPacketPayloadAddress"
            | "sceAgcCbSetShRegisterRangeDirect"
            | "sceAgcDcbEventWrite"
            | "sceAgcSetShRegIndirectPatchAddRegisters"
            | "sceAgcDcbAcquireMem"
            | "sceAgcDcbDrawIndexOffset"
            // libc float/double-returning leaves (SysV: the result travels in
            // XMM0, so `reserve` routes these to the float bridge twin — see
            // `direct_bridge_code_float`). Pure compute plus bounded guest
            // reads (strtod/atof), exactly the strlen/memcmp shape above; a
            // math-heavy title would otherwise pay an exception round-trip
            // per `cosf`/`powf`.
            | "acosf"
            | "asin"
            | "asinf"
            | "atan"
            | "atan2"
            | "atan2f"
            | "atanf"
            | "atof"
            | "cos"
            | "cosf"
            | "difftime"
            | "exp"
            | "exp2"
            | "exp2f"
            | "expf"
            | "fmod"
            | "fmodf"
            | "ldexpf"
            | "log"
            | "log10"
            | "log10f"
            | "log2"
            | "log2f"
            | "logf"
            | "nextafterf"
            | "pow"
            | "powf"
            | "sin"
            | "sinf"
            | "strtod"
            | "tan"
            | "tanf"
            | "tanhf"
    )
}

fn write_call_slot(slot: u64, target: u64) {
    let displacement = i32::try_from(target as i128 - (slot + SLOT_CODE_LEN) as i128)
        .expect("generated HLE bridges remain within rel32 range");
    let mut code = [0x90u8; SLOT_SIZE as usize];
    code[0] = 0xE8;
    code[1..5].copy_from_slice(&displacement.to_le_bytes());
    unsafe {
        core::ptr::copy_nonoverlapping(code.as_ptr(), slot as *mut u8, code.len());
    }
}

fn slow_bridge_code() -> Vec<u8> {
    // Drop the bridge's internal return address before jumping to the
    // corresponding no-access slot. The VEH therefore sees the exact stack
    // shape it saw before executable slots existed.
    let mut code = vec![0x48, 0x8B, 0x04, 0x24, 0x49, 0xBB]; // mov rax,[rsp]; mov r11,imm64
    code.extend_from_slice(&(HLE_TRAMPOLINE_BASE + 5).to_le_bytes());
    code.extend_from_slice(&[0x4C, 0x29, 0xD8, 0x49, 0xBB]); // sub rax,r11; mov r11,imm64
    code.extend_from_slice(&HLE_SLOW_TRAMPOLINE_BASE.to_le_bytes());
    code.extend_from_slice(&[
        0x4C, 0x01, 0xD8, // add rax,r11
        0x48, 0x83, 0xC4, 0x08, // add rsp,8
        0xFF, 0xE0, // jmp rax
    ]);
    code
}

fn direct_bridge_code() -> Vec<u8> {
    // GS:0x28 is the x64 TEB's application-owned ArbitraryUserPointer. `run`
    // installs DirectThreadState there for the duration of guest execution.
    // Unlike the guest FS base, Windows preserves GS/TEB across preemption.
    // Reading the state through FS made the direct gateway intermittently use
    // unrelated TEB bytes after Windows cleared guest FS, crashing retail
    // titles inside this bridge before the HLE handler was reached.
    let mut c = vec![
        0x65, 0x4C, 0x8B, 0x1C, 0x25, 0x28, 0x00, 0x00, 0x00, // mov r11,gs:[28]
        0x49, 0x89, 0xE2, // mov r10,rsp
        0x49, 0x8B, 0x63, 0x08, // mov rsp,[r11+8]
        0x48, 0x83, 0xE4, 0xF0, // and rsp,-16
        0x48, 0x83, 0xEC, 0x60, // sub rsp,96
        0x48, 0x89, 0x44, 0x24, 0x18, // mov [rsp+24],rax
        0x49, 0x8B, 0x03, // mov rax,[r11]
        0x48, 0x89, 0x44, 0x24, 0x10, // mov [rsp+16],rax
        0x4C, 0x89, 0x54, 0x24, 0x08, // mov [rsp+8],r10
        0x49, 0x8B, 0x02, // mov rax,[r10]
        0x49, 0xBB, // mov r11,base+5
    ];
    c.extend_from_slice(&(HLE_TRAMPOLINE_BASE + 5).to_le_bytes());
    c.extend_from_slice(&[
        0x4C, 0x29, 0xD8, // sub rax,r11
        0x48, 0xC1, 0xE8, 0x03, // shr rax,3
        0x48, 0x89, 0x04, 0x24, // mov [rsp],rax
    ]);
    for n in 0u8..8 {
        c.extend_from_slice(&[0x66, 0x0F, 0xD6, 0x44 | (n << 3), 0x24, 32 + n * 8]);
    }
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax,gateway
    c.extend_from_slice(
        &(crate::dispatch::direct_hle_gateway as *const () as usize as u64).to_le_bytes(),
    );
    c.extend_from_slice(&[
        0xFF, 0xD0, // call rax
        0x4C, 0x8B, 0x54, 0x24, 0x08, // mov r10,[rsp+8]
        0x49, 0x8D, 0x62, 0x08, // lea rsp,[r10+8]
        0xC3, // ret to the original guest caller
    ]);
    c
}

/// The float-returning twin of [`direct_bridge_code`]: identical machine
/// code plus one `movq xmm0, rax` (`66 48 0F 6E C0`) immediately after the
/// gateway call, so a `double`/`float` HLE result reaches the guest in the
/// SysV floating-point return register. [`TrampolineGuard::reserve`] picks
/// this bridge for imports its `returns_float` predicate marks — the same
/// registration-time marker the VEH writeback consults
/// (`dispatch::apply_hle_result`), so both call paths hand the result back
/// register-identically.
fn direct_bridge_code_float() -> Vec<u8> {
    let mut code = direct_bridge_code();
    let call_at = code
        .windows(2)
        .position(|w| w == [0xFF, 0xD0])
        .expect("direct bridge contains the gateway call");
    code.splice(call_at + 2..call_at + 2, [0x66, 0x48, 0x0F, 0x6E, 0xC0]);
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trampoline(library: &str, function: &str) -> HleTrampoline {
        HleTrampoline {
            library: library.to_owned(),
            function: function.to_owned(),
            addr: HLE_TRAMPOLINE_BASE,
        }
    }

    #[test]
    fn measured_ordinary_calls_use_the_direct_gateway() {
        for (library, function) in [
            ("libkernel", "scePthreadGetthreadid"),
            ("libkernel", "scePthreadGetspecific"),
            ("libScePosix", "pthread_mutex_lock"),
            ("libScePosix", "pthread_mutex_unlock"),
            ("libScePosix", "recvfrom"),
            ("libSceAgc", "sceAgcDcbAcquireMem"),
        ] {
            assert!(
                direct_dispatchable(&trampoline(library, function)),
                "{library}::{function}"
            );
        }
    }

    #[test]
    fn context_changing_calls_stay_on_veh() {
        for (library, function) in [
            ("libkernel", "scePthreadExit"),
            ("libScePosix", "pthread_once"),
            ("libkernel", "sceKernelLoadStartModule"),
            ("libSceFiber", "sceFiberSwitch"),
            ("libc", "__cxa_throw"),
            ("libkernel", "sceKernelDebugRaiseException"),
        ] {
            assert!(
                !direct_dispatchable(&trampoline(library, function)),
                "{library}::{function}"
            );
        }
    }

    /// The libc float-returning leaves are ordinary compute/read leaves:
    /// they qualify for the zero-fault direct gateway, and `reserve` routes
    /// them to the float bridge twin via the registry's marker.
    #[test]
    fn float_returning_libc_leaves_use_the_direct_gateway() {
        for function in ["cosf", "pow", "atan2", "strtod", "difftime", "ldexpf"] {
            assert!(
                direct_dispatchable(&trampoline("libc", function)),
                "libc::{function} must be direct-dispatchable"
            );
        }
    }

    #[test]
    fn direct_bridge_reads_state_from_preemption_stable_teb_slot() {
        let direct = direct_bridge_code();
        assert_eq!(
            &direct[..9],
            &[0x65, 0x4C, 0x8B, 0x1C, 0x25, 0x28, 0x00, 0x00, 0x00],
            "direct bridge must read GS:[0x28], never the preemption-unstable guest FS base"
        );
    }

    /// The float bridge is byte-for-byte the direct bridge with a single
    /// `movq xmm0, rax` spliced in after the gateway call — the SysV
    /// float-return delivery the VEH path performs in `apply_hle_result`.
    #[test]
    fn float_bridge_is_direct_bridge_plus_movq_xmm0_rax() {
        const MOVQ: [u8; 5] = [0x66, 0x48, 0x0F, 0x6E, 0xC0];
        let direct = direct_bridge_code();
        let float = direct_bridge_code_float();
        assert_eq!(float.len(), direct.len() + MOVQ.len());
        let call_at = direct
            .windows(2)
            .position(|w| w == [0xFF, 0xD0])
            .expect("direct bridge contains the gateway call");
        assert_eq!(float[call_at + 2..call_at + 2 + 5], MOVQ);
        // Everything around the splice is identical.
        assert_eq!(float[..call_at + 2], direct[..call_at + 2]);
        assert_eq!(float[call_at + 2 + 5..], direct[call_at + 2..]);
    }
}
