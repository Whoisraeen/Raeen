//! Cooperative fibers (libSceFiber) via **guest-context transfer**.
//!
//! A fiber is a saved guest CPU context ([`GuestRegs`]); `sceFiberRun`/`Switch`/
//! `ReturnToThread` swap the VEH-delivered Windows `CONTEXT` so the guest resumes
//! executing **natively** on the target fiber's own stack — no host OS fibers, no
//! JIT stub. This is SharpEmu's `FiberExports`/`GuestCpuContinuation` model
//! (GPL-2.0), which maps 1:1 onto Raeen's "rewrite the delivered CONTEXT and
//! `EXCEPTION_CONTINUE_EXECUTION`" seam (the same one the terminating-call and
//! resume-on-missing paths use). shadPS4 `fiber.cpp`/`fiber_context.cpp` is the
//! ABI spec. See the `astro-bot-boot-state` memory.
//!
//! `SceFiber` struct offsets (shadPS4 `fiber.h`; SharpEmu `FiberExports` agree):
//! `0x00` magic_start · `0x04` state · `0x08` entry · `0x10` arg_on_initialize ·
//! `0x18` addr_context · `0x20` size_context · `0x28` name[32] · `0x50` flags.
//! The entry is `void(*)(u64 arg_on_initialize /*rdi*/, u64 arg_on_run /*rsi*/)`
//! and must never return.

use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;
use raeen_hle::GuestMemory;
use raeen_kernel::{FiberThreadState, GuestRegs, OrbisKernel};

const OFF_STATE: u64 = 0x04;
const OFF_ENTRY: u64 = 0x08;
const OFF_ARG_ON_INIT: u64 = 0x10;
const OFF_ADDR_CONTEXT: u64 = 0x18;
const OFF_SIZE_CONTEXT: u64 = 0x20;

const STATE_RUN: u32 = 1;
const STATE_IDLE: u32 = 2;

/// `SetFpuRegs` first-entry MXCSR seed — auto-applied for all PS5 titles
/// (`build_ver >= FW_350`), per shadPS4 `fiber.cpp` / SharpEmu.
const FIRST_RUN_MXCSR: u32 = 0x9FC0;
/// `sceFiberGetSelf` off a fiber → `ORBIS_FIBER_ERROR_PERMISSION`.
const FIBER_ERROR_PERMISSION: u64 = 0x8059_0005;

fn r64(mem: &dyn GuestMemory, addr: u64) -> u64 {
    let mut b = [0u8; 8];
    if mem.read(addr, &mut b) {
        u64::from_le_bytes(b)
    } else {
        0
    }
}
fn w64(mem: &dyn GuestMemory, addr: u64, v: u64) {
    if addr != 0 {
        let _ = mem.write(addr, &v.to_le_bytes());
    }
}
fn w32(mem: &dyn GuestMemory, addr: u64, v: u32) {
    if addr != 0 {
        let _ = mem.write(addr, &v.to_le_bytes());
    }
}

/// Snapshot the caller's resume point: the guest returns PAST the fiber call —
/// `rip` = the return address the `call` pushed (`[Rsp]`), `rsp` = just above it,
/// `rax` = 0 (the fiber call's SCE_OK), the rest verbatim.
fn capture(context: &CONTEXT, mem: &dyn GuestMemory) -> GuestRegs {
    GuestRegs {
        rip: r64(mem, context.Rsp),
        rsp: context.Rsp.wrapping_add(8),
        rax: 0,
        rbx: context.Rbx,
        rcx: context.Rcx,
        rdx: context.Rdx,
        rsi: context.Rsi,
        rdi: context.Rdi,
        rbp: context.Rbp,
        r8: context.R8,
        r9: context.R9,
        r10: context.R10,
        r11: context.R11,
        r12: context.R12,
        r13: context.R13,
        r14: context.R14,
        r15: context.R15,
        rflags: u64::from(context.EFlags),
        mxcsr: context.MxCsr,
        fpucw: 0x037F,
    }
}

/// Overwrite the delivered `CONTEXT` with a target snapshot; after
/// `EXCEPTION_CONTINUE_EXECUTION` the guest resumes at `rip` on stack `rsp`.
fn apply(context: &mut CONTEXT, r: &GuestRegs) {
    context.Rip = r.rip;
    context.Rsp = r.rsp;
    context.Rax = r.rax;
    context.Rbx = r.rbx;
    context.Rcx = r.rcx;
    context.Rdx = r.rdx;
    context.Rsi = r.rsi;
    context.Rdi = r.rdi;
    context.Rbp = r.rbp;
    context.R8 = r.r8;
    context.R9 = r.r9;
    context.R10 = r.r10;
    context.R11 = r.r11;
    context.R12 = r.r12;
    context.R13 = r.r13;
    context.R14 = r.r14;
    context.R15 = r.r15;
    context.EFlags = r.rflags as u32;
    context.MxCsr = r.mxcsr;
}

/// The first-run entry frame for a fiber that has never been scheduled: enter
/// `entry(arg_on_initialize, arg_on_run)` on the top of its context buffer.
fn first_run(mem: &dyn GuestMemory, fiber: u64, arg_on_run: u64) -> GuestRegs {
    let entry = r64(mem, fiber + OFF_ENTRY);
    let addr_context = r64(mem, fiber + OFF_ADDR_CONTEXT);
    let size_context = r64(mem, fiber + OFF_SIZE_CONTEXT);
    let arg_on_init = r64(mem, fiber + OFF_ARG_ON_INIT);
    // SysV entry: rsp & 15 == 8, as if a `call` had just pushed the return addr.
    let top = addr_context.wrapping_add(size_context) & !15u64;
    let rsp = top.wrapping_sub(8);
    // A fiber entry must NEVER return; plant 0 so a stray return faults loudly.
    w64(mem, rsp, 0);
    GuestRegs {
        rip: entry,
        rsp,
        rdi: arg_on_init,
        rsi: arg_on_run,
        rflags: 0x202,
        mxcsr: FIRST_RUN_MXCSR,
        fpucw: 0x037F,
        ..Default::default()
    }
}

/// Resume target `fiber`: its saved snapshot (delivering `arg` into the
/// `*arg_on_run` slot it recorded when it suspended), or a fresh first-run frame.
fn resume_target(kernel: &OrbisKernel, mem: &dyn GuestMemory, fiber: u64, arg: u64) -> GuestRegs {
    match kernel.fibers.remove(&fiber) {
        Some((_, (regs, arg_slot))) => {
            w64(mem, arg_slot, arg);
            regs
        }
        None => first_run(mem, fiber, arg),
    }
}

/// Handle a libSceFiber control-transfer NID. Returns `true` if `function` was
/// one of the fiber calls that rewrites the guest `CONTEXT` (the caller then
/// returns `EXCEPTION_CONTINUE_EXECUTION`); `false` for any non-fiber function.
#[must_use]
pub fn handle(
    function: &str,
    kernel: &OrbisKernel,
    mem: &dyn GuestMemory,
    context: &mut CONTEXT,
    thread_id: u64,
) -> bool {
    match function {
        // sceFiberRun(fiber=rdi, arg_on_run_to=rsi, *arg_on_return=rdx)
        "sceFiberRun" => {
            let fiber = context.Rdi;
            let arg = context.Rsi;
            let ret_slot = context.Rdx;
            let root = capture(context, mem);
            kernel.fiber_threads.insert(
                thread_id,
                FiberThreadState {
                    root,
                    current_fiber: fiber,
                    root_arg_slot: ret_slot,
                },
            );
            let target = resume_target(kernel, mem, fiber, arg);
            w32(mem, fiber + OFF_STATE, STATE_RUN);
            apply(context, &target);
            true
        }
        // sceFiberSwitch(fiber=rdi, arg_on_run_to=rsi, *arg_on_run=rdx)
        "sceFiberSwitch" => {
            let target_fiber = context.Rdi;
            let arg = context.Rsi;
            let arg_slot = context.Rdx;
            let Some(mut ts) = kernel.fiber_threads.get_mut(&thread_id) else {
                return false;
            };
            let cur = ts.current_fiber;
            if cur != 0 {
                let saved = capture(context, mem);
                kernel.fibers.insert(cur, (saved, arg_slot));
                w32(mem, cur + OFF_STATE, STATE_IDLE);
            }
            ts.current_fiber = target_fiber;
            drop(ts);
            let target = resume_target(kernel, mem, target_fiber, arg);
            w32(mem, target_fiber + OFF_STATE, STATE_RUN);
            apply(context, &target);
            true
        }
        // sceFiberReturnToThread(arg_on_return=rdi, *arg_on_run=rsi)
        "sceFiberReturnToThread" => {
            let arg_on_return = context.Rdi;
            let arg_slot = context.Rsi;
            let Some(mut ts) = kernel.fiber_threads.get_mut(&thread_id) else {
                return false;
            };
            let cur = ts.current_fiber;
            if cur != 0 {
                let saved = capture(context, mem);
                kernel.fibers.insert(cur, (saved, arg_slot));
                w32(mem, cur + OFF_STATE, STATE_IDLE);
            }
            w64(mem, ts.root_arg_slot, arg_on_return);
            let root = ts.root;
            ts.current_fiber = 0;
            drop(ts);
            apply(context, &root);
            true
        }
        // sceFiberGetSelf(**out=rdi) — not a switch; return normally.
        "sceFiberGetSelf" => {
            let out = context.Rdi;
            let cur = kernel
                .fiber_threads
                .get(&thread_id)
                .map(|t| t.current_fiber)
                .unwrap_or(0);
            if cur == 0 {
                context.Rax = FIBER_ERROR_PERMISSION;
            } else {
                w64(mem, out, cur);
                context.Rax = 0;
            }
            context.Rip = r64(mem, context.Rsp);
            context.Rsp = context.Rsp.wrapping_add(8);
            true
        }
        _ => false,
    }
}
