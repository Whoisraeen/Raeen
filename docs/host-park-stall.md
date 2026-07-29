# Blasphemous II: the "host park" stall was an unarmed instrument

**Date:** 2026-07-28 · **Branch:** `fix-host-park-stall` · **Title:** Blasphemous II,
PPSA13580, Unity/IL2CPP, user-owned retail copy.

Static investigation only. No retail title was executed for this work; everything
below comes from the preserved captures of the four hardware runs, from the
**exact release binary those runs used**, and from in-tree tests.

---

## Summary

The record in `checklist.md` says the stall "changed class" after three blockers
were cleared — from *14 threads in `sceKernelWaitSema`* to *every thread parked in
Raeen's own host synchronization between guest HLE calls*.

**It did not change class.** The two runs that reported the new class had
`RAEEN_STALL_DUMP` set but **not** `RAEEN_TIME_HLE`, and `in_flight_hle` was only
ever populated under `RAEEN_TIME_HLE`. So `IN-FLIGHT HLE: <none — all threads
between calls>` was not an observation that the threads were between calls; it was
an empty map being printed as though it were one. The same run's
`STALL_DUMP (0 threads)` came from counting entries in `recent_hle_calls`, which
is armed by a third env var (`RAEEN_TRACE_EINVAL`) that was also unset.

The threads were, and are, inside their guest waits. The stall is the same one.

| run | `RAEEN_TIME_HLE` | reported in-flight |
|---|---|---|
| `b2` (01:41Z) | **set** | `t1=libScePosix::pthread_cond_wait`, `t2…t15=libkernel::sceKernelWaitSema` |
| `b3`, `b4` (03:53Z) | unset | `<none — all threads between calls>` |

`b2`'s dump also carries a `TIME IN HLE` section; `b3`/`b4` do not — that absence
is the fingerprint of the missing flag, since `hle_call_time` shares the same gate.

**The empty map is itself evidence, and it clears the mutex.** `in_flight_hle` has
a second producer that is **not** gated on any env var:
`pthread_sync.rs`'s `InFlightWait` publishes
`libkernel::scePthreadMutexLock(waiting mutex=…)` for the whole duration of a
contended mutex wait. So `b3`/`b4` reporting an entirely empty map proves that at
the moment of every dump **no thread was waiting on a guest mutex** — which rules
out the recent direct-handoff change in `try_grant_head` (a grant to a thread that
never runs, or a waiter dequeued without a wake) as the cause of this stall,
independently of everything below.

---

## How the park was identified

### The binary is recoverable, and it symbolizes

`target/release/raeen.exe` still carries the CodeView entry for the `b4` run
(`PDBGUID {3083A030-0961-4C5B-95B3-AD7C600F4962}`, `TimeDateStamp 2026-07-29
03:53:28Z` — 16 s before that run's first log line), and `raeen.pdb` sits beside
it. `SymFromAddr` against it resolves every `raeen.exe+0x…` offset in the capture.

**Why the emulator itself could not do that** is a one-constant bug, now fixed:
`raeen_runtime::thread::symbolize_host_addr` passed `SYMOPT_DEFERRED_LOADS`, and
with deferred loads on, `SymFromAddr` answers `ERROR_MOD_NOT_FOUND` (126) for
every address in the main module. `ntdll`/`KERNELBASE` frames kept resolving
because those come from the DLLs' export tables and need no PDB — which is exactly
why the failure read as "our frames have no symbols" instead of "the PDB was never
loaded", and why every host backtrace in the capture printed our frames as bare
offsets.

### Two park sites, partitioned exactly as `b2`'s two groups

Every thread in `b4` sits at the same RIP (`ZwWaitForAlertByThreadId`), but the
*caller* splits them in two, and the split is `{t1}` against `{t2…t15}` — the same
partition `b2` reported with the instrument armed.

**`t1` — certain, from public symbols alone:**

```
ZwWaitForAlertByThreadId ← RtlWaitOnAddress ← WaitOnAddress
  ← parking_lot::condvar::Condvar::wait_until_internal+0x384
  ← raeen_kernel::GuestWaiter::wait_for_signal+0x7b
  ← (raeen_hle::pthread_cond)
```

That is `pthread_cond_wait` parked on its own `GuestWaiter` — matching `b2`'s
`t1=libScePosix::pthread_cond_wait`.

**`t2…t15` — identified from the enclosing function's contents.** The park is an
inlined futex wait (`WaitOnAddress(addr, cmp, 4, ms)` with a `lock cmpxchgb`
byte-lock beside it), so no public symbol names it directly. The function it is
inlined into contains, within 0xC0 bytes of the park:

* the immediate `0x5F5E100` = 100 000 000 ns — a `Duration::from_millis(100)`;
* `std::time::Instant::now`, `Instant + Duration`, and
  `Instant::checked_duration_since`;
* an indirect (vtable) call to
  `<raeen_runtime::dispatch::ActiveContext as raeen_hle::GuestThreadScheduler>::process_is_terminating`;
* a return into `raeen_hle::HleRegistry::call+0x51f`.

That is the complete ingredient list of `kernel_semaphore.rs::hle_wait`'s loop,
and `from_millis(100)` as a *wait slice* is unique to it in `raeen-hle`
(`libsce_posix::posix_sleep`'s 100 ms slice uses `thread::sleep`/`NtDelayExecution`
and has no deadline math or condvar; the event-flag and equeue waits use 50 ms).
Combined with `b2`'s armed table, `t2…t15` are inside
`libkernel::sceKernelWaitSema`.

### `WaitOnAddress` never meant `parking_lot`

The inference "`WaitOnAddress` is `parking_lot`'s parking primitive, so these
threads are parked in **our** code *between* guest HLE calls" fails twice over:

1. The binary imports `WaitOnAddress`/`WakeByAddress{Single,All}` **and**
   `AcquireSRWLockExclusive`/`SleepConditionVariableSRW`. On this toolchain
   (rustc 1.97.0, `x86_64-pc-windows-msvc`) a `WaitOnAddress` frame does not tell
   you which locking library you are in.
2. Even if it had been `parking_lot`, that would not have implied "between HLE
   calls" — `GuestWaiter` *is* `parking_lot`, and it is reached from inside
   `scePthreadCondWait`, i.e. from inside an HLE call. `t1`'s own backtrace shows
   precisely that.

---

## The park is not a bug, and this is why no locking was changed

Both waits are **bounded re-check loops**, not indefinite waits:

| wait | slice | re-checked each slice |
|---|---|---|
| `kernel_semaphore::hle_wait` (`sceKernelWaitSema`) | 100 ms | `try_consume` (the count), teardown, deadline |
| `pthread_cond.rs` cond wait | 10 ms | the waiter's private wake bit, teardown, deadline |

A lost `notify` therefore costs at most one slice, never the run: the semaphore
waiters re-read the count every 100 ms whether or not anybody signalled, and
`GuestWaiter::wake` sets the private wake bit **under the waiter's own lock**
before notifying, so a wake landing in the register-then-park window is observed
by `wait_for_signal`'s locked check instead of being missed (the mechanism the
type's own docs describe, and what `signal_wakes_only_the_oldest_waiter` pins).

So the direct-handoff mutex change, the process-wide `semaphore_signal` condvar,
the `event_flag_signal` pair and the Vulkan `caches` mutex are all **cleared** as
causes of this stall. 14 threads idling in a job-system semaphore and a main
thread waiting on a condvar is what a correctly-implemented, *starved* Unity job
system looks like. Nothing woke them because nothing in the guest ever posted the
work.

`RtlAllocateHeap` in the captured chains was, as suspected, an artifact: the
backtrace was a raw stack scan that kept any qword landing inside a loaded module.
Function-entry addresses (vtable slots, `&fn`) and adjacent duplicates are now
rejected, since a *return* address is never a function's first byte.

---

## What the instrument now reports

`RAEEN_STALL_DUMP` changes in three ways.

**1. It arms what it reads.** `raeen_runtime::dispatch::stall_instruments_armed`
makes `RAEEN_STALL_DUMP` imply `in_flight_hle`, `hle_call_time` and
`recent_hle_calls`. The report can no longer print an unarmed field as an
observation.

**2. The thread inventory is the host thread sampler**, not an opt-in ring, and
"parked" is a positive observation: `raeen_runtime::host_wait_primitive`
classifies the innermost frame against a whitelist of Windows wait syscalls
(`Nt`/`Zw` `WaitForAlertByThreadId`, `WaitForSingleObject`,
`WaitForMultipleObjects`, `SignalAndWaitForSingleObject`, `WaitForKeyedEvent`,
`DelayExecution`, `RemoveIoCompletion`, `WaitForWorkViaWorkerFactory`,
`WaitForDebugEvent`). It is a whitelist of syscalls rather than "the frame is in
ntdll" because a thread inside `RtlAllocateHeap` is also in ntdll and is running.
`None` means "not observed waiting", never "observed running".

**3. Age comes from diffing samples**, so it costs nothing per HLE call.
`StallTracker` fingerprints each thread's state (in-flight call, park primitive,
guest RIP, newest ring entry) and reports a lower bound — `>=41.4s over 7 dump(s)`
— which resets the moment any of those move, including a guest RIP that advances
while the thread makes no HLE calls at all.

The same state that used to print `STALL_DUMP (0 threads)` now prints:

```
STALL_DUMP: 15 guest thread(s) — 15 host-parked, 0 not observed waiting
VERDICT: every guest thread is parked in a host wait. Nothing in the process
can advance on its own — the wake has to come from outside the guest, or it
never comes.
t1(main) PARKED >=41.4s over 7 dump(s) inside libScePosix::pthread_cond_wait [WaitOnAddress futex (std or parking_lot)]
    last returned from: libScePosix::read
    guest rip: eboot+0xb47c11
t2 PARKED >=41.4s over 7 dump(s) inside libkernel::sceKernelWaitSema [WaitOnAddress futex (std or parking_lot)]
    last returned from: libkernel::sceKernelSignalSema
    guest rip: eboot+0xb47bd0
…
HOST BACKTRACES:
  t1: ntdll.dll+0x163cb4(ZwWaitForAlertByThreadId+0x14) ← … ← raeen_kernel::GuestWaiter::wait_for_signal+0x7b
  t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15: ntdll.dll+0x163cb4(…) ← …
```

A thread parked with no in-flight HLE call — the case the old wording asserted
without evidence — now reads `PARKED >=12.0s over 3 dump(s) in host code between
HLE calls [<primitive>]`, and only when the park was actually observed.

---

## Where the stall actually is (narrowed, not established)

The blocker is upstream of every wait above: the guest never posts the work its
own threads are waiting for. The capture's last events, in order, are

```
t1  sceSysmoduleLoadModule(0xa9) · mkdir('/') · sceSysmoduleLoadModule(0xb4) ×2
t1  sceKernelCreateEventFlag "resumeEvent" -> handle 0x1 attr 0x21
t1  creates guest thread 15 (entry image+0x2088…, 1 310 720-byte stack — a
    different entry and a bigger stack than the 13 identical job workers t2…t14)
—— nothing further for the remaining 40 s ——
```

and `b2`'s per-thread timing says what each did with its life:

```
t1  total 0.4s | top libScePosix::read: 0.4s over 3
t15 total 0.0s | top libkernel_unity::sceKernelRaiseException: 0.0s over 1
t3,t5,t7,t9,t11,t13  top libkernel::sceKernelSignalSema: 1 call
t2,t4,t6,t8,t10,t12,t14  top libkernel::sceKernelAllocateMainDirectMemory: 1 call
```

So `t15` — created last, distinct entry, the only thread that is not a job worker
— **raised an Orbis signal and then blocked on a semaphore**. That is the shape of
a managed runtime's stop-the-world suspend: raise at a target thread, then wait for
its handler to acknowledge. `crates/raeen-hle/src/exception.rs` already documents
the failure mode this produces here, in its own words: a thread whose only imports
are direct-leaf-gateway calls — *"`scePthreadMutexLock`, `scePthreadCondWait`,
`sceKernelWaitSema`"* — never reaches a delivering safe point, "so its exception
handler never runs and whatever raised the signal is waiting on it". Every thread
in this capture is inside exactly one of those three calls.

This is **not established**, and the two things that would settle it are both
below the log's threshold in these runs:

* whether the title ever called `sceKernelInstallExceptionHandler` at all. If it
  did not, `hle_raise_exception` takes its `debug!("no handler installed for this
  signal")` path and returns `SCE_OK` — a silent no-op that the raiser then waits
  on forever. If it did, delivery should have been attempted.
* whether `deliver_pending` hit the `GuestCallError::Unsupported` (direct-gateway)
  arm. Its `GATEWAY_STALL_WARNING` `warn!` never fired — but it *cannot* fire once
  every thread is parked, because it only runs at the end of a dispatch and there
  are no more dispatches.

### What the next hardware run must capture

1. `RAEEN_STALL_DUMP=1` is now sufficient for the thread table. Add
   `RUST_LOG=debug` for `raeen_hle::libkernel`, `raeen_hle::exception`,
   `raeen_hle::kernel_semaphore` and `raeen_hle::pthread_cond` — every fact needed
   below is currently a `debug!`.
2. From those debug lines: the **handle** each of `t2…t15` is waiting on, the
   handle's `initCount`/`maxCount` from `sceKernelCreateSema`, and whether
   `sceKernelSignalSema` is ever called on that handle. One semaphore shared by all
   14 versus 14 distinct ones distinguishes "job queue never fed" from "handshake
   with a thread that does not exist".
3. Whether `sceKernelInstallExceptionHandler` is called, with which `signum`, and
   whether `hle_raise_exception` logged `no handler installed for this signal`.
   `t15`'s raise is the one event in the whole capture that is not a thread going
   idle.
4. An A/B with `RAEEN_DISABLE_DIRECT_HLE=1`. `exception.rs` names this as the
   escape hatch that routes every import through the VEH path where signal
   delivery works. If the stall clears under it, the blocker is exception delivery
   to direct-gateway-parked threads and the fix belongs there. If it does not, the
   raise is a red herring and item 2 is the whole story.
5. Guest RIPs for `t15` and `t1` decoded with `--dump-vaddr`, to name the guest
   call sites rather than the HLE functions.

---

## Also worth fixing, unrelated to the park

`argv[0]` is still a raw host path (`crates/raeen-gui/src/main.rs` passes
`&[path.as_str()]`), which is why the title's own banner prints
`Arg 0 = E:\PS5\PPSA13580-app\eboot.bin`. Recorded in
`docs/blasphemous-next-blockers.md`; not touched here, and not a plausible cause of
this stall (KytyPS5 passes the bare string `"KytyEmu"` and the title runs).
