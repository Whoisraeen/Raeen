# A free-running host vblank source

Closes the design left unimplemented in `docs/silent-zero-frame-cluster.md` §5.
Read §3 of that document for the diagnosis; this one is only about the
implementation, the ownership rule, and how to A/B it.

**Status:** implemented, **default off**, unverified against retail. Nothing
here is a claim that any title now renders. It is a claim that a specific
deadlock is no longer structurally possible when the flag is on.

---

## 1. What was broken

Raeen advanced its vblank sequence from exactly two places, and both were guest
calls into `crates/raeen-hle/src/libsce_video_out.rs`:

* `hle_submit_flip` — "a completed flip implies a display refresh"
* `hle_wait_vblank`

That is sufficient for a **polling** frame loop. It deadlocks an
**event-driven** one. A guest thread that

1. opens video out,
2. registers a vblank event on an equeue (`sceVideoOutAddVblankEvent` — we
   return `SCE_OK` and store the registration), then
3. blocks in `sceKernelWaitEqueue` for the first vblank **before** submitting
   its first flip,

waits forever: the only two things that could fire that event are the two calls
it is now blocked from making. Zero CPU, zero output, zero errors — which is
exactly the signature of the silent zero-frame cluster (Until Dawn, Dragon Ball
Sparking Zero, and the two Unity titles).

This is the same **ack-but-never-deliver** class as `sceKernelRaiseException`
(fixed) and the `HLE_ERROR` sentinel leak (fixed). We acknowledged the
registration and never delivered the event.

KytyPS5 has no such hole: its window loop ticks vblank every displayed host
frame regardless of the guest —
`src/graphics/presentation/window/window.cpp:350-354` calls
`VideoOutBeginVblank()` → `VideoOutFlipWindow(0)` → `VideoOutEndVblank()`, and
those advance the pre-vblank / vblank counters and trigger the VideoOut events
of **every opened handle** (`videoOut.cpp:649-686`), paced against
`Config::GetVblankFrequency()` (`videoOut.cpp:402`).

### One caveat on severity

`sceKernelWaitEqueue` re-evaluates its `ready` predicate every 50 ms wait slice
(`kernel_equeue.rs`, `WAIT_SLICE`), so the missing *wake* was never the reason a
title hung. The missing **state change** was: nothing ever set `triggered` on
the registration, so re-checking it forever changed nothing. That distinction is
why the tests below assert the state transition as the primary contract and the
wake as promptness.

---

## 2. The seam, and why it needs no `HleContext`

`HleContext` (`crates/raeen-hle/src/lib.rs`) is a borrowed struct of `&dyn`
references with a lifetime, so no host thread can own one. Both
`trigger_vblank_events` and `kernel_equeue::wake_equeue` took one. That is why
this was deferred as "a refactor, not a patch".

The refactor turned out to be small, because a vblank delivery touches far less
than an `HleContext` carries. It needs exactly two things:

| Needs | Does **not** need |
|---|---|
| `OrbisKernel::kernel_equeue_events` (the registrations) | `mem` — never reads or writes guest memory |
| `WaitSubsystem::wake` (promptness) | `alloc`, `gpu`, `guest_calls`, `guest_threads`, `caller_*` |

`OrbisKernel` already implements `WaitSubsystem`, that trait is already
`Send + Sync`, and the runtime already holds the kernel in an `Arc`. So the seam
is just those two arguments lifted out of the context:

* `crates/raeen-hle/src/kernel_equeue.rs` — `wake_equeue_via(waker: &dyn
  WaitSubsystem, eq, guest_thread, reason)`; `wake_equeue` now delegates to it.
* `crates/raeen-hle/src/libsce_video_out.rs` — `trigger_vblank_events_via(kernel,
  waker, guest_thread, count)`; `trigger_vblank_events(ctx, count)` now
  delegates to it, and `host_vblank_refresh(kernel, waker)` is the
  advance-then-deliver pair.
* `crates/raeen-hle/src/host_vblank.rs` — the thread, the env gate, and the
  ownership flag.

Why this is sound rather than merely convenient:

* **No new sharing.** `kernel_equeue_events` is a `DashMap` and
  `video_out_vblank_count` an `AtomicU64`; both were already written concurrently
  by several guest threads. The host thread is one more writer of the same shape.
* **No new locking order.** `OrbisKernel::wake` takes one lock — the
  `event_flag_signal` mutex — and notifies. The host thread holds nothing when it
  calls it.
* **`guest_thread` is diagnostics only.** `OrbisKernel::wake` uses the `WaitKey`
  solely for `diagnostics.record` and then does a blanket `notify_all`. The host
  thread passes `0`: no guest thread caused this wake, and the label says so.
* **The kernel cannot be resurrected or outlived.** The thread holds a
  `Weak<OrbisKernel>`, never an `Arc`. A failed upgrade is its cue that the guest
  process is gone, which makes "wake into a torn-down kernel" unrepresentable
  rather than merely unlikely.
* **No guest-visible ABI moved.** Every entry point, error code, event ident,
  and `data` encoding is unchanged.

---

## 3. The double-tick ownership rule

**While the host source is running it is the sole advancer of
`video_out_vblank_count`.** The two guest-driven advances stand down; they check
`host_vblank::owns_sequence()` first.

Chosen this way, and not "whichever fires first", because:

* Both advancing would let the sequence a title uses for frame timing run *ahead*
  of the display clock it is supposed to be measuring — one refresh per real edge
  plus one per flip. Any timestamp-fence logic keyed to it drifts, and the drift
  is load-dependent, so it would not reproduce.
* The host source and `sceVideoOutWaitVblank` share **one clock**: the same
  process-wide epoch and period (`vblank_epoch()`, `vblank_period()`,
  `wait_next_host_vblank_edge`). Edges are absolute (`epoch + n·period`), so a
  guest parked in `WaitVblank` resumes on the very edge the host thread just
  delivered — the sequence it observes is already correct for that refresh, and a
  second increment would be counting the same refresh twice.

Precisely what is and is not conditional:

| Site | With host source | Without (default) |
|---|---|---|
| `hle_wait_vblank` pacing wait | still waits — the guest asked for it | still waits |
| `hle_wait_vblank` sequence advance + vblank events | **skipped** (host owns it) | advances + delivers |
| `hle_submit_flip` flip count, buffer routing, **flip** events | unchanged | unchanged |
| `hle_submit_flip` implied vblank advance + vblank events | **skipped** | advances + delivers |

A completed flip still fires its flip event either way: that a flip completed is
a fact, not an inference. Only "…therefore a display refresh happened" is
dropped.

Ownership is claimed **before** the thread is spawned and released **before** the
join, so there is no window in which both sources are live. When the source
cannot start — disabled, no display clock, thread spawn failed — the flag stays
`false` and the legacy behavior is bit-for-bit what it was.

---

## 4. The flag

`RAEEN_HOST_VBLANK`, resolved by
`host_vblank::configured_host_vblank_period(host, hz)`:

| `RAEEN_HOST_VBLANK` | Result |
|---|---|
| unset | **off** (the default) |
| `0`, `off`, `false`, `no` (any case, trimmed) | off |
| a rate in `24..=480` | on at that rate, overriding `RAEEN_VBLANK_HZ` |
| any other value (`1`, `on`, `true`, empty, …) | on at the configured display rate |

The last row inherits `RAEEN_VBLANK_HZ` through the **existing**
`libsce_video_out::vblank_period()` — deliberately not a second refresh setting.
A consequence worth knowing: `RAEEN_VBLANK_HZ=0` is the explicit unpaced
benchmark mode (`cargo xtask compat run --profile max-fps`), and with no display
clock there is no host source to run, so `RAEEN_HOST_VBLANK=1` yields *off*
there. Requesting the rate directly (`RAEEN_HOST_VBLANK=60`) is how a benchmark
run gets an unpaced guest **and** a real vblank clock.

Why default off: Minecraft renders ~13,400 frames today and ASTRO.BOT holds
`rendering`. An unverified pacing change must not be able to regress them.

**Cost when disabled:** one relaxed `AtomicBool` load and a not-taken branch at
each of the two guest advance sites. No thread, no clock read, no allocation, no
behavior change. Same shape as `frame_path::record`'s guard.

**Cost when enabled and nothing is registered:** one `DashMap` scan per period
and zero wakes — `trigger_vblank_events_via` only wakes queues that actually hold
a vblank or pre-vblank-start registration.

The harness (`xtask`) deliberately does **not** set this. Turning it on for every
compat run would make it the default in practice.

---

## 5. Lifetime and teardown

Started in `crates/raeen-gui/src/main.rs`, immediately before
`execute_process_shared` — the process that owns the guest's
`Arc<OrbisKernel>` — and bound to a local whose `Drop` stops it. That is also the
`--run-eboot` path the compatibility harness uses.

`HostVblankSource::stop()` (idempotent; also called by `Drop`):

1. release ownership of the sequence, so the guest-driven advances resume before
   any last host refresh could land;
2. set the stop flag;
3. `join()` the thread — a real handshake, not a timed wait, so no thread
   outlives the call.

The thread checks the stop flag on **both** sides of every wait, so it cannot
deliver a refresh after `stop()` returned. Worst-case join latency is one period
(≤ 41 ms at the slowest selectable rate, 16.7 ms at 60 Hz): it is parked on an
absolute display edge and checks the flag the moment it wakes. Independently, the
`Weak<OrbisKernel>` ends the loop if the kernel is dropped, so a path that exits
without running the destructor still cannot leave a thread waking a dead kernel.

Thread name: `raeen-host-vblank` (visible in `RAEEN_STALL_DUMP`).

---

## 6. Deviation from the §5 sketch, and what is still missing

* **Started at process bootstrap, not on the first `sceVideoOutOpen`.** Lazy
  start would need an owned kernel handle inside `hle_open`, which only has a
  borrowed `&OrbisKernel`. Bootstrap start is also the more faithful model: a
  real display's vblank counter free-runs from display init, and KytyPS5's window
  loop likewise ticks before the guest opens anything. The observable difference
  is that the guest's first `sceVideoOutGetVblankStatus` reports a non-zero
  absolute count — which is what hardware does, and titles use deltas.
* **"Every opened handle" is "every registered vblank event."** Raeen's
  registrations are keyed by `(equeue, ident)` and `sceVideoOutOpen` only ever
  hands out handle 1, so KytyPS5's per-handle loop and our per-registration loop
  cover the same set. If multi-handle video out is ever implemented, this needs
  revisiting.
* **Pre-vblank-start and vblank still fire together**, with the same sequence
  number, from one tick. KytyPS5 splits them across `VblankBegin` / `VblankEnd`
  with independent counters (`videoOut.cpp:649-686`). Delivering both is what
  matters; the intra-frame ordering is collapsed. Pre-existing, unchanged here.
* **Not measured against a retail title.** See §7.

---

## 7. Tests

`cargo test -p raeen-hle --lib` — 601 (592 before this change).

All deterministic: no test sleeps to observe a tick, and no test asserts on
elapsed time. The wake is observable *without a second thread* because the seam
takes `&dyn WaitSubsystem` — tests pass a `RecordingWaker` that records wakes
instead of performing them.

`crates/raeen-hle/src/host_vblank.rs`:

| Test | Contract |
|---|---|
| `env_gate_is_off_by_default_and_inherits_the_display_rate` | every row of the §4 table, incl. `RAEEN_VBLANK_HZ=0` → off |
| `a_host_refresh_advances_the_sequence_and_wakes_a_registered_waiter` | the headline: **no guest call at all** advances the sequence, triggers the registration, preserves `udata`, encodes `ident \| sequence << 16`, and wakes that queue once with `guest_thread == 0` |
| `a_host_refresh_reaches_every_registered_queue_and_both_classes` | every queue, vblank + pre-vblank-start, each woken once; a non-VideoOut user event is left alone |
| `a_host_refresh_with_no_registration_wakes_nobody` | an enabled source costs a non-registering title nothing but the scan |
| `stopping_releases_ownership_and_joins_the_thread` | ownership claimed before the first tick, released on stop, idempotent stop, `Drop` does not double-join, and the source held only a `Weak` (`Arc::strong_count == 1`) |
| `start_from_env_is_a_no_op_unless_the_flag_is_set` | end to end through the real env read: unset means no source and no ownership |
| `a_zero_period_cannot_start_a_source` | a source that never started must not claim ownership |

`crates/raeen-hle/src/libsce_video_out.rs`:

| Test | Contract |
|---|---|
| `a_running_host_source_is_the_only_advancer_of_the_vblank_sequence` | one host refresh, then a flip **and** a `WaitVblank`, leaves the count at 1; the flip still completes and still fires its flip event; the vblank event carries the host sequence |
| `with_no_host_source_a_flip_still_implies_a_refresh` | the disabled path is unchanged — the Minecraft / ASTRO.BOT regression guard |

The four pre-existing tests that depend on the guest-driven advance
(`wait_vblank_advances_a_separate_frame_sequence`,
`vblank_events_fire_and_get_event_id_classifies`,
`delete_vblank_event_undoes_add_vblank_event`,
`pre_vblank_start_events_register_fire_and_delete`) now hold an
`OwnershipGuard::released()`. The ownership flag is process-global, matching the
process-global vblank clock it governs, so a mutex serializes every test that
reads or writes it and the guard restores the default even on a failing
assertion.

Unchanged baselines: raeen-kernel 67, raeen-runtime 98 lib + 62 execute,
raeen-gpu 310, raeen-gui 204. `cargo clippy --workspace --all-targets -D
warnings` clean.

---

## 8. The A/B that decides whether this ships on by default

Two titles, two runs each. Until Dawn is the candidate beneficiary (UE5, presents
zero frames, and — unlike the other three in the cluster — burns a full core, so
it is the *spinning* member and may not be an equeue-park case at all). Minecraft
is the regression guard.

```powershell
cargo build --release -p raeen-gui

# --- Until Dawn: does a host vblank source unblock it? ---
$env:RAEEN_FRAME_PATH = "5"
$env:RAEEN_HOST_VBLANK = ""          # OFF (baseline)
.\target\release\raeen.exe --run-eboot "<UNTIL-DAWN>\eboot.bin" 2>&1 |
  Tee-Object -FilePath untildawn-vblank-off.log
# Ctrl-C after ~60 s

$env:RAEEN_HOST_VBLANK = "1"         # ON
.\target\release\raeen.exe --run-eboot "<UNTIL-DAWN>\eboot.bin" 2>&1 |
  Tee-Object -FilePath untildawn-vblank-on.log

# --- Minecraft: the regression guard (must not change) ---
$env:RAEEN_HOST_VBLANK = ""
.\target\release\raeen.exe --run-eboot "<MINECRAFT>\eboot.bin" 2>&1 |
  Tee-Object -FilePath minecraft-vblank-off.log

$env:RAEEN_HOST_VBLANK = "1"
.\target\release\raeen.exe --run-eboot "<MINECRAFT>\eboot.bin" 2>&1 |
  Tee-Object -FilePath minecraft-vblank-on.log
```

Read the `frame path: reached=` line in each. Decision rule:

* **Until Dawn's furthest rung advances with the flag on** → the §3 diagnosis is
  confirmed for it and this belongs on by default.
* **Until Dawn is unchanged** → its blocker is upstream of the vblank event, and
  the flag stays off; the fix is still correct (a title that waits on vblank
  before its first flip deadlocks without it) but it is not that title's first
  blocker. Add `RAEEN_STALL_DUMP=1 RAEEN_TIME_HLE=1 RAEEN_TRACE_EINVAL=1` and
  follow `docs/silent-zero-frame-cluster.md` §4 to find the real one.
* **Minecraft's `flips_submitted` or FPS moves in either direction** → do not
  flip the default; the ownership rule interacts with its flip pacing and needs
  another look before anything else.

Dragon Ball Sparking Zero is the better second candidate (same UE5 fingerprint,
hard-blocked at 8.6 s CPU across 180 s — the shape this fix predicts) if Until
Dawn comes back unchanged.

---

## Attribution

`reference/kytyps5` (MIT, © InoriRus / Nmzik). The structure of §1's tick —
advance the counters, then trigger the VideoOut events of every open handle,
paced by the configured refresh — is behaviorally re-implemented from
`src/graphics/presentation/window/window.cpp:350-354` and
`src/graphics/presentation/videoOut.cpp:402,649-686`. No code was copied: the
Rust module, the equeue-wake seam, the ownership flag, the `Weak`-based teardown,
and the tests are original. Recorded in `THIRD_PARTY_NOTICES.md` and
`docs/reference-port-ledger.md`.
