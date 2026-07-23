# Astro Bot residual worker faults — root-cause investigation

Read-only multi-agent investigation (5 agents), 2026-07-23. Verdict: the two
suspected commits are **not** the cause; the crash is a **seed** (guest object
lifecycle bug) amplified by a **lock-release cascade**. The seed's exact nature
is undecided on current logs — one more instrumented run decides it.

## Cleared (strong, unanimous evidence)
- **Guest-GIL (`495e337`) is NOT the cause.** `RAEEN_SINGLE_THREAD_GUEST` is set
  nowhere in-tree → the GIL is inert by default; logs show genuinely concurrent
  native execution. Even if enabled it yields only at HLE boundaries, so it
  can't preempt an alloc+field-write constructor that makes no HLE call.
- **rwlock reader-count is NOT corroborated at fault time.** Every captured
  fault has `rwlock_read_holds=0`, `rwlock_writers=0` — only plain mutexes held.
- **Not memory-ordering, not mutex mutual-exclusion.** x86-64 TSO + DashMap shard
  fences give producer-writes-before-unlock → consumer-reads-after-lock. Mutex
  exclusion demonstrably *worked* (the waiter was correctly blocked).
- Premise correction: **`ce11844` is a gamepad/XInput commit**, not the rwlock
  change. Both suspects collapse to one commit, and it's cleared.

## H2 — AMPLIFIER (confirmed active, has a safe fix)
`release_locks_owned_by` (`crates/raeen-runtime/src/thread.rs:639`, unconditional
on any `Err`; `crates/raeen-kernel/src/lib.rs:783-822` zeroes `owner`/`writer`)
force-unlocks a **faulted** holder mid-critical-section, handing a waiter
half-built/abandoned state. Reconstructed chain: thread 21 (ACB submit worker)
holds mutex `0x300944e00` ~30s with main blocked; thread 21 faults;
`release_locks_owned_by(21)` force-releases; **2.1 ms later** main acquires it,
walks the wreckage, reads `0xffffffff00000000`, dies → teardown. This is an
intentional deadlock-vs-corruption tradeoff, but it turns one seed fault into
the fatal cascade.

**Fix A (safe, high-value, cascade-killing regardless of the seed):** instead of
silently force-releasing a faulted holder's lock, **poison** it so a waiter's
acquire returns a defined `EOWNERDEAD`-style error at the acquire site rather
than dereferencing corrupted state. Collapses N cascade faults into one
localizable error and gives a named seed marker.

## H1 — SEED (undecided: the real bug, needs one instrumented run)
A worker natively reads a shared object field / list-`next` that is null or
recycled. Fault #1 (`mov rax,[r14+0xe8]`=0 → deref `[0x0+0x10]`) precedes any
lock release. The **torn** values (`0xffffffff00000000` = only low dword zeroed;
`0x7ff877ca7ff877ca` = one dword duplicated, non-canonical) are physically
impossible from a reader observing a valid aligned-64-bit atomic publish on x86
TSO → this points at **use-after-free / recycled / wrong-typed memory or an HLE
stub that under-populates a struct**, NOT a clean lock race. Faults #1/#2 are
pure-NULL (maybe not-yet-written / HLE under-population) while #3/#4 are torn
(maybe UAF) → possibly **two distinct defects**.

Faulting sites: `eboot+0xe03f1a` (voice/audio node`+0xe8`), `eboot+0xe47a43`
(list `+0x28`), `eboot+0x33f335` (obj`+0x70`), libc.prx SIMD strlen/memchr.

## Decisive next experiment (cheap, definitive)
1. **`RAEEN_SINGLE_THREAD_GUEST=1`, N≥20 runs.** Seed faults *vanish* → genuine
   guest race (exclusion/wrong-lock hole). Seed faults *persist* → lifecycle /
   UAF / HLE-completeness bug, definitively **not** a lock race (kills H1-as-race
   and Fixes B/C for the seed). This one A/B is the cheapest decider.
2. Capture with `raeen_runtime::dispatch=debug` (the per-call HLE ring was
   filtered out of every prior log), **held-lock addresses** at the fault
   snapshot, an **allocator free-poison** distinct from the torn values, and a
   **write-trace on the three faulting field addresses** (writer-tid+value) — to
   disambiguate use-after-free vs never-written vs torn-partial-write.
3. Baseline fault-per-run probability over N≥20; apply Fix A; expect the cascade
   (fault ~2ms after a force-release) to disappear and ≤1 fault/run.

## H3 — latent real defect (fix regardless; not this crash)
rwlock **write→read downgrade** unlock releases the write hold first:
`pthread_sync.rs:654` admits a read to the current writer (thread can hold
write+read); `hle_rwlock_unlock` at `:757` releases write and `return`s before
consulting read holds → `wrlock; rdlock; unlock` clears `writer=0` while
`readers≥1`. Divergence from KytyPS5 (`pthread.cpp:2500`, LIFO
`RwlockRemoveReader` first) and shadPS4 (`std::shared_mutex` never grants both
modes). Also no writer-preference / `waiting_writers` counter (`:708`) →
reader-barging widens torn windows. Latent — Astro Bot never rdlocks a rwlock it
write-holds in the captured runs.

## Status
Investigation complete; all follow-ups need a build + repeated Astro runs
(blocked while the release build is pending). Land **Fix A** first (safe,
cascade-killing), then run experiment §1 to decide the seed.
