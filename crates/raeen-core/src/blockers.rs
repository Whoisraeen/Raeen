//! The blocker table: every "we refused / could not / did not implement"
//! event, interned once and counted forever.
//!
//! # Why this is not just a log call
//!
//! Every refusal in this tree is currently a `tracing` event, most of them at
//! `debug!`. The level filter is installed as a **global** layer
//! (`logging::init_with_file` puts one `reload::Layer` above stderr, the file
//! writer *and* the console ring), so a `debug!` refusal at the default `info`
//! level never reaches the log, never reaches `logs/crashes/*.report.md`, and
//! never enters the F10 console ring — no console-side level or search can
//! recover it, because it was dropped before the ring saw it. Raising the
//! global level to recover one refusal instead buries it: a Minecraft run at
//! `debug` reaches gigabytes, which is what the 64 MiB cap in
//! [`crate::logging`] exists to survive.
//!
//! [`record`] is a plain function call, not a tracing event, so no filter can
//! drop it. Each *distinct* refusal also emits exactly one `warn!` — visible at
//! the default level, bounded at [`SLOT_CAP`] lines for a whole session — while
//! repeats become a counter. That is the difference between "30,734 identical
//! shader-fetch failures flooded the log" and "shader-fetch failed, ×30734".
//!
//! # Cost
//!
//! Idle: nothing. A healthy run never calls in, so the table is never
//! allocated ([`std::sync::OnceLock`] stays unset) and the only footprint is
//! [`SLOT_CAP`] × 64 B of zeroed BSS the OS commits lazily.
//!
//! Active: the **first** occurrence of a distinct key takes one uncontended
//! mutex, one hash, and one allocation. Every repeat is [`bump`] — a single
//! relaxed `fetch_add` on a counter that owns its cache line, with no lock, no
//! hash and no allocation. Hot sites are expected to intern once into a
//! [`Slot`] (cache it in a `OnceLock`) and call [`bump`] thereafter, which is
//! why the two halves are separate entry points.
//!
//! # Bounded by construction
//!
//! Caps are **per category** ([`PER_CATEGORY_CAP`]), never global: a title with
//! 271 distinct missing NIDs cannot starve the GPU story out of its own crash
//! report. Distinct keys past a category's cap are counted in
//! [`Totals::dropped_distinct`] and reported, so a truncated table always says
//! it was truncated.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

use serde::{Deserialize, Serialize};

/// What kind of refusal a blocker is.
///
/// Declaration order is **severity order**, most-explanatory first — see
/// [`BlockerCategory::rank`]. The slugs are a log/report/JSON contract, exactly
/// like `frame_path::Stage::label`; treat a rename as a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlockerCategory {
    /// The guest faulted, trapped, or aborted itself.
    GuestFault,
    /// A `DT_NEEDED` module could not be found or loaded at all.
    MissingLibrary,
    /// The guest *called* an import that resolved to nothing.
    UnresolvedNid,
    /// Forward progress stopped: no new frame, no new phase, threads parked or
    /// spinning.
    Stall,
    /// A shader could not be translated, so its draw could not be issued.
    ShaderRefused,
    /// A resource descriptor (T#/V#/S#) could not be resolved to real memory
    /// and a placeholder was substituted.
    DescriptorUnresolved,
    /// A draw or dispatch was recorded but deliberately not issued.
    DrawDropped,
    /// The graphics device or its validation layers reported an error.
    GpuError,
    /// An implemented shim that deliberately does not do the real work.
    UnimplementedStub,
    /// An HLE call returned an Orbis error code to the guest.
    OrbisError,
    /// A guest path could not be resolved to a host file.
    VfsMiss,
    /// The host — not the guest — failed: an OS call, an allocation, a device.
    HostError,
}

impl BlockerCategory {
    /// Every category, in severity order.
    pub const ALL: [BlockerCategory; 12] = [
        BlockerCategory::GuestFault,
        BlockerCategory::MissingLibrary,
        BlockerCategory::UnresolvedNid,
        BlockerCategory::Stall,
        BlockerCategory::ShaderRefused,
        BlockerCategory::DescriptorUnresolved,
        BlockerCategory::DrawDropped,
        BlockerCategory::GpuError,
        BlockerCategory::UnimplementedStub,
        BlockerCategory::OrbisError,
        BlockerCategory::VfsMiss,
        BlockerCategory::HostError,
    ];

    /// Stable kebab-case slug used in logs, reports and JSON.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            BlockerCategory::GuestFault => "guest-fault",
            BlockerCategory::MissingLibrary => "missing-library",
            BlockerCategory::UnresolvedNid => "unresolved-nid",
            BlockerCategory::Stall => "stall",
            BlockerCategory::ShaderRefused => "shader-refused",
            BlockerCategory::DescriptorUnresolved => "descriptor-unresolved",
            BlockerCategory::DrawDropped => "draw-dropped",
            BlockerCategory::GpuError => "gpu-error",
            BlockerCategory::UnimplementedStub => "unimplemented-stub",
            BlockerCategory::OrbisError => "orbis-error",
            BlockerCategory::VfsMiss => "vfs-miss",
            BlockerCategory::HostError => "host-error",
        }
    }

    /// Inverse of [`BlockerCategory::slug`].
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        BlockerCategory::ALL.into_iter().find(|c| c.slug() == slug)
    }

    /// Severity rank: **lower is worse**, 0 being the most explanatory thing a
    /// report can lead with. Used by [`worst`] to pick the headline when a
    /// session recorded several kinds of refusal.
    ///
    /// The order is a judgement about what explains a failed session, not about
    /// what is most numerous: a single `GuestFault` explains a dead title,
    /// while ten thousand `OrbisError`s are usually a title being told "no" and
    /// coping fine.
    #[must_use]
    pub const fn rank(self) -> u8 {
        self.index() as u8
    }

    const fn index(self) -> usize {
        match self {
            BlockerCategory::GuestFault => 0,
            BlockerCategory::MissingLibrary => 1,
            BlockerCategory::UnresolvedNid => 2,
            BlockerCategory::Stall => 3,
            BlockerCategory::ShaderRefused => 4,
            BlockerCategory::DescriptorUnresolved => 5,
            BlockerCategory::DrawDropped => 6,
            BlockerCategory::GpuError => 7,
            BlockerCategory::UnimplementedStub => 8,
            BlockerCategory::OrbisError => 9,
            BlockerCategory::VfsMiss => 10,
            BlockerCategory::HostError => 11,
        }
    }
}

impl std::fmt::Display for BlockerCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.slug())
    }
}

/// How many *distinct* keys one category may retain.
pub const PER_CATEGORY_CAP: usize = 32;

/// Total retained keys across all categories.
pub const SLOT_CAP: usize = PER_CATEGORY_CAP * BlockerCategory::ALL.len();

/// A handle to one interned blocker's counter.
///
/// Cache this at a hot call site (a `OnceLock<Option<Slot>>` next to the site)
/// and call [`bump`] on it, so the repeat path never hashes or locks.
/// One-based; the zero value is never handed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot(u32);

/// One distinct refusal, as a report or the console sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blocker {
    pub category: BlockerCategory,
    /// The stable identity of this refusal — a function name, a library name,
    /// a shader reason. Must not carry a changing value (an address, a count),
    /// or every occurrence interns a new entry and the cap is spent on one site.
    pub key: String,
    /// An optional numeric subject that is part of the identity: a NID, an
    /// address, a format code. Zero when the key alone identifies the refusal.
    pub subject: u64,
    /// Free-form context captured at the first occurrence only.
    pub detail: String,
    /// Milliseconds from the process epoch to the first occurrence. `None`
    /// means the refusal happened before `frame_path::mark_origin` — never a
    /// substituted zero.
    pub first_ms: Option<u64>,
    /// Occurrences, including the first.
    pub count: u64,
    /// Interning order, from 0. The tiebreak for "which came first" when two
    /// blockers share a millisecond, and stable across a session.
    pub seq: u64,
}

impl Blocker {
    /// The one-line form used in reports, the console, and the IPC digest.
    #[must_use]
    pub fn line(&self) -> String {
        let mut out = String::with_capacity(96);
        out.push_str(self.category.slug());
        out.push(' ');
        out.push_str(&self.key);
        if self.subject != 0 {
            out.push_str(&format!(" ({:#x})", self.subject));
        }
        if !self.detail.is_empty() {
            out.push_str(" — ");
            out.push_str(&self.detail);
        }
        if self.count > 1 {
            out.push_str(&format!(" ×{}", self.count));
        }
        match self.first_ms {
            Some(ms) => out.push_str(&format!(" [first at +{ms} ms]")),
            None => out.push_str(" [first before timing started]"),
        }
        out
    }
}

/// Aggregate counts over the whole table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    /// Distinct keys retained.
    pub distinct: usize,
    /// Occurrences across every retained key.
    pub total_events: u64,
    /// Distinct keys refused because their category was full.
    pub dropped_distinct: u64,
}

/// An `AtomicU64` owning its own cache line, so a bump on one blocker never
/// invalidates another's line on a different core.
#[repr(align(64))]
struct PaddedU64(AtomicU64);

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: PaddedU64 = PaddedU64(AtomicU64::new(0));
static COUNTS: [PaddedU64; SLOT_CAP] = [ZERO; SLOT_CAP];

/// Metadata for one interned key. The counter itself lives in [`COUNTS`], out
/// from under the lock.
#[derive(Debug, Clone)]
struct Meta {
    category: BlockerCategory,
    key: String,
    subject: u64,
    detail: String,
    first_ms: Option<u64>,
    seq: u64,
}

#[derive(Default)]
struct Table {
    by_key: HashMap<(BlockerCategory, String, u64), Slot>,
    meta: Vec<Meta>,
    per_category_used: [usize; 12],
    per_category_dropped: [u64; 12],
    next_seq: u64,
}

static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();

/// Lock the table, tolerating poisoning.
///
/// Never `.unwrap()`: [`record`] is called from inside the vectored exception
/// handler, and a poisoned-mutex panic *there* is the documented
/// `0xC0000409`-with-no-report failure this whole module exists to prevent.
fn table() -> std::sync::MutexGuard<'static, Table> {
    TABLE
        .get_or_init(|| Mutex::new(Table::default()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Intern a distinct refusal, returning its counter slot.
///
/// `detail` is a closure so a hot site pays no formatting on the repeat path —
/// it is only called when the key is genuinely new. Returns `None` when the
/// category is full, in which case the event is counted as dropped and the
/// caller simply has nothing to bump.
///
/// Interning emits exactly one `warn!`, which is what makes a refusal visible
/// at the default log level without a per-occurrence flood.
pub fn intern(
    category: BlockerCategory,
    key: impl Into<Cow<'static, str>>,
    subject: u64,
    detail: impl FnOnce() -> String,
) -> Option<Slot> {
    let key = key.into();
    let mut table = table();

    let lookup = (category, key.as_ref().to_string(), subject);
    if let Some(slot) = table.by_key.get(&lookup) {
        return Some(*slot);
    }

    let index = category.index();
    if table.per_category_used[index] >= PER_CATEGORY_CAP || table.meta.len() >= SLOT_CAP {
        // The array bound is what actually protects `COUNTS`; enforce it rather
        // than trust the per-category caps to imply it.
        table.per_category_dropped[index] = table.per_category_dropped[index].saturating_add(1);
        return None;
    }

    let detail = detail();
    let first_ms = crate::frame_path::since_origin_ms();
    let seq = table.next_seq;
    table.next_seq += 1;
    table.per_category_used[index] += 1;
    table.meta.push(Meta {
        category,
        key: lookup.1.clone(),
        subject,
        detail: detail.clone(),
        first_ms,
        seq,
    });
    let slot = Slot(table.meta.len() as u32);
    table.by_key.insert(lookup, slot);
    drop(table);

    tracing::warn!(
        target: "raeen::blocker",
        category = %category.slug(),
        key = %key,
        subject = format_args!("{subject:#x}"),
        detail = %detail,
        "blocker"
    );
    Some(slot)
}

/// Count one more occurrence of an interned blocker.
///
/// One relaxed `fetch_add`. No lock, no hash, no allocation.
#[inline]
pub fn bump(slot: Slot) {
    bump_n(slot, 1);
}

/// Count `n` more occurrences at once — for sites that batch (a submission's
/// worth of dropped draws, say).
#[inline]
pub fn bump_n(slot: Slot, n: u64) {
    if let Some(counter) = COUNTS.get(slot.0 as usize - 1) {
        counter.0.fetch_add(n, Ordering::Relaxed);
    }
}

/// Intern-then-count, for cold sites that will not cache a [`Slot`].
///
/// Prefer [`intern`] + [`bump`] anywhere the site can run thousands of times.
pub fn record(
    category: BlockerCategory,
    key: impl Into<Cow<'static, str>>,
    subject: u64,
    detail: impl FnOnce() -> String,
) -> Option<Slot> {
    let slot = intern(category, key, subject, detail)?;
    bump(slot);
    Some(slot)
}

/// Every retained blocker, in interning order.
#[must_use]
pub fn snapshot() -> Vec<Blocker> {
    let table = table();
    table
        .meta
        .iter()
        .enumerate()
        .map(|(index, meta)| Blocker {
            category: meta.category,
            key: meta.key.clone(),
            subject: meta.subject,
            detail: meta.detail.clone(),
            first_ms: meta.first_ms,
            count: COUNTS[index].0.load(Ordering::Relaxed),
            seq: meta.seq,
        })
        .collect()
}

/// Every retained blocker, most explanatory first: by [`BlockerCategory::rank`],
/// then by interning order.
#[must_use]
pub fn ranked() -> Vec<Blocker> {
    let mut all = snapshot();
    all.sort_by_key(|b| (b.category.rank(), b.seq));
    all
}

/// The blocker recorded **first**, chronologically.
///
/// A factual "what went wrong first", not a judgement about what matters most —
/// see [`worst`] for that. The two are reported as separate lines precisely
/// because they are often different, and collapsing them would mean picking one
/// and calling it the other.
#[must_use]
pub fn first() -> Option<Blocker> {
    snapshot().into_iter().min_by_key(|b| b.seq)
}

/// The most explanatory blocker: lowest [`BlockerCategory::rank`], earliest
/// within that category.
#[must_use]
pub fn worst() -> Option<Blocker> {
    ranked().into_iter().next()
}

/// Aggregate counts.
#[must_use]
pub fn totals() -> Totals {
    let table = table();
    Totals {
        distinct: table.meta.len(),
        total_events: (0..table.meta.len())
            .map(|index| COUNTS[index].0.load(Ordering::Relaxed))
            .sum(),
        dropped_distinct: table.per_category_dropped.iter().sum(),
    }
}

/// Per-category distinct-key drops, for reporting a truncated table honestly.
#[must_use]
pub fn dropped_by_category() -> Vec<(BlockerCategory, u64)> {
    let table = table();
    BlockerCategory::ALL
        .into_iter()
        .map(|category| (category, table.per_category_dropped[category.index()]))
        .filter(|(_, dropped)| *dropped > 0)
        .collect()
}

/// A bounded plain-text digest, most explanatory first.
///
/// Truncation is announced in the text itself, so a reader who reaches the end
/// of a digest knows whether they reached the end of the *table*. Used for the
/// child→Shell status channel, which has a hard byte budget.
#[must_use]
pub fn digest(max_entries: usize, max_bytes: usize) -> String {
    let all = ranked();
    let mut out = String::new();
    let mut shown = 0usize;
    for blocker in all.iter().take(max_entries) {
        let line = blocker.line();
        if out.len() + line.len() + 1 > max_bytes {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&line);
        shown += 1;
    }
    let hidden = all.len().saturating_sub(shown);
    let dropped: u64 = dropped_by_category().iter().map(|(_, n)| n).sum();
    if hidden > 0 || dropped > 0 {
        let note = format!("\n… {hidden} more retained, {dropped} distinct dropped at cap");
        if out.len() + note.len() <= max_bytes {
            out.push_str(&note);
        }
    }
    out
}

/// Empty the table. **Tests only** — exposed unconditionally because the crates
/// that record (`raeen-gpu`, `raeen-hle`, `raeen-runtime`) need it from their
/// own test binaries, where a `#[cfg(test)]` in this crate would not apply.
#[doc(hidden)]
pub fn reset_for_tests() {
    let mut table = table();
    // Zero exactly the slots this table interned; `COUNTS` is a fixed-size
    // array, so iterate it and stop at the used length rather than indexing by
    // a range over `meta`.
    for slot in COUNTS.iter().take(table.meta.len()) {
        slot.0.store(0, Ordering::Relaxed);
    }
    *table = Table::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is process-global, so these cases must not run concurrently.
    /// One `#[test]` drives them in order.
    #[test]
    fn the_table_interns_once_counts_repeats_and_bounds_itself() {
        reset_for_tests();

        // --- interning is once per distinct (category, key, subject) ---
        let a = record(BlockerCategory::UnresolvedNid, "sceFooBar", 0xdead, || {
            "library=libSceFoo caller=eboot.bin".to_string()
        })
        .expect("first record interns");
        let b = record(BlockerCategory::UnresolvedNid, "sceFooBar", 0xdead, || {
            panic!("detail must NOT be recomputed for a repeat")
        })
        .expect("second record finds the same slot");
        assert_eq!(a, b, "same identity must reuse the slot");

        // A different subject is a different blocker; a different category too.
        record(
            BlockerCategory::UnresolvedNid,
            "sceFooBar",
            0xbeef,
            String::new,
        )
        .expect("distinct subject");
        record(BlockerCategory::VfsMiss, "sceFooBar", 0xdead, String::new).expect("distinct cat");

        let snap = snapshot();
        assert_eq!(snap.len(), 3, "three distinct keys: {snap:#?}");
        let first_entry = snap.iter().find(|b| b.subject == 0xdead).unwrap();
        assert_eq!(first_entry.count, 2, "repeats are counted, not re-interned");
        assert_eq!(first_entry.detail, "library=libSceFoo caller=eboot.bin");
        assert_eq!(totals().total_events, 4);
        assert_eq!(totals().distinct, 3);

        // --- bump is the lock-free repeat path ---
        bump_n(a, 10);
        assert_eq!(
            snapshot()
                .iter()
                .find(|b| b.subject == 0xdead)
                .unwrap()
                .count,
            12
        );

        // --- first vs worst are genuinely different questions ---
        // UnresolvedNid (rank 2) was recorded first; GuestFault (rank 0) later.
        record(BlockerCategory::GuestFault, "read of 0x0", 0x1234, || {
            "libc.prx+0x103c6".to_string()
        })
        .expect("a fault");
        assert_eq!(first().unwrap().category, BlockerCategory::UnresolvedNid);
        assert_eq!(worst().unwrap().category, BlockerCategory::GuestFault);
        assert_eq!(
            ranked().first().unwrap().category,
            BlockerCategory::GuestFault,
            "ranked leads with the most explanatory category"
        );

        // --- the cap is PER CATEGORY, so one noisy category cannot evict another ---
        reset_for_tests();
        for i in 0..(PER_CATEGORY_CAP + 8) {
            record(
                BlockerCategory::UnresolvedNid,
                format!("nid{i}"),
                i as u64,
                String::new,
            );
        }
        record(BlockerCategory::GpuError, "device lost", 0, String::new)
            .expect("a full UnresolvedNid category must not consume a GpuError slot");
        let counted = totals();
        assert_eq!(counted.distinct, PER_CATEGORY_CAP + 1);
        assert_eq!(counted.dropped_distinct, 8, "over-cap keys are counted");
        assert_eq!(
            dropped_by_category(),
            vec![(BlockerCategory::UnresolvedNid, 8)],
            "the drop is attributed to the category that overflowed"
        );

        // --- a truncated digest says it was truncated ---
        let bounded = digest(4, 4096);
        assert!(bounded.lines().count() <= 5, "bounded: {bounded}");
        assert!(
            bounded.contains("distinct dropped at cap"),
            "a truncated digest must admit it: {bounded}"
        );
        let tiny = digest(64, 120);
        assert!(tiny.len() <= 120, "digest must respect max_bytes: {tiny:?}");

        reset_for_tests();
        assert_eq!(totals(), Totals::default());
    }

    #[test]
    fn slugs_ranks_and_lines_are_a_stable_contract() {
        let slugs: Vec<&str> = BlockerCategory::ALL.iter().map(|c| c.slug()).collect();
        let mut sorted = slugs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), slugs.len(), "slugs must be unique");
        for (index, category) in BlockerCategory::ALL.into_iter().enumerate() {
            assert_eq!(BlockerCategory::from_slug(category.slug()), Some(category));
            assert_eq!(category.rank() as usize, index, "rank is declaration order");
        }
        assert_eq!(BlockerCategory::from_slug("not-a-category"), None);

        // The rendered line carries identity, context, count and timing — and
        // says so plainly when the timing is not knowable.
        let timed = Blocker {
            category: BlockerCategory::ShaderRefused,
            key: "storage_texture_dim_format".to_string(),
            subject: 0,
            detail: "mixed storage image dims".to_string(),
            first_ms: Some(1204),
            count: 30734,
            seq: 0,
        };
        assert_eq!(
            timed.line(),
            "shader-refused storage_texture_dim_format — mixed storage image dims ×30734 \
             [first at +1204 ms]"
        );
        let untimed = Blocker {
            first_ms: None,
            count: 1,
            subject: 0xdeadbeef,
            detail: String::new(),
            ..timed
        };
        let line = untimed.line();
        assert!(line.contains("(0xdeadbeef)"), "{line}");
        assert!(
            !line.contains('×'),
            "a single occurrence needs no count: {line}"
        );
        assert!(
            line.contains("[first before timing started]"),
            "an unknown time must never render as +0 ms: {line}"
        );
    }
}
