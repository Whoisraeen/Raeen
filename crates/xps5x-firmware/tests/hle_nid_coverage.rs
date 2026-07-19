//! Do the NIDs a **real** PS5 title asks for actually resolve against our HLE?
//!
//! A NID is a one-way hash of a function *name*, so an implementation only
//! resolves if it is registered under the exact spelling the title imports.
//! That makes "we implemented X" and "the title's import of X resolves" two
//! different facts, and the gap between them is invisible without a test like
//! this: `sceKernelGettimeofday` was implemented and working for weeks while
//! the measured title's own `libc.prx` died calling `gettimeofday` — a
//! different symbol, a different NID.
//!
//! These NIDs were recovered from the retail title by brute-forcing candidate
//! names against the encoded NID strings in its symbol table, so each one is a
//! symbol a real title genuinely requested.

use xps5x_firmware::ModuleRegistry;
use xps5x_firmware::dynlib::nid::{NidDatabase, encode_nid, nid_of};
use xps5x_hle::HleRegistry;

/// `(name, encoded NID as it appears in the title's symbol table)`.
const REAL_TITLE_IMPORTS: &[(&str, &str)] = &[
    // libc.prx blocked on this one during early init.
    ("gettimeofday", "n88vx3C5nW8"),
    // Recovered from the same title's import table.
    ("__stack_chk_guard", "f7uOxY9mM1U"),
    ("in6addr_any", "ZRAJo-A-ukc"),
    ("in6addr_loopback", "XCuA-GqjA-k"),
    ("__cxa_pure_virtual", "zr094EQ39Ww"),
    ("strcmp", "Ovb2dSJOAuE"),
];

/// The name -> NID hashing must keep producing the exact strings the retail
/// title carries. If this drifts, every resolution silently stops matching.
#[test]
fn recovered_names_still_hash_to_the_nids_the_real_title_carries() {
    for (name, encoded) in REAL_TITLE_IMPORTS {
        assert_eq!(
            &encode_nid(nid_of(name)),
            encoded,
            "name->NID hashing drifted for {name:?}"
        );
    }
}

/// The POSIX spelling and the Sony spelling are DIFFERENT symbols. This is why
/// `libScePosix` has to exist as its own set of registrations rather than being
/// assumed covered by `libkernel`.
#[test]
fn posix_and_sce_spellings_are_distinct_nids() {
    assert_ne!(nid_of("gettimeofday"), nid_of("sceKernelGettimeofday"));
    assert_ne!(nid_of("clock_gettime"), nid_of("sceKernelClockGettime"));
}

/// The NID `libc.prx` actually died on must now resolve to an HLE function.
///
/// This is the acceptance test for the `libScePosix` module: it asserts
/// resolution through the same path the linker uses (NID -> name -> is it
/// implemented?), not merely that a function was registered somewhere.
#[test]
fn libsce_posix_names_resolve_the_nids_the_real_title_asked_for() {
    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());

    for name in ["gettimeofday", "clock_gettime", "usleep", "getpid"] {
        let nid = nid_of(name);
        let resolved = db
            .resolve(nid)
            .unwrap_or_else(|| panic!("NID for {name:?} resolves to no registered HLE name"));
        let (library, function) = resolved
            .split_once("::")
            .expect("NidDatabase stores 'library::function'");
        assert!(
            hle.is_implemented(library, function),
            "{name:?} resolves to {library}::{function}, which is not implemented"
        );
    }
}

/// Building the NID database from the **real** HLE registry must give the same
/// answer every time, however the names arrive.
///
/// This is the end-to-end version of the determinism fix.
/// `HleRegistry::registered_names()` walks a `DashMap`, so its order is not
/// stable between runs — and with the old "first-inserted wins" the winner for
/// a name registered under two libraries flipped. Measured on the real title
/// before the fix: 10 of the 11 duplicated names resolved *both ways* across
/// two runs minutes apart, silently changing which implementation a guest
/// import dispatched to.
///
/// Shuffling here stands in for that instability deterministically, so the test
/// cannot pass by luck of the map's ordering.
#[test]
fn the_real_nid_database_is_identical_however_the_names_are_ordered() {
    let hle = HleRegistry::new();
    let names = hle.registered_names();
    assert!(
        names.len() > 100,
        "expected a substantial HLE surface, got {}",
        names.len()
    );

    let baseline = NidDatabase::from_hle_names(names.clone());

    // A few deterministic permutations: reversed, rotated, and sorted.
    let mut permutations: Vec<Vec<(String, String)>> = Vec::new();
    permutations.push(names.iter().cloned().rev().collect());
    let mut rotated = names.clone();
    rotated.rotate_left(names.len() / 3);
    permutations.push(rotated);
    let mut sorted = names.clone();
    sorted.sort();
    permutations.push(sorted);

    for (i, perm) in permutations.into_iter().enumerate() {
        let db = NidDatabase::from_hle_names(perm);
        assert_eq!(
            db.len(),
            baseline.len(),
            "permutation {i} changed the database size"
        );
        // Every name in the set must resolve to the same winner.
        for (_, function) in &names {
            let nid = nid_of(function);
            assert_eq!(
                db.resolve(nid),
                baseline.resolve(nid),
                "permutation {i} changed the winner for {function:?} — resolution is \
                 order-dependent again"
            );
        }
    }
}

/// A function known ONLY by its NID must actually be reachable.
///
/// `libSceAgc`'s `qj7QZpgr9Uw` has no recovered name; the convention is to label
/// it `sceAgcUnknownQj7QZpgr9Uw`. Resolution normally hashes the *name*, so that
/// placeholder hashes to something else entirely and the implementation — which
/// exists, and is registered — could never be reached, while the measured retail
/// title imports exactly this NID and reported it missing. `register_nid` binds
/// the NID explicitly; `NidDatabase::from_hle` is what applies those bindings.
#[test]
fn a_function_known_only_by_nid_is_reachable() {
    const AGC_UNKNOWN_NID: u64 = 0xaa3e_d066_982b_f54c; // qj7QZpgr9Uw

    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle(&hle);
    let registry = ModuleRegistry::new(db);

    match registry.resolve_import(&hle, "libSceAgc", "libSceAgc", AGC_UNKNOWN_NID) {
        xps5x_firmware::Resolver::Hle { library, .. } => assert_eq!(library, "libSceAgc"),
        other => panic!(
            "a NID-only function must resolve; got {other:?}. Hashing the placeholder name \
             cannot produce {}: the implementation would be unreachable.",
            encode_nid(AGC_UNKNOWN_NID)
        ),
    }

    // And the placeholder label really does hash to something else — i.e. this
    // test is not passing by accident.
    assert_ne!(
        nid_of("sceAgcUnknownQj7QZpgr9Uw"),
        AGC_UNKNOWN_NID,
        "if the label ever hashed to the right NID, register_nid would be unnecessary"
    );

    // The name-only path cannot reach it — the exact bug this guards.
    let names_only = NidDatabase::from_hle_names(hle.registered_names());
    assert!(
        names_only.resolve(AGC_UNKNOWN_NID).is_none(),
        "from_hle_names cannot express a NID-only function; use NidDatabase::from_hle"
    );
}

/// End to end through the real resolver: the exact NID the title requested must
/// come back as an HLE hit, not `Unresolved`.
#[test]
fn the_gettimeofday_nid_resolves_through_the_module_registry() {
    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);

    let nid = nid_of("gettimeofday");
    match registry.resolve_import(&hle, "libkernel", "libScePosix", nid) {
        xps5x_firmware::Resolver::Hle { library, function } => {
            assert_eq!(function, "gettimeofday");
            assert_eq!(library, "libScePosix");
        }
        other => panic!(
            "the NID a real title blocked on must resolve to HLE, got {other:?} \
             (encoded {})",
            encode_nid(nid)
        ),
    }
}
