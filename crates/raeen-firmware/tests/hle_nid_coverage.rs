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

use raeen_firmware::dynlib::nid::{NidDatabase, encode_nid, nid_of};
use raeen_firmware::{ModuleRegistry, Resolver};
use raeen_hle::HleRegistry;

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

/// Provider-aware acceptance for the ASTRO.BOT boot Slice-2 fix. The retail
/// title imports `clock_gettime` (NID 0x94b313f6f240724d) naming provider
/// library **`libkernel`**, not `libScePosix`. Resolution is provider-aware, so
/// a `libScePosix`-only registration does NOT satisfy it — this asserts through
/// the real `ModuleRegistry::resolve(provider, nid)` path the linker uses, not a
/// provider-blind `NidDatabase::resolve`, so it would have caught the missing
/// `libkernel` alias the title actually stopped on.
#[test]
fn clock_gettime_resolves_from_the_libkernel_provider_the_title_names() {
    // The exact NID recovered from the retail symbol table (measured 2026-07-19
    // as the first blocker after the init-once fix restored boot).
    const CLOCK_GETTIME_NID: u64 = 0x94b3_13f6_f240_724d;
    assert_eq!(nid_of("clock_gettime"), CLOCK_GETTIME_NID);

    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    match registry.resolve(&hle, "libkernel", CLOCK_GETTIME_NID) {
        Resolver::Hle { function, .. } => assert_eq!(function, "clock_gettime"),
        other => panic!(
            "clock_gettime imported from provider 'libkernel' must resolve to an HLE \
             function; got {other:?}"
        ),
    }
}

/// Subnautica Below Zero's first blocker once its modules load and its Unity
/// launcher runs. The NID was measured from the failing run; shadPS4's
/// `aerolib.inl` independently maps the same encoded form (`woNpu+45RLk`) to
/// this name, so this also pins that our hash agrees with that spelling — a
/// disagreement would register a different identity than the title imports and
/// leave the run dying on exactly the same import.
#[test]
fn sce_user_service_get_age_level_resolves_from_the_libsceuserservice_provider() {
    const GET_AGE_LEVEL_NID: u64 = 0xc283_69bb_ee39_44b9;
    assert_eq!(nid_of("sceUserServiceGetAgeLevel"), GET_AGE_LEVEL_NID);

    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    match registry.resolve(&hle, "libSceUserService", GET_AGE_LEVEL_NID) {
        Resolver::Hle { function, .. } => assert_eq!(function, "sceUserServiceGetAgeLevel"),
        other => panic!(
            "sceUserServiceGetAgeLevel imported from provider 'libSceUserService' must \
             resolve to an HLE function; got {other:?}"
        ),
    }
}

/// Blasphemous II (PPSA13580) imports `sceVideoOutDeleteFlipEvent` and our
/// loader reported it **unresolved** — measured in that run's own log,
/// `artifacts/compat/raw/baseline-1785285421268/PPSA13580-b5469945261a.stdout.log:322`:
///
/// ```text
/// missing sceVideoOutDeleteFlipEvent — NID 0xfcece7d05d401518 (-Ozn0F1AFRg)
///   wanted from library 'libSceVideoOut'
/// ```
///
/// The title is one of the four in the silent zero-frame cluster
/// (`docs/silent-zero-frame-cluster.md` section 3): it runs, presents nothing,
/// and logs no error. This pins the spelling of the whole VideoOut event family
/// registered alongside it — a NID is a hash of the *name*, so an
/// implementation registered under a near-miss spelling resolves nothing and is
/// indistinguishable, at the log level the compat harness runs at, from not
/// having been written.
#[test]
fn video_out_delete_flip_event_resolves_the_nid_blasphemous_ii_asked_for() {
    const DELETE_FLIP_EVENT_NID: u64 = 0xfcec_e7d0_5d40_1518;
    assert_eq!(nid_of("sceVideoOutDeleteFlipEvent"), DELETE_FLIP_EVENT_NID);
    assert_eq!(encode_nid(DELETE_FLIP_EVENT_NID), "-Ozn0F1AFRg");

    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    match registry.resolve(&hle, "libSceVideoOut", DELETE_FLIP_EVENT_NID) {
        Resolver::Hle { function, .. } => assert_eq!(function, "sceVideoOutDeleteFlipEvent"),
        other => panic!(
            "sceVideoOutDeleteFlipEvent imported from provider 'libSceVideoOut' must \
             resolve to an HLE function; got {other:?}"
        ),
    }

    // The rest of the event family KytyPS5 implements (videoOut.cpp:1059-1095).
    // These are not observed as imports in any measured title yet — unlike the
    // NID above — so this asserts registration under the correct spelling, not
    // that a title needs them today.
    for name in [
        "sceVideoOutAddFlipEvent",
        "sceVideoOutAddVblankEvent",
        "sceVideoOutDeleteVblankEvent",
        "sceVideoOutAddPreVblankStartEvent",
        "sceVideoOutDeletePreVblankStartEvent",
        "sceVideoOutAddOutputModeEvent",
    ] {
        match registry.resolve(&hle, "libSceVideoOut", nid_of(name)) {
            Resolver::Hle { function, .. } => assert_eq!(function, name),
            other => panic!("{name} must resolve from 'libSceVideoOut'; got {other:?}"),
        }
    }
}

/// The first `libSceAgc` import ASTRO.BOT actually calls once boot reaches GPU
/// init. The NID was measured from the retail title; the name was recovered from
/// SharpEmu's `aerolib` catalogue, so this also pins that our NID hash agrees
/// with the recovered spelling (if it did not, the registration would silently
/// bind a different identity than the title imports).
#[test]
fn sce_agc_get_is_trinity_mode_resolves_from_the_libsceagc_provider() {
    const GET_IS_TRINITY_MODE_NID: u64 = 0x05f0_4364_66ed_8bb0;
    assert_eq!(
        nid_of("sceAgcGetIsTrinityMode"),
        GET_IS_TRINITY_MODE_NID,
        "the recovered name must hash to the NID the title imports"
    );
    assert_eq!(encode_nid(GET_IS_TRINITY_MODE_NID), "BfBDZGbti7A");

    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    match registry.resolve(&hle, "libSceAgc", GET_IS_TRINITY_MODE_NID) {
        Resolver::Hle { function, .. } => assert_eq!(function, "sceAgcGetIsTrinityMode"),
        other => panic!(
            "sceAgcGetIsTrinityMode imported from provider 'libSceAgc' must resolve to an \
             HLE function; got {other:?}"
        ),
    }
}

/// The exact import each installed retail title stopped its boot on, as
/// `(provider library, NID, expected function)`. Every one of these is a
/// **provider-aware** resolution: several were already implemented but
/// registered under a different library than the title names, which is
/// invisible to a provider-blind NID lookup and cost each title its boot.
///
/// Measured 2026-07-19 by running each title's eboot through `--run-eboot` and
/// reading the reported unresolved import.
#[test]
fn every_measured_title_boot_blocker_resolves_from_its_own_provider() {
    const TITLE_BLOCKERS: &[(&str, u64, &str, &str)] = &[
        // Minecraft: POSIX spelling imported from `libkernel`, not `libScePosix`.
        (
            "libkernel",
            0x9fcf_2fc7_70b9_9d6f,
            "gettimeofday",
            "Minecraft",
        ),
        // Until Dawn: mutex priority-protocol attribute setter.
        (
            "libkernel",
            0xd451_af53_48bd_b1a4,
            "scePthreadMutexattrSetprotocol",
            "Until Dawn",
        ),
        // Dragon Ball Sparking Zero: Share service under `libSceShare`, while
        // the implementation was registered only as `libSceShareUtility`.
        (
            "libSceShare",
            0x9c10_c3eb_a922_156f,
            "sceShareInitialize",
            "Dragon Ball Sparking Zero",
        ),
        // Until Dawn, round 2: the public (non-`Internal`) Named spelling.
        (
            "libkernel",
            0x98bf_0d0c_7f3a_8902,
            "sceKernelMapNamedFlexibleMemory",
            "Until Dawn",
        ),
        // A Plague Tale Requiem: libc's pre-main environment initialiser, then
        // C++ function-local static guards, then a direct-memory query.
        ("libc", 0x6f34_04c7_2d7c_f592, "_init_env", "A Plague Tale"),
        (
            "libc",
            0xdc63_e98d_0740_313c,
            "__cxa_guard_acquire",
            "A Plague Tale",
        ),
        (
            "libkernel",
            0x0b47_fb4c_971b_7da7,
            "sceKernelAvailableDirectMemorySize",
            "A Plague Tale",
        ),
        // Until Dawn, rounds 3-4: PSN push events, then the PS5-Pro query.
        (
            "libSceNpWebApi2",
            0x595d_46c0_cdf6_3606,
            "sceNpWebApi2PushEventCreateHandle",
            "Until Dawn",
        ),
        (
            "libkernel",
            0xb54e_5edd_ff60_4a25,
            "sceKernelIsTrinityMode",
            "Until Dawn",
        ),
    ];

    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    for (provider, nid, expected, title) in TITLE_BLOCKERS {
        match registry.resolve(&hle, provider, *nid) {
            Resolver::Hle { function, .. } => assert_eq!(
                &function, expected,
                "{title}: {provider}::{nid:#018x} resolved to the wrong function"
            ),
            other => panic!(
                "{title} stops booting here: {expected} (nid {nid:#018x}) imported from \
                 provider '{provider}' must resolve to an HLE function; got {other:?}"
            ),
        }
    }
}

/// Gen5 retail binaries import the AGC *driver* entry points from the
/// `libSceAgcDriver` library, not `libSceAgc`. Both of these were implemented
/// but registered only under `libSceAgc`, so provider-aware resolution left them
/// unreachable and ASTRO.BOT reported them missing at its first GPU submission.
/// These are the measured retail (provider, NID) identities.
#[test]
fn agc_driver_entry_points_resolve_from_the_libsceagcdriver_provider() {
    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    for (nid, expected) in [
        (0x5209_4921_98c6_b2c3u64, "sceAgcDriverSubmitDcb"),
        (0x8124_67af_bf45_f2d4, "sceAgcDriverSubmitAcb"),
        (0xc36a_c986_60fe_76c1, "sceAgcDriverAddEqEvent"),
    ] {
        match registry.resolve(&hle, "libSceAgcDriver", nid) {
            Resolver::Hle { function, .. } => assert_eq!(function, expected),
            other => panic!(
                "{expected} (nid {nid:#018x}) imported from provider 'libSceAgcDriver' must \
                 resolve to an HLE function; got {other:?}"
            ),
        }
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
        raeen_firmware::Resolver::Hle { library, .. } => assert_eq!(library, "libSceAgc"),
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
        raeen_firmware::Resolver::Hle { library, function } => {
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

/// The four `.native` `DT_NEEDED` names Blasphemous II (PPSA13580) carries must
/// reach the HLE libraries that already implement them.
///
/// Measured warnings from that title's launch — all four libraries exist in
/// tree, and all four were reported missing purely because the loader compared
/// a `DT_NEEDED` *file name* against bare HLE *library* names:
///
/// ```text
/// NEEDED libSceAjm.native.prx: no HLE library named 'libSceAjm.native'
/// NEEDED libSceAvPlayer.native.prx: no HLE library named 'libSceAvPlayer.native'
/// NEEDED libSceMsgDialog.native.prx: no HLE library named 'libSceMsgDialog.native'
/// NEEDED libSceSaveDataDialog.native.prx: no HLE library named 'libSceSaveDataDialog.native'
/// ```
///
/// This asserts the property behind that fix: the canonical identity of the
/// `.native` file name is a library the live `HleRegistry` actually registers,
/// and a real import naming the `.native` provider resolves to a real function.
#[test]
fn native_suffixed_needed_names_reach_the_libraries_that_implement_them() {
    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
    let registered: std::collections::HashSet<String> = hle
        .registered_names()
        .into_iter()
        .map(|(library, _)| raeen_firmware::canonical_module_name(&library))
        .collect();

    // (NEEDED file name, one function the title imports from it.)
    const MEASURED: &[(&str, &str)] = &[
        ("libSceAjm.native.prx", "sceAjmInitialize"),
        ("libSceAvPlayer.native.prx", "sceAvPlayerInit"),
        ("libSceMsgDialog.native.prx", "sceMsgDialogInitialize"),
        (
            "libSceSaveDataDialog.native.prx",
            "sceSaveDataDialogInitialize",
        ),
    ];

    for (needed, function) in MEASURED {
        let canonical = raeen_firmware::canonical_module_name(needed);
        assert!(
            registered.contains(&canonical),
            "NEEDED {needed} canonicalizes to {canonical:?}, which no HLE library matches — the \
             loader would warn 'no HLE library named ...' for a library that exists"
        );
        // And the provider spelling in the import table resolves for real.
        let provider = needed.trim_end_matches(".prx");
        match registry.resolve(&hle, provider, nid_of(function)) {
            Resolver::Hle {
                function: got_fn, ..
            } => assert_eq!(&got_fn, function),
            other => panic!(
                "{function} imported from provider {provider:?} must resolve to HLE; got {other:?}"
            ),
        }
    }
}

/// The POSIX filesystem spellings Blasphemous II imports from `libScePosix` and
/// that resolved to nothing.
///
/// Measured with `cargo xtask nids coverage` against the retail title: 9
/// unresolved `libScePosix` imports, of which these seven are the filesystem
/// family. `sceKernelMkdir` worked the whole time — the title was calling
/// `libScePosix::mkdir`, a different NID under a provider nothing registered.
/// `sendmsg`/`recvmsg` are the other two and remain unimplemented (socket
/// message-vector semantics, not a naming gap).
#[test]
fn posix_filesystem_spellings_the_title_imports_resolve() {
    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    for name in [
        "mkdir", "rmdir", "unlink", "chmod", "fchmod", "utimes", "futimes",
    ] {
        match registry.resolve(&hle, "libScePosix", nid_of(name)) {
            Resolver::Hle { function, library } => {
                assert_eq!(function, name);
                assert_eq!(library, "libScePosix");
            }
            other => panic!("libScePosix::{name} must resolve to HLE; got {other:?}"),
        }
    }

    // `sceKernelSleep` is the Sony spelling of `sleep` and a distinct NID; the
    // title imports it from `libkernel` and it resolved to nothing.
    for name in ["sceKernelSleep", "sceKernelFchmod"] {
        match registry.resolve(&hle, "libkernel", nid_of(name)) {
            Resolver::Hle { function, .. } => assert_eq!(function, name),
            other => panic!("libkernel::{name} must resolve to HLE; got {other:?}"),
        }
    }
}

/// The libraries Blasphemous II imports that had **zero** registrations before.
///
/// Names and counts from `cargo xtask nids coverage` against the retail title:
/// 5 unresolved `libSceAudioIn`, 6 unresolved `libSceVrSetupDialog`, 4
/// unresolved `libSceErrorDialog`, plus the three `libSceMsgDialog.native`
/// progress-bar controls. Each must now resolve **from the provider the title
/// names**, which is what a provider-blind check would miss.
#[test]
fn newly_added_libraries_resolve_from_the_providers_the_title_names() {
    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

    const MEASURED: &[(&str, &str)] = &[
        ("libSceAudioIn", "sceAudioInOpen"),
        ("libSceAudioIn", "sceAudioInAsyncOpen"),
        ("libSceAudioIn", "sceAudioInClose"),
        ("libSceAudioIn", "sceAudioInInput"),
        ("libSceAudioIn", "sceAudioInGetSilentState"),
        ("libSceVrSetupDialog", "sceVrSetupDialogInitialize"),
        ("libSceVrSetupDialog", "sceVrSetupDialogOpen"),
        ("libSceVrSetupDialog", "sceVrSetupDialogUpdateStatus"),
        ("libSceVrSetupDialog", "sceVrSetupDialogGetResult"),
        ("libSceVrSetupDialog", "sceVrSetupDialogClose"),
        ("libSceVrSetupDialog", "sceVrSetupDialogTerminate"),
        ("libSceErrorDialog", "sceErrorDialogInitialize"),
        ("libSceErrorDialog", "sceErrorDialogOpen"),
        ("libSceErrorDialog", "sceErrorDialogUpdateStatus"),
        ("libSceErrorDialog", "sceErrorDialogTerminate"),
        // Imported under the `.native` spelling; must reach the bare library.
        ("libSceMsgDialog.native", "sceMsgDialogProgressBarInc"),
        ("libSceMsgDialog.native", "sceMsgDialogProgressBarSetMsg"),
        ("libSceMsgDialog.native", "sceMsgDialogProgressBarSetValue"),
    ];

    for (provider, function) in MEASURED {
        match registry.resolve(&hle, provider, nid_of(function)) {
            Resolver::Hle {
                function: got_fn, ..
            } => assert_eq!(&got_fn, function),
            other => panic!(
                "{function} imported from provider {provider:?} must resolve to HLE; got {other:?}"
            ),
        }
    }
}
