//! `ModuleRegistry` — dispatches each module import NID to either an HLE
//! implementation or a loaded LLE (real, linked) export, per a provider-module
//! policy.
//!
//! This is the LM1 "which implementation answers this import" decision
//! point: [`crate::dynlib::nid::NidDatabase`] maps a NID back to the
//! `"library::function"` name XPS5X's HLE would register it under, and
//! [`xps5x_hle::HleRegistry`] says whether that function is actually
//! implemented. Separately, any module already linked in this session may
//! have exported the same NID (an LLE export) — e.g. because a real,
//! decrypted `.sprx` was loaded instead of relying on HLE. The registry
//! picks between the two according to [`ModulePolicy`].

use std::collections::HashMap;

use xps5x_hle::HleRegistry;

use crate::dynlib::SymbolExport;
use crate::dynlib::nid::NidDatabase;

/// The result of resolving one import NID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolver {
    /// Resolved to an HLE-implemented function.
    Hle { library: String, function: String },
    /// Resolved to a loaded LLE (real module) export address.
    Lle { addr: u64 },
    /// No HLE implementation and no loaded LLE export for this NID.
    Unresolved,
}

/// Per-module dispatch preference: which implementation to try first when
/// both an HLE implementation and an LLE export exist for the same NID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModulePolicy {
    /// Resolve only XPS5X HLE implementations.
    HleOnly,
    /// Try HLE first, fall back to LLE. The default — works without any
    /// module having been linked yet.
    #[default]
    PreferHle,
    /// Try LLE first, fall back to HLE.
    PreferLle,
    /// Resolve only exports from a user-supplied, decryptable module.
    LleOnly,
}

/// Dispatches import NIDs to HLE or LLE implementations, per-module policy.
///
/// LLE exports are keyed by provider module and NID. Two title-supplied
/// modules may legally export the same NID; the importing symbol's provider
/// chooses which address is eligible.
#[derive(Debug, Clone)]
pub struct ModuleRegistry {
    nid_db: NidDatabase,
    policies: HashMap<String, ModulePolicy>,
    /// Loaded LLE exports, keyed by (canonical provider module, NID).
    lle_exports: HashMap<(String, u64), u64>,
    /// NIDs forced to resolve HLE-first regardless of the provider module's
    /// policy — a per-symbol override used to intercept one function of an
    /// otherwise-LLE module (e.g. trapping `__cxa_throw` inside the shipped
    /// libc for diagnostics without redirecting libc's malloc/etc).
    force_hle: std::collections::HashSet<(String, u64)>,
}

impl ModuleRegistry {
    /// Build a registry over the given NID database. All modules default to
    /// [`ModulePolicy::PreferHle`] until [`Self::set_policy`] says otherwise.
    pub fn new(nid_db: NidDatabase) -> Self {
        Self {
            nid_db,
            policies: HashMap::new(),
            lle_exports: HashMap::new(),
            force_hle: std::collections::HashSet::new(),
        }
    }

    /// Force `nid` to resolve HLE-first even when its provider module is
    /// `PreferLle`. Used for targeted single-symbol interception.
    pub fn force_hle_nid(&mut self, module: &str, nid: u64) {
        self.force_hle.insert((canonical_module_name(module), nid));
    }

    /// Set the dispatch policy for `module`. Modules with no explicit policy
    /// use [`ModulePolicy::PreferHle`].
    pub fn set_policy(&mut self, module: &str, policy: ModulePolicy) {
        self.policies.insert(canonical_module_name(module), policy);
    }

    /// Effective provider policy. Unconfigured modules remain HLE-first, so a
    /// normal installation never depends on firmware files or keys.
    #[must_use]
    pub fn policy_for(&self, module: &str) -> ModulePolicy {
        self.policies
            .get(&canonical_module_name(module))
            .copied()
            .unwrap_or_default()
    }

    /// Record `exports` as this module's LLE exports, available to satisfy
    /// imports that name this provider module and NID.
    ///
    /// Exports are registered at their **module-relative** address. Prefer
    /// [`Self::register_module_exports_at`] for a module loaded at a non-zero
    /// base: [`Resolver::Lle`]'s address is written into the importer's
    /// relocation slot verbatim, so it must be the **absolute** guest address.
    pub fn register_module_exports(&mut self, module: &str, exports: &[SymbolExport]) {
        self.register_module_exports_at(module, exports, 0);
    }

    /// Record `exports` for a module loaded at `base`, registering each at its
    /// **absolute** guest address (`base + export.value`).
    ///
    /// A dependency `.prx` is mapped at a non-zero base alongside the main
    /// module, and `link_module` writes a resolved [`Resolver::Lle`] address
    /// straight into the importing module's slot — so a module-relative value
    /// would send the importer to the wrong address entirely.
    pub fn register_module_exports_at(
        &mut self,
        module: &str,
        exports: &[SymbolExport],
        base: u64,
    ) {
        let module = canonical_module_name(module);
        for export in exports {
            let addr = base.wrapping_add(export.value);
            tracing::debug!(
                "registering LLE export {:#x} -> {addr:#x} from module {module:?} (base {base:#x})",
                export.nid,
            );
            self.lle_exports.insert((module.clone(), export.nid), addr);
        }
    }

    /// Resolve `nid` from `provider_module`, per that provider's policy
    /// (default [`ModulePolicy::PreferHle`]).
    ///
    /// This must be the module named by the import symbol, not the module that
    /// happens to contain the relocation. Otherwise a title-supplied runtime
    /// such as `libc.prx` can be split between its own stateful implementation
    /// and unrelated HLE functions according to who called it.
    pub fn resolve(&self, hle: &HleRegistry, provider_module: &str, nid: u64) -> Resolver {
        self.resolve_import(hle, provider_module, provider_module, nid)
    }

    /// Resolve an import that carries **no provider identity** at all.
    ///
    /// Hand-built fixtures without dynlib import tables land here: there is no
    /// provider module whose policy could apply, so the strict
    /// no-cross-provider rule has nothing to bind to. A NID hashes the
    /// function name alone, so the provider-free view can only ever "borrow"
    /// an implementation across libraries when two libraries register the
    /// *same* function (deliberate aliases); genuine name collisions are
    /// already deduped with a warning when the database is built. Real modules
    /// always name their provider and never take this path.
    ///
    /// LLE exports are keyed by provider module; with no provider known, the
    /// only key available is the consumer's own module name — which is how
    /// fixtures register their exports.
    pub fn resolve_unattributed(
        &self,
        hle: &HleRegistry,
        consumer_module: &str,
        nid: u64,
    ) -> Resolver {
        if let Some(name) = self.nid_db.resolve(nid)
            && let Some((library, function)) = name.split_once("::")
            && hle.is_implemented(library, function)
        {
            return Resolver::Hle {
                library: library.to_string(),
                function: function.to_string(),
            };
        }
        self.try_lle(&canonical_module_name(consumer_module), nid)
            .unwrap_or(Resolver::Unresolved)
    }

    /// Resolve an import while preserving both parts of its provider identity.
    /// Module policy and LLE exports belong to `provider_module`; HLE exports
    /// are registered under `provider_library` (for example, module
    /// `libkernel` provides library `libScePosix`).
    pub fn resolve_import(
        &self,
        hle: &HleRegistry,
        provider_module: &str,
        provider_library: &str,
        nid: u64,
    ) -> Resolver {
        // A per-symbol force-HLE override can intercept one function of an
        // HLE-first/LLE-first provider, but never crosses strict `LleOnly`.
        let provider_module = canonical_module_name(provider_module);
        let policy = self.policy_for(&provider_module);
        if policy != ModulePolicy::LleOnly
            && self.force_hle.contains(&(provider_module.clone(), nid))
            && let Some(resolved) = self.try_hle(hle, provider_library, nid)
        {
            return resolved;
        }

        match policy {
            ModulePolicy::HleOnly => self
                .try_hle(hle, provider_library, nid)
                .unwrap_or(Resolver::Unresolved),
            ModulePolicy::PreferHle => self
                .try_hle(hle, provider_library, nid)
                .or_else(|| self.try_lle(&provider_module, nid))
                .unwrap_or(Resolver::Unresolved),
            ModulePolicy::PreferLle => self
                .try_lle(&provider_module, nid)
                .or_else(|| self.try_hle(hle, provider_library, nid))
                .unwrap_or(Resolver::Unresolved),
            ModulePolicy::LleOnly => self
                .try_lle(&provider_module, nid)
                .unwrap_or(Resolver::Unresolved),
        }
    }

    fn try_hle(&self, hle: &HleRegistry, provider_module: &str, nid: u64) -> Option<Resolver> {
        let name = self.nid_db.resolve_for_provider(provider_module, nid)?;
        let (library, function) = name.split_once("::")?;
        if hle.is_implemented(library, function) {
            Some(Resolver::Hle {
                library: library.to_string(),
                function: function.to_string(),
            })
        } else {
            None
        }
    }

    fn try_lle(&self, provider_module: &str, nid: u64) -> Option<Resolver> {
        self.lle_exports
            .get(&(provider_module.to_string(), nid))
            .map(|&addr| Resolver::Lle { addr })
    }
}

/// The identity a module has for policy, LLE exports, and the process
/// loader's visit-set alike: lowercased, `.sprx`/`.prx` suffix stripped.
/// `pub(crate)` so `load_process`'s dependency walk dedupes by exactly the
/// identity the registry resolves providers by.
pub(crate) fn canonical_module_name(module: &str) -> String {
    let lower = module.to_ascii_lowercase();
    lower
        .strip_suffix(".sprx")
        .or_else(|| lower.strip_suffix(".prx"))
        .unwrap_or(&lower)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::nid::nid_of;

    fn build_hle_and_db() -> (HleRegistry, NidDatabase) {
        let hle = HleRegistry::new();
        let db = NidDatabase::from_hle(&hle);
        (hle, db)
    }

    /// Pick a real HLE function whose name is registered under exactly ONE
    /// library, deterministically.
    ///
    /// A test that just takes `registered_names().first()` is doing two unsafe
    /// things at once. The order comes from a `DashMap` and is not stable, and
    /// — worse — some function names are deliberately registered under several
    /// libraries (`getpid` in both `libkernel` and `libScePosix`; the whole
    /// `sce::Json` set in `libSceJson`/`libSceJson2`). A NID hashes the name
    /// alone, so those share one NID and `NidDatabase` keeps exactly one
    /// `library::function` for them. Picking such a name and then asserting the
    /// resolved library equals the one it came from fails whenever the two
    /// disagree — intermittently, since the pick was random.
    ///
    /// Sorting makes the choice reproducible; filtering to names registered
    /// once makes the library assertion meaningful.
    fn uniquely_named_hle_function(hle: &HleRegistry) -> (String, String) {
        let mut names = hle.registered_names();
        names.sort();
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (_, f) in &names {
            *counts.entry(f.as_str()).or_default() += 1;
        }
        names
            .iter()
            .find(|(_, f)| counts[f.as_str()] == 1)
            .cloned()
            .expect("the HLE registers at least one uniquely-named function")
    }

    #[test]
    fn resolves_hle_implemented_function() {
        let (hle, db) = build_hle_and_db();
        let (library, function) = uniquely_named_hle_function(&hle);

        let registry = ModuleRegistry::new(db);
        let nid = nid_of(&function);
        match registry.resolve(&hle, &library, nid) {
            Resolver::Hle {
                library: got_lib,
                function: got_fn,
            } => {
                assert_eq!(got_lib, library);
                assert_eq!(got_fn, function);
            }
            other => panic!("expected Resolver::Hle, got {other:?}"),
        }
        assert_eq!(
            registry.resolve(&hle, "unrelatedProvider", nid),
            Resolver::Unresolved,
            "an unrelated provider must not borrow another module's HLE export"
        );
    }

    #[test]
    fn resolves_lle_export_for_non_hle_nid() {
        let (hle, db) = build_hle_and_db();
        let mut registry = ModuleRegistry::new(db);

        let nid = nid_of("someFreshLleOnlyExport");
        registry.register_module_exports(
            "someModule",
            &[SymbolExport {
                nid,
                value: 0xDEAD_BEEF,
            }],
        );

        match registry.resolve(&hle, "someModule", nid) {
            Resolver::Lle { addr } => assert_eq!(addr, 0xDEAD_BEEF),
            other => panic!("expected Resolver::Lle, got {other:?}"),
        }
    }

    #[test]
    fn unknown_nid_is_unresolved() {
        let (hle, db) = build_hle_and_db();
        let registry = ModuleRegistry::new(db);

        let nid = nid_of("totallyUnknownFunctionNameNobodyRegistered");
        assert_eq!(
            registry.resolve(&hle, "someModule", nid),
            Resolver::Unresolved
        );
    }

    #[test]
    fn unattributed_import_resolves_provider_free() {
        let (hle, db) = build_hle_and_db();
        let (library, function) = uniquely_named_hle_function(&hle);
        let registry = ModuleRegistry::new(db);
        let nid = nid_of(&function);

        // No provider metadata: the provider-free NID view finds the HLE
        // implementation regardless of which library registered it.
        match registry.resolve_unattributed(&hle, "someConsumer", nid) {
            Resolver::Hle {
                library: got_lib,
                function: got_fn,
            } => {
                assert_eq!(got_lib, library);
                assert_eq!(got_fn, function);
            }
            other => panic!("expected Resolver::Hle, got {other:?}"),
        }

        // An unknown NID stays unresolved; nothing can be borrowed.
        assert_eq!(
            registry.resolve_unattributed(
                &hle,
                "someConsumer",
                nid_of("nobodyRegisteredThisEither")
            ),
            Resolver::Unresolved
        );
    }

    #[test]
    fn policy_flips_precedence_between_hle_and_lle() {
        let (hle, db) = build_hle_and_db();
        let (library, function) = uniquely_named_hle_function(&hle);
        let nid = nid_of(&function);

        let mut registry = ModuleRegistry::new(db);
        // Also register an LLE export for the exact same NID.
        registry.register_module_exports(&library, &[SymbolExport { nid, value: 0x1234 }]);

        // Default policy (PreferHle): HLE wins.
        match registry.resolve(&hle, &library, nid) {
            Resolver::Hle {
                library: got_lib,
                function: got_fn,
            } => {
                assert_eq!(got_lib, library);
                assert_eq!(got_fn, function);
            }
            other => panic!("expected Resolver::Hle under default PreferHle, got {other:?}"),
        }

        // Flip to PreferLle: LLE wins for the same NID.
        registry.set_policy(&library, ModulePolicy::PreferLle);
        match registry.resolve(&hle, &library, nid) {
            Resolver::Lle { addr } => assert_eq!(addr, 0x1234),
            other => panic!("expected Resolver::Lle under PreferLle, got {other:?}"),
        }
    }

    #[test]
    fn policy_names_are_case_insensitive_and_ignore_prx_suffixes() {
        let (hle, db) = build_hle_and_db();
        let (_, function) = uniquely_named_hle_function(&hle);
        let nid = nid_of(&function);
        let mut registry = ModuleRegistry::new(db);
        registry.register_module_exports("libc", &[SymbolExport { nid, value: 0x5678 }]);

        registry.set_policy("LiBc.PrX", ModulePolicy::PreferLle);

        assert_eq!(
            registry.resolve(&hle, "libc", nid),
            Resolver::Lle { addr: 0x5678 }
        );
    }

    #[test]
    fn strict_policies_do_not_cross_the_hle_lle_boundary() {
        let (hle, db) = build_hle_and_db();
        let (library, function) = uniquely_named_hle_function(&hle);
        let nid = nid_of(&function);
        let mut registry = ModuleRegistry::new(db);
        registry.register_module_exports(&library, &[SymbolExport { nid, value: 0x9876 }]);

        registry.set_policy(&library, ModulePolicy::HleOnly);
        assert!(matches!(
            registry.resolve(&hle, &library, nid),
            Resolver::Hle { .. }
        ));

        registry.set_policy(&library, ModulePolicy::LleOnly);
        registry.force_hle_nid(&library, nid);
        assert_eq!(
            registry.resolve(&hle, &library, nid),
            Resolver::Lle { addr: 0x9876 }
        );
        assert_eq!(
            registry.policy_for(&format!("{library}.PRX")),
            ModulePolicy::LleOnly
        );
        assert_eq!(registry.policy_for("unconfigured"), ModulePolicy::PreferHle);
    }

    #[test]
    fn same_nid_in_two_lle_modules_resolves_by_provider() {
        let (hle, db) = build_hle_and_db();
        let mut registry = ModuleRegistry::new(db);
        let nid = nid_of("sharedLleExport");
        registry.register_module_exports("libAlpha.prx", &[SymbolExport { nid, value: 0x1111 }]);
        registry.register_module_exports("libBeta.sprx", &[SymbolExport { nid, value: 0x2222 }]);

        assert_eq!(
            registry.resolve(&hle, "LIBALPHA", nid),
            Resolver::Lle { addr: 0x1111 }
        );
        assert_eq!(
            registry.resolve(&hle, "libBeta.prx", nid),
            Resolver::Lle { addr: 0x2222 }
        );
        assert_eq!(
            registry.resolve(&hle, "libGamma", nid),
            Resolver::Unresolved
        );
    }

    #[test]
    fn hle_uses_library_identity_when_module_and_library_differ() {
        let (hle, db) = build_hle_and_db();
        let (_, function) = hle
            .registered_names()
            .into_iter()
            .find(|(library, _)| library == "libScePosix")
            .expect("libScePosix has registered HLE exports");
        let nid = nid_of(&function);
        let registry = ModuleRegistry::new(db);

        assert_eq!(
            registry.resolve_import(&hle, "libkernel", "libScePosix", nid),
            Resolver::Hle {
                library: "libScePosix".to_string(),
                function,
            }
        );
    }
}
