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
    /// Try HLE first, fall back to LLE. The default — works without any
    /// module having been linked yet.
    #[default]
    PreferHle,
    /// Try LLE first, fall back to HLE.
    PreferLle,
}

/// Dispatches import NIDs to HLE or LLE implementations, per-module policy.
///
/// LLE exports are tracked globally, keyed by NID (not per-module): the
/// module name passed to [`ModuleRegistry::register_module_exports`] is used
/// only for policy lookup and diagnostics/logging, matching the LM1 design
/// (a single flat NID export space is sufficient for the homebrew pipeline
/// slice; multiple modules exporting the same NID is out of scope).
#[derive(Debug, Clone)]
pub struct ModuleRegistry {
    nid_db: NidDatabase,
    policies: HashMap<String, ModulePolicy>,
    /// Loaded LLE exports, keyed by NID -> export address.
    lle_exports: HashMap<u64, u64>,
    /// NIDs forced to resolve HLE-first regardless of the provider module's
    /// policy — a per-symbol override used to intercept one function of an
    /// otherwise-LLE module (e.g. trapping `__cxa_throw` inside the shipped
    /// libc for diagnostics without redirecting libc's malloc/etc).
    force_hle: std::collections::HashSet<u64>,
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
    pub fn force_hle_nid(&mut self, nid: u64) {
        self.force_hle.insert(nid);
    }

    /// Set the dispatch policy for `module`. Modules with no explicit policy
    /// use [`ModulePolicy::PreferHle`].
    pub fn set_policy(&mut self, module: &str, policy: ModulePolicy) {
        self.policies.insert(canonical_module_name(module), policy);
    }

    /// Record `exports` as this module's LLE exports, available to satisfy
    /// other modules' imports by NID. `module` is used for diagnostics only
    /// — the export table itself is a single flat NID -> address map.
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
        for export in exports {
            let addr = base.wrapping_add(export.value);
            tracing::debug!(
                "registering LLE export {:#x} -> {addr:#x} from module {module:?} (base {base:#x})",
                export.nid,
            );
            self.lle_exports.insert(export.nid, addr);
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
        // A per-symbol force-HLE override wins over the provider's policy, so
        // one function of an otherwise-LLE module can be intercepted while the
        // rest of that module keeps using its real code.
        if self.force_hle.contains(&nid)
            && let Some(resolved) = self.try_hle(hle, nid)
        {
            return resolved;
        }

        let policy = self
            .policies
            .get(&canonical_module_name(provider_module))
            .copied()
            .unwrap_or_default();

        match policy {
            ModulePolicy::PreferHle => self
                .try_hle(hle, nid)
                .or_else(|| self.try_lle(nid))
                .unwrap_or(Resolver::Unresolved),
            ModulePolicy::PreferLle => self
                .try_lle(nid)
                .or_else(|| self.try_hle(hle, nid))
                .unwrap_or(Resolver::Unresolved),
        }
    }

    fn try_hle(&self, hle: &HleRegistry, nid: u64) -> Option<Resolver> {
        let name = self.nid_db.resolve(nid)?;
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

    fn try_lle(&self, nid: u64) -> Option<Resolver> {
        self.lle_exports
            .get(&nid)
            .map(|&addr| Resolver::Lle { addr })
    }
}

fn canonical_module_name(module: &str) -> String {
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
        match registry.resolve(&hle, "someModule", nid) {
            Resolver::Hle {
                library: got_lib,
                function: got_fn,
            } => {
                assert_eq!(got_lib, library);
                assert_eq!(got_fn, function);
            }
            other => panic!("expected Resolver::Hle, got {other:?}"),
        }
    }

    #[test]
    fn resolves_lle_export_for_non_hle_nid() {
        let (hle, db) = build_hle_and_db();
        let mut registry = ModuleRegistry::new(db);

        let nid = nid_of("someFreshLleOnlyExport");
        registry.register_module_exports(
            "otherModule",
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
    fn policy_flips_precedence_between_hle_and_lle() {
        let (hle, db) = build_hle_and_db();
        let (library, function) = uniquely_named_hle_function(&hle);
        let nid = nid_of(&function);

        let mut registry = ModuleRegistry::new(db);
        // Also register an LLE export for the exact same NID.
        registry.register_module_exports("otherModule", &[SymbolExport { nid, value: 0x1234 }]);

        // Default policy (PreferHle): HLE wins.
        match registry.resolve(&hle, "someModule", nid) {
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
        registry.set_policy("someModule", ModulePolicy::PreferLle);
        match registry.resolve(&hle, "someModule", nid) {
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
}
