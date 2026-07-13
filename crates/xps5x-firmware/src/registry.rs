//! `ModuleRegistry` — dispatches each module import NID to either an HLE
//! implementation or a loaded LLE (real, linked) export, per a per-module
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

use crate::dynlib::nid::NidDatabase;
use crate::dynlib::SymbolExport;

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
}

impl ModuleRegistry {
    /// Build a registry over the given NID database. All modules default to
    /// [`ModulePolicy::PreferHle`] until [`Self::set_policy`] says otherwise.
    pub fn new(nid_db: NidDatabase) -> Self {
        Self {
            nid_db,
            policies: HashMap::new(),
            lle_exports: HashMap::new(),
        }
    }

    /// Set the dispatch policy for `module`. Modules with no explicit policy
    /// use [`ModulePolicy::PreferHle`].
    pub fn set_policy(&mut self, module: &str, policy: ModulePolicy) {
        self.policies.insert(module.to_string(), policy);
    }

    /// Record `exports` as this module's LLE exports, available to satisfy
    /// other modules' imports by NID. `module` is used for diagnostics only
    /// — the export table itself is a single flat NID -> address map.
    pub fn register_module_exports(&mut self, module: &str, exports: &[SymbolExport]) {
        for export in exports {
            tracing::debug!(
                "registering LLE export {:#x} -> {:#x} from module {module:?}",
                export.nid,
                export.value
            );
            self.lle_exports.insert(export.nid, export.value);
        }
    }

    /// Resolve `nid` for an import belonging to `importing_module`, per that
    /// module's policy (default [`ModulePolicy::PreferHle`]).
    pub fn resolve(&self, hle: &HleRegistry, importing_module: &str, nid: u64) -> Resolver {
        let policy = self
            .policies
            .get(importing_module)
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
        self.lle_exports.get(&nid).map(|&addr| Resolver::Lle { addr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::nid::nid_of;

    fn build_hle_and_db() -> (HleRegistry, NidDatabase) {
        let hle = HleRegistry::new();
        let db = NidDatabase::from_hle_names(hle.registered_names());
        (hle, db)
    }

    #[test]
    fn resolves_hle_implemented_function() {
        let (hle, db) = build_hle_and_db();
        let names = hle.registered_names();
        let (library, function) = names
            .first()
            .cloned()
            .expect("HleRegistry::new() registers at least one function");

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
            &[SymbolExport { nid, value: 0xDEAD_BEEF }],
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
        assert_eq!(registry.resolve(&hle, "someModule", nid), Resolver::Unresolved);
    }

    #[test]
    fn policy_flips_precedence_between_hle_and_lle() {
        let (hle, db) = build_hle_and_db();
        let names = hle.registered_names();
        let (library, function) = names
            .first()
            .cloned()
            .expect("HleRegistry::new() registers at least one function");
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
}
