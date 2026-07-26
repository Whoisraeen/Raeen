//! `cargo xtask nids coverage` — per-game NID coverage.
//!
//! For every registered title, parse `eboot.bin` plus every on-disk NEEDED
//! `.prx`/`.sprx` through the *same* static view the loader uses
//! ([`inspect_module`]), union the import tables, and classify each unique
//! (provider, NID) import exactly like the linker does: HLE via the live
//! `HleRegistry`, LLE via the exports of the title's own modules, else
//! unresolved. Render-path libraries (`libSceAgc*`, `libSceVideoOut*`,
//! `libSceGnm*`, `libSceShader*`) are broken out separately — "what does this
//! game need before it can render" is the question that scopes M2/M5.
//!
//! Unresolved imports split into dictionary-named (implementable today) and
//! anonymous (no hash-proven name — the dictionary-fill targets; see
//! `crates/raeen-firmware/src/dynlib/nid_names.rs`).
//!
//! LLE registration is best-case: every found module's exports are registered
//! before any import is classified, so cross-module resolution sees the
//! maximal module set regardless of real load order.
//!
//! Output is local engineering evidence under gitignored `artifacts/`: it
//! names titles and module files and must never feed a public report.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use raeen_firmware::crypto::NoKeysProvider;
use raeen_firmware::dynlib::SymbolExport;
use raeen_firmware::dynlib::nid::{NidDatabase, nid_of};
use raeen_firmware::dynlib::nid_names;
use raeen_firmware::registry::{ModuleRegistry, Resolver};
use raeen_firmware::{hle_data_page_export_names, inspect_module};
use raeen_hle::HleRegistry;
use serde::Serialize;

use crate::schema::{GameRecord, Registry};
use crate::{DEFAULT_REGISTRY, git_output, has, now_ms, option, read_json, sha1_file, write_json};

/// Provider-name prefixes on the render path: a title cannot present a frame
/// without its share of these resolved. Matched lowercase, so `libSceAgc`,
/// `libSceAgcDriver`, `libSceVideoOut`, `libSceVideoOut2`, … all count.
const RENDER_PATH_PREFIXES: [&str; 4] =
    ["libsceagc", "libscevideoout", "libscegnm", "libsceshader"];

fn is_render_path(provider: &str) -> bool {
    let provider = provider.to_ascii_lowercase();
    RENDER_PATH_PREFIXES
        .iter()
        .any(|prefix| provider.starts_with(prefix))
}

fn hex_nid(nid: u64) -> String {
    format!("0x{nid:016x}")
}

#[derive(Serialize)]
struct MissingImport {
    provider: String,
    nid: String,
    name: Option<String>,
}

#[derive(Serialize)]
struct ModuleProblem {
    module: String,
    error: String,
}

#[derive(Serialize)]
struct ParsedModule {
    /// File name (`libc.prx`), not a path — paths stay out of reports. This is
    /// also the module's LLE identity: NEEDED/import-module names are file
    /// stems, so exports register under it (same as `load_process`).
    file: String,
    imports: usize,
    exports: usize,
}

#[derive(Serialize)]
struct GraphicsCoverage {
    imported: usize,
    resolved: usize,
    unresolved: Vec<MissingImport>,
}

#[derive(Serialize)]
struct GameCoverage {
    id: String,
    title: String,
    /// `ok` (everything parsed), `partial` (some modules failed), or
    /// `unreadable` (the eboot itself would not parse — e.g. encrypted).
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    modules_parsed: usize,
    modules: Vec<ParsedModule>,
    needed_not_on_disk: Vec<String>,
    module_problems: Vec<ModuleProblem>,
    imports_unique: usize,
    resolved_hle: usize,
    resolved_lle: usize,
    unresolved: usize,
    unresolved_named: usize,
    unresolved_anonymous: usize,
    unresolved_by_provider: BTreeMap<String, usize>,
    graphics: GraphicsCoverage,
    /// Unresolved imports with no hash-proven name — the dictionary-fill
    /// targets, listed in full.
    anonymous: Vec<MissingImport>,
    /// Every unresolved import (the console shows only a summary).
    unresolved_full: Vec<MissingImport>,
}

#[derive(Serialize)]
struct UnionMissing {
    provider: String,
    nid: String,
    name: Option<String>,
    games: Vec<String>,
}

#[derive(Serialize)]
struct IncompleteRegistration {
    library: String,
    function: String,
    reason: String,
}

#[derive(Serialize)]
struct CoverageReport {
    schema_version: u32,
    generated_unix_ms: u128,
    build_revision: String,
    dictionary_entries: usize,
    hle_registered: usize,
    /// Callable compatibility shims that resolve imports but do not implement
    /// the complete subsystem behavior. Kept explicit so import coverage is
    /// never published as behavioral correctness.
    registered_but_not_implemented: Vec<IncompleteRegistration>,
    games: Vec<GameCoverage>,
    /// Unresolved (provider, NID) across all analyzable games, most-shared
    /// first — the priority order for new HLE work.
    union_unresolved: Vec<UnionMissing>,
}

enum Resolution {
    Hle,
    Lle,
    Unresolved,
}

struct ImportRecord {
    name: Option<&'static str>,
    resolution: Resolution,
}

/// Parse one module file; errors become a string, never abort the walk.
fn inspect_file(path: &Path) -> Result<raeen_firmware::InspectedModule, String> {
    let bytes = fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    inspect_module(&bytes, &NoKeysProvider).map_err(|e| format!("{e}"))
}

/// Runtime-mirrored module discovery bounds (`ModuleIndex::build`).
const MODULE_SCAN_MAX_DEPTH: usize = 4;
const MODULE_SCAN_SKIP_DIRS: [&str; 3] = ["sce_sys", "savedata", "streamingassets"];
/// Directories a title expects already started at boot or activates itself
/// via `sceKernelLoadStartModule` (Unity layout) — mirrored from
/// `load_process` (`EAGER_PLUGIN_DIRS` + scanned plugins), so these modules
/// count in coverage even when no DT_NEEDED names them.
const PLUGIN_DIRS: [&str; 2] = ["media/modules", "media/plugins"];

fn is_module_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("prx") || ext.eq_ignore_ascii_case("sprx"))
}

/// Every `.prx`/`.sprx` under the app root, mirroring the runtime's
/// `ModuleIndex::build`: depth ≤ [`MODULE_SCAN_MAX_DEPTH`], metadata/content
/// dirs pruned, no symlinks, shallowest-first deterministic order.
fn index_modules(eboot_dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<(usize, PathBuf)>) {
        if depth > MODULE_SCAN_MAX_DEPTH {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_symlink() {
                continue;
            }
            let path = entry.path();
            if kind.is_dir() {
                let skip = entry.file_name().to_str().is_some_and(|name| {
                    MODULE_SCAN_SKIP_DIRS
                        .iter()
                        .any(|skip| skip.eq_ignore_ascii_case(name))
                });
                if !skip {
                    walk(&path, depth + 1, out);
                }
            } else if kind.is_file() && is_module_file(&path) {
                out.push((depth, path));
            }
        }
    }
    let mut indexed = Vec::new();
    walk(eboot_dir, 0, &mut indexed);
    indexed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    indexed.into_iter().map(|(_, path)| path).collect()
}

fn canonical_stem(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// Collect every module the runtime could resolve from `eboot`: the eboot
/// itself, boot/plugin modules (Unity `Media/*` layout), and the transitive
/// NEEDED chain. Lookup mirrors `find_dependency_file` precedence (app root,
/// then `sce_module/`, then the recursive index) and `ModuleIndex::find`
/// identity (canonical file stem, so a NEEDED `x.sprx` finds a shipped
/// `x.prx`). Returns parsed modules, NEEDED names with no file on disk, and
/// per-module problems.
fn collect_modules(
    eboot: &Path,
) -> (
    Vec<(String, raeen_firmware::InspectedModule)>,
    BTreeSet<String>,
    Vec<ModuleProblem>,
) {
    let eboot_dir = eboot.parent().unwrap_or_else(|| Path::new("."));
    let index = index_modules(eboot_dir);
    let in_dir = |path: &Path, name: &str| {
        path.parent()
            .and_then(|parent| parent.file_name())
            .is_some_and(|dir| dir.eq_ignore_ascii_case(name))
    };
    let mut on_disk: HashMap<String, PathBuf> = HashMap::new();
    // First insert wins, so insert in precedence order: root, sce_module/,
    // then the rest of the index (already shallowest-first).
    for path in index.iter().filter(|path| path.parent() == Some(eboot_dir)) {
        on_disk
            .entry(canonical_stem(path))
            .or_insert_with(|| path.clone());
    }
    for path in index.iter().filter(|path| in_dir(path, "sce_module")) {
        on_disk
            .entry(canonical_stem(path))
            .or_insert_with(|| path.clone());
    }
    for path in &index {
        on_disk
            .entry(canonical_stem(path))
            .or_insert_with(|| path.clone());
    }

    // Seed exactly what the runtime starts without a NEEDED entry: the eboot
    // plus boot/plugin modules (their exports and imports count in coverage).
    let rel_dir = |path: &Path| {
        path.parent()
            .and_then(|parent| parent.strip_prefix(eboot_dir).ok())
            .map(|rel| {
                rel.to_string_lossy()
                    .replace('\\', "/")
                    .to_ascii_lowercase()
            })
            .unwrap_or_default()
    };
    let mut queue: Vec<PathBuf> = vec![eboot.to_path_buf()];
    for path in index
        .iter()
        .filter(|path| PLUGIN_DIRS.contains(&rel_dir(path).as_str()))
    {
        queue.push(path.clone());
    }

    let mut parsed = Vec::new();
    let mut not_on_disk = BTreeSet::new();
    let mut problems = Vec::new();
    let mut visited = BTreeSet::new();

    while let Some(path) = queue.pop() {
        let key = path.to_string_lossy().to_ascii_lowercase();
        if !visited.insert(key) {
            continue;
        }
        let display = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let inspected = match inspect_file(&path) {
            Ok(inspected) => inspected,
            Err(error) => {
                problems.push(ModuleProblem {
                    module: display,
                    error,
                });
                continue;
            }
        };
        for needed in &inspected.dynlib.needed_modules {
            let key = canonical_stem(Path::new(needed));
            match on_disk.get(&key) {
                Some(needed_path) => queue.push(needed_path.clone()),
                None => {
                    not_on_disk.insert(needed.clone());
                }
            }
        }
        parsed.push((display, inspected));
    }
    (parsed, not_on_disk, problems)
}

fn analyze_game(game: &GameRecord, hle: &HleRegistry, full: bool) -> GameCoverage {
    let mut coverage = GameCoverage {
        id: game.id.clone(),
        title: game.title.clone(),
        status: "ok".into(),
        error: None,
        modules_parsed: 0,
        modules: Vec::new(),
        needed_not_on_disk: Vec::new(),
        module_problems: Vec::new(),
        imports_unique: 0,
        resolved_hle: 0,
        resolved_lle: 0,
        unresolved: 0,
        unresolved_named: 0,
        unresolved_anonymous: 0,
        unresolved_by_provider: BTreeMap::new(),
        graphics: GraphicsCoverage {
            imported: 0,
            resolved: 0,
            unresolved: Vec::new(),
        },
        anonymous: Vec::new(),
        unresolved_full: Vec::new(),
    };

    let Some(local_path) = game.local_path.as_deref() else {
        coverage.status = "unreadable".into();
        coverage.error = Some("registry entry has no local executable path".into());
        return coverage;
    };
    let eboot = PathBuf::from(local_path);
    let (parsed, not_on_disk, problems) = collect_modules(&eboot);
    coverage.needed_not_on_disk = not_on_disk.into_iter().collect();
    coverage.module_problems = problems;
    coverage.modules_parsed = parsed.len();
    coverage.modules = parsed
        .iter()
        .map(|(file, inspected)| ParsedModule {
            file: file.clone(),
            imports: inspected.dynlib.imports.len(),
            exports: inspected.dynlib.exports.len(),
        })
        .collect();

    // An unreadable eboot means zero import data: report it, don't fake it.
    if parsed.is_empty() {
        coverage.status = "unreadable".into();
        coverage.error = Some("eboot did not parse".into());
        return coverage;
    }
    if !coverage.module_problems.is_empty() {
        coverage.status = "partial".into();
    }

    // Best-case LLE: register every module's exports before classifying. The
    // key is the FILE name — NEEDED/import-module identities are file stems
    // (mirrors `load_process`, which registers dependency exports under the
    // NEEDED name), not `SprxModule::name`, which is a placeholder on PS5.
    let mut registry = ModuleRegistry::new(NidDatabase::from_hle(hle));
    for (file, inspected) in &parsed {
        registry.register_module_exports(file, &inspected.dynlib.exports);
    }
    // The runtime also builds its HLE data page for every process
    // (`build_hle_data_page`, called by `load_process`): data exports like
    // `__stack_chk_guard` and `__progname` resolve as LLE, not HLE — model
    // that or coverage reports them missing when the loader resolves them.
    for (provider, name) in hle_data_page_export_names() {
        registry.register_module_exports(
            provider,
            &[SymbolExport {
                nid: nid_of(name),
                value: 0,
            }],
        );
    }

    // Classify each unique (provider, NID) import with the linker's own rules.
    let mut imports: BTreeMap<(String, u64), ImportRecord> = BTreeMap::new();
    for (_, inspected) in &parsed {
        let dynlib = &inspected.dynlib;
        let lib_names: HashMap<u16, &str> = dynlib
            .import_libs
            .iter()
            .map(|(id, name)| (*id, name.as_str()))
            .collect();
        let module_names: HashMap<u16, &str> = dynlib
            .import_modules
            .iter()
            .map(|(id, name)| (*id, name.as_str()))
            .collect();
        for import in &dynlib.imports {
            let provider_module = module_names.get(&import.module_index).copied();
            let provider_library = lib_names
                .get(&import.library_index)
                .copied()
                .or(provider_module);
            let key_provider = provider_library
                .or(provider_module)
                .unwrap_or("<unattributed>")
                .to_string();
            imports
                .entry((key_provider, import.nid))
                .or_insert_with(|| {
                    let resolution = match provider_module {
                        Some(provider_module) => registry.resolve_import(
                            hle,
                            provider_module,
                            provider_library.unwrap_or(provider_module),
                            import.nid,
                        ),
                        None => {
                            registry.resolve_unattributed(hle, &inspected.module.name, import.nid)
                        }
                    };
                    ImportRecord {
                        name: nid_names::name_of(import.nid),
                        resolution: match resolution {
                            Resolver::Hle { .. } => Resolution::Hle,
                            Resolver::Lle { .. } => Resolution::Lle,
                            Resolver::Unresolved => Resolution::Unresolved,
                        },
                    }
                });
        }
    }

    coverage.imports_unique = imports.len();
    for ((provider, nid), record) in &imports {
        match record.resolution {
            Resolution::Hle => coverage.resolved_hle += 1,
            Resolution::Lle => coverage.resolved_lle += 1,
            Resolution::Unresolved => {
                coverage.unresolved += 1;
                let missing = MissingImport {
                    provider: provider.clone(),
                    nid: hex_nid(*nid),
                    name: record.name.map(str::to_string),
                };
                if record.name.is_some() {
                    coverage.unresolved_named += 1;
                } else {
                    coverage.unresolved_anonymous += 1;
                    coverage.anonymous.push(MissingImport {
                        provider: missing.provider.clone(),
                        nid: missing.nid.clone(),
                        name: None,
                    });
                }
                *coverage
                    .unresolved_by_provider
                    .entry(provider.clone())
                    .or_insert(0) += 1;
                if is_render_path(provider) {
                    coverage
                        .graphics
                        .unresolved
                        .push(missing.clone_for_render());
                }
                coverage.unresolved_full.push(missing);
            }
        }
        if is_render_path(provider) {
            coverage.graphics.imported += 1;
            if !matches!(record.resolution, Resolution::Unresolved) {
                coverage.graphics.resolved += 1;
            }
        }
    }

    print_game_summary(&coverage, full);
    coverage
}

// `MissingImport` is Serialize-only; cloning for the render-path list stays
// explicit and allocation-cheap (three small fields).
impl MissingImport {
    fn clone_for_render(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            nid: self.nid.clone(),
            name: self.name.clone(),
        }
    }
}

fn print_game_summary(coverage: &GameCoverage, full: bool) {
    println!("== {} ({}) ==", coverage.title, coverage.id);
    if coverage.status == "unreadable" {
        let reason = coverage.error.as_deref().unwrap_or("unknown");
        println!("  UNREADABLE: {reason}");
        for problem in &coverage.module_problems {
            println!("    {}: {}", problem.module, problem.error);
        }
        return;
    }
    println!(
        "  modules: {} parsed, {} NEEDED not on disk, {} failed{}",
        coverage.modules_parsed,
        coverage.needed_not_on_disk.len(),
        coverage.module_problems.len(),
        if coverage.status == "partial" {
            " [partial]"
        } else {
            ""
        }
    );
    println!(
        "  unique imports: {} | HLE {} | LLE {} | unresolved {}",
        coverage.imports_unique, coverage.resolved_hle, coverage.resolved_lle, coverage.unresolved
    );
    println!(
        "  unresolved: {} dictionary-named, {} anonymous",
        coverage.unresolved_named, coverage.unresolved_anonymous
    );
    let mut providers: Vec<(&String, &usize)> = coverage.unresolved_by_provider.iter().collect();
    providers.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let top: Vec<String> = providers
        .iter()
        .take(8)
        .map(|(provider, count)| format!("{provider} {count}"))
        .collect();
    println!("  top unresolved providers: {}", top.join(" | "));
    println!(
        "  RENDER PATH: {} imported, {} resolved, {} unresolved",
        coverage.graphics.imported,
        coverage.graphics.resolved,
        coverage.graphics.unresolved.len()
    );
    for missing in &coverage.graphics.unresolved {
        println!(
            "    {} {} {}",
            missing.provider,
            missing.nid,
            missing.name.as_deref().unwrap_or("<anonymous>")
        );
    }
    if !coverage.anonymous.is_empty() {
        println!("  anonymous NIDs (dictionary-fill targets):");
        for missing in &coverage.anonymous {
            println!("    {} {}", missing.provider, missing.nid);
        }
    }
    if full {
        println!("  every unresolved import:");
        for missing in &coverage.unresolved_full {
            println!(
                "    {} {} {}",
                missing.provider,
                missing.nid,
                missing.name.as_deref().unwrap_or("<anonymous>")
            );
        }
    }
}

pub fn coverage(args: &[String]) -> Result<()> {
    let full = has(args, "--full");
    let games: Vec<GameRecord> = if let Some(eboot) = option(args, "--eboot") {
        let path = PathBuf::from(&eboot);
        if !path.exists() {
            bail!("--eboot {eboot} does not exist");
        }
        let hash = sha1_file(&path).with_context(|| format!("hash {eboot}"))?;
        let title = path
            .parent()
            .and_then(|parent| parent.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "single eboot".into());
        vec![GameRecord {
            id: format!("sha1-{}", &hash[..12]),
            title,
            content_sha1: hash,
            executable_bytes: fs::metadata(&path)?.len(),
            relative_hint: "eboot.bin".into(),
            local_path: Some(eboot),
            aliases: Vec::new(),
            tags: Vec::new(),
        }]
    } else {
        let registry_path =
            PathBuf::from(option(args, "--registry").unwrap_or_else(|| DEFAULT_REGISTRY.into()));
        let registry: Registry = read_json(&registry_path)?;
        registry.games
    };

    let hle = HleRegistry::new();
    let registered_but_not_implemented = hle
        .incomplete_registrations()
        .into_iter()
        .map(|(library, function, reason)| IncompleteRegistration {
            library,
            function,
            reason,
        })
        .collect::<Vec<_>>();
    let mut coverages = Vec::new();
    for game in &games {
        coverages.push(analyze_game(game, &hle, full));
    }

    // Union of unresolved imports across analyzable games — the priority list.
    let mut union: BTreeMap<(String, u64), (Option<String>, BTreeSet<String>)> = BTreeMap::new();
    for coverage in &coverages {
        if coverage.status == "unreadable" {
            continue;
        }
        for missing in &coverage.unresolved_full {
            let nid = u64::from_str_radix(missing.nid.trim_start_matches("0x"), 16)
                .expect("nid was formatted as hex");
            let entry = union
                .entry((missing.provider.clone(), nid))
                .or_insert_with(|| (missing.name.clone(), BTreeSet::new()));
            entry.1.insert(coverage.title.clone());
        }
    }
    let mut union_unresolved: Vec<UnionMissing> = union
        .into_iter()
        .map(|((provider, nid), (name, games))| UnionMissing {
            provider,
            nid: hex_nid(nid),
            name,
            games: games.into_iter().collect(),
        })
        .collect();
    union_unresolved.sort_by(|a, b| {
        b.games
            .len()
            .cmp(&a.games.len())
            .then(a.provider.cmp(&b.provider))
            .then(a.nid.cmp(&b.nid))
    });

    let analyzable = coverages
        .iter()
        .filter(|coverage| coverage.status != "unreadable")
        .count();
    println!();
    println!("== UNION across {analyzable} analyzable games ==");
    println!("  unique unresolved imports: {}", union_unresolved.len());
    let shared = union_unresolved
        .iter()
        .filter(|missing| missing.games.len() >= 2)
        .count();
    println!("  shared by 2+ games: {shared}");
    println!("  top priorities:");
    for missing in union_unresolved.iter().take(30) {
        println!(
            "    [{:2} games] {} {} {}",
            missing.games.len(),
            missing.provider,
            missing.nid,
            missing.name.as_deref().unwrap_or("<anonymous>")
        );
    }
    let union_render: Vec<&UnionMissing> = union_unresolved
        .iter()
        .filter(|missing| is_render_path(&missing.provider))
        .collect();
    println!(
        "  RENDER PATH unresolved union: {} (what the library needs to render)",
        union_render.len()
    );
    for missing in &union_render {
        println!(
            "    [{:2} games] {} {} {}",
            missing.games.len(),
            missing.provider,
            missing.nid,
            missing.name.as_deref().unwrap_or("<anonymous>")
        );
    }

    println!();
    println!(
        "== REGISTERED BUT NOT FULLY IMPLEMENTED ({}) ==",
        registered_but_not_implemented.len()
    );
    for row in &registered_but_not_implemented {
        println!("  {}::{} — {}", row.library, row.function, row.reason);
    }

    let report = CoverageReport {
        schema_version: 2,
        generated_unix_ms: now_ms(),
        build_revision: git_output(&["rev-parse", "--short=12", "HEAD"])
            .unwrap_or_else(|_| "unknown".into()),
        dictionary_entries: nid_names::len(),
        hle_registered: hle.registered_names().len(),
        registered_but_not_implemented,
        games: coverages,
        union_unresolved,
    };
    let output = PathBuf::from(
        option(args, "--output").unwrap_or_else(|| "artifacts/compat/nid-coverage.json".into()),
    );
    write_json(&output, &report)?;
    println!();
    println!("wrote {}", output.display());
    Ok(())
}
