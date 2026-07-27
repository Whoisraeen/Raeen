use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub id: String,
    pub title: String,
    pub content_sha1: String,
    pub executable_bytes: u64,
    pub relative_hint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_path: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    pub schema_version: u32,
    pub generated_unix_ms: u128,
    pub roots: Vec<String>,
    pub games: Vec<GameRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Detected,
    Launching,
    Rendering,
    Crashed,
    TimedOut,
    Exited,
    Refused,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub wall_ms: u128,
    pub cpu_ms: Option<u128>,
    pub peak_working_set_bytes: Option<u64>,
    pub exit_code: Option<i32>,
    pub flip_events: u64,
    pub shader_errors: u64,
    pub gpu_errors: u64,
    pub audio_errors: u64,
    pub input_events: u64,
    pub observed_fps: Option<f64>,
}

/// One unresolved import the guest actually *called* during a run, harvested
/// from the runtime's `UNRESOLVED NID CALLED` first-occurrence log lines.
/// Static coverage (`nids coverage`) says what could be missing; this says
/// what the title needed on this boot path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct UnresolvedNid {
    pub library: String,
    pub nid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub log_sha1: String,
    pub blocker_signature: Option<String>,
    pub first_blocker: Option<String>,
    pub measured: bool,
    /// `None` = the run predates NID harvesting (field absent in the JSON);
    /// `Some(empty)` = measured, and zero unresolved NIDs were called. The
    /// distinction keeps `baseline diff` honest: it must never report "all
    /// NIDs resolved" against a baseline that simply never measured them.
    /// Additive and optional, so schema_version stays 1 and every existing
    /// report round-trips byte-identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unresolved_nids: Option<Vec<UnresolvedNid>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatResult {
    pub schema_version: u32,
    pub measured_unix_ms: u128,
    pub run_id: String,
    pub build_revision: String,
    pub profile: String,
    pub game_id: String,
    pub title: String,
    pub content_sha1: String,
    pub stage: Stage,
    pub metrics: Metrics,
    pub evidence: Evidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    pub schema_version: u32,
    pub generated_unix_ms: u128,
    pub machine_id: String,
    pub results: Vec<CompatResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub name: String,
    pub directory: String,
    pub upstream_branch: String,
    pub baseline_revision: String,
    pub license: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceState {
    pub schema_version: u32,
    pub references: Vec<Reference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceResult {
    pub name: String,
    pub area: String,
    pub required: bool,
    pub passed: bool,
    pub wall_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceReport {
    pub schema_version: u32,
    pub measured_unix_ms: u128,
    pub build_revision: String,
    pub results: Vec<AcceptanceResult>,
}
