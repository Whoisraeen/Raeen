//! Content-addressed commercial shader failure corpus.

use anyhow::{Context, Result, bail};
use raeen_gpu::{ShaderReplayInput, replay_corpus_shader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

pub const DEFAULT_CORPUS_DIR: &str = "artifacts/shader-corpus";
const CORPUS_SCHEMA_VERSION: u32 = 2;
const REPLAY_SCHEMA_VERSION: u32 = 1;
const REPLAY_INDEX_SCHEMA_VERSION: u32 = 1;
const REPLAY_INDEX_FILE: &str = "replay-index.json";

pub fn run(command: &str, args: &[String]) -> Result<()> {
    match command {
        "capture" => capture(args),
        "report" => report(args),
        "replay" => replay(args),
        _ => {
            bail!("unknown shader-corpus command {command:?}; expected capture, report, or replay")
        }
    }
}

fn capture(args: &[String]) -> Result<()> {
    let mut compat_args = args.to_vec();
    if super::option(&compat_args, "--shader-corpus").is_none() {
        compat_args.push("--shader-corpus".into());
        compat_args.push(DEFAULT_CORPUS_DIR.into());
    }
    super::compat_run(&compat_args)
}

#[derive(Clone, Debug, Deserialize)]
struct FailureRecord {
    schema_version: u32,
    shader_sha1: String,
    stage: String,
    failure_kind: String,
    reason: String,
    binary: String,
    fetched_bytes: usize,
    #[serde(default)]
    binding_identity: Option<Vec<u32>>,
    #[serde(default)]
    replay_input: Option<ShaderReplayInput>,
    title: String,
    game_id: String,
    run_id: String,
    build_revision: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FailureClass {
    family: String,
    opcode: String,
    form: String,
}

#[derive(Default)]
struct Cluster {
    occurrences: usize,
    shaders: BTreeSet<String>,
    titles: BTreeSet<String>,
    example_reason: String,
}

fn report(args: &[String]) -> Result<()> {
    let corpus = corpus_path(args);
    let records = read_records(&corpus)?;
    let mut clusters: BTreeMap<(String, FailureClass), Cluster> = BTreeMap::new();
    for record in &records {
        let key = (record.stage.clone(), classify_reason(&record.reason));
        let cluster = clusters.entry(key).or_default();
        cluster.occurrences += 1;
        cluster.shaders.insert(record.shader_sha1.clone());
        cluster.titles.insert(record.title.clone());
        if cluster.example_reason.is_empty() {
            cluster.example_reason.clone_from(&record.reason);
        }
    }
    let mut ranked: Vec<_> = clusters.into_iter().collect();
    ranked.sort_by(|(left_key, left), (right_key, right)| {
        right
            .titles
            .len()
            .cmp(&left.titles.len())
            .then_with(|| right.shaders.len().cmp(&left.shaders.len()))
            .then_with(|| right.occurrences.cmp(&left.occurrences))
            .then_with(|| left_key.cmp(right_key))
    });

    let unique_shaders = records
        .iter()
        .map(|record| &record.shader_sha1)
        .collect::<BTreeSet<_>>()
        .len();
    let titles = records
        .iter()
        .map(|record| &record.title)
        .collect::<BTreeSet<_>>()
        .len();
    let markdown = render_report(&ranked, records.len(), unique_shaders, titles);
    let output = PathBuf::from(
        super::option(args, "--output")
            .unwrap_or_else(|| corpus.join("report.md").display().to_string()),
    );
    write_file(&output, markdown.as_bytes())?;
    let event_count = records.len();
    let replay_cases = build_replay_cases(records);
    write_json(
        &corpus.join(REPLAY_INDEX_FILE),
        &ReplayIndex {
            schema_version: REPLAY_INDEX_SCHEMA_VERSION,
            event_count,
            cases: replay_cases,
        },
    )?;
    print!("{markdown}");
    println!("wrote ranked shader failure report to {}", output.display());
    Ok(())
}

fn render_report(
    ranked: &[((String, FailureClass), Cluster)],
    occurrences: usize,
    unique_shaders: usize,
    titles: usize,
) -> String {
    let mut out = format!(
        "# Shader failure corpus\n\n{} occurrence(s), {} unique shader(s), {} title(s), {} cluster(s).\n\n\
         | Rank | Stage | Family | Opcode | Operand/encoding form | Titles | Shaders | Occurrences |\n\
         |---:|---|---|---|---|---:|---:|---:|\n",
        occurrences,
        unique_shaders,
        titles,
        ranked.len()
    );
    for (index, ((stage, class), cluster)) in ranked.iter().enumerate() {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            index + 1,
            markdown_cell(stage),
            markdown_cell(&class.family),
            markdown_cell(&class.opcode),
            markdown_cell(&class.form),
            cluster.titles.len(),
            cluster.shaders.len(),
            cluster.occurrences,
        ));
    }
    out
}

fn markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace(['\r', '\n'], " ")
}

fn classify_reason(reason: &str) -> FailureClass {
    let lower = reason.to_ascii_lowercase();
    let mut family = [
        "mubuf", "mtbuf", "mimg", "smem", "ds", "flat", "global", "sopk", "sop2", "sopc", "sop1",
        "vop3p", "vop3", "vop2", "vop1", "vopc", "exp",
    ]
    .into_iter()
    .find(|family| {
        lower.contains(&format!("unknown {family} "))
            || lower.contains(&format!(" {family} instruction"))
            || lower.contains(&format!(" {family} opcode"))
    })
    .unwrap_or("unknown")
    .to_string();

    let opcode_hex = hex_value_after(&lower, "opcode");
    let opcode = extract_after(&lower, &format!("unknown {family} instruction "))
        .map(|tail| token_until(tail, &[',', ':', ' ', '[']))
        .filter(|value| !value.is_empty())
        .or_else(|| {
            extract_after(reason, "Recompile_")
                .map(|tail| token_until(tail, &['_', ':', ' ', '[']))
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            extract_after(reason, "no lowering table entry:")
                .map(str::trim_start)
                .map(|tail| token_until(tail, &[' ', '[', ':']))
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            extract_after(reason, "no recompiler for ")
                .map(|tail| token_until(tail, &['/', ')', ':', ' ', '[']))
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            extract_after(reason, "can't recompile:")
                .map(str::trim_start)
                .map(|tail| token_until(tail, &[' ', '[', ':']))
                .filter(|value| !value.is_empty())
        })
        .or_else(|| opcode_hex.as_ref().map(|value| format!("opcode_{value}")))
        .unwrap_or_else(|| "unknown".into());

    if family == "unknown" {
        let opcode_lower = opcode.to_ascii_lowercase();
        family = if opcode_lower.starts_with("image") {
            "mimg"
        } else if opcode_lower.starts_with("bufferloadformat")
            || opcode_lower.starts_with("bufferstoreformat")
        {
            "mtbuf"
        } else if opcode_lower.starts_with("buffer") {
            "mubuf"
        } else if opcode_lower.starts_with("sload") || opcode_lower.starts_with("sbuffer") {
            "smem"
        } else if opcode_lower.starts_with("ds") {
            "ds"
        } else if opcode_lower.starts_with('v') {
            "valu"
        } else if opcode_lower.starts_with('s') {
            "salu"
        } else {
            "unknown"
        }
        .into();
    }

    let mut form = Vec::new();
    if let Some(value) = opcode_hex {
        form.push(format!("opcode={value}"));
    }
    if let Some(value) = hex_value_after(&lower, "raw") {
        form.push(format!("raw={value}"));
    }
    if let Some(open) = reason.find('[')
        && let Some(close) = reason[open + 1..].find(']')
    {
        let value = reason[open + 1..open + 1 + close].trim();
        if !value.is_empty() {
            form.push(format!("operands={value}"));
        }
    }

    FailureClass {
        family,
        opcode,
        form: if form.is_empty() {
            "unspecified".into()
        } else {
            form.join(" ")
        },
    }
}

fn extract_after<'a>(text: &'a str, marker: &str) -> Option<&'a str> {
    let index = text.find(marker)?;
    Some(&text[index + marker.len()..])
}

fn token_until(text: &str, delimiters: &[char]) -> String {
    text.trim_start()
        .split(|ch| delimiters.contains(&ch))
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn hex_value_after(text: &str, marker: &str) -> Option<String> {
    let tail = extract_after(text, marker)?;
    let start = tail.find("0x")?;
    let value: String = tail[start + 2..]
        .chars()
        .take_while(|ch| ch.is_ascii_hexdigit())
        .collect();
    (!value.is_empty()).then(|| format!("0x{}", value.to_ascii_lowercase()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayResult {
    case_id: String,
    shader_sha1: String,
    stage: String,
    titles: Vec<String>,
    passed: bool,
    outcome: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayReport {
    schema_version: u32,
    elapsed_ms: u128,
    results: Vec<ReplayResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayCase {
    case_id: String,
    shader_sha1: String,
    stage: String,
    binary: String,
    replay_input: Option<ShaderReplayInput>,
    titles: BTreeSet<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReplayIndex {
    schema_version: u32,
    event_count: usize,
    cases: Vec<ReplayCase>,
}

fn replay(args: &[String]) -> Result<()> {
    let started = Instant::now();
    let corpus = corpus_path(args);
    let cases = load_or_build_replay_index(&corpus)?;
    let mut binaries = BTreeMap::new();
    for case in &cases {
        if !binaries.contains_key(&case.shader_sha1) {
            binaries.insert(
                case.shader_sha1.clone(),
                read_case_binary(&corpus, &case.binary, &case.shader_sha1)?,
            );
        }
    }

    let output = PathBuf::from(
        super::option(args, "--output")
            .unwrap_or_else(|| corpus.join("replay-latest.json").display().to_string()),
    );
    let previous = if output.exists() {
        Some(read_json::<ReplayReport>(&output)?)
    } else {
        None
    };
    let mut results: Vec<_> = cases
        .into_par_iter()
        .map(|case| {
            let bytes = binaries
                .get(&case.shader_sha1)
                .expect("replay object cache was populated for every case");
            let replayed = replay_corpus_shader(&case.stage, bytes, case.replay_input.as_ref());
            ReplayResult {
                case_id: case.case_id,
                shader_sha1: case.shader_sha1,
                stage: case.stage,
                titles: case.titles.into_iter().collect(),
                passed: replayed.is_ok(),
                outcome: replayed.unwrap_or_else(|error| error),
            }
        })
        .collect();
    results.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    let report = ReplayReport {
        schema_version: REPLAY_SCHEMA_VERSION,
        elapsed_ms: started.elapsed().as_millis(),
        results,
    };

    let previous_states: BTreeMap<_, _> = previous
        .as_ref()
        .map(|report| {
            report
                .results
                .iter()
                .map(|result| (result.case_id.as_str(), result.passed))
                .collect()
        })
        .unwrap_or_default();
    let improved = report
        .results
        .iter()
        .filter(|result| {
            previous_states.get(result.case_id.as_str()) == Some(&false) && result.passed
        })
        .count();
    let regressed = report
        .results
        .iter()
        .filter(|result| {
            previous_states.get(result.case_id.as_str()) == Some(&true) && !result.passed
        })
        .count();
    let passed = report.results.iter().filter(|result| result.passed).count();
    let failed = report.results.len() - passed;
    write_json(&output, &report)?;
    println!(
        "replayed {} shader case(s) in {:.3}s: {} passed, {} failed, {} improved, {} regressed",
        report.results.len(),
        report.elapsed_ms as f64 / 1000.0,
        passed,
        failed,
        improved,
        regressed
    );
    for result in report
        .results
        .iter()
        .filter(|result| !result.passed)
        .take(20)
    {
        println!(
            "  FAIL {} {} titles={} — {}",
            result.stage,
            result.shader_sha1,
            result.titles.len(),
            result.outcome
        );
    }
    println!("wrote shader replay report to {}", output.display());
    if super::has(args, "--strict") && regressed != 0 {
        bail!("shader corpus replay regressed {regressed} previously passing case(s)");
    }
    Ok(())
}

fn build_replay_cases(records: Vec<FailureRecord>) -> Vec<ReplayCase> {
    let mut cases: BTreeMap<String, ReplayCase> = BTreeMap::new();
    for record in records {
        let case_id = replay_case_id(&record);
        cases
            .entry(case_id.clone())
            .and_modify(|case| {
                case.titles.insert(record.title.clone());
            })
            .or_insert_with(|| ReplayCase {
                case_id,
                shader_sha1: record.shader_sha1,
                stage: record.stage,
                binary: record.binary,
                replay_input: record.replay_input,
                titles: BTreeSet::from([record.title]),
            });
    }
    cases.into_values().collect()
}

fn replay_event_count(corpus: &Path) -> Result<usize> {
    let events = corpus.join("events");
    Ok(fs::read_dir(&events)
        .with_context(|| format!("read shader corpus events at {}", events.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .count())
}

fn load_or_build_replay_index(corpus: &Path) -> Result<Vec<ReplayCase>> {
    let path = corpus.join(REPLAY_INDEX_FILE);
    let event_count = replay_event_count(corpus)?;
    if path.exists() {
        let index: ReplayIndex = read_json(&path)?;
        if index.schema_version == REPLAY_INDEX_SCHEMA_VERSION && index.event_count == event_count {
            return Ok(index.cases);
        }
    }

    let cases = build_replay_cases(read_records(corpus)?);
    write_json(
        &path,
        &ReplayIndex {
            schema_version: REPLAY_INDEX_SCHEMA_VERSION,
            event_count,
            cases: cases.clone(),
        },
    )?;
    println!(
        "rebuilt compact replay index from {event_count} corpus event(s): {} exact case(s)",
        cases.len()
    );
    Ok(cases)
}

fn replay_case_id(record: &FailureRecord) -> String {
    let mut hasher = Sha1::new();
    hasher.update(record.shader_sha1.as_bytes());
    hasher.update([0]);
    hasher.update(record.stage.as_bytes());
    hasher.update([0]);
    if let Some(binding) = &record.binding_identity {
        for value in binding {
            hasher.update(value.to_le_bytes());
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn corpus_path(args: &[String]) -> PathBuf {
    PathBuf::from(super::option(args, "--corpus").unwrap_or_else(|| DEFAULT_CORPUS_DIR.into()))
}

fn read_records(corpus: &Path) -> Result<Vec<FailureRecord>> {
    let events = corpus.join("events");
    let mut paths: Vec<_> = fs::read_dir(&events)
        .with_context(|| format!("read shader corpus events at {}", events.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("shader corpus has no events at {}", events.display());
    }
    let records: Vec<FailureRecord> = paths
        .par_iter()
        .map(|path| {
            let record: FailureRecord = read_json(path)?;
            validate_record_metadata(&record)
                .with_context(|| format!("validate corpus event {}", path.display()))?;
            Ok(record)
        })
        .collect::<Result<_>>()?;

    let mut objects: BTreeMap<&str, (&str, usize)> = BTreeMap::new();
    for record in &records {
        if let Some((binary, fetched_bytes)) = objects.get(record.shader_sha1.as_str()) {
            if *binary != record.binary || *fetched_bytes != record.fetched_bytes {
                bail!(
                    "shader object {} has conflicting corpus declarations",
                    record.shader_sha1
                );
            }
        } else {
            objects.insert(
                record.shader_sha1.as_str(),
                (record.binary.as_str(), record.fetched_bytes),
            );
        }
    }
    for (shader_sha1, (binary, fetched_bytes)) in objects {
        let bytes = read_case_binary(corpus, binary, shader_sha1)?;
        if bytes.len() != fetched_bytes {
            bail!(
                "recorded fetched length {fetched_bytes} does not match object length {}",
                bytes.len()
            );
        }
    }
    Ok(records)
}

fn validate_record_metadata(record: &FailureRecord) -> Result<()> {
    if record.schema_version != CORPUS_SCHEMA_VERSION {
        bail!("unsupported corpus schema {}", record.schema_version);
    }
    if !matches!(record.stage.as_str(), "vs" | "ps" | "cs") {
        bail!("unknown shader stage {:?}", record.stage);
    }
    if !matches!(record.failure_kind.as_str(), "analysis" | "translation") {
        bail!("unknown shader failure kind {:?}", record.failure_kind);
    }
    match (record.failure_kind.as_str(), &record.replay_input) {
        ("analysis", None) => {}
        ("analysis", Some(_)) => bail!("analysis failure unexpectedly contains a replay ABI"),
        ("translation", None) => bail!("translation failure is missing its exact replay ABI"),
        ("translation", Some(input)) => {
            let input_stage = match input {
                ShaderReplayInput::Vs(_) => "vs",
                ShaderReplayInput::Ps(_) => "ps",
                ShaderReplayInput::Cs(_) => "cs",
            };
            if input_stage != record.stage {
                bail!(
                    "captured replay ABI stage {input_stage:?} does not match event stage {:?}",
                    record.stage
                );
            }
            if record.binding_identity.is_none() {
                bail!("translation failure is missing its binding identity");
            }
        }
        _ => unreachable!("failure kind was validated above"),
    }
    if record.shader_sha1.len() != 40
        || !record
            .shader_sha1
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        bail!("invalid shader SHA-1 {:?}", record.shader_sha1);
    }
    if record.title.is_empty()
        || record.game_id.is_empty()
        || record.run_id.is_empty()
        || record.build_revision.is_empty()
        || record.reason.is_empty()
    {
        bail!("corpus event has empty provenance or reason");
    }
    Ok(())
}

fn read_case_binary(corpus: &Path, relative: &str, expected_sha1: &str) -> Result<Vec<u8>> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("unsafe corpus object path {}", relative.display());
    }
    let expected = PathBuf::from("objects").join(format!("{expected_sha1}.bin"));
    if relative != expected {
        bail!(
            "corpus object path {} does not match hash-owned path {}",
            relative.display(),
            expected.display()
        );
    }
    let path = corpus.join(relative);
    let bytes =
        fs::read(&path).with_context(|| format!("read corpus object {}", path.display()))?;
    let actual = corpus_sha1(&bytes);
    if actual != expected_sha1 {
        bail!(
            "corpus object {} hash mismatch: expected {}, got {}",
            path.display(),
            expected_sha1,
            actual
        );
    }
    Ok(bytes)
}

fn corpus_sha1(bytes: &[u8]) -> String {
    hex_digest(Sha1::digest(bytes).as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    write_file(path, &bytes)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_removes_addresses_but_keeps_mubuf_opcode_and_encoding() {
        let first = classify_reason(
            "cs shader at 0x10068012700 (4096 bytes fetched): next_gen: \
             shader_parse_cs: unknown mubuf instruction buffer_store_dwordx3, \
             opcode = 0x1f at addr 0x00000294 (raw 0xe07c2000)",
        );
        let relocated = classify_reason(
            "cs shader at 0x148d47200 (8192 bytes fetched): next_gen: \
             shader_parse_cs: unknown mubuf instruction buffer_store_dwordx3, \
             opcode = 0x1f at addr 0x00000910 (raw 0xe07c2000)",
        );
        assert_eq!(first, relocated);
        assert_eq!(first.family, "mubuf");
        assert_eq!(first.opcode, "buffer_store_dwordx3");
        assert_eq!(first.form, "opcode=0x1f raw=0xe07c2000");
    }

    #[test]
    fn classifier_names_recompiler_function_instead_of_the_stage_wrapper() {
        let class = classify_reason(
            "cs shader at 0x154dfb4000 (4096 bytes fetched): shader_recompile_cs: \
             Recompile_SLoadDwordx8_Sdst8SbaseSoffset: not supported: unresolved \
             register soffset: SLoadDwordx8 [Sdst8SbaseSoffset] x8 dwords",
        );
        assert_eq!(class.family, "smem");
        assert_eq!(class.opcode, "SLoadDwordx8");
        assert_eq!(class.form, "operands=Sdst8SbaseSoffset");
    }

    #[test]
    fn classifier_names_missing_recompiler_opcode_and_mtbuf_family() {
        let class = classify_reason(
            "vs shader at 0x4020031500 (4096 bytes fetched): shader_recompile_vs: \
             can't recompile (no recompiler for BufferLoadFormatXyz/\
             Vdata3VaddrSvSoffsIdxen): BufferLoadFormatXyz \
             [Vdata3VaddrSvSoffsIdxen] v[14:16], v5, s[12:15], 0, idxen",
        );
        assert_eq!(class.family, "mtbuf");
        assert_eq!(class.opcode, "BufferLoadFormatXyz");
        assert_eq!(class.form, "operands=Vdata3VaddrSvSoffsIdxen");
    }

    #[test]
    fn report_ranks_cross_title_fanout_before_occurrence_volume() {
        let broad = Cluster {
            occurrences: 2,
            shaders: BTreeSet::from(["a".into(), "b".into()]),
            titles: BTreeSet::from(["Avatar".into(), "Subnautica".into()]),
            example_reason: "broad".into(),
        };
        let noisy = Cluster {
            occurrences: 10_000,
            shaders: BTreeSet::from(["c".into()]),
            titles: BTreeSet::from(["Avatar".into()]),
            example_reason: "noisy".into(),
        };
        let mut ranked: Vec<((String, FailureClass), Cluster)> = vec![
            (
                (
                    "cs".into(),
                    classify_reason("unknown ds instruction ds_write_b32"),
                ),
                broad,
            ),
            (
                ("ps".into(), classify_reason("unknown mimg opcode: 0x45")),
                noisy,
            ),
        ];
        ranked.sort_by(|(left_key, left), (right_key, right)| {
            right
                .titles
                .len()
                .cmp(&left.titles.len())
                .then_with(|| right.shaders.len().cmp(&left.shaders.len()))
                .then_with(|| right.occurrences.cmp(&left.occurrences))
                .then_with(|| left_key.cmp(right_key))
        });
        assert_eq!(ranked[0].1.example_reason, "broad");
    }

    #[test]
    fn replay_translates_a_known_compute_program_to_valid_spirv() {
        let words = [0x7E00_0280u32, 0x7E02_0280, 0xBF81_0000];
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let outcome = replay_corpus_shader("cs", &bytes, None).expect("known compute program");
        assert!(outcome.contains("validated SPIR-V"), "{outcome}");
    }

    #[test]
    fn replay_index_deduplicates_provenance_but_preserves_structural_bindings() {
        let record = |title: &str, binding_identity: &[u32]| FailureRecord {
            schema_version: CORPUS_SCHEMA_VERSION,
            shader_sha1: "0123456789abcdef0123456789abcdef01234567".into(),
            stage: "cs".into(),
            failure_kind: "translation".into(),
            reason: "missing lowering".into(),
            binary: "objects/0123456789abcdef0123456789abcdef01234567.bin".into(),
            fetched_bytes: 12,
            binding_identity: Some(binding_identity.to_vec()),
            replay_input: Some(ShaderReplayInput::Cs(Default::default())),
            title: title.into(),
            game_id: format!("{title}-id"),
            run_id: format!("{title}-run"),
            build_revision: "test-build".into(),
        };

        let cases = build_replay_cases(vec![
            record("Avatar", &[1, 2, 3]),
            record("Subnautica", &[1, 2, 3]),
            record("Avatar", &[1, 2, 4]),
        ]);
        assert_eq!(cases.len(), 2, "different titles are provenance, not cases");
        assert!(
            cases.iter().any(|case| {
                case.titles == BTreeSet::from(["Avatar".into(), "Subnautica".into()])
            })
        );
    }
}
