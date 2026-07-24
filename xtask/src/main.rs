mod schema;

use anyhow::{Context, Result, anyhow, bail};
use schema::{
    AcceptanceReport, AcceptanceResult, CompatResult, Evidence, GameRecord, Metrics,
    ReferenceState, Registry, RunReport, SCHEMA_VERSION, Stage,
};
use serde::Deserialize;
use sha1::{Digest, Sha1};
use std::{
    collections::BTreeMap,
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DEFAULT_REGISTRY: &str = "artifacts/compat/registry.json";
const DEFAULT_RESULTS: &str = "artifacts/compat/latest.json";

#[derive(Deserialize)]
struct LocalConfig {
    #[serde(default)]
    paths: LocalPaths,
}

#[derive(Default, Deserialize)]
struct LocalPaths {
    #[serde(default)]
    game_folders: Vec<String>,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let area = args.first().map(String::as_str).unwrap_or("--help");
    let command = args.get(1).map(String::as_str).unwrap_or("");
    let rest = args.get(2..).unwrap_or_default();
    if area == "compat" && command == "discover" {
        compat_discover(rest)
    } else if area == "compat" && command == "run" {
        compat_run(rest)
    } else if area == "compat" && command == "publish" {
        compat_publish(rest)
    } else if area == "compat" && command == "compare" {
        compat_compare(rest)
    } else if area == "refs" && command == "report" {
        refs_report(rest)
    } else if area == "acceptance" && command == "run" {
        acceptance_run(rest)
    } else if area == "--help" || area == "-h" {
        print_help();
        Ok(())
    } else {
        bail!("unknown command; run `cargo xtask --help`")
    }
}

fn print_help() {
    println!(
        "Raeen development workflows

  cargo xtask compat discover [--config config.toml] [--library PATH] [--output PATH]
  cargo xtask compat run [--registry PATH] [--output PATH] [--exe PATH]
                          [--timeout SECONDS] [--tier all|nightly] [--profile max-fps]
  cargo xtask compat compare --baseline PATH [--current PATH]
  cargo xtask compat publish [--input PATH] [--output compat/COMPATIBILITY.md]
  cargo xtask refs report [--state compat/reference-state.json] [--output PATH] [--fetch]
  cargo xtask acceptance run [--output PATH] [--timeout SECONDS]

Raw logs, executable paths, and local machine details stay under gitignored artifacts/.
Only `compat publish` emits a sanitized, measured compatibility table."
    );
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn sha1_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha1::new();
    // Heap allocation keeps the Windows xtask entry-point stack small even
    // when the optimized dev profile inlines discovery into command dispatch.
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha1_bytes(value: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}

fn compat_discover(args: &[String]) -> Result<()> {
    let config_path = PathBuf::from(option(args, "--config").unwrap_or("config.toml".into()));
    let mut roots = Vec::new();
    if config_path.exists() {
        let config: LocalConfig = toml::from_str(&fs::read_to_string(&config_path)?)
            .with_context(|| format!("parse {}", config_path.display()))?;
        roots.extend(config.paths.game_folders.into_iter().map(PathBuf::from));
    }
    if let Some(root) = option(args, "--library") {
        roots.push(PathBuf::from(root));
    }
    roots.retain(|root| root.exists());
    roots.sort();
    roots.dedup();
    if roots.is_empty() {
        bail!("no existing game roots found in config or --library");
    }

    let mut candidates = Vec::new();
    for root in &roots {
        find_eboots(root, 0, 6, &mut candidates)?;
    }
    candidates.sort();
    candidates.dedup();

    let mut by_hash: BTreeMap<String, GameRecord> = BTreeMap::new();
    for path in candidates {
        let hash = sha1_file(&path).with_context(|| format!("hash {}", path.display()))?;
        let root = roots
            .iter()
            .find(|root| path.starts_with(root))
            .expect("discovered path has root");
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let title = title_from_relative(relative);
        let relative_hint = sanitize_relative_hint(relative);
        let id = title_id(relative).unwrap_or_else(|| format!("sha1-{}", &hash[..12]));
        let tags = tags_for(&title, &id);
        let size = fs::metadata(&path)?.len();
        let alias = relative.to_string_lossy().replace('\\', "/");
        match by_hash.get_mut(&hash) {
            Some(record) => record.aliases.push(alias),
            None => {
                by_hash.insert(
                    hash.clone(),
                    GameRecord {
                        id,
                        title,
                        content_sha1: hash,
                        executable_bytes: size,
                        relative_hint,
                        local_path: Some(path.to_string_lossy().into_owned()),
                        aliases: Vec::new(),
                        tags,
                    },
                );
            }
        }
    }

    let registry = Registry {
        schema_version: SCHEMA_VERSION,
        generated_unix_ms: now_ms(),
        roots: roots
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        games: by_hash.into_values().collect(),
    };
    let output = PathBuf::from(option(args, "--output").unwrap_or_else(|| DEFAULT_REGISTRY.into()));
    write_json(&output, &registry)?;
    println!(
        "registered {} unique games in {} ({} duplicate images)",
        registry.games.len(),
        output.display(),
        registry
            .games
            .iter()
            .map(|game| game.aliases.len())
            .sum::<usize>()
    );
    Ok(())
}

fn find_eboots(path: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    if depth > max_depth {
        return Ok(());
    }
    for entry in fs::read_dir(path).with_context(|| format!("scan {}", path.display()))? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            continue;
        }
        let child = entry.path();
        if kind.is_file()
            && entry
                .file_name()
                .eq_ignore_ascii_case(OsStr::new("eboot.bin"))
        {
            out.push(child);
        } else if kind.is_dir() {
            find_eboots(&child, depth + 1, max_depth, out)?;
        }
    }
    Ok(())
}

fn title_from_relative(relative: &Path) -> String {
    relative
        .components()
        .next()
        .map(|value| value.as_os_str().to_string_lossy().into_owned())
        .unwrap_or_else(|| "Unknown title".into())
}

fn sanitize_relative_hint(relative: &Path) -> String {
    let count = relative.components().count();
    if let Some(id) = title_id(relative) {
        format!("{id}/eboot.bin")
    } else {
        format!("depth-{}/eboot.bin", count.saturating_sub(1))
    }
}

fn title_id(path: &Path) -> Option<String> {
    for component in path.components() {
        let upper = component.as_os_str().to_string_lossy().to_ascii_uppercase();
        for (start, _) in upper.match_indices("PPSA") {
            let Some(tail) = upper.get(start..start + 9) else {
                continue;
            };
            if tail[4..].chars().all(|c| c.is_ascii_digit()) {
                return Some(tail.into());
            }
        }
    }
    None
}

fn tags_for(title: &str, id: &str) -> Vec<String> {
    let lower = title.to_ascii_lowercase();
    let mut tags = Vec::new();
    if lower.contains("astro") {
        tags.push("astro".into());
    }
    if lower.contains("minecraft") {
        tags.push("minecraft".into());
    }
    if lower.contains("until dawn") || lower.contains("dragon ball") {
        tags.push("ue5".into());
    }
    if lower.contains("subnautica") || lower.contains("plague tale") {
        tags.push("small".into());
    }
    if id.starts_with("PPSA") {
        tags.push("retail".into());
    }
    tags
}

fn compat_run(args: &[String]) -> Result<()> {
    let registry_path =
        PathBuf::from(option(args, "--registry").unwrap_or_else(|| DEFAULT_REGISTRY.into()));
    let registry: Registry = read_json(&registry_path)?;
    if registry.schema_version != SCHEMA_VERSION {
        bail!("unsupported registry schema {}", registry.schema_version);
    }
    let exe =
        PathBuf::from(option(args, "--exe").unwrap_or_else(|| "target/release/raeen.exe".into()));
    if !exe.exists() {
        bail!(
            "{} does not exist; run `cargo build --release -p raeen-gui` first",
            exe.display()
        );
    }
    let timeout_secs = option(args, "--timeout")
        .unwrap_or_else(|| "60".into())
        .parse::<u64>()
        .context("--timeout must be an integer")?;
    let tier = option(args, "--tier").unwrap_or_else(|| "all".into());
    let profile = option(args, "--profile").unwrap_or_else(|| "max-fps".into());
    let selected = select_games(&registry.games, &tier)?;
    let run_id = format!("run-{}", now_ms());
    let raw_dir = PathBuf::from("artifacts/compat/raw").join(&run_id);
    fs::create_dir_all(&raw_dir)?;
    let build_revision =
        git_output(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|_| "unknown".into());
    let machine_id = machine_id();
    let mut results = Vec::new();
    for game in selected {
        println!("measuring {} ({})", game.title, game.id);
        results.push(run_one(
            &exe,
            game,
            &profile,
            timeout_secs,
            &run_id,
            &build_revision,
            &raw_dir,
        )?);
    }
    let report = RunReport {
        schema_version: SCHEMA_VERSION,
        generated_unix_ms: now_ms(),
        machine_id,
        results,
    };
    let output = PathBuf::from(option(args, "--output").unwrap_or_else(|| DEFAULT_RESULTS.into()));
    write_json(&output, &report)?;
    println!(
        "wrote {} measured results to {}",
        report.results.len(),
        output.display()
    );
    Ok(())
}

fn select_games<'a>(games: &'a [GameRecord], tier: &str) -> Result<Vec<&'a GameRecord>> {
    if tier == "all" {
        return Ok(games.iter().collect());
    }
    if tier != "nightly" {
        bail!("unknown tier {tier}; expected all or nightly");
    }
    let mut selected = Vec::new();
    for tag in ["astro", "minecraft", "ue5", "small"] {
        if let Some(game) = games
            .iter()
            .filter(|game| game.tags.iter().any(|value| value == tag))
            .min_by_key(|game| game.executable_bytes)
            && !selected
                .iter()
                .any(|selected: &&GameRecord| selected.content_sha1 == game.content_sha1)
        {
            selected.push(game);
        }
    }
    if selected.len() < 4 {
        eprintln!(
            "warning: nightly tier resolved {} of 4 roles; run discovery after adding titles",
            selected.len()
        );
    }
    Ok(selected)
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    exe: &Path,
    game: &GameRecord,
    profile: &str,
    timeout_secs: u64,
    run_id: &str,
    build_revision: &str,
    raw_dir: &Path,
) -> Result<CompatResult> {
    let local_path = game
        .local_path
        .as_deref()
        .ok_or_else(|| anyhow!("registry {} has no local executable path", game.id))?;
    // A title ID is not a content identity: two installed revisions may share
    // PPSA while carrying different executables. Keep their raw evidence apart.
    let stem = format!("{}-{}", safe_name(&game.id), &game.content_sha1[..12]);
    let stdout_path = raw_dir.join(format!("{stem}.stdout.log"));
    let stderr_path = raw_dir.join(format!("{stem}.stderr.log"));
    let stdout = File::create(&stdout_path)?;
    let stderr = File::create(&stderr_path)?;
    let start = Instant::now();
    let mut command = Command::new(exe);
    command
        .arg("--run-eboot")
        .arg(local_path)
        .env("RAEEN_VBLANK_HZ", "1000")
        .env("RAEEN_TIME_DRAW", "1")
        .env("RAEEN_CALL_STATS", "1")
        .env("RAEEN_COMPAT_RUN_ID", run_id)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if profile != "max-fps" {
        command.env_remove("RAEEN_VBLANK_HZ");
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("launch {}", game.title))?;
    let sampler = ProcessSampler::open(child.id());
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if start.elapsed() >= Duration::from_secs(timeout_secs) {
            timed_out = true;
            child.kill().ok();
            break child.wait().ok();
        }
        thread::sleep(Duration::from_millis(100));
    };
    let wall_ms = start.elapsed().as_millis();
    let (cpu_ms, peak_working_set_bytes) = sampler.finish();
    let mut log = fs::read(&stdout_path)?;
    log.extend_from_slice(&fs::read(&stderr_path)?);
    // `tracing-subscriber` may insert an ANSI reset between a structured
    // field name and `=`. Strip it once before every metric/blocker scan so a
    // measured `total_flips=32` cannot be misreported as zero.
    let text = strip_ansi(&String::from_utf8_lossy(&log));
    let flip_events =
        max_metric(&text, "total_flips").max(count_any(&text, &["sceVideoOutSubmitFlip"]));
    let shader_errors = count_lines(&text, |line| {
        line.contains("shader") && (line.contains("ERROR") || line.contains("not supported"))
    });
    let gpu_errors = count_lines(&text, |line| {
        (line.contains("gpu") || line.contains("graphics")) && line.contains("ERROR")
    });
    let audio_errors = count_lines(&text, |line| {
        line.contains("audio") && line.contains("ERROR")
    });
    let input_events = count_any(&text, &["pad event", "controller connected"]);
    let blocker = first_blocker(&text, &registry_roots_from_path(local_path));
    let blocker_signature = blocker.as_ref().map(|value| sha1_bytes(value.as_bytes()));
    let stage = if timed_out {
        Stage::TimedOut
    } else if status.as_ref().is_some_and(|value| !value.success()) {
        Stage::Crashed
    } else if flip_events > 0 {
        Stage::Rendering
    } else {
        Stage::Exited
    };
    Ok(CompatResult {
        schema_version: SCHEMA_VERSION,
        measured_unix_ms: now_ms(),
        run_id: run_id.into(),
        build_revision: build_revision.trim().into(),
        profile: profile.into(),
        game_id: game.id.clone(),
        title: game.title.clone(),
        content_sha1: game.content_sha1.clone(),
        stage,
        metrics: Metrics {
            wall_ms,
            cpu_ms,
            peak_working_set_bytes,
            exit_code: status.and_then(|value| value.code()),
            flip_events,
            shader_errors,
            gpu_errors,
            audio_errors,
            input_events,
            observed_fps: None,
        },
        evidence: Evidence {
            log_sha1: sha1_bytes(&log),
            blocker_signature,
            first_blocker: blocker,
            measured: true,
        },
    })
}

fn count_lines(text: &str, predicate: impl Fn(&str) -> bool) -> u64 {
    text.lines().filter(|line| predicate(line)).count() as u64
}

fn count_any(text: &str, needles: &[&str]) -> u64 {
    count_lines(text, |line| {
        needles.iter().any(|needle| line.contains(needle))
    })
}

fn max_metric(text: &str, name: &str) -> u64 {
    let prefix = format!("{name}=");
    text.split_whitespace()
        .filter_map(|token| {
            let value = token.strip_prefix(&prefix)?;
            let digits = value
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect::<String>();
            (!digits.is_empty()).then(|| digits.parse::<u64>().ok())?
        })
        .max()
        .unwrap_or(0)
}

fn registry_roots_from_path(path: &str) -> Vec<String> {
    let mut roots = Vec::new();
    if let Some(index) = path.to_ascii_lowercase().find("\\ps5\\") {
        roots.push(path[..index + 4].to_string());
    }
    roots
}

fn first_blocker(text: &str, roots: &[String]) -> Option<String> {
    text.lines()
        .find(|line| is_blocker_line(line))
        .map(|line| sanitize_line(line, roots))
}

fn is_blocker_line(line: &str) -> bool {
    line.contains("ERROR")
        || line.contains("panicked")
        || (line.contains("WARN")
            && (line.contains("not supported") || line.contains("not implemented")))
}

fn sanitize_line(line: &str, roots: &[String]) -> String {
    let mut output = strip_ansi(line.trim());
    for root in roots {
        output = output.replace(root, "<GAME_ROOT>");
    }
    output = output
        .split_whitespace()
        .map(sanitize_token)
        .collect::<Vec<_>>()
        .join(" ");
    output.chars().take(500).collect()
}

fn strip_ansi(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for code in chars.by_ref() {
                if ('@'..='~').contains(&code) {
                    break;
                }
            }
        } else {
            output.push(ch);
        }
    }
    output
}

fn sanitize_token(token: &str) -> String {
    let Some(start) = token.find("0x") else {
        return token.into();
    };
    let digits = token[start + 2..]
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .count();
    if digits < 6 {
        return token.into();
    }
    format!("{}<ADDR>{}", &token[..start], &token[start + 2 + digits..])
}

fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn machine_id() -> String {
    let seed = format!(
        "{}|{}|{}",
        env::var("COMPUTERNAME").unwrap_or_default(),
        env::consts::ARCH,
        env::var("PROCESSOR_IDENTIFIER").unwrap_or_default()
    );
    format!("machine-{}", &sha1_bytes(seed.as_bytes())[..12])
}

fn compat_publish(args: &[String]) -> Result<()> {
    let input = PathBuf::from(option(args, "--input").unwrap_or_else(|| DEFAULT_RESULTS.into()));
    let output =
        PathBuf::from(option(args, "--output").unwrap_or_else(|| "compat/COMPATIBILITY.md".into()));
    let report: RunReport = read_json(&input)?;
    if report.schema_version != SCHEMA_VERSION
        || report
            .results
            .iter()
            .any(|result| result.schema_version != SCHEMA_VERSION || !result.evidence.measured)
    {
        bail!("refusing to publish unmeasured or incompatible results");
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut markdown = String::from(
        "# Measured compatibility\n\n\
         Generated only from Raeen's sanitized compatibility-result schema. \
         A result is evidence for this build and machine class, not a universal compatibility claim.\n\n\
         | Title | Build | Stage | Wall | Peak RAM | Flips | Shader errors | First blocker |\n\
         |---|---:|---|---:|---:|---:|---:|---|\n",
    );
    for result in &report.results {
        let peak = result
            .metrics
            .peak_working_set_bytes
            .map(|bytes| format!("{:.0} MiB", bytes as f64 / 1_048_576.0))
            .unwrap_or_else(|| "n/a".into());
        let blocker = result
            .evidence
            .first_blocker
            .as_deref()
            .filter(|line| is_blocker_line(line))
            .unwrap_or("none observed")
            .to_string();
        // Re-sanitize on publication as a defense in depth: older measured
        // schema-v1 reports may predate a sanitizer improvement.
        let blocker = sanitize_line(&blocker, &[]).replace('|', "\\|");
        let title = result.title.replace('|', "\\|");
        markdown.push_str(&format!(
            "| {} | `{}` | {:?} | {:.1}s | {} | {} | {} | {} |\n",
            title,
            result.build_revision,
            result.stage,
            result.metrics.wall_ms as f64 / 1000.0,
            peak,
            result.metrics.flip_events,
            result.metrics.shader_errors,
            blocker
        ));
    }
    fs::write(&output, markdown)?;
    println!(
        "published {} measured rows to {}",
        report.results.len(),
        output.display()
    );
    Ok(())
}

fn compat_compare(args: &[String]) -> Result<()> {
    let baseline = option(args, "--baseline").ok_or_else(|| anyhow!("--baseline is required"))?;
    let current = option(args, "--current").unwrap_or_else(|| DEFAULT_RESULTS.into());
    let baseline: RunReport = read_json(Path::new(&baseline))?;
    let current: RunReport = read_json(Path::new(&current))?;
    let old = baseline
        .results
        .iter()
        .map(|result| (&result.content_sha1, result))
        .collect::<BTreeMap<_, _>>();
    println!("title\twall change\tRAM change\tshader errors");
    for result in &current.results {
        if let Some(previous) = old.get(&result.content_sha1) {
            let wall = result.metrics.wall_ms as i128 - previous.metrics.wall_ms as i128;
            let ram = match (
                result.metrics.peak_working_set_bytes,
                previous.metrics.peak_working_set_bytes,
            ) {
                (Some(new), Some(old)) => {
                    format!("{:+.1} MiB", (new as f64 - old as f64) / 1_048_576.0)
                }
                _ => "n/a".into(),
            };
            println!(
                "{}\t{:+.1}s\t{}\t{} -> {}",
                result.title,
                wall as f64 / 1000.0,
                ram,
                previous.metrics.shader_errors,
                result.metrics.shader_errors
            );
        }
    }
    Ok(())
}

fn refs_report(args: &[String]) -> Result<()> {
    let state_path = PathBuf::from(
        option(args, "--state").unwrap_or_else(|| "compat/reference-state.json".into()),
    );
    let output = PathBuf::from(
        option(args, "--output").unwrap_or_else(|| "artifacts/reference-delta/latest.md".into()),
    );
    let state: ReferenceState = read_json(&state_path)?;
    let do_fetch = has(args, "--fetch");
    let mut markdown = format!(
        "# Upstream reference delta\n\nGenerated at Unix ms `{}`. Reference trees are read-only inputs.\n\n",
        now_ms()
    );
    for reference in state.references {
        let directory = PathBuf::from(&reference.directory);
        markdown.push_str(&format!("## {}\n\n", reference.name));
        if !directory.join(".git").exists() {
            markdown.push_str(&format!(
                "Missing local clone. Source: <{}> ({}, {}).\n\n",
                reference.url, reference.license, reference.upstream_branch
            ));
            continue;
        }
        if do_fetch {
            let status = Command::new("git")
                .args(["-C", &reference.directory, "fetch", "--prune", "origin"])
                .status()?;
            if !status.success() {
                markdown
                    .push_str("Fetch failed; report uses the existing remote-tracking state.\n\n");
            }
        }
        let target = format!("origin/{}", reference.upstream_branch);
        let head = git_at(&directory, &["rev-parse", "--short=12", &target])
            .unwrap_or_else(|_| "unavailable".into());
        let count = git_at(
            &directory,
            &[
                "rev-list",
                "--count",
                &format!("{}..{}", reference.baseline_revision, target),
            ],
        )
        .unwrap_or_else(|_| "unknown".into());
        markdown.push_str(&format!(
            "- License: {}\n- Baseline: `{}`\n- Upstream: `{}`\n- New commits: {}\n\n",
            reference.license,
            reference.baseline_revision,
            head.trim(),
            count.trim()
        ));
        if count.trim() != "0" && count.trim() != "unknown" {
            let log = git_at(
                &directory,
                &[
                    "log",
                    "--format=- `%h` %cs %s",
                    "--max-count=30",
                    &format!("{}..{}", reference.baseline_revision, target),
                ],
            )
            .unwrap_or_else(|_| "- Commit list unavailable".into());
            markdown.push_str(&log);
            markdown.push_str("\n\n");
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, markdown)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn acceptance_run(args: &[String]) -> Result<()> {
    struct Scenario {
        name: &'static str,
        area: &'static str,
        package: &'static str,
        required: bool,
    }
    let scenarios = [
        Scenario {
            name: "audio-mixer-and-output-contract",
            area: "audio",
            package: "raeen-audio",
            required: true,
        },
        Scenario {
            name: "controller-state-and-mapping-contract",
            area: "input",
            package: "raeen-input",
            required: true,
        },
        Scenario {
            name: "audio-pad-hle-contract",
            area: "audio+input",
            package: "raeen-hle",
            required: true,
        },
        Scenario {
            name: "shell-library-and-presentation-contract",
            area: "ui",
            package: "raeen-gui",
            required: true,
        },
    ];
    let timeout_secs = option(args, "--timeout")
        .unwrap_or_else(|| "180".into())
        .parse::<u64>()
        .context("--timeout must be an integer")?;
    let mut results = Vec::new();
    for scenario in scenarios {
        println!("acceptance: {}", scenario.name);
        let start = Instant::now();
        let mut child = Command::new("cargo")
            .args(["test", "-p", scenario.package])
            .spawn()
            .with_context(|| format!("run {}", scenario.name))?;
        let passed = loop {
            if let Some(status) = child.try_wait()? {
                break status.success();
            }
            if start.elapsed() >= Duration::from_secs(timeout_secs) {
                terminate_process_tree(&mut child);
                eprintln!(
                    "acceptance: {} timed out after {}s",
                    scenario.name, timeout_secs
                );
                break false;
            }
            thread::sleep(Duration::from_millis(100));
        };
        results.push(AcceptanceResult {
            name: scenario.name.into(),
            area: scenario.area.into(),
            required: scenario.required,
            passed,
            wall_ms: start.elapsed().as_millis(),
        });
    }
    let report = AcceptanceReport {
        schema_version: SCHEMA_VERSION,
        measured_unix_ms: now_ms(),
        build_revision: git_output(&["rev-parse", "--short=12", "HEAD"])
            .unwrap_or_else(|_| "unknown".into()),
        results,
    };
    let output = PathBuf::from(
        option(args, "--output")
            .unwrap_or_else(|| "artifacts/compat/acceptance-latest.json".into()),
    );
    write_json(&output, &report)?;
    let failures = report
        .results
        .iter()
        .filter(|result| result.required && !result.passed)
        .count();
    println!("wrote {}", output.display());
    if failures > 0 {
        bail!("{failures} required acceptance scenarios failed");
    }
    Ok(())
}

fn terminate_process_tree(child: &mut std::process::Child) {
    #[cfg(windows)]
    {
        // Cargo launches a test executable as its child. Terminate only this
        // runner-owned process tree so a hung contract cannot outlive the
        // acceptance timeout or interfere with later nightly titles.
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn git_output(args: &[&str]) -> Result<String> {
    git_at(Path::new("."), args)
}

fn git_at(directory: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .output()?;
    if !output.status.success() {
        bail!("git command failed");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().into())
}

struct ProcessSampler {
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
}

impl ProcessSampler {
    fn open(pid: u32) -> Self {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::{
                OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ,
            };
            let handle =
                unsafe { OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid) };
            Self { handle }
        }
        #[cfg(not(windows))]
        {
            let _ = pid;
            Self {}
        }
    }

    fn finish(self) -> (Option<u128>, Option<u64>) {
        #[cfg(windows)]
        {
            use std::mem::{size_of, zeroed};
            use windows_sys::Win32::{
                Foundation::{CloseHandle, FILETIME},
                System::{
                    ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
                    Threading::GetProcessTimes,
                },
            };
            if self.handle.is_null() {
                return (None, None);
            }
            let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { zeroed() };
            let memory_ok = unsafe {
                GetProcessMemoryInfo(
                    self.handle,
                    &mut counters,
                    size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                )
            } != 0;
            let mut creation: FILETIME = unsafe { zeroed() };
            let mut exit: FILETIME = unsafe { zeroed() };
            let mut kernel: FILETIME = unsafe { zeroed() };
            let mut user: FILETIME = unsafe { zeroed() };
            let times_ok = unsafe {
                GetProcessTimes(
                    self.handle,
                    &mut creation,
                    &mut exit,
                    &mut kernel,
                    &mut user,
                )
            } != 0;
            unsafe { CloseHandle(self.handle) };
            let cpu_ms = times_ok.then(|| (filetime_ticks(kernel) + filetime_ticks(user)) / 10_000);
            let peak = memory_ok.then_some(counters.PeakWorkingSetSize as u64);
            (cpu_ms, peak)
        }
        #[cfg(not(windows))]
        {
            (None, None)
        }
    }
}

#[cfg(windows)]
fn filetime_ticks(value: windows_sys::Win32::Foundation::FILETIME) -> u128 {
    ((value.dwHighDateTime as u128) << 32) | value.dwLowDateTime as u128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_ps5_title_id() {
        assert_eq!(
            title_id(Path::new("ASTRO/PPSA21564-app/eboot.bin")).as_deref(),
            Some("PPSA21564")
        );
    }

    #[test]
    fn sanitizer_removes_addresses_and_roots() {
        let value = sanitize_line(
            "\u{1b}[31mERROR\u{1b}[0m E:\\PS5\\ASTRO\\eboot.bin pc=0x556760000 short=0x10",
            &[r"E:\PS5".into()],
        );
        assert!(!value.contains('\u{1b}'));
        assert!(!value.contains(r"E:\PS5"));
        assert!(!value.contains("556760000"));
        assert!(value.contains("0x10"));
    }

    #[test]
    fn informational_capability_messages_are_not_compatibility_blockers() {
        let log = concat!(
            "INFO guest: IPv6 not supported\n",
            "INFO guest: continuing offline\n",
            "WARN shader: opcode not supported\n",
        );
        assert!(!is_blocker_line("INFO guest: IPv6 not supported"));
        let blocker = first_blocker(log, &[]).expect("warning is a blocker");
        assert_eq!(blocker, "WARN shader: opcode not supported");
    }

    #[test]
    fn ansi_structured_flip_counter_uses_high_water_mark() {
        let log = concat!(
            "\u{1b}[32mtotal_flips\u{1b}[0m=1 total_draws=4\n",
            "\u{1b}[32mtotal_flips\u{1b}[0m=19 total_draws=80\n",
        );
        let text = strip_ansi(log);
        assert_eq!(max_metric(&text, "total_flips"), 19);
    }

    #[test]
    fn nightly_selection_is_one_per_role() {
        let game = |title: &str, hash: &str, tag: &str| GameRecord {
            id: title.into(),
            title: title.into(),
            content_sha1: hash.into(),
            executable_bytes: 1,
            relative_hint: "depth-1/eboot.bin".into(),
            local_path: None,
            aliases: Vec::new(),
            tags: vec![tag.into()],
        };
        let games = vec![
            game("Astro", "a", "astro"),
            game("Minecraft", "b", "minecraft"),
            game("Until Dawn", "c", "ue5"),
            game("Subnautica", "d", "small"),
        ];
        assert_eq!(select_games(&games, "nightly").unwrap().len(), 4);
    }
}
