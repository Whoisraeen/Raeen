//! Logging infrastructure for Raeen.
//!
//! Initializes the `tracing` subscriber with configurable log levels,
//! simultaneous stderr + file output, and structured formatting.
//!
//! # Why file logging matters here
//!
//! An emulator's failures are mostly *post-mortem*: an unresolved NID, a guest
//! fault address, a PM4 packet it choked on. Those need to be readable after
//! the fact (and by tooling), not just scrolled past in a terminal — so
//! [`init_with_file`] tees every event to `logs/raeen.log` at a **stable,
//! predictable path** as well as stderr.
//!
//! # Why the file is bounded
//!
//! That stable path used to mean *append forever, with no cap*: every run
//! concatenated onto the last, and a render loop at `debug` emits per-call HLE
//! tracing for every `malloc`/`free`/`memcpy` the guest performs. One
//! observed `logs/raeen.log` reached **15 GB** across a night of Minecraft
//! runs. Two limits keep that bounded, and both matter:
//!
//! * **Rotation** ([`rotate`]) — each run starts a fresh file, keeping exactly
//!   one previous run as `raeen.log.1`. A crash is investigated against *this*
//!   run's log, and the run before it is the one you compare against.
//! * **A byte cap** ([`CappedWriter`]) — a single run cannot outgrow
//!   [`max_log_bytes`]. Rotation alone does not save you here: one Minecraft
//!   session at `debug` was itself multiple gigabytes.
//!
//! The cap truncates the *tail*, keeping the head. Startup — module load, NID
//! resolution, the first frames — is where the diagnosis usually is; the
//! millionth `malloc` of a steady-state render loop is not.

use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt, reload};

/// The log file [`init_with_file`] writes, inside the log directory.
pub const LOG_FILE_NAME: &str = "raeen.log";

/// The previous run's log, kept by [`rotate`] alongside [`LOG_FILE_NAME`].
pub const PREV_LOG_FILE_NAME: &str = "raeen.log.1";

/// The default log directory (relative to the working directory).
pub const DEFAULT_LOG_DIR: &str = "logs";

/// Default cap on one run's log file: 64 MiB.
///
/// Large enough to hold a full boot at `debug` (the observed Minecraft boot
/// reaches its render loop well inside this), small enough that two of them —
/// current plus rotated — cost 128 MiB of disk rather than 15 GB.
pub const DEFAULT_MAX_LOG_BYTES: u64 = 64 * 1024 * 1024;

/// Environment override for the per-run cap, in bytes. `0` disables the cap.
pub const MAX_LOG_BYTES_ENV: &str = "RAEEN_LOG_MAX_BYTES";

/// The per-run byte cap: [`MAX_LOG_BYTES_ENV`] if set and parseable, else
/// [`DEFAULT_MAX_LOG_BYTES`]. `0` means unbounded — for deliberate deep-trace
/// sessions where you have the disk and want every line.
#[must_use]
pub fn max_log_bytes() -> u64 {
    match std::env::var(MAX_LOG_BYTES_ENV) {
        Ok(v) => v.trim().parse().unwrap_or(DEFAULT_MAX_LOG_BYTES),
        Err(_) => DEFAULT_MAX_LOG_BYTES,
    }
}

/// Start a fresh log, keeping the previous run as [`PREV_LOG_FILE_NAME`].
///
/// Any older `raeen.log.1` is dropped — the retention policy is exactly two
/// runs. Rename failures are not fatal: a log we cannot rotate is still a log
/// we can append to, and refusing to boot over it would be absurd.
fn rotate(log_dir: &Path) {
    let current = log_dir.join(LOG_FILE_NAME);
    if !current.exists() {
        return;
    }
    let previous = log_dir.join(PREV_LOG_FILE_NAME);
    let _ = std::fs::remove_file(&previous);
    let _ = std::fs::rename(&current, &previous);
}

/// A writer that stops after `cap` bytes instead of growing without bound.
///
/// Over-cap writes are **discarded, not errors**: they report success so the
/// `tracing` machinery above keeps running normally. A logger that starts
/// failing writes mid-session would be a worse bug than the one this fixes.
/// The cap is announced once, in the file itself, so a reader who hits the end
/// knows the log was truncated rather than that the emulator stopped.
struct CappedWriter<W: Write> {
    inner: W,
    written: u64,
    cap: u64,
    capped: bool,
}

impl<W: Write> CappedWriter<W> {
    /// `cap == 0` means unbounded.
    fn new(inner: W, cap: u64) -> Self {
        Self {
            inner,
            written: 0,
            cap,
            capped: false,
        }
    }
}

impl<W: Write> Write for CappedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.cap == 0 {
            return self.inner.write(buf);
        }
        if self.capped {
            return Ok(buf.len());
        }
        if self.written.saturating_add(buf.len() as u64) > self.cap {
            self.capped = true;
            // Best-effort: if this note cannot be written, the cap still holds.
            let _ = self.inner.write_all(
                format!(
                    "\n--- log size cap of {} bytes reached; further events for this run are \
                     dropped from the file (stderr is unaffected). Raise or disable the cap \
                     with {MAX_LOG_BYTES_ENV}=<bytes> (0 = unbounded). ---\n",
                    self.cap
                )
                .as_bytes(),
            );
            let _ = self.inner.flush();
            return Ok(buf.len());
        }
        let n = self.inner.write(buf)?;
        self.written = self.written.saturating_add(n as u64);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Keeps the background log-writer thread alive.
///
/// **Must be held for the whole process lifetime.** `tracing_appender`'s
/// non-blocking writer runs on a worker thread that is shut down when its
/// `WorkerGuard` drops; dropping the guard early (e.g. binding it to `_guard`
/// inside the init function, as this module previously did) silently loses
/// buffered events — the log file ends up empty or truncated. Holding this in
/// `main` is what makes the file actually complete.
#[must_use = "dropping this stops the background log writer and loses buffered log events"]
pub struct LogGuard {
    _file: Option<WorkerGuard>,
}

/// Build the level filter: `RAEEN_LOG` wins, else `level`.
fn filter(level: &str) -> EnvFilter {
    EnvFilter::try_from_env("RAEEN_LOG").unwrap_or_else(|_| EnvFilter::new(level))
}

// ---------------------------------------------------------------------------
// In-app console buffer
// ---------------------------------------------------------------------------

/// Default cap on buffered console lines (env-overridable via
/// [`CONSOLE_LINES_ENV`]). Bounded the same way the file is: an emulator at
/// `debug` produces millions of lines and the console exists for the recent
/// past, not the whole run (the file keeps the head).
pub const DEFAULT_CONSOLE_LINES: usize = 5_000;

/// Environment override for the console line cap.
pub const CONSOLE_LINES_ENV: &str = "RAEEN_CONSOLE_LINES";

/// One captured log event for the in-app console (Shell log viewer).
#[derive(Clone, Debug)]
pub struct ConsoleLine {
    /// Monotonic sequence number, never reused — lets a reader pull only the
    /// lines it has not seen and detect eviction.
    pub seq: u64,
    /// Milliseconds since logging initialized (the file log carries full
    /// timestamps; the console favors compactness).
    pub elapsed_ms: u64,
    /// Event level.
    pub level: tracing::Level,
    /// Event target (module path).
    pub target: String,
    /// Rendered message plus any structured fields as `key=value`.
    pub message: String,
}

/// Bounded ring of recent log events, fed by a subscriber layer installed in
/// [`init`] / [`init_with_file`] and read by the Shell's console window.
pub struct ConsoleBuffer {
    lines: std::sync::Mutex<std::collections::VecDeque<ConsoleLine>>,
    next_seq: std::sync::atomic::AtomicU64,
    cap: usize,
}

impl ConsoleBuffer {
    fn new(cap: usize) -> Self {
        Self {
            lines: std::sync::Mutex::new(std::collections::VecDeque::with_capacity(
                cap.min(1024),
            )),
            next_seq: std::sync::atomic::AtomicU64::new(0),
            cap,
        }
    }

    fn push(&self, elapsed_ms: u64, level: tracing::Level, target: String, message: String) {
        use std::sync::atomic::Ordering;
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let Ok(mut lines) = self.lines.lock() else {
            return;
        };
        if lines.len() >= self.cap {
            lines.pop_front();
        }
        lines.push_back(ConsoleLine {
            seq,
            elapsed_ms,
            level,
            target,
            message,
        });
    }

    /// Append every buffered line with `seq > after` to `out`, returning the
    /// highest sequence seen (pass it back on the next call; pass `None` on
    /// the first call to read everything buffered).
    pub fn read_since(&self, after: Option<u64>, out: &mut Vec<ConsoleLine>) -> Option<u64> {
        let lines = self.lines.lock().ok()?;
        let mut last = after;
        for line in lines.iter() {
            if after.is_none_or(|a| line.seq > a) {
                out.push(line.clone());
                last = Some(line.seq);
            }
        }
        last
    }

    /// Drop every buffered line (the file log is unaffected).
    pub fn clear(&self) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.clear();
        }
    }
}

/// The process-wide console buffer. Always present; it only receives events
/// once [`init`] / [`init_with_file`] install the capture layer.
pub fn console() -> &'static ConsoleBuffer {
    static CONSOLE: OnceLock<ConsoleBuffer> = OnceLock::new();
    CONSOLE.get_or_init(|| {
        let cap = std::env::var(CONSOLE_LINES_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_CONSOLE_LINES);
        ConsoleBuffer::new(cap)
    })
}

/// Subscriber layer that mirrors every event into [`console`].
struct ConsoleLayer {
    start: std::time::Instant,
}

impl ConsoleLayer {
    fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ConsoleLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        struct Visitor {
            message: String,
            fields: String,
        }
        impl tracing::field::Visit for Visitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                if field.name() == "message" {
                    let _ = write!(self.message, "{value:?}");
                } else {
                    let _ = write!(self.fields, " {}={value:?}", field.name());
                }
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                use std::fmt::Write;
                if field.name() == "message" {
                    self.message.push_str(value);
                } else {
                    let _ = write!(self.fields, " {}={value}", field.name());
                }
            }
        }
        let mut visitor = Visitor {
            message: String::new(),
            fields: String::new(),
        };
        event.record(&mut visitor);
        let mut message = visitor.message;
        message.push_str(&visitor.fields);
        console().push(
            self.start.elapsed().as_millis() as u64,
            *event.metadata().level(),
            event.metadata().target().to_owned(),
            message,
        );
    }
}

/// A boxed closure that swaps the global level filter to a new level string.
type ReloadFn = Box<dyn Fn(&str) + Send + Sync>;

/// Live handle to the global level filter, installed by [`init_with_file`].
/// Boxed so the concrete `reload::Handle<..>` type stays private; `set` only
/// succeeds once, so the first (real) subscriber wins.
static RELOAD: OnceLock<ReloadFn> = OnceLock::new();

/// Change the global log level at runtime — the seam behind Settings ▸ Debug ▸
/// Log Level / Logging. `level` is anything the env-filter accepts
/// (`error`/`warn`/`info`/`debug`/`trace`, or `off` to silence output). A no-op
/// until [`init_with_file`] has installed a reloadable subscriber (so `init`,
/// tests, and already-initialized processes simply ignore it).
pub fn set_level(level: &str) {
    if let Some(reload) = RELOAD.get() {
        reload(level);
    }
}

/// Initialize the global tracing subscriber, stderr only.
///
/// Prefer [`init_with_file`] for the emulator proper — this exists for tools
/// and tests that have nowhere to put a log file. Returns a [`LogGuard`] for
/// signature parity; it owns nothing.
///
/// Idempotent: a second call is a no-op rather than a panic (a global
/// subscriber can only be installed once per process).
///
/// # Examples
/// ```
/// let _log = raeen_core::logging::init("info");
/// ```
pub fn init(level: &str) -> LogGuard {
    let stderr_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(true);

    // `try_init` (not `init`): tolerate an already-installed subscriber
    // instead of panicking.
    let _ = tracing_subscriber::registry()
        .with(filter(level))
        .with(stderr_layer)
        .with(ConsoleLayer::new())
        .try_init();

    tracing::info!("Raeen v{} — PS5 Emulator initialized", crate::VERSION);
    LogGuard { _file: None }
}

/// Initialize the global tracing subscriber, writing to **both** stderr and
/// `<log_dir>/raeen.log`.
///
/// The file lives at a stable path so it can always be found and read (by a
/// human or by tooling) without globbing a date-stamped name. It carries no
/// ANSI escapes and includes target/thread/file/line on every event.
///
/// It is **bounded in both directions**: this call rotates the previous run to
/// `raeen.log.1` rather than appending to it, and the run's own output is
/// capped at [`max_log_bytes`]. See the module docs for why.
///
/// The returned [`LogGuard`] **must** be held for the process lifetime — see
/// its doc comment.
pub fn init_with_file(level: &str, log_dir: &Path) -> anyhow::Result<LogGuard> {
    std::fs::create_dir_all(log_dir)?;
    // Before opening: this run gets a fresh file, last run becomes `.1`.
    rotate(log_dir);

    let cap = max_log_bytes();
    // `rolling::never` opens in append mode; rotation above is what makes this
    // a fresh file, and the cap is what bounds it.
    let file_appender = tracing_appender::rolling::never(log_dir, LOG_FILE_NAME);
    let (non_blocking, guard) =
        tracing_appender::non_blocking(CappedWriter::new(file_appender, cap));

    let file_layer = fmt::layer()
        .with_writer(non_blocking)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        // Escape codes would corrupt a file meant to be grepped/read.
        .with_ansi(false);

    let stderr_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_ansi(true);

    // Wrap the level filter in a reload layer so `set_level` (Settings ▸ Debug)
    // can change verbosity live. Only wire the reload handle up if this call is
    // the one that actually installs the subscriber.
    let (filter_layer, reload_handle) = reload::Layer::new(filter(level));
    let installed = tracing_subscriber::registry()
        .with(filter_layer)
        .with(stderr_layer)
        .with(file_layer)
        .with(ConsoleLayer::new())
        .try_init()
        .is_ok();
    if installed {
        let _ = RELOAD.set(Box::new(move |lvl: &str| {
            let _ = reload_handle.reload(EnvFilter::new(lvl));
        }));
    }

    tracing::info!(
        "Raeen v{} — PS5 Emulator initialized (logging to {}, cap {})",
        crate::VERSION,
        log_dir.join(LOG_FILE_NAME).display(),
        if cap == 0 {
            "unbounded".to_owned()
        } else {
            format!("{} MiB", cap / (1024 * 1024))
        }
    );
    Ok(LogGuard { _file: Some(guard) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_buffer_ring_semantics() {
        let buffer = ConsoleBuffer::new(3);
        for i in 0..5u64 {
            buffer.push(i, tracing::Level::INFO, "t".into(), format!("m{i}"));
        }
        // Capped at 3: the two oldest evicted, sequence numbers preserved.
        let mut all = Vec::new();
        let last = buffer.read_since(None, &mut all);
        assert_eq!(last, Some(4));
        assert_eq!(
            all.iter().map(|l| l.seq).collect::<Vec<_>>(),
            [2, 3, 4],
            "oldest lines evicted, seq survives"
        );
        // Incremental read returns only unseen lines.
        buffer.push(5, tracing::Level::WARN, "t".into(), "m5".into());
        let mut fresh = Vec::new();
        let last = buffer.read_since(last, &mut fresh);
        assert_eq!(last, Some(5));
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].message, "m5");
        // Clear empties the ring but not the sequence counter.
        buffer.clear();
        let mut after_clear = Vec::new();
        assert_eq!(buffer.read_since(None, &mut after_clear), None);
        assert!(after_clear.is_empty());
        buffer.push(6, tracing::Level::ERROR, "t".into(), "m6".into());
        let mut post = Vec::new();
        assert_eq!(buffer.read_since(None, &mut post), Some(6));
    }

    /// The file sink must actually produce a readable file at the stable path.
    /// This is the regression guard for the original bug: `init_with_file`
    /// bound the `WorkerGuard` to `_guard`, dropping it at function exit, which
    /// tore the writer thread down and left the log empty — the function was
    /// also dead code, so nothing ever noticed.
    #[test]
    fn init_with_file_creates_the_log_file_at_the_stable_path() {
        let dir = std::env::temp_dir().join(format!("raeen-log-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // A global subscriber may already be installed by another test in this
        // binary; `init_with_file` tolerates that, and the file sink is created
        // regardless, which is what we assert.
        let guard = init_with_file("info", &dir).expect("init_with_file succeeds");

        let path = dir.join(LOG_FILE_NAME);
        assert!(
            path.exists(),
            "log file must be created at the stable path {}",
            path.display()
        );

        // Holding the guard is what keeps the writer alive; dropping it flushes.
        drop(guard);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The cap is the whole point: a run that logs without bound must produce a
    /// bounded file. Regression guard for the 15 GB `logs/raeen.log`.
    #[test]
    fn capped_writer_stops_growing_at_the_cap() {
        let mut sink = CappedWriter::new(Vec::new(), 64);
        for _ in 0..10_000 {
            sink.write_all(b"per-draw warn spam that used to reach gigabytes\n")
                .expect("over-cap writes are discarded, never errors");
        }

        // The note explaining the truncation is allowed past the cap; the
        // unbounded event stream is not.
        assert!(
            sink.inner.len() < 1024,
            "capped file must stay bounded, got {} bytes from ~470 KB of writes",
            sink.inner.len()
        );
        assert!(
            String::from_utf8_lossy(&sink.inner).contains("log size cap"),
            "a truncated log must say so, or its end looks like a crash"
        );
    }

    /// `cap == 0` is the deliberate deep-trace escape hatch.
    #[test]
    fn capped_writer_with_zero_cap_is_unbounded() {
        let mut sink = CappedWriter::new(Vec::new(), 0);
        for _ in 0..1000 {
            sink.write_all(b"trace\n").expect("write succeeds");
        }
        assert_eq!(sink.inner.len(), 6000, "cap 0 must not truncate");
    }

    /// Rotation is what stops runs from concatenating forever, and it must keep
    /// exactly one previous run.
    #[test]
    fn rotate_moves_the_current_log_aside_and_keeps_one_previous() {
        let dir = std::env::temp_dir().join(format!("raeen-rotate-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let current = dir.join(LOG_FILE_NAME);
        let previous = dir.join(PREV_LOG_FILE_NAME);

        // Rotating with nothing to rotate is a no-op, not an error.
        rotate(&dir);
        assert!(!previous.exists(), "nothing to rotate yet");

        std::fs::write(&current, b"run one").expect("write");
        rotate(&dir);
        assert!(!current.exists(), "current log is moved aside, not copied");
        assert_eq!(std::fs::read(&previous).expect("prev"), b"run one");

        std::fs::write(&current, b"run two").expect("write");
        rotate(&dir);
        assert_eq!(
            std::fs::read(&previous).expect("prev"),
            b"run two",
            "the older run is dropped; retention is exactly two runs"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
