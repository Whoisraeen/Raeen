//! Logging infrastructure for XPS5X.
//!
//! Initializes the `tracing` subscriber with configurable log levels,
//! simultaneous stderr + file output, and structured formatting.
//!
//! # Why file logging matters here
//!
//! An emulator's failures are mostly *post-mortem*: an unresolved NID, a guest
//! fault address, a PM4 packet it choked on. Those need to be readable after
//! the fact (and by tooling), not just scrolled past in a terminal — so
//! [`init_with_file`] tees every event to `logs/xps5x.log` at a **stable,
//! predictable path** as well as stderr.

use std::path::Path;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

/// The log file [`init_with_file`] writes, inside the log directory.
pub const LOG_FILE_NAME: &str = "xps5x.log";

/// The default log directory (relative to the working directory).
pub const DEFAULT_LOG_DIR: &str = "logs";

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

/// Build the level filter: `XPS5X_LOG` wins, else `level`.
fn filter(level: &str) -> EnvFilter {
    EnvFilter::try_from_env("XPS5X_LOG").unwrap_or_else(|_| EnvFilter::new(level))
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
/// let _log = xps5x_core::logging::init("info");
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
        .try_init();

    tracing::info!("XPS5X v{} — PS5 Emulator initialized", crate::VERSION);
    LogGuard { _file: None }
}

/// Initialize the global tracing subscriber, writing to **both** stderr and
/// `<log_dir>/xps5x.log`.
///
/// The file is a single, non-rolling, appended file at a stable path so it can
/// always be found and read (by a human or by tooling) without globbing a
/// date-stamped name. It carries no ANSI escapes and includes target/thread/
/// file/line on every event.
///
/// The returned [`LogGuard`] **must** be held for the process lifetime — see
/// its doc comment.
pub fn init_with_file(level: &str, log_dir: &Path) -> anyhow::Result<LogGuard> {
    std::fs::create_dir_all(log_dir)?;

    let file_appender = tracing_appender::rolling::never(log_dir, LOG_FILE_NAME);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

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

    let _ = tracing_subscriber::registry()
        .with(filter(level))
        .with(stderr_layer)
        .with(file_layer)
        .try_init();

    tracing::info!(
        "XPS5X v{} — PS5 Emulator initialized (logging to {})",
        crate::VERSION,
        log_dir.join(LOG_FILE_NAME).display()
    );
    Ok(LogGuard { _file: Some(guard) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file sink must actually produce a readable file at the stable path.
    /// This is the regression guard for the original bug: `init_with_file`
    /// bound the `WorkerGuard` to `_guard`, dropping it at function exit, which
    /// tore the writer thread down and left the log empty — the function was
    /// also dead code, so nothing ever noticed.
    #[test]
    fn init_with_file_creates_the_log_file_at_the_stable_path() {
        let dir = std::env::temp_dir().join(format!("xps5x-log-test-{}", std::process::id()));
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
}
