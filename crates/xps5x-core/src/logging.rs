//! Logging infrastructure for XPS5X.
//!
//! Initializes the `tracing` subscriber with configurable log levels,
//! optional file output, and structured formatting.

use tracing_subscriber::{fmt, EnvFilter};

/// Initialize the global tracing subscriber.
///
/// Call this once at application startup. The log level can be controlled
/// via the `XPS5X_LOG` environment variable or the configuration file.
///
/// # Examples
/// ```
/// xps5x_core::logging::init("info");
/// ```
pub fn init(level: &str) {
    let env_filter = EnvFilter::try_from_env("XPS5X_LOG")
        .unwrap_or_else(|_| EnvFilter::new(level));

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global tracing subscriber");

    tracing::info!("XPS5X v{} — PS5 Emulator initialized", crate::VERSION);
}

/// Initialize logging with file output in addition to stderr.
pub fn init_with_file(level: &str, log_dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(log_dir)?;

    let env_filter = EnvFilter::try_from_env("XPS5X_LOG")
        .unwrap_or_else(|_| EnvFilter::new(level));

    let file_appender = tracing_appender::rolling::daily(log_dir, "xps5x.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_writer(non_blocking)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global tracing subscriber");

    tracing::info!("XPS5X v{} — PS5 Emulator initialized (file logging)", crate::VERSION);
    Ok(())
}
