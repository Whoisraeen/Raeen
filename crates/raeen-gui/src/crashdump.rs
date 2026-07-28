//! Out-of-process crash reporting for the isolated runner.
//!
//! The Shell (parent) hosts a `minidumper` server; the runner child attaches
//! a `crash-handler` whose crash event asks that server for a minidump over
//! the IPC socket. Dumping from a *different, healthy* process is the whole
//! point — a crashed process cannot reliably walk its own corrupted state.
//! Dumps land under `logs/crashes/` next to `logs/raeen.log`, which already
//! carries the recent-HLE report — together they are the actionable crash
//! report the north star asks for.
//!
//! Coexistence with the VEH trap-and-emulate runtime: the guest's HLE traps
//! are *handled* first-chance exceptions — the vectored handler resolves
//! them and execution continues, so the crash handler (a last-chance
//! unhandled-exception mechanism) never sees them. Only a genuinely fatal
//! fault reaches it.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

/// Where crash dumps are written, relative to the working directory (beside
/// `logs/raeen.log`) — the same folder the assembled `.report.md` files use,
/// so a dump and its report always pair up.
const CRASH_DIR: &str = crate::crash_report::REPORTS_DIR;

/// Domain-socket path shared by this Shell process and its runner children
/// (`minidumper` speaks Unix domain sockets, which Windows supports too).
fn socket_path() -> String {
    std::env::temp_dir()
        .join(format!("raeen-crash-{}.sock", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

struct DumpHandler {
    dir: PathBuf,
}

impl minidumper::ServerHandler for DumpHandler {
    fn create_minidump_file(&self) -> std::io::Result<(std::fs::File, PathBuf)> {
        std::fs::create_dir_all(&self.dir)?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = self.dir.join(format!("raeen-runner-{stamp}.dmp"));
        Ok((std::fs::File::create(&path)?, path))
    }

    fn on_minidump_created(
        &self,
        result: Result<minidumper::MinidumpBinary, minidumper::Error>,
    ) -> minidumper::LoopAction {
        match result {
            Ok(dump) => tracing::error!(
                dump = %dump.path.display(),
                "runner crashed — minidump written (pair it with logs/raeen.log)"
            ),
            Err(error) => tracing::error!(%error, "runner crashed but the minidump failed"),
        }
        // Keep serving: the user may relaunch and crash again this session.
        minidumper::LoopAction::Continue
    }

    fn on_message(&self, kind: u32, buffer: Vec<u8>) {
        tracing::debug!(kind, bytes = buffer.len(), "crash-server message");
    }

    fn on_client_disconnected(&self, clients: usize) -> minidumper::LoopAction {
        tracing::debug!(clients, "crash-server client disconnected");
        minidumper::LoopAction::Continue
    }
}

/// Start (once) the Shell-side minidump server and return the socket name to
/// hand to runner children, or `None` if the server could not start (launch
/// proceeds without crash dumps — never a reason not to play).
pub(crate) fn ensure_server() -> Option<String> {
    static SERVER: OnceLock<Option<String>> = OnceLock::new();
    SERVER
        .get_or_init(|| {
            let name = socket_path();
            let mut server = match minidumper::Server::with_name(minidumper::SocketName::path(&name))
            {
                Ok(server) => server,
                Err(error) => {
                    tracing::warn!(%error, "crash-dump server unavailable — runner crashes will not produce minidumps");
                    return None;
                }
            };
            let socket = name.clone();
            let spawned = std::thread::Builder::new()
                .name("raeen-crash-server".to_owned())
                .spawn(move || {
                    static SHUTDOWN: AtomicBool = AtomicBool::new(false);
                    let handler = Box::new(DumpHandler {
                        dir: PathBuf::from(CRASH_DIR),
                    });
                    // Serves for the Shell's whole lifetime; exit tears it down.
                    if let Err(error) = server.run(handler, &SHUTDOWN, None) {
                        tracing::warn!(%error, "crash-dump server stopped");
                    }
                });
            match spawned {
                Ok(_) => {
                    tracing::info!(socket = %socket, dir = CRASH_DIR, "crash-dump server ready");
                    Some(socket)
                }
                Err(error) => {
                    tracing::warn!(%error, "crash-dump server thread failed to start");
                    None
                }
            }
        })
        .clone()
}

/// Runner-child side: connect to the Shell's dump server and install the
/// crash handler. Both are deliberately leaked — they must outlive everything
/// in the process, including a crashing thread. No-op on failure (the runner
/// still runs, it just cannot produce dumps).
pub(crate) fn attach_client(socket: &str) {
    let client = match minidumper::Client::with_name(minidumper::SocketName::path(socket)) {
        Ok(client) => client,
        Err(error) => {
            tracing::warn!(%error, socket, "crash-dump client could not connect");
            return;
        }
    };
    // SAFETY (make_crash_event): the closure only calls `request_dump`, which
    // is designed to run inside a crash context (no allocation on the happy
    // path, IPC to the healthy parent does the dumping).
    let handler = crash_handler::CrashHandler::attach(unsafe {
        crash_handler::make_crash_event(move |context: &crash_handler::CrashContext| {
            crash_handler::CrashEventResult::Handled(client.request_dump(context).is_ok())
        })
    });
    match handler {
        Ok(handler) => {
            // Keep the handler installed for the process lifetime.
            std::mem::forget(handler);
            tracing::info!(socket, "runner crash handler attached");
        }
        Err(error) => tracing::warn!(%error, "crash handler failed to attach"),
    }
}
