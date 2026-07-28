//! Shell ↔ engine seam (spec §5).
//!
//! The Shell contains **no emulation logic**. It talks to the engine only
//! through [`GameLauncher`], so the Shell can be built and tested against
//! [`StubLauncher`] long before the real engine can run anything. SM3 swaps
//! `StubLauncher` for the real engine implementation without touching Shell
//! navigation or rendering code.

use crate::library::LaunchTarget;
use raeen_core::error::FirmwareError;
use std::collections::HashMap;
#[cfg(target_os = "windows")]
use std::os::windows::io::AsRawHandle;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use thiserror::Error;

/// The `RAEEN_*` variables Settings bridges to the guest runtime: the
/// Advanced dump/trace toggles plus Video ▸ Frame Limit. One list shared by
/// the startup bridge in `main.rs` and the per-launch runner environment
/// below, so the two can never drift.
pub(crate) const RUNNER_ENV_VARS: &[&str] = &[
    "RAEEN_DUMP_SHADERS",
    "RAEEN_DUMP_GPU_RESOURCES",
    "RAEEN_TRACE_HLE",
    "RAEEN_DUMP_FRAMES",
    "RAEEN_CALL_STATS",
    "RAEEN_STALL_DUMP",
    "RAEEN_VBLANK_HZ",
];

/// Names from [`RUNNER_ENV_VARS`] that were already set in the environment
/// when Raeen started — a developer's manual override, which always wins over
/// the Settings toggles. Recorded once in `main` before the startup bridge
/// writes anything.
fn dev_env_overrides() -> &'static std::sync::OnceLock<std::collections::HashSet<String>> {
    static OVERRIDES: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    &OVERRIDES
}

/// Record which [`RUNNER_ENV_VARS`] the environment already carried at
/// startup. Call from `main` **before** the startup env bridge runs.
pub(crate) fn record_dev_env_overrides() {
    let set = RUNNER_ENV_VARS
        .iter()
        .filter(|name| std::env::var_os(name).is_some())
        .map(|name| (*name).to_string())
        .collect();
    let _ = dev_env_overrides().set(set);
}

/// The environment the next isolated-runner child receives, derived from the
/// launching title's *effective* config (global settings + per-game
/// overrides). `Some(value)` sets the variable on the child, `None` removes
/// it — so turning an Advanced toggle Off in Settings really turns the dump
/// off for the next launch even though the Shell's own environment still
/// carries the startup bridge's value.
type RunnerEnv = Mutex<Vec<(String, Option<String>)>>;

fn runner_env() -> &'static RunnerEnv {
    static ENV: std::sync::OnceLock<RunnerEnv> = std::sync::OnceLock::new();
    ENV.get_or_init(|| Mutex::new(Vec::new()))
}

/// Build the per-launch runner environment from an effective config and stage
/// it for the next spawn. Variables the developer set manually at startup are
/// skipped (the child inherits them from the Shell's environment instead).
pub(crate) fn stage_runner_env(config: &raeen_core::config::EmulatorConfig) {
    let flag = |on: bool| if on { Some("1".to_string()) } else { None };
    let pairs = vec![
        ("RAEEN_DUMP_SHADERS", flag(config.debug.dump_shaders)),
        (
            "RAEEN_DUMP_GPU_RESOURCES",
            flag(config.debug.dump_gpu_commands),
        ),
        ("RAEEN_TRACE_HLE", flag(config.debug.trace_syscalls)),
        ("RAEEN_DUMP_FRAMES", flag(config.debug.dump_frames)),
        ("RAEEN_CALL_STATS", flag(config.debug.call_stats)),
        ("RAEEN_STALL_DUMP", flag(config.debug.stall_dump)),
        (
            "RAEEN_VBLANK_HZ",
            Some(config.graphics.frame_limit.to_string()),
        ),
    ];
    let overrides = dev_env_overrides().get();
    *runner_env().lock().unwrap() = pairs
        .into_iter()
        .filter(|(name, _)| !overrides.is_some_and(|o| o.contains(*name)))
        .map(|(name, value)| (name.to_string(), value))
        .collect();
}

/// Test-only view of the currently staged runner environment.
#[cfg(test)]
pub(crate) fn staged_runner_env() -> Vec<(String, Option<String>)> {
    runner_env().lock().unwrap().clone()
}

/// Opaque handle to a launched session, returned by [`GameLauncher::launch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionHandle(u64);

/// Lifecycle state of a launched session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Only ever produced by `StubLauncher`, which simulates a delay before
    /// `Running` — `FirmwareLauncher` loads synchronously (spec: no need to
    /// fake a Loading delay for SM3) and never returns this variant outside
    /// `StubLauncher`'s own tests.
    #[allow(dead_code)]
    Loading,
    Running,
    Faulted,
    Exited,
}

/// Errors returned by a [`GameLauncher`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LaunchError {
    #[error("unknown session handle")]
    UnknownHandle,
    /// Not raised by `StubLauncher` (it never fails); part of the trait's
    /// error contract for the real engine launcher landing in SM3.
    #[allow(dead_code)]
    #[error("launch failed: {0}")]
    Failed(String),
}

/// Implemented by the engine; consumed by the Shell.
pub trait GameLauncher {
    /// Begin launching a title. Returns a handle the Shell polls for state.
    fn launch(&self, target: &LaunchTarget) -> Result<SessionHandle, LaunchError>;
    /// Current state of a running session (Loading, Running, Faulted, Exited).
    fn session_state(&self, handle: &SessionHandle) -> SessionState;
    /// Optional human-readable detail beyond the bare [`SessionState`] — a
    /// fault reason, or (for [`FirmwareLauncher`]) a summary of what linking
    /// actually did, so the Shell's session overlay can be honest about the
    /// current stage instead of just printing a generic "Running"/"Faulted".
    /// `StubLauncher` never has anything more to say, hence the default.
    fn session_detail(&self, _handle: &SessionHandle) -> Option<String> {
        None
    }
    /// Stable process-scoped diagnostic events, when the engine provides
    /// deterministic diagnostics for this session.
    #[allow(dead_code)] // diagnostics UI panel will consume this contract
    fn session_diagnostics(
        &self,
        _handle: &SessionHandle,
    ) -> Option<Vec<raeen_core::diagnostics::DiagnosticEvent>> {
        None
    }
    /// The running session's kernel, so the Shell can push live controller
    /// input into the guest each frame via `OrbisKernel::set_pad_state`. `None`
    /// when the session is unknown or the launcher has no kernel to share
    /// (`StubLauncher`).
    fn session_kernel(
        &self,
        _handle: &SessionHandle,
    ) -> Option<std::sync::Arc<raeen_kernel::OrbisKernel>> {
        None
    }
    /// Request a running session to quit (returns to Shell).
    fn quit(&self, handle: &SessionHandle) -> Result<(), LaunchError>;
}

#[allow(dead_code)] // only constructed by `StubLauncher`, kept for its tests (see module docs)
struct StubSession {
    started: Instant,
    quit_requested: bool,
}

/// A launcher that simulates Loading→Running→Exited over time, so the whole
/// Shell is exercisable end-to-end before the real engine exists.
///
/// `loading_duration` is configurable so tests can set it to `Duration::ZERO`
/// and observe an immediate transition to `Running`.
///
/// SM3 wired the Shell to [`FirmwareLauncher`] instead; `StubLauncher` is
/// kept only for tests that want a deterministic, engine-free launcher
/// (spec: "SM3 swaps `StubLauncher` for the real engine implementation
/// without touching Shell navigation or rendering code" — the swap is in
/// `app.rs`, not here).
#[allow(dead_code)]
pub struct StubLauncher {
    loading_duration: Duration,
    sessions: Mutex<HashMap<u64, StubSession>>,
    next_id: Mutex<u64>,
}

impl StubLauncher {
    #[allow(dead_code)]
    pub fn new(loading_duration: Duration) -> Self {
        Self {
            loading_duration,
            sessions: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }
}

impl Default for StubLauncher {
    /// A realistic default loading duration for interactive use.
    fn default() -> Self {
        Self::new(Duration::from_millis(900))
    }
}

impl GameLauncher for StubLauncher {
    fn launch(&self, _target: &LaunchTarget) -> Result<SessionHandle, LaunchError> {
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;

        self.sessions.lock().unwrap().insert(
            id,
            StubSession {
                started: Instant::now(),
                quit_requested: false,
            },
        );

        Ok(SessionHandle(id))
    }

    fn session_state(&self, handle: &SessionHandle) -> SessionState {
        let sessions = self.sessions.lock().unwrap();
        match sessions.get(&handle.0) {
            None => SessionState::Faulted,
            Some(session) => {
                if session.quit_requested {
                    SessionState::Exited
                } else if session.started.elapsed() >= self.loading_duration {
                    SessionState::Running
                } else {
                    SessionState::Loading
                }
            }
        }
    }

    fn quit(&self, handle: &SessionHandle) -> Result<(), LaunchError> {
        let mut sessions = self.sessions.lock().unwrap();
        match sessions.get_mut(&handle.0) {
            Some(session) => {
                session.quit_requested = true;
                Ok(())
            }
            None => Err(LaunchError::UnknownHandle),
        }
    }
}

/// Guest virtual address linked modules are laid out at. Must equal
/// [`raeen_runtime::GUEST_ARENA_BASE`] — RT2's `GuestArena` always
/// identity-maps a module's image at that fixed base (guest address `A` is
/// host address `A`), so a mismatched link base would make any
/// `R_X86_64_RELATIVE` relocation resolve to the wrong host address.
#[cfg_attr(not(test), allow(dead_code))]
const DEFAULT_LOAD_BASE: u64 = raeen_runtime::GUEST_ARENA_BASE;

/// `argv[0]` every launched module sees (M1-A, crt0/process environment):
/// the PS4/PS5 convention mounts a title's content at `/app0`, so its main
/// module is `/app0/eboot.bin` regardless of where the file lives on the
/// host. The host path is deliberately *not* leaked into the guest — a real
/// filesystem mapping layer (host dir ↔ `/app0`) comes with save-data/file
/// I/O work, but the argv convention is stable now.
#[cfg(target_os = "windows")]
#[cfg_attr(not(test), allow(dead_code))]
const GUEST_ARGV0: &str = "/app0/eboot.bin";

/// What came of trying to load+link (and, on Windows, run) one module for a
/// launch.
#[derive(Debug, Clone)]
enum SessionOutcome {
    /// SELF decrypt -> `.sprx` parse -> dynlibdata decode -> NID link all
    /// succeeded, but the module was not executed. This is RT0/RT1b's
    /// non-Windows fallback: `raeen_runtime::execute_linked` is
    /// Windows-only, so on other targets the pipeline stops at link (spec:
    /// SM3 links, RT1b runs it where the runtime can). Kept as
    /// `#[allow(dead_code)]` because on a Windows build (this crate's
    /// primary target) `load` never constructs it — [`SessionOutcome::Ran`]
    /// or [`SessionOutcome::Faulted`] always win instead.
    #[allow(dead_code)]
    Linked {
        resolved: usize,
        unresolved: usize,
        /// Reserved for a future runtime/diagnostics surface (e.g. Settings
        /// ▸ a "last launch" panel); not yet read anywhere itself, but kept
        /// alongside `resolved`/`unresolved` per the milestone's contract.
        #[allow(dead_code)]
        image_size: usize,
    },
    /// Linked and executed as a process (Windows only, M1-A), but the
    /// module's `_start` *returned* instead of calling an exit-family
    /// function — malformed for a real program (`_start` is entered via
    /// `jmp` with no return address; see `raeen_runtime::execute_process`),
    /// tolerated and reported honestly rather than treated as a fault.
    #[cfg_attr(not(test), allow(dead_code))]
    Ran {
        returned: u64,
        resolved: usize,
        unresolved: usize,
    },
    /// Linked and executed as a real process (Windows only, M1-A):
    /// `raeen_runtime::execute_process` entered the module's `_start` on a
    /// genuine argc/argv/envp/auxv stack and the module ended itself via an
    /// exit-family call — the well-formed way a process run ends. This does
    /// not mean the module "plays" anything; it means the crt0/process
    /// contract held from Shell to exit.
    Exited {
        code: u64,
        resolved: usize,
        unresolved: usize,
    },
    /// The isolated production runner exited cleanly. Import counts stay in
    /// the child's measured report instead of being fabricated in the Shell.
    RunnerExited { code: i32 },
    /// Anything that stopped short of a successful run: no module file at
    /// the target path, an encrypted module with no matching key, a genuine
    /// parse/link error, an unresolved HLE import actually called, or a
    /// guest fault during execution. Carries the message the overlay shows
    /// verbatim.
    Faulted(String),
}

/// Shared control cell for a running guest process: the worker publishes a
/// **weak** process handle here the moment execution starts (so `quit` can
/// request termination). Weak by design — a strong handle would pin the guest
/// arena's fixed-base mapping past teardown, and only one such mapping can
/// exist per host process, so the next launch's reservation would race this
/// cell's release (`MapFailed`). Once the run ends, `upgrade` simply fails.
#[cfg(target_os = "windows")]
#[derive(Default)]
struct ProcessControl {
    handle: Option<std::sync::Weak<raeen_runtime::GuestProcess>>,
    runner: Option<std::process::Child>,
    job: Option<usize>,
    quit_requested: bool,
}

#[cfg(target_os = "windows")]
impl Drop for ProcessControl {
    fn drop(&mut self) {
        if let Some(child) = self.runner.as_mut() {
            let _ = child.kill();
        }
        if let Some(job) = self.job.take() {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(job as *mut core::ffi::c_void);
            }
        }
    }
}

#[cfg(target_os = "windows")]
fn create_runner_job(child: &std::process::Child) -> Result<usize, String> {
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    let job = unsafe { CreateJobObjectW(core::ptr::null(), core::ptr::null()) };
    if job.is_null() {
        return Err("Cannot create runner Job Object".to_string());
    }
    let mut limits = unsafe { core::mem::zeroed::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() };
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    let configured = unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            core::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    } != 0;
    let assigned =
        configured && unsafe { AssignProcessToJobObject(job, child.as_raw_handle()) } != 0;
    if !assigned {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(job) };
        return Err("Cannot assign runner to its kill-on-close Job Object".to_string());
    }
    Ok(job as usize)
}

#[cfg(target_os = "windows")]
#[cfg_attr(test, allow(dead_code))]
fn run_isolated_child(
    path: &Path,
    control: &std::sync::Arc<std::sync::Mutex<ProcessControl>>,
) -> SessionOutcome {
    let frame_receiver = match raeen_gpu::frame_ipc::FrameIpcReceiver::create() {
        Ok(receiver) => std::sync::Arc::new(receiver),
        Err(error) => {
            return SessionOutcome::Faulted(format!(
                "Cannot create isolated runner frame bridge: {error}"
            ));
        }
    };
    raeen_gpu::frame_ipc::install_receiver(std::sync::Arc::clone(&frame_receiver));
    let executable = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            raeen_gpu::frame_ipc::clear_receiver(&frame_receiver);
            return SessionOutcome::Faulted(format!("Cannot locate raeen runner: {error}"));
        }
    };
    let mut command = std::process::Command::new(executable);
    command
        .arg("--run-eboot")
        .arg(path)
        .env("RAEEN_RUNNER_CHILD", "1")
        .env(raeen_gpu::frame_ipc::FRAME_IPC_ENV, frame_receiver.name())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    // Out-of-process crash dumps: the Shell hosts the minidump server; the
    // child attaches a crash handler that requests dumps from it. A server
    // that failed to start simply means no dumps, never no launch.
    if let Some(socket) = crate::crashdump::ensure_server() {
        command.env("RAEEN_CRASH_SOCKET", socket);
    }
    // The launching title's effective Settings (Advanced dumps, Frame Limit),
    // staged by the Shell just before launch. Explicit set/remove per var so
    // the child sees the *current* Settings, not whatever the Shell's own
    // environment was frozen to at startup.
    for (name, value) in runner_env().lock().unwrap().iter() {
        match value {
            Some(value) => {
                command.env(name, value);
            }
            None => {
                command.env_remove(name);
            }
        }
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            raeen_gpu::frame_ipc::clear_receiver(&frame_receiver);
            return SessionOutcome::Faulted(format!("Cannot start isolated runner: {error}"));
        }
    };

    let job = match create_runner_job(&child) {
        Ok(job) => job,
        Err(error) => {
            let _ = child.kill();
            raeen_gpu::frame_ipc::clear_receiver(&frame_receiver);
            return SessionOutcome::Faulted(error);
        }
    };
    if job == 0 {
        let _ = child.kill();
        return SessionOutcome::Faulted("Cannot create runner Job Object".to_string());
    }

    {
        let mut state = control.lock().unwrap();
        if state.quit_requested {
            let _ = child.kill();
        }
        state.runner = Some(child);
        state.job = Some(job);
    }

    loop {
        let status = {
            let mut state = control.lock().unwrap();
            match state.runner.as_mut().expect("runner published").try_wait() {
                Ok(status) => status,
                Err(error) => {
                    raeen_gpu::frame_ipc::clear_receiver(&frame_receiver);
                    return SessionOutcome::Faulted(format!(
                        "Cannot query isolated runner: {error}"
                    ));
                }
            }
        };
        if let Some(status) = status {
            raeen_gpu::frame_ipc::clear_receiver(&frame_receiver);
            let mut state = control.lock().unwrap();
            state.runner.take();
            if let Some(job) = state.job.take() {
                unsafe {
                    windows_sys::Win32::Foundation::CloseHandle(job as *mut core::ffi::c_void);
                }
            }
            return if status.success() {
                SessionOutcome::RunnerExited {
                    code: status.code().unwrap_or(0),
                }
            } else {
                SessionOutcome::Faulted(format!(
                    "Isolated runner stopped with {status}; the Shell survived. See logs/raeen.log \
                     for the guest fault and recent-HLE report."
                ))
            };
        }
        std::thread::sleep(Duration::from_millis(16));
    }
}

struct FirmwareSession {
    /// `None` until the session thread reports — a real title runs for as long
    /// as the player plays, so the outcome does not exist yet.
    outcome: Option<SessionOutcome>,
    /// Delivers the outcome once the guest exits or faults.
    result: Option<std::sync::mpsc::Receiver<SessionOutcome>>,
    worker: Option<std::thread::JoinHandle<()>>,
    #[cfg(target_os = "windows")]
    process: std::sync::Arc<std::sync::Mutex<ProcessControl>>,
    /// Process-scoped kernel state retained for diagnostics after the guest
    /// exits. It is never shared with another launched title.
    kernel: Option<std::sync::Arc<raeen_kernel::OrbisKernel>>,
    quit_requested: bool,
}

impl FirmwareSession {
    /// Take the outcome if the session thread has finished. Non-blocking: the
    /// UI polls this every frame.
    fn poll(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        if let Some(rx) = &self.result {
            match rx.try_recv() {
                Ok(outcome) => {
                    self.outcome = Some(outcome);
                    self.result = None;
                    if let Some(worker) = self.worker.take() {
                        let _ = worker.join();
                    }
                }
                // Sender dropped without sending: the session thread died
                // (panicked). Report it rather than reading as still-running.
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.outcome = Some(SessionOutcome::Faulted(
                        "The session thread stopped unexpectedly".to_string(),
                    ));
                    self.result = None;
                    if let Some(worker) = self.worker.take() {
                        let _ = worker.join();
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }
}

/// Wires the Shell to the real firmware spine: [`raeen_firmware::load_module`]
/// (SELF decrypt-or-passthrough -> `.sprx` parse -> dynlibdata decode -> NID
/// link against HLE). This is SM3's whole point — the Shell no longer talks
/// to a stub, it talks to the actual engine entry point — but the engine
/// itself only *links* a module here; nothing executes it yet (that's the
/// next milestone). See [`SessionOutcome::Linked`].
///
/// Holds a [`raeen_firmware::NoKeysProvider`] and never anything else: the
/// Shell holds no key material of its own (clean-room boundary, spec §2),
/// so an encrypted retail module always faults informatively rather than
/// decrypting anything.
///
/// Each [`GameLauncher::launch`] creates a fresh kernel and module registry.
/// Loaded exports, handles, mounts, diagnostics, and clocks therefore belong
/// to exactly one guest process and cannot leak into a later title.
pub struct FirmwareLauncher {
    #[cfg_attr(not(test), allow(dead_code))]
    hle: std::sync::Arc<raeen_hle::HleRegistry>,
    sessions: Mutex<HashMap<u64, FirmwareSession>>,
    next_id: Mutex<u64>,
}

impl FirmwareLauncher {
    pub fn new() -> Self {
        let hle = raeen_hle::HleRegistry::new();
        Self {
            hle: std::sync::Arc::new(hle),
            sessions: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Read `path` and, if that succeeds, load+link it, then (on Windows)
    /// actually run its entry point through `raeen_runtime::execute_linked`.
    /// Every failure mode — unreadable file, missing key, malformed module,
    /// unresolved-import call, guest fault — becomes a
    /// [`SessionOutcome::Faulted`] with a message fit to show the user; this
    /// never panics.
    ///
    /// `kernel` and `registry` belong to this one session. The immutable HLE
    /// export table is shared; mutable HLE/kernel state is process-scoped.
    #[cfg_attr(not(test), allow(dead_code))]
    fn load_and_run(
        hle: &std::sync::Arc<raeen_hle::HleRegistry>,
        kernel: &std::sync::Arc<raeen_kernel::OrbisKernel>,
        registry: &mut raeen_firmware::ModuleRegistry,
        path: &Path,
        #[cfg(target_os = "windows")] process: std::sync::Arc<std::sync::Mutex<ProcessControl>>,
    ) -> SessionOutcome {
        // Memory-mapped: the eboot is the largest single file on the launch
        // path; mapping starts parsing immediately instead of after a full
        // buffered copy.
        let bytes = match raeen_loader::mapped::MappedFile::open(path) {
            Ok(bytes) => bytes,
            Err(err) => {
                return SessionOutcome::Faulted(format!(
                    "No module file at {}: {err}",
                    path.display()
                ));
            }
        };

        kernel
            .filesystem
            .set_game_directory(path.parent().unwrap_or_else(|| Path::new(".")));
        let title_dir = path.parent().and_then(Path::file_name).unwrap_or_default();
        let writable_root = std::env::temp_dir().join("raeen").join(title_dir);
        let temp_dir = writable_root.join("temp");
        let download_dir = writable_root.join("download");
        let savedata_dir = Path::new("savedata").join(title_dir);
        for writable_dir in [&temp_dir, &download_dir, &savedata_dir] {
            if let Err(error) = std::fs::create_dir_all(writable_dir) {
                return SessionOutcome::Faulted(format!(
                    "Cannot create writable title directory {}: {error}",
                    writable_dir.display()
                ));
            }
        }
        kernel.filesystem.set_temp_directory(&temp_dir);
        kernel.filesystem.set_download_directory(&download_dir);
        kernel.filesystem.set_savedata_directory(&savedata_dir);

        // Load the whole PROCESS — the eboot plus every DT_NEEDED `.prx` beside
        // it — not the eboot alone.
        //
        // `load_module` links the main module only, so every NEEDED library
        // falls back to whatever HLE we happen to provide. Our HLE `libc` is
        // partial, and Minecraft calls `_init_env` from its very first
        // instructions: the import stayed unresolved, the call landed on the
        // stub guard page, and the Shell reported "Unimplemented import:
        // _init_env (libc)" before the title ran at all. The CLI `--run-eboot`
        // path never hit it because it already loads the process — the title
        // ships its own libc.prx (2584 exports), which defines `_init_env`.
        // Loading it makes the Shell and CLI the same launch.
        let game_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let loaded = raeen_firmware::load_process(
            &bytes,
            game_dir,
            &raeen_firmware::NoKeysProvider,
            registry,
            hle,
            DEFAULT_LOAD_BASE,
        );

        let linked = match loaded.map(|process| process.linked) {
            Ok(linked) => linked,
            Err(FirmwareError::MissingKey { .. }) => {
                // Honest about the current state: `key_provider_path` is
                // stored by Settings but no file-based KeyProvider consumes
                // it yet — the launcher always runs `NoKeysProvider`.
                return SessionOutcome::Faulted(
                    "Encrypted module — decryption needs user-supplied keys, and key-provider \
                     support is not implemented yet"
                        .to_string(),
                );
            }
            Err(err) => return SessionOutcome::Faulted(err.to_string()),
        };

        let resolved = linked.hle_trampolines.len();
        let unresolved = linked.unresolved.len();

        // M1-D: the main module joins the kernel's module table, so a guest
        // `sceKernelLoadStartModule` naming it (or a Settings-style module
        // list) can find it by name/handle.
        kernel.register_module(raeen_kernel::ModuleInfo {
            id: 0, // assigned by register_module
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "eboot".to_string()),
            base_address: DEFAULT_LOAD_BASE,
            size: linked.image.len() as u64,
            entry_point: Some(linked.entry),
            initialized: true,
        });
        let linked = std::sync::Arc::new(linked);

        // Stage the package's boot splash (`sce_sys/pic0.png`) before entering
        // the guest: the process GPU session that presents it is created
        // inside `execute_process`, after our last chance to touch it.
        crate::splash::stage_boot_splash(path);

        #[cfg(target_os = "windows")]
        {
            // M1-A: enter the module as a real process — `_start` on a
            // genuine argc/argv/envp/auxv stack (`execute_process`), not a
            // bare 6-register function call. A well-formed run ends via an
            // exit-family call (`Exited`); a `_start` that returns anyway is
            // reported as `Ran` (malformed but tolerated).
            match raeen_runtime::execute_process_shared_with_control(
                std::sync::Arc::clone(&linked),
                std::sync::Arc::clone(hle),
                std::sync::Arc::clone(kernel),
                &[GUEST_ARGV0],
                &[],
                {
                    let process = std::sync::Arc::clone(&process);
                    move |handle| {
                        let mut control = process.lock().unwrap();
                        if control.quit_requested {
                            handle.request_termination(0);
                        }
                        control.handle = Some(handle.downgrade());
                    }
                },
            ) {
                Ok(raeen_runtime::RunOutcome::Exited(code)) => SessionOutcome::Exited {
                    code,
                    resolved,
                    unresolved,
                },
                Ok(raeen_runtime::RunOutcome::Returned(returned)) => SessionOutcome::Ran {
                    returned,
                    resolved,
                    unresolved,
                },
                Err(raeen_runtime::RuntimeError::Faulted { addr, access, kind }) => {
                    SessionOutcome::Faulted(format!(
                        "Faulted at {addr:#x} during execution ({kind} of {access:#x})"
                    ))
                }
                // The guest asked for an import nothing implements. Name it:
                // this is the one fault the user (or we) can actually act on,
                // and it used to read as an anonymous address.
                Err(raeen_runtime::RuntimeError::UnimplementedImport { nid, library, .. }) => {
                    let library = library.as_deref().unwrap_or("unknown library");
                    SessionOutcome::Faulted(format!(
                        "Unimplemented import: {} ({library}) — the game called a function \
                         Raeen does not provide yet",
                        raeen_firmware::dynlib::nid_names::describe(nid)
                    ))
                }
                Err(raeen_runtime::RuntimeError::UnresolvedTrampoline(a)) => {
                    SessionOutcome::Faulted(format!(
                        "Called an unresolved import (trampoline {a:#x})"
                    ))
                }
                Err(e) => SessionOutcome::Faulted(format!("Runtime error: {e:?}")),
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            // `execute_linked` is Windows-only (RT0 design doc §7/§9); every
            // other target stops at "linked" rather than pretending to run.
            SessionOutcome::Linked {
                resolved,
                unresolved,
                image_size: linked.image.len(),
            }
        }
    }
}

impl Default for FirmwareLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl GameLauncher for FirmwareLauncher {
    fn launch(&self, target: &LaunchTarget) -> Result<SessionHandle, LaunchError> {
        // Run the title on its own thread and return NOW.
        //
        // This used to call `load` inline, which runs the guest to completion —
        // `execute_process_shared` only returns when the title exits. For the
        // link-only fixtures this was written against that was instant; a real
        // title boots for a minute and then plays forever, so the Shell simply
        // froze the moment you launched Minecraft (measured: window unresponsive
        // for the whole run). The UI thread must never run the guest.
        let session = match target {
            LaunchTarget::Game { path } => {
                let (tx, rx) = std::sync::mpsc::channel();
                #[cfg(target_os = "windows")]
                let process = std::sync::Arc::new(std::sync::Mutex::new(ProcessControl::default()));
                #[cfg(target_os = "windows")]
                let worker_process = std::sync::Arc::clone(&process);
                #[cfg(test)]
                let hle = std::sync::Arc::clone(&self.hle);
                #[cfg(test)]
                let kernel = std::sync::Arc::new(raeen_kernel::OrbisKernel::new());
                #[cfg(test)]
                let session_kernel = std::sync::Arc::clone(&kernel);
                #[cfg(test)]
                let nid_db = raeen_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
                #[cfg(test)]
                let mut registry = raeen_firmware::ModuleRegistry::new(nid_db);
                let path = path.clone();
                let worker = std::thread::Builder::new()
                    .name("raeen-session".to_owned())
                    .spawn(move || {
                        // The receiver going away (session closed) is not an
                        // error worth reporting — nobody is listening.
                        #[cfg(test)]
                        let outcome = FirmwareLauncher::load_and_run(
                            &hle,
                            &kernel,
                            &mut registry,
                            &path,
                            #[cfg(target_os = "windows")]
                            std::sync::Arc::clone(&worker_process),
                        );
                        #[cfg(all(target_os = "windows", not(test)))]
                        let outcome = run_isolated_child(&path, &worker_process);
                        #[cfg(not(target_os = "windows"))]
                        let outcome = SessionOutcome::Faulted(
                            "Native runner is not implemented on this platform".to_string(),
                        );
                        let _ = tx.send(outcome);
                    })
                    .map_err(|e| LaunchError::Failed(format!("cannot start session: {e}")))?;
                FirmwareSession {
                    outcome: None,
                    result: Some(rx),
                    worker: Some(worker),
                    #[cfg(target_os = "windows")]
                    process,
                    #[cfg(test)]
                    kernel: Some(session_kernel),
                    #[cfg(not(test))]
                    kernel: None,
                    quit_requested: false,
                }
            }
            // Built-in apps (Store, Game Library, Settings) aren't modules;
            // there's no path to read, so this can't even attempt a load.
            LaunchTarget::App { id } => FirmwareSession {
                outcome: Some(SessionOutcome::Faulted(format!(
                    "'{id}' is not a loadable module"
                ))),
                result: None,
                worker: None,
                #[cfg(target_os = "windows")]
                process: std::sync::Arc::new(std::sync::Mutex::new(ProcessControl::default())),
                kernel: None,
                quit_requested: false,
            },
        };

        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;

        self.sessions.lock().unwrap().insert(id, session);

        Ok(SessionHandle(id))
    }

    fn session_state(&self, handle: &SessionHandle) -> SessionState {
        let mut sessions = self.sessions.lock().unwrap();
        let Some(session) = sessions.get_mut(&handle.0) else {
            return SessionState::Faulted;
        };
        session.poll();
        if session.quit_requested && session.outcome.is_some() {
            return SessionState::Exited;
        }
        match &session.outcome {
            // The title is still running on its own thread — which for a real
            // game is the normal, long-lived state, not a brief load.
            None => SessionState::Running,
            Some(SessionOutcome::Linked { .. } | SessionOutcome::Ran { .. }) => {
                SessionState::Running
            }
            // A run that ended via exit() stays `Running` so the session
            // overlay shows the outcome until the user quits — the Shell
            // auto-returns Home the moment it polls `Exited` (see
            // `shell/mod.rs`), which would flash past the detail text.
            Some(SessionOutcome::Exited { .. } | SessionOutcome::RunnerExited { .. }) => {
                SessionState::Running
            }
            Some(SessionOutcome::Faulted(_)) => SessionState::Faulted,
        }
    }

    fn session_detail(&self, handle: &SessionHandle) -> Option<String> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(&handle.0)?;
        session.poll();
        let diagnostic_events = session
            .kernel
            .as_ref()
            .filter(|kernel| kernel.diagnostics.is_enabled())
            .map(|kernel| kernel.diagnostics.snapshot().len());
        let mut detail = match session.outcome.as_ref()? {
            SessionOutcome::Linked {
                resolved,
                unresolved,
                ..
            } => format!(
                "Linked — {resolved} imports resolved to HLE, {unresolved} unresolved · execution not yet implemented (Esc to return)"
            ),
            SessionOutcome::Ran {
                returned,
                resolved,
                unresolved,
            } => format!(
                "Executed — _start returned {returned:#x} (malformed: a process ends via exit) · \
                 {resolved} HLE imports resolved, {unresolved} unresolved"
            ),
            SessionOutcome::Exited {
                code,
                resolved,
                unresolved,
            } => format!(
                "Ran to exit({code:#x}) — {resolved} HLE imports resolved, {unresolved} unresolved \
                 (early runtime — full game execution needs more HLE breadth)"
            ),
            SessionOutcome::RunnerExited { code } => format!(
                "Isolated runner exited cleanly with host status {code}; measured compatibility \
                 details are in logs/raeen.log"
            ),
            SessionOutcome::Faulted(message) => message.clone(),
        };
        if let Some(count) = diagnostic_events {
            detail.push_str(&format!(" · {count} deterministic events"));
        }
        Some(detail)
    }

    fn session_diagnostics(
        &self,
        handle: &SessionHandle,
    ) -> Option<Vec<raeen_core::diagnostics::DiagnosticEvent>> {
        self.sessions
            .lock()
            .unwrap()
            .get(&handle.0)
            .and_then(|session| session.kernel.as_ref())
            .map(|kernel| kernel.diagnostics.snapshot())
    }

    fn session_kernel(
        &self,
        handle: &SessionHandle,
    ) -> Option<std::sync::Arc<raeen_kernel::OrbisKernel>> {
        self.sessions
            .lock()
            .unwrap()
            .get(&handle.0)
            .and_then(|session| session.kernel.clone())
    }

    fn quit(&self, handle: &SessionHandle) -> Result<(), LaunchError> {
        let mut sessions = self.sessions.lock().unwrap();
        match sessions.get_mut(&handle.0) {
            Some(session) => {
                session.poll();
                session.quit_requested = true;
                #[cfg(target_os = "windows")]
                {
                    let mut control = session.process.lock().unwrap();
                    control.quit_requested = true;
                    if let Some(process) =
                        control.handle.as_ref().and_then(std::sync::Weak::upgrade)
                    {
                        process.request_termination(0);
                    }
                    if let Some(runner) = control.runner.as_mut() {
                        let _ = runner.kill();
                    }
                }
                Ok(())
            }
            None => Err(LaunchError::UnknownHandle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[cfg(target_os = "windows")]
    #[test]
    fn isolated_runner_crash_does_not_kill_shell() {
        if std::env::var_os("RAEEN_RUNNER_CRASH_PROBE").is_some() {
            std::process::abort();
        }

        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "launcher::tests::isolated_runner_crash_does_not_kill_shell",
                "--nocapture",
            ])
            .env("RAEEN_RUNNER_CRASH_PROBE", "1")
            .spawn()
            .expect("crash-probe runner must start");
        let job = create_runner_job(&child).expect("runner must enter kill-on-close Job Object");
        let status = child.wait().expect("runner status must be observable");
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(job as *mut core::ffi::c_void);
        }
        assert!(
            !status.success(),
            "the deliberate runner abort must be visible"
        );
        // Reaching this assertion is the acceptance property: the parent
        // Shell/test process survived the child's hard crash.
        assert_eq!(2 + 2, 4);
    }

    fn target() -> LaunchTarget {
        LaunchTarget::Game {
            path: PathBuf::from("Games/nova/eboot.bin"),
        }
    }

    #[test]
    fn stage_runner_env_maps_settings_to_set_and_remove() {
        let mut config = raeen_core::config::EmulatorConfig::default();
        config.debug.dump_shaders = true;
        config.debug.dump_frames = false;
        config.graphics.frame_limit = 120;
        stage_runner_env(&config);
        let staged = staged_runner_env();
        let get = |name: &str| {
            staged
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
                .expect("var must be staged")
        };
        // On → set "1"; Off → explicit remove (the child must not inherit a
        // stale startup value); Frame Limit always carries its number.
        assert_eq!(get("RAEEN_DUMP_SHADERS"), Some("1".to_string()));
        assert_eq!(get("RAEEN_DUMP_FRAMES"), None);
        assert_eq!(get("RAEEN_VBLANK_HZ"), Some("120".to_string()));
        assert_eq!(
            staged.len(),
            RUNNER_ENV_VARS.len(),
            "every bridged variable is staged when no dev override exists"
        );
    }

    #[test]
    fn zero_loading_duration_goes_straight_to_running() {
        let launcher = StubLauncher::new(Duration::ZERO);
        let handle = launcher.launch(&target()).expect("launch should succeed");
        assert_eq!(launcher.session_state(&handle), SessionState::Running);
    }

    #[test]
    fn nonzero_loading_duration_starts_in_loading() {
        let launcher = StubLauncher::new(Duration::from_secs(60));
        let handle = launcher.launch(&target()).expect("launch should succeed");
        assert_eq!(launcher.session_state(&handle), SessionState::Loading);
    }

    #[test]
    fn quit_transitions_to_exited() {
        let launcher = StubLauncher::new(Duration::ZERO);
        let handle = launcher.launch(&target()).expect("launch should succeed");
        assert_eq!(launcher.session_state(&handle), SessionState::Running);

        launcher.quit(&handle).expect("quit should succeed");
        assert_eq!(launcher.session_state(&handle), SessionState::Exited);
    }

    #[test]
    fn quit_on_unknown_handle_errors() {
        let launcher = StubLauncher::new(Duration::ZERO);
        let bogus = SessionHandle(9999);
        assert_eq!(launcher.quit(&bogus), Err(LaunchError::UnknownHandle));
    }

    #[test]
    fn each_launch_gets_a_distinct_handle() {
        let launcher = StubLauncher::new(Duration::ZERO);
        let a = launcher.launch(&target()).unwrap();
        let b = launcher.launch(&target()).unwrap();
        assert_ne!(a, b);
    }
}

#[cfg(test)]
mod firmware_launcher_tests {
    use super::*;
    use std::path::PathBuf;

    // --- minimal synthetic-.sprx helpers -----------------------------------
    // Mirrors the hand-built-buffer helpers in
    // `raeen-firmware/tests/homebrew_pipeline.rs` (not importable across
    // crates), trimmed to the bare minimum this test needs: a plaintext SELF
    // wrapping an `ET_SCE_DYNAMIC` ELF with a single `PT_LOAD` and no
    // `PT_DYNAMIC`/`PT_SCE_DYNLIBDATA` at all — the load-module pipeline
    // treats that as a module with zero imports/exports, not an error, so it
    // links cleanly with `resolved == 0` and `unresolved == 0`. On Windows,
    // M1-A's `FirmwareLauncher` executes this as a *process* (`e_entry`
    // defaults to 0, and the segment's first byte is a bare `ret` — a
    // malformed `_start` that pops `argc` off the process stack and jumps to
    // it as an address, faulting at Rip == argc == 1). Entirely synthetic
    // bytes; no real firmware anywhere.

    const EHDR_SIZE: usize = 64;
    const PHDR_SIZE: usize = 56;
    const EM_X86_64: u16 = 62;
    const ET_SCE_DYNAMIC: u16 = 0xFE18;
    const PT_LOAD: u32 = 1;

    const SELF_MAGIC: u32 = 0x4F15_D17E;
    const SELF_HEADER_SIZE: usize = 32;
    const SELF_ENTRY_SIZE: usize = 32;

    fn build_minimal_elf() -> Vec<u8> {
        let mut load_bytes = vec![0u8; 0x10];
        load_bytes[0] = 0xC3; // ret — entry (e_entry defaults to 0) returns immediately
        let phoff = EHDR_SIZE as u64;

        let mut header = vec![0u8; EHDR_SIZE];
        header[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        header[4] = 2; // ELFCLASS64
        header[5] = 1; // ELFDATA2LSB
        header[6] = 1; // EV_CURRENT
        header[16..18].copy_from_slice(&ET_SCE_DYNAMIC.to_le_bytes());
        header[18..20].copy_from_slice(&EM_X86_64.to_le_bytes());
        header[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
        header[32..40].copy_from_slice(&phoff.to_le_bytes()); // e_phoff
        header[52..54].copy_from_slice(&(EHDR_SIZE as u16).to_le_bytes()); // e_ehsize
        header[54..56].copy_from_slice(&(PHDR_SIZE as u16).to_le_bytes()); // e_phentsize
        header[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

        let mut ph = [0u8; PHDR_SIZE];
        ph[0..4].copy_from_slice(&PT_LOAD.to_le_bytes());
        ph[4..8].copy_from_slice(&6u32.to_le_bytes()); // R+W
        ph[8..16].copy_from_slice(&(EHDR_SIZE as u64 + PHDR_SIZE as u64).to_le_bytes()); // p_offset
        ph[16..24].copy_from_slice(&0u64.to_le_bytes()); // p_vaddr
        ph[32..40].copy_from_slice(&(load_bytes.len() as u64).to_le_bytes()); // p_filesz
        ph[40..48].copy_from_slice(&(load_bytes.len() as u64).to_le_bytes()); // p_memsz

        let mut buf = header;
        buf.extend_from_slice(&ph);
        buf.extend_from_slice(&load_bytes);
        buf
    }

    /// Wrap `inner_elf` in a plaintext (unencrypted) SELF header — mirrors
    /// `self_crypto.rs`'s private `build_self` test helper.
    fn build_plaintext_self(inner_elf: &[u8]) -> Vec<u8> {
        let header_size = SELF_HEADER_SIZE + SELF_ENTRY_SIZE;

        let mut buf = vec![0u8; header_size];
        buf[0..4].copy_from_slice(&SELF_MAGIC.to_le_bytes());
        buf[4] = 1; // version
        buf[5] = 0; // mode
        buf[6] = 1; // endian
        buf[7] = 0; // attributes
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // key_type
        buf[12..14].copy_from_slice(&(header_size as u16).to_le_bytes());
        buf[14..16].copy_from_slice(&0u16.to_le_bytes()); // meta_size
        buf[24..26].copy_from_slice(&1u16.to_le_bytes()); // num_entries
        buf[26..28].copy_from_slice(&0u16.to_le_bytes()); // flags

        let base = SELF_HEADER_SIZE;
        buf[base..base + 8].copy_from_slice(&0u64.to_le_bytes()); // properties: plaintext
        buf[base + 8..base + 16].copy_from_slice(&(header_size as u64).to_le_bytes()); // offset
        buf[base + 16..base + 24].copy_from_slice(&(inner_elf.len() as u64).to_le_bytes());
        buf[base + 24..base + 32].copy_from_slice(&(inner_elf.len() as u64).to_le_bytes());

        buf.extend_from_slice(inner_elf);

        let file_size = buf.len() as u64;
        buf[16..24].copy_from_slice(&file_size.to_le_bytes());

        buf
    }

    fn write_synthetic_sprx(dir: &Path, name: &str) -> PathBuf {
        let elf = build_minimal_elf();
        let sprx_bytes = build_plaintext_self(&elf);
        let path = dir.join(name);
        std::fs::write(&path, &sprx_bytes).expect("write synthetic .sprx to temp dir");
        path
    }

    /// Block until the session thread reports, then return its state.
    ///
    /// `launch` is asynchronous — it starts the title on its own thread and
    /// returns immediately, because a real title runs for as long as the player
    /// plays and the UI thread must stay live. So a test that wants the OUTCOME
    /// has to wait for it; reading `session_state` straight after `launch`
    /// races the thread and usually just sees `Running`.
    ///
    /// The fixtures here link or fault in milliseconds; the timeout only exists
    /// so a regression hangs the one test instead of the whole suite.
    fn settled_state(launcher: &FirmwareLauncher, handle: &SessionHandle) -> SessionState {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let state = launcher.session_state(handle);
            let settled = launcher
                .sessions
                .lock()
                .unwrap()
                .get(&handle.0)
                .is_some_and(|s| s.outcome.is_some());
            if settled || std::time::Instant::now() >= deadline {
                return state;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    // --- tests --------------------------------------------------------------

    #[test]
    fn missing_module_file_faults_without_panicking() {
        let launcher = FirmwareLauncher::new();
        let target = LaunchTarget::Game {
            path: PathBuf::from("this/path/does/not/exist/eboot.bin"),
        };

        let handle = launcher
            .launch(&target)
            .expect("launch always returns a handle, even on fault");
        assert_eq!(settled_state(&launcher, &handle), SessionState::Faulted);
        let detail = launcher
            .session_detail(&handle)
            .expect("fault carries a message");
        assert!(
            detail.starts_with("No module file at"),
            "unexpected message: {detail}"
        );
    }

    #[test]
    fn app_target_faults_cleanly() {
        let launcher = FirmwareLauncher::new();
        let target = LaunchTarget::App {
            id: "settings".to_string(),
        };

        let handle = launcher
            .launch(&target)
            .expect("launch always returns a handle, even on fault");
        assert_eq!(settled_state(&launcher, &handle), SessionState::Faulted);
        let detail = launcher
            .session_detail(&handle)
            .expect("fault carries a message");
        assert!(
            detail.contains("not a loadable module"),
            "unexpected message: {detail}"
        );
    }

    #[test]
    fn each_game_launch_gets_isolated_process_kernel_state() {
        let launcher = FirmwareLauncher::new();
        let first = launcher
            .launch(&LaunchTarget::Game {
                path: PathBuf::from("missing/first/eboot.bin"),
            })
            .expect("first launch");
        let second = launcher
            .launch(&LaunchTarget::Game {
                path: PathBuf::from("missing/second/eboot.bin"),
            })
            .expect("second launch");
        assert_eq!(settled_state(&launcher, &first), SessionState::Faulted);
        assert_eq!(settled_state(&launcher, &second), SessionState::Faulted);

        let sessions = launcher.sessions.lock().unwrap();
        let first_kernel = sessions[&first.0]
            .kernel
            .as_ref()
            .expect("game session has a kernel");
        let second_kernel = sessions[&second.0]
            .kernel
            .as_ref()
            .expect("game session has a kernel");
        assert!(!std::sync::Arc::ptr_eq(first_kernel, second_kernel));

        first_kernel.register_module(raeen_kernel::ModuleInfo {
            id: 0,
            name: "first-only".to_string(),
            base_address: 0,
            size: 1,
            entry_point: None,
            initialized: true,
        });
        assert_eq!(first_kernel.modules.len(), 1);
        assert!(second_kernel.modules.is_empty());
    }

    #[test]
    fn valid_synthetic_module_links_and_exposes_resolved_counts() {
        let tmp =
            std::env::temp_dir().join(format!("raeen-gui-launcher-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        let path = write_synthetic_sprx(&tmp, "eboot.bin");

        let launcher = FirmwareLauncher::new();
        let target = LaunchTarget::Game { path };
        let handle = launcher
            .launch(&target)
            .expect("launch always returns a handle");

        // On Windows, M1-A runs this module as a process. A bare `ret` at
        // `_start` is malformed (entered via `jmp`, there is no return
        // address): it pops `argc` (== 1, one argv entry) and jumps to it,
        // faulting at Rip == 0x1 — itself proof the process stack delivered
        // a real `argc` at the entry's first instruction. Every other target
        // has no runtime backend, so execution is never reached and the
        // pipeline stops at `Linked` (see `load`'s `#[cfg]` gate).
        #[cfg(target_os = "windows")]
        {
            assert_eq!(settled_state(&launcher, &handle), SessionState::Faulted);
            let detail = launcher
                .session_detail(&handle)
                .expect("a faulted session has detail text");
            assert!(
                detail.starts_with("Faulted at 0x1 during execution"),
                "a bare-ret _start must fault at Rip == argc == 1 — unexpected message: {detail}"
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(settled_state(&launcher, &handle), SessionState::Running);
            let detail = launcher
                .session_detail(&handle)
                .expect("a running session has detail text");
            assert!(
                detail.contains("0 imports resolved to HLE"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("0 unresolved"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("execution not yet implemented"),
                "unexpected message: {detail}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn quit_transitions_a_linked_session_to_exited() {
        let tmp = std::env::temp_dir().join(format!(
            "raeen-gui-launcher-quit-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        let path = write_synthetic_sprx(&tmp, "eboot.bin");

        let launcher = FirmwareLauncher::new();
        let handle = launcher
            .launch(&LaunchTarget::Game { path })
            .expect("launch should succeed");
        // The minimal bare-`ret` fixture faults as a process on Windows (see
        // `valid_synthetic_module_links_and_exposes_resolved_counts`) and
        // stops at `Linked` (`Running`) elsewhere — either way, `quit` must
        // transition the session to `Exited`.
        #[cfg(target_os = "windows")]
        assert_eq!(settled_state(&launcher, &handle), SessionState::Faulted);
        #[cfg(not(target_os = "windows"))]
        assert_eq!(settled_state(&launcher, &handle), SessionState::Running);

        launcher.quit(&handle).expect("quit should succeed");
        assert_eq!(settled_state(&launcher, &handle), SessionState::Exited);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn quit_on_unknown_handle_errors() {
        let launcher = FirmwareLauncher::new();
        let bogus = SessionHandle(9999);
        assert_eq!(launcher.quit(&bogus), Err(LaunchError::UnknownHandle));
    }

    #[test]
    fn session_state_of_unknown_handle_is_faulted() {
        let launcher = FirmwareLauncher::new();
        let bogus = SessionHandle(9999);
        assert_eq!(launcher.session_state(&bogus), SessionState::Faulted);
        assert!(launcher.session_detail(&bogus).is_none());
    }

    // --- RT1b: shell Play genuinely executes the module (Windows only) -----
    // Mirrors `raeen-firmware/tests/homebrew_pipeline.rs`'s
    // `build_dynlib_and_dynamic`/`build_elf` (a real PT_SCE_DYNLIBDATA +
    // PT_DYNAMIC declaring one import) and `raeen-runtime/tests/execute.rs`'s
    // entry-stub writer (`call qword ptr [rip+disp32]; ret`), combined into a
    // module whose entry calls a single HLE-registered import — so this
    // exercises the *entire* pipeline the shell's Play button now drives:
    // load_module (real SELF/`.sprx`/dynlibdata parse + NID link) ->
    // execute_linked (real VirtualAlloc mapping + VEH-guarded native call) ->
    // the trampoline dispatching to a test-registered HLE function. Not
    // importable across crates, so replicated here. Entirely synthetic
    // bytes; no real firmware anywhere.
    #[cfg(target_os = "windows")]
    mod executes_module {
        use super::*;

        const PT_DYNAMIC: u32 = 2;
        const DT_SCE_JMPREL: u64 = 0x6100_0029;
        const DT_SCE_PLTRELSZ: u64 = 0x6100_002D;
        const DT_SCE_RELA: u64 = 0x6100_002F;
        const DT_SCE_RELASZ: u64 = 0x6100_0031;
        const DT_SCE_RELAENT: u64 = 0x6100_0033;
        const DT_SCE_STRTAB: u64 = 0x6100_0035;
        const DT_SCE_STRSZ: u64 = 0x6100_0037;
        const DT_SCE_SYMTAB: u64 = 0x6100_0039;
        const DT_SCE_SYMENT: u64 = 0x6100_003B;
        const DT_SCE_SYMTABSZ: u64 = 0x6100_003F;
        const DT_NULL: u64 = 0;
        const R_X86_64_JUMP_SLOT: u64 = 7;
        const RELOC_SLOT_OFFSET: u64 = 0x10;

        struct PhdrSpec {
            p_type: u32,
            p_flags: u32,
            p_vaddr: u64,
            data: Vec<u8>,
            /// `p_memsz`. Usually equals `data.len()` (`p_filesz`), but a real
            /// compiler-produced `PT_LOAD` can have `p_memsz > p_filesz` (a
            /// `.bss`/`.relro_padding` tail); the `PT_TLS` template likewise
            /// needs its own `p_memsz`. Kept explicit so the M1 compiler
            /// fixture can round-trip guest.so's segments faithfully.
            p_memsz: u64,
            /// `p_align` — carried through so a `PT_TLS`'s alignment reaches
            /// the loader's `TlsTemplate` (the runtime's TLS-block placement
            /// reads it). `0` for segments that don't care.
            p_align: u64,
        }

        /// General-purpose ELF64 builder (unlike `build_minimal_elf`, takes
        /// an arbitrary phdr list and an explicit `e_entry`) — mirrors
        /// `homebrew_pipeline.rs`'s `build_elf`.
        fn build_elf_with_entry(e_type: u16, entry: u64, phdrs: &[PhdrSpec]) -> Vec<u8> {
            let phnum = phdrs.len();
            let phoff = EHDR_SIZE as u64;

            let mut header = vec![0u8; EHDR_SIZE];
            header[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
            header[4] = 2; // ELFCLASS64
            header[5] = 1; // ELFDATA2LSB
            header[6] = 1; // EV_CURRENT
            header[16..18].copy_from_slice(&e_type.to_le_bytes());
            header[18..20].copy_from_slice(&EM_X86_64.to_le_bytes());
            header[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
            header[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry
            header[32..40].copy_from_slice(&phoff.to_le_bytes()); // e_phoff
            header[52..54].copy_from_slice(&(EHDR_SIZE as u16).to_le_bytes()); // e_ehsize
            header[54..56].copy_from_slice(&(PHDR_SIZE as u16).to_le_bytes()); // e_phentsize
            header[56..58].copy_from_slice(&(phnum as u16).to_le_bytes()); // e_phnum

            let mut offset = (EHDR_SIZE + phnum * PHDR_SIZE) as u64;
            let mut phdr_bytes = Vec::new();
            let mut seg_bytes = Vec::new();
            for spec in phdrs {
                let mut ph = [0u8; PHDR_SIZE];
                ph[0..4].copy_from_slice(&spec.p_type.to_le_bytes());
                ph[4..8].copy_from_slice(&spec.p_flags.to_le_bytes());
                ph[8..16].copy_from_slice(&offset.to_le_bytes()); // p_offset
                ph[16..24].copy_from_slice(&spec.p_vaddr.to_le_bytes()); // p_vaddr
                ph[32..40].copy_from_slice(&(spec.data.len() as u64).to_le_bytes()); // p_filesz
                ph[40..48].copy_from_slice(&spec.p_memsz.to_le_bytes()); // p_memsz
                ph[48..56].copy_from_slice(&spec.p_align.to_le_bytes()); // p_align
                phdr_bytes.extend_from_slice(&ph);

                seg_bytes.extend_from_slice(&spec.data);
                offset += spec.data.len() as u64;
            }

            let mut buf = header;
            buf.extend_from_slice(&phdr_bytes);
            buf.extend_from_slice(&seg_bytes);
            buf
        }

        /// The `PT_SCE_DYNLIBDATA` blob (strtab + one undefined `Elf64_Sym` +
        /// one JMPREL `Elf64_Rela`) and matching `PT_DYNAMIC` bytes for a
        /// single import identified by `import_nid`. Mirrors
        /// `homebrew_pipeline.rs`'s `build_dynlib_and_dynamic`.
        fn build_dynlib_and_dynamic(import_nid: u64) -> (Vec<u8>, Vec<u8>) {
            let import_name = format!(
                "{}#A#A",
                raeen_firmware::dynlib::nid::encode_nid(import_nid)
            );
            let mut strtab = vec![0u8];
            let import_off = strtab.len() as u32;
            strtab.extend_from_slice(import_name.as_bytes());
            strtab.push(0);

            let mut symtab = Vec::new();
            symtab.extend_from_slice(&import_off.to_le_bytes());
            symtab.push(0);
            symtab.push(0);
            symtab.extend_from_slice(&0u16.to_le_bytes());
            symtab.extend_from_slice(&0u64.to_le_bytes());
            symtab.extend_from_slice(&0u64.to_le_bytes());

            let mut jmprel = Vec::new();
            jmprel.extend_from_slice(&RELOC_SLOT_OFFSET.to_le_bytes());
            jmprel.extend_from_slice(&R_X86_64_JUMP_SLOT.to_le_bytes());
            jmprel.extend_from_slice(&0i64.to_le_bytes());

            let strtab_off = 0u64;
            let symtab_off = strtab.len() as u64;
            let jmprel_off = symtab_off + symtab.len() as u64;

            let mut blob = Vec::new();
            blob.extend_from_slice(&strtab);
            blob.extend_from_slice(&symtab);
            blob.extend_from_slice(&jmprel);

            let mut dynamic = Vec::new();
            let mut push_tag = |tag: u64, val: u64| {
                dynamic.extend_from_slice(&tag.to_le_bytes());
                dynamic.extend_from_slice(&val.to_le_bytes());
            };
            push_tag(DT_SCE_STRTAB, strtab_off);
            push_tag(DT_SCE_STRSZ, strtab.len() as u64);
            push_tag(DT_SCE_SYMTAB, symtab_off);
            push_tag(DT_SCE_SYMTABSZ, symtab.len() as u64);
            push_tag(DT_SCE_SYMENT, 24);
            push_tag(DT_SCE_JMPREL, jmprel_off);
            push_tag(DT_SCE_PLTRELSZ, jmprel.len() as u64);
            push_tag(DT_NULL, 0);

            (blob, dynamic)
        }

        /// A plaintext-SELF-wrapped `.sprx` whose entry calls an unresolved
        /// import, forwards its return value to `exit`, and therefore proves
        /// the default unresolved-call path resumed with `rax = 0`.
        fn build_executable_sprx(import_nid: u64) -> Vec<u8> {
            const SLOT_UNRESOLVED_OFF: usize = 0x40;
            const SLOT_EXIT_OFF: usize = 0x48;
            let exit_nid = raeen_firmware::dynlib::nid::nid_of("exit");
            let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic_multi(&[
                (import_nid, SLOT_UNRESOLVED_OFF as u64),
                (exit_nid, SLOT_EXIT_OFF as u64),
            ]);

            let mut load_bytes = vec![0u8; 0x100];
            // call qword ptr [rip+disp32] -> unresolved slot
            let call1_disp32 = (SLOT_UNRESOLVED_OFF as i64 - 6) as i32;
            load_bytes[0] = 0xFF;
            load_bytes[1] = 0x15;
            load_bytes[2..6].copy_from_slice(&call1_disp32.to_le_bytes());
            // mov rdi, rax — unresolved calls resume with zero by default.
            load_bytes[6..9].copy_from_slice(&[0x48, 0x89, 0xC7]);
            // call qword ptr [rip+disp32] -> exit slot (never returns)
            let call2_disp32 = (SLOT_EXIT_OFF as i64 - 15) as i32;
            load_bytes[9] = 0xFF;
            load_bytes[10] = 0x15;
            load_bytes[11..15].copy_from_slice(&call2_disp32.to_le_bytes());

            let elf = build_elf_with_entry(
                ET_SCE_DYNAMIC,
                0x0,
                &[
                    PhdrSpec {
                        p_type: PT_LOAD,
                        p_flags: 5,
                        p_vaddr: 0,
                        p_memsz: load_bytes.len() as u64,
                        p_align: 0,
                        data: load_bytes,
                    },
                    PhdrSpec {
                        p_type: 0x6100_0000, /* PT_SCE_DYNLIBDATA */
                        p_flags: 4,
                        p_vaddr: 0,
                        p_memsz: dynlib_blob.len() as u64,
                        p_align: 0,
                        data: dynlib_blob,
                    },
                    PhdrSpec {
                        p_type: PT_DYNAMIC,
                        p_flags: 6,
                        p_vaddr: 0x2000,
                        p_memsz: dynamic_bytes.len() as u64,
                        p_align: 0,
                        data: dynamic_bytes,
                    },
                ],
            );

            build_plaintext_self(&elf)
        }

        fn write_executable_sprx(dir: &Path, name: &str, import_nid: u64) -> PathBuf {
            let bytes = build_executable_sprx(import_nid);
            let path = dir.join(name);
            std::fs::write(&path, &bytes).expect("write synthetic executable .sprx to temp dir");
            path
        }

        fn sentinel(_ctx: &raeen_hle::HleContext, _args: &[u64]) -> u64 {
            0xC0DE
        }

        /// The genuine HLE-dispatch acceptance test: the shell's
        /// `FirmwareLauncher` loads a synthetic `_start`-shaped `.sprx` whose
        /// entry calls a real HLE-registered import, moves its return value
        /// into `rdi`, and passes it to the imported `exit` — asserting that
        /// `session_detail` reports the sentinel value as the exit code,
        /// i.e. the module really ran, HLE dispatch really happened, and the
        /// run ended the well-formed process way (M1-A).
        ///
        /// Constructs `FirmwareLauncher` via its private-field struct
        /// literal (this test module is a descendant of `launcher`, so the
        /// fields are visible) rather than `FirmwareLauncher::new()`,
        /// because the sentinel HLE function must be registered on the
        /// *same* `HleRegistry` instance that seeds the `NidDatabase`/
        /// `ModuleRegistry` used to link — exactly the "same instance"
        /// requirement `FirmwareLauncher::new()` itself upholds internally.
        #[test]
        fn play_executes_module_and_reports_sentinel_return_value() {
            let hle = raeen_hle::HleRegistry::new();
            hle.register("libtest", "sceTestSentinel", sentinel);
            let sentinel_nid = raeen_firmware::dynlib::nid::nid_of("sceTestSentinel");
            let exit_nid = raeen_firmware::dynlib::nid::nid_of("exit");

            let launcher = FirmwareLauncher {
                hle: hle.into(),
                sessions: Mutex::new(HashMap::new()),
                next_id: Mutex::new(0),
            };

            const SLOT_SENTINEL_OFF: usize = 0x40;
            const SLOT_EXIT_OFF: usize = 0x48;
            let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic_multi(&[
                (sentinel_nid, SLOT_SENTINEL_OFF as u64),
                (exit_nid, SLOT_EXIT_OFF as u64),
            ]);

            let mut load_bytes = vec![0u8; 0x100];
            // call qword ptr [rip+disp32] -> sentinel slot
            let call1_disp32 = (SLOT_SENTINEL_OFF as i64 - 6) as i32;
            load_bytes[0] = 0xFF;
            load_bytes[1] = 0x15;
            load_bytes[2..6].copy_from_slice(&call1_disp32.to_le_bytes());
            // mov rdi, rax — the sentinel's return value becomes exit's arg
            load_bytes[6..9].copy_from_slice(&[0x48, 0x89, 0xC7]);
            // call qword ptr [rip+disp32] -> exit slot (never returns)
            let call2_disp32 = (SLOT_EXIT_OFF as i64 - 15) as i32;
            load_bytes[9] = 0xFF;
            load_bytes[10] = 0x15;
            load_bytes[11..15].copy_from_slice(&call2_disp32.to_le_bytes());

            let elf = build_elf_with_entry(
                ET_SCE_DYNAMIC,
                0x0,
                &[
                    PhdrSpec {
                        p_type: PT_LOAD,
                        p_flags: 5,
                        p_vaddr: 0,
                        p_memsz: load_bytes.len() as u64,
                        p_align: 0,
                        data: load_bytes,
                    },
                    PhdrSpec {
                        p_type: 0x6100_0000, /* PT_SCE_DYNLIBDATA */
                        p_flags: 4,
                        p_vaddr: 0,
                        p_memsz: dynlib_blob.len() as u64,
                        p_align: 0,
                        data: dynlib_blob,
                    },
                    PhdrSpec {
                        p_type: PT_DYNAMIC,
                        p_flags: 6,
                        p_vaddr: 0x2000,
                        p_memsz: dynamic_bytes.len() as u64,
                        p_align: 0,
                        data: dynamic_bytes,
                    },
                ],
            );
            let sprx_bytes = build_plaintext_self(&elf);

            let tmp = std::env::temp_dir().join(format!(
                "raeen-gui-launcher-exec-test-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = tmp.join("eboot.bin");
            std::fs::write(&path, &sprx_bytes).expect("write synthetic .sprx to temp dir");

            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(
                settled_state(&launcher, &handle),
                SessionState::Running,
                "detail: {:?}",
                launcher.session_detail(&handle)
            );
            let detail = launcher
                .session_detail(&handle)
                .expect("a ran session has detail text");
            assert!(
                detail.starts_with("Ran to exit(0xc0de)"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("2 HLE imports resolved"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("0 unresolved"),
                "unexpected message: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }

        /// A guest call to an unresolved import is inventoried and resumes with
        /// zero by default. The following imported `exit` receives that zero,
        /// proving the compatibility miss did not become a process fault.
        /// `RAEEN_STRICT_NIDS=1` retains fail-fast debugging.
        #[test]
        fn play_resumes_after_an_unresolved_import_by_default() {
            let hle = raeen_hle::HleRegistry::new();
            let bogus_nid =
                raeen_firmware::dynlib::nid::nid_of("totallyUnknownFunctionNobodyRegistered");

            let launcher = FirmwareLauncher {
                hle: hle.into(),
                sessions: Mutex::new(HashMap::new()),
                next_id: Mutex::new(0),
            };

            let tmp = std::env::temp_dir().join(format!(
                "raeen-gui-launcher-unresolved-test-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = write_executable_sprx(&tmp, "eboot.bin", bogus_nid);

            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(settled_state(&launcher, &handle), SessionState::Running);
            let detail = launcher
                .session_detail(&handle)
                .expect("completed run carries a message");
            assert!(
                detail.starts_with("Ran to exit(0x0)"),
                "unresolved call must resume with rax=0; got: {detail}"
            );
            assert!(
                detail.contains("1 unresolved"),
                "the run must retain the unresolved-link inventory: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }

        /// M1-A acceptance test (crt0/process environment through the Shell):
        /// a hand-assembled `_start`-shaped module — entry reads `argc` off
        /// the process stack (`mov rdi, [rsp]`) and passes it straight to the
        /// imported `exit` — launched through the Shell's real
        /// `FirmwareLauncher::launch` path. The launcher passes exactly one
        /// argv entry, so the reported exit code must be 1 == argc: proof the
        /// Shell now enters modules as a real process (`execute_process` +
        /// argc/argv/envp/auxv stack), not a bare 6-register function call
        /// (under which `[rsp]` at entry would be a return address, never 1).
        #[test]
        fn start_shaped_module_reads_argc_from_process_stack_and_exits_with_it() {
            let exit_nid = raeen_firmware::dynlib::nid::nid_of("exit");
            let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic(exit_nid);

            let mut load_bytes = vec![0u8; 0x100];
            // mov rdi, [rsp] — argc, the first thing a real _start reads.
            load_bytes[0..4].copy_from_slice(&[0x48, 0x8B, 0x3C, 0x24]);
            // call qword ptr [rip+disp32] -> the exit relocation slot. exit
            // never returns (the runtime's exit-longjmp ends the run), so
            // nothing follows.
            let rip_after = 4i64 + 6;
            let disp32 = (RELOC_SLOT_OFFSET as i64 - rip_after) as i32;
            load_bytes[4] = 0xFF;
            load_bytes[5] = 0x15;
            load_bytes[6..10].copy_from_slice(&disp32.to_le_bytes());

            let elf = build_elf_with_entry(
                ET_SCE_DYNAMIC,
                0x0,
                &[
                    PhdrSpec {
                        p_type: PT_LOAD,
                        p_flags: 5,
                        p_vaddr: 0,
                        p_memsz: load_bytes.len() as u64,
                        p_align: 0,
                        data: load_bytes,
                    },
                    PhdrSpec {
                        p_type: 0x6100_0000, /* PT_SCE_DYNLIBDATA */
                        p_flags: 4,
                        p_vaddr: 0,
                        p_memsz: dynlib_blob.len() as u64,
                        p_align: 0,
                        data: dynlib_blob,
                    },
                    PhdrSpec {
                        p_type: PT_DYNAMIC,
                        p_flags: 6,
                        p_vaddr: 0x2000,
                        p_memsz: dynamic_bytes.len() as u64,
                        p_align: 0,
                        data: dynamic_bytes,
                    },
                ],
            );
            let sprx_bytes = build_plaintext_self(&elf);

            let tmp = std::env::temp_dir().join(format!(
                "raeen-gui-launcher-argc-test-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = tmp.join("eboot.bin");
            std::fs::write(&path, &sprx_bytes)
                .expect("write synthetic _start-shaped .sprx to temp dir");

            // `FirmwareLauncher::new()`'s default registry already has libc's
            // `exit` registered — no test-local registration needed.
            let launcher = FirmwareLauncher::new();
            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(settled_state(&launcher, &handle), SessionState::Running);
            let detail = launcher
                .session_detail(&handle)
                .expect("a completed session has detail text");
            assert!(
                detail.starts_with("Ran to exit(0x1)"),
                "exit code must equal argc == 1 (one argv entry) — unexpected message: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }

        // --- Homebrew gap-analysis milestone: a "more realistic" module -----
        // Everything above proves a single sentinel/HLE import dispatches
        // correctly. This proves something stronger: a synthetic module
        // whose entry does *real work* through several distinct real
        // HLE-registered imports (not a test-local sentinel) — `malloc`,
        // `memset`, and libkernel's `sceKernelMapFlexibleMemory` — resolved
        // via genuine NIDs, run through the Shell's actual load path
        // (`FirmwareLauncher::launch` -> `load` -> `execute_linked`), both
        // by loading a temp file directly and by discovering it on disk via
        // the real `scan_dir` first. Entirely hand-built buffers; no real
        // firmware bytes anywhere; `NoKeysProvider` throughout (inherited
        // from `FirmwareLauncher::new()`).

        const SLOT_MALLOC_OFF: usize = 0x80;
        const SLOT_MEMSET_OFF: usize = 0x88;
        /// `sceKernelMapFlexibleMemory`'s relocation slot — resolved to a
        /// real HLE trampoline during linking (so it counts toward
        /// `resolved`) even though the entry stub below never actually
        /// calls through it; the point is proving the linker handles
        /// *several* distinct imports, not just the ones a given entry
        /// happens to execute.
        const SLOT_MAP_OFF: usize = 0x90;
        const SCRATCH_OFF: usize = 0x98;
        const SLOT_EXIT_OFF: usize = 0xA0;
        const MALLOC_SIZE: u64 = 0x40;
        const MEMSET_VALUE: u32 = 0xAB;

        /// Writes `mov rdi, malloc_size; call [malloc]; mov [scratch], rax;
        /// mov rdi, [scratch]; mov esi, memset_value; mov edx, malloc_size;
        /// call [memset]; mov rdi, [scratch]; movzx edi, byte [rdi];
        /// call [exit]` — `malloc(N)`, `memset(ptr, byte, N)`, then read byte
        /// 0 of the block back and end the process with it as the exit code
        /// (M1-A: a `_start`-shaped entry ends via exit, it never `ret`s).
        /// Derived from `raeen-runtime/tests/execute.rs`'s
        /// `write_malloc_memset_readback_stub` (not importable across
        /// crates, so replicated here), retailed for process mode.
        #[allow(clippy::too_many_arguments)] // test-fixture assembler; args mirror the stub's slots
        fn write_malloc_memset_readback_stub(
            buf: &mut [u8],
            entry_off: usize,
            slot_malloc_off: usize,
            slot_memset_off: usize,
            slot_exit_off: usize,
            scratch_off: usize,
            malloc_size: u64,
            memset_value: u32,
        ) {
            let mut off = entry_off;

            // mov rdi, malloc_size
            buf[off] = 0x48;
            buf[off + 1] = 0xBF;
            buf[off + 2..off + 10].copy_from_slice(&malloc_size.to_le_bytes());
            off += 10;

            // call qword ptr [rip+disp32]  -> slot_malloc_off
            let call1_rip_after = off as i64 + 6;
            let call1_disp32 = (slot_malloc_off as i64 - call1_rip_after) as i32;
            buf[off] = 0xFF;
            buf[off + 1] = 0x15;
            buf[off + 2..off + 6].copy_from_slice(&call1_disp32.to_le_bytes());
            off += 6;

            // mov [rip+disp32], rax  -> scratch_off (stash the malloc'd pointer)
            let store_rip_after = off as i64 + 7;
            let store_disp32 = (scratch_off as i64 - store_rip_after) as i32;
            buf[off] = 0x48;
            buf[off + 1] = 0x89;
            buf[off + 2] = 0x05;
            buf[off + 3..off + 7].copy_from_slice(&store_disp32.to_le_bytes());
            off += 7;

            // mov rdi, [rip+disp32]  <- scratch_off (memset's dst arg)
            let load1_rip_after = off as i64 + 7;
            let load1_disp32 = (scratch_off as i64 - load1_rip_after) as i32;
            buf[off] = 0x48;
            buf[off + 1] = 0x8B;
            buf[off + 2] = 0x3D;
            buf[off + 3..off + 7].copy_from_slice(&load1_disp32.to_le_bytes());
            off += 7;

            // mov esi, memset_value
            buf[off] = 0xBE;
            buf[off + 1..off + 5].copy_from_slice(&memset_value.to_le_bytes());
            off += 5;

            // mov edx, malloc_size (low 32 bits; MALLOC_SIZE is small here)
            buf[off] = 0xBA;
            buf[off + 1..off + 5].copy_from_slice(&(malloc_size as u32).to_le_bytes());
            off += 5;

            // call qword ptr [rip+disp32]  -> slot_memset_off
            let call2_rip_after = off as i64 + 6;
            let call2_disp32 = (slot_memset_off as i64 - call2_rip_after) as i32;
            buf[off] = 0xFF;
            buf[off + 1] = 0x15;
            buf[off + 2..off + 6].copy_from_slice(&call2_disp32.to_le_bytes());
            off += 6;

            // mov rdi, [rip+disp32]  <- scratch_off (reload the pointer for read-back)
            let load2_rip_after = off as i64 + 7;
            let load2_disp32 = (scratch_off as i64 - load2_rip_after) as i32;
            buf[off] = 0x48;
            buf[off + 1] = 0x8B;
            buf[off + 2] = 0x3D;
            buf[off + 3..off + 7].copy_from_slice(&load2_disp32.to_le_bytes());
            off += 7;

            // movzx edi, byte [rdi] — the read-back byte becomes exit's arg
            buf[off] = 0x0F;
            buf[off + 1] = 0xB6;
            buf[off + 2] = 0x3F;
            off += 3;

            // call qword ptr [rip+disp32]  -> slot_exit_off (never returns)
            let call3_rip_after = off as i64 + 6;
            let call3_disp32 = (slot_exit_off as i64 - call3_rip_after) as i32;
            buf[off] = 0xFF;
            buf[off + 1] = 0x15;
            buf[off + 2..off + 6].copy_from_slice(&call3_disp32.to_le_bytes());
        }

        /// The `PT_SCE_DYNLIBDATA` blob + matching `PT_DYNAMIC` bytes for
        /// several imports at once, each bound to its own relocation slot —
        /// generalizes this module's `build_dynlib_and_dynamic` (single
        /// import) to prove a realistic homebrew module importing several
        /// distinct HLE functions via NID. `imports` is `(nid, slot_offset)`
        /// pairs; symtab index order matches `imports`' order, so each
        /// relocation's `r_sym` is just that import's position.
        fn build_dynlib_and_dynamic_multi(imports: &[(u64, u64)]) -> (Vec<u8>, Vec<u8>) {
            let mut strtab = vec![0u8];
            let mut name_offsets = Vec::with_capacity(imports.len());
            for (nid, _) in imports {
                let name = format!("{}#A#A", raeen_firmware::dynlib::nid::encode_nid(*nid));
                name_offsets.push(strtab.len() as u32);
                strtab.extend_from_slice(name.as_bytes());
                strtab.push(0);
            }

            let mut symtab = Vec::new();
            for &name_off in &name_offsets {
                symtab.extend_from_slice(&name_off.to_le_bytes());
                symtab.push(0);
                symtab.push(0);
                symtab.extend_from_slice(&0u16.to_le_bytes());
                symtab.extend_from_slice(&0u64.to_le_bytes());
                symtab.extend_from_slice(&0u64.to_le_bytes());
            }

            let mut jmprel = Vec::new();
            for (index, (_, slot_off)) in imports.iter().enumerate() {
                jmprel.extend_from_slice(&slot_off.to_le_bytes());
                jmprel.extend_from_slice(
                    &(((index as u64) << 32) | R_X86_64_JUMP_SLOT).to_le_bytes(),
                );
                jmprel.extend_from_slice(&0i64.to_le_bytes());
            }

            let strtab_off = 0u64;
            let symtab_off = strtab.len() as u64;
            let jmprel_off = symtab_off + symtab.len() as u64;

            let mut blob = Vec::new();
            blob.extend_from_slice(&strtab);
            blob.extend_from_slice(&symtab);
            blob.extend_from_slice(&jmprel);

            let mut dynamic = Vec::new();
            let mut push_tag = |tag: u64, val: u64| {
                dynamic.extend_from_slice(&tag.to_le_bytes());
                dynamic.extend_from_slice(&val.to_le_bytes());
            };
            push_tag(DT_SCE_STRTAB, strtab_off);
            push_tag(DT_SCE_STRSZ, strtab.len() as u64);
            push_tag(DT_SCE_SYMTAB, symtab_off);
            push_tag(DT_SCE_SYMTABSZ, symtab.len() as u64);
            push_tag(DT_SCE_SYMENT, 24);
            push_tag(DT_SCE_JMPREL, jmprel_off);
            push_tag(DT_SCE_PLTRELSZ, jmprel.len() as u64);
            push_tag(DT_NULL, 0);

            (blob, dynamic)
        }

        /// A plaintext-SELF-wrapped `.sprx` whose entry calls `malloc` then
        /// `memset` then reads a byte back and exits with it (see
        /// `write_malloc_memset_readback_stub`), importing `malloc`,
        /// `memset`, `sceKernelMapFlexibleMemory`, and `exit` via real NIDs
        /// — `sceKernelMapFlexibleMemory` resolved but never called by this
        /// entry.
        fn build_realistic_homebrew_sprx() -> Vec<u8> {
            let malloc_nid = raeen_firmware::dynlib::nid::nid_of("malloc");
            let memset_nid = raeen_firmware::dynlib::nid::nid_of("memset");
            let map_nid = raeen_firmware::dynlib::nid::nid_of("sceKernelMapFlexibleMemory");
            let exit_nid = raeen_firmware::dynlib::nid::nid_of("exit");

            let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic_multi(&[
                (malloc_nid, SLOT_MALLOC_OFF as u64),
                (memset_nid, SLOT_MEMSET_OFF as u64),
                (map_nid, SLOT_MAP_OFF as u64),
                (exit_nid, SLOT_EXIT_OFF as u64),
            ]);

            let mut load_bytes = vec![0u8; 0x100];
            write_malloc_memset_readback_stub(
                &mut load_bytes,
                0x0,
                SLOT_MALLOC_OFF,
                SLOT_MEMSET_OFF,
                SLOT_EXIT_OFF,
                SCRATCH_OFF,
                MALLOC_SIZE,
                MEMSET_VALUE,
            );

            let elf = build_elf_with_entry(
                ET_SCE_DYNAMIC,
                0x0,
                &[
                    PhdrSpec {
                        p_type: PT_LOAD,
                        p_flags: 7,
                        p_vaddr: 0,
                        p_memsz: load_bytes.len() as u64,
                        p_align: 0,
                        data: load_bytes,
                    },
                    PhdrSpec {
                        p_type: 0x6100_0000, /* PT_SCE_DYNLIBDATA */
                        p_flags: 4,
                        p_vaddr: 0,
                        p_memsz: dynlib_blob.len() as u64,
                        p_align: 0,
                        data: dynlib_blob,
                    },
                    PhdrSpec {
                        p_type: PT_DYNAMIC,
                        p_flags: 6,
                        p_vaddr: 0x2000,
                        p_memsz: dynamic_bytes.len() as u64,
                        p_align: 0,
                        data: dynamic_bytes,
                    },
                ],
            );

            build_plaintext_self(&elf)
        }

        fn write_realistic_homebrew_sprx(dir: &Path, name: &str) -> PathBuf {
            let bytes = build_realistic_homebrew_sprx();
            let path = dir.join(name);
            std::fs::write(&path, &bytes)
                .expect("write synthetic realistic-homebrew .sprx to temp dir");
            path
        }

        /// Part 2 "Direct" of the homebrew end-to-end proof: the Shell's
        /// real `FirmwareLauncher::launch` (-> `load` -> `execute_linked`)
        /// runs this realistic synthetic module to completion — a real
        /// `malloc`, a real `memset`, and a real read-back all actually
        /// happen through the genuine HLE registry and runtime, and the
        /// linker resolves all three imports even though the guest entry
        /// only calls two of them.
        #[test]
        fn realistic_homebrew_module_executes_malloc_memset_readback_through_firmware_launcher() {
            let tmp = std::env::temp_dir().join(format!(
                "raeen-gui-realistic-homebrew-direct-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = write_realistic_homebrew_sprx(&tmp, "eboot.bin");

            let launcher = FirmwareLauncher::new();
            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(settled_state(&launcher, &handle), SessionState::Running);
            let detail = launcher
                .session_detail(&handle)
                .expect("a ran session has detail text");
            assert!(
                detail.starts_with("Ran to exit(0xab)"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("4 HLE imports resolved"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("0 unresolved"),
                "unexpected message: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }

        /// Part 2 "Full scan→launch" of the homebrew end-to-end proof: the
        /// same realistic module, written as `Games/<name>/eboot.bin` (with
        /// an `raeen-title.toml` alongside it), discovered by the Shell's
        /// real `scan_dir`, then launched through `FirmwareLauncher` using
        /// the discovered `LaunchTarget` exactly as the Shell's Play button
        /// would — proving the entire path: discover on disk -> load -> run
        /// -> outcome.
        #[test]
        fn realistic_homebrew_discovered_by_scan_dir_then_launched_through_firmware_launcher() {
            let tmp = std::env::temp_dir().join(format!(
                "raeen-gui-realistic-homebrew-scan-{}",
                std::process::id()
            ));
            let game_dir = tmp.join("Games").join("realistic-homebrew-demo");
            std::fs::create_dir_all(&game_dir).expect("create temp game dir");

            let bytes = build_realistic_homebrew_sprx();
            let eboot_path = game_dir.join("eboot.bin");
            std::fs::write(&eboot_path, &bytes)
                .expect("write synthetic realistic-homebrew .sprx to temp dir");
            std::fs::write(
                game_dir.join("raeen-title.toml"),
                "title = \"Realistic Homebrew Demo\"\n",
            )
            .expect("write optional title metadata");

            let games_root = tmp.join("Games");
            let items = crate::library::scan::scan_dir(&games_root);
            assert_eq!(
                items.len(),
                1,
                "scan_dir should discover exactly the one synthetic game folder"
            );
            let item = &items[0];
            assert_eq!(item.title, "Realistic Homebrew Demo");
            let LaunchTarget::Game { path } = &item.launch else {
                panic!("expected a Game launch target");
            };
            assert_eq!(path, &eboot_path);

            let launcher = FirmwareLauncher::new();
            let handle = launcher
                .launch(&item.launch)
                .expect("launch always returns a handle");

            assert_eq!(settled_state(&launcher, &handle), SessionState::Running);
            let detail = launcher
                .session_detail(&handle)
                .expect("a ran session has detail text");
            assert!(
                detail.starts_with("Ran to exit(0xab)"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("4 HLE imports resolved"),
                "unexpected message: {detail}"
            );
            assert!(
                detail.contains("0 unresolved"),
                "unexpected message: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }

        // --- M1 FINAL acceptance: compiler-produced homebrew ---------------
        //
        // Everything above is *synthetic* (hand-assembled machine code +
        // hand-laid dynlib blobs). The M1 gate proper requires a
        // **toolchain-built** binary: nightly `rustc` compiles a real
        // `no_std` guest to a Linux `cdylib`, which is re-wrapped as an SCE
        // `eboot.bin` and driven through the *same* Shell launch path. It
        // proves, with compiler-emitted code, the full M1 wall stack at once:
        // crt0 `argc` read at `_start`, a `#[thread_local]` resolved through a
        // real `R_X86_64_TPOFF64`, the `fs:0x28` stack canary (`-Z
        // stack-protector=all`), and real `printf`/`write` HLE dispatch.

        /// The exact validated guest program (see the module-level rationale):
        /// `guest_main` must return normally (else `-Z stack-protector=all`
        /// elides the canary), `-Bsymbolic` keeps defined symbols from
        /// becoming imports, and the `rust_eh_personality` stub avoids a
        /// spurious unresolved import.
        const GUEST_SRC: &str = r####"#![no_std]
#![no_main]
#![feature(thread_local)]
use core::panic::PanicInfo;
#[thread_local]
static mut TLS_VAR: u64 = 0xAB;
extern "C" {
    fn printf(fmt: *const u8, ...) -> i32;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn exit(code: i32) -> !;
}
#[no_mangle]
pub extern "C" fn guest_main(argc: u64) -> u64 {
    let mut buf = [0u8; 24];
    let mut i = 0usize;
    while i < buf.len() { buf[i] = (argc as u8).wrapping_add(i as u8); i += 1; }
    unsafe {
        let tls = TLS_VAR.wrapping_add(buf[0] as u64);
        TLS_VAR = tls;
        printf(b"argc=%d tls=%d\n\0".as_ptr(), argc as i32, tls as i32);
        let msg = b"bye\n\0";
        write(1, msg.as_ptr(), 4);
        tls
    }
}
core::arch::global_asm!(
    ".global _start", "_start:",
    "  mov rdi, [rsp]", "  call guest_main", "  xor edi, edi", "  call exit",
);
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
"####;

        /// Read a NUL-terminated string from `buf` starting at `off`.
        fn read_cstr(buf: &[u8], off: usize) -> Option<String> {
            let rest = buf.get(off..)?;
            let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
            Some(String::from_utf8_lossy(&rest[..end]).into_owned())
        }

        /// Compile [`GUEST_SRC`] with the validated nightly-rustc invocation.
        /// Returns `None` (the test then skips) if the toolchain — nightly
        /// `rustc`, the `x86_64-unknown-linux-gnu` target, or `rust-lld` —
        /// isn't available or the build fails for any reason.
        fn build_guest_so(dir: &Path) -> Option<Vec<u8>> {
            let src = dir.join("guest.rs");
            std::fs::write(&src, GUEST_SRC).ok()?;
            let so = dir.join("guest.so");

            // A single space-joined `-C link-args=...` value (the linker
            // splits it on spaces).
            let link_args = "-shared -z now -nostdlib -e _start \
                 --no-dynamic-linker --allow-shlib-undefined -Bsymbolic";
            let output = std::process::Command::new("rustc")
                .arg("+nightly")
                .arg(&src)
                .args([
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "--edition",
                    "2021",
                    "--crate-type",
                    "cdylib",
                    "-C",
                    "panic=abort",
                    "-C",
                    "relocation-model=pic",
                    "-Z",
                    "stack-protector=all",
                    "-Z",
                    "tls-model=initial-exec",
                    "-C",
                    "linker=rust-lld",
                    "-C",
                    "linker-flavor=ld.lld",
                    "-C",
                ])
                .arg(format!("link-args={link_args}"))
                .arg("-o")
                .arg(&so)
                .output()
                .ok()?;
            if !output.status.success() {
                eprintln!(
                    "SKIP: guest.so build failed (nightly rustc / linux target / rust-lld \
                     unavailable):\n{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return None;
            }
            std::fs::read(&so).ok()
        }

        /// Re-synthesize a compiler-built Linux `cdylib` (`elf_bytes`) as a
        /// plaintext-SELF SCE `eboot.bin`: every `PT_LOAD` and the `PT_TLS`
        /// are carried through with their original vaddrs (the linker lays a
        /// flat `image[p_vaddr..]`), the `.dynsym` is copied 1:1 with each
        /// undefined import's name rewritten to its `encode_nid(nid)#A#A` SCE
        /// form, and every dynamic relocation is copied verbatim — `r_offset`,
        /// `r_info` (so `r_sym`/`r_type` are preserved), and `r_addend` (load
        /// -bearing for `TPOFF64`, whose TLS offset lives in the addend, and
        /// for the `RELATIVE` addend-based fixups) — bucketed into
        /// `DT_SCE_JMPREL` (JUMP_SLOT) vs `DT_SCE_RELA` (everything else).
        fn synthesize_sce_eboot(elf_bytes: &[u8]) -> Option<Vec<u8>> {
            use goblin::elf::Elf;
            const PT_TLS: u32 = 7;
            const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;
            const R_JUMP_SLOT: u32 = 7;

            let elf = Elf::parse(elf_bytes).ok()?;
            let entry = elf.header.e_entry;

            // PT_LOAD (vaddrs preserved) + the single PT_TLS template.
            let mut phdrs: Vec<PhdrSpec> = Vec::new();
            let mut tls_spec: Option<PhdrSpec> = None;
            for ph in elf.program_headers.iter() {
                if ph.p_type != PT_LOAD && ph.p_type != PT_TLS {
                    continue;
                }
                let start = ph.p_offset as usize;
                let end = start.checked_add(ph.p_filesz as usize)?;
                let data = elf_bytes.get(start..end)?.to_vec();
                let spec = PhdrSpec {
                    p_type: ph.p_type,
                    p_flags: ph.p_flags,
                    p_vaddr: ph.p_vaddr,
                    p_memsz: ph.p_memsz,
                    p_align: ph.p_align,
                    data,
                };
                if ph.p_type == PT_TLS {
                    tls_spec = Some(spec);
                } else {
                    phdrs.push(spec);
                }
            }
            // The fixture has a `#[thread_local]`, so a PT_TLS must exist —
            // its absence means the compiler produced something unexpected.
            let tls_spec = tls_spec?;

            // Locate the dynamic-linking sections by name, read raw.
            let section = |name: &str| -> Option<(usize, usize)> {
                elf.section_headers.iter().find_map(|sh| {
                    if elf.shdr_strtab.get_at(sh.sh_name) == Some(name) {
                        Some((sh.sh_offset as usize, sh.sh_size as usize))
                    } else {
                        None
                    }
                })
            };
            let (dynsym_off, dynsym_sz) = section(".dynsym")?;
            let (dynstr_off, dynstr_sz) = section(".dynstr")?;
            let dynsym_raw = elf_bytes.get(dynsym_off..dynsym_off.checked_add(dynsym_sz)?)?;
            let dynstr_raw = elf_bytes.get(dynstr_off..dynstr_off.checked_add(dynstr_sz)?)?;

            // Index-preserving `.dynsym` copy: undefined named symbols become
            // NID imports (so the linker NID-resolves them to HLE); every
            // other entry (the null symbol, defined symbols) is a benign,
            // never-relocation-referenced placeholder with `is_import` false.
            let mut strtab: Vec<u8> = vec![0u8];
            let mut symtab: Vec<u8> = Vec::new();
            for sym in dynsym_raw.chunks_exact(24) {
                let st_name = u32::from_le_bytes(sym[0..4].try_into().unwrap());
                let st_shndx = u16::from_le_bytes(sym[6..8].try_into().unwrap());
                if st_shndx == 0 && st_name != 0 {
                    let cname = read_cstr(dynstr_raw, st_name as usize)?;
                    let nid = raeen_firmware::dynlib::nid::nid_of(&cname);
                    let sce_name = format!("{}#A#A", raeen_firmware::dynlib::nid::encode_nid(nid));
                    let name_off = strtab.len() as u32;
                    strtab.extend_from_slice(sce_name.as_bytes());
                    strtab.push(0);
                    symtab.extend_from_slice(&name_off.to_le_bytes()); // st_name
                    symtab.push(0x10); // st_info = STB_GLOBAL | STT_NOTYPE
                    symtab.push(0); // st_other
                    symtab.extend_from_slice(&0u16.to_le_bytes()); // st_shndx = UNDEF
                    symtab.extend_from_slice(&0u64.to_le_bytes()); // st_value
                    symtab.extend_from_slice(&0u64.to_le_bytes()); // st_size
                } else {
                    symtab.extend_from_slice(&0u32.to_le_bytes()); // st_name = 0
                    symtab.push(0x10);
                    symtab.push(0);
                    symtab.extend_from_slice(&1u16.to_le_bytes()); // st_shndx != 0 (defined)
                    symtab.extend_from_slice(&1u64.to_le_bytes()); // st_value != 0
                    symtab.extend_from_slice(&0u64.to_le_bytes());
                }
            }

            // Every dynamic relocation, verbatim, bucketed by type.
            let mut rela: Vec<u8> = Vec::new();
            let mut jmprel: Vec<u8> = Vec::new();
            for name in [".rela.dyn", ".rela.plt"] {
                let Some((off, sz)) = section(name) else {
                    continue;
                };
                let raw = elf_bytes.get(off..off.checked_add(sz)?)?;
                for r in raw.chunks_exact(24) {
                    let r_info = u64::from_le_bytes(r[8..16].try_into().unwrap());
                    if (r_info & 0xFFFF_FFFF) as u32 == R_JUMP_SLOT {
                        jmprel.extend_from_slice(r);
                    } else {
                        rela.extend_from_slice(r);
                    }
                }
            }

            // Blob layout: [strtab][symtab][rela][jmprel]; all dynamic-table
            // offsets below are offsets *into this blob*.
            let strtab_off = 0u64;
            let symtab_off = strtab.len() as u64;
            let rela_off = symtab_off + symtab.len() as u64;
            let jmprel_off = rela_off + rela.len() as u64;
            let mut blob = Vec::new();
            blob.extend_from_slice(&strtab);
            blob.extend_from_slice(&symtab);
            blob.extend_from_slice(&rela);
            blob.extend_from_slice(&jmprel);

            let mut dynamic = Vec::new();
            let mut push_tag = |tag: u64, val: u64| {
                dynamic.extend_from_slice(&tag.to_le_bytes());
                dynamic.extend_from_slice(&val.to_le_bytes());
            };
            push_tag(DT_SCE_STRTAB, strtab_off);
            push_tag(DT_SCE_STRSZ, strtab.len() as u64);
            push_tag(DT_SCE_SYMTAB, symtab_off);
            push_tag(DT_SCE_SYMTABSZ, symtab.len() as u64);
            push_tag(DT_SCE_SYMENT, 24);
            push_tag(DT_SCE_RELA, rela_off);
            push_tag(DT_SCE_RELASZ, rela.len() as u64);
            push_tag(DT_SCE_RELAENT, 24);
            push_tag(DT_SCE_JMPREL, jmprel_off);
            push_tag(DT_SCE_PLTRELSZ, jmprel.len() as u64);
            push_tag(DT_NULL, 0);

            // PT_LOADs + PT_TLS + PT_SCE_DYNLIBDATA + PT_DYNAMIC. The dynlib
            // and dynamic segments are parsed by file offset (not mapped by
            // vaddr), so their vaddrs are irrelevant — set to 0.
            phdrs.push(tls_spec);
            phdrs.push(PhdrSpec {
                p_type: PT_SCE_DYNLIBDATA,
                p_flags: 4,
                p_vaddr: 0,
                p_memsz: blob.len() as u64,
                p_align: 0,
                data: blob,
            });
            phdrs.push(PhdrSpec {
                p_type: PT_DYNAMIC,
                p_flags: 6,
                p_vaddr: 0,
                p_memsz: dynamic.len() as u64,
                p_align: 0,
                data: dynamic,
            });

            let sce_elf = build_elf_with_entry(ET_SCE_DYNAMIC, entry, &phdrs);
            Some(build_plaintext_self(&sce_elf))
        }

        /// Build the M1 fixture end-to-end, writing it as
        /// `<dir>/Games/compiler-homebrew/eboot.bin`. `None` ⇒ skip.
        fn build_compiler_homebrew_eboot(dir: &Path) -> Option<PathBuf> {
            let elf_bytes = build_guest_so(dir)?;
            let sce = synthesize_sce_eboot(&elf_bytes)?;
            let game_dir = dir.join("Games").join("compiler-homebrew");
            std::fs::create_dir_all(&game_dir).ok()?;
            let eboot = game_dir.join("eboot.bin");
            std::fs::write(&eboot, &sce).ok()?;
            Some(eboot)
        }

        /// M1 FINAL acceptance. Compiler-emitted machine code runs through the
        /// Shell's real `FirmwareLauncher::launch` → `load_module` →
        /// `execute_process` path and produces byte-exact observable output.
        #[test]
        fn compiler_built_homebrew_runs_through_shell_and_prints() {
            let tmp = std::env::temp_dir().join(format!(
                "raeen-gui-compiler-homebrew-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");

            let Some(eboot) = build_compiler_homebrew_eboot(&tmp) else {
                // Toolchain unavailable — skip cleanly (see `build_guest_so`).
                let _ = std::fs::remove_dir_all(&tmp);
                return;
            };

            let launcher = FirmwareLauncher::new();
            let handle = launcher
                .launch(&LaunchTarget::Game { path: eboot })
                .expect("launch always returns a handle");

            let detail = launcher
                .session_detail(&handle)
                .expect("a ran session has detail text");
            assert_eq!(
                settled_state(&launcher, &handle),
                SessionState::Running,
                "detail: {detail}"
            );
            // `_start` runs `guest_main` then `exit(0)`, the well-formed way.
            assert!(
                detail.starts_with("Ran to exit(0x0)"),
                "compiler homebrew should exit(0) — detail: {detail}"
            );
            // printf, write, exit, memset, __stack_chk_fail — all HLE.
            assert!(
                detail.contains("5 HLE imports resolved"),
                "detail: {detail}"
            );
            assert!(detail.contains("0 unresolved"), "detail: {detail}");

            // Byte-exact guest stdout: printf's `argc=1 tls=172\n` (argc == 1
            // from the process stack; tls == 0xAB + buf[0] == 0xAB + argc ==
            // 172) followed by write's `bye\n`.
            let console = launcher
                .sessions
                .lock()
                .unwrap()
                .get(&handle.0)
                .and_then(|session| session.kernel.as_ref())
                .expect("game session retains its process kernel")
                .console
                .contents();
            assert_eq!(
                console, "argc=1 tls=172\nbye\n",
                "unexpected guest console output; detail: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }
    }
}
