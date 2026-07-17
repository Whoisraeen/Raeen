//! Shell ↔ engine seam (spec §5).
//!
//! The Shell contains **no emulation logic**. It talks to the engine only
//! through [`GameLauncher`], so the Shell can be built and tested against
//! [`StubLauncher`] long before the real engine can run anything. SM3 swaps
//! `StubLauncher` for the real engine implementation without touching Shell
//! navigation or rendering code.

use crate::library::LaunchTarget;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use thiserror::Error;
use xps5x_core::error::FirmwareError;

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
/// [`xps5x_runtime::GUEST_ARENA_BASE`] — RT2's `GuestArena` always
/// identity-maps a module's image at that fixed base (guest address `A` is
/// host address `A`), so a mismatched link base would make any
/// `R_X86_64_RELATIVE` relocation resolve to the wrong host address.
const DEFAULT_LOAD_BASE: u64 = xps5x_runtime::GUEST_ARENA_BASE;

/// `argv[0]` every launched module sees (M1-A, crt0/process environment):
/// the PS4/PS5 convention mounts a title's content at `/app0`, so its main
/// module is `/app0/eboot.bin` regardless of where the file lives on the
/// host. The host path is deliberately *not* leaked into the guest — a real
/// filesystem mapping layer (host dir ↔ `/app0`) comes with save-data/file
/// I/O work, but the argv convention is stable now.
#[cfg(target_os = "windows")]
const GUEST_ARGV0: &str = "/app0/eboot.bin";

/// What came of trying to load+link (and, on Windows, run) one module for a
/// launch.
#[derive(Debug, Clone)]
enum SessionOutcome {
    /// SELF decrypt -> `.sprx` parse -> dynlibdata decode -> NID link all
    /// succeeded, but the module was not executed. This is RT0/RT1b's
    /// non-Windows fallback: `xps5x_runtime::execute_linked` is
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
    /// `jmp` with no return address; see `xps5x_runtime::execute_process`),
    /// tolerated and reported honestly rather than treated as a fault.
    Ran {
        returned: u64,
        resolved: usize,
        unresolved: usize,
    },
    /// Linked and executed as a real process (Windows only, M1-A):
    /// `xps5x_runtime::execute_process` entered the module's `_start` on a
    /// genuine argc/argv/envp/auxv stack and the module ended itself via an
    /// exit-family call — the well-formed way a process run ends. This does
    /// not mean the module "plays" anything; it means the crt0/process
    /// contract held from Shell to exit.
    Exited {
        code: u64,
        resolved: usize,
        unresolved: usize,
    },
    /// Anything that stopped short of a successful run: no module file at
    /// the target path, an encrypted module with no matching key, a genuine
    /// parse/link error, an unresolved HLE import actually called, or a
    /// guest fault during execution. Carries the message the overlay shows
    /// verbatim.
    Faulted(String),
}

struct FirmwareSession {
    outcome: SessionOutcome,
    quit_requested: bool,
}

/// Wires the Shell to the real firmware spine: [`xps5x_firmware::load_module`]
/// (SELF decrypt-or-passthrough -> `.sprx` parse -> dynlibdata decode -> NID
/// link against HLE). This is SM3's whole point — the Shell no longer talks
/// to a stub, it talks to the actual engine entry point — but the engine
/// itself only *links* a module here; nothing executes it yet (that's the
/// next milestone). See [`SessionOutcome::Linked`].
///
/// Holds a [`xps5x_firmware::NoKeysProvider`] and never anything else: the
/// Shell holds no key material of its own (clean-room boundary, spec §2),
/// so an encrypted retail module always faults informatively rather than
/// decrypting anything.
///
/// `load_module` takes `&mut ModuleRegistry`, but [`GameLauncher::launch`]
/// only gets `&self` (the Shell stores one launcher behind `Box<dyn
/// GameLauncher>` and never needs `&mut` access to it). `StubLauncher`
/// solves the same problem with per-field `Mutex`es; `FirmwareLauncher`
/// follows suit — the registry lives behind a `std::sync::Mutex` and is
/// locked only for the duration of one `load_module` call.
pub struct FirmwareLauncher {
    hle: std::sync::Arc<xps5x_hle::HleRegistry>,
    /// The live emulated kernel HLE calls get access to via
    /// [`xps5x_hle::HleContext::kernel`] (dispatch-context milestone — see
    /// `xps5x_runtime::execute_linked`'s doc comment). One instance per
    /// launcher, not per launch: kernel state (the virtual memory manager,
    /// thread manager, ...) is process-wide, matching a real PS5's single
    /// kernel serving every loaded module.
    kernel: std::sync::Arc<xps5x_kernel::OrbisKernel>,
    registry: Mutex<xps5x_firmware::ModuleRegistry>,
    sessions: Mutex<HashMap<u64, FirmwareSession>>,
    next_id: Mutex<u64>,
}

impl FirmwareLauncher {
    pub fn new() -> Self {
        let hle = xps5x_hle::HleRegistry::new();
        let nid_db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        Self {
            hle: std::sync::Arc::new(hle),
            kernel: std::sync::Arc::new(xps5x_kernel::OrbisKernel::new()),
            registry: Mutex::new(xps5x_firmware::ModuleRegistry::new(nid_db)),
            sessions: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Read `path` and, if that succeeds, load+link it, then (on Windows)
    /// actually run its entry point through `xps5x_runtime::execute_linked`.
    /// Every failure mode — unreadable file, missing key, malformed module,
    /// unresolved-import call, guest fault — becomes a
    /// [`SessionOutcome::Faulted`] with a message fit to show the user; this
    /// never panics.
    ///
    /// `self.registry`'s lock is held only for the `load_module` call
    /// itself — execution (which needs `&self.hle`, not the registry) runs
    /// with the lock already released, since `self.hle` lives outside the
    /// `Mutex<ModuleRegistry>` and is borrowed directly.
    fn load(&self, path: &Path) -> SessionOutcome {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) => {
                return SessionOutcome::Faulted(format!(
                    "No module file at {}: {err}",
                    path.display()
                ));
            }
        };

        self.kernel
            .filesystem
            .set_game_directory(path.parent().unwrap_or_else(|| Path::new(".")));
        let title_dir = path.parent().and_then(Path::file_name).unwrap_or_default();
        let writable_root = std::env::temp_dir().join("xps5x").join(title_dir);
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
        self.kernel.filesystem.set_temp_directory(&temp_dir);
        self.kernel.filesystem.set_download_directory(&download_dir);
        self.kernel.filesystem.set_savedata_directory(&savedata_dir);

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
        let loaded = {
            let mut registry = self.registry.lock().unwrap();
            xps5x_firmware::load_process(
                &bytes,
                game_dir,
                &xps5x_firmware::NoKeysProvider,
                &mut registry,
                &self.hle,
                DEFAULT_LOAD_BASE,
            )
        };

        let linked = match loaded.map(|process| process.linked) {
            Ok(linked) => linked,
            Err(FirmwareError::MissingKey { .. }) => {
                return SessionOutcome::Faulted(
                    "Encrypted module — no KeyProvider configured (Settings ▸ Key Provider)"
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
        self.kernel.register_module(xps5x_kernel::ModuleInfo {
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

        #[cfg(target_os = "windows")]
        {
            // M1-A: enter the module as a real process — `_start` on a
            // genuine argc/argv/envp/auxv stack (`execute_process`), not a
            // bare 6-register function call. A well-formed run ends via an
            // exit-family call (`Exited`); a `_start` that returns anyway is
            // reported as `Ran` (malformed but tolerated).
            match xps5x_runtime::execute_process_shared(
                std::sync::Arc::clone(&linked),
                std::sync::Arc::clone(&self.hle),
                std::sync::Arc::clone(&self.kernel),
                &[GUEST_ARGV0],
                &[],
            ) {
                Ok(xps5x_runtime::RunOutcome::Exited(code)) => SessionOutcome::Exited {
                    code,
                    resolved,
                    unresolved,
                },
                Ok(xps5x_runtime::RunOutcome::Returned(returned)) => SessionOutcome::Ran {
                    returned,
                    resolved,
                    unresolved,
                },
                Err(xps5x_runtime::RuntimeError::Faulted { addr, access, kind }) => {
                    SessionOutcome::Faulted(format!(
                        "Faulted at {addr:#x} during execution ({kind} of {access:#x})"
                    ))
                }
                // The guest asked for an import nothing implements. Name it:
                // this is the one fault the user (or we) can actually act on,
                // and it used to read as an anonymous address.
                Err(xps5x_runtime::RuntimeError::UnimplementedImport { nid, .. }) => {
                    let library = linked
                        .unresolved_stubs
                        .iter()
                        .find(|s| s.nid == nid)
                        .and_then(|s| s.library.as_deref())
                        .unwrap_or("unknown library");
                    SessionOutcome::Faulted(format!(
                        "Unimplemented import: {} ({library}) — the game called a function \
                         XPS5X does not provide yet",
                        xps5x_firmware::dynlib::nid_names::describe(nid)
                    ))
                }
                Err(xps5x_runtime::RuntimeError::UnresolvedTrampoline(a)) => {
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
        let outcome = match target {
            LaunchTarget::Game { path } => self.load(path),
            // Built-in apps (Store, Game Library, Settings) aren't modules;
            // there's no path to read, so this can't even attempt a load.
            LaunchTarget::App { id } => {
                SessionOutcome::Faulted(format!("'{id}' is not a loadable module"))
            }
        };

        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;

        self.sessions.lock().unwrap().insert(
            id,
            FirmwareSession {
                outcome,
                quit_requested: false,
            },
        );

        Ok(SessionHandle(id))
    }

    fn session_state(&self, handle: &SessionHandle) -> SessionState {
        let sessions = self.sessions.lock().unwrap();
        match sessions.get(&handle.0) {
            None => SessionState::Faulted,
            Some(session) if session.quit_requested => SessionState::Exited,
            Some(session) => match &session.outcome {
                SessionOutcome::Linked { .. } => SessionState::Running,
                SessionOutcome::Ran { .. } => SessionState::Running,
                // A run that ended via exit() stays `Running` so the session
                // overlay shows the outcome until the user quits — the Shell
                // auto-returns Home the moment it polls `Exited` (see
                // `shell/mod.rs`), which would flash past the detail text.
                SessionOutcome::Exited { .. } => SessionState::Running,
                SessionOutcome::Faulted(_) => SessionState::Faulted,
            },
        }
    }

    fn session_detail(&self, handle: &SessionHandle) -> Option<String> {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(&handle.0).map(|session| match &session.outcome {
            SessionOutcome::Linked { resolved, unresolved, .. } => format!(
                "Linked — {resolved} imports resolved to HLE, {unresolved} unresolved · execution not yet implemented (Esc to return)"
            ),
            SessionOutcome::Ran { returned, resolved, unresolved } => format!(
                "Executed — _start returned {returned:#x} (malformed: a process ends via exit) · \
                 {resolved} HLE imports resolved, {unresolved} unresolved"
            ),
            SessionOutcome::Exited { code, resolved, unresolved } => format!(
                "Ran to exit({code:#x}) — {resolved} HLE imports resolved, {unresolved} unresolved \
                 (early runtime — full game execution needs more HLE breadth)"
            ),
            SessionOutcome::Faulted(message) => message.clone(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn target() -> LaunchTarget {
        LaunchTarget::Game {
            path: PathBuf::from("Games/nova/eboot.bin"),
        }
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
    // `xps5x-firmware/tests/homebrew_pipeline.rs` (not importable across
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
        assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
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
        assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
        let detail = launcher
            .session_detail(&handle)
            .expect("fault carries a message");
        assert!(
            detail.contains("not a loadable module"),
            "unexpected message: {detail}"
        );
    }

    #[test]
    fn valid_synthetic_module_links_and_exposes_resolved_counts() {
        let tmp =
            std::env::temp_dir().join(format!("xps5x-gui-launcher-test-{}", std::process::id()));
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
            assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
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
            assert_eq!(launcher.session_state(&handle), SessionState::Running);
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
            "xps5x-gui-launcher-quit-test-{}",
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
        assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
        #[cfg(not(target_os = "windows"))]
        assert_eq!(launcher.session_state(&handle), SessionState::Running);

        launcher.quit(&handle).expect("quit should succeed");
        assert_eq!(launcher.session_state(&handle), SessionState::Exited);

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
    // Mirrors `xps5x-firmware/tests/homebrew_pipeline.rs`'s
    // `build_dynlib_and_dynamic`/`build_elf` (a real PT_SCE_DYNLIBDATA +
    // PT_DYNAMIC declaring one import) and `xps5x-runtime/tests/execute.rs`'s
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

        /// `call qword ptr [rip+disp32]; ret` at `entry_off`, calling through
        /// the pointer slot at `slot_off`. Mirrors `execute.rs`'s
        /// `write_entry_stub`.
        fn write_entry_stub(buf: &mut [u8], entry_off: usize, slot_off: usize) {
            let rip_after_instr = entry_off as i64 + 6; // FF 15 <disp32> is 6 bytes.
            let disp32 = (slot_off as i64 - rip_after_instr) as i32;

            buf[entry_off] = 0xFF;
            buf[entry_off + 1] = 0x15;
            buf[entry_off + 2..entry_off + 6].copy_from_slice(&disp32.to_le_bytes());
            buf[entry_off + 6] = 0xC3; // ret
        }

        /// The `PT_SCE_DYNLIBDATA` blob (strtab + one undefined `Elf64_Sym` +
        /// one JMPREL `Elf64_Rela`) and matching `PT_DYNAMIC` bytes for a
        /// single import identified by `import_nid`. Mirrors
        /// `homebrew_pipeline.rs`'s `build_dynlib_and_dynamic`.
        fn build_dynlib_and_dynamic(import_nid: u64) -> (Vec<u8>, Vec<u8>) {
            let import_name = format!(
                "{}#A#A",
                xps5x_firmware::dynlib::nid::encode_nid(import_nid)
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

        /// A plaintext-SELF-wrapped `.sprx` whose entry (offset 0) calls the
        /// import `import_nid` through a real JUMP_SLOT relocation.
        fn build_executable_sprx(import_nid: u64) -> Vec<u8> {
            let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic(import_nid);

            let mut load_bytes = vec![0u8; 0x100];
            write_entry_stub(&mut load_bytes, 0x0, RELOC_SLOT_OFFSET as usize);

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

        fn sentinel(_ctx: &xps5x_hle::HleContext, _args: &[u64]) -> u64 {
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
            let hle = xps5x_hle::HleRegistry::new();
            hle.register("libtest", "sceTestSentinel", sentinel);
            let sentinel_nid = xps5x_firmware::dynlib::nid::nid_of("sceTestSentinel");
            let exit_nid = xps5x_firmware::dynlib::nid::nid_of("exit");

            let nid_db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
            let launcher = FirmwareLauncher {
                hle: hle.into(),
                kernel: xps5x_kernel::OrbisKernel::new().into(),
                registry: Mutex::new(xps5x_firmware::ModuleRegistry::new(nid_db)),
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
                "xps5x-gui-launcher-exec-test-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = tmp.join("eboot.bin");
            std::fs::write(&path, &sprx_bytes).expect("write synthetic .sprx to temp dir");

            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(launcher.session_state(&handle), SessionState::Running);
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

        /// A guest `call` to an import nobody registered must tell the user
        /// **which function is missing**, by name and library — not just that
        /// something faulted.
        ///
        /// The linker gives each distinct unresolved NID its own stub address
        /// (`UNRESOLVED_STUB_BASE + i*8`); calling one is a genuine access
        /// violation outside RT0's trampoline guard, which the VEH recovers and
        /// then maps back through the stub table. Before that per-NID scheme
        /// every missing import shared one address and this test could only
        /// assert "Faulted at 0x5000000000000" — a message that named nothing
        /// and gave nobody a next step.
        #[test]
        fn play_faults_cleanly_when_the_module_calls_an_unresolved_import() {
            let hle = xps5x_hle::HleRegistry::new();
            let bogus_nid =
                xps5x_firmware::dynlib::nid::nid_of("totallyUnknownFunctionNobodyRegistered");

            let nid_db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
            let launcher = FirmwareLauncher {
                hle: hle.into(),
                kernel: xps5x_kernel::OrbisKernel::new().into(),
                registry: Mutex::new(xps5x_firmware::ModuleRegistry::new(nid_db)),
                sessions: Mutex::new(HashMap::new()),
                next_id: Mutex::new(0),
            };

            let tmp = std::env::temp_dir().join(format!(
                "xps5x-gui-launcher-unresolved-test-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = write_executable_sprx(&tmp, "eboot.bin", bogus_nid);

            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
            let detail = launcher
                .session_detail(&handle)
                .expect("fault carries a message");
            // The message must NAME the missing import, not just an address.
            let encoded = xps5x_firmware::dynlib::nid::encode_nid(bogus_nid);
            assert!(
                detail.contains("Unimplemented import") && detail.contains(&encoded),
                "the fault must name the missing import ({encoded}); got: {detail}"
            );
            // Pin the old, useless message as gone.
            assert!(
                !detail.starts_with("Faulted at 0x"),
                "an unresolved-import call must not degrade to a bare address: {detail}"
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
            let exit_nid = xps5x_firmware::dynlib::nid::nid_of("exit");
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
                "xps5x-gui-launcher-argc-test-{}",
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

            assert_eq!(launcher.session_state(&handle), SessionState::Running);
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
        /// Derived from `xps5x-runtime/tests/execute.rs`'s
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
                let name = format!("{}#A#A", xps5x_firmware::dynlib::nid::encode_nid(*nid));
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
            let malloc_nid = xps5x_firmware::dynlib::nid::nid_of("malloc");
            let memset_nid = xps5x_firmware::dynlib::nid::nid_of("memset");
            let map_nid = xps5x_firmware::dynlib::nid::nid_of("sceKernelMapFlexibleMemory");
            let exit_nid = xps5x_firmware::dynlib::nid::nid_of("exit");

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
                "xps5x-gui-realistic-homebrew-direct-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = write_realistic_homebrew_sprx(&tmp, "eboot.bin");

            let launcher = FirmwareLauncher::new();
            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(launcher.session_state(&handle), SessionState::Running);
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
        /// an `xps5x-title.toml` alongside it), discovered by the Shell's
        /// real `scan_dir`, then launched through `FirmwareLauncher` using
        /// the discovered `LaunchTarget` exactly as the Shell's Play button
        /// would — proving the entire path: discover on disk -> load -> run
        /// -> outcome.
        #[test]
        fn realistic_homebrew_discovered_by_scan_dir_then_launched_through_firmware_launcher() {
            let tmp = std::env::temp_dir().join(format!(
                "xps5x-gui-realistic-homebrew-scan-{}",
                std::process::id()
            ));
            let game_dir = tmp.join("Games").join("realistic-homebrew-demo");
            std::fs::create_dir_all(&game_dir).expect("create temp game dir");

            let bytes = build_realistic_homebrew_sprx();
            let eboot_path = game_dir.join("eboot.bin");
            std::fs::write(&eboot_path, &bytes)
                .expect("write synthetic realistic-homebrew .sprx to temp dir");
            std::fs::write(
                game_dir.join("xps5x-title.toml"),
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

            assert_eq!(launcher.session_state(&handle), SessionState::Running);
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
                    let nid = xps5x_firmware::dynlib::nid::nid_of(&cname);
                    let sce_name = format!("{}#A#A", xps5x_firmware::dynlib::nid::encode_nid(nid));
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
                "xps5x-gui-compiler-homebrew-{}",
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
                launcher.session_state(&handle),
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
            let console = launcher.kernel.console.contents();
            assert_eq!(
                console, "argc=1 tls=172\nbye\n",
                "unexpected guest console output; detail: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }
    }
}
