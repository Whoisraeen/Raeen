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
            StubSession { started: Instant::now(), quit_requested: false },
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

/// Guest virtual address linked modules are laid out at. Matches the
/// convention `xps5x-firmware`'s own homebrew-pipeline test and the
/// `xps5x --load-sprx` diagnostic use; nothing claims this range as real
/// memory yet since this milestone links modules but does not run them.
const DEFAULT_LOAD_BASE: u64 = 0x8000_0000;

/// What came of trying to load+link one module for a launch.
#[derive(Debug, Clone)]
enum SessionOutcome {
    /// SELF decrypt -> `.sprx` parse -> dynlibdata decode -> NID link all
    /// succeeded. The module is *linked* into a flat image, not executed —
    /// there is no runtime yet (spec: SM3 links, a later milestone runs).
    Linked {
        resolved: usize,
        unresolved: usize,
        /// Reserved for a future runtime/diagnostics surface (e.g. Settings
        /// ▸ a "last launch" panel); not yet read anywhere itself, but kept
        /// alongside `resolved`/`unresolved` per the milestone's contract.
        #[allow(dead_code)]
        image_size: usize,
    },
    /// Anything that stopped short of a link: no module file at the target
    /// path, an encrypted module with no matching key, or a genuine
    /// parse/link error. Carries the message the overlay shows verbatim.
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
    hle: xps5x_hle::HleRegistry,
    registry: Mutex<xps5x_firmware::ModuleRegistry>,
    sessions: Mutex<HashMap<u64, FirmwareSession>>,
    next_id: Mutex<u64>,
}

impl FirmwareLauncher {
    pub fn new() -> Self {
        let hle = xps5x_hle::HleRegistry::new();
        let nid_db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle_names(hle.registered_names());
        Self {
            hle,
            registry: Mutex::new(xps5x_firmware::ModuleRegistry::new(nid_db)),
            sessions: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Read `path` and, if that succeeds, load+link it. Every failure mode
    /// — unreadable file, missing key, malformed module — becomes a
    /// [`SessionOutcome::Faulted`] with a message fit to show the user;
    /// this never panics.
    fn load(&self, path: &Path) -> SessionOutcome {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) => return SessionOutcome::Faulted(format!("No module file at {}: {err}", path.display())),
        };

        let mut registry = self.registry.lock().unwrap();
        match xps5x_firmware::load_module(&bytes, &xps5x_firmware::NoKeysProvider, &mut registry, &self.hle, DEFAULT_LOAD_BASE)
        {
            Ok(linked) => SessionOutcome::Linked {
                resolved: linked.hle_trampolines.len(),
                unresolved: linked.unresolved.len(),
                image_size: linked.image.len(),
            },
            Err(FirmwareError::MissingKey { .. }) => SessionOutcome::Faulted(
                "Encrypted module — no KeyProvider configured (Settings ▸ Key Provider)".to_string(),
            ),
            Err(err) => SessionOutcome::Faulted(err.to_string()),
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
            LaunchTarget::App { id } => SessionOutcome::Faulted(format!("'{id}' is not a loadable module")),
        };

        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;

        self.sessions.lock().unwrap().insert(id, FirmwareSession { outcome, quit_requested: false });

        Ok(SessionHandle(id))
    }

    fn session_state(&self, handle: &SessionHandle) -> SessionState {
        let sessions = self.sessions.lock().unwrap();
        match sessions.get(&handle.0) {
            None => SessionState::Faulted,
            Some(session) if session.quit_requested => SessionState::Exited,
            Some(session) => match &session.outcome {
                SessionOutcome::Linked { .. } => SessionState::Running,
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
        LaunchTarget::Game { path: PathBuf::from("Games/nova/eboot.bin") }
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
    // links cleanly with `resolved == 0` and `unresolved == 0`. Entirely
    // synthetic bytes; no real firmware anywhere.

    const EHDR_SIZE: usize = 64;
    const PHDR_SIZE: usize = 56;
    const EM_X86_64: u16 = 62;
    const ET_SCE_DYNAMIC: u16 = 0xFE18;
    const PT_LOAD: u32 = 1;

    const SELF_MAGIC: u32 = 0x4F15_D17E;
    const SELF_HEADER_SIZE: usize = 32;
    const SELF_ENTRY_SIZE: usize = 32;

    fn build_minimal_elf() -> Vec<u8> {
        let load_bytes = vec![0u8; 0x10];
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
        let target = LaunchTarget::Game { path: PathBuf::from("this/path/does/not/exist/eboot.bin") };

        let handle = launcher.launch(&target).expect("launch always returns a handle, even on fault");
        assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
        let detail = launcher.session_detail(&handle).expect("fault carries a message");
        assert!(detail.starts_with("No module file at"), "unexpected message: {detail}");
    }

    #[test]
    fn app_target_faults_cleanly() {
        let launcher = FirmwareLauncher::new();
        let target = LaunchTarget::App { id: "settings".to_string() };

        let handle = launcher.launch(&target).expect("launch always returns a handle, even on fault");
        assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
        let detail = launcher.session_detail(&handle).expect("fault carries a message");
        assert!(detail.contains("not a loadable module"), "unexpected message: {detail}");
    }

    #[test]
    fn valid_synthetic_module_links_and_exposes_resolved_counts() {
        let tmp = std::env::temp_dir().join(format!("xps5x-gui-launcher-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        let path = write_synthetic_sprx(&tmp, "eboot.bin");

        let launcher = FirmwareLauncher::new();
        let target = LaunchTarget::Game { path };
        let handle = launcher.launch(&target).expect("launch always returns a handle");

        assert_eq!(launcher.session_state(&handle), SessionState::Running);
        let detail = launcher.session_detail(&handle).expect("a linked session has detail text");
        assert!(detail.contains("0 imports resolved to HLE"), "unexpected message: {detail}");
        assert!(detail.contains("0 unresolved"), "unexpected message: {detail}");
        assert!(detail.contains("execution not yet implemented"), "unexpected message: {detail}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn quit_transitions_a_linked_session_to_exited() {
        let tmp = std::env::temp_dir().join(format!("xps5x-gui-launcher-quit-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        let path = write_synthetic_sprx(&tmp, "eboot.bin");

        let launcher = FirmwareLauncher::new();
        let handle = launcher.launch(&LaunchTarget::Game { path }).expect("launch should succeed");
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
}
