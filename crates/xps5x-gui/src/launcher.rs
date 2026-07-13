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
    /// Linked *and* executed (Windows only, RT1b): `xps5x_runtime::execute_linked`
    /// ran the module's entry point to completion and returned. This does
    /// not mean the module "plays" anything — RT0/RT1 is an early native
    /// call-and-return runtime with no process environment yet (see
    /// `session_detail`'s honest wording).
    Ran {
        returned: u64,
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
            Err(err) => return SessionOutcome::Faulted(format!("No module file at {}: {err}", path.display())),
        };

        let linked = {
            let mut registry = self.registry.lock().unwrap();
            xps5x_firmware::load_module(&bytes, &xps5x_firmware::NoKeysProvider, &mut registry, &self.hle, DEFAULT_LOAD_BASE)
        };

        let linked = match linked {
            Ok(linked) => linked,
            Err(FirmwareError::MissingKey { .. }) => {
                return SessionOutcome::Faulted(
                    "Encrypted module — no KeyProvider configured (Settings ▸ Key Provider)".to_string(),
                );
            }
            Err(err) => return SessionOutcome::Faulted(err.to_string()),
        };

        let resolved = linked.hle_trampolines.len();
        let unresolved = linked.unresolved.len();

        #[cfg(target_os = "windows")]
        {
            match xps5x_runtime::execute_linked(&linked, &self.hle, linked.entry, &[]) {
                Ok(returned) => SessionOutcome::Ran { returned, resolved, unresolved },
                Err(xps5x_runtime::RuntimeError::Faulted { addr }) => {
                    SessionOutcome::Faulted(format!("Faulted at {addr:#x} during execution"))
                }
                Err(xps5x_runtime::RuntimeError::UnresolvedTrampoline(a)) => {
                    SessionOutcome::Faulted(format!("Called an unresolved import (trampoline {a:#x})"))
                }
                Err(e) => SessionOutcome::Faulted(format!("Runtime error: {e:?}")),
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            // `execute_linked` is Windows-only (RT0 design doc §7/§9); every
            // other target stops at "linked" rather than pretending to run.
            SessionOutcome::Linked { resolved, unresolved, image_size: linked.image.len() }
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
                SessionOutcome::Ran { .. } => SessionState::Running,
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
                "Executed — entry returned {returned:#x} · {resolved} HLE calls resolved, {unresolved} unresolved \
                 (early runtime — full game execution needs more HLE + a process environment)"
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
    // links cleanly with `resolved == 0` and `unresolved == 0`. On Windows,
    // RT1b's `FirmwareLauncher` now actually *executes* this too (`e_entry`
    // defaults to 0, and the segment's first byte is a bare `ret`, so the
    // entry returns immediately instead of running past whatever garbage
    // follows). Entirely synthetic bytes; no real firmware anywhere.

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
        let detail = launcher.session_detail(&handle).expect("a running session has detail text");

        // On Windows, RT1b actually runs this module's entry (a bare `ret`)
        // through the runtime, so the outcome is `Ran`, not `Linked`. Every
        // other target has no runtime backend, so `execute_linked` is never
        // reached and the pipeline stops at `Linked` (see `load`'s `#[cfg]`
        // gate).
        #[cfg(target_os = "windows")]
        {
            assert!(detail.starts_with("Executed — entry returned 0x"), "unexpected message: {detail}");
            assert!(detail.contains("0 HLE calls resolved"), "unexpected message: {detail}");
            assert!(detail.contains("0 unresolved"), "unexpected message: {detail}");
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert!(detail.contains("0 imports resolved to HLE"), "unexpected message: {detail}");
            assert!(detail.contains("0 unresolved"), "unexpected message: {detail}");
            assert!(detail.contains("execution not yet implemented"), "unexpected message: {detail}");
        }

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
                ph[40..48].copy_from_slice(&(spec.data.len() as u64).to_le_bytes()); // p_memsz
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
            let import_name = format!("{}#A#A", xps5x_firmware::dynlib::nid::encode_nid(import_nid));
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
                    PhdrSpec { p_type: PT_LOAD, p_flags: 5, p_vaddr: 0, data: load_bytes },
                    PhdrSpec { p_type: 0x6100_0000 /* PT_SCE_DYNLIBDATA */, p_flags: 4, p_vaddr: 0, data: dynlib_blob },
                    PhdrSpec { p_type: PT_DYNAMIC, p_flags: 6, p_vaddr: 0x2000, data: dynamic_bytes },
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

        fn sentinel(_args: &[u64]) -> u64 {
            0xC0DE
        }

        /// The genuine RT1b acceptance test: the shell's `FirmwareLauncher`
        /// loads a synthetic `.sprx` whose entry calls a real HLE-registered
        /// import and asserts that `session_detail` reports the sentinel
        /// value the HLE function returned — i.e. the module really ran.
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
            let import_nid = xps5x_firmware::dynlib::nid::nid_of("sceTestSentinel");

            let nid_db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle_names(hle.registered_names());
            let launcher = FirmwareLauncher {
                hle,
                registry: Mutex::new(xps5x_firmware::ModuleRegistry::new(nid_db)),
                sessions: Mutex::new(HashMap::new()),
                next_id: Mutex::new(0),
            };

            let tmp = std::env::temp_dir().join(format!("xps5x-gui-launcher-exec-test-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = write_executable_sprx(&tmp, "eboot.bin", import_nid);

            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(launcher.session_state(&handle), SessionState::Running);
            let detail = launcher.session_detail(&handle).expect("a ran session has detail text");
            assert!(detail.starts_with("Executed — entry returned 0xc0de"), "unexpected message: {detail}");
            assert!(detail.contains("1 HLE calls resolved"), "unexpected message: {detail}");
            assert!(detail.contains("0 unresolved"), "unexpected message: {detail}");

            let _ = std::fs::remove_dir_all(&tmp);
        }

        /// A guest `call` to an import nobody registered resolves (at link
        /// time) to `UNRESOLVED_STUB_ADDR` — a sentinel address outside RT0's
        /// trampoline guard region, so *calling* it at runtime is a genuine
        /// wild access violation there, not a recognized-trampoline dispatch.
        /// Either way it must surface as a clean `Faulted` detail — not a
        /// crash and not a silently-successful "Ran".
        #[test]
        fn play_faults_cleanly_when_the_module_calls_an_unresolved_import() {
            let hle = xps5x_hle::HleRegistry::new();
            let bogus_nid = xps5x_firmware::dynlib::nid::nid_of("totallyUnknownFunctionNobodyRegistered");

            let nid_db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle_names(hle.registered_names());
            let launcher = FirmwareLauncher {
                hle,
                registry: Mutex::new(xps5x_firmware::ModuleRegistry::new(nid_db)),
                sessions: Mutex::new(HashMap::new()),
                next_id: Mutex::new(0),
            };

            let tmp = std::env::temp_dir().join(format!("xps5x-gui-launcher-unresolved-test-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).expect("create temp dir");
            let path = write_executable_sprx(&tmp, "eboot.bin", bogus_nid);

            let handle = launcher
                .launch(&LaunchTarget::Game { path })
                .expect("launch always returns a handle");

            assert_eq!(launcher.session_state(&handle), SessionState::Faulted);
            let detail = launcher.session_detail(&handle).expect("fault carries a message");
            assert!(
                detail.starts_with("Faulted at 0x") && detail.contains("during execution"),
                "unexpected message: {detail}"
            );

            let _ = std::fs::remove_dir_all(&tmp);
        }
    }
}
