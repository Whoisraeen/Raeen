//! Shell ↔ engine seam (spec §5).
//!
//! The Shell contains **no emulation logic**. It talks to the engine only
//! through [`GameLauncher`], so the Shell can be built and tested against
//! [`StubLauncher`] long before the real engine can run anything. SM3 swaps
//! `StubLauncher` for the real engine implementation without touching Shell
//! navigation or rendering code.

use crate::library::LaunchTarget;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Opaque handle to a launched session, returned by [`GameLauncher::launch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionHandle(u64);

/// Lifecycle state of a launched session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
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
    /// Request a running session to quit (returns to Shell).
    fn quit(&self, handle: &SessionHandle) -> Result<(), LaunchError>;
}

struct StubSession {
    started: Instant,
    quit_requested: bool,
}

/// A launcher that simulates Loading→Running→Exited over time, so the whole
/// Shell is exercisable end-to-end before the real engine exists.
///
/// `loading_duration` is configurable so tests can set it to `Duration::ZERO`
/// and observe an immediate transition to `Running`.
pub struct StubLauncher {
    loading_duration: Duration,
    sessions: Mutex<HashMap<u64, StubSession>>,
    next_id: Mutex<u64>,
}

impl StubLauncher {
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
