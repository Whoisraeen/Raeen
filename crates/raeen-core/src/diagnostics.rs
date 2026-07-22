//! Deterministic, process-scoped emulator diagnostics.
//!
//! Records contain no host timestamps or host-thread identifiers. A single
//! monotonically increasing sequence number gives HLE, wait, event, task, and
//! GPU activity one order that can be compared without parsing interleaved log
//! lines from several host threads.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Categories kept deliberately independent of the concrete HLE/kernel types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticKind {
    HleEnter,
    HleExit,
    WaitBegin,
    WaitEnd,
    Wake,
    EventTransition,
    TaskOwned,
    TaskReleased,
    GpuSubmit,
    /// A module initializer (`module_start`/`DT_INIT`) the loader ran or, for
    /// the main executable under a crt0 entry, deliberately withheld. The
    /// `subject` is the module name, `object` the image offset, and `detail`
    /// carries the initializer's role and whether it ran or was deferred.
    ModuleInit,
}

/// One stable event in a guest process's diagnostic stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticEvent {
    pub sequence: u64,
    pub guest_thread: u64,
    pub kind: DiagnosticKind,
    pub subject: String,
    pub object: u64,
    pub detail: String,
}

/// Bounded recorder. Disabled recorders are effectively free (one relaxed
/// atomic load) and allocate nothing.
pub struct DiagnosticRecorder {
    enabled: bool,
    capacity: usize,
    next_sequence: AtomicU64,
    events: Mutex<VecDeque<DiagnosticEvent>>,
}

impl DiagnosticRecorder {
    pub const DEFAULT_CAPACITY: usize = 65_536;

    /// Build from `RAEEN_DETERMINISTIC_DIAGNOSTICS`. Any non-empty value other
    /// than `0` enables recording. `RAEEN_DIAGNOSTIC_CAPACITY` may bound the
    /// retained tail without changing sequence numbering.
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var_os("RAEEN_DETERMINISTIC_DIAGNOSTICS")
            .is_some_and(|value| !value.is_empty() && value != "0");
        let capacity = std::env::var("RAEEN_DIAGNOSTIC_CAPACITY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|capacity| *capacity > 0)
            .unwrap_or(Self::DEFAULT_CAPACITY);
        Self::new(enabled, capacity)
    }

    #[must_use]
    pub fn new(enabled: bool, capacity: usize) -> Self {
        Self {
            enabled,
            capacity: capacity.max(1),
            next_sequence: AtomicU64::new(1),
            events: Mutex::new(VecDeque::new()),
        }
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an event and return its stable sequence number.
    pub fn record(
        &self,
        guest_thread: u64,
        kind: DiagnosticKind,
        subject: impl Into<String>,
        object: u64,
        detail: impl Into<String>,
    ) -> Option<u64> {
        if !self.enabled {
            return None;
        }
        // Serialize sequence assignment with retention. Assigning the number
        // before taking this lock lets concurrent callers push 2 before 1,
        // making a snapshot disagree with the advertised stable order.
        let mut events = self
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let event = DiagnosticEvent {
            sequence,
            guest_thread,
            kind,
            subject: subject.into(),
            object,
            detail: detail.into(),
        };
        tracing::info!(
            target: "raeen::deterministic",
            sequence = event.sequence,
            guest_thread = event.guest_thread,
            kind = ?event.kind,
            subject = %event.subject,
            object = format_args!("{:#x}", event.object),
            detail = %event.detail,
            "guest event"
        );
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        Some(sequence)
    }

    /// Snapshot in sequence order. The bounded recorder may have discarded an
    /// older prefix, but retained sequence values are never renumbered.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DiagnosticEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}

impl Default for DiagnosticRecorder {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_is_stable_across_categories_and_capacity_keeps_tail() {
        let recorder = DiagnosticRecorder::new(true, 2);
        assert_eq!(
            recorder.record(1, DiagnosticKind::HleEnter, "libc::malloc", 0, ""),
            Some(1)
        );
        assert_eq!(
            recorder.record(2, DiagnosticKind::WaitBegin, "event-flag", 7, "bits=1"),
            Some(2)
        );
        assert_eq!(
            recorder.record(1, DiagnosticKind::Wake, "event-flag", 7, "set"),
            Some(3)
        );
        let events = recorder.snapshot();
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn disabled_recorder_does_not_advance_or_retain() {
        let recorder = DiagnosticRecorder::new(false, 4);
        assert_eq!(
            recorder.record(1, DiagnosticKind::HleEnter, "x", 0, ""),
            None
        );
        assert!(recorder.snapshot().is_empty());
    }

    #[test]
    fn concurrent_records_are_retained_in_sequence_order() {
        let recorder = std::sync::Arc::new(DiagnosticRecorder::new(true, 800));
        let mut workers = Vec::new();
        for guest_thread in 1..=8 {
            let recorder = std::sync::Arc::clone(&recorder);
            workers.push(std::thread::spawn(move || {
                for object in 0..100 {
                    recorder.record(
                        guest_thread,
                        DiagnosticKind::TaskOwned,
                        "concurrent",
                        object,
                        "",
                    );
                }
            }));
        }
        for worker in workers {
            worker.join().expect("diagnostic writer should not panic");
        }

        let events = recorder.snapshot();
        assert_eq!(events.len(), 800);
        assert!(
            events
                .iter()
                .enumerate()
                .all(|(index, event)| event.sequence == index as u64 + 1)
        );
    }
}
