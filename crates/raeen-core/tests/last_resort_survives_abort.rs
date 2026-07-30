//! The breadcrumb must survive the exact death that used to be silent.
//!
//! `raeen-runtime`'s HLE thunk gateway logs `tracing::error!` and then calls
//! [`std::process::abort`]. Because the file log is asynchronous, that error
//! line never reaches the disk: the process is gone before the writer thread
//! runs, and the `WorkerGuard` that would flush it only flushes on `Drop`,
//! which an abort skips. The result was a run that stopped dead with a log
//! truncated mid-line and no stated cause.
//!
//! Unit tests can prove the *wording* of a breadcrumb, but not that one
//! survives an abort — nothing inside a process outlives its own abort. So
//! this test re-executes the test binary as a child, has the child reproduce
//! the gateway's pattern (panic, catch it, abort), and then reads the file back
//! from the parent. That is the only way to demonstrate the property that
//! actually matters.

use std::path::Path;

/// Env var carrying the child's log directory; its presence selects the child
/// role, so one test binary plays both parts.
const CHILD_DIR_ENV: &str = "RAEEN_TEST_LAST_RESORT_CHILD_DIR";

/// What the child panics with — asserted verbatim in the parent, so the test
/// fails if the message is dropped or mangled rather than merely present.
const CHILD_PANIC_MESSAGE: &str = "synthetic gateway panic 0xC0FFEE";

/// The child half: install the breadcrumbs, then die the way the gateway dies.
///
/// Runs on a *named, detached* thread because that is the shape of the real
/// failure (the `raeen-gpu` worker and the guest thunk gateway are both off the
/// main thread), and because a thread name in the record is a large part of its
/// value.
fn run_as_child(dir: &Path) -> ! {
    raeen_core::last_resort::install(dir);

    let worker = std::thread::Builder::new()
        .name("synthetic-worker".to_owned())
        .spawn(|| {
            // Exactly `dispatch.rs`: the panic is *caught*, so no unwind
            // escapes and nothing else would ever report it...
            let caught = std::panic::catch_unwind(|| panic!("{CHILD_PANIC_MESSAGE}"));
            assert!(caught.is_err(), "the panic must have been caught");
            // ...and then the process aborts anyway. The panic hook already
            // fired (hooks run before the unwind), which is what makes the
            // record exist despite this.
            std::process::abort();
        })
        .expect("spawn the synthetic worker");

    let _ = worker.join();
    unreachable!("the worker aborts the process");
}

#[test]
fn a_panic_followed_by_abort_still_leaves_a_breadcrumb() {
    // Child role: do the dying, never return.
    if let Some(dir) = std::env::var_os(CHILD_DIR_ENV) {
        run_as_child(Path::new(&dir));
    }

    // Parent role: run ourselves as the child, in a scratch directory.
    let dir = std::env::temp_dir().join(format!(
        "raeen-last-resort-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("create the scratch log dir");

    let status = std::process::Command::new(
        std::env::current_exe().expect("locate this test binary for re-execution"),
    )
    .args([
        "--exact",
        "a_panic_followed_by_abort_still_leaves_a_breadcrumb",
        "--nocapture",
    ])
    .env(CHILD_DIR_ENV, &dir)
    // The child aborts by design; its stderr is noise for a passing run.
    .stderr(std::process::Stdio::null())
    .stdout(std::process::Stdio::null())
    .status()
    .expect("re-execute this test binary as the dying child");

    assert!(
        !status.success(),
        "the child was supposed to abort, but exited cleanly ({status:?})"
    );

    let breadcrumb = dir.join(raeen_core::last_resort::BREADCRUMB_FILE_NAME);
    let recorded = std::fs::read_to_string(&breadcrumb).unwrap_or_else(|e| {
        panic!(
            "no breadcrumb at {} after the child aborted ({e}) — the abort was silent, \
             which is the whole bug this file exists to prevent",
            breadcrumb.display()
        )
    });

    // The cause, named.
    assert!(
        recorded.contains(CHILD_PANIC_MESSAGE),
        "breadcrumb does not carry the panic message:\n{recorded}"
    );
    // The thread, so a reader knows which worker died.
    assert!(
        recorded.contains("panic on thread 'synthetic-worker'"),
        "breadcrumb does not name the panicking thread:\n{recorded}"
    );
    // The source location, so it is actionable without a debugger.
    assert!(
        recorded.contains("last_resort_survives_abort.rs:"),
        "breadcrumb does not carry a source location:\n{recorded}"
    );
    assert!(
        recorded.contains("backtrace:"),
        "breadcrumb does not carry a backtrace:\n{recorded}"
    );
    // An abort must NOT look like a clean shutdown: that distinction is how a
    // reader tells "caught and carried on" from "this killed the process".
    assert!(
        !recorded.contains("session ended normally"),
        "an aborting run must not claim a clean exit:\n{recorded}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Env var selecting the clean-exit child role (see below).
const CLEAN_DIR_ENV: &str = "RAEEN_TEST_LAST_RESORT_CLEAN_DIR";

/// The clean-exit marker is what makes the *absence* of a marker meaningful, so
/// prove a normally-ending process writes it — and writes no panic record.
///
/// Needs its own child because [`raeen_core::last_resort::install`] is
/// process-global and one-shot: the parent must stay uninstalled so the abort
/// test above is not perturbed.
#[test]
fn a_clean_exit_is_recorded_and_carries_no_panic() {
    if let Some(dir) = std::env::var_os(CLEAN_DIR_ENV) {
        raeen_core::last_resort::install(Path::new(&dir));
        raeen_core::last_resort::mark_clean_exit();
        return;
    }

    let dir = std::env::temp_dir().join(format!("raeen-last-resort-clean-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the scratch log dir");

    let status = std::process::Command::new(
        std::env::current_exe().expect("locate this test binary for re-execution"),
    )
    .args([
        "--exact",
        "a_clean_exit_is_recorded_and_carries_no_panic",
        "--nocapture",
    ])
    .env(CLEAN_DIR_ENV, &dir)
    .stderr(std::process::Stdio::null())
    .stdout(std::process::Stdio::null())
    .status()
    .expect("re-execute this test binary as the clean child");
    assert!(
        status.success(),
        "the clean child should exit 0 ({status:?})"
    );

    let breadcrumb = dir.join(raeen_core::last_resort::BREADCRUMB_FILE_NAME);
    let recorded = std::fs::read_to_string(&breadcrumb).expect("clean child wrote no breadcrumb");
    assert!(
        recorded.contains("session ended normally"),
        "clean exit not recorded:\n{recorded}"
    );
    assert!(
        !recorded.contains("panic on thread"),
        "a clean run must record no panic:\n{recorded}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
