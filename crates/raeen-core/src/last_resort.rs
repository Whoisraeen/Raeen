//! Last-resort crash breadcrumbs: say what killed the process, even when the
//! normal log cannot.
//!
//! # Why this exists
//!
//! The emulator had a measured failure class where a guest run simply stopped
//! between 18 and 25 s with **no** terminal message: no `RESULT:` line, no
//! teardown, no session report, and a `logs/raeen.log` that truncates
//! mid-line. Nothing in the process said what happened, so every diagnosis was
//! inference.
//!
//! Three properties of the runner conspire to produce exactly that silence:
//!
//! 1. **The log is asynchronous.** `tracing_appender::non_blocking` hands
//!    events to a writer thread. Its `WorkerGuard` flushes on `Drop` — and
//!    `Drop` does not run when the process dies abnormally. Any event logged in
//!    the last instants before an abort is buffered and then lost, which is
//!    precisely why the file ends mid-line.
//! 2. **There are `abort()` calls on the guest hot path.** The HLE thunk
//!    gateway logs `tracing::error!` and then calls [`std::process::abort`]
//!    (see `raeen-runtime`'s `dispatch.rs`). By (1) that error line never
//!    reaches the file, so the loudest thing the runner can say about a
//!    panicking HLE handler is nothing at all.
//! 3. **Nothing catches a host SEH fault.** The runtime's vectored handler
//!    declines any exception on a thread with no guest context, and the
//!    minidump client is attached *only* when the Shell launched the runner and
//!    set `RAEEN_CRASH_SOCKET`. A `raeen --run-eboot` invocation — the way
//!    titles are actually measured — therefore has no unhandled-exception
//!    filter whatsoever. An access violation on the detached `raeen-gpu`
//!    worker, or in the Vulkan driver it calls, kills the process in total
//!    silence. Note that `catch_unwind` cannot help here: it catches Rust
//!    unwinds, never a hardware access violation.
//!
//! # What this module does
//!
//! [`install`] adds three breadcrumbs, all of which write **synchronously** to
//! `logs/last-resort.log`, bypassing the async writer entirely:
//!
//! - A **panic hook** — fires before the unwind, so a panic that is about to be
//!   swallowed by `catch_unwind` *or* followed by `abort()` still records its
//!   message, thread, source location, and backtrace. This alone converts
//!   route (2) from silent to fully diagnosed, and it also gives the GPU
//!   worker's existing "GPU submission panicked" warning the location it never
//!   had.
//! - A **process-wide unhandled-exception filter** — records the exception
//!   code, the faulting instruction and the module that owns it, the
//!   read/write/execute target of an access violation, the thread, and `rip`/
//!   `rsp`. Naming the owning module is the point: it distinguishes our own
//!   unsafe code from the graphics driver at a glance. Covers route (3).
//! - [`mark_clean_exit`] — an explicit "ended normally" line. Its **absence**
//!   is itself evidence: a breadcrumb file holding a panic record but no clean
//!   exit means the process aborted, and an empty one points at a route that
//!   bypasses both hooks (below).
//!
//! # Honest limits
//!
//! `SetUnhandledExceptionFilter` is not a universal net. Rust's `abort()`,
//! stack-overflow detection, and allocation failure all funnel into
//! `__fastfail`, which is *designed* to bypass top-level filters (and signal
//! handlers). Those cases are covered only indirectly — via the panic hook
//! when a panic precedes them, and otherwise by the *absence* of any record,
//! which narrows the cause rather than naming it. This module makes the silent
//! class visible; it does not claim to catch every possible death.
//!
//! The filter chains to whatever filter was installed before it and returns
//! `EXCEPTION_CONTINUE_SEARCH`, so it observes without suppressing: the
//! `minidumper` crash client and Windows Error Reporting both still run.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// File the breadcrumbs are appended to, inside the log directory.
pub const BREADCRUMB_FILE_NAME: &str = "last-resort.log";

/// Set to any non-empty value to skip installation entirely.
///
/// An escape hatch for the rare case where a host hook interferes with another
/// debugger; the diagnostic is otherwise unconditional, because a diagnostic
/// that must be enabled ahead of time is never on when the rare death happens.
pub const DISABLE_ENV: &str = "RAEEN_NO_LAST_RESORT";

/// Every line this module writes starts with this, so one `findstr`/`grep`
/// finds the whole story regardless of which breadcrumb fired.
pub const TAG: &str = "LAST RESORT";

/// Resolved breadcrumb path, set once by [`install`].
static BREADCRUMB_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Guards against recursion: a panic *inside* the panic hook (or a fault inside
/// the fault filter) must not spin forever.
static IN_HOOK: AtomicBool = AtomicBool::new(false);

/// How many panics get a full symbolized backtrace before the hook degrades to
/// message-and-location only.
///
/// `Backtrace::force_capture` resolves symbols, which costs milliseconds. The
/// GPU worker catches a panic per submission, so an unbounded hook would make a
/// title that panics every frame pay that cost every frame — a diagnostic that
/// changes the performance it is meant to measure. The first few backtraces are
/// all anyone reads.
const BACKTRACE_LIMIT: usize = 8;

/// Count of panics seen by the hook, for [`BACKTRACE_LIMIT`].
static PANICS_SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Stable crash code for a panic at `location` (`file:line:column`):
/// `RAEEN-PANIC-<8 hex>`, an FNV-1a hash of the location string.
///
/// The code names the *crash site*, not the run: the same panic location
/// produces the same code across runs and machines, so recurring crashes in
/// user reports and compat sweeps can be bucketed by a grep for one token
/// instead of by fuzzy-matching message text.
#[must_use]
pub fn panic_crash_code(location: &str) -> String {
    format!("RAEEN-PANIC-{:08X}", fnv1a64(location) as u32)
}

/// Stable crash code for an unhandled SEH exception: `RAEEN-SEH-<code hex>`,
/// e.g. `RAEEN-SEH-C0000094` for `STATUS_INTEGER_DIVIDE_BY_ZERO`.
#[must_use]
pub fn seh_crash_code(exception_code: u32) -> String {
    format!("RAEEN-SEH-{exception_code:08X}")
}

/// FNV-1a 64-bit: tiny, dependency-free, and stable across builds — all this
/// hash has to be. (Rust's `DefaultHasher` is explicitly unstable across
/// releases, which would silently re-key every crash bucket on a toolchain
/// bump.)
fn fnv1a64(s: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Install the panic hook and, on Windows, the unhandled-exception filter.
///
/// Idempotent — only the first call takes effect, so the Shell and the runner
/// can both call it unconditionally. `log_dir` is where
/// [`BREADCRUMB_FILE_NAME`] is written (the same directory as `raeen.log`).
///
/// Call this as early in `main` as possible: it protects everything that runs
/// after it, and nothing before it.
pub fn install(log_dir: &Path) {
    if std::env::var_os(DISABLE_ENV).is_some_and(|v| !v.is_empty()) {
        return;
    }
    if BREADCRUMB_PATH
        .set(log_dir.join(BREADCRUMB_FILE_NAME))
        .is_err()
    {
        // Already installed.
        return;
    }
    install_panic_hook();
    #[cfg(windows)]
    windows_filter::install();
}

/// Record that this process is ending on the normal path.
///
/// Pair with the records above when reading a breadcrumb file: a panic record
/// followed by this line means the panic was caught and the run continued to a
/// normal end; a panic record with no such line means the process died on it.
pub fn mark_clean_exit() {
    append(&format!("{TAG}: session ended normally\n"));
}

/// Append `record` to the breadcrumb file, synchronously, and mirror it to
/// stderr.
///
/// Deliberately does **not** go through `tracing`: the whole purpose is to
/// survive a death that the async log writer cannot, and taking a logging lock
/// from inside a crash handler risks deadlocking against the very thread that
/// crashed while holding it.
///
/// Best-effort by construction — a breadcrumb that cannot be written must
/// never become a second failure — so every I/O error here is discarded.
fn append(record: &str) {
    // stderr first: it needs no path, no directory, and no allocation beyond
    // what the caller already did, so it works even if the log dir is gone.
    let _ = std::io::stderr().write_all(record.as_bytes());

    let Some(path) = BREADCRUMB_PATH.get() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(record.as_bytes());
        // Flushing is the entire point: the default `File` write already
        // reaches the OS, but an explicit flush documents that nothing may sit
        // in a userspace buffer when the process is about to die.
        let _ = file.flush();
    }
}

/// Chain a recording hook in front of the existing panic hook.
///
/// The previous hook is still called, so Rust's default panic message on
/// stderr — and anything else already installed — keeps working exactly as
/// before. This observes; it does not replace.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // A panic raised while formatting a panic record would recurse; take
        // the flag and fall straight through to the previous hook if we are
        // already inside.
        if !IN_HOOK.swap(true, Ordering::SeqCst) {
            // Message and location are cheap and always recorded. The
            // BACKTRACE is not: `force_capture` resolves symbols, which costs
            // milliseconds. The GPU worker catches a panic per submission
            // (`agc_exec.rs`), so a title that panics every frame would pay
            // that cost every frame and lose measurable FPS to its own
            // diagnostic. Capture the first few in full — which is all anyone
            // reads — and degrade to message-only afterwards.
            let seen = PANICS_SEEN.fetch_add(1, Ordering::Relaxed);
            let backtrace = if seen < BACKTRACE_LIMIT {
                std::backtrace::Backtrace::force_capture().to_string()
            } else {
                format!("<omitted after {BACKTRACE_LIMIT} recorded backtraces>")
            };
            append(&format_panic_record(
                std::thread::current().name(),
                &payload_of(info),
                info.location().map(|l| (l.file(), l.line(), l.column())),
                &backtrace,
            ));
            IN_HOOK.store(false, Ordering::SeqCst);
        }
        previous(info);
    }));
}

/// The human-readable panic message, for the two payload types `panic!`
/// produces (`&str` for a literal, `String` for a format).
fn payload_of(info: &std::panic::PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

/// Render a panic breadcrumb.
///
/// Pure and separated from the hook so the exact wording is testable without
/// panicking a test process.
fn format_panic_record(
    thread: Option<&str>,
    message: &str,
    location: Option<(&str, u32, u32)>,
    backtrace: &str,
) -> String {
    let where_ = match location {
        Some((file, line, column)) => format!("{file}:{line}:{column}"),
        None => "<unknown location>".to_owned(),
    };
    format!(
        "{TAG}: panic on thread '{}' at {where_}\n\
         {TAG}:   message: {message}\n\
         {TAG}:   backtrace:\n{backtrace}\n",
        thread.unwrap_or("<unnamed>"),
    )
}

#[cfg(windows)]
mod windows_filter {
    //! The unhandled-exception filter half, which is necessarily Win32.

    use super::{IN_HOOK, TAG, append};
    use std::sync::atomic::Ordering;

    use windows_sys::Win32::System::Diagnostics::Debug::{
        EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS, SetUnhandledExceptionFilter,
    };
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;

    /// Filter installed before ours, called after we record, so this module
    /// never suppresses a minidump or Windows Error Reporting.
    ///
    /// Stored as a raw address because a Win32 callback pointer is not `Sync`;
    /// it is only ever set once during `install` and transmuted back on the
    /// crash path.
    static PREVIOUS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    type Filter = unsafe extern "system" fn(*const EXCEPTION_POINTERS) -> i32;

    pub(super) fn install() {
        // SAFETY: `filter` has exactly the `LPTOP_LEVEL_EXCEPTION_FILTER`
        // signature and is a `fn` item, valid for the whole process lifetime.
        let previous = unsafe { SetUnhandledExceptionFilter(Some(filter)) };
        PREVIOUS.store(
            previous.map_or(0, |f| f as usize),
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    /// Record the fault, then defer to whatever was installed before us.
    unsafe extern "system" fn filter(exception: *const EXCEPTION_POINTERS) -> i32 {
        if !IN_HOOK.swap(true, Ordering::SeqCst) && !exception.is_null() {
            // SAFETY: Windows guarantees `exception`, its `ExceptionRecord`,
            // and its `ContextRecord` are valid for the duration of this
            // callback. Both records are plain PODs read by value below.
            unsafe {
                let record = &*(*exception).ExceptionRecord;
                let context = (*exception).ContextRecord;
                let (rip, rsp) = if context.is_null() {
                    (0, 0)
                } else {
                    ((*context).Rip, (*context).Rsp)
                };
                let address = record.ExceptionAddress as usize;
                append(&format_exception_record(
                    record.ExceptionCode as u32,
                    address,
                    &module_of(address),
                    access_violation_detail(
                        record.ExceptionCode as u32,
                        record.NumberParameters as usize,
                        &record.ExceptionInformation,
                    ),
                    std::thread::current().name(),
                    GetCurrentThreadId(),
                    rip,
                    rsp,
                ));
            }
            IN_HOOK.store(false, Ordering::SeqCst);
        }

        let previous = PREVIOUS.load(std::sync::atomic::Ordering::SeqCst);
        if previous != 0 {
            // SAFETY: `previous` is either 0 (handled above) or the exact
            // pointer `SetUnhandledExceptionFilter` returned, which is a
            // valid filter with this signature.
            let previous: Filter = unsafe { std::mem::transmute::<usize, Filter>(previous) };
            // SAFETY: forwarding the same valid `exception` pointer Windows
            // gave us to a filter with the matching signature.
            return unsafe { previous(exception) };
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    /// `(module path, offset)` owning `address`, for naming *whose* code
    /// faulted — our binary, a system DLL, the graphics driver, or nothing
    /// mapped at all.
    fn module_of(address: usize) -> String {
        use windows_sys::Win32::Foundation::HMODULE;
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            GetModuleFileNameA, GetModuleHandleExA,
        };

        let mut module: HMODULE = std::ptr::null_mut();
        // SAFETY: `GetModuleHandleExA` with the FROM_ADDRESS flag treats
        // `address` as a lookup key and never dereferences it, so any value
        // (mapped or not) is sound; `module` is a valid out-parameter.
        let found = unsafe {
            GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                address as *const u8,
                &mut module,
            )
        };
        if found == 0 || module.is_null() {
            return "<no module mapped at that address>".to_owned();
        }

        let mut name = [0u8; 512];
        // SAFETY: `module` is a live handle (UNCHANGED_REFCOUNT means we must
        // not free it, and we do not); `name` is a valid buffer of the length
        // passed.
        let len = unsafe { GetModuleFileNameA(module, name.as_mut_ptr(), name.len() as u32) };
        let path = String::from_utf8_lossy(&name[..len as usize]).into_owned();
        let offset = address.wrapping_sub(module as usize);
        format!("{path}+{offset:#x}")
    }

    /// The read/write/execute detail Windows attaches to an access violation,
    /// or `None` for any other exception code.
    fn access_violation_detail(
        code: u32,
        number_parameters: usize,
        information: &[usize],
    ) -> Option<(&'static str, usize)> {
        const ACCESS_VIOLATION: u32 = 0xC000_0005;
        const IN_PAGE_ERROR: u32 = 0xC000_0006;
        if (code != ACCESS_VIOLATION && code != IN_PAGE_ERROR) || number_parameters < 2 {
            return None;
        }
        let kind = match information[0] {
            0 => "read",
            1 => "write",
            8 => "execute",
            _ => "access",
        };
        Some((kind, information[1]))
    }

    /// Render an unhandled-exception breadcrumb. Pure, so the wording is
    /// testable without faulting a test process.
    #[allow(clippy::too_many_arguments)] // one flat record; a struct here would only relocate the same fields
    fn format_exception_record(
        code: u32,
        address: usize,
        module: &str,
        access: Option<(&str, usize)>,
        thread: Option<&str>,
        thread_id: u32,
        rip: u64,
        rsp: u64,
    ) -> String {
        let mut out = format!(
            "{TAG}: unhandled exception {code:#010x} {} on thread '{}' (tid {thread_id})\n\
             {TAG}:   faulting instruction {address:#x} in {module}\n",
            exception_code_name(code),
            thread.unwrap_or("<unnamed>"),
        );
        if let Some((kind, target)) = access {
            out.push_str(&format!("{TAG}:   invalid {kind} of {target:#x}\n"));
        }
        out.push_str(&format!("{TAG}:   rip={rip:#x} rsp={rsp:#x}\n"));
        out.push_str(&format!(
            "{TAG}:   the async log may be truncated; this record is the authoritative cause\n"
        ));
        out
    }

    /// Name the exception codes that actually kill this process, so a reader
    /// does not have to look up a hex constant.
    fn exception_code_name(code: u32) -> &'static str {
        match code {
            0xC000_0005 => "ACCESS_VIOLATION",
            0xC000_0006 => "IN_PAGE_ERROR",
            0xC000_001D => "ILLEGAL_INSTRUCTION",
            0xC000_0025 => "NONCONTINUABLE_EXCEPTION",
            0xC000_0026 => "INVALID_DISPOSITION",
            0xC000_008C => "ARRAY_BOUNDS_EXCEEDED",
            0xC000_008D => "FLT_DENORMAL_OPERAND",
            0xC000_008E => "FLT_DIVIDE_BY_ZERO",
            0xC000_0090 => "FLT_INVALID_OPERATION",
            0xC000_0091 => "FLT_OVERFLOW",
            0xC000_0093 => "FLT_UNDERFLOW",
            0xC000_0094 => "INT_DIVIDE_BY_ZERO",
            0xC000_0095 => "INT_OVERFLOW",
            0xC000_0096 => "PRIVILEGED_INSTRUCTION",
            0xC000_00FD => "STACK_OVERFLOW",
            0xC000_0374 => "HEAP_CORRUPTION",
            0xC000_0409 => "STACK_BUFFER_OVERRUN",
            0x8000_0003 => "BREAKPOINT",
            0x4000_001F => "MS_VC_EXCEPTION (thread name)",
            0xE06D_7363 => "CXX_EXCEPTION",
            _ => "UNKNOWN",
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn names_the_access_violation_code() {
            assert_eq!(exception_code_name(0xC000_0005), "ACCESS_VIOLATION");
            assert_eq!(exception_code_name(0xC000_00FD), "STACK_OVERFLOW");
            assert_eq!(exception_code_name(0x1234), "UNKNOWN");
        }

        #[test]
        fn reports_the_read_target_of_an_access_violation() {
            let info = [0usize, 0xdead_beef, 0, 0];
            assert_eq!(
                access_violation_detail(0xC000_0005, 2, &info),
                Some(("read", 0xdead_beef))
            );
        }

        #[test]
        fn reports_the_write_target_of_an_access_violation() {
            let info = [1usize, 0x1557c00000, 0, 0];
            assert_eq!(
                access_violation_detail(0xC000_0005, 2, &info),
                Some(("write", 0x1557c00000))
            );
        }

        #[test]
        fn ignores_access_detail_for_other_codes_and_short_records() {
            let info = [0usize, 0x10, 0, 0];
            assert_eq!(access_violation_detail(0xC000_001D, 2, &info), None);
            // A record claiming fewer than two parameters must not be indexed.
            assert_eq!(access_violation_detail(0xC000_0005, 1, &info), None);
        }

        #[test]
        fn exception_record_names_code_module_and_access() {
            let record = format_exception_record(
                0xC000_0005,
                0x7ffb_1234_abcd,
                "C:\\Windows\\System32\\amdvlk64.dll+0x1234",
                Some(("read", 0x0)),
                Some("raeen-gpu"),
                4242,
                0x7ffb_1234_abcd,
                0x9_0000,
            );
            assert!(
                record.contains("unhandled exception 0xc0000005 ACCESS_VIOLATION on thread 'raeen-gpu' (tid 4242)"),
                "{record}"
            );
            assert!(record.contains("amdvlk64.dll+0x1234"), "{record}");
            assert!(record.contains("invalid read of 0x0"), "{record}");
            assert!(
                record.contains("rip=0x7ffb1234abcd rsp=0x90000"),
                "{record}"
            );
            // Every line must carry the tag so one grep finds the whole record.
            for line in record.lines() {
                assert!(line.starts_with(TAG), "untagged line: {line}");
            }
        }

        #[test]
        fn module_lookup_names_this_executable_for_a_local_code_address() {
            // A function pointer into this test binary must resolve to a real
            // module path, proving the lookup works against live mappings.
            let here =
                module_lookup_names_this_executable_for_a_local_code_address as *const () as usize;
            let module = module_of(here);
            assert!(
                module.contains(".exe") || module.contains(".dll"),
                "expected a module path, got {module}"
            );
        }

        #[test]
        fn module_lookup_reports_nothing_mapped_for_a_null_address() {
            assert_eq!(module_of(0), "<no module mapped at that address>");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_record_names_thread_message_and_location() {
        let record = format_panic_record(
            Some("raeen-gpu"),
            "index out of bounds: the len is 1080 but the index is 2160",
            Some(("crates/raeen-gpu/src/vulkan/offscreen.rs", 512, 9)),
            "   0: some_frame\n   1: another_frame",
        );
        assert!(
            record.contains(
                "panic on thread 'raeen-gpu' at crates/raeen-gpu/src/vulkan/offscreen.rs:512:9"
            ),
            "{record}"
        );
        assert!(
            record.contains("message: index out of bounds: the len is 1080 but the index is 2160"),
            "{record}"
        );
        assert!(record.contains("backtrace:"), "{record}");
        assert!(record.contains("some_frame"), "{record}");
    }

    #[test]
    fn panic_record_tolerates_a_missing_thread_name_and_location() {
        let record = format_panic_record(None, "boom", None, "");
        assert!(
            record.contains("panic on thread '<unnamed>' at <unknown location>"),
            "{record}"
        );
    }

    #[test]
    fn every_panic_record_line_is_tagged_for_grep() {
        let record = format_panic_record(Some("t"), "m", Some(("f.rs", 1, 2)), "");
        // The backtrace body is verbatim (indented frames), but the three
        // structural lines must each be findable by the tag.
        let tagged = record.lines().filter(|l| l.starts_with(TAG)).count();
        assert_eq!(tagged, 3, "{record}");
    }

    #[test]
    fn payload_of_reads_str_and_string_panics() {
        // `panic!("literal")` produces a `&str` payload; a formatted panic
        // produces a `String`. Both must render.
        let caught = std::panic::catch_unwind(|| panic!("literal payload"));
        let payload = caught.unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("literal payload")
        );

        let n = 7;
        let caught = std::panic::catch_unwind(move || panic!("formatted {n}"));
        let payload = caught.unwrap_err();
        assert_eq!(
            payload.downcast_ref::<String>().map(String::as_str),
            Some("formatted 7")
        );
    }
}
