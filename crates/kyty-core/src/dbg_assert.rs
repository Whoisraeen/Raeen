//! Port of Kyty's `Core/DbgAssert` (`reference/kyty/.../Core/DbgAssert.{h,cpp}`).
//!
//! Kyty's assertion layer is used pervasively across the whole emulator
//! (`EXIT_IF` in particular appears thousands of times). Its behavior: on a
//! failed check, print `expr`/file/line, then *halt* — break into an attached
//! debugger if present, otherwise terminate the process with status 321.
//!
//! The faithful Rust equivalent is `panic!`: it prints the same diagnostic
//! (message + `file!()`/`line!()`), and terminates the failing operation. We
//! use `panic!` rather than `std::process::exit(321)` because it is
//! catchable and testable (`#[should_panic]`), and because the Raeen runtime
//! already builds on `panic = unwind` for its fault machinery. The mapping
//! ported call sites use:
//!
//! | Kyty (C++)                | Rust (this module)          |
//! |---------------------------|-----------------------------|
//! | `ASSERT(x)`               | `assert_kyty!(x)`           |
//! | `EXIT_IF(x)`              | `exit_if!(x)`               |
//! | `EXIT("fmt", ..)`         | `exit!("fmt", ..)`          |
//! | `EXIT_NOT_IMPLEMENTED(x)` | `exit_not_implemented!(x)`  |
//! | `KYTY_NOT_IMPLEMENTED`    | `not_implemented!()`        |

/// Whether a debugger is attached. Kyty uses this to choose between breaking
/// (debugger) and exiting (no debugger) on `ASSERT_HALT`. In the Rust port the
/// halt is always a `panic!`, so this is informational only; it is wired to
/// the OS check on Windows and returns `false` elsewhere.
#[must_use]
pub fn dbg_is_debugger_present() -> bool {
    #[cfg(windows)]
    {
        // SAFETY: `IsDebuggerPresent` takes no arguments, reads only the
        // calling process's PEB `BeingDebugged` flag, and has no failure mode
        // — it always returns a `BOOL`. Declared inline to avoid pulling a
        // Win32 crate dependency into this leaf module.
        unsafe extern "system" {
            fn IsDebuggerPresent() -> i32;
        }
        unsafe { IsDebuggerPresent() != 0 }
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// `EXIT("fmt", ..)` — unconditional fatal error with a formatted message.
/// Faithful to Kyty's `EXIT`: reports the location and message, then halts.
#[macro_export]
macro_rules! exit {
    ($($arg:tt)*) => {
        ::std::panic!("KYTY EXIT: {}", ::std::format_args!($($arg)*))
    };
}

/// `EXIT_IF(x)` — fatal if the condition holds (the inverse of `assert`).
#[macro_export]
macro_rules! exit_if {
    ($cond:expr $(,)?) => {
        if $cond {
            ::std::panic!("KYTY EXIT_IF failed: {}", ::std::stringify!($cond));
        }
    };
}

/// `ASSERT(x)` — fatal unless the condition holds.
#[macro_export]
macro_rules! assert_kyty {
    ($cond:expr $(,)?) => {
        if !($cond) {
            ::std::panic!("KYTY ASSERT failed: {}", ::std::stringify!($cond));
        }
    };
}

/// `EXIT_NOT_IMPLEMENTED(x)` — fatal if the condition holds, flagging an
/// unimplemented path Kyty had not handled.
#[macro_export]
macro_rules! exit_not_implemented {
    ($cond:expr $(,)?) => {
        if $cond {
            ::std::panic!("KYTY NOT IMPLEMENTED: {}", ::std::stringify!($cond));
        }
    };
}

/// `KYTY_NOT_IMPLEMENTED` — unconditional unimplemented-path halt.
#[macro_export]
macro_rules! not_implemented {
    () => {
        ::std::panic!("KYTY NOT IMPLEMENTED")
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn dbg_is_debugger_present_is_callable() {
        // Just exercises the OS probe without asserting a value (depends on
        // whether the test harness runs under a debugger).
        let _ = super::dbg_is_debugger_present();
    }

    #[test]
    fn exit_if_false_does_not_panic() {
        exit_if!(1 + 1 == 3);
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn exit_if_true_panics() {
        exit_if!(1 + 1 == 2);
    }

    #[test]
    fn assert_kyty_true_does_not_panic() {
        assert_kyty!(1 + 1 == 2);
    }

    #[test]
    #[should_panic(expected = "KYTY ASSERT failed")]
    fn assert_kyty_false_panics() {
        assert_kyty!(1 + 1 == 3);
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT: bad value 7")]
    fn exit_formats_and_panics() {
        exit!("bad value {}", 7);
    }

    #[test]
    #[should_panic(expected = "KYTY NOT IMPLEMENTED")]
    fn not_implemented_panics() {
        not_implemented!();
    }
}
