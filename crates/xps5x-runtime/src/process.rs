//! The initial process stack (design doc §2, wall #1 W1a): lays out
//! `argc`/`argv`/`envp`/`auxv` top-down in the arena's stack region, in
//! exactly the layout a real ELF `_start` expects at `rsp` — see
//! [`build_process_stack`].

use xps5x_hle::GuestMemory;

use crate::RuntimeError;

/// `AT_PAGESZ`: the auxv entry type for the system page size.
const AT_PAGESZ: u64 = 6;
/// `AT_NULL`: terminates the auxv array.
const AT_NULL: u64 = 0;
/// The page-size value reported via `AT_PAGESZ` (design doc §2's minimal
/// auxv) — matches `arena.rs`'s `PAGE_SIZE`... except the design doc pins
/// this to `0x4000` specifically (not the literal 4 KiB host page size), so
/// it is duplicated here as its own named constant rather than importing
/// `arena`'s private one.
const AUXV_PAGESZ_VALUE: u64 = 0x4000;

/// Build the initial process stack a real ELF `_start` expects, top-down in
/// `[*, stack_top)`: `argv`'s and `envp`'s NUL-terminated strings first
/// (highest addresses), then — below them, 16-byte aligned — `argc`, the
/// `argv[]` pointer table, a `NULL`, the `envp[]` pointer table, another
/// `NULL`, and the auxv (`AT_PAGESZ`, `AT_NULL`). Returns the final `rsp`
/// (pointing at `argc`), which is always `% 16 == 0` (the `_start` ABI).
///
/// Every write goes through `mem` (bounds-checked, design doc §2): overflow
/// (too many/too-long `argv`/`envp` entries for the region, or arithmetic
/// overflow) and any out-of-region write both surface as
/// `Err(RuntimeError::MapFailed)` rather than a panic — `argv`/`envp` content
/// is caller-controlled (ultimately guest-adjacent), so no path here trusts
/// its length to fit without a checked computation first.
pub(crate) fn build_process_stack(
    stack_top: u64,
    argv: &[&str],
    envp: &[&str],
    mem: &dyn GuestMemory,
) -> Result<u64, RuntimeError> {
    // 1. Strings first, highest addresses, in argv-then-envp order — each
    // write moves `cursor` further down, recording where that string ended
    // up so the pointer table below can reference it.
    let mut cursor = stack_top;

    let mut argv_addrs = Vec::with_capacity(argv.len());
    for s in argv {
        cursor = write_cstr_below(mem, cursor, s.as_bytes())?;
        argv_addrs.push(cursor);
    }
    let mut envp_addrs = Vec::with_capacity(envp.len());
    for s in envp {
        cursor = write_cstr_below(mem, cursor, s.as_bytes())?;
        envp_addrs.push(cursor);
    }

    // 2. The pointer-table block: argc, argv[argc], NULL, envp[..], NULL,
    // then the auxv (two (type, value) pairs: AT_PAGESZ and AT_NULL). Every
    // field is one u64 slot.
    let n_argv = argv_addrs.len() as u64;
    let n_envp = envp_addrs.len() as u64;
    let n_slots = 1u64 // argc
        .checked_add(n_argv)
        .and_then(|v| v.checked_add(1)) // argv NULL terminator
        .and_then(|v| v.checked_add(n_envp))
        .and_then(|v| v.checked_add(1)) // envp NULL terminator
        .and_then(|v| v.checked_add(4)) // auxv: 2 (type, value) pairs
        .ok_or(RuntimeError::MapFailed)?;
    let block_bytes = n_slots.checked_mul(8).ok_or(RuntimeError::MapFailed)?;

    // Place the block so it ends at or below `cursor` (never overlapping the
    // strings just written above it), then align its start down to 16 bytes
    // — rounding down only ever moves the start *further* from `cursor`, so
    // the no-overlap property still holds after alignment.
    let block_end = cursor.checked_sub(block_bytes).ok_or(RuntimeError::MapFailed)?;
    let rsp = block_end & !0xF;

    let mut slot = rsp;
    write_u64(mem, &mut slot, n_argv)?;
    for addr in &argv_addrs {
        write_u64(mem, &mut slot, *addr)?;
    }
    write_u64(mem, &mut slot, 0)?; // argv NULL terminator
    for addr in &envp_addrs {
        write_u64(mem, &mut slot, *addr)?;
    }
    write_u64(mem, &mut slot, 0)?; // envp NULL terminator
    write_u64(mem, &mut slot, AT_PAGESZ)?;
    write_u64(mem, &mut slot, AUXV_PAGESZ_VALUE)?;
    write_u64(mem, &mut slot, AT_NULL)?;
    write_u64(mem, &mut slot, 0)?;

    Ok(rsp)
}

/// Write `bytes` plus a trailing NUL just below `cursor`, returning the
/// address the string now starts at (the new, lower `cursor`). `false` from
/// `mem.write` (out of the committed region) or a length computation
/// overflow both become `Err(MapFailed)`.
fn write_cstr_below(mem: &dyn GuestMemory, cursor: u64, bytes: &[u8]) -> Result<u64, RuntimeError> {
    let len = bytes.len().checked_add(1).ok_or(RuntimeError::MapFailed)?;
    let len_u64 = u64::try_from(len).map_err(|_| RuntimeError::MapFailed)?;
    let addr = cursor.checked_sub(len_u64).ok_or(RuntimeError::MapFailed)?;

    let mut buf = Vec::with_capacity(len);
    buf.extend_from_slice(bytes);
    buf.push(0);
    if !mem.write(addr, &buf) {
        return Err(RuntimeError::MapFailed);
    }
    Ok(addr)
}

/// Write `value` at `*slot` and advance `*slot` by 8 bytes. `false` from
/// `mem.write` becomes `Err(MapFailed)`.
fn write_u64(mem: &dyn GuestMemory, slot: &mut u64, value: u64) -> Result<(), RuntimeError> {
    if !mem.write(*slot, &value.to_le_bytes()) {
        return Err(RuntimeError::MapFailed);
    }
    *slot = slot.checked_add(8).ok_or(RuntimeError::MapFailed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A tiny `Vec<u8>`-backed [`GuestMemory`] for these unit tests — a
    /// zero-based address space (unlike the arena's identity mapping), which
    /// is all `build_process_stack` itself needs (it only ever writes
    /// through the `mem` trait object, never assuming identity mapping).
    struct VecMemory(RefCell<Vec<u8>>);

    impl VecMemory {
        fn new(size: usize) -> Self {
            Self(RefCell::new(vec![0u8; size]))
        }

        fn read_u64(&self, addr: u64) -> u64 {
            let mut buf = [0u8; 8];
            assert!(GuestMemory::read(self, addr, &mut buf), "read at {addr:#x} should succeed");
            u64::from_le_bytes(buf)
        }
    }

    impl GuestMemory for VecMemory {
        fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
            let Ok(addr) = usize::try_from(guest_addr) else { return false };
            let buf = self.0.borrow();
            let Some(end) = addr.checked_add(out.len()) else { return false };
            if end > buf.len() {
                return false;
            }
            out.copy_from_slice(&buf[addr..end]);
            true
        }

        fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
            let Ok(addr) = usize::try_from(guest_addr) else { return false };
            let mut buf = self.0.borrow_mut();
            let Some(end) = addr.checked_add(data.len()) else { return false };
            if end > buf.len() {
                return false;
            }
            buf[addr..end].copy_from_slice(data);
            true
        }
    }

    /// `rsp` is 16-aligned, `argc == 1`, `argv[0]`'s pointer resolves back to
    /// the exact string (NUL-terminated) — the W1a acceptance shape, at the
    /// `build_process_stack` layer (the `_start` asm-level acceptance lives
    /// in `tests/execute.rs`).
    #[test]
    fn layout_argc_argv0_pointer_and_string_round_trip() {
        let mem = VecMemory::new(0x1000);
        let rsp = build_process_stack(0x1000, &["/app/eboot.bin"], &[], &mem).expect("layout should succeed");

        assert_eq!(rsp % 16, 0, "rsp must be 16-byte aligned (the _start ABI)");

        let argc = mem.read_u64(rsp);
        assert_eq!(argc, 1, "argc must equal argv.len()");

        let argv0_ptr = mem.read_u64(rsp + 8);
        let mut string_bytes = [0u8; 15]; // "/app/eboot.bin\0"
        assert!(mem.read(argv0_ptr, &mut string_bytes));
        assert_eq!(&string_bytes, b"/app/eboot.bin\0");
    }

    /// Empty `argv`/`envp` still produce a well-formed layout: `argc == 0`,
    /// immediately followed by the argv NULL terminator, then the envp NULL
    /// terminator, then the auxv.
    #[test]
    fn empty_argv_and_envp_still_produce_null_terminators_and_auxv() {
        let mem = VecMemory::new(0x1000);
        let rsp = build_process_stack(0x1000, &[], &[], &mem).expect("layout should succeed");

        assert_eq!(rsp % 16, 0);
        assert_eq!(mem.read_u64(rsp), 0, "argc == 0");
        assert_eq!(mem.read_u64(rsp + 8), 0, "argv NULL terminator immediately follows argc");
        assert_eq!(mem.read_u64(rsp + 16), 0, "envp NULL terminator immediately follows argv's");
        assert_eq!(mem.read_u64(rsp + 24), AT_PAGESZ);
        assert_eq!(mem.read_u64(rsp + 32), AUXV_PAGESZ_VALUE);
        assert_eq!(mem.read_u64(rsp + 40), AT_NULL);
        assert_eq!(mem.read_u64(rsp + 48), 0);
    }

    /// A populated `argv` *and* `envp`: every slot in the pointer table lines
    /// up where the layout doc says it should, and every pointer resolves
    /// back to the right string.
    #[test]
    fn populated_argv_and_envp_layout_is_fully_correct() {
        let mem = VecMemory::new(0x1000);
        let rsp = build_process_stack(0x1000, &["a", "bb"], &["FOO=1"], &mem).expect("layout should succeed");

        assert_eq!(rsp % 16, 0);
        assert_eq!(mem.read_u64(rsp), 2, "argc == 2");

        let argv0 = mem.read_u64(rsp + 8);
        let argv1 = mem.read_u64(rsp + 16);
        assert_eq!(mem.read_u64(rsp + 24), 0, "argv NULL terminator after both entries");

        let envp0 = mem.read_u64(rsp + 32);
        assert_eq!(mem.read_u64(rsp + 40), 0, "envp NULL terminator after the one entry");

        assert_eq!(mem.read_u64(rsp + 48), AT_PAGESZ);
        assert_eq!(mem.read_u64(rsp + 56), AUXV_PAGESZ_VALUE);
        assert_eq!(mem.read_u64(rsp + 64), AT_NULL);
        assert_eq!(mem.read_u64(rsp + 72), 0);

        let mut buf = [0u8; 2];
        assert!(mem.read(argv0, &mut buf[..2]));
        assert_eq!(&buf, b"a\0");
        let mut buf3 = [0u8; 3];
        assert!(mem.read(argv1, &mut buf3));
        assert_eq!(&buf3, b"bb\0");
        let mut buf6 = [0u8; 6];
        assert!(mem.read(envp0, &mut buf6));
        assert_eq!(&buf6, b"FOO=1\0");
    }

    /// A string too large for the backing region fails the write and
    /// surfaces `Err(MapFailed)`, not a panic.
    #[test]
    fn write_failure_when_string_does_not_fit_returns_map_failed() {
        let mem = VecMemory::new(4);
        let err = build_process_stack(4, &["way too long for four bytes"], &[], &mem).unwrap_err();
        assert_eq!(err, RuntimeError::MapFailed);
    }
}
