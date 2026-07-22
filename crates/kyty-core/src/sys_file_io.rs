//! Port of Kyty's Windows `Sys::FileIO` layer
//! (`reference/kyty/source/include/Kyty/Sys/Windows/SysWindowsFileIO.h` +
//! `reference/kyty/source/lib/Sys/src/SysWindowsFileIO.cpp`), which
//! implements the cross-platform interface declared in
//! `reference/kyty/source/include/Kyty/Sys/SysFileIO.h`.
//!
//! This is Kyty's OS-abstraction layer for file I/O: a small C-style API of
//! free functions operating on an opaque `sys_file_t` handle that is either a
//! real Win32 file (`SYS_FILE_FILE`, backed by a `HANDLE` from `CreateFileW`)
//! or one of two in-memory buffers used to serialize/deserialize without disk
//! I/O: `SYS_FILE_MEMORY_STAT` (a fixed-capacity buffer that never grows,
//! originally an externally-owned `uint8_t*`/`size` pair) and
//! `SYS_FILE_MEMORY_DYN` (a buffer that reallocates/grows on write, starting
//! empty). This port keeps the free-function, `sys_file_*`-named API
//! (matching how other, not-yet-ported Kyty modules call it) rather than
//! turning it into an object-oriented `impl` surface, since that *is* Kyty's
//! public API shape here.
//!
//! Std/FFI mapping:
//! - `sys_file_t` (a `type` tag + `HANDLE`/`sys_file_mem_buf_t*` union) ->
//!   [`SysFile`], an enum-backed struct (`SysFileRepr`) so the "union" is a
//!   safe Rust enum instead of a C union with a manually-tracked type tag.
//!   The two in-memory buffer kinds share one representation
//!   ([`MemBuf`], wrapping an owned `Vec<u8>` + cursor) distinguished by a
//!   `growable` flag, since `Vec<u8>` already provides safe, correct
//!   growable-or-fixed byte storage — no raw pointers, no manual
//!   `mem_realloc`.
//! - Because the original externally-owned `uint8_t* buf` for
//!   `SYS_FILE_MEMORY_STAT` would require unsafe pointer aliasing to port
//!   1:1, [`sys_file_open_mem`] instead takes ownership of a `Vec<u8>` whose
//!   length is the fixed capacity — same bounded, non-growing read/write
//!   behavior, just safe ownership instead of a borrowed raw pointer (per
//!   this crate's "prefer zero unsafe" mandate for std-reimplementation
//!   types).
//! - C++'s two `sys_file_open`/`sys_file_create` *overloads* (by arg count)
//!   become distinctly-named functions, since Rust has no overloading:
//!   `sys_file_create(const String&)` -> [`sys_file_create_file`];
//!   `sys_file_create()` -> [`sys_file_create_mem`]; `sys_file_open(uint8_t*,
//!   uint32_t)` -> [`sys_file_open_mem`]. Likewise the two
//!   `sys_file_size`/`sys_file_get_last_access_and_write_time_utc` overloads
//!   (by parameter type) are disambiguated by name suffix (see each
//!   function's doc comment for its C++ counterpart).
//! - Out-parameters (`uint32_t* bytes_read`, `SysFileTimeStruct& a/w`) become
//!   return values (a plain `u32`, or a `(SysFileTimeStruct,
//!   SysFileTimeStruct)` tuple) — idiomatic Rust, same information.
//! - `SysFileTimeStruct` (`FILETIME time; bool is_invalid;`) -> here, `time`
//!   is the `FILETIME` packed into a single `u64` (`(dwHighDateTime << 32) |
//!   dwLowDateTime`, the same value a `FILETIME` represents) instead of the
//!   raw two-`u32` FFI struct, so the public type doesn't leak `windows-sys`
//!   and gets the usual derives (`Debug`/`Eq`/`Hash`/...) for free. The
//!   `SysTimeStruct`/`sys_get_system_time_utc`/`sys_system_to_file_time_utc`
//!   round trip the original uses for the memory-backed
//!   `sys_file_get_last_access_and_write_time_utc(sys_file_t&, ...)` overload
//!   is replaced by a single `GetSystemTimeAsFileTime` call (same resulting
//!   FILETIME value, no intermediate `SYSTEMTIME`); `Kyty::SysTimeStruct`
//!   itself is out of scope for this module (a `Sys::Timer` concept) and is
//!   not ported here.
//! - Win32 calls (`CreateFileW`, `ReadFile`, `WriteFile`, `FindFirstFileW`,
//!   ...) have no `std`-only equivalent that preserves this module's exact
//!   flag/attribute semantics (cache-hint flags, `WIN32_FIND_DATAW`
//!   timestamps, etc.), so they're called directly via `windows-sys` FFI,
//!   as expected for this Sys layer; every `unsafe` block has a `SAFETY`
//!   comment.
//! - `Kyty::String`/`Kyty::Vector<T>` are this crate's already-ported
//!   [`crate::string::String`] / [`crate::vector::Vector`] (aliased here as
//!   `String` — shadowing `std::string::String`, used internally as
//!   `StdString` where needed).
//!
//! Faithfully-preserved upstream quirks (documented at their call sites
//! below, not "fixed" by this port): `get_cache_access_type`'s
//! `SYS_FILE_CACHE_SEQUENTIAL_SCAN` branch returns the *enum discriminant*
//! instead of `FILE_FLAG_SEQUENTIAL_SCAN`, and `sys_file_get_dents` (unlike
//! `sys_file_find_files`) does not skip `.`/`..` entries.
//!
//! One deliberate, safety-driven deviation: growing a `SYS_FILE_MEMORY_DYN`
//! buffer after seeking past its current end zero-fills the gap (`Vec::resize`)
//! instead of leaving it uninitialized as the original's `mem_realloc` would
//! — reading uninitialized memory is undefined behavior we won't reproduce
//! under the "prefer zero unsafe" mandate.

#![cfg(windows)]

use std::ptr;

use windows_sys::Win32::Foundation::{
    CloseHandle, FILETIME, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{
    CREATE_ALWAYS, CopyFileW, CreateDirectoryW, CreateFileW, DeleteFileW, FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_READONLY, FILE_BEGIN, FILE_CURRENT,
    FILE_FLAG_RANDOM_ACCESS, FILE_SHARE_READ, FindClose, FindFirstFileW, FindNextFileW,
    FlushFileBuffers, GetFileAttributesExW, GetFileAttributesW, GetFileExInfoStandard,
    GetFileSizeEx, GetFileTime, INVALID_FILE_ATTRIBUTES, MoveFileW, OPEN_EXISTING, ReadFile,
    RemoveDirectoryW, SetEndOfFile, SetFileAttributesW, SetFilePointerEx, SetFileTime,
    WIN32_FILE_ATTRIBUTE_DATA, WIN32_FIND_DATAW, WriteFile,
};
use windows_sys::Win32::System::SystemInformation::GetSystemTimeAsFileTime;

use crate::string::{Case, String};
use crate::vector::Vector;

/// `sys_file_cache_type_t`: a hint for how the OS should cache reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SysFileCacheType {
    #[default]
    Auto,
    RandomAccess,
    SequentialScan,
}

/// `sys_file_type_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysFileType {
    Error,
    MemoryStat,
    File,
    MemoryDyn,
}

/// `SysFileTimeStruct` (`Windows/SysWindowsTimer.h`): see the module doc
/// comment for why `time` is a packed `u64` rather than a raw `FILETIME`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct SysFileTimeStruct {
    pub time: u64,
    pub is_invalid: bool,
}

fn filetime_to_u64(ft: FILETIME) -> u64 {
    (u64::from(ft.dwHighDateTime) << 32) | u64::from(ft.dwLowDateTime)
}

fn u64_to_filetime(v: u64) -> FILETIME {
    FILETIME {
        dwLowDateTime: v as u32,
        dwHighDateTime: (v >> 32) as u32,
    }
}

/// `sys_file_find_t`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SysFileFind {
    pub path_with_name: String,
    pub last_access_time: SysFileTimeStruct,
    pub last_write_time: SysFileTimeStruct,
    pub size: u64,
}

/// `sys_dir_entry_t`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SysDirEntry {
    pub name: String,
    pub is_file: bool,
}

/// The in-memory backing store shared by `SYS_FILE_MEMORY_STAT` (fixed
/// capacity, `growable == false`) and `SYS_FILE_MEMORY_DYN` (`growable ==
/// true`). See the module doc comment for the `Vec<u8>`-vs-raw-pointer
/// rationale.
#[derive(Debug, Default)]
struct MemBuf {
    data: Vec<u8>,
    pos: usize,
    growable: bool,
}

impl MemBuf {
    fn read(&mut self, out: &mut [u8]) -> u32 {
        let avail = self.data.len().saturating_sub(self.pos);
        let n = out.len().min(avail);
        out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        n as u32
    }

    fn write(&mut self, input: &[u8]) -> u32 {
        if self.growable {
            let end = self.pos + input.len();
            if end > self.data.len() {
                // Zero-fills any gap left by a prior seek-past-end; see the
                // module doc comment's "deliberate deviation" note.
                self.data.resize(end, 0);
            }
            self.data[self.pos..end].copy_from_slice(input);
            self.pos = end;
            input.len() as u32
        } else {
            let avail = self.data.len().saturating_sub(self.pos);
            let n = input.len().min(avail);
            self.data[self.pos..self.pos + n].copy_from_slice(&input[..n]);
            self.pos += n;
            n as u32
        }
    }
}

#[derive(Debug)]
enum SysFileRepr {
    Error,
    File(HANDLE),
    Mem(MemBuf),
}

/// `sys_file_t`: an opened file, either a real Win32 file or an in-memory
/// buffer. Created by [`sys_file_create_file`], [`sys_file_open_r`],
/// [`sys_file_open_w`], [`sys_file_open_rw`], [`sys_file_open_mem`] or
/// [`sys_file_create_mem`]; released by [`sys_file_close`] (or simply by
/// being dropped — [`Drop`] performs the same cleanup as a safety net, since
/// leaking a Win32 `HANDLE` on an unwind/forgotten-close would be a real OS
/// resource leak, unlike the original's fully-manual `delete`).
#[derive(Debug)]
pub struct SysFile {
    repr: SysFileRepr,
}

impl SysFile {
    /// `sys_file_t::type` read access (the original has no getter — the
    /// field was public — so this is the natural Rust equivalent).
    #[must_use]
    pub fn file_type(&self) -> SysFileType {
        match &self.repr {
            SysFileRepr::Error => SysFileType::Error,
            SysFileRepr::File(_) => SysFileType::File,
            SysFileRepr::Mem(buf) if buf.growable => SysFileType::MemoryDyn,
            SysFileRepr::Mem(_) => SysFileType::MemoryStat,
        }
    }
}

impl Drop for SysFile {
    fn drop(&mut self) {
        if let SysFileRepr::File(handle) = self.repr {
            // SAFETY: `handle` was returned by a `CreateFileW` call made by
            // one of this module's constructors and is owned exclusively by
            // this `SysFile` (never duplicated); closed at most once since
            // `Drop` runs exactly once. Matches the original's unconditional
            // `CloseHandle(f->handle)` (no invalid-handle check).
            unsafe {
                CloseHandle(handle);
            }
        }
        // `SysFileRepr::Mem`'s `Vec<u8>` frees itself; `Error` owns nothing.
    }
}

fn to_wide_z(s: &String) -> Vec<u16> {
    let mut units = s.utf16_str();
    units.push(0);
    units
}

fn wide_z_to_string(buf: &[u16]) -> String {
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf16(&buf[..len])
}

/// `get_cache_access_type` (internal helper). Faithfully preserves the
/// upstream bug where the `SequentialScan` branch returns the enum
/// discriminant `2` instead of `FILE_FLAG_SEQUENTIAL_SCAN` — see the module
/// doc comment.
fn get_cache_access_type(t: SysFileCacheType) -> u32 {
    match t {
        SysFileCacheType::RandomAccess => FILE_FLAG_RANDOM_ACCESS,
        SysFileCacheType::SequentialScan => 2,
        SysFileCacheType::Auto => 0,
    }
}

/// `sys_file_io_init()`.
#[must_use]
pub fn sys_file_io_init() -> bool {
    true
}

/// `sys_file_read(void*, uint32_t, sys_file_t&, uint32_t*)`: returns the
/// number of bytes actually read (the original's optional `bytes_read`
/// out-param), reading `data.len()` bytes (the original's separate `size`
/// parameter, made redundant by the slice's own length).
pub fn sys_file_read(data: &mut [u8], f: &mut SysFile) -> u32 {
    match &mut f.repr {
        SysFileRepr::File(handle) => {
            let mut read: u32 = 0;
            // SAFETY: `handle` is a live `HANDLE` from `CreateFileW`
            // exclusively borrowed for this call; `data` is a valid,
            // uniquely-owned buffer of `data.len()` bytes for `ReadFile` to
            // write into, and `read` is a valid `u32` out-param.
            unsafe {
                ReadFile(
                    *handle,
                    data.as_mut_ptr(),
                    data.len() as u32,
                    &mut read,
                    ptr::null_mut(),
                );
            }
            read
        }
        SysFileRepr::Mem(buf) => buf.read(data),
        SysFileRepr::Error => 0,
    }
}

/// `sys_file_write(const void*, uint32_t, sys_file_t&, uint32_t*)`: returns
/// bytes actually written.
pub fn sys_file_write(data: &[u8], f: &mut SysFile) -> u32 {
    match &mut f.repr {
        SysFileRepr::File(handle) => {
            let mut written: u32 = 0;
            // SAFETY: `handle` is a live `HANDLE`; `data` is a valid,
            // immutably-borrowed buffer `WriteFile` only reads from.
            unsafe {
                WriteFile(
                    *handle,
                    data.as_ptr(),
                    data.len() as u32,
                    &mut written,
                    ptr::null_mut(),
                );
            }
            written
        }
        SysFileRepr::Mem(buf) => buf.write(data),
        SysFileRepr::Error => 0,
    }
}

/// `sys_file_read_r(void*, uint32_t, sys_file_t&)`: reads normally, then
/// reverses the bytes in place — `[u8]::reverse` performs exactly the same
/// pairwise `data[i] <-> data[size-i-1]` swaps as the original's
/// hand-written loop.
pub fn sys_file_read_r(data: &mut [u8], f: &mut SysFile) {
    sys_file_read(data, f);
    data.reverse();
}

/// `sys_file_write_r(const void*, uint32_t, sys_file_t&)`: writes a
/// byte-reversed copy of `data`.
pub fn sys_file_write_r(data: &[u8], f: &mut SysFile) {
    let mut reversed = data.to_vec();
    reversed.reverse();
    sys_file_write(&reversed, f);
}

/// `sys_file_write(uint32_t, sys_file_t&)`: writes `n`'s raw (native-endian)
/// byte representation, matching the original's `memcpy(&n, 4)`.
pub fn sys_file_write_u32(n: u32, f: &mut SysFile) {
    sys_file_write(&n.to_ne_bytes(), f);
}

/// `sys_file_write_r(uint32_t, sys_file_t&)`.
pub fn sys_file_write_u32_r(n: u32, f: &mut SysFile) {
    sys_file_write_r(&n.to_ne_bytes(), f);
}

/// `sys_file_create(const String&)`: creates (or truncates) `file_name` for
/// read+write. Renamed from the overload set — see the module doc comment.
#[must_use]
pub fn sys_file_create_file(file_name: &String) -> SysFile {
    let wide = to_wide_z(file_name);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string alive for the
    // duration of this call; the remaining pointer arguments are
    // intentionally null (no security attributes / template handle), which
    // `CreateFileW` documents as valid.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            ptr::null(),
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            ptr::null_mut(),
        )
    };
    // No error check here, matching the original: `is_error` still detects
    // a failed create via `handle == INVALID_HANDLE_VALUE`.
    SysFile {
        repr: SysFileRepr::File(handle),
    }
}

/// `sys_file_open_r(const String&, sys_file_cache_type_t)`.
#[must_use]
pub fn sys_file_open_r(file_name: &String, cache_type: SysFileCacheType) -> SysFile {
    let wide = to_wide_z(file_name);
    // SAFETY: see `sys_file_create_file`.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_READ,
            ptr::null(),
            OPEN_EXISTING,
            get_cache_access_type(cache_type),
            ptr::null_mut(),
        )
    };
    let repr = if handle == INVALID_HANDLE_VALUE {
        SysFileRepr::Error
    } else {
        SysFileRepr::File(handle)
    };
    SysFile { repr }
}

/// `sys_file_open_w(const String&, sys_file_cache_type_t)`.
#[must_use]
pub fn sys_file_open_w(file_name: &String, cache_type: SysFileCacheType) -> SysFile {
    let wide = to_wide_z(file_name);
    // SAFETY: see `sys_file_create_file`.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_WRITE,
            0,
            ptr::null(),
            OPEN_EXISTING,
            get_cache_access_type(cache_type),
            ptr::null_mut(),
        )
    };
    let repr = if handle == INVALID_HANDLE_VALUE {
        SysFileRepr::Error
    } else {
        SysFileRepr::File(handle)
    };
    SysFile { repr }
}

/// `sys_file_open_rw(const String&, sys_file_cache_type_t)`.
#[must_use]
pub fn sys_file_open_rw(file_name: &String, cache_type: SysFileCacheType) -> SysFile {
    let wide = to_wide_z(file_name);
    // SAFETY: see `sys_file_create_file`.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            ptr::null(),
            OPEN_EXISTING,
            get_cache_access_type(cache_type),
            ptr::null_mut(),
        )
    };
    let repr = if handle == INVALID_HANDLE_VALUE {
        SysFileRepr::Error
    } else {
        SysFileRepr::File(handle)
    };
    SysFile { repr }
}

/// `sys_file_open(uint8_t*, uint32_t)`: opens a fixed-capacity in-memory
/// file (`SYS_FILE_MEMORY_STAT`) over `buf`; `buf.len()` is the capacity.
/// Reads/writes never grow it. Renamed — see the module doc comment.
#[must_use]
pub fn sys_file_open_mem(buf: Vec<u8>) -> SysFile {
    SysFile {
        repr: SysFileRepr::Mem(MemBuf {
            data: buf,
            pos: 0,
            growable: false,
        }),
    }
}

/// `sys_file_create()`: creates an empty, growable in-memory file
/// (`SYS_FILE_MEMORY_DYN`). Renamed — see the module doc comment.
#[must_use]
pub fn sys_file_create_mem() -> SysFile {
    SysFile {
        repr: SysFileRepr::Mem(MemBuf {
            data: Vec::new(),
            pos: 0,
            growable: true,
        }),
    }
}

/// `sys_file_close(sys_file_t*)`: releases `f`. Equivalent to simply
/// dropping it (see [`SysFile`]'s `Drop` impl) — kept as a named function to
/// match the original API.
pub fn sys_file_close(f: SysFile) {
    drop(f);
}

/// `sys_file_size(sys_file_t&)`.
#[must_use]
pub fn sys_file_size(f: &SysFile) -> u64 {
    match &f.repr {
        SysFileRepr::File(handle) => {
            let mut size: i64 = 0;
            // SAFETY: `handle` is a live `HANDLE`; `size` is a valid i64 out-param.
            unsafe {
                GetFileSizeEx(*handle, &mut size);
            }
            size as u64
        }
        SysFileRepr::Mem(buf) => buf.data.len() as u64,
        SysFileRepr::Error => 0,
    }
}

/// `sys_file_size(const String&)`: the size of a file given its path,
/// without opening it. Renamed with a `_of_path` suffix — Rust has no
/// overloading.
#[must_use]
pub fn sys_file_size_of_path(file_name: &String) -> u64 {
    let wide = to_wide_z(file_name);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string; `data` is a
    // valid, zero-initialized (all-zero is a valid bit pattern for this
    // repr(C) struct of integers/FILETIMEs) out-param for `GetFileAttributesExW`.
    let (ok, data): (i32, WIN32_FILE_ATTRIBUTE_DATA) = unsafe {
        let mut data: WIN32_FILE_ATTRIBUTE_DATA = std::mem::zeroed();
        let ok = GetFileAttributesExW(
            wide.as_ptr(),
            GetFileExInfoStandard,
            (&mut data as *mut WIN32_FILE_ATTRIBUTE_DATA).cast(),
        );
        (ok, data)
    };
    if ok == 0 {
        return 0;
    }
    (u64::from(data.nFileSizeHigh) << 32) + u64::from(data.nFileSizeLow)
}

/// `sys_file_truncate(sys_file_t&, uint64_t)`: only meaningful for real
/// files, matching the original (memory-backed files are a no-op returning
/// `false`).
pub fn sys_file_truncate(f: &mut SysFile, size: u64) -> bool {
    let SysFileRepr::File(handle) = f.repr else {
        return false;
    };
    let mut current: i64 = 0;
    // SAFETY: `handle` is a live `HANDLE`; `current`/(new-pointer args) are
    // valid out-params or null where the original passes null.
    unsafe {
        SetFilePointerEx(handle, 0, &mut current, FILE_CURRENT);
    }
    let moved = unsafe { SetFilePointerEx(handle, size as i64, ptr::null_mut(), FILE_BEGIN) };
    let ok = moved != 0 && unsafe { SetEndOfFile(handle) } != 0;
    unsafe {
        SetFilePointerEx(handle, current, ptr::null_mut(), FILE_BEGIN);
    }
    ok
}

/// `sys_file_seek(sys_file_t&, uint64_t)`.
pub fn sys_file_seek(f: &mut SysFile, offset: u64) -> bool {
    match &mut f.repr {
        SysFileRepr::File(handle) => {
            // SAFETY: `handle` is a live `HANDLE`.
            unsafe { SetFilePointerEx(*handle, offset as i64, ptr::null_mut(), FILE_BEGIN) != 0 }
        }
        SysFileRepr::Mem(buf) => {
            buf.pos = offset as usize;
            true
        }
        SysFileRepr::Error => true,
    }
}

/// `sys_file_tell(sys_file_t&)`.
#[must_use]
pub fn sys_file_tell(f: &SysFile) -> u64 {
    match &f.repr {
        SysFileRepr::File(handle) => {
            let mut current: i64 = 0;
            // SAFETY: `handle` is a live `HANDLE`; `current` is a valid out-param.
            unsafe {
                SetFilePointerEx(*handle, 0, &mut current, FILE_CURRENT);
            }
            current as u64
        }
        SysFileRepr::Mem(buf) => buf.pos as u64,
        SysFileRepr::Error => 0,
    }
}

/// `sys_file_is_error(sys_file_t&)`.
#[must_use]
pub fn sys_file_is_error(f: &SysFile) -> bool {
    match f.repr {
        SysFileRepr::Error => true,
        SysFileRepr::File(handle) => handle == INVALID_HANDLE_VALUE,
        SysFileRepr::Mem(_) => false,
    }
}

/// `sys_file_is_directory_existing(const String&)`.
#[must_use]
pub fn sys_file_is_directory_existing(path: &String) -> bool {
    let wide = to_wide_z(path);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string.
    let attrs = unsafe { GetFileAttributesW(wide.as_ptr()) };
    attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0
}

/// `sys_file_is_file_existing(const String&)`.
#[must_use]
pub fn sys_file_is_file_existing(name: &String) -> bool {
    let wide = to_wide_z(name);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string.
    let attrs = unsafe { GetFileAttributesW(wide.as_ptr()) };
    attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0
}

/// `sys_file_create_directory(const String&)`.
pub fn sys_file_create_directory(path: &String) -> bool {
    let wide = to_wide_z(path);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string; a null
    // security-attributes pointer is valid (default security descriptor).
    unsafe { CreateDirectoryW(wide.as_ptr(), ptr::null()) != 0 }
}

/// `sys_file_delete_directory(const String&)`.
pub fn sys_file_delete_directory(path: &String) -> bool {
    let wide = to_wide_z(path);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string.
    unsafe { RemoveDirectoryW(wide.as_ptr()) != 0 }
}

/// `sys_file_delete_file(const String&)`.
pub fn sys_file_delete_file(name: &String) -> bool {
    let wide = to_wide_z(name);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string.
    unsafe { DeleteFileW(wide.as_ptr()) != 0 }
}

/// `sys_file_flush(sys_file_t&)`.
pub fn sys_file_flush(f: &mut SysFile) -> bool {
    if let SysFileRepr::File(handle) = f.repr {
        if handle != INVALID_HANDLE_VALUE {
            // SAFETY: `handle` is a live, non-invalid `HANDLE`.
            return unsafe { FlushFileBuffers(handle) != 0 };
        }
    }
    false
}

/// `sys_file_get_last_access_time_utc(const String&)`.
#[must_use]
pub fn sys_file_get_last_access_time_utc(name: &String) -> SysFileTimeStruct {
    let f = sys_file_open_r(name, SysFileCacheType::Auto);
    let mut result = SysFileTimeStruct {
        time: 0,
        is_invalid: true,
    };
    if let SysFileRepr::File(handle) = f.repr {
        let mut ft = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        // SAFETY: `handle` is a live `HANDLE` (the `Error` variant is never
        // constructed with an invalid one — see `sys_file_open_r`); `ft` is
        // a valid out-param, the other two time slots are intentionally null.
        let ok = unsafe { GetFileTime(handle, ptr::null_mut(), &mut ft, ptr::null_mut()) != 0 };
        result = SysFileTimeStruct {
            time: filetime_to_u64(ft),
            is_invalid: !ok,
        };
    }
    sys_file_close(f);
    result
}

/// `sys_file_get_last_write_time_utc(const String&)`.
#[must_use]
pub fn sys_file_get_last_write_time_utc(name: &String) -> SysFileTimeStruct {
    let f = sys_file_open_r(name, SysFileCacheType::Auto);
    let mut result = SysFileTimeStruct {
        time: 0,
        is_invalid: true,
    };
    if let SysFileRepr::File(handle) = f.repr {
        let mut ft = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        // SAFETY: see `sys_file_get_last_access_time_utc`.
        let ok = unsafe { GetFileTime(handle, ptr::null_mut(), ptr::null_mut(), &mut ft) != 0 };
        result = SysFileTimeStruct {
            time: filetime_to_u64(ft),
            is_invalid: !ok,
        };
    }
    sys_file_close(f);
    result
}

/// `sys_file_get_last_access_and_write_time_utc(const String&,
/// SysFileTimeStruct&, SysFileTimeStruct&)`: the path-based overload,
/// returning `(access, write)` instead of writing through two out-params.
#[must_use]
pub fn sys_file_get_last_access_and_write_time_utc(
    name: &String,
) -> (SysFileTimeStruct, SysFileTimeStruct) {
    let f = sys_file_open_r(name, SysFileCacheType::Auto);
    let mut access = SysFileTimeStruct {
        time: 0,
        is_invalid: true,
    };
    let mut write = SysFileTimeStruct {
        time: 0,
        is_invalid: true,
    };
    if let SysFileRepr::File(handle) = f.repr {
        let mut a = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut w = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        // SAFETY: see `sys_file_get_last_access_time_utc`.
        let ok = unsafe { GetFileTime(handle, ptr::null_mut(), &mut a, &mut w) != 0 };
        access = SysFileTimeStruct {
            time: filetime_to_u64(a),
            is_invalid: !ok,
        };
        write = SysFileTimeStruct {
            time: filetime_to_u64(w),
            is_invalid: !ok,
        };
    }
    sys_file_close(f);
    (access, write)
}

/// `sys_file_get_last_access_and_write_time_utc(sys_file_t&,
/// SysFileTimeStruct&, SysFileTimeStruct&)`: the already-open-file overload
/// — renamed with a `_file` infix since Rust can't overload by parameter
/// type. For memory-backed files this returns the current UTC time (both
/// values), matching the original's `sys_get_system_time_utc` +
/// `sys_system_to_file_time_utc` round trip (collapsed here into one
/// `GetSystemTimeAsFileTime` call producing the same `FILETIME` value).
#[must_use]
pub fn sys_file_get_file_last_access_and_write_time_utc(
    f: &SysFile,
) -> (SysFileTimeStruct, SysFileTimeStruct) {
    match f.repr {
        SysFileRepr::File(handle) => {
            let mut a = FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            };
            let mut w = FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            };
            // SAFETY: `handle` is a live `HANDLE`.
            let ok = unsafe { GetFileTime(handle, ptr::null_mut(), &mut a, &mut w) != 0 };
            (
                SysFileTimeStruct {
                    time: filetime_to_u64(a),
                    is_invalid: !ok,
                },
                SysFileTimeStruct {
                    time: filetime_to_u64(w),
                    is_invalid: !ok,
                },
            )
        }
        SysFileRepr::Mem(_) => {
            let mut now = FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            };
            // SAFETY: `now` is a valid `FILETIME` out-param.
            unsafe {
                GetSystemTimeAsFileTime(&mut now);
            }
            let t = SysFileTimeStruct {
                time: filetime_to_u64(now),
                is_invalid: false,
            };
            (t, t)
        }
        SysFileRepr::Error => (
            SysFileTimeStruct {
                time: 0,
                is_invalid: true,
            },
            SysFileTimeStruct {
                time: 0,
                is_invalid: true,
            },
        ),
    }
}

/// `sys_file_set_last_access_time_utc(const String&, SysFileTimeStruct&)`.
pub fn sys_file_set_last_access_time_utc(name: &String, access: &SysFileTimeStruct) -> bool {
    if access.is_invalid {
        return false;
    }
    let f = sys_file_open_w(name, SysFileCacheType::Auto);
    let ok = match f.repr {
        SysFileRepr::File(handle) => {
            let ft = u64_to_filetime(access.time);
            // SAFETY: `handle` is a live `HANDLE` (opened for write); `ft`
            // is a valid `FILETIME` to set, the other two are null (unchanged).
            unsafe { SetFileTime(handle, ptr::null(), &ft, ptr::null()) != 0 }
        }
        _ => false,
    };
    sys_file_close(f);
    ok
}

/// `sys_file_set_last_write_time_utc(const String&, SysFileTimeStruct&)`.
pub fn sys_file_set_last_write_time_utc(name: &String, write: &SysFileTimeStruct) -> bool {
    if write.is_invalid {
        return false;
    }
    let f = sys_file_open_w(name, SysFileCacheType::Auto);
    let ok = match f.repr {
        SysFileRepr::File(handle) => {
            let ft = u64_to_filetime(write.time);
            // SAFETY: see `sys_file_set_last_access_time_utc`.
            unsafe { SetFileTime(handle, ptr::null(), ptr::null(), &ft) != 0 }
        }
        _ => false,
    };
    sys_file_close(f);
    ok
}

/// `sys_file_set_last_access_and_write_time_utc(const String&,
/// SysFileTimeStruct&, SysFileTimeStruct&)`.
pub fn sys_file_set_last_access_and_write_time_utc(
    name: &String,
    access: &SysFileTimeStruct,
    write: &SysFileTimeStruct,
) -> bool {
    if access.is_invalid || write.is_invalid {
        return false;
    }
    let f = sys_file_open_w(name, SysFileCacheType::Auto);
    let ok = match f.repr {
        SysFileRepr::File(handle) => {
            let a = u64_to_filetime(access.time);
            let w = u64_to_filetime(write.time);
            // SAFETY: see `sys_file_set_last_access_time_utc`.
            unsafe { SetFileTime(handle, ptr::null(), &a, &w) != 0 }
        }
        _ => false,
    };
    sys_file_close(f);
    ok
}

/// `sys_file_find_files(const String&, Vector<sys_file_find_t>&)`: recurses
/// into subdirectories, appending every *file* (not directory) found to
/// `out`. Skips `.`/`..` (unlike [`sys_file_get_dents`] — an intentional,
/// faithfully-preserved difference from the original).
pub fn sys_file_find_files(path: &String, out: &mut Vector<SysFileFind>) {
    let mut real_path = path.replace_char('\\', '/', Case::Sensitive);
    if !real_path.ends_with(&String::from("/"), Case::Sensitive) {
        real_path += "/";
    }

    let pattern = real_path.clone() + "*";
    let wide_pattern = to_wide_z(&pattern);

    // SAFETY: `wide_pattern` is a valid NUL-terminated UTF-16 string; `data`
    // is a valid out-param for `FindFirstFileW`/`FindNextFileW`.
    let mut data: WIN32_FIND_DATAW = unsafe { std::mem::zeroed() };
    let handle = unsafe { FindFirstFileW(wide_pattern.as_ptr(), &mut data) };
    if handle == INVALID_HANDLE_VALUE {
        return;
    }

    loop {
        let file_name = wide_z_to_string(&data.cFileName);

        if file_name != "." && file_name != ".." {
            if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 {
                sys_file_find_files(&(real_path.clone() + &file_name), out);
            } else {
                let size = (u64::from(data.nFileSizeHigh) << 32) + u64::from(data.nFileSizeLow);
                out.add(SysFileFind {
                    path_with_name: real_path.clone() + &file_name,
                    last_access_time: SysFileTimeStruct {
                        time: filetime_to_u64(data.ftLastAccessTime),
                        is_invalid: false,
                    },
                    last_write_time: SysFileTimeStruct {
                        time: filetime_to_u64(data.ftLastWriteTime),
                        is_invalid: false,
                    },
                    size,
                });
            }
        }

        // SAFETY: `handle` is the live find-handle from `FindFirstFileW`.
        let has_next = unsafe { FindNextFileW(handle, &mut data) != 0 };
        if !has_next {
            break;
        }
    }

    // SAFETY: `handle` is the live find-handle from `FindFirstFileW`, closed
    // exactly once here.
    unsafe {
        FindClose(handle);
    }
}

/// `sys_file_get_dents(const String&, Vector<sys_dir_entry_t>&)`: lists the
/// immediate contents of `path` (files and directories, non-recursive),
/// *including* `.`/`..` — matching the original exactly (unlike
/// [`sys_file_find_files`], which skips them).
pub fn sys_file_get_dents(path: &String, out: &mut Vector<SysDirEntry>) {
    let mut real_path = path.replace_char('\\', '/', Case::Sensitive);
    if !real_path.ends_with(&String::from("/"), Case::Sensitive) {
        real_path += "/";
    }

    let pattern = real_path + "*";
    let wide_pattern = to_wide_z(&pattern);

    // SAFETY: see `sys_file_find_files`.
    let mut data: WIN32_FIND_DATAW = unsafe { std::mem::zeroed() };
    let handle = unsafe { FindFirstFileW(wide_pattern.as_ptr(), &mut data) };
    if handle == INVALID_HANDLE_VALUE {
        return;
    }

    loop {
        let file_name = wide_z_to_string(&data.cFileName);
        let is_file = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
        out.add(SysDirEntry {
            name: file_name,
            is_file,
        });

        // SAFETY: see `sys_file_find_files`.
        let has_next = unsafe { FindNextFileW(handle, &mut data) != 0 };
        if !has_next {
            break;
        }
    }

    // SAFETY: see `sys_file_find_files`.
    unsafe {
        FindClose(handle);
    }
}

/// `sys_file_copy_file(const String&, const String&)`.
pub fn sys_file_copy_file(src: &String, dst: &String) -> bool {
    let s = to_wide_z(src);
    let d = to_wide_z(dst);
    // SAFETY: `s`/`d` are valid NUL-terminated UTF-16 strings.
    unsafe { CopyFileW(s.as_ptr(), d.as_ptr(), 0) != 0 }
}

/// `sys_file_move_file(const String&, const String&)`.
pub fn sys_file_move_file(src: &String, dst: &String) -> bool {
    let s = to_wide_z(src);
    let d = to_wide_z(dst);
    // SAFETY: `s`/`d` are valid NUL-terminated UTF-16 strings.
    unsafe { MoveFileW(s.as_ptr(), d.as_ptr()) != 0 }
}

/// `sys_file_remove_readonly(const String&)`.
pub fn sys_file_remove_readonly(name: &String) {
    let wide = to_wide_z(name);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string, read then
    // written back within this single call.
    unsafe {
        let attrs = GetFileAttributesW(wide.as_ptr());
        SetFileAttributesW(wide.as_ptr(), attrs & !FILE_ATTRIBUTE_READONLY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::string::String as StdString;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(tag: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("raeen_kyty_sysfileio_{tag}_{nanos}"));
        String::from(dir.to_str().unwrap())
    }

    /// Test-only helper: `Kyty::String` (UTF-32) -> `std::string::String`,
    /// for convenient assertions (mirrors what `String::utf8_str` does
    /// internally, without going through the separate `String8` type).
    fn to_std(s: &String) -> StdString {
        s.get_data_const().iter().collect()
    }

    // ---- in-memory files (SYS_FILE_MEMORY_STAT / SYS_FILE_MEMORY_DYN) ----

    #[test]
    fn mem_stat_file_reports_fixed_type_and_size() {
        let f = sys_file_open_mem(vec![0u8; 16]);
        assert_eq!(f.file_type(), SysFileType::MemoryStat);
        assert_eq!(sys_file_size(&f), 16);
        assert!(!sys_file_is_error(&f));
    }

    #[test]
    fn mem_stat_write_read_round_trip() {
        let mut f = sys_file_open_mem(vec![0u8; 8]);
        let written = sys_file_write(b"hello", &mut f);
        assert_eq!(written, 5);
        assert_eq!(sys_file_tell(&f), 5);

        assert!(sys_file_seek(&mut f, 0));
        let mut buf = [0u8; 5];
        let read = sys_file_read(&mut buf, &mut f);
        assert_eq!(read, 5);
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn mem_stat_write_clamps_at_capacity_never_grows() {
        let mut f = sys_file_open_mem(vec![0u8; 4]);
        let written = sys_file_write(b"abcdef", &mut f); // 6 bytes into a 4-byte buffer
        assert_eq!(written, 4, "STAT buffer must clamp, not grow");
        assert_eq!(sys_file_size(&f), 4);

        // Nothing more fits once the cursor reaches capacity.
        let more = sys_file_write(b"z", &mut f);
        assert_eq!(more, 0);
    }

    #[test]
    fn mem_dyn_file_starts_empty_and_grows_on_write() {
        let mut f = sys_file_create_mem();
        assert_eq!(f.file_type(), SysFileType::MemoryDyn);
        assert_eq!(sys_file_size(&f), 0);

        let written = sys_file_write(b"grow me", &mut f);
        assert_eq!(written, 7);
        assert_eq!(sys_file_size(&f), 7);

        assert!(sys_file_seek(&mut f, 0));
        let mut buf = [0u8; 7];
        assert_eq!(sys_file_read(&mut buf, &mut f), 7);
        assert_eq!(&buf, b"grow me");
    }

    #[test]
    fn mem_dyn_seek_past_end_then_write_zero_fills_gap() {
        let mut f = sys_file_create_mem();
        sys_file_write(b"AB", &mut f);
        assert!(sys_file_seek(&mut f, 5)); // past current end (len 2)
        sys_file_write(b"C", &mut f);
        assert_eq!(sys_file_size(&f), 6);

        assert!(sys_file_seek(&mut f, 0));
        let mut buf = [0u8; 6];
        sys_file_read(&mut buf, &mut f);
        assert_eq!(&buf, b"AB\0\0\0C");
    }

    #[test]
    fn read_past_end_returns_fewer_bytes_than_requested() {
        let mut f = sys_file_open_mem(vec![1, 2, 3]);
        let mut buf = [0u8; 10];
        let read = sys_file_read(&mut buf, &mut f);
        assert_eq!(read, 3);
        assert_eq!(&buf[..3], &[1, 2, 3]);
    }

    #[test]
    fn read_r_and_write_r_reverse_bytes() {
        let mut f = sys_file_create_mem();
        sys_file_write_r(b"ABCD", &mut f); // stored reversed: "DCBA"
        assert!(sys_file_seek(&mut f, 0));
        let mut plain = [0u8; 4];
        sys_file_read(&mut plain, &mut f);
        assert_eq!(&plain, b"DCBA");

        assert!(sys_file_seek(&mut f, 0));
        let mut round_tripped = [0u8; 4];
        sys_file_read_r(&mut round_tripped, &mut f); // reversed again -> back to "ABCD"
        assert_eq!(&round_tripped, b"ABCD");
    }

    #[test]
    fn write_u32_and_write_u32_r_use_native_and_reversed_byte_order() {
        let mut f = sys_file_create_mem();
        sys_file_write_u32(0x0102_0304, &mut f);
        sys_file_write_u32_r(0x0102_0304, &mut f);
        assert!(sys_file_seek(&mut f, 0));

        let mut native = [0u8; 4];
        sys_file_read(&mut native, &mut f);
        assert_eq!(native, 0x0102_0304u32.to_ne_bytes());

        let mut reversed = [0u8; 4];
        sys_file_read(&mut reversed, &mut f);
        let mut expected = 0x0102_0304u32.to_ne_bytes();
        expected.reverse();
        assert_eq!(reversed, expected);
    }

    #[test]
    fn seeking_on_error_file_is_a_harmless_no_op_returning_true() {
        // Opening a nonexistent file yields the `Error` repr.
        let mut f = sys_file_open_r(
            &String::from("R:/does/not/exist_raeen.bin"),
            SysFileCacheType::Auto,
        );
        assert!(sys_file_is_error(&f));
        assert!(sys_file_seek(&mut f, 123));
        assert_eq!(sys_file_tell(&f), 0);
        assert_eq!(sys_file_read(&mut [0u8; 4], &mut f), 0);
    }

    // ---- real filesystem ----

    #[test]
    fn create_write_close_reopen_and_read_back_a_real_file() {
        let dir = unique_temp_dir("basic");
        assert!(sys_file_create_directory(&dir));
        assert!(sys_file_is_directory_existing(&dir));

        let path = dir.clone() + "/thing.bin";
        {
            let mut f = sys_file_create_file(&path);
            assert!(!sys_file_is_error(&f));
            assert_eq!(sys_file_write(b"payload", &mut f), 7);
            sys_file_close(f);
        }

        assert!(sys_file_is_file_existing(&path));
        assert!(!sys_file_is_directory_existing(&path));
        assert_eq!(sys_file_size_of_path(&path), 7);

        {
            let mut f = sys_file_open_r(&path, SysFileCacheType::Auto);
            assert!(!sys_file_is_error(&f));
            assert_eq!(sys_file_size(&f), 7);
            let mut buf = [0u8; 7];
            assert_eq!(sys_file_read(&mut buf, &mut f), 7);
            assert_eq!(&buf, b"payload");
            sys_file_close(f);
        }

        sys_file_delete_file(&path);
        sys_file_delete_directory(&dir);
    }

    #[test]
    fn open_r_on_missing_file_is_an_error() {
        let dir = unique_temp_dir("missing");
        let path = dir + "/nope.bin";
        let f = sys_file_open_r(&path, SysFileCacheType::Auto);
        assert!(sys_file_is_error(&f));
        assert_eq!(f.file_type(), SysFileType::Error);
    }

    #[test]
    fn truncate_shrinks_a_real_file_and_preserves_position() {
        let dir = unique_temp_dir("truncate");
        assert!(sys_file_create_directory(&dir));
        let path = dir.clone() + "/t.bin";

        let mut f = sys_file_create_file(&path);
        sys_file_write(b"0123456789", &mut f);
        assert!(sys_file_seek(&mut f, 3));
        assert!(sys_file_truncate(&mut f, 5));
        assert_eq!(sys_file_size(&f), 5);
        assert_eq!(
            sys_file_tell(&f),
            3,
            "position must be restored after truncate"
        );
        sys_file_close(f);

        sys_file_delete_file(&path);
        sys_file_delete_directory(&dir);
    }

    #[test]
    fn copy_then_move_then_delete_a_real_file() {
        let dir = unique_temp_dir("copymove");
        assert!(sys_file_create_directory(&dir));

        let original = dir.clone() + "/orig.bin";
        let copy = dir.clone() + "/copy.bin";
        let moved = dir.clone() + "/moved.bin";

        let mut f = sys_file_create_file(&original);
        sys_file_write(b"data", &mut f);
        sys_file_close(f);

        assert!(sys_file_copy_file(&original, &copy));
        assert!(sys_file_is_file_existing(&copy));

        assert!(sys_file_move_file(&copy, &moved));
        assert!(!sys_file_is_file_existing(&copy));
        assert!(sys_file_is_file_existing(&moved));

        assert!(sys_file_delete_file(&original));
        assert!(sys_file_delete_file(&moved));
        assert!(sys_file_delete_directory(&dir));
    }

    #[test]
    fn set_and_get_last_write_time_round_trips() {
        let dir = unique_temp_dir("times");
        assert!(sys_file_create_directory(&dir));
        let path = dir.clone() + "/stamped.bin";

        let mut f = sys_file_create_file(&path);
        sys_file_write(b"x", &mut f);
        sys_file_close(f);

        // An arbitrary, clearly-non-invalid FILETIME value (some point in
        // 2000), used purely to check the round trip.
        let stamp = SysFileTimeStruct {
            time: 125_911_584_000_000_000,
            is_invalid: false,
        };
        assert!(sys_file_set_last_write_time_utc(&path, &stamp));

        let read_back = sys_file_get_last_write_time_utc(&path);
        assert!(!read_back.is_invalid);
        assert_eq!(read_back.time, stamp.time);

        sys_file_delete_file(&path);
        sys_file_delete_directory(&dir);
    }

    #[test]
    fn set_time_rejects_an_invalid_timestamp() {
        let dir = unique_temp_dir("invalid_time");
        assert!(sys_file_create_directory(&dir));
        let path = dir.clone() + "/f.bin";
        let mut f = sys_file_create_file(&path);
        sys_file_write(b"x", &mut f);
        sys_file_close(f);

        let invalid = SysFileTimeStruct {
            time: 0,
            is_invalid: true,
        };
        assert!(!sys_file_set_last_access_time_utc(&path, &invalid));
        assert!(!sys_file_set_last_write_time_utc(&path, &invalid));
        assert!(!sys_file_set_last_access_and_write_time_utc(
            &path, &invalid, &invalid
        ));

        sys_file_delete_file(&path);
        sys_file_delete_directory(&dir);
    }

    #[test]
    fn find_files_recurses_and_skips_dot_entries() {
        let dir = unique_temp_dir("find");
        let sub = dir.clone() + "/sub";
        assert!(sys_file_create_directory(&dir));
        assert!(sys_file_create_directory(&sub));

        let top_file = dir.clone() + "/top.bin";
        let nested_file = sub.clone() + "/nested.bin";
        for (p, contents) in [
            (&top_file, b"aa".as_slice()),
            (&nested_file, b"bbb".as_slice()),
        ] {
            let mut f = sys_file_create_file(p);
            sys_file_write(contents, &mut f);
            sys_file_close(f);
        }

        let mut found: Vector<SysFileFind> = Vector::new();
        sys_file_find_files(&dir, &mut found);

        assert_eq!(
            found.size(),
            2,
            "must recurse into `sub` and find both files, skipping . and .."
        );
        let names: Vec<StdString> = found.iter().map(|e| to_std(&e.path_with_name)).collect();
        assert!(names.iter().any(|n| n.ends_with("top.bin")));
        assert!(names.iter().any(|n| n.ends_with("nested.bin")));
        let nested_entry = found
            .iter()
            .find(|e| to_std(&e.path_with_name).ends_with("nested.bin"))
            .unwrap();
        assert_eq!(nested_entry.size, 3);

        sys_file_delete_file(&top_file);
        sys_file_delete_file(&nested_file);
        sys_file_delete_directory(&sub);
        sys_file_delete_directory(&dir);
    }

    #[test]
    fn get_dents_is_non_recursive_and_includes_dot_entries() {
        let dir = unique_temp_dir("dents");
        let sub = dir.clone() + "/sub";
        assert!(sys_file_create_directory(&dir));
        assert!(sys_file_create_directory(&sub));

        let top_file = dir.clone() + "/top.bin";
        let mut f = sys_file_create_file(&top_file);
        sys_file_write(b"x", &mut f);
        sys_file_close(f);

        let mut entries: Vector<SysDirEntry> = Vector::new();
        sys_file_get_dents(&dir, &mut entries);

        let names: Vec<StdString> = entries.iter().map(|e| to_std(&e.name)).collect();
        // Non-recursive: the file inside `sub` must NOT appear directly.
        assert!(names.contains(&".".to_string()));
        assert!(names.contains(&"..".to_string()));
        assert!(names.contains(&"top.bin".to_string()));
        assert!(names.contains(&"sub".to_string()));
        assert_eq!(entries.size(), 4);

        let sub_entry = entries.iter().find(|e| to_std(&e.name) == "sub").unwrap();
        assert!(!sub_entry.is_file);
        let file_entry = entries
            .iter()
            .find(|e| to_std(&e.name) == "top.bin")
            .unwrap();
        assert!(file_entry.is_file);

        sys_file_delete_file(&top_file);
        sys_file_delete_directory(&sub);
        sys_file_delete_directory(&dir);
    }

    #[test]
    fn remove_readonly_clears_the_attribute_so_delete_succeeds() {
        let dir = unique_temp_dir("readonly");
        assert!(sys_file_create_directory(&dir));
        let path = dir.clone() + "/ro.bin";

        let mut f = sys_file_create_file(&path);
        sys_file_write(b"x", &mut f);
        sys_file_close(f);

        let wide = to_wide_z(&path);
        // SAFETY: `wide` is a valid NUL-terminated UTF-16 string; this
        // directly sets the read-only attribute to set up the test
        // scenario `sys_file_remove_readonly` is meant to undo.
        unsafe {
            SetFileAttributesW(wide.as_ptr(), FILE_ATTRIBUTE_READONLY);
        }
        assert!(
            !sys_file_delete_file(&path),
            "a read-only file should fail to delete"
        );

        sys_file_remove_readonly(&path);
        assert!(
            sys_file_delete_file(&path),
            "delete must succeed once read-only is cleared"
        );

        sys_file_delete_directory(&dir);
    }

    #[test]
    fn io_init_returns_true() {
        assert!(sys_file_io_init());
    }
}
