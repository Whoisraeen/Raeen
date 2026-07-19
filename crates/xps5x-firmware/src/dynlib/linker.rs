//! SCE relocation applier / linker — the final LM1 step: lays a module's
//! `PT_LOAD` segments into a flat image at `base`, then walks its SCE
//! relocations, resolving each symbol relocation's NID through the
//! [`ModuleRegistry`] (HLE trampoline / LLE export / unresolved stub) and
//! patching the relocation slot.
//!
//! # Link-time marker addresses
//!
//! [`HLE_TRAMPOLINE_BASE`] and [`UNRESOLVED_STUB_BASE`] are **not real code
//! addresses** — they are distinct, obviously-synthetic high sentinel
//! ranges a later runtime milestone will recognize and trap (dispatching to
//! the HLE registry, or diagnosing an unresolved import) rather than ever
//! executing as code. LM1's job is only to reach a *linked* state with
//! every relocation slot holding a defined value.
//!
//! # Image layout assumption
//!
//! `link_module` assumes the module's `PT_LOAD` segment `p_vaddr`s are
//! already image-relative, starting near 0 (as this crate's synthetic test
//! fixtures build them) — the flat `image` buffer is sized to
//! `max(seg.vaddr + seg.mem_size)` and each segment's bytes are copied to
//! `image[vaddr..]` directly. `base` is folded in only when *writing*
//! resolved addresses into relocation slots (`R_X86_64_RELATIVE`, and any
//! `Lle` export address, are added to `base`/looked up independent of the
//! image's own vaddr space). A real `.sprx`'s segments may need an explicit
//! rebase before this image-relative assumption holds; that's out of LM1
//! scope.

use std::collections::HashMap;

use iced_x86::{Decoder, DecoderOptions, Mnemonic};
use xps5x_core::error::FirmwareError;
use xps5x_hle::HleRegistry;

use crate::dynlib::DynlibData;
use crate::registry::{ModuleRegistry, Resolver};
use crate::sprx::SprxModule;

/// Deterministic base address for synthetic HLE-trampoline slots. See the
/// module docs: this is a link-time marker, not a real code address.
pub const HLE_TRAMPOLINE_BASE: u64 = 0x0000_4000_0000_0000;
/// Base of the per-NID unresolved-stub region. A relocation whose symbol
/// resolved to neither HLE nor a loaded LLE export gets
/// `UNRESOLVED_STUB_BASE + i * 8`, where `i` indexes
/// [`LinkedModule::unresolved_stubs`] — exactly the scheme
/// [`HLE_TRAMPOLINE_BASE`] uses.
///
/// # Why per-NID and not one sentinel
///
/// Every unresolved symbol used to share this single address. When the guest
/// called it the runtime could only report `Faulted { addr: 0x5000_0000_0000 }`
/// — the one thing it could not say was *which import the guest wanted*, which
/// is the only thing worth knowing. Giving each NID its own slot makes the
/// faulting address itself the identity of the missing function, so the fault
/// reports "guest called `<nid>` from `<library>`, unimplemented" and names the
/// next thing to implement.
///
/// The region is deduped by NID, so it is bounded by the module's distinct
/// import count (876 on the measured retail title ≈ 7 KB), not by its
/// relocation count (87k).
pub const UNRESOLVED_STUB_BASE: u64 = 0x0000_5000_0000_0000;

/// Private two-byte invalid instruction substituted for a decoded guest
/// `syscall`. Native execution must never let an Orbis syscall number reach
/// the Windows kernel; the runtime VEH recognizes this exact marker and
/// dispatches it through the Orbis kernel. `0F 04` is undefined
/// in x86-64 mode and, unlike the common `UD2` (`0F 0B`), is not emitted as a
/// normal compiler abort instruction.
pub const SYSCALL_TRAP_BYTES: [u8; 2] = [0x0F, 0x04];

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const R_X86_64_DTPMOD64: u32 = 16;
const R_X86_64_DTPOFF64: u32 = 17;
const R_X86_64_TPOFF64: u32 = 18;

/// The TLS module ID a module's `DTPMOD64` relocations resolve to when no
/// process-wide assignment says otherwise: the main executable's. A
/// multi-module process passes each module its real [`TlsAssignment`];
/// single-module linking (tests, fixtures, `--load-sprx`) keeps this default.
const MAIN_TLS_MODULE_ID: u64 = 1;

/// A module's slot in the process-wide static TLS layout, as the linker needs
/// it: which TLS module ID its `DTPMOD64`s name, and how far below the thread
/// pointer its block sits (the basis of its `TPOFF64` values).
///
/// `None` at the [`link_module_into`] boundary means "this module is the whole
/// TLS world" — id 1, block directly below the TCB — which is exactly the
/// pre-layout behavior and correct for every single-module caller. Passing
/// nothing in a *multi*-module process is how four modules ended up sharing
/// the eboot's TLS block; see [`crate::sprx::StaticTlsModule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsAssignment {
    /// Value for this module's `DTPMOD64` relocations.
    pub module_id: u64,
    /// Distance from the thread pointer down to this module's block; its
    /// `TPOFF64` values are `template_offset - tp_offset` (negative, wrapped).
    pub tp_offset: u64,
}

/// One HLE import resolved during linking: which `library::function` a
/// relocation's slot now points at, via its deterministic trampoline
/// address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HleTrampoline {
    pub library: String,
    pub function: String,
    pub addr: u64,
}

/// One relocation whose symbol resolved to neither HLE nor a loaded LLE
/// export. Recorded per **relocation**, not per symbol: a single NID
/// referenced by 48292 relocations appears 48292 times, so a raw count of
/// [`LinkedModule::unresolved`] measures patched slots, not missing functions.
///
/// `r_type` is what makes the count readable, because the two are wildly
/// different work items:
///
/// * `R_X86_64_JUMP_SLOT` — a PLT entry the guest will **call**. One per
///   distinct imported function. This is the real HLE gap.
/// * `R_X86_64_64` / `R_X86_64_GLOB_DAT` — a **data pointer** slot. On a big
///   C++ title these are dominated by RTTI: every polymorphic class's typeinfo
///   object points at a handful of `__cxxabiv1` vtables, so a few symbols
///   generate tens of thousands of relocations that are never called.
///
///
/// Measured on a retail title, 87414 import relocations break down as 86592
/// `R_X86_64_64`, 64 `GLOB_DAT`, and **758** `JUMP_SLOT`. The honest size of
/// the HLE gap is 758, not 87414.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnresolvedImport {
    pub nid: u64,
    /// The ELF relocation type (`R_X86_64_*`) of the slot left unresolved.
    pub r_type: u32,
}

/// One **distinct** unresolved NID and the stub address reserved for it.
/// Deduped: a NID referenced by many relocations owns exactly one slot, and
/// every one of those relocations is patched with that slot's address.
///
/// This is the reverse map the runtime needs: a fault at `addr` names `nid`,
/// and `library` says which library should have supplied it — turning an
/// opaque access violation into "implement this function next".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedStub {
    pub nid: u64,
    /// The importing library's name (from the module's import-library table),
    /// if the symbol's `#lib#` id maps to one. `None` when the module carries
    /// no library table, or the id is not in it.
    pub library: Option<String>,
    /// `UNRESOLVED_STUB_BASE + i * 8`.
    pub addr: u64,
}

/// Whether an initializer belongs to a file-backed dependency or to the main
/// executable itself.
///
/// The distinction decides *ownership of the call*, which is what makes a real
/// title boot: a retail crt0 `_start` walks the executable's own init array,
/// so a process loader that *also* calls the main initializer runs those
/// constructors twice. Measured on ASTRO.BOT, a list-building constructor then
/// formed a cyclic list its own later walk spun on forever (t1 frozen at a
/// `mov rdx,[rcx]; lea rcx,[rdx+0x10]; jnz` cycle at `module+0x7426c00`). A
/// `.prx` dependency has no crt0 that re-enters, so its initializer is the
/// loader's to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleInitRole {
    /// A file-backed `DT_NEEDED` dependency's `module_start`. The loader owns
    /// this call in every entry mode — nothing else runs it.
    Dependency,
    /// The main executable's own `DT_INIT`. Its crt0 `_start` re-runs it, so a
    /// process entry must withhold it (see [`ModuleInitRole`]); a direct,
    /// crt0-less entry runs it because nothing else will.
    Main,
}

impl std::fmt::Display for ModuleInitRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModuleInitRole::Dependency => "dependency",
            ModuleInitRole::Main => "main",
        })
    }
}

/// One module's `module_start`, to be called before the process entry.
///
/// The Orbis ABI (confirmed against Kyty's `RuntimeLinker::StartModule` ->
/// `run_ini_fini`, MIT © InoriRus) is
/// `int module_start(size_t args, const void *argp, module_func_t func)`,
/// SysV, called with `(0, NULL, NULL)` for a plain load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleInit {
    /// The `.prx` this belongs to, for diagnostics.
    pub name: String,
    /// Offset of `module_start` within the composed process image.
    pub image_offset: u64,
    /// Whether this is a dependency's initializer or the main executable's own
    /// (which its crt0 re-runs). Decides whether a process entry calls it; see
    /// [`ModuleInitRole`].
    pub role: ModuleInitRole,
}

/// One ELF's load range and exception tables within a composed process image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedUnwindModule {
    pub name: String,
    /// Placement of this ELF in [`LinkedModule::image`].
    pub image_offset: u64,
    pub unwind: crate::sprx::UnwindInfo,
    /// Module-relative LLE exports, retained for handle-scoped `dlsym`.
    pub exports: Vec<crate::dynlib::SymbolExport>,
    /// Module-relative `DT_INIT`, if present. Optional plugins run this when
    /// `sceKernelLoadStartModule` activates their preplaced image.
    pub init_vaddr: Option<u64>,
}

/// The result of [`link_module`]: a flat, relocated image plus a record of
/// what each symbol relocation resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedModule {
    pub image: Vec<u8>,
    pub base: u64,
    /// Relocations whose symbol resolved to [`Resolver::Unresolved`] — logged,
    /// non-fatal. **One entry per relocation**; see [`UnresolvedImport`].
    ///
    /// Every entry is a genuine import: [`link_module`] skips defined symbols
    /// before it ever consults the registry, so nothing here is an artifact of
    /// misclassifying the module's own symbols.
    pub unresolved: Vec<UnresolvedImport>,
    /// The distinct unresolved NIDs, in first-encountered order — index `i`
    /// owns stub address `UNRESOLVED_STUB_BASE + i * 8`. The runtime inverts
    /// this to name the import a faulting guest call wanted.
    pub unresolved_stubs: Vec<UnresolvedStub>,
    /// `module_start` entry points to run **before** the process entry, as
    /// offsets into [`Self::image`], in the order they must be called
    /// (dependencies first).
    ///
    /// A `.prx` runs its C++ global constructors from its `DT_INIT`, and a real
    /// loader calls each dependency's before entering the main module. Nothing
    /// did, so every dependency's globals stayed null — on the measured title
    /// the eboot's own constructors then called a virtual method through a null
    /// vtable. See `dynlib::DT_INIT`.
    ///
    /// Populated by `load_process` (which knows the dependency order), with
    /// dependency initializers first and the main executable's initializer
    /// last. A single-module `link_module` leaves it empty because it has no
    /// process-loader context to schedule calls.
    pub module_inits: Vec<ModuleInit>,
    /// Distinct HLE imports resolved, in first-encountered order, deduped
    /// by NID (a NID referenced by more than one relocation reuses the same
    /// trampoline address and appears here once).
    pub hle_trampolines: Vec<HleTrampoline>,
    /// [`SprxModule::entry`] (the ELF `e_entry`), carried through unchanged
    /// as an offset into `image` — a later milestone that gives modules a
    /// non-zero load bias will need to rebase this (see the module docs'
    /// "Image layout assumption").
    pub entry: u64,
    /// The module's `PT_TLS` static TLS template (M1-B, wall #2), carried
    /// through from [`SprxModule::tls`] so the runtime can materialize the
    /// per-thread TLS block whose layout this module's `TPOFF64`/`DTPOFF64`
    /// relocations were resolved against.
    pub tls: Option<crate::sprx::TlsTemplate>,
    /// The **process-wide** static TLS layout the runtime must materialize per
    /// thread: one entry per module with a `PT_TLS`, each at its assigned
    /// distance below the thread pointer. For a single-module link this is
    /// just the module's own template at `tp_offset = block_size()`;
    /// `load_process` replaces it with the layout spanning every loaded
    /// module. Empty when nothing in the process has TLS.
    pub tls_layout: Vec<crate::sprx::StaticTlsModule>,
    /// Image offset of the `PT_SCE_PROCPARAM` block (from
    /// [`SprxModule::procparam`]'s vaddr), if the module has one. The runtime
    /// exposes `base + procparam_offset` as the guest address
    /// `sceKernelGetProcParam` returns.
    pub procparam_offset: Option<u64>,
    /// Every executable/PRX in this process, used by the guest unwinder to
    /// resolve an instruction address to its `.eh_frame` tables.
    pub unwind_modules: Vec<LinkedUnwindModule>,
}

/// The marker-address tables shared by every module in one process.
///
/// # Why this must be shared
///
/// [`HLE_TRAMPOLINE_BASE`] and [`UNRESOLVED_STUB_BASE`] name **process-global**
/// address spaces: at runtime a fault at `BASE + i*8` is inverted through a
/// single table to recover the function it stands for. So `i` must be unique
/// across the whole process.
///
/// Linking each module with its own tables breaks that, silently. Every module
/// restarts at `i = 0`, so dependency A's import #3 and the main module's
/// import #3 claim the same address — and since the runtime only ever holds one
/// table, A's call to *its* #3 dispatches to the main module's #3 instead: the
/// wrong function, with the wrong signature, and no diagnostic. Threading one
/// `ProcessTables` through every [`link_module_into`] call in a process is what
/// makes the indices mean one thing.
#[derive(Debug, Default)]
pub struct ProcessTables {
    hle_trampolines: Vec<HleTrampoline>,
    hle_addrs: HashMap<(String, String), u64>,
    unresolved_stubs: Vec<UnresolvedStub>,
    stub_addrs: HashMap<(Option<String>, u64), u64>,
}

impl ProcessTables {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every HLE import resolved across the process, deduped by resolved
    /// library/function identity; index `i` owns `HLE_TRAMPOLINE_BASE + i*8`.
    pub fn hle_trampolines(&self) -> &[HleTrampoline] {
        &self.hle_trampolines
    }

    /// Every distinct unresolved provider/NID across the process; index `i`
    /// owns `UNRESOLVED_STUB_BASE + i*8`.
    pub fn unresolved_stubs(&self) -> &[UnresolvedStub] {
        &self.unresolved_stubs
    }

    /// The address for `nid`'s HLE trampoline, allocating one if this is the
    /// first module to import it.
    fn hle_addr(&mut self, library: String, function: String) -> u64 {
        let trampolines = &mut self.hle_trampolines;
        let key = (library.clone(), function.clone());
        *self.hle_addrs.entry(key).or_insert_with(|| {
            let addr = HLE_TRAMPOLINE_BASE + (trampolines.len() as u64 * 8);
            trampolines.push(HleTrampoline {
                library,
                function,
                addr,
            });
            addr
        })
    }

    /// The address for `nid`'s unresolved stub, allocating one if this is the
    /// first module to fail to resolve it.
    fn stub_addr(&mut self, nid: u64, library: Option<String>) -> u64 {
        let stubs = &mut self.unresolved_stubs;
        let key = (library.clone(), nid);
        *self.stub_addrs.entry(key).or_insert_with(|| {
            let addr = UNRESOLVED_STUB_BASE + (stubs.len() as u64 * 8);
            stubs.push(UnresolvedStub { nid, library, addr });
            addr
        })
    }
}

/// Lay `module`'s `PT_LOAD` segments into a flat image at `base` and apply
/// every relocation in `dynlib.relocations`, allocating trampoline/stub
/// addresses from `tables`.
///
/// Use this (not [`link_module`]) whenever more than one module is linked into
/// the same process, so their marker addresses share one index space — see
/// [`ProcessTables`] for what goes wrong otherwise. The returned
/// [`LinkedModule`]'s own `hle_trampolines`/`unresolved_stubs` are left empty;
/// the process-wide tables live in `tables`.
pub fn link_module_into(
    module: &SprxModule,
    dynlib: &DynlibData,
    registry: &ModuleRegistry,
    hle: &HleRegistry,
    base: u64,
    tables: &mut ProcessTables,
    tls: Option<TlsAssignment>,
) -> Result<LinkedModule, FirmwareError> {
    link_inner(module, dynlib, registry, hle, base, tables, tls)
}

/// Lay `module`'s `PT_LOAD` segments into a flat image at `base` and apply
/// every relocation in `dynlib.relocations`. See the module docs for the
/// image-layout assumption and the meaning of the HLE-trampoline /
/// unresolved-stub marker addresses.
///
/// Single-module convenience: allocates a private [`ProcessTables`] and moves
/// it into the returned [`LinkedModule`]. For a multi-module process use
/// [`link_module_into`] with one shared `ProcessTables`.
///
/// Never panics: every offset/index derived from `module`/`dynlib` is
/// bounds-checked and returns [`FirmwareError::MalformedDynlibData`] on
/// overflow or out-of-range access. An unresolved import is recorded in
/// `unresolved` and does not fail the call; only a genuinely unsupported
/// relocation type fails it ([`FirmwareError::UnsupportedRelocation`]).
pub fn link_module(
    module: &SprxModule,
    dynlib: &DynlibData,
    registry: &ModuleRegistry,
    hle: &HleRegistry,
    base: u64,
) -> Result<LinkedModule, FirmwareError> {
    let mut tables = ProcessTables::new();
    let mut linked = link_inner(module, dynlib, registry, hle, base, &mut tables, None)?;
    linked.hle_trampolines = tables.hle_trampolines;
    linked.unresolved_stubs = tables.unresolved_stubs;
    Ok(linked)
}

/// The shared body of [`link_module`] / [`link_module_into`].
fn link_inner(
    module: &SprxModule,
    dynlib: &DynlibData,
    registry: &ModuleRegistry,
    hle: &HleRegistry,
    base: u64,
    tables: &mut ProcessTables,
    tls_assignment: Option<TlsAssignment>,
) -> Result<LinkedModule, FirmwareError> {
    // No process-wide assignment means this module is the whole TLS world:
    // module id 1, block directly below the TCB — the single-module layout
    // every existing fixture and test was linked against.
    let tls_assignment = tls_assignment.or_else(|| {
        module.tls.as_ref().map(|t| TlsAssignment {
            module_id: MAIN_TLS_MODULE_ID,
            tp_offset: t.block_size(),
        })
    });
    let mut image = vec![0u8; image_size(module)?];

    for seg in &module.segments {
        let start = usize::try_from(seg.vaddr).map_err(|_| {
            FirmwareError::MalformedDynlibData(format!(
                "segment vaddr {:#x} overflows usize",
                seg.vaddr
            ))
        })?;
        let end = start.checked_add(seg.data.len()).ok_or_else(|| {
            FirmwareError::MalformedDynlibData(format!(
                "segment vaddr {:#x} + data len {:#x} overflows",
                seg.vaddr,
                seg.data.len()
            ))
        })?;
        if end > image.len() {
            return Err(FirmwareError::MalformedDynlibData(format!(
                "segment range [{start:#x}, {end:#x}) exceeds image size {:#x}",
                image.len()
            )));
        }
        image[start..end].copy_from_slice(&seg.data);
    }

    let mut unresolved: Vec<UnresolvedImport> = Vec::new();

    // Provider names are resolved per symtab entry (`r_sym`), not per NID:
    // the same NID can be imported from two modules in one consumer.
    let lib_names: HashMap<u16, &str> = dynlib
        .import_libs
        .iter()
        .map(|(id, n)| (*id, n.as_str()))
        .collect();
    let module_names: HashMap<u16, &str> = dynlib
        .import_modules
        .iter()
        .map(|(id, n)| (*id, n.as_str()))
        .collect();

    for reloc in &dynlib.relocations {
        let r_type = (reloc.info & 0xFFFF_FFFF) as u32;
        let r_sym = reloc.info >> 32;

        let value = match r_type {
            R_X86_64_RELATIVE => base.wrapping_add(reloc.addend as u64),
            // TLS relocations (M1-B, wall #2). Symbol semantics: `r_sym == 0`
            // means a local TLS reference whose template-relative offset is
            // carried entirely in the addend; otherwise the symbol's `value`
            // is its offset within the module's `PT_TLS` template.
            R_X86_64_TPOFF64 => {
                // Variant-II x86-64 static TLS: this module's block sits
                // `tp_offset` bytes below the TCB the FS base points at, so
                // the fs-relative offset of template offset `o` is
                // `o - tp_offset` (negative, wrapped). The assignment's
                // `tp_offset` is the single source of truth both here and in
                // the runtime's block placement — they must agree exactly.
                let assignment =
                    tls_assignment
                        .filter(|_| module.tls.is_some())
                        .ok_or_else(|| {
                            FirmwareError::MalformedDynlibData(
                                "TPOFF64 relocation in a module with no PT_TLS segment".to_string(),
                            )
                        })?;
                let sym_off = tls_symbol_offset(dynlib, r_sym)?;
                sym_off
                    .wrapping_add(reloc.addend as u64)
                    .wrapping_sub(assignment.tp_offset)
            }
            R_X86_64_DTPMOD64 => tls_assignment
                .map(|a| a.module_id)
                .unwrap_or(MAIN_TLS_MODULE_ID),
            R_X86_64_DTPOFF64 => {
                if module.tls.is_none() {
                    return Err(FirmwareError::MalformedDynlibData(
                        "DTPOFF64 relocation in a module with no PT_TLS segment".to_string(),
                    ));
                }
                let sym_off = tls_symbol_offset(dynlib, r_sym)?;
                sym_off.wrapping_add(reloc.addend as u64)
            }
            R_X86_64_64 | R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => {
                let sym_index = usize::try_from(r_sym).map_err(|_| {
                    FirmwareError::MalformedDynlibData(format!(
                        "relocation r_sym {r_sym:#x} overflows usize"
                    ))
                })?;
                let symbol = dynlib.symbols.get(sym_index).ok_or_else(|| {
                    FirmwareError::MalformedDynlibData(format!(
                        "relocation r_sym {sym_index} is out of range of symbols (len {})",
                        dynlib.symbols.len()
                    ))
                })?;

                // A **defined** symbol is not an import: its slot holds the
                // symbol's own address, and there is nothing to look up. Only
                // undefined symbols name something another module must supply.
                //
                // Resolving defined symbols through the NID registry (as this
                // used to do unconditionally) is actively destructive on a real
                // title: its ~717k relocations mostly target its OWN internal
                // symbols, whose NIDs are naturally absent from the HLE
                // registry, so every one of them had `UNRESOLVED_STUB_BASE`
                // written into it — corrupting the module's internal pointers
                // and vtables wholesale, and guaranteeing a fault at the stub
                // the moment any of them was used. It went unnoticed because
                // in-tree fixtures only ever relocate imports.
                if !symbol.is_import {
                    let value =
                        base.wrapping_add(symbol.value)
                            .wrapping_add(if r_type == R_X86_64_64 {
                                reloc.addend as u64
                            } else {
                                0
                            });
                    write_slot(&mut image, reloc.offset, value)?;
                    continue;
                }

                // Dispatch policy belongs to the module that PROVIDES this
                // import. Using `module.name` (the consumer) split a shipped
                // libc between LLE and HLE depending on which object called
                // it, leaving C++ runtime globals owned by different worlds.
                let provider_ref = dynlib
                    .symbol_providers
                    .get(r_sym as usize)
                    .and_then(Option::as_ref)
                    .or_else(|| {
                        // Compatibility for hand-built fixtures predating the
                        // aligned provider table. A unique NID is unambiguous;
                        // duplicate-NID imports must carry per-symbol data.
                        let mut matching = dynlib
                            .imports
                            .iter()
                            .filter(|provider| provider.nid == symbol.nid);
                        let provider = matching.next()?;
                        matching.next().is_none().then_some(provider)
                    });
                let provider_library = provider_ref
                    .and_then(|provider| lib_names.get(&provider.library_index).copied());
                let provider_module = provider_ref
                    .and_then(|provider| module_names.get(&provider.module_index).copied())
                    .or(provider_library)
                    .unwrap_or(&module.name);
                // Forensic (XPS5X_TRACE_DRAWS): name the import whose PLT stub
                // (GOT slot module-vaddr 0xE123280) returns EINVAL and kills the
                // Streaming Pool threads. Logs the NID + provider so the failing
                // function can be identified and fixed.
                if reloc.offset == 0xE12_3280 && std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
                    tracing::warn!(
                        offset = format_args!("{:#x}", reloc.offset),
                        nid = format_args!("{:#018x}", symbol.nid),
                        provider = provider_module,
                        "TRACE_DRAWS: PLT-0xb5 import (returns EINVAL, kills Streaming Pool)"
                    );
                }
                match registry.resolve_import(
                    hle,
                    provider_module,
                    provider_library.unwrap_or(provider_module),
                    symbol.nid,
                ) {
                    Resolver::Hle { library, function } => {
                        let addr = tables.hle_addr(library, function);
                        if r_type == R_X86_64_64 {
                            addr.wrapping_add(reloc.addend as u64)
                        } else {
                            addr
                        }
                    }
                    Resolver::Lle { addr } => addr.wrapping_add(reloc.addend as u64),
                    Resolver::Unresolved => {
                        let nid = symbol.nid;
                        unresolved.push(UnresolvedImport { nid, r_type });
                        // Deliberately NOT `+ addend`, unlike the HLE arm: the
                        // stub address IS the symbol's identity, and the
                        // runtime inverts it by exact slot. An addend would
                        // land mid-slot and lose the NID — the whole point.
                        tables.stub_addr(nid, provider_library.map(str::to_string))
                    }
                }
            }
            other => return Err(FirmwareError::UnsupportedRelocation(other)),
        };

        write_slot(&mut image, reloc.offset, value)?;
    }

    let patched_syscalls = patch_guest_syscalls(module, &mut image)?;
    if patched_syscalls > 0 {
        tracing::info!(
            "{}: patched {patched_syscalls} native syscall instruction(s) into Orbis traps",
            module.name
        );
    }

    Ok(LinkedModule {
        image,
        base,
        unresolved,
        // Left empty here: the marker tables are process-wide and live in
        // `tables`. `link_module` (the single-module wrapper) moves them in
        // afterwards; `load_process` installs the merged tables once, after
        // every module is linked.
        unresolved_stubs: Vec::new(),
        hle_trampolines: Vec::new(),
        // Only `load_process` knows the dependency order, so it fills this in.
        module_inits: Vec::new(),
        entry: module.entry,
        tls: module.tls.clone(),
        // A single module's layout is itself; `load_process` overwrites this
        // with the layout spanning every module in the process.
        tls_layout: match (&module.tls, tls_assignment) {
            (Some(t), Some(a)) => vec![crate::sprx::StaticTlsModule {
                name: module.name.clone(),
                module_id: a.module_id,
                tp_offset: a.tp_offset,
                template: t.clone(),
            }],
            _ => Vec::new(),
        },
        procparam_offset: module.procparam.as_ref().map(|p| p.vaddr),
        unwind_modules: module
            .unwind
            .clone()
            .map(|unwind| LinkedUnwindModule {
                name: module.name.clone(),
                image_offset: 0,
                unwind,
                exports: dynlib.exports.clone(),
                init_vaddr: dynlib.init,
            })
            .into_iter()
            .collect(),
    })
}

/// Decode file-backed executable segments and replace only instruction-boundary
/// `syscall`s. A raw byte search is not safe: `0F 05` can occur inside an
/// immediate or embedded table, and changing those bytes silently corrupts
/// otherwise unrelated guest code/data.
fn patch_guest_syscalls(module: &SprxModule, image: &mut [u8]) -> Result<usize, FirmwareError> {
    let mut sites = Vec::new();
    for segment in module
        .segments
        .iter()
        .filter(|segment| segment.flags & 1 != 0)
    {
        let start = usize::try_from(segment.vaddr).map_err(|_| {
            FirmwareError::MalformedDynlibData(format!(
                "executable segment vaddr {:#x} overflows usize",
                segment.vaddr
            ))
        })?;
        let end = start.checked_add(segment.data.len()).ok_or_else(|| {
            FirmwareError::MalformedDynlibData(format!(
                "executable segment at {:#x} length {:#x} overflows",
                segment.vaddr,
                segment.data.len()
            ))
        })?;
        let bytes = image.get(start..end).ok_or_else(|| {
            FirmwareError::MalformedDynlibData(format!(
                "executable segment [{start:#x}, {end:#x}) exceeds linked image {:#x}",
                image.len()
            ))
        })?;
        let mut decoder = Decoder::with_ip(64, bytes, segment.vaddr, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.mnemonic() != Mnemonic::Syscall {
                continue;
            }
            if instruction.len() != 2 {
                return Err(FirmwareError::MalformedDynlibData(format!(
                    "decoded syscall at {:#x} has impossible length {}",
                    instruction.ip(),
                    instruction.len()
                )));
            }
            let offset =
                usize::try_from(instruction.ip().saturating_sub(segment.vaddr)).map_err(|_| {
                    FirmwareError::MalformedDynlibData(
                        "decoded syscall offset overflows usize".to_string(),
                    )
                })?;
            sites.push(start + offset);
        }
    }

    sites.sort_unstable();
    sites.dedup();
    for site in &sites {
        image[*site..*site + SYSCALL_TRAP_BYTES.len()].copy_from_slice(&SYSCALL_TRAP_BYTES);
    }
    Ok(sites.len())
}

/// The template-relative offset a TLS relocation's symbol contributes:
/// 0 for `r_sym == 0` (local reference, offset in the addend), else the
/// symbol's `value`. Out-of-range `r_sym` is
/// [`FirmwareError::MalformedDynlibData`], mirroring the symbol lookup for
/// the address relocation types above.
fn tls_symbol_offset(dynlib: &DynlibData, r_sym: u64) -> Result<u64, FirmwareError> {
    if r_sym == 0 {
        return Ok(0);
    }
    let sym_index = usize::try_from(r_sym).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("relocation r_sym {r_sym:#x} overflows usize"))
    })?;
    let symbol = dynlib.symbols.get(sym_index).ok_or_else(|| {
        FirmwareError::MalformedDynlibData(format!(
            "TLS relocation r_sym {sym_index} is out of range of symbols (len {})",
            dynlib.symbols.len()
        ))
    })?;
    Ok(symbol.value)
}

/// Size the flat image to `max(seg.vaddr + seg.mem_size)` over `module`'s
/// `PT_LOAD` segments (also covering each segment's actual file-backed
/// `data` length, in case it exceeds `mem_size` for malformed input).
/// Bounds-checked: overflow returns [`FirmwareError::MalformedDynlibData`].
pub fn image_size(module: &SprxModule) -> Result<usize, FirmwareError> {
    let mut max_end: u64 = 0;
    for seg in &module.segments {
        let mem_end = seg.vaddr.checked_add(seg.mem_size).ok_or_else(|| {
            FirmwareError::MalformedDynlibData(format!(
                "segment vaddr {:#x} + mem_size {:#x} overflows u64",
                seg.vaddr, seg.mem_size
            ))
        })?;
        let data_end = seg
            .vaddr
            .checked_add(seg.data.len() as u64)
            .ok_or_else(|| {
                FirmwareError::MalformedDynlibData(format!(
                    "segment vaddr {:#x} + data len overflows u64",
                    seg.vaddr
                ))
            })?;
        max_end = max_end.max(mem_end).max(data_end);
    }
    usize::try_from(max_end).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("image size {max_end:#x} overflows usize"))
    })
}

/// Write `value` little-endian into `image[offset .. offset + 8]`,
/// bounds-checked. Never panics; out-of-range `offset` returns
/// [`FirmwareError::MalformedDynlibData`].
fn write_slot(image: &mut [u8], offset: u64, value: u64) -> Result<(), FirmwareError> {
    let start = usize::try_from(offset).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("relocation offset {offset:#x} overflows usize"))
    })?;
    let end = start.checked_add(8).ok_or_else(|| {
        FirmwareError::MalformedDynlibData(format!("relocation offset {offset:#x} + 8 overflows"))
    })?;
    if end > image.len() {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "relocation slot [{start:#x}, {end:#x}) exceeds image size {:#x}",
            image.len()
        )));
    }
    image[start..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::nid::{NidDatabase, nid_of};
    use crate::dynlib::{DynSymbol, SceRela, SymbolExport, SymbolRef};
    use crate::registry::ModulePolicy;
    use crate::sprx::SprxSegment;

    /// A single-`PT_LOAD`-segment module big enough (`mem_size` bytes) to
    /// hold whatever relocation slots a test writes into it. Entirely
    /// synthetic — no real firmware bytes.
    fn test_module(mem_size: u64) -> SprxModule {
        SprxModule {
            name: "someModule".to_string(),
            e_type: 0xFE18, // ET_SCE_DYNAMIC
            segments: vec![SprxSegment {
                vaddr: 0,
                data: vec![0u8; mem_size as usize],
                flags: 5,
                mem_size,
            }],
            dynlib_data: None,
            relro: None,
            dynamic: None,
            entry: 0,
            tls: None,
            procparam: None,
            unwind: None,
        }
    }

    /// `test_module` plus a `PT_TLS` template: 3 bytes of `.tdata`
    /// (`[0xAB, 0xCD, 0xEF]`), `mem_size` 0x30 (so 0x2D bytes of `.tbss`),
    /// align 0x20 — `block_size()` = 0x40.
    fn test_module_with_tls(mem_size: u64) -> SprxModule {
        let mut module = test_module(mem_size);
        module.tls = Some(crate::sprx::TlsTemplate {
            vaddr: 0x800,
            data: vec![0xAB, 0xCD, 0xEF],
            mem_size: 0x30,
            align: 0x20,
        });
        module
    }

    fn empty_registry() -> (HleRegistry, ModuleRegistry) {
        let hle = HleRegistry::new();
        let db = NidDatabase::from_hle_names(hle.registered_names());
        (hle, ModuleRegistry::new(db))
    }

    fn uniquely_named_hle_function(hle: &HleRegistry) -> (String, String) {
        let mut names = hle.registered_names();
        names.sort();
        let mut counts = HashMap::<&str, usize>::new();
        for (_, function) in &names {
            *counts.entry(function).or_default() += 1;
        }
        names
            .iter()
            .find(|(_, function)| counts[function.as_str()] == 1)
            .cloned()
            .expect("HLE has a uniquely named function")
    }

    fn import_dynlib(library: &str, nid: u64, relocations: Vec<SceRela>) -> DynlibData {
        let provider = SymbolRef {
            nid,
            module_index: 1,
            library_index: 1,
        };
        DynlibData {
            imports: vec![provider],
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            symbol_providers: vec![Some(provider)],
            import_modules: vec![(1, library.to_string())],
            import_libs: vec![(1, library.to_string())],
            relocations,
            ..Default::default()
        }
    }

    fn read_slot(image: &[u8], offset: u64) -> u64 {
        let start = offset as usize;
        u64::from_le_bytes(image[start..start + 8].try_into().unwrap())
    }

    #[test]
    fn relative_relocation_writes_base_plus_addend() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x8,
                info: R_X86_64_RELATIVE as u64, // r_sym = 0, unused for RELATIVE
                addend: 0x20,
            }],
            ..Default::default()
        };

        let base = 0x1_0000_0000u64;
        let linked =
            link_module(&module, &dynlib, &registry, &hle, base).expect("relative reloc links");

        assert_eq!(read_slot(&linked.image, 0x8), base + 0x20);
        assert!(linked.unresolved.is_empty());
        assert!(linked.hle_trampolines.is_empty());
        assert_eq!(linked.base, base);
    }

    /// Two modules sharing one [`ProcessTables`] must get DISTINCT marker
    /// addresses for DISTINCT symbols — and the shared table must name both.
    ///
    /// This pins the aliasing bug: `HLE_TRAMPOLINE_BASE`/`UNRESOLVED_STUB_BASE`
    /// are process-global address spaces, but `link_module` numbers from 0. A
    /// process that linked each module with private tables gave module B's
    /// import #0 the same address as module A's import #0, and since the
    /// runtime holds one table, B's call dispatched to A's function — silently.
    /// Caught for real: the composed title reported its blocking import as
    /// `libSceNpManager` when it was actually `libkernel`'s `__stack_chk_guard`,
    /// because a dependency's stub index was read out of the main module's table.
    #[test]
    fn two_modules_sharing_process_tables_get_distinct_marker_slots() {
        let (hle, registry) = empty_registry();
        let nid_a = nid_of("someUnknownFunctionA");
        let nid_b = nid_of("someUnknownFunctionB");

        let make = |nid: u64| {
            (
                test_module(0x100),
                DynlibData {
                    symbols: vec![DynSymbol {
                        nid,
                        value: 0,
                        is_import: true,
                    }],
                    relocations: vec![SceRela {
                        offset: 0x10,
                        info: R_X86_64_JUMP_SLOT as u64,
                        addend: 0,
                    }],
                    ..Default::default()
                },
            )
        };
        let (mod_a, dyn_a) = make(nid_a);
        let (mod_b, dyn_b) = make(nid_b);

        let mut tables = ProcessTables::new();
        let a =
            link_module_into(&mod_a, &dyn_a, &registry, &hle, 0, &mut tables, None).expect("links");
        let b =
            link_module_into(&mod_b, &dyn_b, &registry, &hle, 0, &mut tables, None).expect("links");

        let slot_a = read_slot(&a.image, 0x10);
        let slot_b = read_slot(&b.image, 0x10);
        assert_eq!(
            slot_a, UNRESOLVED_STUB_BASE,
            "first distinct NID owns slot 0"
        );
        assert_eq!(
            slot_b,
            UNRESOLVED_STUB_BASE + 8,
            "the SECOND module's distinct NID must get its OWN slot, not restart at 0"
        );
        assert_ne!(
            slot_a, slot_b,
            "two symbols must never share a stub address"
        );

        // The shared table names both, in slot order — this is what the runtime
        // inverts, so it must cover every module in the process.
        assert_eq!(tables.unresolved_stubs().len(), 2);
        assert_eq!(tables.unresolved_stubs()[0].nid, nid_a);
        assert_eq!(tables.unresolved_stubs()[1].nid, nid_b);

        // Per-module tables are empty: they live in `tables` now.
        assert!(a.unresolved_stubs.is_empty() && b.unresolved_stubs.is_empty());
    }

    /// The same provider/NID imported twice shares one process-wide slot.
    #[test]
    fn the_same_nid_in_two_modules_shares_one_stub_slot() {
        let (hle, registry) = empty_registry();
        let nid = nid_of("someUnknownFunctionSharedByTwoModules");

        let dynlib = DynlibData {
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_JUMP_SLOT as u64,
                addend: 0,
            }],
            ..Default::default()
        };
        let module = test_module(0x100);

        let mut tables = ProcessTables::new();
        let a = link_module_into(&module, &dynlib, &registry, &hle, 0, &mut tables, None)
            .expect("links");
        let b = link_module_into(&module, &dynlib, &registry, &hle, 0, &mut tables, None)
            .expect("links");

        assert_eq!(read_slot(&a.image, 0x10), read_slot(&b.image, 0x10));
        assert_eq!(
            tables.unresolved_stubs().len(),
            1,
            "one NID, one slot, however many modules import it"
        );
    }

    #[test]
    fn same_unresolved_nid_from_two_libraries_keeps_distinct_diagnostics() {
        let (hle, registry) = empty_registry();
        let nid = nid_of("unknownSharedName");
        let module = test_module(0x100);
        let dynlib = DynlibData {
            imports: vec![
                SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 1,
                },
                SymbolRef {
                    nid,
                    module_index: 2,
                    library_index: 2,
                },
            ],
            symbols: vec![
                DynSymbol {
                    nid,
                    value: 0,
                    is_import: true,
                },
                DynSymbol {
                    nid,
                    value: 0,
                    is_import: true,
                },
            ],
            symbol_providers: vec![
                Some(SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 1,
                }),
                Some(SymbolRef {
                    nid,
                    module_index: 2,
                    library_index: 2,
                }),
            ],
            import_modules: vec![(1, "modAlpha".to_string()), (2, "modBeta".to_string())],
            import_libs: vec![(1, "libAlpha".to_string()), (2, "libBeta".to_string())],
            relocations: vec![
                SceRela {
                    offset: 0x18,
                    info: R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
                SceRela {
                    offset: 0x20,
                    info: (1u64 << 32) | R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
            ],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");
        assert_ne!(
            read_slot(&linked.image, 0x18),
            read_slot(&linked.image, 0x20)
        );
        assert_eq!(linked.unresolved_stubs.len(), 2);
        assert_eq!(
            linked.unresolved_stubs[0].library.as_deref(),
            Some("libAlpha")
        );
        assert_eq!(
            linked.unresolved_stubs[1].library.as_deref(),
            Some("libBeta")
        );
    }

    #[test]
    fn same_hle_nid_from_two_libraries_keeps_distinct_trampolines() {
        let (hle, registry) = empty_registry();
        let nid = nid_of("getpid");
        let module = test_module(0x100);
        let dynlib = DynlibData {
            imports: vec![
                SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 1,
                },
                SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 2,
                },
            ],
            symbols: vec![
                DynSymbol {
                    nid,
                    value: 0,
                    is_import: true,
                },
                DynSymbol {
                    nid,
                    value: 0,
                    is_import: true,
                },
            ],
            symbol_providers: vec![
                Some(SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 1,
                }),
                Some(SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 2,
                }),
            ],
            import_modules: vec![(1, "libkernel".to_string())],
            import_libs: vec![(1, "libkernel".to_string()), (2, "libScePosix".to_string())],
            relocations: vec![
                SceRela {
                    offset: 0x18,
                    info: R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
                SceRela {
                    offset: 0x20,
                    info: (1u64 << 32) | R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
            ],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");
        assert_ne!(
            read_slot(&linked.image, 0x18),
            read_slot(&linked.image, 0x20)
        );
        assert_eq!(linked.hle_trampolines.len(), 2);
        assert_eq!(linked.hle_trampolines[0].library, "libkernel");
        assert_eq!(linked.hle_trampolines[1].library, "libScePosix");
    }

    #[test]
    fn jump_slot_import_resolves_to_hle_trampoline() {
        let (hle, registry) = empty_registry();
        let (library, function) = uniquely_named_hle_function(&hle);
        let nid = nid_of(&function);

        let module = test_module(0x100);
        let dynlib = import_dynlib(
            &library,
            nid,
            vec![SceRela {
                offset: 0x10,
                info: R_X86_64_JUMP_SLOT as u64, // r_sym = 0
                addend: 0,
            }],
        );

        let linked =
            link_module(&module, &dynlib, &registry, &hle, 0).expect("HLE-resolved reloc links");

        assert_eq!(read_slot(&linked.image, 0x10), HLE_TRAMPOLINE_BASE);
        assert_eq!(linked.hle_trampolines.len(), 1);
        assert_eq!(linked.hle_trampolines[0].library, library);
        assert_eq!(linked.hle_trampolines[0].function, function);
        assert_eq!(linked.hle_trampolines[0].addr, HLE_TRAMPOLINE_BASE);
        assert!(linked.unresolved.is_empty());
    }

    #[test]
    fn repeated_hle_import_reuses_same_trampoline_address() {
        let (hle, registry) = empty_registry();
        let (library, function) = uniquely_named_hle_function(&hle);
        let nid = nid_of(&function);

        let module = test_module(0x100);
        let dynlib = import_dynlib(
            &library,
            nid,
            vec![
                SceRela {
                    offset: 0x10,
                    info: R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
                SceRela {
                    offset: 0x18,
                    info: R_X86_64_GLOB_DAT as u64,
                    addend: 0,
                },
            ],
        );

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");

        assert_eq!(read_slot(&linked.image, 0x10), HLE_TRAMPOLINE_BASE);
        assert_eq!(read_slot(&linked.image, 0x18), HLE_TRAMPOLINE_BASE);
        // Same NID referenced twice -> one trampoline record, not two.
        assert_eq!(linked.hle_trampolines.len(), 1);
    }

    #[test]
    fn import_resolves_to_lle_export_plus_addend() {
        let (hle, mut registry) = empty_registry();
        let nid = nid_of("someUniqueLleOnlyExport");
        registry.register_module_exports("someModule", &[SymbolExport { nid, value: 0x9999 }]);

        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x18,
                info: R_X86_64_64 as u64, // r_sym = 0
                addend: 0x5,
            }],
            ..Default::default()
        };

        let linked =
            link_module(&module, &dynlib, &registry, &hle, 0).expect("LLE-resolved reloc links");

        assert_eq!(read_slot(&linked.image, 0x18), 0x9999 + 0x5);
        assert!(linked.unresolved.is_empty());
        assert!(linked.hle_trampolines.is_empty());
    }

    #[test]
    fn equal_nids_from_two_providers_resolve_per_symbol_index() {
        let (hle, mut registry) = empty_registry();
        let nid = nid_of("sameNamedExport");
        registry.register_module_exports("libAlpha", &[SymbolExport { nid, value: 0x1111 }]);
        registry.register_module_exports("libBeta", &[SymbolExport { nid, value: 0x2222 }]);

        let module = test_module(0x100);
        let dynlib = DynlibData {
            imports: vec![
                SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 1,
                },
                SymbolRef {
                    nid,
                    module_index: 2,
                    library_index: 2,
                },
            ],
            import_modules: vec![(1, "libAlpha".to_string()), (2, "libBeta".to_string())],
            import_libs: vec![(1, "libAlpha".to_string()), (2, "libBeta".to_string())],
            symbols: vec![
                DynSymbol {
                    nid,
                    value: 0,
                    is_import: true,
                },
                DynSymbol {
                    nid,
                    value: 0,
                    is_import: true,
                },
            ],
            symbol_providers: vec![
                Some(SymbolRef {
                    nid,
                    module_index: 1,
                    library_index: 1,
                }),
                Some(SymbolRef {
                    nid,
                    module_index: 2,
                    library_index: 2,
                }),
            ],
            relocations: vec![
                SceRela {
                    offset: 0x18,
                    info: R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
                SceRela {
                    offset: 0x20,
                    info: (1u64 << 32) | R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
            ],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");
        assert_eq!(read_slot(&linked.image, 0x18), 0x1111);
        assert_eq!(read_slot(&linked.image, 0x20), 0x2222);
    }

    #[test]
    fn decoded_syscall_is_trapped_without_corrupting_the_same_bytes_in_an_immediate() {
        let (hle, registry) = empty_registry();
        let mut module = test_module(8);
        // mov eax, 0x0000050f ; syscall ; ret
        module.segments[0].data = vec![0xB8, 0x0F, 0x05, 0x00, 0x00, 0x0F, 0x05, 0xC3];
        let linked =
            link_module(&module, &DynlibData::default(), &registry, &hle, 0).expect("links");

        assert_eq!(&linked.image[1..3], &[0x0F, 0x05]);
        assert_eq!(&linked.image[5..7], &SYSCALL_TRAP_BYTES);
    }

    #[test]
    fn syscall_bytes_in_a_non_executable_segment_are_not_patched() {
        let (hle, registry) = empty_registry();
        let mut module = test_module(2);
        module.segments[0].flags = 4;
        module.segments[0].data = vec![0x0F, 0x05];
        let linked =
            link_module(&module, &DynlibData::default(), &registry, &hle, 0).expect("links");

        assert_eq!(&linked.image[..2], &[0x0F, 0x05]);
    }

    #[test]
    fn shipped_provider_policy_owns_imports_from_an_eboot_consumer() {
        let (hle, mut registry) = empty_registry();
        let (_, function) = hle
            .registered_names()
            .into_iter()
            .next()
            .expect("HleRegistry::new() registers at least one function");
        let nid = nid_of(&function);
        registry.register_module_exports("libc.prx", &[SymbolExport { nid, value: 0x7777 }]);
        registry.set_policy("libc.prx", ModulePolicy::PreferLle);

        let module = test_module(0x100);
        let dynlib = DynlibData {
            imports: vec![SymbolRef {
                nid,
                module_index: 9,
                library_index: 7,
            }],
            import_libs: vec![(7, "libc".to_string())],
            import_modules: vec![(9, "libc".to_string())],
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x18,
                info: R_X86_64_JUMP_SLOT as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links through LLE");

        assert_eq!(read_slot(&linked.image, 0x18), 0x7777);
        assert!(linked.hle_trampolines.is_empty());
    }

    #[test]
    fn unknown_nid_writes_stub_and_is_recorded_unresolved_without_failing() {
        let (hle, registry) = empty_registry();
        let nid = nid_of("totallyUnknownFunctionNameNobodyRegistered");

        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x20,
                info: R_X86_64_GLOB_DAT as u64, // r_sym = 0
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0)
            .expect("unresolved import is non-fatal");

        assert_eq!(read_slot(&linked.image, 0x20), UNRESOLVED_STUB_BASE);
        assert_eq!(
            linked.unresolved,
            vec![UnresolvedImport {
                nid,
                r_type: R_X86_64_GLOB_DAT,
            }]
        );
    }

    #[test]
    fn entry_offset_propagates_from_module_to_linked_module() {
        let (hle, registry) = empty_registry();
        let mut module = test_module(0x100);
        module.entry = 0x40;
        let dynlib = DynlibData::default();

        let linked = link_module(&module, &dynlib, &registry, &hle, 0x1000).expect("links");
        assert_eq!(linked.entry, 0x40);
    }

    // --- TLS relocations (M1-B, wall #2) ---------------------------------

    #[test]
    fn tpoff64_writes_negative_block_relative_offset() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            // r_sym == 0: local TLS reference, offset carried in the addend.
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_TPOFF64 as u64,
                addend: 0x8,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("TPOFF64 links");

        // block_size = 0x40 (0x30 rounded to align 0x20, min 16); the
        // fs-relative offset of template offset 0x8 is 0x8 - 0x40 = -0x38.
        assert_eq!(read_slot(&linked.image, 0x10), (-0x38i64) as u64);
    }

    #[test]
    fn tpoff64_uses_symbol_value_for_nonzero_r_sym() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            symbols: vec![
                DynSymbol {
                    nid: 0,
                    value: 0,
                    is_import: false,
                },
                DynSymbol {
                    nid: nid_of("someTlsVar"),
                    value: 0x10,
                    is_import: false,
                },
            ],
            relocations: vec![SceRela {
                offset: 0x18,
                info: (1u64 << 32) | R_X86_64_TPOFF64 as u64,
                addend: 0x4,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("TPOFF64 links");
        // 0x10 (symbol) + 0x4 (addend) - 0x40 (block) = -0x2C.
        assert_eq!(read_slot(&linked.image, 0x18), (-0x2Ci64) as u64);
    }

    #[test]
    fn dtpmod64_writes_main_module_id() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x20,
                info: R_X86_64_DTPMOD64 as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("DTPMOD64 links");
        assert_eq!(
            read_slot(&linked.image, 0x20),
            1,
            "single-module world: TLS module id is 1"
        );
    }

    #[test]
    fn dtpoff64_writes_template_relative_offset() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x28,
                info: R_X86_64_DTPOFF64 as u64,
                addend: 0xC,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("DTPOFF64 links");
        assert_eq!(read_slot(&linked.image, 0x28), 0xC);
    }

    /// A module linked with a process-wide [`TlsAssignment`] must resolve its
    /// TLS relocations against ITS slot: `DTPMOD64` names its real module id
    /// and `TPOFF64` subtracts its assigned distance below the thread pointer
    /// — not its own block size, which is only correct for a module sitting
    /// alone directly below the TCB. Ignoring the assignment folds every
    /// module's thread-locals onto the main executable's block (the measured
    /// retail-title TLS corruption; see `sprx::StaticTlsModule`).
    #[test]
    fn tls_assignment_overrides_module_id_and_tp_offset() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            relocations: vec![
                SceRela {
                    offset: 0x10,
                    info: R_X86_64_TPOFF64 as u64,
                    addend: 0x8,
                },
                SceRela {
                    offset: 0x20,
                    info: R_X86_64_DTPMOD64 as u64,
                    addend: 0,
                },
            ],
            ..Default::default()
        };

        let mut tables = ProcessTables::new();
        let linked = link_module_into(
            &module,
            &dynlib,
            &registry,
            &hle,
            0,
            &mut tables,
            Some(TlsAssignment {
                module_id: 3,
                tp_offset: 0xA0,
            }),
        )
        .expect("assigned TLS links");

        // fs-relative offset of template offset 0x8 with the block 0xA0 below
        // the thread pointer: 0x8 - 0xA0 = -0x98 — NOT 0x8 - block_size(0x40).
        assert_eq!(read_slot(&linked.image, 0x10), (-0x98i64) as u64);
        assert_eq!(
            read_slot(&linked.image, 0x20),
            3,
            "DTPMOD64 is the assigned id"
        );

        // The layout entry the runtime will materialize records the same slot.
        assert_eq!(linked.tls_layout.len(), 1);
        assert_eq!(linked.tls_layout[0].module_id, 3);
        assert_eq!(linked.tls_layout[0].tp_offset, 0xA0);
    }

    #[test]
    fn tpoff64_without_pt_tls_is_a_hard_link_error() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100); // no TLS template
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_TPOFF64 as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn tls_template_is_carried_into_linked_module() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData::default();

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");
        let tls = linked.tls.expect("TLS template carried through");
        assert_eq!(tls.data, vec![0xAB, 0xCD, 0xEF]);
        assert_eq!(tls.mem_size, 0x30);
        assert_eq!(tls.block_size(), 0x40);
    }

    #[test]
    fn unsupported_relocation_type_errors() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x0,
                info: 99,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::UnsupportedRelocation(99)));
    }

    #[test]
    fn relocation_offset_past_image_end_errors_not_panics() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x10);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x1_0000, // far past the 0x10-byte image
                info: R_X86_64_RELATIVE as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn relocation_r_sym_past_symbols_len_errors_not_panics() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: Vec::new(), // r_sym 5 is out of range of an empty table
            relocations: vec![SceRela {
                offset: 0x8,
                info: (5u64 << 32) | R_X86_64_JUMP_SLOT as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }
}
