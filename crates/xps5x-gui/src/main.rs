//! # XPS5X — PlayStation 5 Emulator
//!
//! Main entry point for the XPS5X desktop application.
//! Initializes the emulator subsystems and launches the GUI.

mod app;
mod launcher;
mod library;
mod shell;
mod splash;
mod theme;
mod updater;

use tracing::info;

/// Say what the guest was executing when it faulted: which module the faulting
/// `rip` lands in, its offset within that module, and the bytes there.
///
/// A bare `rip` names nothing. In a 250 MB stripped C++ binary one address looks
/// like any other, and the temptation is to reason from whatever HLE call
/// happened to be last — which is adjacency, not causation, and has already cost
/// this project one wrong diagnosis. The module and offset make a fault
/// comparable across runs and greppable against a disassembly; the bytes make it
/// decodable on the spot.
///
/// Reads the composed process image the loader already holds, so it cannot
/// perturb the guest or the fault path.
fn report_fault_site(
    linked: &xps5x_firmware::LinkedModule,
    deps: &[xps5x_firmware::LoadedDependency],
    addr: u64,
) {
    let Some(offset) = addr.checked_sub(xps5x_runtime::GUEST_ARENA_BASE) else {
        info!("        rip is below the guest image — not guest code");
        return;
    };
    let Ok(offset) = usize::try_from(offset) else {
        return;
    };
    if offset >= linked.image.len() {
        info!(
            "        rip is past the loaded image ({:#x} byte(s)) — not guest code",
            linked.image.len()
        );
        return;
    }

    // Dependencies are composed above the eboot at known offsets, so the last
    // one at or below the rip owns it; below them all, it is the eboot's.
    match deps
        .iter()
        .filter(|d| usize::try_from(d.image_offset).is_ok_and(|off| off <= offset))
        .max_by_key(|d| d.image_offset)
    {
        Some(d) => info!(
            "        module: {} at +{:#x}",
            d.name,
            offset as u64 - d.image_offset
        ),
        None => info!("        module: eboot.bin at +{offset:#x}"),
    }
    let end = (offset + 16).min(linked.image.len());
    info!("        bytes at rip: {:02x?}", &linked.image[offset..end]);
}

fn main() -> anyhow::Result<()> {
    // Initialize logging to BOTH stderr and `logs/xps5x.log`. `_log` must stay
    // alive for the whole process — dropping it shuts down the background
    // writer thread and loses buffered events (see `LogGuard`). Binding it here
    // in `main` is what makes the log file complete on exit.
    //
    // Falls back to stderr-only if the log directory can't be created (e.g. a
    // read-only working directory) — never a reason to refuse to boot.
    let _log = match xps5x_core::logging::init_with_file(
        "info",
        std::path::Path::new(xps5x_core::logging::DEFAULT_LOG_DIR),
    ) {
        Ok(guard) => guard,
        Err(e) => {
            let guard = xps5x_core::logging::init("info");
            tracing::warn!("file logging unavailable ({e}); continuing with stderr only");
            guard
        }
    };

    // Diagnostic: `xps5x --firmware-info <PUP>` inspects a firmware package
    // and exits without launching the GUI. It never decrypts anything.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--firmware-info") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--firmware-info requires a path to a PUP file"))?;
        let firmware = xps5x_firmware::Firmware::open(path)?;
        print!("{}", xps5x_firmware::summarize(&firmware));
        return Ok(());
    }

    // Diagnostic: `xps5x --load-sprx <sprx>` runs the LM1 homebrew pipeline
    // (SELF decrypt-or-passthrough -> .sprx parse -> dynlibdata decode ->
    // NID link against HLE) over a file and prints a summary, then exits
    // without launching the GUI. Uses `NoKeysProvider` throughout — it never
    // decrypts anything without a user-supplied key.
    if let Some(pos) = args.iter().position(|a| a == "--load-sprx") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--load-sprx requires a path to a .sprx/SELF file"))?;
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let dyn_tags = match &module.dynamic {
            Some(d) => xps5x_firmware::dynlib::parse_sce_dynamic(d)?,
            None => Vec::new(),
        };
        // Two dynamic models: the PT_SCE_DYNLIBDATA blob (homebrew/.sprx) or
        // standard vaddr-based tags with no such segment (real PS5 titles).
        let standard = xps5x_firmware::dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
        let dynlib_data = match &standard {
            Some((image, tags)) => xps5x_firmware::dynlib::parse_dynlibdata(image, tags)?,
            None => xps5x_firmware::dynlib::parse_dynlibdata(
                module.dynlib_data.as_deref().unwrap_or(&[]),
                &dyn_tags,
            )?,
        };
        if let Ok(value) = std::env::var("XPS5X_RELOC_OFFSET") {
            let value = value.trim_start_matches("0x");
            if let Ok(offset) = u64::from_str_radix(value, 16) {
                for relocation in dynlib_data
                    .relocations
                    .iter()
                    .filter(|relocation| relocation.offset == offset)
                {
                    let symbol_index = (relocation.info >> 32) as usize;
                    let r_type = relocation.info as u32;
                    match dynlib_data.symbols.get(symbol_index) {
                        Some(symbol) => println!(
                            "relocation {offset:#x}: type={r_type} sym={symbol_index} \
                             nid={:#018x} ({}) value={:#x} import={} {}",
                            symbol.nid,
                            xps5x_firmware::dynlib::nid::encode_nid(symbol.nid),
                            symbol.value,
                            symbol.is_import,
                            xps5x_firmware::dynlib::nid_names::describe(symbol.nid),
                        ),
                        None => println!(
                            "relocation {offset:#x}: type={r_type} invalid sym={symbol_index}"
                        ),
                    }
                }
            }
        }
        let hle = std::sync::Arc::new(xps5x_hle::HleRegistry::new());
        let db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        let mut registry = xps5x_firmware::ModuleRegistry::new(db);
        registry.register_module_exports(&module.name, &dynlib_data.exports);
        let linked = xps5x_firmware::link_module(&module, &dynlib_data, &registry, &hle, 0)?;
        println!("module: {}", module.name);
        println!(
            "imports: {}  exports: {}",
            dynlib_data.imports.len(),
            dynlib_data.exports.len()
        );
        // The export *table* and the symbol table are different things, and the
        // gap between them is where Minecraft's boot dies: a symbol this module
        // defines but never publishes as an export is invisible to dlsym. If
        // `defined` here materially exceeds `exports`, the export parse is
        // dropping symbols the guest can legitimately ask for.
        let defined = dynlib_data.symbols.iter().filter(|s| !s.is_import).count();
        println!(
            "dynsym: {} symbol(s) — {} defined, {} imported",
            dynlib_data.symbols.len(),
            defined,
            dynlib_data.symbols.len() - defined
        );
        // What a module exports is the whole contract a dependent links against,
        // and "41 exports" says nothing about whether the three symbols a title
        // actually wants are among them. Minecraft's boot dies on exactly that
        // gap: MediaDecoders parses 41 exports and none is CreateMP3Decoder.
        for export in &dynlib_data.exports {
            println!(
                "  export nid={:#018x} ({}) vaddr={:#x}  {}",
                export.nid,
                xps5x_firmware::dynlib::nid::encode_nid(export.nid),
                export.value,
                xps5x_firmware::dynlib::nid_names::describe(export.nid)
            );
        }
        println!(
            "resolved HLE trampolines: {}  unresolved: {}",
            linked.hle_trampolines.len(),
            linked.unresolved.len()
        );

        // Turn "N unresolved" into an actionable list. A NID is a one-way hash,
        // so an unresolved one can't be turned back into a function name — but
        // each import symbol carries a library_index, and DT_SCE_IMPORT_LIB_1
        // maps that to a real library name.
        //
        // Two things this must NOT do, both of which produced a badly wrong
        // headline before:
        //  * index library_index into the *needed-module* table (renames every
        //    library: "libc" became "libfmod"), and
        //  * report the raw relocation count as if it were a count of missing
        //    functions. It is one entry PER RELOCATION, and on a real C++ title
        //    a handful of RTTI symbols generate ~99% of them. The JUMP_SLOT
        //    subtotal is the number that actually sizes the HLE work.
        if !linked.unresolved.is_empty() {
            use std::collections::HashMap;
            const R_X86_64_64: u32 = 1;
            const R_X86_64_GLOB_DAT: u32 = 6;
            const R_X86_64_JUMP_SLOT: u32 = 7;

            let lib_names: HashMap<u16, &str> = dynlib_data
                .import_libs
                .iter()
                .map(|(id, n)| (*id, n.as_str()))
                .collect();
            let nid_to_lib: HashMap<u64, u16> = dynlib_data
                .imports
                .iter()
                .map(|s| (s.nid, s.library_index))
                .collect();

            let mut by_type: HashMap<u32, usize> = HashMap::new();
            // (relocations, distinct called NIDs) per library.
            let mut per_lib: HashMap<&str, usize> = HashMap::new();
            let mut called_per_lib: HashMap<&str, std::collections::HashSet<u64>> = HashMap::new();
            let mut unknown = 0usize;
            for u in &linked.unresolved {
                *by_type.entry(u.r_type).or_default() += 1;
                match nid_to_lib.get(&u.nid).and_then(|i| lib_names.get(i)) {
                    Some(name) => {
                        *per_lib.entry(name).or_default() += 1;
                        if u.r_type == R_X86_64_JUMP_SLOT {
                            called_per_lib.entry(name).or_default().insert(u.nid);
                        }
                    }
                    None => unknown += 1,
                }
            }

            println!("\nunresolved relocations by type:");
            let mut types: Vec<_> = by_type.into_iter().collect();
            types.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            for (t, n) in &types {
                let what = match *t {
                    R_X86_64_JUMP_SLOT => "JUMP_SLOT  - a function the guest CALLS",
                    R_X86_64_64 => "R_X86_64_64 - a data pointer slot (RTTI/vtable)",
                    R_X86_64_GLOB_DAT => "GLOB_DAT   - a data pointer slot",
                    _ => "other",
                };
                println!("  {n:>6}  {what}");
            }

            let called: usize = linked
                .unresolved
                .iter()
                .filter(|u| u.r_type == R_X86_64_JUMP_SLOT)
                .count();
            println!(
                "\nunresolved imports by library (relocations, then distinct CALLED functions).\n\
                 The second column is the real work: {called} function(s) are actually called."
            );
            let mut ranked: Vec<_> = per_lib.into_iter().collect();
            ranked.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            println!("  {:>8}  {:>8}  library", "relocs", "called");
            for (lib, n) in ranked.iter().take(20) {
                let c = called_per_lib.get(lib).map_or(0, |s| s.len());
                println!("  {n:>8}  {c:>8}  {lib}");
            }
            if unknown > 0 {
                println!("  {unknown:>8}  {:>8}  <library unknown>", "?");
            }
        }
        return Ok(());
    }

    // Diagnostic: `xps5x --dump-vaddr <eboot.bin> <hex-vaddr> [<len>]` prints
    // hex + ASCII at a module-relative virtual address, straight from the
    // parsed segments without executing anything.
    //
    // This exists because a fault report can only print what was in registers
    // at the crash; the *static* strings an assert site references (its
    // expression text, source-file path) live at RIP-relative addresses the
    // report names but cannot read once the process is gone. This turns those
    // offsets into the message the game was trying to log.
    if let Some(pos) = args.iter().position(|a| a == "--dump-vaddr") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--dump-vaddr requires a path to an eboot.bin"))?;
        let vaddr = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--dump-vaddr requires a hex vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad vaddr {s:?}: {e}"))
            })?;
        let len = match args.get(pos + 3) {
            Some(s) => s.parse::<usize>()?,
            None => 256,
        };
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let segment = module
            .segments
            .iter()
            .find(|s| vaddr >= s.vaddr && vaddr < s.vaddr + s.mem_size)
            .ok_or_else(|| anyhow::anyhow!("vaddr {vaddr:#x} is not in any PT_LOAD segment"))?;
        let start = (vaddr - segment.vaddr) as usize;
        let file_backed = segment.data.len().saturating_sub(start);
        if file_backed == 0 {
            println!(
                "vaddr {vaddr:#x} is in BSS (zero-initialized) of segment {:#x}",
                segment.vaddr
            );
            return Ok(());
        }
        let slice = &segment.data[start..start + len.min(file_backed)];
        for (i, chunk) in slice.chunks(16).enumerate() {
            let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
            let ascii: String = chunk
                .iter()
                .map(|&b| {
                    if (0x20..0x7f).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            println!(
                "{:#014x}  {:<47}  {ascii}",
                vaddr + (i * 16) as u64,
                hex.join(" ")
            );
        }
        return Ok(());
    }

    // Diagnostic: `xps5x --disas <eboot.bin> <hex-vaddr> [<len-decimal>]`
    // disassembles x86-64 from a module vaddr — the missing piece for RE'ing
    // guest boot-decision logic (e.g. Minecraft's never-taken CreateView
    // branch). Reuses the workspace iced-x86 decoder the VEH already links.
    if let Some(pos) = args.iter().position(|a| a == "--disas") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--disas requires a path to an eboot.bin"))?;
        let vaddr = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--disas requires a hex vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad vaddr {s:?}: {e}"))
            })?;
        let len = match args.get(pos + 3) {
            Some(s) => s.parse::<usize>()?,
            None => 256,
        };
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let segment = module
            .segments
            .iter()
            .find(|s| vaddr >= s.vaddr && vaddr < s.vaddr + s.mem_size)
            .ok_or_else(|| anyhow::anyhow!("vaddr {vaddr:#x} is not in any PT_LOAD segment"))?;
        let start = (vaddr - segment.vaddr) as usize;
        let file_backed = segment.data.len().saturating_sub(start);
        if file_backed == 0 {
            println!("vaddr {vaddr:#x} is in BSS");
            return Ok(());
        }
        let slice = &segment.data[start..start + len.min(file_backed)];
        use iced_x86::Formatter as _;
        let mut decoder =
            iced_x86::Decoder::with_ip(64, slice, vaddr, iced_x86::DecoderOptions::NONE);
        let mut formatter = iced_x86::IntelFormatter::new();
        let mut out = String::new();
        let mut inst = iced_x86::Instruction::default();
        while decoder.can_decode() {
            decoder.decode_out(&mut inst);
            out.clear();
            formatter.format(&inst, &mut out);
            // Flag control-flow so a branch that gates a subsystem stands out.
            // Keyed off the formatted mnemonic to avoid the `code_asm`/flow
            // feature; conditional jumps start with "j" but not "jmp".
            let m = out.split_whitespace().next().unwrap_or("");
            let mark = if m == "call" {
                " (call)"
            } else if m == "ret" {
                " (ret)"
            } else if m == "jmp" {
                " (jmp)"
            } else if m.starts_with('j') {
                " <-- COND"
            } else if m == "test" || m == "cmp" {
                " <-- TEST"
            } else {
                ""
            };
            println!("{:#014x}  {out}{mark}", inst.ip());
        }
        return Ok(());
    }

    // Diagnostic: `xps5x --find-calls <eboot.bin> <hex-target-vaddr>` scans every
    // executable segment for a call whose target is the given vaddr — the core
    // "who calls X" RE operation. Finds direct `call rel32` (e8) and
    // `call [rip+disp32]` (ff 15) through a GOT slot holding the target. Prints
    // each call-site vaddr, so the guarding branch upstream can be disassembled.
    if let Some(pos) = args.iter().position(|a| a == "--find-calls") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--find-calls requires an eboot.bin path"))?;
        let target = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--find-calls requires a hex target vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad target {s:?}: {e}"))
            })?;
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let mut hits = 0usize;
        for seg in &module.segments {
            // Executable segments only (PF_X = bit 0).
            if seg.flags & 1 == 0 {
                continue;
            }
            let data = &seg.data;
            let base = seg.vaddr;
            // Direct `call rel32`: e8 <rel32>. Target = insn_end + rel32.
            for i in 0..data.len().saturating_sub(5) {
                if data[i] == 0xe8 {
                    let rel =
                        i32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
                    let site = base + i as u64;
                    let dest = (site + 5).wrapping_add(rel as i64 as u64);
                    if dest == target {
                        println!("{site:#014x}  call {target:#x} (direct)");
                        hits += 1;
                    }
                }
            }
        }
        eprintln!("{hits} direct call site(s) to {target:#x}");
        return Ok(());
    }

    // Diagnostic: `xps5x --find-lea <eboot.bin> <hex-target-vaddr>` scans every
    // executable segment for `lea reg, [rip+disp32]` whose target is the given
    // vaddr — "who references this data/string". Pairs with --find-calls: locate
    // a .rodata string (e.g. "coui://", "index.html") then find the code that
    // loads its address to build a navigate URL.
    if let Some(pos) = args.iter().position(|a| a == "--find-lea") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--find-lea requires an eboot.bin path"))?;
        let target = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--find-lea requires a hex target vaddr"))
            .and_then(|s| {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|e| anyhow::anyhow!("bad target {s:?}: {e}"))
            })?;
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let mut hits = 0usize;
        for seg in &module.segments {
            if seg.flags & 1 == 0 {
                continue;
            }
            let data = &seg.data;
            let base = seg.vaddr;
            // REX.W lea rip-relative: 48 8d <modrm> disp32, modrm in {05,0d,15,
            // 1d,25,2d,35,3d} (mod=00, rm=101). insn length = 7. Target =
            // (site+7) + disp32.
            for i in 0..data.len().saturating_sub(7) {
                if data[i] == 0x48 && data[i + 1] == 0x8d && (data[i + 2] & 0xc7) == 0x05 {
                    let disp =
                        i32::from_le_bytes([data[i + 3], data[i + 4], data[i + 5], data[i + 6]]);
                    let site = base + i as u64;
                    let dest = (site + 7).wrapping_add(disp as i64 as u64);
                    if dest == target {
                        let reg = (data[i + 2] >> 3) & 7;
                        println!("{site:#014x}  lea r{reg}, [{target:#x}]");
                        hits += 1;
                    }
                }
            }
        }
        eprintln!("{hits} lea reference(s) to {target:#x}");
        return Ok(());
    }

    // Diagnostic: `xps5x --find-str <eboot.bin> <needle>` searches the decrypted
    // image segments for an ASCII substring and prints each match's vaddr — the
    // string can then be fed to --find-lea to locate the code that references
    // it. (Raw `strings` fails: the eboot is an encrypted SELF.)
    if let Some(pos) = args.iter().position(|a| a == "--find-str") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--find-str requires an eboot.bin path"))?;
        let needle = args
            .get(pos + 2)
            .ok_or_else(|| anyhow::anyhow!("--find-str requires a search string"))?
            .as_bytes();
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let mut hits = 0usize;
        for seg in &module.segments {
            let data = &seg.data;
            let mut i = 0usize;
            while i + needle.len() <= data.len() {
                if &data[i..i + needle.len()] == needle {
                    let vaddr = seg.vaddr + i as u64;
                    // Show a little context so the exact string is visible.
                    let end = (i + needle.len() + 24).min(data.len());
                    let ctx: String = data[i..end]
                        .iter()
                        .map(|&b| {
                            if (0x20..0x7f).contains(&b) {
                                b as char
                            } else {
                                '.'
                            }
                        })
                        .collect();
                    println!("{vaddr:#014x}  {ctx:?}");
                    hits += 1;
                    if hits >= 64 {
                        break;
                    }
                }
                i += 1;
            }
        }
        eprintln!("{hits} match(es) for {:?}", String::from_utf8_lossy(needle));
        return Ok(());
    }

    // Diagnostic: `xps5x --resolve-got <eboot.bin> <hex-got-vaddr>...` maps a
    // PLT/GOT slot vaddr (the `jmp qword [rip+disp]` target of an import thunk)
    // back to the import symbol it binds — NID, recovered name, and library.
    // A JMPREL relocation's `offset` *is* the GOT slot vaddr, so a call site's
    // `ff 25 <disp>` thunk can be turned into "which libc function is this".
    if let Some(pos) = args.iter().position(|a| a == "--resolve-got") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--resolve-got requires a path to an eboot.bin"))?;
        let bytes = std::fs::read(path)?;
        let decrypted = xps5x_firmware::decrypt_self(&bytes, &xps5x_firmware::NoKeysProvider)?;
        let module = xps5x_firmware::parse_sprx(&decrypted.elf)?;
        let dyn_tags = match &module.dynamic {
            Some(d) => xps5x_firmware::dynlib::parse_sce_dynamic(d)?,
            None => Vec::new(),
        };
        // Real PS5 titles use the standard dynamic model (vaddr-addressed
        // tables), not the `PT_SCE_DYNLIBDATA` blob — mirror `load_module`.
        let standard = xps5x_firmware::dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
        let dynlib = match &standard {
            Some((image, tags)) => xps5x_firmware::dynlib::parse_dynlibdata(image, tags)?,
            None => xps5x_firmware::dynlib::parse_dynlibdata(
                module.dynlib_data.as_deref().unwrap_or(&[]),
                &dyn_tags,
            )?,
        };
        for gs in &args[pos + 2..] {
            let Ok(target) = u64::from_str_radix(gs.trim_start_matches("0x"), 16) else {
                continue;
            };
            match dynlib.relocations.iter().find(|r| r.offset == target) {
                Some(r) => {
                    let symidx = (r.info >> 32) as usize;
                    let nid = dynlib.symbols.get(symidx).map_or(0, |s| s.nid);
                    let lib = dynlib
                        .imports
                        .iter()
                        .find(|i| i.nid == nid)
                        .and_then(|i| {
                            dynlib
                                .import_libs
                                .iter()
                                .find(|(idx, _)| *idx == i.library_index)
                        })
                        .map_or("?", |(_, n)| n.as_str());
                    println!(
                        "GOT {target:#x} -> sym#{symidx} nid={nid:#018x} {} [{lib}] rtype={}",
                        xps5x_firmware::dynlib::nid_names::describe(nid),
                        r.info & 0xffff_ffff
                    );
                }
                None => {
                    println!("GOT {target:#x} -> no exact relocation; nearest by offset:");
                    let mut near: Vec<&xps5x_firmware::dynlib::SceRela> =
                        dynlib.relocations.iter().collect();
                    near.sort_by_key(|r| r.offset.abs_diff(target));
                    for r in near.iter().take(4) {
                        let symidx = (r.info >> 32) as usize;
                        let nid = dynlib.symbols.get(symidx).map_or(0, |s| s.nid);
                        println!(
                            "    off={:#x} (Δ{:#x}) sym#{symidx} {} rtype={}",
                            r.offset,
                            r.offset.abs_diff(target),
                            xps5x_firmware::dynlib::nid_names::describe(nid),
                            r.info & 0xffff_ffff
                        );
                    }
                }
            }
        }
        println!(
            "[{} relocations total; offset range {:#x}..={:#x}]",
            dynlib.relocations.len(),
            dynlib
                .relocations
                .iter()
                .map(|r| r.offset)
                .min()
                .unwrap_or(0),
            dynlib
                .relocations
                .iter()
                .map(|r| r.offset)
                .max()
                .unwrap_or(0),
        );
        return Ok(());
    }

    // Diagnostic: `xps5x --missing-nids <eboot.bin>` loads a title exactly as
    // `--run-eboot` does but stops before executing, and prints every DISTINCT
    // import nothing resolves — encoded NID, raw NID, and library — grouped by
    // library and ranked.
    //
    // This exists because a NID is a one-way hash: "352 unresolved" is a number
    // nobody can act on, while `n88vx3C5nW8 (libScePosix)` can be brute-forced
    // back to `gettimeofday` and implemented. It is the input to sizing the
    // remaining HLE work at all.
    if let Some(pos) = args.iter().position(|a| a == "--missing-nids") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--missing-nids requires a path to an eboot.bin"))?;
        let bytes = std::fs::read(path)?;
        let hle = std::sync::Arc::new(xps5x_hle::HleRegistry::new());
        let db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        let mut registry = xps5x_firmware::ModuleRegistry::new(db);
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let process = xps5x_firmware::load_process(
            &bytes,
            dir,
            &xps5x_firmware::NoKeysProvider,
            &mut registry,
            &hle,
            xps5x_runtime::GUEST_ARENA_BASE,
        )?;

        use std::collections::BTreeMap;
        let mut by_lib: BTreeMap<&str, Vec<&xps5x_firmware::UnresolvedStub>> = BTreeMap::new();
        for s in &process.linked.unresolved_stubs {
            by_lib
                .entry(s.library.as_deref().unwrap_or("<unknown library>"))
                .or_default()
                .push(s);
        }
        let mut ranked: Vec<_> = by_lib.into_iter().collect();
        ranked.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));

        println!(
            "# {} distinct unresolved import NID(s) across {} librar(ies)",
            process.linked.unresolved_stubs.len(),
            ranked.len()
        );
        // `name` is hash-verified when known (see dynlib::nid_names), else it
        // repeats the encoded NID — an anonymous import nothing can name yet.
        println!("# encoded_nid  nid  library  name");
        for (lib, stubs) in ranked {
            let named = stubs
                .iter()
                .filter(|s| xps5x_firmware::dynlib::nid_names::name_of(s.nid).is_some())
                .count();
            println!("\n## {lib}  ({} missing, {named} named)", stubs.len());
            let mut rows: Vec<String> = stubs
                .iter()
                .map(|s| {
                    format!(
                        "{}  {:#018x}  {lib}  {}",
                        xps5x_firmware::dynlib::nid::encode_nid(s.nid),
                        s.nid,
                        xps5x_firmware::dynlib::nid_names::describe(s.nid),
                    )
                })
                .collect();
            rows.sort();
            for r in rows {
                println!("{r}");
            }
        }
        return Ok(());
    }

    // Diagnostic: `xps5x --run-eboot <eboot.bin>` drives the **real** launch
    // path headlessly — exactly what the Shell does (`load_module` at
    // `GUEST_ARENA_BASE`, then `execute_process` into the module's `_start` on a
    // genuine argc/argv/envp/auxv stack) — and reports the outcome. Same
    // `NoKeysProvider`: it never decrypts anything without a user-supplied key.
    // This exists so a real title's execution can be observed (and its logs
    // read) without standing up the GUI.
    if let Some(pos) = args.iter().position(|a| a == "--run-eboot") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--run-eboot requires a path to an eboot.bin"))?;
        let bytes = std::fs::read(path)?;
        let hle = std::sync::Arc::new(xps5x_hle::HleRegistry::new());
        let db = xps5x_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
        let mut registry = xps5x_firmware::ModuleRegistry::new(db);
        let kernel = std::sync::Arc::new(xps5x_kernel::OrbisKernel::new());
        // Load as a whole process: the eboot plus every DT_NEEDED .prx that
        // ships beside it (M1-D). A real title's imports are overwhelmingly
        // satisfied by those bundled libraries, not by HLE.
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        // `/app0` is the directory containing the selected title's eboot,
        // for every title and every host layout. Never leave the VFS on its
        // placeholder `games/current` mount during a real launch.
        kernel.filesystem.set_game_directory(dir);
        let title_dir = dir.file_name().unwrap_or_default();
        let writable_root = std::env::temp_dir().join("xps5x").join(title_dir);
        let temp_dir = writable_root.join("temp");
        let download_dir = writable_root.join("download");
        let savedata_dir = std::path::Path::new("savedata").join(title_dir);
        for writable_dir in [&temp_dir, &download_dir, &savedata_dir] {
            std::fs::create_dir_all(writable_dir)?;
        }
        kernel.filesystem.set_temp_directory(&temp_dir);
        kernel.filesystem.set_download_directory(&download_dir);
        kernel.filesystem.set_savedata_directory(&savedata_dir);
        let process = xps5x_firmware::load_process(
            &bytes,
            dir,
            &xps5x_firmware::NoKeysProvider,
            &mut registry,
            &hle,
            xps5x_runtime::GUEST_ARENA_BASE,
        )?;
        for d in &process.dependencies {
            info!(
                "  dep {} at +{:#x}: {} exports, {} unresolved",
                d.name, d.image_offset, d.exports, d.unresolved
            );
        }
        // Diagnostic: XPS5X_TRAP_CXA_THROW patches the eboot's STATICALLY-LINKED
        // __cxa_throw (which is not an import, so it can't be redirected by the
        // linker) so the C++ exception a title's worker threads throw gets
        // NAMED. The function is found by its relocation-free prologue
        // fingerprint (matched against libc.prx's copy); each hit's entry is
        // overwritten with `movabs rax, <trampoline>; jmp rax` into an appended
        // libc::__cxa_throw HLE trampoline. __cxa_throw is noreturn, so the
        // trap only reads tinfo and exits the thread — the original prologue is
        // never needed. Gated so normal runs are untouched.
        let mut linked = process.linked;
        if std::env::var_os("XPS5X_TRAP_CXA_THROW").is_some() {
            // Prologue of __cxa_throw up to its first relative call — the
            // relocation-free bytes are a reliable fingerprint (see libc.prx
            // +0x18a30): push rbp; mov rbp,rsp; push r15..rbx; push rax;
            // mov r14,rdx; mov r15,rsi; mov r12,rdi.
            const PROLOGUE: &[u8] = &[
                0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50,
                0x49, 0x89, 0xd6, 0x49, 0x89, 0xf7, 0x49, 0x89, 0xfc,
            ];
            let tramp_addr = xps5x_firmware::dynlib::linker::HLE_TRAMPOLINE_BASE
                + (linked.hle_trampolines.len() as u64) * 8;
            linked.hle_trampolines.push(xps5x_firmware::HleTrampoline {
                library: "libc".to_string(),
                function: "__cxa_throw".to_string(),
                addr: tramp_addr,
            });
            let mut patch = Vec::with_capacity(12);
            patch.extend_from_slice(&[0x48, 0xb8]); // movabs rax, imm64
            patch.extend_from_slice(&tramp_addr.to_le_bytes());
            patch.extend_from_slice(&[0xff, 0xe0]); // jmp rax
            let mut patched = 0usize;
            let mut search = 0usize;
            while let Some(rel) = linked.image[search..]
                .windows(PROLOGUE.len())
                .position(|w| w == PROLOGUE)
            {
                let at = search + rel;
                linked.image[at..at + patch.len()].copy_from_slice(&patch);
                info!(
                    "__cxa_throw trap: patched eboot __cxa_throw at guest {:#x} -> trampoline {:#x}",
                    xps5x_runtime::GUEST_ARENA_BASE + at as u64,
                    tramp_addr
                );
                patched += 1;
                search = at + patch.len();
            }
            info!(
                "__cxa_throw trap: patched {patched} internal copies, trampoline at {tramp_addr:#x}"
            );
        }
        // Diagnostic: XPS5X_TRAP_MSPACE installs a log-and-continue detour on the
        // title's native libc `sceLibcMspaceCreate` impl (`libc.prx+0xbe50`) so
        // its args + return value are logged — it returns null over valid memory
        // under our native execution while succeeding interpreted in SharpEmu.
        // The impl's prologue is `push rbp; mov rbp,rsp; push r15/r14/r13/r12/rbx`
        // = 13 relocation-free bytes, safe to relocate into the continuation stub.
        if std::env::var_os("XPS5X_TRAP_MSPACE").is_some() {
            if let Some(libc) = process
                .dependencies
                .iter()
                .find(|d| d.name.contains("libc"))
            {
                // (libc.prx offset, whole-prologue bytes, label). Create impl at
                // 0xbe50 and the sceLibcMspaceFree wrapper at 0xf600 — both start
                // with relocation-free register pushes + a reg-reg mov.
                for (off, plen, name) in [
                    (0xbe50u64, 13usize, "MspaceCreate"),
                    (0xf600, 13, "MspaceFree"),
                ] {
                    xps5x_runtime::native_trap::install(
                        &mut linked.image,
                        &mut linked.hle_trampolines,
                        libc.image_offset + off,
                        plen,
                        name,
                    );
                }
            } else {
                info!("XPS5X_TRAP_MSPACE: no libc dependency found to trap");
            }
        } else {
            // Permanent, always-on fix (not gated behind the diagnostic detour):
            // resolve the title's native `sceLibcMspaceFree` by NID from the
            // linked libc module and install the null-mspace-free guard on it, so
            // the title's `sceLibcMspaceFree(0, ptr)` returns 0 instead of the
            // retail impl faulting on the null. Title-agnostic and zero-overhead —
            // the common (non-null) free path runs the real function directly.
            const SCE_LIBC_MSPACE_FREE_NID: u64 = 0x5656_bf67_e797_971a;
            if let Some(target) = linked
                .unwind_modules
                .iter()
                .find(|m| m.name.contains("libc"))
                .and_then(|m| {
                    m.exports
                        .iter()
                        .find(|e| e.nid == SCE_LIBC_MSPACE_FREE_NID)
                        .map(|e| m.image_offset + e.value)
                })
            {
                xps5x_runtime::native_trap::install_null_free_guard(
                    &mut linked.image,
                    target,
                    13,
                    "sceLibcMspaceFree",
                );
            }
        }
        // Diagnostic: XPS5X_TRAP_MODULE_EXPORTS=<substring> plants a one-shot
        // `int3` on the entry byte of EVERY export of each matching loaded
        // module. The first call to each export logs module + NID + caller and
        // restores the byte permanently — near-zero overhead after the hit.
        // Answers "is this module ever actually entered, and from where"
        // (e.g. `=cohtml` to see whether the title drives its HTML engine).
        if let Ok(filter) = std::env::var("XPS5X_TRAP_MODULE_EXPORTS") {
            struct TrapTarget {
                name: String,
                image_offset: u64,
                exports: Vec<(u64, u64)>,
                exec_range: Option<(u64, u64)>,
            }
            let targets: Vec<TrapTarget> = linked
                .unwind_modules
                .iter()
                .filter(|m| xps5x_runtime::export_trap::module_matches(&filter, &m.name))
                .map(|m| TrapTarget {
                    name: m.name.clone(),
                    image_offset: m.image_offset,
                    exports: m.exports.iter().map(|e| (e.nid, e.value)).collect(),
                    // First PT_LOAD = the text segment; exports outside it are
                    // data and must not be patched (see export_trap docs).
                    exec_range: (m.unwind.seg0_size > 0).then(|| {
                        (
                            m.unwind.seg0_vaddr,
                            m.unwind.seg0_vaddr + m.unwind.seg0_size,
                        )
                    }),
                })
                .collect();
            if targets.is_empty() {
                tracing::warn!(
                    "XPS5X_TRAP_MODULE_EXPORTS={filter}: no loaded module matches — nothing armed"
                );
            }
            for t in targets {
                xps5x_runtime::export_trap::install_module_exports(
                    &mut linked.image,
                    xps5x_runtime::GUEST_ARENA_BASE,
                    &t.name,
                    t.image_offset,
                    &t.exports,
                    t.exec_range,
                );
            }
        }
        // Diagnostic: XPS5X_TRAP_ADDR=<hex>[,<hex>...] plants a one-shot int3 at
        // each eboot-relative address — an RE probe for "does this code ever
        // execute", logging the caller when hit. Used to confirm whether a code
        // path (e.g. Minecraft's per-screen view-create loop) ever runs.
        if let Ok(list) = std::env::var("XPS5X_TRAP_ADDR") {
            let addrs: Vec<u64> = list
                .split(',')
                .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .collect();
            xps5x_runtime::export_trap::install_addr_traps(
                &mut linked.image,
                xps5x_runtime::GUEST_ARENA_BASE,
                &addrs,
            );
        }
        let linked = std::sync::Arc::new(linked);
        info!(
            "loaded: entry={:#x} image={:#x} byte(s) resolved={} unresolved={}",
            linked.entry,
            linked.image.len(),
            linked.hle_trampolines.len(),
            linked.unresolved.len()
        );
        // The module's thread-local template. `tdata` is the part that must be
        // *copied* into every thread's block; the rest is `.tbss` and is zero.
        // A title whose thread-locals silently read as zero is the shape of a
        // whole class of null-dereference bugs, so say what the template is.
        match &linked.tls {
            Some(tls) => info!(
                "  PT_TLS: vaddr={:#x} tdata={:#x} memsz={:#x} align={:#x} init={:02x?}",
                tls.vaddr,
                tls.data.len(),
                tls.mem_size,
                tls.align,
                &tls.data[..tls.data.len().min(32)]
            ),
            None => info!("  PT_TLS: none"),
        }
        // Stall diagnosis: with XPS5X_STALL_DUMP set (and XPS5X_TRACE_EINVAL to
        // populate the ring), periodically log every guest thread's most recent
        // HLE calls. A thread blocked at a boot gate shows its last call frozen
        // or a tight spin — naming exactly what the game is waiting on.
        if std::env::var_os("XPS5X_STALL_DUMP").is_some() {
            let kmon = std::sync::Arc::clone(&kernel);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(6));
                    let mut lines: Vec<String> = kmon
                        .recent_hle_calls
                        .iter()
                        .map(|entry| {
                            let tid = *entry.key();
                            let name = kmon
                                .thread_names
                                .get(&tid)
                                .map_or_else(String::new, |n| n.clone());
                            let ring = entry.value().lock();
                            let recent: Vec<String> = ring.iter().rev().take(5).cloned().collect();
                            format!("t{tid}({name}): {}", recent.join(" <- "))
                        })
                        .collect();
                    lines.sort();
                    // The call ring goes blank when a thread spins in GUEST code
                    // (it calls nothing). RIP is the only thing that still says
                    // where it is — resolve it against the loaded modules so it
                    // reads as module+offset, which `--dump-vaddr` can decode.
                    let mut rips: Vec<String> = xps5x_runtime::sample_guest_rips(&kmon)
                        .into_iter()
                        .map(|(id, rip)| {
                            let site = kmon.unwind_module_for_addr(rip).map_or_else(
                                || format!("{rip:#x}"),
                                |m| format!("{}+{:#x}", m.name, rip - m.start),
                            );
                            format!("t{id}@{site}")
                        })
                        .collect();
                    rips.sort();
                    // With XPS5X_TIME_HLE, name where each thread's wall-clock
                    // actually went. A thread whose top entry accounts for most
                    // of the run is parked in that one call — that is the thing
                    // to fix, and it is invisible in the call ring above.
                    // Per THREAD, not globally: every idle worker parks ~the
                    // whole run in scePthreadCondWait, so a global top-N is 12
                    // rows of "idle" and buries the one thread that matters.
                    // Report each thread's own biggest sink plus its total, so a
                    // busy thread (many short calls) is distinguishable at a
                    // glance from a parked one (all its time in a single wait).
                    let mut per_thread: std::collections::HashMap<u64, (u128, u128, u64, String)> =
                        std::collections::HashMap::new();
                    for e in kmon.hle_call_time.iter() {
                        let ((tid, func), (calls, micros)) = (e.key().clone(), *e.value());
                        let slot = per_thread.entry(tid).or_insert((0, 0, 0, String::new()));
                        slot.0 += micros; // total across all calls
                        if micros > slot.1 {
                            *slot = (slot.0, micros, calls, func);
                        }
                    }
                    let mut spent: Vec<(u128, String)> = per_thread
                        .into_iter()
                        .map(|(tid, (total, top_us, calls, func))| {
                            let name = kmon
                                .thread_names
                                .get(&tid)
                                .map_or_else(String::new, |n| n.clone());
                            (
                                total,
                                format!(
                                    "t{tid}({name}) total {:.1}s | top {func}: {:.1}s over {calls}",
                                    total as f64 / 1e6,
                                    top_us as f64 / 1e6
                                ),
                            )
                        })
                        .collect();
                    spent.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
                    let top: Vec<String> = spent.into_iter().map(|(_, s)| s).collect();
                    // The HLE call each thread is CURRENTLY inside (empty = not in
                    // one → blocked in guest code or runtime infra). Names exactly
                    // which blocking call a frozen thread never returns from.
                    let mut inflight: Vec<String> = kmon
                        .in_flight_hle
                        .iter()
                        .map(|e| format!("t{}={}", e.key(), e.value()))
                        .collect();
                    inflight.sort();
                    // Shallow host backtrace per thread (module+offset), so a
                    // thread parked in a host wait OUTSIDE any HLE call is shown
                    // with the call chain through our code that reached it.
                    let mut bt: Vec<String> = xps5x_runtime::sample_host_backtraces(&kmon)
                        .into_iter()
                        .map(|(id, chain)| format!("t{id}: {chain}"))
                        .collect();
                    bt.sort();
                    // The title's OWN log output (its `write`s to fd 1/2) is the
                    // single most informative thing during a stall — it says what
                    // the game thinks it is doing. It is otherwise only printed
                    // when the run ends normally, which a hung title never does,
                    // so surface the tail here.
                    let console = kmon.console.contents();
                    let console_tail = if console.is_empty() {
                        "<empty>".to_owned()
                    } else {
                        let tail: String = console
                            .lines()
                            .rev()
                            .take(25)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("({} bytes, last 25 lines)\n{tail}", console.len())
                    };
                    info!(
                        "STALL_DUMP ({} threads):\n{}\nIN-FLIGHT HLE: {}\nHOST BACKTRACES:\n{}\nRIPs: {}{}\nGUEST CONSOLE: {}",
                        lines.len(),
                        lines.join("\n"),
                        if inflight.is_empty() {
                            "<none — all threads between calls>".to_owned()
                        } else {
                            inflight.join("  ")
                        },
                        bt.join("\n"),
                        rips.join(" "),
                        if top.is_empty() {
                            String::new()
                        } else {
                            format!("\nTIME IN HLE (top):\n{}", top.join("\n"))
                        },
                        console_tail
                    );
                }
            });
        }
        // Poll-gate diagnosis: with XPS5X_CALL_STATS set, the dispatch path
        // counts every HLE call per function, split into a boot window (first
        // 30 s) and steady state. Dump the top of each ranking periodically —
        // a title that polls a "not ready" value in short cycles puts the
        // polled function at the top of the STEADY window (timing/allocator
        // noise like clock_gettime ranks high too; the status queries below
        // them are the gate). Periodic, not at-exit: diagnosis runs are
        // usually killed hard (timeout -s KILL), so an exit hook never fires.
        if std::env::var_os("XPS5X_CALL_STATS").is_some() {
            let kmon = std::sync::Arc::clone(&kernel);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(10));
                    let mut boot: Vec<(u64, String)> = Vec::new();
                    let mut steady: Vec<(u64, String)> = Vec::new();
                    for e in kmon.hle_call_counts.iter() {
                        let (b, s) = e.value();
                        let b = b.load(std::sync::atomic::Ordering::Relaxed);
                        let s = s.load(std::sync::atomic::Ordering::Relaxed);
                        if b > 0 {
                            boot.push((b, e.key().clone()));
                        }
                        if s > 0 {
                            steady.push((s, e.key().clone()));
                        }
                    }
                    boot.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
                    steady.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
                    let render = |v: &[(u64, String)]| {
                        v.iter()
                            .take(40)
                            .map(|(n, f)| format!("  {n:>9}  {f}"))
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    info!(
                        "CALL_STATS t=+{:.0}s\nBOOT WINDOW (first 30s, {} distinct) top 40:\n{}\nSTEADY STATE (after 30s, {} distinct) top 40:\n{}",
                        kmon.uptime().as_secs_f64(),
                        boot.len(),
                        render(&boot),
                        steady.len(),
                        render(&steady),
                    );
                }
            });
        }
        // Same boot-splash staging as the Shell launcher, so `--run-eboot`
        // and the Shell remain one launch path observably.
        splash::stage_boot_splash(std::path::Path::new(&path));

        info!("entering guest _start via execute_process ...");
        let outcome = xps5x_runtime::execute_process_shared(
            std::sync::Arc::clone(&linked),
            std::sync::Arc::clone(&hle),
            std::sync::Arc::clone(&kernel),
            &[path.as_str()],
            &[],
        );
        match &outcome {
            Ok(o) => info!("RESULT: {o:?}"),
            // The whole point of the per-NID unresolved stub: say WHICH import
            // the guest wanted. Report it as a worklist item, not an address.
            Err(xps5x_runtime::RuntimeError::UnimplementedImport {
                nid,
                library,
                stub_addr,
                rip,
            }) => {
                let library = library.as_deref().unwrap_or("<unknown library>");
                info!(
                    "RESULT: guest needs an UNIMPLEMENTED import — {} \
                     — nid {nid:#018x} (encoded {}) from library '{library}'",
                    xps5x_firmware::dynlib::nid_names::describe(*nid),
                    xps5x_firmware::dynlib::nid::encode_nid(*nid)
                );
                info!("        guest rip {rip:#x}; its stub is {stub_addr:#x}");
                info!("        implement it, or supply the module that exports it, and re-run");
            }
            // A fault reports where the guest *was*, which on its own names
            // nothing: 250 MB into a stripped C++ binary, one address looks like
            // any other. The bytes there are the difference between a lead and a
            // dead end — they say whether the guest was loading a vtable,
            // calling through a pointer, or reading a thread-local. The image is
            // right here in the loader, so this costs nothing and never touches
            // the fault path itself.
            Err(xps5x_runtime::RuntimeError::Faulted { addr, access, kind }) => {
                info!("RESULT: guest fault at {addr:#x} ({kind} {access:#x})");
                report_fault_site(&linked, &process.dependencies, *addr);
            }
            Err(e) => info!("RESULT: {e:?}"),
        }
        let console = kernel.console.contents();
        if console.is_empty() {
            info!("guest console: <empty>");
        } else {
            info!("guest console ({} byte(s)):\n{console}", console.len());
        }
        return Ok(());
    }

    info!("╔══════════════════════════════════════════════╗");
    info!(
        "║          XPS5X — PS5 Emulator v{}        ║",
        xps5x_core::VERSION
    );
    info!("║        Cross-Platform Compatibility Layer     ║");
    info!("╚══════════════════════════════════════════════╝");

    // Load configuration.
    let config_path = std::path::Path::new("config.toml");
    let config = xps5x_core::config::EmulatorConfig::load(config_path)?;
    info!("Configuration loaded from {}", config_path.display());

    // Initialize the kernel.
    let _kernel = xps5x_kernel::OrbisKernel::new();
    info!("Orbis kernel HLE initialized");

    // Initialize the HLE registry.
    let _hle = xps5x_hle::HleRegistry::new();
    info!("HLE library registry initialized");

    // Launch the GUI.
    info!("Launching XPS5X GUI...");

    // The Shell is a full-screen, PS5-style console experience by default
    // (spec §7): borderless fullscreen, sized by the OS to the active
    // monitor. Forcing an inner_size alongside fullscreen used to strand a
    // 1920x1080 window in the corner of larger displays, so the configured
    // window size only applies when `general.fullscreen = false` opts into
    // a normal desktop window.
    let viewport = egui::ViewportBuilder::default()
        .with_title("XPS5X")
        .with_min_inner_size([800.0, 600.0]);
    let viewport = if config.general.fullscreen {
        viewport.with_fullscreen(true).with_decorations(false)
    } else {
        viewport.with_inner_size([
            config.general.window_width as f32,
            config.general.window_height as f32,
        ])
    };
    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "XPS5X",
        native_options,
        Box::new(|cc| {
            // Set dark theme.
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app::XPS5XApp::new(
                &cc.egui_ctx,
                config,
                config_path.to_path_buf(),
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;

    info!("XPS5X shutting down");
    Ok(())
}
