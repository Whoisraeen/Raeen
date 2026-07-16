use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: guest_disasm <decrypted-self> <vaddr>...");
    let bytes = std::fs::read(path).expect("read input");
    let decrypted =
        xps5x_firmware::crypto::decrypt_self(&bytes, &xps5x_firmware::crypto::NoKeysProvider)
            .expect("decrypted SELF passthrough");
    let module = xps5x_firmware::parse_sprx(&decrypted.elf).expect("parse inner ELF");

    for value in args {
        if let Some(value) = value.strip_prefix("e:") {
            let target =
                u64::from_str_radix(value.trim_start_matches("0x"), 16).expect("hex export NID");
            let tags = module
                .dynamic
                .as_deref()
                .map(xps5x_firmware::dynlib::parse_sce_dynamic)
                .transpose()
                .expect("parse dynamic tags")
                .unwrap_or_default();
            let standard = xps5x_firmware::dynlib::standard_dynamic_view(&module.segments, &tags);
            let dynlib = match &standard {
                Some((image, tags)) => xps5x_firmware::dynlib::parse_dynlibdata(image, tags),
                None => xps5x_firmware::dynlib::parse_dynlibdata(
                    module.dynlib_data.as_deref().unwrap_or(&[]),
                    &tags,
                ),
            }
            .expect("parse dynlib");
            let export = dynlib
                .exports
                .iter()
                .find(|export| export.nid == target)
                .unwrap_or_else(|| panic!("no export for NID {target:#x}"));
            println!(
                "\n=== export {target:#018x} ===\nname={} value={:#x}",
                xps5x_firmware::dynlib::nid_names::describe(target),
                export.value
            );
            continue;
        }
        if let Some(value) = value.strip_prefix("r:") {
            let target =
                u64::from_str_radix(value.trim_start_matches("0x"), 16).expect("hex reloc vaddr");
            let tags = module
                .dynamic
                .as_deref()
                .map(xps5x_firmware::dynlib::parse_sce_dynamic)
                .transpose()
                .expect("parse dynamic tags")
                .unwrap_or_default();
            let standard = xps5x_firmware::dynlib::standard_dynamic_view(&module.segments, &tags);
            let dynlib = match &standard {
                Some((image, tags)) => xps5x_firmware::dynlib::parse_dynlibdata(image, tags),
                None => xps5x_firmware::dynlib::parse_dynlibdata(
                    module.dynlib_data.as_deref().unwrap_or(&[]),
                    &tags,
                ),
            }
            .expect("parse dynlib");
            let reloc = dynlib
                .relocations
                .iter()
                .find(|reloc| reloc.offset == target)
                .unwrap_or_else(|| panic!("no relocation at {target:#x}"));
            let symbol_index = (reloc.info >> 32) as usize;
            let symbol = dynlib
                .symbols
                .get(symbol_index)
                .expect("relocation symbol index");
            let import = dynlib
                .imports
                .iter()
                .find(|import| import.nid == symbol.nid);
            let library = import.and_then(|import| {
                dynlib
                    .import_libs
                    .iter()
                    .find(|(index, _)| *index == import.library_index)
                    .map(|(_, name)| name.as_str())
            });
            println!(
                "\n=== relocation {target:#x} ===\ntype={} symbol={} nid={:#018x} name={} library={}",
                reloc.info as u32,
                symbol_index,
                symbol.nid,
                xps5x_firmware::dynlib::nid_names::describe(symbol.nid),
                library.unwrap_or("<unknown>")
            );
            continue;
        }
        let (value, before, after) = value
            .strip_prefix("l:")
            .map_or((value.as_str(), 64, 96), |value| (value, 2048, 256));
        let target = u64::from_str_radix(value.trim_start_matches("0x"), 16).expect("hex vaddr");
        let segment = module
            .segments
            .iter()
            .find(|segment| {
                target >= segment.vaddr && target < segment.vaddr + segment.data.len() as u64
            })
            .unwrap_or_else(|| panic!("vaddr {target:#x} is outside file-backed segments"));
        let start = target.saturating_sub(before).max(segment.vaddr);
        let end = (target + after).min(segment.vaddr + segment.data.len() as u64);
        let offset = (start - segment.vaddr) as usize;
        let code = &segment.data[offset..offset + (end - start) as usize];
        let mut decoder = Decoder::with_ip(64, code, start, DecoderOptions::NONE);
        let mut formatter = IntelFormatter::new();
        let mut rendered = String::new();

        println!("\n=== {target:#x} ===");
        while decoder.can_decode() {
            let instruction = decoder.decode();
            rendered.clear();
            formatter.format(&instruction, &mut rendered);
            let marker = if instruction.ip() <= target && target < instruction.next_ip() {
                ">"
            } else {
                " "
            };
            println!("{marker} {:012x}  {rendered}", instruction.ip());
        }
    }
}
