use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("missing eboot path")?;
    let selector = args.get(2).ok_or("missing hex offset")?;
    let bytes = std::fs::read(path)?;
    let hle = raeen_hle::HleRegistry::new();
    let db = raeen_firmware::dynlib::nid::NidDatabase::from_hle(&hle);
    let mut registry = raeen_firmware::ModuleRegistry::new(db);
    let dir = Path::new(path).parent().unwrap_or(Path::new("."));
    let process = raeen_firmware::load_process(
        &bytes,
        dir,
        &raeen_firmware::NoKeysProvider,
        &mut registry,
        &hle,
        0x1000_0000_0000,
    )?;
    if let Some(target) = selector.strip_prefix("ref=") {
        let target = u64::from_str_radix(target.trim_start_matches("0x"), 16)?;
        for offset in 0..process.linked.image.len().saturating_sub(4) {
            let encoded = i32::from_le_bytes(
                process.linked.image[offset..offset + 4]
                    .try_into()
                    .expect("four-byte window"),
            ) as i64;
            for instruction_end in offset + 4..=offset + 12 {
                if instruction_end as i64 + encoded == target as i64 {
                    println!(
                        "candidate disp32 at {offset:#x}, instruction end {instruction_end:#x}"
                    );
                }
            }
        }
        return Ok(());
    }
    if let Some(pattern) = selector.strip_prefix("pat=") {
        let pattern = pattern.replace('_', "");
        let bytes = (0..pattern.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&pattern[index..index + 2], 16))
            .collect::<Result<Vec<_>, _>>()?;
        let start = args
            .get(3)
            .map(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16))
            .transpose()?
            .unwrap_or(0) as usize;
        let end = args
            .get(4)
            .map(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16))
            .transpose()?
            .unwrap_or(process.linked.image.len() as u64) as usize;
        for offset in start
            ..=end
                .min(process.linked.image.len())
                .saturating_sub(bytes.len())
        {
            if process.linked.image[offset..].starts_with(&bytes) {
                println!("pattern at {offset:#x}");
            }
        }
        return Ok(());
    }
    let offset = u64::from_str_radix(selector.trim_start_matches("0x"), 16)? as usize;
    let length = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(128);
    let end = offset.checked_add(length).ok_or("range overflow")?;
    let code = process
        .linked
        .image
        .get(offset..end)
        .ok_or("range outside linked image")?;
    for (line, chunk) in code.chunks(16).enumerate() {
        print!("{:#010x}:", offset + line * 16);
        for byte in chunk {
            print!(" {byte:02x}");
        }
        println!();
    }
    Ok(())
}
