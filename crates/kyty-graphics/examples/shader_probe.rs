//! Parse a raw guest shader dump and report the first unsupported instruction.

use std::env;
use std::fs;
use std::path::Path;

use kyty_graphics::shader::{ShaderCode, ShaderType, shader_parse};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: shader_probe <file> <vs|ps|cs|fetch>")?;
    let stage = args
        .next()
        .ok_or("usage: shader_probe <file> <vs|ps|cs|fetch>")?;
    if args.next().is_some() {
        return Err("usage: shader_probe <file> <vs|ps|cs|fetch>".into());
    }
    let stage = stage.to_string_lossy();
    let shader_type = match stage.as_ref() {
        "vs" => ShaderType::Vertex,
        "ps" => ShaderType::Pixel,
        "cs" => ShaderType::Compute,
        "fetch" => ShaderType::Fetch,
        _ => return Err(format!("unknown shader stage: {stage}").into()),
    };

    let bytes = fs::read(Path::new(&path))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("shader length {} is not dword-aligned", bytes.len()).into());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
        .collect();
    let mut code = ShaderCode::new();
    code.set_type(shader_type);
    match shader_parse(0, &words, &mut code, true) {
        Ok(consumed) => {
            println!(
                "parsed {} instructions, consumed {consumed} dwords",
                code.get_instructions().len()
            );
            for (index, instruction) in code.get_instructions().iter().enumerate() {
                println!("{index:04}: {instruction:?}");
            }
            Ok(())
        }
        Err(error) => {
            println!(
                "decoded {} instructions before failure: {error}",
                code.get_instructions().len()
            );
            Err(error.into())
        }
    }
}
