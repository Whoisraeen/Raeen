//! Convert an `RAEEN_DUMP_FRAMES` P6 PPM frame dump to PNG for inspection.
//!
//! ```text
//! cargo run --release --example ppm2png -- scratch/mc-frames/run2/frame_000003.ppm
//! ```

use std::io::Read;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ppm2png <in.ppm> — writes <in>.png beside it");
    let mut f = std::fs::File::open(&path).expect("open ppm");
    let mut magic = [0u8; 2];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"P6", "not a P6 ppm");
    let mut read_token = || -> String {
        let mut t = String::new();
        loop {
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            if b[0] == b'#' {
                // Header comment: skip to end of line.
                loop {
                    let mut c = [0u8; 1];
                    f.read_exact(&mut c).unwrap();
                    if c[0] == b'\n' {
                        break;
                    }
                }
                continue;
            }
            if b[0].is_ascii_whitespace() {
                if !t.is_empty() {
                    break;
                }
                continue;
            }
            t.push(b[0] as char);
        }
        t
    };
    let w: u32 = read_token().parse().unwrap();
    let h: u32 = read_token().parse().unwrap();
    let max: u32 = read_token().parse().unwrap();
    assert_eq!(max, 255);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    f.read_exact(&mut rgb).unwrap();
    let out = path.replace(".ppm", ".png");
    image::save_buffer(&out, &rgb, w, h, image::ColorType::Rgb8).unwrap();
    println!("{out} ({w}x{h})");
}
