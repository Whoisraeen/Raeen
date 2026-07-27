//! NID hot-path benchmarks: hashing a symbol name to its NID, encoding it,
//! and resolving it through a registry-sized lookup table. These are the
//! per-relocation costs of linking — the measured retail title carries
//! ~719k relocations, so nanoseconds here are milliseconds on the loading
//! screen. Run with `cargo bench -p raeen-firmware`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use raeen_firmware::dynlib::nid::{NidDatabase, encode_nid, nid_of};

/// A registry-shaped corpus: enough distinct names to make the lookup table
/// realistic (the real HLE registry is in the same order of magnitude).
fn corpus() -> Vec<(String, String)> {
    (0..4_000)
        .map(|i| (format!("libSceBench{}", i % 40), format!("sceBenchFn{i}")))
        .collect()
}

fn bench_nid_of(c: &mut Criterion) {
    c.bench_function("nid_of (name -> NID hash)", |b| {
        b.iter(|| nid_of(black_box("sceKernelLoadStartModule")))
    });
}

fn bench_encode(c: &mut Criterion) {
    let nid = nid_of("sceKernelLoadStartModule");
    c.bench_function("encode_nid (NID -> import string)", |b| {
        b.iter(|| encode_nid(black_box(nid)))
    });
}

fn bench_resolve(c: &mut Criterion) {
    let table = NidDatabase::from_hle_names(corpus());
    let hit = nid_of("sceBenchFn1234");
    let miss = nid_of("sceDoesNotExist");
    c.bench_function("NidTable::resolve hit", |b| {
        b.iter(|| table.resolve(black_box(hit)))
    });
    c.bench_function("NidTable::resolve miss", |b| {
        b.iter(|| table.resolve(black_box(miss)))
    });
}

criterion_group!(benches, bench_nid_of, bench_encode, bench_resolve);
criterion_main!(benches);
