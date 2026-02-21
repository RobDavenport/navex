//! Steering behavior benchmarks.

use criterion::{criterion_group, criterion_main, Criterion};

fn steering_benchmarks(_c: &mut Criterion) {
    // TODO: Add benchmarks
}

criterion_group!(benches, steering_benchmarks);
criterion_main!(benches);
