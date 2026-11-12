//! The bind-group cache (`context/binds.rs`): reuse with each dispatch's own parameters, invalidation when the
//! pool lets a buffer go, and its size bound.

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use super::gpu;
use crate::context::binds::{BindCache, GENERATION};
use crate::pipeline::{Access, Kernel, KernelDef, Params, Source, Spec};
use crate::pool::{BufferPool, Retired};
use crate::{DType, GpuContext, GpuTensor};

/// `X[i] += pf(1)` for `i < pu(0)`.
static ADD_PARAM: KernelDef = KernelDef {
    name: "test_add_param",
    storage: &[Access::ReadWrite],
    source: Source::Static(concat!(
        "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n",
        "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n",
        "    if (g.x < pu(0u)) { X[g.x] += pf(1u); }\n",
        "}\n",
    )),
};

/// A context of its own, so other tests' dispatches do not move its counters.
fn private() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("note: skipping, no second context: {e}");
            None
        }
    }
}

/// Dispatches on the same buffer share one bind group, but each reads its own parameters (the dynamic offset),
/// also across submissions and, with a long flush interval, across uniform chunks.
#[test]
fn reused_bind_groups_read_their_own_parameters() {
    let _shared = gpu!();
    let Some(ctx) = private() else { return };
    let k = ctx.kernel(&ADD_PARAM, Spec::new()).unwrap();
    let a = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    let b = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    ctx.sync().unwrap();
    let before = ctx.work_stats().bind_groups;
    for i in 1..=300u32 {
        ctx.dispatch(&k, &[a.raw()], &Params::new().u(100).f(i as f32), [2, 1, 1]).unwrap();
    }
    assert_eq!(a.to_vec().unwrap(), vec![(300 * 301 / 2) as f32; 100]);
    assert_eq!(ctx.work_stats().bind_groups - before, 1, "300 dispatches on one buffer, one uniform chunk");
    ctx.dispatch(&k, &[b.raw()], &Params::new().u(50).f(2.5), [1, 1, 1]).unwrap();
    let got = b.to_vec().unwrap();
    assert_eq!((got[49], got[50]), (2.5, 0.0));
    assert_eq!(ctx.work_stats().bind_groups - before, 2, "another buffer, another bind group");
    // A batch of 1024 dispatches fills four uniform chunks: one bind group per chunk, each slot its own value. The
    // batch limit doubles up to the flush interval after a wait (64 + 128 + ... + 1024 at most), so 3100
    // dispatches hold at least one full batch.
    ctx.set_flush_interval(1024);
    let c = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    ctx.sync().unwrap();
    let before = ctx.work_stats().bind_groups;
    for _ in 0..3100 {
        ctx.dispatch(&k, &[c.raw()], &Params::new().u(64).f(0.5), [1, 1, 1]).unwrap();
    }
    assert_eq!(c.to_vec().unwrap(), vec![1550.0; 64]);
    assert_eq!(ctx.work_stats().bind_groups - before, 4);
}

/// A buffer the pool drops (here by `trim_pool`) leaves the cache with every bind group that holds it, and so
/// does everything on `release_memory`.
#[test]
fn dropped_buffers_leave_the_bind_cache() {
    let _shared = gpu!();
    let Some(ctx) = private() else { return };
    let k = ctx.kernel(&ADD_PARAM, Spec::new()).unwrap();
    let keep = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    let gone = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    for t in [&keep, &gone] {
        ctx.dispatch(&k, &[t.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
    }
    ctx.sync().unwrap();
    assert_eq!(ctx.bind_cache_len(), 2);
    drop(gone);
    assert_eq!(ctx.bind_cache_len(), 2, "back in the pool's free list: still the pool's, still cached");
    ctx.trim_pool();
    assert_eq!(ctx.bind_cache_len(), 1, "the pool let it go");
    ctx.dispatch(&k, &[keep.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
    assert_eq!(keep.to_vec().unwrap(), vec![2.0; 64]);
    ctx.release_memory().unwrap();
    assert_eq!(ctx.bind_cache_len(), 0);
    ctx.dispatch(&k, &[keep.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
    assert_eq!(keep.to_vec().unwrap(), vec![3.0; 64], "a new uniform chunk and bind group after the release");
}

/// The pool reports the buffers it evicts or clears, and gives up on ids (`All`) past its bound.
#[test]
fn pool_reports_the_buffers_it_lets_go() {
    let ctx = gpu!();
    let usage = wgpu::BufferUsages::STORAGE;
    let mut pool = BufferPool::new(1024);
    let flag = pool.retired_flag();
    let a = pool.take(ctx.wgpu_device(), 1024, usage);
    let id_a = crate::context::buffer_id(&a);
    pool.give(a, 1024);
    assert!(matches!(pool.take_retired(), Retired::None));
    assert!(!flag.load(Ordering::Relaxed));
    // A 2 KiB buffer freed while the 1 KiB one is cached evicts it (cap and high-water mark: 2 KiB).
    let x = pool.take(ctx.wgpu_device(), 2048, usage);
    let id_x = crate::context::buffer_id(&x);
    pool.give(x, 2048);
    assert!(flag.load(Ordering::Relaxed));
    match pool.take_retired() {
        Retired::Some(ids) => assert_eq!(ids, HashSet::from([id_a])),
        other => panic!("{other:?}"),
    }
    assert!(!flag.load(Ordering::Relaxed));
    pool.clear();
    match pool.take_retired() {
        Retired::Some(ids) => assert_eq!(ids, HashSet::from([id_x])),
        other => panic!("{other:?}"),
    }
    let mut big = BufferPool::new(1 << 30);
    let held: Vec<_> = (0..5000).map(|_| big.take(ctx.wgpu_device(), 256, usage)).collect();
    for x in held {
        big.give(x, 256);
    }
    big.clear();
    assert!(matches!(big.take_retired(), Retired::All));
}

/// The cache holds at most two generations, and an entry used again moves to the current one.
#[test]
fn bind_cache_is_bounded() {
    static THREE: KernelDef = KernelDef {
        name: "test_three_buffers",
        storage: &[Access::Read, Access::Read, Access::ReadWrite],
        source: Source::Static(concat!(
            "@group(0) @binding(1) var<storage, read> A: array<f32>;\n",
            "@group(0) @binding(2) var<storage, read> B: array<f32>;\n",
            "@group(0) @binding(3) var<storage, read_write> C: array<f32>;\n",
            "@compute @workgroup_size(1) fn main() { C[0] = A[0] + B[0] + pf(0u); }\n",
        )),
    };
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let k = Kernel::compile(dev, &THREE, &Spec::new()).unwrap();
    let mk = |usage| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage,
            mapped_at_creation: false,
        })
    };
    let uniform = mk(wgpu::BufferUsages::UNIFORM);
    let bufs: Vec<_> = (0..24).map(|_| mk(wgpu::BufferUsages::STORAGE)).collect();
    let mut cache = BindCache::default();
    let mut created = 0;
    let mut get = |cache: &mut BindCache, i: usize| {
        let (x, y, z) = (&bufs[i % 24], &bufs[i / 24 % 24], &bufs[i / 576 % 24]);
        let mut made = false;
        cache.get(dev, &k.layout, &uniform, &[x, y, z], &mut made);
        created += usize::from(made);
    };
    let hot = 0;
    for i in 0..3 * GENERATION {
        get(&mut cache, i + 1);
        if i % 1000 == 0 {
            get(&mut cache, hot);
        }
        assert!(cache.len() <= 2 * GENERATION);
    }
    drop(get);
    assert_eq!(created, 3 * GENERATION + 1, "every combination once, the hot one once");
}
