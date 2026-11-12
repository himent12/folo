//! The bind-group cache (`context/binds.rs`): reuse with each dispatch's own parameters, second-sight admission,
//! only pool-owned buffers cached, invalidation when the pool lets a buffer go, and its size and index bounds.

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

/// `C[0] = A[0] + B[0] + pf(0)`.
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

fn buffer(dev: &wgpu::Device, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage, mapped_at_creation: false })
}

/// Dispatches on the same buffer share one bind group, but each reads its own parameters (the dynamic offset),
/// also across submissions and, with a long flush interval, across uniform chunks. A combination's first dispatch
/// creates a bind group without caching it, its second creates and caches one, later ones reuse it.
#[test]
fn reused_bind_groups_read_their_own_parameters() {
    let _shared = gpu!();
    let Some(ctx) = private() else { return };
    let k = ctx.kernel(&ADD_PARAM, Spec::new()).unwrap();
    let a = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    let b = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    ctx.sync().unwrap();
    let before = ctx.work_stats().bind_groups;
    ctx.dispatch(&k, &[a.raw()], &Params::new().u(100).f(1.0), [2, 1, 1]).unwrap();
    assert_eq!((ctx.work_stats().bind_groups - before, ctx.bind_cache_len()), (1, 0), "first sighting");
    for i in 2..=300u32 {
        ctx.dispatch(&k, &[a.raw()], &Params::new().u(100).f(i as f32), [2, 1, 1]).unwrap();
    }
    assert_eq!(a.to_vec().unwrap(), vec![(300 * 301 / 2) as f32; 100]);
    assert_eq!(ctx.work_stats().bind_groups - before, 2, "300 dispatches on one buffer, one uniform chunk");
    assert_eq!(ctx.bind_cache_len(), 1);
    ctx.dispatch(&k, &[b.raw()], &Params::new().u(50).f(2.5), [1, 1, 1]).unwrap();
    let got = b.to_vec().unwrap();
    assert_eq!((got[49], got[50]), (2.5, 0.0));
    assert_eq!(ctx.work_stats().bind_groups - before, 3, "another buffer, another bind group");
    assert_eq!(ctx.bind_cache_len(), 1, "seen once: not cached");
    // A batch of 1024 dispatches fills four uniform chunks: two bind groups per chunk (first and second sighting),
    // each slot its own value, so a bind group of the wrong chunk would add another dispatch's value. The batch
    // limit doubles up to the flush interval after a wait (64 + 128 + ... + 1024 at most), so 3100 dispatches hold
    // at least one full batch. The sum stays below 2^24, exact in f32.
    ctx.set_flush_interval(1024);
    let c = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    ctx.sync().unwrap();
    let before = ctx.work_stats().bind_groups;
    for i in 1..=3100u32 {
        ctx.dispatch(&k, &[c.raw()], &Params::new().u(64).f(i as f32), [1, 1, 1]).unwrap();
    }
    assert_eq!(c.to_vec().unwrap(), vec![(3100 * 3101 / 2) as f32; 64]);
    assert_eq!(ctx.work_stats().bind_groups - before, 8);
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
    for t in [&keep, &gone, &keep, &gone] {
        ctx.dispatch(&k, &[t.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
    }
    ctx.sync().unwrap();
    assert_eq!(ctx.bind_cache_len(), 2);
    drop(gone);
    assert_eq!(ctx.bind_cache_len(), 2, "back in the pool's free list: still the pool's, still cached");
    ctx.trim_pool();
    assert_eq!(ctx.bind_cache_len(), 1, "the pool let it go");
    assert_eq!(ctx.bind_index_size(), (1, 1));
    ctx.dispatch(&k, &[keep.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
    assert_eq!(keep.to_vec().unwrap(), vec![3.0; 64]);
    ctx.release_memory().unwrap();
    assert_eq!((ctx.bind_cache_len(), ctx.bind_index_size()), (0, (0, 0)));
    ctx.dispatch(&k, &[keep.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
    assert_eq!(keep.to_vec().unwrap(), vec![4.0; 64], "a new uniform chunk and bind group after the release");
}

/// A buffer the pool evicts leaves the cache at the next submission, so `sync()` alone releases it (no later
/// dispatch needed). `trim_pool` sweeps the cache itself.
#[test]
fn sync_drops_the_entries_of_evicted_buffers() {
    let _shared = gpu!();
    let Some(ctx) = private() else { return };
    let k = ctx.kernel(&ADD_PARAM, Spec::new()).unwrap();
    ctx.set_pool_cap(0);
    // The pool's cap is its live-bytes high-water mark now: 256 bytes while `a` is the only buffer.
    let a = ctx.alloc(256).unwrap();
    for _ in 0..2 {
        ctx.dispatch(&k, &[a.raw()], &Params::new().u(0).f(1.0), [1, 1, 1]).unwrap();
    }
    ctx.sync().unwrap();
    ctx.trim_pool();
    assert_eq!(ctx.bind_cache_len(), 1, "a is lent out, not dropped");
    drop(a);
    // Freeing a 1 KiB buffer while `a` is cached exceeds the 1 KiB high-water mark: `a` is evicted.
    drop(ctx.alloc(1024).unwrap());
    assert_eq!(ctx.bind_cache_len(), 1, "no dispatch or submission since");
    ctx.sync().unwrap();
    assert_eq!((ctx.bind_cache_len(), ctx.bind_index_size()), (0, (0, 0)));
}

/// Buffers outside the pool are never cached (nothing would report their release); the context's dummy buffer is.
#[test]
fn only_pool_buffers_are_cached() {
    let _shared = gpu!();
    let Some(ctx) = private() else { return };
    let k = ctx.kernel(&ADD_PARAM, Spec::new()).unwrap();
    let raw = buffer(ctx.wgpu_device(), wgpu::BufferUsages::STORAGE);
    let before = ctx.work_stats().bind_groups;
    for _ in 0..5 {
        ctx.dispatch(&k, &[&raw], &Params::new().u(0).f(1.0), [1, 1, 1]).unwrap();
    }
    assert_eq!((ctx.work_stats().bind_groups - before, ctx.bind_cache_len()), (5, 0));
    // The dummy buffer is never written: `u(0)` threads write.
    for _ in 0..5 {
        ctx.dispatch(&k, &[ctx.dummy()], &Params::new().u(0).f(1.0), [1, 1, 1]).unwrap();
    }
    assert_eq!((ctx.work_stats().bind_groups - before, ctx.bind_cache_len()), (7, 1));
    ctx.sync().unwrap();
}

/// The pool reports the buffers it evicts or clears, gives up on ids (`All`) past its bound, and owns exactly the
/// buffers it created and still holds or lends.
#[test]
fn pool_reports_the_buffers_it_lets_go() {
    let ctx = gpu!();
    let usage = wgpu::BufferUsages::STORAGE;
    let mut pool = BufferPool::new(1024);
    let flag = pool.retired_flag();
    let a = pool.take(ctx.wgpu_device(), 1024, usage);
    let id_a = crate::context::buffer_id(&a);
    assert!(pool.owns(id_a));
    pool.give(a, 1024);
    assert!(pool.owns(id_a), "cached");
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
    assert!(!pool.owns(id_a) && pool.owns(id_x));
    assert!(!flag.load(Ordering::Relaxed));
    pool.clear();
    match pool.take_retired() {
        Retired::Some(ids) => assert_eq!(ids, HashSet::from([id_x])),
        other => panic!("{other:?}"),
    }
    assert!(!pool.owns(id_x));
    assert!(!pool.owns(crate::context::buffer_id(ctx.dummy())));
    let mut big = BufferPool::new(1 << 30);
    let held: Vec<_> = (0..5000).map(|_| big.take(ctx.wgpu_device(), 256, usage)).collect();
    let ids: Vec<u64> = held.iter().map(crate::context::buffer_id).collect();
    for x in held {
        big.give(x, 256);
    }
    big.clear();
    assert!(matches!(big.take_retired(), Retired::All));
    assert!(ids.iter().all(|&id| !big.owns(id)), "dropped past the bound too");
}

/// The cache holds at most two generations, an entry used again moves to the current one, and retiring a buffer
/// drops exactly the entries that bind it.
#[test]
fn bind_cache_is_bounded() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let k = Kernel::compile(dev, &THREE, &Spec::new()).unwrap();
    let uniform = buffer(dev, wgpu::BufferUsages::UNIFORM);
    let bufs: Vec<_> = (0..24).map(|_| buffer(dev, wgpu::BufferUsages::STORAGE)).collect();
    // Combination `i` (distinct for i < 24^3).
    let parts = |i: usize| [i % 24, i / 24 % 24, i / 576 % 24];
    let three = |i: usize| parts(i).map(|j| &bufs[j]);
    let mut cache = BindCache::default();
    // Whether `get` created a bind group.
    let get = |cache: &mut BindCache, i: usize| {
        let mut made = false;
        cache.get(dev, &k.layout, &uniform, &three(i), |_, _| true, &mut made);
        made
    };
    let holds = |cache: &BindCache, i: usize| cache.holds(&k.layout, &uniform, &three(i));
    let hot = 0;
    let mut created = 0;
    for i in 1..=3 * GENERATION {
        // The first sighting is not cached, the second is.
        assert!(get(&mut cache, i) && holds(&cache, i).is_none());
        assert!(get(&mut cache, i) && holds(&cache, i) == Some(true));
        created += 2;
        if i % 1000 == 1 {
            created += usize::from(get(&mut cache, hot));
        }
        assert!(cache.len() <= 2 * GENERATION);
        assert!(cache.index_size().1 <= 3 * cache.len(), "the index lists only resident entries");
    }
    assert_eq!(created, 6 * GENERATION + 2, "every combination twice, the hot one twice");
    assert!(holds(&cache, hot).is_some(), "used every 1000 combinations, it stays");
    let newest = 3 * GENERATION;
    // Probing an old entry moves it to the current generation, which must not roll over meanwhile.
    assert!(cache.len() > GENERATION && cache.len() + 8 < 2 * GENERATION, "{}", cache.len());
    // Retire the first buffer of the newest combination (in the current generation). Two old entries bind it too:
    // one is probed (a hit, which moves it to the current generation), the other stays in the old one. A resident
    // entry without it must survive.
    let r = parts(newest)[0];
    let mut old = (1..newest).rev().filter(|&i| parts(i).contains(&r) && holds(&cache, i) == Some(false));
    let (probed, stays_old) = (old.next().expect("an old entry binds it"), old.next().expect("and another"));
    let unaffected = (1..newest)
        .rev()
        .find(|&i| !parts(i).contains(&r) && holds(&cache, i).is_some())
        .expect("a resident entry without it");
    assert_eq!(holds(&cache, newest), Some(true));
    for i in [newest, probed, unaffected] {
        assert!(!get(&mut cache, i), "{i} resident: a hit");
    }
    assert_eq!((holds(&cache, probed), holds(&cache, stays_old)), (Some(true), Some(false)));
    let before = cache.len();
    cache.retire(&HashSet::from([crate::context::buffer_id(&bufs[r])]));
    assert!(cache.len() < before);
    assert!(cache.index_size().1 <= 3 * cache.len());
    assert!(holds(&cache, stays_old).is_none(), "gone from the old generation");
    for i in [newest, probed] {
        assert!(holds(&cache, i).is_none() && get(&mut cache, i), "{i} binds the retired buffer: gone");
    }
    assert!(!get(&mut cache, unaffected), "an entry without it stays");
}

/// A persistent buffer bound with many short-lived ones: when those are retired, their entries leave the
/// persistent buffer's index set too, so the index stays as small as the cache.
#[test]
fn the_reverse_index_stays_bounded() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let k = Kernel::compile(dev, &THREE, &Spec::new()).unwrap();
    let uniform = buffer(dev, wgpu::BufferUsages::UNIFORM);
    let keep = buffer(dev, wgpu::BufferUsages::STORAGE);
    let out = buffer(dev, wgpu::BufferUsages::STORAGE);
    let mut cache = BindCache::default();
    let mut made = false;
    for _ in 0..1000 {
        let short = buffer(dev, wgpu::BufferUsages::STORAGE);
        for _ in 0..2 {
            cache.get(dev, &k.layout, &uniform, &[&keep, &short, &out], |_, _| true, &mut made);
        }
        assert_eq!((cache.len(), cache.index_size()), (1, (3, 3)));
        cache.retire(&HashSet::from([crate::context::buffer_id(&short)]));
        assert_eq!((cache.len(), cache.index_size()), (0, (0, 0)));
    }
}

/// `set_bind_cache(false)` empties the cache and makes every dispatch create its bind group; on again, it caches.
#[test]
fn the_bind_cache_can_be_switched_off() {
    let _shared = gpu!();
    let Some(ctx) = private() else { return };
    let k = ctx.kernel(&ADD_PARAM, Spec::new()).unwrap();
    let a = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    let run = |n: usize| {
        let before = ctx.work_stats().bind_groups;
        for _ in 0..n {
            ctx.dispatch(&k, &[a.raw()], &Params::new().u(64).f(1.0), [1, 1, 1]).unwrap();
        }
        ctx.work_stats().bind_groups - before
    };
    assert_eq!(run(5), 2);
    ctx.set_bind_cache(false);
    assert_eq!(ctx.bind_cache_len(), 0);
    assert_eq!(run(5), 5);
    assert_eq!(ctx.bind_cache_len(), 0);
    ctx.set_bind_cache(true);
    assert_eq!(run(5), 2, "the recent keys went with the cache");
    assert_eq!(a.to_vec().unwrap(), vec![15.0; 64]);
}
