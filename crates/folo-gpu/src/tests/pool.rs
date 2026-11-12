//! The buffer pool's hand-out order and cache bound (`pool.rs`): the same request of every step gets the same buffer
//! whatever order the last step freed its buffers in (also through a context, where it keeps the bind groups), a step
//! whose size classes peak at different times keeps all its buffers, classes that stop being requested leave the
//! cache, and the pool never holds more than a cache bounded by the high-water mark alone would.

use super::{Rng, gpu};
use crate::context::buffer_id;
use crate::pool::{BufferPool, NEED_WINDOW};
use crate::{GpuTensor, UnaryOp};

const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
const MIB: u64 = 1 << 20;

/// Requests of one "step": several buffers of a few classes, freed in a different order every step (as autograd and
/// a hash map of gradients free them), some freed mid-step and their classes requested again.
#[test]
fn every_step_gets_the_same_buffers_whatever_the_free_order() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let mut pool = BufferPool::new(1 << 30);
    let mut rng = Rng::new(7);
    let classes = [256u64, 512, 1024, 256, 256, 512, 2048, 1024, 256, 512, 256, 1024];
    let mut first: Option<Vec<u64>> = None;
    let mut misses = 0;
    for step in 0..12 {
        let mut ids = Vec::new();
        let mut held = Vec::new();
        for (i, &c) in classes.iter().enumerate() {
            let b = pool.take(dev, c, USAGE);
            ids.push(buffer_id(&b));
            held.push((i, b, c));
            // Every third request, the even requests held so far are freed, in a shuffled order.
            if i % 3 == 2 {
                let (mut free, keep): (Vec<_>, Vec<_>) = held.drain(..).partition(|(j, _, _)| j % 2 == 0);
                held = keep;
                shuffle(&mut free, &mut rng);
                for (_, b, c) in free {
                    pool.give(b, c);
                }
            }
        }
        shuffle(&mut held, &mut rng);
        for (_, b, c) in held {
            pool.give(b, c);
        }
        if step == 0 {
            misses = pool.stats().misses;
        }
        match &first {
            None => first = Some(ids),
            Some(f) => assert_eq!(&ids, f, "step {step} got other buffers"),
        }
    }
    assert_eq!(pool.stats().misses, misses, "no buffer created after the first step");
}

/// The same through a context: a "step" of elementwise ops on three sizes, its temporaries dropped in a shuffled
/// order, creates no buffer and no bind group once warmed up. With newest-first reuse each op got other buffers every
/// step (the free order decided which), and its bind groups missed the cache.
#[test]
fn a_context_step_reuses_its_buffers_and_bind_groups_whatever_the_free_order() {
    let _shared = gpu!();
    // A context of its own, so other tests' dispatches and allocations do not move its counters.
    let ctx = match crate::GpuContext::new() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping, no second context: {e}"),
    };
    let mut rng = Rng::new(21);
    let step = |rng: &mut Rng| {
        let mut temps = Vec::new();
        for n in [3000, 5000, 9000] {
            let x = rng.vec(n, -1.0, 1.0);
            let a = GpuTensor::from_slice(&ctx, &x, &[n]).unwrap();
            let b = a.mul(&a).unwrap();
            let c = b.add(&a).unwrap();
            let d = c.unary(UnaryOp::Neg).unwrap();
            let e = d.mul(&b).unwrap();
            let got = e.to_vec().unwrap();
            for (i, (&g, &v)) in got.iter().zip(&x).enumerate() {
                let want = -(v * v + v) * (v * v);
                assert!((g - want).abs() <= 1e-6, "n {n}, element {i}: {g} vs {want}");
            }
            temps.extend([a, b, c, d, e]);
        }
        shuffle(&mut temps, rng);
        drop(temps);
    };
    for _ in 0..3 {
        step(&mut rng);
    }
    let (binds, misses) = (ctx.work_stats().bind_groups, ctx.pool_stats().misses);
    for i in 0..6 {
        step(&mut rng);
        let s = ctx.pool_stats();
        assert_eq!(s.misses, misses, "step {i} created a buffer: {s:?}");
        assert_eq!(ctx.work_stats().bind_groups, binds, "step {i} created a bind group");
    }
}

/// A miss evicts before it allocates: 4 MiB then 5 MiB taken and freed, then 6 MiB taken. The need (15 MiB, bounded
/// to twice the 6 MiB high-water mark) less the 6 MiB live leaves a 6 MiB cap, so the 4 MiB buffer goes before the
/// 6 MiB one is created: 6 + 5 = 11 MiB held, what a cache bounded by the high-water mark alone held (it evicted the
/// 4 MiB buffer when the 5 MiB one came back). Without the eviction on the miss the pool held 6 + 9 = 15 MiB.
#[test]
fn a_miss_evicts_before_it_allocates() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let mut pool = BufferPool::new(1024);
    for class in [4 * MIB, 5 * MIB] {
        let b = pool.take(dev, class, USAGE);
        pool.give(b, class);
    }
    assert_eq!(pool.stats().cached_bytes, 9 * MIB, "both kept: the need is 9 MiB");
    let six = pool.take(dev, 6 * MIB, USAGE);
    let s = pool.stats();
    assert_eq!((s.live_bytes, s.cached_bytes, s.peak_held_bytes), (6 * MIB, 5 * MIB, 11 * MIB), "{s:?}");

    // `reset_peak` restarts the held peak from what is held now, cached buffers included.
    pool.give(six, 6 * MIB);
    let four = pool.take(dev, 4 * MIB, USAGE);
    let s = pool.stats();
    assert_eq!(s.peak_held_bytes, 11 * MIB, "{s:?}");
    assert!(s.cached_bytes > 0, "{s:?}");
    pool.reset_peak();
    let s = pool.stats();
    assert_eq!(
        (s.peak_live_bytes, s.peak_held_bytes),
        (s.live_bytes, s.live_bytes + s.cached_bytes),
        "{s:?}"
    );
    assert!(s.peak_held_bytes < 11 * MIB, "{s:?}");
    pool.give(four, 4 * MIB);
}

/// Three workloads of the same live size (16 MiB) in one window: 4 x 4 MiB, then 8 x 2 MiB, then 16 x 1 MiB, each
/// taken at once and freed. By hand (cap `c = max(16, min(need, 32) - live)` MiB, misses evict before allocating):
/// the first holds 16 MiB and leaves 16 cached; the second's takes see `c` = 16, so 16 cached + 16 live = 32 at its
/// last take, and its frees (`c` = 32 - live) keep everything, 32 cached; the third's take `k` sees `c` = 32 - k and
/// evicts a 4 MiB buffer whenever 32 - k drops below the cache, so live + cached peaks at 32 (= 16 + 16 at its last
/// take). A cache bounded by the 16 MiB high-water mark alone held the same 32 MiB at most (16 live, 16 cached); a
/// bound checked only on frees held 16 live on top of 32 cached, 48 MiB.
#[test]
fn successive_workloads_hold_no_more_than_twice_the_high_water_mark() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let mut pool = BufferPool::new(1024);
    for (count, class) in [(4, 4 * MIB), (8, 2 * MIB), (16, MIB)] {
        let held: Vec<_> = (0..count).map(|_| pool.take(dev, class, USAGE)).collect();
        for b in held {
            pool.give(b, class);
        }
    }
    let s = pool.stats();
    assert_eq!((s.misses, s.peak_live_bytes), (28, 16 * MIB), "{s:?}");
    assert_eq!(s.peak_held_bytes, 32 * MIB, "{s:?}");
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
}

/// A step whose size classes peak at different times (4 MiB of one class, freed, then 5 MiB of another) needs 9 MiB
/// of buffers for 5 MiB live at most. A cache bounded by the live high-water mark evicted the first class every step
/// and created it again the next; the need keeps both.
#[test]
fn a_step_keeps_the_buffers_of_classes_that_peak_apart() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let mut pool = BufferPool::new(1024);
    let step = |pool: &mut BufferPool| {
        for class in [MIB, MIB + MIB / 4] {
            let held: Vec<_> = (0..4).map(|_| pool.take(dev, class, USAGE)).collect();
            for b in held {
                pool.give(b, class);
            }
        }
    };
    step(&mut pool);
    let created = pool.stats().misses;
    assert_eq!(created, 8);
    // Eight requests per step: past two need-window rollovers, which must not evict what the step needs.
    for _ in 0..(2 * NEED_WINDOW / 8 + 20) {
        step(&mut pool);
    }
    let s = pool.stats();
    assert_eq!(s.misses, created, "{s:?}");
    assert_eq!(s.cached_bytes, 9 * MIB, "{s:?}");
    assert_eq!(s.need_bytes, 9 * MIB, "{s:?}");
    assert_eq!(s.peak_held_bytes, 9 * MIB, "{s:?}");
}

/// Classes no longer requested drop out of the need after two windows, and their cached buffers are evicted by the
/// next frees that do not fit; the need never lets the cache grow past twice the live high-water mark.
#[test]
fn classes_no_longer_requested_leave_the_cache() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let mut pool = BufferPool::new(1024);
    // Shape A: 4 buffers of 1 MiB at once.
    let held: Vec<_> = (0..4).map(|_| pool.take(dev, MIB, USAGE)).collect();
    let a: Vec<u64> = held.iter().map(buffer_id).collect();
    for b in held {
        pool.give(b, MIB);
    }
    // Shape B: 4 buffers of 1.25 MiB at once, for two need windows.
    let b_class = MIB + MIB / 4;
    let mut takes = 4;
    while takes < 2 * NEED_WINDOW + 8 {
        let held: Vec<_> = (0..4).map(|_| pool.take(dev, b_class, USAGE)).collect();
        for b in held {
            pool.give(b, b_class);
        }
        takes += 4;
    }
    let s = pool.stats();
    assert_eq!(s.need_bytes, 4 * b_class, "{s:?}");
    assert_eq!(s.cached_bytes, 4 * b_class, "{s:?}");
    assert!(a.iter().all(|&id| !pool.owns(id)), "shape A's buffers were let go");

    // Many classes requested one at a time: the need sums them, the cache stays within twice the high-water mark.
    let mut pool = BufferPool::new(1024);
    for i in 1..=40u64 {
        let b = pool.take(dev, i * 64 * 1024, USAGE);
        pool.give(b, i * 64 * 1024);
    }
    let s = pool.stats();
    assert_eq!(s.need_bytes, (1..=40u64).sum::<u64>() * 64 * 1024, "{s:?}");
    assert!(s.cached_bytes <= 2 * 40 * 64 * 1024, "{s:?}");
}

/// A need window that rolls over on a request served from the cache applies the lower need at once: classes that
/// left the need are evicted on that request, not only at the next miss or free.
#[test]
fn a_window_rollover_on_a_hit_applies_the_lower_need() {
    let ctx = gpu!();
    let dev = ctx.wgpu_device();
    let mut pool = BufferPool::new(1024);
    for class in [4 * MIB, 5 * MIB] {
        let b = pool.take(dev, class, USAGE);
        pool.give(b, class);
    }
    assert_eq!(pool.stats().cached_bytes, 9 * MIB);
    // Small requests only, all hits after the first, up to the last request of the second window: the final take
    // below is request 2 * NEED_WINDOW, a hit that rolls the window over.
    let small = 256;
    for _ in 2..2 * NEED_WINDOW - 1 {
        let b = pool.take(dev, small, USAGE);
        pool.give(b, small);
    }
    let b = pool.take(dev, small, USAGE);
    let s = pool.stats();
    assert_eq!(s.need_bytes, small, "{s:?}");
    // The cap is now the high-water mark (5 MiB): the 4 MiB buffer, freed first, is gone.
    assert!(s.live_bytes + s.cached_bytes <= 5 * MIB + small, "{s:?}");
    pool.give(b, small);
}
