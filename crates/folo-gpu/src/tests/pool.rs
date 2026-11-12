//! The buffer pool's hand-out order and cache bound (`pool.rs`): the same request of every step gets the same buffer
//! whatever order the last step freed its buffers in, a step whose size classes peak at different times keeps all
//! its buffers, and classes that stop being requested leave the cache.

use super::{Rng, gpu};
use crate::context::buffer_id;
use crate::pool::{BufferPool, NEED_WINDOW};

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
    for _ in 0..20 {
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
