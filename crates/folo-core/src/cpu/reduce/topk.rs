//! Top-k selection along the last dimension: `k == 1` zero-alloc scan, bounded-`k` max-heap, and quickselect.

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::cpu::{PAR_MIN, alloc_uninit};

/// Restore max-heap property under `cmp` (so `heap[0]` is the worst of the kept `k` elements).
#[inline]
fn sift_down(
    heap: &mut [(f32, usize)],
    mut i: usize,
    cmp: &impl Fn(&(f32, usize), &(f32, usize)) -> Ordering,
) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let mut worst = left;
        if right < n && cmp(&heap[right], &heap[left]) == Ordering::Greater {
            worst = right;
        }
        if cmp(&heap[worst], &heap[i]) == Ordering::Greater {
            heap.swap(i, worst);
            i = worst;
        } else {
            break;
        }
    }
}

pub(crate) fn topk(x: &[f32], rows: usize, d: usize, k: usize, largest: bool) -> (Vec<f32>, Vec<i64>) {
    if k == 0 || rows == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut vals = alloc_uninit(rows * k);
    let mut idx = vec![0i64; rows * k];
    // `total_cmp` orders -0.0 before 0.0; they are equal, so they tie (broken by index) like any other values.
    let key = |v: f32| if v == 0.0 { 0.0 } else { v };
    let cmp = |a: &(f32, usize), b: &(f32, usize)| {
        let o = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
        o.then(a.1.cmp(&b.1))
    };
    let use_heap = k > 1 && k <= 1024 && k.saturating_mul(4) < d;
    let init_cap = if k == 1 {
        0
    } else if use_heap {
        k
    } else {
        d
    };
    let row = |buf: &mut Vec<(f32, usize)>, r: usize, v: &mut [f32], ix: &mut [i64]| {
        let src = &x[r * d..(r + 1) * d];
        if k == 1 {
            let mut best = (src[0], 0usize);
            for (i, &val) in src.iter().enumerate().skip(1) {
                // Fast reject: since `i > best.1`, any non-NaN `val <= best.0` (or `>=` for smallest)
                // cannot beat `best` even on ties (`-0.0 == +0.0` included). NaN comparisons are false,
                // so NaN falls through to exact `total_cmp`.
                if largest {
                    if val <= best.0 {
                        continue;
                    }
                } else if val >= best.0 {
                    continue;
                }
                if cmp(&(val, i), &best) == Ordering::Less {
                    best = (val, i);
                }
            }
            v[0] = best.0;
            ix[0] = best.1 as i64;
            return;
        }
        buf.clear();
        if use_heap {
            buf.extend(src[..k].iter().copied().zip(0..k));
            for start in (0..k / 2).rev() {
                sift_down(buf, start, &cmp);
            }
            for (i, &val) in src.iter().enumerate().skip(k) {
                if largest {
                    if val <= buf[0].0 {
                        continue;
                    }
                } else if val >= buf[0].0 {
                    continue;
                }
                if cmp(&(val, i), &buf[0]) == Ordering::Less {
                    buf[0] = (val, i);
                    sift_down(buf, 0, &cmp);
                }
            }
            buf.sort_unstable_by(cmp);
        } else {
            buf.extend(src.iter().copied().zip(0..));
            if k < d {
                buf.select_nth_unstable_by(k - 1, cmp);
                buf.truncate(k);
            }
            buf.sort_unstable_by(cmp);
        }
        for (j, &(val, i)) in buf.iter().enumerate() {
            v[j] = val;
            ix[j] = i as i64;
        }
    };
    if rows * d >= PAR_MIN && rows > 1 {
        vals.par_chunks_mut(k)
            .zip(idx.par_chunks_mut(k))
            .enumerate()
            .for_each_init(|| Vec::with_capacity(init_cap), |buf, (r, (v, ix))| row(buf, r, v, ix));
    } else {
        let mut buf = Vec::with_capacity(init_cap);
        for (r, (v, ix)) in vals.chunks_mut(k).zip(idx.chunks_mut(k)).enumerate() {
            row(&mut buf, r, v, ix);
        }
    }
    (vals, idx)
}
