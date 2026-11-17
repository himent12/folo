//! The int8 kernels against a scalar integer reference: the same codes bit for bit, on every path (VNNI or
//! portable kernel, vectorised or portable epilogue), at several thread counts.

use super::conv::{TASK_MACS, TEST_TASK_MACS, plan};
use super::kernel::{MR, NR, Tile, kernel_scalar};
use super::*;
use crate::rng::Rng;
use std::sync::atomic::Ordering;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
}

fn codes(rng: &mut Rng, n: usize) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n).map(|_| (rng.next_u64() >> 7) as u32).collect();
    v.extend(std::iter::repeat_n(0xDEAD_BEEF, SLACK));
    v
}

fn byte_of(w: &[u32], g: usize, hw: usize, p: usize, t: usize) -> u8 {
    w[g * hw + p].to_le_bytes()[t]
}

struct Case {
    n: usize,
    parts: Vec<usize>,
    o: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    pad: usize,
    silu: bool,
    res: bool,
    f32_out: bool,
}

/// The reference: the conv sum per output in i64 from the codes, then the epilogue written out.
fn reference(
    c: &Case,
    xs: &[(Vec<u32>, Grid)],
    wt: &[f32],
    bias: &[f32],
    res: Option<&(Vec<u32>, Grid)>,
    out: Grid,
) -> (Vec<u32>, Vec<f32>) {
    let cin: usize = c.parts.iter().sum();
    let (hw, taps) = (c.h * c.w, c.k * c.k);
    let oh = (c.h + 2 * c.pad - c.k) / c.s + 1;
    let ow = (c.w + 2 * c.pad - c.k) / c.s + 1;
    let ohw = oh * ow;
    // Channel ci's part and its index within the part.
    let mut chan = Vec::new();
    for (pi, &pc) in c.parts.iter().enumerate() {
        chan.extend((0..pc).map(|j| (pi, j)));
    }
    let scales: Vec<f32> = chan.iter().map(|&(pi, _)| xs[pi].1.scale).collect();
    let folded: Vec<f32> = wt.iter().enumerate().map(|(i, &v)| v * scales[(i / taps) % cin]).collect();
    let (q, sw) = quantize_weights(&folded, c.o);
    let mut qout = vec![0u32; c.n * c.o / 4 * ohw];
    let mut fout = vec![0f32; c.n * c.o * ohw];
    for img in 0..c.n {
        for o in 0..c.o {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc: i64 = 0;
                    let mut corr: i64 = 0;
                    for (ci, &(pi, j)) in chan.iter().enumerate() {
                        let (data, g) = (&xs[pi].0, xs[pi].1);
                        let per_img = c.parts[pi] / 4 * hw;
                        for ky in 0..c.k {
                            for kx in 0..c.k {
                                let wq = q[(o * cin + ci) * taps + ky * c.k + kx] as i64;
                                let y = (oy * c.s + ky) as isize - c.pad as isize;
                                let x = (ox * c.s + kx) as isize - c.pad as isize;
                                let v = if y < 0 || x < 0 || y >= c.h as isize || x >= c.w as isize {
                                    g.zero
                                } else {
                                    byte_of(
                                        &data[img * per_img..],
                                        j / 4,
                                        hw,
                                        y as usize * c.w + x as usize,
                                        j % 4,
                                    )
                                };
                                acc += wq * v as i64;
                                corr += wq * g.zero as i64;
                            }
                        }
                    }
                    let p = oy * ow + ox;
                    let mut yv = ((acc - corr) as i32 as f32).mul_add(sw[o], bias[o]);
                    if c.silu {
                        yv = silu_q8(yv);
                    }
                    if let Some((r, rg)) = res {
                        let b = byte_of(&r[img * c.o / 4 * ohw..], o / 4, ohw, p, o % 4);
                        yv = rg.scale.mul_add((b as i32 - rg.zero as i32) as f32, yv);
                    }
                    if c.f32_out {
                        fout[(img * c.o + o) * ohw + p] = yv;
                    } else {
                        let code = out.quantize(yv) as u32;
                        qout[(img * c.o / 4 + o / 4) * ohw + p] |= code << (8 * (o % 4));
                    }
                }
            }
        }
    }
    (qout, fout)
}

fn run_case(c: &Case, seed: u64) {
    let mut rng = Rng::seed(seed);
    let cin: usize = c.parts.iter().sum();
    let hw = c.h * c.w;
    let xs: Vec<(Vec<u32>, Grid)> = c
        .parts
        .iter()
        .map(|&pc| {
            (
                codes(&mut rng, c.n * pc / 4 * hw),
                Grid { scale: rng.uniform(0.01, 0.06), zero: (rng.next_u64() % 256) as u8 },
            )
        })
        .collect();
    let wt: Vec<f32> = (0..c.o * cin * c.k * c.k).map(|_| rng.uniform(-0.5, 0.5)).collect();
    let bias: Vec<f32> = (0..c.o).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let oh = (c.h + 2 * c.pad - c.k) / c.s + 1;
    let ow = (c.w + 2 * c.pad - c.k) / c.s + 1;
    let ohw = oh * ow;
    let res = c.res.then(|| (codes(&mut rng, c.n * c.o / 4 * ohw), Grid { scale: 0.03, zero: 17 }));
    let out_grid = Grid { scale: 0.05, zero: 40 };
    let grids: Vec<Grid> =
        xs.iter().zip(&c.parts).flat_map(|((_, g), &pc)| std::iter::repeat_n(*g, pc)).collect();
    let cv = QConv::new(&wt, &bias, (c.o, cin, c.k), c.s, c.pad, c.silu, &grids);
    let (want_q, want_f) = reference(c, &xs, &wt, &bias, res.as_ref(), out_grid);
    for threads in [1, 3, 8] {
        let views: Vec<C4> = xs
            .iter()
            .zip(&c.parts)
            .map(|((d, g), &pc)| C4 { data: d, offset: 0, img_stride: pc / 4 * hw, c: pc, zero: g.zero })
            .collect();
        let rv = res
            .as_ref()
            .map(|(d, g)| (C4 { data: d, offset: 0, img_stride: c.o / 4 * ohw, c: c.o, zero: g.zero }, *g));
        if c.f32_out {
            let mut got = vec![0f32; c.n * c.o * ohw];
            pool(threads).install(|| conv(&views, c.n, c.h, c.w, &cv, rv, QOut::F32 { data: &mut got }));
            let same = got.iter().zip(&want_f).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "f32 out differs at {threads} threads");
        } else {
            let mut got = alloc(c.n, c.o, ohw);
            pool(threads).install(|| {
                conv(&views, c.n, c.h, c.w, &cv, rv, QOut::U8 { data: &mut got, grid: out_grid })
            });
            let bad = got[..want_q.len()].iter().zip(&want_q).position(|(a, b)| a != b);
            assert_eq!(bad, None, "u8 out differs at {threads} threads");
        }
    }
}

#[test]
fn conv_matches_the_integer_reference() {
    let base = Case {
        n: 1,
        parts: vec![8],
        o: 8,
        h: 7,
        w: 9,
        k: 3,
        s: 1,
        pad: 1,
        silu: true,
        res: false,
        f32_out: false,
    };
    let cases = [
        // 3×3 stride 1 with a residual, odd sizes (ragged tiles and dropped columns).
        Case { res: true, ..base.clone() },
        // 3×3 stride 2, two images.
        Case { n: 2, s: 2, h: 11, w: 10, parts: vec![12], o: 16, ..base.clone() },
        // 1×1 over three parts with different grids (a C3k2 cv2), identity, a residual.
        Case {
            k: 1,
            pad: 0,
            parts: vec![4, 8, 4],
            o: 12,
            silu: false,
            res: true,
            h: 13,
            w: 13,
            ..base.clone()
        },
        // 1×1 with an f32 output (qkv).
        Case { k: 1, pad: 0, parts: vec![16], o: 8, silu: false, f32_out: true, ..base.clone() },
        // A long K (more than one task's worth of steps) and a stride-2 3×3 with f32 out.
        Case { parts: vec![64], o: 4, h: 5, w: 6, ..base.clone() },
        Case { s: 2, f32_out: true, h: 9, w: 9, ..base.clone() },
    ];
    // With the real task size (one task for shapes this small) and with one-MAC tasks (as many tasks as the
    // pool allows: every split of pixel chunks and channel groups).
    for task_macs in [0, 1] {
        TEST_TASK_MACS.store(task_macs, Ordering::Relaxed);
        for (i, c) in cases.iter().enumerate() {
            run_case(c, 100 + i as u64);
        }
    }
    TEST_TASK_MACS.store(0, Ordering::Relaxed);
}

#[test]
fn plan_covers_the_conv_in_few_big_tasks() {
    // (images, chunks, groups, MACs a chunk of a group, threads)
    let cases = [
        (1, 67, 32, 4608, 28),
        (1, 17, 64, 3072, 28),
        (1, 1067, 16, 1152, 28),
        (2, 5, 3, 100, 8),
        (1, 9, 40, 9216, 1),
    ];
    // Both orders: a small input (channel groups first) and a big one (pixel chunks first).
    for ((n, chunks, og, unit, threads), input) in
        cases.into_iter().flat_map(|c| [(c, 1 << 10), (c, 1 << 30)])
    {
        let (cb, ob) = plan(n, chunks, og, unit, input, threads);
        assert!((1..=chunks).contains(&cb) && (1..=og).contains(&ob));
        let tasks = n * chunks.div_ceil(cb) * og.div_ceil(ob);
        let macs = n * chunks * og * unit;
        assert!(tasks <= (3 * threads).max(1), "{tasks} tasks on {threads} threads");
        // At least one task an image.
        assert!(tasks <= n || macs / tasks >= TASK_MACS / 2, "tasks of {} MACs", macs / tasks);
        if threads == 1 {
            assert_eq!(tasks, 1);
        }
    }
}

impl Clone for Case {
    fn clone(&self) -> Self {
        Case { parts: self.parts.clone(), ..*self }
    }
}

#[test]
fn kernels_agree() {
    // The VNNI kernel (when present) and the portable one: the same integers on extreme codes.
    let mut rng = Rng::seed(3);
    let ks = 37;
    let a: Vec<i32> = (0..ks * MR)
        .map(
            |i| if i % 5 == 0 { i32::from_le_bytes([0x81, 0x7F, 0x81, 0x7F]) } else { rng.next_u64() as i32 },
        )
        .collect();
    let rows_data: Vec<Vec<u32>> = (0..ks)
        .map(|i| {
            (0..NR + 5).map(|j| if (i + j) % 3 == 0 { u32::MAX } else { rng.next_u64() as u32 }).collect()
        })
        .collect();
    let rows: Vec<*const u32> = rows_data.iter().map(|r| r.as_ptr()).collect();
    let mut want: Tile = [[0; NR]; MR];
    let mut got: Tile = [[0; NR]; MR];
    for col in [0, 5] {
        // SAFETY: `a` has ks · MR words and every row col + NR.
        unsafe {
            kernel_scalar(ks, a.as_ptr(), rows.as_ptr(), col, &mut want);
            super::kernel::kernel()(ks, a.as_ptr(), rows.as_ptr(), col, &mut got);
        }
        assert_eq!(want, got);
    }
}

#[test]
fn silu_q8_is_silu() {
    // Within 2e-6 of SiLU in f64 (relative, or absolute near zero), and finite at the extremes.
    let mut rng = Rng::seed(9);
    for _ in 0..20_000 {
        let x = rng.uniform(-90.0, 90.0);
        let want = x as f64 / (1.0 + (-(x as f64)).exp());
        let got = silu_q8(x) as f64;
        assert!((got - want).abs() <= 2e-6 * want.abs().max(1.0), "silu_q8({x}) = {got}, want {want}");
    }
    assert_eq!(silu_q8(f32::MAX), f32::MAX);
    assert!(silu_q8(f32::MIN).abs() < 1e-30 && silu_q8(-200.0).abs() < 1e-30);
    assert_eq!(silu_q8(0.0), 0.0);
    assert!(silu_q8(f32::NAN).is_nan());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn silu_q8_scalar_and_vector_agree() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
        return;
    }
    let mut rng = Rng::seed(10);
    let mut xs: Vec<f32> = (0..4096).map(|_| rng.uniform(-100.0, 100.0)).collect();
    xs.extend([
        0.0,
        -0.0,
        1e-30,
        -1e-30,
        86.9,
        -86.9,
        87.0,
        -87.0,
        88.0,
        -88.0,
        -200.0,
        200.0,
        f32::MAX,
        f32::MIN,
    ]);
    for ch in xs.chunks(8) {
        let mut v = [0f32; 8];
        v[..ch.len()].copy_from_slice(ch);
        // SAFETY: AVX2 and FMA were detected.
        let got: [f32; 8] = unsafe {
            use std::arch::x86_64::*;
            let mut o = [0f32; 8];
            _mm256_storeu_ps(o.as_mut_ptr(), super::kernel::silu_q8_avx2(_mm256_loadu_ps(v.as_ptr())));
            o
        };
        for (x, g) in v.iter().zip(got) {
            assert_eq!(silu_q8(*x).to_bits(), g.to_bits(), "silu_q8({x})");
        }
    }
}

#[test]
fn grid_round_trips_and_pools_are_exact() {
    let g = Grid::from_range(-0.3, 5.0);
    assert_eq!(g.dequantize(g.zero), 0.0);
    for q in 0..=255u8 {
        assert_eq!(g.quantize(g.dequantize(q)), q);
    }
    let mut rng = Rng::seed(4);
    let (n, c, h, w) = (2, 8, 6, 7);
    let hw = h * w;
    let x: Vec<f32> = (0..n * c * hw).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let mut q = alloc(n, c, hw);
    quantize(&x, n, c, hw, g, &mut q);
    let view = C4 { data: &q, offset: 0, img_stride: c / 4 * hw, c, zero: g.zero };
    let mut back = vec![0f32; n * c * hw];
    dequantize(view, n, hw, g, &mut back);
    for (a, b) in x.iter().zip(&back) {
        assert_eq!(g.dequantize(g.quantize(*a)), *b);
    }
    // Max pool on codes = max pool of the dequantised values, re-quantised.
    let mut mp = alloc(n, c, hw);
    max_pool(view, n, h, w, 5, &mut mp);
    let mut up = alloc(n, c, 4 * hw);
    upsample2(view, n, h, w, &mut up);
    for img in 0..n {
        for ch in 0..c {
            let plane = &back[(img * c + ch) * hw..][..hw];
            for y in 0..h {
                for xo in 0..w {
                    let mut m = f32::NEG_INFINITY;
                    for yy in y.saturating_sub(2)..(y + 3).min(h) {
                        for xx in xo.saturating_sub(2)..(xo + 3).min(w) {
                            m = m.max(plane[yy * w + xx]);
                        }
                    }
                    let got = (mp[(img * c / 4 + ch / 4) * hw + y * w + xo] >> (8 * (ch % 4))) as u8;
                    assert_eq!(g.dequantize(got), m);
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let u = (up[(img * c / 4 + ch / 4) * 4 * hw + (2 * y + dy) * 2 * w + 2 * xo + dx]
                            >> (8 * (ch % 4))) as u8;
                        assert_eq!(g.dequantize(u), plane[y * w + xo]);
                    }
                }
            }
        }
    }
}
