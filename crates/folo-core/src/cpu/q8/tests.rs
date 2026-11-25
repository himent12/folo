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
    /// Words of other data before each image of the inputs and the residual (views with an offset and a longer
    /// image stride, as `narrow` makes them).
    gap: usize,
}

/// `dense` (`n` images of `per_img` words) with `gap` other words before each image and [`SLACK`] after.
fn spread(rng: &mut Rng, dense: &[u32], n: usize, per_img: usize, gap: usize) -> Vec<u32> {
    let mut v = Vec::new();
    for img in 0..n {
        v.extend((0..gap).map(|_| rng.next_u64() as u32));
        v.extend_from_slice(&dense[img * per_img..(img + 1) * per_img]);
    }
    v.extend((0..SLACK).map(|_| rng.next_u64() as u32));
    v
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
    let gap = c.gap;
    let xs_v: Vec<Vec<u32>> =
        xs.iter().zip(&c.parts).map(|((d, _), &pc)| spread(&mut rng, d, c.n, pc / 4 * hw, gap)).collect();
    let res_v = res.as_ref().map(|(d, _)| spread(&mut rng, d, c.n, c.o / 4 * ohw, gap));
    for threads in [1, 3, 8] {
        let views: Vec<C4> = xs_v
            .iter()
            .zip(xs.iter().zip(&c.parts))
            .map(|(d, ((_, g), &pc))| C4 {
                data: d,
                offset: gap,
                img_stride: pc / 4 * hw + gap,
                c: pc,
                zero: g.zero,
            })
            .collect();
        let rv = res.as_ref().zip(res_v.as_ref()).map(|((_, g), d)| {
            (C4 { data: d, offset: gap, img_stride: c.o / 4 * ohw + gap, c: c.o, zero: g.zero }, *g)
        });
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
        gap: 0,
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
        // 1×1 over two images of offset, strided views (two parts, a residual): what `narrow` hands the conv.
        Case { n: 2, k: 1, pad: 0, parts: vec![8, 4], o: 8, res: true, gap: 37, h: 6, w: 11, ..base.clone() },
        // And a 3×3 with them.
        Case { n: 2, res: true, gap: 5, ..base.clone() },
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
        let (cb, ob) = plan(n, chunks, og, unit, input, (threads, TASK_MACS));
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

#[test]
#[should_panic(expected = "zero point")]
fn conv_refuses_inputs_on_other_zero_points() {
    // Packed for zero points [1, 2, 3, 4]: one C4 part cannot carry them (its padding and correction use one).
    let grids: Vec<Grid> = (1..=4).map(|z| Grid { scale: 0.05, zero: z }).collect();
    let cv = QConv::new(&[0.1; 4 * 4 * 9], &[0.0; 4], (4, 4, 3), 1, 1, false, &grids);
    let x = codes(&mut Rng::seed(1), 9);
    let mut out = alloc(1, 4, 9);
    let view = C4 { data: &x, offset: 0, img_stride: 9, c: 4, zero: 1 };
    conv(&[view], 1, 3, 3, &cv, None, QOut::U8 { data: &mut out, grid: Grid { scale: 0.1, zero: 0 } });
}

#[test]
#[should_panic(expected = "c · k²")]
fn qconv_refuses_reductions_that_overflow_i32() {
    let c = 4 * 1024;
    let grids = vec![Grid { scale: 1.0, zero: 255 }; c];
    QConv::new(&vec![1.0; 4 * c * 9], &[0.0; 4], (4, c, 3), 1, 1, false, &grids);
}

#[test]
fn quantize_round_trips_planes_over_one_block() {
    // Planes longer than quantize's parallel block (4096 values), so the blocks after the first are used.
    let g = Grid::from_range(-1.0, 3.0);
    let (n, c, hw) = (2, 8, 4096 + 1500);
    let mut rng = Rng::seed(9);
    let x: Vec<f32> = (0..n * c * hw).map(|_| rng.uniform(-1.5, 3.5)).collect();
    let mut q = alloc(n, c, hw);
    quantize(&x, n, c, hw, g, &mut q);
    for img in 0..n {
        for ch in 0..c {
            for p in (0..hw).step_by(7) {
                let want = g.quantize(x[(img * c + ch) * hw + p]);
                assert_eq!(
                    byte_of(&q[img * c / 4 * hw..], ch / 4, hw, p, ch % 4),
                    want,
                    "({img}, {ch}, {p})"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "under a 3×3 kernel")]
fn conv_refuses_an_input_smaller_than_its_kernel() {
    let grids = vec![Grid { scale: 0.05, zero: 3 }; 4];
    let cv = QConv::new(&[0.1; 4 * 4 * 9], &[0.0; 4], (4, 4, 3), 1, 0, false, &grids);
    let x = codes(&mut Rng::seed(2), 2);
    let mut out = alloc(1, 4, 1);
    let view = C4 { data: &x, offset: 0, img_stride: 2, c: 4, zero: 3 };
    conv(&[view], 1, 1, 2, &cv, None, QOut::U8 { data: &mut out, grid: Grid { scale: 0.1, zero: 0 } });
}

#[test]
#[should_panic(expected = "C4: buffer")]
fn views_whose_extent_wraps_are_refused() {
    // Offset 128 plus a stride of usize::MAX − 127 wraps to 128: it must not pass as inside 128 + SLACK words.
    let data = vec![0u32; 128 + SLACK];
    let view = C4 { data: &data, offset: 128, img_stride: usize::MAX - 127, c: 4, zero: 0 };
    view.check(2, 64);
}

#[test]
fn supports_and_out_size_name_the_kernels_limits() {
    assert!(QConv::supports((4, 4, 3), 1, 1));
    assert!(QConv::supports((8, 4, 1), MAX_STRIDE, MAX_PAD));
    assert!(!QConv::supports((6, 4, 1), 1, 0), "output channels not a multiple of 4");
    assert!(!QConv::supports((4, 4, 0), 1, 0), "no kernel");
    assert!(!QConv::supports((4, 4, 1), 0, 0), "stride 0");
    assert!(!QConv::supports((4, 4, 1), MAX_STRIDE + 1, 0), "stride past the limit");
    assert!(!QConv::supports((4, 4, 1), 1, MAX_PAD + 1), "padding past the limit");
    assert!(!QConv::supports((4, 4 * 1024, 3), 1, 1), "c · k² past the limit");
    assert!(!QConv::supports((4, 4, usize::MAX), 1, 0), "k² overflows");
    assert_eq!(out_size(80, 80, 3, 1, 1), Some((80, 80)));
    assert_eq!(out_size(80, 81, 3, 2, 1), Some((40, 41)));
    assert_eq!(out_size(2, 2, 1, 65, 32), Some((2, 2)));
    assert_eq!(out_size(1, 2, 3, 1, 0), None, "kernel larger than the input");
    assert_eq!(out_size(4, 4, 1, 0, 0), None, "stride 0");
    assert_eq!(out_size(usize::MAX, 4, 1, 1, 1), None, "padded size overflows");
}

#[test]
fn quantize_weights_has_its_known_answers() {
    // Row max 127: scale 1 exactly, so the codes are the values rounded, ties away from zero.
    let row = [127.0, -127.0, 2.5, -2.5, 0.5, -0.5, 1.49, 0.0];
    let (q, s) = quantize_weights(&row, 1);
    assert_eq!(s, [1.0]);
    assert_eq!(q, [127, -127, 3, -3, 1, -1, 1, 0]);
    // Whatever the row's largest magnitude, it maps to ±127 (the clamp holds when `m / 127` rounds), and a row
    // of zeros gets scale 1.
    for m in [1.0f32, 0.1, 3.3, 1e-20, 1e30, f32::MAX] {
        let (q, s) = quantize_weights(&[m, -m, m / 2.0, 0.0], 1);
        assert_eq!((q[0], q[1]), (127, -127), "max {m}");
        assert!((q[2] - 64).abs() <= 1, "max {m}: half is {}", q[2]);
        assert_eq!(s[0], m / 127.0);
    }
    let (q, s) = quantize_weights(&[0.0; 8], 2);
    assert_eq!((q, s), (vec![0; 8], vec![1.0, 1.0]));
}

#[test]
fn epilogue_rounds_exact_ties_to_even_on_both_paths() {
    use super::kernel::{Epi, Out};
    // acc − corr = 2j + 1 − 24 (odd) and scale ½: every y · inv is an exact tie, j − 11.5.
    let (corr, scale, bias) = ([24; MR], [0.5; MR], [0.0; MR]);
    let mut acc: Tile = [[0; NR]; MR];
    for row in acc.iter_mut() {
        for (j, a) in row.iter_mut().enumerate() {
            *a = 2 * j as i32 + 1;
        }
    }
    let zero = 128;
    let epi = Epi {
        corr: &corr,
        scale: &scale,
        bias: &bias,
        silu: false,
        res: None,
        out: Out::U8 { inv: &[1.0; MR], zero: &[zero; MR] },
    };
    let want: Vec<u32> = (0..NR)
        .map(|j| {
            let n = 2 * j as i32 + 1 - 24;
            let lo = n.div_euclid(2);
            let v = if lo % 2 == 0 { lo } else { lo + 1 };
            u32::from_le_bytes([(v + zero) as u8; 4])
        })
        .collect();
    let (res, mut f) = ([0u32; NR], [[0f32; NR]; MR]);
    let mut q = [0u32; NR];
    epi.apply(&acc, NR, &res, &mut q, &mut f);
    assert_eq!(q.to_vec(), want, "portable epilogue");
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let mut q = [0u32; NR];
        // SAFETY: AVX2 and FMA were detected.
        unsafe { epi.apply_avx2(&acc, &res, &mut q, &mut f) };
        assert_eq!(q.to_vec(), want, "AVX2 epilogue");
    }
}

#[test]
fn conv_size_refuses_buffers_that_overflow() {
    assert_eq!(conv_size(2, 64, 32, 80, 80, 3, 1, 1), Some((80, 80)));
    // The padded copy, `c · (h + 2p + s) · (w + 2p + s)`, overflows although the output size does not.
    let big = 1usize << (usize::BITS / 2 - 1);
    assert_eq!(out_size(big, big, 1, 64, 0).map(|_| ()), Some(()));
    assert_eq!(conv_size(1, 64, 4, big, big, 1, 64, 0), None);
    // Only the output overflows (n · c fits, n · o does not).
    let n = usize::MAX / 64 + 1;
    assert!(n.checked_mul(4).is_some() && n.checked_mul(64).is_none());
    assert_eq!(conv_size(n, 4, 64, 1, 1, 1, 1, 0), None);
    // No size overflows `usize`, but the input and the padded copy are over `isize::MAX` bytes.
    let side = 1usize << (usize::BITS / 2 - 3);
    assert!((8 * side * side).checked_mul(4).is_none_or(|b| b > isize::MAX as usize));
    assert_eq!(conv_size(1, 8, 4, side, side, 1, 1, 0), None);
    assert_eq!(conv_size(1, 8, 4, side / 2, side / 2, 1, 1, 0).map(|_| ()), Some(()));
    // A padded copy over `isize::MAX` bytes on 32-bit targets (8194² · 32 codes).
    let wide = conv_size(1, 32, 4, 1, 1, 1, 1, 4096);
    assert_eq!(wide.is_none(), usize::BITS == 32);
    assert_eq!(conv_size(1, 4, 4, 1, 2, 3, 1, 0), None, "kernel larger than the input");
}

#[test]
fn qconv_rejects_zero_channels_and_handles_empty_batch() {
    assert!(!QConv::supports((0, 4, 3), 1, 1), "zero output channels");
    assert!(!QConv::supports((4, 0, 3), 1, 1), "zero input channels");
    assert!(!QConv::supports((usize::MAX & !3, 4, 1), 1, 0), "weight count overflow");
    assert!(!QConv::supports(((usize::MAX / 4) & !3, 4, 1), 1, 0), "weight byte overflow");
    assert!(!QConv::supports((((isize::MAX as usize) / 4) & !3, 4, 1), 1, 0), "weight exceeds isize::MAX");
    assert!(
        !QConv::supports(((((isize::MAX as usize) / 16) + 4) & !3, 4, 1), 1, 0),
        "weight bytes exceed isize::MAX without usize overflow"
    );
    let g = Grid { scale: 0.05, zero: 10 };
    let cv = QConv::new(&[0.1; 4 * 4 * 9], &[0.0; 4], (4, 4, 3), 1, 1, true, &[g; 4]);
    let x = vec![0u32; SLACK];
    let view = C4 { data: &x, offset: 0, img_stride: 0, c: 4, zero: g.zero };
    let mut out = vec![0u32; SLACK];
    conv(&[view], 0, 8, 8, &cv, None, QOut::U8 { data: &mut out, grid: g });
}

/// Wide (`w >= 25`) tests exercising AVX2 `max_pool`, `upsample2`, `quantize`/`dequantize` (both serial and
/// parallel `>= PAR_MIN`), `k=3` within-row `contiguous` direct-store (with/without residual, single-part and
/// decoupled multi-part), stride-2 `image_rows` (`p=0` and `p=1`), and wide `stem_u8`.
#[test]
fn wide_avx2_helpers_and_k3_direct_store_match_reference() {
    let mut rng = Rng::seed(42);
    // Wide max_pool (k = 3, 5, and huge k = usize::MAX saturating check) + upsample2.
    for (h, w) in [(11usize, 27usize), (27, 35)] {
        let (n, c, hw) = (2, 8, h * w);
        let g = Grid::from_range(-1.2, 3.4);
        let x: Vec<f32> = (0..n * c * hw).map(|_| rng.uniform(-1.5, 3.8)).collect();
        let mut q = alloc(n, c, hw);
        quantize(&x, n, c, hw, g, &mut q);
        let view = C4 { data: &q, offset: 0, img_stride: c / 4 * hw, c, zero: g.zero };
        let mut back = vec![0f32; n * c * hw];
        dequantize(view, n, hw, g, &mut back);
        for (a, b) in x.iter().zip(&back) {
            assert_eq!(g.dequantize(g.quantize(*a)), *b);
        }
        // Per-channel quantize + dequantize on large plane (>= PAR_MIN on 27x35x8x2? 15120 -> also test c=32).
        for k in [3usize, 5, usize::MAX] {
            let mut mp = alloc(n, c, hw);
            max_pool(view, n, h, w, k, &mut mp);
            let r = k / 2;
            for img in 0..n {
                for ch in 0..c {
                    let plane = &back[(img * c + ch) * hw..][..hw];
                    for y in (0..h).step_by(3) {
                        for xo in (0..w).step_by(3) {
                            let mut m = f32::NEG_INFINITY;
                            for yy in y.saturating_sub(r)..=y.saturating_add(r).min(h - 1) {
                                for xx in xo.saturating_sub(r)..=xo.saturating_add(r).min(w - 1) {
                                    m = m.max(plane[yy * w + xx]);
                                }
                            }
                            let got = (mp[(img * c / 4 + ch / 4) * hw + y * w + xo] >> (8 * (ch % 4))) as u8;
                            assert_eq!(
                                g.dequantize(got),
                                m,
                                "max_pool k={k} ({h}x{w}) ({img},{ch},{y},{xo})"
                            );
                        }
                    }
                }
            }
        }
        let mut up = alloc(n, c, 4 * hw);
        upsample2(view, n, h, w, &mut up);
        for img in 0..n {
            for ch in 0..c {
                let plane = &back[(img * c + ch) * hw..][..hw];
                for y in (0..h).step_by(3) {
                    for xo in (0..w).step_by(3) {
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
    // Parallel quantize / quantize_per_channel / dequantize_per_channel (>= PAR_MIN = 32768).
    {
        let (n, c, hw) = (2, 16, 1200);
        let grids: Vec<Grid> =
            (0..c).map(|i| Grid::from_range(-0.5 - 0.1 * i as f32, 2.0 + 0.2 * i as f32)).collect();
        let x: Vec<f32> = (0..n * c * hw).map(|_| rng.uniform(-2.0, 4.0)).collect();
        let mut q = alloc(n, c, hw);
        quantize_per_channel(&x, n, c, hw, &grids, &mut q);
        let view = C4 { data: &q, offset: 0, img_stride: c / 4 * hw, c, zero: grids[0].zero };
        let mut back = vec![0f32; n * c * hw];
        dequantize_per_channel(view, n, hw, &grids, &mut back);
        for img in 0..n {
            for (ch, grid) in grids.iter().enumerate() {
                for p in (0..hw).step_by(17) {
                    let idx = (img * c + ch) * hw + p;
                    assert_eq!(back[idx], grid.dequantize(grid.quantize(x[idx])));
                }
            }
        }
    }
    // Wide k=3 convs hitting within-row contiguous direct-store (ow >= 24) and AVX2 stride-2 (p=0, p=1).
    let base = Case {
        n: 1,
        parts: vec![8],
        o: 8,
        h: 9,
        w: 35,
        k: 3,
        s: 1,
        pad: 1,
        silu: true,
        res: false,
        f32_out: false,
        gap: 0,
    };
    let wide_cases = [
        Case { res: false, ..base.clone() },
        Case { res: true, ..base.clone() },
        Case { res: true, silu: false, w: 27, ..base.clone() },
        Case { s: 2, pad: 1, h: 15, w: 55, ..base.clone() },
        Case { s: 2, pad: 0, h: 15, w: 55, res: true, ..base.clone() },
        Case { s: 2, pad: 1, h: 11, w: 34, parts: vec![4, 8], o: 12, ..base.clone() },
    ];
    for (i, c) in wide_cases.iter().enumerate() {
        run_case(c, 500 + i as u64);
    }
    // Wide decoupled multi-part k=3 (hits `apply_f32_full_u8_direct` with & without residual).
    for has_res in [false, true] {
        let (h, w, o) = (7usize, 31usize, 8usize);
        let hw = h * w;
        let g0 = Grid { scale: 0.01, zero: 15 };
        let g1 = Grid { scale: 0.25, zero: 90 };
        let gout = Grid { scale: 0.05, zero: 32 };
        let gres = Grid { scale: 0.04, zero: 20 };
        let x0 = codes(&mut rng, hw);
        let x1 = codes(&mut rng, 2 * hw);
        let res_buf = codes(&mut rng, o / 4 * hw);
        let wt: Vec<f32> = (0..o * 12 * 9).map(|_| rng.uniform(-0.4, 0.4)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let pg0 = [g0; 4];
        let pg1 = [g1; 8];
        let cv = QConv::new_parts(&wt, &bias, (o, 12, 3), 1, 1, true, &[&pg0, &pg1], true);
        assert_eq!(cv.num_parts(), 2);
        let v0 = C4 { data: &x0, offset: 0, img_stride: hw, c: 4, zero: g0.zero };
        let v1 = C4 { data: &x1, offset: 0, img_stride: 2 * hw, c: 8, zero: g1.zero };
        let rv = has_res.then_some((
            C4 { data: &res_buf, offset: 0, img_stride: o / 4 * hw, c: o, zero: gres.zero },
            gres,
        ));
        let mut got1 = alloc(1, o, hw);
        let mut got8 = alloc(1, o, hw);
        pool(1).install(|| conv(&[v0, v1], 1, h, w, &cv, rv, QOut::U8 { data: &mut got1, grid: gout }));
        pool(8).install(|| conv(&[v0, v1], 1, h, w, &cv, rv, QOut::U8 { data: &mut got8, grid: gout }));
        assert_eq!(got1[..o / 4 * hw], got8[..o / 4 * hw], "multi-part k3 res={has_res}");
    }
    // Wide stem_u8 (w = 54, ow = 27 >= NR: hits AVX2 de-interleave, rotating slots, direct store and tail).
    {
        let (h, w, o) = (14usize, 54usize, 8usize);
        let rgb: Vec<f32> = (0..3 * h * w).map(|_| rng.uniform(0.0, 1.0)).collect();
        let mut rgb0 = alloc(1, 4, h * w);
        pack_f32_to_rgb0_u8(&rgb, 1, h, w, &mut rgb0);
        let wt: Vec<f32> = (0..o * 3 * 9).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let qs = QStem::new(&wt, &bias, (o, 3), 2, 1, true);
        let (oh, ow) = qs.out_hw(h, w);
        assert_eq!((oh, ow), (7, 27));
        let gout = Grid::from_range(-0.3, 4.0);
        let mut out1 = alloc(1, o, oh * ow);
        let mut out4 = alloc(1, o, oh * ow);
        pool(1).install(|| stem_u8(&rgb0, 1, h, w, &qs, QOut::U8 { data: &mut out1, grid: gout }));
        pool(4).install(|| stem_u8(&rgb0, 1, h, w, &qs, QOut::U8 { data: &mut out4, grid: gout }));
        assert_eq!(out1[..o / 4 * oh * ow], out4[..o / 4 * oh * ow]);
    }
}

/// Every public q8 helper must panic cleanly on size overflow instead of wrapping and writing out of bounds.
#[test]
fn q8_public_apis_reject_size_overflow() {
    let g = Grid { scale: 0.1, zero: 0 };
    let big_n = (usize::MAX / 4) + 10_000;
    let x_f = vec![0.0f32; 64];
    let in_q = vec![0u32; 64 + SLACK];
    let mut out_q = vec![0u32; 64 + SLACK];
    let mut out_f = vec![0.0f32; 64];
    let view = C4 { data: &in_q, offset: 0, img_stride: 0, c: 4, zero: 0 };

    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            quantize(&x_f, big_n, 4, 1, g, &mut out_q);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            quantize_per_channel(&x_f, big_n, 4, 1, &[g; 4], &mut out_q);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dequantize(view, big_n, 1, g, &mut out_f);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dequantize_per_channel(view, big_n, 1, &[g; 4], &mut out_f);
        }))
        .is_err()
    );
    let mut out_mp = vec![0u32; 64];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            max_pool(view, usize::MAX, 1, 2, 3, &mut out_mp);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            upsample2(view, big_n, 1, 1, &mut out_mp);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pack_f32_to_rgb0_u8(&x_f, (usize::MAX / 3) + 4, 1, 1, &mut out_mp);
        }))
        .is_err()
    );
}

/// Parallel `q8::max_pool` (`>= PAR_MIN`, both `hw < MIN_POOLED` and `hw >= MIN_POOLED`) and cache-line-padded
/// `image_rows` on non-16-aligned planes (`22×22` and `42×42`, hitting parallel `fill` when `body.len() >= par_words`)
/// match the scalar reference bit-for-bit.
#[test]
fn parallel_max_pool_and_cacheline_padded_image_rows_match_reference() {
    let mut rng = Rng::seed(808);
    // 1. Parallel `max_pool` with `hw = 20*20 = 400 < MIN_POOLED` (`total = 2 * 64 * 400 = 51200 >= PAR_MIN`)
    //    and with `hw = 64*65 = 4160 >= MIN_POOLED` (`total = 2 * 8 * 4160 = 66560 >= PAR_MIN`).
    for (n, c, h, w, k) in [(2usize, 256usize, 20usize, 20usize, 5usize), (2, 32, 64, 65, 3)] {
        let hw = h * w;
        assert!(n * (c / 4) * hw >= crate::cpu::PAR_MIN);
        let q = codes(&mut rng, n * (c / 4) * hw);
        let view = C4 { data: &q, offset: 0, img_stride: (c / 4) * hw, c, zero: 0 };
        let mut par_out = alloc(n, c, hw);
        let mut ser_out = alloc(n, c, hw);
        pool(4).install(|| max_pool(view, n, h, w, k, &mut par_out));
        pool(1).install(|| max_pool(view, n, h, w, k, &mut ser_out));
        assert_eq!(
            par_out[..n * (c / 4) * hw],
            ser_out[..n * (c / 4) * hw],
            "max_pool ({n},{c},{h},{w},k={k})"
        );
    }

    // 2. Non-16-aligned `plane` (`h=40, w=40, pad=1` => `hq=wq=42, plane=1764, plane % 16 == 4`) with enough
    //    groups (`c = 320`, `groups = 80`, `body.len() = 80 * 1776 = 142080 >= par_words`) to exercise parallel
    //    `image_rows` `fill` across adjacent cache-line-padded channel groups, plus `s=2, p=1` (`21×21` phase).
    for (c, h, w, s) in [(320usize, 40usize, 40usize, 1usize), (160, 40, 40, 2)] {
        let o = 8;
        let g_in = Grid { scale: 0.05, zero: 17 };
        let g_out = Grid { scale: 0.1, zero: 33 };
        let wt: Vec<f32> = (0..o * c * 9).map(|_| rng.uniform(-0.2, 0.2)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let cv = QConv::new(&wt, &bias, (o, c, 3), s, 1, true, &vec![g_in; c]);
        let (oh, ow) = cv.out_hw(h, w);
        let x = codes(&mut rng, (c / 4) * h * w);
        let view = C4 { data: &x, offset: 0, img_stride: (c / 4) * h * w, c, zero: g_in.zero };
        let mut out1 = alloc(1, o, oh * ow);
        let mut out4 = alloc(1, o, oh * ow);
        pool(1).install(|| conv(&[view], 1, h, w, &cv, None, QOut::U8 { data: &mut out1, grid: g_out }));
        pool(4).install(|| conv(&[view], 1, h, w, &cv, None, QOut::U8 { data: &mut out4, grid: g_out }));
        assert_eq!(out1[..(o / 4) * oh * ow], out4[..(o / 4) * oh * ow], "conv s={s} c={c} {h}x{w}");
    }
}
