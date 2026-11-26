//! Candidate-first detection kernels: the conv at listed sites against the dense conv, bit for bit (regular
//! kernels in f32 and f16, tensor cores), and the class pass, field lists and decode against
//! [`vision::detect_candidates`] on the same head.

use super::coop::coop_ctx;
use super::{Rng, f16_round, gpu, to_f16};
use crate::kernels::conv::{self, Conv2dDesc, ConvInput};
use crate::kernels::detect_sites::{self, DecodeLevel, SiteLists};
use crate::kernels::pooling::Window2d;
use crate::kernels::vision;
use crate::{DType, GpuContext, GpuTensor, UnaryOp};

fn upload(ctx: &GpuContext, v: &[f32], dtype: DType) -> GpuTensor {
    match dtype {
        DType::F32 => GpuTensor::from_slice(ctx, v, &[v.len()]).unwrap(),
        DType::F16 => GpuTensor::from_f16(ctx, &to_f16(v), &[v.len()]).unwrap(),
    }
}

fn upload_u32(ctx: &GpuContext, v: &[u32]) -> GpuTensor {
    let f: Vec<f32> = v.iter().map(|&u| f32::from_bits(u)).collect();
    GpuTensor::from_slice(ctx, &f, &[f.len().max(1)]).unwrap()
}

/// The stored bits of every element (f16 values widened exactly).
fn bits(t: &GpuTensor, dtype: DType) -> Vec<u32> {
    match dtype {
        DType::F32 => t.to_vec().unwrap().into_iter().map(f32::to_bits).collect(),
        DType::F16 => t.to_f16_vec().unwrap().into_iter().map(|h| u32::from(h.to_bits())).collect(),
    }
}

/// Per image `cap + 1` words: the count, then the listed pixels (the first `cap` of `pixels[img]`).
fn site_rows(pixels: &[Vec<u32>], cap: usize) -> Vec<u32> {
    let mut v = Vec::new();
    for p in pixels {
        v.push(p.len() as u32);
        v.extend(p.iter().take(cap));
        v.extend(std::iter::repeat_n(u32::MAX, cap - p.len().min(cap)));
    }
    v
}

/// `d` (bias, SiLU) densely and at sites: every listed site's values bit for bit.
fn check_sites(ctx: &GpuContext, d: &Conv2dDesc, seed: u64) {
    let mut rng = Rng::new(seed);
    let (kh, kw) = d.win.kernel;
    let (oh, ow) = d.output().unwrap();
    let rnd = |rng: &mut Rng, n: usize, s: f32| {
        let v = rng.vec(n, -s, s);
        if d.dtype == DType::F16 { f16_round(&v) } else { v }
    };
    let x = rnd(&mut rng, d.n * d.c_in * d.h * d.w, 1.0);
    let w = rnd(&mut rng, d.c_out * d.c_in * kh * kw, 0.3);
    let b = rnd(&mut rng, d.c_out, 0.5);
    let (gx, gw, gb) = (upload(ctx, &x, d.dtype), upload(ctx, &w, d.dtype), upload(ctx, &b, d.dtype));
    let hw = oh * ow;
    let dense = GpuTensor::empty(ctx, &[d.n * d.c_out * hw], d.dtype).unwrap();
    conv::conv2d(ctx, d, gx.raw(), gw.raw(), Some(gb.raw()), dense.raw()).unwrap();
    let dense = bits(&dense, d.dtype);
    // Per image a different, shuffled list: fewer sites than slots, as many, more (the excess is not computed),
    // none at all.
    let cap = 37.min(hw);
    let counts = [cap / 2, cap, cap + 5, 0];
    let pixels: Vec<Vec<u32>> = (0..d.n)
        .map(|i| {
            let mut all: Vec<u32> = (0..hw as u32).collect();
            for k in (1..all.len()).rev() {
                all.swap(k, (rng.next_u64() % (k as u64 + 1)) as usize);
            }
            all.truncate(counts[i % counts.len()].min(hw));
            all
        })
        .collect();
    let sites = upload_u32(ctx, &site_rows(&pixels, cap));
    let y = GpuTensor::empty(ctx, &[d.n * d.c_out * cap], d.dtype).unwrap();
    let at = ConvInput { offset: 0, n_stride: d.c_in * d.h * d.w };
    conv::conv2d_sites(ctx, d, (gx.raw(), at), gw.raw(), Some(gb.raw()), sites.raw(), cap, y.raw()).unwrap();
    let got = bits(&y, d.dtype);
    for (img, p) in pixels.iter().enumerate() {
        for (s, &px) in p.iter().take(cap).enumerate() {
            for oc in 0..d.c_out {
                let want = dense[(img * d.c_out + oc) * hw + px as usize];
                let g = got[(img * d.c_out + oc) * cap + s];
                assert_eq!(g, want, "{d:?}: image {img}, slot {s} (pixel {px}), channel {oc}");
            }
        }
    }
}

fn desc(n: usize, c_in: usize, h: usize, w: usize, c_out: usize, win: Window2d, dtype: DType) -> Conv2dDesc {
    Conv2dDesc { n, c_in, h, w, c_out, groups: 1, win, act: UnaryOp::Silu, dtype }
}

/// The box branch's convs (folo-n's P5 and P3 maps, split along K and not), odd maps, a strided conv and a 1×1.
fn shapes(dtype: DType) -> Vec<Conv2dDesc> {
    let k3 = Window2d::square(3, 1, 1);
    vec![
        desc(2, 64, 20, 20, 64, k3, dtype),
        desc(4, 64, 20, 20, 64, k3, dtype),
        desc(1, 64, 80, 80, 64, k3, dtype),
        desc(3, 32, 13, 11, 48, k3, dtype),
        desc(2, 8, 10, 11, 16, Window2d::square(3, 2, 1), dtype),
        desc(2, 64, 9, 7, 64, Window2d::square(1, 1, 0), dtype),
    ]
}

#[test]
fn conv_sites_are_the_dense_values() {
    let ctx = gpu!();
    for (i, d) in shapes(DType::F32).iter().enumerate() {
        check_sites(&ctx, d, 10 + i as u64);
    }
    if ctx.caps().shader_f16 && !ctx.tensor_cores() {
        for (i, d) in shapes(DType::F16).iter().enumerate() {
            check_sites(&ctx, d, 20 + i as u64);
        }
    }
}

#[test]
fn conv_sites_on_tensor_cores_and_off() {
    let Some((ctx, _g)) = coop_ctx() else { return };
    for on in [true, false] {
        ctx.set_tensor_cores(on);
        // A 1×1 conv is the tensor-core GEMM densely: not served at sites.
        for (i, d) in shapes(DType::F16).iter().filter(|d| !on || d.win.kernel != (1, 1)).enumerate() {
            check_sites(&ctx, d, 30 + i as u64);
        }
        // A tensor-core kernel that fails to build turns them off (and the regular kernels serve).
        assert_eq!(ctx.tensor_cores(), on, "{}", ctx.tensor_cores_status());
    }
    ctx.set_tensor_cores(true);
}

/// A random head of `levels` (h, w) maps: class pass, the boxes at the sites (compacted on the host from the
/// dense logits) and the decode give `detect_candidates`' records on the concatenated head; the field lists are
/// the slotted candidates' windows; a level with more candidates than slots is flagged.
fn check_decode(ctx: &GpuContext, dtype: DType, seed: u64) {
    let (b, nc, rm) = (3, 5, 16);
    let levels = [(16usize, 12usize, 8.0f32), (8, 6, 16.0), (4, 3, 32.0)];
    let mut rng = Rng::new(seed);
    let a: usize = levels.iter().map(|l| l.0 * l.1).sum();
    let rnd = |rng: &mut Rng, n: usize, lo: f32, hi: f32| {
        let v = rng.vec(n, lo, hi);
        if dtype == DType::F16 { f16_round(&v) } else { v }
    };
    // Low scores, so a few percent of the anchors are candidates; every P5 anchor of image 2 is one (more than
    // its slots).
    let mut cls_l: Vec<Vec<f32>> =
        levels.iter().map(|l| rnd(&mut rng, b * nc * l.0 * l.1, -14.0, -0.9)).collect();
    let p5 = levels[2].0 * levels[2].1;
    for v in &mut cls_l[2][2 * nc * p5..2 * nc * p5 + p5] {
        *v = 3.0;
    }
    let box_l: Vec<Vec<f32>> =
        levels.iter().map(|l| rnd(&mut rng, b * 4 * rm * l.0 * l.1, -4.0, 4.0)).collect();
    let cat = |parts: &[Vec<f32>], rows: usize| {
        let mut v = Vec::new();
        for img in 0..b {
            for r in 0..rows {
                for (l, p) in parts.iter().enumerate() {
                    let hw = levels[l].0 * levels[l].1;
                    v.extend_from_slice(&p[(img * rows + r) * hw..(img * rows + r + 1) * hw]);
                }
            }
        }
        v
    };
    let (cls, boxes) = (cat(&cls_l, nc), cat(&box_l, 4 * rm));
    let mut anc = Vec::new();
    for k in 0..3 {
        for (l, &(h, w, s)) in levels.iter().enumerate() {
            let _ = l;
            for p in 0..h * w {
                anc.push([(p % w) as f32 + 0.5, (p / w) as f32 + 0.5, s][k]);
            }
        }
    }
    let out = GpuTensor::empty(ctx, &[b * (1 + 7 * a)], DType::F32).unwrap();
    let (gcls, gbox, ganc) =
        (upload(ctx, &cls, dtype), upload(ctx, &boxes, dtype), upload(ctx, &anc, DType::F32));
    let conf = 0.25;
    vision::detect_candidates(ctx, gbox.raw(), gcls.raw(), ganc.raw(), out.raw(), b, a, nc, rm, conf, dtype)
        .unwrap();
    let dense = bits(&out, DType::F32);
    let records = |v: &[u32], n: usize| {
        let mut r: Vec<[u32; 7]> = v[..7 * n].as_chunks::<7>().0.to_vec();
        r.sort_unstable();
        r
    };
    let want: Vec<Vec<[u32; 7]>> = (0..b)
        .map(|i| {
            let row = &dense[i * (1 + 7 * a)..(i + 1) * (1 + 7 * a)];
            records(&row[1..], row[0] as usize)
        })
        .collect();

    let caps = [40, 16, 8];
    let mut lists = Vec::new();
    let mut start = 0;
    for (l, &(h, w, _)) in levels.iter().enumerate() {
        let cap = caps[l];
        let sl = SiteLists { b, h, w, nc, cap, fcap: (9 * cap).min(h * w) };
        let t = |n| GpuTensor::empty(ctx, &[n], DType::F32).unwrap();
        let (sites, recs, field, mark) =
            (t(b * (cap + 1)), t(b * 2 * cap), t(b * (sl.fcap + 1)), t(b * h * w));
        let gc = upload(ctx, &cls_l[l], dtype);
        detect_sites::class_sites(
            ctx,
            gc.raw(),
            &sl,
            conf,
            dtype,
            (sites.raw(), recs.raw(), field.raw(), mark.raw()),
        )
        .unwrap();
        let sv = bits(&sites, DType::F32);
        let fv = bits(&field, DType::F32);
        // The boxes at the slots, gathered from the dense logits.
        let hw = h * w;
        let mut raw = vec![0f32; b * 4 * rm * cap];
        for img in 0..b {
            let row = &sv[img * (cap + 1)..(img + 1) * (cap + 1)];
            let live = (row[0] as usize).min(cap);
            let mut want_field: Vec<u32> = Vec::new();
            for s in 0..live {
                let j = row[1 + s] as usize;
                for r in 0..4 * rm {
                    raw[(img * 4 * rm + r) * cap + s] = box_l[l][(img * 4 * rm + r) * hw + j];
                }
                let (y, x) = (j / w, j % w);
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        want_field.push((yy * w + xx) as u32);
                    }
                }
            }
            want_field.sort_unstable();
            want_field.dedup();
            let frow = &fv[img * (sl.fcap + 1)..(img + 1) * (sl.fcap + 1)];
            let mut got_field = frow[1..1 + frow[0] as usize].to_vec();
            got_field.sort_unstable();
            assert_eq!(got_field, want_field, "level {l}, image {img}: field");
        }
        lists.push((sites, recs, upload(ctx, &raw, dtype), cap, w, levels[l].2, start));
        start += hw;
    }
    let total = detect_sites::HEADER_WORDS + 7 * caps.iter().sum::<usize>();
    let out = GpuTensor::empty(ctx, &[b * total], DType::F32).unwrap();
    let dl: Vec<DecodeLevel<'_>> = lists
        .iter()
        .map(|(s, r, raw, cap, w, st, start)| DecodeLevel {
            raw: raw.raw(),
            sites: s.raw(),
            recs: r.raw(),
            cap: *cap,
            w: *w,
            stride: *st,
            start: *start,
        })
        .collect();
    detect_sites::decode_sites(ctx, &dl, b, rm, dtype, out.raw()).unwrap();
    let got = bits(&out, DType::F32);
    for img in 0..b {
        let row = &got[img * total..(img + 1) * total];
        if img == 2 {
            assert_eq!(row[1], 1 << 2, "image 2's P5 overflows");
            continue;
        }
        assert_eq!(row[1], 0, "image {img}: no level overflows");
        assert!(!want[img].is_empty(), "image {img} has candidates");
        assert_eq!(records(&row[2..], row[0] as usize), want[img], "image {img}: records");
    }
}

#[test]
fn class_pass_and_decode_match_detect_candidates() {
    let ctx = gpu!();
    check_decode(&ctx, DType::F32, 1);
    if ctx.caps().shader_f16 {
        check_decode(&ctx, DType::F16, 2);
    }
}

/// Runs both `detect_candidates` and `class_sites` + `decode_sites` on a single-level `[1, nc, h, w]` head and
/// asserts they produce bit-identical sorted `(anchor, class, score_bits)` triples, returning them.
fn run_head_both(
    ctx: &GpuContext,
    cls_per_anchor: &[Vec<f32>],
    h: usize,
    w: usize,
    conf: f32,
    dtype: DType,
) -> Vec<(u32, u32, u32)> {
    let a = h * w;
    let nc = cls_per_anchor[0].len();
    assert_eq!(cls_per_anchor.len(), a);
    let mut cls = vec![0f32; nc * a];
    for (p, row) in cls_per_anchor.iter().enumerate() {
        assert_eq!(row.len(), nc);
        for (c, &v) in row.iter().enumerate() {
            cls[c * a + p] = v;
        }
    }
    let rm = 4;
    let boxes = vec![0f32; 4 * rm * a];
    let mut anc = vec![0f32; 3 * a];
    for p in 0..a {
        anc[p] = (p % w) as f32 + 0.5;
        anc[a + p] = (p / w) as f32 + 0.5;
        anc[2 * a + p] = 8.0;
    }
    let (gcls, gbox, ganc) =
        (upload(ctx, &cls, dtype), upload(ctx, &boxes, dtype), upload(ctx, &anc, DType::F32));
    let out = GpuTensor::empty(ctx, &[1 + 7 * a], DType::F32).unwrap();
    vision::detect_candidates(ctx, gbox.raw(), gcls.raw(), ganc.raw(), out.raw(), 1, a, nc, rm, conf, dtype)
        .unwrap();
    let dense = bits(&out, DType::F32);
    let mut want: Vec<[u32; 7]> = dense[1..1 + 7 * dense[0] as usize].as_chunks::<7>().0.to_vec();
    want.sort_unstable();

    let cap = a;
    let sl = SiteLists { b: 1, h, w, nc, cap, fcap: a };
    let t = |n| GpuTensor::empty(ctx, &[n], DType::F32).unwrap();
    let (sites, recs, field, mark) = (t(cap + 1), t(2 * cap), t(sl.fcap + 1), t(a));
    detect_sites::class_sites(
        ctx,
        gcls.raw(),
        &sl,
        conf,
        dtype,
        (sites.raw(), recs.raw(), field.raw(), mark.raw()),
    )
    .unwrap();
    let raw = upload(ctx, &vec![0f32; 4 * rm * cap], dtype);
    let dl =
        [DecodeLevel { raw: raw.raw(), sites: sites.raw(), recs: recs.raw(), cap, w, stride: 8.0, start: 0 }];
    let total = detect_sites::HEADER_WORDS + 7 * cap;
    let sout = GpuTensor::empty(ctx, &[total], DType::F32).unwrap();
    detect_sites::decode_sites(ctx, &dl, 1, rm, dtype, sout.raw()).unwrap();
    let sbits = bits(&sout, DType::F32);
    assert_eq!(sbits[1], 0, "{dtype:?}: unexpected overflow");
    let mut got: Vec<[u32; 7]> = sbits[2..2 + 7 * sbits[0] as usize].as_chunks::<7>().0.to_vec();
    got.sort_unstable();
    assert_eq!(got, want, "{dtype:?}: sites_class+decode diverged from detect_candidates");
    want.into_iter().map(|r| (r[0], r[1], r[2])).collect()
}

/// Edge cases for `detect.wgsl` and `sites_class.wgsl`: NaN at c=0 and c>0, ±inf, large-logit sigmoid
/// saturation ties, distinct near-saturated rounding ties, signed-zero ties, negative conf with `-inf` ties,
/// single-class heads, and NaN conf, in both f32 and f16.
fn check_edge_cases(ctx: &GpuContext, dtype: DType) {
    let inf = f32::INFINITY;
    let ninf = f32::NEG_INFINITY;
    let nan = f32::NAN;
    let rows: Vec<Vec<f32>> = vec![
        // 0: large-logit saturation tie starting at c=0 -> class 0, score 1.0
        vec![20.0, 25.0, -5.0, -5.0, -5.0],
        // 1: saturation tie starting at c=1, with larger logits later -> class 1, score 1.0
        vec![-5.0, 20.0, 10.0, 25.0, 30.0],
        // 2: duplicate saturating logits before strictly larger -> class 1, score 1.0
        vec![-5.0, 20.0, 20.0, 25.0, -5.0],
        // 3: saturating finite logit before +inf -> class 0, score 1.0
        vec![20.0, inf, -5.0, -5.0, -5.0],
        // 4: +inf tie -> class 1, score 1.0
        vec![-5.0, inf, inf, -5.0, -5.0],
        // 5: NaN at c=0 poisons the anchor -> dropped
        vec![nan, 5.0, 20.0, inf, 10.0],
        // 6: NaN at c>0 is skipped -> class 2
        vec![-5.0, nan, 3.0, nan, 1.0],
        // 7: NaN between saturating logits -> class 1, score 1.0
        vec![-5.0, 20.0, nan, 25.0, -5.0],
        // 8: -inf at c=0 followed by finite winner -> class 2
        vec![ninf, ninf, 2.0, -5.0, -5.0],
        // 9: all below conf=0.25 -> dropped
        vec![ninf, -100.0, -50.0, -20.0, -10.0],
        // 10: +0.0 vs -0.0 tie -> class 0, score 0.5
        vec![0.0, -0.0, -1.0, -2.0, -3.0],
        // 11: -0.0 vs +0.0 tie -> class 0, score 0.5
        vec![-0.0, 0.0, -1.0, -2.0, -3.0],
        // 12: exact duplicate winning logit -> class 1, score 0.5
        vec![-5.0, 0.0, 0.0, -1.0, -2.0],
        // 13: distinct near-saturated logits (both representable in f16) whose f32 sigmoid ties -> class 1
        vec![-5.0, 16.0, 16.015625, -5.0, -5.0],
        // 14: winner at last class, no tie -> class 4
        vec![-5.0, -5.0, -5.0, -5.0, 2.0],
        // 15: NaN after a winning c=0 must not overwrite it -> class 0
        vec![2.0, nan, -5.0, -5.0, -5.0],
        // 16: NaN before a saturation tie during the 0..best rescan -> class 2, score 1.0
        vec![-5.0, nan, 20.0, 25.0, -5.0],
    ];
    let got = run_head_both(ctx, &rows, 1, rows.len(), 0.25, dtype);
    let anchors_and_classes: Vec<(u32, u32)> = got.iter().map(|&(a, c, _)| (a, c)).collect();
    assert_eq!(
        anchors_and_classes,
        vec![
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 0),
            (4, 1),
            (6, 2),
            (7, 1),
            (8, 2),
            (10, 0),
            (11, 0),
            (12, 1),
            (13, 1),
            (14, 4),
            (15, 0),
            (16, 2),
        ],
        "{dtype:?}: candidate (anchor, class) mismatch"
    );
    let score_of = |anchor: u32| f32::from_bits(got.iter().find(|r| r.0 == anchor).unwrap().2);
    for a in [0, 1, 2, 3, 4, 7, 16] {
        assert_eq!(score_of(a), 1.0, "{dtype:?}: anchor {a} expected 1.0");
    }
    for a in [10, 11, 12] {
        assert_eq!(score_of(a), 0.5, "{dtype:?}: anchor {a} expected 0.5");
    }
    assert!((score_of(6) - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-6);
    assert!((score_of(8) - 1.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-6);
    assert!((score_of(14) - 1.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-6);

    // Negative conf: -inf and large negative logits tie at sig == 0.0 -> first class wins; c=0 NaN still drops.
    let neg_rows = vec![vec![ninf, -100.0, -90.0], vec![-100.0, ninf, -90.0], vec![nan, -100.0, 5.0]];
    let neg_got = run_head_both(ctx, &neg_rows, 1, 3, -0.1, dtype);
    assert_eq!(neg_got, vec![(0, 0, 0.0f32.to_bits()), (1, 0, 0.0f32.to_bits())], "{dtype:?}: neg conf");

    // Single class (nc = 1) and NaN conf.
    let one_cls = vec![vec![2.0], vec![-5.0], vec![nan], vec![inf]];
    let one_got = run_head_both(ctx, &one_cls, 2, 2, 0.25, dtype);
    assert_eq!(
        one_got.iter().map(|&(a, c, _)| (a, c)).collect::<Vec<_>>(),
        vec![(0, 0), (3, 0)],
        "{dtype:?}: nc=1"
    );
    assert_eq!(one_got[1].2, 1.0f32.to_bits());
    assert!(run_head_both(ctx, &one_cls, 2, 2, nan, dtype).is_empty(), "{dtype:?}: nan conf");

    // Partial NaN in box logits must poison that decoded coordinate to NaN (not be masked by finite sibling bins).
    let rm = 4;
    let cls_ok = upload(ctx, &[5.0f32], dtype);
    let mut box_bad = vec![0.0f32; 4 * rm];
    box_bad[1] = nan; // side 0 (left) bin 1 is NaN; bins 0,2,3 and sides 1..3 are finite
    let gbox_bad = upload(ctx, &box_bad, dtype);
    let ganc_one = upload(ctx, &[0.5f32, 0.5, 8.0], DType::F32);
    let out_dense = GpuTensor::empty(ctx, &[8], DType::F32).unwrap();
    vision::detect_candidates(
        ctx,
        gbox_bad.raw(),
        cls_ok.raw(),
        ganc_one.raw(),
        out_dense.raw(),
        1,
        1,
        1,
        rm,
        0.25,
        dtype,
    )
    .unwrap();
    let dbits = bits(&out_dense, DType::F32);
    assert_eq!(dbits[0], 1, "{dtype:?}: partial-NaN box candidate should emit");
    assert!(
        f32::from_bits(dbits[4]).is_nan(),
        "{dtype:?}: detect x1 must be NaN, got {}",
        f32::from_bits(dbits[4])
    );
    assert!(f32::from_bits(dbits[5]).is_finite(), "{dtype:?}: detect y1 should stay finite");

    let sl = SiteLists { b: 1, h: 1, w: 1, nc: 1, cap: 1, fcap: 1 };
    let t = |n| GpuTensor::empty(ctx, &[n], DType::F32).unwrap();
    let (sites, recs, field, mark) = (t(2), t(2), t(2), t(1));
    detect_sites::class_sites(
        ctx,
        cls_ok.raw(),
        &sl,
        0.25,
        dtype,
        (sites.raw(), recs.raw(), field.raw(), mark.raw()),
    )
    .unwrap();
    let sout = GpuTensor::empty(ctx, &[detect_sites::HEADER_WORDS + 7], DType::F32).unwrap();
    let dl = [DecodeLevel {
        raw: gbox_bad.raw(),
        sites: sites.raw(),
        recs: recs.raw(),
        cap: 1,
        w: 1,
        stride: 8.0,
        start: 0,
    }];
    detect_sites::decode_sites(ctx, &dl, 1, rm, dtype, sout.raw()).unwrap();
    let sbits = bits(&sout, DType::F32);
    assert_eq!(sbits[0], 1, "{dtype:?}: sites partial-NaN candidate should emit");
    assert!(
        f32::from_bits(sbits[5]).is_nan(),
        "{dtype:?}: sites_decode x1 must be NaN, got {}",
        f32::from_bits(sbits[5])
    );
    assert!(f32::from_bits(sbits[6]).is_finite(), "{dtype:?}: sites_decode y1 should stay finite");
}

#[test]
fn detect_and_class_sites_edge_cases() {
    let ctx = gpu!();
    check_edge_cases(&ctx, DType::F32);
    if ctx.caps().shader_f16 {
        check_edge_cases(&ctx, DType::F16);
    }
}
