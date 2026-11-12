//! Convs that read their operands in place: a 1×1 conv over a channel concatenation (`conv2d_act_cat`, the parts
//! never copied into one tensor) and any non-depthwise conv on a channel slice (`narrow` along dim 1), against the
//! CPU backend. The routing checks trace the operations a call encodes: an in-place read is one GEMM on the
//! parts' own buffers, with no copy or gather of them first.

use std::sync::Mutex;

use folo_core::backend::{Conv2dParams, UnaryOp, View};
use folo_core::{DType, Device, Error, Result, Storage, Tensor, no_grad};

use super::fixtures::{assert_close_t, dev, randn};
use crate::backend::storage_buffer;
use crate::kernels::gemm::{BatchStrides, GemmBias, GemmDesc, TileConfig};
use crate::kernels::gemm_cat::{CatPart, gemm_cat};
use crate::tests::gemm::cpu_gemm;
use crate::tests::{Rng, assert_close, gpu};
use crate::{DType as GpuDType, GpuTensor};

/// One test traces at a time: [`crate::GpuContext::trace_begin`] restarts the shared context's trace.
static TRACE: Mutex<()> = Mutex::new(());

/// Runs `f` and returns its result with the names of the operations it encoded that touch one of `ts`' buffers
/// (other tests share the context and encode meanwhile, but on their own buffers).
fn traced<R>(ts: &[&Tensor], f: impl FnOnce() -> R) -> (R, Vec<&'static str>) {
    let ctx = crate::tests::ctx().expect("a GPU test");
    let bufs: Vec<wgpu::Buffer> = ts.iter().map(|t| storage_buffer(t)).collect();
    let _g = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    ctx.trace_begin();
    let r = f();
    let ops = ctx.trace_end();
    (r, ops.iter().filter(|o| bufs.iter().any(|b| o.touches(b))).map(|o| o.name).collect())
}

/// Whether `ops` is the single in-place GEMM over a concatenation (regular or tensor-core kernel).
fn one_cat_gemm(ops: &[&str]) -> bool {
    matches!(ops, ["gemm_cat" | "coop_gemm_cat"])
}

/// `t` rounded through f16 when the test runs in f16, so the CPU reference sees the GPU's inputs.
fn rounded(t: Tensor, dtype: DType) -> Tensor {
    if dtype == DType::F16 { t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap() } else { t }
}

/// `f` on views of all of `ts`, in order.
fn with_views<R>(ts: &[&Tensor], views: Vec<View<'_>>, f: impl FnOnce(&[View<'_>]) -> R) -> R {
    match ts.split_first() {
        None => f(&views),
        Some((t, rest)) => t.with_view(|v| {
            let mut views = views;
            views.push(v);
            with_views(rest, views, f)
        }),
    }
}

/// The GPU backend's `conv2d_act_cat` called directly (not through `Tensor`, which would fall back to `cat` then
/// a conv): `Ok` means the backend ran it without concatenating.
fn direct(xs: &[&Tensor], w: &Tensor, b: &Tensor, p: &Conv2dParams, act: Option<UnaryOp>) -> Result<Storage> {
    let be = folo_core::backend::backend(xs[0].device())?;
    let mut ts = xs.to_vec();
    ts.extend([w, b]);
    let k = xs.len();
    with_views(&ts, Vec::new(), |v| be.conv2d_act_cat(&v[..k], v[k], Some(v[k + 1]), p, act))
}

/// The CPU's `conv2d_act_cat` of GPU tensors.
fn cpu_cat(xs: &[&Tensor], w: &Tensor, b: &Tensor, p: &Conv2dParams, act: Option<UnaryOp>) -> Tensor {
    let cpu = |t: &Tensor| t.to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
    let parts: Vec<Tensor> = xs.iter().map(|t| cpu(t)).collect();
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::conv2d_act_cat(&refs, &cpu(w), Some(&cpu(b)), p, act).unwrap()
}

/// Parts of 2 to 4 tensors, some whole and some channel slices of a larger tensor (offset, larger image stride),
/// batch 1 and 2, `H·W` a multiple of 4 (vector loads) and not, f32 and f16: the backend hook's own output and
/// `Tensor::conv2d_act_cat` both match the CPU, and each is one GEMM on the parts' buffers.
#[test]
fn pointwise_conv_reads_a_concatenation_in_place() {
    let dev = dev!();
    let _g = no_grad();
    let act = Some(UnaryOp::Silu);
    let p = Conv2dParams::default();
    for dtype in [DType::F32, DType::F16] {
        if dtype == DType::F16 && !crate::tests::ctx().is_some_and(|c| c.caps().shader_f16) {
            continue;
        }
        let (atol, rtol) = if dtype == DType::F16 { (1e-2, 1e-2) } else { (1e-4, 1e-4) };
        for (n, h, w) in [(1, 4, 5), (2, 4, 5), (2, 3, 5), (1, 20, 20)] {
            let to = |t: Tensor| rounded(t, dtype).to_device(dev).unwrap().to_dtype(dtype).unwrap();
            // The slices stay slices on the GPU: `narrow` of the uploaded larger tensors.
            let (big, whole, other) =
                (to(randn(&[n, 12, h, w], 1)), to(randn(&[n, 3, h, w], 2)), to(randn(&[n, 8, h, w], 3)));
            let s =
                [big.narrow(1, 2, 5).unwrap(), big.narrow(1, 9, 3).unwrap(), other.narrow(1, 0, 4).unwrap()];
            let cases: [Vec<&Tensor>; 3] =
                [vec![&s[0], &whole], vec![&whole, &s[1], &s[0]], vec![&s[0], &whole, &s[2], &s[1]]];
            for (i, parts) in cases.iter().enumerate() {
                let c: usize = parts.iter().map(|t| t.dims()[1]).sum();
                let wt = to(randn(&[7, c, 1, 1], 10 + i as u64));
                let bias = to(randn(&[7], 20 + i as u64));
                let what = format!("{dtype:?} n{n} {h}x{w} case {i}");
                let cpu = cpu_cat(parts, &wt, &bias, &p, act);
                let (out, ops) = traced(parts, || direct(parts, &wt, &bias, &p, act));
                assert!(one_cat_gemm(&ops), "{what}: the hook encoded {ops:?}");
                let got = Tensor::from_storage(out.expect("read in place"), [n, 7, h, w]).unwrap();
                assert_close_t(
                    &got.to_dtype(DType::F32).unwrap(),
                    &cpu,
                    atol,
                    rtol,
                    &format!("{what} (hook)"),
                );
                let (gpu, ops) = traced(parts, || Tensor::conv2d_act_cat(parts, &wt, Some(&bias), &p, act));
                assert!(one_cat_gemm(&ops), "{what}: Tensor::conv2d_act_cat encoded {ops:?}");
                let gpu = gpu.unwrap();
                assert!(gpu.device() == dev, "{what}");
                assert_close_t(&gpu.to_dtype(DType::F32).unwrap(), &cpu, atol, rtol, &what);
            }
        }
    }
}

/// What the in-place concatenation refuses with `NotSupported` (the caller then concatenates first): 3×3
/// kernels, groups, five parts, mismatched sizes and parts whose images are not contiguous. Through `Tensor`
/// every one of them that is a valid concatenation still gives the concatenation's conv.
#[test]
fn concatenations_it_cannot_read_in_place_fall_back() {
    let dev = dev!();
    let _g = no_grad();
    let to = |t: &Tensor| t.to_device(dev).unwrap();
    let a = to(&randn(&[2, 4, 6, 6], 1));
    let b = to(&randn(&[2, 4, 6, 6], 2));
    let e = to(&randn(&[2, 2, 6, 6], 8));
    let p = Conv2dParams::default();
    let grouped = Conv2dParams { groups: 2, ..p };
    // A transposed (H, W) part: its images are not contiguous.
    let t = b.transpose(2, 3).unwrap();
    let w = |c_out: usize, c_in: usize, k: usize, seed: u64| to(&randn(&[c_out, c_in, k, k], seed));
    let bias = |c_out: usize, seed: u64| to(&randn(&[c_out], seed));
    let cases: [(&str, Vec<&Tensor>, Tensor, Tensor, Conv2dParams); 4] = [
        ("3x3", vec![&a, &b], w(5, 8, 3, 4), bias(5, 3), p),
        ("grouped", vec![&a, &b], w(4, 4, 1, 5), bias(4, 6), grouped),
        ("strided part", vec![&a, &t], w(5, 8, 1, 7), bias(5, 3), p),
        ("five parts", vec![&a, &b, &e, &a, &e], w(5, 16, 1, 9), bias(5, 3), p),
    ];
    for (what, parts, wt, bt, pp) in &cases {
        let err = direct(parts, wt, bt, pp, None).err();
        assert!(matches!(err, Some(Error::NotSupported { .. })), "{what}: {err:?}");
        let gpu = Tensor::conv2d_act_cat(parts, wt, Some(bt), pp, None).unwrap();
        assert!(gpu.device() == dev, "{what}");
        assert_close_t(&gpu, &cpu_cat(parts, wt, bt, pp, None), 1e-4, 1e-4, what);
    }
    // Parts that do not concatenate (another batch or image size) are refused before anything runs.
    let (w1, b1) = (w(5, 8, 1, 7), bias(5, 3));
    for (what, other) in [("batch", to(&randn(&[1, 4, 6, 6], 10))), ("size", to(&randn(&[2, 4, 5, 6], 11)))] {
        let err = direct(&[&a, &other], &w1, &b1, &p, None).err();
        assert!(matches!(err, Some(Error::NotSupported { .. })), "mismatched {what}: {err:?}");
    }
}

/// Empty work: an empty batch gives an empty output, through the concatenation hook and the plain conv (an
/// empty image is an error before any kernel, as for the plain conv); parts without channels are left out (one
/// part left is the plain conv on it). `gemm_cat_tile_multiples_bias_strides_and_misuse` covers empty GEMMs.
#[test]
fn empty_batches_images_and_parts() {
    let dev = dev!();
    let _g = no_grad();
    let to = |t: &Tensor| t.to_device(dev).unwrap();
    let p = Conv2dParams::default();
    let (w4, b4) = (to(&randn(&[4, 4, 1, 1], 1)), to(&randn(&[4], 2)));
    for dims in [[0, 2, 3, 3], [0, 2, 4, 4]] {
        let (x, y) = (to(&randn(&dims, 3)), to(&randn(&dims, 4)));
        let want = [dims[0], 4, dims[2], dims[3]];
        let got = Tensor::from_storage(direct(&[&x, &y], &w4, &b4, &p, None).unwrap(), want).unwrap();
        assert_eq!(got.dims(), want);
        let got = Tensor::conv2d_act_cat(&[&x, &y], &w4, Some(&b4), &p, Some(UnaryOp::Silu)).unwrap();
        assert_eq!((got.dims(), got.numel()), (&want[..], 0), "cat {dims:?}");
    }
    // The plain conv on an empty batch: dense, and a channel slice read in place.
    let (w3, b3) = (to(&randn(&[5, 3, 3, 3], 5)), to(&randn(&[5], 6)));
    let pad = Conv2dParams { padding: [1, 1], ..p };
    let x = to(&randn(&[0, 6, 5, 7], 7));
    for (what, x) in
        [("dense", x.narrow(1, 0, 3).unwrap().contiguous().unwrap()), ("slice", x.narrow(1, 2, 3).unwrap())]
    {
        let got = x.conv2d_act(&w3, Some(&b3), &pad, Some(UnaryOp::Silu)).unwrap();
        assert_eq!((got.dims(), got.numel()), (&[0, 5, 5, 7][..], 0), "{what}");
    }
    // Empty parts: left out, so one GEMM over the others or the plain conv on the one left.
    let none = to(&randn(&[1, 0, 4, 4], 10));
    let (two, more) = (to(&randn(&[1, 2, 4, 4], 11)), to(&randn(&[1, 3, 4, 4], 12)));
    let (w2, b2) = (to(&randn(&[3, 2, 1, 1], 8)), to(&randn(&[3], 9)));
    let (w5, b5) = (to(&randn(&[4, 5, 1, 1], 13)), to(&randn(&[4], 14)));
    let cases: [(Vec<&Tensor>, &Tensor, &Tensor, &[&str]); 2] = [
        (vec![&none, &two], &w2, &b2, &["gemm", "coop_gemm"]),
        (vec![&two, &none, &more, &none], &w5, &b5, &["gemm_cat", "coop_gemm_cat"]),
    ];
    for (i, (parts, wt, bt, kernels)) in cases.into_iter().enumerate() {
        let (out, ops) = traced(&[&two, &more], || direct(&parts, wt, bt, &p, None));
        assert!(matches!(&ops[..], [k] if kernels.contains(k)), "empty parts {i}: {ops:?}");
        let got = Tensor::from_storage(out.unwrap(), [1, wt.dims()[0], 4, 4]).unwrap();
        assert_close_t(&got, &cpu_cat(&parts, wt, bt, &p, None), 1e-4, 1e-4, &format!("empty parts {i}"));
    }
}

/// A conv whose input is a channel slice reads it in place (1×1 and 3×3, batch 1 and 2, f32 and f16): the conv
/// is the only operation on the larger tensor's buffer, no copy or gather first.
#[test]
fn conv_reads_a_channel_slice_in_place() {
    let dev = dev!();
    let _g = no_grad();
    for dtype in [DType::F32, DType::F16] {
        if dtype == DType::F16 && !crate::tests::ctx().is_some_and(|c| c.caps().shader_f16) {
            continue;
        }
        for (n, k, pad) in [(1, 1, 0), (2, 1, 0), (1, 3, 1), (2, 3, 1)] {
            let big = rounded(randn(&[n, 10, 7, 9], 30), dtype);
            let slice = big.narrow(1, 3, 6).unwrap();
            let wt = rounded(randn(&[5, 6, k, k], 31), dtype);
            let bias = rounded(randn(&[5], 32), dtype);
            let p = Conv2dParams { padding: [pad, pad], ..Conv2dParams::default() };
            let act = Some(UnaryOp::Silu);
            let cpu = slice.conv2d_act(&wt, Some(&bias), &p, act).unwrap();
            let to = |t: &Tensor| t.to_device(dev).unwrap().to_dtype(dtype).unwrap();
            let gbig = to(&big);
            let gslice = gbig.narrow(1, 3, 6).unwrap();
            let at =
                gslice.with_view(|v| crate::backend::conv_input(&v)).expect("a channel slice is addressable");
            assert_eq!(at.offset, 3 * 7 * 9);
            let (gw, gb) = (to(&wt), to(&bias));
            let what = format!("{dtype:?} n{n} k{k}");
            let (gpu, ops) = traced(&[&gbig], || gslice.conv2d_act(&gw, Some(&gb), &p, act));
            assert!(
                matches!(&ops[..], [k] if !["copy", "move", "scatter", "cast_any"].contains(k)),
                "{what}: the slice was not read in place ({ops:?})"
            );
            let (atol, rtol) = if dtype == DType::F16 { (1e-2, 1e-2) } else { (1e-4, 1e-4) };
            assert_close_t(&gpu.unwrap().to_dtype(DType::F32).unwrap(), &cpu, atol, rtol, &what);
        }
    }
}

/// `gemm_cat` at the kernel level: a problem whose M, N, K are multiples of the tile (no bounds checks in the
/// kernel, `CHECK` off) with batch-strided parts and bias against the CPU GEMM, a bias too short for its batch
/// stride, an output that aliases a part, a part without rows, and empty work.
#[test]
fn gemm_cat_tile_multiples_bias_strides_and_misuse() {
    let ctx = gpu!();
    let cfg = TileConfig::SMALL;
    let (m, n, batch) = (cfg.bm as usize, 2 * cfg.bn as usize, 2);
    let rows = [cfg.bk as usize, cfg.bk as usize];
    let k: usize = rows.iter().sum();
    let mut rng = Rng::new(9);
    let mut d = GemmDesc::new(m, n, k, false, false, batch);
    d.a = BatchStrides { offset: 0, outer: 0, inner: 0 };
    d.bias = GemmBias::Row;
    d.bias_strides = BatchStrides { offset: 3, outer: m, inner: 0 };
    let a = rng.vec(m * k, -1.0, 1.0);
    let bias = rng.vec(3 + batch * m, -1.0, 1.0);
    // Part 0 starts 4 elements in and holds 8 spare rows per image; part 1 is packed.
    let (off0, outer0) = (4, (rows[0] + 8) * n);
    let p0 = rng.vec(off0 + batch * outer0, -1.0, 1.0);
    let p1 = rng.vec(batch * rows[1] * n, -1.0, 1.0);
    let mut dense = Vec::with_capacity(batch * k * n);
    for b1 in 0..batch {
        dense.extend_from_slice(&p0[off0 + b1 * outer0..][..rows[0] * n]);
        dense.extend_from_slice(&p1[b1 * rows[1] * n..][..rows[1] * n]);
    }
    let want = cpu_gemm(&d, &a, &dense, Some(&bias));
    let up = |v: &[f32]| GpuTensor::from_slice(&ctx, v, &[v.len()]).unwrap();
    let (ga, gbias, g0, g1) = (up(&a), up(&bias), up(&p0), up(&p1));
    let gc = GpuTensor::zeros(&ctx, &[want.len()], GpuDType::F32).unwrap();
    let parts = [
        CatPart { buf: g0.raw(), rows: rows[0], offset: off0, outer: outer0 },
        CatPart { buf: g1.raw(), rows: rows[1], offset: 0, outer: rows[1] * n },
    ];
    gemm_cat(&ctx, &d, ga.raw(), &parts, Some(gbias.raw()), gc.raw(), Some(cfg)).unwrap();
    assert_close(&gc.to_vec().unwrap(), &want, 1e-4, 1e-5, "gemm_cat on tile multiples");

    // The kernel reads the bias at `offset + b1·outer + row`: a buffer that only covers batch 0 is refused.
    let short = up(&bias[..3 + m]);
    assert!(
        gemm_cat(&ctx, &d, ga.raw(), &parts, Some(short.raw()), gc.raw(), Some(cfg)).is_err(),
        "bias extent"
    );
    let err = gemm_cat(&ctx, &d, ga.raw(), &parts, Some(gbias.raw()), g1.raw(), Some(cfg)).unwrap_err();
    assert!(err.to_string().contains("must not share a buffer"), "aliasing: {err}");
    let empty = [parts[0], CatPart { rows: 0, ..parts[1] }, parts[1]];
    assert!(
        gemm_cat(&ctx, &d, ga.raw(), &empty, Some(gbias.raw()), gc.raw(), Some(cfg)).is_err(),
        "empty part"
    );
    // No work: nothing is checked against the buffers and nothing runs.
    let tiny = GpuTensor::zeros(&ctx, &[1], GpuDType::F32).unwrap();
    for (bm, bn, bb) in [(0, n, batch), (m, 0, batch), (m, n, 0)] {
        let e = GemmDesc { m: bm, n: bn, batch: (bb, 1), ..d.clone() };
        gemm_cat(&ctx, &e, tiny.raw(), &parts, Some(tiny.raw()), tiny.raw(), None).unwrap();
    }
}

/// `conv2d_act_add`: the residual added in the conv's epilogue (3×3 implicit GEMM and 1×1 GEMM; dense, channel
/// slice and odd-offset residuals; vector and scalar stores), against the CPU and, in f32, to an ulp against the
/// GPU's own conv then add.
#[test]
fn conv_adds_a_residual_in_its_epilogue() {
    let dev = dev!();
    let _g = no_grad();
    let act = Some(UnaryOp::Silu);
    for dtype in [DType::F32, DType::F16] {
        if dtype == DType::F16 && !crate::tests::ctx().is_some_and(|c| c.caps().shader_f16) {
            continue;
        }
        for (n, h, w, k) in [(1, 8, 8, 3), (2, 8, 8, 3), (2, 5, 7, 3), (1, 8, 8, 1), (2, 5, 7, 1)] {
            let pad = k / 2;
            let p = Conv2dParams { padding: [pad, pad], ..Conv2dParams::default() };
            let x = rounded(randn(&[n, 6, h, w], 40), dtype);
            let wt = rounded(randn(&[6, 6, k, k], 41), dtype);
            let bias = rounded(randn(&[6], 42), dtype);
            // The residual four ways: dense, a channel slice, at an odd offset (scalar epilogue), and with strided
            // images (not addressable: the add runs after the conv).
            let big = rounded(randn(&[n, 9, h, w], 43), dtype);
            let flat = rounded(randn(&[n * 6 * h * w + 1], 44), dtype);
            let wide = rounded(randn(&[n, 6, h, w * 2 + 1], 46), dtype);
            for (ri, which) in ["dense", "slice", "odd", "strided"].iter().enumerate() {
                let pick = |t: &Tensor, big: &Tensor, flat: &Tensor, wide: &Tensor| -> Tensor {
                    match ri {
                        0 => t.clone(),
                        1 => big.narrow(1, 2, 6).unwrap(),
                        2 => flat.narrow(0, 1, n * 6 * h * w).unwrap().reshape([n, 6, h, w]).unwrap(),
                        _ => wide.narrow(3, 1, w).unwrap(),
                    }
                };
                let r_dense = rounded(randn(&[n, 6, h, w], 45), dtype);
                let r = pick(&r_dense, &big, &flat, &wide);
                let cpu = x.conv2d_act_add(&wt, Some(&bias), &p, act, &r).unwrap();
                let to = |t: &Tensor| t.to_device(dev).unwrap().to_dtype(dtype).unwrap();
                let gr = pick(&to(&r_dense), &to(&big), &to(&flat), &to(&wide));
                let (gx, gw, gb) = (to(&x), to(&wt), to(&bias));
                let what = format!("{dtype:?} n{n} {h}x{w} k{k} residual {which}");
                let gpu = gx.conv2d_act_add(&gw, Some(&gb), &p, act, &gr).unwrap();
                let direct = folo_core::backend::backend(dev).unwrap();
                let fused = gx.with_view(|xv| {
                    gw.with_view(|wv| {
                        gb.with_view(|bv| {
                            gr.with_view(|rv| direct.conv2d_act_add(xv, wv, Some(bv), &p, act, rv))
                        })
                    })
                });
                assert_eq!(fused.is_ok(), ri < 3, "{what}: read in place");
                let (atol, rtol) = if dtype == DType::F16 { (1e-2, 1e-2) } else { (1e-4, 1e-4) };
                assert_close_t(&gpu.to_dtype(DType::F32).unwrap(), &cpu, atol, rtol, &what);
                if dtype == DType::F32 {
                    // The GPU's own conv then add, to an ulp: the driver may contract `act(v) + r` into an FMA.
                    let composite = gr.add(&gx.conv2d_act(&gw, Some(&gb), &p, act).unwrap()).unwrap();
                    assert_close_t(&gpu, &composite, 1e-6, 1e-6, &format!("{what} against conv + add"));
                }
            }
        }
    }
}
