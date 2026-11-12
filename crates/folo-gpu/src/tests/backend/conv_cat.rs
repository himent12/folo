//! Convs that read their input in place: a 1×1 conv over a channel concatenation (`conv2d_act_cat`, the parts
//! never copied into one tensor) and any non-depthwise conv on a channel slice (`narrow` along dim 1), against the
//! CPU backend.

use folo_core::backend::{Conv2dParams, UnaryOp};
use folo_core::{DType, Device, Result, Storage, Tensor, no_grad};

use super::fixtures::{assert_close_t, dev, randn};

/// `t` rounded through f16 when the test runs in f16, so the CPU reference sees the GPU's inputs.
fn rounded(t: Tensor, dtype: DType) -> Tensor {
    if dtype == DType::F16 { t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap() } else { t }
}

/// The GPU backend's `conv2d_act_cat` called directly (not through `Tensor`, which would fall back to `cat` then
/// a conv): `Ok` means the parts were read in place.
fn direct(xs: &[&Tensor], w: &Tensor, b: &Tensor, p: &Conv2dParams, act: Option<UnaryOp>) -> Result<Storage> {
    let be = folo_core::backend::backend(xs[0].device())?;
    w.with_view(|wv| {
        b.with_view(|bv| match xs {
            [a, c] => a.with_view(|a| c.with_view(|c| be.conv2d_act_cat(&[a, c], wv, Some(bv), p, act))),
            [a, c, e] => a.with_view(|a| {
                c.with_view(|c| e.with_view(|e| be.conv2d_act_cat(&[a, c, e], wv, Some(bv), p, act)))
            }),
            [a, c, e, g] => a.with_view(|a| {
                c.with_view(|c| {
                    e.with_view(|e| g.with_view(|g| be.conv2d_act_cat(&[a, c, e, g], wv, Some(bv), p, act)))
                })
            }),
            _ => unreachable!("2 to 4 parts"),
        })
    })
}

/// Parts of 2 to 4 tensors, some whole and some channel slices of a larger tensor (offset, larger image stride),
/// batch 1 and 2, `H·W` a multiple of 4 (vector loads) and not, f32 and f16.
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
        for (n, h, w) in [(1, 4, 5), (2, 4, 5), (2, 3, 5), (1, 20, 20)] {
            let big = rounded(randn(&[n, 12, h, w], 1), dtype);
            let whole = rounded(randn(&[n, 3, h, w], 2), dtype);
            let other = rounded(randn(&[n, 8, h, w], 3), dtype);
            let slices =
                [big.narrow(1, 2, 5).unwrap(), big.narrow(1, 9, 3).unwrap(), other.narrow(1, 0, 4).unwrap()];
            let cases: [Vec<&Tensor>; 3] = [
                vec![&slices[0], &whole],
                vec![&whole, &slices[1], &slices[0]],
                vec![&slices[0], &whole, &slices[2], &slices[1]],
            ];
            for (i, parts) in cases.iter().enumerate() {
                let c: usize = parts.iter().map(|t| t.dims()[1]).sum();
                let wt = rounded(randn(&[7, c, 1, 1], 10 + i as u64), dtype);
                let bias = rounded(randn(&[7], 20 + i as u64), dtype);
                let what = format!("{dtype:?} n{n} {h}x{w} case {i}");
                let cpu = Tensor::conv2d_act_cat(parts, &wt, Some(&bias), &p, act).unwrap();
                let to = |t: &Tensor| t.to_device(dev).unwrap().to_dtype(dtype).unwrap();
                // The slices stay slices on the GPU: `narrow` of the uploaded larger tensors.
                let (gbig, gother) = (to(&big), to(&other));
                let gs = [
                    gbig.narrow(1, 2, 5).unwrap(),
                    gbig.narrow(1, 9, 3).unwrap(),
                    gother.narrow(1, 0, 4).unwrap(),
                ];
                let gwhole = to(&whole);
                let pick = |t: &Tensor| -> Tensor {
                    if std::ptr::eq(t, &whole) {
                        gwhole.clone()
                    } else {
                        let k = slices.iter().position(|s| std::ptr::eq(s, t)).unwrap();
                        gs[k].clone()
                    }
                };
                let gparts: Vec<Tensor> = parts.iter().map(|t| pick(t)).collect();
                let refs: Vec<&Tensor> = gparts.iter().collect();
                let (gw, gb) = (to(&wt), to(&bias));
                assert!(direct(&refs, &gw, &gb, &p, act).is_ok(), "{what}: not read in place");
                let gpu = Tensor::conv2d_act_cat(&refs, &gw, Some(&gb), &p, act).unwrap();
                assert!(gpu.device() == dev, "{what}");
                let (atol, rtol) = if dtype == DType::F16 { (1e-2, 1e-2) } else { (1e-4, 1e-4) };
                assert_close_t(&gpu.to_dtype(DType::F32).unwrap(), &cpu, atol, rtol, &what);
            }
        }
    }
}

/// What the in-place concatenation refuses (the caller then concatenates first): 3×3 kernels, groups, five
/// parts, mismatched sizes and parts whose images are not contiguous.
#[test]
fn concatenations_it_cannot_read_in_place_fall_back() {
    let dev = dev!();
    let _g = no_grad();
    let to = |t: &Tensor| t.to_device(dev).unwrap();
    let a = to(&randn(&[2, 4, 6, 6], 1));
    let b = to(&randn(&[2, 4, 6, 6], 2));
    let bias = to(&randn(&[5], 3));
    let p = Conv2dParams::default();
    let w3 = to(&randn(&[5, 8, 3, 3], 4));
    assert!(direct(&[&a, &b], &w3, &bias, &p, None).is_err(), "3x3");
    let grouped = Conv2dParams { groups: 2, ..p };
    assert!(direct(&[&a, &b], &to(&randn(&[4, 4, 1, 1], 5)), &to(&randn(&[4], 6)), &grouped, None).is_err());
    let w1 = to(&randn(&[5, 8, 1, 1], 7));
    // A transposed (H, W) part: its images are not contiguous.
    let t = b.transpose(2, 3).unwrap();
    assert!(direct(&[&a, &t], &w1, &bias, &p, None).is_err(), "strided part");
    // Through `Tensor` all of them still give the concatenation's conv.
    for (parts, wt, pp) in [(vec![&a, &b], &w3, p), (vec![&a, &t], &w1, p)] {
        let gpu = Tensor::conv2d_act_cat(&parts, wt, Some(&bias), &pp, None).unwrap();
        let cpu_parts: Vec<Tensor> = parts.iter().map(|t| t.to_device(Device::Cpu).unwrap()).collect();
        let refs: Vec<&Tensor> = cpu_parts.iter().collect();
        let cpu = Tensor::conv2d_act_cat(
            &refs,
            &wt.to_device(Device::Cpu).unwrap(),
            Some(&bias.to_device(Device::Cpu).unwrap()),
            &pp,
            None,
        )
        .unwrap();
        assert_close_t(&gpu, &cpu, 1e-4, 1e-4, "fallback");
    }
}

/// A conv whose input is a channel slice reads it in place (1×1 and 3×3, batch 1 and 2, f32 and f16).
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
            let gslice = to(&big).narrow(1, 3, 6).unwrap();
            let at =
                gslice.with_view(|v| crate::backend::conv_input(&v)).expect("a channel slice is addressable");
            assert_eq!(at.offset, 3 * 7 * 9);
            let gpu = gslice.conv2d_act(&to(&wt), Some(&to(&bias)), &p, act).unwrap();
            let (atol, rtol) = if dtype == DType::F16 { (1e-2, 1e-2) } else { (1e-4, 1e-4) };
            let what = format!("{dtype:?} n{n} k{k}");
            assert_close_t(&gpu.to_dtype(DType::F32).unwrap(), &cpu, atol, rtol, &what);
        }
    }
}
