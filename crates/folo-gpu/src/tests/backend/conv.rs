//! Conv2d forward and gradients, split-K, strides beyond the kernel and batches beyond one dispatch.

use folo_core::{DType, Device, Tensor};

use super::fixtures::{assert_close_t, check, dev, randn};

struct ConvCase {
    x: [usize; 4],
    w: [usize; 4],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
}

#[test]
fn conv2d_forward_and_backward() {
    let dev = dev!();
    let cases = [
        ConvCase {
            x: [2, 5, 9, 11],
            w: [7, 5, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 6, 13, 10],
            w: [8, 6, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 8, 12, 12],
            w: [16, 8, 1, 1],
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 3, 17, 17],
            w: [5, 3, 1, 1],
            stride: 2,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 4, 15, 14],
            w: [6, 4, 5, 5],
            stride: 1,
            padding: 4,
            dilation: 2,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 8, 9, 9],
            w: [12, 4, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 2,
            bias: true,
        },
        ConvCase {
            x: [2, 6, 11, 11],
            w: [6, 1, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 6,
            bias: true,
        },
        ConvCase {
            x: [1, 4, 12, 13],
            w: [8, 1, 5, 5],
            stride: 2,
            padding: 2,
            dilation: 1,
            groups: 4,
            bias: false,
        },
        ConvCase {
            x: [1, 3, 32, 32],
            w: [16, 3, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        // Stride phases (the input gradient's decomposition): groups, no padding on odd sizes, a kernel smaller
        // than the stride (phases without taps), a kernel of 4 at stride 2, stride 2 with dilation 2 (not phased).
        ConvCase {
            x: [2, 8, 11, 9],
            w: [12, 4, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 2,
            bias: false,
        },
        ConvCase {
            x: [2, 5, 13, 12],
            w: [7, 5, 3, 3],
            stride: 2,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 6, 14, 13],
            w: [9, 6, 2, 2],
            stride: 3,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [2, 4, 12, 15],
            w: [8, 4, 4, 4],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 4, 15, 15],
            w: [6, 4, 3, 3],
            stride: 2,
            padding: 2,
            dilation: 2,
            groups: 1,
            bias: false,
        },
        // Weight-gradient tiles picked by shape (`conv::pick_config_weight_grad`): 32×128 for few output
        // channels, 128×128 and 64×128 for more, each with at least 128 columns (C_in·KH·KW).
        ConvCase {
            x: [2, 16, 20, 19],
            w: [24, 16, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 16, 12, 12],
            w: [128, 16, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 20, 10, 11],
            w: [70, 20, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        let x = randn(&c.x, 30 + i as u64);
        let w = randn(&c.w, 40 + i as u64).mul_scalar(0.3).unwrap();
        let b = randn(&[c.w[0]], 50 + i as u64);
        let k = (c.w[1] * c.w[2] * c.w[3]) as f32;
        let atol = 1e-4 * k.sqrt();
        let what = format!("conv2d case {i}");
        let (s, p, d, g, has_bias) = (c.stride, c.padding, c.dilation, c.groups, c.bias);
        check(dev, &[x, w, b], atol, 1e-4, &what, move |v| {
            v[0].conv2d(&v[1], if has_bias { Some(&v[2]) } else { None }, s, p, d, g)
        });
    }
}

#[test]
fn conv2d_weight_gradient_split_k() {
    // A small weight with a large image: the weight gradient splits K over many workgroups.
    let dev = dev!();
    let x = randn(&[2, 3, 96, 96], 60);
    let w = randn(&[4, 3, 3, 3], 61).mul_scalar(0.3).unwrap();
    check(dev, &[x.clone(), w.clone()], 2e-3, 1e-4, "split-k weight gradient", |v| {
        v[0].conv2d(&v[1], None, 1, 1, 1, 1)
    });
    let dw = randn(&[4, 1, 3, 3], 62);
    let xd = randn(&[2, 4, 80, 80], 63);
    check(dev, &[xd, dw], 2e-3, 1e-4, "depthwise weight gradient chunks", |v| {
        v[0].conv2d(&v[1], None, 1, 1, 1, 4)
    });
}

#[test]
fn conv2d_input_gradient_stride_beyond_kernel() {
    // Strides far beyond the kernel leave most stride phases without taps: the input gradient's scratch and host
    // work must scale with the phases that have taps, not with stride².
    let dev = dev!();
    let ones = |shape: &[usize]| Tensor::ones(shape);
    check(dev, &[ones(&[1, 2, 4, 4]), ones(&[3, 2, 1, 1])], 1e-5, 0.0, "stride 65536", |v| {
        v[0].conv2d(&v[1], None, 65536, 0, 1, 1)
    });
    let x = ones(&[1, 2, 4, 4]).to_device(dev).unwrap().requires_grad_();
    let w = ones(&[3, 2, 1, 1]).to_device(dev).unwrap();
    let y = x.conv2d(&w, None, 65536, 0, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 3, 1, 1]);
    let gx = y.sum_all().unwrap().backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap();
    let want: Vec<f32> = (0..32).map(|i| if i % 16 == 0 { 3.0 } else { 0.0 }).collect();
    assert_eq!(gx, want, "stride 65536 input gradient");
    for (i, (xs, ws, s, p)) in [
        ([2, 3, 17, 19], [4, 3, 1, 1], 4, 0),
        ([1, 4, 23, 21], [5, 4, 3, 3], 5, 1),
        ([1, 2, 9, 9], [3, 2, 2, 2], 7, 3),
    ]
    .into_iter()
    .enumerate()
    {
        let x = randn(&xs, 70 + i as u64);
        let w = randn(&ws, 80 + i as u64).mul_scalar(0.3).unwrap();
        let atol = 1e-4 * ((ws[1] * ws[2] * ws[3]) as f32).sqrt();
        check(dev, &[x, w], atol, 1e-4, &format!("kernel {} stride {s}", ws[2]), move |v| {
            v[0].conv2d(&v[1], None, s, p, 1, 1)
        });
    }
}

#[test]
fn conv2d_batches_beyond_one_dispatch() {
    // The conv kernels put (image, group) on the grid's z axis, at most 65535 workgroups: larger N · groups run
    // in pieces of the batch. A 3×3 conv, a strided one (stride-phase input gradient), a grouped 1×1 (80000
    // layers) and a depthwise one, forward and both gradients; the bias gradient sums 70000 images.
    let dev = dev!();
    for (i, (xs, ws, s, p, g)) in [
        ([70000, 1, 3, 3], [2, 1, 3, 3], 1, 1, 1),
        ([70000, 2, 4, 4], [2, 2, 3, 3], 2, 1, 1),
        ([40000, 4, 2, 2], [4, 2, 1, 1], 1, 0, 2),
        ([70000, 2, 3, 3], [2, 1, 3, 3], 1, 1, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let x = randn(&xs, 110 + i as u64);
        let w = randn(&ws, 120 + i as u64).mul_scalar(0.3).unwrap();
        let b = randn(&[ws[0]], 130 + i as u64);
        check(dev, &[x, w, b], 2e-3, 1e-4, &format!("conv {xs:?} {ws:?} s{s} g{g}"), move |v| {
            v[0].conv2d(&v[1], Some(&v[2]), s, p, 1, g)
        });
    }
    // f16 with an odd image size (9 halves) and an odd last piece: the piece copies stay 4-byte aligned.
    if folo_core::backend::backend(dev).is_ok_and(|b| b.computes_f16()) {
        let x = randn(&[70001, 1, 3, 3], 140).to_dtype(DType::F16).unwrap();
        let w = randn(&[2, 1, 3, 3], 141).mul_scalar(0.3).unwrap().to_dtype(DType::F16).unwrap();
        let y = x.to_device(dev).unwrap().conv2d(&w.to_device(dev).unwrap(), None, 1, 1, 1, 1).unwrap();
        let (xf, wf) = (x.to_dtype(DType::F32).unwrap(), w.to_dtype(DType::F32).unwrap());
        let want = xf.conv2d(&wf, None, 1, 1, 1, 1).unwrap();
        assert_close_t(&y.to_dtype(DType::F32).unwrap(), &want, 2e-3, 2e-3, "f16 conv batch 70001");
    }
}

#[test]
fn conv2d_many_groups_run_one_image_at_a_time() {
    // With 32768 groups or more not even two images fit one dispatch, so the pieces are single images; f16 ones
    // start at odd element offsets when an image holds an odd number of elements (input, output or both).
    let dev = dev!();
    let q = |t: Tensor| t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
    for (i, (xs, ws, g)) in [
        ([3, 65536, 1, 1], [32768, 2, 1, 1], 32768),
        ([3, 32769, 1, 1], [32769, 1, 1, 1], 32769),
        ([3, 65538, 1, 1], [32769, 2, 1, 1], 32769),
        ([3, 32769, 1, 1], [65538, 1, 1, 1], 32769),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 150 + 3 * i as u64;
        let (x, w, gy) =
            (q(randn(&xs, seed)), q(randn(&ws, seed + 1)), q(randn(&[3, ws[0], 1, 1], seed + 2)));
        // Forward and input gradient: on the CPU in f32, on the GPU in f32 and (autocast) in f16.
        let run = |dev: Device, amp: bool| {
            let xl = x.to_device(dev).unwrap().detach().requires_grad_();
            let y = {
                let _a = amp.then(folo_core::autocast);
                xl.conv2d(&w.to_device(dev).unwrap(), None, 1, 0, 1, g)
                    .unwrap_or_else(|e| panic!("conv {xs:?} {ws:?} on {dev:?} amp {amp}: {e}"))
            };
            assert_eq!(y.dtype(), if amp { DType::F16 } else { DType::F32 });
            let gr = y.backward_with(&gy.to_device(dev).unwrap().to_dtype(y.dtype()).unwrap()).unwrap();
            (y.to_dtype(DType::F32).unwrap(), gr.get(&xl).unwrap().clone())
        };
        let (yc, gc) = run(Device::Cpu, false);
        let mut runs = vec![(false, 1e-5)];
        if folo_core::backend::computes_f16(dev) {
            // Sums of at most two f16 products in f32, rounded once to f16.
            runs.push((true, 2e-3));
        }
        for (amp, tol) in runs {
            let (yg, gg) = run(dev, amp);
            let what = format!("conv {xs:?} {ws:?} g{g} amp {amp}");
            assert_close_t(&yg, &yc, tol, tol, &what);
            assert_close_t(&gg, &gc, tol, tol, &format!("{what}: input gradient"));
        }
    }
}

#[test]
fn depthwise_grad_vx4_and_subgroup_vec4_parity() {
    use crate::kernels::FORCE_NO_SUBGROUPS;

    let dev = dev!();
    let has_f16 = folo_core::backend::computes_f16(dev);
    let q = |t: &Tensor| t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();

    // Covers VX4 dX (stride 1, dilation 1, odd/even/tail widths w = 1, 2, 3, 5, 16, 20), general fallback dX
    // (stride 2, dilation 2), VEC4 dW (ow % 4 == 0, SW == 1, DW == 1), scalar dW (odd ow, stride 2, dilation 2),
    // kernel sizes 1x1 / 3x3 / 5x5 / 7x7, and depthwise multiplier MULT = 2.
    let cases = [
        ([2, 8, 9, 16], [8, 1, 3, 3], 1, 1, 1, 8),
        ([2, 6, 7, 5], [6, 1, 3, 3], 1, 1, 1, 6),
        ([1, 4, 3, 1], [4, 1, 3, 3], 1, 1, 1, 4),
        ([1, 4, 4, 2], [4, 1, 3, 3], 1, 1, 1, 4),
        ([1, 4, 4, 3], [4, 1, 3, 3], 1, 1, 1, 4),
        ([2, 4, 12, 20], [4, 1, 5, 5], 1, 2, 1, 4),
        ([2, 3, 10, 12], [3, 1, 7, 7], 1, 3, 1, 3),
        ([2, 4, 8, 8], [8, 1, 3, 3], 1, 1, 1, 4), // MULT = 2, VEC4 dW + VX4 dX
        ([2, 4, 8, 8], [4, 1, 1, 1], 1, 0, 1, 4), // 1x1 depthwise
        ([2, 4, 11, 16], [4, 1, 3, 3], 2, 1, 1, 4), // stride 2 fallback
        ([2, 4, 9, 12], [4, 1, 3, 3], 1, 2, 2, 4), // dilation 2 fallback
        ([2, 2, 41, 64], [2, 1, 3, 3], 1, 1, 1, 2), // multi-chunk VEC4 dW (> 4096 pixels, short last chunk)
    ];

    struct ResetSg(bool);
    impl Drop for ResetSg {
        fn drop(&mut self) {
            FORCE_NO_SUBGROUPS.with(|f| f.set(self.0));
        }
    }

    for (idx, &(xs, ws, s, p, dil, g)) in cases.iter().enumerate() {
        let seed = 200 + idx as u64 * 4;
        let x = randn(&xs, seed);
        let w = randn(&ws, seed + 1).mul_scalar(0.3).unwrap();
        // Run both native (subgroups when supported) and forced-portable fallback.
        for force_portable in [false, true] {
            let prev = FORCE_NO_SUBGROUPS.with(|f| f.replace(force_portable));
            let _guard = ResetSg(prev);
            check(
                dev,
                &[x.clone(), w.clone()],
                5e-4,
                2e-4,
                &format!("dw f32 {xs:?} {ws:?} s{s} p{p} d{dil} g{g} portable={force_portable}"),
                move |v| v[0].conv2d(&v[1], None, s, p, dil, g),
            );
            if has_f16 {
                let xq = q(&x);
                let wq = q(&w);
                let xc = xq.detach().requires_grad_();
                let wc = wq.detach().requires_grad_();
                let yc = xc.conv2d(&wc, None, s, p, dil, g).unwrap();
                let gy = q(&randn(yc.dims(), seed + 2));
                let gc = yc.backward_with(&gy).unwrap();

                let xg = xq.to_device(dev).unwrap().to_dtype(DType::F16).unwrap().detach().requires_grad_();
                let wg = wq.to_device(dev).unwrap().to_dtype(DType::F16).unwrap().detach().requires_grad_();
                let yg = xg.conv2d(&wg, None, s, p, dil, g).unwrap();
                let gg = yg.backward_with(&gy.to_device(dev).unwrap().to_dtype(DType::F16).unwrap()).unwrap();
                let label = format!("dw f16 {xs:?} {ws:?} s{s} p{p} d{dil} g{g} portable={force_portable}");
                assert_close_t(
                    &gg.get(&xg).unwrap().to_dtype(DType::F32).unwrap(),
                    gc.get(&xc).unwrap(),
                    3e-3,
                    3e-3,
                    &format!("{label} dX"),
                );
                assert_close_t(
                    &gg.get(&wg).unwrap().to_dtype(DType::F32).unwrap(),
                    gc.get(&wc).unwrap(),
                    5e-3,
                    5e-3,
                    &format!("{label} dW"),
                );
            }
        }
    }

    // Asymmetric stride (SH=2, SW=1, ow % 4 == 0): exercises VEC4 dW with vertical stride > 1.
    {
        let params = folo_core::Conv2dParams { stride: [2, 1], padding: [1, 1], dilation: [1, 1], groups: 4 };
        let x = randn(&[2, 4, 11, 12], 290);
        let w = randn(&[4, 1, 3, 3], 291).mul_scalar(0.3).unwrap();
        check(dev, &[x, w], 5e-4, 2e-4, "dw asymmetric stride [2, 1]", move |v| {
            v[0].conv2d_with(&v[1], None, &params)
        });
    }

    // Non-finite border checks: padded terms must not multiply 0.0 * ±inf into NaN in VX4 dX or VEC4 dW
    // (`groups == c_in >= 2` so `is_depthwise()` is true).
    {
        // VX4 dX: 1x3 kernel with padding [0, 1], weight [0.0, 1.0, +inf]. Input col 0 has ox(kx=2) = -1
        // (padded, must skip +inf) and ox(kx=1) = 0, so dX[0,*,0,0] must stay finite (= G[0,*,0,0]).
        let x = Tensor::ones([1, 2, 1, 4]);
        let w =
            Tensor::from_slice(&[0.0f32, 1.0, f32::INFINITY, 0.0, 1.0, f32::INFINITY], [2, 1, 1, 3]).unwrap();
        let gy = Tensor::from_slice(&[2.0f32, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0], [1, 2, 1, 4]).unwrap();
        let params = folo_core::Conv2dParams { stride: [1, 1], padding: [0, 1], dilation: [1, 1], groups: 2 };
        let xg = x.to_device(dev).unwrap().detach().requires_grad_();
        let wg = w.to_device(dev).unwrap();
        let yg = xg.conv2d_with(&wg, None, &params).unwrap();
        let dx =
            yg.backward_with(&gy.to_device(dev).unwrap()).unwrap().get(&xg).unwrap().to_vec::<f32>().unwrap();
        assert_eq!((dx[0], dx[4]), (2.0, 2.0), "VX4 dX padded border must not turn 0 * inf into NaN: {dx:?}");

        // VEC4 dW: 1x3 kernel with padding [0, 1], finite input, G[*,0] = +inf. Tap kx=0 at ox=0 reads padded
        // ix = -1 (must skip +inf), while ox = 1..3 read finite G -> dW[*,kx=0] must stay finite!
        let x2 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], [1, 2, 1, 4]).unwrap();
        let w2 = Tensor::ones([2, 1, 1, 3]);
        let gy2 =
            Tensor::from_slice(&[f32::INFINITY, 1.0, 1.0, 1.0, f32::INFINITY, 1.0, 1.0, 1.0], [1, 2, 1, 4])
                .unwrap();
        let wg2 = w2.to_device(dev).unwrap().detach().requires_grad_();
        let yg2 = x2.to_device(dev).unwrap().conv2d_with(&wg2, None, &params).unwrap();
        let dw = yg2
            .backward_with(&gy2.to_device(dev).unwrap())
            .unwrap()
            .get(&wg2)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            (dw[0], dw[3]),
            (6.0, 6.0),
            "VEC4 dW padded border must not turn inf * 0 into NaN: {dw:?}"
        );
    }
}

#[test]
fn depthwise_backward_input_tiled_s2_and_vx4_parity() {
    let dev = dev!();
    let ctx = crate::context_for(dev).unwrap();
    let has_f16 = ctx.caps().shader_f16;

    let cases = [
        // Tiled stride-1 (`h, w >= 64`): even & odd spatial dims, k=3/5/7, MULT=1 & MULT=2
        ([1, 4, 64, 64], [4, 1, 3, 3], 1, 1, 4, "dw_dx tiled 3x3 even"),
        ([1, 3, 65, 67], [3, 1, 5, 5], 1, 2, 3, "dw_dx tiled 5x5 odd"),
        ([1, 2, 64, 68], [4, 1, 7, 7], 1, 3, 2, "dw_dx tiled 7x7 mult2"),
        // Phase-unrolled stride-2 (`S2`): even & odd spatial dims, pad 0/1/2, k=2/3/5, MULT=1 & MULT=2
        ([2, 4, 16, 16], [4, 1, 3, 3], 2, 1, 4, "dw_dx s2 3x3 p1 even"),
        ([2, 3, 15, 17], [3, 1, 3, 3], 2, 1, 3, "dw_dx s2 3x3 p1 odd"),
        ([1, 4, 14, 15], [4, 1, 3, 3], 2, 0, 4, "dw_dx s2 3x3 p0 odd"),
        ([2, 3, 18, 19], [6, 1, 5, 5], 2, 2, 3, "dw_dx s2 5x5 p2 mult2"),
        ([1, 4, 12, 14], [4, 1, 2, 2], 2, 0, 4, "dw_dx s2 2x2 p0"),
        // VX4 + VEC4_OUT (h, w < 64 stride-1, including k=1)
        ([2, 4, 12, 16], [4, 1, 3, 3], 1, 1, 4, "dw_dx vx4 vec4_out 3x3"),
        ([2, 4, 10, 12], [4, 1, 1, 1], 1, 0, 4, "dw_dx vx4 vec4_out 1x1"),
        ([2, 3, 9, 11], [6, 1, 5, 5], 1, 2, 3, "dw_dx vx4 scalar_out 5x5 mult2"),
    ];
    for (idx, &(xs, ws, s, p, g, label)) in cases.iter().enumerate() {
        let x = randn(&xs, 600 + idx as u64);
        let w = randn(&ws, 700 + idx as u64).mul_scalar(0.3).unwrap();
        let atol = 2e-4 * ((ws[2] * ws[3]) as f32).sqrt();
        check(dev, &[x.clone(), w.clone()], atol, 1e-4, label, move |v| v[0].conv2d(&v[1], None, s, p, 1, g));
        if has_f16 && xs[2] <= 64 {
            let x16 = x.to_dtype(DType::F16).unwrap();
            let w16 = w.to_dtype(DType::F16).unwrap();
            let xc = x16.to_dtype(DType::F32).unwrap().detach().requires_grad_();
            let wc = w16.to_dtype(DType::F32).unwrap();
            let yc = xc.conv2d(&wc, None, s, p, 1, g).unwrap();
            let gy = randn(yc.dims(), 800 + idx as u64).to_dtype(DType::F16).unwrap();
            let gxc = yc.backward_with(&gy.to_dtype(DType::F32).unwrap()).unwrap().get(&xc).unwrap().clone();

            let xg = x16.to_device(dev).unwrap().detach().requires_grad_();
            let wg = w16.to_device(dev).unwrap();
            let yg = xg.conv2d(&wg, None, s, p, 1, g).unwrap();
            let gxg = yg.backward_with(&gy.to_device(dev).unwrap()).unwrap().get(&xg).unwrap().clone();
            assert_eq!(gxg.dtype(), DType::F16);
            assert_close_t(&gxg.to_dtype(DType::F32).unwrap(), &gxc, 5e-3, 5e-3, &format!("f16 {label}"));
        }
    }

    // Infinite weight at a padded corner must not turn zero-padded out-of-bounds taps into NaN on TILED or S2
    // (use groups=2 so both CPU and GPU take their depthwise paths).
    for &(h, w_dim, stride) in &[(64usize, 64usize, 1usize), (16, 16, 2)] {
        let mut wv = vec![1.0f32; 2 * 9];
        wv[0] = f32::INFINITY;
        wv[9] = f32::INFINITY;
        let wt = Tensor::from_vec(wv, [2, 1, 3, 3]).unwrap();
        let x = Tensor::ones([1, 2, h, w_dim]);
        let xg = x.to_device(dev).unwrap().detach().requires_grad_();
        let yg = xg.conv2d(&wt.to_device(dev).unwrap(), None, stride, 1, 1, 2).unwrap();
        let gy = Tensor::ones(yg.dims());
        let gxg = yg.backward_with(&gy.to_device(dev).unwrap()).unwrap().get(&xg).unwrap().clone();
        let gv = gxg.to_vec::<f32>().unwrap();
        // Bottom-right corner of both channels: ky=0 / kx=0 fall outside (OH, OW), so finite taps only -> no NaN.
        let want = if stride == 1 { 4.0f32 } else { 1.0f32 };
        assert_eq!(gv[h * w_dim - 1], want, "stride={stride} ch0 corner: {}", gv[h * w_dim - 1]);
        assert_eq!(gv[2 * h * w_dim - 1], want, "stride={stride} ch1 corner: {}", gv[2 * h * w_dim - 1]);
    }
    assert!(ctx.kernels_without_registry().is_empty());
}
