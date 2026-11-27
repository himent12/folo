use super::*;

#[test]
fn fused_conv_noncontiguous_and_rowband_fill_match_exact_reference() {
    let mut rng = Rng::seed(7009);
    // Exercise non-contiguous cross-row writes/residual reads (`ow = 1, 20, 40`), Pointwise1x1 past-SLACK
    // boundary (`1x73` -> final `col = 72 > SLACK = 64`), and row-band parallel `image_rows` (`s = 1` & `s = 2`).
    for &(n, c, o, h, w, k, s, pad, silu, has_res, f32_out, check_ref) in &[
        (1usize, 8usize, 8usize, 5usize, 1usize, 3usize, 1usize, 1usize, true, true, false, true),
        (1, 8, 8, 5, 1, 3, 1, 1, false, true, true, true),
        (1, 8, 8, 1, 73, 1, 1, 0, true, true, false, true), // Pointwise1x1 final col=72 > SLACK=64
        (2, 12, 12, 20, 20, 3, 1, 1, true, true, false, true),
        (1, 12, 12, 20, 20, 3, 1, 1, false, false, true, true),
        (1, 16, 16, 40, 40, 3, 2, 1, true, true, false, true),
        (1, 8, 8, 96, 96, 3, 2, 1, true, false, false, true), // groups=2, hq=49, body=19208 >= 16384 -> row-band par
        (1, 8, 8, 128, 128, 3, 1, 1, true, false, false, false), // groups=2, hq=130, body=33824 >= 16384 -> row-band par
        (1, 16, 16, 25, 29, 1, 1, 0, true, true, false, true),   // Pointwise1x1 u8 + silu + residual + tail
        (1, 16, 16, 25, 29, 1, 1, 0, false, true, true, true),   // Pointwise1x1 f32 + residual + tail
    ] {
        let hw = h * w;
        let taps = k * k;
        let (oh, ow) = q8::out_size(h, w, k, s, pad).unwrap();
        let ohw = oh * ow;
        let in_g: Vec<Grid> = (0..c).map(|i| Grid::from_range(-0.4 - 0.05 * i as f32, 2.5)).collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.5, 3.0 + 0.1 * i as f32)).collect();
        let res_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-1.0, 2.0 + 0.05 * i as f32)).collect();
        let in_z: Vec<u8> = in_g.iter().map(|g| g.zero).collect();
        let xf: Vec<f32> = (0..n * c * hw).map(|_| rng.uniform(-1.0, 2.5)).collect();
        let mut xq = alloc(n, c, hw);
        quantize_per_channel(&xf, n, c, hw, &in_g, &mut xq);
        let rf: Vec<f32> = (0..n * o * ohw).map(|_| rng.uniform(-1.0, 2.0)).collect();
        let mut rq = alloc(n, o, ohw);
        quantize_per_channel(&rf, n, o, ohw, &res_g, &mut rq);
        let wt: Vec<f32> = (0..o * c * taps).map(|_| rng.uniform(-0.3, 0.3)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let cv = QConv::new(&wt, &bias, (o, c, k), s, pad, silu, &in_g);
        let vx = C4 { data: &xq, offset: 0, img_stride: (c / 4) * hw, c, zero: in_z[0] };
        let vr = C4 { data: &rq, offset: 0, img_stride: (o / 4) * ohw, c: o, zero: res_g[0].zero };
        let res_arg = has_res.then_some((vr, res_g.as_slice()));
        let expected = check_ref.then(|| {
            ref_conv_parts(
                &[(&xq, &in_g)],
                n,
                h,
                w,
                &wt,
                &bias,
                (o, k, s, pad),
                silu,
                false,
                has_res.then_some((rq.as_slice(), res_g.as_slice())),
                &out_g,
            )
        });

        if f32_out {
            let mut out1 = vec![0f32; n * o * ohw];
            let mut out8 = vec![0f32; n * o * ohw];
            pool(1).install(|| {
                conv_per_channel(&[vx], &[&in_z], n, h, w, &cv, res_arg, QOut::F32 { data: &mut out1 });
            });
            pool(8).install(|| {
                conv_per_channel(&[vx], &[&in_z], n, h, w, &cv, res_arg, QOut::F32 { data: &mut out8 });
            });
            assert!(
                out1.iter().zip(&out8).all(|(a, b)| a.to_bits() == b.to_bits()),
                "f32 thread mismatch ({n},{c}->{o},{h}x{w},k{k}s{s})"
            );
            if let Some((_, want_f)) = &expected {
                assert!(
                    out1.iter().zip(want_f).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "f32 scalar-ref mismatch ({n},{c}->{o},{h}x{w},k{k}s{s})"
                );
            }
        } else {
            let mut out1 = alloc(n, o, ohw);
            let mut out8 = alloc(n, o, ohw);
            pool(1).install(|| {
                conv_per_channel(
                    &[vx],
                    &[&in_z],
                    n,
                    h,
                    w,
                    &cv,
                    res_arg,
                    QOut::U8PerChannel { data: &mut out1, grids: &out_g },
                );
            });
            pool(8).install(|| {
                conv_per_channel(
                    &[vx],
                    &[&in_z],
                    n,
                    h,
                    w,
                    &cv,
                    res_arg,
                    QOut::U8PerChannel { data: &mut out8, grids: &out_g },
                );
            });
            let words = n * (o / 4) * ohw;
            assert_eq!(&out1[..words], &out8[..words], "u8 thread mismatch ({n},{c}->{o},{h}x{w},k{k}s{s})");
            if let Some((want_q, _)) = &expected {
                assert_eq!(
                    &out1[..words],
                    want_q.as_slice(),
                    "u8 scalar-ref mismatch ({n},{c}->{o},{h}x{w},k{k}s{s})"
                );
            }
        }
    }

    // Decoupled multi-part `run_multi_spec` bit-exact against scalar integer reference (k=1 and k=3, non-contiguous, with residual).
    for &(h, w, k, pad, silu) in &[(13usize, 20usize, 3usize, 1usize, true), (9, 29, 1, 0, false)] {
        let (n, c0, c1, o) = (2, 8, 8, 8);
        let hw = h * w;
        let g0: Vec<Grid> = (0..c0).map(|i| Grid { scale: 0.003 + 0.0004 * i as f32, zero: 25 }).collect();
        let g1: Vec<Grid> = (0..c1).map(|i| Grid { scale: 0.04 + 0.005 * i as f32, zero: 85 }).collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.4, 3.0 + 0.1 * i as f32)).collect();
        let res_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.8, 2.0 + 0.1 * i as f32)).collect();
        let z0: Vec<u8> = g0.iter().map(|g| g.zero).collect();
        let z1: Vec<u8> = g1.iter().map(|g| g.zero).collect();
        let mut x0 = alloc(n, c0, hw);
        let mut x1 = alloc(n, c1, hw);
        let mut rq = alloc(n, o, hw);
        let x0f: Vec<f32> = (0..n * c0 * hw).map(|_| rng.uniform(-0.1, 0.4)).collect();
        let x1f: Vec<f32> = (0..n * c1 * hw).map(|_| rng.uniform(-2.0, 5.0)).collect();
        let rf: Vec<f32> = (0..n * o * hw).map(|_| rng.uniform(-1.0, 2.0)).collect();
        quantize_per_channel(&x0f, n, c0, hw, &g0, &mut x0);
        quantize_per_channel(&x1f, n, c1, hw, &g1, &mut x1);
        quantize_per_channel(&rf, n, o, hw, &res_g, &mut rq);
        let wt: Vec<f32> = (0..o * (c0 + c1) * k * k).map(|_| rng.uniform(-0.4, 0.4)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let cv = QConv::new_parts(&wt, &bias, (o, c0 + c1, k), 1, pad, silu, &[&g0, &g1], true);
        assert_eq!(cv.num_parts(), 2);
        let (want_q, want_f) = ref_conv_parts(
            &[(&x0, &g0), (&x1, &g1)],
            n,
            h,
            w,
            &wt,
            &bias,
            (o, k, 1, pad),
            silu,
            true,
            Some((&rq, &res_g)),
            &out_g,
        );
        let v0 = C4 { data: &x0, offset: 0, img_stride: (c0 / 4) * hw, c: c0, zero: z0[0] };
        let v1 = C4 { data: &x1, offset: 0, img_stride: (c1 / 4) * hw, c: c1, zero: z1[0] };
        let vr = C4 { data: &rq, offset: 0, img_stride: (o / 4) * hw, c: o, zero: res_g[0].zero };
        let mut got_q = alloc(n, o, hw);
        let mut got_f = vec![0f32; n * o * hw];
        pool(4).install(|| {
            conv_per_channel(
                &[v0, v1],
                &[&z0, &z1],
                n,
                h,
                w,
                &cv,
                Some((vr, res_g.as_slice())),
                QOut::U8PerChannel { data: &mut got_q, grids: &out_g },
            );
            conv_per_channel(
                &[v0, v1],
                &[&z0, &z1],
                n,
                h,
                w,
                &cv,
                Some((vr, res_g.as_slice())),
                QOut::F32 { data: &mut got_f },
            );
        });
        assert_eq!(&got_q[..want_q.len()], want_q.as_slice(), "multi-part u8 mismatch (k={k})");
        assert!(
            got_f.iter().zip(&want_f).all(|(a, b)| a.to_bits() == b.to_bits()),
            "multi-part f32 mismatch (k={k})"
        );
    }
}

#[test]
fn fused_3x3_strip_and_multipart_1x1_match_exact_reference() {
    let mut rng = Rng::seed(7012);

    // 1. `k=3, s=2, p=1` single-barrier rotating strip (tiny 1x1, multi-chunk `ow > NR`, odd 29x35, multi-part,
    //    residual, F32/U8).
    for &(n, c0, c1, o, h, w, silu, has_res) in &[
        (1usize, 4usize, 0usize, 4usize, 1usize, 1usize, true, false),
        (1, 4, 0, 4, 1, 49, true, false),
        (2, 8, 0, 8, 2, 3, false, true),
        (1, 8, 4, 12, 3, 55, false, true),
        (1, 8, 0, 8, 8, 16, true, true), // ow = 8 (exact VECS=1 direct store + residual)
        (1, 8, 0, 8, 8, 32, false, true), // ow = 16 (exact VECS=2 direct store + residual)
        (1, 8, 0, 8, 8, 64, true, true), // ow = 32 = 24 + 8 (full + exact VECS=1)
        (1, 8, 0, 8, 8, 80, true, false), // ow = 40 = 24 + 16 (full + exact VECS=2)
        (2, 8, 4, 12, 29, 35, true, true),
        (1, 16, 0, 16, 40, 41, false, true),
    ] {
        let c = c0 + c1;
        let hw = h * w;
        let (oh, ow) = q8::out_size(h, w, 3, 2, 1).unwrap();
        let ohw = oh * ow;
        let g0: Vec<Grid> = (0..c0).map(|i| Grid::from_range(-0.4, 2.0 + 0.1 * i as f32)).collect();
        let g1: Vec<Grid> = (0..c1).map(|i| Grid::from_range(-0.3, 1.5 + 0.1 * i as f32)).collect();
        let all_g: Vec<Grid> = g0.iter().chain(&g1).copied().collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.5, 3.0 + 0.1 * i as f32)).collect();
        let res_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.6, 2.2 + 0.1 * i as f32)).collect();
        let mut x0 = alloc(n, c0, hw);
        let x0f: Vec<f32> = (0..n * c0 * hw).map(|_| rng.uniform(-0.4, 2.0)).collect();
        quantize_per_channel(&x0f, n, c0, hw, &g0, &mut x0);
        let mut x1 = if c1 > 0 { alloc(n, c1, hw) } else { Vec::new() };
        if c1 > 0 {
            let x1f: Vec<f32> = (0..n * c1 * hw).map(|_| rng.uniform(-0.3, 1.5)).collect();
            quantize_per_channel(&x1f, n, c1, hw, &g1, &mut x1);
        }
        let mut rq = alloc(n, o, ohw);
        let rf: Vec<f32> = (0..n * o * ohw).map(|_| rng.uniform(-0.6, 2.2)).collect();
        quantize_per_channel(&rf, n, o, ohw, &res_g, &mut rq);
        let wt: Vec<f32> = (0..o * c * 9).map(|_| rng.uniform(-0.35, 0.35)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.4, 0.4)).collect();
        let cv = QConv::new(&wt, &bias, (o, c, 3), 2, 1, silu, &all_g);
        let z0: Vec<u8> = g0.iter().map(|g| g.zero).collect();
        let z1: Vec<u8> = g1.iter().map(|g| g.zero).collect();
        let ref_parts: Vec<(&[u32], &[Grid])> =
            if c1 > 0 { vec![(&x0, &g0), (&x1, &g1)] } else { vec![(&x0, &g0)] };
        let (want_q, want_f) = ref_conv_parts(
            &ref_parts,
            n,
            h,
            w,
            &wt,
            &bias,
            (o, 3, 2, 1),
            silu,
            false,
            has_res.then_some((&rq, &res_g)),
            &out_g,
        );
        let v0 = C4 { data: &x0, offset: 0, img_stride: (c0 / 4) * hw, c: c0, zero: z0[0] };
        let views: Vec<C4<'_>> = if c1 > 0 {
            let v1 = C4 { data: &x1, offset: 0, img_stride: (c1 / 4) * hw, c: c1, zero: z1[0] };
            vec![v0, v1]
        } else {
            vec![v0]
        };
        let zslices: Vec<&[u8]> = if c1 > 0 { vec![&z0, &z1] } else { vec![&z0] };
        let vr = C4 { data: &rq, offset: 0, img_stride: (o / 4) * ohw, c: o, zero: res_g[0].zero };
        let res_arg = has_res.then_some((vr, res_g.as_slice()));
        for threads in [1usize, 4] {
            let mut got_q = alloc(n, o, ohw);
            let mut got_f = vec![f32::NAN; n * o * ohw];
            pool(threads).install(|| {
                conv_per_channel(
                    &views,
                    &zslices,
                    n,
                    h,
                    w,
                    &cv,
                    res_arg,
                    QOut::U8PerChannel { data: &mut got_q, grids: &out_g },
                );
                conv_per_channel(&views, &zslices, n, h, w, &cv, res_arg, QOut::F32 { data: &mut got_f });
            });
            assert_eq!(
                &got_q[..want_q.len()],
                want_q.as_slice(),
                "s2 strip u8 mismatch ({n},{c0}+{c1}->{o},{h}x{w},silu={silu},res={has_res},thr={threads})"
            );
            assert!(
                got_f.iter().zip(&want_f).all(|(a, b)| a.to_bits() == b.to_bits()),
                "s2 strip f32 mismatch ({n},{c0}+{c1}->{o},{h}x{w},silu={silu},res={has_res},thr={threads})"
            );
        }
    }

    // 2. Large-input `k=3, s=1, p=1` (> 384 KiB triggers the single-barrier s1 row-strip path):
    //    exact VECS=1 (`w=80`), exact VECS=2 (`w=88`), and odd partial (`w=81`) tails with residual.
    for &w in &[80usize, 81, 88] {
        let (n, c, o, h) = (1usize, 64usize, 8usize, 80usize);
        let hw = h * w;
        let in_g: Vec<Grid> = (0..c).map(|i| Grid::from_range(-0.3, 1.8 + 0.02 * i as f32)).collect();
        let in_z: Vec<u8> = in_g.iter().map(|g| g.zero).collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.5, 2.5 + 0.1 * i as f32)).collect();
        let res_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.4, 2.0 + 0.1 * i as f32)).collect();
        let mut xq = alloc(n, c, hw);
        let xf: Vec<f32> = (0..c * hw).map(|i| ((i.wrapping_mul(37) % 251) as f32) * 0.008 - 0.3).collect();
        quantize_per_channel(&xf, n, c, hw, &in_g, &mut xq);
        let mut rq = alloc(n, o, hw);
        let rf: Vec<f32> = (0..o * hw).map(|i| ((i.wrapping_mul(19) % 197) as f32) * 0.01 - 0.4).collect();
        quantize_per_channel(&rf, n, o, hw, &res_g, &mut rq);
        let wt: Vec<f32> = (0..o * c * 9).map(|_| rng.uniform(-0.2, 0.2)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.3, 0.3)).collect();
        let cv = QConv::new(&wt, &bias, (o, c, 3), 1, 1, true, &in_g);
        let vx = C4 { data: &xq, offset: 0, img_stride: (c / 4) * hw, c, zero: in_z[0] };
        let vr = C4 { data: &rq, offset: 0, img_stride: (o / 4) * hw, c: o, zero: res_g[0].zero };
        let (want_q, want_f) = ref_conv_parts(
            &[(&xq, &in_g)],
            n,
            h,
            w,
            &wt,
            &bias,
            (o, 3, 1, 1),
            true,
            false,
            Some((&rq, &res_g)),
            &out_g,
        );
        let mut q1 = alloc(n, o, hw);
        let mut q4 = alloc(n, o, hw);
        let mut f4 = vec![f32::NAN; o * hw];
        pool(1).install(|| {
            conv_per_channel(
                &[vx],
                &[&in_z],
                n,
                h,
                w,
                &cv,
                Some((vr, res_g.as_slice())),
                QOut::U8PerChannel { data: &mut q1, grids: &out_g },
            );
        });
        pool(4).install(|| {
            conv_per_channel(
                &[vx],
                &[&in_z],
                n,
                h,
                w,
                &cv,
                Some((vr, res_g.as_slice())),
                QOut::U8PerChannel { data: &mut q4, grids: &out_g },
            );
            conv_per_channel(
                &[vx],
                &[&in_z],
                n,
                h,
                w,
                &cv,
                Some((vr, res_g.as_slice())),
                QOut::F32 { data: &mut f4 },
            );
        });
        assert_eq!(&q1[..want_q.len()], want_q.as_slice(), "large s1 strip u8 (1 thr) mismatch");
        assert_eq!(&q4[..want_q.len()], want_q.as_slice(), "large s1 strip u8 (4 thr) mismatch");
        assert!(
            f4.iter().zip(&want_f).all(|(a, b)| a.to_bits() == b.to_bits()),
            "large s1 strip f32 mismatch"
        );
    }

    // 3. Multi-part 1×1 inline (`<= 8` parts) and fallback (`9` parts), plus decoupled 1×1 with mismatched `xs`.
    for &part_count in &[2usize, 4, 8, 9] {
        let (n, pc, o, h, w) = (2usize, 4usize, 8usize, 7usize, 13usize);
        let c = pc * part_count;
        let hw = h * w;
        let part_grids: Vec<Vec<Grid>> = (0..part_count)
            .map(|p| (0..pc).map(|i| Grid::from_range(-0.3, 1.5 + 0.1 * (p + i) as f32)).collect())
            .collect();
        let all_g: Vec<Grid> = part_grids.iter().flatten().copied().collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.4, 2.5 + 0.1 * i as f32)).collect();
        let part_bufs: Vec<Vec<u32>> = part_grids
            .iter()
            .map(|gs| {
                let mut q = alloc(n, pc, hw);
                let xf: Vec<f32> = (0..n * pc * hw).map(|_| rng.uniform(-0.3, 1.5)).collect();
                quantize_per_channel(&xf, n, pc, hw, gs, &mut q);
                q
            })
            .collect();
        let part_zs: Vec<Vec<u8>> = part_grids.iter().map(|gs| gs.iter().map(|g| g.zero).collect()).collect();
        let wt: Vec<f32> = (0..o * c).map(|_| rng.uniform(-0.4, 0.4)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.4, 0.4)).collect();
        let cv = QConv::new(&wt, &bias, (o, c, 1), 1, 0, true, &all_g);
        let ref_parts: Vec<(&[u32], &[Grid])> =
            part_bufs.iter().zip(&part_grids).map(|(b, g)| (b.as_slice(), g.as_slice())).collect();
        let (want_q, want_f) =
            ref_conv_parts(&ref_parts, n, h, w, &wt, &bias, (o, 1, 1, 0), true, false, None, &out_g);
        let views: Vec<C4<'_>> = part_bufs
            .iter()
            .zip(&part_zs)
            .map(|(b, z)| C4 { data: b, offset: 0, img_stride: (pc / 4) * hw, c: pc, zero: z[0] })
            .collect();
        let zslices: Vec<&[u8]> = part_zs.iter().map(|z| z.as_slice()).collect();
        let mut got_q = alloc(n, o, hw);
        let mut got_f = vec![f32::NAN; n * o * hw];
        pool(4).install(|| {
            conv_per_channel(
                &views,
                &zslices,
                n,
                h,
                w,
                &cv,
                None,
                QOut::U8PerChannel { data: &mut got_q, grids: &out_g },
            );
            conv_per_channel(&views, &zslices, n, h, w, &cv, None, QOut::F32 { data: &mut got_f });
        });
        assert_eq!(&got_q[..want_q.len()], want_q.as_slice(), "1x1 {part_count}-part u8 mismatch");
        assert!(
            got_f.iter().zip(&want_f).all(|(a, b)| a.to_bits() == b.to_bits()),
            "1x1 {part_count}-part f32 mismatch"
        );
    }

    // 4. Decoupled 1×1 `QConv` (`[8, 8]`) invoked with mismatched `xs` partition (`[4, 12]`) falls back cleanly.
    {
        let (n, h, w, o) = (1usize, 7usize, 11usize, 8usize);
        let hw = h * w;
        let g0: Vec<Grid> = (0..8).map(|i| Grid { scale: 0.004 + 0.0002 * i as f32, zero: 20 }).collect();
        let g1: Vec<Grid> = (0..8).map(|i| Grid { scale: 0.04 + 0.003 * i as f32, zero: 90 }).collect();
        let all_g: Vec<Grid> = g0.iter().chain(&g1).copied().collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.4, 3.0 + 0.1 * i as f32)).collect();
        let xf: Vec<f32> = (0..16 * hw).map(|_| rng.uniform(-0.2, 1.0)).collect();
        let mut x_all = alloc(n, 16, hw);
        quantize_per_channel(&xf, n, 16, hw, &all_g, &mut x_all);
        let wt: Vec<f32> = (0..o * 16).map(|_| rng.uniform(-0.3, 0.3)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.2, 0.2)).collect();
        let cv = QConv::new_parts(&wt, &bias, (o, 16, 1), 1, 0, true, &[&g0, &g1], true);
        let z_all: Vec<u8> = all_g.iter().map(|g| g.zero).collect();
        // Split `[4, 12]` across two views into `x_all`, differing from `cv`'s `[8, 8]` decoupled parts.
        let va = C4 { data: &x_all, offset: 0, img_stride: 4 * hw, c: 4, zero: z_all[0] };
        let vb = C4 { data: &x_all, offset: hw, img_stride: 4 * hw, c: 12, zero: z_all[4] };
        let v_m0 = C4 { data: &x_all, offset: 0, img_stride: 4 * hw, c: 8, zero: z_all[0] };
        let v_m1 = C4 { data: &x_all, offset: 2 * hw, img_stride: 4 * hw, c: 8, zero: z_all[8] };
        let mut q_match = alloc(n, o, hw);
        let mut q_mismatch = alloc(n, o, hw);
        conv_per_channel(
            &[v_m0, v_m1],
            &[&z_all[..8], &z_all[8..]],
            n,
            h,
            w,
            &cv,
            None,
            QOut::U8PerChannel { data: &mut q_match, grids: &out_g },
        );
        conv_per_channel(
            &[va, vb],
            &[&z_all[..4], &z_all[4..]],
            n,
            h,
            w,
            &cv,
            None,
            QOut::U8PerChannel { data: &mut q_mismatch, grids: &out_g },
        );
        assert_eq!(&q_match[..(o / 4) * hw], &q_mismatch[..(o / 4) * hw]);
    }
}

#[test]
fn packed_multipart_1x1_matches_unpacked_bit_exact() {
    let mut rng = Rng::seed(7099);
    // Exercises `should_pack_1x1_multi` across every `VECS ∈ {1, 2, 3}` copied and direct tail width
    // (`hw % 24 ∈ {0, 5, 8, 13, 16, 17}`), SiLU on/off, residual on/off, single- and multi-thread (`og_blk >= 4`),
    // multi-image (`n > 1`), deep-K (`total_ks >= 48, hw < 1024`), and `U8PerChannel` + `F32` outputs, comparing
    // against mismatched-partition unpacked multi-part execution (guaranteed bit-exact).
    for &(n, h, w, c0, c1, o, threads, silu, has_res) in &[
        (1usize, 32usize, 33usize, 16usize, 24usize, 20usize, 1usize, true, true), // hw=1056 (tail 0)
        (1, 17, 61, 16, 24, 20, 1, true, false), // hw=1037 (tail 5 -> VECS=1 copy)
        (1, 32, 34, 16, 24, 20, 1, false, true), // hw=1088 (tail 8 -> VECS=1 direct)
        (1, 19, 55, 16, 24, 20, 1, true, true),  // hw=1045 (tail 13 -> VECS=2 copy)
        (1, 40, 28, 16, 24, 20, 1, false, true), // hw=1120 (tail 16 -> VECS=2 direct)
        (1, 29, 37, 16, 24, 64, 4, true, false), // hw=1073 (tail 17 -> VECS=3 copy, mt og_blk>=4)
        (2, 7, 59, 96, 100, 16, 1, true, true),  // hw=413 (tail 5, n=2, total_ks=49>=48)
    ] {
        let c = c0 + c1;
        let hw = h * w;
        let g0: Vec<Grid> =
            (0..c0).map(|i| Grid { scale: 0.004 + 0.0002 * i as f32, zero: 15 + (i % 7) as u8 }).collect();
        let g1: Vec<Grid> =
            (0..c1).map(|i| Grid { scale: 0.04 + 0.002 * i as f32, zero: 95 + (i % 11) as u8 }).collect();
        let all_g: Vec<Grid> = g0.iter().chain(&g1).copied().collect();
        let out_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.4, 3.0 + 0.05 * i as f32)).collect();
        let res_g: Vec<Grid> = (0..o).map(|i| Grid::from_range(-0.6, 2.0 + 0.07 * i as f32)).collect();
        let xf: Vec<f32> = (0..n * c * hw).map(|_| rng.uniform(-0.3, 1.1)).collect();
        let mut x_all = alloc(n, c, hw);
        quantize_per_channel(&xf, n, c, hw, &all_g, &mut x_all);
        let mut r_buf = alloc(n, o, hw);
        let rf: Vec<f32> = (0..n * o * hw).map(|_| rng.uniform(-0.4, 1.2)).collect();
        quantize_per_channel(&rf, n, o, hw, &res_g, &mut r_buf);
        let wt: Vec<f32> = (0..o * c).map(|_| rng.uniform(-0.25, 0.25)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.2, 0.2)).collect();
        let cv = QConv::new_parts(&wt, &bias, (o, c, 1), 1, 0, silu, &[&g0, &g1], true);
        let z_all: Vec<u8> = all_g.iter().map(|g| g.zero).collect();
        // Matching partition hits `run_packed_1x1_multi_spec`; mismatched `[4, c-4]` hits unpacked `BMode::Rows`.
        let v_p0 = C4 { data: &x_all, offset: 0, img_stride: c / 4 * hw, c: c0, zero: z_all[0] };
        let v_p1 = C4 { data: &x_all, offset: c0 / 4 * hw, img_stride: c / 4 * hw, c: c1, zero: z_all[c0] };
        let v_u0 = C4 { data: &x_all, offset: 0, img_stride: c / 4 * hw, c: 4, zero: z_all[0] };
        let v_u1 = C4 { data: &x_all, offset: hw, img_stride: c / 4 * hw, c: c - 4, zero: z_all[4] };
        let rv = C4 { data: &r_buf, offset: 0, img_stride: o / 4 * hw, c: o, zero: res_g[0].zero };
        let res_arg = has_res.then_some((rv, res_g.as_slice()));
        let mut q_pack = alloc(n, o, hw);
        let mut q_ref = alloc(n, o, hw);
        let mut f_pack = vec![0f32; n * o * hw];
        let mut f_ref = vec![0f32; n * o * hw];
        pool(threads).install(|| {
            conv_per_channel(
                &[v_p0, v_p1],
                &[&z_all[..c0], &z_all[c0..]],
                n,
                h,
                w,
                &cv,
                res_arg,
                QOut::U8PerChannel { data: &mut q_pack, grids: &out_g },
            );
            conv_per_channel(
                &[v_u0, v_u1],
                &[&z_all[..4], &z_all[4..]],
                n,
                h,
                w,
                &cv,
                res_arg,
                QOut::U8PerChannel { data: &mut q_ref, grids: &out_g },
            );
            conv_per_channel(
                &[v_p0, v_p1],
                &[&z_all[..c0], &z_all[c0..]],
                n,
                h,
                w,
                &cv,
                res_arg,
                QOut::F32 { data: &mut f_pack },
            );
            conv_per_channel(
                &[v_u0, v_u1],
                &[&z_all[..4], &z_all[4..]],
                n,
                h,
                w,
                &cv,
                res_arg,
                QOut::F32 { data: &mut f_ref },
            );
        });
        assert_eq!(
            &q_pack[..n * (o / 4) * hw],
            &q_ref[..n * (o / 4) * hw],
            "packed multi u8 (n={n} {h}x{w} c={c0}+{c1} o={o})"
        );
        assert!(
            f_pack.iter().zip(&f_ref).all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed multi f32 (n={n} {h}x{w} c={c0}+{c1} o={o})"
        );
    }
}
