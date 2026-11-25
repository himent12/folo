//! Results must be bitwise identical from run to run and for every rayon thread count: reductions use a fixed
//! blocking and combine order, accumulating scatters partition by destination, and the GEMM's K split is a
//! function of the shape only.

use folo_core::rng::Rng;
use folo_core::{AllDims, Tensor};

/// Bit patterns of every output of `f`, computed inside a pool of `threads` threads.
fn bits_in_pool(threads: usize, f: &(dyn Fn() -> Vec<Tensor> + Sync)) -> Vec<Vec<u32>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        f().iter().map(|t| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()).collect()
    })
}

fn assert_reproducible(name: &str, f: &(dyn Fn() -> Vec<Tensor> + Sync)) {
    let want = bits_in_pool(1, f);
    for threads in [1, 2, 3, 5, 8, 16, 28] {
        assert!(bits_in_pool(threads, f) == want, "{name}: result differs with {threads} threads");
    }
}

#[test]
fn full_reductions_are_bitwise_reproducible() {
    let mut rng = Rng::seed(11);
    // Values over many magnitudes so that a different summation order would change the bits.
    let n = 3_000_017;
    let v: Vec<f32> = (0..n).map(|_| rng.normal() * 10f32.powf(rng.uniform(-3.0, 3.0))).collect();
    let x = Tensor::from_vec(v, [n]).unwrap();
    let x4 = Tensor::randn([8, 16, 96, 96], 0.0, 1.0, &mut rng);
    assert_reproducible("sum_all", &|| {
        vec![
            x.sum_all().unwrap(),
            x.mean_all().unwrap(),
            x.max_all().unwrap(),
            x.min(AllDims, false).unwrap(),
            x4.sum_all().unwrap(),
            x4.sum([0, 2, 3], false).unwrap(),
            x4.mean([0, 2], true).unwrap(),
            x4.var([0, 2, 3], false, false).unwrap(),
            x4.var(3, false, false).unwrap(),
            x4.var(1, true, false).unwrap(),
            x4.var(AllDims, false, false).unwrap(),
            x4.sum(1, false).unwrap(),
            x4.transpose(1, 3).unwrap().sum_all().unwrap(),
        ]
    });
    // The fixed blocking stays accurate: compare with an f64 sum.
    let exact: f64 = x.to_vec::<f32>().unwrap().iter().map(|&v| v as f64).sum();
    let got = x.sum_all().unwrap().item::<f32>().unwrap() as f64;
    let scale: f64 = x.abs().unwrap().sum_all().unwrap().item::<f32>().unwrap() as f64;
    assert!((got - exact).abs() <= 1e-6 * scale, "{got} vs {exact}");
}

#[test]
fn training_gradients_are_bitwise_reproducible() {
    let mut rng = Rng::seed(12);
    // A stem-like conv: tiny C (M = 16, N = 27) and a long K = N·OH·OW, the K-split case of the GEMM.
    let x = Tensor::randn([4, 3, 96, 96], 0.0, 1.0, &mut rng).requires_grad_();
    let w = Tensor::randn([16, 3, 3, 3], 0.0, 0.3, &mut rng).requires_grad_();
    let w2 = Tensor::randn([64, 16, 3, 3], 0.0, 0.1, &mut rng).requires_grad_();
    let b2 = Tensor::randn([64], 0.0, 0.1, &mut rng).requires_grad_();
    // A pointwise conv wide enough for the weight gradient over input rows read in place.
    let w3 = Tensor::randn([24, 64, 1, 1], 0.0, 0.1, &mut rng).requires_grad_();
    let table = Tensor::randn([500, 64], 0.0, 1.0, &mut rng).requires_grad_();
    let rows: Vec<i64> = (0..4000).map(|_| rng.below(500) as i64).collect();
    let rows = Tensor::from_vec(rows, [4000]).unwrap();
    let cols: Vec<i64> = (0..4000 * 8).map(|_| rng.below(64) as i64).collect();
    let cols = Tensor::from_vec(cols, [4000, 8]).unwrap();
    assert_reproducible("grads", &|| {
        let y = x.conv2d(&w, None, 1, 1, 1, 1).unwrap().silu().unwrap();
        let y = y.conv2d(&w2, Some(&b2), 2, 1, 1, 1).unwrap();
        let y = y.conv2d(&w3, None, 1, 0, 1, 1).unwrap();
        let picked = table.index_select(0, &rows).unwrap();
        let gathered = picked.gather(1, &cols).unwrap();
        let loss = y.square().unwrap().mean_all().unwrap().add(&gathered.sum_all().unwrap()).unwrap();
        let g = loss.backward().unwrap();
        vec![
            loss,
            g.get(&x).unwrap().clone(),
            g.get(&w).unwrap().clone(),
            g.get(&w2).unwrap().clone(),
            g.get(&b2).unwrap().clone(),
            g.get(&w3).unwrap().clone(),
            g.get(&table).unwrap().clone(),
        ]
    });
}

#[test]
fn fused_inference_convs_are_bitwise_reproducible() {
    use folo_core::{Conv2dParams, UnaryOp, no_grad};
    let mut rng = Rng::seed(13);
    let p = |s: usize, pad: usize, g: usize| Conv2dParams {
        stride: [s; 2],
        padding: [pad; 2],
        dilation: [1; 2],
        groups: g,
    };
    // Ragged spatial sizes, so GEMM tiles (which move with the pool size) cut output rows unevenly.
    let x = Tensor::randn([1, 16, 41, 37], 0.0, 1.0, &mut rng);
    let x2 = Tensor::randn([2, 8, 41, 37], 0.0, 1.0, &mut rng);
    let w3 = Tensor::randn([24, 16, 3, 3], 0.0, 0.2, &mut rng);
    let ws2 = Tensor::randn([24, 16, 3, 3], 0.0, 0.2, &mut rng);
    let w1 = Tensor::randn([40, 24, 1, 1], 0.0, 0.2, &mut rng);
    let wd = Tensor::randn([16, 1, 3, 3], 0.0, 0.2, &mut rng);
    let b24 = Tensor::randn([24], 0.0, 0.2, &mut rng);
    let b40 = Tensor::randn([40], 0.0, 0.2, &mut rng);
    let wide = Tensor::randn([2, 24, 41, 37], 0.0, 1.0, &mut rng);
    let silu = Some(UnaryOp::Silu);
    assert_reproducible("fused convs", &|| {
        let _g = no_grad();
        // A conv over a contiguous part and a channel slice (strided across the batch), read in place.
        let parts = [&x2, &wide.narrow(1, 4, 16).unwrap()];
        vec![
            x.conv2d_act(&w3, Some(&b24), &p(1, 1, 1), silu).unwrap(),
            x.conv2d_act(&ws2, Some(&b24), &p(2, 1, 1), silu).unwrap(),
            x.conv2d_act(&wd, None, &p(1, 1, 16), silu).unwrap(),
            Tensor::conv2d_act_cat(&parts, &w1, Some(&b40), &p(1, 0, 1), silu).unwrap(),
        ]
    });
}

#[test]
fn elementwise_simd_ops_are_bitwise_reproducible() {
    let mut rng = Rng::seed(14);
    // Above PAR_MIN with an odd length so parallel chunks and padded AVX2 tails are both exercised.
    let x = Tensor::randn([40_005], 0.0, 2.5, &mut rng).requires_grad_();
    let xp = Tensor::rand_uniform([40_005], 0.05, 50.0, &mut rng).requires_grad_();
    let g = Tensor::randn([40_005], 0.0, 1.0, &mut rng);
    assert_reproducible("elementwise simd", &|| {
        let ye = x.exp().unwrap();
        let yl = xp.log().unwrap();
        let yt = x.tanh().unwrap();
        let ys = x.sigmoid().unwrap();
        let ysi = x.silu().unwrap();
        let ysp = x.softplus().unwrap();
        let ge = ye.backward_with(&g).unwrap().get(&x).unwrap().clone();
        let gl = yl.backward_with(&g).unwrap().get(&xp).unwrap().clone();
        let gt = yt.backward_with(&g).unwrap().get(&x).unwrap().clone();
        let gs = ys.backward_with(&g).unwrap().get(&x).unwrap().clone();
        let gsi = ysi.backward_with(&g).unwrap().get(&x).unwrap().clone();
        let gsp = ysp.backward_with(&g).unwrap().get(&x).unwrap().clone();
        vec![ye, yl, yt, ys, ysi, ysp, ge, gl, gt, gs, gsi, gsp]
    });
}

#[test]
fn softmax_and_backward_are_bitwise_reproducible() {
    let mut rng = Rng::seed(15);
    // Exercises inner == 1 short, inner == 1 online (> 4096), and inner > 1 column-tiled parallel paths.
    let x_row = Tensor::randn([64, 513], 0.0, 2.0, &mut rng).requires_grad_();
    let g_row = Tensor::randn([64, 513], 0.0, 1.0, &mut rng);
    let x_long = Tensor::randn([8, 5003], 0.0, 2.0, &mut rng).requires_grad_();
    let g_long = Tensor::randn([8, 5003], 0.0, 1.0, &mut rng);
    let x_col = Tensor::randn([2, 19, 1029], 0.0, 2.0, &mut rng).requires_grad_();
    let g_col = Tensor::randn([2, 19, 1029], 0.0, 1.0, &mut rng);
    assert_reproducible("softmax", &|| {
        let s1 = x_row.softmax(-1).unwrap();
        let ls1 = x_row.log_softmax(-1).unwrap();
        let s2 = x_long.softmax(-1).unwrap();
        let ls2 = x_long.log_softmax(-1).unwrap();
        let s3 = x_col.softmax(1).unwrap();
        let ls3 = x_col.log_softmax(1).unwrap();
        let gs1 = s1.backward_with(&g_row).unwrap().get(&x_row).unwrap().clone();
        let gls1 = ls1.backward_with(&g_row).unwrap().get(&x_row).unwrap().clone();
        let gs2 = s2.backward_with(&g_long).unwrap().get(&x_long).unwrap().clone();
        let gls2 = ls2.backward_with(&g_long).unwrap().get(&x_long).unwrap().clone();
        let gs3 = s3.backward_with(&g_col).unwrap().get(&x_col).unwrap().clone();
        let gls3 = ls3.backward_with(&g_col).unwrap().get(&x_col).unwrap().clone();
        vec![s1, ls1, s2, ls2, s3, ls3, gs1, gls1, gs2, gls2, gs3, gls3]
    });
}

#[test]
fn avg_pool_and_upsample2x2_are_bitwise_reproducible() {
    let mut rng = Rng::seed(16);
    let x = Tensor::randn([4, 32, 37, 41], 0.0, 1.0, &mut rng).requires_grad_();
    let g_avg = Tensor::randn([4, 32, 18, 20], 0.0, 1.0, &mut rng);
    let g_up = Tensor::randn([4, 32, 74, 82], 0.0, 1.0, &mut rng);
    assert_reproducible("avg_pool_upsample", &|| {
        let a2 = x.avg_pool2d(2, 2, 0).unwrap();
        let a3 = x.avg_pool2d(3, 1, 1).unwrap();
        let u2 = x.upsample_nearest2d(2).unwrap();
        let ga2 = a2.backward_with(&g_avg).unwrap().get(&x).unwrap().clone();
        let gu2 = u2.backward_with(&g_up).unwrap().get(&x).unwrap().clone();
        vec![a2, a3, u2, ga2, gu2]
    });
}

#[test]
fn cast_minmax_and_topk_are_bitwise_reproducible() {
    use folo_core::DType;
    let mut rng = Rng::seed(17);
    let x = Tensor::randn([64, 600], 0.0, 2.0, &mut rng);
    let x3 = Tensor::randn([4, 37, 257], 0.0, 2.0, &mut rng);
    assert_reproducible("cast_minmax_topk", &|| {
        let h = x.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
        let (tv, _) = x.topk(10, true).unwrap();
        let (sv, _) = x.topk(1, false).unwrap();
        let mx_row = x.max(1, false).unwrap();
        let mn_col = x3.min(1, false).unwrap();
        vec![h, tv, sv, mx_row, mn_col]
    });
}

#[test]
fn layer_norm_and_rms_norm_are_bitwise_reproducible() {
    let mut rng = Rng::seed(18);
    // Multi-row parallel (outer > 1), single long row (outer == 1, d > REDUCE_BLOCK), and var backward.
    let x = Tensor::randn([128, 389], 0.0, 1.5, &mut rng).requires_grad_();
    let w = Tensor::randn([389], 1.0, 0.3, &mut rng).requires_grad_();
    let b = Tensor::randn([389], 0.0, 0.2, &mut rng).requires_grad_();
    let g = Tensor::randn([128, 389], 0.0, 1.0, &mut rng);
    let x_long = Tensor::randn([1, 40_003], 0.0, 1.5, &mut rng).requires_grad_();
    let w_long = Tensor::randn([40_003], 1.0, 0.2, &mut rng).requires_grad_();
    let g_long = Tensor::randn([1, 40_003], 0.0, 1.0, &mut rng);
    assert_reproducible("norm_and_var", &|| {
        let ln = x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap();
        let gln = ln.backward_with(&g).unwrap();
        let rn = x.rms_norm(Some(&w), 1e-5).unwrap();
        let grn = rn.backward_with(&g).unwrap();
        let ln_l = x_long.layer_norm(Some(&w_long), None, 1e-5).unwrap();
        let gln_l = ln_l.backward_with(&g_long).unwrap();
        let rn_l = x_long.rms_norm(Some(&w_long), 1e-5).unwrap();
        let grn_l = rn_l.backward_with(&g_long).unwrap();
        vec![
            ln,
            gln.get(&x).unwrap().clone(),
            gln.get(&w).unwrap().clone(),
            gln.get(&b).unwrap().clone(),
            rn,
            grn.get(&x).unwrap().clone(),
            grn.get(&w).unwrap().clone(),
            ln_l,
            gln_l.get(&x_long).unwrap().clone(),
            gln_l.get(&w_long).unwrap().clone(),
            rn_l,
            grn_l.get(&x_long).unwrap().clone(),
            grn_l.get(&w_long).unwrap().clone(),
        ]
    });
}
