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
    let g = Tensor::randn([40_005], 0.0, 1.0, &mut rng);
    assert_reproducible("elementwise simd", &|| {
        let ye = x.exp().unwrap();
        let ys = x.sigmoid().unwrap();
        let ysi = x.silu().unwrap();
        let ge = ye.backward_with(&g).unwrap().get(&x).unwrap().clone();
        let gs = ys.backward_with(&g).unwrap().get(&x).unwrap().clone();
        let gsi = ysi.backward_with(&g).unwrap().get(&x).unwrap().clone();
        vec![ye, ys, ysi, ge, gs, gsi]
    });
}
