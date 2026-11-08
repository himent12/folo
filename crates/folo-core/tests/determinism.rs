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
    let w2 = Tensor::randn([32, 16, 3, 3], 0.0, 0.1, &mut rng).requires_grad_();
    let b2 = Tensor::randn([32], 0.0, 0.1, &mut rng).requires_grad_();
    let table = Tensor::randn([500, 64], 0.0, 1.0, &mut rng).requires_grad_();
    let rows: Vec<i64> = (0..4000).map(|_| rng.below(500) as i64).collect();
    let rows = Tensor::from_vec(rows, [4000]).unwrap();
    let cols: Vec<i64> = (0..4000 * 8).map(|_| rng.below(64) as i64).collect();
    let cols = Tensor::from_vec(cols, [4000, 8]).unwrap();
    assert_reproducible("grads", &|| {
        let y = x.conv2d(&w, None, 1, 1, 1, 1).unwrap().silu().unwrap();
        let y = y.conv2d(&w2, Some(&b2), 2, 1, 1, 1).unwrap();
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
            g.get(&table).unwrap().clone(),
        ]
    });
}
