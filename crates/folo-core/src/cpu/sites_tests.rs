//! `conv2d_act_at` on the CPU: at every site, bit for bit what the dense inference conv (`conv2d_act`) computes,
//! for the Winograd tile list, the pointwise gathered GEMM and the dense fallback; and the tile list writes its
//! own tiles only.

use super::conv::{
    ConvGeom, ConvInput, ConvPart, transform_weights, winograd_calls, winograd_forward, winograd_forward_at,
};
use crate::rng::Rng;
use crate::{Conv2dParams, Tensor, UnaryOp, no_grad};

const P3: Conv2dParams = Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };
const P1: Conv2dParams = Conv2dParams { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 };

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// Sites of an `[n, *, oh, ow]` output: every corner and border midpoint of every image, plus `extra` random ones.
fn sites(n: usize, oh: usize, ow: usize, extra: usize, rng: &mut Rng) -> Vec<u32> {
    let mut v = Vec::new();
    for img in 0..n {
        for (y, x) in
            [(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1), (oh / 2, 0), (0, ow / 2), (oh - 1, ow / 2)]
        {
            v.push((img * oh * ow + y * ow + x) as u32);
        }
    }
    v.extend(
        (0..extra).map(|_| (rng.uniform(0.0, (n * oh * ow) as f32) as usize).min(n * oh * ow - 1) as u32),
    );
    v
}

/// `conv2d_act_at` against `conv2d_act` at `at`, bit for bit, and the Winograd forwards it ran.
fn check(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    p: &Conv2dParams,
    act: Option<UnaryOp>,
    at: &[u32],
) -> usize {
    let _g = no_grad();
    let dense = bits(&x.conv2d_act(w, b, p, act).unwrap());
    let before = winograd_calls()[1];
    let t = Tensor::from_vec(at.to_vec(), [at.len()]).unwrap();
    let y = x.conv2d_act_at(w, b, p, act, &t).unwrap();
    let ran = winograd_calls()[1] - before;
    let (_, o, oh, ow) = y.dims4().unwrap();
    assert_eq!(y.dims(), x.conv2d_act(w, b, p, act).unwrap().dims());
    let y = bits(&y);
    for &s in at {
        let (img, pix) = (s as usize / (oh * ow), s as usize % (oh * ow));
        for oc in 0..o {
            let i = (img * o + oc) * oh * ow + pix;
            assert_eq!(y[i], dense[i], "site {s}, channel {oc}");
        }
    }
    ran
}

/// Winograd convs (odd sides, two images, a single-tile-wide map, every site) give the dense bits at their sites.
#[test]
fn winograd_sites_are_the_dense_values() {
    let mut rng = Rng::seed(3);
    for (n, c, o, h, w) in [(2, 64, 64, 13, 11), (1, 96, 64, 20, 20), (3, 64, 64, 1, 7), (1, 64, 72, 11, 11)]
    {
        let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng);
        let wt = Tensor::randn([o, c, 3, 3], 0.0, 0.05, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let at = sites(n, h, w, 9, &mut rng);
        assert_eq!(
            check(&x, &wt, Some(&b), &P3, Some(UnaryOp::Silu), &at),
            1,
            "{n}x{c}x{h}x{w}: not Winograd"
        );
        let all: Vec<u32> = (0..(n * h * w) as u32).rev().collect();
        assert_eq!(check(&x, &wt, None, &P3, None, &all), 1);
        assert_eq!(check(&x, &wt, Some(&b), &P3, None, &[]), 1);
    }
}

/// Pointwise convs, with and without bias and SiLU, and with two images, give the dense bits at their sites.
#[test]
fn pointwise_sites_are_the_dense_values() {
    let mut rng = Rng::seed(4);
    for (n, c, o, h, w) in [(1, 64, 64, 20, 20), (2, 37, 13, 9, 7), (1, 300, 70, 5, 6)] {
        let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng);
        let wt = Tensor::randn([o, c, 1, 1], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let at = sites(n, h, w, 11, &mut rng);
        assert_eq!(check(&x, &wt, Some(&b), &P1, None, &at), 0);
        assert_eq!(check(&x, &wt, Some(&b), &P1, Some(UnaryOp::Silu), &at), 0);
        assert_eq!(check(&x, &wt, None, &P1, Some(UnaryOp::Relu), &at), 0);
    }
}

/// Convs neither path takes (stride 2, depthwise, a 3x3 conv too small for Winograd) run densely: same values.
#[test]
fn other_convs_fall_back_to_dense() {
    let mut rng = Rng::seed(5);
    let x = Tensor::randn([2, 16, 10, 9], 0.0, 1.0, &mut rng);
    let at = sites(2, 5, 5, 4, &mut rng);
    let s2 = Conv2dParams { stride: [2, 2], ..P3 };
    let w = Tensor::randn([8, 16, 3, 3], 0.0, 0.1, &mut rng);
    assert_eq!(check(&x, &w, None, &s2, Some(UnaryOp::Silu), &at), 0);
    let dw = Conv2dParams { groups: 16, ..P3 };
    let wd = Tensor::randn([16, 1, 3, 3], 0.0, 0.1, &mut rng);
    assert_eq!(check(&x, &wd, None, &dw, Some(UnaryOp::Silu), &sites(2, 10, 9, 4, &mut rng)), 0);
    assert_eq!(check(&x, &w, None, &P3, None, &sites(2, 10, 9, 4, &mut rng)), 0, "8x16 took Winograd");
}

/// Sites outside the output, and inputs that do not fit, are errors.
#[test]
fn bad_sites_are_errors() {
    let _g = no_grad();
    let x = Tensor::zeros([1, 64, 4, 4]);
    let w = Tensor::zeros([64, 64, 3, 3]);
    let at = Tensor::from_vec(vec![16u32], [1]).unwrap();
    assert!(x.conv2d_act_at(&w, None, &P3, None, &at).is_err(), "site 16 of a 4x4 output");
    let f = Tensor::zeros([1]);
    assert!(x.conv2d_act_at(&w, None, &P3, None, &f).is_err(), "F32 sites");
}

/// The tile list writes exactly its tiles, with the dense values, at every thread count (one block on one
/// thread, the two-pass split on three, many blocks on the full pool); every other element keeps its value.
#[test]
fn tile_list_writes_its_tiles_only() {
    let mut rng = Rng::seed(6);
    let (n, c, o, h, w) = (2, 64, 48, 15, 13);
    let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng).to_vec::<f32>().unwrap();
    let wt = Tensor::randn([o, c, 3, 3], 0.0, 0.05, &mut rng).to_vec::<f32>().unwrap();
    let g = ConvGeom { n, c, h, w, o, kh: 3, kw: 3, oh: h, ow: w, p: P3 };
    let u = transform_weights(&wt, o, c);
    let part = [ConvPart::contiguous(&x, c, h * w)];
    let input = ConvInput { parts: &part };
    let (tw, all) = (w.div_ceil(2), h.div_ceil(2) * w.div_ceil(2));
    let mut dense = vec![0f32; n * o * h * w];
    winograd_forward(input, &u, &mut dense, &g, None);
    // Image 0: a scattered list with runs, the last column and the last row; image 1: nothing.
    let list: Vec<u32> =
        vec![0, 1, 2, 5, (tw + 3) as u32, (2 * tw - 1) as u32, (all - 2) as u32, (all - 1) as u32];
    let lists = vec![list.clone(), Vec::new()];
    for threads in [1, 3, rayon::current_num_threads()] {
        let mut out = vec![f32::NAN; n * o * h * w];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| winograd_forward_at(input, &u, &mut out, &g, None, &lists));
        for (i, (&got, &want)) in out.iter().zip(&dense).enumerate() {
            let (img, pix) = (i / (o * h * w), i % (h * w));
            let tile = ((pix / w) / 2 * tw + (pix % w) / 2) as u32;
            if img == 0 && list.contains(&tile) {
                assert_eq!(got.to_bits(), want.to_bits(), "{threads} threads: element {i}");
            } else {
                assert!(got.is_nan(), "{threads} threads: element {i} outside the list was written");
            }
        }
    }
}
