//! In-place accumulation (`Backend::add_into`, the fan-in of autograd) and ops writing into recycled buffers
//! (`Backend::empty`).

use folo_core::{DType, Tensor, no_grad};

use super::fixtures::{assert_close_t, assert_same, dev, randn};

/// `dst_view += src_view` on the GPU leaves exactly the CPU's bits in the whole destination storage (the
/// elements outside the view untouched), for dense (`vec4` and scalar) and strided views of either side.
#[test]
fn add_assign_in_place_matches_cpu() {
    let dev = dev!();
    type View = fn(&Tensor) -> Tensor;
    let cases: Vec<(&str, Vec<usize>, View, View)> = vec![
        ("dense vec4", vec![1024], |t| t.clone(), |t| t.clone()),
        ("dense odd", vec![1027], |t| t.clone(), |t| t.clone()),
        // Dense at an offset in their storages: the linear kernel, vec4 (offset 256) and scalar (offset 7).
        (
            "dense vec4 at an offset",
            vec![4, 256],
            |t| t.narrow(0, 1, 2).unwrap(),
            |t| t.narrow(0, 2, 2).unwrap(),
        ),
        (
            "dense odd at an offset",
            vec![3, 7],
            |t| t.narrow(0, 1, 2).unwrap(),
            |t| t.narrow(0, 1, 2).unwrap(),
        ),
        // A channel slice of NCHW (a fan-in into one part of a concatenation) from a dense gradient.
        ("channel slice", vec![2, 5, 3, 4], |t| t.narrow(1, 1, 3).unwrap(), |t| t.narrow(1, 0, 3).unwrap()),
        // A strided destination from a transposed source.
        (
            "transposed source",
            vec![6, 6],
            |t| t.narrow(0, 2, 3).unwrap(),
            |t| t.transpose(0, 1).unwrap().narrow(0, 1, 3).unwrap(),
        ),
        (
            "broadcast source",
            vec![3, 8],
            |t| t.clone(),
            |t| t.narrow(0, 1, 1).unwrap().broadcast_to([3, 8]).unwrap(),
        ),
    ];
    for (i, (what, shape, dst_view, src_view)) in cases.into_iter().enumerate() {
        let base = randn(&shape, 10 + i as u64);
        let src = randn(&shape, 20 + i as u64);
        let (bc, bg) = (base.clone(), base.to_device(dev).unwrap());
        let (sc, sg) = (src.clone(), src.to_device(dev).unwrap());
        let _g = no_grad();
        dst_view(&bc).add_assign(&src_view(&sc)).unwrap();
        dst_view(&bg).add_assign(&src_view(&sg)).unwrap();
        assert_same(&bg, &bc, what);
    }
}

/// Gradients that fan in (one tensor read by several ops) sum to the CPU's on the GPU, in `F32` and in `F16`
/// (mixed precision's activations), with the accumulator dense, exclusive and therefore added into in place.
#[test]
fn fan_in_gradients_match_cpu() {
    let dev = dev!();
    // `[2, 4, 6, 6]` takes the vec4 kernel, `[2, 4, 5, 7]` (280 elements, odd rows) the scalar one at f16.
    for shape in [[2, 4, 6, 6], [2, 4, 5, 7]] {
        fan_in_case(dev, &shape);
    }
}

fn fan_in_case(dev: folo_core::Device, shape: &[usize; 4]) {
    let x = randn(shape, 1);
    let w = randn(shape, 2);
    // `h` is read four times: two products, a narrow (whose gradient is scattered into zeros) and a sum.
    let f = |x: &Tensor, w: &Tensor, dtype: DType| -> Tensor {
        let h = x.to_dtype(dtype).unwrap();
        let a = h.mul(&w.to_dtype(dtype).unwrap()).unwrap();
        let b = (&h * 3.0).unwrap();
        let c = h.narrow(1, 1, 2).unwrap().sum_all().unwrap().to_dtype(DType::F32).unwrap();
        let d = a.add(&b).unwrap().to_dtype(DType::F32).unwrap().sum_all().unwrap();
        let hf = h.to_dtype(DType::F32).unwrap();
        let e = hf.mul(&hf).unwrap().sum_all().unwrap();
        d.add(&c).unwrap().add(&e).unwrap()
    };
    let xc = x.detach().requires_grad_();
    let gc = f(&xc, &w, DType::F32).backward().unwrap();
    let want = gc.get(&xc).unwrap();
    for dtype in [DType::F32, DType::F16] {
        let xg = x.detach().to_device(dev).unwrap().requires_grad_();
        let gg = f(&xg, &w.to_device(dev).unwrap(), dtype).backward().unwrap();
        let got =
            gg.get(&xg).unwrap_or_else(|| panic!("{dtype}: no gradient for x: {gg:?}, x {:?}", xg.shape()));
        assert!(got.device().is_gpu());
        // F16: x, w and the products are rounded to f16 (rel. 2^-11 each, a few roundings deep).
        let (atol, rtol) = if dtype == DType::F16 { (2e-2, 4e-3) } else { (1e-5, 1e-5) };
        assert_close_t(got, want, atol, rtol, &format!("fan-in gradient {dtype} {shape:?}"));
    }
}

/// `Tensor::cat` writes into a buffer that is not cleared first: every element of a recycled, NaN-filled one
/// is overwritten.
#[test]
fn cat_overwrites_a_recycled_buffer() {
    let dev = dev!();
    // Even runs (whole words at f16) and odd ones (`[.., 5, 3]`: f16 halves written into words of the dirty buffer).
    for (hw, seed) in [([5, 4], 3), ([5, 3], 13)] {
        cat_case(dev, hw, seed);
    }
}

fn cat_case(dev: folo_core::Device, [h, w]: [usize; 2], seed: u64) {
    let parts = [randn(&[2, 3, h, w], seed), randn(&[2, 1, h, w], seed + 1), randn(&[2, 4, h, w], seed + 2)];
    let want = Tensor::cat(&parts, 1).unwrap();
    let gparts: Vec<Tensor> = parts.iter().map(|p| p.to_device(dev).unwrap()).collect();
    for dtype in [DType::F32, DType::F16] {
        for _ in 0..3 {
            // Same size class as the result: the pool hands this buffer back to the concatenation.
            drop(Tensor::full_with(want.shape().clone(), f64::NAN, dtype, dev).unwrap());
            let p: Vec<Tensor> = gparts.iter().map(|p| p.to_dtype(dtype).unwrap()).collect();
            let got = Tensor::cat(&p, 1).unwrap();
            assert_same(&got, &want.to_dtype(dtype).unwrap(), &format!("cat {dtype} {h}x{w}"));
        }
    }
}

/// Opposing in-place additions on two threads (`a += b` and `b += a`) both finish: the two storage locks are taken
/// in one order.
#[test]
fn opposing_in_place_additions_do_not_deadlock() {
    let dev = dev!();
    let a = Tensor::ones([256]).to_device(dev).unwrap();
    let b = Tensor::ones([256]).to_device(dev).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    for (dst, src) in [(a.clone(), b.clone()), (b.clone(), a.clone())] {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let _g = no_grad();
            for _ in 0..300 {
                dst.add_assign(&src).unwrap();
            }
            tx.send(()).unwrap();
        });
    }
    for _ in 0..2 {
        rx.recv_timeout(std::time::Duration::from_secs(60)).expect("opposing additions deadlocked");
    }
}
