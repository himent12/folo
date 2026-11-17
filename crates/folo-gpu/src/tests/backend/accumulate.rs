//! In-place accumulation (`Backend::add_into`, the fan-in of autograd) and ops writing into recycled buffers
//! (`Backend::empty`).

use folo_core::{DType, Tensor, no_grad};

use super::fixtures::{assert_close_t, assert_same, dev, randn};
use crate::backend::ADD_INTO_DONE;

/// F32, and F16 where the device computes in it.
fn float_dtypes(dev: folo_core::Device) -> Vec<DType> {
    [DType::F32, DType::F16]
        .into_iter()
        .filter(|&d| d == DType::F32 || folo_core::backend::computes_f16(dev))
        .collect()
}

/// In-place additions the GPU backend ran on this thread so far.
fn in_place_adds() -> u64 {
    ADD_INTO_DONE.with(|n| n.get())
}

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
    // `[2, 4, 6, 6]` takes the vec4 kernel, `[2, 3, 5, 7]` (210 elements, not a multiple of 4) the scalar one.
    for shape in [[2, 4, 6, 6], [2, 3, 5, 7]] {
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
    for dtype in float_dtypes(dev) {
        let xg = x.detach().to_device(dev).unwrap().requires_grad_();
        let n0 = in_place_adds();
        let gg = f(&xg, &w.to_device(dev).unwrap(), dtype).backward().unwrap();
        assert!(in_place_adds() > n0, "{dtype}: the fan-in did not add in place");
        let got =
            gg.get(&xg).unwrap_or_else(|| panic!("{dtype}: no gradient for x: {gg:?}, x {:?}", xg.shape()));
        assert!(got.device().is_gpu());
        // F16: x, w and the products are rounded to f16 (rel. 2^-11 each, a few roundings deep).
        let (atol, rtol) = if dtype == DType::F16 { (2e-2, 4e-3) } else { (1e-5, 1e-5) };
        assert_close_t(got, want, atol, rtol, &format!("fan-in gradient {dtype} {shape:?}"));
    }
}

/// The in-place fan-in is bit-identical to the out-of-place add it replaces, at f32 and f16, through the vec4 and
/// the scalar kernel: `x` is read by two products, so its gradient is `w1 · 1` accumulated with `w2 · 1` (exact
/// copies of `w1` and `w2`), and must equal the GPU's own `w1 + w2` bit for bit.
#[test]
fn fan_in_is_bit_identical_to_the_out_of_place_add() {
    let dev = dev!();
    for (i, shape) in [[2, 4, 6, 6], [2, 3, 5, 7]].into_iter().enumerate() {
        for dtype in float_dtypes(dev) {
            let on = |t: Tensor| t.to_device(dev).unwrap().to_dtype(dtype).unwrap();
            let (w1, w2) = (on(randn(&shape, 40 + i as u64)), on(randn(&shape, 50 + i as u64)));
            let x = on(randn(&shape, 60 + i as u64)).requires_grad_();
            let loss =
                x.mul(&w1).unwrap().sum_all().unwrap().add(&x.mul(&w2).unwrap().sum_all().unwrap()).unwrap();
            let n0 = in_place_adds();
            let g = loss.backward().unwrap();
            assert_eq!(in_place_adds() - n0, 1, "{dtype} {shape:?}: one fan-in, added in place");
            let got = g.get(&x).unwrap();
            let want = {
                let _g = no_grad();
                w1.add(&w2).unwrap()
            };
            assert_eq!(got.dtype(), dtype);
            assert_same(got, &want, &format!("fan-in {dtype} {shape:?}"));
        }
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

/// Runs `jobs` on their own threads and waits for all of them: a panic fails the test as itself, a hang after 60 s
/// as `what`.
fn finish_all(what: &str, jobs: Vec<Box<dyn FnOnce() + Send>>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let n = jobs.len();
    let handles: Vec<_> = jobs
        .into_iter()
        .map(|job| {
            let tx = tx.clone();
            std::thread::spawn(move || {
                job();
                let _ = tx.send(());
            })
        })
        .collect();
    drop(tx);
    for _ in 0..n {
        match rx.recv_timeout(std::time::Duration::from_secs(60)) {
            Ok(()) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("{what}: deadlocked"),
            // A worker panicked before it sent: joining re-raises its panic.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    for h in handles {
        if let Err(e) = h.join() {
            std::panic::resume_unwind(e);
        }
    }
}

/// Opposing in-place additions on two threads (`a += b` and `b += a`), started together, both finish.
#[test]
fn opposing_in_place_additions_do_not_deadlock() {
    let dev = dev!();
    let a = Tensor::ones([256]).to_device(dev).unwrap();
    let b = Tensor::ones([256]).to_device(dev).unwrap();
    let start = std::sync::Arc::new(std::sync::Barrier::new(2));
    let jobs: Vec<Box<dyn FnOnce() + Send>> = [(a.clone(), b.clone()), (b.clone(), a.clone())]
        .into_iter()
        .map(|(dst, src)| {
            let start = start.clone();
            Box::new(move || {
                let _g = no_grad();
                start.wait();
                for _ in 0..300 {
                    dst.add_assign(&src).unwrap();
                }
            }) as Box<dyn FnOnce() + Send>
        })
        .collect();
    finish_all("opposing additions", jobs);
}
