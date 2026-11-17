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

/// The in-place kernel (`ew::add_into`) leaves exactly the CPU's bits in the whole destination buffer, the elements
/// outside the view untouched: dense views (`vec4` and scalar, also at an offset: the linear kernel) and strided ones
/// on either side, f32 and f16 (the sum in f32, rounded to f16 once, as the CPU's `(a as f32 + b as f32) as f16`).
#[test]
fn add_into_kernel_matches_cpu() {
    let Some(ctx) = crate::tests::ctx() else { return };
    type View = fn(&Tensor) -> Tensor;
    let cases: Vec<(&str, Vec<usize>, View, View)> = vec![
        ("dense vec4", vec![1024], |t| t.clone(), |t| t.clone()),
        ("dense odd", vec![1027], |t| t.clone(), |t| t.clone()),
        // Dense at an offset in their buffers: the linear kernel, vec4 (offset 256) and scalar (offset 7).
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
        // A channel slice of NCHW from another channel slice.
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
    let strided = |t: &Tensor| crate::kernels::strided::Strided {
        offset: t.layout().offset(),
        dims: t.layout().dims().to_vec(),
        strides: t.layout().strides().to_vec(),
    };
    for (dtype, gdt) in [(DType::F32, crate::DType::F32), (DType::F16, crate::DType::F16)] {
        if dtype == DType::F16 && !ctx.caps().shader_f16 {
            continue;
        }
        for (i, (what, shape, dst_view, src_view)) in cases.iter().enumerate() {
            // Values representable in `dtype`, so the CPU and the GPU start from the same bits.
            let round = |t: Tensor| t.to_dtype(dtype).unwrap().to_dtype(DType::F32).unwrap();
            let (base, src) = (round(randn(shape, 10 + i as u64)), round(randn(shape, 20 + i as u64)));
            let want = {
                let _g = no_grad();
                // A copy: `clone` would share (and so change) `base`'s storage.
                let b = Tensor::from_vec(base.to_vec::<f32>().unwrap(), base.shape().clone()).unwrap();
                dst_view(&b).add_assign(&src_view(&src)).unwrap();
                b.to_dtype(dtype).unwrap().to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap()
            };
            let bg = super::super::coop::upload(&ctx, &base.to_vec::<f32>().unwrap(), gdt);
            let sg = super::super::coop::upload(&ctx, &src.to_vec::<f32>().unwrap(), gdt);
            let (dv, sv) = (strided(&dst_view(&base)), strided(&src_view(&src)));
            let x = crate::kernels::ew::EwInput { buf: sg.raw(), view: &sv };
            crate::kernels::ew::add_into(&ctx, &dv.dims, x, bg.raw(), &dv, gdt).unwrap();
            let got = bg.to_vec().unwrap();
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{what} {dtype}");
        }
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

/// `Tensor::cat` writes into a buffer that is not cleared first: every element of a NaN-filled one is overwritten.
/// `poison_uninit` fills every result buffer the backend allocates with NaN, so the test does not depend on which
/// recycled buffer the pool hands out (other threads share the pool). It stays on: the other tests of this crate
/// write every element of their results, and an element they did not would show up as NaN, which is a bug anyway.
#[test]
fn cat_overwrites_a_recycled_buffer() {
    let dev = dev!();
    crate::poison_uninit(true);
    // Even runs (whole words at f16) and odd ones (`[.., 5, 3]`: f16 halves written into words of the dirty buffer).
    for (hw, seed) in [([5, 4], 3), ([5, 3], 13)] {
        cat_case(dev, hw, seed);
    }
}

fn cat_case(dev: folo_core::Device, [h, w]: [usize; 2], seed: u64) {
    let parts = [randn(&[2, 3, h, w], seed), randn(&[2, 1, h, w], seed + 1), randn(&[2, 4, h, w], seed + 2)];
    let want = Tensor::cat(&parts, 1).unwrap();
    let gparts: Vec<Tensor> = parts.iter().map(|p| p.to_device(dev).unwrap()).collect();
    for dtype in float_dtypes(dev) {
        let p: Vec<Tensor> = gparts.iter().map(|p| p.to_dtype(dtype).unwrap()).collect();
        let got = Tensor::cat(&p, 1).unwrap();
        assert_same(&got, &want.to_dtype(dtype).unwrap(), &format!("cat {dtype} {h}x{w}"));
    }
}
