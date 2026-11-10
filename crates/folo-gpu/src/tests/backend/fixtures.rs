//! Shared fixtures of the backend tests (also used by the other test modules): the GPU device, seeded inputs,
//! and the CPU-versus-GPU comparison of an op's outputs and gradients.

use folo_core::rng::Rng;
use folo_core::{DType, Device, Result, Tensor};

/// The registered GPU device, or `None` (with a note) when there is no suitable adapter.
pub(crate) fn device() -> Option<Device> {
    crate::tests::ctx()?;
    match crate::init() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("note: skipping GPU backend test: {e}");
            None
        }
    }
}

/// `let dev = dev!();`: the GPU device or an early return.
macro_rules! dev {
    () => {
        match $crate::tests::backend::device() {
            Some(d) => d,
            None => return,
        }
    };
}
pub(crate) use dev;

pub(crate) fn randn(shape: &[usize], seed: u64) -> Tensor {
    Tensor::randn(shape, 0.0, 1.0, &mut Rng::seed(seed))
}

pub(super) fn uniform(shape: &[usize], lo: f32, hi: f32, seed: u64) -> Tensor {
    Tensor::rand_uniform(shape, lo, hi, &mut Rng::seed(seed))
}

/// Asserts two f32 tensors (on any devices) match elementwise within `atol + rtol·|cpu|`.
#[track_caller]
pub(crate) fn assert_close_t(gpu: &Tensor, cpu: &Tensor, atol: f32, rtol: f32, what: &str) {
    assert_eq!(gpu.shape(), cpu.shape(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    let g = gpu.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let c = cpu.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    crate::tests::assert_close(&g, &c, atol, rtol, what);
}

#[track_caller]
pub(super) fn assert_same(gpu: &Tensor, cpu: &Tensor, what: &str) {
    assert_eq!(gpu.shape(), cpu.shape(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    assert!(gpu.device().is_gpu(), "{what}: result left the GPU");
    let (g, c) = (gpu.to_cpu_storage().unwrap(), cpu.to_cpu_storage().unwrap());
    // Compare bit patterns so NaN payloads and signed zeros count.
    assert_eq!(g.as_bytes(), c.as_bytes(), "{what}: {g:?} vs {c:?}");
}

/// Runs `f` on CPU and GPU copies of `inputs`; compares outputs and, when the output requires grad, the
/// gradient of every f32 input for a random output gradient.
#[track_caller]
pub(crate) fn check<F>(dev: Device, inputs: &[Tensor], atol: f32, rtol: f32, what: &str, f: F)
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let lift = |t: &Tensor, d: Device| -> Tensor {
        let t = t.to_device(d).unwrap().detach();
        if t.dtype() == DType::F32 { t.requires_grad_() } else { t }
    };
    let cpu: Vec<Tensor> = inputs.iter().map(|t| lift(t, Device::Cpu)).collect();
    let gpu: Vec<Tensor> = inputs.iter().map(|t| lift(t, dev)).collect();
    let yc = f(&cpu).unwrap_or_else(|e| panic!("{what}: cpu: {e}"));
    let yg = f(&gpu).unwrap_or_else(|e| panic!("{what}: gpu: {e}"));
    assert!(yg.device().is_gpu(), "{what}: result left the GPU");
    if yc.dtype() == DType::F32 {
        assert_close_t(&yg, &yc, atol, rtol, what);
    } else {
        assert_same(&yg, &yc, what);
    }
    if !yc.requires_grad() {
        return;
    }
    let seed = randn(yc.dims(), 0xfeed);
    let gc = yc.backward_with(&seed).unwrap();
    let gg = yg
        .backward_with(&seed.to_device(dev).unwrap())
        .unwrap_or_else(|e| panic!("{what}: gpu backward: {e}"));
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        if c.dtype() != DType::F32 {
            continue;
        }
        match (gc.get(c), gg.get(g)) {
            (Some(a), Some(b)) => {
                assert!(b.device().is_gpu(), "{what}: grad {i} left the GPU");
                assert_close_t(b, a, atol, rtol, &format!("{what}: grad of input {i}"));
            }
            (None, None) => {}
            (a, b) => panic!("{what}: grad {i} present on cpu {} / gpu {}", a.is_some(), b.is_some()),
        }
    }
}
