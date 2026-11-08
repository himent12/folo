//! Context, buffer pool, pipeline cache and batching behaviour.

use super::{Rng, gpu};
use std::sync::Arc;

use crate::context::parse_backends;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
use crate::{DType, GpuTensor, UnaryOp};

#[test]
fn adapter_summary_is_populated() {
    let ctx = gpu!();
    let s = ctx.summary();
    assert!(!s.name.is_empty());
    assert!(s.max_compute_workgroup_storage_size >= 16 * 1024);
    assert!(format!("{s}").contains(&s.name));
}

#[test]
fn backend_parsing() {
    assert_eq!(parse_backends("vulkan"), Some(wgpu::Backends::VULKAN));
    assert_eq!(parse_backends(" DX12 , metal"), Some(wgpu::Backends::DX12 | wgpu::Backends::METAL));
    assert_eq!(parse_backends("nonsense"), None);
    assert_eq!(parse_backends(""), None);
}

#[test]
fn pool_reuses_freed_buffers() {
    let ctx = gpu!();
    // A size no other test uses, so concurrent tests do not steal the freed buffer.
    let bytes = 3 * 1024 * 1024 + 4 * 977;
    let a = ctx.alloc(bytes).unwrap();
    let size = a.size();
    assert!(size >= bytes);
    drop(a);
    let before = ctx.pool_stats();
    let b = ctx.alloc(bytes).unwrap();
    let after = ctx.pool_stats();
    assert_eq!(b.size(), size);
    assert!(after.hits > before.hits, "{before:?} -> {after:?}");
}

#[test]
fn upload_after_reuse_does_not_clobber_pending_reads() {
    let ctx = gpu!();
    let n = 4096 + 17;
    let x1 = vec![1.5f32; n];
    let a = GpuTensor::from_slice(&ctx, &x1, &[n]).unwrap();
    let y = a.unary(UnaryOp::Square).unwrap(); // encoded, not yet submitted
    drop(a); // its buffer goes back to the pool while `y` still depends on it
    let b = GpuTensor::from_slice(&ctx, &vec![-7.0f32; n], &[n]).unwrap(); // may reuse that buffer
    assert!(y.to_vec().unwrap().iter().all(|&v| v == 2.25));
    assert!(b.to_vec().unwrap().iter().all(|&v| v == -7.0));
}

#[test]
fn long_chains_span_several_batches() {
    let ctx = gpu!();
    let mut rng = Rng::new(40);
    let x = rng.vec(1000, -1.0, 1.0);
    let one = GpuTensor::from_slice(&ctx, &[1.0], &[1]).unwrap();
    let mut t = GpuTensor::from_slice(&ctx, &x, &[1000]).unwrap();
    // More dispatches than one batch holds, and more than one uniform chunk.
    for _ in 0..2500 {
        t = t.add(&one).unwrap();
    }
    ctx.sync().unwrap();
    let got = t.to_vec().unwrap();
    for (g, x) in got.iter().zip(&x) {
        assert!((g - (x + 2500.0)).abs() < 1e-3, "{g} vs {}", x + 2500.0);
    }
}

#[test]
fn pipeline_cache_hits() {
    static ADD_ONE: KernelDef = KernelDef {
        name: "test_add_one",
        storage: &[Access::ReadWrite],
        source: Source::Static(concat!(
            "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n",
            "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n",
            "    if (g.x < pu(0u)) { X[g.x] += f32(N); }\n",
            "}\n",
        )),
    };
    let ctx = gpu!();
    let k1 = ctx.kernel(&ADD_ONE, Spec::new().u32("N", 1)).unwrap();
    let k2 = ctx.kernel(&ADD_ONE, Spec::new().u32("N", 1)).unwrap();
    let k3 = ctx.kernel(&ADD_ONE, Spec::new().u32("N", 2)).unwrap();
    assert!(Arc::ptr_eq(&k1, &k2), "same key, same pipeline");
    assert!(!Arc::ptr_eq(&k1, &k3), "different constants, different pipeline");
    // The specialisation constant really changes the code.
    let t = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    let p = Params::new().u(100);
    ctx.dispatch(&k1, &[t.raw()], &p, [2, 1, 1]).unwrap();
    ctx.dispatch(&k3, &[t.raw()], &p, [2, 1, 1]).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![3.0; 100]);
    // A shader that fails to compile is an error, not a panic.
    static BROKEN: KernelDef =
        KernelDef { name: "test_broken", storage: &[], source: Source::Static("fn main() { nope }") };
    assert!(ctx.kernel(&BROKEN, Spec::new()).is_err());
}

#[test]
fn shape_and_dtype_errors() {
    let ctx = gpu!();
    assert!(GpuTensor::from_slice(&ctx, &[1.0, 2.0], &[3]).is_err());
    let t = GpuTensor::zeros(&ctx, &[2, 3], DType::F32).unwrap();
    assert!(t.reshape(&[4, 2]).is_err());
    assert_eq!(t.reshape(&[3, usize::MAX]).unwrap().shape(), &[3, 2]);
    assert_eq!(t.to_vec().unwrap(), vec![0.0; 6]);
    let h = t.to_dtype(DType::F16).unwrap();
    assert!(h.unary(UnaryOp::Relu).is_err());
    assert!(t.to_f16_vec().is_err());
    // Shapes whose element count (or any sub-product) overflows are rejected before allocating.
    assert!(GpuTensor::empty(&ctx, &[1 << 32, 1 << 32], DType::F32).is_err());
    assert!(GpuTensor::empty(&ctx, &[0, 1 << 40, 1 << 40], DType::F32).is_err());
    assert!(GpuTensor::empty(&ctx, &[1 << 40], DType::F32).is_err());
    assert!(t.reshape(&[0, 1 << 40, 1 << 40]).is_err());
    let empty = GpuTensor::zeros(&ctx, &[0, 4], DType::F32).unwrap();
    assert_eq!(empty.unary(UnaryOp::Exp).unwrap().to_vec().unwrap(), Vec::<f32>::new());
    assert_eq!(empty.sum(1, false).unwrap().shape(), &[0]);
}
