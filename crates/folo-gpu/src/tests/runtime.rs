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
fn pool_evicts_least_recently_freed_and_caches_up_to_the_high_water_mark() {
    let ctx = gpu!();
    let usage = wgpu::BufferUsages::STORAGE;
    let mut pool = crate::pool::BufferPool::new(4096);
    let (a, b, c) = (
        pool.take(ctx.device(), 1024, usage),
        pool.take(ctx.device(), 2048, usage),
        pool.take(ctx.device(), 1536, usage),
    );
    // 4.5 KiB were live at once, so the cache may hold that much even though the cap is 4 KiB.
    pool.give(a, 1024);
    pool.give(b, 2048);
    pool.give(c, 1536);
    assert_eq!(pool.stats().cached_bytes, 4608);
    // A 3 KiB buffer pushes out the two least recently freed (1 KiB, then 2 KiB).
    let d = pool.take(ctx.device(), 3072, usage);
    pool.give(d, 3072);
    assert_eq!(pool.stats().cached_bytes, 1536 + 3072);
    let before = pool.stats();
    let _ = pool.take(ctx.device(), 1536, usage);
    let _ = pool.take(ctx.device(), 3072, usage);
    let _ = pool.take(ctx.device(), 1024, usage);
    let after = pool.stats();
    assert_eq!((after.hits - before.hits, after.misses - before.misses), (2, 1));
    pool.clear();
    assert_eq!(pool.stats().cached_bytes, 0);
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
    assert!(GpuTensor::empty(&ctx, &[usize::MAX], DType::F32).is_err());
    assert!(ctx.alloc(u64::MAX - 3).is_err());
    assert!(t.reshape(&[0, 1 << 40, 1 << 40]).is_err());
    let empty = GpuTensor::zeros(&ctx, &[0, 4], DType::F32).unwrap();
    assert_eq!(empty.unary(UnaryOp::Exp).unwrap().to_vec().unwrap(), Vec::<f32>::new());
    assert_eq!(empty.sum(1, false).unwrap().shape(), &[0]);
}

/// `stage` on another thread, then `upload_staged`: the bytes arrive in stream order (after work encoded earlier
/// into the same pooled buffer), including a length that is not a multiple of 4.
#[test]
fn staged_upload_from_another_thread() {
    let ctx = gpu!();
    for len in [1usize, 7, 4096, 300_001] {
        let data: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        let staged = {
            let (c, d) = (ctx.clone(), data.clone());
            std::thread::spawn(move || c.stage(&d).unwrap()).join().unwrap()
        };
        assert_eq!(staged.len(), len as u64);
        let buf = ctx.upload_staged(&staged).unwrap();
        drop(staged);
        let back = ctx.read(buf.raw(), 0, len as u64).unwrap();
        assert_eq!(back, data, "length {len}");
    }
}

/// A lost device (here: destroyed on purpose) makes waits and readbacks return `DeviceLost` instead of panicking
/// inside wgpu, and stays lost. Uses a private context so the shared one other tests use is untouched.
#[test]
fn lost_device_is_an_error_not_a_panic() {
    let _shared = gpu!();
    let ctx = crate::GpuContext::new().unwrap();
    let buf = ctx.upload(&[1, 2, 3, 4]).unwrap();
    ctx.sync().unwrap();
    ctx.device().destroy();
    let err = ctx.sync().and_then(|_| ctx.read(buf.raw(), 0, 4).map(|_| ())).unwrap_err();
    assert!(matches!(err, crate::GpuError::DeviceLost(_)), "{err}");
    assert!(ctx.lost_reason().is_some());
    assert!(matches!(ctx.sync(), Err(crate::GpuError::DeviceLost(_))));
    // As folo-core sees it through the backend: recognisable, unlike other backend errors.
    let core = |msg: String| folo_core::Error::Backend { backend: crate::backend::BACKEND_NAME, msg };
    assert!(crate::is_device_lost(&core(err.to_string())));
    assert!(!crate::is_device_lost(&core(crate::GpuError::Wgpu("validation".into()).to_string())));
}
