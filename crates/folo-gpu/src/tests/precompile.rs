//! Parallel precompile on a real device: a context records the kernels it used, the next one compiles them on
//! worker threads before they are asked for, falls back to a list of another shape, and a corrupt, stale or broken
//! list only means compiling lazily.

use super::gpu;
use crate::kernel_list;
use crate::{BinaryOp, GpuContext, GpuOptions, GpuTensor, UnaryOp};
use folo_test_support::TestDir;

fn context(dir: &std::path::Path) -> GpuContext {
    GpuContext::with_options(&GpuOptions {
        pipeline_cache_dir: Some(dir.to_path_buf()),
        ..GpuOptions::from_env()
    })
    .unwrap()
}

/// Three kernels (unary, binary, GEMM) with `extra` adding a fourth; checks the results.
fn work(ctx: &GpuContext, extra: bool) {
    let x = GpuTensor::from_slice(ctx, &[1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
    let y = x.unary(UnaryOp::Abs).unwrap().binary(BinaryOp::Add, &x).unwrap();
    assert_eq!(y.to_vec().unwrap(), [2.0, 0.0, 6.0, 0.0]);
    assert_eq!(y.matmul(&x).unwrap().to_vec().unwrap(), [2.0, -4.0, 6.0, -12.0]);
    if extra {
        assert_eq!(x.unary(UnaryOp::Neg).unwrap().to_vec().unwrap(), [-1.0, 2.0, -3.0, 4.0]);
    }
}

/// The kernels of `ctx`'s list for `kind` / `shape`, by name.
fn listed(ctx: &GpuContext, kind: &str, shape: &str) -> Vec<&'static str> {
    let path = ctx.kernel_list_file(kind, shape).unwrap();
    let id = kernel_list::identity_for(&ctx.adapter().get_info(), kind);
    kernel_list::read(&path, &id).iter().map(|k| k.name).collect()
}

#[test]
fn kernels_are_recorded_and_precompiled_next_time() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile");
    // No list yet: nothing to compile ahead; the run's kernels are recorded in order of first use.
    let a = context(&dir);
    assert_eq!(a.precompile("test", "s1"), 0);
    assert_eq!(a.precompile("test", "s1"), 0, "a second call does nothing");
    work(&a, false);
    a.save_kernel_lists();
    let first = listed(&a, "test", "s1");
    assert!(first.len() >= 3, "{first:?}");
    assert!(first.contains(&"gemm"), "{first:?}");
    assert!(a.kernels_without_registry().is_empty(), "{:?}", a.kernels_without_registry());
    drop(a);

    // The next context compiles them on the workers, and every first use takes the worker's pipeline.
    let b = context(&dir);
    let n = b.precompile("test", "s1");
    assert_eq!(n, first.len());
    b.precompile_wait();
    work(&b, false);
    assert_eq!(b.precompiled_taken(), n as u64);
    let file = b.kernel_list_file("test", "s1").unwrap();
    let before = std::fs::read(&file).unwrap();
    b.save_kernel_lists();
    assert_eq!(std::fs::read(&file).unwrap(), before, "nothing new: the list is not rewritten");
    drop(b);

    // A shorter run keeps the longer list; a run with a new kernel rewrites it with what it used.
    let c = context(&dir);
    c.precompile("test", "s1");
    let x = GpuTensor::from_slice(&c, &[1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
    x.unary(UnaryOp::Abs).unwrap().to_vec().unwrap();
    c.save_kernel_lists();
    assert_eq!(std::fs::read(&file).unwrap(), before, "a run that used fewer kernels keeps the list");
    work(&c, true);
    c.save_kernel_lists();
    assert_eq!(listed(&c, "test", "s1").len(), first.len() + 1);
    drop(c);

    // Another shape of the same kind starts from that list; another kind has none.
    let d = context(&dir);
    assert_eq!(d.precompile("test", "s2"), first.len() + 1);
    assert_eq!(d.precompile("other", "s1"), 0);
    d.precompile_wait();
    work(&d, false);
    d.save_kernel_lists();
    assert_eq!(listed(&d, "test", "s2").len(), first.len(), "its own list holds what it used");
}

#[test]
fn bad_lists_only_mean_compiling_lazily() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile-bad");
    let a = context(&dir);
    a.precompile("test", "s");
    work(&a, false);
    a.save_kernel_lists();
    let file = a.kernel_list_file("test", "s").unwrap();
    let id = kernel_list::identity_for(&a.adapter().get_info(), "test");
    let n = kernel_list::read(&file, &id).len();
    drop(a);

    // Corrupt: ignored, and rewritten valid by the run.
    let good = std::fs::read(&file).unwrap();
    let mut bad = good.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x55;
    std::fs::write(&file, &bad).unwrap();
    let b = context(&dir);
    assert_eq!(b.precompile("test", "s"), 0);
    work(&b, false);
    b.save_kernel_lists();
    assert_eq!(kernel_list::read(&file, &id).len(), n);
    drop(b);

    // Stale and broken lines: an unknown kernel is skipped; one whose constants do not build (a constant named like
    // the kernel's storage array) fails on a worker without failing anything else.
    let mut text = String::from_utf8(kernel_list::decode_payload_for_test(&good, &id)).unwrap();
    text.push_str("no_such_kernel - X=u1\nfill - Y=u1\n");
    std::fs::write(&file, kernel_list::encode_text_for_test(&text, &id)).unwrap();
    let c = context(&dir);
    assert_eq!(c.precompile("test", "s"), n + 1);
    c.precompile_wait();
    work(&c, false);
    assert_eq!(c.precompiled_taken(), n as u64);
}
