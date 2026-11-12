//! A fused folo-n eval forward on the GPU reads its concatenations and channel slices in place: at batch 1 it
//! encodes no buffer copy, and it matches the CPU. Its own test binary, so nothing else encodes on the shared
//! context while the forward is traced.
//!
//! Skips (with a note) without a hardware adapter; set `FOLO_GPU_TEST_SOFTWARE=1` to run on WARP / llvmpipe.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::Module;
use folo_vision::model::{FoloModel, ModelConfig};

#[track_caller]
fn close(got: &Tensor, want: &Tensor, atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (g, w) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
        assert!((a - b).abs() <= atol + rtol * b.abs(), "{what}[{i}]: got {a}, want {b}");
    }
}

#[test]
fn fused_forward_reads_concatenations_in_place() {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping GPU test, no adapter: {e}"),
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        return eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
    }
    let dev = folo_gpu::init().unwrap();
    let _g = no_grad();
    let mut m = FoloModel::new(ModelConfig::new("n", 4).unwrap()).unwrap();
    m.eval();
    m.fuse().unwrap();
    let x = Tensor::rand_uniform([1, 3, 96, 64], 0.0, 1.0, &mut Rng::seed(1));
    let want = m.forward(&x).unwrap();
    m.to_device(dev).unwrap();
    let xg = x.to_device(dev).unwrap();
    // Once untraced (pipelines built), then traced.
    m.forward(&xg).unwrap();
    ctx.trace_begin();
    let got = m.forward(&xg).unwrap();
    let ops = ctx.trace_end();
    let names: Vec<&str> = ops.iter().map(|o| o.name).collect();
    let copies = names.iter().filter(|n| **n == "copy").count();
    assert_eq!(copies, 0, "buffer copies in a fused batch-1 forward: {names:?}");
    assert!(names.iter().any(|n| n.ends_with("gemm_cat")), "no concatenation read in place: {names:?}");
    close(&got.box_raw, &want.box_raw, 2e-3, 2e-3, "box_raw");
    close(&got.cls_raw, &want.cls_raw, 2e-3, 2e-3, "cls_raw");
}
