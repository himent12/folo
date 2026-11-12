//! The persistent pipeline cache on a real device: written on drop, reused, and a corrupt or foreign file falls back
//! to an empty cache without an error.

use super::gpu;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
use crate::{DType, GpuContext, GpuOptions, GpuTensor};
use folo_test_support::TestDir;

static ADD_N: KernelDef = KernelDef {
    name: "test_cache_add_n",
    storage: &[Access::ReadWrite],
    source: Source::Static(concat!(
        "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n",
        "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n",
        "    if (g.x < pu(0u)) { X[g.x] += f32(N); }\n",
        "}\n",
    )),
};

/// A context with the cache in `dir`, compiling and running two variants of a kernel: 1 + 2 added to zeros.
fn run(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let ctx = GpuContext::with_options(&GpuOptions {
        pipeline_cache_dir: Some(dir.to_path_buf()),
        ..GpuOptions::from_env()
    })
    .unwrap();
    let file = ctx.pipeline_cache_file()?.to_path_buf();
    let t = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    for n in [1, 2] {
        let k = ctx.kernel(&ADD_N, Spec::new().u32("N", n)).unwrap();
        ctx.dispatch(&k, &[t.raw()], &Params::new().u(100), [2, 1, 1]).unwrap();
    }
    assert_eq!(t.to_vec().unwrap(), vec![3.0; 100]);
    Some(file)
}

#[test]
fn pipeline_cache_is_written_reused_and_survives_corruption() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-pipeline-cache");
    let Some(file) = run(&dir) else {
        eprintln!("adapter has no pipeline cache (not Vulkan); skipping");
        return;
    };
    // Dropping the context wrote the file, and it is valid for this adapter.
    let good = std::fs::read(&file).expect("cache file written on drop");
    assert!(good.len() > 64, "{} bytes", good.len());
    run(&dir);
    assert!(std::fs::read(&file).unwrap().len() >= good.len());
    // Corrupt, truncated and foreign files: an empty cache, the same results, and a valid file again.
    let mut flipped = good.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0xff;
    let mut foreign = good.clone();
    foreign[20] ^= 1;
    for bad in [flipped, good[..good.len() / 3].to_vec(), foreign] {
        std::fs::write(&file, &bad).unwrap();
        run(&dir);
        let again = std::fs::read(&file).unwrap();
        assert!(crate::pipeline_cache::decode_for_test(&again), "rewritten as a valid cache file");
    }
}

#[test]
fn no_folder_no_cache() {
    let _shared = gpu!();
    let ctx =
        GpuContext::with_options(&GpuOptions { pipeline_cache_dir: None, ..GpuOptions::from_env() }).unwrap();
    assert!(ctx.pipeline_cache_file().is_none());
}
