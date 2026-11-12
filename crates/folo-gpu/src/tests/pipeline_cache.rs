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

/// What one context saw: its cache file, whether it loaded the file, and the size of the cache's data right after
/// the open (before any pipeline was compiled), which is what wgpu was handed.
#[derive(Debug, PartialEq)]
struct Seen {
    file: std::path::PathBuf,
    loaded: bool,
    data_at_open: usize,
}

/// A context with the cache in `dir`, compiling and running two variants of a kernel: 1 + 2 added to zeros.
fn run(dir: &std::path::Path) -> Option<Seen> {
    let ctx = GpuContext::with_options(&GpuOptions {
        pipeline_cache_dir: Some(dir.to_path_buf()),
        ..GpuOptions::from_env()
    })
    .unwrap();
    let file = ctx.pipeline_cache_file()?.to_path_buf();
    let data_at_open = ctx.pipeline_cache_data_len().unwrap_or(0);
    let t = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    for n in [1, 2] {
        let k = ctx.kernel(&ADD_N, Spec::new().u32("N", n)).unwrap();
        ctx.dispatch(&k, &[t.raw()], &Params::new().u(100), [2, 1, 1]).unwrap();
    }
    assert_eq!(t.to_vec().unwrap(), vec![3.0; 100]);
    Some(Seen { file, loaded: ctx.pipeline_cache_loaded(), data_at_open })
}

#[test]
fn pipeline_cache_is_written_reused_and_survives_corruption() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-pipeline-cache");
    let Some(first) = run(&dir) else {
        eprintln!("adapter has no pipeline cache (not Vulkan); skipping");
        return;
    };
    assert!(!first.loaded, "an empty folder has nothing to load");
    let file = first.file.clone();
    // Dropping the context wrote the file, and the next context hands it to wgpu: its cache starts with the two
    // pipelines.
    let good = std::fs::read(&file).expect("cache file written on drop");
    assert!(good.len() > 64, "{} bytes", good.len());
    let second = run(&dir).unwrap();
    assert!(second.loaded, "the second context reuses the file");
    assert!(second.data_at_open > first.data_at_open, "{second:?} vs {first:?}");
    // A corrupt file: an empty cache, the same results, and a valid file again (truncated and foreign files are
    // covered by the header test in pipeline_cache.rs).
    let mut flipped = good.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0xff;
    std::fs::write(&file, &flipped).unwrap();
    let corrupt = run(&dir).unwrap();
    assert_eq!(corrupt, Seen { loaded: false, ..first }, "a corrupt file is not loaded");
    let again = std::fs::read(&file).unwrap();
    assert!(crate::pipeline_cache::decode_for_test(&again), "rewritten as a valid cache file");
}

#[test]
fn no_folder_no_cache() {
    let _shared = gpu!();
    let ctx =
        GpuContext::with_options(&GpuOptions { pipeline_cache_dir: None, ..GpuOptions::from_env() }).unwrap();
    assert!(ctx.pipeline_cache_file().is_none());
    assert!(!ctx.pipeline_cache_loaded());
}
