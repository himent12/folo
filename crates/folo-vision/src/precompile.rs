//! Parallel GPU pipeline precompile for training and inference ([`folo_gpu::GpuContext::precompile`]): a run
//! names its kind (command, model, precision, class count) and shape (image size, batch), and the kernels an
//! earlier run of the same kind and shape used compile on every core while this one opens its data and builds its
//! model (docs/GPU.md, Pipeline cache).

use folo_core::{DType, Device};
use folo_nn::io::ModelFile;

/// Starts precompiling on `device`'s GPU context the kernels an earlier run of `kind` at `shape` used, and
/// recording this run's for the next one. Returns how many kernels went to the workers (0 on the CPU, without a
/// pipeline cache folder, or with no list yet).
pub fn precompile(device: Device, kind: &str, shape: &str) -> usize {
    if device.is_cpu() {
        return 0;
    }
    folo_gpu::context_for(device).map_or(0, |ctx| ctx.precompile(kind, shape))
}

/// [`precompile`] for inference with `file`'s model on `device` at `dtype`: `command` is what runs it (`predict`,
/// `val`, `serve`) and `shape` the settings that change its kernels (image size, batch, head).
pub fn precompile_inference(
    file: &ModelFile,
    device: Device,
    dtype: DType,
    command: &str,
    shape: &str,
) -> usize {
    let m = file.metadata();
    let precision = if dtype == DType::F16 { "f16" } else { "f32" };
    let kind = format!("{command} {} {precision} nc {}", m.arch, m.classes.len());
    precompile(device, &kind, shape)
}
