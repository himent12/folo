//! The devices a run can use: placing a model ([`place_model`], [`can_place`]), whether mixed precision is
//! available ([`amp_support`]), and recognising and explaining a lost GPU.

use std::path::Path;

use folo_core::Device;

use crate::error::{Error, Result};
use crate::model::FoloModel;

/// Whether a run on `device` can use mixed precision (`amp`): a GPU whose folo-gpu context has tensor cores (on
/// the RTX 4080 SUPER, Vulkan with `FOLO_GPU_TENSOR_CORES=1`, which `folo train --amp` requests itself and
/// `folo serve --tensor-cores` for the server). Switches them on if they were switched off at run time. Anything
/// else is `not_supported`, naming why: the f16 gradient kernels exist without tensor cores too, but they bring no
/// speed (on Ada, non-tensor f16 arithmetic is no faster than f32), so AMP refuses to run there.
pub fn amp_support(device: Device) -> Result<()> {
    amp_check(device, true)
}

/// [`amp_support`]'s answer without switching anything on: what `/api/system` reports before a run (a context
/// whose tensor cores were switched off at run time qualifies, since a run switches them back on).
pub fn amp_available(device: Device) -> Result<()> {
    amp_check(device, false)
}

fn amp_check(device: Device, activate: bool) -> Result<()> {
    let why = if device.is_cpu() {
        "the run is on the CPU".to_string()
    } else {
        match folo_gpu::context_for(device) {
            None => format!("{device} has no folo-gpu context"),
            Some(ctx) if ctx.can_enable_tensor_cores() && (!activate || ctx.set_tensor_cores(true)) => {
                return Ok(());
            }
            Some(ctx) => format!("{device} has tensor cores {}", ctx.tensor_cores_status()),
        }
    };
    Err(amp_not_supported(&why))
}

/// The `not_supported` error [`amp_support`] returns, naming `why` (for example `gpu:0 has tensor cores off
/// (...)`); public so a stand-in for the trainer (folo-server's mock) words it the same.
pub fn amp_not_supported(why: &str) -> Error {
    Error::NotSupported(format!(
        "mixed-precision training (amp) needs a GPU with tensor cores (f16 cooperative matrices): {why}; train \
         without amp (f32), or on such a GPU (`folo train --amp` asks for them; start `folo serve` or `folo mcp` with \
         `--tensor-cores`)"
    ))
}

/// Put a freshly built (CPU) model on `device` (`Module::to_device`: parameters, buffers and, through them, the
/// optimizer state and EMA built from it). [`can_place`] says whether that device can be used.
pub fn place_model(mut model: FoloModel, device: Device) -> Result<FoloModel> {
    can_place(device)?;
    if !device.is_cpu() {
        model.to_device(device)?;
    }
    Ok(model)
}

/// Whether [`place_model`] can put a model on `device`: the CPU always, a GPU once its folo-core backend is
/// registered (folo-gpu's `init()`).
pub fn can_place(device: Device) -> Result<()> {
    if device.is_cpu() || folo_core::backend::backend(device).is_ok() {
        Ok(())
    } else {
        Err(device_not_found(device))
    }
}

/// `device` cannot be used: no GPU backend is registered for it (no such adapter, or no usable GPU at all).
pub(crate) fn device_not_found(device: Device) -> Error {
    let have: Vec<String> = folo_core::backend::available_devices().iter().map(|d| d.to_string()).collect();
    let why = if have.len() > 1 { "this machine has no such GPU" } else { "no usable GPU adapter was found" };
    Error::DeviceNotFound(format!(
        "device {device} is not available: {why}; FOLO can use {}",
        have.join(", ")
    ))
}

/// Whether `e` is folo-gpu reporting a lost device, from the trainer's own tensor work, a layer, or the batch
/// feeder.
pub(super) fn device_lost(e: &Error) -> bool {
    match e {
        Error::Core(c) | Error::Nn(folo_nn::Error::Core(c)) => folo_gpu::is_device_lost(c),
        _ => false,
    }
}

/// The failure message of a run whose GPU was lost: the cause, and the checkpoint to resume from (the last one
/// written or resumed from, with the epochs it holds) or that there is none.
pub(super) fn device_loss_message(e: &Error, last: Option<(&Path, u32)>) -> String {
    let resume = match last {
        Some((p, n)) => format!(
            "{} holds the state after epoch {n}: {}",
            folo_api::json::path(p),
            folo_api::json::cli_command(
                "resume with `folo train --resume {checkpoint}`",
                &[("checkpoint", p)]
            )
        ),
        None => "no checkpoint was written yet: start the run again".into(),
    };
    format!("the GPU device was lost ({e}); {resume}")
}
