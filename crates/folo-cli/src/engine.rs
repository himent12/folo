//! Glue between the CLI and the engine: opening datasets, registering the GPU backend for `--device`, and mapping
//! engine errors to CLI errors.

use std::path::Path;

use crate::output::{CliError, ExitCode};

/// Open a dataset folder or YAML/JSON file. Returns the dataset and a short name for it (the folder name, or the
/// YAML's folder for `data.yaml`).
pub fn open_dataset(
    arg: &str,
    format: folo_data::Format,
) -> Result<(folo_data::LoadedDataset, String), CliError> {
    open_dataset_splits(arg, format, &folo_data::Split::ALL)
}

/// [`open_dataset`] building only `splits` (folo_data::open_splits); the others come back empty.
pub fn open_dataset_splits(
    arg: &str,
    format: folo_data::Format,
    splits: &[folo_data::Split],
) -> Result<(folo_data::LoadedDataset, String), CliError> {
    let path = Path::new(arg);
    if !path.exists() {
        return Err(CliError::new(ExitCode::NotFound, "dataset_not_found", format!("{arg} does not exist"))
            .hint("pass a YOLO / COCO dataset folder or its data.yaml"));
    }
    let opts = folo_data::OpenOptions { format, ..folo_data::OpenOptions::default() };
    let ds = folo_data::open_splits(path, &opts, splits).map_err(|e| match e {
        e @ folo_data::DataError::Unfinished(_) => {
            CliError::new(ExitCode::Runtime, "dataset_unfinished", e.to_string())
                .hint(folo_data::unfinished::HINT)
        }
        e => CliError::new(ExitCode::Runtime, "invalid_dataset", e.to_string())
            .hint("see docs/DATA.md for the supported layouts"),
    })?;
    let named = if path.is_dir() { path } else { path.parent().unwrap_or(path) };
    let abs = std::path::absolute(named).unwrap_or_else(|_| named.to_path_buf());
    let name = abs.file_name().map_or_else(|| "dataset".to_string(), |n| n.to_string_lossy().into_owned());
    Ok((ds, name))
}

/// Map an engine error of `folo <command>` to a CLI error: the code of `folo_api::codes` with its exit code, the
/// message capped ([`folo_vision::Error::summary`]) and a hint for the failures an agent can fix.
pub fn engine_error(command: &str, e: folo_vision::Error) -> CliError {
    let code = e.code();
    let hint = match code {
        "invalid_config" => Some(format!("see `folo describe {command} --json` for the accepted ranges")),
        "training_failed" => {
            Some("the last completed epoch's checkpoint is in <run>/weights/last.folo".into())
        }
        "gpu_out_of_memory" => {
            Some("lower --batch or --imgsz, or use a smaller --model (folo-n, folo-s)".into())
        }
        "device_not_found" => Some(format!(
            "this machine has {}; use --device auto (the GPU when there is one) or --device cpu",
            folo_server::system::available_devices()
        )),
        _ => None,
    };
    let err = CliError::new(ExitCode::of(code), code, e.summary());
    match hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

/// Register the GPU backend when `--device` may use a GPU (`auto`, `gpu`, `gpu:<n>`), so the engine can resolve
/// it. Without an adapter, `auto` runs on the CPU quietly and an explicit GPU fails in the engine.
pub fn init_device(device: &str) {
    let d = device.trim().to_ascii_lowercase();
    let explicit = d.starts_with("gpu") || d.starts_with("cuda");
    if (explicit || d == "auto" || d.is_empty())
        && let Err(e) = folo_gpu::init()
        && explicit
    {
        eprintln!("warning: no usable GPU: {e}");
    }
}
