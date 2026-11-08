//! Glue between the CLI and the engine: opening datasets and mapping engine errors to CLI errors.

use std::path::Path;

use crate::output::{CliError, ExitCode};

/// Open a dataset folder or YAML/JSON file. Returns the dataset and a short name for it (the folder name, or the
/// YAML's folder for `data.yaml`).
pub fn open_dataset(arg: &str) -> Result<(folo_data::LoadedDataset, String), CliError> {
    let path = Path::new(arg);
    if !path.exists() {
        return Err(CliError::new(ExitCode::NotFound, "dataset_not_found", format!("{arg} does not exist"))
            .hint("pass a YOLO / COCO dataset folder or its data.yaml"));
    }
    let ds = folo_data::open(path).map_err(|e| {
        CliError::new(ExitCode::Runtime, "invalid_dataset", e.to_string())
            .hint("see docs/DATA.md for the supported layouts")
    })?;
    let named = if path.is_dir() { path } else { path.parent().unwrap_or(path) };
    let abs = std::path::absolute(named).unwrap_or_else(|_| named.to_path_buf());
    let name = abs.file_name().map_or_else(|| "dataset".to_string(), |n| n.to_string_lossy().into_owned());
    Ok((ds, name))
}

/// Map an engine error to a CLI error with the matching exit code.
pub fn engine_error(e: folo_vision::Error) -> CliError {
    use folo_vision::Error as E;
    let exit = match &e {
        E::Config(_) => ExitCode::Usage,
        E::Unsupported(_) => ExitCode::NotImplemented,
        _ => ExitCode::Runtime,
    };
    let hint = match &e {
        E::Config(_) => Some("see `folo describe --json` for the accepted ranges"),
        E::Failed(_) => Some("the last completed epoch's checkpoint is in <run>/weights/last.folo"),
        E::Unsupported(_) => Some("use --device cpu"),
        _ => None,
    };
    let err = CliError::new(exit, e.code(), e.to_string());
    match hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

/// Fail with `not_found` unless `path` is an existing file.
pub fn require_file(path: &Path, code: &str) -> Result<(), CliError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(CliError::new(ExitCode::NotFound, code, format!("{} does not exist", path.display())))
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
