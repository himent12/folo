//! The error type of the trainer, the validator and the predictor (the layers that combine data, model and IO).

use std::path::PathBuf;

/// Result alias for [`crate::train`], [`crate::val`] and [`crate::predict`].
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong while training, validating or predicting.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A `TrainConfig` field (or a predict / val setting) is out of range; the message names it.
    #[error("invalid config: {0}")]
    Config(String),
    /// The dataset cannot be used (no classes, no images, a class list that does not match the model, ...).
    #[error("invalid dataset: {0}")]
    Dataset(String),
    /// Reading the data failed.
    #[error(transparent)]
    Data(#[from] folo_data::DataError),
    /// A layer, optimizer, schedule or `.folo` file failed.
    #[error(transparent)]
    Nn(#[from] folo_nn::Error),
    /// A tensor op failed.
    #[error(transparent)]
    Core(#[from] folo_core::Error),
    /// A file or directory could not be written or read.
    #[error("{path}: {source}")]
    Io {
        /// The file involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The request is valid but this build cannot do it yet.
    #[error("{0}")]
    Unsupported(String),
    /// The request is valid but this machine cannot run it (mixed-precision training without GPU tensor cores).
    #[error("{0}")]
    NotSupported(String),
    /// A device (`gpu:<n>`) this machine has no usable adapter for; the message names the devices it has.
    #[error("{0}")]
    DeviceNotFound(String),
    /// Training diverged or produced no usable batch (docs/MODEL.md §8.3).
    #[error("training failed: {0}")]
    Failed(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }

    /// True when the GPU ran out of memory (wgpu reports "Out of Memory"; buffers it could not create then fail
    /// validation, so the whole report is searched).
    pub fn is_gpu_out_of_memory(&self) -> bool {
        matches!(self, Error::Core(_) | Error::Nn(_)) && {
            let m = self.to_string().to_ascii_lowercase();
            m.contains("out of memory") || m.contains("outofmemory")
        }
    }

    /// The message an agent sees: at most [`folo_api::codes::MAX_MESSAGE_CHARS`] characters (driver reports are
    /// cut), and for a GPU out-of-memory failure one sentence plus the start of the driver's report.
    pub fn summary(&self) -> String {
        let max = folo_api::codes::MAX_MESSAGE_CHARS;
        if self.is_gpu_out_of_memory() {
            let report = self.to_string();
            let first = report.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
            return folo_api::codes::cap(&format!("the GPU ran out of memory ({first})"), max);
        }
        folo_api::codes::cap(&self.to_string(), max)
    }

    /// A stable snake_case code (`folo_api::codes`, shared by docs/API.md, docs/CLI.md and docs/MCP.md).
    pub fn code(&self) -> &'static str {
        if self.is_gpu_out_of_memory() {
            return "gpu_out_of_memory";
        }
        match self {
            Error::Config(_) => "invalid_config",
            Error::Dataset(_) => "invalid_dataset",
            Error::Data(_) => "data_error",
            Error::Nn(folo_nn::Error::Io { .. }) | Error::Io { .. } => "io_error",
            Error::Nn(folo_nn::Error::Load(_) | folo_nn::Error::Format(_) | folo_nn::Error::Metadata(_)) => {
                "invalid_weights"
            }
            Error::Nn(_) | Error::Core(_) => "internal",
            Error::Unsupported(_) => "not_implemented",
            Error::NotSupported(_) => "not_supported",
            Error::DeviceNotFound(_) => "device_not_found",
            Error::Failed(_) => "training_failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_out_of_memory_is_its_own_short_error() {
        let report = format!(
            "Out of Memory
Validation Error

Caused by:
{}",
            "  In a buffer
"
            .repeat(3000)
        );
        let e = Error::Core(folo_core::Error::Backend { backend: "wgpu", msg: report });
        assert!(e.is_gpu_out_of_memory());
        assert_eq!(e.code(), "gpu_out_of_memory");
        let s = e.summary();
        assert!(s.starts_with("the GPU ran out of memory (wgpu backend: Out of Memory)"), "{s}");
        assert!(s.chars().count() <= folo_api::codes::MAX_MESSAGE_CHARS + 1);
        // Other errors keep their code; long ones are capped too.
        let long = Error::Failed("x".repeat(40_000));
        assert_eq!(long.code(), "training_failed");
        assert_eq!(long.summary().chars().count(), folo_api::codes::MAX_MESSAGE_CHARS + 1);
        assert!(!Error::Config("out of memory? no".into()).is_gpu_out_of_memory());
    }
}
