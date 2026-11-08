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
    /// The request is valid but this build cannot do it yet (e.g. a device without a backend).
    #[error("{0}")]
    Unsupported(String),
    /// Training diverged or produced no usable batch (docs/MODEL.md §8.3).
    #[error("training failed: {0}")]
    Failed(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }

    /// A stable snake_case code (the vocabulary of docs/API.md and docs/CLI.md where they overlap).
    pub fn code(&self) -> &'static str {
        match self {
            Error::Config(_) => "invalid_config",
            Error::Dataset(_) => "invalid_dataset",
            Error::Data(_) => "data_error",
            Error::Nn(folo_nn::Error::Io { .. }) | Error::Io { .. } => "io_error",
            Error::Nn(folo_nn::Error::Load(_) | folo_nn::Error::Format(_) | folo_nn::Error::Metadata(_)) => {
                "invalid_weights"
            }
            Error::Nn(_) | Error::Core(_) => "internal",
            Error::Unsupported(_) => "not_supported",
            Error::Failed(_) => "training_failed",
        }
    }
}
