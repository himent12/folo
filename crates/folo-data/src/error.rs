//! Error type for the data pipeline.

use std::path::PathBuf;

/// Result alias used throughout folo-data.
pub type Result<T, E = DataError> = std::result::Result<T, E>;

/// Everything that can go wrong while opening, reading or generating data.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// A filesystem operation failed.
    #[error("io error at {}: {source}", crate::paths::shown(path))]
    Io {
        /// The file or directory involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// An image could not be decoded or encoded (`image error: …` for in-memory data, which has no path).
    #[error("{}", image_message(path, message))]
    Image {
        /// The image file (empty for in-memory data).
        path: PathBuf,
        /// What the codec reported.
        message: String,
    },
    /// A label file is malformed.
    #[error("bad label {}:{line}: {message}", crate::paths::shown(path))]
    Label {
        /// The label file.
        path: PathBuf,
        /// 1-based line number (0 when the whole file is affected).
        line: usize,
        /// What is wrong with it.
        message: String,
    },
    /// A dataset description (data.yaml, COCO JSON, directory layout) is invalid or not recognised.
    #[error("dataset format error: {0}")]
    Format(String),
    /// A dataset folder whose writer has not finished it ([`crate::unfinished`]).
    #[error(
        "{} is an unfinished dataset: `folo dataset synth` or `convert` is writing it or was stopped while writing \
         it (it has a {} file)",
        crate::paths::shown(.0),
        crate::unfinished::MARKER
    )]
    Unfinished(PathBuf),
    /// An invalid configuration value.
    #[error("invalid config: {0}")]
    Config(String),
    /// An index past the end of a dataset.
    #[error("index {index} out of range for dataset of length {len}")]
    OutOfRange {
        /// The requested index.
        index: usize,
        /// The dataset length.
        len: usize,
    },
}

fn image_message(path: &std::path::Path, message: &str) -> String {
    if path.as_os_str().is_empty() {
        format!("image error: {message}")
    } else {
        format!("image error at {}: {message}", crate::paths::shown(path))
    }
}

impl DataError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        DataError::Io { path: path.into(), source }
    }

    pub(crate) fn image(path: impl Into<PathBuf>, message: impl ToString) -> Self {
        DataError::Image { path: path.into(), message: message.to_string() }
    }
}
