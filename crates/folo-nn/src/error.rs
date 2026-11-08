//! Error type for folo-nn.

/// Result alias used throughout folo-nn.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong in a layer, an optimizer or a `.folo` file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A tensor op failed.
    #[error(transparent)]
    Core(#[from] folo_core::Error),

    /// Reading or writing a file failed.
    #[error("{path}: {source}")]
    Io {
        /// The file involved.
        path: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// The bytes are not a valid safetensors container (docs/FORMAT.md, reader acceptance rules).
    #[error("invalid safetensors file: {0}")]
    Format(String),

    /// The container is valid but its FOLO metadata is missing, malformed or unsupported.
    #[error("invalid .folo metadata: {0}")]
    Metadata(String),

    /// The file does not fit the model, optimizer or schedule it is being loaded into. The message lists every
    /// offending name.
    #[error("cannot load weights: {0}")]
    Load(String),

    /// A layer, optimizer or schedule was configured or used incorrectly.
    #[error("{0}")]
    Invalid(String),
}

impl Error {
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Error::Io { path: path.display().to_string(), source }
    }
}
