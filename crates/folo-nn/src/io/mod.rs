//! `.folo` files (docs/FORMAT.md): the safetensors container, metadata, namespaces, strict and partial
//! loading, checkpoints and export.
//!
//! - [`safetensors`]: the container, read with docs/FORMAT.md's acceptance rules and written in FOLO's layout.
//! - `meta.rs`: `__metadata__` ([`Metadata`]), the v1 [`ModelConfig`] and the scale table ([`scale_row`]).
//! - `file.rs`: [`ModelFile`], an opened file, its namespaces and the strict / partial loading rules
//!   ([`LoadOptions`], [`LoadReport`]).
//! - `checkpoint.rs`: the full training state ([`save_checkpoint`], [`checkpoint_contents`]).
//! - `export.rs`: weights files and export ([`save_weights`], [`write_export`]): Conv+BN fusion, optional `F16`.
//! - `replace.rs`: [`rename_over`], the temporary-file-then-rename step of every atomic write, retried while
//!   another program holds the target open on Windows.

mod checkpoint;
mod export;
mod file;
mod meta;
mod replace;
pub mod safetensors;

pub use checkpoint::{
    BestState, CheckpointMeta, CheckpointParts, FileContents, Resumed, RngState, checkpoint_contents,
    save_checkpoint,
};
pub use export::{export_module, export_tensors, fuse_bn_tensors, save_weights, write_export, write_weights};
pub use file::{
    LoadOptions, LoadReport, ModelFile, Namespace, Purpose, SkipReason, Skipped, Source, is_class_output,
};
pub use meta::{FORMAT_VERSION, FileKind, FloatDType, Metadata, Metrics, ModelConfig, scale_row};
pub use replace::{RenameRetry, rename_over, rename_over_observed, rename_over_with};
