//! Playground: open or drop an image or folder batch, run a model, compare A/B models
//! side-by-side, inspect boxes, tune thresholds, and export annotated PNGs, JSON, or Batch CSV/JSON.

mod batch;
mod batch_export;
mod compare;
mod device;
mod export;
mod export_text;
mod page;
mod samples;
mod side;
mod side_compare;
mod stage;
mod stage_compare;

#[cfg(test)]
pub(crate) use batch::batch_action_labels;
pub use batch::{
    BatchItemResult, BatchItemStatus, MAX_BATCH_FILES, PlayBatchItem, PlayBatchSource, PlayBatchState,
    bytes_fingerprint, disambiguate_file_labels, is_supported_image_ext, scan_image_folder,
};
pub use compare::{
    ComparePaneCaption, CompareSideExport, PlayCompareState, PlayPredMeta, compare_detections_json,
    compare_pane_caption_parts, pick_default_model_b, summarize_comparison,
};
pub use export::{
    ExportPalette, export_filename, render_annotated_compare_png, render_annotated_single_png,
    render_image_to_rgba, write_export_bytes,
};
pub(crate) use page::{detections_json, filter_detections};
pub(crate) use samples::pick_val_samples;
