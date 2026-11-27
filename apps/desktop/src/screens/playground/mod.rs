//! Playground: open or drop an image or folder batch, run a model, compare A/B models
//! side-by-side, inspect boxes, tune thresholds, and export annotated PNGs, JSON, or Batch CSV/JSON.

mod batch;
mod batch_export;
mod batch_qa;
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
mod thresholds;
mod thresholds_ui;

#[cfg(test)]
pub(crate) use batch::batch_action_labels;
pub use batch::{
    BatchItemResult, BatchItemStatus, MAX_BATCH_FILES, PlayBatchItem, PlayBatchSource, PlayBatchState,
    bytes_fingerprint, disambiguate_file_labels, is_supported_image_ext, scan_image_folder,
};
pub use batch_qa::BatchQaFilter;
pub use compare::{
    CompareClassFilter, ComparePaneCaption, CompareSideExport, PlayCompareState, PlayPredMeta,
    compare_detections_json, compare_pane_caption_parts, pick_default_model_b, summarize_comparison,
};
pub use export::{
    ExportPalette, export_filename, render_annotated_compare_png, render_annotated_single_png,
    render_image_to_rgba, write_export_bytes,
};
pub(crate) use page::detections_json;
#[cfg(test)]
pub(crate) use page::filter_detections;
pub(crate) use samples::pick_val_samples;
pub use thresholds::{
    MAX_THRESHOLDS_JSON_BYTES, PlayThresholdsState, calibration_to_overrides, effective_conf,
    export_calibration_json, export_thresholds_json, filter_detections_with_per_class, fmt_threshold,
    map_per_class_to_classes, parse_thresholds_json, round_conf, thresholds_match,
};
