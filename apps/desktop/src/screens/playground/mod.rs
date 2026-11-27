//! Playground: open or drop an image, run a model, compare A/B models side-by-side, inspect
//! boxes, tune thresholds, and export annotated PNGs or structured JSON.

mod compare;
mod device;
mod export;
mod page;
mod samples;
mod side;
mod side_compare;
mod stage;
mod stage_compare;

pub use compare::{
    CompareSideExport, PlayCompareState, PlayPredMeta, compare_detections_json, compare_pane_caption,
    pick_default_model_b, summarize_comparison,
};
pub use export::{
    ExportPalette, export_filename, render_annotated_compare_png, render_annotated_single_png,
    render_image_to_rgba, write_export_bytes,
};
pub(crate) use page::{detections_json, filter_detections};
