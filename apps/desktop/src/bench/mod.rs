//! Vision benchmarks (servers with `vision_bench`, docs/BENCH.md): what the Benchmarks screens
//! hold, how they load, the open benchmark's live stream and the actions they take.

mod compare;
mod create;
mod delete;
mod design;
mod external;
mod images;
mod keys;
mod load;
mod loops;
mod merge;
mod run;
mod state;
mod stream;

#[cfg(test)]
pub use compare::reconcile_compare_entries;
pub use compare::{
    COMPARE_CLS_SHOWN, COMPARE_MAX, COMPARE_SLOTS, best_in_row, compare_caveats, compare_class_rows,
    compare_condition_rows, default_compare_entries, intervals_domain, intervals_verdict, is_comparable,
};
pub use create::{SourceChoice, SourceId, SourcePick, is_generated_source, validate_generated_bench};
#[cfg(test)]
pub use create::{parse_bench_gen_classes, parse_bench_gen_images, parse_bench_gen_seed};
pub use external::validate_box_scores;
#[cfg(test)]
pub use external::{
    check_predictions, parse_predictions_text, parts_of, suggest_external_name, upload_external_predictions,
};
pub use images::preview_severity;
pub use keys::BenchForm;
pub use loops::{loop_pool_likely_empty, loop_pool_overlaps};
pub use run::model_fits;
pub use state::{
    BENCH_GEN_IMGSZ, BENCH_SPLITS, BenchListFilter, BenchSourceKind, BenchState, BenchTab, Contender,
    ENTRY_IMAGES_PAGE, EntryImageReveal, EntryImages, ImageSort, LOOP_EPOCHS, LOOP_GENERATIONS,
};
