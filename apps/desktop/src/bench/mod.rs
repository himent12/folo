//! Vision benchmarks (servers with `vision_bench`, docs/BENCH.md): what the Benchmarks screens
//! hold, how they load, the open benchmark's live stream and the actions they take.

mod create;
mod delete;
mod design;
mod images;
mod keys;
mod load;
mod loops;
mod merge;
mod run;
mod state;
mod stream;

pub use create::{SourceChoice, SourceId, SourcePick};
pub use images::preview_severity;
pub use keys::BenchForm;
pub use run::model_fits;
pub use state::{
    BENCH_SPLITS, BenchState, BenchTab, Contender, ENTRY_IMAGES_PAGE, EntryImages, ImageSort, LOOP_EPOCHS,
    LOOP_GENERATIONS,
};
