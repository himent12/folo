//! Benchmarks (servers with `vision_bench`, docs/BENCH.md): the benchmark list with a New benchmark
//! form, a benchmark's leaderboard, plan and loops, and an entry's result on the five axes.

mod cells;
mod compare;
mod design;
mod entry;
mod entry_image_detail;
mod entry_images;
mod error_analysis;
mod external;
mod format;
mod leaderboard;
mod list;
mod loops;
mod new_bench;
mod new_loop;
mod page;
mod plan;

pub use error_analysis::{EntryImageFilter, EntryImageOrder, entry_image_matches_filter};
