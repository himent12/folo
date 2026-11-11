//! Vision benchmarks (docs/API.md, *Vision benchmarks*; docs/BENCH.md defines every number): a frozen set of
//! labeled images plus a plan of stress conditions, the entries that ran it (FOLO models, coding agents, or
//! predictions made anywhere else) with their scores, and self-improvement loops. Everything under `/api/bench`.
//!
//! Not the speed probe of `/api/system/benchmark` ([`crate::BenchmarkRequest`]): that one times a model, this
//! one grades what a model sees.

mod benchmarks;
mod catalog;
mod entries;
mod events;
mod loops;
mod results;

pub use benchmarks::*;
pub use catalog::*;
pub use entries::*;
pub use events::*;
pub use loops::*;
pub use results::*;

/// [`SystemResponse::features`](crate::SystemResponse::features): the `/api/bench` endpoints exist.
pub const FEATURE_VISION_BENCH: &str = "vision_bench";

/// Severities of a condition run from 1 (barely visible) to this (severe, still solvable by a person).
pub const BENCH_SEVERITIES: u8 = 5;

#[cfg(test)]
mod tests;
