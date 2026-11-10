//! Vision benchmarks end to end over HTTP (docs/API.md, Vision benchmarks; docs/BENCH.md), on the CPU with tiny
//! synthetic datasets and untrained (or one-epoch) folo-n models, and scripted agents in place of the Claude Code
//! and Codex CLIs: benchmarks from a dataset split and from a labeling job, their images and variants, external,
//! model and agent entries, designing a plan, events, restarts, and a self-improvement loop.

mod agents;
mod benchmarks;
mod external;
mod harness;
mod loops;
mod model_entries;
mod restarts;
