//! The real store's runs, models and inference end to end on a tiny synthetic dataset (CPU, 64 px): create the
//! dataset, train over HTTP while reading the SSE stream, find the run's model, evaluate, export and predict with
//! it, stop / resume / delete runs, and reload everything after a store restart.

mod hardware;
mod harness;
mod published;
mod queue;
mod training;

// For `crate::app` in hardware.rs, where a local `app` hides the function.
use harness::app;
