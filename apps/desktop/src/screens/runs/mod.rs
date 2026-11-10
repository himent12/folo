//! Runs: the run list and the live training dashboard.

mod charts;
mod cloud;
mod config;
mod detail;
mod list;
mod logs;
mod samples;

use crate::api::{Run, time};

pub(crate) fn duration_of(r: &Run, now: i64) -> Option<f64> {
    let start = time::parse(r.started.as_deref()?)?;
    let end = r.finished.as_deref().and_then(time::parse).unwrap_or(now);
    Some((end - start) as f64)
}
