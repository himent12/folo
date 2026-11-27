//! Runs: the run list, the live training dashboard, and multi-run comparison.

mod charts;
mod cloud;
mod compare;
pub(crate) mod compare_data;
mod compare_table;
mod config;
mod detail;
pub(crate) mod export;
mod list;
mod logs;
mod per_class;
pub(crate) mod sample_errors;
mod samples;
pub(crate) mod spot;

pub(crate) use charts::RunChartOpts;
#[cfg(test)]
pub(crate) use compare::compare_legend_scope;
pub(crate) use compare_data::RunCompareState;
#[cfg(test)]
pub(crate) use compare_table::compare_best_badge_label;
pub(crate) use export::CheckpointCliMode;
#[cfg(test)]
pub(crate) use list::filter_runs;
pub(crate) use logs::LogFilterCache;
pub(crate) use sample_errors::{SampleErrorFilter, SampleSort};
pub(crate) use samples::SAMPLE_CONF;
#[cfg(test)]
pub(crate) use spot::{SpotRecoveryState, spot_recovery_status};

use crate::api::{Run, time};

/// How long a run trained (or has trained, while it does): from `started` to `finished` (or now),
/// less the time it sat stopped or failed before a resume (`paused_s`), as Studio counts it.
pub(crate) fn duration_of(r: &Run, now: i64) -> Option<f64> {
    let start = time::parse(r.started.as_deref()?)?;
    let end = r.finished.as_deref().and_then(time::parse).unwrap_or(now);
    Some(((end - start) as f64 - r.paused_s.unwrap_or(0.)).max(0.))
}

#[cfg(test)]
mod tests {
    use super::duration_of;
    use crate::api::{Backend, time};

    #[test]
    fn a_resumed_runs_duration_leaves_out_the_time_it_sat_stopped() {
        let base = crate::api::mock::MockBackend::new(4200, true).runs().unwrap().remove(0);
        let r = crate::api::Run {
            started: Some("2026-11-10T18:00:00Z".into()),
            finished: Some("2026-11-11T08:30:00Z".into()),
            paused_s: Some(14. * 3600.),
            ..base
        };
        assert_eq!(duration_of(&r, 0), Some(1800.), "14 h 30 min less 14 h stopped");
        let live = crate::api::Run { finished: None, paused_s: None, ..r };
        let now = time::parse("2026-11-10T18:00:50Z").unwrap();
        assert_eq!(duration_of(&live, now), Some(50.));
    }
}
