//! Which model is "the best": the one rule every default pick and ranking uses (Playground,
//! a benchmark's Run, Overview's Best models), the same as Studio's (`rankByContext`).

use std::cmp::Ordering;

use super::Store;
use crate::api::Model;

/// `a` before `b` when it is the better model: the higher mAP50-95 (an unevaluated model last); on
/// a tie a `-best` checkpoint before a `-last` one, then the earlier made (Studio's tie-break).
pub fn better_model(a: &Model, b: &Model) -> Ordering {
    let score = |m: &Model| m.metrics.as_ref().map_or(-1., |x| x.map50_95);
    let checkpoint = |m: &Model| {
        if m.name.ends_with("-best") {
            0
        } else if m.name.ends_with("-last") {
            2
        } else {
            1
        }
    };
    score(b)
        .total_cmp(&score(a))
        .then_with(|| checkpoint(a).cmp(&checkpoint(b)))
        .then_with(|| a.created.cmp(&b.created))
}

impl Store {
    /// The dataset a model was trained on, from the run that made it (unknown for imported weights,
    /// or while the runs aren't in).
    pub fn model_dataset(&self, m: &Model) -> Option<&str> {
        m.run_id.as_ref().and_then(|r| self.run(r)).map(|r| r.config.dataset_id.as_str())
    }

    /// The best model to offer first: among those trained on `dataset` when any is, else among all
    /// of them.
    pub fn best_model(&self, dataset: Option<&str>) -> Option<&Model> {
        let models = self.models.ready()?;
        let on = |m: &&Model| dataset.is_some_and(|d| self.model_dataset(m) == Some(d));
        models
            .iter()
            .filter(on)
            .min_by(|a, b| better_model(a, b))
            .or_else(|| models.iter().min_by(|a, b| better_model(a, b)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::api::Backend;
    use crate::api::mock::MockBackend;

    #[test]
    fn the_best_model_is_the_highest_map_then_best_before_last() {
        let api = MockBackend::new(4200, true);
        let mut models = api.models().unwrap();
        let base = models[0].clone();
        let model = |id: &str, name: &str, created: &str, map: Option<f32>| Model {
            id: id.into(),
            name: name.into(),
            created: created.into(),
            run_id: None,
            metrics: map.map(|v| crate::api::ModelMetrics {
                map50: v,
                map50_95: v,
                precision: 0.,
                recall: 0.,
                best_f1_conf: None,
            }),
            ..base.clone()
        };
        // The newest checkpoint learned nothing; a tie at 0 takes the "-best" one, even when its
        // "-last" twin is older.
        let mut list = [
            model("m_new", "bccd-n-last", "2026-11-10T02:00:00Z", Some(0.)),
            model("m_tie", "bccd-n-best", "2026-11-10T02:00:01Z", Some(0.)),
            model("m_raw", "imported", "2026-11-10T03:00:00Z", None),
            model("m_good", "shapes-n-best", "2026-11-09T01:00:00Z", Some(0.254)),
        ];
        list.sort_by(better_model);
        let ids: Vec<&str> = list.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["m_good", "m_tie", "m_new", "m_raw"]);

        // A benchmark on a dataset prefers what was trained on it, even when weaker elsewhere.
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let runs = api.runs().unwrap();
        let (on_ds, other) =
            (&runs[0], runs.iter().find(|r| r.config.dataset_id != runs[0].config.dataset_id));
        let other = other.expect("the demo has runs on two datasets");
        models = vec![
            Model { run_id: Some(other.id.clone()), ..list[0].clone() },
            Model { run_id: Some(on_ds.id.clone()), ..list[1].clone() },
        ];
        store.runs = crate::store::Load::Ready(runs.clone());
        store.models = crate::store::Load::Ready(models);
        assert_eq!(store.best_model(Some(&on_ds.config.dataset_id)).map(|m| m.id.as_str()), Some("m_tie"));
        assert_eq!(store.best_model(None).map(|m| m.id.as_str()), Some("m_good"));
        // Nothing trained on it: the best overall.
        assert_eq!(store.best_model(Some("ds_none")).map(|m| m.id.as_str()), Some("m_good"));
    }
}
