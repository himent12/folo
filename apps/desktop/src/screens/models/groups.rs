//! The Models list's groups: a model's score compares only with scores on the same dataset and
//! split, so the list is grouped by what each model was evaluated on and ranked inside each group
//! (Studio's `scoreSummary` and `bestMark`, the same rules).
//!
//! A trained model is scored on its run's dataset: its validation split, or its training split when
//! the dataset has no validation images. Published weights and imported files carry no dataset of
//! this server's: a group each, after the datasets, never ranked. A trained model whose run or
//! dataset is not known here (deleted, or not loaded) has an unknown split: its own group, not ranked.

use std::cmp::Ordering;

use crate::api::{Model, ModelEval, ModelSource, RunStatus, Split, time};
use crate::store::Store;

/// What a group's scores are on, and so whether they rank.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalSet {
    /// One key per group: `{dataset}:{split}` for a ranked one, `~published`, `~imported` or
    /// `~unknown:{dataset}` otherwise.
    pub key: String,
    /// Its heading: "Validation split of fasteners", "Imported · split unknown".
    pub label: String,
    pub ranked: bool,
    /// Ranked groups first (by dataset name, validation before training), then published, imported,
    /// unknown.
    order: (u8, String, u8),
}

/// A group and its models, sorted.
pub struct Group {
    pub set: EvalSet,
    pub rows: Vec<Model>,
}

/// What `m`'s score is on.
pub fn eval_set(store: &Store, m: &Model) -> EvalSet {
    let unranked = |key: &str, label: String, rank: u8| EvalSet {
        key: key.to_string(),
        label,
        ranked: false,
        order: (rank, String::new(), 0),
    };
    if m.source == Some(ModelSource::Published) {
        return unranked("~published", "Published weights · split unknown".into(), 1);
    }
    let Some(run_id) = &m.run_id else {
        return unranked("~imported", "Imported · split unknown".into(), 2);
    };
    let run = store.run(run_id);
    let ds_id = run.map(|r| r.config.dataset_id.clone());
    match ds_id.as_deref().and_then(|d| store.dataset_summary(d)) {
        Some(ds) => {
            let (split, word, rank) =
                if ds.splits.val == 0 { ("train", "Training", 1) } else { ("val", "Validation", 0) };
            EvalSet {
                key: format!("{}:{split}", ds.id),
                label: format!("{word} split of {}", ds.name),
                ranked: true,
                order: (0, ds.name.to_lowercase(), rank),
            }
        }
        // A removed dataset by the name its run recorded (never its id).
        None => {
            let name = run.map_or("Unknown dataset".to_string(), |r| store.run_dataset_name(r));
            let key = format!("~unknown:{}", ds_id.unwrap_or_default());
            let mut set = unranked(&key, format!("{name} · split unknown"), 3);
            set.order.1 = name.to_lowercase();
            set
        }
    }
}

/// What `m`'s evaluation report actually scored on when a loaded [`ModelEval`] carries concrete
/// `dataset_id` / `split` provenance for a run-produced model, falling back to [`eval_set`]
/// otherwise. Published and imported files (`run_id: None`) never inherit `/eval`'s local dataset
/// for their stored file header metrics (`m.metrics`), matching Model Detail's rule.
pub fn eval_set_with_report(store: &Store, m: &Model, eval: Option<&ModelEval>) -> EvalSet {
    if m.source == Some(ModelSource::Published) || m.run_id.is_none() {
        return eval_set(store, m);
    }
    let Some(ev) = eval else {
        return eval_set(store, m);
    };
    let Some(split) = ev.split else {
        return eval_set(store, m);
    };
    let (word, split_rank) = match split {
        Split::Val => ("Validation", 0),
        Split::Train => ("Training", 1),
        Split::Test => ("Test", 2),
    };
    let ds_name = ev
        .dataset_name
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .or_else(|| ev.dataset_id.as_deref().and_then(|id| store.dataset_summary(id)).map(|d| d.name.clone()))
        .or_else(|| m.run_id.as_deref().and_then(|rid| store.run(rid)).map(|r| store.run_dataset_name(r)))
        .or_else(|| ev.dataset_id.clone())
        .unwrap_or_else(|| "Unknown dataset".into());
    if let Some(ds_id) = ev.dataset_id.as_deref().filter(|s| !s.trim().is_empty()) {
        EvalSet {
            key: format!("{ds_id}:{}", split.as_str()),
            label: format!("{word} split of {ds_name}"),
            ranked: true,
            order: (0, ds_name.to_lowercase(), split_rank),
        }
    } else {
        EvalSet {
            key: format!("~eval:{}:{}", ds_name.to_lowercase(), split.as_str()),
            label: format!("{word} split of {ds_name}"),
            ranked: false,
            order: (3, ds_name.to_lowercase(), split_rank),
        }
    }
}

/// A run's last checkpoint that is also its best epoch: the same weights as its `-best`, so not a
/// second contender (Studio's "same as best").
pub fn same_as_best(store: &Store, m: &Model) -> bool {
    m.name.ends_with("-last")
        && m.run_id.as_ref().and_then(|r| store.run(r)).is_some_and(|r| {
            r.status == RunStatus::Completed && r.best.as_ref().is_some_and(|b| b.epoch == r.epochs)
        })
}

/// The value a column sorts by; missing or non-finite values return `None` so they always sort last.
fn sort_value(m: &Model, key: &str) -> Option<f64> {
    match key {
        "map" => m.metrics.as_ref().map(|x| x.map50_95 as f64).filter(|v| v.is_finite()),
        "map50" => m.metrics.as_ref().map(|x| x.map50 as f64).filter(|v| v.is_finite()),
        "size" => Some(m.bytes as f64),
        "classes" => Some(m.classes.len() as f64),
        _ => time::parse(&m.created).map(|t| t as f64),
    }
}

/// Compares optional numeric sort keys so `None` (and non-finite values) always sort last,
/// regardless of ascending or descending direction.
fn cmp_opt_f64(a: Option<f64>, b: Option<f64>, desc: bool) -> Ordering {
    let a = a.filter(|v| v.is_finite());
    let b = b.filter(|v| v.is_finite());
    match (a, b) {
        (Some(va), Some(vb)) => {
            let o = va.total_cmp(&vb);
            if desc { o.reverse() } else { o }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// `models` in their groups, in group order, each group's rows sorted by `sort` (ties by name; the
/// server's order without a sort). Missing/unscored rows always sort last in both directions.
pub fn group_models(store: &Store, models: &[Model], sort: Option<(&str, bool)>) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for m in models {
        let set = eval_set(store, m);
        match groups.iter_mut().find(|g| g.set.key == set.key) {
            Some(g) => g.rows.push(m.clone()),
            None => groups.push(Group { set, rows: vec![m.clone()] }),
        }
    }
    groups.sort_by(|a, b| a.set.order.cmp(&b.set.order));
    if let Some((key, desc)) = sort {
        for g in &mut groups {
            g.rows.sort_by(|a, b| {
                cmp_opt_f64(sort_value(a, key), sort_value(b, key), desc).then_with(|| a.name.cmp(&b.name))
            });
        }
    }
    groups
}

/// Which values of a comparable group are its best (higher is better).
#[derive(Clone, Debug, PartialEq)]
pub struct BestMark {
    /// Indexes of the best values: one, or every one of an exact tie. Empty with fewer than two
    /// values to compare, or a top of 0.
    pub best: Vec<usize>,
    /// Whether the best is shared (an exact tie): "joint best".
    pub joint: bool,
    /// Decimals that tell the best apart from every other value (at least `decimals`, at most
    /// `max`), so "best" never sits beside a value that reads the same.
    pub decimals: usize,
}

pub fn best_mark(values: &[Option<f32>], decimals: usize, max: usize) -> BestMark {
    let finite: Vec<(usize, f32)> =
        values.iter().enumerate().filter_map(|(i, v)| v.filter(|v| v.is_finite()).map(|v| (i, v))).collect();
    let none = BestMark { best: Vec::new(), joint: false, decimals };
    if finite.len() < 2 {
        return none;
    }
    let top = finite.iter().map(|f| f.1).fold(f32::NEG_INFINITY, f32::max);
    // Nothing scored: a 0 is not a best.
    if top.partial_cmp(&0.) != Some(Ordering::Greater) {
        return none;
    }
    let best: Vec<usize> = finite.iter().filter(|f| f.1 == top).map(|f| f.0).collect();
    let rest: Vec<f32> = finite.iter().filter(|f| f.1 != top).map(|f| f.1).collect();
    let mut d = decimals;
    while d < max && rest.iter().any(|v| format!("{v:.d$}") == format!("{top:.d$}")) {
        d += 1;
    }
    BestMark { joint: best.len() > 1, best, decimals: d }
}

/// The word beside a best value: "best", or "joint best" for a tie.
pub fn best_word(m: &BestMark) -> &'static str {
    if m.joint { "joint best" } else { "best" }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::api::Backend;
    use crate::api::mock::MockBackend;
    use crate::store::Load;

    #[test]
    fn the_best_is_marked_once_or_jointly_with_decimals_that_tell_it_apart() {
        let m = best_mark(&[Some(0.604), Some(0.5), None], 3, 6);
        assert_eq!((m.best.clone(), m.joint, m.decimals, best_word(&m)), (vec![0], false, 3, "best"));
        let m = best_mark(&[Some(0.604), Some(0.604), Some(0.2)], 3, 6);
        assert_eq!((m.best.clone(), best_word(&m)), (vec![0, 1], "joint best"));
        // 0.6041 beside 0.6039 reads 0.604 twice at three decimals: four tell them apart.
        assert_eq!(best_mark(&[Some(0.6041), Some(0.6039)], 3, 6).decimals, 4);
        // One value, or nothing scored, is no best.
        assert!(best_mark(&[Some(0.7), None], 3, 6).best.is_empty());
        assert!(best_mark(&[Some(0.), Some(0.)], 3, 6).best.is_empty());
    }

    #[test]
    fn models_group_by_what_they_were_evaluated_on_and_sort_inside_each_group() {
        let api = MockBackend::new(4200, true);
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        store.runs = Load::Ready(api.runs().unwrap());
        store.datasets = Load::Ready(api.datasets().unwrap());
        let mut models = api.models().unwrap();
        // An imported file and published weights join the demo's trained models.
        let base = models[0].clone();
        models.push(Model {
            id: "m_imp".into(),
            name: "zz-imported".into(),
            run_id: None,
            source: None,
            ..base.clone()
        });
        models.push(Model {
            id: "m_pub".into(),
            name: "folo-n-coco".into(),
            run_id: None,
            source: Some(ModelSource::Published),
            ..base.clone()
        });
        let groups = group_models(&store, &models, Some(("map", true)));
        let keys: Vec<&str> = groups.iter().map(|g| g.set.key.as_str()).collect();
        // Every model once, in some group.
        assert_eq!(groups.iter().map(|g| g.rows.len()).sum::<usize>(), models.len());
        // Ranked dataset groups first, then published, then imported.
        let n = keys.len();
        assert_eq!(&keys[n - 2..], ["~published", "~imported"]);
        assert!(groups[..n - 2].iter().all(|g| g.set.ranked), "{keys:?}");
        assert!(
            groups[0]
                .set
                .label
                .ends_with(&format!("of {}", store.dataset_name(keys[0].split(':').next().unwrap())))
        );
        // Dataset groups by name.
        let names: Vec<String> = groups[..n - 2].iter().map(|g| g.set.order.1.clone()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        // Inside each group, the highest mAP50-95 first; flipping the sort flips measured scores
        // while keeping unscored models last.
        for g in &groups {
            let scored: Vec<f32> = g.rows.iter().filter_map(|m| m.metrics.map(|x| x.map50_95)).collect();
            assert!(scored.windows(2).all(|w| w[0] >= w[1]), "{}: {scored:?}", g.set.label);
            let first_none = g.rows.iter().position(|m| m.metrics.is_none()).unwrap_or(g.rows.len());
            assert!(g.rows[first_none..].iter().all(|m| m.metrics.is_none()), "unscored models last");
        }
        let up = group_models(&store, &models, Some(("map", false)));
        assert_eq!(up.iter().map(|g| g.set.key.clone()).collect::<Vec<_>>(), keys);
        for g in &up {
            let scored: Vec<f32> = g.rows.iter().filter_map(|m| m.metrics.map(|x| x.map50_95)).collect();
            assert!(scored.windows(2).all(|w| w[0] <= w[1]), "{}: {scored:?}", g.set.label);
            let first_none = g.rows.iter().position(|m| m.metrics.is_none()).unwrap_or(g.rows.len());
            assert!(g.rows[first_none..].iter().all(|m| m.metrics.is_none()), "unscored models last asc");
        }
        // A model whose run is gone has an unknown split: not ranked.
        let orphan = Model { run_id: Some("run_gone".into()), ..base };
        let set = eval_set(&store, &orphan);
        assert!(!set.ranked && set.label.ends_with("split unknown"), "{set:?}");
    }

    #[test]
    fn unscored_and_missing_models_always_sort_last_in_both_directions() {
        let api = MockBackend::new(4200, true);
        let store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let base = api.models().unwrap().remove(0);
        let mk = |id: &str, name: &str, metrics: Option<(f32, f32)>, created: &str| Model {
            id: id.into(),
            name: name.into(),
            run_id: None,
            source: None,
            metrics: metrics.map(|(m95, m50)| crate::api::ModelMetrics {
                map50_95: m95,
                map50: m50,
                precision: 0.5,
                recall: 0.5,
                best_f1_conf: None,
            }),
            created: created.into(),
            ..base.clone()
        };
        let models = vec![
            mk("m_none_b", "unscored-b", None, "bad-date"),
            mk("m_low", "low", Some((0.25, 0.40)), "2026-11-01T00:00:00Z"),
            mk("m_none_a", "unscored-a", None, ""),
            mk("m_high", "high", Some((0.75, 0.90)), "2026-11-05T00:00:00Z"),
        ];
        for key in ["map", "map50", "created"] {
            let desc = group_models(&store, &models, Some((key, true)));
            let desc_ids: Vec<&str> = desc[0].rows.iter().map(|m| m.id.as_str()).collect();
            assert_eq!(desc_ids, ["m_high", "m_low", "m_none_a", "m_none_b"], "{key} desc");

            let asc = group_models(&store, &models, Some((key, false)));
            let asc_ids: Vec<&str> = asc[0].rows.iter().map(|m| m.id.as_str()).collect();
            assert_eq!(asc_ids, ["m_low", "m_high", "m_none_a", "m_none_b"], "{key} asc");
        }
    }
}
