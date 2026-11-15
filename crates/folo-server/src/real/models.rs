//! Models: those of finished runs, the models folder and published weights, and the dataset `/eval` uses.

use axum::http::StatusCode;

use crate::error::{Error, Result};
use crate::models::{ModelRef, Models, model_of};
use crate::runs::{DatasetRef, RunEntry, run_model_id};

use super::RealStore;

/// What `folo train --weights` takes for model `m` on this machine: a published model's name, else its file.
pub(super) fn cli_weights(m: &ModelRef) -> String {
    match m.model.source {
        Some(folo_api::ModelSource::Published) => m.model.id.clone(),
        _ => folo_api::json::path(&m.path),
    }
}

impl RealStore {
    /// The models of runs that finished with weights: `best.folo` (`m_<run>`) and `last.folo` (`m_<run>-last`).
    fn run_models(&self) -> Vec<ModelRef> {
        let finished = self.runs.entries().into_iter().filter(|e| e.run().model_id.is_some());
        finished
            .flat_map(|e| [false, true].into_iter().filter_map(move |last| run_model(&self.models, &e, last)))
            .collect()
    }

    /// Every model, newest first.
    pub(super) fn all_models(&self) -> Vec<ModelRef> {
        let mut all = self.run_models();
        all.extend(self.models.scan());
        all.extend(self.models.published());
        all.sort_by(|a, b| b.model.created.cmp(&a.model.created).then_with(|| a.model.id.cmp(&b.model.id)));
        all
    }

    pub(super) fn model(&self, id: &str) -> Result<ModelRef> {
        // Run models are found without scanning the models folder, also those of a run in progress (not listed
        // until it ends): a cloud control plane copies a worker run's checkpoint while it trains.
        let run = id
            .strip_prefix("m_")
            .map(|s| (format!("run_{}", s.strip_suffix("-last").unwrap_or(s)), s.ends_with("-last")));
        if let Some((r, last)) = run
            && let Ok(e) = self.runs.get(&r)
            && let Some(m) = run_model(&self.models, &e, last).filter(|m| m.model.id == id)
        {
            return Ok(m);
        }
        if let Some(p) = folo_api::weights::find(id) {
            return self.models.published_model(id).ok_or_else(|| self.published_not_found(p));
        }
        self.models.scan().into_iter().find(|m| m.model.id == id).ok_or_else(|| Error::not_found("model", id))
    }

    /// `model_not_found` for published weights that are not downloaded (or that this store does not list).
    fn published_not_found(&self, p: &folo_api::weights::Published) -> Error {
        let (message, hint) = match self.models.published_dir() {
            Some(dir) => (
                format!(
                    "published weights '{}' are not downloaded ({} does not exist)",
                    p.name,
                    folo_api::json::path(dir.join(p.asset))
                ),
                format!(
                    "run `folo weights pull {}` on the server's machine; it lists as model `{}` from then on, \
                     no restart needed",
                    p.name, p.name
                ),
            ),
            None => (
                format!("model '{}' does not exist (this server does not list published weights)", p.name),
                format!(
                    "copy the file `folo weights pull {}` downloads into <data dir>/models; list_models (MCP) or \
                     GET /api/models then lists it",
                    p.name
                ),
            ),
        };
        Error::new(StatusCode::NOT_FOUND, "model_not_found", message).hint(hint)
    }

    /// The dataset `/eval` validates `m` on and its index fingerprint: its run's dataset (the registered copy when
    /// it still is, else the path the run trained on, with no fingerprint: such results are not cached), else the
    /// first registered dataset with the model's classes.
    pub(super) fn eval_dataset(&self, m: &ModelRef) -> Result<(DatasetRef, Option<String>)> {
        if let Some(d) = &m.dataset {
            return Ok(
                match self.entry(&d.id).and_then(|e| Ok((self.dataset_ref(&e), self.fresh_index(&e)?))) {
                    Ok((r, idx)) => (r, Some(idx.fingerprint.hash.clone())),
                    Err(_) => (d.clone(), None),
                },
            );
        }
        let list = self.entries.read().unwrap().clone();
        for e in list {
            if let Ok(idx) = self.fresh_index(&e)
                && idx.classes == m.model.classes
            {
                return Ok((self.dataset_ref(&e), Some(idx.fingerprint.hash.clone())));
            }
        }
        Err(no_matching_dataset(&m.model.classes))
    }
}

/// `404 no_matching_dataset`: no registered dataset has exactly `classes`, so `/eval` has nothing to run on. The
/// message names the count and the first few classes, never the whole list (a COCO model has 80).
pub(crate) fn no_matching_dataset(classes: &[String]) -> Error {
    const SHOWN: usize = 4;
    let mut names = classes.iter().take(SHOWN).map(String::as_str).collect::<Vec<_>>().join(", ");
    if classes.len() > SHOWN {
        names.push_str(", …");
    }
    let n = classes.len();
    let what = if n == 1 { "class".to_string() } else { format!("{n} classes") };
    Error::new(
        StatusCode::NOT_FOUND,
        "no_matching_dataset",
        format!("no registered dataset has this model's {what} ({names})"),
    )
    .hint("register a dataset with exactly these classes, in this order, to evaluate it")
}

/// Run `e`'s `best.folo` (`m_<run>`) or `last.folo` (`m_<run>-last`) as a model, when the file is there.
fn run_model(models: &Models, e: &RunEntry, last: bool) -> Option<ModelRef> {
    let (suffix, file) = if last { ("last", "last.folo") } else { ("best", "best.folo") };
    let path = e.dir().join("weights").join(file);
    if !path.is_file() {
        return None;
    }
    let run = e.run();
    match models.meta(&path) {
        Ok(meta) => {
            // A checkpoint's `folo.metrics` are the run's best; last.folo gets its own epoch's (unknown when a crash
            // kept that epoch out of the history).
            let metrics = if last { e.last_metrics() } else { meta.metrics };
            let model = model_of(
                run_model_id(&run.id, last),
                format!("{}-{suffix}", run.name),
                &meta,
                Some(run.id.clone()),
                metrics,
            );
            Some(ModelRef::new(model, path, Some(e.dataset())))
        }
        Err(err) => {
            models.warn_skipped(&path, &err);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_matching_dataset_names_a_few_classes_not_all() {
        let coco: Vec<String> = (0..80).map(|i| format!("c{i}")).collect();
        let e = no_matching_dataset(&coco);
        assert_eq!((e.status, e.body.code.as_str()), (StatusCode::NOT_FOUND, "no_matching_dataset"));
        assert_eq!(e.body.message, "no registered dataset has this model's 80 classes (c0, c1, c2, c3, …)");
        assert!(!e.body.hint.unwrap().contains("c4"));
        let one = no_matching_dataset(&["person".to_string()]);
        assert_eq!(one.body.message, "no registered dataset has this model's class (person)");
    }
}
