//! New run's Start from: the weights a run can fine-tune from, as Studio's field offers them (from
//! scratch, the trained models of this size best first, those of the other sizes, published weights).

use folo_ui::Context;

use crate::api::{Model, ModelSource};
use crate::ui::f3;

use super::FoloApp;
use super::forms::ARCHS;

/// One Start from option: the weights' id (`None`: from scratch), its name, what it says beside it,
/// and the architecture index picking it switches to.
#[derive(Clone, Debug, PartialEq)]
pub struct StartChoice {
    pub id: Option<String>,
    pub label: String,
    pub detail: Option<String>,
    pub arch: Option<usize>,
}

/// Weights a run can start from: not an export (fused for inference only).
fn trainable(m: &Model) -> bool {
    m.fused != Some(true)
}

fn arch_index(m: &Model) -> Option<usize> {
    ARCHS.iter().position(|a| *a == m.arch)
}

fn score(m: &Model) -> f32 {
    m.metrics.as_ref().map_or(-1., |s| s.map50_95)
}

/// The options in the field's order, for a form on architecture `arch`; `current` (the form's
/// weights) is kept as an option when the server doesn't list it but it is `base`'s, the weights
/// of the run the form was prefilled from (one that started from a file). Weights picked here and
/// gone since (deleted elsewhere) are not: [`FoloApp::nr_weights_gone`] says so.
pub fn start_choices(
    models: &[Model],
    arch: usize,
    current: Option<&str>,
    base: Option<&str>,
) -> Vec<StartChoice> {
    let mut out = vec![StartChoice { id: None, label: "From scratch".into(), detail: None, arch: None }];
    let published = |m: &&Model| m.source == Some(ModelSource::Published);
    let mut trained: Vec<&Model> = models.iter().filter(|m| trainable(m) && !published(m)).collect();
    // This size first, then the others (picking one switches the size), each best first.
    trained.sort_by(|a, b| {
        (arch_index(a) != Some(arch)).cmp(&(arch_index(b) != Some(arch))).then(score(b).total_cmp(&score(a)))
    });
    for m in trained {
        let other = arch_index(m).filter(|&i| i != arch);
        let s = m.metrics.as_ref().map(|s| f3(s.map50_95));
        let detail = match (other, s) {
            (Some(i), Some(s)) => format!("{} · {s}", ARCHS[i].as_str()),
            (Some(i), None) => ARCHS[i].as_str().to_string(),
            (None, Some(s)) => s,
            (None, None) => String::new(),
        };
        out.push(StartChoice {
            id: Some(m.id.clone()),
            label: m.name.clone(),
            detail: (!detail.is_empty()).then_some(detail),
            arch: arch_index(m),
        });
    }
    for m in models.iter().filter(|m| trainable(m) && published(m)) {
        out.push(StartChoice {
            id: Some(m.id.clone()),
            label: m.name.clone(),
            detail: Some("published".into()),
            arch: arch_index(m),
        });
    }
    if let Some(w) = current.filter(|w| base == Some(*w) && !out.iter().any(|c| c.id.as_deref() == Some(*w)))
    {
        let file = w.rsplit(['/', '\\']).next().unwrap_or(w);
        out.push(StartChoice {
            id: Some(w.to_string()),
            label: file.to_string(),
            detail: Some("the earlier run's".into()),
            arch: None,
        });
    }
    out
}

/// The dataset published weights' description names, the usual way ("… on COCO train2017 …"):
/// a capitalised word after "on", as Studio reads it (`web/src/lib/published.ts`).
fn pretrained_on(about: &str) -> Option<&str> {
    about.match_indices("on ").find_map(|(i, _)| {
        let before_ok = i == 0 || about[..i].ends_with(' ');
        let rest = &about[i + 3..];
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '-')).unwrap_or(rest.len());
        let word = rest[..end].trim_end_matches('-');
        (before_ok && word.starts_with(|c: char| c.is_ascii_uppercase())).then_some(word)
    })
}

/// Published weights in plain words, as Studio says them: "Pretrained on COCO (80 classes) · good
/// for fine-tuning" (the server's own description is provenance).
fn published_summary(m: &Model) -> String {
    let on = m
        .about
        .as_deref()
        .and_then(pretrained_on)
        .map_or("Pretrained".to_string(), |d| format!("Pretrained on {d}"));
    let n = m.classes.len();
    format!("{on} ({n} {}) · good for fine-tuning", if n == 1 { "class" } else { "classes" })
}

impl FoloApp {
    /// Start from's options for the open New run form.
    pub fn nr_start_choices(&self) -> Vec<StartChoice> {
        let Some(f) = self.new_run.as_ref() else { return Vec::new() };
        let models = self.store.models.ready().map(Vec::as_slice).unwrap_or_default();
        start_choices(
            models,
            f.arch,
            f.weights.as_deref(),
            f.base.as_ref().and_then(|b| b.weights.as_deref()),
        )
    }

    /// The picked weights are no longer listed (their run deleted elsewhere, also while the form
    /// waited as a draft): Start waits for another choice, and this says why.
    pub fn nr_weights_gone(&self) -> Option<String> {
        let f = self.new_run.as_ref()?;
        let w = f.weights.as_deref()?;
        let models = self.store.models.ready()?;
        let from_base = f.base.as_ref().and_then(|b| b.weights.as_deref()) == Some(w);
        let listed = models.iter().any(|m| m.id == w);
        (!listed && !from_base).then(|| {
            let name = f.weights_name.clone().unwrap_or_else(|| w.to_string());
            format!("{name} was deleted. Pick other weights, or From scratch.")
        })
    }

    /// Picks a Start from option: its weights, and their size (a fine-tune trains the weights' own).
    pub fn pick_start(&mut self, c: &StartChoice, cx: &mut Context<Self>) {
        let Some(f) = self.new_run.as_mut() else { return };
        f.weights = c.id.clone();
        f.weights_name = c.id.as_ref().map(|_| c.label.clone());
        if let Some(a) = c.arch {
            f.arch = a;
        }
        self.resuggest_run_name(cx);
        cx.notify();
    }

    /// Which of `choices` the form has: `None` when its weights are none of them (gone since).
    pub fn nr_start_selected(&self, choices: &[StartChoice]) -> Option<usize> {
        let current = self.new_run.as_ref()?.weights.clone();
        choices.iter().position(|c| c.id == current)
    }

    /// The size picked by hand; a press on the size already chosen changes nothing (its weights
    /// stay).
    pub fn nr_set_arch(&mut self, arch: usize, cx: &mut Context<Self>) {
        let Some(f) = self.new_run.as_mut().filter(|f| f.arch != arch) else { return };
        f.arch = arch;
        self.nr_arch_changed(cx);
    }

    /// The architecture changed by hand: weights of another size can't start this run (the server
    /// would train the weights' own size), so it trains from scratch again; so do weights whose
    /// size isn't known here (not listed: a checkpoint the earlier run started from).
    pub fn nr_arch_changed(&mut self, cx: &mut Context<Self>) {
        let models = self.store.models.ready().map(Vec::as_slice).unwrap_or_default();
        if let Some(f) = self.new_run.as_mut()
            && let Some(w) = f.weights.as_deref()
            && models.iter().find(|m| m.id == w).is_none_or(|m| arch_index(m) != Some(f.arch))
        {
            f.weights = None;
            f.weights_name = None;
        }
        self.resuggest_run_name(cx);
    }

    /// Start from's hint: what the picked weights are, or why to pick some.
    pub fn nr_start_hint(&self) -> String {
        let Some(f) = self.new_run.as_ref() else { return String::new() };
        let models = self.store.models.ready().map(Vec::as_slice).unwrap_or_default();
        let Some(w) = f.weights.as_deref() else {
            return "Trained weights learn a small dataset much faster".into();
        };
        let Some(m) = models.iter().find(|m| m.id == w) else { return format!("Weights {w}") };
        if m.source == Some(ModelSource::Published) {
            return published_summary(m);
        }
        let on = m.run_id.as_deref().and_then(|r| self.store.run(r)).map(|r| self.store.run_dataset_name(r));
        match (m.metrics.as_ref(), on) {
            (Some(s), Some(ds)) => format!("mAP50-95 {} on {ds}", f3(s.map50_95)),
            (Some(s), None) => format!("mAP50-95 {}", f3(s.map50_95)),
            (None, Some(ds)) => format!("Trained on {ds}"),
            (None, None) => "Trained weights".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::tests::{drive, open, open_with};
    use crate::api::mock::MockBackend;
    use crate::api::{Backend, ModelSource};

    #[test]
    fn start_from_offers_this_size_first_then_others_then_published_and_sends_the_pick() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            // A published weights file among the demo's models (as `folo weights pull` lists one).
            let models = this.store.models.ready_mut().unwrap();
            let mut p = models[0].clone();
            (p.id, p.name, p.source, p.about) = (
                "folo-n-coco".into(),
                "folo-n-coco".into(),
                Some(ModelSource::Published),
                Some("folo-n trained from scratch on COCO train2017 at 640 px".into()),
            );
            p.arch = crate::api::Arch::N;
            models.push(p);
            this.open_new_run(None, w, cx);
        });
        drive(&mut app, window, |this, _, cx| {
            let arch = this.new_run.as_ref().unwrap().arch;
            let choices = this.nr_start_choices();
            assert_eq!(choices[0].label, "From scratch");
            // This size's trained models before the other sizes', the published ones last.
            let sizes: Vec<bool> = choices[1..]
                .iter()
                .filter(|c| c.detail.as_deref() != Some("published"))
                .map(|c| c.arch == Some(arch))
                .collect();
            assert!(sizes.windows(2).all(|w| w[0] >= w[1]), "{sizes:?}");
            let published = choices.iter().position(|c| c.id.as_deref() == Some("folo-n-coco")).unwrap();
            assert_eq!(published, choices.len() - 1);
            // Picking it sends it and switches to its size.
            let pick = choices[published].clone();
            this.pick_start(&pick, cx);
            let config = this.form_config(cx).unwrap();
            assert_eq!((config.weights.as_deref(), config.arch), (Some("folo-n-coco"), crate::api::Arch::N));
            let n = this.store.models.ready().unwrap().last().unwrap().classes.len();
            assert_eq!(
                this.nr_start_hint(),
                format!("Pretrained on COCO ({n} classes) · good for fine-tuning")
            );
            // Another size by hand: those weights can't start it, so from scratch again.
            this.new_run.as_mut().unwrap().arch = 1;
            this.nr_arch_changed(cx);
            assert_eq!(this.form_config(cx).unwrap().weights, None);
        });
    }

    #[test]
    fn a_dataset_with_no_training_image_keeps_new_run_from_starting() {
        let api = Arc::new(MockBackend::new(4200, true).with_no_training_images("ds_8f2c"));
        let (mut app, window) = open_with(api.clone());
        drive(&mut app, window, |this, w, cx| this.open_new_run(Some("ds_8f2c"), w, cx));
        let runs = api.runs().unwrap().len();
        drive(&mut app, window, |this, w, cx| {
            let why = this.dataset_blocked().expect("blocked");
            assert!(why.ends_with("has no training images. Pick another dataset."), "{why}");
            this.submit_new_run(w, cx);
            assert!(!this.new_run.as_ref().unwrap().submitting, "nothing is sent");
        });
        assert_eq!(api.runs().unwrap().len(), runs);
    }

    #[test]
    fn a_dataset_counted_empty_before_is_asked_for_again() {
        // The list says no training image (read before images were added); the server, asked
        // again when New run opens on it, counts them: Start is not held back.
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            this.store
                .datasets
                .ready_mut()
                .unwrap()
                .iter_mut()
                .find(|d| d.id == "ds_8f2c")
                .unwrap()
                .splits
                .train = 0;
            this.open_new_run(Some("ds_8f2c"), w, cx);
        });
        drive(&mut app, window, |this, _, _| assert_eq!(this.dataset_blocked(), None));
    }

    #[test]
    fn weights_deleted_while_picked_hold_start_until_another_pick() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
        drive(&mut app, window, |this, w, cx| {
            let pick = this.nr_start_choices()[1].clone();
            this.pick_start(&pick, cx);
            // Its run deleted elsewhere: the next list no longer has it.
            let id = pick.id.clone().unwrap();
            this.store.models.ready_mut().unwrap().retain(|m| m.id != id);
            assert!(this.nr_start_choices().iter().all(|c| c.id.as_deref() != Some(id.as_str())));
            let why = this.nr_weights_gone().expect("said");
            assert_eq!(why, format!("{} was deleted. Pick other weights, or From scratch.", pick.label));
            this.submit_new_run(w, cx);
            assert!(!this.new_run.as_ref().unwrap().submitting, "nothing is sent");
            // The select shows none of its options as the choice, so From scratch is a change.
            assert_eq!(this.nr_start_selected(&this.nr_start_choices()), None);
            let scratch = this.nr_start_choices()[0].clone();
            this.pick_start(&scratch, cx);
            assert_eq!(this.nr_weights_gone(), None);
        });
    }

    #[test]
    fn weights_of_an_unknown_size_go_when_the_size_changes() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            let mut config = this.store.runs.ready().unwrap()[0].config.clone();
            config.weights = Some("runs/a/weights/last.folo".into());
            this.open_new_run_from(&config, None, w, cx);
        });
        drive(&mut app, window, |this, _, cx| {
            // Prefilled: kept, and listed as the earlier run's.
            assert_eq!(this.form_config(cx).unwrap().weights.as_deref(), Some("runs/a/weights/last.folo"));
            assert!(this.nr_start_choices().iter().any(|c| c.label == "last.folo"));
            // A press on the size it has keeps them; another size drops them.
            let arch = this.new_run.as_ref().unwrap().arch;
            this.nr_set_arch(arch, cx);
            assert_eq!(this.form_config(cx).unwrap().weights.as_deref(), Some("runs/a/weights/last.folo"));
            this.nr_set_arch((arch + 1) % 5, cx);
            assert_eq!(this.form_config(cx).unwrap().weights, None);
        });
    }

    #[test]
    fn a_late_empty_count_never_undoes_a_newer_one() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, _, _| {
            let list = this.store.datasets.ready_mut().unwrap();
            let fresh = list[0].clone();
            let mut stale = fresh.clone();
            stale.splits.train = 0;
            list[0].splits.train = 0;
            // The newer answer (images added) lands first, the older empty one after it.
            assert!(super::super::new_run::recount_dataset(list, fresh.clone()));
            assert!(!super::super::new_run::recount_dataset(list, stale));
            assert_eq!(list[0].splits.train, fresh.splits.train);
        });
    }
}
