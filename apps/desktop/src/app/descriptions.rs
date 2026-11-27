//! How runs and models are described across pages: a run's identity line, the split its scores
//! come from, its subtitle, and whether a model is end-to-end.

use crate::api::*;

use super::FoloApp;

impl FoloApp {
    /// A run's identity line, "folo-m · shapes-hard": architecture, then dataset, the one order every
    /// surface uses (the hero, Recent runs, the run page, focus tips).
    pub fn run_meta(&self, r: &Run) -> String {
        let ds = self.store.run_dataset_name(r);
        format!("{} · {ds}", self.store.run_arch(r).as_str())
    }

    /// The split a run's per-epoch scores come from: "val", or "train" when its dataset has no
    /// validation images (the trainer then evaluates on the training images), so those scores are
    /// never called validation.
    pub fn eval_split(&self, r: &Run) -> &'static str {
        self.store
            .dataset_summary(&r.config.dataset_id)
            .map_or("val", |d| if d.splits.val == 0 { "train" } else { "val" })
    }

    /// The mAP chart's title for that split.
    pub fn map_title(&self, r: &Run) -> &'static str {
        if self.eval_split(r) == "train" { "mAP on training images" } else { "Validation mAP" }
    }

    /// A run's subtitle, the same in the Overview hero and on its page: identity, device, start.
    pub fn run_subtitle(
        &self,
        t: &crate::theme::Theme,
        r: &Run,
        size: crate::theme::TypeStep,
        color: folo_ui::Hsla,
        id: &str,
    ) -> folo_ui::Div {
        let ds = self.store.run_dataset_name(r);
        let mut rest = vec![ds];
        // A cloud run's device is where it rents it ("RunPod · RTX 4090"); a run here names the
        // device it trains on once it has started (waiting in the queue, it has none yet).
        match &r.cloud {
            Some(c) => rest.push(self.cloud_device(r, c)),
            None if r.started.is_some() => rest.push(crate::ui::device_label(&r.device)),
            None => {}
        }
        // Precision is part of what trains (the e2e tag names the head).
        if r.config.amp {
            rest.push("mixed precision".into());
        }
        if let Some(s) = r.started.as_deref().and_then(crate::api::time::parse) {
            // Training, it sits by an elapsed clock: the same seconds, not "just now".
            let now = self.store.api.now();
            let when = if r.status.is_active() {
                crate::api::time::ago_live(s, now)
            } else {
                crate::api::time::ago(s, now)
            };
            rest.push(format!("started {when}"));
        }
        // The architecture leads, its e2e tag right after it, as in every list.
        let arch = self.store.run_arch(r);
        let tag = (r.config.head == Head::O2o).then(|| crate::ui::e2e_tag(t, format!("e2e-{id}-{}", r.id)));
        crate::ui::meta_row(size, color, arch.as_str(), tag, &rest)
    }

    /// Whether `m` is end-to-end (the one-to-one head: it predicts without NMS). The model's own
    /// record says (`Model.head`, from its folo.config), so imported and orphaned models read right.
    pub fn model_e2e(&self, m: &Model) -> bool {
        m.head == Head::O2o
    }

    /// The help shown wherever a model or run is marked end-to-end.
    pub const E2E_HELP: &'static str =
        "End-to-end: predicts boxes directly, without non-maximum suppression (NMS)";
}
