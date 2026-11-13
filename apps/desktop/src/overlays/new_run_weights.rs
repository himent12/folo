//! New run's Start from select (the options and what picking one does: `app/start_from.rs`).

use std::rc::Rc;

use folo_ui::{AnyElement, Context, SharedString, Window};

use crate::app::FoloApp;

impl FoloApp {
    /// The Start from select: from scratch, then the weights a run can fine-tune from.
    pub(super) fn start_from_select(&self, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let choices = self.nr_start_choices();
        let current = self.new_run.as_ref().and_then(|f| f.weights.clone());
        let selected = choices.iter().position(|c| c.id == current).unwrap_or(0);
        let shown =
            SharedString::from(choices.get(selected).map_or("From scratch".into(), |c| c.label.clone()));
        let options = choices
            .iter()
            .map(|c| (SharedString::from(c.label.clone()), c.detail.clone().map(SharedString::from)))
            .collect();
        self.select(
            "nr-weights",
            shown,
            options,
            selected,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                if let Some(c) = choices.get(i) {
                    this.pick_start(c, cx);
                }
            }),
        )
    }
}
