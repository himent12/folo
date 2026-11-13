//! New run's Start from select (the options and what picking one does: `app/start_from.rs`).

use std::rc::Rc;

use folo_ui::{AnyElement, Context, SharedString, Window};

use crate::app::FoloApp;

impl FoloApp {
    /// The Start from select: from scratch, then the weights a run can fine-tune from.
    pub(super) fn start_from_select(&self, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let choices = self.nr_start_choices();
        // Weights gone since they were picked: none of the options is the choice (so picking
        // From scratch counts as a change), and the field asks for one.
        let selected = self.nr_start_selected(&choices).unwrap_or(choices.len());
        let shown = SharedString::from(
            choices.get(selected).map_or("Pick what to start from".into(), |c| c.label.clone()),
        );
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
