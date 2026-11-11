//! What an empty Runs or Models page asks for first: a dataset, then a run. The empty state says
//! it and carries the button that does it.

use folo_ui::{AnyElement, Context, IntoElement, Window};

use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;

/// The step a person on an empty server takes next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirstStep {
    /// No dataset yet: training needs one.
    AddDataset,
    /// Datasets, but no run yet.
    NewRun,
    /// Nothing to ask for (runs are under way, or the lists are still loading).
    Wait,
}

/// The next step from what the server has (`None`: that list isn't in yet).
pub fn first_step(datasets: Option<usize>, runs: Option<usize>) -> FirstStep {
    match (datasets, runs) {
        (Some(0), _) => FirstStep::AddDataset,
        (Some(_), Some(0)) => FirstStep::NewRun,
        _ => FirstStep::Wait,
    }
}

impl FoloApp {
    /// The server's next step, from the lists it has sent.
    pub(crate) fn next_first_step(&self) -> FirstStep {
        first_step(self.store.datasets.ready().map(Vec::len), self.store.runs.ready().map(Vec::len))
    }

    /// The button for `step`: Add dataset (to Datasets, with the folder picker open, as Overview's
    /// Getting started does) or New run.
    pub(crate) fn first_step_button(
        &self,
        key: &str,
        step: FirstStep,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let button = match step {
            FirstStep::AddDataset => self.button(
                key,
                "Add dataset",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    this.navigate(Route::Datasets, w, cx);
                    this.add_dataset_folder(cx);
                }),
            ),
            FirstStep::NewRun => self.button(
                key,
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_new_run(None, w, cx)),
            ),
            FirstStep::Wait => return None,
        };
        Some(button.into_any_element())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_server_asks_for_a_dataset_then_a_run() {
        assert_eq!(first_step(Some(0), Some(0)), FirstStep::AddDataset);
        assert_eq!(first_step(Some(0), None), FirstStep::AddDataset);
        assert_eq!(first_step(Some(2), Some(0)), FirstStep::NewRun);
        assert_eq!(first_step(Some(2), Some(3)), FirstStep::Wait);
        // Still loading: no advice yet.
        assert_eq!(first_step(None, Some(0)), FirstStep::Wait);
        assert_eq!(first_step(Some(2), None), FirstStep::Wait);
    }
}
