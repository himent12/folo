//! Agent labeling (servers with `agent_labeling`, docs/LABELING.md): state, loading, live SSE
//! events, and mutations backing the Labeling screens (`screens/label/`).

mod actions;
mod forms;
mod load;
mod state;
mod stream;

#[cfg(test)]
pub use actions::validate_new_job;
pub use actions::{LABEL_VAL_FRACTIONS, filter_label_images, filter_label_jobs, parse_class_list};
#[cfg(test)]
pub use load::route_label;
#[cfg(test)]
pub(crate) use load::suffix_prefix_overlap;
pub use state::{LabelImageFilter, LabelJobFilter, LabelState, SelectedBox};
