//! Agent labeling (servers with `agent_labeling`, docs/LABELING.md): state, loading, live SSE
//! events, and mutations backing the Labeling screens (`screens/label/`).

mod actions;
mod editor;
mod editor_actions;
mod forms;
mod load;
mod state;
mod stream;

#[cfg(test)]
pub use actions::validate_new_job;
pub use actions::{LABEL_VAL_FRACTIONS, filter_label_images, filter_label_jobs, parse_class_list};
pub use editor::{EditTool, PendingImageSwitch, PendingRouteNav};
#[cfg(test)]
pub(crate) use load::merge_log_lines;
#[cfg(test)]
pub use load::route_label;
pub use state::{LabelImageFilter, LabelJobFilter, LabelState, SelectedBox};
