//! Models: the model list and a model's evaluation (PR curves, confusion matrix, per-class
//! metrics) with export.

mod class_table;
mod compare;
mod confusion;
mod confusion_ink;
mod detail;
mod details;
pub(crate) mod export_builder;
mod groups;
mod list;
mod pr;

pub(crate) use compare::ModelCompareState;
pub(crate) use export_builder::{MODEL_CLI_CONF, MODEL_CLI_IOU, ModelCliState};
#[cfg(test)]
pub(crate) use groups::group_models;
#[cfg(test)]
pub(crate) use list::filter_models;
