//! Models: the model list and a model's evaluation (PR curves, confusion matrix, per-class
//! metrics) with export.

mod class_table;
mod confusion;
mod confusion_ink;
mod detail;
mod details;
mod groups;
mod list;
mod pr;

#[cfg(test)]
pub(crate) use groups::group_models;
#[cfg(test)]
pub(crate) use list::filter_models;
