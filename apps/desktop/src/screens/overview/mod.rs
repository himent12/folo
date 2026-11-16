//! Overview: the live run, the machine, recent runs, best models and datasets at a glance.

mod datasets;
mod getting_started;
mod idle;
mod live;
mod page;
mod tables;

#[cfg(test)]
pub(crate) use datasets::card_facts;
pub(crate) use datasets::dataset_problem;
