//! Label assignment algorithms beyond the default task-aligned assigner ([`crate::assign`]).
//!
//! - [`simota`]: **SimOTA** (Simplified Optimal Transport Assignment, Ge et al. YOLOX 2021) — dynamic-`k`
//!   assignment using pairwise classification + IoU regression costs, optional in-box and center-prior
//!   spatial gating, top-`q` IoU dynamic-`k` estimation, and lowest-cost conflict resolution with
//!   deterministic tie-breaking.

pub mod simota;
#[cfg(test)]
mod tests;

pub use simota::{SimOtaAssignment, SimOtaConfig, assign_simota};
