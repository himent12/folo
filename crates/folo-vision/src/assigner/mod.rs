//! Label assignment algorithms beyond the default task-aligned assigner ([`crate::assign`]).
//!
//! - [`simota`]: **SimOTA** (Simplified Optimal Transport Assignment, Ge et al. YOLOX 2021) — dynamic-`k`
//!   assignment using pairwise classification + IoU regression costs, optional in-box and center-prior
//!   spatial gating, top-`q` IoU dynamic-`k` estimation, and lowest-cost conflict resolution with
//!   deterministic tie-breaking.
//! - [`hungarian`]: Exact `O(min(N, M)² · max(N, M))` Hungarian / linear-sum assignment
//!   ([`hungarian_assign`]) and [`OneToOneMatcher`] combining classification + L1 + GIoU/CIoU costs for
//!   NMS-free detection heads.

pub mod hungarian;
pub mod simota;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_hungarian;

pub use hungarian::{
    ClsCostKind, HungarianResult, OneToOneAssignment, OneToOneConfig, OneToOneMatcher, hungarian_assign,
    hungarian_assign_f32,
};
pub use simota::{SimOtaAssignment, SimOtaConfig, assign_simota};
