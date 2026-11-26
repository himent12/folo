//! Label assignment algorithms beyond the default task-aligned assigner ([`crate::assign`]).
//!
//! - [`atss`]: **ATSS** (Adaptive Training Sample Selection, Zhang et al. CVPR 2020) — multi-scale
//!   center-distance top-`k` candidate selection, dynamic `mean + std` IoU thresholding, in-GT filtering,
//!   and highest-IoU conflict resolution.
//! - [`paa`]: **PAA** (Probabilistic Anchor Assignment, Kim & Lee ECCV 2020) — multi-scale candidate
//!   selection, joint classification + localization anchor loss, and deterministic 2-component 1D Gaussian
//!   Mixture EM separation ([`fit_gmm_1d_two_component`]).
//! - [`simota`]: **SimOTA** (Simplified Optimal Transport Assignment, Ge et al. YOLOX 2021) — dynamic-`k`
//!   assignment using pairwise classification + IoU regression costs, optional in-box and center-prior
//!   spatial gating, top-`q` IoU dynamic-`k` estimation, and lowest-cost conflict resolution with
//!   deterministic tie-breaking.
//! - [`hungarian`]: Exact `O(min(N, M)² · max(N, M))` Hungarian / linear-sum assignment
//!   ([`hungarian_assign`]) and [`OneToOneMatcher`] combining classification + L1 + GIoU/CIoU costs for
//!   NMS-free detection heads.

pub mod atss;
pub mod hungarian;
pub mod paa;
pub mod simota;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_atss_paa;
#[cfg(test)]
mod tests_hungarian;

pub use atss::{AtssAssignment, AtssConfig, anchor_boxes_from_points, assign_atss};
pub use hungarian::{
    ClsCostKind, HungarianResult, OneToOneAssignment, OneToOneConfig, OneToOneMatcher, hungarian_assign,
    hungarian_assign_f32,
};
pub use paa::{Gmm1dResult, PaaAssignment, PaaConfig, assign_paa, fit_gmm_1d_two_component};
pub use simota::{SimOtaAssignment, SimOtaConfig, assign_simota};
