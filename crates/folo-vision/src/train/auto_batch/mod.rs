//! GPU VRAM-budget auto-batch-size prober and OOM-safe micro-batch planner (`folo-gpu` `MemoryBudget`
//! integration).
//!
//! Separates pure, deterministic planning ([`estimate_training_footprint`], [`auto_batch_plan`]) from live GPU
//! queries ([`probe_batch_size`]), so callers (the [`super::Trainer`], CLI, server, Studio, Desktop, and unit
//! tests) can evaluate safe physical micro-batch sizes against both dedicated VRAM budgets and per-buffer
//! `wgpu::Limits` (`max_buffer_size`, `max_storage_buffer_binding_size`), pair them cleanly with gradient
//! accumulation (`accumulate`) and epoch validation batching (`chosen_val_batch`), and trigger
//! `GpuContext::release_memory()` when cached pool buffers need to be reclaimed.

mod footprint;

use std::fmt;

use folo_core::Device;
use folo_gpu::{AllocationKind, HeadroomBlocker, HeadroomEstimate, MemoryBudget};
use folo_nn::optim::{OptimKind, accumulate_steps};
use serde::{Deserialize, Serialize};

use footprint::{FIXED_WORKSPACE_BYTES, estimate_footprint_details, max_wgrad_scratch_bytes};
pub use footprint::{MemoryFootprint, estimate_training_footprint};

use super::TrainConfig;
use crate::error::{Error, Result};
use crate::model::HeadMode;

/// Default target VRAM utilization fraction when auto-batch sizing is enabled (`80%`, reserving `20%` safety
/// headroom for driver, OS, and transient allocator slack).
pub const DEFAULT_AUTO_BATCH_UTILIZATION: f64 = 0.80;
/// Lowest valid [`TrainConfig::auto_batch_target`] / [`AutoBatchRequest::target_utilization`].
pub const MIN_AUTO_BATCH_UTILIZATION: f64 = 0.25;
/// Highest valid [`TrainConfig::auto_batch_target`] / [`AutoBatchRequest::target_utilization`].
pub const MAX_AUTO_BATCH_UTILIZATION: f64 = 0.95;
/// Maximum physical micro-batch size allowed by [`TrainConfig::validate`].
pub const MAX_AUTO_BATCH: usize = 1024;
/// Upper bound on `num_classes` accepted by [`AutoBatchRequest::validate`].
pub const MAX_AUTO_BATCH_CLASSES: usize = 1_000_000;
/// Maximum validation batch size allowed by [`TrainConfig::validate`].
const MAX_VAL_BATCH: usize = 4096;

/// Inputs describing a training run for [`estimate_training_footprint`], [`auto_batch_plan`], and
/// [`probe_batch_size`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoBatchRequest {
    /// Detector architecture (`"folo-n"` … `"folo-x"`, `"folo2-n"` … `"folo2-x"`, or scale letter `"n"` … `"x"`).
    pub arch: String,
    /// Square training image size (multiple of 32 in `64..=4096`).
    pub imgsz: u32,
    /// Number of detection classes (`1..=1_000_000`).
    pub num_classes: usize,
    /// Detection head (`o2m` or dual `o2o`).
    pub head: HeadMode,
    /// Mixed-precision (`F16` activations / `F32` master weights, BN stats, softmax, loss, optimizer).
    pub amp: bool,
    /// Whether an EMA shadow model is kept on the device.
    pub ema: bool,
    /// Resolved optimizer kind (`Sgd`: 1× state; `MuSgd`: 1× SGD momentum + 1× Muon buffer on conv weights;
    /// `AdamW`: 2× state).
    pub optimizer: OptimKind,
    /// User-requested physical micro-batch size (`1..=1024`).
    pub requested_batch: usize,
    /// Nominal effective batch size (`nbs`, `>= 1`) used for gradient accumulation.
    pub nominal_batch: usize,
    /// Target fraction of dedicated VRAM to stay within (default `0.80`, valid in `0.25..=0.95`).
    pub target_utilization: f64,
    /// Optional upper cap on the chosen physical micro-batch (for example, `Some(n_train)` on small datasets).
    /// When `None`, defaults to `requested_batch.max(nominal_batch).min(MAX_AUTO_BATCH)`.
    pub max_batch: Option<usize>,
    /// Requested validation batch size (`0` = default `2 * chosen_batch`, clamped against `F32` eval limits).
    #[serde(default)]
    pub val_batch: usize,
    /// Optional frozen distillation teacher architecture (`"folo-n"` … `"folo-x"`, `"folo2-n"` … `"folo2-x"`, or
    /// scale letter `"n"` … `"x"`).
    pub teacher_arch: Option<String>,
    /// Detection head of the frozen distillation teacher (`o2m` by default; `o2o` includes both resident branches).
    #[serde(default)]
    pub teacher_head: HeadMode,
    /// Optional exact frozen teacher parameter count (overrides the analytic count for `teacher_arch` when set).
    pub teacher_params: Option<usize>,
}

impl AutoBatchRequest {
    /// Build a probe request from a validated [`TrainConfig`], class count, training split size, and optional
    /// distillation teacher info (`scale`, `head`, `params`).
    pub fn from_train_config(
        cfg: &TrainConfig,
        num_classes: usize,
        n_train: usize,
        teacher_arch: Option<String>,
        teacher_head: Option<HeadMode>,
        teacher_params: Option<usize>,
    ) -> Result<Self> {
        let resolved = cfg.resolve_optimizer(n_train, num_classes);
        let orig_eff = cfg.batch.saturating_mul(cfg.accumulate());
        let cap = orig_eff.min(n_train.max(1));
        let req = AutoBatchRequest {
            arch: cfg.arch.clone(),
            imgsz: cfg.imgsz,
            num_classes,
            head: cfg.head,
            amp: cfg.amp,
            ema: cfg.ema,
            optimizer: resolved.kind,
            requested_batch: cfg.batch,
            nominal_batch: cfg.nbs,
            target_utilization: cfg.auto_batch_target,
            max_batch: Some(cap),
            val_batch: cfg.val_batch,
            teacher_arch,
            teacher_head: teacher_head.unwrap_or(HeadMode::O2m),
            teacher_params,
        };
        req.validate()?;
        Ok(req)
    }

    /// Validate every request field before estimating memory or planning batch sizes.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(Error::Config(format!("auto_batch: {msg}")));
        parse_scale(&self.arch)?;
        if !self.imgsz.is_multiple_of(32) || !(64..=4096).contains(&self.imgsz) {
            return bad(format!("imgsz must be a multiple of 32 in 64..=4096, got {}", self.imgsz));
        }
        if !(1..=MAX_AUTO_BATCH_CLASSES).contains(&self.num_classes) {
            return bad(format!(
                "num_classes must be in 1..={MAX_AUTO_BATCH_CLASSES}, got {}",
                self.num_classes
            ));
        }
        if !(1..=MAX_AUTO_BATCH).contains(&self.requested_batch) {
            return bad(format!(
                "requested_batch must be in 1..={MAX_AUTO_BATCH}, got {}",
                self.requested_batch
            ));
        }
        if self.nominal_batch == 0 {
            return bad("nominal_batch must be at least 1".into());
        }
        if !(self.target_utilization.is_finite()
            && (MIN_AUTO_BATCH_UTILIZATION..=MAX_AUTO_BATCH_UTILIZATION).contains(&self.target_utilization))
        {
            return bad(format!(
                "target_utilization must be finite in {MIN_AUTO_BATCH_UTILIZATION}..={MAX_AUTO_BATCH_UTILIZATION}, \
                 got {}",
                self.target_utilization
            ));
        }
        if let Some(m) = self.max_batch
            && m == 0
        {
            return bad("max_batch must be at least 1 when set".into());
        }
        if self.val_batch > MAX_VAL_BATCH {
            return bad(format!("val_batch must be at most {MAX_VAL_BATCH}, got {}", self.val_batch));
        }
        if let Some(t) = &self.teacher_arch {
            parse_scale(t)?;
        }
        Ok(())
    }
}

/// Which constraint determined [`AutoBatchPlan::chosen_batch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchLimitFactor {
    /// Hardware headroom exceeded the target batch; kept `max(requested_batch, nominal_batch)`.
    RequestedBatch,
    /// Capped by `max_batch` (such as the training dataset size).
    MaxBatchCap,
    /// Capped by [`MAX_AUTO_BATCH`] (`1024`).
    ConfigMaxBatch,
    /// Limited by remaining usable dedicated VRAM under `target_utilization`.
    VramBudget,
    /// Limited by `wgpu::Limits::max_storage_buffer_binding_size` for a single tensor or kernel workspace.
    StorageBindingLimit,
    /// Limited by `wgpu::Limits::max_buffer_size` for a single tensor or kernel workspace.
    MaxBufferSize,
    /// Dedicated VRAM is unknown on this backend and single-buffer limits did not cap `requested_batch`.
    DeviceUnbounded,
}

impl fmt::Display for BatchLimitFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RequestedBatch => "requested/nominal batch",
            Self::MaxBatchCap => "max_batch cap",
            Self::ConfigMaxBatch => "config max batch (1024)",
            Self::VramBudget => "VRAM budget",
            Self::StorageBindingLimit => "max_storage_buffer_binding_size",
            Self::MaxBufferSize => "max_buffer_size",
            Self::DeviceUnbounded => "device VRAM unknown (kept requested batch)",
        })
    }
}

/// Serializable summary of `folo_gpu::HeadroomEstimate` for the planned micro-batch's peak single storage buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadroomSummary {
    /// Largest single storage allocation requested for the planned micro-batch (or batch 1 when `!fits`),
    /// covering both model tensors and convolution weight-gradient scratch.
    pub requested_bytes: u64,
    /// Pool-aligned allocation size, or `None` if it exceeds the single-buffer limit.
    pub aligned_bytes: Option<u64>,
    /// Effective storage buffer limit (`min(max_buffer_size, max_storage_buffer_binding_size)`).
    pub single_buffer_limit: u64,
    /// Whether this single-buffer allocation fits.
    pub fits: bool,
    /// Whether reclaiming cached pool buffers is needed for headroom.
    pub needs_reclaim: bool,
    /// Human-readable blocker name if `!fits`.
    pub blocker: Option<String>,
}

impl HeadroomSummary {
    fn from_estimate(est: HeadroomEstimate, needs_reclaim: bool) -> Self {
        let blocker = est.blocker.map(|b| {
            match b {
                HeadroomBlocker::ExceedsStorageBindingLimit => "exceeds_storage_buffer_binding_size",
                HeadroomBlocker::ExceedsMaxBufferSize => "exceeds_max_buffer_size",
                HeadroomBlocker::ExceedsDeviceMemory => "exceeds_device_memory",
            }
            .to_string()
        });
        HeadroomSummary {
            requested_bytes: est.requested_bytes,
            aligned_bytes: est.aligned_bytes,
            single_buffer_limit: est.single_buffer_limit,
            fits: est.fits,
            needs_reclaim: est.needs_reclaim || needs_reclaim,
            blocker,
        }
    }
}

/// Structured auto-batch sizing plan and VRAM headroom diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoBatchPlan {
    /// Originally requested micro-batch size.
    pub requested_batch: usize,
    /// Original resolved effective batch (`requested_batch * accumulate_steps(nominal_batch, requested_batch)`)
    /// targeted for schedule preservation.
    pub target_effective_batch: usize,
    /// Chosen safe physical micro-batch size (`0` when `!fits`), rounded down to a divisor of
    /// `target_effective_batch` when hardware-limited so effective batch size and optimizer steps per epoch stay
    /// consistent.
    pub chosen_batch: usize,
    /// Chosen safe epoch-validation batch size (`0` when `!fits`), clamped against `F32` eval buffer and VRAM
    /// limits so epoch-end validation does not overflow after training succeeds.
    pub chosen_val_batch: usize,
    /// Gradient accumulation steps paired with `chosen_batch` (`0` when `!fits`).
    pub grad_accum: usize,
    /// Effective nominal batch size (`nbs`) paired with `chosen_batch` to keep effective batch & LR schedule
    /// consistent when `chosen_batch < requested_batch`.
    pub nominal_batch: usize,
    /// Resulting effective batch size per optimizer step (`chosen_batch * grad_accum`).
    pub effective_batch: usize,
    /// Raw maximum batch size that fits hardware limits (`None` when unbounded).
    pub max_fit_batch: Option<usize>,
    /// Constraint that bounded `chosen_batch`.
    pub limiting_factor: BatchLimitFactor,
    /// Target VRAM utilization fraction used (`reserve_fraction = 1 - target_utilization`).
    pub target_utilization: f64,
    /// Adapter's dedicated VRAM total in bytes, when reported by the backend.
    pub dedicated_vram_bytes: Option<u64>,
    /// Usable VRAM bytes after subtracting the `(1 - target_utilization)` safety reserve.
    pub usable_vram_bytes: Option<u64>,
    /// Estimated free VRAM available for new live allocations before placing this run.
    pub free_vram_bytes: Option<u64>,
    /// Effective per-buffer storage limit (`min(max_buffer_size, max_storage_buffer_binding_size)`).
    pub single_buffer_limit_bytes: u64,
    /// Adapter's `wgpu::Limits::max_buffer_size` in bytes.
    pub max_buffer_size_bytes: u64,
    /// Estimated fixed resident bytes for this run.
    pub fixed_bytes: u64,
    /// Estimated per-sample working bytes (including pool overhead).
    pub per_sample_bytes: u64,
    /// Largest single storage tensor per sample in bytes.
    pub max_single_tensor_per_sample_bytes: u64,
    /// Estimated peak device bytes (`fixed_bytes + chosen_batch * per_sample_bytes` plus any excess wgrad scratch,
    /// or batch 1 if `!fits`).
    pub estimated_peak_bytes: u64,
    /// True when cached pool/plan/anchor buffers should be reclaimed (`GpuContext::release_memory`) before
    /// placing the run to avoid evicting or spiking past usable VRAM.
    pub needs_reclaim: bool,
    /// Reclaimable cached bytes currently held on the device.
    pub reclaimable_bytes: u64,
    /// Whether at least batch 1 fits within both VRAM and per-buffer limits.
    pub fits: bool,
    /// Headroom details for the largest single storage buffer of the planned batch.
    pub headroom: HeadroomSummary,
}

impl AutoBatchPlan {
    /// Concise, actionable diagnostic line suitable for training logs and CLI/UI surfaces.
    pub fn summary_line(&self) -> String {
        let vram = match (self.dedicated_vram_bytes, self.free_vram_bytes) {
            (Some(total), Some(free)) => format!(
                ", VRAM {} total / {} est. free (target {:.0}%)",
                fmt_bytes(total),
                fmt_bytes(free),
                self.target_utilization * 100.0
            ),
            _ => String::new(),
        };
        if !self.fits {
            return format!(
                "batch 1 does not fit (limited by {}: est. fixed {}, per-sample {}, peak {}, single-buffer limit \
                 {}{vram})",
                self.limiting_factor,
                fmt_bytes(self.fixed_bytes),
                fmt_bytes(self.per_sample_bytes),
                fmt_bytes(self.estimated_peak_bytes),
                fmt_bytes(self.single_buffer_limit_bytes),
            );
        }
        let reclaim = if self.needs_reclaim {
            format!("; reclaiming {} cache", fmt_bytes(self.reclaimable_bytes))
        } else {
            String::new()
        };
        let drift = if self.effective_batch != self.target_effective_batch {
            format!(" [target effective {}]", self.target_effective_batch)
        } else {
            String::new()
        };
        format!(
            "auto-batch: batch {} × grad_accum {} = {} effective{drift}, val_batch {} (requested {}, limited by \
             {}; est. {} fixed + {}/sample = {} peak, buffer limit {}{vram}{reclaim})",
            self.chosen_batch,
            self.grad_accum,
            self.effective_batch,
            self.chosen_val_batch,
            self.requested_batch,
            self.limiting_factor,
            fmt_bytes(self.fixed_bytes),
            fmt_bytes(self.per_sample_bytes),
            fmt_bytes(self.estimated_peak_bytes),
            fmt_bytes(self.single_buffer_limit_bytes),
        )
    }
}

/// Probe a live device's `folo-gpu` [`MemoryBudget`] and compute an [`AutoBatchPlan`].
///
/// Returns `Ok(None)` when `device` is not backed by a `folo-gpu` context (for example `Device::Cpu`).
pub fn probe_batch_size(device: Device, req: &AutoBatchRequest) -> Result<Option<AutoBatchPlan>> {
    req.validate()?;
    let Some(ctx) = folo_gpu::context_for(device) else {
        return Ok(None);
    };
    let budget = ctx.memory_budget();
    Ok(Some(auto_batch_plan(req, &budget)?))
}

/// Deterministic auto-batch planner over any [`MemoryBudget`] (live GPU or mock limits/usage in tests).
///
/// Respects BOTH dedicated VRAM budget (with safety reserve `1.0 - req.target_utilization`) and per-buffer
/// `wgpu::Limits` (`max_storage_buffer_binding_size` and `max_buffer_size`, covering model tensors and conv
/// weight-gradient scratch), chooses a divisor-compatible micro-batch when hardware-limited so effective batch
/// size and LR schedule stay consistent, and caps epoch validation batching (`chosen_val_batch`) against `F32`
/// eval limits.
pub fn auto_batch_plan(req: &AutoBatchRequest, budget: &MemoryBudget) -> Result<AutoBatchPlan> {
    req.validate()?;
    let details = estimate_footprint_details(req)?;
    let fp = details.fp;
    let budgeted = budget.with_reserve_fraction(1.0 - req.target_utilization);
    let storage_limit = budgeted.limits.single_buffer_limit(AllocationKind::Storage);

    let wgrad_for = |b: usize| max_wgrad_scratch_bytes(&details.convs, b, req.amp);
    let single_req_for = |b: usize| {
        (b as u64)
            .saturating_mul(fp.max_single_tensor_per_sample)
            .max(wgrad_for(b))
            .max(details.max_fixed_tensor_bytes)
    };
    let peak_for = |b: usize| {
        let wgrad_extra = wgrad_for(b).saturating_sub(FIXED_WORKSPACE_BYTES);
        fp.fixed_bytes
            .saturating_add((b as u64).saturating_mul(fp.per_sample_bytes))
            .saturating_add(wgrad_extra)
    };
    let fits_single_buf =
        |b: usize| budgeted.aligned_bytes(single_req_for(b), AllocationKind::Storage).is_some();
    let fits_vram = |b: usize| budgeted.free_vram_bytes().is_none_or(|free| peak_for(b) <= free);
    let fits_both = |b: usize| fits_vram(b) && fits_single_buf(b);

    // Original resolved effective batch targeted for schedule & LR preservation (e.g. `batch=24, nbs=64` ->
    // `acc=3`, `target_effective=72`).
    let orig_acc = accumulate_steps(req.nominal_batch, req.requested_batch);
    let target_effective_batch = req.requested_batch.saturating_mul(orig_acc);
    let default_target = target_effective_batch.min(MAX_AUTO_BATCH);
    let cap = match req.max_batch {
        Some(m) => m.min(MAX_AUTO_BATCH),
        None => default_target,
    };

    // Base analytic ceilings from `MemoryBudget::estimate_max_batch`, then check the exact combined predicate
    // (`fits_both`) at `target_cap` and downward so non-monotonic conv weight-gradient split-K scratch or large
    // fixed weight tensors can never sneak past a higher fitting batch.
    let raw_by_vram = budgeted.estimate_max_batch(fp.fixed_bytes, fp.per_sample_bytes, 0);
    let raw_by_buf = budgeted.estimate_max_batch(0, 0, fp.max_single_tensor_per_sample);
    let raw_hw_cap = match (raw_by_vram, raw_by_buf) {
        (Some(v), Some(b)) => Some(v.min(b)),
        (Some(v), None) => Some(v),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let target_cap =
        if budgeted.limits.dedicated_bytes.is_none() { req.requested_batch.min(cap) } else { cap };
    let search_upper = raw_hw_cap.map_or(target_cap, |hw| hw.min(target_cap));
    let mut best_in_target = search_upper;
    while best_in_target > 0 && !fits_both(best_in_target) {
        best_in_target -= 1;
    }

    let classify_fail = |fail_b: usize| -> BatchLimitFactor {
        if !fits_single_buf(fail_b) {
            buffer_limit_factor(budgeted, single_req_for(fail_b))
        } else {
            BatchLimitFactor::VramBudget
        }
    };

    let (raw_chosen, mut limiting_factor, mut fits, hardware_limited) = if best_in_target == 0 {
        (0, classify_fail(1), false, true)
    } else if best_in_target < target_cap {
        let fail_b = best_in_target.saturating_add(1).min(target_cap);
        (best_in_target, classify_fail(fail_b), true, true)
    } else if budgeted.limits.dedicated_bytes.is_none() {
        let factor = if target_cap < req.requested_batch {
            BatchLimitFactor::MaxBatchCap
        } else {
            BatchLimitFactor::DeviceUnbounded
        };
        (target_cap, factor, true, false)
    } else {
        let factor = if req.max_batch.unwrap_or(target_effective_batch) > MAX_AUTO_BATCH {
            BatchLimitFactor::ConfigMaxBatch
        } else if req.max_batch.is_some_and(|m| m != default_target && m <= MAX_AUTO_BATCH) {
            BatchLimitFactor::MaxBatchCap
        } else {
            BatchLimitFactor::RequestedBatch
        };
        (target_cap, factor, true, false)
    };

    let mut max_fit_batch = if !fits {
        Some(0)
    } else if hardware_limited {
        Some(raw_chosen)
    } else {
        raw_hw_cap.map(|hw| hw.max(raw_chosen))
    };

    // When hardware-limited below `target_effective_batch`, round `chosen_batch` down to a compatible micro-batch
    // (exact divisor when within 2× of `raw_chosen`) so `chosen_batch * grad_accum` and optimizer steps/epoch stay
    // faithful to the configured schedule.
    let (mut chosen_batch, mut nominal_batch, mut grad_accum, mut effective_batch) = if fits {
        let chosen = if hardware_limited && raw_chosen < target_effective_batch {
            compatible_micro_batch(raw_chosen, target_effective_batch, fits_both)
        } else {
            raw_chosen
        };
        let keep_orig_nbs =
            chosen.saturating_mul(accumulate_steps(req.nominal_batch, chosen)) == target_effective_batch;
        let nominal = if keep_orig_nbs { req.nominal_batch } else { target_effective_batch };
        let acc = accumulate_steps(nominal, chosen);
        (chosen, nominal, acc, chosen.saturating_mul(acc))
    } else {
        (0, req.nominal_batch, 0, 0)
    };

    // Plan epoch-validation batch size against `F32` eval buffer & VRAM limits (the default `val_batch = 0` runs
    // `2 * batch` in `F32`). If even validation batch 1 cannot fit (e.g. AMP training fit F16 under a binding
    // limit that F32 validation exceeds), report `!fits` up front rather than failing after epoch 1.
    let eval_single_1 = details.eval_max_single_tensor.max(details.max_fixed_tensor_bytes);
    let max_val_by_buf = if budgeted.aligned_bytes(eval_single_1, AllocationKind::Storage).is_none() {
        Some(0)
    } else {
        budgeted.estimate_max_batch(0, 0, details.eval_max_single_tensor)
    };
    let max_val_by_vram = if budgeted.limits.dedicated_bytes.is_some() {
        budgeted.estimate_max_batch(details.eval_fixed_bytes, details.eval_per_sample_bytes, 0)
    } else {
        None
    };
    let max_val_fit = match (max_val_by_vram, max_val_by_buf) {
        (Some(v), Some(b)) => Some(v.min(b)),
        (Some(v), None) => Some(v),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let mut val_blocked = false;
    let chosen_val_batch = if fits {
        let desired_val =
            if req.val_batch > 0 { req.val_batch } else { chosen_batch.saturating_mul(2) }.min(MAX_VAL_BATCH);
        match max_val_fit {
            Some(0) => {
                fits = false;
                val_blocked = true;
                chosen_batch = 0;
                grad_accum = 0;
                effective_batch = 0;
                nominal_batch = req.nominal_batch;
                max_fit_batch = Some(0);
                limiting_factor = if max_val_by_buf == Some(0) {
                    buffer_limit_factor(budgeted, eval_single_1)
                } else {
                    BatchLimitFactor::VramBudget
                };
                0
            }
            Some(m) => desired_val.min(m),
            None => desired_val,
        }
    } else {
        0
    };

    let probe_b = chosen_batch.max(1);
    let estimated_peak_bytes = if val_blocked {
        peak_for(1).max(details.eval_fixed_bytes.saturating_add(details.eval_per_sample_bytes))
    } else {
        peak_for(probe_b)
    };
    let single_req_bytes =
        if val_blocked { single_req_for(1).max(eval_single_1) } else { single_req_for(probe_b) };
    let single_est = budgeted.estimate_headroom(single_req_bytes, AllocationKind::Storage);

    let reclaimable_bytes = budgeted.usage.reclaimable_bytes();
    let total_needs_reclaim = fits
        && reclaimable_bytes > 0
        && budgeted.free_without_reclaim_bytes().is_some_and(|avail| estimated_peak_bytes > avail);
    let needs_reclaim = fits && (single_est.needs_reclaim || total_needs_reclaim);

    Ok(AutoBatchPlan {
        requested_batch: req.requested_batch,
        target_effective_batch,
        chosen_batch,
        chosen_val_batch,
        grad_accum,
        nominal_batch,
        effective_batch,
        max_fit_batch,
        limiting_factor,
        target_utilization: req.target_utilization,
        dedicated_vram_bytes: budgeted.limits.dedicated_bytes,
        usable_vram_bytes: budgeted.usable_vram_bytes(),
        free_vram_bytes: budgeted.free_vram_bytes(),
        single_buffer_limit_bytes: storage_limit,
        max_buffer_size_bytes: budgeted.limits.max_buffer_size,
        fixed_bytes: fp.fixed_bytes,
        per_sample_bytes: fp.per_sample_bytes,
        max_single_tensor_per_sample_bytes: fp.max_single_tensor_per_sample,
        estimated_peak_bytes,
        needs_reclaim,
        reclaimable_bytes,
        fits,
        headroom: HeadroomSummary::from_estimate(single_est, needs_reclaim),
    })
}

/// Chooses the largest micro-batch `b <= max_b` that preserves `nominal` cleanly under `accumulate_steps`:
/// 1. Prefers the largest exact divisor of `nominal` in `(max_b / 2).max(1)..=max_b` (0% effective-batch drift,
///    >= 50% of hardware batch utilization).
/// 2. Otherwise (e.g. prime `nominal`), picks `b` in `(max_b / 2).max(1)..=max_b` minimizing
///    `|b * accumulate_steps(nominal, b) - nominal|`, breaking ties by larger `b`.
fn compatible_micro_batch(max_b: usize, nominal: usize, mut fits: impl FnMut(usize) -> bool) -> usize {
    let lo = (max_b / 2).max(1);
    for b in (lo..=max_b).rev() {
        if nominal.is_multiple_of(b) && fits(b) {
            return b;
        }
    }
    let mut best = max_b;
    let mut best_err = usize::MAX;
    for b in (lo..=max_b).rev() {
        if !fits(b) {
            continue;
        }
        let eff = b.saturating_mul(accumulate_steps(nominal, b));
        let err = eff.abs_diff(nominal);
        if err < best_err {
            best_err = err;
            best = b;
            if err == 0 {
                break;
            }
        }
    }
    best
}

fn buffer_limit_factor(budget: MemoryBudget, over_bytes: u64) -> BatchLimitFactor {
    match budget.estimate_headroom(over_bytes, AllocationKind::Storage).blocker {
        Some(HeadroomBlocker::ExceedsStorageBindingLimit) => BatchLimitFactor::StorageBindingLimit,
        Some(HeadroomBlocker::ExceedsMaxBufferSize) => BatchLimitFactor::MaxBufferSize,
        _ => {
            if budget.limits.max_storage_buffer_binding_size < budget.limits.max_buffer_size {
                BatchLimitFactor::StorageBindingLimit
            } else {
                BatchLimitFactor::MaxBufferSize
            }
        }
    }
}

pub(super) fn parse_arch(arch: &str) -> Result<(bool, &str)> {
    let (is_v2, s) = if let Some(rest) = arch.strip_prefix("folo2-") {
        (true, rest)
    } else if let Some(rest) = arch.strip_prefix("folo-") {
        (false, rest)
    } else {
        (false, arch)
    };
    match s {
        "n" | "s" | "m" | "l" | "x" => Ok((is_v2, s)),
        _ => Err(Error::Config(format!("arch `{arch}` is not one of folo-n..x or folo2-n..x"))),
    }
}

pub(super) fn parse_scale(arch: &str) -> Result<&str> {
    parse_arch(arch).map(|(_, s)| s)
}

fn fmt_bytes(bytes: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}
