//! GPU VRAM-budget auto-batch-size prober and OOM-safe micro-batch planner (`folo-gpu` `MemoryBudget`
//! integration).
//!
//! Separates pure, deterministic planning ([`estimate_training_footprint`], [`auto_batch_plan`]) from live GPU
//! queries ([`probe_batch_size`]), so callers (the [`super::Trainer`], CLI, server, Studio, Desktop, and unit
//! tests) can evaluate safe physical micro-batch sizes against both dedicated VRAM budgets and per-buffer
//! `wgpu::Limits` (`max_buffer_size`, `max_storage_buffer_binding_size`), pair them cleanly with gradient
//! accumulation (`accumulate`), and trigger `GpuContext::release_memory()` when cached pool buffers need to be
//! reclaimed.

use std::fmt;

use folo_core::Device;
use folo_gpu::{AllocationKind, HeadroomBlocker, HeadroomEstimate, MemoryBudget};
use folo_nn::io::ModelConfig;
use folo_nn::optim::{OptimKind, accumulate_steps};
use serde::{Deserialize, Serialize};

use super::TrainConfig;
use crate::error::{Error, Result};
use crate::model::{HeadMode, Scaling};

/// Default target VRAM utilization fraction when auto-batch sizing is enabled (`80%`, reserving `20%` safety
/// headroom for driver, OS, and transient allocator slack).
pub const DEFAULT_AUTO_BATCH_UTILIZATION: f64 = 0.80;
/// Lowest valid [`TrainConfig::auto_batch_target`] / [`AutoBatchRequest::target_utilization`].
pub const MIN_AUTO_BATCH_UTILIZATION: f64 = 0.25;
/// Highest valid [`TrainConfig::auto_batch_target`] / [`AutoBatchRequest::target_utilization`].
pub const MAX_AUTO_BATCH_UTILIZATION: f64 = 0.95;
/// Maximum physical micro-batch size allowed by [`TrainConfig::validate`].
pub const MAX_AUTO_BATCH: usize = 1024;

/// Pool size-class rounding overhead factor (power-of-2 / 25% buckets in `folo-gpu::pool::size_class`).
const POOL_OVERHEAD_NUM: u64 = 5;
const POOL_OVERHEAD_DEN: u64 = 4;
/// Fixed baseline device workspace (cached anchor tables, uniform arenas, kernel scratch).
const FIXED_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;

/// Inputs describing a training run for [`estimate_training_footprint`], [`auto_batch_plan`], and
/// [`probe_batch_size`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoBatchRequest {
    /// Detector architecture (`"folo-n"` … `"folo-x"`, or scale letter `"n"` … `"x"`).
    pub arch: String,
    /// Square training image size (multiple of 32 in `64..=4096`).
    pub imgsz: u32,
    /// Number of detection classes (`>= 1`).
    pub num_classes: usize,
    /// Detection head (`o2m` or dual `o2o`).
    pub head: HeadMode,
    /// Mixed-precision (`F16` activations / `F32` master weights, BN, loss, optimizer).
    pub amp: bool,
    /// Whether an EMA shadow model is kept on the device.
    pub ema: bool,
    /// Resolved optimizer kind (determines per-parameter state buffers: 1× for SGD/MuSGD, 2× for AdamW).
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
    /// Optional frozen distillation teacher architecture (`"folo-n"` … `"folo-x"`).
    pub teacher_arch: Option<String>,
    /// Optional exact frozen teacher parameter count (overrides the analytic count for `teacher_arch` when set).
    pub teacher_params: Option<usize>,
}

impl AutoBatchRequest {
    /// Build a probe request from a validated [`TrainConfig`], class count, training split size, and optional
    /// distillation teacher info.
    pub fn from_train_config(
        cfg: &TrainConfig,
        num_classes: usize,
        n_train: usize,
        teacher_arch: Option<String>,
        teacher_params: Option<usize>,
    ) -> Result<Self> {
        let resolved = cfg.resolve_optimizer(n_train, num_classes);
        let cap = cfg.batch.max(cfg.nbs).min(n_train.max(1)).min(MAX_AUTO_BATCH);
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
            teacher_arch,
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
        if self.num_classes == 0 {
            return bad("num_classes must be at least 1".into());
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
        if let Some(t) = &self.teacher_arch {
            parse_scale(t)?;
        }
        Ok(())
    }
}

/// Analytic GPU training memory footprint for a [`FoloModel`] configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFootprint {
    /// Exact trainable parameter count of the student `FoloModel`.
    pub model_params: usize,
    /// Fixed resident bytes (weights, BN buffers, grad accumulator, optimizer state, optional EMA & teacher,
    /// plus pool overhead and fixed workspace).
    pub fixed_bytes: u64,
    /// Estimated per-sample activation, autograd backward, loss/assigner, and optional teacher output bytes
    /// (including 25% `folo-gpu` pool size-class rounding overhead).
    pub per_sample_bytes: u64,
    /// Largest single GPU storage buffer per sample (in bytes), checked against
    /// `min(max_buffer_size, max_storage_buffer_binding_size)`.
    pub max_single_tensor_per_sample: u64,
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
    /// Limited by `wgpu::Limits::max_storage_buffer_binding_size` for a single per-batch tensor.
    StorageBindingLimit,
    /// Limited by `wgpu::Limits::max_buffer_size` for a single per-batch tensor.
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

/// Serializable summary of `folo_gpu::HeadroomEstimate` for the planned micro-batch's peak single tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadroomSummary {
    /// Largest single storage allocation requested for the planned micro-batch (or batch 1 when `!fits`).
    pub requested_bytes: u64,
    /// Pool-aligned allocation size, or `None` if it exceeds the single-buffer limit.
    pub aligned_bytes: Option<u64>,
    /// Effective storage buffer limit (`min(max_buffer_size, max_storage_buffer_binding_size)`).
    pub single_buffer_limit: u64,
    /// Whether this single-tensor allocation fits.
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
    /// Chosen safe physical micro-batch size (`0` when `!fits`).
    pub chosen_batch: usize,
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
    /// Estimated peak device bytes (`fixed_bytes + chosen_batch * per_sample_bytes`, or batch 1 if `!fits`).
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
        format!(
            "auto-batch: batch {} × grad_accum {} = {} effective (requested {}, limited by {}; est. {} fixed + \
             {}/sample = {} peak, buffer limit {}{vram}{reclaim})",
            self.chosen_batch,
            self.grad_accum,
            self.effective_batch,
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
/// `wgpu::Limits` (`max_storage_buffer_binding_size` and `max_buffer_size`), pairing any clamped micro-batch
/// with gradient accumulation so effective batch size and LR schedule remain consistent.
pub fn auto_batch_plan(req: &AutoBatchRequest, budget: &MemoryBudget) -> Result<AutoBatchPlan> {
    req.validate()?;
    let fp = estimate_training_footprint(req)?;
    let budgeted = budget.with_reserve_fraction(1.0 - req.target_utilization);

    let max_fit =
        budgeted.estimate_max_batch(fp.fixed_bytes, fp.per_sample_bytes, fp.max_single_tensor_per_sample);
    let max_by_vram = budgeted.estimate_max_batch(fp.fixed_bytes, fp.per_sample_bytes, 0);
    let max_by_buf = budgeted.estimate_max_batch(0, 0, fp.max_single_tensor_per_sample);

    let default_target = req.requested_batch.max(req.nominal_batch).min(MAX_AUTO_BATCH);
    let cap = match req.max_batch {
        Some(m) => m.min(MAX_AUTO_BATCH),
        None => default_target,
    };

    let storage_limit = budgeted.limits.single_buffer_limit(AllocationKind::Storage);
    let classify_hardware_limit = |vram_cap: Option<usize>, buf_cap: Option<usize>| -> BatchLimitFactor {
        match (vram_cap, buf_cap) {
            (Some(v), Some(b)) if b < v => {
                buffer_limit_factor(budgeted, fp.max_single_tensor_per_sample, b + 1)
            }
            (Some(_), _) => BatchLimitFactor::VramBudget,
            (None, Some(b)) => buffer_limit_factor(budgeted, fp.max_single_tensor_per_sample, b + 1),
            (None, None) => BatchLimitFactor::DeviceUnbounded,
        }
    };

    let (chosen_batch, limiting_factor, fits) = if budgeted.limits.dedicated_bytes.is_none() {
        // Without dedicated VRAM totals (e.g. DX12/Metal/WebGPU), never scale up past `requested_batch`,
        // but still clamp down cleanly if a single storage tensor exceeds device buffer limits.
        let target = req.requested_batch.min(cap);
        match max_by_buf {
            Some(0) => (0, buffer_limit_factor(budgeted, fp.max_single_tensor_per_sample, 1), false),
            Some(b) if b < target => {
                (b, buffer_limit_factor(budgeted, fp.max_single_tensor_per_sample, b + 1), true)
            }
            _ => (target, BatchLimitFactor::DeviceUnbounded, true),
        }
    } else {
        match max_fit {
            Some(0) => (0, classify_hardware_limit(max_by_vram, max_by_buf), false),
            Some(fit) if fit < cap => (fit, classify_hardware_limit(max_by_vram, max_by_buf), true),
            Some(_) => {
                let factor = if req.max_batch.is_some_and(|m| m < default_target && m <= MAX_AUTO_BATCH) {
                    BatchLimitFactor::MaxBatchCap
                } else if req.max_batch.unwrap_or(req.requested_batch.max(req.nominal_batch)) > MAX_AUTO_BATCH
                {
                    BatchLimitFactor::ConfigMaxBatch
                } else {
                    BatchLimitFactor::RequestedBatch
                };
                (cap, factor, true)
            }
            None => {
                let chosen = req.requested_batch.min(cap);
                (chosen, BatchLimitFactor::DeviceUnbounded, true)
            }
        }
    };

    let probe_b = chosen_batch.max(1) as u64;
    let estimated_peak_bytes = fp.fixed_bytes.saturating_add(probe_b.saturating_mul(fp.per_sample_bytes));
    let single_req_bytes = probe_b.saturating_mul(fp.max_single_tensor_per_sample);
    let single_est = budgeted.estimate_headroom(single_req_bytes, AllocationKind::Storage);

    let reclaimable_bytes = budgeted.usage.reclaimable_bytes();
    let total_needs_reclaim = fits
        && reclaimable_bytes > 0
        && budgeted.free_without_reclaim_bytes().is_some_and(|avail| estimated_peak_bytes > avail);
    let needs_reclaim = fits && (single_est.needs_reclaim || total_needs_reclaim);

    let (nominal_batch, grad_accum, effective_batch) = if fits {
        // Preserve the user's effective batch when clamping below `requested_batch` (e.g. `batch=128, nbs=64`
        // clamped to `batch=32` keeps `nominal_batch=128`, `grad_accum=4`).
        let nominal = if chosen_batch < req.requested_batch {
            req.nominal_batch.max(req.requested_batch)
        } else {
            req.nominal_batch
        };
        let acc = accumulate_steps(nominal, chosen_batch);
        (nominal, acc, chosen_batch.saturating_mul(acc))
    } else {
        (req.nominal_batch, 0, 0)
    };

    Ok(AutoBatchPlan {
        requested_batch: req.requested_batch,
        chosen_batch,
        grad_accum,
        nominal_batch,
        effective_batch,
        max_fit_batch: max_fit,
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

fn buffer_limit_factor(budget: MemoryBudget, per_sample_single: u64, over_batch: usize) -> BatchLimitFactor {
    let over_bytes = (over_batch.max(1) as u64).saturating_mul(per_sample_single);
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

/// Analytically estimate the GPU memory footprint (exact parameter count, fixed resident bytes, per-sample
/// activation/backward bytes, and largest single per-sample storage tensor) for a [`FoloModel`] training run.
///
/// Allocates no tensors and completes in microseconds, matching `FoloModel::with_rng`'s exact stage widths,
/// repeats, and head branches.
pub fn estimate_training_footprint(req: &AutoBatchRequest) -> Result<MemoryFootprint> {
    req.validate()?;
    let scale = parse_scale(&req.arch)?;
    let mut mconf = ModelConfig::new(scale, req.num_classes)?;
    mconf.head = req.head.as_str().into();
    let student = analyze_folo_model(&mconf, req.imgsz as usize, req.amp);

    let opt_state_mult: u64 = match req.optimizer {
        OptimKind::Sgd => 1,
        OptimKind::MuSgd => 1,
        OptimKind::AdamW => 2,
    };
    let params_bytes = (student.params as u64).saturating_mul(4);
    let buffers_bytes = (student.bn_buffers as u64).saturating_mul(4);
    // Live weights + BN buffers + grad accumulator + optimizer state + optional EMA shadow (weights + BN buffers).
    let mut fixed = params_bytes.saturating_mul(2 + opt_state_mult).saturating_add(buffers_bytes);
    if req.ema {
        fixed = fixed.saturating_add(params_bytes).saturating_add(buffers_bytes);
    }

    let mut per_sample = student.per_sample_bytes;
    let mut max_single = student.max_single_tensor;

    if req.teacher_arch.is_some() || req.teacher_params.is_some() {
        let teacher_elem: u64 = if req.amp { 2 } else { 4 };
        let teacher_params = match (req.teacher_params, &req.teacher_arch) {
            (Some(p), _) => p as u64,
            (None, Some(t_arch)) => {
                let ts = parse_scale(t_arch)?;
                let tconf = ModelConfig::new(ts, req.num_classes)?;
                analyze_folo_model(&tconf, req.imgsz as usize, req.amp).params as u64
            }
            (None, None) => 0,
        };
        // Fused frozen teacher weights (`F16` under AMP, `F32` otherwise).
        fixed = fixed.saturating_add(teacher_params.saturating_mul(teacher_elem));
        // Retained `TeacherBatch` raw outputs (`box_raw` + `cls_raw`) stay live across the student forward+backward.
        let anchors = total_anchors(req.imgsz as usize);
        let t_box = (4 * mconf.reg_max as u64).saturating_mul(anchors).saturating_mul(teacher_elem);
        let t_cls = (req.num_classes as u64).saturating_mul(anchors).saturating_mul(teacher_elem);
        per_sample = per_sample.saturating_add(t_box).saturating_add(t_cls);
        if let Some(t_arch) = &req.teacher_arch {
            let ts = parse_scale(t_arch)?;
            let tconf = ModelConfig::new(ts, req.num_classes)?;
            let t_stats = analyze_folo_model(&tconf, req.imgsz as usize, req.amp);
            max_single = max_single.max(t_stats.max_single_tensor);
        }
    }

    // Apply pool size-class rounding overhead (+25% per buffer) and fixed workspace floor.
    let fixed_bytes = fixed
        .saturating_mul(POOL_OVERHEAD_NUM)
        .div_ceil(POOL_OVERHEAD_DEN)
        .saturating_add(FIXED_WORKSPACE_BYTES);
    let per_sample_bytes = per_sample.saturating_mul(POOL_OVERHEAD_NUM).div_ceil(POOL_OVERHEAD_DEN);

    Ok(MemoryFootprint {
        model_params: student.params,
        fixed_bytes,
        per_sample_bytes,
        max_single_tensor_per_sample: max_single,
    })
}

struct ModelStats {
    amp: bool,
    params: usize,
    bn_buffers: usize,
    per_sample_bytes: u64,
    max_single_tensor: u64,
}

impl ModelStats {
    fn new(amp: bool) -> Self {
        ModelStats { amp, params: 0, bn_buffers: 0, per_sample_bytes: 0, max_single_tensor: 0 }
    }

    fn elem_bytes(&self) -> u64 {
        if self.amp { 2 } else { 4 }
    }

    fn record_single(&mut self, bytes: u64) {
        self.max_single_tensor = self.max_single_tensor.max(bytes);
    }

    /// Account for one `ConvBnAct(cin -> cout, k, s)` producing `[cout, h, w]` during training.
    fn conv_bn_act(&mut self, cin: usize, cout: usize, k: usize, groups: usize, h: usize, w: usize) {
        self.params += cout * (cin / groups) * k * k + 2 * cout; // conv weight + BN gamma/beta
        self.bn_buffers += 2 * cout; // running_mean + running_var
        let elems = (cout as u64) * (h as u64) * (w as u64);
        // Under AMP: conv out & SiLU activations are F16 (2 B), while streaming BN keeps F32 intermediate (4 B);
        // under F32: saved conv/BN/SiLU activations + backward grad buffer ≈ 3 * F32.
        let bytes_per_elem: u64 = if self.amp { 8 } else { 12 };
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(elems.saturating_mul(bytes_per_elem));
        // BatchNorm always binds F32 storage buffers even under AMP.
        self.record_single(elems.saturating_mul(4));
    }

    /// Account for a bare `Conv2d(cin -> cout, k)` with bias.
    fn conv2d_bias(&mut self, cin: usize, cout: usize, k: usize, h: usize, w: usize) {
        self.params += cout * cin * k * k + cout;
        let elems = (cout as u64) * (h as u64) * (w as u64);
        let elem = self.elem_bytes();
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(elems.saturating_mul(elem * 2));
        self.record_single(elems.saturating_mul(elem));
    }

    fn bottleneck(&mut self, c1: usize, c2: usize, e: f64, h: usize, w: usize) {
        let hid = (c2 as f64 * e) as usize;
        self.conv_bn_act(c1, hid, 3, 1, h, w);
        self.conv_bn_act(hid, c2, 3, 1, h, w);
    }

    fn c3k(&mut self, c1: usize, c2: usize, n: usize, h: usize, w: usize) {
        let mid = c2 / 2;
        self.conv_bn_act(c1, mid, 1, 1, h, w);
        self.conv_bn_act(c1, mid, 1, 1, h, w);
        for _ in 0..n {
            self.bottleneck(mid, mid, 1.0, h, w);
        }
        let elem = self.elem_bytes();
        self.record_single(((2 * mid) as u64) * (h as u64) * (w as u64) * elem);
        self.conv_bn_act(2 * mid, c2, 1, 1, h, w);
    }

    #[allow(clippy::too_many_arguments)]
    fn c3k2(&mut self, c1: usize, c2: usize, n: usize, use_c3k: bool, e: f64, h: usize, w: usize) {
        let mid = ((c2 as f64) * e) as usize;
        self.conv_bn_act(c1, 2 * mid, 1, 1, h, w);
        for _ in 0..n {
            if use_c3k {
                self.c3k(mid, mid, 2, h, w);
            } else {
                self.bottleneck(mid, mid, 0.5, h, w);
            }
        }
        let cat_c = (2 + n) * mid;
        let elem = self.elem_bytes();
        let cat_bytes = (cat_c as u64) * (h as u64) * (w as u64) * elem;
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(cat_bytes);
        self.record_single(cat_bytes);
        self.conv_bn_act(cat_c, c2, 1, 1, h, w);
    }

    fn sppf(&mut self, c1: usize, c2: usize, h: usize, w: usize) {
        let mid = c1 / 2;
        self.conv_bn_act(c1, mid, 1, 1, h, w);
        let elem = self.elem_bytes();
        // Three MaxPool outputs + 4-way concat.
        let pool_bytes = (mid as u64) * (h as u64) * (w as u64) * elem;
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(pool_bytes * 7);
        self.record_single(pool_bytes * 4);
        self.conv_bn_act(4 * mid, c2, 1, 1, h, w);
    }

    fn c2psa(&mut self, c: usize, n: usize, h: usize, w: usize) {
        let mid = c / 2;
        self.conv_bn_act(c, 2 * mid, 1, 1, h, w);
        let heads = (mid / 64).max(1);
        let head_dim = mid / heads;
        let key_dim = head_dim / 2;
        let qkv_out = heads * (2 * key_dim + head_dim);
        let tokens = (h as u64) * (w as u64);
        let elem = self.elem_bytes();
        for _ in 0..n {
            // Attention: qkv, pe (depthwise 3x3), proj + N×N attention logits/weights.
            self.conv_bn_act(mid, qkv_out, 1, 1, h, w);
            let attn_bytes = (heads as u64) * tokens * tokens * elem;
            self.per_sample_bytes = self.per_sample_bytes.saturating_add(attn_bytes * 3);
            self.record_single(attn_bytes);
            self.conv_bn_act(mid, mid, 3, mid, h, w); // pe depthwise
            self.conv_bn_act(mid, mid, 1, 1, h, w); // proj
            // FFN: 1x1 expand 2*mid, 1x1 contract mid.
            self.conv_bn_act(mid, 2 * mid, 1, 1, h, w);
            self.conv_bn_act(2 * mid, mid, 1, 1, h, w);
        }
        self.conv_bn_act(2 * mid, c, 1, 1, h, w);
    }

    fn detect_branch(&mut self, chs: [usize; 3], imgsz: usize, nc: usize, reg_max: usize) {
        let c2 = 16usize.max(chs[0] / 4).max(4 * reg_max);
        let c3 = chs[0].max(nc.min(100));
        let strides = [8usize, 16, 32];
        for (&cin, &s) in chs.iter().zip(&strides) {
            let hw = imgsz / s;
            // Box branch: conv0(3x3), conv1(3x3), out(1x1)
            self.conv_bn_act(cin, c2, 3, 1, hw, hw);
            self.conv_bn_act(c2, c2, 3, 1, hw, hw);
            self.conv2d_bias(c2, 4 * reg_max, 1, hw, hw);
            // Cls branch: dw0(3x3, g=cin), pw0(1x1), dw1(3x3, g=c3), pw1(1x1), out(1x1)
            self.conv_bn_act(cin, cin, 3, cin, hw, hw);
            self.conv_bn_act(cin, c3, 1, 1, hw, hw);
            self.conv_bn_act(c3, c3, 3, c3, hw, hw);
            self.conv_bn_act(c3, c3, 1, 1, hw, hw);
            self.conv2d_bias(c3, nc, 1, hw, hw);
        }
    }
}

fn analyze_folo_model(cfg: &ModelConfig, imgsz: usize, amp: bool) -> ModelStats {
    let s = Scaling::of(cfg);
    let (ch, rep) = (|c| s.ch(c), |n| s.rep(n));
    let mut st = ModelStats::new(amp);

    // Input image tensor [3, H, W] in F32 (uploaded as u8/f32 before autocast).
    let in_bytes = (cfg.in_channels as u64) * (imgsz as u64) * (imgsz as u64) * 4;
    st.per_sample_bytes = st.per_sample_bytes.saturating_add(in_bytes);
    st.record_single(in_bytes);

    let (h2, h4, h8, h16, h32) = (imgsz / 2, imgsz / 4, imgsz / 8, imgsz / 16, imgsz / 32);
    let (c0, c1, c2, c3, c4) = (ch(64), ch(128), ch(256), ch(256), ch(512));
    let (c5, c6, c7, c8, c9) = (ch(512), ch(512), ch(1024), ch(1024), ch(1024));
    let c10 = c9;
    let (c13, c16, c17, c19, c20, c22) = (ch(512), ch(256), ch(256), ch(512), ch(512), ch(1024));

    // Backbone (layers 0-10)
    st.conv_bn_act(cfg.in_channels, c0, 3, 1, h2, h2); // stem
    st.conv_bn_act(c0, c1, 3, 1, h4, h4); // down2
    st.c3k2(c1, c2, rep(2), s.c3k(false), 0.25, h4, h4); // stage2
    st.conv_bn_act(c2, c3, 3, 1, h8, h8); // down3
    st.c3k2(c3, c4, rep(2), s.c3k(false), 0.25, h8, h8); // stage3
    st.conv_bn_act(c4, c5, 3, 1, h16, h16); // down4
    st.c3k2(c5, c6, rep(2), s.c3k(true), 0.5, h16, h16); // stage4
    st.conv_bn_act(c6, c7, 3, 1, h32, h32); // down5
    st.c3k2(c7, c8, rep(2), s.c3k(true), 0.5, h32, h32); // stage5
    st.sppf(c8, c9, h32, h32); // sppf
    st.c2psa(c9, rep(2), h32, h32); // psa

    // Neck (layers 11-22)
    let elem = st.elem_bytes();
    st.record_single(((c10 + c6) as u64) * (h16 as u64) * (h16 as u64) * elem);
    st.c3k2(c10 + c6, c13, rep(2), s.c3k(false), 0.5, h16, h16); // td4
    st.record_single(((c13 + c4) as u64) * (h8 as u64) * (h8 as u64) * elem);
    st.c3k2(c13 + c4, c16, rep(2), s.c3k(false), 0.5, h8, h8); // out3
    st.conv_bn_act(c16, c17, 3, 1, h16, h16); // down3
    st.record_single(((c17 + c13) as u64) * (h16 as u64) * (h16 as u64) * elem);
    st.c3k2(c17 + c13, c19, rep(2), s.c3k(false), 0.5, h16, h16); // out4
    st.conv_bn_act(c19, c20, 3, 1, h32, h32); // down4
    st.record_single(((c20 + c10) as u64) * (h32 as u64) * (h32 as u64) * elem);
    st.c3k2(c20 + c10, c22, rep(2), s.c3k(true), 0.5, h32, h32); // out5

    // Head (layer 23): o2m + optional o2o branch
    let reg_max = cfg.reg_max as usize;
    let nc = cfg.num_classes;
    let o2o = cfg.has_o2o();
    st.detect_branch([c16, c19, c22], imgsz, nc, reg_max);
    if o2o {
        st.detect_branch([c16, c19, c22], imgsz, nc, reg_max);
    }

    // Loss & TAL working buffers (always F32, even under AMP; doubled for o2o dual loss).
    let anchors = total_anchors(imgsz);
    let box_raw_f32 = (4 * reg_max as u64).saturating_mul(anchors).saturating_mul(4);
    let cls_raw_f32 = (nc as u64).saturating_mul(anchors).saturating_mul(4);
    let head_branches: u64 = if o2o { 2 } else { 1 };
    let loss_work =
        box_raw_f32.saturating_add(cls_raw_f32).saturating_mul(3).saturating_add(anchors.saturating_mul(32));
    st.per_sample_bytes = st.per_sample_bytes.saturating_add(loss_work.saturating_mul(head_branches));
    st.record_single(box_raw_f32);
    st.record_single(cls_raw_f32);
    st
}

fn total_anchors(imgsz: usize) -> u64 {
    let a = |s: usize| {
        let g = (imgsz / s) as u64;
        g * g
    };
    a(8) + a(16) + a(32)
}

fn parse_scale(arch: &str) -> Result<&str> {
    let s = arch.strip_prefix("folo-").unwrap_or(arch);
    match s {
        "n" | "s" | "m" | "l" | "x" => Ok(s),
        _ => {
            Err(Error::Config(format!("arch `{arch}` is not one of folo-n, folo-s, folo-m, folo-l, folo-x")))
        }
    }
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
