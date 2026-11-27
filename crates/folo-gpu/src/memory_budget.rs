//! Deterministic GPU memory visibility, peak-watermark tracking, and VRAM budget / headroom estimation.
//!
//! [`MemoryLimits`], [`MemoryUsage`], [`MemoryBudget`], and [`HeadroomEstimate`] separate pure budget math from
//! live `wgpu` handles so callers (trainers, auto-batch sizers, `/api/system`, diagnostics) can inspect pool
//! pressure, estimate headroom for upcoming allocations or batch sizes, and format actionable out-of-memory /
//! device-limit messages — and every calculation is unit-tested deterministically without requiring a GPU.

use std::fmt;

use crate::context::Capabilities;
use crate::pool::{PoolStats, size_class};

/// Live and peak buffer-pool allocation bytes captured atomically under the pool lock.
///
/// Returned by [`crate::GpuContext::peak_memory`] (read-only) and [`crate::GpuContext::take_peak_memory`]
/// (capture and reset peak watermarks in one critical section).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeakMemory {
    /// Bytes in pooled buffers currently lent out.
    pub live_bytes: u64,
    /// Bytes in pooled buffers currently sitting in free lists for reuse.
    pub cached_bytes: u64,
    /// Highest [`Self::live_bytes`] observed since creation or the last peak reset.
    pub peak_live_bytes: u64,
    /// Highest `live_bytes + cached_bytes` observed since creation or the last peak reset.
    pub peak_held_bytes: u64,
}

impl PeakMemory {
    /// Builds a [`PeakMemory`] view from a [`PoolStats`] snapshot.
    pub const fn from_pool_stats(stats: PoolStats) -> Self {
        Self {
            live_bytes: stats.live_bytes,
            cached_bytes: stats.cached_bytes,
            peak_live_bytes: stats.peak_live_bytes,
            peak_held_bytes: stats.peak_held_bytes,
        }
    }

    /// Total pooled bytes held at snapshot time (`live_bytes + cached_bytes`).
    pub const fn held_bytes(&self) -> u64 {
        self.live_bytes.saturating_add(self.cached_bytes)
    }

    /// Transient peak spike above current live bytes (`peak_live_bytes - live_bytes`).
    pub const fn peak_delta_bytes(&self) -> u64 {
        self.peak_live_bytes.saturating_sub(self.live_bytes)
    }
}

/// Hardware allocation limits and total dedicated VRAM backing a [`MemoryBudget`].
///
/// Constructible directly from [`wgpu::Limits`], from [`Capabilities`] + [`crate::AdapterSummary::memory_bytes`],
/// or with explicit numbers in deterministic unit tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLimits {
    /// Total dedicated / device-local VRAM in bytes when reported by the backend (`Some` on native Vulkan,
    /// `None` where the backend cannot query it, such as DX12, Metal, or browser WebGPU).
    pub dedicated_bytes: Option<u64>,
    /// Maximum size of a single GPU buffer (`wgpu::Limits::max_buffer_size`).
    pub max_buffer_size: u64,
    /// Maximum size of a single storage buffer binding (`wgpu::Limits::max_storage_buffer_binding_size`).
    pub max_storage_buffer_binding_size: u64,
}

impl MemoryLimits {
    /// Creates explicit limits (`dedicated_bytes: Some(0)` is normalized to `None`).
    pub const fn new(
        dedicated_bytes: Option<u64>,
        max_buffer_size: u64,
        max_storage_buffer_binding_size: u64,
    ) -> Self {
        let dedicated_bytes = match dedicated_bytes {
            Some(0) | None => None,
            Some(v) => Some(v),
        };
        Self { dedicated_bytes, max_buffer_size, max_storage_buffer_binding_size }
    }

    /// Builds [`MemoryLimits`] from [`wgpu::Limits`] and an optional dedicated VRAM total.
    pub fn from_wgpu(limits: &wgpu::Limits, dedicated_bytes: Option<u64>) -> Self {
        Self::new(dedicated_bytes, limits.max_buffer_size, limits.max_storage_buffer_binding_size)
    }

    /// Builds [`MemoryLimits`] from FOLO's [`Capabilities`] and [`crate::AdapterSummary::memory_bytes`]
    /// (`0` becomes `None`).
    pub const fn from_capabilities(caps: &Capabilities, dedicated_bytes: u64) -> Self {
        let dedicated = if dedicated_bytes == 0 { None } else { Some(dedicated_bytes) };
        Self::new(dedicated, caps.max_buffer_size, caps.max_storage_buffer_binding_size)
    }

    /// Effective per-buffer allocation ceiling in bytes for `kind`.
    pub const fn single_buffer_limit(&self, kind: AllocationKind) -> u64 {
        match kind {
            AllocationKind::Storage => {
                if self.max_buffer_size < self.max_storage_buffer_binding_size {
                    self.max_buffer_size
                } else {
                    self.max_storage_buffer_binding_size
                }
            }
            AllocationKind::Buffer => self.max_buffer_size,
        }
    }
}

/// Snapshot of current and peak GPU memory held by a [`crate::GpuContext`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Buffer pool counters (live, cached, peak live, peak held, need, hits/misses).
    pub pool: PoolStats,
    /// Bytes pinned by cached forward plans (`GpuContext::plan_stats().1`).
    pub plan_bytes: u64,
    /// Bytes held by cached device-resident anchor tables.
    pub anchor_bytes: u64,
    /// Host-visible mapped staging bytes held by `GpuContext::upload_with`.
    pub staging_bytes: u64,
    /// Bytes held by open command-batch uniform arena chunks.
    pub uniform_bytes: u64,
    /// Total allocated bytes reported by wgpu's allocator, if supported by the backend.
    pub device_allocated_bytes: Option<u64>,
    /// Total reserved memory-block bytes reported by wgpu's allocator, if supported by the backend.
    pub device_reserved_bytes: Option<u64>,
}

impl MemoryUsage {
    /// Creates a [`MemoryUsage`] from pool counters alone (zero auxiliary caches, no driver allocator report).
    pub const fn from_pool(pool: PoolStats) -> Self {
        Self {
            pool,
            plan_bytes: 0,
            anchor_bytes: 0,
            staging_bytes: 0,
            uniform_bytes: 0,
            device_allocated_bytes: None,
            device_reserved_bytes: None,
        }
    }

    /// Live and peak buffer-pool watermarks.
    pub const fn peak(&self) -> PeakMemory {
        PeakMemory::from_pool_stats(self.pool)
    }

    /// Total bytes held in the buffer pool (`live + cached`).
    pub const fn pool_held_bytes(&self) -> u64 {
        self.pool.held_bytes()
    }

    /// Tracked non-reclaimable live device bytes (live pool buffers + open uniform chunks).
    pub const fn tracked_live_bytes(&self) -> u64 {
        self.pool.live_bytes.saturating_add(self.uniform_bytes)
    }

    /// Tracked device bytes held across the pool, forward plans, anchor cache, and uniform chunks.
    pub const fn tracked_held_bytes(&self) -> u64 {
        self.pool_held_bytes()
            .saturating_add(self.plan_bytes)
            .saturating_add(self.anchor_bytes)
            .saturating_add(self.uniform_bytes)
    }

    /// Bytes that [`crate::GpuContext::release_memory`] can reclaim without dropping live tensors.
    pub const fn reclaimable_bytes(&self) -> u64 {
        self.pool
            .cached_bytes
            .saturating_add(self.plan_bytes)
            .saturating_add(self.anchor_bytes)
            .saturating_add(self.staging_bytes)
    }

    /// Best estimate of committed non-reclaimable live device bytes.
    ///
    /// When wgpu reports `device_allocated_bytes`, strictly disjoint cached pool buffers, cached anchor tables,
    /// and idle mapped upload staging buffers are subtracted so reusable/reclaimable caches are not miscounted as
    /// unavailable VRAM, while any untracked live allocations on the same device still count against headroom.
    pub const fn effective_live_bytes(&self) -> u64 {
        let tracked = self.tracked_live_bytes();
        match self.device_allocated_bytes {
            Some(alloc) => {
                let disjoint_cached = self
                    .pool
                    .cached_bytes
                    .saturating_add(self.anchor_bytes)
                    .saturating_add(self.staging_bytes);
                let wgpu_live = alloc.saturating_sub(disjoint_cached);
                if wgpu_live > tracked { wgpu_live } else { tracked }
            }
            None => tracked,
        }
    }

    /// Best estimate of total device bytes currently held from the driver, including reusable pool caches and
    /// driver allocator blocks.
    pub const fn effective_reserved_bytes(&self) -> u64 {
        let tracked = self.tracked_held_bytes();
        let driver = match (self.device_allocated_bytes, self.device_reserved_bytes) {
            (Some(a), Some(r)) => {
                if r > a {
                    r
                } else {
                    a
                }
            }
            (Some(a), None) => a,
            (None, Some(r)) => r,
            (None, None) => 0,
        };
        if driver > tracked { driver } else { tracked }
    }
}

/// Whether an allocation will be bound as a compute storage buffer or used only as a plain/staging buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationKind {
    /// Storage buffer (`STORAGE`), bounded by `min(max_buffer_size, max_storage_buffer_binding_size)`.
    Storage,
    /// Non-storage buffer (e.g. staging or copy-only buffer), bounded by `max_buffer_size`.
    Buffer,
}

/// Specific reason an allocation is rejected by [`MemoryBudget::estimate_headroom`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadroomBlocker {
    /// Single allocation exceeds `wgpu::Limits::max_storage_buffer_binding_size`.
    ExceedsStorageBindingLimit,
    /// Single allocation exceeds `wgpu::Limits::max_buffer_size`.
    ExceedsMaxBufferSize,
    /// Allocation exceeds remaining usable device VRAM even after accounting for reusable caches.
    ExceedsDeviceMemory,
}

/// Deterministic headroom assessment for a candidate GPU allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadroomEstimate {
    /// Raw requested byte count.
    pub requested_bytes: u64,
    /// Size-class (or 4-byte aligned near-limit) allocation size, or `None` if alignment itself overflows or
    /// exceeds the single-buffer limit.
    pub aligned_bytes: Option<u64>,
    /// Allocation class checked (`Storage` vs `Buffer`).
    pub kind: AllocationKind,
    /// Applicable per-buffer device limit in bytes.
    pub single_buffer_limit: u64,
    /// Usable VRAM remaining above committed live usage before this allocation (`None` when VRAM total is unknown).
    pub available_vram_bytes: Option<u64>,
    /// Usable VRAM remaining above driver-reserved bytes without reclaiming cached buffers (`None` when unknown).
    pub available_without_reclaim_bytes: Option<u64>,
    /// Cached bytes that `release_memory()` / `trim_pool()` can reclaim without touching live tensors.
    pub reclaimable_bytes: u64,
    /// Whether the allocation fits within both per-buffer limits and known usable VRAM.
    pub fits: bool,
    /// True when the allocation fits overall but exceeds unreserved VRAM, so pool eviction or
    /// `GpuContext::release_memory()` will be needed.
    pub needs_reclaim: bool,
    /// Why the allocation does not fit, if `!fits`.
    pub blocker: Option<HeadroomBlocker>,
}

impl fmt::Display for HeadroomEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind_label = match self.kind {
            AllocationKind::Storage => "storage buffer",
            AllocationKind::Buffer => "buffer",
        };
        let aligned = match self.aligned_bytes {
            Some(a) if a != self.requested_bytes => format!(" (aligned {a} B / {})", fmt_bytes(a)),
            _ => String::new(),
        };
        match self.blocker {
            Some(HeadroomBlocker::ExceedsStorageBindingLimit) => write!(
                f,
                "{kind_label} of {} bytes ({}){aligned} exceeds max_storage_buffer_binding_size ({} bytes / {})",
                self.requested_bytes,
                fmt_bytes(self.requested_bytes),
                self.single_buffer_limit,
                fmt_bytes(self.single_buffer_limit),
            ),
            Some(HeadroomBlocker::ExceedsMaxBufferSize) => write!(
                f,
                "{kind_label} of {} bytes ({}){aligned} exceeds max_buffer_size ({} bytes / {})",
                self.requested_bytes,
                fmt_bytes(self.requested_bytes),
                self.single_buffer_limit,
                fmt_bytes(self.single_buffer_limit),
            ),
            Some(HeadroomBlocker::ExceedsDeviceMemory) => {
                let avail = self.available_vram_bytes.unwrap_or(0);
                write!(
                    f,
                    "{kind_label} of {} bytes ({}){aligned} exceeds estimated remaining VRAM ({} bytes / {}; \
                     reclaimable cache {})",
                    self.requested_bytes,
                    fmt_bytes(self.requested_bytes),
                    avail,
                    fmt_bytes(avail),
                    fmt_bytes(self.reclaimable_bytes),
                )
            }
            None => {
                write!(
                    f,
                    "{kind_label} of {} bytes ({}){aligned} fits (limit {})",
                    self.requested_bytes,
                    fmt_bytes(self.requested_bytes),
                    fmt_bytes(self.single_buffer_limit),
                )?;
                if let Some(avail) = self.available_vram_bytes {
                    write!(f, "; estimated free VRAM {}", fmt_bytes(avail))?;
                    if self.needs_reclaim {
                        write!(f, " (reclaim/evict from {} cache)", fmt_bytes(self.reclaimable_bytes))?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Deterministic GPU VRAM budget built from hardware [`MemoryLimits`] and a [`MemoryUsage`] snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Device per-buffer limits and dedicated VRAM total.
    pub limits: MemoryLimits,
    /// Snapshot of current and peak memory usage.
    pub usage: MemoryUsage,
    /// Safety reserve kept aside from dedicated VRAM for driver/OS/desktop overhead.
    pub reserve_bytes: u64,
}

impl MemoryBudget {
    /// Creates a [`MemoryBudget`] with zero safety reserve (exact hardware limits).
    pub const fn new(limits: MemoryLimits, usage: MemoryUsage) -> Self {
        Self { limits, usage, reserve_bytes: 0 }
    }

    /// Sets an explicit safety reserve in bytes subtracted from dedicated VRAM.
    pub const fn with_reserve_bytes(mut self, reserve_bytes: u64) -> Self {
        self.reserve_bytes = reserve_bytes;
        self
    }

    /// Sets the safety reserve as a fraction of dedicated VRAM (clamped to `[0.0, 1.0]`; `0` when VRAM is unknown).
    pub fn with_reserve_fraction(mut self, fraction: f64) -> Self {
        let clamped = if fraction.is_finite() { fraction.clamp(0.0, 1.0) } else { 0.0 };
        self.reserve_bytes = self
            .limits
            .dedicated_bytes
            .map(|total| ((total as f64) * clamped).round().clamp(0.0, total as f64) as u64)
            .unwrap_or(0);
        self
    }

    /// Usable dedicated VRAM after subtracting [`Self::reserve_bytes`], or `None` when total VRAM is unknown.
    pub const fn usable_vram_bytes(&self) -> Option<u64> {
        match self.limits.dedicated_bytes {
            Some(total) => Some(total.saturating_sub(self.reserve_bytes)),
            None => None,
        }
    }

    /// Estimated free VRAM available for new live allocations (assuming cached pool buffers can be reused or
    /// evicted), or `None` when total VRAM is unknown.
    pub const fn free_vram_bytes(&self) -> Option<u64> {
        match self.usable_vram_bytes() {
            Some(usable) => Some(usable.saturating_sub(self.usage.effective_live_bytes())),
            None => None,
        }
    }

    /// Estimated free VRAM without evicting cached buffers or calling `release_memory()`, or `None` when unknown.
    pub const fn free_without_reclaim_bytes(&self) -> Option<u64> {
        match self.usable_vram_bytes() {
            Some(usable) => Some(usable.saturating_sub(self.usage.effective_reserved_bytes())),
            None => None,
        }
    }

    /// Fraction of dedicated VRAM currently used by live allocations (`effective_live_bytes / dedicated_bytes`),
    /// or `None` when total VRAM is unknown.
    pub fn utilization(&self) -> Option<f64> {
        let total = self.limits.dedicated_bytes?;
        (total > 0).then(|| self.usage.effective_live_bytes() as f64 / total as f64)
    }

    /// Fraction of dedicated VRAM reached at the buffer pool's live peak (`peak_live_bytes / dedicated_bytes`),
    /// or `None` when total VRAM is unknown.
    pub fn peak_utilization(&self) -> Option<f64> {
        let total = self.limits.dedicated_bytes?;
        (total > 0).then(|| self.usage.pool.peak_live_bytes as f64 / total as f64)
    }

    /// Computes the aligned allocation size `GpuContext` uses for `requested_bytes` under `kind`, or `None` when
    /// `requested_bytes` or its required 4-byte copy alignment exceeds the per-buffer device limit.
    pub fn aligned_bytes(&self, requested_bytes: u64, kind: AllocationKind) -> Option<u64> {
        aligned_within_limit(requested_bytes, self.limits.single_buffer_limit(kind))
    }

    /// Evaluates whether a single allocation of `requested_bytes` fits within per-buffer `wgpu::Limits` and
    /// estimated remaining VRAM.
    pub fn estimate_headroom(&self, requested_bytes: u64, kind: AllocationKind) -> HeadroomEstimate {
        let single_buffer_limit = self.limits.single_buffer_limit(kind);
        let aligned = aligned_within_limit(requested_bytes, single_buffer_limit);
        let available_vram_bytes = self.free_vram_bytes();
        let available_without_reclaim_bytes = self.free_without_reclaim_bytes();
        let reclaimable_bytes = self.usage.reclaimable_bytes();

        let limit_blocker = if aligned.is_none() {
            Some(match kind {
                AllocationKind::Storage
                    if self.limits.max_storage_buffer_binding_size < self.limits.max_buffer_size =>
                {
                    let fits_buf =
                        aligned_within_limit(requested_bytes, self.limits.max_buffer_size).is_some();
                    if fits_buf {
                        HeadroomBlocker::ExceedsStorageBindingLimit
                    } else {
                        HeadroomBlocker::ExceedsMaxBufferSize
                    }
                }
                AllocationKind::Storage => HeadroomBlocker::ExceedsMaxBufferSize,
                AllocationKind::Buffer => HeadroomBlocker::ExceedsMaxBufferSize,
            })
        } else {
            None
        };

        let effective_need = aligned.unwrap_or(requested_bytes);
        let blocker = limit_blocker.or_else(|| {
            available_vram_bytes
                .is_some_and(|avail| effective_need > avail)
                .then_some(HeadroomBlocker::ExceedsDeviceMemory)
        });
        let fits = blocker.is_none();
        let needs_reclaim =
            fits && available_without_reclaim_bytes.is_some_and(|avail| effective_need > avail);

        HeadroomEstimate {
            requested_bytes,
            aligned_bytes: aligned,
            kind,
            single_buffer_limit,
            available_vram_bytes,
            available_without_reclaim_bytes,
            reclaimable_bytes,
            fits,
            needs_reclaim,
            blocker,
        }
    }

    /// Shorthand for `self.estimate_headroom(requested_bytes, kind).fits`.
    pub fn fits(&self, requested_bytes: u64, kind: AllocationKind) -> bool {
        self.estimate_headroom(requested_bytes, kind).fits
    }

    /// Estimates the largest batch size that fits within BOTH:
    /// - remaining usable VRAM (`fixed_bytes + batch * per_sample_bytes <= free_vram_bytes`), when dedicated VRAM
    ///   is known, and
    /// - the single storage-buffer limit (`batch * max_single_tensor_per_sample <= max_storage_limit`), when
    ///   `max_single_tensor_per_sample > 0`.
    ///
    /// Returns `None` when neither constraint bounds the batch size (for example `per_sample_bytes == 0` or unknown
    /// VRAM with `max_single_tensor_per_sample == 0`). Returns `Some(0)` when even batch 1 does not fit.
    pub fn estimate_max_batch(
        &self,
        fixed_bytes: u64,
        per_sample_bytes: u64,
        max_single_tensor_per_sample: u64,
    ) -> Option<usize> {
        let mut cap: Option<u64> = None;
        if let Some(free) = self.free_vram_bytes() {
            if fixed_bytes > free {
                return Some(0);
            }
            if let Some(by_vram) = (free - fixed_bytes).checked_div(per_sample_bytes) {
                cap = Some(cap.map_or(by_vram, |c| c.min(by_vram)));
            }
        }
        if let Some(by_limit) =
            self.limits.single_buffer_limit(AllocationKind::Storage).checked_div(max_single_tensor_per_sample)
        {
            cap = Some(cap.map_or(by_limit, |c| c.min(by_limit)));
        }
        cap.map(|b| usize::try_from(b).unwrap_or(usize::MAX))
    }

    /// Formats an actionable error diagnostic when an allocation exceeds a device limit (or fails near an OOM
    /// boundary), including which limit was exceeded, pool live/cached/peak watermarks, and VRAM headroom when
    /// known.
    pub fn format_limit_diagnostic(&self, requested_bytes: u64, kind: AllocationKind) -> String {
        let est = self.estimate_headroom(requested_bytes, kind);
        let pool = self.usage.pool;
        let vram = match (self.limits.dedicated_bytes, est.available_vram_bytes) {
            (Some(total), Some(free)) => {
                format!(", VRAM {} total / ~{} est. free", fmt_bytes(total), fmt_bytes(free))
            }
            _ => String::new(),
        };
        format!(
            "{est} [pool live {}, cached {}, peak live {}, peak held {}{vram}]",
            fmt_bytes(pool.live_bytes),
            fmt_bytes(pool.cached_bytes),
            fmt_bytes(pool.peak_live_bytes),
            fmt_bytes(pool.peak_held_bytes),
        )
    }
}

/// Rounds `requested_bytes` up to its pool size class (falling back to 4-byte copy alignment right near `limit`),
/// returning `None` if `requested_bytes` or its required alignment exceeds `limit`.
pub(crate) fn aligned_within_limit(requested_bytes: u64, limit: u64) -> Option<u64> {
    if requested_bytes > limit {
        return None;
    }
    let class = size_class(requested_bytes);
    if class <= limit {
        return Some(class);
    }
    let aligned = requested_bytes
        .max(wgpu::COPY_BUFFER_ALIGNMENT)
        .checked_next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)?;
    (aligned <= limit).then_some(aligned)
}

/// Formats a byte count cleanly (`B`, `KiB`, `MiB`, `GiB`) for diagnostics.
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

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    fn sample_limits() -> MemoryLimits {
        MemoryLimits::new(Some(16 * GIB), 4 * GIB, 2 * GIB)
    }

    #[test]
    fn peak_memory_deltas_and_held_bytes() {
        let stats = PoolStats {
            live_bytes: 512 * MIB,
            cached_bytes: 256 * MIB,
            peak_live_bytes: 1536 * MIB,
            peak_held_bytes: 2048 * MIB,
            ..PoolStats::default()
        };
        let peak = PeakMemory::from_pool_stats(stats);
        assert_eq!(peak.held_bytes(), 768 * MIB);
        assert_eq!(peak.peak_delta_bytes(), 1024 * MIB);
        assert_eq!(stats.held_bytes(), 768 * MIB);
    }

    #[test]
    fn limits_from_wgpu_and_zero_dedicated_normalization() {
        let wl = wgpu::Limits {
            max_buffer_size: 512 * MIB,
            max_storage_buffer_binding_size: 128 * MIB,
            ..wgpu::Limits::default()
        };
        let lim = MemoryLimits::from_wgpu(&wl, Some(0));
        assert_eq!(lim.dedicated_bytes, None);
        assert_eq!(lim.single_buffer_limit(AllocationKind::Storage), 128 * MIB);
        assert_eq!(lim.single_buffer_limit(AllocationKind::Buffer), 512 * MIB);
    }

    #[test]
    fn headroom_distinguishes_storage_binding_vs_max_buffer_vs_vram() {
        let usage = MemoryUsage {
            pool: PoolStats {
                live_bytes: 13 * GIB,
                cached_bytes: 2 * GIB,
                peak_live_bytes: 14 * GIB,
                peak_held_bytes: 15 * GIB,
                ..PoolStats::default()
            },
            device_allocated_bytes: Some(15 * GIB),
            device_reserved_bytes: Some(15 * GIB),
            ..MemoryUsage::default()
        };
        // Reserve 1 GiB -> 15 GiB usable, 13 GiB live -> 2 GiB free (0 without reclaim).
        let budget = MemoryBudget::new(sample_limits(), usage).with_reserve_bytes(GIB);
        assert_eq!(budget.usable_vram_bytes(), Some(15 * GIB));
        assert_eq!(budget.free_vram_bytes(), Some(2 * GIB));
        assert_eq!(budget.free_without_reclaim_bytes(), Some(0));

        // 1 GiB storage fits, but needs cache reclaim/eviction.
        let ok = budget.estimate_headroom(GIB, AllocationKind::Storage);
        assert!(ok.fits);
        assert!(ok.needs_reclaim);
        assert_eq!(ok.blocker, None);
        assert_eq!(ok.aligned_bytes, Some(GIB));

        // 2.5 GiB storage exceeds max_storage_buffer_binding_size (2 GiB), while <= max_buffer_size (4 GiB).
        let storage_over = budget.estimate_headroom(2 * GIB + 512 * MIB, AllocationKind::Storage);
        assert!(!storage_over.fits);
        assert_eq!(storage_over.blocker, Some(HeadroomBlocker::ExceedsStorageBindingLimit));
        assert!(storage_over.to_string().contains("max_storage_buffer_binding_size"), "{storage_over}");

        // 5 GiB plain buffer exceeds max_buffer_size (4 GiB).
        let buf_over = budget.estimate_headroom(5 * GIB, AllocationKind::Buffer);
        assert!(!buf_over.fits);
        assert_eq!(buf_over.blocker, Some(HeadroomBlocker::ExceedsMaxBufferSize));
        assert!(buf_over.to_string().contains("max_buffer_size"), "{buf_over}");

        // 3 GiB plain buffer is within max_buffer_size (4 GiB) but exceeds 2 GiB remaining VRAM.
        let vram_over = budget.estimate_headroom(3 * GIB, AllocationKind::Buffer);
        assert!(!vram_over.fits);
        assert_eq!(vram_over.blocker, Some(HeadroomBlocker::ExceedsDeviceMemory));
        assert!(vram_over.to_string().contains("remaining VRAM"), "{vram_over}");
    }

    #[test]
    fn device_allocated_subtracts_disjoint_pool_cache_but_keeps_untracked_live_buffers() {
        let usage = MemoryUsage {
            pool: PoolStats { live_bytes: 2 * GIB, cached_bytes: 8 * GIB, ..PoolStats::default() },
            anchor_bytes: 64 * MIB,
            // 1 GiB untracked external buffer on the same device: 2 + 8 + 0.0625 + 1 = 11.0625 GiB.
            device_allocated_bytes: Some(11 * GIB + 64 * MIB),
            device_reserved_bytes: Some(12 * GIB),
            ..MemoryUsage::default()
        };
        assert_eq!(usage.effective_live_bytes(), 3 * GIB);
        assert_eq!(usage.effective_reserved_bytes(), 12 * GIB);
        let budget = MemoryBudget::new(sample_limits(), usage);
        assert_eq!(budget.free_vram_bytes(), Some(13 * GIB));
        assert_eq!(budget.free_without_reclaim_bytes(), Some(4 * GIB));
    }

    #[test]
    fn estimate_max_batch_respects_vram_reserve_and_single_tensor_limit() {
        let usage = MemoryUsage::from_pool(PoolStats { live_bytes: 2 * GIB, ..PoolStats::default() });
        // 16 GiB * 10% reserve = 1.6 GiB -> 14.4 GiB usable -> 12.4 GiB free.
        let budget = MemoryBudget::new(sample_limits(), usage).with_reserve_fraction(0.10);
        // Fixed 1 GiB, 512 MiB/sample -> VRAM allows floor(11.4 GiB / 0.5 GiB) = 22.
        assert_eq!(budget.estimate_max_batch(GIB, 512 * MIB, 0), Some(22));
        // If one tensor is 256 MiB/sample, 2 GiB storage limit caps batch at 8.
        assert_eq!(budget.estimate_max_batch(GIB, 512 * MIB, 256 * MIB), Some(8));
        // Fixed exceeds free VRAM -> 0, even when per_sample_bytes == 0.
        assert_eq!(budget.estimate_max_batch(20 * GIB, 512 * MIB, 64 * MIB), Some(0));
        assert_eq!(budget.estimate_max_batch(20 * GIB, 0, 64 * MIB), Some(0));

        // Unknown dedicated VRAM: single-tensor limit still bounds batch; zero single-tensor returns None.
        let unknown = MemoryBudget::new(MemoryLimits::new(None, 4 * GIB, 2 * GIB), usage);
        assert_eq!(unknown.estimate_max_batch(GIB, 512 * MIB, 0), None);
        assert_eq!(unknown.estimate_max_batch(GIB, 512 * MIB, 512 * MIB), Some(4));
    }

    #[test]
    fn near_limit_alignment_and_diagnostic_format_never_overflow() {
        let lim = MemoryLimits::new(Some(8 * GIB), 1024, 1024);
        let budget = MemoryBudget::new(lim, MemoryUsage::default());
        assert_eq!(budget.aligned_bytes(1024, AllocationKind::Storage), Some(1024));
        assert_eq!(budget.aligned_bytes(1025, AllocationKind::Storage), None);
        assert_eq!(budget.aligned_bytes(u64::MAX, AllocationKind::Storage), None);
        assert_eq!(budget.aligned_bytes(u64::MAX - 1, AllocationKind::Buffer), None);

        // Non-multiple-of-4 limit: 1021 bytes rounds to 1024 > 1022, rejected cleanly.
        let tight = MemoryLimits::new(None, 1022, 1022);
        let b2 = MemoryBudget::new(tight, MemoryUsage::default());
        assert_eq!(b2.aligned_bytes(1020, AllocationKind::Storage), Some(1020));
        assert_eq!(b2.aligned_bytes(1021, AllocationKind::Storage), None);

        let diag = budget.format_limit_diagnostic(2048, AllocationKind::Storage);
        assert!(diag.contains("2048 bytes") && diag.contains("pool live") && diag.contains("VRAM"), "{diag}");
    }
}
