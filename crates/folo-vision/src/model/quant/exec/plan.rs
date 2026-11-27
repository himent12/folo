//! Per-model int8 execution plan ([`Int8Plan`]), boundary-grid table, warm [`ScratchPool`] cache, and
//! O(tensors) weight-stamp validation for fingerprinted calibrations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use folo_core::TensorId;
use folo_core::cpu::q8::Grid;
use folo_nn::{Module, Result};

use crate::model::quant::Calibration;
use crate::model::quant::cache::{BoundaryGrids, FnvBuildHasher};
use crate::model::quant::pool::{ScratchPool, ScratchStats};

/// Maximum number of warm [`ScratchPool`]s retained per [`Int8Plan`] for concurrent `predict_batch` threads.
const MAX_CACHED_POOLS: usize = 8;

struct BoundaryEntry {
    ch: Option<BoundaryGrids>,
    tensor: Option<Grid>,
    is_f32: bool,
}

/// Borrowed (v2) or shared (v1 lazy uniform) reference to a [`BoundaryGrids`] so v2 forwards incur zero atomic
/// reference-count traffic per activation.
#[derive(Clone)]
pub(crate) enum GridOwner<'a> {
    Borrowed(&'a BoundaryGrids),
    Owned(Arc<BoundaryGrids>),
}

impl GridOwner<'_> {
    #[inline]
    pub(super) fn as_ref(&self) -> &BoundaryGrids {
        match self {
            GridOwner::Borrowed(bg) => bg,
            GridOwner::Owned(arc) => arc,
        }
    }
}

fn collect_stamp(m: &dyn Module, out: &mut Vec<(TensorId, u64)>) {
    m.visit_tensors(&mut |_, t, _| out.push((t.id(), t.storage_version())));
    m.visit_children(&mut |_, c| collect_stamp(c, out));
}

fn stamp_matches(m: &dyn Module, stamp: &[(TensorId, u64)], idx: &mut usize, ok: &mut bool) {
    m.visit_tensors(&mut |_, t, _| {
        if *ok {
            if let Some(&(id, ver)) = stamp.get(*idx) {
                if t.id() != id || t.storage_version() != ver {
                    *ok = false;
                }
            } else {
                *ok = false;
            }
            *idx += 1;
        }
    });
    if *ok {
        m.visit_children(&mut |_, c| stamp_matches(c, stamp, idx, ok));
    }
}

/// A calibrated model's int8 plan. A conv's packed weights are kept with its weight tensor
/// ([`folo_core::Tensor::derived_f32`]) under a key of everything else they are made from: a write to the
/// weights drops them, even one that overlaps a forward, and the next forward packs them again.
pub struct Int8Plan {
    pub(super) cal: Calibration,
    /// Whether `cal.f32_convs` is non-empty (lets `wants_f32` skip HashMap lookups when empty).
    has_f32_convs: bool,
    /// Unified FNV-1a boundary table: maps boundary name to its slot in `boundary_list`.
    boundaries: HashMap<String, usize, FnvBuildHasher>,
    /// Flat immutable table of `(name, BoundaryEntry)` indexed by per-step cached slot indices for O(1) lookups
    /// without re-hashing boundary strings on every steady-state forward pass.
    boundary_list: Vec<(String, BoundaryEntry)>,
    /// Lazily expanded per-boundary uniform grids for `version: 1` per-tensor calibrations.
    uniform_grids: RwLock<HashMap<String, Arc<BoundaryGrids>, FnvBuildHasher>>,
    /// Warm reusable C4 activation buffer pools shared across forward calls (one checked out per active forward).
    scratch: Mutex<Vec<ScratchPool>>,
    /// Cumulative scratch pool stats across all checked-in forwards (for diagnostics and tests).
    stats: Mutex<ScratchStats>,
    /// Set when the owning model's weights change and no longer match `cal.weights`.
    invalidated: AtomicBool,
    /// Snapshot of `(TensorId, storage_version)` for every tensor of the owning model when `cal.weights` is
    /// recorded, letting `active_int8` detect arbitrary weight replacements (including direct `ModelFile::load`,
    /// `state()` writes, or trait `Module::to_dtype` round-trips) in O(tensors) atomic reads.
    weight_stamp: Option<RwLock<Vec<(TensorId, u64)>>>,
}

impl Int8Plan {
    /// The plan of `cal`.
    pub fn new(cal: Calibration) -> Self {
        Self::new_inner(cal, None)
    }

    /// The plan of `cal`, recording `module`'s current tensor `(id, storage_version)` stamp when `cal.weights`
    /// is set so out-of-band weight writes invalidate a mismatched plan automatically.
    pub(crate) fn new_for_module(cal: Calibration, module: &dyn Module) -> Self {
        let stamp = cal.weights.is_some().then(|| {
            let mut v = Vec::new();
            collect_stamp(module, &mut v);
            RwLock::new(v)
        });
        Self::new_inner(cal, stamp)
    }

    fn new_inner(cal: Calibration, weight_stamp: Option<RwLock<Vec<(TensorId, u64)>>>) -> Self {
        let cap = cal.ranges.len().max(cal.f32_convs.len()) + 4;
        let mut boundaries: HashMap<String, usize, FnvBuildHasher> =
            HashMap::with_capacity_and_hasher(cap, FnvBuildHasher);
        let mut boundary_list: Vec<(String, BoundaryEntry)> = Vec::with_capacity(cap);
        if cal.check_version().is_ok() {
            for name in cal.ranges.keys() {
                let tensor = cal.param(name).ok();
                let ch = cal
                    .channel_ranges
                    .get(name)
                    .and_then(|crs| cal.channel_params(name, crs.len()).ok())
                    .map(BoundaryGrids::new);
                let idx = boundary_list.len();
                boundary_list.push((name.clone(), BoundaryEntry { ch, tensor, is_f32: false }));
                boundaries.insert(name.clone(), idx);
            }
        }
        for name in &cal.f32_convs {
            if let Some(&idx) = boundaries.get(name) {
                boundary_list[idx].1.is_f32 = true;
            } else {
                let idx = boundary_list.len();
                boundary_list.push((name.clone(), BoundaryEntry { ch: None, tensor: None, is_f32: true }));
                boundaries.insert(name.clone(), idx);
            }
        }
        let has_f32_convs = !cal.f32_convs.is_empty();
        Int8Plan {
            cal,
            has_f32_convs,
            boundaries,
            boundary_list,
            uniform_grids: RwLock::new(HashMap::with_hasher(FnvBuildHasher)),
            scratch: Mutex::new(Vec::new()),
            stats: Mutex::new(ScratchStats::default()),
            invalidated: AtomicBool::new(false),
            weight_stamp,
        }
    }

    /// The calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.cal
    }

    /// Cumulative activation scratch-buffer reuse and allocation counters across completed forwards on this plan.
    pub fn scratch_stats(&self) -> ScratchStats {
        self.stats.lock().map(|g| *g).unwrap_or_default()
    }

    /// Whether this plan has been permanently invalidated.
    #[inline]
    pub(crate) fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }

    /// Mark this plan stale after its model's weights were replaced with incompatible ones, immediately releasing
    /// pooled scratch buffers and cached packed weight handles.
    pub(crate) fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
        if let Ok(mut v) = self.scratch.lock() {
            v.clear();
        }
    }

    /// Whether this plan is still valid for `module`'s current weights. Fast-paths in O(tensors) atomic reads
    /// when no tensor storage changed; if any tensor changed on a fingerprinted calibration, re-verifies the
    /// fingerprint and either refreshes the stamp (if weights still match) or permanently invalidates the plan.
    pub(crate) fn check_module(
        &self,
        module: &dyn Module,
        fingerprint: impl FnOnce() -> Result<String>,
    ) -> bool {
        if self.invalidated.load(Ordering::Acquire) {
            return false;
        }
        let (Some(want), Some(lock)) = (self.cal.weights.as_deref(), &self.weight_stamp) else {
            return true;
        };
        if let Ok(guard) = lock.read() {
            let mut idx = 0usize;
            let mut ok = true;
            stamp_matches(module, &guard, &mut idx, &mut ok);
            if ok && idx == guard.len() {
                return true;
            }
        }
        let mut fresh = Vec::new();
        collect_stamp(module, &mut fresh);
        let matches = fingerprint().is_ok_and(|fp| fp == want);
        if matches {
            if let Ok(mut guard) = lock.write() {
                *guard = fresh;
            }
            true
        } else {
            self.invalidate();
            false
        }
    }

    #[inline]
    pub(super) fn is_f32(&self, name: &str) -> bool {
        self.has_f32_convs && self.boundaries.get(name).is_some_and(|&idx| self.boundary_list[idx].1.is_f32)
    }

    /// Fast O(1) boundary hit check using a cached slot index from [`ScratchPool`].
    #[inline]
    pub(super) fn boundary_at_slot(
        &self,
        slot: usize,
        name: &str,
        c: usize,
    ) -> Option<(GridOwner<'_>, bool)> {
        let (entry_name, entry) = self.boundary_list.get(slot)?;
        if entry_name == name
            && let Some(bg) = &entry.ch
            && bg.grids.len() == c
        {
            return Some((GridOwner::Borrowed(bg), entry.is_f32));
        }
        None
    }

    /// Single-lookup resolution of `(grids, is_f32, optional_slot)` for boundary `name` of channel count `c`.
    pub(super) fn boundary_for_slot(
        &self,
        name: &str,
        c: usize,
    ) -> Result<(GridOwner<'_>, bool, Option<usize>)> {
        if let Some(&idx) = self.boundaries.get(name) {
            let entry = &self.boundary_list[idx].1;
            if let Some(bg) = &entry.ch
                && bg.grids.len() == c
            {
                return Ok((GridOwner::Borrowed(bg), entry.is_f32, Some(idx)));
            }
            if !self.cal.channel_ranges.contains_key(name)
                && let Some(g) = entry.tensor
            {
                if let Ok(guard) = self.uniform_grids.read()
                    && let Some(bg) = guard.get(name)
                    && bg.grids.len() == c
                {
                    return Ok((GridOwner::Owned(bg.clone()), entry.is_f32, None));
                }
                if let Ok(mut guard) = self.uniform_grids.write() {
                    if let Some(bg) = guard.get(name)
                        && bg.grids.len() == c
                    {
                        return Ok((GridOwner::Owned(bg.clone()), entry.is_f32, None));
                    }
                    let bg = Arc::new(BoundaryGrids::new(vec![g; c]));
                    guard.insert(name.to_string(), bg.clone());
                    return Ok((GridOwner::Owned(bg), entry.is_f32, None));
                }
                return Ok((GridOwner::Owned(Arc::new(BoundaryGrids::new(vec![g; c]))), entry.is_f32, None));
            }
        }
        let gs = self.cal.channel_params(name, c)?;
        Ok((GridOwner::Owned(Arc::new(BoundaryGrids::new(gs))), self.is_f32(name), None))
    }

    pub(super) fn checkout_scratch(&self) -> ScratchPool {
        self.scratch.lock().ok().and_then(|mut v| v.pop()).unwrap_or_default()
    }

    pub(super) fn checkin_scratch(&self, mut pool: ScratchPool, prev_stats: ScratchStats) {
        let cur = pool.stats();
        let d_reuses = cur.reuses.saturating_sub(prev_stats.reuses);
        let d_allocs = cur.allocs.saturating_sub(prev_stats.allocs);
        if let Ok(mut total) = self.stats.lock() {
            total.reuses += d_reuses;
            total.allocs += d_allocs;
        }
        pool.trim_for_checkin();
        if !self.is_invalidated()
            && let Ok(mut v) = self.scratch.lock()
            && v.len() < MAX_CACHED_POOLS
        {
            v.push(pool);
        }
    }
}
