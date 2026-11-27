//! ByteTrack-style multi-object tracking ([`ByteTracker`]) wired into [`Predictor::predict_sequence`].
//!
//! Each tracked object maintains a decoupled 8-state constant-velocity Kalman filter `[cx, cy, a, h, vx, vy, va, vh]`
//! (`a = w / h`), and each frame runs ByteTrack's signature two-stage optimal IoU association (via an exact
//! zero-dependency Hungarian solver):
//! 1. **Stage 1 (high-confidence):** Confirmed active and coasting lost tracks match detections with
//!    `score >= TrackConfig::high_thresh` (`IoU >= TrackConfig::match_iou`).
//! 2. **Stage 2 (low-confidence occlusion recovery):** Remaining unmatched active tracks match detections with
//!    `low_thresh <= score < high_thresh` (`IoU >= TrackConfig::second_match_iou`), recovering temporarily occluded
//!    objects without spawning false-positive tracks from low-score noise.
//! 3. **Tentative & spawn:** Unconfirmed tracks (when `min_hits > 1`) match remaining high-confidence detections,
//!    and unmatched high-confidence detections with `score >= new_track_thresh` spawn new tracks. Lost tracks coast
//!    up to `track_buffer` missed frames before retiring.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_data::Image;
use serde::{Deserialize, Serialize};

use crate::boxes::{IouKind, Xyxy, iou as box_iou};
use crate::error::{Error, Result};
use crate::nms::Detection;
use crate::predict::{Predictor, Timing, add_scaled_timing, ms, one_canvas};
use crate::tile::check_image;
use crate::tta::run_views_with_oom_fallback;

/// Maximum frames batched per detector pass inside [`Predictor::predict_sequence`].
const SEQUENCE_BATCH: usize = 8;

/// Sentinel cost for disallowed track-detection pairs in the Hungarian assignment matrix.
const DISALLOWED_COST: f64 = 1e6;

/// ByteTrack Kalman position std weight (`1 / 20`).
const STD_WEIGHT_POS: f64 = 1.0 / 20.0;
/// ByteTrack Kalman velocity std weight (`1 / 160`).
const STD_WEIGHT_VEL: f64 = 1.0 / 160.0;

/// Settings for [`ByteTracker`] and [`Predictor::predict_sequence`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackConfig {
    /// Minimum score for high-confidence detections (stage-1 association). Default `0.5`.
    pub high_thresh: f32,
    /// Minimum score for low-confidence detections (stage-2 occlusion recovery). Default `0.1`.
    /// Must satisfy `0.0 <= low_thresh <= high_thresh <= 1.0`.
    pub low_thresh: f32,
    /// Minimum score for an unmatched high-confidence detection to spawn a new track. Default `0.6`.
    /// Must satisfy `high_thresh <= new_track_thresh <= 1.0`.
    pub new_track_thresh: f32,
    /// Minimum IoU for stage-1 association (active & lost tracks vs high-conf detections). Default `0.2`.
    pub match_iou: f32,
    /// Minimum IoU for stage-2 occlusion recovery (unmatched active tracks vs low-conf detections). Default `0.5`.
    pub second_match_iou: f32,
    /// Minimum IoU for tentative (unconfirmed) tracks vs remaining high-conf detections. Default `0.3`.
    pub tentative_match_iou: f32,
    /// Maximum consecutive missed frames a lost track coasts before retiring. Default `30`.
    pub track_buffer: u32,
    /// Consecutive matched detections required before a newly spawned track is confirmed and emitted.
    /// Default `1` (emit immediately on creation frame).
    pub min_hits: u32,
    /// Require identical `class_id` when associating detections with tracks (`true`, the default).
    pub class_aware: bool,
}

impl Default for TrackConfig {
    fn default() -> Self {
        TrackConfig {
            high_thresh: 0.5,
            low_thresh: 0.1,
            new_track_thresh: 0.6,
            match_iou: 0.2,
            second_match_iou: 0.5,
            tentative_match_iou: 0.3,
            track_buffer: 30,
            min_hits: 1,
            class_aware: true,
        }
    }
}

impl TrackConfig {
    /// Validate all thresholds and buffer parameters.
    pub fn validate(&self) -> Result<()> {
        let check_unit = |name: &str, v: f32| -> Result<()> {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(Error::Config(format!("track {name} must be finite and in [0, 1], got {v}")));
            }
            Ok(())
        };
        check_unit("low_thresh", self.low_thresh)?;
        check_unit("high_thresh", self.high_thresh)?;
        check_unit("new_track_thresh", self.new_track_thresh)?;
        check_unit("match_iou", self.match_iou)?;
        check_unit("second_match_iou", self.second_match_iou)?;
        check_unit("tentative_match_iou", self.tentative_match_iou)?;
        if self.low_thresh > self.high_thresh {
            return Err(Error::Config(format!(
                "track low_thresh ({}) must not exceed high_thresh ({})",
                self.low_thresh, self.high_thresh
            )));
        }
        if self.new_track_thresh < self.high_thresh {
            return Err(Error::Config(format!(
                "track new_track_thresh ({}) must be >= high_thresh ({})",
                self.new_track_thresh, self.high_thresh
            )));
        }
        if self.min_hits == 0 {
            return Err(Error::Config("track min_hits must be at least 1".into()));
        }
        Ok(())
    }
}

/// One active tracked detection emitted by [`ByteTracker::update`] or [`Predictor::predict_sequence`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackedDetection {
    /// Persistent monotonically increasing track ID (starts at `1`).
    pub track_id: u64,
    /// Total frames elapsed since this track was created (`1` on the creation frame).
    pub age: u32,
    /// Total matched detection updates accumulated by this track.
    pub hits: u32,
    /// Object class index (`Detection::class`).
    pub class_id: u32,
    /// Latest matched detection confidence score.
    pub score: f32,
    /// Current Kalman-filtered bounding box in image pixel coordinates.
    pub bbox: Xyxy,
}

/// Tracked output of one frame from [`Predictor::predict_sequence`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceFrame {
    /// Zero-based index of this frame in the input slice.
    pub frame_index: usize,
    /// Original frame width in pixels.
    pub width: u32,
    /// Original frame height in pixels.
    pub height: u32,
    /// Active confirmed tracked objects on this frame, sorted by ascending `track_id`.
    pub tracks: Vec<TrackedDetection>,
    /// Per-frame timing breakdown (tracking cost included in `postprocess_ms`).
    pub timing: Timing,
}

/// 1D constant-velocity Kalman state `[position, velocity]` with symmetric 2×2 covariance `[[p00, p01], [p01, p11]]`.
///
/// ByteTrack's 8-state filter `[cx, cy, a, h, vx, vy, va, vh]` has block-diagonal `F`, `H`, `P0`, `Q`, and `R`,
/// so running four independent 2×2 filters is algebraically identical to the full 8×8 matrix update with O(1)
/// flops, zero heap allocation, and guaranteed positive-definite variance.
#[derive(Clone, Copy, Debug)]
struct Kalman1D {
    pos: f64,
    vel: f64,
    p00: f64,
    p01: f64,
    p11: f64,
}

impl Kalman1D {
    #[inline]
    fn new(pos: f64, std_pos: f64, std_vel: f64) -> Self {
        Kalman1D { pos, vel: 0.0, p00: std_pos * std_pos, p01: 0.0, p11: std_vel * std_vel }
    }

    #[inline]
    fn predict(&mut self, q_std_pos: f64, q_std_vel: f64) {
        self.pos += self.vel;
        self.p00 = self.p00 + 2.0 * self.p01 + self.p11 + q_std_pos * q_std_pos;
        self.p01 += self.p11;
        self.p11 += q_std_vel * q_std_vel;
    }

    #[inline]
    fn update(&mut self, z: f64, r_std: f64) {
        let r = r_std * r_std;
        let s = self.p00 + r;
        if s <= 0.0 || !s.is_finite() {
            return;
        }
        let y = z - self.pos;
        let k0 = self.p00 / s;
        let k1 = self.p01 / s;
        self.pos += k0 * y;
        self.vel += k1 * y;
        let one_minus_k0 = (1.0 - k0).max(0.0);
        self.p00 = (one_minus_k0 * self.p00).max(1e-12);
        self.p01 *= one_minus_k0;
        self.p11 = (self.p11 - k1 * self.p01).max(1e-12);
    }
}

/// Decoupled 8-state constant-velocity Kalman box tracker over `(cx, cy, aspect_ratio, height)`.
#[derive(Clone, Copy, Debug)]
struct KalmanBox {
    cx: Kalman1D,
    cy: Kalman1D,
    a: Kalman1D,
    h: Kalman1D,
}

impl KalmanBox {
    fn from_bbox(b: Xyxy) -> Self {
        let w = (b.x2 as f64 - b.x1 as f64).max(1e-3);
        let h = (b.y2 as f64 - b.y1 as f64).max(1e-3);
        let cx = b.x1 as f64 + 0.5 * w;
        let cy = b.y1 as f64 + 0.5 * h;
        let a = w / h;
        let sp = 2.0 * STD_WEIGHT_POS * h;
        let sv = 10.0 * STD_WEIGHT_VEL * h;
        KalmanBox {
            cx: Kalman1D::new(cx, sp, sv),
            cy: Kalman1D::new(cy, sp, sv),
            a: Kalman1D::new(a, 1e-2, 1e-5),
            h: Kalman1D::new(h, sp, sv),
        }
    }

    fn predict(&mut self, coasting: bool) {
        if coasting || self.h.pos + self.h.vel <= 1e-3 {
            // ByteTrack guard: dampen height velocity while coasting or if predicted height would collapse.
            self.h.vel = 0.0;
        }
        let h_ref = self.h.pos.max(1.0);
        let qp = STD_WEIGHT_POS * h_ref;
        let qv = STD_WEIGHT_VEL * h_ref;
        self.cx.predict(qp, qv);
        self.cy.predict(qp, qv);
        self.a.predict(1e-2, 1e-5);
        self.h.predict(qp, qv);
        self.h.pos = self.h.pos.max(1e-3);
        self.a.pos = self.a.pos.max(1e-3);
    }

    fn update(&mut self, b: Xyxy) {
        let w = (b.x2 as f64 - b.x1 as f64).max(1e-3);
        let h = (b.y2 as f64 - b.y1 as f64).max(1e-3);
        let cx = b.x1 as f64 + 0.5 * w;
        let cy = b.y1 as f64 + 0.5 * h;
        let a = w / h;
        let r = STD_WEIGHT_POS * self.h.pos.max(1.0);
        self.cx.update(cx, r);
        self.cy.update(cy, r);
        self.a.update(a, 1e-1);
        self.h.update(h, r);
        self.h.pos = self.h.pos.max(1e-3);
        self.a.pos = self.a.pos.max(1e-3);
    }

    fn bbox(&self) -> Xyxy {
        let h = self.h.pos.max(1e-3);
        let w = (self.a.pos.max(1e-3) * h).max(1e-3);
        let x1 = (self.cx.pos - 0.5 * w) as f32;
        let y1 = (self.cy.pos - 0.5 * h) as f32;
        let x2 = (self.cx.pos + 0.5 * w) as f32;
        let y2 = (self.cy.pos + 0.5 * h) as f32;
        Xyxy::new(x1, y1, x2, y2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackState {
    Tentative,
    Tracked,
    Lost,
}

#[derive(Clone, Debug)]
struct Track {
    id: u64,
    kalman: KalmanBox,
    class_id: u32,
    score: f32,
    age: u32,
    hits: u32,
    time_since_update: u32,
    state: TrackState,
}

impl Track {
    fn to_detection(&self) -> TrackedDetection {
        TrackedDetection {
            track_id: self.id,
            age: self.age,
            hits: self.hits,
            class_id: self.class_id,
            score: self.score,
            bbox: self.kalman.bbox(),
        }
    }
}

/// Exact minimum-cost linear assignment (Kuhn-Munkres / Hungarian algorithm) for rectangular `n_rows × n_cols`
/// matrices. Returns matched `(row, col)` pairs sorted by ascending `row`.
fn hungarian(n_rows: usize, n_cols: usize, costs: &[f64]) -> Vec<(usize, usize)> {
    if n_rows == 0 || n_cols == 0 {
        return Vec::new();
    }
    if n_rows > n_cols {
        let mut transposed = vec![0.0f64; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                transposed[c * n_rows + r] = costs[r * n_cols + c];
            }
        }
        let mut pairs: Vec<(usize, usize)> =
            hungarian(n_cols, n_rows, &transposed).into_iter().map(|(c, r)| (r, c)).collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        return pairs;
    }

    let mut u = vec![0.0f64; n_rows + 1];
    let mut v = vec![0.0f64; n_cols + 1];
    let mut p = vec![0usize; n_cols + 1];
    let mut way = vec![0usize; n_cols + 1];
    let mut minv = vec![0.0f64; n_cols + 1];
    let mut used = vec![false; n_cols + 1];

    for i in 1..=n_rows {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(f64::INFINITY);
        used.fill(false);
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            let row_off = (i0 - 1) * n_cols;
            for j in 1..=n_cols {
                if !used[j] {
                    let cur = costs[row_off + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n_cols {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs = Vec::with_capacity(n_rows);
    for (j, &row) in p.iter().enumerate().skip(1) {
        if row != 0 {
            pairs.push((row - 1, j - 1));
        }
    }
    pairs.sort_unstable_by_key(|&(r, _)| r);
    pairs
}

/// Associate `tracks` with `dets` by optimal IoU matching (`IoU >= min_iou`, class-matched when `class_aware`).
/// Returns `(matched_pairs, unmatched_track_positions, unmatched_det_positions)`.
fn associate(
    tracks: &[&Track],
    dets: &[&Detection],
    min_iou: f32,
    class_aware: bool,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let nt = tracks.len();
    let nd = dets.len();
    if nt == 0 || nd == 0 {
        return (Vec::new(), (0..nt).collect(), (0..nd).collect());
    }
    let track_boxes: Vec<Xyxy> = tracks.iter().map(|t| t.kalman.bbox()).collect();
    let mut costs = vec![DISALLOWED_COST; nt * nd];
    let mut any_valid = false;
    for (ti, (track, tb)) in tracks.iter().zip(&track_boxes).enumerate() {
        for (di, det) in dets.iter().enumerate() {
            if class_aware && track.class_id != det.class {
                continue;
            }
            let iou = box_iou(tb, &det.bbox, IouKind::Iou);
            if iou.is_finite() && iou >= min_iou {
                costs[ti * nd + di] = (1.0 - iou as f64).clamp(0.0, 1.0);
                any_valid = true;
            }
        }
    }
    if !any_valid {
        return (Vec::new(), (0..nt).collect(), (0..nd).collect());
    }

    let assigned = hungarian(nt, nd, &costs);
    let mut matched = Vec::with_capacity(assigned.len());
    let mut t_used = vec![false; nt];
    let mut d_used = vec![false; nd];
    for (ti, di) in assigned {
        if costs[ti * nd + di] < DISALLOWED_COST * 0.5 {
            matched.push((ti, di));
            t_used[ti] = true;
            d_used[di] = true;
        }
    }
    let unmatched_t = (0..nt).filter(|&i| !t_used[i]).collect();
    let unmatched_d = (0..nd).filter(|&i| !d_used[i]).collect();
    (matched, unmatched_t, unmatched_d)
}

/// Deterministic two-stage ByteTrack multi-object tracker.
#[derive(Clone, Debug)]
pub struct ByteTracker {
    cfg: TrackConfig,
    frame_id: u64,
    next_id: u64,
    tracked: Vec<Track>,
    tentative: Vec<Track>,
    lost: Vec<Track>,
}

impl Default for ByteTracker {
    fn default() -> Self {
        ByteTracker::new(TrackConfig::default()).expect("default TrackConfig is valid")
    }
}

impl ByteTracker {
    /// Create a new tracker with validated [`TrackConfig`].
    pub fn new(cfg: TrackConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(ByteTracker {
            cfg,
            frame_id: 0,
            next_id: 1,
            tracked: Vec::new(),
            tentative: Vec::new(),
            lost: Vec::new(),
        })
    }

    /// The tracker's configuration.
    pub fn config(&self) -> &TrackConfig {
        &self.cfg
    }

    /// Number of frames processed since creation or the last [`ByteTracker::reset`].
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Number of currently active confirmed tracks.
    pub fn active_count(&self) -> usize {
        self.tracked.len()
    }

    /// Number of currently coasting lost tracks in the recovery buffer.
    pub fn lost_count(&self) -> usize {
        self.lost.len()
    }

    /// Clear all active, tentative, and lost tracks and reset frame and track ID counters.
    pub fn reset(&mut self) {
        self.frame_id = 0;
        self.next_id = 1;
        self.tracked.clear();
        self.tentative.clear();
        self.lost.clear();
    }

    /// Process one frame's detections and return all active confirmed [`TrackedDetection`]s sorted by ascending
    /// `track_id`.
    ///
    /// Input detection order does not affect output (detections are canonically sorted before association), and
    /// detections with non-finite scores or non-finite/degenerate boxes are ignored cleanly.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedDetection> {
        self.frame_id += 1;

        // Clean and canonically sort detections so association is independent of input order.
        let mut valid: Vec<Detection> = detections
            .iter()
            .copied()
            .filter(|d| {
                d.score.is_finite()
                    && d.score >= self.cfg.low_thresh
                    && d.bbox.x1.is_finite()
                    && d.bbox.y1.is_finite()
                    && d.bbox.x2.is_finite()
                    && d.bbox.y2.is_finite()
                    && d.bbox.x2 > d.bbox.x1
                    && d.bbox.y2 > d.bbox.y1
            })
            .collect();
        valid.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.class.cmp(&b.class))
                .then_with(|| a.bbox.x1.total_cmp(&b.bbox.x1))
                .then_with(|| a.bbox.y1.total_cmp(&b.bbox.y1))
                .then_with(|| a.bbox.x2.total_cmp(&b.bbox.x2))
                .then_with(|| a.bbox.y2.total_cmp(&b.bbox.y2))
        });

        let split = valid.partition_point(|d| d.score >= self.cfg.high_thresh);
        let high_dets: Vec<&Detection> = valid[..split].iter().collect();
        let low_dets: Vec<&Detection> = valid[split..].iter().collect();

        // Advance Kalman state and age for all existing tracks.
        for t in &mut self.tracked {
            t.age += 1;
            t.kalman.predict(false);
        }
        for t in &mut self.tentative {
            t.age += 1;
            t.kalman.predict(false);
        }
        for t in &mut self.lost {
            t.age += 1;
            t.kalman.predict(true);
        }

        // Stage 1: match confirmed active + coasting lost tracks against high-confidence detections.
        let mut pool: Vec<Track> = Vec::with_capacity(self.tracked.len() + self.lost.len());
        pool.append(&mut self.tracked);
        pool.append(&mut self.lost);
        pool.sort_unstable_by_key(|t| t.id);

        let pool_refs: Vec<&Track> = pool.iter().collect();
        let (matched1, unmatched_pool, unmatched_high_idx) =
            associate(&pool_refs, &high_dets, self.cfg.match_iou, self.cfg.class_aware);

        let mut next_tracked: Vec<Track> = Vec::new();
        let mut next_lost: Vec<Track> = Vec::new();
        let mut pool_slots: Vec<Option<Track>> = pool.into_iter().map(Some).collect();

        for (ti, di) in matched1 {
            let mut tr = pool_slots[ti].take().expect("unique track match");
            let det = high_dets[di];
            tr.kalman.update(det.bbox);
            tr.hits += 1;
            tr.score = det.score;
            tr.time_since_update = 0;
            if !self.cfg.class_aware {
                tr.class_id = det.class;
            }
            tr.state = TrackState::Tracked;
            next_tracked.push(tr);
        }

        let mut remain_active: Vec<Track> = Vec::new();
        for ti in unmatched_pool {
            let mut tr = pool_slots[ti].take().expect("unmatched track");
            match tr.state {
                TrackState::Tracked => remain_active.push(tr),
                TrackState::Lost => {
                    tr.time_since_update += 1;
                    if tr.time_since_update <= self.cfg.track_buffer {
                        next_lost.push(tr);
                    }
                }
                TrackState::Tentative => {}
            }
        }

        // Stage 2 (ByteTrack): recover occluded active tracks from low-confidence detections.
        let remain_refs: Vec<&Track> = remain_active.iter().collect();
        let (matched2, unmatched_active, _) =
            associate(&remain_refs, &low_dets, self.cfg.second_match_iou, self.cfg.class_aware);
        let mut active_slots: Vec<Option<Track>> = remain_active.into_iter().map(Some).collect();

        for (ti, di) in matched2 {
            let mut tr = active_slots[ti].take().expect("unique stage-2 match");
            let det = low_dets[di];
            tr.kalman.update(det.bbox);
            tr.hits += 1;
            tr.score = det.score;
            tr.time_since_update = 0;
            if !self.cfg.class_aware {
                tr.class_id = det.class;
            }
            tr.state = TrackState::Tracked;
            next_tracked.push(tr);
        }
        for ti in unmatched_active {
            let mut tr = active_slots[ti].take().expect("unmatched active track");
            tr.state = TrackState::Lost;
            tr.time_since_update = 1;
            if tr.time_since_update <= self.cfg.track_buffer {
                next_lost.push(tr);
            }
        }

        // Stage 3: match tentative (unconfirmed) tracks against remaining high-confidence detections.
        let rem_high: Vec<&Detection> = unmatched_high_idx.iter().map(|&i| high_dets[i]).collect();
        let mut prev_tent = std::mem::take(&mut self.tentative);
        prev_tent.sort_unstable_by_key(|t| t.id);
        let tent_refs: Vec<&Track> = prev_tent.iter().collect();
        let (matched3, _, unmatched_rem_high) =
            associate(&tent_refs, &rem_high, self.cfg.tentative_match_iou, self.cfg.class_aware);
        let mut tent_slots: Vec<Option<Track>> = prev_tent.into_iter().map(Some).collect();
        let mut next_tentative: Vec<Track> = Vec::new();

        for (ti, di) in matched3 {
            let mut tr = tent_slots[ti].take().expect("unique tentative match");
            let det = rem_high[di];
            tr.kalman.update(det.bbox);
            tr.hits += 1;
            tr.score = det.score;
            tr.time_since_update = 0;
            if !self.cfg.class_aware {
                tr.class_id = det.class;
            }
            if tr.hits >= self.cfg.min_hits {
                tr.state = TrackState::Tracked;
                next_tracked.push(tr);
            } else {
                next_tentative.push(tr);
            }
        }

        // Spawn new tracks for remaining unmatched high-confidence detections above `new_track_thresh`.
        for di in unmatched_rem_high {
            let det = rem_high[di];
            if det.score < self.cfg.new_track_thresh {
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            let confirmed = self.cfg.min_hits <= 1;
            let tr = Track {
                id,
                kalman: KalmanBox::from_bbox(det.bbox),
                class_id: det.class,
                score: det.score,
                age: 1,
                hits: 1,
                time_since_update: 0,
                state: if confirmed { TrackState::Tracked } else { TrackState::Tentative },
            };
            if confirmed {
                next_tracked.push(tr);
            } else {
                next_tentative.push(tr);
            }
        }

        next_tracked.sort_unstable_by_key(|t| t.id);
        next_tentative.sort_unstable_by_key(|t| t.id);
        next_lost.sort_unstable_by_key(|t| t.id);

        let out: Vec<TrackedDetection> = next_tracked.iter().map(Track::to_detection).collect();
        self.tracked = next_tracked;
        self.tentative = next_tentative;
        self.lost = next_lost;
        out
    }
}

impl Predictor {
    /// Detect and track objects across an ordered sequence of frames using [`ByteTracker`].
    ///
    /// Frames run in bounded batches (up to 8 frames per forward pass, inheriting CPU/int8/GPU/`f16`/`o2o`
    /// fast paths and GPU OOM half-batch fallback) with the detector's confidence threshold lowered to
    /// `self.config().conf.min(track_cfg.low_thresh)` so ByteTrack's second-stage low-confidence occlusion
    /// recovery receives low-score candidates even when `PredictConfig::conf` is higher, without mutating
    /// `self`. Tracker updates execute strictly in frame order.
    pub fn predict_sequence(&self, frames: &[Image], track_cfg: &TrackConfig) -> Result<Vec<SequenceFrame>> {
        let mut tracker = ByteTracker::new(track_cfg.clone())?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        for frame in frames {
            check_image(frame)?;
        }

        let det_conf = self.config().conf.min(track_cfg.low_thresh);
        let imgsz = self.config().imgsz;
        let mut out = Vec::with_capacity(frames.len());
        let mut frame_index = 0usize;

        for chunk in frames.chunks(SEQUENCE_BATCH) {
            let refs: Vec<&Image> = chunk.iter().collect();
            let rect = self.config().rect && one_canvas(chunk, imgsz);
            let t_chunk = Instant::now();
            let preds = run_views_with_oom_fallback(&refs, &mut |sub| {
                self.run_with_imgsz_conf(sub, rect, imgsz, Some(det_conf))
            })?;
            let chunk_wall = ms(t_chunk);
            let inv_n = 1.0 / chunk.len() as f64;
            let (mut b_pre, mut b_inf, mut b_post) = (0.0f64, 0.0f64, 0.0f64);
            for p in &preds {
                b_pre += p.timing.preprocess_ms;
                b_inf += p.timing.inference_ms;
                b_post += p.timing.postprocess_ms;
            }
            let (mut pre_ms, mut inf_ms, mut post_ms) = (0.0f64, 0.0f64, 0.0f64);
            add_scaled_timing(
                &mut pre_ms,
                &mut inf_ms,
                &mut post_ms,
                b_pre,
                b_inf,
                b_post,
                chunk_wall * inv_n,
            );

            for pred in preds {
                let t_trk = Instant::now();
                let tracks = tracker.update(&pred.detections);
                let trk_ms = ms(t_trk);
                out.push(SequenceFrame {
                    frame_index,
                    width: pred.width,
                    height: pred.height,
                    tracks,
                    timing: Timing::new(pre_ms, inf_ms, post_ms + trk_ms),
                });
                frame_index += 1;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
        Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
    }

    #[test]
    fn config_validation_rejects_invalid_thresholds() {
        assert!(TrackConfig::default().validate().is_ok());
        assert!(TrackConfig { low_thresh: -0.1, ..Default::default() }.validate().is_err());
        assert!(TrackConfig { high_thresh: 1.1, ..Default::default() }.validate().is_err());
        assert!(TrackConfig { low_thresh: 0.6, high_thresh: 0.5, ..Default::default() }.validate().is_err());
        assert!(
            TrackConfig { high_thresh: 0.5, new_track_thresh: 0.4, ..Default::default() }.validate().is_err()
        );
        assert!(TrackConfig { match_iou: f32::NAN, ..Default::default() }.validate().is_err());
        assert!(TrackConfig { min_hits: 0, ..Default::default() }.validate().is_err());
    }

    #[test]
    fn hungarian_matches_bruteforce_optimum_on_rectangular_matrices() {
        fn brute_best(r: usize, c: usize, costs: &[f64]) -> f64 {
            fn dfs(
                row: usize,
                r: usize,
                c: usize,
                costs: &[f64],
                used: &mut [bool],
                acc: f64,
                best: &mut f64,
            ) {
                if row == r {
                    if acc < *best {
                        *best = acc;
                    }
                    return;
                }
                for col in 0..c {
                    if !used[col] {
                        used[col] = true;
                        dfs(row + 1, r, c, costs, used, acc + costs[row * c + col], best);
                        used[col] = false;
                    }
                }
            }
            let mut best = f64::INFINITY;
            if r <= c {
                let mut used = vec![false; c];
                dfs(0, r, c, costs, &mut used, 0.0, &mut best);
            } else {
                let mut t = vec![0.0; r * c];
                for i in 0..r {
                    for j in 0..c {
                        t[j * r + i] = costs[i * c + j];
                    }
                }
                let mut used = vec![false; r];
                dfs(0, c, r, &t, &mut used, 0.0, &mut best);
            }
            best
        }

        let mut seed = 0x517c_c1b7_2722_0a95u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for &(r, c) in &[(1, 1), (2, 3), (3, 2), (4, 4), (3, 5), (5, 3)] {
            for _ in 0..25 {
                let costs: Vec<f64> =
                    (0..r * c).map(|_| if next() < 0.25 { DISALLOWED_COST } else { next() }).collect();
                let got = hungarian(r, c, &costs);
                assert_eq!(got.len(), r.min(c));
                let got_cost: f64 = got.iter().map(|&(ri, ci)| costs[ri * c + ci]).sum();
                let want_cost = brute_best(r, c, &costs);
                assert!((got_cost - want_cost).abs() < 1e-9, "{r}x{c}: got {got_cost} vs {want_cost}");
            }
        }
    }

    #[test]
    fn second_stage_recovers_occluded_object_from_low_confidence_detection() {
        let mut trk = ByteTracker::new(TrackConfig {
            high_thresh: 0.5,
            low_thresh: 0.1,
            new_track_thresh: 0.6,
            match_iou: 0.2,
            second_match_iou: 0.5,
            ..Default::default()
        })
        .unwrap();

        // Frame 1: high-confidence object spawns track 1; lone low-conf noise must NOT spawn a track.
        let f1 =
            trk.update(&[det(100.0, 100.0, 160.0, 180.0, 0.85, 0), det(300.0, 300.0, 350.0, 350.0, 0.25, 0)]);
        assert_eq!(f1.len(), 1);
        assert_eq!((f1[0].track_id, f1[0].age, f1[0].hits), (1, 1, 1));

        // Frames 2-4: object moves steadily right (+10px/frame), confidence drops to 0.20 (partial occlusion).
        for step in 1..=3 {
            let x = 100.0 + step as f32 * 10.0;
            let out = trk.update(&[det(x, 100.0, x + 60.0, 180.0, 0.20, 0)]);
            assert_eq!(out.len(), 1, "stage-2 must keep track 1 alive at low confidence (step {step})");
            assert_eq!(out[0].track_id, 1);
            assert_eq!(out[0].age, 1 + step);
            assert_eq!(out[0].hits, 1 + step);
            assert!((out[0].score - 0.20).abs() < 1e-6);
        }

        // Frame 5: object emerges back at high confidence.
        let f5 = trk.update(&[det(140.0, 100.0, 200.0, 180.0, 0.90, 0)]);
        assert_eq!(f5.len(), 1);
        assert_eq!((f5[0].track_id, f5[0].age, f5[0].hits), (1, 5, 5));
    }

    #[test]
    fn coasting_reidentification_crossing_stability_and_reset() {
        let mut trk = ByteTracker::new(TrackConfig { track_buffer: 3, ..Default::default() }).unwrap();

        // Two objects moving toward and crossing each other with constant velocity.
        // Track A moves right (+12px/frame), Track B moves left (-12px/frame), slightly offset vertically.
        let mut id_a = 0u64;
        let mut id_b = 0u64;
        for f in 0..8 {
            let ax = 50.0 + f as f32 * 12.0;
            let bx = 170.0 - f as f32 * 12.0;
            // Deliberately alternate input detection order every frame to verify order invariance.
            let mut frame_dets =
                vec![det(ax, 100.0, ax + 50.0, 160.0, 0.90, 0), det(bx, 110.0, bx + 50.0, 170.0, 0.88, 0)];
            if f % 2 == 1 {
                frame_dets.reverse();
            }
            let out = trk.update(&frame_dets);
            assert_eq!(out.len(), 2);
            if f == 0 {
                id_a = out.iter().find(|t| t.bbox.x1 < 100.0).unwrap().track_id;
                id_b = out.iter().find(|t| t.bbox.x1 > 100.0).unwrap().track_id;
                assert_ne!(id_a, id_b);
            } else {
                let ta = out.iter().find(|t| t.track_id == id_a).unwrap();
                let tb = out.iter().find(|t| t.track_id == id_b).unwrap();
                assert!((ta.bbox.x1 - ax).abs() < 15.0, "frame {f}: track A drifted {:?}", ta.bbox);
                assert!((tb.bbox.x1 - bx).abs() < 15.0, "frame {f}: track B drifted {:?}", tb.bbox);
            }
        }

        // Complete 2-frame occlusion (empty frames), then reappearance along predicted trajectory -> same IDs!
        assert!(trk.update(&[]).is_empty());
        assert_eq!(trk.lost_count(), 2);
        assert!(trk.update(&[]).is_empty());
        assert_eq!(trk.lost_count(), 2);

        let f_reappear = 10;
        let ax = 50.0 + f_reappear as f32 * 12.0;
        let bx = 170.0 - f_reappear as f32 * 12.0;
        let re = trk
            .update(&[det(ax, 100.0, ax + 50.0, 160.0, 0.91, 0), det(bx, 110.0, bx + 50.0, 170.0, 0.89, 0)]);
        assert_eq!(re.len(), 2);
        assert!(re.iter().any(|t| t.track_id == id_a && t.hits == 9 && t.age == 11));
        assert!(re.iter().any(|t| t.track_id == id_b && t.hits == 9 && t.age == 11));

        // Exceed track_buffer (3 missed frames -> retired on 4th missed frame).
        for _ in 0..4 {
            assert!(trk.update(&[]).is_empty());
        }
        assert_eq!(trk.active_count(), 0);
        assert_eq!(trk.lost_count(), 0);

        // Reset restores frame_id = 0 and next_id = 1.
        trk.reset();
        assert_eq!(trk.frame_id(), 0);
        let fresh = trk.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9, 1)]);
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].track_id, 1);
        assert_eq!(fresh[0].class_id, 1);
    }

    #[test]
    fn min_hits_and_class_awareness_and_non_finite_filtering() {
        let mut trk =
            ByteTracker::new(TrackConfig { min_hits: 2, class_aware: true, ..Default::default() }).unwrap();
        let b = det(20.0, 20.0, 80.0, 80.0, 0.9, 0);
        let bad_nan = det(20.0, 20.0, 80.0, 80.0, f32::NAN, 0);
        let bad_deg = det(80.0, 20.0, 20.0, 80.0, 0.9, 0);

        // Frame 1: tentative, not yet emitted.
        assert!(trk.update(&[b, bad_nan, bad_deg]).is_empty());
        // Frame 2: different class on same box does NOT confirm class-0 tentative track.
        assert!(trk.update(&[det(20.0, 20.0, 80.0, 80.0, 0.9, 1)]).is_empty());
        // Frame 3: class-1 second hit confirms class-1 track!
        let f3 = trk.update(&[det(20.0, 20.0, 80.0, 80.0, 0.9, 1)]);
        assert_eq!(f3.len(), 1);
        assert_eq!((f3[0].class_id, f3[0].hits, f3[0].age), (1, 2, 2));
    }
}
