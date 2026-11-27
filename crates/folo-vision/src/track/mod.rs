//! ByteTrack-style multi-object tracking ([`ByteTracker`]) wired into [`Predictor::predict_sequence`].
//!
//! Each tracked object maintains a decoupled 8-state constant-velocity Kalman filter `[cx, cy, a, h, vx, vy, va, vh]`
//! ([`kalman`]), and each frame runs ByteTrack's signature two-stage optimal IoU association via a sparse-pruned
//! zero-dependency Hungarian solver ([`assign`]):
//! 1. **Stage 1 (high-confidence):** Confirmed active and coasting lost tracks match detections with
//!    `score >= TrackConfig::high_thresh` (`IoU >= TrackConfig::match_iou`).
//! 2. **Stage 2 (low-confidence occlusion recovery):** Remaining unmatched active tracks match detections with
//!    `low_thresh <= score < high_thresh` (`IoU >= TrackConfig::second_match_iou`), recovering temporarily occluded
//!    objects without spawning false-positive tracks from low-score noise.
//! 3. **Tentative & spawn:** Unconfirmed tracks (when `min_hits > 1`) match remaining high-confidence detections,
//!    and unmatched high-confidence detections with `score >= new_track_thresh` spawn new tracks. Lost tracks coast
//!    up to `track_buffer` missed frames before retiring.

mod assign;
mod kalman;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_data::{Image, Letterbox, LetterboxOptions};
use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::fusion::ScoreRule;
use crate::model::HeadMode;
use crate::nms::Detection;
use crate::predict::{Predictor, Timing, add_scaled_timing, ms};
use crate::tile::check_image;
use crate::tta::run_views_with_oom_fallback;

use self::assign::associate;
use self::kalman::KalmanBox;

/// Maximum consecutive same-canvas frames batched per detector pass inside [`Predictor::predict_sequence`].
const SEQUENCE_BATCH: usize = 8;

/// Logit headroom subtracted from `low_thresh` when pre-extracting raw candidates for calibrated `ScoreRule`s
/// (`Elc` / `DflConsensus` / `DflRepulse` / `DflCgr`) so clusters that calibrate upward past `low_thresh` are
/// preserved before tracking.
const CALIBRATED_LOGIT_HEADROOM: f64 = 1.5;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrackState {
    Tentative,
    Tracked,
    Lost,
}

#[derive(Clone, Debug)]
pub(crate) struct Track {
    pub(crate) id: u64,
    pub(crate) kalman: KalmanBox,
    pub(crate) class_id: u32,
    pub(crate) score: f32,
    pub(crate) age: u32,
    pub(crate) hits: u32,
    pub(crate) time_since_update: u32,
    pub(crate) state: TrackState,
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

/// Raw pre-NMS detector confidence floor for [`Predictor::predict_sequence`]:
/// - Uses `target.next_down()` when `target > 0.0` so boundary detections with `score == low_thresh` survive
///   the detector's strict `>` candidate filter (both `o2m` and `o2o`) before `ByteTracker::update`'s `>=` check.
/// - For calibrated `ScoreRule`s (`Elc` / `DflConsensus` / `DflRepulse` / `DflCgr`) on `o2m`, lowers the pre-NMS
///   floor by `CALIBRATED_LOGIT_HEADROOM` in logit space so clusters that calibrate upward past `low_thresh` are
///   not pruned prematurely; `ByteTracker::update` still filters final calibrated scores by `>= low_thresh`.
pub(crate) fn sequence_det_conf(
    pred_conf: f32,
    low_thresh: f32,
    score_rule: ScoreRule,
    head_mode: HeadMode,
) -> f32 {
    let target = pred_conf.min(low_thresh);
    if target <= 0.0 {
        return 0.0;
    }
    let strict_floor = target.next_down().max(0.0);
    if head_mode == HeadMode::O2m && score_rule.is_calibrated() {
        let p = (target as f64).clamp(1e-6, 1.0 - 1e-6);
        let logit = (p / (1.0 - p)).ln() - CALIBRATED_LOGIT_HEADROOM;
        let lowered = (1.0 / (1.0 + (-logit).exp())) as f32;
        lowered.min(strict_floor).max(0.0)
    } else {
        strict_floor
    }
}

/// Split `frames` into consecutive runs of up to `SEQUENCE_BATCH` frames that share the exact same `rect`
/// letterbox canvas (or any consecutive frames when `rect == false`). Because runs never mix incompatible
/// `rect` canvases, every frame's letterbox matches single-image [`Predictor::predict`] regardless of later
/// frames' dimensions.
fn canvas_runs(frames: &[Image], imgsz: u32, rect: bool) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let opts = LetterboxOptions { stride: Some(32), ..LetterboxOptions::default() };
    let mut start = 0usize;
    while start < frames.len() {
        let first_canvas = rect
            .then(|| Letterbox::compute(frames[start].width, frames[start].height, imgsz, imgsz, &opts).out);
        let mut end = start + 1;
        while end < frames.len() && end - start < SEQUENCE_BATCH {
            if let Some(c0) = first_canvas {
                let c = Letterbox::compute(frames[end].width, frames[end].height, imgsz, imgsz, &opts).out;
                if c != c0 {
                    break;
                }
            }
            end += 1;
        }
        runs.push((start, end));
        start = end;
    }
    runs
}

impl Predictor {
    /// Detect and track objects across an ordered sequence of frames using [`ByteTracker`].
    ///
    /// Consecutive frames sharing the same letterbox canvas run in bounded batches (up to 8 frames per forward
    /// pass, inheriting CPU/int8/GPU/`f16`/`o2o` fast paths and GPU OOM half-batch fallback) so every frame uses
    /// the exact canvas of [`Predictor::predict`] independent of other frames' dimensions. The detector's raw
    /// candidate threshold is lowered internally for `track_cfg.low_thresh` (including strict-boundary and
    /// calibrated `ScoreRule` headroom) without mutating `self`, and tracker updates execute strictly in frame
    /// order.
    pub fn predict_sequence(&self, frames: &[Image], track_cfg: &TrackConfig) -> Result<Vec<SequenceFrame>> {
        let mut tracker = ByteTracker::new(track_cfg.clone())?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        for frame in frames {
            check_image(frame)?;
        }

        let score_rule = self.config().score;
        let head_mode = self.detector().head_mode();
        let conf_override = crate::predict::ConfOverride {
            conf: sequence_det_conf(self.config().conf, track_cfg.low_thresh, score_rule, head_mode),
            per_class_conf: self.per_class_conf().map(|map| {
                map.iter()
                    .map(|(&c, &v)| (c, sequence_det_conf(v, track_cfg.low_thresh, score_rule, head_mode)))
                    .collect()
            }),
            conf_inclusive: false,
        };
        let imgsz = self.config().imgsz;
        let rect = self.config().rect;
        let mut out = Vec::with_capacity(frames.len());
        let mut frame_index = 0usize;

        for (start, end) in canvas_runs(frames, imgsz, rect) {
            let chunk = &frames[start..end];
            let refs: Vec<&Image> = chunk.iter().collect();
            let t_chunk = Instant::now();
            let preds = run_views_with_oom_fallback(&refs, &mut |sub| {
                self.run_with_imgsz_conf(sub, rect, imgsz, Some(&conf_override))
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
    fn second_stage_recovers_occluded_object_including_exact_kalman_velocity_gate() {
        let mut trk = ByteTracker::new(TrackConfig::default()).unwrap();

        // Regression for prior-p01 Kalman update: 60x80 box centered at x = 100, 110, ..., 160, followed by a
        // low-confidence detection centered at x = 149 (must pass default stage-2 IoU gate 0.5 and keep ID 1).
        for step in 0..=6 {
            let cx = 100.0 + step as f32 * 10.0;
            let out = trk.update(&[
                det(cx - 30.0, 60.0, cx + 30.0, 140.0, 0.90, 0),
                det(400.0, 400.0, 450.0, 450.0, 0.20, 0), // lone low-conf noise must never spawn
            ]);
            assert_eq!(out.len(), 1);
            assert_eq!((out[0].track_id, out[0].age, out[0].hits), (1, step + 1, step + 1));
        }

        // Low-confidence occluded detection at cx = 149 (score 0.20 in [0.1, 0.5)).
        let recovered = trk.update(&[det(149.0 - 30.0, 60.0, 149.0 + 30.0, 140.0, 0.20, 0)]);
        assert_eq!(recovered.len(), 1, "stage-2 must retain ID 1 under exact Kalman prediction");
        assert_eq!((recovered[0].track_id, recovered[0].age, recovered[0].hits), (1, 8, 8));
        assert!((recovered[0].score - 0.20).abs() < 1e-6);
    }

    #[test]
    fn coasting_reidentification_crossing_stability_and_reset() {
        let mut trk = ByteTracker::new(TrackConfig { track_buffer: 3, ..Default::default() }).unwrap();

        let mut id_a = 0u64;
        let mut id_b = 0u64;
        for f in 0..8 {
            let ax = 50.0 + f as f32 * 12.0;
            let bx = 170.0 - f as f32 * 12.0;
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

        // Complete 2-frame occlusion, then reappearance along trajectory -> same IDs!
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
        assert_eq!((trk.active_count(), trk.lost_count()), (0, 0));

        // Reset restores frame_id = 0 and next_id = 1.
        trk.reset();
        assert_eq!(trk.frame_id(), 0);
        let fresh = trk.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9, 1)]);
        assert_eq!(fresh.len(), 1);
        assert_eq!((fresh[0].track_id, fresh[0].class_id), (1, 1));
    }

    #[test]
    fn min_hits_class_agnostic_track_buffer_zero_and_det_conf_floor() {
        // Class-aware + min_hits = 2 + NaN/degenerate filtering.
        let mut trk =
            ByteTracker::new(TrackConfig { min_hits: 2, class_aware: true, ..Default::default() }).unwrap();
        let b = det(20.0, 20.0, 80.0, 80.0, 0.9, 0);
        let bad_nan = det(20.0, 20.0, 80.0, 80.0, f32::NAN, 0);
        let bad_deg = det(80.0, 20.0, 20.0, 80.0, 0.9, 0);
        assert!(trk.update(&[b, bad_nan, bad_deg]).is_empty());
        assert!(trk.update(&[det(20.0, 20.0, 80.0, 80.0, 0.9, 1)]).is_empty());
        let f3 = trk.update(&[det(20.0, 20.0, 80.0, 80.0, 0.9, 1)]);
        assert_eq!(f3.len(), 1);
        assert_eq!((f3[0].class_id, f3[0].hits, f3[0].age), (1, 2, 2));

        // Class-agnostic tracking updates class_id across stages and track_buffer = 0 retires immediately on miss.
        let mut ag =
            ByteTracker::new(TrackConfig { class_aware: false, track_buffer: 0, ..Default::default() })
                .unwrap();
        let t1 = ag.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9, 0)]);
        assert_eq!((t1[0].track_id, t1[0].class_id), (1, 0));
        // Stage-2 low-conf detection with different class updates class_id and keeps ID 1.
        let t2 = ag.update(&[det(12.0, 10.0, 62.0, 60.0, 0.25, 2)]);
        assert_eq!((t2.len(), t2[0].track_id, t2[0].class_id), (1, 1, 2));
        // One missed frame immediately retires when track_buffer == 0.
        assert!(ag.update(&[]).is_empty());
        assert_eq!((ag.active_count(), ag.lost_count()), (0, 0));

        // sequence_det_conf preserves strict boundary `score == low_thresh` and calibrated ScoreRule headroom.
        assert_eq!(sequence_det_conf(0.25, 0.0, ScoreRule::None, HeadMode::O2m), 0.0);
        let f_o2o = sequence_det_conf(0.25, 0.1, ScoreRule::None, HeadMode::O2o);
        assert!((0.0..0.1).contains(&f_o2o));
        let f_elc = sequence_det_conf(0.25, 0.1, ScoreRule::Elc, HeadMode::O2m);
        assert!(f_elc < 0.05 && f_elc > 0.0, "ELC pre-NMS floor should leave headroom below 0.1: {f_elc}");
    }
}
