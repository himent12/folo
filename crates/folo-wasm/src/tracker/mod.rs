//! Stateful ByteTrack multi-object tracker behind `FoloTracker`, free of `wasm-bindgen` types so native tests
//! exercise every code path directly.
//!
//! Wraps [`folo_vision::ByteTracker`] over browser/JSON box shapes ([`TrackInputBox`] matches `FoloBox` /
//! `folo_api::ScoredBox`, and [`TrackedBox`] extends `FoloBox` with `track_id`, `age`, `hits`, and `state`).

use std::collections::HashMap;

use folo_vision::boxes::Xyxy;
use folo_vision::nms::Detection;
use folo_vision::{ByteTracker, TrackConfig};
use serde::{Deserialize, Serialize};

/// Lifecycle state reported on emitted [`TrackedBox`]es.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackStateLabel {
    /// Confirmed active track matched on the current frame.
    Tracked,
}

/// Options accepted by [`Tracker::new`] (JS: `FoloTrackOptions`; camelCase, unknown fields ignored).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TrackerOptions {
    /// Minimum score for stage-1 high-confidence association (default `0.5`, in `[0, 1]`).
    pub high_thresh: f32,
    /// Minimum score for stage-2 low-confidence occlusion recovery (`<= high_thresh`, in `[0, 1]`).
    /// When omitted, defaults to `0.1_f32.min(high_thresh)`.
    pub low_thresh: Option<f32>,
    /// Minimum score for an unmatched high-confidence detection to spawn a new track (`>= high_thresh`,
    /// in `[0, 1]`). When omitted, defaults to `0.6_f32.max(high_thresh)`.
    pub new_track_thresh: Option<f32>,
    /// Minimum IoU for stage-1 association (default `0.2`, in `[0, 1]`).
    pub match_iou: f32,
    /// Minimum IoU for stage-2 occlusion recovery (default `0.5`, in `[0, 1]`).
    pub second_match_iou: f32,
    /// Minimum IoU for tentative (unconfirmed) tracks (default `0.3`, in `[0, 1]`).
    pub tentative_match_iou: f32,
    /// Maximum consecutive missed frames a lost track coasts before retiring (default `30`).
    pub track_buffer: u32,
    /// Consecutive matched detections required before a newly spawned track is confirmed and emitted
    /// (default `1`, must be `>= 1`).
    pub min_hits: u32,
    /// Require identical `class` when associating detections with tracks (default `true`).
    pub class_aware: bool,
    /// Optional class name table (indexed by `class`), used to populate [`TrackedBox::name`].
    pub classes: Vec<String>,
}

impl Default for TrackerOptions {
    fn default() -> Self {
        let def = TrackConfig::default();
        TrackerOptions {
            high_thresh: def.high_thresh,
            low_thresh: None,
            new_track_thresh: None,
            match_iou: def.match_iou,
            second_match_iou: def.second_match_iou,
            tentative_match_iou: def.tentative_match_iou,
            track_buffer: def.track_buffer,
            min_hits: def.min_hits,
            class_aware: def.class_aware,
            classes: Vec::new(),
        }
    }
}

impl TrackerOptions {
    /// Validate and resolve into [`TrackConfig`].
    pub fn to_track_config(&self) -> Result<TrackConfig, String> {
        let def = TrackConfig::default();
        if !self.high_thresh.is_finite() || !(0.0..=1.0).contains(&self.high_thresh) {
            return Err(format!("track high_thresh must be finite and in [0, 1], got {}", self.high_thresh));
        }
        let low_thresh = self.low_thresh.unwrap_or_else(|| def.low_thresh.min(self.high_thresh));
        let new_track_thresh =
            self.new_track_thresh.unwrap_or_else(|| def.new_track_thresh.max(self.high_thresh));
        let cfg = TrackConfig {
            high_thresh: self.high_thresh,
            low_thresh,
            new_track_thresh,
            match_iou: self.match_iou,
            second_match_iou: self.second_match_iou,
            tentative_match_iou: self.tentative_match_iou,
            track_buffer: self.track_buffer,
            min_hits: self.min_hits,
            class_aware: self.class_aware,
        };
        cfg.validate().map_err(|e| e.to_string())?;
        Ok(cfg)
    }
}

/// Resolved tracker configuration returned by [`Tracker::config`] (JS: `FoloTrackConfig`).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackerConfig {
    /// Stage-1 high-confidence threshold.
    pub high_thresh: f32,
    /// Stage-2 low-confidence occlusion recovery threshold.
    pub low_thresh: f32,
    /// Score threshold required to spawn a new track.
    pub new_track_thresh: f32,
    /// Stage-1 IoU gate.
    pub match_iou: f32,
    /// Stage-2 IoU gate.
    pub second_match_iou: f32,
    /// Tentative-track IoU gate.
    pub tentative_match_iou: f32,
    /// Maximum missed frames a lost track coasts before retiring.
    pub track_buffer: u32,
    /// Consecutive hits required to confirm a new track.
    pub min_hits: u32,
    /// Whether association requires matching class IDs.
    pub class_aware: bool,
}

impl From<&TrackConfig> for TrackerConfig {
    fn from(c: &TrackConfig) -> Self {
        TrackerConfig {
            high_thresh: c.high_thresh,
            low_thresh: c.low_thresh,
            new_track_thresh: c.new_track_thresh,
            match_iou: c.match_iou,
            second_match_iou: c.second_match_iou,
            tentative_match_iou: c.tentative_match_iou,
            track_buffer: c.track_buffer,
            min_hits: c.min_hits,
            class_aware: c.class_aware,
        }
    }
}

/// Live tracker telemetry and configuration returned by [`Tracker::info`] (JS: `FoloTrackerInfo`).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackerInfo {
    /// Number of frames processed since creation or the last [`Tracker::reset`].
    pub frame_id: u64,
    /// Number of currently active confirmed tracks.
    pub active_tracks: usize,
    /// Number of currently coasting lost tracks in the recovery buffer.
    pub lost_tracks: usize,
    /// Effective validated tracker configuration.
    pub config: TrackerConfig,
}

/// One detection box passed into [`Tracker::update`].
///
/// Matches `FoloBox` / `folo_api::ScoredBox` directly (with optional `name` and `class_id`/`classId` aliases for
/// `class`), so `res.boxes` from `FoloDetector.detect()` passes in unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackInputBox {
    /// Object class index.
    #[serde(alias = "class_id", alias = "classId")]
    pub class: u32,
    /// Optional class label (remembered for matching tracked output boxes).
    #[serde(default)]
    pub name: Option<String>,
    /// Detection confidence score.
    pub score: f32,
    /// Left coordinate in image pixels.
    pub x1: f32,
    /// Top coordinate in image pixels.
    pub y1: f32,
    /// Right coordinate in image pixels.
    pub x2: f32,
    /// Bottom coordinate in image pixels.
    pub y2: f32,
}

impl From<&folo_api::ScoredBox> for TrackInputBox {
    fn from(b: &folo_api::ScoredBox) -> Self {
        TrackInputBox {
            class: b.class,
            name: Some(b.name.clone()),
            score: b.score,
            x1: b.x1,
            y1: b.y1,
            x2: b.x2,
            y2: b.y2,
        }
    }
}

/// One active tracked detection emitted by [`Tracker::update`] (JS: `FoloTrackedBox`).
///
/// Extends `FoloBox` (`class`, `name`, `score`, `x1`, `y1`, `x2`, `y2`) with ByteTrack identity and lifecycle
/// fields (`track_id`, `age`, `hits`, `state`) so existing box renderers can draw tracked boxes directly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackedBox {
    /// Persistent monotonically increasing track ID (starts at `1`).
    pub track_id: u64,
    /// Total frames elapsed since this track was created (`1` on the creation frame).
    pub age: u32,
    /// Total matched detection updates accumulated by this track.
    pub hits: u32,
    /// Lifecycle state of emitted tracks (`"tracked"`).
    pub state: TrackStateLabel,
    /// Object class index (`FoloBox.class`).
    pub class: u32,
    /// Human-readable class name (`TrackerOptions::classes`, input box `name`, or `"class{N}"`).
    pub name: String,
    /// Latest matched detection confidence score.
    pub score: f32,
    /// Kalman-filtered left coordinate in image pixels.
    pub x1: f32,
    /// Kalman-filtered top coordinate in image pixels.
    pub y1: f32,
    /// Kalman-filtered right coordinate in image pixels.
    pub x2: f32,
    /// Kalman-filtered bottom coordinate in image pixels.
    pub y2: f32,
}

/// Stateful ByteTrack multi-object tracker.
#[derive(Clone, Debug)]
pub struct Tracker {
    inner: ByteTracker,
    configured_classes: Vec<String>,
    seen_names: HashMap<u32, String>,
}

impl Default for Tracker {
    fn default() -> Self {
        Tracker::new(TrackerOptions::default()).expect("default TrackerOptions is valid")
    }
}

impl Tracker {
    /// Create a new tracker from validated [`TrackerOptions`].
    pub fn new(opts: TrackerOptions) -> Result<Self, String> {
        let cfg = opts.to_track_config()?;
        let inner = ByteTracker::new(cfg).map_err(|e| e.to_string())?;
        Ok(Tracker { inner, configured_classes: opts.classes, seen_names: HashMap::new() })
    }

    /// Process one frame's detection boxes and return all active confirmed [`TrackedBox`]es sorted by ascending
    /// `track_id`.
    ///
    /// Input box order does not affect output (including learned class labels when multiple eligible boxes of the
    /// same class appear in one frame); non-finite scores, sub-`low_thresh` boxes, and non-finite/degenerate boxes
    /// are ignored cleanly by both name learning and the underlying [`ByteTracker`].
    pub fn update(&mut self, boxes: &[TrackInputBox]) -> Vec<TrackedBox> {
        let low_thresh = self.inner.config().low_thresh;
        let mut dets = Vec::with_capacity(boxes.len());
        let mut best_name: HashMap<u32, (f32, f32, f32, f32, f32, &str)> = HashMap::new();
        for b in boxes {
            let eligible = b.score.is_finite()
                && b.score >= low_thresh
                && b.x1.is_finite()
                && b.y1.is_finite()
                && b.x2.is_finite()
                && b.y2.is_finite()
                && b.x2 > b.x1
                && b.y2 > b.y1;
            if eligible && let Some(name) = b.name.as_deref().filter(|s| !s.is_empty()) {
                let cand = (b.score, b.x1, b.y1, b.x2, b.y2, name);
                let replace = match best_name.get(&b.class) {
                    None => true,
                    Some(prev) => cand
                        .0
                        .total_cmp(&prev.0)
                        .then_with(|| prev.1.total_cmp(&cand.1))
                        .then_with(|| prev.2.total_cmp(&cand.2))
                        .then_with(|| prev.3.total_cmp(&cand.3))
                        .then_with(|| prev.4.total_cmp(&cand.4))
                        .then_with(|| prev.5.cmp(cand.5))
                        .is_gt(),
                };
                if replace {
                    best_name.insert(b.class, cand);
                }
            }
            dets.push(Detection { bbox: Xyxy::new(b.x1, b.y1, b.x2, b.y2), score: b.score, class: b.class });
        }
        for (class_id, (_, _, _, _, _, name)) in best_name {
            self.seen_names.insert(class_id, name.to_string());
        }
        let tracks = self.inner.update(&dets);
        tracks
            .into_iter()
            .map(|t| TrackedBox {
                track_id: t.track_id,
                age: t.age,
                hits: t.hits,
                state: TrackStateLabel::Tracked,
                class: t.class_id,
                name: self.class_name(t.class_id),
                score: t.score,
                x1: t.bbox.x1,
                y1: t.bbox.y1,
                x2: t.bbox.x2,
                y2: t.bbox.y2,
            })
            .collect()
    }

    /// Clear all active, tentative, and lost tracks and reset frame and track ID counters.
    ///
    /// Preserves the tracker configuration and any explicit `classes` table passed at construction, while
    /// clearing dynamically observed class names from earlier frames.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.seen_names.clear();
    }

    /// Return the resolved tracker configuration.
    pub fn config(&self) -> TrackerConfig {
        TrackerConfig::from(self.inner.config())
    }

    /// Return current frame/track counters alongside the resolved configuration.
    pub fn info(&self) -> TrackerInfo {
        TrackerInfo {
            frame_id: self.inner.frame_id(),
            active_tracks: self.inner.active_count(),
            lost_tracks: self.inner.lost_count(),
            config: self.config(),
        }
    }

    /// Number of frames processed since creation or the last [`Tracker::reset`].
    pub fn frame_id(&self) -> u64 {
        self.inner.frame_id()
    }

    /// Number of currently active confirmed tracks.
    pub fn active_tracks(&self) -> usize {
        self.inner.active_count()
    }

    /// Number of currently coasting lost tracks in the recovery buffer.
    pub fn lost_tracks(&self) -> usize {
        self.inner.lost_count()
    }

    fn class_name(&self, class_id: u32) -> String {
        if let Some(name) = self.configured_classes.get(class_id as usize).filter(|s| !s.is_empty()) {
            return name.clone();
        }
        if let Some(name) = self.seen_names.get(&class_id) {
            return name.clone();
        }
        format!("class{class_id}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_defaults_and_ergonomic_threshold_resolution() {
        let def = TrackerOptions::default().to_track_config().unwrap();
        assert_eq!(def, TrackConfig::default());

        // Raising high_thresh without setting new_track_thresh raises new_track_thresh to match.
        let hi = TrackerOptions { high_thresh: 0.75, ..Default::default() }.to_track_config().unwrap();
        assert_eq!((hi.low_thresh, hi.high_thresh, hi.new_track_thresh), (0.1, 0.75, 0.75));

        // Lowering high_thresh below 0.1 without setting low_thresh lowers low_thresh to match.
        let lo = TrackerOptions { high_thresh: 0.05, ..Default::default() }.to_track_config().unwrap();
        assert_eq!((lo.low_thresh, lo.high_thresh, lo.new_track_thresh), (0.05, 0.05, 0.6));

        // Explicit conflicting bounds are still rejected.
        assert!(
            TrackerOptions { high_thresh: 0.5, new_track_thresh: Some(0.4), ..Default::default() }
                .to_track_config()
                .is_err()
        );
        assert!(
            TrackerOptions { high_thresh: 0.5, low_thresh: Some(0.6), ..Default::default() }
                .to_track_config()
                .is_err()
        );
        assert!(TrackerOptions { min_hits: 0, ..Default::default() }.to_track_config().is_err());
    }
}
