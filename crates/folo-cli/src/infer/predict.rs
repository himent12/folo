//! `folo predict`: recursive image walking, `--save` confinement / collision avoidance, and atomic annotated image
//! writing.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use folo_data::ImageFormat;
use folo_vision::fusion::ScoreRule;
use folo_vision::model::HeadMode;
use folo_vision::tta::DEFAULT_TTA_SCALES;
use folo_vision::{
    ByteTracker, Detection, PredictConfig, PredictOptions, Predictor, TileConfig, TileMergeMetric,
    TrackConfig, TrackedDetection, TtaConfig,
};
use serde_json::json;

use crate::cli::{PredictArgs, bounds};
use crate::draw::{annotate, annotate_tracked};
use crate::engine::engine_error;
use crate::output::{CliError, ExitCode, Out};

use super::{check_fusion_iou, check_infer_args, open_model, precision_of};

/// Refuses an empty `--save` path or an existing non-directory before loading the model.
fn check_save_dir(save: Option<&Path>) -> Result<(), CliError> {
    let Some(save) = save else { return Ok(()) };
    if save.as_os_str().is_empty() {
        return Err(CliError::new(ExitCode::Usage, "usage", "--save folder path must not be empty")
            .hint("pass a folder path to --save; each annotated image is written inside it"));
    }
    if std::fs::metadata(save).is_ok_and(|m| !m.is_dir()) {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--save {} is a file, not a folder", folo_api::json::path(save)),
        )
        .hint("pass a folder path to --save; each annotated image is written inside it"));
    }
    Ok(())
}

/// Every image under `sources` (folders recursively, in name order), with its path relative to the folder it was
/// found in (just the file name for a file source).
fn collect_images(sources: &[PathBuf]) -> Result<Vec<(PathBuf, PathBuf)>, CliError> {
    fn walk(dir: &Path, ancestors: &mut Vec<PathBuf>, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        let mut entries: Vec<PathBuf> =
            std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                match std::fs::canonicalize(&p) {
                    Ok(canon) => {
                        // Skip only real cycles (a directory already among the ancestors on the current walk),
                        // so non-cyclic links to the same folder (e.g. `latest -> run-3` beside `run-3/`) are
                        // still walked as before.
                        if ancestors.contains(&canon) {
                            continue;
                        }
                        ancestors.push(canon);
                        let res = walk(&p, ancestors, out);
                        ancestors.pop();
                        res?;
                    }
                    // When identity cannot be established (e.g. a share whose ancestor metadata cannot be
                    // queried), walk it as before rather than silently dropping its images.
                    Err(_) => walk(&p, ancestors, out)?,
                }
            } else if folo_data::is_image_path(&p) {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    for s in sources {
        if s.is_dir() {
            let mut ancestors: Vec<PathBuf> = std::fs::canonicalize(s).into_iter().collect();
            let mut found = Vec::new();
            walk(s, &mut ancestors, &mut found).map_err(|e| {
                CliError::new(
                    ExitCode::Runtime,
                    "io_error",
                    format!("cannot read {}: {e}", folo_api::json::path(s)),
                )
            })?;
            out.extend(found.into_iter().map(|p| {
                let rel = p.strip_prefix(s).map(Path::to_path_buf).unwrap_or_else(|_| p.clone());
                (p, rel)
            }));
        } else if s.is_file() {
            out.push((s.clone(), s.file_name().map_or_else(|| PathBuf::from("image"), PathBuf::from)));
        } else {
            return Err(CliError::new(
                ExitCode::NotFound,
                "path_not_found",
                format!("{} does not exist", folo_api::json::path(s)),
            ));
        }
    }
    if out.is_empty() {
        return Err(CliError::new(
            ExitCode::NotFound,
            "path_not_found",
            "no images found in the given sources",
        )
        .hint(format!("supported extensions: {}", folo_data::IMAGE_EXTENSIONS.join(", "))));
    }
    Ok(out)
}

/// Where `--save` writes each image: `<dir>/<relative path>` with a `.png` (PNG sources) or `.jpg` extension,
/// suffixed `_2`, `_3`, … when two sources map to the same name. Refuses any destination that is one of the
/// inputs, so annotations never overwrite a source image.
fn save_paths(dir: &Path, images: &[(PathBuf, PathBuf)]) -> Result<Vec<(PathBuf, ImageFormat)>, CliError> {
    let canon = |p: &Path| std::fs::canonicalize(p).ok();
    let inputs: std::collections::HashSet<PathBuf> = images.iter().filter_map(|(p, _)| canon(p)).collect();
    let mut used = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(images.len());
    for (path, rel) in images {
        let png = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"));
        let (fmt, ext) = if png { (ImageFormat::Png, "png") } else { (ImageFormat::Jpeg(92), "jpg") };
        let base = dir.join(rel).with_extension("");
        // Not `with_extension(ext)` again: for `a.b.jpg` it would replace the `b`.
        let mut dst = PathBuf::from(format!("{}.{ext}", base.display()));
        let mut i = 2;
        // Keyed case-insensitively: Windows and macOS file systems usually are, so `Cat.jpg` and `cat.jpg` must not
        // share an output (on a case-sensitive one this only adds a harmless suffix).
        while !used.insert(dst.to_string_lossy().to_lowercase()) {
            dst = PathBuf::from(format!("{}_{i}.{ext}", base.display()));
            i += 1;
        }
        if canon(&dst).is_some_and(|c| inputs.contains(&c)) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("--save would overwrite the input image {}", folo_api::json::path(path)),
            )
            .hint("pass a --save folder outside the source folders"));
        }
        out.push((dst, fmt));
    }
    Ok(out)
}

/// Write `bytes` to `dst` through a fresh short-named temporary file in `dst`'s folder renamed over it, so an
/// existing `dst` (even a hard link to a source image) is replaced atomically as a directory entry, never
/// overwritten in place, failed writes leave no temporary files behind, and filenames up to the filesystem's
/// component limit still succeed.
fn write_new(dst: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::sync::atomic::AtomicU64;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    write_new_with(dst, bytes, &NEXT)
}

fn write_new_with(dst: &Path, bytes: &[u8], next: &std::sync::atomic::AtomicU64) -> Result<(), String> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    let dir = dst.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let dst_name = dst.file_name().map(|n| n.to_string_lossy().to_lowercase());
    let (tmp, mut f) = loop {
        let n = next.fetch_add(1, Ordering::Relaxed);
        let name = format!("{}{}-{n}.tmp", folo_nn::io::TEMP_PREFIX, std::process::id());
        // Skip before creating so we never briefly publish an empty file at `dst`.
        if Some(name.to_lowercase()) == dst_name {
            continue;
        }
        let candidate = dir.join(&name);
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(f) => break (candidate, f),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    };
    let r = (|| -> std::io::Result<()> {
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        folo_nn::io::rename_over(&tmp, dst)
    })();
    if let Err(e) = r {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    Ok(())
}

/// `predict --save DIR` must lie outside every source folder and hold no source, else the annotated copies are read
/// as images on the next run (or replace a source). Canonical paths are compared; a `DIR` that does not exist yet
/// is its nearest existing ancestor's canonical path plus the rest, so the first run gets the same answer. A source
/// that does not exist is left to `collect_images` (after the model, as before), so its `path_not_found` stays.
fn check_save_outside(save: Option<&Path>, sources: &[PathBuf]) -> Result<(), CliError> {
    let Some(save) = save else { return Ok(()) };
    if sources.iter().any(|s| !s.exists()) {
        return Ok(());
    }
    // `.` and `..` are folded by components before any link is followed (the accepted approximation: a `..` after a
    // link goes back up the spelling, not the target), so the missing rest appended below holds no `..`.
    let mut abs = PathBuf::new();
    for c in std::path::absolute(save).unwrap_or_else(|_| save.to_path_buf()).components() {
        match c {
            std::path::Component::CurDir => {}
            // `pop` stops at the root.
            std::path::Component::ParentDir => _ = abs.pop(),
            c => abs.push(c),
        }
    }
    let canon = |p: &Path| std::fs::canonicalize(p).ok();
    let Some(dir) = abs.ancestors().find_map(|a| Some(canon(a)?.join(abs.strip_prefix(a).ok()?))) else {
        return Ok(());
    };
    for s in sources {
        let Some(src) = canon(s) else { continue };
        let (sp, sv) = (folo_api::json::path(s), folo_api::json::path(save));
        let message = if src.is_dir() && dir.starts_with(&src) {
            let rel = if dir == src { "is" } else { "is inside" };
            format!("--save {sv} {rel} the input folder {sp}: the next run would read the copies as images")
        } else if src.starts_with(&dir) {
            format!("the input {sp} is inside the --save folder {sv}: the copies could replace it")
        } else {
            continue;
        };
        return Err(
            CliError::new(ExitCode::Usage, "usage", message).hint("save to a folder outside the inputs")
        );
    }
    Ok(())
}

/// Builds and validates SAHI tiled inference or Test-Time Augmentation options before the model is opened.
fn build_predict_options(args: &PredictArgs) -> Result<PredictOptions, CliError> {
    let invalid = |msg: String| {
        CliError::new(ExitCode::Usage, "invalid_config", msg)
            .hint("see `folo describe predict --json` for the accepted ranges")
    };
    if args.tile.is_some() && args.tta {
        return Err(invalid("--tile and --tta cannot be combined in the same prediction call".into()));
    }
    let tile = if let Some(tile_size) = args.tile {
        if !bounds::POSITIVE_INT.contains_u64(u64::from(tile_size)) {
            return Err(invalid("--tile must be at least 1".into()));
        }
        let overlap = args.tile_overlap.unwrap_or(0.2);
        if !bounds::TILE_OVERLAP.contains_f64(f64::from(overlap)) {
            return Err(invalid(format!("--tile-overlap must be in [0, 1), got {overlap}")));
        }
        let tile_batch = args.tile_batch.unwrap_or(8);
        if !bounds::POSITIVE_INT.contains_u64(tile_batch as u64) {
            return Err(invalid("--tile-batch must be at least 1".into()));
        }
        let cfg = TileConfig {
            tile_width: tile_size,
            tile_height: tile_size,
            overlap_x: overlap,
            overlap_y: overlap,
            full_image: !args.no_tile_full_image,
            tile_batch,
            merge_metric: args.tile_merge.map_or(TileMergeMetric::Ios, Into::into),
            max_det: args.max_det as usize,
            ..TileConfig::default()
        };
        cfg.validate().map_err(|e| engine_error("predict", e))?;
        Some(cfg)
    } else {
        None
    };
    let tta = if args.tta {
        let scales = args.tta_scales.clone().unwrap_or_else(|| DEFAULT_TTA_SCALES.to_vec());
        let cfg = TtaConfig { hflip: !args.no_tta_hflip, scales, ..TtaConfig::default() };
        cfg.validate().map_err(|e| engine_error("predict", e))?;
        if let Some(imgsz) = args.imgsz {
            cfg.resolve_imgszs(imgsz).map_err(|e| engine_error("predict", e))?;
        }
        Some(cfg)
    } else {
        None
    };
    Ok(PredictOptions { tta, tile })
}

/// Builds and validates [`TrackConfig`] when `--track` is enabled, before the model is opened.
fn build_track_config(args: &PredictArgs) -> Result<Option<TrackConfig>, CliError> {
    if !args.track {
        return Ok(None);
    }
    let invalid = |msg: String| {
        CliError::new(ExitCode::Usage, "invalid_config", msg)
            .hint("see `folo describe predict --json` for the accepted ranges")
    };
    let d = TrackConfig::default();
    let high_thresh = args.track_high_thresh.unwrap_or(d.high_thresh);
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(high_thresh)) {
        return Err(invalid(format!("--track-high-thresh must be in 0..=1, got {high_thresh}")));
    }
    // When omitted, adapt default low / new thresholds to a custom --track-high-thresh so passing e.g.
    // `--track --track-high-thresh 0.7` works cleanly while explicit out-of-order pairs still fail validation.
    let low_thresh = args.track_low_thresh.unwrap_or_else(|| d.low_thresh.min(high_thresh));
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(low_thresh)) {
        return Err(invalid(format!("--track-low-thresh must be in 0..=1, got {low_thresh}")));
    }
    let new_track_thresh = args.track_new_thresh.unwrap_or_else(|| d.new_track_thresh.max(high_thresh));
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(new_track_thresh)) {
        return Err(invalid(format!("--track-new-thresh must be in 0..=1, got {new_track_thresh}")));
    }
    let match_iou = args.track_match_iou.unwrap_or(d.match_iou);
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(match_iou)) {
        return Err(invalid(format!("--track-match-iou must be in 0..=1, got {match_iou}")));
    }
    let min_hits = args.track_min_hits.unwrap_or(d.min_hits);
    if !bounds::POSITIVE_INT.contains_u64(u64::from(min_hits)) {
        return Err(invalid("--track-min-hits must be at least 1".into()));
    }
    let cfg = TrackConfig {
        high_thresh,
        low_thresh,
        new_track_thresh,
        match_iou,
        track_buffer: args.track_buffer.unwrap_or(d.track_buffer),
        min_hits,
        class_aware: !args.track_class_agnostic,
        ..d
    };
    cfg.validate().map_err(|e| invalid(e.to_string()))?;
    Ok(Some(cfg))
}

/// Detector confidence floor when ByteTrack is active so stage-2 low-confidence detections down to `low_thresh`
/// survive pre-NMS filtering (matching `folo_vision::track::sequence_det_conf`).
fn track_det_conf(pred_conf: f32, low_thresh: f32, score_rule: ScoreRule, head_mode: HeadMode) -> f32 {
    const CALIBRATED_LOGIT_HEADROOM: f64 = 1.5;
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

/// Filters and stages detections before [`ByteTracker::update`] when calibrated per-class thresholds are active.
///
/// Detections meeting their calibrated threshold (`score >= cls_thresh`, falling back to `global_conf`) pass through
/// unchanged so they can participate in stage-1 association and new-track admission. Detections below their calibrated
/// threshold can never spawn new tracks or match lost/tentative tracks: when there are active tracks and a stage-2
/// recovery band exists (`low_thresh < high_thresh`), candidates with `score >= low_thresh` are routed exclusively to
/// ByteTrack's stage-2 low-confidence pool (demoting any `high_thresh <= score < cls_thresh` into unique bit patterns
/// strictly below `high_thresh` and returning a `bits -> real_score` map so emitted [`TrackedDetection::score`] values
/// preserve the exact detector score).
fn filter_detections_for_tracker(
    detections: &[Detection],
    tc: &TrackConfig,
    has_active_tracks: bool,
    global_conf: f32,
    per_class: &BTreeMap<u32, f32>,
) -> (Vec<Detection>, HashMap<u32, f32>) {
    let mut kept = Vec::with_capacity(detections.len());
    let mut demoted = Vec::new();
    let mut used_low_bits: HashSet<u32> = HashSet::new();
    let allow_stage2 = has_active_tracks && tc.low_thresh < tc.high_thresh;

    for &d in detections {
        if !d.score.is_finite() {
            continue;
        }
        let cls_thresh = per_class.get(&d.class).copied().unwrap_or(global_conf);
        if d.score >= cls_thresh {
            if d.score >= tc.low_thresh && d.score < tc.high_thresh {
                used_low_bits.insert(d.score.to_bits());
            }
            kept.push(d);
        } else if allow_stage2 && d.score >= tc.low_thresh {
            if d.score < tc.high_thresh {
                used_low_bits.insert(d.score.to_bits());
                kept.push(d);
            } else {
                demoted.push(d);
            }
        }
    }

    let mut restore_scores = HashMap::new();
    if !demoted.is_empty() {
        // Preserve descending real-score order inside ByteTrack's stage-2 pool.
        demoted.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        let mut cursor = tc.high_thresh.next_down();
        for mut d in demoted {
            while cursor >= tc.low_thresh && used_low_bits.contains(&cursor.to_bits()) {
                cursor = cursor.next_down();
            }
            if cursor < tc.low_thresh || !cursor.is_finite() {
                break;
            }
            let real = d.score;
            d.score = cursor;
            let bits = cursor.to_bits();
            used_low_bits.insert(bits);
            restore_scores.insert(bits, real);
            kept.push(d);
            cursor = cursor.next_down();
        }
    }
    (kept, restore_scores)
}

fn tracked_boxes_json(tracks: &[TrackedDetection], classes: &[String]) -> Vec<serde_json::Value> {
    let name = |c: u32| classes.get(c as usize).cloned().unwrap_or_else(|| c.to_string());
    tracks
        .iter()
        .map(|tr| {
            json!({
                "x1": tr.bbox.x1,
                "y1": tr.bbox.y1,
                "x2": tr.bbox.x2,
                "y2": tr.bbox.y2,
                "score": tr.score,
                "class": tr.class_id,
                "name": name(tr.class_id),
                "track_id": tr.track_id,
                "age": tr.age,
                "hits": tr.hits,
            })
        })
        .collect()
}

pub fn predict(args: PredictArgs, out: Out) -> Result<(), CliError> {
    check_fusion_iou(args.fusion_iou)?;
    check_infer_args("predict", args.conf, args.iou, args.max_det, args.imgsz, bounds::INFER_IMGSZ, None)?;
    let predict_opts = build_predict_options(&args)?;
    let track_cfg = build_track_config(&args)?;
    let calibration = match &args.thresholds {
        Some(path) => Some((path.as_path(), super::thresholds::load_thresholds(path)?)),
        None => None,
    };
    check_save_dir(args.save.as_deref())?;
    check_save_outside(args.save.as_deref(), &args.sources)?;
    let shape = format!("imgsz {:?} head {:?} tile {:?} tta {}", args.imgsz, args.head, args.tile, args.tta);
    let model = open_model(
        "predict",
        &args.model,
        &args.device,
        (args.half, args.precision),
        args.tensor_cores,
        &shape,
    )?;
    if let Some((path, cal)) = &calibration {
        super::thresholds::check_thresholds_for_model(path, cal, model.classes().len())?;
    }
    let effective_imgsz = args.imgsz.unwrap_or(model.imgsz());
    predict_opts.validate(effective_imgsz).map_err(|e| engine_error("predict", e))?;
    let images = collect_images(&args.sources)?;
    let score_rule: ScoreRule = args.score.into();
    let effective_head = args.head.map(Into::into).unwrap_or_else(|| model.head_mode());
    let cfg = PredictConfig {
        imgsz: effective_imgsz,
        conf: args.conf,
        iou: args.iou,
        max_det: args.max_det as usize,
        rect: true,
        head: args.head.map(Into::into),
        fusion: args.fusion.into(),
        fusion_iou: args.fusion_iou,
        score: score_rule,
    };
    let mut predictor = Predictor::from_detector(model, cfg).map_err(|e| engine_error("predict", e))?;
    if let Some((_, cal)) = &calibration {
        predictor = predictor.with_calibrated_thresholds(cal).map_err(|e| engine_error("predict", e))?;
    }
    let calibrated_gate = calibration
        .as_ref()
        .filter(|(_, cal)| !cal.is_empty())
        .map(|(_, cal)| (cal.global.best_conf, cal.per_class_map()));
    if let Some(tc) = &track_cfg {
        let mut pcfg = predictor.config().clone();
        pcfg.conf = track_det_conf(pcfg.conf, tc.low_thresh, score_rule, effective_head);
        predictor.set_config(pcfg).map_err(|e| engine_error("predict", e))?;
        if let Some(map) = predictor.per_class_conf() {
            let lowered: std::collections::BTreeMap<u32, f32> = map
                .iter()
                .map(|(&cls, &v)| (cls, track_det_conf(v, tc.low_thresh, score_rule, effective_head)))
                .collect();
            predictor.set_per_class_conf(Some(lowered)).map_err(|e| engine_error("predict", e))?;
            predictor.set_conf_inclusive(tc.low_thresh == 0.0);
        }
    }
    let mut tracker = match &track_cfg {
        Some(tc) => Some(ByteTracker::new(tc.clone()).map_err(|e| engine_error("predict", e))?),
        None => None,
    };
    let destinations = match &args.save {
        Some(dir) => Some(save_paths(dir, &images)?),
        None => None,
    };
    let classes = predictor.classes().to_vec();
    let device = predictor.device().to_string();
    let precision = precision_of(predictor.detector());
    let (mut boxes, mut failed) = (0usize, 0usize);
    let mut frame_index = 0usize;
    let mut unique_tracks: BTreeSet<u64> = BTreeSet::new();
    // The first image's timing (it includes pipeline compilation and the first allocations) and the sum over the
    // rest: the mean leaves the first out once there are others.
    let mut first: Option<[f64; 4]> = None;
    let mut sum = [0f64; 4];
    let t_all = Instant::now();
    for (i, (path, _)) in images.iter().enumerate() {
        let t = Instant::now();
        let img = match folo_data::read_image(path) {
            Ok(i) => i,
            Err(e) => {
                failed += 1;
                out.record(&json!({ "event": "error", "file": folo_api::json::path(path), "error": { "code": "image_decode_failed", "message": e.to_string() } }));
                out.human(format!("{}: {e}", folo_api::json::path(path)));
                continue;
            }
        };
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut p =
            predictor.predict_with_options(&img, &predict_opts).map_err(|e| engine_error("predict", e))?;
        p.timing.preprocess_ms += decode_ms;
        p.timing.total_ms += decode_ms;
        let tracks = if let Some(trk) = &mut tracker {
            let t_trk = Instant::now();
            let active = if let Some((global_conf, per_class)) = &calibrated_gate {
                let (filtered, restore) = filter_detections_for_tracker(
                    &p.detections,
                    trk.config(),
                    trk.active_count() > 0,
                    *global_conf,
                    per_class,
                );
                let mut active = trk.update(&filtered);
                if !restore.is_empty() {
                    for tr in &mut active {
                        if let Some(&real) = restore.get(&tr.score.to_bits()) {
                            tr.score = real;
                        }
                    }
                }
                active
            } else {
                trk.update(&p.detections)
            };
            let trk_ms = t_trk.elapsed().as_secs_f64() * 1e3;
            p.timing.postprocess_ms += trk_ms;
            p.timing.total_ms += trk_ms;
            for tr in &active {
                unique_tracks.insert(tr.track_id);
            }
            Some(active)
        } else {
            None
        };
        let ti = p.timing;
        let parts = [ti.preprocess_ms, ti.inference_ms, ti.postprocess_ms, ti.total_ms];
        if first.is_none() {
            first = Some(parts);
        } else {
            for (s, v) in sum.iter_mut().zip(parts) {
                *s += v;
            }
        }
        let count = tracks.as_ref().map_or(p.detections.len(), Vec::len);
        boxes += count;
        let mut saved = None;
        if let Some(dests) = &destinations {
            let (dst, fmt) = &dests[i];
            let mut canvas = img;
            if let Some(active) = &tracks {
                annotate_tracked(&mut canvas, active);
            } else {
                annotate(&mut canvas, &p.detections);
            }
            let io = |e: String| {
                CliError::new(
                    ExitCode::Runtime,
                    "io_error",
                    format!("cannot write {}: {e}", folo_api::json::path(dst)),
                )
            };
            if let Some(parent) = dst.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|e| io(e.to_string()))?;
            }
            write_new(dst, &folo_data::encode(&canvas, *fmt).map_err(|e| io(e.to_string()))?)
                .map_err(|e| io(e.to_string()))?;
            saved = Some(dst.clone());
        }
        if let Some(active) = &tracks {
            let boxes_json = tracked_boxes_json(active, &classes);
            out.record(&json!({
                "event": "prediction",
                "frame_index": frame_index,
                "file": crate::output::record_path(path),
                "image": { "width": p.width, "height": p.height },
                "boxes": boxes_json,
                "timing": ti.to_api(),
                "device": device,
                "dtype": precision,
                "saved": saved.as_ref().map(crate::output::record_path),
            }));
            let name = |c: u32| classes.get(c as usize).cloned().unwrap_or_else(|| c.to_string());
            let summary: Vec<String> = active
                .iter()
                .take(6)
                .map(|tr| format!("#{} {} {:.2}", tr.track_id, name(tr.class_id), tr.score))
                .collect();
            out.human(image_line(&folo_api::json::path(path), active.len(), &summary, ti.total_ms));
            frame_index += 1;
        } else {
            out.record(&json!({
                "event": "prediction",
                "file": crate::output::record_path(path),
                "image": { "width": p.width, "height": p.height },
                "boxes": p.scored_boxes(&classes),
                "timing": ti.to_api(),
                "device": device,
                "dtype": precision,
                "saved": saved.as_ref().map(crate::output::record_path),
            }));
            let summary: Vec<String> = p
                .scored_boxes(&classes)
                .iter()
                .take(6)
                .map(|b| format!("{} {:.2}", b.name, b.score))
                .collect();
            out.human(image_line(&folo_api::json::path(path), p.detections.len(), &summary, ti.total_ms));
        }
    }
    let done = images.len() - failed;
    if done == 1 {
        sum = first.unwrap_or_default();
    }
    let n = done.saturating_sub(1).max(1) as f64;
    // Rounded like the per-image timings (`folo_api::json::round_ms`): `49.0546`, not `49.054555000000015`.
    let mean = |i: usize| folo_api::json::round_ms(sum[i] / n);
    let first_ms = |i: usize| first.map(|f| folo_api::json::round_ms(f[i]));
    let mut done_record = json!({
        "event": "done",
        "images": images.len() - failed,
        "failed": failed,
        "boxes": boxes,
        "mean_timing": { "preprocess_ms": mean(0), "inference_ms": mean(1), "postprocess_ms": mean(2), "total_ms": mean(3) },
        "first_timing": first.map(|_| json!({ "preprocess_ms": first_ms(0), "inference_ms": first_ms(1), "postprocess_ms": first_ms(2), "total_ms": first_ms(3) })),
        "seconds": t_all.elapsed().as_secs_f64(),
        "saved_to": args.save.as_ref().map(crate::output::record_path),
    });
    if tracker.is_some() {
        done_record["tracked"] = json!(true);
        done_record["unique_tracks"] = json!(unique_tracks.len());
    }
    if let Some((path, cal)) = &calibration {
        done_record["thresholds"] = super::thresholds::predict_thresholds_json(path, cal);
    }
    out.record(&done_record);
    let track_note =
        if tracker.is_some() { format!(", {} unique track(s)", unique_tracks.len()) } else { String::new() };
    let thresh_note = calibration.as_ref().map_or(String::new(), |(_, cal)| {
        if cal.is_empty() {
            "; thresholds file empty (default conf)".into()
        } else {
            format!(
                "; calibrated thresholds ({} classes, global {:.3})",
                cal.per_class.len(),
                cal.global.best_conf
            )
        }
    });
    out.human(format!(
        "{} images, {boxes} boxes{track_note}; mean {:.1} ms (pre {:.1}, inference {:.1}, post {:.1}{}){thresh_note}{}",
        done,
        sum[3] / n,
        sum[0] / n,
        sum[1] / n,
        sum[2] / n,
        if done > 1 { "; the first image, with warm-up, left out" } else { "" },
        args.save
            .as_ref()
            .map_or(String::new(), |d| format!("; annotated images in {}", folo_api::json::path(d)))
    ));
    if failed == images.len() {
        return Err(CliError::new(ExitCode::Runtime, "image_decode_failed", "no image could be decoded"));
    }
    Ok(())
}

/// `predict`'s line for one image: "a.jpg: 2 boxes (cat 0.91, dog 0.40) 9.2 ms", "1 box", "0 boxes" without an empty
/// list; `shown` is the first boxes' "name score", at most six, and the rest are counted.
fn image_line(path: &str, boxes: usize, shown: &[String], ms: f64) -> String {
    let noun = if boxes == 1 { "box" } else { "boxes" };
    let list = match boxes {
        0 => String::new(),
        n if n > shown.len() => format!(" ({}, +{} more)", shown.join(", "), n - shown.len()),
        _ => format!(" ({})", shown.join(", ")),
    };
    format!("{path}: {boxes} {noun}{list} {ms:.1} ms")
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::*;

    fn dir_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn an_image_line_counts_its_boxes_and_lists_none_for_none() {
        let two = ["cat 0.91".to_string(), "dog 0.40".to_string()];
        assert_eq!(image_line("a.jpg", 2, &two, 9.24), "a.jpg: 2 boxes (cat 0.91, dog 0.40) 9.2 ms");
        assert_eq!(image_line("a.jpg", 1, &two[..1], 9.0), "a.jpg: 1 box (cat 0.91) 9.0 ms");
        assert_eq!(image_line("a.jpg", 0, &[], 14.7), "a.jpg: 0 boxes 14.7 ms");
        assert_eq!(image_line("a.jpg", 9, &two, 1.0), "a.jpg: 9 boxes (cat 0.91, dog 0.40, +7 more) 1.0 ms");
    }

    #[test]
    fn check_save_dir_rejects_empty_and_existing_file() {
        let dir = TestDir::new("infer-save-dir");
        let file = dir.join("not-a-dir.png");
        std::fs::write(&file, b"png").unwrap();

        check_save_dir(None).unwrap();
        check_save_dir(Some(dir.path())).unwrap();
        check_save_dir(Some(&dir.join("new-folder"))).unwrap();

        let empty = check_save_dir(Some(Path::new(""))).unwrap_err();
        assert_eq!((empty.exit, empty.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(empty.message.contains("--save"));

        let is_file = check_save_dir(Some(&file)).unwrap_err();
        assert_eq!((is_file.exit, is_file.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(is_file.message.contains("not a folder"), "{}", is_file.message);
    }

    #[test]
    fn write_new_replaces_atomically_handles_long_names_and_cleans_up_on_error() {
        let dir = TestDir::new("infer-write-new");

        // Fresh write + atomic overwrite leave no `.tmp` files behind.
        let dst = dir.join("out.png");
        write_new(&dst, b"first").unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"first");
        write_new(&dst, b"second").unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"second");
        assert_eq!(dir_names(dir.path()), ["out.png"]);

        // Hard-linked destination is replaced as a directory entry without touching the original link.
        let src = dir.join("source.png");
        let linked = dir.join("linked.png");
        std::fs::write(&src, b"original").unwrap();
        std::fs::hard_link(&src, &linked).unwrap();
        write_new(&linked, b"annotated").unwrap();
        assert_eq!(std::fs::read(&src).unwrap(), b"original");
        assert_eq!(std::fs::read(&linked).unwrap(), b"annotated");

        // Long filename near the filesystem's 255-byte single-component limit still succeeds.
        let long_name = format!("{}.png", "a".repeat(246));
        let long_dst = dir.join(&long_name);
        write_new(&long_dst, b"long").unwrap();
        assert_eq!(std::fs::read(&long_dst).unwrap(), b"long");

        // Destination matching the next candidate temp name skips that slot before creation.
        let colliding_name = format!("{}{}-0.tmp", folo_nn::io::TEMP_PREFIX, std::process::id());
        let colliding = dir.join(&colliding_name);
        let counter = std::sync::atomic::AtomicU64::new(0);
        write_new_with(&colliding, b"exact", &counter).unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(std::fs::read(&colliding).unwrap(), b"exact");

        // Failed rename (directory in destination's place) cleans up the temporary file and leaves no leftovers.
        let existing_dir = dir.join("taken");
        std::fs::create_dir(&existing_dir).unwrap();
        assert!(write_new(&existing_dir, b"nope").is_err());
        assert!(existing_dir.is_dir());
        assert!(
            dir_names(dir.path()).iter().all(|n| n == &colliding_name || !n.ends_with(".tmp")),
            "leftover temp file: {:?}",
            dir_names(dir.path())
        );
    }

    fn det(class: u32, score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection { bbox: folo_vision::Xyxy { x1, y1, x2, y2 }, score, class }
    }

    fn step_calibrated(
        trk: &mut ByteTracker,
        dets: &[Detection],
        global_conf: f32,
        per_class: &BTreeMap<u32, f32>,
    ) -> Vec<TrackedDetection> {
        let (filtered, restore) =
            filter_detections_for_tracker(dets, trk.config(), trk.active_count() > 0, global_conf, per_class);
        let mut active = trk.update(&filtered);
        for tr in &mut active {
            if let Some(&real) = restore.get(&tr.score.to_bits()) {
                tr.score = real;
            }
        }
        active
    }

    #[test]
    fn calibrated_thresholds_gate_track_admission_and_preserve_stage2_recovery() {
        // 1. All-1.0 calibration blocks new tracks even with low tracker thresholds.
        let low_tc = TrackConfig {
            high_thresh: 0.001,
            low_thresh: 0.0,
            new_track_thresh: 0.001,
            ..TrackConfig::default()
        };
        let mut trk = ByteTracker::new(low_tc.clone()).unwrap();
        let all_one = BTreeMap::from([(0, 1.0), (1, 1.0)]);
        let frame0 = vec![det(0, 0.002, 10.0, 10.0, 30.0, 30.0), det(1, 0.006, 40.0, 40.0, 60.0, 60.0)];
        assert!(step_calibrated(&mut trk, &frame0, 1.0, &all_one).is_empty());

        // 2. Per-class `{0: 0.0, 1: 1.0}` admits class 0 while blocking class 1 on frames 0 and 1.
        let mut trk = ByteTracker::new(low_tc).unwrap();
        let mixed = BTreeMap::from([(0, 0.0), (1, 1.0)]);
        let out0 = step_calibrated(&mut trk, &frame0, 0.0, &mixed);
        assert_eq!(out0.len(), 1);
        assert_eq!(out0[0].class_id, 0);
        let out1 = step_calibrated(&mut trk, &frame0, 0.0, &mixed);
        assert_eq!(out1.len(), 1);
        assert_eq!(out1[0].class_id, 0);

        // 3. Stage-2 occlusion recovery keeps an existing active track alive at both `high <= score < cls_thresh`
        //    and `low <= score < high`, restores the exact detector score, and refuses spawning a new track below
        //    `cls_thresh`.
        let tc = TrackConfig {
            high_thresh: 0.5,
            low_thresh: 0.1,
            new_track_thresh: 0.6,
            ..TrackConfig::default()
        };
        let mut trk = ByteTracker::new(tc).unwrap();
        let cal = BTreeMap::from([(0, 0.8)]);
        let t0 = step_calibrated(&mut trk, &[det(0, 0.85, 10.0, 10.0, 50.0, 50.0)], 0.8, &cal);
        assert_eq!(t0.len(), 1);
        let id = t0[0].track_id;

        // Frame 1: track 1 drops to 0.65 (in `[high_thresh, cls_thresh)`), and an unrelated box at 0.72 tries to spawn.
        let t1 = step_calibrated(
            &mut trk,
            &[det(0, 0.65, 11.0, 10.0, 51.0, 50.0), det(0, 0.72, 120.0, 120.0, 160.0, 160.0)],
            0.8,
            &cal,
        );
        assert_eq!(t1.len(), 1, "unrelated 0.72 box below 0.8 calibration must not spawn a track");
        assert_eq!(t1[0].track_id, id);
        assert_eq!(t1[0].score, 0.65, "exact score must be restored after stage-2 demotion");

        // Frame 2: track 1 drops further to 0.25 (in `[low_thresh, high_thresh)`): still recovered with exact score.
        let t2 = step_calibrated(&mut trk, &[det(0, 0.25, 12.0, 10.0, 52.0, 50.0)], 0.8, &cal);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].track_id, id);
        assert_eq!(t2[0].score, 0.25);

        // 4. When `low_thresh == high_thresh`, stage-2 is disabled and sub-calibrated detections are dropped cleanly.
        let tight = TrackConfig {
            high_thresh: 0.5,
            low_thresh: 0.5,
            new_track_thresh: 0.5,
            ..TrackConfig::default()
        };
        let mut trk = ByteTracker::new(tight).unwrap();
        assert_eq!(step_calibrated(&mut trk, &[det(0, 0.9, 10.0, 10.0, 50.0, 50.0)], 0.8, &cal).len(), 1);
        assert!(step_calibrated(&mut trk, &[det(0, 0.6, 10.0, 10.0, 50.0, 50.0)], 0.8, &cal).is_empty());
    }
}
