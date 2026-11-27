//! Per-image detection error diagnosis for `folo val --diagnostics` / `--worst <N>`, on top of
//! `folo_vision::eval::{diagnose_image, worst_images}`.

use std::path::Path;

use folo_data::FileDataset;
use folo_vision::eval::{ImageDiagnostics, diagnose_image, worst_images};
use folo_vision::{Detection, Truth, Xyxy};
use serde_json::{Value, json};

use crate::cli::{ValArgs, bounds};
use crate::output::{CliError, ExitCode};

/// Default number of worst failing images shown when `--diagnostics` is passed without `--worst`.
pub const DEFAULT_WORST: usize = 10;
/// Default confidence threshold for per-image error diagnosis (`EvalConfig::default().diag_conf`).
pub const DEFAULT_DIAG_CONF: f32 = 0.25;
/// Default TP IoU threshold for per-image error diagnosis (`EvalConfig::default().diag_tp_iou`).
pub const DEFAULT_DIAG_TP_IOU: f32 = 0.50;
/// Default background IoU threshold for per-image error diagnosis (`EvalConfig::default().diag_bg_iou`).
pub const DEFAULT_DIAG_BG_IOU: f32 = 0.10;

/// Validated settings for `folo val --diagnostics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DiagSettings {
    pub worst: usize,
    pub conf: f32,
    pub tp_iou: f32,
    pub bg_iou: f32,
    pub coco_compatible: bool,
}

/// Validates `--diagnostics` / `--worst` / `--diag-*` upfront before opening any model or dataset.
pub(super) fn check_diag_args(args: &ValArgs) -> Result<Option<DiagSettings>, CliError> {
    let invalid = |msg: String| {
        CliError::new(ExitCode::Usage, "invalid_config", msg)
            .hint("see `folo describe val --json` for the accepted ranges")
    };
    let enabled = args.diagnostics || args.worst.is_some();
    if !enabled {
        if args.diag_conf.is_some() || args.diag_tp_iou.is_some() || args.diag_bg_iou.is_some() {
            return Err(invalid(
                "--diag-conf, --diag-tp-iou and --diag-bg-iou require --diagnostics or --worst".into(),
            ));
        }
        return Ok(None);
    }
    let worst = args.worst.unwrap_or(DEFAULT_WORST);
    if !bounds::POSITIVE_INT.contains_u64(worst as u64) {
        return Err(invalid("--worst must be at least 1".into()));
    }
    let conf = args.diag_conf.unwrap_or(DEFAULT_DIAG_CONF);
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(conf)) {
        return Err(invalid(format!("--diag-conf must be in 0..=1, got {conf}")));
    }
    let tp_iou = args.diag_tp_iou.unwrap_or(DEFAULT_DIAG_TP_IOU);
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(tp_iou)) {
        return Err(invalid(format!("--diag-tp-iou must be in 0..=1, got {tp_iou}")));
    }
    let bg_iou = args.diag_bg_iou.unwrap_or(DEFAULT_DIAG_BG_IOU);
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(bg_iou)) {
        return Err(invalid(format!("--diag-bg-iou must be in 0..=1, got {bg_iou}")));
    }
    if bg_iou > tp_iou {
        return Err(invalid(format!("--diag-bg-iou ({bg_iou}) must not exceed --diag-tp-iou ({tp_iou})")));
    }
    Ok(Some(DiagSettings { worst, conf, tp_iou, bg_iou, coco_compatible: args.coco }))
}

/// Aggregated per-image diagnosis report ready for `--json` and human output.
pub(super) struct DiagOutput {
    pub json: Value,
    pub lines: Vec<String>,
}

/// Diagnoses every evaluated image in `detections` against `part`'s ground truths in original-image pixel
/// coordinates, using `folo_vision::eval::{diagnose_image, worst_images}`.
pub(super) fn compute(
    part: &FileDataset,
    root: &Path,
    detections: &[(usize, Vec<Detection>)],
    settings: DiagSettings,
) -> DiagOutput {
    let entries = part.entries();
    let mut diags: Vec<ImageDiagnostics> = Vec::with_capacity(detections.len());
    for (img_idx, preds) in detections {
        let Some(entry) = entries.get(*img_idx) else {
            continue;
        };
        let (w, h) = entry.size.or_else(|| folo_data::probe_size(&entry.image).ok()).unwrap_or((1, 1));
        let (fw, fh) = (w as f32, h as f32);
        let truths: Vec<Truth> = entry
            .boxes
            .iter()
            .map(|b| {
                let (cx, cy, hw, hh) = (b.cx * fw, b.cy * fh, b.w * fw * 0.5, b.h * fh * 0.5);
                Truth { bbox: Xyxy::new(cx - hw, cy - hh, cx + hw, cy + hh), class: b.class }
            })
            .collect();
        let id = folo_data::paths::relative_to(&entry.image, root);
        diags.push(diagnose_image(
            *img_idx as u64,
            Some(id),
            preds,
            &truths,
            settings.conf,
            settings.tp_iou,
            settings.bg_iou,
            settings.coco_compatible,
        ));
    }

    let mut num_truths = 0u64;
    let mut num_preds = 0u64;
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_count = 0u64;
    let mut localization_errors = 0u64;
    let mut class_confusions = 0u64;
    let mut duplicate_errors = 0u64;
    let mut background_fp = 0u64;
    let mut failing_images = 0u64;
    let mut f1_sum = 0.0f64;
    let mut tp_iou_weighted = 0.0f64;

    for d in &diags {
        num_truths += u64::from(d.num_truths);
        num_preds += u64::from(d.num_preds);
        tp += u64::from(d.tp);
        fp += u64::from(d.fp);
        fn_count += u64::from(d.fn_count);
        localization_errors += u64::from(d.localization_errors);
        class_confusions += u64::from(d.class_confusions);
        duplicate_errors += u64::from(d.duplicate_errors);
        background_fp += u64::from(d.background_fp);
        if d.total_errors() > 0 {
            failing_images += 1;
        }
        f1_sum += f64::from(d.f1);
        tp_iou_weighted += f64::from(d.mean_tp_iou) * f64::from(d.tp);
    }

    let mean_f1 = if diags.is_empty() { 1.0 } else { (f1_sum / diags.len() as f64) as f32 };
    let mean_tp_iou = if tp > 0 { (tp_iou_weighted / tp as f64) as f32 } else { 0.0 };
    let worst = worst_images(&diags, settings.worst);

    let json = json!({
        "conf": settings.conf,
        "tp_iou": settings.tp_iou,
        "bg_iou": settings.bg_iou,
        "images_diagnosed": diags.len(),
        "failing_images": failing_images,
        "summary": {
            "num_truths": num_truths,
            "num_preds": num_preds,
            "tp": tp,
            "fp": fp,
            "fn_count": fn_count,
            "localization_errors": localization_errors,
            "class_confusions": class_confusions,
            "duplicate_errors": duplicate_errors,
            "background_fp": background_fp,
            "mean_f1": mean_f1,
            "mean_tp_iou": mean_tp_iou,
        },
        "worst_images": worst,
    });

    let mut lines = vec![format!(
        "Error diagnostics (conf {:.2}, TP IoU {:.2}, bg IoU {:.2}): {} / {} images with errors — TP {tp}, FP {fp} \
         (loc {localization_errors}, cls {class_confusions}, dup {duplicate_errors}, bg {background_fp}), FN \
         {fn_count}, mean F1 {mean_f1:.3}, mean TP IoU {mean_tp_iou:.3}",
        settings.conf,
        settings.tp_iou,
        settings.bg_iou,
        failing_images,
        diags.len(),
    )];
    if worst.is_empty() {
        lines.push("  no failing images".into());
    } else {
        lines.push(format!(
            "  {:<32} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>5} {:>5}",
            "worst image", "err", "TP", "FP", "FN", "loc", "cls", "dup", "bg", "F1"
        ));
        for d in &worst {
            let label = d.image_id.as_deref().unwrap_or("<image>");
            lines.push(format!(
                "  {:<32} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>5} {:>5.2}",
                truncate_left(label, 32),
                d.total_errors(),
                d.tp,
                d.fp,
                d.fn_count,
                d.localization_errors,
                d.class_confusions,
                d.duplicate_errors,
                d.background_fp,
                d.f1,
            ));
        }
    }

    DiagOutput { json, lines }
}

fn truncate_left(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    let skip = n.saturating_sub(max_chars.saturating_sub(1));
    format!("…{}", s.chars().skip(skip).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn check_diag_args_validates_thresholds_and_worst() {
        let parse = |argv: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Command::Val(v) = cli.command else { panic!() };
            v
        };
        assert!(check_diag_args(&parse(&["folo", "val", "--model", "m", "--data", "d"])).unwrap().is_none());

        let d = check_diag_args(&parse(&["folo", "val", "--model", "m", "--data", "d", "--diagnostics"]))
            .unwrap()
            .unwrap();
        assert_eq!((d.worst, d.conf, d.tp_iou, d.bg_iou), (10, 0.25, 0.50, 0.10));

        let w = check_diag_args(&parse(&["folo", "val", "--model", "m", "--data", "d", "--worst", "5"]))
            .unwrap()
            .unwrap();
        assert_eq!(w.worst, 5);

        for argv in [
            &["folo", "val", "--model", "m", "--data", "d", "--worst", "0"][..],
            &["folo", "val", "--model", "m", "--data", "d", "--diagnostics", "--diag-conf", "1.5"][..],
            &[
                "folo",
                "val",
                "--model",
                "m",
                "--data",
                "d",
                "--diagnostics",
                "--diag-tp-iou",
                "0.3",
                "--diag-bg-iou",
                "0.5",
            ][..],
        ] {
            let err = check_diag_args(&parse(argv)).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "invalid_config"), "{argv:?}");
        }
    }
}
