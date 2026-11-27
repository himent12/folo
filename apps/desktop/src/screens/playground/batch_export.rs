//! Playground batch export serializers and export action bar: structured Batch JSON, RFC 4180
//! Batch Detections CSV (`filename,class,confidence,x1,y1,x2,y2,width,height`), and per-image
//! Batch Summary CSV (`filename,status,detection_count,classes,top_confidence,total_ms,width,height,error`).

use std::collections::BTreeSet;

use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::batch::{BatchItemResult, BatchItemStatus, batch_tile_label};
use super::page::filter_detections;
use crate::api::{PredictResponse, ScoredBox, Timing};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::ui::{counted, plural, push_csv_safe_text};

type BatchExportRows<'a> = (&'a str, &'a str, bool, Vec<(String, &'a BatchItemResult, Vec<&'a ScoredBox>)>);

/// One completed image entry included in a Playground batch export.
pub struct BatchImageExport<'a> {
    pub filename: &'a str,
    pub width: u32,
    pub height: u32,
    pub pred: &'a PredictResponse,
    pub boxes: &'a [&'a ScoredBox],
}

/// Suggested filename for a Playground batch export (`.json` or `.csv`).
pub fn batch_export_filename(model_name: &str, count: usize, ext: &str) -> String {
    let mut slug = String::with_capacity(model_name.len());
    let mut prev_dash = false;
    for ch in model_name.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
    }
    let mut slug: String = slug.chars().take(48).collect();
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = "model".to_string();
    }
    format!("playground-batch-{count}-{slug}.{ext}")
}

/// Formats completed Playground batch predictions as pretty-printed JSON.
#[allow(clippy::too_many_arguments)]
pub fn batch_detections_json(
    model: &str,
    model_id: &str,
    conf: f32,
    iou: Option<f32>,
    class_filter: Option<u32>,
    total_images: usize,
    entries: &[BatchImageExport<'_>],
) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct ImageEntry<'a> {
        filename: &'a str,
        width: u32,
        height: u32,
        device: &'a str,
        cold: bool,
        timing: &'a Timing,
        detection_count: usize,
        detections: &'a [&'a ScoredBox],
    }
    #[derive(serde::Serialize)]
    struct BatchExport<'a> {
        mode: &'static str,
        model: &'a str,
        model_id: &'a str,
        conf: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        iou: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        class_filter: Option<u32>,
        total_images: usize,
        exported_images: usize,
        total_detections: usize,
        images: Vec<ImageEntry<'a>>,
    }

    let total_detections = entries.iter().map(|e| e.boxes.len()).sum();
    let images: Vec<ImageEntry<'_>> = entries
        .iter()
        .map(|e| ImageEntry {
            filename: e.filename,
            width: e.width,
            height: e.height,
            device: &e.pred.device,
            cold: e.pred.cold,
            timing: &e.pred.timing,
            detection_count: e.boxes.len(),
            detections: e.boxes,
        })
        .collect();
    let payload = BatchExport {
        mode: "batch",
        model,
        model_id,
        conf,
        iou,
        class_filter,
        total_images,
        exported_images: entries.len(),
        total_detections,
        images,
    };
    Ok(serde_json::to_string_pretty(&payload)?)
}

/// Formats completed Playground batch detections as CSV with RFC 4180 field quoting, spreadsheet
/// formula-injection protection, and LF (`\n`) record endings, using columns:
/// `filename,class,confidence,x1,y1,x2,y2,width,height`.
pub fn batch_detections_csv(entries: &[BatchImageExport<'_>]) -> String {
    let mut out = String::from("filename,class,confidence,x1,y1,x2,y2,width,height\n");
    for entry in entries {
        for b in entry.boxes {
            let score = if b.score.is_finite() { b.score.clamp(0.0, 1.0) } else { 0.0 };
            let x1 = if b.x1.is_finite() { b.x1 } else { 0.0 };
            let y1 = if b.y1.is_finite() { b.y1 } else { 0.0 };
            let x2 = if b.x2.is_finite() { b.x2 } else { 0.0 };
            let y2 = if b.y2.is_finite() { b.y2 } else { 0.0 };
            let width = (x2 - x1).max(0.0);
            let height = (y2 - y1).max(0.0);
            let cls = if b.name.trim().is_empty() { format!("class {}", b.class) } else { b.name.clone() };
            push_csv_safe_text(&mut out, entry.filename);
            out.push(',');
            push_csv_safe_text(&mut out, &cls);
            out.push_str(&format!(",{score:.4},{x1:.2},{y1:.2},{x2:.2},{y2:.2},{width:.2},{height:.2}\n"));
        }
    }
    out
}

/// One per-image row in the Playground Batch Summary CSV export.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchSummaryRow {
    pub filename: String,
    pub status: &'static str,
    pub detection_count: Option<usize>,
    pub classes: String,
    pub top_confidence: Option<f32>,
    pub total_ms: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub error: String,
}

/// Suggested filename for a Playground Batch Summary CSV export.
pub fn batch_summary_export_filename(model_name: &str, count: usize) -> String {
    let base = batch_export_filename(model_name, count, "csv");
    base.replacen("playground-batch-", "playground-batch-summary-", 1)
}

/// Formats per-image Playground batch summary rows as RFC 4180 CSV (CRLF record separators) with
/// formula-injection protection and columns:
/// `filename,status,detection_count,classes,top_confidence,total_ms,width,height,error`.
pub fn batch_summary_csv(rows: &[BatchSummaryRow]) -> String {
    let mut out = String::from(
        "filename,status,detection_count,classes,top_confidence,total_ms,width,height,error\r\n",
    );
    for r in rows {
        push_csv_safe_text(&mut out, &r.filename);
        out.push(',');
        push_csv_safe_text(&mut out, r.status);
        out.push(',');
        if let Some(n) = r.detection_count {
            out.push_str(&n.to_string());
        }
        out.push(',');
        push_csv_safe_text(&mut out, &r.classes);
        out.push(',');
        if let Some(c) = r.top_confidence.filter(|v| v.is_finite()) {
            out.push_str(&format!("{:.4}", c.clamp(0.0, 1.0)));
        }
        out.push(',');
        if let Some(ms) = r.total_ms.filter(|v| v.is_finite()) {
            out.push_str(&format!("{ms:.2}"));
        }
        out.push(',');
        if let Some(w) = r.width {
            out.push_str(&w.to_string());
        }
        out.push(',');
        if let Some(h) = r.height {
            out.push_str(&h.to_string());
        }
        out.push(',');
        push_csv_safe_text(&mut out, &r.error);
        out.push_str("\r\n");
    }
    out
}

impl FoloApp {
    /// True when at least one fresh completed batch image is ready for Batch JSON / Batch CSV export.
    pub(crate) fn has_play_batch_export(&self) -> bool {
        if !self.play_batch.is_active() {
            return false;
        }
        let Some(model_id) = self.play_model.as_deref() else { return false };
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        (0..self.play_batch.items.len()).any(|i| {
            self.play_batch
                .fresh_result(i, model_id, self.play_iou, self.play_target, self.play_conf, e2e)
                .is_some()
        })
    }

    /// Collects fresh batch entries for export, returning `(model_name, model_id, e2e, rows)`.
    fn collect_batch_export_rows(&self) -> Option<BatchExportRows<'_>> {
        if !self.play_batch.is_active() {
            return None;
        }
        let model_id = self.play_model.as_deref()?;
        let shown = self.play_shown_model();
        let model_name = shown.map(|m| m.name.as_str()).unwrap_or(model_id);
        let e2e = shown.is_some_and(|m| self.model_e2e(m));
        let mut owned = Vec::new();
        for (i, item) in self.play_batch.items.iter().enumerate() {
            if let Some(res) = self.play_batch.fresh_result(
                i,
                model_id,
                self.play_iou,
                self.play_target,
                self.play_conf,
                e2e,
            ) {
                let boxes = filter_detections(&res.pred.boxes, self.play_conf, self.play_class);
                owned.push((batch_tile_label(item.name.as_ref()).to_string(), res, boxes));
            }
        }
        (!owned.is_empty()).then_some((model_name, model_id, e2e, owned))
    }

    /// Builds the Batch JSON payload, suggested filename, and summary string from fresh batch results.
    pub(crate) fn build_play_batch_export_json(&self) -> Option<anyhow::Result<(String, String, String)>> {
        let (model_name, model_id, e2e, owned) = self.collect_batch_export_rows()?;
        let entries: Vec<BatchImageExport<'_>> = owned
            .iter()
            .map(|(fname, res, boxes)| BatchImageExport {
                filename: fname.as_str(),
                width: res.width,
                height: res.height,
                pred: &res.pred,
                boxes: boxes.as_slice(),
            })
            .collect();
        let total_dets: usize = entries.iter().map(|e| e.boxes.len()).sum();
        let iou = (!e2e).then_some(self.play_iou);
        let json = batch_detections_json(
            model_name,
            model_id,
            self.play_conf,
            iou,
            self.play_class,
            self.play_batch.items.len(),
            &entries,
        );
        let fname = batch_export_filename(model_name, entries.len(), "json");
        let summary = format!(
            "{} · {}",
            counted(entries.len() as u64, "image", "images"),
            plural(total_dets, "detection")
        );
        Some(json.map(|s| (s, fname, summary)))
    }

    /// Builds the Batch CSV payload, suggested filename, and summary string from fresh batch results.
    pub(crate) fn build_play_batch_export_csv(&self) -> Option<(String, String, String)> {
        let (model_name, _, _, owned) = self.collect_batch_export_rows()?;
        let entries: Vec<BatchImageExport<'_>> = owned
            .iter()
            .map(|(fname, res, boxes)| BatchImageExport {
                filename: fname.as_str(),
                width: res.width,
                height: res.height,
                pred: &res.pred,
                boxes: boxes.as_slice(),
            })
            .collect();
        let total_dets: usize = entries.iter().map(|e| e.boxes.len()).sum();
        let csv = batch_detections_csv(&entries);
        let fname = batch_export_filename(model_name, entries.len(), "csv");
        let summary = format!(
            "{} · {}",
            counted(entries.len() as u64, "image", "images"),
            plural(total_dets, "detection")
        );
        Some((csv, fname, summary))
    }

    /// Builds the per-image Batch Summary CSV payload respecting the active Batch QA filter.
    pub(crate) fn build_play_batch_summary_csv(&self) -> Option<(String, String, String)> {
        if !self.play_batch.is_active() {
            return None;
        }
        let filtered = self.play_batch_filtered_indices();
        if filtered.is_empty() {
            return None;
        }
        let model_id = self.play_model.as_deref();
        let shown = self.play_shown_model();
        let model_name = shown.map(|m| m.name.as_str()).or(model_id).unwrap_or("model");
        let e2e = shown.is_some_and(|m| self.model_e2e(m));

        let eff_class = self.play_batch.effective_class(self.play_class);
        let mut rows: Vec<BatchSummaryRow> = Vec::with_capacity(filtered.len());
        for idx in filtered {
            let item = &self.play_batch.items[idx];
            let filename = batch_tile_label(item.name.as_ref()).to_string();
            match &item.status {
                BatchItemStatus::Done(res) => {
                    let fresh = model_id.is_some_and(|m| {
                        res.matches_current(m, self.play_iou, self.play_target, self.play_conf, e2e)
                    });
                    if fresh {
                        let vis = filter_detections(&res.pred.boxes, self.play_conf, eff_class);
                        let mut class_names: BTreeSet<String> = BTreeSet::new();
                        let mut top_conf: Option<f32> = None;
                        for b in &vis {
                            let cname = if b.name.trim().is_empty() {
                                format!("class {}", b.class)
                            } else {
                                b.name.clone()
                            };
                            class_names.insert(cname);
                            if b.score.is_finite() {
                                top_conf = Some(top_conf.map_or(b.score, |prev| prev.max(b.score)));
                            }
                        }
                        rows.push(BatchSummaryRow {
                            filename,
                            status: "done",
                            detection_count: Some(vis.len()),
                            classes: class_names.into_iter().collect::<Vec<_>>().join("; "),
                            top_confidence: top_conf,
                            total_ms: Some(res.pred.timing.total_ms),
                            width: Some(res.width),
                            height: Some(res.height),
                            error: String::new(),
                        });
                    } else {
                        rows.push(BatchSummaryRow {
                            filename,
                            status: "stale",
                            detection_count: None,
                            classes: String::new(),
                            top_confidence: None,
                            total_ms: None,
                            width: Some(res.width),
                            height: Some(res.height),
                            error: String::new(),
                        });
                    }
                }
                BatchItemStatus::Pending => rows.push(BatchSummaryRow {
                    filename,
                    status: "pending",
                    detection_count: None,
                    classes: String::new(),
                    top_confidence: None,
                    total_ms: None,
                    width: None,
                    height: None,
                    error: String::new(),
                }),
                BatchItemStatus::Running => rows.push(BatchSummaryRow {
                    filename,
                    status: "running",
                    detection_count: None,
                    classes: String::new(),
                    top_confidence: None,
                    total_ms: None,
                    width: None,
                    height: None,
                    error: String::new(),
                }),
                BatchItemStatus::Failed(msg) => rows.push(BatchSummaryRow {
                    filename,
                    status: "failed",
                    detection_count: None,
                    classes: String::new(),
                    top_confidence: None,
                    total_ms: None,
                    width: None,
                    height: None,
                    error: msg.clone(),
                }),
            }
        }
        let csv = batch_summary_csv(&rows);
        let fname = batch_summary_export_filename(model_name, rows.len());
        let summary = counted(rows.len() as u64, "image", "images");
        Some((csv, fname, summary))
    }

    /// Renders the **Batch JSON**, **Batch CSV**, and **Summary CSV** export buttons with focus
    /// continuity across disabled/enabled state transitions. In Compare A/B mode labels explicitly
    /// denote Model A.
    pub(super) fn play_batch_export_row(&self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let has_batch = self.has_play_batch_export();
        let filtered_n = self.play_batch_filtered_indices().len();
        let has_summary = self.play_batch.is_active() && filtered_n > 0;
        let compare_on = self.play_compare.enabled && self.store.models.ready().is_none_or(|m| !m.is_empty());
        let (_, json_label, csv_label) = super::batch::batch_action_labels(compare_on, false);
        let summary_label = if compare_on { "Summary CSV (A)" } else { "Summary CSV" };
        let json_tip = if compare_on {
            "Save completed Model A batch detections as JSON"
        } else {
            "Save completed batch detections as JSON"
        };
        let csv_tip = if compare_on {
            "Save completed Model A batch detections as CSV"
        } else {
            "Save completed batch detections as CSV"
        };
        let model_tag = if compare_on { "Model A " } else { "" };
        let summary_tip = if self.play_batch.qa_filter == super::batch_qa::BatchQaFilter::All {
            format!(
                "Save per-image {model_tag}batch summary CSV ({})",
                counted(filtered_n as u64, "image", "images")
            )
        } else {
            format!(
                "Save filtered per-image {model_tag}batch summary CSV ({} matching QA filter)",
                counted(filtered_n as u64, "image", "images")
            )
        };
        let disabled_tip = if compare_on {
            "Run Model A on at least one batch image first"
        } else {
            "Run at least one batch image first"
        };
        for (key, enabled) in [
            ("play-export-batch-json", has_batch),
            ("play-export-batch-csv", has_batch),
            ("play-export-batch-summary-csv", has_summary),
        ] {
            let dis = format!("{key}-disabled");
            if !enabled && self.fh(key, cx).is_focused(window) {
                window.focus(&self.fh(dis, cx));
            } else if enabled && self.fh(&dis, cx).is_focused(window) {
                window.focus(&self.fh(key, cx));
            }
        }
        let batch_json_btn = if has_batch {
            self.button(
                "play-export-batch-json",
                json_label,
                Btn::Secondary,
                Some("icons/file-text.svg"),
                window,
                cx,
                act(|this, _, cx| this.export_play_batch_json(cx)),
            )
            .tooltip(crate::controls::tip(json_tip, None))
            .into_any_element()
        } else {
            self.button_disabled(
                "play-export-batch-json",
                json_label,
                Btn::Secondary,
                Some("icons/file-text.svg"),
                disabled_tip,
                window,
                cx,
            )
            .into_any_element()
        };
        let batch_csv_btn = if has_batch {
            self.button(
                "play-export-batch-csv",
                csv_label,
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(|this, _, cx| this.export_play_batch_csv(cx)),
            )
            .tooltip(crate::controls::tip(csv_tip, None))
            .into_any_element()
        } else {
            self.button_disabled(
                "play-export-batch-csv",
                csv_label,
                Btn::Secondary,
                Some("icons/download.svg"),
                disabled_tip,
                window,
                cx,
            )
            .into_any_element()
        };
        let summary_csv_btn = if has_summary {
            self.button(
                "play-export-batch-summary-csv",
                summary_label,
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(|this, _, cx| this.export_play_batch_summary_csv(cx)),
            )
            .tooltip(crate::controls::tip(summary_tip, None))
            .into_any_element()
        } else {
            self.button_disabled(
                "play-export-batch-summary-csv",
                summary_label,
                Btn::Secondary,
                Some("icons/download.svg"),
                "No batch images match the active QA filter",
                window,
                cx,
            )
            .into_any_element()
        };
        div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(6.))
            .child(batch_json_btn)
            .child(batch_csv_btn)
            .child(summary_csv_btn)
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ImageSize;

    fn fake_pred(boxes: Vec<ScoredBox>) -> PredictResponse {
        PredictResponse {
            model_id: "m_1".into(),
            image: ImageSize { width: 640, height: 480 },
            boxes,
            timing: Timing { preprocess_ms: 1.0, inference_ms: 8.0, postprocess_ms: 0.5, total_ms: 9.5 },
            device: "cpu".into(),
            cold: false,
        }
    }

    #[test]
    fn batch_csv_formats_required_columns_and_escapes_special_characters() {
        let b1 = ScoredBox {
            class: 0,
            name: "stop, \"sign\"".into(),
            score: 0.91234,
            x1: 10.5,
            y1: 20.0,
            x2: 110.25,
            y2: 70.5,
        };
        let b2 = ScoredBox {
            class: 2,
            name: String::new(),
            score: f32::NAN,
            x1: 50.0,
            y1: 40.0,
            x2: 30.0,
            y2: 60.0,
        };
        let p1 = fake_pred(vec![b1.clone(), b2.clone()]);
        let p2 = fake_pred(vec![]);
        let refs1 = vec![&b1, &b2];
        let refs2: Vec<&ScoredBox> = vec![];
        let entries = vec![
            BatchImageExport {
                filename: "dir/img, 01.jpg",
                width: 640,
                height: 480,
                pred: &p1,
                boxes: &refs1,
            },
            BatchImageExport { filename: "empty.png", width: 320, height: 240, pred: &p2, boxes: &refs2 },
        ];
        let csv = batch_detections_csv(&entries);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "filename,class,confidence,x1,y1,x2,y2,width,height");
        assert_eq!(
            lines[1],
            "\"dir/img, 01.jpg\",\"stop, \"\"sign\"\"\",0.9123,10.50,20.00,110.25,70.50,99.75,50.50"
        );
        assert_eq!(lines[2], "\"dir/img, 01.jpg\",class 2,0.0000,50.00,40.00,30.00,60.00,0.00,20.00");
        assert_eq!(lines.len(), 3, "empty image contributes no detection rows");
    }

    #[test]
    fn batch_json_serializes_summary_and_per_image_entries() {
        let b1 = ScoredBox {
            class: 0,
            name: "circle".into(),
            score: 0.88,
            x1: 12.0,
            y1: 14.0,
            x2: 64.0,
            y2: 80.0,
        };
        let p1 = fake_pred(vec![b1.clone()]);
        let p2 = fake_pred(vec![]);
        let refs1 = vec![&b1];
        let refs2: Vec<&ScoredBox> = vec![];
        let entries = vec![
            BatchImageExport { filename: "a.jpg", width: 640, height: 480, pred: &p1, boxes: &refs1 },
            BatchImageExport { filename: "b.jpg", width: 640, height: 480, pred: &p2, boxes: &refs2 },
        ];
        let json = batch_detections_json("folo-n", "m_1", 0.25, Some(0.7), None, 3, &entries).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["mode"], "batch");
        assert_eq!(v["model"], "folo-n");
        assert_eq!(v["total_images"], 3);
        assert_eq!(v["exported_images"], 2);
        assert_eq!(v["total_detections"], 1);
        assert_eq!(v["images"][0]["filename"], "a.jpg");
        assert_eq!(v["images"][0]["detection_count"], 1);
        assert_eq!(v["images"][1]["filename"], "b.jpg");
        assert_eq!(v["images"][1]["detection_count"], 0);
        assert_eq!(batch_export_filename("folo-n v1.2", 6, "csv"), "playground-batch-6-folo-n-v1-2.csv");
        assert_eq!(
            batch_summary_export_filename("folo-n v1.2", 4),
            "playground-batch-summary-4-folo-n-v1-2.csv"
        );
    }

    #[test]
    fn batch_summary_csv_formats_all_columns_and_protects_against_formulas() {
        let rows = vec![
            BatchSummaryRow {
                filename: "=cmd.png".into(),
                status: "done",
                detection_count: Some(2),
                classes: "circle; square".into(),
                top_confidence: Some(0.94218),
                total_ms: Some(12.345),
                width: Some(640),
                height: Some(480),
                error: String::new(),
            },
            BatchSummaryRow {
                filename: "broken, file.jpg".into(),
                status: "failed",
                detection_count: None,
                classes: String::new(),
                top_confidence: None,
                total_ms: None,
                width: None,
                height: None,
                error: "+decode failed".into(),
            },
        ];
        let csv = batch_summary_csv(&rows);
        assert!(
            csv.starts_with(
                "filename,status,detection_count,classes,top_confidence,total_ms,width,height,error\r\n"
            ),
            "RFC 4180 CSV uses CRLF record endings"
        );
        assert!(csv.ends_with("\r\n"));
        let lines: Vec<&str> = csv.split("\r\n").filter(|l| !l.is_empty()).collect();
        assert_eq!(
            lines[0],
            "filename,status,detection_count,classes,top_confidence,total_ms,width,height,error"
        );
        assert_eq!(lines[1], "'=cmd.png,done,2,circle; square,0.9422,12.35,640,480,");
        assert_eq!(lines[2], "\"broken, file.jpg\",failed,,,,,,,'+decode failed");

        // Embedded newlines inside quoted fields stay preserved verbatim while record endings remain CRLF.
        let multiline = batch_summary_csv(&[BatchSummaryRow {
            filename: "a.png".into(),
            status: "failed",
            detection_count: None,
            classes: String::new(),
            top_confidence: None,
            total_ms: None,
            width: None,
            height: None,
            error: "line1\nline2".into(),
        }]);
        assert_eq!(
            multiline,
            "filename,status,detection_count,classes,top_confidence,total_ms,width,height,error\r\na.png,failed,,,,,,,\"line1\nline2\"\r\n"
        );
    }
}
