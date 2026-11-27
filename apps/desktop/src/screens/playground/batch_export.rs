//! Playground batch export serializers: structured Batch JSON and RFC 4180 Batch CSV
//! (`filename,class,confidence,x1,y1,x2,y2,width,height`).

use crate::api::{PredictResponse, ScoredBox, Timing};

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

/// Formats completed Playground batch detections as CSV with RFC 4180 field quoting and LF (`\n`)
/// record endings, using columns: `filename,class,confidence,x1,y1,x2,y2,width,height`.
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
            push_csv_field(&mut out, entry.filename);
            out.push(',');
            push_csv_field(&mut out, &cls);
            out.push_str(&format!(",{score:.4},{x1:.2},{y1:.2},{x2:.2},{y2:.2},{width:.2},{height:.2}\n"));
        }
    }
    out
}

fn push_csv_field(out: &mut String, field: &str) {
    if field.contains([',', '"', '\n', '\r']) {
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
    } else {
        out.push_str(field);
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
    }
}
