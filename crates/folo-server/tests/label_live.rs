//! Agent labeling against the real CLIs: Claude Code and Codex label synthetic shapes whose boxes are known, and
//! the tests check what the integration promises: one call per batch, a small token bill per call, and boxes that
//! land on the objects (so the coordinate convention holds end to end).
//!
//! Ignored by default: they need the CLI installed and signed in, and they spend (a few thousand) tokens. Run
//! one with `cargo test -p folo-server --test label_live -- --ignored --nocapture claude` (or `codex`);
//! `FOLO_LABEL_LIVE_MODEL` names another model to try.

use std::path::Path;
use std::time::Duration;

use folo_server::api::*;
use folo_server::{RealStore, Store};
use folo_test_support::TestDir;

const IMAGES: usize = 8;
const CLASSES: [&str; 3] = ["circle", "square", "triangle"];

/// Intersection over union of two pixel boxes.
fn iou(a: &BBox, b: &BBox) -> f32 {
    let w = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let h = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let area = |r: &BBox| (r.x2 - r.x1) * (r.y2 - r.y1);
    w * h / (area(a) + area(b) - w * h)
}

/// Labels [`IMAGES`] synthetic images with `agent` and checks the run against the images' true boxes.
/// `max_input_per_call` bounds the input tokens of one call.
fn label_shapes(agent: LabelAgentId, model: Option<&str>, max_input_per_call: u64) {
    let t = TestDir::new(&format!("label-live-{}", agent.as_str()));
    let shapes = t.join("shapes");
    let cfg = folo_data::synthetic::SyntheticConfig {
        images: IMAGES,
        classes: CLASSES.map(String::from).to_vec(),
        imgsz: 320,
        seed: 11,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        edge_fraction: 0.0,
        max_occlusion: 0.0,
        val_fraction: 0.0,
        ..Default::default()
    };
    folo_data::synthetic::write_yolo(&cfg, &shapes, folo_data::ImageFormat::Jpeg(90)).unwrap();
    let truth = folo_data::open(&shapes).unwrap();

    let store = RealStore::open(t.join("home"));
    let labeler = store.labeler().unwrap();
    let info = labeler.agents().into_iter().find(|a| a.id == agent).unwrap();
    assert!(info.available, "{} is not installed: {:?}", info.name, info.hint);
    let job = labeler
        .create(CreateLabelJob {
            name: "shapes".into(),
            classes: CLASSES.map(String::from).to_vec(),
            instructions: "Flat coloured shapes on a textured background.".into(),
            path: Some(shapes.join("images").join("train").to_string_lossy().into_owned()),
        })
        .unwrap();
    assert_eq!(job.counts.images as usize, IMAGES);

    let model = std::env::var("FOLO_LABEL_LIVE_MODEL").ok().or(model.map(String::from));
    let start: StartLabeling =
        serde_json::from_value(serde_json::json!({ "agent": agent, "model": model })).unwrap();
    labeler.start(&job.id, start).unwrap();
    let job = loop {
        let job = labeler.get(&job.id).unwrap();
        if job.status == LabelJobStatus::Idle {
            break job;
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let run = job.run.clone().unwrap();
    let logs = labeler.logs(&job.id, 100).unwrap();
    let log = logs.iter().map(|l| l.message.as_str()).collect::<Vec<_>>().join("\n");
    assert_eq!(run.outcome, Some(LabelOutcome::Completed), "{:?} {:?}\n{log}", run.error, run.hint);
    assert_eq!((job.counts.labeled as usize, job.counts.failed), (IMAGES, 0), "{log}");

    // One call per batch, and no call near what a default (tool-laden) agent session costs.
    let batch = if agent == LabelAgentId::Claude { 4 } else { 8 };
    assert_eq!(run.usage.calls as usize, IMAGES.div_ceil(batch), "{log}");
    let per_call = run.usage.input_tokens / run.usage.calls;
    assert!(per_call <= max_input_per_call, "{per_call} input tokens per call\n{log}");

    // Every true box against the best labeled box of its class.
    let labeled = labeler.images(&job.id, None, 0, 1000).unwrap().images;
    let (mut boxes, mut found, mut iou_sum) = (0usize, 0usize, 0.0f32);
    for entry in truth.train.entries() {
        let name = Path::new(&entry.image).file_name().unwrap().to_string_lossy();
        let image = labeled.iter().find(|i| i.file == name).unwrap();
        for t in &entry.boxes {
            let [x1, y1, x2, y2] = t.xyxy();
            let (w, h) = (image.width as f32, image.height as f32);
            let t = BBox { class: t.class, x1: x1 * w, y1: y1 * h, x2: x2 * w, y2: y2 * h };
            let best =
                image.boxes.iter().filter(|b| b.class == t.class).map(|b| iou(&t, b)).fold(0.0, f32::max);
            boxes += 1;
            iou_sum += best;
            found += usize::from(best >= 0.5);
        }
    }
    println!(
        "{} ({}): {found} of {boxes} boxes found at IoU 0.5, mean IoU {:.2}; {} calls, {} input tokens \
         ({} cached), {} output, cost {:?}",
        info.name,
        run.model.as_deref().unwrap_or("default"),
        iou_sum / boxes as f32,
        run.usage.calls,
        run.usage.input_tokens,
        run.usage.cached_input_tokens,
        run.usage.output_tokens,
        run.usage.cost_usd,
    );
    assert!(found * 10 >= boxes * 6, "only {found} of {boxes} boxes were found\n{log}");
}

#[test]
#[ignore = "calls the real Claude Code CLI: needs it installed and signed in, and spends tokens"]
fn claude_code_labels_shapes() {
    // Four 320 px images and the instructions are about 1500 tokens; a default Claude Code session starts at over
    // ten times that.
    label_shapes(LabelAgentId::Claude, Some("haiku"), 4_000);
}

#[test]
#[ignore = "calls the real Codex CLI: needs it installed and signed in, and spends tokens"]
fn codex_labels_shapes() {
    // Codex's own instructions are most of this; a default `codex exec` call is about 20000.
    label_shapes(LabelAgentId::Codex, None, 12_000);
}
