//! Deterministic unit and headless UI tests for side-by-side Model Compare (`Route::ModelCompare`)
//! and CSV / Markdown comparison export.

use super::*;
use crate::api::{Arch, ClassMetrics, Head, Model, ModelEval, ModelMetrics};
use crate::app::tests::{drive, open};
use crate::arch::ModelArch;

fn sample_model(id: &str, name: &str, arch: Arch, map50_95: f32, map50: f32) -> Model {
    Model {
        id: id.into(),
        name: name.into(),
        arch,
        imgsz: 640,
        classes: vec!["circle".into(), "square".into()],
        params: 2_630_000,
        flops_g: 6.6,
        bytes: 5_400_000,
        created: "2026-11-08T00:00:00Z".into(),
        run_id: None,
        metrics: Some(ModelMetrics {
            map50,
            map50_95,
            precision: 0.90,
            recall: 0.85,
            best_f1_conf: Some(0.32),
        }),
        head: Head::O2m,
        trained_with: None,
        precision: None,
        fused: Some(false),
        source: None,
        about: None,
        path: None,
    }
}

fn sample_eval(pairs: &[(u32, f32, f32)]) -> ModelEval {
    ModelEval {
        pr: Vec::new(),
        confusion: Vec::new(),
        per_class: pairs
            .iter()
            .map(|&(class, map50_95, map50)| ClassMetrics {
                class,
                map50,
                map50_95,
                precision: 0.9,
                recall: 0.85,
            })
            .collect(),
        images: 100,
        best_f1_conf: Some(0.32),
        dataset_id: None,
        dataset_name: None,
        split: None,
        split_fallback: None,
        confusion_conf: None,
        confusion_iou: None,
    }
}

#[test]
fn csv_export_has_uniform_columns_signed_deltas_and_formula_injection_protection() {
    let mut a = sample_model("m_a", "=CMD(\"calc\")", Arch::N, 0.700, 0.900);
    a.classes = vec!["=circle".into(), "square".into()];
    let mut b = sample_model("m_b", "folo2-n-best", Arch::N, 0.742, 0.915);
    b.head = Head::O2o;
    b.classes = vec!["=circle".into(), "square".into(), "star".into()];
    let ea = sample_eval(&[(0, 0.650, 0.880), (1, 0.750, 0.920)]);
    let eb = sample_eval(&[(0, 0.700, 0.910), (1, 0.730, 0.900), (2, 0.600, 0.800)]);

    let csv = build_model_compare_csv(
        &[&a, &b],
        &[ModelArch::v1(Arch::N), ModelArch::v2(Arch::N)],
        &["shapes".into(), "shapes".into()],
        &[Some(&ea), Some(&eb)],
        true,
    );
    assert!(csv.contains("\r\n"), "RFC 4180 CRLF");
    let lines: Vec<&str> = csv.split("\r\n").filter(|l| !l.is_empty()).collect();
    assert!(lines.len() >= 18, "summary + per-class rows: {lines:#?}");
    // Formula-leading model name & class name must be neutralized with a leading apostrophe.
    assert!(lines[0].contains("'=CMD("), "neutralizes formula header: {}", lines[0]);
    assert!(csv.contains("per_class_ap50_95,'=circle,0.650,0.700,+0.050"), "{csv}");
    assert!(csv.contains("per_class_ap50_95,square,0.750,0.730,-0.020"), "{csv}");
    assert!(csv.contains("per_class_ap50_95,star,,0.600,"), "{csv}");
    assert!(csv.contains("summary,Architecture,folo-n,folo2-n,"), "{csv}");
    assert!(csv.contains("summary,Family,FOLO v1,FOLO v2,"), "{csv}");
    assert!(csv.contains("summary,mAP50-95,0.700,0.742,+0.042"), "{csv}");
    assert!(!csv.contains("Evaluation caveat"), "no caveat when same_set=true");

    // Mixed evaluation sets emit a caveat row with identical column count.
    let mixed_csv = build_model_compare_csv(
        &[&a, &b],
        &[ModelArch::v1(Arch::N), ModelArch::v2(Arch::N)],
        &["Validation split of shapes".into(), "Training split of shapes".into()],
        &[Some(&ea), Some(&eb)],
        false,
    );
    assert!(mixed_csv.contains("summary,Evaluation caveat,"), "{mixed_csv}");
    let caveat_line = mixed_csv.split("\r\n").find(|l| l.starts_with("summary,Evaluation caveat,")).unwrap();
    assert_eq!(caveat_line.matches(',').count(), 4, "uniform 5-column CSV row: {caveat_line}");
}

#[test]
fn markdown_export_escapes_markup_and_renders_summary_and_per_class_tables() {
    let mut a = sample_model("m_a", "base | [v1](http://x)", Arch::S, 0.624, 0.874);
    a.classes = vec!["part [left](right) | *a*".into()];
    let mut b = sample_model("m_b", "cand\nv2_`test`", Arch::S, 0.660, 0.900);
    b.classes = vec!["part [left](right) | *a*".into()];
    let ea = sample_eval(&[(0, 0.624, 0.874)]);
    let eb = sample_eval(&[(0, 0.660, 0.900)]);

    let md = build_model_compare_markdown(
        &[&a, &b],
        &[ModelArch::v1(Arch::S), ModelArch::v2(Arch::S)],
        &["shapes-hard".into(), "shapes-hard".into()],
        &[Some(&ea), Some(&eb)],
        false,
    );
    assert!(md.contains("# Model comparison"));
    assert!(md.contains("base \\| \\[v1\\]\\(http://x\\)"), "escapes link & pipe in model name: {md}");
    assert!(md.contains("cand v2\\_\\`test\\`"), "collapses newline & escapes code/underscore: {md}");
    assert!(
        md.contains("part \\[left\\]\\(right\\) \\| \\*a\\*"),
        "escapes link & emphasis in class name: {md}"
    );
    assert!(md.contains("different datasets"), "mixed dataset caveat included");
    assert!(md.contains("## Summary metrics"));
    assert!(md.contains("## Per-class AP50-95"));
    assert!(md.contains("| mAP50-95 | 0.624 | 0.660 | +0.036 |"));
    assert!(md.contains("| Architecture | folo-s | folo2-s | — |"));
}

#[test]
fn compare_uses_loaded_eval_provenance_for_split_labels_and_mixed_caveat() {
    use crate::api::Split;
    use crate::screens::models::groups::eval_set_with_report;
    use crate::store::Store;

    let api = std::sync::Arc::new(crate::api::mock::MockBackend::new(4200, true));
    let store = Store::new(api);

    // Two imported models (`run_id: None`) where the server's loaded ModelEval supplies concrete
    // dataset_id + split provenance: one on train, one on val -> must NOT claim same_eval_set!
    let a = sample_model("m_imp_a", "imported-a", Arch::N, 0.70, 0.90);
    let b = sample_model("m_imp_b", "imported-b", Arch::N, 0.75, 0.92);

    let mut ea = sample_eval(&[(0, 0.70, 0.90)]);
    ea.dataset_id = Some("ds_8f2c".into());
    ea.dataset_name = Some("shapes".into());
    ea.split = Some(Split::Train);

    let mut eb = sample_eval(&[(0, 0.75, 0.92)]);
    eb.dataset_id = Some("ds_8f2c".into());
    eb.dataset_name = Some("shapes".into());
    eb.split = Some(Split::Val);

    assert_eq!(eval_set_with_report(&store, &a, Some(&ea)).label, "Training split of shapes");
    assert_eq!(eval_set_with_report(&store, &b, Some(&eb)).label, "Validation split of shapes");
    assert!(!same_eval_set_with_evals(&store, &[&a, &b], &[Some(&ea), Some(&eb)]));

    // When both eval reports share dataset_id + split, they ARE directly comparable even if imported.
    ea.split = Some(Split::Val);
    assert!(same_eval_set_with_evals(&store, &[&a, &b], &[Some(&ea), Some(&eb)]));
}

#[test]
fn filename_slugs_cleanly() {
    let a = sample_model("m_1", "Shapes Hard (v2)", Arch::S, 0.7, 0.9);
    let b = sample_model("m_2", "Baseline/v1", Arch::S, 0.6, 0.8);
    assert_eq!(model_compare_filename(&[&a, &b], "csv"), "shapes-hard-v2-vs-baseline-v1-compare.csv");
    assert_eq!(model_compare_filename(&[&a, &b], "md"), "shapes-hard-v2-vs-baseline-v1-compare.md");
}

#[test]
fn model_compare_screen_opens_switches_baseline_and_copies_csv_and_markdown() {
    let (mut app, window) = open();
    // Open Model Compare seeded from the demo's FOLO v2 model (`m_7c12`) and verify UI + clipboard.
    drive(&mut app, window, |this, w, cx| {
        this.open_model_compare(Some("m_7c12".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.route, Route::ModelCompare);
        assert_eq!(this.model_compare.ids.first().map(String::as_str), Some("m_7c12"));
        assert!(this.model_compare.ids.len() >= 2);
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "model-cmp-copy-csv"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "model-cmp-save-csv"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "model-cmp-copy-md"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "model-cmp-save-md"), "{keys:?}");

        this.copy_model_compare_csv(cx);
        let csv = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert!(csv.contains("summary,Architecture,folo2-n,"), "v2 arch in CSV: {csv}");
        assert!(csv.contains("per_class_ap50_95,"), "per-class AP deltas included: {csv}");

        this.copy_model_compare_markdown(cx);
        let md = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert!(md.contains("## Summary metrics") && md.contains("## Per-class AP50-95"), "{md}");
        assert!(md.contains("folo2-n"), "{md}");

        // Promote candidate slot 1 to baseline.
        let cand = this.model_compare.ids[1].clone();
        this.model_compare_make_baseline(&cand, cx);
        assert_eq!(this.model_compare.ids[0], cand);
    });
}
