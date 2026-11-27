//! Sprint 17 end-to-end Desktop tests: FOLO v2 (`folo2-n..x`) parity across New Run, Runs & Models
//! filters/badges, INT8 PTQ CLI parity, and Models Compare (`Route::ModelCompare`) navigation.

use std::sync::Arc;

use super::tests::{drive, open};
use super::{FoloApp, Route, ServerMode};
use crate::api::mock::MockBackend;
use crate::api::{Arch, Backend};
use crate::arch::{ArchFamily, ModelArch};
use crate::screens::models::export_builder::{
    CliExportFormat, ModelCliMode, ModelCliState, QuantF32Policy, build_model_cli_with_arch,
};

fn drawn_keys(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

#[test]
fn new_run_supports_folo2_selection_name_suggestion_cloud_guard_and_submit() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        assert!(this.store.supports_folo2(), "mock advertises folo2_arch");
        this.open_new_run(Some("ds_8f2c"), w, cx);
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "nr-arch-fam"), "family toggle rendered: {keys:?}");

    // Switch to FOLO v2 + medium and verify suggested run name updates to `shapes-folo2-m-*`.
    drive(&mut app, window, |this, _, cx| {
        this.nr_set_arch_family(ArchFamily::Folo2, cx);
        this.nr_set_arch(2, cx); // m
        let f = this.new_run.as_ref().unwrap();
        assert_eq!(f.model_arch(), ModelArch::v2(Arch::M));
        let name = f.name.read(cx).text().to_string();
        assert!(name.starts_with("shapes-folo2-m-"), "unexpected suggested name: {name}");
        assert!(this.nr_folo2_blocked().is_none(), "local folo2 run is allowed");

        // Selecting a cloud connection blocks folo2 until switched back to Local or v1.
        this.new_run.as_mut().unwrap().run_on = Some("conn_runpod".into());
        assert!(
            this.nr_folo2_blocked().is_some_and(|m| m.contains("local-only")),
            "cloud + folo2 must be blocked"
        );
        // Back to local and submit.
        this.new_run.as_mut().unwrap().run_on = None;
    });

    drive(&mut app, window, |this, w, cx| {
        this.submit_new_run(w, cx);
    });
    app.run_until_idle();
    app.frame(window.into());

    drive(&mut app, window, |this, w, cx| {
        assert!(this.new_run.is_none(), "dialog closed after submit");
        assert_eq!(this.new_run_last.map(|t| (t.0, t.1)), Some((ArchFamily::Folo2, 2)));
        let created = this.store.runs.ready().unwrap().first().unwrap().clone();
        assert_eq!(this.store.run_arch(&created), ModelArch::v2(Arch::M));
        let cli = this.store.api.run_cli(&created.id).unwrap();
        assert!(cli.cli.contains("folo2-m"), "run_cli emits folo2-m: {}", cli.cli);

        // Clone config on a folo2 run carries over ArchFamily::Folo2.
        let cfg = created.config.clone();
        let arch = this.store.run_arch(&created);
        this.open_new_run_from_arch(&cfg, arch, None, w, cx);
        assert_eq!(this.new_run.as_ref().unwrap().model_arch(), ModelArch::v2(Arch::M));
    });
}

#[test]
fn runs_and_models_lists_filter_v1_and_v2_families_and_search_folo2() {
    let (mut app, window) = open();

    // Runs list: seeded demo has `run_d204` as `folo2-n` alongside legacy `folo-*` runs.
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Runs, w, cx));
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "runs-family-filter"), "{keys:?}");

    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.run_family_filter = Some(ArchFamily::Folo2);
        cx.notify();
    });
    let v2_runs: Vec<String> =
        drawn_keys(&mut app, window).into_iter().filter(|k| k.starts_with("run-row-")).collect();
    assert_eq!(v2_runs, vec!["run-row-run_d204"], "v2 filter isolates folo2 runs");

    // Searching "folo2" with family filter cleared also finds `run_d204`.
    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.run_family_filter = None;
        this.run_search.update(cx, |i, cx| i.set_text("folo2".to_string(), cx));
        cx.notify();
    });
    let search_v2: Vec<String> =
        drawn_keys(&mut app, window).into_iter().filter(|k| k.starts_with("run-row-")).collect();
    assert_eq!(search_v2, vec!["run-row-run_d204"]);

    // Models list: seeded demo has `m_7c12` as `folo2-n` alongside legacy `folo-*` models.
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Models, w, cx));
    let m_keys = drawn_keys(&mut app, window);
    assert!(m_keys.iter().any(|k| k == "models-family-filter"), "{m_keys:?}");
    assert!(m_keys.iter().any(|k| k == "models-compare"), "Compare models action rendered: {m_keys:?}");

    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.models_family_filter = Some(ArchFamily::Folo2);
        cx.notify();
    });
    let v2_models: Vec<String> =
        drawn_keys(&mut app, window).into_iter().filter(|k| k.starts_with("model-row-")).collect();
    assert_eq!(v2_models, vec!["model-row-m_7c12"]);

    // Backend switch clears family filters and model compare ids.
    drive(&mut app, window, |this, _, cx| {
        this.run_family_filter = Some(ArchFamily::Folo2);
        this.models_family_filter = Some(ArchFamily::Folo2);
        this.model_compare.ids = vec!["m_7c12".into(), "m_44c0".into()];
        this.set_backend(Arc::new(MockBackend::new(4200, true)), ServerMode::Mock, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.run_family_filter, None);
        assert_eq!(this.models_family_filter, None);
        assert!(this.model_compare.ids.is_empty());
    });
}

#[test]
fn folo2_int8_quantize_cli_builder_emits_selective_f32_flags_and_v2_hints() {
    let api = MockBackend::new(4200, true);
    let mut m = api.models().unwrap().into_iter().find(|m| m.id == "m_7c12").unwrap();
    m.path = Some("/weights/shapes-folo2-n.folo".into());

    let mut st = ModelCliState {
        mode: ModelCliMode::Export,
        export_format: CliExportFormat::Int8Quantize,
        quant_f32_policy: QuantF32Policy::Auto,
        ..Default::default()
    };
    let auto = build_model_cli_with_arch(&m, ModelArch::v2(Arch::N), Some("/data/shapes"), &st);
    assert!(
        auto.command.starts_with("folo quantize --model /weights/shapes-folo2-n.folo --data /data/shapes")
    );
    assert!(auto.hint.as_deref().is_some_and(|h| h.contains("FOLO v2 selective-f32")));

    st.quant_f32_policy = QuantF32Policy::P5Tail;
    let p5 = build_model_cli_with_arch(&m, ModelArch::v2(Arch::N), Some("/data/shapes"), &st);
    assert!(p5.command.contains("--f32-policy p5-tail"));
    assert!(p5.hint.as_deref().is_some_and(|h| h.contains("FOLO v2 P5 neck tail")));
}

#[test]
fn review_fixes_arch_caches_probe_sync_and_finetune_resolution() {
    let (mut app, window) = open();

    // 1. Store::run_arch resolves fine-tune runs from config.weights even before a checkpoint exists.
    drive(&mut app, window, |this, _, _| {
        let mut fake_run = this.store.runs.ready().unwrap()[0].clone();
        fake_run.id = "run_ft_v2".into();
        fake_run.model_id = None;
        fake_run.config.arch = Arch::N;
        fake_run.config.weights = Some("m_7c12".into()); // seeded folo2-n model
        assert_eq!(this.store.run_arch(&fake_run), ModelArch::v2(Arch::N));
    });

    // 2. MockBackend infers folo2-* from config.weights when arch_override is None (and blocks cloud).
    let mock = MockBackend::new(4200, true);
    let mut cfg = mock.runs().unwrap()[0].config.clone();
    cfg.name = "ft-from-v2-weights".into();
    cfg.arch = Arch::N;
    cfg.weights = Some("m_7c12".into());
    cfg.compute = None;
    let created = mock.create_run(cfg.clone()).unwrap();
    let (run_archs, _) = mock.known_archs();
    assert!(
        run_archs.iter().any(|(id, a)| id == &created.id && a == "folo2-n"),
        "fine-tune from v2 weights records folo2-n: {run_archs:?}"
    );

    // 3. In-flight probe blocks Start and late probe completion promotes open New Run form to v2.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(Some("ds_8f2c"), w, cx);
        let f = this.new_run.as_mut().unwrap();
        f.arch_family = ArchFamily::Folo;
        f.arch = 0;
        f.weights = Some("m_late_v2".into());
        this.store.probing_model_archs.insert("m_late_v2".into());
        assert_eq!(this.nr_folo2_blocked(), Some("Checking model architecture…"));

        // Probe completes with folo2-l -> unblocks & promotes the open form.
        this.store.probing_model_archs.remove("m_late_v2");
        this.store.probed_model_archs.insert("m_late_v2".into());
        this.store.note_model_arch("m_late_v2", ModelArch::v2(Arch::L));
        this.sync_new_run_arch_after_probe(cx);
        assert!(this.nr_folo2_blocked().is_none());
        assert_eq!(this.new_run.as_ref().unwrap().model_arch(), ModelArch::v2(Arch::L));
        this.close_new_run(w, cx);
    });

    // 4. set_backend clears stale arch caches and new_run_last; open_model_compare preserves seed before models load.
    drive(&mut app, window, |this, w, cx| {
        this.store.probed_model_archs.insert("stale_m".into());
        this.store.probed_run_archs.insert("stale_r".into());
        this.store.run_archs.insert("stale_r".into(), ModelArch::v2(Arch::X));
        this.new_run_last = Some((ArchFamily::Folo2, 3, 2, 100.0));
        this.set_backend(Arc::new(MockBackend::new(4200, true)), ServerMode::Mock, cx);
        assert!(!this.store.probed_model_archs.contains("stale_m"));
        assert!(!this.store.probed_run_archs.contains("stale_r"));
        assert!(!this.store.run_archs.contains_key("stale_r"));
        assert!(this.new_run_last.is_none());
        // Models are Load::Idle right after set_backend: open_model_compare must keep seed.
        this.open_model_compare(Some("m_7c12".into()), w, cx);
        assert_eq!(this.model_compare.ids, vec!["m_7c12"]);
    });
}
