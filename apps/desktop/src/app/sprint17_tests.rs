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

    // 3. In-flight probe sets nr_arch_probe_pending (neutral pending, not error), late probe
    // promotes an open form AND falls through to source_run_id when weights is a file path, and
    // reopening a dismissed draft also reconciles the newly resolved v2 arch.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(Some("ds_8f2c"), w, cx);
        let f = this.new_run.as_mut().unwrap();
        f.arch_family = ArchFamily::Folo;
        f.arch = 0;
        f.weights = Some("m_late_v2".into());
        this.store.probing_model_archs.insert("m_late_v2".into());
        assert!(this.nr_arch_probe_pending());
        assert!(this.nr_folo2_blocked().is_none(), "pending probe is not an error");

        // Probe completes with folo2-l -> unblocks & promotes the open form.
        this.store.probing_model_archs.remove("m_late_v2");
        this.store.probed_model_archs.insert("m_late_v2".into());
        this.store.note_model_arch("m_late_v2", ModelArch::v2(Arch::L));
        this.sync_new_run_arch_after_probe(cx);
        assert!(!this.nr_arch_probe_pending());
        assert_eq!(this.new_run.as_ref().unwrap().model_arch(), ModelArch::v2(Arch::L));

        // Cloned fine-tune run with raw `.folo` weights path falls through to source_run_id
        // and also reconciles when restored from a dismissed draft.
        let run_v2 = this.store.run("run_d204").unwrap().clone();
        let f = this.new_run.as_mut().unwrap();
        f.arch_family = ArchFamily::Folo;
        f.arch = 0;
        f.weights = Some("runs/prev/weights/best.folo".into());
        f.source_run_id = Some(run_v2.id.clone());
        f.arch_touched = false;
        this.close_new_run(w, cx);
        // Reopen draft: must reconcile from provisional v1 to run_d204's folo2-n.
        this.open_new_run(None, w, cx);
        assert_eq!(this.new_run.as_ref().unwrap().model_arch(), ModelArch::v2(Arch::N));
        this.close_new_run(w, cx);
    });

    // 4. hold_loading never strands probing_*_archs.
    drive(&mut app, window, |this, _, cx| {
        this.hold_loading = true;
        this.server_mode = ServerMode::Local;
        this.probe_model_arch("m_hold_test", cx);
        this.probe_run_arch("run_d204", cx);
        assert!(!this.store.probing_model_archs.contains("m_hold_test"));
        assert!(!this.store.probing_run_archs.contains("run_d204"));
        this.hold_loading = false;
        this.server_mode = ServerMode::Mock;
    });

    // 5. set_backend clears stale arch caches, bumps arch_probe_gen, and clears new_run_last;
    // open_model_compare preserves seed before models load.
    drive(&mut app, window, |this, w, cx| {
        let gen_before = this.store.arch_probe_gen;
        this.store.probed_model_archs.insert("stale_m".into());
        this.store.probed_run_archs.insert("stale_r".into());
        this.store.run_archs.insert("stale_r".into(), ModelArch::v2(Arch::X));
        this.new_run_last = Some((ArchFamily::Folo2, 3, 2, 100.0));
        this.set_backend(Arc::new(MockBackend::new(4200, true)), ServerMode::Mock, cx);
        assert_ne!(this.store.arch_probe_gen, gen_before);
        assert!(!this.store.probed_model_archs.contains("stale_m"));
        assert!(!this.store.probed_run_archs.contains("stale_r"));
        assert!(!this.store.run_archs.contains_key("stale_r"));
        assert!(this.new_run_last.is_none());
        // Models are Load::Idle right after set_backend: open_model_compare must keep seed.
        this.open_model_compare(Some("m_7c12".into()), w, cx);
        assert_eq!(this.model_compare.ids, vec!["m_7c12"]);
    });
}

#[test]
fn playtest_r17_fixes_short_run_clone_mock_v2_metadata_and_estimate_basis() {
    let (mut app, window) = open();

    // 1. Cloning a valid 1-epoch folo2 run with default `close_mosaic = 10` clamps close_mosaic
    // so `close_invalid` is false and Start training succeeds with the same v2 arch.
    drive(&mut app, window, |this, w, cx| {
        let mut short_run = this.store.run("run_d204").unwrap().clone();
        short_run.config.epochs = 1;
        short_run.config.augment.close_mosaic = 10;
        short_run.config.head = crate::api::Head::O2o;
        let cfg = short_run.config.clone();
        this.open_new_run_from_run(&short_run, &cfg, None, w, cx);
        let f = this.new_run.as_ref().unwrap();
        assert!(!f.close_invalid, "short-run clone must not mark close_mosaic invalid");
        assert_eq!(f.close_mosaic.round() as u32, 0);
        assert_eq!(f.model_arch(), ModelArch::v2(Arch::N));
        assert!(f.e2e);
    });
    // Flush TextInput::set_text subscribers and verify close_invalid stays false before submit.
    app.frame(window.into());
    app.run_until_idle();
    drive(&mut app, window, |this, w, cx| {
        let f = this.new_run.as_ref().unwrap();
        assert!(!f.close_invalid, "close_invalid stays false after input subscribers flush");
        this.submit_new_run(w, cx);
    });
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(this.new_run.is_none(), "short-run clone submitted cleanly");
        let created = this.store.runs.ready().unwrap().first().unwrap().clone();
        assert_eq!(this.store.run_arch(&created), ModelArch::v2(Arch::N));
        // 3. Startup logs for a demo folo2-n run identify `folo2-n` and `2.85 M` params.
        let logs = this.store.api.logs(&created.id).unwrap();
        assert!(logs.lines.iter().any(|l| l.message.contains("folo train folo2-n")));
        assert!(logs.lines.iter().any(|l| l.message.contains("model: folo2-n · 2.85 M params")));
    });

    // 3b & 4. Seeded demo v2 model (`m_7c12`) has consistent 2.85M params, and demo v2 estimates
    // use Reference basis instead of inheriting v1 Measured rows.
    let mock = MockBackend::new(4200, true);
    let m_v2 = mock.models().unwrap().into_iter().find(|m| m.id == "m_7c12").unwrap();
    assert_eq!(m_v2.params, 2_850_000);
    let d204_logs = mock.logs("run_d204").unwrap();
    assert!(d204_logs.lines.iter().any(|l| l.message.contains("folo train folo2-n")));
    assert!(d204_logs.lines.iter().any(|l| l.message.contains("model: folo2-n · 2.85 M params")));

    let mut est_cfg = mock.runs().unwrap()[0].config.clone();
    est_cfg.arch = Arch::S;
    est_cfg.batch = 16;
    est_cfg.imgsz = 640;
    est_cfg.amp = false;
    est_cfg.compute = None;
    let est_v2 = mock.estimate_run_with_arch(&est_cfg, Some("folo2-s")).unwrap();
    assert_eq!(
        est_v2.basis.source,
        crate::api::ThroughputSource::Reference,
        "demo v2 estimate must not inherit v1 measured throughput rows"
    );
}

#[test]
fn playtest_r17_fixes_compare_fourth_model_focus_and_reconnected_arch_cache() {
    let (mut app, window) = open();

    // 2. Adding the 4th model in Compare (which unmounts `model-cmp-add`) moves focus to the
    // newly added model chip rather than stranding keyboard focus on Overview.
    drive(&mut app, window, |this, w, cx| {
        this.open_model_compare(Some("m_7c12".into()), w, cx);
    });
    drawn_keys(&mut app, window);

    drive(&mut app, window, |this, _, cx| {
        let all: Vec<String> = this.store.models.ready().unwrap().iter().map(|m| m.id.clone()).collect();
        let third = all.iter().find(|id| !this.model_compare.ids.contains(id)).unwrap().clone();
        this.model_compare_add(third, cx);
    });
    drawn_keys(&mut app, window);

    drive(&mut app, window, |this, w, cx| {
        let all: Vec<String> = this.store.models.ready().unwrap().iter().map(|m| m.id.clone()).collect();
        let fourth = all.iter().find(|id| !this.model_compare.ids.contains(id)).unwrap().clone();
        assert_eq!(this.model_compare.ids.len(), 3);
        let will_unmount = this.model_compare.ids.len() + 1 >= 4;
        this.model_compare_add(fourth.clone(), cx);
        assert!(will_unmount);
        let target = this.fh(format!("model-cmp-open-{fourth}"), cx);
        w.focus(&target);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        let fourth = this.model_compare.ids[3].clone();
        assert!(
            this.fh(format!("model-cmp-open-{fourth}"), cx).is_focused(w),
            "focus must stay on the 4th chip after model-cmp-add unmounts"
        );

        // Reconnect clears stale arch caches and bumps arch_probe_gen.
        let gen_before = this.store.arch_probe_gen;
        this.store.probed_model_archs.insert("stale_after_outage".into());
        this.reconnected(cx);
        assert_ne!(this.store.arch_probe_gen, gen_before);
        assert!(!this.store.probed_model_archs.contains("stale_after_outage"));
    });

    // 8. Dataset Statistics scroll container is tracked by `page_scroll` so keyboard Tabbing
    // down to the Export COCO Copy button can scroll it fully into view.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Dataset("ds_3b91".into()), w, cx);
        this.ds_tab = crate::app::DsTab::Statistics;
        this.ds_cli.mode = crate::screens::datasets::DatasetCliMode::ExportCoco;
        cx.notify();
    });
    drawn_keys(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(
            f32::from(this.page_scroll.max_offset().height) > 0.0,
            "ds-stats must track page_scroll so Tab reveal scrolls Copy command into view"
        );
    });
}
