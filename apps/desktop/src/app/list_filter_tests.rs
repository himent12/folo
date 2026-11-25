//! Runs and Models live search + architecture filtering, completed/stopped run config cloning,
//! and Model Details inline class expansion + copy.

use std::sync::Arc;

use folo_ui::Focusable;

use super::tests::{drive, open};
use super::{FoloApp, Route, RunFilter, ServerMode};
use crate::api::mock::MockBackend;
use crate::api::{Arch, RunStatus};
use crate::screens::models::filter_models;
use crate::screens::runs::filter_runs;

fn drawn_keys(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

fn press_enter(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<FoloApp>) {
    let keystroke = folo_ui::Keystroke::parse("enter").unwrap();
    app.simulate_input(
        window.into(),
        folo_ui::PlatformInput::KeyDown(folo_ui::KeyDownEvent { keystroke, is_held: false }),
    );
    app.run_until_idle();
    app.frame(window.into());
}

#[test]
fn runs_list_filters_by_arch_and_search_and_resets_on_clear_or_reentry() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Runs, w, cx));
    let all_keys = drawn_keys(&mut app, window);
    assert!(all_keys.iter().any(|k| k == "runs-arch-filter"), "{all_keys:?}");
    let total_rows = all_keys.iter().filter(|k| k.starts_with("run-row-")).count();
    assert!(total_rows >= 2, "demo has multiple runs: {total_rows}");

    let (first_id, first_arch, first_name) = drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().first().unwrap();
        (r.id.clone(), r.config.arch, r.name.clone())
    });
    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.run_arch_filter = Some(first_arch);
        this.run_search.update(cx, |i, cx| i.set_text(first_name.clone(), cx));
        cx.notify();
    });
    let filtered = drawn_keys(&mut app, window);
    let shown_rows: Vec<&String> = filtered.iter().filter(|k| k.starts_with("run-row-")).collect();
    assert_eq!(shown_rows, vec![&format!("run-row-{first_id}")], "excludes non-matching runs");
    assert!(shown_rows.len() < total_rows);
    assert!(filtered.iter().any(|k| k == "runs-search-clear"), "{filtered:?}");

    // Navigating away and back to Runs clears status, arch, and search filters together.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Overview, w, cx);
        this.navigate(Route::Runs, w, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.run_filter, RunFilter::All);
        assert_eq!(this.run_arch_filter, None);
        assert!(this.run_search.read(cx).text().is_empty());
    });

    // Zero matches show Clear filters; activating it clears all filters and focuses run_search.
    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.run_filter = RunFilter::Failed;
        this.run_arch_filter = Some(Arch::X);
        this.run_search.update(cx, |i, cx| i.set_text("no-such-run-xyz", cx));
        cx.notify();
    });
    let empty = drawn_keys(&mut app, window);
    assert!(empty.iter().any(|k| k == "runs-show-all"), "{empty:?}");
    assert!(!empty.iter().any(|k| k.starts_with("run-row-")), "{empty:?}");
    drive(&mut app, window, |this, w, cx| this.focus_key("runs-show-all", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.run_filter, RunFilter::All);
        assert_eq!(this.run_arch_filter, None);
        assert!(this.run_search.read(cx).text().is_empty());
        assert!(this.run_search.read(cx).focus_handle(cx).is_focused(w), "focus returned to run_search");
    });
}

#[test]
fn models_list_filters_by_arch_and_search_and_resets_on_clear_and_backend_switch() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Models, w, cx));
    let all_keys = drawn_keys(&mut app, window);
    assert!(all_keys.iter().any(|k| k == "models-arch-filter"), "{all_keys:?}");
    let total_rows = all_keys.iter().filter(|k| k.starts_with("model-row-")).count();
    assert!(total_rows >= 2, "demo has multiple models: {total_rows}");

    let (first_id, first_arch, first_name) = drive(&mut app, window, |this, _, _| {
        let m = this.store.models.ready().unwrap().first().unwrap();
        (m.id.clone(), m.arch, m.name.clone())
    });
    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.models_arch_filter = Some(first_arch);
        this.model_search.update(cx, |i, cx| i.set_text(first_name.clone(), cx));
        cx.notify();
    });
    let filtered = drawn_keys(&mut app, window);
    let shown_rows: Vec<&String> = filtered.iter().filter(|k| k.starts_with("model-row-")).collect();
    assert_eq!(shown_rows, vec![&format!("model-row-{first_id}")], "excludes non-matching models");
    assert!(shown_rows.len() < total_rows);

    // Pure filter helpers match by dataset, source run, id, class, and status.
    drive(&mut app, window, |this, _, _| {
        let runs = this.store.runs.ready().unwrap();
        let ds_name = this.store.run_dataset_name(&runs[0]);
        let by_ds = filter_runs(runs, RunFilter::All, None, &ds_name, |r| this.store.run_dataset_name(r));
        assert!(!by_ds.is_empty());
        let models = this.store.models.ready().unwrap();
        let by_id = filter_models(&this.store, models, None, &models[0].id);
        assert_eq!(by_id.len(), 1);
    });

    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.models_arch_filter = Some(Arch::X);
        this.model_search.update(cx, |i, cx| i.set_text("no-such-model-xyz", cx));
        cx.notify();
    });
    let empty = drawn_keys(&mut app, window);
    assert!(empty.iter().any(|k| k == "models-filter-clear"), "{empty:?}");
    assert!(!empty.iter().any(|k| k.starts_with("model-row-")), "{empty:?}");
    drive(&mut app, window, |this, w, cx| this.focus_key("models-filter-clear", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.models_arch_filter, None);
        assert!(this.model_search.read(cx).text().is_empty());
        assert!(this.model_search.read(cx).focus_handle(cx).is_focused(w), "focus returned to model_search");
    });

    // Switching backends resets every filter, search box, and expanded class state.
    drive(&mut app, window, |this, _, cx| {
        this.run_arch_filter = Some(Arch::S);
        this.run_search.update(cx, |i, cx| i.set_text("abc", cx));
        this.models_arch_filter = Some(Arch::M);
        this.model_search.update(cx, |i, cx| i.set_text("xyz", cx));
        this.model_classes_expanded.insert("m_1".into());
        this.set_backend(Arc::new(MockBackend::new(4200, true)), ServerMode::Mock, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.run_arch_filter, None);
        assert!(this.run_search.read(cx).text().is_empty());
        assert_eq!(this.models_arch_filter, None);
        assert!(this.model_search.read(cx).text().is_empty());
        assert!(this.model_classes_expanded.is_empty());
    });
}

#[test]
fn completed_and_stopped_runs_offer_clone_config_preserving_custom_batch_and_imgsz() {
    let (mut app, window) = open();
    for want_status in [RunStatus::Completed, RunStatus::Stopped] {
        let run = drive(&mut app, window, |this, _, _| {
            let r = this
                .store
                .runs
                .ready_mut()
                .unwrap()
                .iter_mut()
                .find(|r| r.status == want_status)
                .expect("demo has run with status");
            // Non-preset API-valid batch and imgsz must survive Clone config verbatim.
            r.config.batch = 12;
            r.config.imgsz = 512;
            r.clone()
        });
        drive(&mut app, window, |this, w, cx| {
            this.new_run = None;
            this.new_run_draft = None;
            this.focus.borrow_mut().clear();
            this.navigate(Route::Run(run.id.clone()), w, cx);
        });
        let keys = drawn_keys(&mut app, window);
        assert!(keys.iter().any(|k| k == "run-clone-cfg"), "Clone config on {want_status:?}: {keys:?}");
        drive(&mut app, window, |this, w, cx| this.focus_key("run-clone-cfg", w, cx));
        app.frame(window.into());
        press_enter(&mut app, window);
        drive(&mut app, window, |this, _, cx| {
            let cfg = this.form_config(cx).expect("Clone config builds a valid TrainConfig");
            assert_eq!(cfg.dataset_id, run.config.dataset_id);
            assert_eq!(cfg.arch, run.config.arch);
            assert_eq!(cfg.epochs, run.config.epochs);
            assert_eq!(cfg.batch, 12, "preserves non-preset batch on {want_status:?}");
            assert_eq!(cfg.imgsz, 512, "preserves non-preset imgsz on {want_status:?}");
        });
    }
}

#[test]
fn model_details_expands_all_classes_and_copies_class_list() {
    let (mut app, window) = open();
    // <= 10 classes: Copy is shown, no Show all toggle.
    let small_id = drive(&mut app, window, |this, _, _| {
        let m = this.store.models.ready_mut().unwrap().first_mut().unwrap();
        m.classes = vec!["circle".into(), "square".into(), "triangle".into()];
        m.id.clone()
    });
    drive(&mut app, window, |this, w, cx| {
        this.focus.borrow_mut().clear();
        this.navigate(Route::Model(small_id.clone()), w, cx);
    });
    let small_keys = drawn_keys(&mut app, window);
    assert!(!small_keys.iter().any(|k| k == "model-classes-toggle"), "no toggle for <= 10 classes");
    assert!(small_keys.iter().any(|k| k == "model-classes-copy"), "copy offered for <= 10 classes");

    // > 10 classes: Show all toggle expands/collapses and Copy writes exact class list to clipboard.
    let expected_classes: Vec<String> = (0..15).map(|i| format!("class_{i}")).collect();
    let expected_clip = expected_classes.join(", ");
    drive(&mut app, window, |this, _, _| {
        let m = this.store.models.ready_mut().unwrap().first_mut().unwrap();
        m.classes = expected_classes.clone();
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "model-classes-toggle"), "expand toggle shown: {keys:?}");
    assert!(keys.iter().any(|k| k == "model-classes-copy"), "copy action shown: {keys:?}");

    drive(&mut app, window, |this, w, cx| this.focus_key("model-classes-toggle", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.model_classes_expanded.contains(&small_id), "classes expanded");
    });

    press_enter(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.model_classes_expanded.contains(&small_id), "classes collapsed");
    });

    drive(&mut app, window, |this, w, cx| this.focus_key("model-classes-copy", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert!(this.toasts.iter().any(|t| t.title.as_ref() == "Copied 15 classes"));
        let clip = cx.read_from_clipboard().and_then(|c| c.text());
        assert_eq!(clip.as_deref(), Some(expected_clip.as_str()), "clipboard holds full class list");
    });
}
