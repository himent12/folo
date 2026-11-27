//! Deterministic tests for side-by-side Multi-Run Comparison (`Route::RunCompare`).

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, WindowBounds, WindowHandle, WindowOptions, point, px,
    size,
};

use crate::api::mock::MockBackend;
use crate::api::{ApiError, Backend};
use crate::app::tests::drive;
use crate::app::{FoloApp, Route, Section};
use crate::screens::runs::compare_data::{
    CompareMapMetric, CompareXAxis, RUN_COMPARE_MAX, default_compare_runs, hyperparam_rows, same_dataset,
};
use crate::store::Load;
use crate::theme::{Mode, Theme};

fn open_with(api: Arc<dyn Backend>, w: f32, h: f32, mode: Mode) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let mut app = HeadlessApp::new(HeadlessOptions { size: size(px(w), px(h)), ..Default::default() });
    let window = app.update(|cx| {
        cx.set_global(Theme::new(mode));
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(w), px(h)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |win, cx| cx.new(|cx| FoloApp::new(api, true, win, cx)))
            .expect("open test window")
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    (app, window)
}

fn settle(app: &mut HeadlessApp, w: WindowHandle<FoloApp>) {
    for _ in 0..4 {
        app.frame(w.into());
        app.run_until_idle();
    }
}

#[test]
fn route_parsing_pinning_baseline_add_remove_and_delete_pruning() {
    assert_eq!(Route::parse("runs-compare"), Some(Route::RunCompare));
    assert_eq!(Route::parse("run-compare"), Some(Route::RunCompare));
    assert_eq!(Route::RunCompare.section(), Section::Runs);

    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);

    // Navigate to Runs and verify Compare runs & Pin to compare controls appear.
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Runs, w, cx));
    settle(&mut app, window);

    let all_runs = drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "runs-compare"), "Compare runs button visible: {keys:?}");
        assert!(keys.iter().any(|k| k == "runs-select-cmp"), "Pin to compare button visible: {keys:?}");
        this.store.runs.ready().expect("runs loaded").clone()
    });
    assert!(all_runs.len() >= 3, "mock provides >= 3 runs");

    // Pin runs in select mode: toggle on/off and enforce RUN_COMPARE_MAX cap.
    drive(&mut app, window, |this, _, cx| {
        this.run_compare.select_mode = true;
        this.toggle_run_compare_pin(all_runs[0].id.clone(), cx);
        this.toggle_run_compare_pin(all_runs[1].id.clone(), cx);
        assert_eq!(this.run_compare.ids, vec![all_runs[0].id.clone(), all_runs[1].id.clone()]);
        // Toggling off removes it.
        this.toggle_run_compare_pin(all_runs[0].id.clone(), cx);
        assert_eq!(this.run_compare.ids, vec![all_runs[1].id.clone()]);
        // Pin back up to RUN_COMPARE_MAX.
        for r in all_runs.iter().take(RUN_COMPARE_MAX + 2) {
            if !this.run_compare.ids.contains(&r.id) {
                this.toggle_run_compare_pin(r.id.clone(), cx);
            }
        }
        assert!(this.run_compare.ids.len() <= RUN_COMPARE_MAX);
    });
    settle(&mut app, window);

    // Open RunCompare from pinned selection; verify parent_route is Runs and metrics load.
    drive(&mut app, window, |this, w, cx| {
        this.open_run_compare(None, w, cx);
        assert_eq!(this.route, Route::RunCompare);
        assert_eq!(this.parent_route(), Some(Route::Runs));
    });
    settle(&mut app, window);

    // Promote second run to baseline, remove a run to expose Add run picker, and add it back.
    let (second_id, removed_id) = drive(&mut app, window, |this, _, cx| {
        let ids = this.run_compare.ids.clone();
        assert!(ids.len() >= 2);
        let second = ids[1].clone();
        this.run_compare_make_baseline(&second, cx);
        assert_eq!(this.run_compare.ids[0], second, "promoted run moved to index 0 (baseline)");
        let last = this.run_compare.ids.last().unwrap().clone();
        this.run_compare_remove(&last, cx);
        assert_eq!(this.run_compare.ids.len(), ids.len() - 1);
        (second, last)
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, _, cx| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(
            keys.iter().any(|k| k == "run-cmp-add"),
            "Add run selector appears when below RUN_COMPARE_MAX: {keys:?}"
        );
        this.run_compare_add(removed_id.clone(), cx);
        assert!(this.run_compare.ids.contains(&removed_id));
        assert_eq!(this.run_compare.ids[0], second_id);
    });
    settle(&mut app, window);

    // Command Palette exposes "Compare runs" action when >= 2 runs exist.
    drive(&mut app, window, |this, w, cx| this.open_palette(w, cx));
    settle(&mut app, window);
    let shown = drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().shown.clone());
    assert!(shown.iter().any(|(g, t)| *g == "Actions" && t == "Compare runs"), "{shown:?}");
    drive(&mut app, window, |this, w, cx| this.close_palette(w, cx));

    // Opening Compare from a specific Run detail page puts that run first as baseline.
    let target_run = all_runs.last().unwrap().id.clone();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Run(target_run.clone()), w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "run-compare"), "Run detail shows Compare button: {keys:?}");
        this.open_run_compare(Some(&target_run), w, cx);
        assert_eq!(this.route, Route::RunCompare);
        assert_eq!(this.run_compare.ids.first().map(String::as_str), Some(target_run.as_str()));
    });
    settle(&mut app, window);

    // Deleting a compared run prunes it cleanly from run_compare.ids.
    let deletable = drive(&mut app, window, |this, _, _| {
        this.store
            .runs
            .ready()
            .unwrap()
            .iter()
            .find(|r| !r.status.is_active() && this.run_compare.ids.contains(&r.id))
            .map(|r| r.id.clone())
    });
    if let Some(del_id) = deletable {
        drive(&mut app, window, |this, w, cx| {
            this.ask_delete(crate::app::DeleteTarget::Run(del_id.clone()), w, cx);
            this.delete_confirmed(cx);
        });
        settle(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert!(!this.run_compare.ids.contains(&del_id), "deleted run pruned from compare selection");
        });
    }
}

#[test]
fn chart_controls_hyperparam_diff_partial_failure_and_viewports() {
    let api = Arc::new(MockBackend::new(4200, true));
    let runs = api.runs().unwrap();
    let defaults = default_compare_runs(&runs, None);
    assert!(defaults.len() >= 2 && defaults.len() <= RUN_COMPARE_MAX);
    let selected: Vec<_> =
        defaults.iter().filter_map(|id| runs.iter().find(|r| &r.id == id).cloned()).collect();
    let rows = hyperparam_rows(&selected, |r| r.config.dataset_id.clone(), |m| m.to_string());
    assert!(!rows.is_empty());
    assert!(rows.iter().any(|r| r.label == "Architecture"));
    assert!(rows.iter().any(|r| r.label == "Learning rate"));
    assert!(rows.iter().any(|r| r.label == "Batch"));
    let _ = same_dataset(&selected);

    // Render RunCompare across toggles, partial metric load failure, and both target viewports.
    for (w, h, mode) in [(1280.0_f32, 800.0_f32, Mode::Dark), (1120.0, 780.0, Mode::Light)] {
        let (mut app, win) = open_with(Arc::new(MockBackend::new(4200, true)), w, h, mode);
        drive(&mut app, win, |this, w, cx| this.open_run_compare(None, w, cx));
        settle(&mut app, win);

        // Toggle all chart modes, diff filter, and isolate pin.
        drive(&mut app, win, |this, _, cx| {
            this.run_compare.map_metric = CompareMapMetric::Map50;
            this.run_compare.x_axis = CompareXAxis::Progress;
            this.run_compare.loss_part = 1;
            this.run_compare.loss_smooth = false;
            this.run_compare.loss_log_y = true;
            this.run_compare.show_identical = true;
            this.run_compare.pinned_run = this.run_compare.ids.first().cloned();
            cx.notify();
        });
        settle(&mut app, win);

        // Inject a partial metric load failure for one compared run -> Retry button appears without
        // hiding the other runs' curves.
        let failed_id = drive(&mut app, win, |this, _, cx| {
            let fid = this.run_compare.ids[1].clone();
            this.store
                .run_data
                .insert(fid.clone(), Load::Failed(ApiError::new("not_found", "Metrics file unreadable")));
            cx.notify();
            fid
        });
        settle(&mut app, win);
        drive(&mut app, win, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            assert!(
                keys.iter().any(|k| k == &format!("run-cmp-mretry-{failed_id}")),
                "failed run gets per-run Retry button: {keys:?}"
            );
        });
    }
}
