//! Regression tests for Sprint 8 desktop & `folo-ui` hardening:
//! 1. Inspector focus recovery when stepping unmounts a per-image box/pred focus target, plus
//!    preserving newly stepped images across background page arrivals and sample rapid stepping.
//! 2. `submit_new_run` routes through `navigate`/`after_navigate`, resetting prior run filters,
//!    scroll offsets, and focus.
//! 3. `forget_run` tears down active SSE subscriptions and bumps `sub_attempt`; `forget_dataset`
//!    evicts cached issues, decoded pixels, filter caches, and restores inspector focus.
//! 4. Non-finite `imgs_per_s` samples do not poison `RunData::recent_ips`, Run detail, or Overview.
//! 5. Dataset Statistics `ds-empty-tile` and Images `ds-empty-chip` preserve keyboard focus.
//! 6. Playground `Open image` button is disabled while models are loading or failed.
//! 7. Modified arrow/Home/End keys (`Ctrl+...`) do not hijack segmented controls, tabs, selects, or
//!    chart cursors, and narrow charts clamp `future`/marker/`pow_caption`/histogram labels cleanly.

use std::sync::Arc;
use std::sync::mpsc::channel;

use folo_ui::{
    AppContext, HeadlessApp, HeadlessOptions, KeyDownEvent, Keystroke, PlatformInput, WindowBounds,
    WindowOptions, point, px, size,
};

use super::super::select_driving::press;
use super::super::tests::{drive, open};
use super::super::{DsTab, LogFilter, Route, SampleView};
use super::settle;
use crate::api::{
    ApiError, BBox, ImageInfo, IssueCounts, Loss, RunMetrics, RunSamples, Sample, Step, Subscription,
};
use crate::store::{ImgKind, Load, RunData};
use crate::theme::{Mode, Theme};

#[test]
fn inspector_shortcuts_survive_unmounted_box_focus_and_background_page_arrival() {
    let (mut app, window) = open();
    let ds_id = drive(&mut app, window, |this, w, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].clone();
        this.navigate(Route::Dataset(ds.id.clone()), w, cx);
        ds.id
    });
    settle(&mut app, window);

    // Ensure image 0 has 3 boxes and image 1 has 0 boxes so stepping from focused `insp-box-2`
    // unmounts that focus handle.
    let (img0_id, img1_id) = drive(&mut app, window, |this, w, cx| {
        let imgs = this.store.images.get_mut(&ds_id).and_then(|l| l.ready_mut()).unwrap();
        assert!(imgs.len() >= 2);
        let sample_box =
            imgs[0].boxes.first().copied().unwrap_or(BBox { class: 0, x1: 10., y1: 10., x2: 50., y2: 50. });
        while imgs[0].boxes.len() < 3 {
            imgs[0].boxes.push(sample_box);
        }
        imgs[1].boxes.clear();
        let ids = (imgs[0].id.clone(), imgs[1].id.clone());
        this.open_inspector(ds_id.clone(), 0, w, cx);
        ids
    });
    settle(&mut app, window);

    // Focus `insp-box-2`, then step right to image 1 (which has 0 boxes).
    drive(&mut app, window, |this, w, cx| {
        this.kb = true;
        w.focus(&this.fh("insp-box-2", cx));
        cx.notify();
    });
    settle(&mut app, window);

    press(&mut app, window, "right");
    settle(&mut app, window);
    drive(&mut app, window, |this, w, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert_eq!(insp.index, 1, "stepped from focused box row to image 1");
        assert!(
            this.inspector_focus.is_focused(w),
            "unmounted insp-box-2 must restore focus to inspector_focus"
        );
    });

    // Inspector shortcuts (`b` toggle boxes, `left` step back) must keep working immediately.
    press(&mut app, window, "b");
    assert!(
        !drive(&mut app, window, |this, _, _| this.inspector.as_ref().unwrap().boxes),
        "B shortcut still works after unmounted box focus recovery"
    );

    // Background page arrival between `step_inspector` and next render must anchor on the newly
    // stepped image rather than snapping back to `img0_id`.
    drive(&mut app, window, |this, _, cx| {
        this.step_inspector(-1, cx); // back to index 0 (`img0_id`)
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.inspector.as_ref().unwrap().index, 0);
        // Step to image 1 (`img1_id`), then simulate a new image page appending before render.
        this.step_inspector(1, cx);
        let imgs = this.store.images.get_mut(&ds_id).and_then(|l| l.ready_mut()).unwrap();
        let mut extra: ImageInfo = imgs[0].clone();
        extra.id = "img_appended_page".into();
        extra.file = "appended.jpg".into();
        imgs.push(extra);
    });
    settle(&mut app, window);
    let shown_id = drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().unwrap();
        this.filtered_image(&ds_id, insp.index).unwrap().id.clone()
    });
    assert_eq!(
        shown_id, img1_id,
        "background page arrival must stay on stepped image {img1_id}, not snap back to {img0_id}"
    );
}

#[test]
fn inspector_waiting_state_keeps_focus_and_steps_back_from_unloaded_sample() {
    let (mut app, window) = open();
    let (ds_id, run_id, img0_id) = drive(&mut app, window, |this, w, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].clone();
        let run = this.store.runs.ready().unwrap()[0].clone();
        this.navigate(Route::Dataset(ds.id.clone()), w, cx);
        (ds.id, run.id, String::new())
    });
    let _ = img0_id;
    settle(&mut app, window);

    // Give image 0 boxes, attach a 2-sample set whose 2nd image is on an unloaded page, and hold
    // that page pending via `images_more`.
    let img0_id = drive(&mut app, window, |this, w, cx| {
        let imgs = this.store.images.get_mut(&ds_id).and_then(|l| l.ready_mut()).unwrap();
        let sample_box =
            imgs[0].boxes.first().copied().unwrap_or(BBox { class: 0, x1: 10., y1: 10., x2: 50., y2: 50. });
        while imgs[0].boxes.len() < 3 {
            imgs[0].boxes.push(sample_box);
        }
        let img0_id = imgs[0].id.clone();
        let make_sample = |id: String| Sample {
            dataset_id: ds_id.clone(),
            image_id: id,
            width: 100,
            height: 100,
            truth: vec![],
            pred: vec![],
        };
        this.store.samples.insert(
            run_id.clone(),
            Load::Ready(RunSamples {
                epoch: 1,
                samples: vec![make_sample(img0_id.clone()), make_sample("img_unloaded_page_2".into())],
            }),
        );
        this.images_more.insert(ds_id.clone(), 999);
        this.open_inspector(ds_id.clone(), 0, w, cx);
        if let Some(i) = this.inspector.as_mut() {
            i.sample = Some((run_id.clone(), SampleView::Both));
        }
        img0_id
    });
    settle(&mut app, window);

    // Focus a per-image box row, step right onto the unloaded sample, and let the waiting frame
    // render. Focus must recover to `inspector_focus`, and Left must step back immediately.
    drive(&mut app, window, |this, w, cx| {
        this.kb = true;
        w.focus(&this.fh("insp-box-2", cx));
        cx.notify();
    });
    settle(&mut app, window);

    press(&mut app, window, "right");
    settle(&mut app, window);
    drive(&mut app, window, |this, w, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert_eq!(insp.want.as_deref(), Some("img_unloaded_page_2"));
        assert!(
            this.inspector_focus.is_focused(w),
            "waiting state must recover focus to inspector_focus after box row unmounts"
        );
    });

    press(&mut app, window, "left");
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert_eq!(insp.want, None, "Left during waiting state stepped back to the loaded sample");
        assert_eq!(this.filtered_image(&ds_id, insp.index).unwrap().id, img0_id);
    });
}

#[test]
fn submit_new_run_navigates_cleanly_resetting_route_state_and_focus() {
    let (mut app, window) = open();
    settle(&mut app, window);

    // Open an existing run and dirty its page-scoped state (scroll, log filter/search, sample_epoch).
    let first_run = drive(&mut app, window, |this, w, cx| {
        let id = this.store.runs.ready().unwrap()[0].id.clone();
        this.navigate(Route::Run(id.clone()), w, cx);
        this.log_filter = LogFilter::Error;
        this.log_search.update(cx, |i, cx| i.set_text("OOM", cx));
        this.sample_epoch = Some(1);
        this.run_x_fit = Some(true);
        this.page_scroll.set_offset(point(px(0.), px(-200.)));
        id
    });
    settle(&mut app, window);

    // Open New run, submit, and settle until `route_pending` runs `navigate` + `after_navigate`.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(None, w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        this.submit_new_run(w, cx);
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, w, cx| {
        assert!(this.new_run.is_none(), "dialog closed after submit");
        match &this.route {
            Route::Run(id) => assert_ne!(id, &first_run, "navigated to the newly created run"),
            other => panic!("expected Route::Run, got {other:?}"),
        }
        assert_eq!(this.log_filter, LogFilter::All, "after_navigate reset log_filter");
        assert!(this.log_search.read(cx).text().is_empty(), "after_navigate cleared log_search");
        assert_eq!(this.sample_epoch, None, "after_navigate cleared sample_epoch");
        assert_eq!(this.run_x_fit, None, "after_navigate cleared run_x_fit");
        assert_eq!(this.page_scroll.offset().y, px(0.), "after_navigate reset page_scroll");
        assert!(this.root_focus.is_focused(w), "navigate restored focus to root_focus");
    });
}

#[test]
fn forget_run_and_forget_dataset_clean_up_streams_issues_and_cached_pixels() {
    let (mut app, window) = open();
    settle(&mut app, window);

    // 1. Removing a queued/active run drops its active subscription and bumps `sub_attempt`.
    let (run_id, cancel_flag, before_attempt) = drive(&mut app, window, |this, _, _| {
        let id = this.store.runs.ready().unwrap()[0].id.clone();
        let (_tx, rx) = channel();
        let (sub, cancel) = Subscription::new(rx);
        this.subscriptions.insert(id.clone(), sub);
        this.stream_lost.insert(id.clone());
        let attempt = this.sub_attempt.get(&id).copied().unwrap_or(0);
        (id, cancel, attempt)
    });
    drive(&mut app, window, |this, _, cx| {
        this.remove_queued(run_id.clone(), cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.subscriptions.contains_key(&run_id), "removed run drops its subscription");
        assert!(!this.stream_lost.contains(&run_id), "removed run drops stream_lost entry");
        assert!(
            this.sub_attempt.get(&run_id).copied().unwrap_or(0) > before_attempt,
            "removed run bumps sub_attempt so in-flight resubscribes are fenced off"
        );
    });
    assert!(
        cancel_flag.load(std::sync::atomic::Ordering::Relaxed),
        "dropping subscription cancels its SSE reader"
    );

    // 2. Removing a dataset evicts its issues, decoded pixels, filter cache, and closes its
    // inspector with `restore_pending = true`.
    let ds_id = drive(&mut app, window, |this, w, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].clone();
        this.navigate(Route::Dataset(ds.id.clone()), w, cx);
        ds.id
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        this.store.issues.insert(ds_id.clone(), Load::Ready(IssueCounts::default()));
        this.open_inspector(ds_id.clone(), 0, w, cx);
        let img = this.filtered_image(&ds_id, 0).unwrap().id.clone();
        let _ = this.image(&ds_id, &img, ImgKind::Thumb, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        this.confirm_delete = Some(super::super::DeleteTarget::Dataset(ds_id.clone()));
        this.delete_confirmed(cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.issues.contains_key(&ds_id), "forget_dataset evicted cached issues");
        assert!(!this.ds_reads.contains_key(&ds_id), "forget_dataset cleared ds_reads");
        assert!(this.inspector.is_none(), "forget_dataset closed open inspector");
        assert!(
            this.store.images_px.forget_dataset(&ds_id).is_empty(),
            "forget_dataset already evicted cached decoded textures"
        );
    });
}

#[test]
fn non_finite_throughput_samples_never_poison_recent_ips_or_overview_kpis() {
    let mk_step = |step: u64, ips: f32| Step {
        epoch: 1,
        step,
        lr: 0.01,
        loss: Loss { box_: 1.0, cls: 0.8, dfl: 0.5, total: 2.3 },
        imgs_per_s: ips,
        eta_s: 12.0,
        steps_per_epoch: 10,
        epochs: 5,
    };
    let rd = RunData::new(
        RunMetrics {
            steps: vec![
                mk_step(1, 100.0),
                mk_step(2, f32::NAN),
                mk_step(3, f32::INFINITY),
                mk_step(4, 140.0),
            ],
            epochs: vec![],
        },
        10,
    );
    assert_eq!(rd.recent_ips(10), vec![100.0, 140.0]);

    // Also render Overview & Run detail with a trailing NaN step to confirm neither panics or
    // breaks layout.
    let (mut app, window) = open();
    settle(&mut app, window);
    let live_id = drive(&mut app, window, |this, _, _| {
        let id = this.store.live_run().unwrap().id.clone();
        if let Some(Load::Ready(d)) = this.store.run_data.get_mut(&id) {
            let next = d.metrics.steps.last().map_or(1, |s| s.step + 1);
            d.push_step(mk_step(next, f32::NAN));
        }
        id
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(live_id), w, cx));
    settle(&mut app, window);
}

#[test]
fn dataset_empty_tile_and_chip_preserve_keyboard_focus() {
    let (mut app, window) = open();
    let ds_id = drive(&mut app, window, |this, w, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].clone();
        this.navigate(Route::Dataset(ds.id.clone()), w, cx);
        ds.id
    });
    settle(&mut app, window);

    // Ensure at least one image has no boxes so Statistics mounts `ds-empty-tile`.
    drive(&mut app, window, |this, _, cx| {
        if let Some(imgs) = this.store.images.get_mut(&ds_id).and_then(|l| l.ready_mut()) {
            imgs[0].boxes.clear();
        }
        *this.ds_filter_cache.borrow_mut() = None;
        this.ds_tab = DsTab::Statistics;
        this.kb = true;
        cx.notify();
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, w, cx| {
        w.focus(&this.fh("ds-empty-tile", cx));
        cx.notify();
    });
    settle(&mut app, window);

    // Pressing Enter on `ds-empty-tile` jumps to Images and focuses `ds-empty-chip`.
    press(&mut app, window, "enter");
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.ds_tab, DsTab::Images);
        assert!(this.empty_only(&ds_id));
        assert!(
            this.fh("ds-empty-chip", cx).is_focused(w),
            "jumping from Statistics empty tile must focus ds-empty-chip"
        );
    });

    // Pressing Enter on `ds-empty-chip` clears the filter and moves focus to `ds-class`.
    press(&mut app, window, "enter");
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(!this.empty_only(&ds_id));
        assert!(
            this.fh("ds-class", cx).is_focused(w),
            "clearing ds-empty-chip must move focus to ds-class instead of stranding it"
        );
    });
}

#[test]
fn playground_open_button_respects_open_image_blocked_while_loading_or_failed() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.store.models = Load::Loading;
        cx.notify();
    });
    settle(&mut app, window);
    // `button_disabled` mounts `{key}` without registering a Tab stop on `fh("play-open")`,
    // whereas enabled `button` tracks `fh("play-open")` as mounted.
    drive(&mut app, window, |this, w, cx| {
        assert!(this.open_image_blocked().is_some());
        let fh = this.fh("play-open", cx);
        assert!(!w.is_focus_mounted(&fh), "play-open must be disabled while models are Loading");
    });

    drive(&mut app, window, |this, _, cx| {
        this.store.models = Load::Failed(ApiError::new("server_unreachable", "offline"));
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(this.open_image_blocked().is_some());
        let fh = this.fh("play-open", cx);
        assert!(!w.is_focus_mounted(&fh), "play-open must be disabled when models Failed");
    });
}

#[test]
fn modified_arrows_do_not_activate_segmented_tabs_or_charts_and_narrow_charts_render_cleanly() {
    let (mut app, window) = open();
    let send = |app: &mut HeadlessApp, spec: &str| {
        let ks = Keystroke::parse(spec).unwrap();
        app.simulate_input(
            window.into(),
            PlatformInput::KeyDown(KeyDownEvent { keystroke: ks, is_held: false }),
        );
        settle(app, window);
    };

    // 1. Underline tabs: Ctrl+Right must not switch tabs.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_3c90".into()), w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("bench-tab-0", cx)));
    settle(&mut app, window);
    send(&mut app, "ctrl-right");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.bench.tab),
        crate::bench::BenchTab::Leaderboard,
        "Ctrl+Right must not switch underline tabs"
    );

    // 2. Segmented control & 3. Chart cursor on a Run page: Ctrl+arrows must not trigger either.
    let run_id = drive(&mut app, window, |this, w, cx| {
        let id = this.store.runs.ready().unwrap()[0].id.clone();
        this.navigate(Route::Run(id.clone()), w, cx);
        id
    });
    let _ = run_id;
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.focus_key("run-loss-smooth", w, cx));
    settle(&mut app, window);
    let smooth_before = drive(&mut app, window, |this, _, _| this.run_chart_opts.loss_smooth);
    send(&mut app, "ctrl-right");
    send(&mut app, "ctrl-end");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.run_chart_opts.loss_smooth),
        smooth_before,
        "Ctrl+Right/End must not toggle segmented control"
    );

    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("run-charts", cx)));
    settle(&mut app, window);
    send(&mut app, "ctrl-left");
    send(&mut app, "ctrl-right");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.chart_hover),
        None,
        "Ctrl+Left/Right must not activate chart cursor"
    );

    // 4. Select menu: Ctrl+Down/End must not move menu highlight.
    let ds_id = drive(&mut app, window, |this, w, cx| {
        let id = this.store.datasets.ready().unwrap()[0].id.clone();
        this.navigate(Route::Dataset(id.clone()), w, cx);
        id
    });
    let _ = ds_id;
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-class", w, cx));
    settle(&mut app, window);
    press(&mut app, window, "enter");
    settle(&mut app, window);
    let (open_before, hl_before) = drive(&mut app, window, |this, _, _| (this.menu_open, this.menu_hl));
    assert_eq!(open_before, Some("ds-class"), "ds-class select opened");
    send(&mut app, "ctrl-down");
    send(&mut app, "ctrl-end");
    assert_eq!(
        drive(&mut app, window, |this, _, _| (this.menu_open, this.menu_hl)),
        (open_before, hl_before),
        "Ctrl+Down/End must not move select highlight"
    );

    // 5. Narrow chart rendering with positive plot height: marker, future, pow_caption,
    // axis_titles, cursor chip, and histogram labels must all paint and stay inside [0..140].
    let mut narrow =
        HeadlessApp::new(HeadlessOptions { size: size(px(140.), px(280.)), ..Default::default() });
    let w = narrow.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(folo_ui::Bounds::new(
                    point(px(0.), px(0.)),
                    size(px(140.), px(280.)),
                ))),
                ..Default::default()
            },
            |_, cx| cx.new(|_| ChartClampView),
        )
        .unwrap()
    });
    narrow.frame(w.into());
    let scene = narrow.scene(w.into());
    let mono: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            folo_ui::Primitive::MonoSprite(s) => Some(s),
            _ => None,
        })
        .collect();
    let chart_sprites: Vec<_> = mono.iter().filter(|s| s.bounds.origin.y < px(160.0).scale(1.0)).collect();
    assert!(!chart_sprites.is_empty(), "narrow line chart must have positive plot_h and paint labels");
    // With `readout = false`, the cursor chip quad (`height == 16`) must be painted and stay inside
    // `[0..140]`.
    let chip_quads: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            folo_ui::Primitive::Quad(q)
                if q.bounds.size.height == px(16.0).scale(1.0)
                    && q.bounds.origin.y < px(160.0).scale(1.0) =>
            {
                Some(q)
            }
            _ => None,
        })
        .collect();
    assert_eq!(chip_quads.len(), 1, "narrow line chart with readout=false must paint its cursor chip");
    assert!(
        chip_quads[0].bounds.origin.x >= px(-1.0).scale(1.0)
            && chip_quads[0].bounds.right() <= px(141.0).scale(1.0),
        "cursor chip escaped 140 px width: {:?}",
        chip_quads[0].bounds
    );
    for s in &mono {
        assert!(
            s.bounds.origin.x >= px(-1.0).scale(1.0) && s.bounds.right() <= px(141.0).scale(1.0),
            "narrow chart glyph escaped 140 px width: {:?}",
            s.bounds
        );
    }
}

struct ChartClampView;
impl folo_ui::Render for ChartClampView {
    fn render(
        &mut self,
        _: &mut folo_ui::Window,
        cx: &mut folo_ui::Context<Self>,
    ) -> impl folo_ui::IntoElement {
        use folo_ui::{ParentElement, Styled, div};
        let t = crate::theme::theme(cx);
        let mut lc = crate::charts::LineChart::new(
            "narrow-lc",
            vec![crate::charts::Series::new(
                "lr",
                t.chart_series[0],
                Arc::new(vec![(0.0, 0.0001), (1.0, 0.0005), (2.0, 0.0003)]),
            )],
        );
        lc.x_range = Some((0.0, 10.0));
        lc.height = 160.;
        lc.readout = false;
        lc.x_name = "epoch";
        lc.pow_caption = true;
        lc.axis_titles = Some(("Training epoch number", "Rate"));
        lc.future = Some((2.0, "ETA 2h 15m remaining".into()));
        lc.markers = vec![crate::charts::Marker {
            x: 9.5,
            label: "very long close mosaic marker".into(),
            color: t.text_tertiary,
            lane: Some(0),
        }];
        lc.hover = Some(0.15);
        let hist = crate::charts::histogram(
            &t,
            "narrow-hist",
            Arc::new(vec![10, 0, 25, 5, 18, 0, 12, 30]),
            Arc::new((0..8).map(|i| format!("bin-{i}")).collect()),
            t.chart_series[1],
            90.,
            Some(0.3),
            None,
        );
        div().size_full().flex().flex_col().child(crate::charts::line_chart(&t, lc)).child(hist)
    }
}

#[test]
fn rtl_text_input_partial_selection_paints_positive_width_highlight() {
    use folo_ui::{Entity, Focusable, IntoElement, ParentElement, Styled, div};

    struct RtlInputView(Entity<crate::input::TextInput>);
    impl folo_ui::Render for RtlInputView {
        fn render(&mut self, _: &mut folo_ui::Window, cx: &mut folo_ui::Context<Self>) -> impl IntoElement {
            let t = crate::theme::theme(cx);
            div().size_full().bg(t.bg_panel).p(px(8.)).child(self.0.clone())
        }
    }

    let mut app = HeadlessApp::new(HeadlessOptions { size: size(px(240.), px(60.)), ..Default::default() });
    let (window, input) = app.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        crate::input::bind_keys(cx);
        let input = cx.new(|cx| crate::input::TextInput::new("", cx));
        let input_for_view = input.clone();
        let w = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(folo_ui::Bounds::new(
                        point(px(0.), px(0.)),
                        size(px(240.), px(60.)),
                    ))),
                    ..Default::default()
                },
                |_, cx| cx.new(|_| RtlInputView(input_for_view)),
            )
            .unwrap();
        (w, input)
    });
    app.update(|cx| {
        input.update(cx, |inp, cx| {
            inp.set_text("שלום", cx);
        });
        cx.update_window(window.into(), |_, w, cx| {
            let fh = input.read(cx).focus_handle(cx);
            w.focus(&fh);
        })
        .unwrap();
    });
    app.frame(window.into());
    for spec in ["home", "shift-right", "shift-right"] {
        app.simulate_input(
            window.into(),
            PlatformInput::KeyDown(KeyDownEvent {
                keystroke: Keystroke::parse(spec).unwrap(),
                is_held: false,
            }),
        );
        app.frame(window.into());
    }
    let scene = app.scene(window.into());
    let accent_sel = crate::theme::alpha(Theme::new(Mode::Dark).accent_fill(), 0.28);
    let sel_quads: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            folo_ui::Primitive::Quad(q) if q.background.solid == accent_sel => Some(q),
            _ => None,
        })
        .collect();
    assert_eq!(sel_quads.len(), 1, "focused RTL partial selection must paint one highlight quad");
    assert!(
        sel_quads[0].bounds.size.width > px(0.).scale(1.0),
        "RTL partial selection highlight must have positive width, got {:?}",
        sel_quads[0].bounds
    );

    // Select All on mixed LTR + RTL ("hello שלום") must highlight the full width including the
    // Hebrew run, not stop at the left edge of the Hebrew run.
    app.update(|cx| {
        input.update(cx, |inp, cx| inp.set_text("hello שלום", cx));
    });
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent {
            keystroke: Keystroke::parse("ctrl-a").unwrap(),
            is_held: false,
        }),
    );
    app.frame(window.into());
    let scene = app.scene(window.into());
    let rightmost_glyph = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            folo_ui::Primitive::MonoSprite(s) => Some(s.bounds.right()),
            _ => None,
        })
        .fold(px(0.).scale(1.0), |a, b| a.max(b));
    let sel_right = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            folo_ui::Primitive::Quad(q) if q.background.solid == accent_sel => Some(q.bounds.right()),
            _ => None,
        })
        .fold(px(0.).scale(1.0), |a, b| a.max(b));
    assert!(
        sel_right + px(2.).scale(1.0) >= rightmost_glyph,
        "Select All on mixed text must cover rightmost Hebrew glyph: sel_right={sel_right:?}, rightmost_glyph={rightmost_glyph:?}"
    );
}
