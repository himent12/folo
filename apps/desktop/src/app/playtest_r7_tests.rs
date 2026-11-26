//! Regression tests for Sol Playtest Round 7 fixes:
//! 1. Benchmarks surfaces `/api/system` failures (unreachable server, auth/address errors) via
//!    the standard error panel (with Retry + Connection settings) instead of hanging on "Loading…"
//!    or showing "not found", and recovers automatically once capability discovery succeeds.
//! 2. Open `Select` dropdowns dismiss when their enclosing scroll container scrolls (wheel,
//!    scrollbar, or programmatic scroll) while still allowing wheel scrolling inside the open
//!    dropdown menu itself.
//! 3. Model Export menu dismisses on page scroll / scrollbar interaction and never scrolls over the
//!    title bar.
//! 4. Failed-run `Validation mAP` header (`So far / All N` + zoomed-axis caption) stays strictly
//!    inside the mAP card at narrow and high-DPI viewports (960×640 @1.5×, 1120×800 @2×) without
//!    overlapping `Training loss`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, MouseButton, MouseDownEvent, PlatformInput, Primitive,
    ScrollDelta, ScrollWheelEvent, TouchPhase, WindowBounds, WindowHandle, WindowOptions, point, px, size,
};

use super::super::tests::drive;
use super::super::{FoloApp, Route};
use super::{settle, sp};
use crate::api::mock::MockBackend;
use crate::api::{
    ApiError, ApiResult, Backend, Bench, BenchCatalog, BenchEntry, BenchLoop, Dataset, ExportResult, Health,
    ImageData, ImagePage, ImageQuery, Model, ModelEval, PredictRequest, PredictResponse, Run, RunLogs,
    RunMetrics, RunSamples, RunStatus, Subscription, SystemResponse, TrainConfig,
};
use crate::theme::{Mode, Theme, theme};

/// Wrapper over [`MockBackend`] that can fail `/api/system` and `/api/health` on demand to
/// reproduce an unreachable server before benchmark capabilities are discovered.
struct ToggleSystemBackend {
    inner: MockBackend,
    fail_system: Arc<AtomicBool>,
}

impl Backend for ToggleSystemBackend {
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn now(&self) -> i64 {
        self.inner.now()
    }
    fn health(&self) -> ApiResult<Health> {
        if self.fail_system.load(Ordering::Relaxed) {
            return Err(ApiError::new("server_unreachable", "Could not reach folo serve"));
        }
        self.inner.health()
    }
    fn system(&self) -> ApiResult<SystemResponse> {
        if self.fail_system.load(Ordering::Relaxed) {
            return Err(ApiError::new("server_unreachable", "Could not reach folo serve"));
        }
        self.inner.system()
    }
    fn datasets(&self) -> ApiResult<Vec<Dataset>> {
        self.inner.datasets()
    }
    fn dataset(&self, id: &str) -> ApiResult<Dataset> {
        self.inner.dataset(id)
    }
    fn images(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage> {
        self.inner.images(id, q)
    }
    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        self.inner.image(dataset_id, image_id, thumb)
    }
    fn models(&self) -> ApiResult<Vec<Model>> {
        self.inner.models()
    }
    fn model_eval(&self, id: &str) -> ApiResult<ModelEval> {
        self.inner.model_eval(id)
    }
    fn export_model(&self, id: &str, format: &str) -> ApiResult<ExportResult> {
        self.inner.export_model(id, format)
    }
    fn runs(&self) -> ApiResult<Vec<Run>> {
        self.inner.runs()
    }
    fn run(&self, id: &str) -> ApiResult<Run> {
        self.inner.run(id)
    }
    fn create_run(&self, config: TrainConfig) -> ApiResult<Run> {
        self.inner.create_run(config)
    }
    fn stop_run(&self, id: &str) -> ApiResult<Run> {
        self.inner.stop_run(id)
    }
    fn delete_run(&self, id: &str) -> ApiResult<()> {
        self.inner.delete_run(id)
    }
    fn delete_dataset(&self, id: &str, files: bool) -> ApiResult<()> {
        self.inner.delete_dataset(id, files)
    }
    fn metrics(&self, id: &str) -> ApiResult<RunMetrics> {
        self.inner.metrics(id)
    }
    fn samples(&self, id: &str, epoch: Option<u32>) -> ApiResult<RunSamples> {
        self.inner.samples(id, epoch)
    }
    fn logs(&self, id: &str) -> ApiResult<RunLogs> {
        self.inner.logs(id)
    }
    fn events(&self, id: &str) -> ApiResult<Subscription> {
        self.inner.events(id)
    }
    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        self.inner.predict(req)
    }
    fn benches(&self) -> ApiResult<Vec<Bench>> {
        self.inner.benches()
    }
    fn bench_catalog(&self) -> ApiResult<BenchCatalog> {
        self.inner.bench_catalog()
    }
    fn bench_entries(&self, bench: &str) -> ApiResult<Vec<BenchEntry>> {
        self.inner.bench_entries(bench)
    }
    fn bench_loops(&self, bench: &str) -> ApiResult<Vec<BenchLoop>> {
        self.inner.bench_loops(bench)
    }
}

fn open_with_backend(
    api: Arc<dyn Backend>,
    width: f32,
    height: f32,
    scale_factor: f32,
) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let window_size = size(px(width), px(height));
    let mut app = HeadlessApp::new(HeadlessOptions { size: window_size, scale_factor, ..Default::default() });
    let window = app.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), window_size);
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
            .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    settle(&mut app, window);
    (app, window)
}

fn wheel_at(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, x: f32, y: f32, delta_y: f32) {
    app.simulate_input(
        window.into(),
        PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: point(px(x), px(y)),
            delta: ScrollDelta::Pixels(point(px(0.), px(delta_y))),
            modifiers: Default::default(),
            touch_phase: TouchPhase::Moved,
        }),
    );
    app.frame(window.into());
    app.run_until_idle();
}

#[test]
fn benchmarks_unreachable_server_shows_error_panel_and_recovers() {
    let fail = Arc::new(AtomicBool::new(true));
    let api: Arc<dyn Backend> =
        Arc::new(ToggleSystemBackend { inner: MockBackend::new(4200, true), fail_system: fail.clone() });
    let (mut app, window) = open_with_backend(api, 960., 640., 1.0);

    // 1. All three benchmark routes surface the unreachable `/api/system` error via `error_panel`
    //    (mounting both Retry and Connection settings) rather than hanging on "Loading…" or
    //    showing "not found".
    let cases = [
        (Route::Benches, "benches-retry", "benches-retry-settings"),
        (Route::Bench("bn_3c90".into()), "bench-retry", "bench-retry-settings"),
        (
            Route::BenchEntry("bn_3c90".into(), "be_3c01".into()),
            "bench-entry-retry",
            "bench-entry-retry-settings",
        ),
    ];
    for (route, retry_key, settings_key) in cases {
        drive(&mut app, window, |this, w, cx| this.navigate(route.clone(), w, cx));
        settle(&mut app, window);
        drive(&mut app, window, |this, w, cx| {
            assert!(
                this.bench_blocked_error().is_some(),
                "{route:?}: system failure should be reported as bench_blocked_error"
            );
            assert!(!this.bench_discovering());
            assert!(
                w.is_focus_mounted(&this.fh(retry_key, cx)),
                "{route:?}: Retry button should be mounted in error panel"
            );
            assert!(
                w.is_focus_mounted(&this.fh(settings_key, cx)),
                "{route:?}: Connection settings button should be mounted in error panel"
            );
        });
    }

    // 2. Once the server responds again and Retry runs, Benchmarks automatically loads its list.
    fail.store(false, Ordering::Relaxed);
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Benches, w, cx);
        this.retry_all(cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.bench_on(), "vision_bench feature should be discovered after recovery");
        assert!(this.bench_blocked_error().is_none());
        assert!(
            this.bench.list.ready().is_some_and(|l| !l.is_empty()),
            "benchmark list should populate after server recovery"
        );
    });

    // 3. While capability discovery is still loading (`hold_loading`), detail/entry routes treat
    //    it as discovering (skeleton) rather than "not found" or error.
    drive(&mut app, window, |this, w, cx| {
        this.hold_loading = true;
        this.retry_all(cx);
        this.navigate(Route::Bench("bn_3c90".into()), w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(this.bench_discovering());
        assert!(this.bench_blocked_error().is_none());
        assert!(!w.is_focus_mounted(&this.fh("bench-retry", cx)));
    });
}

#[test]
fn select_dropdown_dismisses_on_dialog_scroll_but_scrolls_inside_menu() {
    let api: Arc<dyn Backend> = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with_backend(api, 960., 640., 1.5);

    // Open New training run dialog and open the Dataset select.
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    settle(&mut app, window);
    let dialog_bounds = drive(&mut app, window, |this, _, cx| {
        assert!(
            this.dialog_scroll.max_offset().height > px(0.),
            "new-run dialog body should overflow at 960x640"
        );
        this.menu_open = Some("nr-dataset");
        this.menu_trigger.set(None);
        cx.notify();
        this.dialog_scroll.bounds()
    });
    settle(&mut app, window);
    let trigger = drive(&mut app, window, |this, _, _| {
        this.menu_trigger.get().expect("dataset select trigger bounds measured")
    });

    // 1. Scrolling INSIDE the open deferred dropdown menu must NOT scroll the dialog or close it.
    let menu_x = f32::from(trigger.left()) + 24.;
    let menu_y = f32::from(trigger.bottom()) + 24.;
    wheel_at(&mut app, window, menu_x, menu_y, -60.);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(
            this.menu_open,
            Some("nr-dataset"),
            "scrolling inside the open select menu should keep it open"
        );
        assert_eq!(
            this.dialog_scroll.offset().y,
            px(0.),
            "occluded select menu must not scroll dialog-body behind it"
        );
    });

    // 2. Scrolling OUTSIDE the open menu over dialog-body scrolls the dialog and dismisses the menu.
    let outside_x = f32::from(dialog_bounds.left()) + 40.;
    let outside_y = f32::from(dialog_bounds.top()) + 40.;
    wheel_at(&mut app, window, outside_x, outside_y, -120.);
    drive(&mut app, window, |this, _, _| {
        assert!(
            this.dialog_scroll.offset().y < px(0.),
            "dialog-body should scroll when wheeled outside the menu"
        );
        assert_eq!(this.menu_open, None, "scrolling dialog-body must dismiss the open select dropdown");
        assert!(this.menu_trigger.get().is_none());
    });

    // 3. Defense-in-depth: programmatic scroll moving the trigger out of `dialog-body` also closes it.
    drive(&mut app, window, |this, _, cx| {
        this.dialog_scroll.set_offset(point(px(0.), px(0.)));
        this.menu_open = Some("nr-dataset");
        this.menu_trigger.set(None);
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        let max = this.dialog_scroll.max_offset().height;
        this.dialog_scroll.set_offset(point(px(0.), -max));
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(
            this.menu_open, None,
            "programmatically scrolling trigger outside dialog-body must close menu_open"
        );
    });
}

#[test]
fn model_export_menu_dismisses_on_page_scroll_and_scrollbar_click() {
    let api: Arc<dyn Backend> = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with_backend(api, 960., 640., 1.5);

    drive(&mut app, window, |this, w, cx| {
        let id = this.store.models.ready().unwrap()[0].id.clone();
        this.navigate(Route::Model(id), w, cx);
    });
    settle(&mut app, window);

    // Open Export menu and measure its trigger.
    let page_bounds = drive(&mut app, window, |this, _, cx| {
        assert!(this.page_scroll.max_offset().height > px(0.), "model page should overflow at 960x640");
        this.export_open = true;
        this.menu_trigger.set(None);
        cx.notify();
        this.page_scroll.bounds()
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.export_open);
        assert!(this.menu_trigger.get().is_some(), "export trigger bounds should be measured");
    });

    // 1. Wheeling over the page body (outside Export menu) scrolls the page and closes Export.
    let scroll_x = f32::from(page_bounds.left()) + 120.;
    let scroll_y = f32::from(page_bounds.top()) + 300.;
    wheel_at(&mut app, window, scroll_x, scroll_y, -120.);
    drive(&mut app, window, |this, _, _| {
        assert!(this.page_scroll.offset().y < px(0.), "model page should scroll on wheel");
        assert!(!this.export_open, "page scroll must dismiss the open Export menu");
    });

    // 2. Clicking the page scrollbar track also dismisses Export.
    drive(&mut app, window, |this, _, cx| {
        this.page_scroll.set_offset(point(px(0.), px(0.)));
        this.export_open = true;
        this.menu_trigger.set(None);
        cx.notify();
    });
    settle(&mut app, window);
    let track_x = f32::from(page_bounds.right()) - 4.;
    let track_y = f32::from(page_bounds.bottom()) - 40.;
    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: point(px(track_x), px(track_y)),
            modifiers: Default::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.frame(window.into());
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(!this.export_open, "scrollbar click must dismiss the open Export menu");
    });
}

#[test]
fn failed_run_range_switch_stays_inside_map_card_at_narrow_and_hidpi_viewports() {
    for (width, height, scale) in [(960., 640., 1.5), (1120., 800., 2.0)] {
        let api: Arc<dyn Backend> = Arc::new(MockBackend::new(4200, true));
        let (mut app, window) = open_with_backend(api, width, height, scale);

        // Pick a run and configure it as a failed partial run with low mAP (< 0.25) so both the
        // `So far / All 100` segmented switch and the zoomed-axis caption are active.
        let run_id = drive(&mut app, window, |this, w, cx| {
            let runs = this.store.runs.ready().unwrap().clone();
            let target = runs
                .iter()
                .find(|r| r.status == RunStatus::Failed)
                .or_else(|| runs.first())
                .unwrap()
                .id
                .clone();
            if let crate::store::Load::Ready(list) = &mut this.store.runs
                && let Some(r) = list.iter_mut().find(|r| r.id == target)
            {
                r.status = RunStatus::Failed;
                r.epoch = 4;
                r.epochs = 100;
            }
            this.navigate(Route::Run(target.clone()), w, cx);
            target
        });
        settle(&mut app, window);
        // Clamp epoch mAP values below 0.25 and cap steps to epoch 4 so both `unit_zoomed(map_max)`
        // and the `So far / All 100` switch are guaranteed active.
        drive(&mut app, window, |this, _, cx| {
            if let Some(crate::store::Load::Ready(rd)) = this.store.run_data.get(&run_id) {
                let mut m = rd.metrics.clone();
                let spe = rd.spe;
                m.steps.retain(|s| s.epoch <= 4);
                m.epochs.retain(|e| e.epoch <= 4);
                for ep in &mut m.epochs {
                    ep.val.map50_95 = ep.val.map50_95.min(0.08);
                    ep.val.map50 = ep.val.map50.min(0.15);
                }
                this.store
                    .run_data
                    .insert(run_id.clone(), crate::store::Load::Ready(crate::store::RunData::new(m, spe)));
            }
            this.run_x_fit = Some(true);
            cx.notify();
        });
        settle(&mut app, window);
        drive(&mut app, window, |this, w, cx| {
            assert!(
                w.is_focus_mounted(&this.fh("run-x-fit", cx)),
                "So far / All 100 switch should be mounted at {width}x{height}@{scale}x"
            );
        });

        let (bg_panel, bg_inset) = drive(&mut app, window, |_, _, cx| {
            let t = theme(cx);
            (t.bg_panel, t.bg_inset)
        });
        let s = scale;
        let scene = app.scene(window.into());

        // Find the two side-by-side primary chart cards (Validation mAP and Training loss): two
        // matching-top `bg_panel` quads at least 250 logical px wide and 150 logical px tall.
        let mut chart_panels: Vec<(f32, f32, f32, f32)> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Quad(q) if q.background == bg_panel.into() => {
                    let x0 = sp(q.bounds.origin.x) / s;
                    let y0 = sp(q.bounds.origin.y) / s;
                    let w = sp(q.bounds.size.width) / s;
                    let h = sp(q.bounds.size.height) / s;
                    (w >= 250. && h >= 150.).then_some((x0, y0, x0 + w, y0 + h))
                }
                _ => None,
            })
            .collect();
        chart_panels.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)));

        // Pick the first side-by-side pair (same top edge within 2 px): left = mAP, right = loss.
        let (map_card, loss_card) = chart_panels
            .windows(2)
            .find_map(|pair| {
                let (a, b) = (pair[0], pair[1]);
                ((a.1 - b.1).abs() <= 2. && a.2 < b.0).then_some((a, b))
            })
            .unwrap_or_else(|| {
                panic!(
                    "expected side-by-side primary chart panels at {width}x{height}@{scale}x; panels={chart_panels:?}"
                )
            });

        // Locate the `run-x-fit` segmented control inside the mAP header band (`bg_inset` quad,
        // ~28 px tall, starting inside `map_card`'s horizontal span; note `folo-ui` paints borders
        // in a separate pass after children).
        let header_top = map_card.1;
        let header_bot = map_card.1 + 44.;
        let segs: Vec<(f32, f32)> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Quad(q) if q.background == bg_inset.into() => {
                    let x0 = sp(q.bounds.origin.x) / s;
                    let y0 = sp(q.bounds.origin.y) / s;
                    let x1 = x0 + sp(q.bounds.size.width) / s;
                    let h = sp(q.bounds.size.height) / s;
                    (y0 >= header_top
                        && y0 <= header_bot
                        && (20. ..=34.).contains(&h)
                        && x0 >= map_card.0
                        && x0 < loss_card.0)
                        .then_some((x0, x1))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            segs.len(),
            1,
            "expected one segmented switch (So far / All 100) in mAP header at {width}x{height}@{scale}x; got {segs:?}"
        );
        let (seg_left, seg_right) = segs[0];
        assert!(
            seg_right <= map_card.2 - 8.,
            "at {width}x{height}@{scale}x, So far / All 100 ({seg_left:.1}..{seg_right:.1}) overflows mAP card right edge ({:.1}) toward Training loss ({:.1})",
            map_card.2,
            loss_card.0
        );

        // No header-band glyph may bleed across the gap between `map_card` and `loss_card`.
        for p in &scene.primitives {
            if let Primitive::MonoSprite(g) = p
                && g.kind == folo_ui::SpriteKind::Glyph
            {
                let gy = sp(g.bounds.origin.y) / s;
                let gx0 = sp(g.bounds.origin.x) / s;
                let gx1 = gx0 + sp(g.bounds.size.width) / s;
                if gy >= header_top && gy <= header_bot {
                    assert!(
                        gx1 <= map_card.2 || gx0 >= loss_card.0,
                        "at {width}x{height}@{scale}x, header glyph ({gx0:.1}..{gx1:.1}) bleeds into gap between mAP ({:.1}) and Training loss ({:.1})",
                        map_card.2,
                        loss_card.0
                    );
                }
            }
        }
    }
}
