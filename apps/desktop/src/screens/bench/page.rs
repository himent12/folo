//! A benchmark's page: its header and its tabs.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use super::format::source_line;
use crate::api::time;
use crate::app::{FoloApp, act};
use crate::bench::BenchTab;
use crate::controls::{Btn, Tab};
use crate::screens::{detail_header, loading_rows, page_header};
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    // ---------- a benchmark ----------

    pub fn bench_view(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_bench(id, cx);
        if self.bench_unsupported() {
            let header = page_header(&t, "Benchmark", "Not available on this server", vec![]);
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let Some(b) = self.bench.list.ready().and_then(|l| l.iter().find(|b| b.id == id)).cloned() else {
            let header = page_header(&t, "Benchmark", "", vec![]);
            let body = if let Some(e) = self.bench.list.error().cloned() {
                self.error_panel(&e, "bench-retry", window, cx)
            } else if self.bench.list.is_loading() {
                panel(&t).child(loading_rows(&t, "bench", 5)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Benchmark not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body, cx);
        };
        let now = self.store.api.now();
        let created = time::parse(&b.created).map_or(String::new(), |c| time::ago(c, now));
        let title = div()
            .w_full()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).min_w_0().ellipsis().child(b.name.clone()))
            .when(b.active, |d| {
                d.child(div().flex_none().child(status_badge_of(
                    &t,
                    StatusKind::Running,
                    "Running",
                    format!("bench-hdr-{}", b.id),
                )))
            });
        let subtitle = format!(
            "{} · {} · {} · {} · created {created}",
            source_line(&b),
            counted(b.images, "image", "images"),
            counted(b.boxes, "box", "boxes"),
            plural(b.plan.conditions.len(), "condition"),
        );
        let confirming = self.bench.confirm_delete.as_ref().is_some_and(|(x, i)| *x == b.id && i.is_none());
        let mut actions = Vec::new();
        if confirming {
            actions.push(
                self.button(
                    "bench-del-cancel",
                    "Keep it",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, w, cx| {
                        this.bench.confirm_delete = None;
                        this.focus_key("bench-del", w, cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
            actions.push(
                self.button(
                    "bench-del-confirm",
                    "Delete benchmark and entries",
                    Btn::DangerCommit,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.confirm_bench_delete(cx)),
                )
                .into_any_element(),
            );
        } else if b.active {
            actions.push(
                self.button_disabled(
                    "bench-del",
                    "Delete",
                    Btn::Secondary,
                    Some("icons/trash-2.svg"),
                    "Stop its running entries and loops first",
                    window,
                    cx,
                )
                .into_any_element(),
            );
        } else {
            let bid = b.id.clone();
            actions.push(
                self.button(
                    "bench-del",
                    "Delete",
                    Btn::Secondary,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(move |this, w, cx| {
                        this.bench.confirm_delete = Some((bid.clone(), None));
                        this.focus_key("bench-del-cancel", w, cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        let header = detail_header(&t, title, subtitle, actions);

        let entries = self.bench.entries.get(id).cloned().unwrap_or_default();
        let loops = self.bench.loops.get(id).cloned().unwrap_or_default();
        let n_entries = entries.ready().map_or(b.entries as usize, |l| l.len());
        let n_loops = loops.ready().map(|l| l.len());
        let tab = self.bench.tab;
        // The page's sections are underline tabs with their counts (a segmented control is a filter).
        let tabs = self.tabs(
            "bench-tab",
            vec![
                Tab::new("Leaderboard", Some(n_entries.to_string())),
                Tab::new("Plan", Some(b.plan.conditions.len().to_string())),
                Tab::new("Loops", n_loops.map(|n| n.to_string())),
            ],
            tab as usize,
            window,
            cx,
            std::rc::Rc::new(|this, i, _, _| {
                let next = [BenchTab::Leaderboard, BenchTab::Plan, BenchTab::Loops][i.min(2)];
                if this.bench.tab != next {
                    this.bench.tab = next;
                    if matches!(this.bench.confirm_delete, Some((_, Some(_)))) {
                        this.bench.confirm_delete = None;
                    }
                }
            }),
        );
        let description = (!b.description.is_empty())
            .then(|| text(ty::BASE, t.text_secondary).max_w(px(820.)).child(b.description.clone()));
        let content = match tab {
            BenchTab::Leaderboard => self.bench_leaderboard(&t, &b, &entries, window, cx),
            BenchTab::Plan => self.bench_plan(&t, &b, window, cx),
            BenchTab::Loops => self.bench_loops(&t, &b, &loops, window, cx),
        };
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .children(description)
            .child(div().flex().child(tabs))
            .child(content);
        self.page_frame(&t, header, body, cx)
    }
}

#[cfg(test)]
mod tests {
    use crate::app::Route;

    #[test]
    fn routes_round_trip() {
        assert_eq!(Route::parse("benchmarks"), Some(Route::Benches));
        assert_eq!(Route::parse("bench:bn_1"), Some(Route::Bench("bn_1".into())));
        assert_eq!(
            Route::parse("bench-entry:bn_1/be_2"),
            Some(Route::BenchEntry("bn_1".into(), "be_2".into()))
        );
        assert_eq!(Route::parse("bench-entry:bn_1"), None);
        assert_eq!(Route::BenchEntry("b".into(), "e".into()).section(), crate::app::Section::Benchmarks);
    }

    #[test]
    fn a_benchmark_opens_on_three_underline_tabs() {
        use crate::app::tests::{drive, open};
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Bench("bn_3c90".into()), w, cx));
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            // Leaderboard, Plan and Loops are tabs (each a Tab stop), no longer one segmented control.
            for k in ["bench-tab-0", "bench-tab-1", "bench-tab-2"] {
                assert!(keys.iter().any(|x| x == k), "{k}: {keys:?}");
            }
        });
    }

    #[test]
    fn leaderboard_draws_sort_headers_pin_and_expandable_failed_error() {
        use crate::api::mock::MockBackend;
        use crate::api::{BenchEntry, BenchEntryStatus};
        use crate::app::FoloApp;
        use crate::app::tests::drive;
        use folo_ui::{
            AppContext, Bounds, HeadlessApp, HeadlessOptions, Modifiers, MouseButton, MouseDownEvent,
            MouseUpEvent, PlatformInput, Primitive, SpriteKind, WindowBounds, WindowOptions, point, px, size,
        };
        let click_at = |app: &mut HeadlessApp, window: folo_ui::WindowHandle<FoloApp>, x: f32, y: f32| {
            let pos = point(px(x), px(y));
            app.simulate_input(
                window.into(),
                PlatformInput::MouseDown(MouseDownEvent {
                    button: MouseButton::Left,
                    position: pos,
                    modifiers: Modifiers::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
            );
            app.simulate_input(
                window.into(),
                PlatformInput::MouseUp(MouseUpEvent {
                    button: MouseButton::Left,
                    position: pos,
                    modifiers: Modifiers::default(),
                    click_count: 1,
                }),
            );
            app.run_until_idle();
        };
        use std::sync::Arc;

        let mut app = HeadlessApp::new(HeadlessOptions::default()).with_assets(crate::Assets);
        let api = Arc::new(MockBackend::new(4200, true));
        let window = app.update(|cx| {
            cx.set_global(crate::theme::Theme::new(crate::theme::Mode::Dark));
            let bounds = Bounds::new(point(px(0.), px(0.)), size(px(1280.), px(800.)));
            let options =
                WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
            cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
                .expect("open window")
        });
        drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
        let bid = "bn_3c90";
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Bench(bid.into()), w, cx));
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());

        // Inject a failed entry with a detailed error alongside the done entries.
        let (done_id, failed_id) = drive(&mut app, window, |this, _, cx| {
            let list = this.bench.entries.get_mut(bid).and_then(|l| l.ready_mut()).expect("entries loaded");
            let done_id = list.iter().find(|e| e.status == BenchEntryStatus::Done).unwrap().id.clone();
            let mut fail: BenchEntry = list[0].clone();
            fail.id = "be_fail_1".into();
            fail.name = "broken-model".into();
            fail.status = BenchEntryStatus::Failed;
            fail.score = None;
            fail.error = Some("CUDA out of memory while evaluating severity 3".into());
            list.push(fail);
            this.focus.borrow_mut().clear();
            cx.notify();
            (done_id, "be_fail_1".to_string())
        });
        app.frame(window.into());

        // Collect action icon pairs `[left_action, delete]` in the right-hand actions column:
        // each finished/failed row paints two 15–16 px icons inside 28 px buttons 4 px apart (dx ≈ 32 px),
        // independent of titlebar height or header wrapping across platforms.
        let sp = |v: folo_ui::ScaledPixels| f64::from(v) as f32;
        let row_left_actions = |app: &mut folo_ui::HeadlessApp| -> Vec<(f32, f32)> {
            let mut pts: Vec<(f32, f32)> = app
                .scene(window.into())
                .primitives
                .iter()
                .filter_map(|p| match p {
                    Primitive::MonoSprite(s)
                        if s.kind == SpriteKind::Svg
                            && sp(s.bounds.origin.x) >= 1100.
                            && sp(s.bounds.size.width) >= 14. =>
                    {
                        Some((
                            sp(s.bounds.origin.x) + sp(s.bounds.size.width) / 2.,
                            sp(s.bounds.origin.y) + sp(s.bounds.size.height) / 2.,
                        ))
                    }
                    _ => None,
                })
                .collect();
            pts.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.total_cmp(&b.0)));
            let mut rows: Vec<(f32, f32)> = Vec::new();
            let mut i = 0;
            while i < pts.len() {
                let mut band = vec![pts[i]];
                let y0 = pts[i].1;
                i += 1;
                while i < pts.len() && (pts[i].1 - y0).abs() < 8. {
                    band.push(pts[i]);
                    i += 1;
                }
                band.sort_by(|a, b| a.0.total_cmp(&b.0));
                if let Some(pair) = band.windows(2).find(|w| (24.0..=40.0).contains(&(w[1].0 - w[0].0))) {
                    rows.push(pair[0]);
                }
            }
            rows
        };
        let row_actions = row_left_actions(&mut app);
        assert!(row_actions.len() >= 2, "expected done + failed action pairs: {row_actions:?}");
        // First row is the first Done entry (Pin); last row is the injected Failed entry (Chevron).
        let pin_pt = *row_actions.first().expect("pin icon painted");
        let chevron_pt = *row_actions.last().expect("chevron icon painted");

        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            for k in [
                "bench-sort-score",
                "bench-sort-accuracy",
                "bench-sort-robustness",
                "bench-sort-self_labeling",
                "bench-sort-calibration",
                "bench-sort-localization",
            ] {
                assert!(keys.iter().any(|x| x == k), "missing sort header {k}: {keys:?}");
            }
            assert!(keys.iter().any(|x| x == &format!("bench-entry-{done_id}-pin")), "pin button rendered");
            assert!(
                keys.iter().any(|x| x == &format!("bench-entry-{failed_id}-err-toggle")),
                "failed entry has error expand toggle"
            );
            assert!(
                !keys.iter().any(|x| x == &format!("bench-entry-{failed_id}-err-copy")),
                "error panel collapsed by default"
            );
        });

        // Pointer click on Pin must pin the entry AND stay on the leaderboard (no row navigation!).
        click_at(&mut app, window, pin_pt.0, pin_pt.1);
        drive(&mut app, window, |this, _, _| {
            assert_eq!(this.route, Route::Bench(bid.into()), "clicking Pin must not open the entry");
            assert_eq!(this.bench.pinned_entry, Some((bid.into(), done_id.clone())));
        });

        // Pointer click on failed-entry chevron expands once (no double-toggle); second click collapses.
        click_at(&mut app, window, chevron_pt.0, chevron_pt.1);
        drive(&mut app, window, |this, _, _| {
            assert!(
                this.bench.expanded_errors.contains(&(bid.into(), failed_id.clone())),
                "one chevron click expands error details"
            );
            this.bench.leaderboard_sort = Some(("accuracy", true));
            this.focus.borrow_mut().clear();
        });
        app.frame(window.into());

        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            assert!(keys.iter().any(|x| x == "bench-pin-clear"), "Clear baseline button shown");
            assert!(keys.iter().any(|x| x == "bench-sort-reset"), "Reset order button shown");
            assert!(
                keys.iter().any(|x| x == &format!("bench-entry-{failed_id}-err-copy")),
                "expanded error panel shows Copy error button"
            );
        });

        // Second chevron click collapses the error panel again.
        click_at(&mut app, window, chevron_pt.0, chevron_pt.1);
        drive(&mut app, window, |this, w, cx| {
            assert!(
                !this.bench.expanded_errors.contains(&(bid.into(), failed_id.clone())),
                "second chevron click collapses error details"
            );
            // Navigating back to the Benchmarks list resets leaderboard_sort.
            this.navigate(Route::Benches, w, cx);
            assert_eq!(this.bench.leaderboard_sort, None);
        });
    }

    #[test]
    fn bench_list_filter_plan_severity_chips_and_entry_image_inspector() {
        use crate::app::tests::{drive, open};
        use crate::bench::BenchTab;
        let (mut app, window) = open();
        let send_enter = |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<crate::app::FoloApp>| {
            let keystroke = folo_ui::Keystroke::parse("enter").unwrap();
            app.simulate_input(
                w.into(),
                folo_ui::PlatformInput::KeyDown(folo_ui::KeyDownEvent { keystroke, is_held: false }),
            );
            app.run_until_idle();
            app.frame(w.into());
        };
        // 1. Benchmarks list toolbar, sort headers, search filter, and Reset.
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Benches, w, cx));
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            for k in [
                "bench-filter",
                "bench-list-sort-images",
                "bench-list-sort-entries",
                "bench-list-sort-leader",
            ] {
                assert!(keys.iter().any(|x| x == k), "missing {k}: {keys:?}");
            }
        });
        drive(&mut app, window, |this, _, cx| {
            this.bench_search.update(cx, |i, cx| i.set_text("zz-no-match-zz", cx));
            this.focus.borrow_mut().clear();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            assert!(keys.iter().any(|x| x == "bench-empty-reset"), "empty reset shown: {keys:?}");
            this.focus_key("bench-empty-reset", w, cx);
        });
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, cx| {
            assert!(this.bench_search.read(cx).text().is_empty(), "empty reset cleared search");
        });

        // 2. Plan tab interactive severity preview chips.
        let bid = "bn_5e12";
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Bench(bid.into()), w, cx);
            this.ensure_bench(bid, cx);
            this.bench.tab = BenchTab::Plan;
        });
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        let (cond_id, sev) = drive(&mut app, window, |this, _, _| {
            let b = this.bench.list.ready().unwrap().iter().find(|b| b.id == bid).unwrap();
            let c = b.plan.conditions.first().expect("demo benchmark has conditions");
            let s = *c.severities.first().unwrap();
            (c.condition.clone(), s)
        });
        let sev_key = format!("plan-sev-{cond_id}-{sev}");
        drive(&mut app, window, |this, w, cx| this.focus_key(&sev_key, w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert_eq!(
                this.bench.plan_preview_sev.get(&(bid.to_string(), cond_id.clone())),
                Some(&sev),
                "activating severity chip updates preview severity"
            );
        });

        // 3. Entry image inline inspector: open, toggle Truth/Detections, step Next, and Close.
        let eid = drive(&mut app, window, |this, w, cx| {
            let eid = this
                .bench
                .entries
                .get(bid)
                .and_then(|l| l.ready())
                .unwrap()
                .iter()
                .find(|e| e.status == crate::api::BenchEntryStatus::Done)
                .unwrap()
                .id
                .clone();
            this.navigate(Route::BenchEntry(bid.into(), eid.clone()), w, cx);
            eid
        });
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        let (first_img, second_img) = drive(&mut app, window, |this, _, _| {
            let page = this
                .bench
                .entry_images
                .get(&(bid.into(), eid.clone()))
                .and_then(|x| x.page.ready())
                .expect("entry images loaded");
            assert!(page.images.len() >= 2, "demo entry has at least 2 images");
            (page.images[0].image.id.clone(), page.images[1].image.id.clone())
        });
        drive(&mut app, window, |this, w, cx| this.focus_key(&format!("eimg-open-{first_img}"), w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            assert_eq!(
                this.bench.selected_entry_image,
                Some((bid.into(), eid.clone(), first_img.clone())),
                "activating tile opens inline inspector"
            );
            assert!(this.fh("eimg-prev", cx).is_focused(w), "opening inspector moves focus to its bar");
        });
        drive(&mut app, window, |this, w, cx| this.focus_key("eimg-toggle-truth", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert!(this.bench.entry_inspect_hide_truth, "Truth chip toggles ground truth off");
        });
        drive(&mut app, window, |this, w, cx| this.focus_key("eimg-next", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert_eq!(
                this.bench.selected_entry_image,
                Some((bid.into(), eid.clone(), second_img.clone())),
                "Next steps to the second image"
            );
        });
        drive(&mut app, window, |this, w, cx| this.focus_key("eimg-close", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            assert_eq!(this.bench.selected_entry_image, None, "Close dismisses inline inspector");
            assert!(
                this.fh(format!("eimg-open-{second_img}"), cx).is_focused(w),
                "Close returns focus to the inspected tile"
            );
        });
    }

    #[test]
    fn benchmark_error_panels_render_and_trigger_retry() {
        use std::sync::Arc;

        use folo_ui::{KeyDownEvent, Keystroke, PlatformInput};

        use crate::api::mock::MockBackend;
        use crate::api::{ApiError, BenchEntryStatus};
        use crate::app::Route;
        use crate::app::tests::{drive, open_with};
        use crate::bench::{BenchTab, EntryImages, ImageSort};
        use crate::store::Load;

        let send_enter = |app: &mut folo_ui::HeadlessApp,
                          window: folo_ui::WindowHandle<crate::app::FoloApp>| {
            app.simulate_input(
                window.into(),
                PlatformInput::KeyDown(KeyDownEvent {
                    keystroke: Keystroke { key: "enter".into(), ..Default::default() },
                    is_held: false,
                }),
            );
        };
        let err = || ApiError {
            status: Some(500),
            code: "internal".into(),
            message: "Temporary failure".into(),
            hint: None,
        };
        let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true)));
        let bid = "bn_3c90";

        // 1. Leaderboard failed -> `bench-board-retry` reloads entries.
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Bench(bid.into()), w, cx));
        app.frame(window.into());
        app.run_until_idle();
        drive(&mut app, window, |this, _, cx| {
            this.bench.entries.insert(bid.into(), Load::Failed(err()));
            cx.notify();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.focus_key("bench-board-retry", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        // While reloading (button replaced by passing_stop anchor), focus stays anchored in place.
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            assert!(
                this.fh("bench-board-retry", cx).is_focused(w),
                "board retry keeps its focus anchor while loading"
            );
        });
        app.run_until_idle();
        app.frame(window.into());
        let eid = drive(&mut app, window, |this, w, cx| {
            assert!(
                this.fh("bench-board-retry", cx).is_focused(w),
                "board retry keeps its focus anchor after reload"
            );
            w.focus_next();
            assert!(
                this.fh("bench-sort-score", cx).is_focused(w),
                "Tab after board retry continues into the reloaded leaderboard instead of restarting"
            );
            let list = this.bench.entries.get(bid).and_then(|l| l.ready()).expect("entries reloaded");
            list.iter().find(|e| e.status == BenchEntryStatus::Done).unwrap().id.clone()
        });

        // 2. Loops tab failed -> `bench-loops-retry` reloads loops.
        drive(&mut app, window, |this, _, cx| {
            this.bench.tab = BenchTab::Loops;
            this.bench.loops.insert(bid.into(), Load::Failed(err()));
            cx.notify();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.focus_key("bench-loops-retry", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        app.run_until_idle();
        drive(&mut app, window, |this, _, _| {
            assert!(this.bench.loops.get(bid).and_then(|l| l.ready()).is_some(), "loops reloaded");
        });

        // 3. Entry result failed -> `bench-result-retry` reloads result.
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::BenchEntry(bid.into(), eid.clone()), w, cx)
        });
        app.frame(window.into());
        app.run_until_idle();
        drive(&mut app, window, |this, _, cx| {
            this.bench.results.insert((bid.into(), eid.clone()), Load::Failed(err()));
            cx.notify();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.focus_key("bench-result-retry", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        app.run_until_idle();
        drive(&mut app, window, |this, _, _| {
            assert!(
                this.bench.results.get(&(bid.into(), eid.clone())).and_then(|l| l.ready()).is_some(),
                "entry result reloaded"
            );
        });

        // 4. Entry images failed -> `bench-entry-images-retry` reloads images.
        drive(&mut app, window, |this, _, cx| {
            this.bench.entry_images.insert(
                (bid.into(), eid.clone()),
                EntryImages {
                    sort: ImageSort::Worst,
                    page: Load::Failed(err()),
                    more: false,
                    more_error: None,
                    seq: 999,
                },
            );
            cx.notify();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.focus_key("bench-entry-images-retry", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        app.run_until_idle();
        drive(&mut app, window, |this, _, _| {
            assert!(
                this.bench
                    .entry_images
                    .get(&(bid.into(), eid.clone()))
                    .and_then(|x| x.page.ready())
                    .is_some(),
                "entry images reloaded"
            );
        });
    }
}
