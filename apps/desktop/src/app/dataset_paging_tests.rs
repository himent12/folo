//! Dataset index format/search filtering and later-page failure recovery (bounded backoff,
//! revision validation, inspector error overlay, and keyboard focus restoration).

use std::sync::Arc;

use super::Route;
use super::data::IMAGE_PAGE;
use super::tests::{drive, open_with};
use crate::api::SplitsExt;
use crate::api::mock::MockBackend;

/// The focus keys a page draws, once what it asked for has come.
fn drawn_keys(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<super::FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

#[test]
fn a_permanent_later_page_error_stops_retrying_and_recovers_on_retry() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_from_with(
        IMAGE_PAGE,
        crate::api::ApiError::new("permission_denied", "Cannot read dataset folder"),
        false,
    );
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Dataset("ds_8f2c".into()), w, cx));
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "ds-page-retry"), "inline retry action shown: {keys:?}");
    let calls_after_fail = api.image_lists().len();
    drive(&mut app, window, |this, _, _| {
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len());
        assert_eq!(got, Some(IMAGE_PAGE as usize), "keeps the already-loaded first page");
        assert!(!this.images_pending("ds_8f2c"), "does not spin loading indefinitely");
        let failed = this.failed_image_page("ds_8f2c").expect("caches failed page");
        assert_eq!(failed.offset, IMAGE_PAGE);
        assert_eq!(failed.error.code, "permission_denied");
    });
    // Advancing the clock must NOT auto-retry a permanent error.
    app.advance_clock(std::time::Duration::from_secs(60));
    assert_eq!(api.image_lists().len(), calls_after_fail, "no background retry loop");

    // Once fixed, clicking Retry clears the cached failure and loads the rest of the dataset.
    api.clear_fail_images();
    drive(&mut app, window, |this, _, cx| this.retry_failed_image_page("ds_8f2c", cx));
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len() as u64);
        assert_eq!(got, total);
        assert!(!this.images_pending("ds_8f2c"));
        assert!(this.failed_image_page("ds_8f2c").is_none());
    });
}

#[test]
fn repeated_transient_later_page_errors_stop_after_bounded_backoff() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_from_with(
        IMAGE_PAGE,
        crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878"),
        false,
    );
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    // Page 0 succeeded + initial page 1 failed -> 2 calls total.
    assert_eq!(api.image_lists().len(), 2, "page 0 + initial failed page 1");
    // Exact exponential backoff intervals: 3s, 6s, 12s (3 automatic retries -> 5 calls total).
    for (wait_secs, expected_calls) in [(3u64, 3usize), (6, 4), (12, 5)] {
        app.advance_clock(std::time::Duration::from_secs(wait_secs - 1));
        app.run_until_idle();
        assert_eq!(api.image_lists().len(), expected_calls - 1, "not before {wait_secs}s backoff");
        app.advance_clock(std::time::Duration::from_secs(1));
        app.run_until_idle();
        assert_eq!(api.image_lists().len(), expected_calls, "retry fires at {wait_secs}s");
    }
    drive(&mut app, window, |this, _, _| {
        assert!(!this.images_pending("ds_8f2c"), "bounded retries stopped");
        let failed = this.failed_image_page("ds_8f2c").expect("failed page surfaced after exhaustion");
        assert_eq!(failed.offset, IMAGE_PAGE);
    });
    // Further time passes with zero extra fetches.
    app.advance_clock(std::time::Duration::from_secs(60));
    app.run_until_idle();
    assert_eq!(api.image_lists().len(), 5, "no infinite retry loop after backoff exhaustion");

    // Reconnecting resumes exhausted server_unreachable page failures automatically.
    api.clear_fail_images();
    drive(&mut app, window, |this, _, cx| this.reconnected(cx));
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len() as u64);
        assert_eq!(got, total);
        assert!(this.failed_image_page("ds_8f2c").is_none());
    });
}

#[test]
fn transient_http_503_retries_while_permanent_http_404_stops_immediately() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    api.set_dataset_revision("ds_8f2c", Some("rev-1"));
    let (mut app, window) = open_with(api.clone());
    // Structured error body with HTTP 503 also retries and revalidates revision during backoff.
    api.fail_images_from_with(
        IMAGE_PAGE,
        crate::api::ApiError::new("service_unavailable", "Busy").with_status(503),
        true,
    );
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.images_pending("ds_8f2c"), "HTTP 503 schedules retry");
        assert!(this.failed_image_page("ds_8f2c").is_none());
    });
    // Revision changes during backoff: automatic retry restarts from offset 0 instead of splicing.
    let before = api.image_lists().len();
    api.set_dataset_revision("ds_8f2c", Some("rev-2"));
    app.advance_clock(std::time::Duration::from_secs(3));
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len() as u64);
        assert_eq!(got, total, "recovered after transient HTTP 503");
    });
    let after: Vec<u32> = api.image_lists()[before..].iter().map(|(_, q)| q.offset).collect();
    assert_eq!(after.first().copied(), Some(0), "changed revision during backoff restarts at 0: {after:?}");

    // Permanent HTTP 404 stops immediately without scheduling retries.
    let api2 = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app2, window2) = open_with(api2.clone());
    api2.fail_images_from_with(
        IMAGE_PAGE,
        crate::api::ApiError::new("http_error", "folo serve answered HTTP 404").with_status(404),
        false,
    );
    drive(&mut app2, window2, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    assert_eq!(api2.image_lists().len(), 2);
    app2.advance_clock(std::time::Duration::from_secs(30));
    app2.run_until_idle();
    assert_eq!(api2.image_lists().len(), 2, "HTTP 404 does not auto-retry");
    drive(&mut app2, window2, |this, _, _| {
        assert_eq!(this.failed_image_page("ds_8f2c").map(|f| f.error.status), Some(Some(404)));
    });
}

#[test]
fn manual_retry_checks_dataset_revision_before_resuming() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    api.set_dataset_revision("ds_8f2c", Some("rev-1"));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_from_with(IMAGE_PAGE, crate::api::ApiError::new("permission_denied", "Denied"), false);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    let before = api.image_lists().len();
    api.clear_fail_images();
    drive(&mut app, window, |this, _, cx| this.retry_failed_image_page("ds_8f2c", cx));
    let resumed: Vec<u32> = api.image_lists()[before..].iter().map(|(_, q)| q.offset).collect();
    assert_eq!(
        resumed.first().copied(),
        Some(IMAGE_PAGE),
        "same revision resumes at failed offset: {resumed:?}"
    );

    // When revision changes on disk, manual Retry restarts cleanly from offset 0.
    let api2 = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    api2.set_dataset_revision("ds_8f2c", Some("rev-1"));
    let (mut app2, window2) = open_with(api2.clone());
    api2.fail_images_from_with(IMAGE_PAGE, crate::api::ApiError::new("permission_denied", "Denied"), false);
    drive(&mut app2, window2, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    let before2 = api2.image_lists().len();
    api2.clear_fail_images();
    api2.set_dataset_revision("ds_8f2c", Some("rev-2"));
    drive(&mut app2, window2, |this, _, cx| this.retry_failed_image_page("ds_8f2c", cx));
    let restarted: Vec<u32> = api2.image_lists()[before2..].iter().map(|(_, q)| q.offset).collect();
    assert_eq!(restarted.first().copied(), Some(0), "changed revision restarts from page 0: {restarted:?}");
}

#[test]
fn failed_page_in_inspector_keeps_requested_image_and_recovers_on_retry() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_from_with(IMAGE_PAGE, crate::api::ApiError::new("permission_denied", "Denied"), false);
    let want = format!("img_{}", IMAGE_PAGE + 50);
    drive(&mut app, window, |this, w, cx| {
        this.open_sample("ds_8f2c".into(), want.clone(), "run_x".into(), w, cx);
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "insp-page-retry"), "inspector exposes retry: {keys:?}");
    let (overlay_bg, accent_bg) = drive(&mut app, window, |_, _, cx| {
        (crate::theme::theme(cx).bg_overlay, crate::theme::theme(cx).accent_fill())
    });
    let scene = app.scene(window.into());
    let sp = |v: folo_ui::ScaledPixels| f64::from(v) as f32;
    let card_b = scene
        .primitives
        .iter()
        .find_map(|p| match p {
            folo_ui::Primitive::Quad(q)
                if q.background.solid == overlay_bg && sp(q.bounds.size.width) >= 400. =>
            {
                Some(q.bounds)
            }
            _ => None,
        })
        .expect("error card quad painted");
    assert!(
        sp(card_b.origin.y) >= 0. && sp(card_b.origin.y + card_b.size.height) <= 800.,
        "error card centered inside viewport: {card_b:?}"
    );
    let retry_b = scene
        .primitives
        .iter()
        .rev()
        .find_map(|p| match p {
            folo_ui::Primitive::Quad(q)
                if q.background.solid == accent_bg
                    && sp(q.bounds.origin.y) >= sp(card_b.origin.y)
                    && sp(q.bounds.origin.y) <= sp(card_b.origin.y + card_b.size.height) =>
            {
                Some(q.bounds)
            }
            _ => None,
        })
        .expect("Retry button quad inside card");
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().expect("inspector stays open on page error");
        assert_eq!(insp.want.as_deref(), Some(want.as_str()), "does not fall back to index 0");
    });
    api.clear_fail_images();
    let rx = sp(retry_b.origin.x) + sp(retry_b.size.width) / 2.;
    let ry = sp(retry_b.origin.y) + sp(retry_b.size.height) / 2.;
    super::select_driving::click_n_at(&mut app, window, rx, ry, 1);
    app.run_until_idle();
    drawn_keys(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().expect("inspector lands on target");
        assert!(insp.want.is_none());
        assert_eq!(this.filtered_image("ds_8f2c", insp.index).map(|m| m.id.as_str()), Some(want.as_str()));
    });
}

#[test]
fn dropping_image_list_mid_paging_does_not_leak_failed_page() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_from_with(IMAGE_PAGE, crate::api::ApiError::new("dataset_not_found", "Gone"), false);
    let (first, second) = (api.hold("images"), api.hold("images"));
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_dataset("ds_8f2c", cx)).unwrap());
    app.run_until_held(1);
    drop(first);
    while !second.is_held() {
        app.run_until_held(1);
    }
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                this.store.images.remove("ds_8f2c");
            })
            .unwrap()
    });
    drop(second);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(this.failed_image_page("ds_8f2c").is_none(), "stale failed page not recorded");
        assert!(!this.images_pending("ds_8f2c"));
    });
}

#[test]
fn datasets_list_filters_by_format_and_search_before_loading_card_previews() {
    use folo_ui::Focusable;
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    // Filter down to a single dataset by search before visiting Datasets.
    drive(&mut app, window, |this, w, cx| {
        this.ds_search.update(cx, |i, cx| i.set_text("shapes-hard", cx));
        this.navigate(Route::Datasets, w, cx);
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "ds-format-filter"), "{keys:?}");
    assert!(keys.iter().any(|k| k == "ds-search-clear"), "{keys:?}");
    let cards: Vec<&String> = keys.iter().filter(|k| k.starts_with("ds-card-")).collect();
    assert_eq!(cards.len(), 1, "only the matching dataset card renders: {cards:?}");
    let previewed: std::collections::HashSet<String> =
        api.image_lists().into_iter().map(|(id, _)| id).collect();
    assert_eq!(previewed.len(), 1, "hidden cards did not trigger preview fetches: {previewed:?}");

    // Activating Clear filters restores focus to the search input instead of stranding it.
    let press_enter = |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<super::FoloApp>| {
        let keystroke = folo_ui::Keystroke::parse("enter").unwrap();
        app.simulate_input(
            w.into(),
            folo_ui::PlatformInput::KeyDown(folo_ui::KeyDownEvent { keystroke, is_held: false }),
        );
        app.run_until_idle();
        app.frame(w.into());
    };
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-search-clear", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(this.ds_search.read(cx).text().is_empty());
        assert!(this.ds_search.read(cx).focus_handle(cx).is_focused(w), "ds-search-clear restores focus");
    });

    drive(&mut app, window, |this, _, cx| {
        this.focus.borrow_mut().clear();
        this.ds_search.update(cx, |i, cx| i.set_text("no-such-dataset-xyz", cx));
        cx.notify();
    });
    let empty_keys = drawn_keys(&mut app, window);
    assert!(empty_keys.iter().any(|k| k == "ds-filter-clear"), "{empty_keys:?}");
    assert!(!empty_keys.iter().any(|k| k.starts_with("ds-card-")), "{empty_keys:?}");
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-filter-clear", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(this.ds_search.read(cx).text().is_empty());
        assert!(this.ds_search.read(cx).focus_handle(cx).is_focused(w), "ds-filter-clear restores focus");
    });
}

#[test]
fn dataset_images_search_sort_reset_and_clickable_class_balance_bars() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api);
    let ds_id = "ds_8f2c";
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Dataset(ds_id.into()), w, cx);
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "ds-img-sort"), "{keys:?}");
    assert!(!keys.iter().any(|k| k == "ds-img-reset"), "reset hidden when unfiltered: {keys:?}");

    // Filtering by filename updates filtered_indices and surfaces the Reset button.
    let first_file = app
        .update(|cx| {
            window.update(cx, |this, _, _| this.filtered_image(ds_id, 0).map(|m| m.file.clone())).unwrap()
        })
        .expect("dataset has images");
    let query = crate::screens::datasets::file_name(&first_file).to_string();
    drive(&mut app, window, |this, _, cx| {
        this.ds_img_search.update(cx, |i, cx| i.set_text(query.clone(), cx));
        this.ds_img_sort = crate::screens::datasets::ImageSort::MostBoxes;
        cx.notify();
    });
    let filtered_keys = drawn_keys(&mut app, window);
    assert!(filtered_keys.iter().any(|k| k == "ds-img-reset"), "{filtered_keys:?}");
    drive(&mut app, window, |this, _, _| {
        assert!(this.dataset_image_filters_active(ds_id));
        assert!(!this.filtered_indices(ds_id).is_empty());
    });

    // Activating Reset clears search, sort, split, and class filters.
    let press_enter = |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<super::FoloApp>| {
        let keystroke = folo_ui::Keystroke::parse("enter").unwrap();
        app.simulate_input(
            w.into(),
            folo_ui::PlatformInput::KeyDown(folo_ui::KeyDownEvent { keystroke, is_held: false }),
        );
        app.run_until_idle();
        app.frame(w.into());
    };
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-img-reset", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.dataset_image_filters_active(ds_id));
        assert!(this.ds_img_search.read(cx).text().is_empty());
        assert_eq!(this.ds_img_sort, crate::screens::datasets::ImageSort::Default);
    });

    // Statistics tab renders clickable class-balance bars that filter and jump to Images.
    drive(&mut app, window, |this, _, cx| {
        this.ds_tab = super::DsTab::Statistics;
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    let stats_keys = drawn_keys(&mut app, window);
    assert!(stats_keys.iter().any(|k| k == "ds-class-bar-0"), "{stats_keys:?}");
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-class-bar-0", w, cx));
    app.frame(window.into());
    press_enter(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.ds_tab, super::DsTab::Images);
        assert_eq!(this.class_filter(ds_id), Some(0));
        assert!(
            this.fh("ds-class", cx).is_focused(w),
            "activating a class bar moves focus to a persistent Images toolbar control"
        );
    });

    // Filename search & sort are scoped to their dataset: opening a run sample on another dataset
    // neither filters the other dataset nor erases the first dataset's search.
    let other_ds = "ds_3b91";
    drive(&mut app, window, |this, _, cx| {
        this.reset_dataset_image_filters(ds_id, cx);
        this.ds_img_search.update(cx, |i, cx| i.set_text(query.clone(), cx));
        this.ds_img_sort = crate::screens::datasets::ImageSort::NameAsc;
        this.ensure_dataset(other_ds, cx);
    });
    app.run_until_idle();
    app.frame(window.into());
    let other_first = drive(&mut app, window, |this, _, _| {
        assert_eq!(this.img_query(other_ds), "");
        assert_eq!(this.img_sort(other_ds), crate::screens::datasets::ImageSort::Default);
        this.filtered_image(other_ds, 0).map(|m| m.id.clone()).expect("other dataset loaded")
    });
    drive(&mut app, window, |this, w, cx| {
        this.open_inspector(other_ds.into(), 0, w, cx);
        if let Some(i) = this.inspector.as_mut() {
            i.want = Some(other_first.clone());
        }
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.img_query(ds_id), query, "first dataset's search stays intact");
        assert_eq!(this.img_sort(ds_id), crate::screens::datasets::ImageSort::NameAsc);
        this.close_inspector(w, cx);
    });

    // Under a non-default sort, appending a page that sorts ahead of the inspected image keeps the
    // inspector locked onto the same image id rather than jumping to the shifted index.
    drive(&mut app, window, |this, w, cx| {
        this.ds_img_search.update(cx, |i, cx| i.set_text("", cx));
        this.ds_img_sort = crate::screens::datasets::ImageSort::NameAsc;
        this.open_inspector(ds_id.into(), 0, w, cx);
    });
    app.frame(window.into());
    let shown_before = drive(&mut app, window, |this, _, _| {
        let idx = this.inspector.as_ref().unwrap().index;
        this.filtered_image(ds_id, idx).unwrap().id.clone()
    });
    drive(&mut app, window, |this, _, cx| {
        if let Some(crate::store::Load::Ready(imgs)) = this.store.images.get_mut(ds_id) {
            let mut ahead = imgs[0].clone();
            ahead.id = "img_aaa_new_page".into();
            ahead.file = "images/train/000000_aaa_first.jpg".into();
            imgs.push(ahead);
        }
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let idx = this.inspector.as_ref().unwrap().index;
        let shown_after = this.filtered_image(ds_id, idx).unwrap().id.clone();
        assert_eq!(shown_after, shown_before, "inspector keeps its image when a new page sorts ahead");
        assert!(idx > 0, "inspector index shifted to follow the same image");
    });
}

#[test]
fn inspector_toggles_click_to_zoom_and_try_in_playground() {
    use crate::api::PredictInput;
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api);
    let send_key = |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<super::FoloApp>, k: &str| {
        let keystroke = folo_ui::Keystroke::parse(k).unwrap();
        app.simulate_input(
            w.into(),
            folo_ui::PlatformInput::KeyDown(folo_ui::KeyDownEvent { keystroke, is_held: false }),
        );
        app.run_until_idle();
        app.frame(w.into());
    };
    let ds_id = "ds_8f2c";
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Dataset(ds_id.into()), w, cx));
    let _ = drawn_keys(&mut app, window);

    // Open an image that has at least one label box.
    let img_id = drive(&mut app, window, |this, w, cx| {
        let imgs = this.store.images.get(ds_id).and_then(|l| l.ready()).unwrap();
        let (idx, meta) = imgs.iter().enumerate().find(|(_, m)| !m.boxes.is_empty()).unwrap();
        let id = meta.id.clone();
        this.open_inspector(ds_id.into(), idx, w, cx);
        id
    });
    app.run_until_idle();
    app.frame(window.into());

    // B hides boxes; Ctrl+B does not; L while boxes are hidden restores both boxes and labels; L again hides labels.
    send_key(&mut app, window, "b");
    drive(&mut app, window, |this, _, _| {
        assert!(!this.inspector.as_ref().unwrap().boxes, "B hides boxes");
    });
    send_key(&mut app, window, "ctrl-b");
    drive(&mut app, window, |this, _, _| {
        assert!(!this.inspector.as_ref().unwrap().boxes, "Ctrl+B must not toggle inspector boxes");
    });
    send_key(&mut app, window, "l");
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert!(insp.boxes && insp.labels, "L with boxes hidden turns both boxes and labels on");
    });
    send_key(&mut app, window, "l");
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert!(insp.boxes && !insp.labels, "L toggles labels off while keeping boxes on");
    });

    // Activating a label row zooms onto that box; activating it a second time resets to fit.
    drive(&mut app, window, |this, w, cx| this.focus_key("insp-box-0", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert!(insp.zoom > 1.0, "activating label row zooms in");
        assert_eq!(insp.focused_box, Some(0));
    });
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        let insp = this.inspector.as_ref().unwrap();
        assert_eq!(insp.zoom, 1.0, "activating the same label row again fits back to 1x");
        assert_eq!(insp.pan, (0., 0.));
        assert_eq!(insp.focused_box, None);
    });

    // Activating Try in Playground closes the inspector, navigates to Playground, and loads this image.
    drive(&mut app, window, |this, w, cx| this.focus_key("insp-playground", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(this.inspector.is_none(), "inspector closed on Try in Playground");
        assert_eq!(this.route, Route::Playground);
        let input = &this.play_image.as_ref().expect("image loaded into Playground").input;
        assert!(
            matches!(input, PredictInput::Dataset { dataset_id, image_id } if dataset_id == ds_id && image_id == &img_id),
            "unexpected play input: {input:?}"
        );
    });
}
