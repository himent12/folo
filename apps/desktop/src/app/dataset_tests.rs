//! Datasets as a person meets them: a large one's page, one with no usable image, generating one in
//! the background.

use std::sync::Arc;

use super::ServerMode;
use super::data::IMAGE_PAGE;
use super::tests::{drive, open_with};
use crate::api::mock::MockBackend;
use crate::api::{Split, SplitsExt};
use crate::ui::Tone;

#[test]
fn a_dataset_page_holds_every_image_past_a_hundred_thousand() {
    // COCO-sized: the demo's shapes-1k with 100,000 more val images after its own.
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 100_000));
    let (mut app, window) = open_with(api);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    drive(&mut app, window, |this, _, _| {
        let ds = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).cloned().expect("loaded");
        let images = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).expect("listed");
        assert_eq!(images.len() as u64, ds.splits.total(), "the grid lists every image");
        this.ds_split = Some(Split::Val);
        assert_eq!(this.filtered_indices("ds_8f2c").len() as u64, ds.splits.val, "the last ones too");
    });
}

#[test]
fn a_large_list_shows_its_first_page_while_the_rest_arrives() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    let len = |this: &super::FoloApp| {
        this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map_or(0, |v| v.len())
    };
    // The first page lands; the second is held at the server.
    let (first, second) = (api.hold("images"), api.hold("images"));
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_dataset("ds_8f2c", cx)).unwrap());
    app.run_until_held(1);
    drop(first);
    // Until the first page is applied and the second asked for (bounded by the condition, not
    // a time: a call that never comes fails `run_until_held`).
    while !second.is_held() {
        app.run_until_held(1);
    }
    // (Plain updates while it is held: `drive` waits for every call.)
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                assert_eq!(len(this), IMAGE_PAGE as usize, "the grid has the first page");
                assert!(this.images_pending("ds_8f2c"), "and knows more is coming");
                // The list reloaded meanwhile (a change elsewhere): the old load's pages are
                // dropped, never appended twice.
                this.store.images.remove("ds_8f2c");
                this.ensure_dataset("ds_8f2c", cx);
            })
            .unwrap()
    });
    drop(second);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        assert_eq!(Some(len(this) as u64), total);
        assert!(!this.images_pending("ds_8f2c"));
    });
}

#[test]
fn a_page_that_fails_after_the_first_is_asked_for_again() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_once_from(IMAGE_PAGE);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    drive(&mut app, window, |this, _, _| {
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len());
        assert_eq!(got, Some(IMAGE_PAGE as usize));
        assert!(this.images_pending("ds_8f2c"), "still loading, not passed off as the whole list");
    });
    app.advance_clock(std::time::Duration::from_secs(5));
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len() as u64);
        assert_eq!(got, total);
        assert!(!this.images_pending("ds_8f2c"));
    });
}

#[test]
fn a_retry_ends_with_the_list_it_was_for() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_once_from(IMAGE_PAGE);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    // The dataset removed (here or elsewhere) while its page waits to be asked again.
    drive(&mut app, window, |this, _, _| {
        assert!(this.images_pending("ds_8f2c"));
        this.store.images.remove("ds_8f2c");
    });
    // Asked again, it would fail again (a removed dataset is a 404) and wait for another turn.
    api.fail_images_once_from(IMAGE_PAGE);
    app.advance_clock(std::time::Duration::from_secs(5));
    drive(&mut app, window, |this, _, _| {
        assert!(!this.images_pending("ds_8f2c"), "the retry ends instead of asking forever");
        assert!(!this.store.images.contains_key("ds_8f2c"), "nothing comes back for a dropped list");
    });
}

#[test]
fn a_sample_past_the_first_page_opens_on_itself() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    let last = drive(&mut app, window, |this, _, _| {
        this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total()).unwrap()
    });
    let want = format!("img_{last}");
    drive(&mut app, window, |this, _, _| this.store.images.remove("ds_8f2c"));
    // The first page in, the second held: the inspector waits for its image.
    let (first, second) = (api.hold("images"), api.hold("images"));
    app.update(|cx| {
        window
            .update(cx, |this, w, cx| this.open_sample("ds_8f2c".into(), want.clone(), "run_x".into(), w, cx))
            .unwrap()
    });
    app.run_until_held(1);
    drop(first);
    while !second.is_held() {
        app.run_until_held(1);
    }
    app.frame(window.into());
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                let i = this.inspector.as_ref().expect("still open");
                assert_eq!(i.want.as_deref(), Some(want.as_str()), "waiting, not on image 0");
            })
            .unwrap()
    });
    drop(second);
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let i = this.inspector.as_ref().expect("open");
        assert!(i.want.is_none());
        assert_eq!(this.filtered_image("ds_8f2c", i.index).map(|m| m.id.clone()), Some(want.clone()));
    });
}

#[test]
fn a_folder_with_no_usable_image_is_added_with_a_warning() {
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true)));
    drive(&mut app, window, |this, _, cx| {
        let mut ds = this.store.datasets.ready().unwrap()[0].clone();
        ds.id = "ds_0000".into();
        ds.name = "broken".into();
        ds.splits = Default::default();
        this.dataset_added(ds, "Added", cx);
        let t = this.toasts.last().expect("a toast");
        assert_eq!((t.tone, t.title.as_ref()), (Tone::Warning, "broken has no usable images"));
        assert!(t.action.is_some(), "with Open, to its page");
    });
}

#[test]
fn closing_generate_while_it_works_says_it_goes_on() {
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true)));
    let toasts = |app: &mut _| drive(app, window, |this, _, _| this.toasts.len());
    let before = toasts(&mut app);
    // Closed idle: nothing to say.
    drive(&mut app, window, |this, w, cx| {
        this.server_mode = ServerMode::Local;
        this.open_synth(w, cx);
        this.dismiss(w, cx);
    });
    assert_eq!(toasts(&mut app), before);
    // Esc while the server generates: the dialog goes, and a toast says the job goes on.
    drive(&mut app, window, |this, w, cx| {
        this.open_synth(w, cx);
        let f = this.synth.as_mut().unwrap();
        (f.submitting, f.sent_name) = (true, "shapes-sent".into());
        // Edited after Generate: the toast names what was sent.
        f.name.update(cx, |i, cx| i.set_text("other", cx));
        this.dismiss(w, cx);
        assert!(this.synth.is_none());
        let t = this.toasts.last().expect("a toast");
        assert_eq!(t.title.as_ref(), "Generating shapes-sent in the background");
    });
    assert_eq!(toasts(&mut app), before + 1);
}

#[test]
fn a_dataset_page_has_its_tabs_and_one_toolbar_and_statistics_holds_the_strip() {
    let (mut app, window) = super::tests::open();
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Dataset("ds_3b91".into()), w, cx));
    let images = drawn_keys(&mut app, window);
    // Images / Statistics as underline tabs; the split, the class picker (a select, not a chip per
    // class), Boxes, S / M / L (not a slider) and Fit / Fill on one toolbar.
    for k in ["ds-tab-0", "ds-tab-1", "ds-split", "ds-class", "ds-boxes", "ds-size", "ds-fit"] {
        assert!(images.iter().any(|x| x == k), "{k}: {images:?}");
    }
    assert!(!images.iter().any(|k| k == "ds-tile" || k.starts_with("ds-class-")), "{images:?}");
    // The empty-images and broken-labels tiles moved into Statistics with the rest of the strip.
    assert!(!images.iter().any(|k| k == "ds-empty-tile"), "{images:?}");
    drive(&mut app, window, |this, _, cx| {
        this.ds_tab = super::DsTab::Statistics;
        cx.notify();
    });
    let stats = drawn_keys(&mut app, window);
    assert!(stats.iter().any(|k| k == "ds-empty-tile"), "Show them stays: {stats:?}");
    // The health summary is a Tab stop (its focus tip lists every finding).
    assert!(stats.iter().any(|k| k == "ds-health"), "{stats:?}");
    // "Show them": the empty-images chip takes the class picker's place on the toolbar.
    drive(&mut app, window, |this, _, cx| {
        this.ds_tab = super::DsTab::Images;
        this.ds_empty_only = true;
        cx.notify();
    });
    let empty = drawn_keys(&mut app, window);
    assert!(empty.iter().any(|k| k == "ds-empty-chip"), "{empty:?}");
}

/// The focus keys a page draws, once what it asked for has come.
fn drawn_keys(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<super::FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

#[test]
fn the_health_summary_asks_for_the_check_even_when_every_label_parses() {
    let (mut app, window) = super::tests::open();
    // A dataset whose labels all parse: the check can still find unreadable images or warnings.
    let id = drive(&mut app, window, |this, _, _| {
        let ids: Vec<String> = this.store.datasets.ready().unwrap().iter().map(|d| d.id.clone()).collect();
        ids.into_iter().find(|id| id != "ds_3b91").unwrap()
    });
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Dataset(id.clone()), w, cx));
    drawn_keys(&mut app, window);
    drawn_keys(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let stats = this.store.dataset.get(&id).and_then(|l| l.ready()).and_then(|d| d.stats.clone());
        assert_eq!(stats.map(|s| s.broken_labels), Some(0), "{id}");
        assert!(this.store.issues.contains_key(&id), "the check's counts were asked for");
    });
}
