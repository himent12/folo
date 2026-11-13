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
