//! Datasets as a person meets them: a large one's page, one with no usable image, generating one in
//! the background.

use std::sync::Arc;

use super::tests::{drive, open_with};
use crate::api::mock::MockBackend;
use crate::api::{Split, SplitsExt};

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
