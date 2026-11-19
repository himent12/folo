//! A dataset card's images: the first few of its list, not the whole of it (COCO's is 123k records,
//! about 70 MB), read when a card shows it. Opening the dataset reads every image
//! ([`FoloApp::ensure_dataset`]); a list read that way, or still arriving, serves the card from then on.

use folo_ui::Context;

use super::FoloApp;
use crate::api::{Dataset, ImageInfo, ImageQuery, SplitsExt};
use crate::store::Load;

/// The images a card reads of a larger dataset: the four with boxes it shows, with room for a few
/// without any.
pub(crate) const PREVIEW_PAGE: u32 = 32;
/// A dataset up to this size has its whole list read for its card instead: one request either way
/// (the most a server's page holds, docs/API.md), and its page then opens on its full grid.
pub(crate) const CARD_WHOLE_LIST: u64 = 1_000;

impl FoloApp {
    /// What dataset `d`'s card shows: its own record (the class balance, what is wrong with it) and
    /// its first images.
    pub fn ensure_dataset_card(&mut self, d: &Dataset, cx: &mut Context<Self>) {
        if d.splits.total() <= CARD_WHOLE_LIST {
            self.ensure_dataset(&d.id, cx);
            return;
        }
        self.ensure_dataset_record(&d.id, cx);
        if self.store.images.contains_key(&d.id)
            || self.store.previews.get(&d.id).is_some_and(|l| !l.needs_fetch())
        {
            return;
        }
        self.store.previews.insert(d.id.clone(), Load::Loading);
        self.images_seq += 1;
        let ticket = self.images_seq;
        self.preview_reads.insert(d.id.clone(), ticket);
        let (a, b) = (d.id.clone(), d.id.clone());
        self.fetch(
            cx,
            move |api| {
                api.images(&a, &ImageQuery { limit: PREVIEW_PAGE, ..Default::default() }).map(|p| p.images)
            },
            move |this, r, _| {
                // Only the latest read, of a preview still wanted: one dropped since (the dataset's
                // files changed, another server) or asked for again is about what was there.
                if this.preview_reads.get(&b) != Some(&ticket) {
                    return;
                }
                this.preview_reads.remove(&b);
                if this.store.previews.get(&b).is_some_and(|l| l.is_loading()) {
                    this.store.previews.insert(b, r.into());
                }
            },
        );
    }

    /// The first `n` images with boxes of dataset `id` that a card shows: from its full list when
    /// that is read (or arriving), else from the card's own read.
    pub fn card_images(&self, id: &str, n: usize) -> Vec<ImageInfo> {
        let held = self.store.images.get(id).and_then(|l| l.ready());
        held.or_else(|| self.store.previews.get(id).and_then(|l| l.ready()))
            .map(|v| v.iter().filter(|m| !m.boxes.is_empty()).take(n).cloned().collect())
            .unwrap_or_default()
    }
}
