//! A dataset card's images: the first page of its list, not the whole of it (COCO's is 123k
//! records, about 70 MB), read when a card shows it, whatever the dataset's size. Opening the
//! dataset reads every image ([`FoloApp::ensure_dataset`]); a list read that way, or still
//! arriving, serves the card from then on.

use folo_ui::Context;

use super::FoloApp;
use crate::api::{Dataset, ImageInfo, ImageQuery};
use crate::store::Load;

/// The images a card reads: the four it shows, those with boxes first, with room for a few
/// without any.
pub(crate) const PREVIEW_PAGE: u32 = 32;

impl FoloApp {
    /// What dataset `d`'s card shows: its own record (the class balance, what is wrong with it) and
    /// its first images.
    pub fn ensure_dataset_card(&mut self, d: &Dataset, cx: &mut Context<Self>) {
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

    /// The first `n` images of dataset `id` that a card shows, those with boxes first, then the rest
    /// (a card is never empty when the dataset has images): from its full list when that is read
    /// (or arriving), else from the card's own read.
    pub fn card_images(&self, id: &str, n: usize) -> Vec<ImageInfo> {
        let held = self.store.images.get(id).and_then(|l| l.ready());
        let Some(v) = held.or_else(|| self.store.previews.get(id).and_then(|l| l.ready())) else {
            return Vec::new();
        };
        let boxed = v.iter().filter(|m| !m.boxes.is_empty());
        boxed.chain(v.iter().filter(|m| m.boxes.is_empty())).take(n).cloned().collect()
    }
}
