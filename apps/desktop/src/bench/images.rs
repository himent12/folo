//! Benchmark images: an entry's paged images, the Plan tab's previews and the image cache's keys.

use std::sync::Arc;

use folo_ui::Context;

use super::state::{ENTRY_IMAGES_PAGE, EntryImages};
use crate::app::FoloApp;
use crate::store::{ImgKey, ImgKind, ImgSlot, Load};

/// The severity a condition is previewed at: the middle of the ones the plan runs.
pub fn preview_severity(severities: &[u8]) -> u8 {
    let mut s = severities.to_vec();
    s.sort_unstable();
    s.get(s.len() / 2).copied().unwrap_or(3)
}

/// The image cache's key for a benchmark image: one "dataset" per benchmark, and the condition in the
/// image's key, so every variant has its own slot.
pub fn bench_img_key(bench: &str, image: &str, condition: Option<(&str, u8)>) -> (String, String) {
    let img = match condition {
        Some((c, s)) => format!("{image}@{c}:{s}"),
        None => image.to_string(),
    };
    (format!("bench:{bench}"), img)
}

impl FoloApp {
    /// An entry's first page of images in the chosen order (again when the order changes).
    pub fn ensure_entry_images(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        let key = (bench.to_string(), entry.to_string());
        let sort = self.bench.image_sort;
        if self.bench.entry_images.get(&key).is_some_and(|x| x.sort == sort && !x.page.needs_fetch()) {
            return;
        }
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        let state = EntryImages { sort, page: Load::Loading, more: false, more_error: None, seq };
        self.bench.entry_images.insert(key.clone(), state);
        let (b, e) = key.clone();
        self.fetch(
            cx,
            move |api| api.bench_entry_images(&b, &e, sort.as_str(), 0, ENTRY_IMAGES_PAGE),
            move |this, r, _| {
                if let Some(x) = this.bench.entry_images.get_mut(&key).filter(|x| x.seq == seq) {
                    x.page = r.into();
                }
            },
        );
    }

    /// The next page of an entry's images, after the ones there are.
    pub fn more_entry_images(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        let key = (bench.to_string(), entry.to_string());
        let Some(x) = self.bench.entry_images.get_mut(&key) else { return };
        let Some(page) = x.page.ready() else { return };
        let offset = page.images.len() as u32;
        if x.more || u64::from(offset) >= page.total {
            return;
        }
        x.more = true;
        x.more_error = None;
        let (sort, seq) = (x.sort, x.seq);
        let (b, e) = key.clone();
        self.fetch(
            cx,
            move |api| api.bench_entry_images(&b, &e, sort.as_str(), offset, ENTRY_IMAGES_PAGE),
            move |this, r, _| {
                let Some(x) = this.bench.entry_images.get_mut(&key).filter(|x| x.seq == seq) else { return };
                x.more = false;
                match (r, x.page.ready_mut()) {
                    (Ok(next), Some(page)) if page.images.len() as u32 == offset => {
                        page.images.extend(next.images);
                        page.total = next.total;
                    }
                    (Err(e), _) => x.more_error = Some(e),
                    _ => {}
                }
            },
        );
    }

    /// A benchmark's first image, the one the Plan tab shows under each condition.
    pub fn ensure_previews(&mut self, bench: &str, cx: &mut Context<Self>) {
        if self.bench.previews.get(bench).is_some_and(|l| !l.needs_fetch()) {
            return;
        }
        self.bench.previews.insert(bench.to_string(), Load::Loading);
        let (a, b) = (bench.to_string(), bench.to_string());
        self.fetch(
            cx,
            move |api| api.bench_images(&a, 0, 1, None),
            move |this, r, _| {
                this.bench.previews.insert(b, r.into());
            },
        );
    }

    /// A benchmark image (under a condition, or unchanged) as a thumbnail, decoded and cached with the
    /// app's other images; `None` until it is in.
    pub fn bench_image(
        &mut self,
        bench: &str,
        image: &str,
        condition: Option<(&str, u8)>,
        cx: &mut Context<Self>,
    ) -> Option<Arc<folo_ui::RenderImage>> {
        let (dataset, img) = bench_img_key(bench, image, condition);
        let key = ImgKey { dataset, image: img, kind: ImgKind::Thumb };
        match self.store.images_px.get(&key) {
            Some(ImgSlot::Ready(img)) => return Some(img.clone()),
            Some(_) => {
                self.store.image_queue.touch(&key);
                return None;
            }
            None => {}
        }
        if self.hold_loading || !self.store.images_px.begin(&key) {
            return None;
        }
        let (b, i, c) = (bench.to_string(), image.to_string(), condition.map(|(c, s)| (c.to_string(), s)));
        self.load_cached_image(
            key,
            move |api| api.bench_image(&b, &i, c.as_ref().map(|(c, s)| (c.as_str(), *s)), Some(320)),
            cx,
        );
        None
    }

    /// A benchmark image's thumbnail failed (it waits for its next automatic retry, or for a Retry
    /// by hand once those are spent): its frame says so instead of looking as if it still loads.
    pub fn bench_image_failed(&self, bench: &str, image: &str, condition: Option<(&str, u8)>) -> bool {
        let (dataset, image) = bench_img_key(bench, image, condition);
        self.store.images_px.failed(&ImgKey { dataset, image, kind: ImgKind::Thumb })
    }

    /// Retry by hand: the benchmark image's thumbnail is asked for again, its retries afresh.
    pub fn retry_bench_image(
        &mut self,
        bench: &str,
        image: &str,
        condition: Option<(&str, u8)>,
        cx: &mut Context<Self>,
    ) {
        let (dataset, image) = bench_img_key(bench, image, condition);
        self.store.images_px.retry(&ImgKey { dataset, image, kind: ImgKind::Thumb });
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previews_use_the_middle_planned_severity_and_their_own_cache_slot() {
        assert_eq!(preview_severity(&[5, 1, 3]), 3);
        assert_eq!(preview_severity(&[2, 4]), 4);
        assert_eq!(preview_severity(&[]), 3);
        let plain = bench_img_key("bn_1", "img_7", None);
        let hazy = bench_img_key("bn_1", "img_7", Some(("haze", 3)));
        assert_eq!(plain, ("bench:bn_1".to_string(), "img_7".to_string()));
        assert_ne!(plain, hazy);
        assert_ne!(hazy, bench_img_key("bn_1", "img_7", Some(("haze", 4))));
    }
}
