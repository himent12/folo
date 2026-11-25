//! Decoded image fetching, queue scheduling, and bounded per-image retry timers.

use std::sync::Arc;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;

use super::super::FoloApp;

impl FoloApp {
    /// A decoded image, requesting it if needed.
    pub fn image(
        &mut self,
        dataset: &str,
        image: &str,
        kind: ImgKind,
        cx: &mut Context<Self>,
    ) -> Option<Arc<folo_ui::RenderImage>> {
        let key = ImgKey { dataset: dataset.to_string(), image: image.to_string(), kind };
        match self.store.images_px.get(&key) {
            Some(ImgSlot::Ready(img)) => return Some(img.clone()),
            Some(_) => {
                // Drawn again while loading (a grid scrolled back): ahead of the ones queued after it.
                self.store.image_queue.touch(&key);
                return None;
            }
            None => {}
        }
        if self.hold_loading || !self.store.images_px.begin(&key) {
            return None;
        }
        let thumb = (kind == ImgKind::Thumb).then_some(320);
        let (ds, im) = (key.dataset.clone(), key.image.clone());
        self.load_cached_image(key, move |api| api.image(&ds, &im, thumb), cx);
        None
    }

    /// Whether a dataset image failed at that size (its retries may be spent): the inspector then
    /// says so and offers [`Self::retry_failed_image`].
    pub fn image_failed(&self, dataset: &str, image: &str, kind: ImgKind) -> bool {
        let key = ImgKey { dataset: dataset.to_string(), image: image.to_string(), kind };
        self.store.images_px.failed(&key)
    }

    /// Retry by hand: the image (both sizes) is asked for again, its automatic retries afresh.
    pub fn retry_failed_image(&mut self, dataset: &str, image: &str, cx: &mut Context<Self>) {
        for kind in [ImgKind::Full, ImgKind::Thumb] {
            let key = ImgKey { dataset: dataset.to_string(), image: image.to_string(), kind };
            self.store.images_px.retry(&key);
        }
        cx.notify();
    }

    /// Fetches and decodes an image off the UI thread into the slot `key` (already marked loading),
    /// dropping it if the data source changed meanwhile. Queued: a few run at a time, the newest
    /// first, and one evicted from the cache before its turn is never fetched (`LoadQueue`).
    pub fn load_cached_image(
        &mut self,
        key: ImgKey,
        call: impl FnOnce(&dyn Backend) -> ApiResult<ImageData> + Send + 'static,
        cx: &mut Context<Self>,
    ) {
        self.store.image_queue.push(key, Box::new(call));
        self.start_queued_images(cx);
    }

    /// Starts queued image fetches while there is room for them.
    fn start_queued_images(&mut self, cx: &mut Context<Self>) {
        loop {
            let store = &mut self.store;
            let next =
                store.image_queue.start_next(|k| matches!(store.images_px.get(k), Some(ImgSlot::Loading)));
            let Some((key, call)) = next else { return };
            let api = self.store.api.clone();
            let generation = self.backend_gen;
            let attempt = self.store.images_px.attempt(&key);
            // Full-size images are bounded too: a 12000 px photo would otherwise hold 400 MB.
            let max_side = (key.kind == ImgKind::Full).then_some(ImageCache::FULL_MAX_SIDE);
            cx.spawn(async move |this, cx| {
                let (img, err) = cx
                    .background_executor()
                    .spawn(async move {
                        match call(&*api) {
                            Ok(d) => (to_render_image_within(d, max_side), None),
                            Err(e) => (None, Some(e)),
                        }
                    })
                    .await;
                this.update(cx, |this, cx| {
                    if this.backend_gen != generation {
                        if let Some(img) = img {
                            cx.drop_image(img, None);
                        }
                        return;
                    }
                    this.store.image_queue.done();
                    // An image that couldn't reach the server has it asked at once whether it is there:
                    // an outage that confirms is noticed (`reconnected` asks for the failed images again)
                    // even when only images saw it. The image alone doesn't mark it down: one that times
                    // out on a server that answers would have its retries reset at every next answer.
                    if err.is_some_and(|e| e.code == "server_unreachable") && !this.store.unreachable() {
                        this.refresh_system(cx);
                        this.ensure_core(cx);
                    }
                    // A slot evicted while its fetch was on its way (scrolled far past) has dropped
                    // it; drawn again since, it is on a newer fetch, whose answer is the one it gets.
                    let current = attempt.is_some() && this.store.images_px.attempt(&key) == attempt;
                    if current {
                        let failed = img.is_none();
                        let evicted = this.store.images_px.finish(key.clone(), img);
                        for e in evicted {
                            cx.drop_image(e, None);
                        }
                        // Any other failure (a body cut off, a file being rewritten) is asked for again
                        // a few times on its own: nothing else would, as no outage was seen.
                        if failed && let Some((wait, token)) = this.store.images_px.retry_delay(&key) {
                            this.retry_image(key, wait, token, cx);
                        }
                    } else if let Some(img) = img {
                        cx.drop_image(img, None);
                    }
                    this.start_queued_images(cx);
                    if this.store.image_queue.in_flight() == 0 {
                        // `FOLO_UI_TRACE`: the first time the queue ran dry, every image asked for answered (an
                        // error included: see `trace_first_data`).
                        static DONE: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            folo_ui::trace_point("images-drained");
                            folo_ui::trace_at_next_frame("images-drained-frame");
                        }
                    }
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
    }

    /// After `wait`, forgets the failed image `key` so the next draw asks for it again, unless the
    /// failure `token` names was answered meanwhile (a reconnect, a Retry, a newer failure).
    fn retry_image(&mut self, key: ImgKey, wait: std::time::Duration, token: u64, cx: &mut Context<Self>) {
        let generation = self.backend_gen;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(wait).await;
            this.update(cx, |this, cx| {
                if this.backend_gen == generation {
                    this.store.images_px.forget(&key, token);
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }
}
