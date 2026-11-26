//! Dataset record loading, label-issue counts, and paged image-list fetching with bounded
//! automatic retries for transient failures and on-demand Retry for persistent page errors.

use std::time::Duration;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;

use super::super::FoloApp;

/// Image records asked for at a time when a dataset page loads its list (ten of the server's
/// pages): the grid paints after the first.
pub(crate) const IMAGE_PAGE: u32 = 10_000;
/// Base delay before automatically retrying a transiently failed later image page.
const IMAGE_PAGE_RETRY: Duration = Duration::from_secs(3);
/// Maximum automatic retries (with exponential backoff: 3s, 6s, 12s) for a transient failure on
/// a later image page before surfacing an inline Retry action.
pub(crate) const IMAGE_PAGE_MAX_RETRIES: u8 = 3;

/// A later image page that failed to load, stored alongside the already-loaded images so the grid
/// keeps what arrived and offers an inline Retry action instead of spinning forever.
#[derive(Clone, Debug)]
pub struct FailedImagePage {
    /// First image offset that failed to load.
    pub offset: u32,
    /// The backend error reported for that page.
    pub error: ApiError,
}

/// Place in a dataset's image grid saved across an external revision refresh and restored once the
/// batch carrying that file arrives. Keyed by `(file, split)` because image IDs are positional.
#[derive(Clone, Debug)]
pub struct DatasetScrollAnchor {
    pub dataset: String,
    pub file: String,
    pub split: Split,
    /// Row index captured with the anchor, used as a clamped fallback if the exact file was
    /// removed externally so large paged grids stay in the same region instead of snapping to row 0.
    pub fallback_row: usize,
    /// Pixel scroll offset within the anchor row (`0..row_h`) when captured from a scrolled grid.
    pub intra_row: f32,
    /// Pending `scroll_to_item` strategy (for example from closing the inspector right before a
    /// deferred recheck invalidated the image list).
    pub strategy: Option<folo_ui::ScrollStrategy>,
    /// True while this anchor is speculative (captured on `close_inspector` while waiting on the
    /// record recheck to decide whether `forget_dataset_contents` actually fires). Flipped to
    /// `false` once committed to an image reload, so a second unchanged recheck mid-paging never
    /// discards an active reload anchor.
    pub pending_recheck: bool,
    /// Focus handle held when closing the inspector: deferred tile focus only applies if no
    /// overlay is open and focus has not moved to another control since closing.
    pub focus_from: Option<folo_ui::FocusHandle>,
}

/// Whether a failed image-list page is likely transient (worth bounded automatic backoff) rather
/// than a permanent permission, missing-dataset, or client error.
pub(crate) fn is_transient_page_error(error: &ApiError) -> bool {
    if (error.code == "server_unreachable" && !error.is_bad_address()) || error.code == "timeout" {
        return true;
    }
    let status = error.status.or_else(|| {
        (error.code == "http_error")
            .then(|| {
                error.message.strip_prefix("folo serve answered HTTP ").and_then(|s| s.parse::<u16>().ok())
            })
            .flatten()
    });
    matches!(status, Some(408 | 429 | 502 | 503 | 504))
}

impl FoloApp {
    /// A dataset's issue counts, once (the Broken labels tile splits its count by severity). Only
    /// the latest check's answer is kept: one asked for before the dataset's files changed
    /// ([`FoloApp::forget_dataset_contents`]) counted the old files, and is dropped.
    pub fn ensure_issues(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.issues.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.issues.insert(id.to_string(), Load::Loading);
            self.issue_read_seq += 1;
            let ticket = self.issue_read_seq;
            self.issue_reads.insert(id.to_string(), ticket);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset_issues(&a).map(|p| p.counts),
                move |this, r, _| {
                    if this.issue_reads.get(&b) != Some(&ticket) {
                        return;
                    }
                    this.issue_reads.remove(&b);
                    this.store.issues.insert(b, r.into());
                },
            );
        }
    }

    /// A dataset's own record and every one of its images (its page).
    pub fn ensure_dataset(&mut self, id: &str, cx: &mut Context<Self>) {
        self.ensure_dataset_record(id, cx);
        if self.store.images.get(id).is_none_or(|l| l.needs_fetch()) {
            self.failed_pages.remove(id);
            self.store.images.insert(id.to_string(), Load::Loading);
            self.images_seq += 1;
            self.images_more.insert(id.to_string(), self.images_seq);
            self.load_images_from(id.to_string(), self.images_seq, 0, 0, cx);
        }
    }

    /// A dataset's own record (its stats, or what is wrong with it), without its image list: what
    /// a card needs besides its first images ([`FoloApp::ensure_dataset_card`]).
    pub(crate) fn ensure_dataset_record(&mut self, id: &str, cx: &mut Context<Self>) {
        // "Not found" once, listed now (its folder registered again, here or elsewhere): ask again,
        // its images too.
        let back = matches!(self.store.dataset.get(id), Some(Load::Failed(e)) if e.code == "dataset_not_found")
            && self.store.dataset_summary(id).is_some();
        if back {
            self.store.dataset.remove(id);
            if self.store.images.get(id).is_some_and(|l| l.error().is_some()) {
                self.store.images.remove(id);
            }
            self.failed_pages.remove(id);
            if self.store.previews.get(id).is_some_and(|l| l.error().is_some()) {
                self.store.previews.remove(id);
            }
        }
        // Read before and shown again on a page with cards (`ds_recheck`): read again in place, the
        // card keeping what it shows until the answer. Only an answer about the dataset replaces it
        // (a connection blip leaves a healthy card healthy); back from a failure, its images reload.
        // Never under an open inspector: the read waits until it closes ([`FoloApp::take_inspector`]).
        if !self.inspecting(id)
            && self.ds_recheck.remove(id)
            && self.store.dataset.get(id).is_some_and(|l| !l.is_loading())
        {
            let ticket = self.ds_read_ticket(id);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset(&a),
                move |this, r, cx| {
                    // A newer read was asked for since (another Retry or visit): its answer stands.
                    if !this.ds_read_done(&b, ticket) {
                        return;
                    }
                    // Cleared since (another server): this answer is about what was there.
                    if this.store.dataset.get(&b).is_none_or(|l| l.is_loading()) {
                        return;
                    }
                    match r {
                        Ok(d) => {
                            // Back from a failure, or its contents changed: the images, their labels and
                            // their pixels are read again (image ids are positions, docs/API.md).
                            if this.store.dataset.get(&b).and_then(|l| l.ready()) != Some(&d) {
                                // An inspector opened on it since: all stays, read again when it closes.
                                if this.inspecting(&b) {
                                    this.ds_recheck.insert(b);
                                    return;
                                }
                                this.forget_dataset_contents(&b, cx);
                            } else if this
                                .ds_scroll_anchor
                                .as_ref()
                                .is_some_and(|a| a.dataset == b && a.pending_recheck)
                            {
                                // Record unchanged: no image reload will run, so drop only a
                                // speculative anchor captured on inspector close (never an anchor
                                // already committed to an in-progress paged reload).
                                this.ds_scroll_anchor = None;
                            }
                            this.store.dataset.insert(b, Load::Ready(d));
                        }
                        Err(e)
                            if e.code == "dataset_not_found"
                                || crate::screens::dataset_problem(&e).is_some() =>
                        {
                            // The same under an open inspector: its record and list stay until it closes.
                            if this.inspecting(&b) {
                                this.ds_recheck.insert(b);
                                return;
                            }
                            if this.ds_scroll_anchor.as_ref().is_some_and(|a| a.dataset == b) {
                                this.ds_scroll_anchor = None;
                            }
                            this.store.dataset.insert(b, Load::Failed(e));
                        }
                        Err(_) => {
                            if this
                                .ds_scroll_anchor
                                .as_ref()
                                .is_some_and(|a| a.dataset == b && a.pending_recheck)
                            {
                                this.ds_scroll_anchor = None;
                            }
                        }
                    }
                },
            );
        }
        if self.store.dataset.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.dataset.insert(id.to_string(), Load::Loading);
            let ticket = self.ds_read_ticket(id);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset(&a),
                move |this, r, _| {
                    if this.ds_read_done(&b, ticket) {
                        this.store.dataset.insert(b, r.into());
                    }
                },
            );
        }
    }

    /// A number for a new read of dataset `id`, the latest from now on ([`FoloApp::ds_read_done`]).
    fn ds_read_ticket(&mut self, id: &str) -> u64 {
        self.ds_read_seq += 1;
        self.ds_reads.insert(id.to_string(), self.ds_read_seq);
        self.ds_read_seq
    }

    /// Whether the read numbered `ticket` is still dataset `id`'s latest: its answer may be applied
    /// (and nothing older will be). An older one is dropped.
    fn ds_read_done(&mut self, id: &str, ticket: u64) -> bool {
        if self.ds_reads.get(id) != Some(&ticket) {
            return false;
        }
        self.ds_reads.remove(id);
        true
    }

    /// Drops what the app holds of a dataset's files (its image list and the page of it still on its
    /// way, its card's first images, its label issues, its decoded images, the grid's filter): read
    /// again when shown.
    pub(crate) fn forget_dataset_contents(&mut self, id: &str, cx: &mut Context<Self>) {
        self.capture_dataset_scroll_anchor(id);
        self.store.images.remove(id);
        self.images_more.remove(id);
        self.failed_pages.remove(id);
        self.store.previews.remove(id);
        self.store.issues.remove(id);
        self.issue_reads.remove(id);
        *self.ds_filter_cache.borrow_mut() = None;
        *self.ds_split_counts.borrow_mut() = None;
        let mut textures = self.store.images_px.forget_dataset(id);
        // The Playground's image from it too: read again, its old pixels never under new boxes.
        textures.extend(self.forget_play_dataset(id, cx));
        // Freed after this update, when no window is mid-draw (as on a server switch).
        cx.defer(move |cx| {
            for img in textures {
                cx.drop_image(img, None);
            }
        });
    }

    /// Every image of a dataset (the grid filters by split and class here, so a list cut short
    /// would lose the end of train and all of val: COCO has 123k), a page of
    /// [`IMAGE_PAGE`] at a time: the grid shows the first page while the rest arrives.
    /// `generation` is this load's: a page of a load since replaced (the list refetched or
    /// dropped) is not appended, and ends that load.
    fn load_images_from(
        &mut self,
        id: String,
        generation: u64,
        offset: u32,
        attempt: u8,
        cx: &mut Context<Self>,
    ) {
        let a = id.clone();
        self.fetch(
            cx,
            move |api| api.images(&a, &ImageQuery { offset, limit: IMAGE_PAGE, ..Default::default() }),
            move |this, r, cx| {
                if this.images_more.get(&id) != Some(&generation) {
                    return;
                }
                let page = match r {
                    Ok(p) => p,
                    // A failed first page is the list's error.
                    Err(e) if offset == 0 => {
                        this.images_more.remove(&id);
                        this.failed_pages.remove(&id);
                        if this.ds_scroll_anchor.as_ref().is_some_and(|a| a.dataset == id) {
                            this.ds_scroll_anchor = None;
                        }
                        this.store.images.insert(id, Load::Failed(e));
                        return;
                    }
                    Err(e) => {
                        // A removed or reloaded list no longer owns this offset: drop the error
                        // instead of recording a stale failed page or scheduling another retry.
                        let aligned = this
                            .store
                            .images
                            .get(&id)
                            .and_then(|l| l.ready())
                            .is_some_and(|l| l.len() == offset as usize);
                        if !aligned {
                            this.images_more.remove(&id);
                            this.failed_pages.remove(&id);
                            return;
                        }
                        // Transient later-page failures (connection drop / timeout / 502-504 / 429)
                        // retry a bounded number of times with exponential backoff while keeping
                        // loaded pages visible.
                        if is_transient_page_error(&e) && attempt < IMAGE_PAGE_MAX_RETRIES {
                            this.schedule_image_page_retry(id, generation, offset, attempt, cx);
                            return;
                        }
                        // Permanent errors (permission denied, missing dataset, HTTP 4xx/500) or
                        // exhausted transient retries stop spinning and surface an inline Retry action.
                        this.images_more.remove(&id);
                        this.failed_pages.insert(id, FailedImagePage { offset, error: e });
                        cx.notify();
                        return;
                    }
                };
                let list = match this.store.images.get_mut(&id) {
                    Some(Load::Ready(l)) if l.len() == offset as usize => l,
                    Some(Load::Loading) if offset == 0 => {
                        this.store.images.insert(id.clone(), Load::Ready(Vec::new()));
                        this.store.images.get_mut(&id).and_then(|l| l.ready_mut()).expect("just set")
                    }
                    _ => {
                        this.images_more.remove(&id);
                        return;
                    }
                };
                let got = page.images.len() as u32;
                list.extend(page.images);
                let next = offset + got;
                this.failed_pages.remove(&id);
                // New list, new truth: nothing derived from the old one may survive.
                *this.ds_filter_cache.borrow_mut() = None;
                *this.ds_split_counts.borrow_mut() = None;
                if got > 0 && u64::from(next) < page.total {
                    this.load_images_from(id, generation, next, 0, cx);
                } else {
                    this.images_more.remove(&id);
                }
                cx.notify();
            },
        );
    }

    /// Schedules one exponentially backed-off retry for `offset`. If the dataset carries a
    /// server revision, revalidates it before resuming so files edited during backoff never splice
    /// records from two revisions together.
    fn schedule_image_page_retry(
        &mut self,
        id: String,
        generation: u64,
        offset: u32,
        attempt: u8,
        cx: &mut Context<Self>,
    ) {
        let backend = self.backend_gen;
        let wait = IMAGE_PAGE_RETRY * (1u32 << attempt);
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(wait).await;
            this.update(cx, |this, cx| {
                // Only the same load, on the same server, of a list still aligned.
                if this.images_more.get(&id) != Some(&generation) {
                    return;
                }
                let still_aligned = this
                    .store
                    .images
                    .get(&id)
                    .and_then(|l| l.ready())
                    .is_some_and(|l| l.len() == offset as usize);
                if this.backend_gen != backend || !still_aligned {
                    this.images_more.remove(&id);
                    return;
                }
                let cached = this.store.dataset.get(&id).and_then(|l| l.ready()).cloned();
                if let Some(cached) = cached
                    && cached.revision.is_some()
                {
                    this.revalidate_and_load_images_from(id, generation, offset, attempt + 1, cached, cx);
                } else {
                    this.load_images_from(id, generation, offset, attempt + 1, cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Revalidates `cached`'s revision via `api.dataset(&id)` before requesting `offset`. If the
    /// revision changed (or disappeared), invalidates cached dataset contents and restarts from
    /// page zero.
    fn revalidate_and_load_images_from(
        &mut self,
        id: String,
        generation: u64,
        offset: u32,
        attempt: u8,
        cached: Dataset,
        cx: &mut Context<Self>,
    ) {
        let req_id = id.clone();
        self.fetch(
            cx,
            move |api| api.dataset(&req_id),
            move |this, r, cx| {
                if this.images_more.get(&id) != Some(&generation) {
                    return;
                }
                let still_aligned = this
                    .store
                    .images
                    .get(&id)
                    .and_then(|l| l.ready())
                    .is_some_and(|l| l.len() == offset as usize);
                match r {
                    Ok(fresh)
                        if still_aligned
                            && fresh.revision.is_some()
                            && fresh.revision == cached.revision
                            && fresh == cached =>
                    {
                        this.store.dataset.insert(id.clone(), Load::Ready(fresh));
                        this.load_images_from(id, generation, offset, attempt, cx);
                    }
                    Ok(fresh) => {
                        // Files changed on disk or the server dropped revision tracking:
                        // discard stale prefix + decoded images and reload from page zero.
                        this.forget_dataset_contents(&id, cx);
                        this.store.dataset.insert(id.clone(), Load::Ready(fresh));
                        this.ensure_dataset(&id, cx);
                    }
                    Err(e)
                        if still_aligned
                            && is_transient_page_error(&e)
                            && attempt < IMAGE_PAGE_MAX_RETRIES =>
                    {
                        this.schedule_image_page_retry(id, generation, offset, attempt, cx);
                    }
                    Err(e) => {
                        this.images_more.remove(&id);
                        if still_aligned {
                            this.failed_pages.insert(id, FailedImagePage { offset, error: e });
                        }
                    }
                }
                cx.notify();
            },
        );
    }

    /// Cached later-page failure for dataset `id`, if paging stopped after partial images loaded.
    pub fn failed_image_page(&self, id: &str) -> Option<&FailedImagePage> {
        self.failed_pages.get(id)
    }

    /// Clears a cached failed image page, revalidates the dataset revision before resuming from
    /// that offset, and restarts from page zero (invalidating stale cached images) if the revision
    /// changed or cannot be verified.
    pub fn retry_failed_image_page(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(failed) = self.failed_pages.remove(id) else {
            return;
        };
        let aligned = self.store.images.get(id).and_then(|l| l.ready()).map(|l| l.len())
            == Some(failed.offset as usize);
        let cached = self.store.dataset.get(id).and_then(|l| l.ready()).cloned();
        if aligned
            && let Some(cached) = cached
            && cached.revision.is_some()
        {
            self.images_seq += 1;
            let generation = self.images_seq;
            self.images_more.insert(id.to_string(), generation);
            self.revalidate_and_load_images_from(id.to_string(), generation, failed.offset, 0, cached, cx);
        } else {
            // Revision cannot be verified or list is no longer aligned: drop cached contents and
            // reload from page zero so positional image IDs never mix across dataset revisions.
            self.forget_dataset_contents(id, cx);
            self.ensure_dataset(id, cx);
        }
        cx.notify();
    }

    /// Whether an inspector (the grid's or a run sample's) is open on dataset `id`: its list stays.
    pub(crate) fn inspecting(&self, id: &str) -> bool {
        self.inspector.as_ref().is_some_and(|i| i.dataset == id)
    }

    /// Whether more of a dataset's image list is still on its way (the grid shows what came).
    pub fn images_pending(&self, id: &str) -> bool {
        self.images_more.contains_key(id)
    }

    /// Saves the user's visible place in dataset `id`'s image grid before a revision reload drops
    /// `store.images`, so paged reloads return to the same file and offset once its batch arrives.
    fn capture_dataset_scroll_anchor(&mut self, id: &str) {
        if !matches!(&self.route, super::super::Route::Dataset(cur) if cur == id)
            || self.ds_tab != super::super::DsTab::Images
        {
            return;
        }
        // Keep an anchor captured before an earlier in-flight reload (or on inspector close),
        // committing any speculative anchor to this active image reload.
        if let Some(existing) = self.ds_scroll_anchor.as_mut().filter(|a| a.dataset == id) {
            existing.pending_recheck = false;
            return;
        }
        let (deferred, scroll_y, item_h) = {
            let state = self.grid_scroll.0.borrow();
            (
                state.deferred_scroll_to_item,
                f32::from(-state.base_handle.offset().y).max(0.),
                state.last_item_height.map(f32::from),
            )
        };
        let (target_row, intra_row, strategy) = if let Some((row, strategy)) = deferred {
            if row == 0 && scroll_y <= 0. {
                return;
            }
            (row, 0., Some(strategy))
        } else {
            let Some(item_h) = item_h.filter(|&h| h > 0.) else { return };
            if scroll_y <= 0. {
                return;
            }
            let row = (scroll_y / item_h).floor().max(0.) as usize;
            let rem = (scroll_y - row as f32 * item_h).clamp(0., item_h);
            (row, rem, None)
        };
        let cols = self.grid_cols();
        let indices = self.filtered_indices(id);
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else { return };
        let Some(&raw_idx) = indices.get(target_row * cols).or_else(|| indices.last()) else { return };
        let Some(meta) = all.get(raw_idx) else { return };
        self.ds_scroll_anchor = Some(DatasetScrollAnchor {
            dataset: id.to_string(),
            file: meta.file.clone(),
            split: meta.split,
            fallback_row: target_row,
            intra_row,
            strategy,
            pending_recheck: false,
            focus_from: None,
        });
    }

    /// Restores a saved dataset grid scroll anchor once the image batch carrying `(file, split)`
    /// has arrived. Keeps waiting across earlier batches while `more` is true or a retryable later
    /// page failed, keeps re-anchoring through the final batch under non-default sorts that can
    /// move rows as later pages arrive, and clears once the final position is applied (or when a
    /// completed load no longer contains the file).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_restore_dataset_scroll_anchor(
        &mut self,
        id: &str,
        cols: usize,
        row_h: f32,
        more: bool,
        has_failed_page: bool,
        window: &mut folo_ui::Window,
        cx: &mut Context<Self>,
    ) {
        let Some(anchor) = self.ds_scroll_anchor.as_ref().filter(|a| a.dataset == id).cloned() else {
            return;
        };
        // Wait until a speculative inspector-close anchor is either committed by
        // `forget_dataset_contents` or cleared by an unchanged/failed recheck.
        if anchor.pending_recheck {
            return;
        }
        let still_incomplete = more || has_failed_page;
        let indices = self.filtered_indices(id);
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else {
            if !still_incomplete {
                self.ds_scroll_anchor = None;
            }
            return;
        };
        let found = indices.iter().enumerate().find_map(|(pos, &raw)| {
            let meta = all.get(raw)?;
            (meta.split == anchor.split && meta.file == anchor.file).then(|| (pos, meta.id.clone()))
        });
        let Some((pos, tile_id)) = found else {
            if !still_incomplete {
                if !indices.is_empty() && (anchor.fallback_row > 0 || anchor.strategy.is_some()) {
                    let rows = indices.len().div_ceil(cols);
                    let row = anchor.fallback_row.min(rows - 1);
                    if let Some(strategy) = anchor.strategy {
                        self.grid_scroll.scroll_to_item(row, strategy);
                    } else {
                        let clamped_intra = anchor.intra_row.min((row_h - 1.).max(0.));
                        let target_y = row as f32 * row_h + clamped_intra;
                        let mut scroll = self.grid_scroll.0.borrow_mut();
                        scroll.deferred_scroll_to_item = None;
                        scroll
                            .base_handle
                            .set_offset(folo_ui::point(folo_ui::px(0.), -folo_ui::px(target_y)));
                    }
                }
                self.ds_scroll_anchor = None;
            }
            return;
        };
        let row = pos / cols;
        let rows = indices.len().div_ceil(cols);
        let content_h = row_h * rows as f32;
        let viewport_h = f32::from(self.grid_scroll.0.borrow().base_handle.bounds().size.height);
        let default_sort = self.img_sort(id) == crate::screens::datasets::ImageSort::Default;
        // Under non-default sorts (name / box count), later pages can sort ahead of the anchor and
        // shift its row index, so only finish early while `more` when using Default order.
        let can_finish = if let Some(strategy) = anchor.strategy {
            self.grid_scroll.scroll_to_item(row, strategy);
            !still_incomplete
                || (default_sort && viewport_h > 0. && (row + 1) as f32 * row_h + viewport_h <= content_h)
        } else {
            let clamped_intra = anchor.intra_row.min((row_h - 1.).max(0.));
            let target_y = row as f32 * row_h + clamped_intra;
            {
                let mut scroll = self.grid_scroll.0.borrow_mut();
                scroll.deferred_scroll_to_item = None;
                scroll.base_handle.set_offset(folo_ui::point(folo_ui::px(0.), -folo_ui::px(target_y)));
            }
            !still_incomplete || (default_sort && viewport_h > 0. && target_y + viewport_h <= content_h)
        };
        if can_finish {
            if let Some(prev) = anchor.focus_from.as_ref()
                && !self.any_overlay()
                && (prev.is_focused(window) || self.root_focus.is_focused(window))
            {
                let fh = self.fh(format!("tile-{tile_id}"), cx);
                window.focus(&fh);
            }
            self.ds_scroll_anchor = None;
        }
    }
}
