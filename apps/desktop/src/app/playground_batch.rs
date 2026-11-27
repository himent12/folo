//! Playground multi-image / folder batch loading, navigation, sequential Run All with strict
//! generation/cancellation guards, and Batch JSON / Batch CSV export.

use std::path::PathBuf;
use std::sync::Arc;

use folo_ui::{Context, Window};

use crate::api::*;
use crate::screens::playground::{
    BatchItemResult, BatchItemStatus, MAX_BATCH_FILES, PlayBatchItem, PlayBatchSource, PlayPredMeta,
    bytes_fingerprint, disambiguate_file_labels, is_supported_image_ext, pick_val_samples, scan_image_folder,
    write_export_bytes,
};
use crate::store::{ImgKey, ImgKind, Load, decode_upright, to_render_image};
use crate::ui::*;

use super::playground::{OpenRequest, PLAY_CONF_FLOOR};
use super::playground_compare::{export_default_dir, picked_save};
use super::{FoloApp, PlayImage, Route, act};

/// Decodes `bytes` upright into a GPU-ready `RenderImage` with `(width, height)`.
fn decode_render(bytes: &[u8]) -> Option<(Arc<folo_ui::RenderImage>, u32, u32)> {
    let img = decode_upright(bytes)?;
    let (w, h) = img.dimensions();
    let image = to_render_image(ImageData::Rgba { width: w, height: h, pixels: img.into_raw() })?;
    Some((image, w, h))
}

/// Collects all paths from a multi-file or folder picker; `None` when cancelled.
pub(super) fn picked_paths(
    this: &folo_ui::WeakEntity<FoloApp>,
    answer: Result<anyhow::Result<Option<Vec<PathBuf>>>, folo_ui::oneshot::Canceled>,
    cx: &mut folo_ui::AsyncApp,
) -> Option<Vec<PathBuf>> {
    match answer {
        Ok(Ok(paths)) => paths.filter(|v| !v.is_empty()),
        Ok(Err(e)) => {
            this.update(cx, |this, cx| {
                this.toast(Tone::Danger, "The file dialog didn't open", Some(e.to_string().into()), cx)
            })
            .ok();
            None
        }
        Err(_) => None,
    }
}

impl FoloApp {
    /// Prompts for a folder of images to load into Playground as a batch.
    pub fn pick_play_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open a folder yet", Some(why.into()), cx);
            return;
        }
        let req = self.open_request();
        let handle = window.window_handle();
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Open folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Some(paths) = picked_paths(&this, rx.await, cx)
                && let Some(dir) = paths.into_iter().next()
            {
                handle
                    .update(cx, |_, window, cx| {
                        this.update(cx, |this, cx| this.open_play_folder(dir, req, window, cx))
                    })
                    .ok();
            }
        })
        .detach();
    }

    /// Scans `dir` off the UI thread and loads its supported images into Playground (single image
    /// if 1 file, batch filmstrip if 2+ files).
    pub(super) fn open_play_folder(
        &mut self,
        dir: PathBuf,
        req: OpenRequest,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if req.ticket != self.play_open_seq {
            return;
        }
        self.play_batch.load_gen += 1;
        let load_gen = self.play_batch.load_gen;
        let handle = window.window_handle();
        cx.spawn(async move |this, cx| {
            let scanned = cx.background_executor().spawn(async move { scan_image_folder(&dir) }).await;
            handle
                .update(cx, |_, window, cx| {
                    this.update(cx, |this, cx| {
                        if this.play_batch.load_gen != load_gen || req.ticket != this.play_open_seq {
                            return;
                        }
                        match scanned {
                            Ok((paths, truncated)) => {
                                if paths.is_empty() {
                                    this.toast(
                                        Tone::Neutral,
                                        "No supported images in that folder",
                                        Some(format!("Use {} files.", super::IMAGE_FORMATS).into()),
                                        cx,
                                    );
                                    return;
                                }
                                this.apply_file_batch(paths, truncated, req, window, cx);
                            }
                            Err(e) => {
                                this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx);
                            }
                        }
                    })
                })
                .ok();
        })
        .detach();
    }

    /// Opens one or more picked/dropped file or folder paths in Playground.
    /// A single file opens in single-image mode; a folder scans its images; 2+ images load a batch.
    pub(super) fn open_play_paths(
        &mut self,
        paths: Vec<PathBuf>,
        req: OpenRequest,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if paths.is_empty() || req.ticket != self.play_open_seq {
            return;
        }
        if paths.len() == 1 {
            let p = paths.into_iter().next().unwrap();
            if p.is_dir() {
                self.open_play_folder(p, req, window, cx);
            } else {
                self.open_play_file(p, req, window, cx);
            }
            return;
        }
        // Expand any dropped directories off the UI thread while preserving dropped files.
        let has_dir = paths.iter().any(|p| p.is_dir());
        if has_dir {
            self.play_batch.load_gen += 1;
            let load_gen = self.play_batch.load_gen;
            let handle = window.window_handle();
            cx.spawn(async move |this, cx| {
                let expanded = cx
                    .background_executor()
                    .spawn(async move {
                        let mut out = Vec::new();
                        let mut truncated = 0usize;
                        for p in paths {
                            if p.is_dir() {
                                if let Ok((dir_paths, t)) = scan_image_folder(&p) {
                                    out.extend(dir_paths);
                                    truncated += t;
                                }
                            } else if is_supported_image_ext(&p) {
                                out.push(p);
                            }
                        }
                        if out.len() > MAX_BATCH_FILES {
                            truncated += out.len() - MAX_BATCH_FILES;
                            out.truncate(MAX_BATCH_FILES);
                        }
                        (out, truncated)
                    })
                    .await;
                handle
                    .update(cx, |_, window, cx| {
                        this.update(cx, |this, cx| {
                            if this.play_batch.load_gen != load_gen || req.ticket != this.play_open_seq {
                                return;
                            }
                            let (files, truncated) = expanded;
                            if files.is_empty() {
                                this.toast(
                                    Tone::Neutral,
                                    "No supported images found",
                                    Some(format!("Use {} files.", super::IMAGE_FORMATS).into()),
                                    cx,
                                );
                                return;
                            }
                            this.apply_file_batch(files, truncated, req, window, cx);
                        })
                    })
                    .ok();
            })
            .detach();
            return;
        }
        self.apply_file_batch(paths, 0, req, window, cx);
    }

    fn apply_file_batch(
        &mut self,
        paths: Vec<PathBuf>,
        extra_truncated: usize,
        req: OpenRequest,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if paths.is_empty() || req.ticket != self.play_open_seq {
            return;
        }
        if paths.len() == 1 {
            self.open_play_file(paths.into_iter().next().unwrap(), req, window, cx);
            return;
        }
        let mut supported: Vec<PathBuf> = Vec::with_capacity(paths.len());
        let mut skipped = 0usize;
        for p in paths {
            if is_supported_image_ext(&p) {
                supported.push(p);
            } else {
                skipped += 1;
            }
        }
        if supported.is_empty() {
            self.toast(
                Tone::Neutral,
                "No supported images selected",
                Some(format!("Use {} files.", super::IMAGE_FORMATS).into()),
                cx,
            );
            return;
        }
        let single = supported.len() == 1;
        if let Some((id, name)) = &req.model
            && self.store.models.ready().is_some_and(|m| !m.iter().any(|m| m.id == *id))
        {
            let body = format!("Model {name} is no longer available.");
            let title = if single { "The image was not opened" } else { "The batch was not opened" };
            return self.toast(Tone::Neutral, title, Some(body.into()), cx);
        }
        if let Some(why) = self.open_image_blocked() {
            let title = if single { "Can't open an image yet" } else { "Can't open images yet" };
            return self.toast(Tone::Neutral, title, Some(why.into()), cx);
        }
        if skipped > 0 {
            self.toast(
                Tone::Info,
                format!("Skipped {}", counted(skipped as u64, "unsupported file", "unsupported files")),
                Some(format!("Supported: {}.", super::IMAGE_FORMATS).into()),
                cx,
            );
        }
        if single {
            self.open_play_file(supported.into_iter().next().unwrap(), req, window, cx);
            return;
        }
        let mut truncated = extra_truncated;
        if supported.len() > MAX_BATCH_FILES {
            truncated += supported.len() - MAX_BATCH_FILES;
            supported.truncate(MAX_BATCH_FILES);
        }
        if truncated > 0 {
            self.toast(
                Tone::Info,
                format!("Loaded first {MAX_BATCH_FILES} images"),
                Some(format!("{} additional images were skipped.", int(truncated as u64)).into()),
                cx,
            );
        }
        self.navigate(Route::Playground, window, cx);
        if let Some((id, _)) = req.model {
            if self.play_model.as_deref() != Some(id.as_str()) {
                self.play_class = None;
            }
            self.play_model_run = self.store.model(&id).and_then(|m| m.run_id.clone());
            self.play_model = Some(id);
        }
        let labels = disambiguate_file_labels(&supported);
        let items: Vec<PlayBatchItem> = supported
            .into_iter()
            .zip(labels)
            .map(|(p, name)| PlayBatchItem {
                name,
                source: PlayBatchSource::File(p),
                status: BatchItemStatus::Pending,
            })
            .collect();
        self.install_play_batch(items, cx);
    }

    /// Loads up to 6 validation images from the active sample dataset as a Playground batch
    /// (ideal in Demo mode and for quick multi-image inspection on real datasets).
    pub fn load_play_demo_batch(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't load a batch yet", Some(why.into()), cx);
            return;
        }
        let Some(ds_id) = self.play_sample_dataset() else {
            self.toast(
                Tone::Neutral,
                "No validation samples available",
                Some("Select a model with a matching dataset, or pick local images/folders.".into()),
                cx,
            );
            return;
        };
        self.navigate(Route::Playground, window, cx);
        if let Some(imgs) = self.store.images.get(&ds_id).and_then(|l| l.ready()).cloned() {
            self.install_dataset_batch(&ds_id, &imgs, cx);
            return;
        }
        // Fetch a small page of validation images on demand when the dataset hasn't been opened yet.
        self.play_batch.load_gen += 1;
        let load_gen = self.play_batch.load_gen;
        let ds_req = ds_id.clone();
        self.fetch(
            cx,
            move |api| {
                api.images(
                    &ds_req,
                    &ImageQuery { split: Some(Split::Val), offset: 0, limit: 24, class: None },
                )
            },
            move |this, r, cx| {
                if this.play_batch.load_gen != load_gen {
                    return;
                }
                match r {
                    Ok(page) => this.install_dataset_batch(&ds_id, &page.images, cx),
                    Err(e) => {
                        this.toast(Tone::Danger, "Could not load batch samples", Some(e.message.into()), cx)
                    }
                }
            },
        );
    }

    fn install_dataset_batch(&mut self, ds_id: &str, imgs: &[ImageInfo], cx: &mut Context<Self>) {
        let samples = pick_val_samples(imgs, 6);
        if samples.len() < 2 {
            self.toast(
                Tone::Neutral,
                "Need at least 2 validation images for a batch",
                Some("Pick multiple local images or open a folder instead.".into()),
                cx,
            );
            return;
        }
        let ds_name = self.store.dataset_name(ds_id);
        let items: Vec<PlayBatchItem> = samples
            .into_iter()
            .map(|m| {
                let file = crate::screens::datasets::file_name(&m.file);
                let shown = if file.is_empty() { m.id.as_str() } else { file };
                PlayBatchItem {
                    name: format!("{ds_name} / {shown}").into(),
                    source: PlayBatchSource::Dataset { dataset_id: ds_id.to_string(), image_id: m.id },
                    status: BatchItemStatus::Pending,
                }
            })
            .collect();
        self.install_play_batch(items, cx);
    }

    /// Replaces the current Playground batch with `items` (must have >= 2) and opens item 0.
    pub(crate) fn install_play_batch(&mut self, items: Vec<PlayBatchItem>, cx: &mut Context<Self>) {
        self.play_batch.clear();
        self.play_batch.items = items;
        self.play_batch.active = 0;
        self.load_active_batch_item(cx);
        cx.notify();
    }

    /// Cancels any in-flight sequential Run All without clearing the loaded batch strip.
    pub fn cancel_play_batch(&mut self) {
        self.play_batch.cancel();
    }

    /// Exits batch mode and cancels any in-flight batch work while keeping the currently staged image.
    pub fn clear_play_batch(&mut self) {
        self.play_batch.clear();
    }

    /// Cancels in-flight batch work and clears stale inference results after a model, device, or
    /// IoU change, while preserving an active unstaged item's load failure so its failure banner
    /// stays visible while the previous image remains on screen.
    pub(super) fn invalidate_play_batch(&mut self) {
        self.play_batch.cancel();
        self.play_batch.qa_filter = crate::screens::playground::BatchQaFilter::All;
        let active = self.play_batch.active;
        let active_unstaged_failure = (self.play_image.as_ref().and_then(|i| i.batch_ref)
            != Some((self.play_batch.batch_id, active)))
        .then(|| {
            self.play_batch.items.get(active).and_then(|it| match &it.status {
                BatchItemStatus::Failed(msg) => Some(msg.clone()),
                _ => None,
            })
        })
        .flatten();
        self.play_batch.invalidate_results();
        if let (Some(msg), Some(slot)) = (active_unstaged_failure, self.play_batch.items.get_mut(active)) {
            slot.status = BatchItemStatus::Failed(msg);
        }
    }

    /// Selects item `idx` in the batch filmstrip and stages it on Playground.
    pub fn select_play_batch_item(&mut self, idx: usize, cx: &mut Context<Self>) {
        if idx >= self.play_batch.items.len() {
            return;
        }
        let already_staged = self.play_image.as_ref().and_then(|i| i.batch_ref)
            == Some((self.play_batch.batch_id, idx))
            && !matches!(self.play_batch.items[idx].status, BatchItemStatus::Failed(_));
        if self.play_batch.active == idx && already_staged {
            return;
        }
        self.play_batch.active = idx;
        self.load_active_batch_item(cx);
        cx.notify();
    }

    /// Keeps the staged batch item inside the active QA filter: preserves `play_batch.active`
    /// when it still matches, otherwise selects the next matching item (wrapping to the first).
    /// Returns `true` when it switched to a different batch item.
    pub(crate) fn reconcile_play_batch_active(&mut self, cx: &mut Context<Self>) -> bool {
        if !self.play_batch.is_active()
            || self.play_batch.qa_filter == crate::screens::playground::BatchQaFilter::All
        {
            return false;
        }
        let filtered = self.play_batch_filtered_indices();
        if filtered.is_empty() || filtered.contains(&self.play_batch.active) {
            return false;
        }
        let next = filtered.iter().copied().find(|&i| i > self.play_batch.active).unwrap_or(filtered[0]);
        self.select_play_batch_item(next, cx);
        true
    }

    /// Sets the active Batch QA filter, clears any conflicting Single-mode class filter when
    /// selecting `ByClass`, and jumps to the first matching image when the active item falls
    /// outside the filtered set.
    pub fn set_play_batch_qa_filter(
        &mut self,
        filter: crate::screens::playground::BatchQaFilter,
        cx: &mut Context<Self>,
    ) {
        self.play_batch.qa_filter = filter;
        if matches!(filter, crate::screens::playground::BatchQaFilter::ByClass(_)) {
            self.play_class = None;
        }
        self.clear_play_hover();
        self.play_compare.clear_hover();
        if !self.reconcile_play_batch_active(cx) {
            cx.notify();
        }
    }

    /// Steps backward (`delta = -1`) or forward (`delta = +1`) through the filtered batch filmstrip
    /// (wrapping within the active QA filter).
    pub fn step_play_batch(&mut self, delta: i32, cx: &mut Context<Self>) {
        if self.play_batch.items.len() < 2 {
            return;
        }
        let filtered = self.play_batch_filtered_indices();
        if let Some(next) = self.play_batch.next_filtered_index(&filtered, delta) {
            self.select_play_batch_item(next, cx);
        }
    }

    /// Loads the currently active batch item onto the Playground stage (reusing its cached fresh
    /// prediction if already computed under the current settings).
    fn load_active_batch_item(&mut self, cx: &mut Context<Self>) {
        let idx = self.play_batch.active;
        let Some(item) = self.play_batch.items.get(idx).cloned() else { return };
        self.play_load_seq += 1;
        let seq = self.play_load_seq;
        let batch_id = self.play_batch.batch_id;
        let name = item.name.clone();
        match item.source.clone() {
            PlayBatchSource::Dataset { dataset_id, image_id } => {
                let input =
                    PredictInput::Dataset { dataset_id: dataset_id.clone(), image_id: image_id.clone() };
                let (d, i) = (dataset_id.clone(), image_id.clone());
                let thumb = ImgKey { dataset: d.clone(), image: i.clone(), kind: ImgKind::Thumb };
                self.play_loading = Some(Some((d.clone(), i.clone())));
                self.fetch(
                    cx,
                    move |api| {
                        let failed = || ApiError::new("decode_failed", "Could not decode the image");
                        match api.image(&d, &i, None)? {
                            ImageData::Encoded(b) => {
                                let (image, w, h) = decode_render(&b).ok_or_else(failed)?;
                                Ok((image, w, h, Some(Arc::new(b))))
                            }
                            data @ ImageData::Rgba { width, height, .. } => {
                                Ok((to_render_image(data).ok_or_else(failed)?, width, height, None))
                            }
                        }
                    },
                    move |this, r, cx| {
                        if seq != this.play_load_seq {
                            if let Ok((image, _, _, _)) = r {
                                cx.drop_image(image, None);
                            }
                            return;
                        }
                        this.play_loading = None;
                        if batch_id != this.play_batch.batch_id {
                            if let Ok((image, _, _, _)) = r {
                                cx.drop_image(image, None);
                            }
                            return;
                        }
                        match r {
                            Ok((image, width, height, encoded)) => {
                                if this.store.images_px.failed(&thumb) {
                                    this.store.images_px.retry(&thumb);
                                }
                                this.stage_batch_image(
                                    batch_id,
                                    idx,
                                    PlayImage {
                                        name,
                                        input,
                                        encoded,
                                        image,
                                        width,
                                        height,
                                        batch_ref: Some((batch_id, idx)),
                                    },
                                    cx,
                                );
                            }
                            Err(e) => {
                                if let Some(slot) = this.play_batch.items.get_mut(idx) {
                                    slot.status = BatchItemStatus::Failed(e.message.clone());
                                }
                                this.toast(
                                    Tone::Danger,
                                    "Could not open the image",
                                    Some(e.message.into()),
                                    cx,
                                );
                            }
                        }
                    },
                );
            }
            PlayBatchSource::File(path) => {
                self.play_loading = Some(None);
                let n2 = name.to_string();
                let read = cx.background_executor().spawn(async move {
                    let bytes = std::fs::read(&path)
                        .map_err(|e| ApiError::new("read_failed", format!("Could not read {n2}: {e}")))?;
                    let (image, width, height) = decode_render(&bytes).ok_or_else(|| {
                        ApiError::new("decode_failed", format!("{n2} could not be decoded"))
                    })?;
                    let input = PredictInput::File { name: n2, bytes: Arc::new(bytes) };
                    Ok::<PlayImage, ApiError>(PlayImage {
                        name,
                        input,
                        encoded: None,
                        image,
                        width,
                        height,
                        batch_ref: Some((batch_id, idx)),
                    })
                });
                cx.spawn(async move |this, cx| {
                    let r = read.await;
                    this.update(cx, |this, cx| {
                        if seq != this.play_load_seq {
                            if let Ok(img) = r {
                                cx.drop_image(img.image, None);
                            }
                            return;
                        }
                        this.play_loading = None;
                        if batch_id != this.play_batch.batch_id {
                            if let Ok(img) = r {
                                cx.drop_image(img.image, None);
                            }
                            return;
                        }
                        match r {
                            Ok(img) => this.stage_batch_image(batch_id, idx, img, cx),
                            Err(e) => {
                                if let Some(slot) = this.play_batch.items.get_mut(idx) {
                                    slot.status = BatchItemStatus::Failed(e.message.clone());
                                }
                                this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx);
                            }
                        }
                        cx.notify();
                    })
                    .ok();
                })
                .detach();
            }
        }
    }

    fn stage_batch_image(&mut self, batch_id: u64, idx: usize, img: PlayImage, cx: &mut Context<Self>) {
        if self.play_batch.batch_id != batch_id
            || self.play_batch.active != idx
            || idx >= self.play_batch.items.len()
        {
            cx.drop_image(img.image, None);
            return;
        }
        let img_hash = img.content_hash();
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        let cached = self.play_model.as_deref().and_then(|m| {
            self.play_batch
                .fresh_result(idx, m, self.play_iou, self.play_target, self.play_conf, e2e)
                .filter(|res| res.content_hash == img_hash)
                .cloned()
        });
        if let Some(res) = cached {
            self.set_play_image_with_cached_pred(img, res, cx);
        } else {
            if matches!(self.play_batch.items[idx].status, BatchItemStatus::Failed(_)) {
                self.play_batch.items[idx].status = BatchItemStatus::Pending;
            }
            self.set_play_image(img, cx);
        }
    }

    /// Records the stage's completed Model A prediction back into its exact staged batch item
    /// (verified via `(batch_id, item_index)` so same-basename files or concurrent stage loads
    /// never cross-contaminate cached detections).
    pub(super) fn sync_stage_pred_to_batch(
        &mut self,
        meta: Option<&PlayPredMeta>,
        r: &ApiResult<PredictResponse>,
    ) {
        if !self.play_batch.is_active() {
            return;
        }
        let Some(img) = self.play_image.as_ref() else { return };
        let Some((bid, idx)) = img.batch_ref else { return };
        if bid != self.play_batch.batch_id {
            return;
        }
        let content_hash = img.content_hash();
        let (width, height) = (img.width, img.height);
        let Some(item) = self.play_batch.items.get_mut(idx) else { return };
        match (r, meta) {
            (Ok(pred), Some(meta)) => {
                item.status = BatchItemStatus::Done(BatchItemResult {
                    pred: pred.clone(),
                    width,
                    height,
                    model_id: meta.model_id.clone(),
                    iou_bits: meta.iou_bits,
                    target: meta.target,
                    floor: meta.floor,
                    content_hash,
                });
            }
            (Err(e), _) => {
                item.status = BatchItemStatus::Failed(e.message.clone());
            }
            _ => {}
        }
    }

    /// Starts sequential Run All across every pending/failed/stale item in the Playground batch.
    /// When `force_rerun` is true, invalidates existing results first and runs every item fresh.
    pub fn start_play_batch(&mut self, force_rerun: bool, cx: &mut Context<Self>) {
        if !self.play_batch.is_active() {
            return;
        }
        let Some(model_id) = self.play_model.clone() else {
            self.toast(Tone::Neutral, "Select a model first", None, cx);
            return;
        };
        self.play_batch.cancel();
        if force_rerun {
            self.play_batch.invalidate_results();
        }
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        let first = self.next_pending_batch_index(0, &model_id, e2e);
        let Some(start_idx) = first else {
            self.toast(Tone::Neutral, "All batch images are already up to date", None, cx);
            return;
        };
        self.play_batch.run_gen += 1;
        let run_gen = self.play_batch.run_gen;
        self.play_batch.running = true;
        self.run_play_batch_step(run_gen, start_idx, cx);
        cx.notify();
    }

    fn next_pending_batch_index(&self, from: usize, model_id: &str, e2e: bool) -> Option<usize> {
        (from..self.play_batch.items.len()).find(|&i| {
            self.play_batch
                .fresh_result(i, model_id, self.play_iou, self.play_target, self.play_conf, e2e)
                .is_none()
        })
    }

    fn run_play_batch_step(&mut self, run_gen: u64, idx: usize, cx: &mut Context<Self>) {
        if !self.play_batch.running
            || self.play_batch.run_gen != run_gen
            || idx >= self.play_batch.items.len()
        {
            return;
        }
        let Some(model_id) = self.play_model.clone() else {
            self.play_batch.cancel();
            return;
        };
        let item = &mut self.play_batch.items[idx];
        item.status = BatchItemStatus::Running;
        let source = item.source.clone();
        let name = item.name.to_string();
        let floor = self.play_conf.min(PLAY_CONF_FLOOR);
        let iou = self.play_iou;
        let iou_bits = iou.to_bits();
        let target = self.play_target;
        let local_gen = crate::local::generation();
        let ticket = crate::local::Ticket::take(&self.play_batch.line);
        let model_req = model_id.clone();

        self.fetch(
            cx,
            move |api| {
                let (input, dims, content_hash) = match source {
                    PlayBatchSource::File(path) => {
                        let bytes = std::fs::read(&path).map_err(|e| {
                            ApiError::new("read_failed", format!("Could not read {name}: {e}"))
                        })?;
                        let hash = bytes_fingerprint(&bytes);
                        let img = decode_upright(&bytes).ok_or_else(|| {
                            ApiError::new("decode_failed", format!("{name} could not be decoded"))
                        })?;
                        let dims = img.dimensions();
                        (
                            PredictInput::File { name: name.clone(), bytes: Arc::new(bytes) },
                            Some(dims),
                            Some(hash),
                        )
                    }
                    PlayBatchSource::Dataset { dataset_id, image_id } => {
                        if target.is_some() {
                            match api.image(&dataset_id, &image_id, None)? {
                                ImageData::Encoded(bytes) => {
                                    let dims = decode_upright(&bytes).map(|im| im.dimensions());
                                    (
                                        PredictInput::File { name: name.clone(), bytes: Arc::new(bytes) },
                                        dims,
                                        None,
                                    )
                                }
                                ImageData::Rgba { width, height, .. } => (
                                    PredictInput::Dataset { dataset_id, image_id },
                                    Some((width, height)),
                                    None,
                                ),
                            }
                        } else {
                            (PredictInput::Dataset { dataset_id, image_id }, None, None)
                        }
                    }
                };
                let req =
                    PredictRequest { model_id: model_req.clone(), input, conf: floor, iou, max_det: 300 };
                let mut pred = match target {
                    Some(t) => crate::local::predict(api, &req, t, local_gen, &ticket)?,
                    None => api.predict(&req)?,
                };
                for b in &mut pred.boxes {
                    if b.name.trim().is_empty() {
                        b.name = format!("class {}", b.class);
                    }
                }
                let (width, height) = dims.unwrap_or((pred.image.width, pred.image.height));
                Ok(BatchItemResult {
                    pred,
                    width,
                    height,
                    model_id: model_req,
                    iou_bits,
                    target,
                    floor,
                    content_hash,
                })
            },
            move |this, r, cx| {
                this.play_batch_step_answered(run_gen, idx, &model_id, r, cx);
            },
        );
    }

    /// Handles one sequential batch prediction completion, guarded strictly by `run_gen` and `running`.
    pub(crate) fn play_batch_step_answered(
        &mut self,
        run_gen: u64,
        idx: usize,
        model_id: &str,
        r: ApiResult<BatchItemResult>,
        cx: &mut Context<Self>,
    ) {
        if !self.play_batch.running || self.play_batch.run_gen != run_gen {
            return;
        }
        if idx >= self.play_batch.items.len() || self.play_model.as_deref() != Some(model_id) {
            self.play_batch.cancel();
            return;
        }
        match r {
            Ok(res) => {
                let stage_matches = self.play_image.as_ref().is_some_and(|i| {
                    i.batch_ref == Some((self.play_batch.batch_id, idx))
                        && i.content_hash() == res.content_hash
                });
                if idx == self.play_batch.active
                    && stage_matches
                    && self.play_loading.is_none()
                    && !self.play_pred_pending
                {
                    let meta = PlayPredMeta {
                        image_gen: self.play_image_gen,
                        model_id: res.model_id.clone(),
                        iou_bits: res.iou_bits,
                        target: res.target,
                        floor: res.floor,
                    };
                    self.play_compare.pred_a_meta = Some(meta);
                    self.play_pred = Load::Ready(res.pred.clone());
                    self.play_pred_for = Some((self.play_image_gen, res.model_id.clone()));
                    self.play_fetched_floor = res.floor;
                    self.clear_play_hover();
                }
                self.play_batch.items[idx].status = BatchItemStatus::Done(res);
                self.reconcile_play_batch_active(cx);
            }
            Err(e) => {
                if e.code == "server_unreachable" {
                    self.store.note_reachability(Some(&e));
                }
                self.play_batch.items[idx].status = BatchItemStatus::Failed(e.message);
            }
        }
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        if let Some(next_idx) = self.next_pending_batch_index(idx + 1, model_id, e2e) {
            self.run_play_batch_step(run_gen, next_idx, cx);
        } else {
            self.play_batch.running = false;
            let s = self.play_batch.summary(
                Some(model_id),
                self.play_iou,
                self.play_target,
                self.play_conf,
                self.play_class,
                e2e,
            );
            // Clean completion is already shown inline in the batch filmstrip header; avoid
            // covering the bottom-right Batch JSON / Batch CSV export buttons with a success toast.
            if s.failed > 0 {
                let body = format!("{} succeeded · {} failed", s.done, s.failed);
                self.toast(Tone::Warning, "Batch run finished with errors", Some(body.into()), cx);
            }
        }
    }

    /// Prompts for a save path and writes the Playground Batch JSON file off the UI thread.
    pub fn export_play_batch_json(&mut self, cx: &mut Context<Self>) {
        let Some(built) = self.build_play_batch_export_json() else {
            self.toast(
                Tone::Neutral,
                "Nothing to export yet",
                Some("Run at least one batch image first.".into()),
                cx,
            );
            return;
        };
        let (json, suggested, summary) = match built {
            Ok(v) => v,
            Err(e) => {
                self.toast(Tone::Danger, "Batch JSON export failed", Some(e.to_string().into()), cx);
                return;
            }
        };
        self.save_batch_export_file("Batch JSON", json.into_bytes(), suggested, summary, cx);
    }

    /// Prompts for a save path and writes the Playground Batch CSV file off the UI thread.
    pub fn export_play_batch_csv(&mut self, cx: &mut Context<Self>) {
        let Some((csv, suggested, summary)) = self.build_play_batch_export_csv() else {
            self.toast(
                Tone::Neutral,
                "Nothing to export yet",
                Some("Run at least one batch image first.".into()),
                cx,
            );
            return;
        };
        self.save_batch_export_file("Batch CSV", csv.into_bytes(), suggested, summary, cx);
    }

    /// Prompts for a save path and writes the filtered per-image Batch Summary CSV off the UI thread.
    pub fn export_play_batch_summary_csv(&mut self, cx: &mut Context<Self>) {
        let Some((csv, suggested, summary)) = self.build_play_batch_summary_csv() else {
            self.toast(
                Tone::Neutral,
                "Nothing to export",
                Some("No batch images match the active QA filter.".into()),
                cx,
            );
            return;
        };
        self.save_batch_export_file("Batch Summary CSV", csv.into_bytes(), suggested, summary, cx);
    }

    fn save_batch_export_file(
        &mut self,
        label: &'static str,
        bytes_out: Vec<u8>,
        suggested: String,
        summary: String,
        cx: &mut Context<Self>,
    ) {
        let dir = export_default_dir();
        let rx = cx.prompt_for_new_path(&dir, Some(&suggested));
        cx.spawn(async move |this, cx| {
            let Some(path) = picked_save(&this, rx.await, cx) else { return };
            let write_path = path.clone();
            let res = cx
                .background_executor()
                .spawn(async move { write_export_bytes(&write_path, &bytes_out) })
                .await;
            this.update(cx, |this, cx| match res {
                Ok(()) => {
                    let reveal = path.clone();
                    let shown = display_path(&path.to_string_lossy());
                    this.toast_with(
                        Tone::Success,
                        format!("Exported {label}"),
                        Some(format!("{shown} · {summary}").into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(
                        Tone::Danger,
                        format!("{label} export failed"),
                        Some(e.to_string().into()),
                        cx,
                    );
                }
            })
            .ok();
        })
        .detach();
    }
}
