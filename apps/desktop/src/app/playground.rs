//! Playground: the image on stage, the model's predictions for it, and where the model runs.

use std::sync::Arc;

use folo_ui::{Context, SharedString, Window};

use crate::api::*;
use crate::store::*;
use crate::ui::*;

use super::{FoloApp, PlayImage, Route};

/// The Playground's lowest default confidence: predictions are fetched down to it once, and the
/// slider filters above it locally.
pub const PLAY_CONF_FLOOR: f32 = 0.05;

/// The image files the Playground opens: what the engine decodes (folo_data::IMAGE_EXTENSIONS).
pub const IMAGE_FORMATS: &str = "JPEG, PNG, BMP or WebP";

/// An encoded image upright ([`decode_upright`]) as a render image, with its size.
fn upright_render(bytes: &[u8]) -> Option<(Arc<folo_ui::RenderImage>, u32, u32)> {
    let img = decode_upright(bytes)?;
    let (w, h) = img.dimensions();
    let image = to_render_image(ImageData::Rgba { width: w, height: h, pixels: img.into_raw() })?;
    Some((image, w, h))
}

/// True when `head` (a file's first bytes) starts a JPEG, PNG, BMP or WebP file, the formats the
/// engine reads (Studio's `serverReads`, web/src/routes/playground/intake.ts).
fn engine_reads(head: &[u8]) -> bool {
    let at = |i: usize, b: &[u8]| head.get(i..i + b.len()) == Some(b);
    at(0, &[0xff, 0xd8, 0xff]) || at(0, b"\x89PNG") || at(0, b"BM") || (at(0, b"RIFF") && at(8, b"WEBP"))
}

/// Why a picked or dropped file did not decode, in Studio's words: one that starts as a format the
/// engine reads is damaged (cut short, a broken download), for which "Use a PNG file" would be
/// wrong (s4-models-2); anything else is not an image FOLO reads.
fn undecodable(name: &str, bytes: &[u8]) -> ApiError {
    if engine_reads(bytes) {
        return ApiError::new("decode_failed", format!("{name} could not be decoded"))
            .with_hint("The file may be truncated or damaged.");
    }
    ApiError::new("unsupported_image", format!("{name} is not an image FOLO can read"))
        .with_hint(format!("Use a {IMAGE_FORMATS} file."))
}

impl PlayImage {
    /// What a prediction of this image sends: a dataset image by its id to the server (which reads
    /// it from disk), but its bytes, already here, to a local run, so that needs no server.
    pub fn predict_input(&self, local: bool) -> PredictInput {
        match (&self.input, &self.encoded) {
            (PredictInput::Dataset { .. }, Some(bytes)) if local => {
                PredictInput::File { name: self.name.to_string(), bytes: bytes.clone() }
            }
            _ => self.input.clone(),
        }
    }
}

impl FoloApp {
    // ---------- playground ----------

    pub fn play_dataset_image(&mut self, dataset: &str, image_id: &str, cx: &mut Context<Self>) {
        let input = PredictInput::Dataset { dataset_id: dataset.to_string(), image_id: image_id.to_string() };
        let (d, i) = (dataset.to_string(), image_id.to_string());
        // Named by its file, as the dataset grid names it; by its id while the list isn't loaded.
        let file = self.store.images.get(dataset).and_then(|l| l.ready()).and_then(|list| {
            list.iter().find(|m| m.id == image_id).map(|m| crate::screens::datasets::file_name(&m.file))
        });
        let shown = file.filter(|f| !f.is_empty()).unwrap_or(image_id);
        let name: SharedString = format!("{} / {}", self.store.dataset_name(dataset), shown).into();
        self.play_load_seq += 1;
        let seq = self.play_load_seq;
        self.play_loading = Some(Some((d.clone(), i.clone())));
        let thumb = ImgKey { dataset: d.clone(), image: i.clone(), kind: ImgKind::Thumb };
        self.fetch(
            cx,
            move |api| {
                let failed = || ApiError::new("decode_failed", "Could not decode the image");
                match api.image(&d, &i, None)? {
                    // Kept as fetched: a local run predicts on these bytes, with or without the server.
                    ImageData::Encoded(b) => {
                        let (image, w, h) = upright_render(&b).ok_or_else(failed)?;
                        Ok((image, w, h, Some(Arc::new(b))))
                    }
                    data @ ImageData::Rgba { width, height, .. } => {
                        Ok((to_render_image(data).ok_or_else(failed)?, width, height, None))
                    }
                }
            },
            move |this, r, cx| {
                if seq != this.play_load_seq {
                    return;
                }
                this.play_loading = None;
                match r {
                    Ok((image, width, height, encoded)) => {
                        // Its pixels came (fetched past the image cache): a sample tile marked
                        // unavailable is asked for again, as a full size loading in the cache does.
                        if this.store.images_px.failed(&thumb) {
                            this.store.images_px.retry(&thumb);
                        }
                        this.set_play_image(PlayImage { name, input, encoded, image, width, height }, cx);
                    }
                    Err(e) => {
                        this.toast(Tone::Danger, "Could not open the image", Some(e.message.into()), cx)
                    }
                }
            },
        );
    }

    pub fn play_file(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) {
        // Also after a picker that opened while models were there, if they've gone since.
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
            return;
        }
        let name: SharedString =
            path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default().into();
        let n2 = name.to_string();
        self.play_load_seq += 1;
        let seq = self.play_load_seq;
        self.play_loading = Some(None);
        self.fetch(
            cx,
            move |_| {
                let bytes = std::fs::read(&path)
                    .map_err(|e| ApiError::new("read_failed", format!("Could not read {n2}: {e}")))?;
                let (render, w, h) = upright_render(&bytes).ok_or_else(|| undecodable(&n2, &bytes))?;
                Ok((Arc::new(bytes), render, w, h))
            },
            move |this, r, cx| {
                if seq != this.play_load_seq {
                    return;
                }
                this.play_loading = None;
                match r {
                    Ok((bytes, image, width, height)) => {
                        let input = PredictInput::File { name: name.to_string(), bytes };
                        this.set_play_image(
                            PlayImage { name: name.clone(), input, encoded: None, image, width, height },
                            cx,
                        );
                    }
                    Err(e) => this.toast(Tone::Danger, e.message.clone(), e.hint.map(Into::into), cx),
                }
            },
        );
    }

    pub fn play_zoom_by(&mut self, f: f32, cx: &mut Context<Self>) {
        (self.play_zoom, self.play_pan) = crate::viz::zoom_step(self.play_zoom, self.play_pan, f);
        cx.notify();
    }

    /// Put a new image on the Playground stage; its old detections never paint over it.
    fn set_play_image(&mut self, img: PlayImage, cx: &mut Context<Self>) {
        self.play_image_gen += 1;
        // Images live in the GPU sprite atlas until dropped explicitly. Called from fetch
        // callbacks, where no window is mid-update, so every window's atlas is reached.
        if let Some(old) = self.play_image.replace(img) {
            cx.drop_image(old.image, None);
        }
        self.play_pred = Load::Idle;
        self.play_pred_pending = false;
        self.play_pred_dim = false;
        self.play_pred_for = None;
        self.play_hover = None;
        self.play_zoom = 1.0;
        self.play_pan = (0., 0.);
        self.run_predict(cx);
    }

    /// Dataset `dataset`'s files changed ([`FoloApp::forget_dataset_contents`]): its image ids are
    /// positions, so the id of the Playground's image may name another picture now, and the server
    /// predicts on the file the id names. The old pixels and every prediction for them are dropped
    /// (the texture is returned to free) and the image is read again under its id, so what is shown
    /// and what is predicted are one picture. A load of one of its images still on its way is asked
    /// for again too (its answer may be the old picture); a newer pick from elsewhere is left alone.
    pub(super) fn forget_play_dataset(
        &mut self,
        dataset: &str,
        cx: &mut Context<Self>,
    ) -> Option<Arc<folo_ui::RenderImage>> {
        let shown = self.play_image.as_ref().and_then(|p| match &p.input {
            PredictInput::Dataset { dataset_id, image_id } if dataset_id == dataset => Some(image_id.clone()),
            _ => None,
        });
        let dropped = if shown.is_some() {
            self.play_image_gen += 1;
            self.play_pred_seq += 1;
            self.play_pred = Load::Idle;
            self.play_pred_pending = false;
            self.play_pred_dim = false;
            self.play_pred_for = None;
            self.play_hover = None;
            self.play_image.take().map(|p| p.image)
        } else {
            None
        };
        let again = match &self.play_loading {
            Some(Some((d, i))) if d == dataset => Some(i.clone()),
            Some(_) => None,
            None => shown,
        };
        if let Some(image) = again {
            self.play_dataset_image(dataset, &image, cx);
        }
        dropped
    }

    /// Choose where Playground runs the model, and rerun it there.
    pub fn set_play_target(&mut self, target: Option<crate::local::Target>, cx: &mut Context<Self>) {
        if self.play_target != target {
            self.play_target = target;
            if matches!(self.play_pred, Load::Ready(_)) {
                // The same image and model: the boxes stay up, dimmed and "Updating…", until the
                // other device answers, instead of blanking to placeholder rows.
                self.play_pred_dim = true;
                self.run_predict(cx);
            } else {
                // An error (an OOM and its Retry) isn't left live during the fallback.
                self.rerun_predict(cx);
            }
        }
    }

    /// The dataset whose validation images the Playground offers for its model, as the server picks
    /// one to evaluate it on: its run's dataset while that is registered, else the first dataset with
    /// exactly the model's classes, else none (a COCO model next to a blood-cell set gets no samples).
    pub fn play_sample_dataset(&self) -> Option<String> {
        let model = self.play_shown_model()?;
        let datasets = self.store.datasets.ready()?;
        let of_run = model.run_id.as_ref().and_then(|r| self.store.run(r)).map(|r| &r.config.dataset_id);
        of_run
            .and_then(|id| datasets.iter().find(|d| d.id == *id))
            .or_else(|| datasets.iter().find(|d| d.classes == model.classes))
            .map(|d| d.id.clone())
    }

    /// Look for a GPU for local inference (once; setting up folo-gpu blocks for a moment).
    pub fn probe_local_gpu(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.local_gpu, Load::Idle) || !self.store.api.local_inference() {
            return;
        }
        self.local_gpu = Load::Loading;
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async { crate::local::gpu() }).await;
            this.update(cx, |this, cx| {
                this.local_gpu = match r {
                    Ok((_, name)) => Load::Ready(name),
                    Err(e) => Load::Failed(crate::api::ApiError::new("no_gpu", e)),
                };
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Playground uses model `id` (picked from its list, or a model's Try in Playground) and runs it on
    /// the image on stage.
    pub fn pick_play_model(&mut self, id: String, cx: &mut Context<Self>) {
        self.play_model_run = self.store.model(&id).and_then(|m| m.run_id.clone());
        self.play_model = Some(id);
        self.run_predict(cx);
    }

    /// The model Playground shows and runs: only `play_model`, never another in its place.
    pub fn play_shown_model(&self) -> Option<&Model> {
        let id = self.play_model.as_ref()?;
        self.store.models.ready()?.iter().find(|m| m.id == *id)
    }

    /// Whether the Playground's model is one that run `run` made (`made`, deleted with it), also while
    /// the model list is loading.
    pub(super) fn play_model_made_by(&self, run: &str, made: &[String]) -> bool {
        self.play_model.as_ref().is_some_and(|m| made.contains(m))
            || self.play_model_run.as_deref() == Some(run)
    }

    /// The Playground's model against the model list as it now stands (after each list installed, and
    /// a delete here): gone (deleted here or by another client), everything it predicted, shown or still
    /// out, is dropped, the best model left is picked and run on the image on stage, and the person is
    /// told once, when Playground is in use. Without a model yet (the first list), the best is picked.
    pub(super) fn reconcile_play_model(&mut self, cx: &mut Context<Self>) {
        let Some(models) = self.store.models.ready() else { return };
        if let Some(id) = self.play_model.clone()
            && !models.iter().any(|m| m.id == id)
        {
            let name = self.deleted.get(&id).cloned().unwrap_or(id);
            self.drop_play_model(Some(name));
        }
        if self.play_model.is_some() {
            return;
        }
        // The best model (not the newest checkpoint, which may have learned nothing yet).
        let best = self.store.best_model(None).map(|m| (m.id.clone(), m.name.clone(), m.run_id.clone()));
        let in_use = self.play_image.is_some() || self.route == Route::Playground;
        if let Some(gone) = self.play_model_gone.take().filter(|_| in_use) {
            let body = match &best {
                Some((_, next, _)) => format!("Playground now uses {next}."),
                None => "No other model is left: train one first.".to_string(),
            };
            self.toast(Tone::Neutral, format!("{gone} was deleted"), Some(body.into()), cx);
        }
        if let Some((id, _, run)) = best {
            self.play_model = Some(id);
            self.play_model_run = run;
            self.run_predict(cx);
        }
    }

    /// The Playground's model is gone (its run was deleted, here or elsewhere; `name` says which):
    /// nothing it predicted is shown, and an answer still out for it is dropped when it comes.
    /// [`Self::reconcile_play_model`] picks another.
    pub(super) fn drop_play_model(&mut self, name: Option<String>) {
        self.play_model = None;
        self.play_model_run = None;
        self.play_model_gone = name;
        self.play_pred_seq += 1;
        self.play_pred = Load::Idle;
        self.play_pred_for = None;
        self.play_pred_pending = false;
        self.play_pred_dim = false;
        self.play_hover = None;
    }

    /// Predict again from scratch, showing "Detecting…" meanwhile (Retry, or another device). A plain
    /// rerun of the same image and model, an IoU change, keeps the old result up instead.
    pub fn rerun_predict(&mut self, cx: &mut Context<Self>) {
        self.play_pred_for = None;
        self.play_pred_dim = false;
        self.run_predict(cx);
    }

    /// Sets the Playground threshold, refetching when it drops below what was fetched (filtering
    /// is local only above that floor).
    pub fn set_play_conf(&mut self, conf: f32, cx: &mut Context<Self>) {
        self.play_conf = conf;
        if conf < self.play_fetched_floor - 1e-6 {
            self.run_predict(cx);
        }
    }

    pub fn run_predict(&mut self, cx: &mut Context<Self>) {
        let (Some(model), Some(img)) = (self.play_model.clone(), self.play_image.as_ref()) else { return };
        // Fetch everything above the lowest threshold once; the conf slider filters locally. The
        // floor is 0.05, or lower when the threshold is (a model's best-F1 confidence can be), so
        // local filtering never claims a threshold the fetch cut off.
        let floor = self.play_conf.min(PLAY_CONF_FLOOR);
        let req = PredictRequest {
            model_id: model.clone(),
            input: img.predict_input(self.play_target.is_some()),
            conf: floor,
            iou: self.play_iou,
            max_det: 300,
        };
        self.play_fetched_floor = floor;
        let key = (self.play_image_gen, model);
        // Same image and model (an IoU change): keep showing the old boxes until the new ones land.
        if self.play_pred_for.as_ref() != Some(&key) {
            self.play_pred = Load::Loading;
            self.play_hover = None;
        }
        self.play_pred_seq += 1;
        self.play_pred_pending = true;
        let seq = self.play_pred_seq;
        let target = self.play_target;
        // Taken with the backend handle: a switch after this makes the load's model throwaway.
        let generation = crate::local::generation();
        // Every prediction asked for supersedes the ones before: a local run still waiting for the
        // predictor when a newer one comes is skipped (its answer would be dropped anyway).
        let ticket = crate::local::Ticket::take(&self.play_line);
        self.fetch(
            cx,
            move |api| match target {
                Some(t) => crate::local::predict(api, &req, t, generation, &ticket),
                None => api.predict(&req),
            },
            move |this, r, _| this.play_pred_answered(seq, key, r),
        );
    }

    /// A prediction's answer: shown if it is still the latest asked for (`seq`). One that couldn't
    /// reach the server says so to the connection state too, so the next answer from the server
    /// (the 2 s refresh) counts as a reconnect and runs it again, however short the outage was.
    pub(super) fn play_pred_answered(&mut self, seq: u64, key: (u64, String), r: ApiResult<PredictResponse>) {
        if let Err(e) = &r
            && e.code == "server_unreachable"
        {
            self.store.note_reachability(Some(e));
        }
        if seq != self.play_pred_seq {
            return;
        }
        // Every detection has a name to show: a class outside the model's list (a mismatched
        // server, an old export) reads "class N", never a blank row.
        let r = r.map(|mut p| {
            for b in &mut p.boxes {
                if b.name.trim().is_empty() {
                    b.name = format!("class {}", b.class);
                }
            }
            p
        });
        self.play_pred = r.into();
        self.play_pred_for = Some(key);
        self.play_pred_pending = false;
        self.play_pred_dim = false;
    }

    /// Why Open image can't run now, if it can't: the model list failed (with the runs list too:
    /// not connected) or is still loading, or there's no model yet. Only the models decide: a
    /// failed runs list alone doesn't stop inference. The button, Ctrl+O and the palette share this.
    pub fn open_image_blocked(&self) -> Option<&'static str> {
        if self.store.models.error().is_some() {
            Some(if self.store.runs.error().is_some() {
                "Not connected"
            } else {
                "Models didn't load: retry under Best models"
            })
        } else if self.store.models.ready().is_none() {
            Some("Loading models…")
        } else if self.store.models.ready().is_some_and(|m| m.is_empty()) {
            Some("Train a model first")
        } else {
            None
        }
    }

    /// Open image (the button, Ctrl+O, the palette): the picker, then [`Self::open_play_file`] with the
    /// page it was asked for on. A cancelled picker changes nothing.
    pub fn pick_image(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // The shortcut and palette obey the button's rule, and say why.
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
            return;
        }
        let from = self.route.clone();
        let handle = window.window_handle();
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open image".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Some(p) = super::actions::picked(&this, rx.await, cx) {
                handle
                    .update(cx, |_, window, cx| {
                        this.update(cx, |this, cx| this.open_play_file(p, &from, window, cx))
                    })
                    .ok();
            }
        })
        .detach();
    }

    /// An image file into the Playground (picked, or dropped on the window) from page `from`: there
    /// through navigation, so Back returns to `from`, running the model of `from` when it has one
    /// ([`Self::model_in_context`]), else the Playground's own choice.
    pub(super) fn open_play_file(
        &mut self,
        path: std::path::PathBuf,
        from: &Route,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Also after a picker that opened while models were there, if they've gone since: nothing moves.
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
            return;
        }
        let model = self.model_in_context(from);
        self.navigate(Route::Playground, window, cx);
        if let Some(id) = model.filter(|id| self.play_model.as_ref() != Some(id)) {
            self.pick_play_model(id, cx);
        }
        self.play_file(path, cx);
    }

    /// The model page `route` is about, as its Try in Playground picks it: a model's page, that model;
    /// a finished (or stopped) run's page, its model. Other pages have none.
    pub(super) fn model_in_context(&self, route: &Route) -> Option<String> {
        let id = match route {
            Route::Model(id) => id.clone(),
            Route::Run(id) => self
                .store
                .run(id)
                .filter(|r| matches!(r.status, RunStatus::Completed | RunStatus::Stopped))?
                .model_id
                .clone()?,
            _ => return None,
        };
        self.store.model(&id).map(|m| m.id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{undecodable, upright_render};

    /// s4-models-2: a PNG cut short was called an unsupported format ("Use a JPEG, PNG, BMP or WebP
    /// file."). A file that starts as one the engine reads is damaged; only anything else is not an
    /// image FOLO reads.
    #[test]
    fn a_damaged_png_is_called_damaged_not_an_unsupported_format() {
        let mut cut = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        cut.extend([0u8; 5]);
        assert!(upright_render(&cut).is_none(), "the cut PNG does not decode");
        let e = undecodable("cut.png", &cut);
        assert_eq!(e.message, "cut.png could not be decoded");
        assert_eq!(e.hint.as_deref(), Some("The file may be truncated or damaged."));
        for head in [&b"\xff\xd8\xff\xe0"[..], b"BM\0\0", b"RIFF\0\0\0\0WEBPVP8 "] {
            assert_eq!(undecodable("x", head).code, "decode_failed", "{head:?}");
        }
        for other in [&b"GIF89a"[..], b"hello", b"", b"RIFF\0\0\0\0WAVE"] {
            let e = undecodable("notes.txt", other);
            assert_eq!(e.message, "notes.txt is not an image FOLO can read");
            assert_eq!(e.hint.as_deref(), Some("Use a JPEG, PNG, BMP or WebP file."));
        }
    }
}
