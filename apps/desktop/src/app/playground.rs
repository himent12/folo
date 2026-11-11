//! Playground: the image on stage, the model's predictions for it, and where the model runs.

use std::sync::Arc;

use gpui::{Context, SharedString};

use crate::api::*;
use crate::store::*;
use crate::ui::*;

use super::{FoloApp, PlayImage, Route};

/// The Playground's lowest default confidence: predictions are fetched down to it once, and the
/// slider filters above it locally.
pub const PLAY_CONF_FLOOR: f32 = 0.05;

impl FoloApp {
    // ---------- playground ----------

    pub fn play_dataset_image(&mut self, dataset: &str, image_id: &str, cx: &mut Context<Self>) {
        let input = PredictInput::Dataset { dataset_id: dataset.to_string(), image_id: image_id.to_string() };
        let (d, i) = (dataset.to_string(), image_id.to_string());
        let name: SharedString = format!(
            "{} / {}",
            self.store.dataset_summary(dataset).map_or(dataset, |d| d.name.as_str()),
            image_id
        )
        .into();
        self.play_load_seq += 1;
        let seq = self.play_load_seq;
        self.fetch(
            cx,
            move |api| {
                let data = api.image(&d, &i, None)?;
                let (w, h) = match &data {
                    ImageData::Rgba { width, height, .. } => (*width, *height),
                    ImageData::Encoded(b) => {
                        image::load_from_memory(b).map(|i| (i.width(), i.height())).unwrap_or((0, 0))
                    }
                };
                to_render_image(data)
                    .map(|img| (img, w, h))
                    .ok_or_else(|| ApiError::new("decode_failed", "Could not decode the image"))
            },
            move |this, r, cx| {
                if seq != this.play_load_seq {
                    return;
                }
                match r {
                    Ok((image, width, height)) => {
                        this.set_play_image(PlayImage { name, input, image, width, height }, cx);
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
        self.fetch(
            cx,
            move |_| {
                let bytes = std::fs::read(&path)
                    .map_err(|e| ApiError::new("read_failed", format!("Could not read {n2}: {e}")))?;
                let img = image::load_from_memory(&bytes).map_err(|_| {
                    ApiError::new("unsupported_image", format!("{n2} is not an image FOLO can read"))
                        .with_hint("Use a JPEG or PNG.")
                })?;
                let (w, h) = (img.width(), img.height());
                let render = to_render_image(ImageData::Rgba {
                    width: w,
                    height: h,
                    pixels: img.to_rgba8().into_raw(),
                })
                .ok_or_else(|| ApiError::new("decode_failed", "Could not decode the image"))?;
                Ok((Arc::new(bytes), render, w, h))
            },
            move |this, r, cx| {
                if seq != this.play_load_seq {
                    return;
                }
                match r {
                    Ok((bytes, image, width, height)) => {
                        let input = PredictInput::File { name: name.to_string(), bytes };
                        this.set_play_image(
                            PlayImage { name: name.clone(), input, image, width, height },
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
        self.play_pred_for = None;
        self.play_hover = None;
        self.play_zoom = 1.0;
        self.play_pan = (0., 0.);
        self.run_predict(cx);
    }

    /// Choose where Playground runs the model, and rerun it there.
    pub fn set_play_target(&mut self, target: Option<crate::local::Target>, cx: &mut Context<Self>) {
        if self.play_target != target {
            self.play_target = target;
            // Another device's answer (and timing): "Detecting…" while it runs, not the previous
            // result, or an OOM error and its Retry still live during the CPU fallback.
            self.rerun_predict(cx);
        }
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

    /// Predict again from scratch, showing "Detecting…" meanwhile (Retry, or another device). A plain
    /// rerun of the same image and model, an IoU change, keeps the old result up instead.
    pub fn rerun_predict(&mut self, cx: &mut Context<Self>) {
        self.play_pred_for = None;
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
            input: img.input.clone(),
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
        self.fetch(
            cx,
            move |api| match target {
                Some(t) => crate::local::predict(api, &req, t, generation),
                None => api.predict(&req),
            },
            move |this, r, _| {
                if seq == this.play_pred_seq {
                    // Every detection has a name to show: a class outside the model's list (a
                    // mismatched server, an old export) reads "class N", never a blank row.
                    let r = r.map(|mut p| {
                        for b in &mut p.boxes {
                            if b.name.trim().is_empty() {
                                b.name = format!("class {}", b.class);
                            }
                        }
                        p
                    });
                    this.play_pred = r.into();
                    this.play_pred_for = Some(key);
                    this.play_pred_pending = false;
                }
            },
        );
    }

    /// Why Open image can't run now, if it can't: the model list failed (with the runs list too:
    /// not connected) or is still loading, or there's no model yet. Only the models decide: a
    /// failed runs list alone doesn't stop inference. The button, Ctrl+O and the palette share this.
    pub fn open_image_blocked(&self) -> Option<&'static str> {
        if self.store.models.error().is_some() {
            Some(if self.store.runs.error().is_some() {
                "Not connected"
            } else {
                "Models didn't load: retry under Best model per dataset"
            })
        } else if self.store.models.ready().is_none() {
            Some("Loading models…")
        } else if self.store.models.ready().is_some_and(|m| m.is_empty()) {
            Some("Train a model first")
        } else {
            None
        }
    }

    pub fn pick_image(&mut self, cx: &mut Context<Self>) {
        // The shortcut and palette obey the button's rule, and say why.
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
            return;
        }
        let rx = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open image".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| {
                    this.route = Route::Playground;
                    this.play_file(p, cx);
                })
                .ok();
            }
        })
        .detach();
    }
}
