//! The image inspector over a dataset grid: the image with its labels (and a run's predictions),
//! zoom and pan, box/label toggles, click-to-zoom labels, Playground handoff, and stepping through
//! the filtered images.

mod header;
mod sidebar;
mod stage;

use folo_ui::{
    AnyElement, Bounds, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, Pixels,
    Styled, Window, div, px,
};

use crate::app::{FoloApp, act};
use crate::store::ImgKind;
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

/// Computes `(zoom, pan)` to center and frame `box_xyxy` comfortably inside `bounds` (fitting the
/// box to ~45% of the stage, clamped to `1.5×..=8.0×` and clamped via [`crate::viz::clamp_pan`]).
pub fn focus_box_transform(
    bounds: Bounds<Pixels>,
    (w, h): (f32, f32),
    (x1, y1, x2, y2): (f32, f32, f32, f32),
) -> (f32, (f32, f32)) {
    let (bw, bh) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    if !w.is_finite() || !h.is_finite() || w <= 0. || h <= 0. || bw <= 0. || bh <= 0. {
        return (1.0, (0., 0.));
    }
    let bx1 = x1.min(x2).clamp(0., w);
    let bx2 = x1.max(x2).clamp(0., w);
    let by1 = y1.min(y2).clamp(0., h);
    let by2 = y1.max(y2).clamp(0., h);
    let box_w = (bx2 - bx1).max(1.);
    let box_h = (by2 - by1).max(1.);
    let cx = (bx1 + bx2) * 0.5;
    let cy = (by1 + by2) * 0.5;
    let base_s = (bw / w).min(bh / h);
    if !base_s.is_finite() || base_s <= 0. {
        return (1.0, (0., 0.));
    }
    let zoom = ((bw * 0.45) / (box_w * base_s)).min((bh * 0.45) / (box_h * base_s)).clamp(1.5, 8.0);
    let s = base_s * zoom;
    let raw_pan = ((w * 0.5 - cx) * s, (h * 0.5 - cy) * s);
    let pan = crate::viz::clamp_pan(bounds, w, h, Fit::Contain, zoom, raw_pan);
    (zoom, pan)
}

impl FoloApp {
    /// The inspector was asked for an image by id (`Inspector::want`): once the list is in, its
    /// place in the filtered list. An image the dataset page's filters hide clears them (the
    /// inspector walks that list); one the dataset doesn't have opens the first image. The class
    /// filter turned on or off under the inspector (its class list read again, after its images)
    /// moves every place: it lands on the image it showed again.
    fn land_inspector(&mut self, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_ref()
            && i.want.is_none()
            && let Some((image, class, empty, query, sort, len)) = &i.shown
        {
            let cur_len = self.store.images.get(&i.dataset).and_then(|l| l.ready()).map_or(0, Vec::len);
            let changed = (*class, *empty, query.as_str(), *sort, *len)
                != (
                    self.class_filter(&i.dataset),
                    self.empty_only(&i.dataset),
                    self.img_query(&i.dataset),
                    self.img_sort(&i.dataset),
                    cur_len,
                );
            if changed {
                let image = image.clone();
                let i = self.inspector.as_mut().unwrap();
                i.want = Some(image);
                i.shown = None;
            }
        }
        let Some((ds, want)) =
            self.inspector.as_ref().and_then(|i| Some((i.dataset.clone(), i.want.clone()?)))
        else {
            return;
        };
        let Some(all) = self.store.images.get(&ds).and_then(|l| l.ready()) else { return };
        let in_all = all.iter().position(|m| m.id == want);
        let shown = self.filtered_indices(&ds).iter().position(|&i| all[i].id == want);
        let index = match (shown, in_all) {
            (Some(i), _) => i,
            (None, Some(i)) => {
                self.ds_split = None;
                // Another dataset's filters hid nothing here: they stay with that dataset.
                if self.ds_class.as_ref().is_some_and(|(of, _)| *of == ds) {
                    self.ds_class = None;
                }
                if self.empty_only(&ds) {
                    self.ds_empty_of = None;
                }
                if self.ds_img_dataset.as_deref() == Some(ds.as_str()) && !self.ds_img_query.is_empty() {
                    self.ds_img_query.clear();
                    self.ds_img_search.update(cx, |inp, cx| inp.set_text("", cx));
                }
                // Re-resolve in case a non-default sort reordered the unfiltered image list.
                self.filtered_indices(&ds).iter().position(|&raw| raw == i).unwrap_or(i)
            }
            // Not in yet: a large list arrives a page at a time (a run's sample can be past the
            // first page), so wait for it (or keep waiting on its page error) rather than open
            // another image.
            (None, None) if self.images_pending(&ds) || self.failed_image_page(&ds).is_some() => return,
            (None, None) => 0,
        };
        if let Some(i) = self.inspector.as_mut() {
            i.index = index;
            i.want = None;
        }
    }

    pub(super) fn inspector_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.land_inspector(cx);
        let insp = self.inspector.as_ref().unwrap();
        let ds_id = insp.dataset.clone();
        let n = self.filtered_indices(&ds_id).len();
        // Its image not in yet (the list still arriving, or a later page failed): never fall back
        // to an unrelated image.
        if n == 0 || insp.want.is_some() {
            return self.inspector_waiting(&t, &ds_id, window, cx);
        }
        let index = insp.index.min(n - 1);
        let (zoom, pan, hover, focused_box, show_boxes, show_labels) =
            (insp.zoom, insp.pan, insp.hover_box, insp.focused_box, insp.boxes, insp.labels);
        let Some(meta) = self.filtered_image(&ds_id, index).cloned() else { return div().into_any_element() };
        let (class, empty) = (self.class_filter(&ds_id), self.empty_only(&ds_id));
        let query = self.img_query(&ds_id).to_string();
        let sort = self.img_sort(&ds_id);
        let loaded_len = self.store.images.get(&ds_id).and_then(|l| l.ready()).map_or(0, Vec::len);
        if let Some(i) = self.inspector.as_mut() {
            i.shown = Some((meta.id.clone(), class, empty, query, sort, loaded_len));
        }
        // Opened from a run's samples: this image's predictions at that epoch, if it is one of them.
        let sample_ctx = self.inspector.as_ref().and_then(|i| i.sample.clone());
        let sample_set =
            sample_ctx.as_ref().and_then(|(run, _)| self.store.samples.get(run)?.ready().cloned());
        let sample_preds = sample_set.as_ref().and_then(|s| {
            let smp = s.samples.iter().find(|x| x.image_id == meta.id)?;
            Some((s.epoch, smp.pred.clone()))
        });
        let show_preds = sample_ctx.as_ref().is_some_and(|(_, v)| *v != crate::app::SampleView::Truth);
        // Shown predictions (at the run page's minimum confidence, as it draws them). Box indices:
        // the labels first (0..n), then these (n..), for hover, focus and the list rows alike.
        let conf = self.sample_conf;
        let preds: Option<Vec<crate::api::ScoredBox>> = sample_preds
            .as_ref()
            .filter(|_| show_preds)
            .map(|(_, ps)| ps.iter().filter(|p| p.score >= conf).cloned().collect());
        let n_labels = meta.boxes.len();
        // A keyboard-focused label row highlights its box like the pointer does; while the keyboard
        // is in use it wins over a pointer hover left behind on the image.
        let focused =
            (0..n_labels).find(|i| self.fh(format!("insp-box-{i}"), cx).is_focused(window)).or_else(|| {
                (0..preds.as_ref().map_or(0, Vec::len))
                    .find(|j| self.fh(format!("insp-pred-{j}"), cx).is_focused(window))
                    .map(|j| n_labels + j)
            });
        let hover = if self.kb { focused.or(hover) } else { hover.or(focused) }.or(focused_box);
        let classes = self.store.dataset_summary(&ds_id).map(|d| d.classes.clone()).unwrap_or_default();
        let img = self
            .image(&ds_id, &meta.id, ImgKind::Full, cx)
            .or_else(|| self.image(&ds_id, &meta.id, ImgKind::Thumb, cx));
        let dims = (meta.width as f32, meta.height as f32);
        let mut boxes: Vec<DrawBox> = meta
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| DrawBox {
                label: classes.get(b.class as usize).cloned().map(Into::into),
                highlight: hover == Some(i),
                dim: hover.is_some_and(|h| h != i),
                // With predictions over them, the labels are drawn as the run page draws ground
                // truth: dashed, no chip (the list beside names them).
                truth: preds.is_some() && hover != Some(i),
                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
            })
            .collect();
        if let Some(ps) = &preds {
            // After the labels, so a label row's index is still its box's.
            boxes.extend(ps.iter().enumerate().map(|(j, p)| DrawBox {
                label: classes.get(p.class as usize).cloned().map(Into::into),
                score: Some(p.score),
                highlight: hover == Some(n_labels + j),
                dim: hover.is_some_and(|h| h != n_labels + j),
                ..DrawBox::new(p.class, p.x1, p.y1, p.x2, p.y2)
            }));
        }
        // What the image shows, said beside it: the run, the epoch, and how each kind is drawn.
        let sample_line: Option<(String, String)> = sample_ctx.as_ref().map(|(run, _)| {
            let name = self.store.run(run).map_or(run.clone(), |r| r.name.clone());
            match (&sample_preds, show_preds) {
                (Some((epoch, _)), true) => (
                    format!("{name} · epoch {epoch}"),
                    format!("Predictions ≥ {conf:.2} solid · labels dashed"),
                ),
                (Some((epoch, _)), false) => {
                    (format!("{name} · epoch {epoch}"), "Labels (the run page's view)".to_string())
                }
                (None, _) => (name, "Not one of this run's samples: labels only".to_string()),
            }
        });
        let stage = self.inspector_stage(
            &t,
            &ds_id,
            &meta.id,
            img,
            dims,
            &boxes,
            zoom,
            pan,
            show_boxes,
            show_labels,
            window,
            cx,
        );
        let side = self.inspector_sidebar(
            &t,
            &meta,
            dims,
            &classes,
            preds.as_deref(),
            sample_line,
            hover,
            window,
            cx,
        );
        let sample_run = sample_ctx.as_ref().map(|(run, _)| run.as_str());
        let counter = match sample_set.as_ref().and_then(|s| {
            let at = s.samples.iter().position(|x| x.image_id == meta.id)?;
            Some((at, s.samples.len(), s.epoch))
        }) {
            // Opened from a run's samples: ← / → walk those.
            Some((at, len, epoch)) => format!("Sample {} of {len} · epoch {epoch}", at + 1),
            None => format!("{} of {}", int(index as u64 + 1), int(n as u64)),
        };
        let header = self.inspector_header(
            &t,
            &ds_id,
            &meta.id,
            sample_run,
            counter,
            zoom,
            show_boxes,
            show_labels,
            window,
            cx,
        );
        // Fades in like the dialogs and palette (160 ms; none under Reduce motion).
        crate::ui::appear(
            div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .child(self.scrim(
                    &t,
                    "insp-scrim",
                    window,
                    cx,
                    act(|this, w, cx| this.close_inspector(w, cx)),
                ))
                .child(
                    div()
                        .id("inspector")
                        .track_focus(&self.inspector_focus)
                        .absolute()
                        .top(px(24.))
                        .left(px(24.))
                        .right(px(24.))
                        .bottom(px(24.))
                        .occlude()
                        .capture_any_mouse_down(Self::guard_presses(cx))
                        .flex()
                        .flex_col()
                        .rounded(px(radius::XL))
                        .border_1()
                        .border_color(t.line_default)
                        .shadow(t.shadow_lg())
                        .overflow_hidden()
                        .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _, cx| {
                            this.inspector_key_down(ev, cx);
                        }))
                        .child(header)
                        .child(div().flex_1().min_h_0().flex().child(stage).child(side)),
                ),
            "inspector-in",
        )
    }

    fn inspector_waiting(
        &mut self,
        t: &crate::theme::Theme,
        ds_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        if let Some(failed) = self.failed_image_page(ds_id).cloned() {
            let retry_id = ds_id.to_string();
            let msg = format!(
                "Couldn't load remaining images from offset {}: {}",
                int(failed.offset as u64),
                failed.error.message
            );
            let retry = self.button(
                "insp-page-retry",
                "Retry",
                crate::controls::Btn::Primary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.retry_failed_image_page(&retry_id, cx)),
            );
            let close = self.button(
                "insp-page-close",
                "Close",
                crate::controls::Btn::Secondary,
                None,
                window,
                cx,
                act(|this, w, cx| this.close_inspector(w, cx)),
            );
            let card = div()
                .id("insp-page-error")
                .w(px(420.))
                .p(px(20.))
                .occlude()
                .capture_any_mouse_down(Self::guard_presses(cx))
                .rounded(px(radius::XL))
                .bg(t.bg_overlay)
                .border_1()
                .border_color(t.line_default)
                .shadow(t.shadow_lg())
                .flex()
                .flex_col()
                .gap(px(12.))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(icon("icons/circle-alert.svg", 16., t.warning_text))
                        .child(text(ty::MD, t.text_primary).child("Couldn't open this image")),
                )
                .child(text(ty::SM, t.text_secondary).whitespace_normal().child(msg))
                .child(div().flex().justify_end().gap(px(8.)).child(close).child(retry));
            return div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .child(self.scrim(
                    t,
                    "insp-scrim",
                    window,
                    cx,
                    act(|this, w, cx| this.close_inspector(w, cx)),
                ))
                .child(
                    div()
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(card),
                )
                .into_any_element();
        }
        // Still loading the image list (or the filter emptied it): keep the scrim up.
        if self.store.images.get(ds_id).is_some_and(|l| !l.is_loading()) && !self.images_pending(ds_id) {
            self.take_inspector(cx);
        }
        // The page's scrim (it takes the clicks, and a click there closes, as Esc does).
        self.scrim(t, "insp-scrim", window, cx, act(|this, w, cx| this.close_inspector(w, cx)))
            .into_any_element()
    }

    fn inspector_key_down(&mut self, ev: &KeyDownEvent, cx: &mut Context<Self>) {
        let mods = &ev.keystroke.modifiers;
        let plain = !mods.control && !mods.alt && !mods.platform;
        // Zoomed in, arrows pan (Shift: further); otherwise ←/→ browse. Page Up/Down always browse.
        let zoomed = self.inspector.as_ref().is_some_and(|i| i.zoom > 1.0);
        let d = if mods.shift { 128. } else { 32. };
        let pan = |this: &mut Self, dx: f32, dy: f32| {
            if let Some(i) = this.inspector.as_mut() {
                i.pan = (i.pan.0 + dx, i.pan.1 + dy);
                i.focused_box = None;
            }
        };
        match ev.keystroke.key.as_str() {
            "left" if zoomed => pan(self, d, 0.),
            "right" if zoomed => pan(self, -d, 0.),
            "up" if zoomed => pan(self, 0., d),
            "down" if zoomed => pan(self, 0., -d),
            "left" | "pageup" => self.step_inspector(-1, cx),
            "right" | "pagedown" => self.step_inspector(1, cx),
            "+" | "=" => self.zoom_by(1.4, cx),
            "-" => self.zoom_by(1.0 / 1.4, cx),
            "0" => {
                if let Some(i) = self.inspector.as_mut() {
                    i.zoom = 1.0;
                    i.pan = (0., 0.);
                    i.focused_box = None;
                }
            }
            k if plain && k.eq_ignore_ascii_case("b") => self.toggle_inspector_boxes(cx),
            k if plain && k.eq_ignore_ascii_case("l") => self.toggle_inspector_labels(cx),
            _ => return,
        }
        cx.stop_propagation();
        cx.notify();
    }

    /// Toggles bounding box overlays in the inspector (`B`).
    pub fn toggle_inspector_boxes(&mut self, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_mut() {
            i.boxes = !i.boxes;
            if !i.boxes {
                i.hover_box = None;
                i.focused_box = None;
            }
            cx.notify();
        }
    }

    /// Toggles label chips on bounding boxes in the inspector (`L`). When boxes are hidden, turns
    /// both boxes and labels back on so the chip toggle always produces a visible change.
    pub fn toggle_inspector_labels(&mut self, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_mut() {
            if !i.boxes {
                i.boxes = true;
                i.labels = true;
            } else {
                i.labels = !i.labels;
            }
            cx.notify();
        }
    }

    /// Zooms and centers the inspector stage onto a box (or fits back to `1×` if clicked again).
    pub fn focus_inspector_box(
        &mut self,
        box_idx: usize,
        dims: (f32, f32),
        coords: (f32, f32, f32, f32),
        cx: &mut Context<Self>,
    ) {
        let bounds = self.insp_view.get();
        let Some(insp) = self.inspector.as_mut() else { return };
        if insp.focused_box == Some(box_idx) && insp.zoom > 1.0 {
            insp.zoom = 1.0;
            insp.pan = (0., 0.);
            insp.focused_box = None;
            insp.hover_box = Some(box_idx);
        } else {
            // Reveal boxes so the focused box is always visible.
            insp.boxes = true;
            insp.hover_box = Some(box_idx);
            insp.focused_box = Some(box_idx);
            if let Some(b) = bounds {
                (insp.zoom, insp.pan) = focus_box_transform(b, dims, coords);
            }
        }
        cx.notify();
    }

    pub fn step_inspector(&mut self, d: i32, cx: &mut Context<Self>) {
        // From the image on screen: the class filter may have changed since it was drawn.
        self.land_inspector(cx);
        // Opened from a run's samples: the previous / next sample (the inspector lands on it by id).
        let next_sample = self.inspector.as_ref().and_then(|i| {
            let (run, _) = i.sample.as_ref()?;
            let set = self.store.samples.get(run)?.ready()?;
            let cur = self.filtered_image(&i.dataset, i.index).map(|m| m.id.clone());
            let at = cur.and_then(|c| set.samples.iter().position(|x| x.image_id == c))?;
            let len = set.samples.len() as i32;
            Some(set.samples[(at as i32 + d).rem_euclid(len) as usize].image_id.clone())
        });
        if let (Some(id), Some(i)) = (next_sample, self.inspector.as_mut()) {
            i.want = Some(id);
            i.zoom = 1.0;
            i.pan = (0., 0.);
            i.hover_box = None;
            i.focused_box = None;
            cx.notify();
            return;
        }
        let n = self.inspector.as_ref().map_or(0, |i| self.filtered_indices(&i.dataset).len());
        if let Some(i) = self.inspector.as_mut()
            && n > 0
        {
            i.index = ((i.index as i32 + d).rem_euclid(n as i32)) as usize;
            i.zoom = 1.0;
            i.pan = (0., 0.);
            i.hover_box = None;
            i.focused_box = None;
            cx.notify();
        }
    }

    pub fn zoom_by(&mut self, f: f32, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_mut() {
            (i.zoom, i.pan) = crate::viz::zoom_step(i.zoom, i.pan, f);
            i.focused_box = None;
            cx.notify();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::focus_box_transform;
    use crate::viz::{Fit, to_image};
    use folo_ui::{Bounds, point, px, size};

    #[test]
    fn focus_box_transform_centers_and_clamps_zoom() {
        let bounds = Bounds::new(point(px(24.), px(24.)), size(px(800.), px(600.)));
        let dims = (640., 480.);
        // A moderate off-center box zooms in and places the box center at the stage center.
        let (z, pan) = focus_box_transform(bounds, dims, (160., 140., 260., 220.));
        assert!((1.5..=8.0).contains(&z), "zoom={z}");
        let center_pt =
            point(bounds.origin.x + bounds.size.width * 0.5, bounds.origin.y + bounds.size.height * 0.5);
        let (ix, iy) = to_image(bounds, dims, Fit::Contain, z, pan, center_pt);
        assert!((ix - 210.).abs() < 0.5 && (iy - 180.).abs() < 0.5, "mapped=({ix}, {iy})");

        // A tiny box caps at 8x zoom.
        let (z_tiny, _) = focus_box_transform(bounds, dims, (318., 238., 322., 242.));
        assert_eq!(z_tiny, 8.0);

        // A huge box clamps to at least 1.5x zoom so clicking always gives clear focus feedback.
        let (z_huge, pan_huge) = focus_box_transform(bounds, dims, (0., 0., 640., 480.));
        assert_eq!(z_huge, 1.5);
        assert!(pan_huge.0.abs() < 0.01 && pan_huge.1.abs() < 0.01);

        // Degenerate dimensions fall back safely.
        assert_eq!(focus_box_transform(bounds, (0., 480.), (0., 0., 10., 10.)), (1.0, (0., 0.)));
    }
}
