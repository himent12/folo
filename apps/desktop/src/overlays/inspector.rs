//! The image inspector over a dataset grid: the image with its labels (and a run's predictions),
//! zoom and pan, the label list, and stepping through the filtered images.

use crate::api::Label;
use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, ParentElement, ScrollWheelEvent, SharedString, StatefulInteractiveElement, Styled,
    Window, canvas, div, px,
};

use crate::app::{FoloApp, act};
use crate::store::ImgKind;
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, image_rect, paint_image_with_boxes, to_image};

impl FoloApp {
    /// The inspector was asked for an image by id (`Inspector::want`): once the list is in, its
    /// place in the filtered list. An image the dataset page's filters hide clears them (the
    /// inspector walks that list); one the dataset doesn't have opens the first image. The class
    /// filter turned on or off under the inspector (its class list read again, after its images)
    /// moves every place: it lands on the image it showed again.
    fn land_inspector(&mut self, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_ref()
            && i.want.is_none()
            && let Some((image, class, empty)) = &i.shown
            && (*class, *empty) != (self.class_filter(&i.dataset), self.empty_only(&i.dataset))
        {
            let image = image.clone();
            let i = self.inspector.as_mut().unwrap();
            i.want = Some(image);
            i.shown = None;
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
                if !self.ds_img_query.is_empty() {
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
            if let Some(failed) = self.failed_image_page(&ds_id).cloned() {
                let retry_id = ds_id.clone();
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
                        &t,
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
            if self.store.images.get(&ds_id).is_some_and(|l| !l.is_loading()) && !self.images_pending(&ds_id)
            {
                self.take_inspector(cx);
            }
            // The page's scrim (it takes the clicks, and a click there closes, as Esc does).
            return self
                .scrim(&t, "insp-scrim", window, cx, act(|this, w, cx| this.close_inspector(w, cx)))
                .into_any_element();
        }
        let index = insp.index.min(n - 1);
        let (zoom, pan, hover) = (insp.zoom, insp.pan, insp.hover_box);
        let Some(meta) = self.filtered_image(&ds_id, index).cloned() else { return div().into_any_element() };
        let (class, empty) = (self.class_filter(&ds_id), self.empty_only(&ds_id));
        if let Some(i) = self.inspector.as_mut() {
            i.shown = Some((meta.id.clone(), class, empty));
        }
        // A keyboard-focused label row highlights its box like the pointer does; while the keyboard
        // is in use it wins over a pointer hover left behind on the image.
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
        let focused =
            (0..n_labels).find(|i| self.fh(format!("insp-box-{i}"), cx).is_focused(window)).or_else(|| {
                (0..preds.as_ref().map_or(0, Vec::len))
                    .find(|j| self.fh(format!("insp-pred-{j}"), cx).is_focused(window))
                    .map(|j| n_labels + j)
            });
        let hover = if self.kb { focused.or(hover) } else { hover.or(focused) };
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
        // An image that would not load (its automatic retries may be spent) says so, with a Retry
        // that asks for it again (Tab reaches it; Enter or Space presses it). A full-size one shown
        // whose thumbnail failed after it loaded says so too: the grid would stay blank.
        let full_failed = self.image_failed(&ds_id, &meta.id, ImgKind::Full);
        let thumb_failed = img.is_some() && self.image_failed(&ds_id, &meta.id, ImgKind::Thumb);
        let unavailable = (full_failed || thumb_failed).then(|| {
            let (ds, id) = (ds_id.clone(), meta.id.clone());
            let retry = self.button(
                "insp-retry-image",
                "Retry",
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.retry_failed_image(&ds, &id, cx)),
            );
            div().absolute().bottom(px(40.)).left_0().right_0().flex().justify_center().child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .px(px(12.))
                    .py(px(8.))
                    .rounded(px(radius::MD))
                    .bg(t.bg_overlay)
                    .child(text(ty::SM, t.text_secondary).child(if !full_failed {
                        "Thumbnail unavailable: the grid can't show it"
                    } else if img.is_some() {
                        "Full-size image unavailable: showing the thumbnail"
                    } else {
                        "Image unavailable"
                    }))
                    .child(retry),
            )
        });
        let bounds_cell = self.insp_view.clone();
        // A pan pushed past the edge is stored clamped, so reversing it responds at once.
        if let (Some(b), Some(i)) = (bounds_cell.get(), self.inspector.as_mut()) {
            i.pan = crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, i.zoom, i.pan);
        }
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let down_cell = bounds_cell.clone();
        let paint_boxes = boxes.clone();
        let hit_boxes = boxes.clone();
        let tt = t;
        let img2 = img.clone();
        let stage = div()
            .id("insp-stage")
            .relative()
            .flex_1()
            .min_w_0()
            .bg(t.bg_inset)
            .cursor(if zoom > 1.0 { folo_ui::CursorStyle::OpenHand } else { folo_ui::CursorStyle::Arrow })
            .on_scroll_wheel(cx.listener(move |this, ev: &ScrollWheelEvent, w, cx| {
                let Some(b) = wheel_cell.get() else { return };
                let Some(insp) = this.inspector.as_mut() else { return };
                let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                let factor = (1.0 + dy * 0.0025).clamp(0.5, 2.0);
                let new_zoom = (insp.zoom * factor).clamp(1.0, 12.0);
                // Zoom about the cursor: keep the image point under it fixed.
                let rect = image_rect(b, dims.0, dims.1, Fit::Contain, insp.zoom, insp.pan);
                let (px_, py_) = (
                    f32::from(ev.position.x - rect.left()) / f32::from(rect.size.width),
                    f32::from(ev.position.y - rect.top()) / f32::from(rect.size.height),
                );
                let new_rect = image_rect(b, dims.0, dims.1, Fit::Contain, new_zoom, (0., 0.));
                let want_x = f32::from(ev.position.x) - px_ * f32::from(new_rect.size.width);
                let want_y = f32::from(ev.position.y) - py_ * f32::from(new_rect.size.height);
                insp.pan = if new_zoom <= 1.0 {
                    (0., 0.)
                } else {
                    (want_x - f32::from(new_rect.left()), want_y - f32::from(new_rect.top()))
                };
                insp.zoom = new_zoom;
                cx.notify();
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, _, cx| {
                    let _ = down_cell.get();
                    if let Some(insp) = this.inspector.as_mut() {
                        insp.drag_from = Some((ev.position, insp.pan));
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if let Some(insp) = this.inspector.as_mut() {
                        insp.drag_from = None;
                        cx.notify();
                    }
                }),
            )
            .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                let Some(insp) = this.inspector.as_mut() else { return };
                if let Some((from, pan0)) = insp.drag_from
                    && ev.pressed_button == Some(MouseButton::Left)
                {
                    if insp.zoom > 1.0 {
                        insp.pan = (
                            pan0.0 + f32::from(ev.position.x - from.x),
                            pan0.1 + f32::from(ev.position.y - from.y),
                        );
                        cx.notify();
                    }
                    return;
                }
                let Some(b) = bounds_cell.get() else { return };
                let (x, y) = to_image(b, dims, Fit::Contain, insp.zoom, insp.pan, ev.position);
                let hit = hit_box(&hit_boxes, x, y);
                if hit != insp.hover_box {
                    insp.hover_box = hit;
                    // The pointer moved over the image: hover leads again (not a keyboard
                    // focus left on a label row).
                    this.kb = false;
                    cx.notify();
                }
            }))
            .child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, cx| {
                        let inner = folo_ui::Bounds::new(
                            b.origin + folo_ui::point(px(24.), px(24.)),
                            // 24 px on every side (the zoom controls live in the header now).
                            b.size - folo_ui::size(px(48.), px(48.)),
                        );
                        paint_cell.set(Some(inner));
                        window.with_content_mask(Some(folo_ui::ContentMask { bounds: b }), |window| {
                            paint_image_with_boxes(
                                &tt,
                                window,
                                cx,
                                inner,
                                img2.as_ref(),
                                dims,
                                &paint_boxes,
                                Fit::Contain,
                                zoom,
                                pan,
                                true,
                                false,
                                0.,
                            );
                        });
                    },
                )
                .size_full(),
            )
            .children(unavailable);
        // By class, then largest first, like the PR legend; indices stay the image's.
        let mut box_order: Vec<(usize, &crate::api::BBox)> = meta.boxes.iter().enumerate().collect();
        box_order.sort_by(|(_, a), (_, b)| {
            let area = |x: &crate::api::BBox| (x.x2 - x.x1) * (x.y2 - x.y1);
            a.class.cmp(&b.class).then(area(b).total_cmp(&area(a)))
        });
        let insp_chrome = self.scroll_chrome(self.insp_list_scroll.clone(), t.bg_overlay, 0., 0., cx);
        let mut box_rows: Vec<AnyElement> = box_order
            .into_iter()
            .map(|(i, b)| {
                let hot = hover == Some(i);
                let key = format!("insp-box-{i}");
                // The class column is 70 px: a name that may not fit shows in full on hover and on
                // keyboard focus (short ones need neither).
                let full_name = classes.get(b.class as usize).cloned().unwrap_or_default();
                let long = crate::ui::may_overflow(&full_name, ty::SM.size, 70.);
                let row = self
                    .disclose(
                        div().id(SharedString::from(key.clone())),
                        &key,
                        full_name,
                        long,
                        30.,
                        false,
                        window,
                        cx,
                    )
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(&t)))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            let next = if *h { Some(i) } else { None };
                            if insp.hover_box != next {
                                insp.hover_box = next;
                                cx.notify();
                            }
                        }
                    }))
                    .child(crate::ui::class_swatch(&t, b.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .ellipsis()
                            .child(classes.get(b.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(
                        text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                            .tnum()
                            .flex_1()
                            .text_right()
                            .child(format!(
                                "{:.0}, {:.0} · {:.0}×{:.0}",
                                b.x1,
                                b.y1,
                                b.x2 - b.x1,
                                b.y2 - b.y1
                            )),
                    );
                // Focusable: Tab through labels to highlight each box from the keyboard.
                self.bind(
                    row,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            insp.hover_box = if insp.hover_box == Some(i) { None } else { Some(i) };
                        }
                        cx.notify();
                    }),
                )
                .into_any_element()
            })
            .collect();
        // The epoch's predictions, after the labels: score and box, each a Tab stop that
        // highlights its box (as a label row does).
        if let Some(ps) = &preds {
            box_rows.push(
                div()
                    .px(px(8.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(&t, "Predictions"))
                    .child(text(ty::SM, t.text_tertiary).child("score · x, y · w×h"))
                    .into_any_element(),
            );
            for (j, p) in ps.iter().enumerate() {
                let idx = n_labels + j;
                let hot = hover == Some(idx);
                let key = format!("insp-pred-{j}");
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(&t)))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            let next = if *h { Some(idx) } else { None };
                            if insp.hover_box != next {
                                insp.hover_box = next;
                                cx.notify();
                            }
                        }
                    }))
                    .child(crate::ui::class_swatch(&t, p.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .ellipsis()
                            .child(classes.get(p.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(text(ty::SM, t.text_primary).tnum().w(px(32.)).child(format!("{:.2}", p.score)))
                    .child(
                        text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                            .tnum()
                            .flex_1()
                            .text_right()
                            .child(format!(
                                "{:.0}, {:.0} · {:.0}×{:.0}",
                                p.x1,
                                p.y1,
                                p.x2 - p.x1,
                                p.y2 - p.y1
                            )),
                    );
                box_rows.push(
                    self.bind(
                        row,
                        &key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            if let Some(insp) = this.inspector.as_mut() {
                                insp.hover_box = if insp.hover_box == Some(idx) { None } else { Some(idx) };
                            }
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                );
            }
        }
        let prev = self.icon_button(
            "insp-prev",
            "icons/chevron-left.svg",
            "Previous (←)",
            window,
            cx,
            act(|this, _, cx| this.step_inspector(-1, cx)),
        );
        let next = self.icon_button(
            "insp-next",
            "icons/chevron-right.svg",
            "Next (→)",
            window,
            cx,
            act(|this, _, cx| this.step_inspector(1, cx)),
        );
        let close = self.icon_button(
            "insp-close",
            "icons/x.svg",
            "Close (Esc)",
            window,
            cx,
            act(|this, w, cx| this.close_inspector(w, cx)),
        );
        let side = div()
            .w(px(300.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(t.bg_overlay)
            .border_l_1()
            .border_color(t.line_subtle)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .p(px(16.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child({
                        // The panel's 268 px hold ~33 monospace glyphs (0.6 em each); a longer
                        // path ends in "…" and shows in full on hover.
                        let long = meta.file.chars().count() as f32 * 0.6 * ty::BASE.size > 268.;
                        let el = div().id("insp-file");
                        self.disclose(el, "insp-file", meta.file.clone(), long, 24., true, window, cx).child(
                            text(ty::BASE, t.text_primary)
                                .font_family(crate::theme::MONO)
                                .ellipsis()
                                .child(meta.file.clone()),
                        )
                    })
                    .child(
                        div()
                            .flex()
                            .gap(px(6.))
                            .child(badge(&t, meta.split.label(), Tone::Neutral))
                            .child(badge(&t, format!("{} × {}", meta.width, meta.height), Tone::Neutral))
                            .child(badge(
                                &t,
                                match &preds {
                                    Some(_) => counted(meta.boxes.len() as u64, "label", "labels"),
                                    None => counted(meta.boxes.len() as u64, "box", "boxes"),
                                },
                                Tone::Neutral,
                            ))
                            .when_some(preds.as_ref().map(Vec::len), |d, m| {
                                d.child(badge(
                                    &t,
                                    format!("{m} {}", if m == 1 { "prediction" } else { "predictions" }),
                                    Tone::Neutral,
                                ))
                            }),
                    )
                    .when_some(sample_line, |d, (run, how)| {
                        d.child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .child(text(ty::SM, t.text_secondary).ellipsis().child(run))
                                .child(text(ty::SM, t.text_tertiary).child(how)),
                        )
                    }),
            )
            .child(
                div()
                    .px(px(16.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(&t, "Labels"))
                    .child(text(ty::SM, t.text_tertiary).child("top-left x, y · w×h (px)")),
            )
            .child(
                div()
                    .relative()
                    .flex_shrink()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("insp-boxes")
                            // As tall as its rows (up to the room left), so the key hints sit right
                            // under it.
                            .flex_shrink()
                            .min_h_0()
                            .overflow_y_scroll()
                            .track_scroll(&self.insp_list_scroll)
                            .px(px(8.))
                            .children(box_rows)
                            .when(meta.boxes.is_empty(), |d| {
                                d.child(div().p(px(8.)).child(
                                    text(ty::SM, t.text_tertiary).child("No labels: a background image."),
                                ))
                            }),
                    )
                    .children(insp_chrome),
            )
            .child(
                // One action per row on a fixed key column, so the keys line up and nothing wraps.
                {
                    let row = |keys: AnyElement, what: &'static str| {
                        div()
                            .flex()
                            .items_center()
                            .child(
                                div()
                                    .w(px(96.))
                                    .flex_none()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap(px(4.))
                                    .child(keys),
                            )
                            // At 12 px the longest hint needs two lines in the 300 px panel: it
                            // wraps inside its column rather than running past the edge.
                            .child(
                                text(ty::SM, t.text_tertiary)
                                    .flex_1()
                                    .min_w_0()
                                    .whitespace_normal()
                                    .child(what),
                            )
                    };
                    div()
                        .p(px(12.))
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .border_t_1()
                        .border_color(t.line_subtle)
                        // Both pairs step the same way: one row, the pairs stacked in the key column.
                        .child(row(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(4.))
                                .child(shortcut(&t, &["←", "→"]))
                                .child(shortcut(&t, &["PgUp", "PgDn"]))
                                .into_any_element(),
                            "previous / next",
                        ))
                        .child(row(
                            div()
                                .flex()
                                .gap(px(4.))
                                .child(kbd(&t, "Scroll"))
                                .child(shortcut(&t, &["+", "−"]))
                                .into_any_element(),
                            "zoom",
                        ))
                        .child(row(kbd(&t, "0").into_any_element(), "fit"))
                        .child(row(kbd(&t, "Drag").into_any_element(), "pan (arrows too, when zoomed)"))
                },
            );
        // Fitted by any path (the readout, 0, −, another image), focus left on the readout moves
        // to its neighbour.
        if zoom <= 1.0 && self.fh("insp-fit", cx).is_focused(window) {
            self.focus_key("insp-zoom-in", window, cx);
        }
        // Zoom in the top bar, right-aligned before Close, as Playground keeps it in its header.
        let zoom_ctrls = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(self.icon_button(
                "insp-zoom-out",
                "icons/zoom-out.svg",
                "Zoom out",
                window,
                cx,
                act(|this, _, cx| this.zoom_by(1.0 / 1.4, cx)),
            ))
            .child(self.zoom_readout(
                "insp-fit",
                zoom,
                window,
                cx,
                act(|this, _, cx| {
                    if let Some(i) = this.inspector.as_mut() {
                        i.zoom = 1.0;
                        i.pan = (0., 0.);
                    }
                    cx.notify();
                }),
            ))
            .child(self.icon_button(
                "insp-zoom-in",
                "icons/zoom-in.svg",
                "Zoom in",
                window,
                cx,
                act(|this, _, cx| this.zoom_by(1.4, cx)),
            ));
        let header = div()
            .h(px(48.))
            .flex_none()
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(8.))
            .bg(t.bg_overlay)
            .border_b_1()
            .border_color(t.line_subtle)
            .child(prev)
            .child(next)
            .child(text(ty::SM, t.text_secondary).tnum().child(
                match sample_set.as_ref().and_then(|s| {
                    let at = s.samples.iter().position(|x| x.image_id == meta.id)?;
                    Some((at, s.samples.len(), s.epoch))
                }) {
                    // Opened from a run's samples: ← / → walk those.
                    Some((at, len, epoch)) => format!("Sample {} of {len} · epoch {epoch}", at + 1),
                    None => format!("{} of {}", int(index as u64 + 1), int(n as u64)),
                },
            ))
            .child(div().flex_1())
            .child(zoom_ctrls)
            .child(div().w(px(1.)).h(px(16.)).bg(t.line_subtle))
            .child(close);
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
                            // Zoomed in, arrows pan (Shift: further); otherwise ←/→ browse. Page Up/Down
                            // always browse.
                            let zoomed = this.inspector.as_ref().is_some_and(|i| i.zoom > 1.0);
                            let d = if ev.keystroke.modifiers.shift { 128. } else { 32. };
                            let pan = |this: &mut Self, dx: f32, dy: f32| {
                                if let Some(i) = this.inspector.as_mut() {
                                    i.pan = (i.pan.0 + dx, i.pan.1 + dy);
                                }
                            };
                            match ev.keystroke.key.as_str() {
                                "left" if zoomed => pan(this, d, 0.),
                                "right" if zoomed => pan(this, -d, 0.),
                                "up" if zoomed => pan(this, 0., d),
                                "down" if zoomed => pan(this, 0., -d),
                                "left" | "pageup" => this.step_inspector(-1, cx),
                                "right" | "pagedown" => this.step_inspector(1, cx),
                                "+" | "=" => this.zoom_by(1.4, cx),
                                "-" => this.zoom_by(1.0 / 1.4, cx),
                                "0" => {
                                    if let Some(i) = this.inspector.as_mut() {
                                        i.zoom = 1.0;
                                        i.pan = (0., 0.);
                                    }
                                }
                                _ => return,
                            }
                            cx.stop_propagation();
                            cx.notify();
                        }))
                        .child(header)
                        .child(div().flex_1().min_h_0().flex().child(stage).child(side)),
                ),
            "inspector-in",
        )
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
            cx.notify();
        }
    }

    pub fn zoom_by(&mut self, f: f32, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_mut() {
            (i.zoom, i.pan) = crate::viz::zoom_step(i.zoom, i.pan, f);
            cx.notify();
        }
    }
}
