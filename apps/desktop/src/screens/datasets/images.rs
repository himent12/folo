//! The image browser: the virtualised grid of a dataset's images.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, canvas, div, px, uniform_list,
};

use crate::api::{Dataset, ImageInfo, Split};
use crate::app::{FoloApp, act};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

const GAP: f32 = 8.;

impl FoloApp {
    pub(super) fn image_browser(
        &mut self,
        ds: &Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let id = ds.id.clone();
        let total = self.store.images.get(&id).and_then(|l| l.ready()).map_or(0, |v| v.len());
        let count = self.filtered_indices(&id).len();
        let splits = [None, Some(Split::Train), Some(Split::Val), Some(Split::Test)];
        let labels: Vec<String> = ["All", "Train", "Val", "Test"].iter().map(|s| s.to_string()).collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let usable: Vec<&str> =
            if ds.splits.test == 0 { label_refs[..3].to_vec() } else { label_refs.clone() };
        let sel = splits.iter().position(|s| *s == self.ds_split).unwrap_or(0);
        let seg = self.segmented(
            "ds-split",
            &usable,
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, _| {
                this.ds_split = splits[i];
                this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
            }),
        );
        // Counts follow the split filter: boxes per class among the images shown.
        let counts = match self.ds_split {
            None => ds.stats.as_ref().map(|s| s.class_counts.clone()).unwrap_or_default(),
            Some(split) => self.split_class_counts(&ds.id, split, ds.classes.len()),
        };
        let chips: Vec<AnyElement> = ds
            .classes
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let on = self.ds_class == Some(i as u32);
                // The count is boxes; the chip filters images (the tooltip says both).
                let n = counts.get(i).copied();
                let tip_text = if on {
                    format!("{} {name} boxes · showing images with one; click to clear", int(n.unwrap_or(0)))
                } else {
                    format!("{} {name} boxes · click to show only images with one", int(n.unwrap_or(0)))
                };
                self.chip(
                    &format!("ds-class-{i}"),
                    name.clone(),
                    on,
                    Some(i as u32),
                    counts.get(i).copied(),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.ds_class = if this.ds_class == Some(i as u32) { None } else { Some(i as u32) };
                        if this.ds_class.is_some() {
                            this.ds_empty_only = false;
                        }
                        this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                        cx.notify();
                    }),
                )
                .tooltip(crate::controls::tip(tip_text, None))
                .into_any_element()
            })
            .collect();
        let boxes_on = self.ds_boxes;
        let boxes_switch = self.switch(
            "ds-boxes",
            boxes_on,
            window,
            cx,
            act(|this, _, cx| {
                this.ds_boxes = !this.ds_boxes;
                cx.notify();
            }),
        );
        let size_slider = self.slider("ds-tile", self.ds_tile, 112., 272., 8., window, cx);
        // What the slider sets and its value, on hover and on keyboard focus.
        let tile_tip = format!("Thumbnail size · {:.0} px", self.ds_tile);
        let tile_ring = self.ring(&self.fh("ds-tile", cx), window);
        // Fit shows every pixel (boxes at the edge stay visible); Fill crops for a denser look.
        let fit_seg = self.segmented(
            "ds-fit",
            &["Fit", "Fill"],
            usize::from(self.ds_fill),
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.ds_fill = i == 1;
                cx.notify();
            }),
        );
        let (total, of) = match self.ds_split {
            Some(Split::Train) => (ds.splits.train as usize, "train images"),
            Some(Split::Val) => (ds.splits.val as usize, "val images"),
            Some(Split::Test) => (ds.splits.test as usize, "test images"),
            None => (total, "images"),
        };
        let count_label = if count == total {
            format!("{} {of}", int(total as u64))
        } else {
            format!("{} of {} {of} match", int(count as u64), int(total as u64))
        };
        // The split filter joins the view controls on the tab line; the class chips get one row
        // below it, with the result count at its end.
        let toolbar = div()
            .min_h(px(32.))
            .flex_wrap()
            .mb(px(12.))
            .flex()
            .items_center()
            .gap(px(6.))
            .child(text(ty::SM, t.text_tertiary).mr(px(2.)).child("Classes · box counts"))
            .children(chips)
            .when(self.ds_empty_only, |d| {
                d.child(self.chip(
                    "ds-empty-chip",
                    "Empty images ×",
                    true,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_empty_only = false;
                        cx.notify();
                    }),
                ))
            })
            .child(div().flex_1())
            .child(text(ty::SM, t.text_tertiary).tnum().whitespace_nowrap().child(count_label));
        let view_controls = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .child(seg)
            .child(div().w(px(1.)).h(px(20.)).bg(t.line_subtle))
            .child(fit_seg)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_secondary).child("Boxes"))
                    .child(boxes_switch),
            )
            // Thumbnail size, not image zoom: a small and a large tile at the ends.
            .child(
                div()
                    .id("ds-tile-size")
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .w(px(140.))
                    .flex_none()
                    .relative()
                    .tooltip(crate::controls::tip(tile_tip.clone(), Some("← →")))
                    .when(tile_ring, |d| {
                        d.child(crate::controls::focus_tip(&t, format!("{tile_tip} · ← → adjust"), 20.))
                    })
                    .child(div().w(px(14.)).flex().justify_center().child(icon(
                        "icons/image.svg",
                        9.,
                        t.text_tertiary,
                    )))
                    .child(size_slider)
                    .child(icon("icons/image.svg", 14., t.text_tertiary)),
            );
        self.ds_view_controls = Some(view_controls.into_any_element());
        if self.store.images.get(&id).is_none_or(|l| l.is_loading()) {
            return div()
                .flex()
                .flex_col()
                .flex_1()
                .child(toolbar)
                .child(div().flex_1().child(skeleton(&t, "grid")))
                .into_any_element();
        }
        if count == 0 {
            return div()
                .flex()
                .flex_col()
                .flex_1()
                .child(toolbar)
                .child(panel(&t).flex_1().flex().items_center().justify_center().child(empty_state(
                    &t,
                    "icons/list-filter.svg",
                    "No images match",
                    "Try another split or clear the class filter.",
                    Tone::Neutral,
                )))
                .into_any_element();
        }
        let tile = self.ds_tile;
        let width = f32::from(self.grid_width.get());
        let cols = (((width + GAP) / (tile + GAP)).floor() as usize).max(1);
        let rows = count.div_ceil(cols);
        let cell_w = (width - GAP * (cols as f32 - 1.)) / cols as f32;
        let row_h = cell_w * 0.75 + GAP + 1.;
        let width_cell = self.grid_width.clone();
        let ds_id: SharedString = id.clone().into();
        // Class names, so a tile can say what it holds, not only how many boxes.
        let class_names: Vec<String> =
            self.store.dataset_summary(&id).map(|d| d.classes.clone()).unwrap_or_default();
        let fit = if self.ds_fill { crate::viz::Fit::Cover } else { crate::viz::Fit::Contain };
        // A focused image tile names its keys (decided here: the list's rows render after the
        // page has handed its hint to the strip).
        if self.kb && self.focus.borrow().iter().any(|(k, f)| k.starts_with("tile-") && f.is_focused(window))
        {
            self.hint_keys("Arrows move between images · Enter opens one in the inspector");
        }
        let grid = uniform_list(
            "ds-grid",
            rows,
            cx.processor(move |this, range: std::ops::Range<usize>, window, cx| {
                let t = theme(cx);
                // Clone only the visible rows' metadata.
                let visible: Vec<Option<ImageInfo>> = (range.start * cols..range.end * cols)
                    .map(|i| this.filtered_image(&ds_id, i).cloned())
                    .collect();
                let first = range.start * cols;
                let mut out = Vec::new();
                for r in range {
                    let mut row = div().flex().gap(px(GAP)).h(px(row_h)).pb(px(GAP));
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let Some(meta) = visible.get(idx - first).and_then(|m| m.as_ref()) else {
                            row = row.child(div().w(px(cell_w)).flex_none());
                            continue;
                        };
                        let key = format!("tile-{}", meta.id);
                        let reveal = this.fh(key.clone(), cx).is_focused(window)
                            || this.ds_hover_tile.as_deref() == Some(meta.id.as_str());
                        // Filtered, each tile at rest names its largest match (see `tile`).
                        let tile_el = this.tile(&ds_id, meta, this.ds_boxes, true, fit, reveal, cx);
                        // "2 square · 1 triangle · +2": the two commonest classes, then the rest as a
                        // count, so the box colours aren't the only way to tell what a tile holds.
                        let (what, all) = {
                            let mut counts: Vec<(u32, usize)> = Vec::new();
                            for b in &meta.boxes {
                                match counts.iter_mut().find(|(c, _)| *c == b.class) {
                                    Some((_, k)) => *k += 1,
                                    None => counts.push((b.class, 1)),
                                }
                            }
                            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                            let name = |c: u32| {
                                class_names.get(c as usize).cloned().unwrap_or_else(|| format!("class {c}"))
                            };
                            let named: Vec<String> =
                                counts.iter().map(|(c, k)| format!("{k} {}", name(*c))).collect();
                            let mut parts: Vec<String> = named.iter().take(2).cloned().collect();
                            let rest: usize = counts.iter().skip(2).map(|(_, k)| k).sum();
                            if rest > 0 {
                                parts.push(format!("+{rest}"));
                            }
                            // The strip keeps two; the tile's tip lists every class it holds.
                            let all = format!(
                                "{} · {}",
                                meta.id,
                                if named.is_empty() { "no boxes".into() } else { named.join(" · ") }
                            );
                            (if parts.is_empty() { "no boxes".to_string() } else { parts.join(" · ") }, all)
                        };
                        let dsid = ds_id.to_string();
                        let focused = this.fh(key.clone(), cx).is_focused(window);
                        let el = div()
                            .id(SharedString::from(key.clone()))
                            // Everything it holds, by class, on hover and keyboard focus (labels a
                            // tile can't fit, a tiny box's class included).
                            .tooltip(crate::controls::tip(all.clone(), Some("Enter opens")))
                            .when(focused, |d| {
                                d.child(crate::controls::focus_tip(&t, all.clone(), row_h - GAP))
                            })
                            .group(SharedString::from(key.clone()))
                            .w(px(cell_w))
                            .h_full()
                            .flex_none()
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .border_1()
                            .border_color(folo_ui::transparent_black())
                            // Over an image a fill wouldn't show: the edge answers instead.
                            .hover(|s| s.border_color(t.line_strong))
                            .active(|s| s.border_color(t.text_tertiary))
                            .on_hover({
                                let mid = meta.id.clone();
                                cx.listener(move |this, h: &bool, _, cx| {
                                    let next = h.then(|| mid.clone());
                                    if *h || this.ds_hover_tile.as_deref() == Some(mid.as_str()) {
                                        this.ds_hover_tile = next;
                                        cx.notify();
                                    }
                                })
                            })
                            .child(
                                div()
                                    .absolute()
                                    .inset_0()
                                    .rounded(px(radius::MD))
                                    .overflow_hidden()
                                    .bg(if this.ds_fill {
                                        t.bg_inset
                                    } else {
                                        crate::theme::over(t.bg_panel, t.bg_active)
                                    })
                                    .child(tile_el)
                                    .child(
                                        div()
                                            .absolute()
                                            .bottom_0()
                                            .left_0()
                                            .right_0()
                                            .h(px(28.))
                                            .px(px(8.))
                                            .flex()
                                            .items_center()
                                            .justify_between()
                                            .bg(folo_ui::linear_gradient(
                                                180.,
                                                folo_ui::linear_color_stop(folo_ui::hsla(0., 0., 0., 0.), 0.),
                                                folo_ui::linear_color_stop(
                                                    folo_ui::hsla(0., 0., 0., 0.7),
                                                    1.,
                                                ),
                                            ))
                                            .when(!focused, |d| d.invisible())
                                            .group_hover(SharedString::from(key.clone()), |s| s.visible())
                                            .child(
                                                text(ty::SM, folo_ui::white())
                                                    .font_family(crate::theme::MONO)
                                                    .child(meta.id.clone()),
                                            )
                                            .child(
                                                text(ty::SM, folo_ui::hsla(0., 0., 1., 0.75))
                                                    .tnum()
                                                    .ellipsis()
                                                    .child(what),
                                            ),
                                    ),
                            );
                        // Arrow keys walk the grid; the list scrolls the target row into view.
                        let nav_ds = ds_id.clone();
                        let el = el.on_key_down(cx.listener(move |this, ev: &KeyDownEvent, w, cx| {
                            let target = match ev.keystroke.key.as_str() {
                                "left" => idx.checked_sub(1),
                                "right" => Some(idx + 1),
                                "up" => idx.checked_sub(cols),
                                "down" => Some(idx + cols),
                                _ => return,
                            };
                            cx.stop_propagation();
                            let Some(j) = target.filter(|&j| j < count) else { return };
                            let Some(next) = this.filtered_image(&nav_ds, j).map(|m| m.id.clone()) else {
                                return;
                            };
                            let strategy = if j < idx {
                                folo_ui::ScrollStrategy::Top
                            } else {
                                folo_ui::ScrollStrategy::Bottom
                            };
                            this.grid_scroll.scroll_to_item(j / cols, strategy);
                            this.kb = true;
                            let fh = this.fh(format!("tile-{next}"), cx);
                            w.focus(&fh);
                            cx.notify();
                        }));
                        let el = this.bind(
                            el,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.open_inspector(dsid.clone(), idx, w, cx)),
                        );
                        row = row.child(el);
                    }
                    out.push(row.into_any_element());
                }
                out
            }),
        )
        .track_scroll(self.grid_scroll.clone())
        // A scroll moves tiles under a still pointer (and recycles the hovered one away), which
        // `on_hover` never hears: drop the hovered tile, so the next pointer move sets it afresh.
        .on_scroll_wheel(cx.listener(|this, _, _, cx| {
            if this.ds_hover_tile.take().is_some() {
                cx.notify();
            }
        }))
        .flex_1();
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h_0()
            .child(toolbar)
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .child(
                        canvas(move |b, _, _| width_cell.set(b.size.width), |_, _, _, _| {})
                            .absolute()
                            .top_0()
                            .left_0()
                            .right_0()
                            .h(px(1.)),
                    )
                    .child(grid),
            )
            .into_any_element()
    }
}
