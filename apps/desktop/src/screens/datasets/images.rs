//! The image browser: one toolbar (the split, the class, the count; Boxes, the size, Fit / Fill)
//! and straight under it the virtualised grid of a dataset's images.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    canvas, div, px, uniform_list, AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent,
    ParentElement, SharedString, StatefulInteractiveElement, Styled, Window,
};

use crate::api::{Dataset, ImageInfo, Split, SplitsExt};
use crate::app::{act, FoloApp};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

const GAP: f32 = 8.;
/// The thumbnail widths behind S / M / L.
const TILE_SIZES: [f32; 3] = [120., 168., 240.];

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
        let split_n = [ds.splits.total(), ds.splits.train, ds.splits.val, ds.splits.test];
        // Each split with its images; a dataset without a test split offers none. A no-break space
        // marks the count for the segmented control's quieter style.
        let labels: Vec<String> = ["All", "Train", "Val", "Test"]
            .iter()
            .zip(split_n)
            .map(|(s, n)| format!("{s}\u{a0}{}", int(n)))
            .collect();
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
        // The class filter: a select that holds 80 classes as well as 3, each with its swatch and
        // its boxes (it filters images with one).
        let mut options: Vec<(SharedString, Option<SharedString>)> =
            vec![("All classes".into(), Some(counted(counts.iter().sum(), "box", "boxes").into()))];
        options.extend(
            ds.classes.iter().enumerate().map(|(i, name)| {
                (name.clone().into(), Some(int(counts.get(i).copied().unwrap_or(0)).into()))
            }),
        );
        let classes: Vec<Option<u32>> =
            std::iter::once(None).chain((0..ds.classes.len() as u32).map(Some)).collect();
        let picked = self.ds_class.filter(|&c| (c as usize) < ds.classes.len());
        let current: SharedString =
            picked.map_or("All classes".into(), |c| ds.classes[c as usize].clone().into());
        // The 960 px window: a narrower picker and tighter gaps, so the count keeps its room.
        let narrow = window.viewport_size().width < px(1120.);
        let picker = div()
            .id("ds-class-pick")
            .w(px(if narrow { 160. } else { 184. }))
            .flex_none()
            .tooltip(crate::controls::tip("Show only images with a box of this class", None))
            .child(self.class_select(
                "ds-class",
                current,
                options,
                &classes,
                picked.map_or(0, |c| c as usize + 1),
                window,
                cx,
                Rc::new(|this, i, _, cx| {
                    this.ds_class = i.checked_sub(1).map(|c| c as u32);
                    // Empty images have no class: the two filters can't both apply.
                    if this.ds_class.is_some() {
                        this.ds_empty_only = false;
                    }
                    this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                    cx.notify();
                }),
            ));
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
        // Thumbnail size, not image zoom: three steps.
        let size_at = TILE_SIZES.iter().position(|&w| w >= self.ds_tile).unwrap_or(1);
        let size_seg = self
            .segmented(
                "ds-size",
                &["S", "M", "L"],
                size_at,
                window,
                cx,
                Rc::new(|this, i, _, cx| {
                    this.ds_tile = TILE_SIZES[i.min(2)];
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip("Thumbnail size", None));
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
        let loading = self.store.images.get(&id).is_none_or(|l| l.is_loading());
        // A large list arrives a page at a time: the grid shows what came, and the count says
        // how much of it is in (never "0 images" or a split's partial count as if final).
        let more = self.images_pending(&id);
        let count_label = if loading {
            let total = if self.ds_split.is_none() { ds.splits.total() } else { total as u64 };
            format!("Loading {} {of}…", int(total))
        } else if more {
            let loaded = self.store.images.get(&id).and_then(|l| l.ready()).map_or(0, |v| v.len());
            format!("Loading… {} of {} images", int(loaded as u64), int(ds.splits.total()))
        } else if count == total {
            format!("{} {of}", int(total as u64))
        } else {
            format!("{} of {} {of}", int(count as u64), int(total as u64))
        };
        // One toolbar over the grid: what is shown (the split, the class, the count) on the left,
        // how it is drawn (Boxes, the size, Fit / Fill) on the right.
        let toolbar = div()
            .h(px(tsize::CONTROL_LG))
            .mb(px(12.))
            .flex()
            .items_center()
            .gap(px(if narrow { 8. } else { 12. }))
            .child(seg)
            // Empty images have no class, so the two filters never both apply: while "Show them"
            // (Statistics) holds the empty images, its chip takes the class picker's place.
            .when(!self.ds_empty_only, |d| d.child(picker))
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
            // A count cut short says itself in full on hover.
            .child(
                div()
                    .id("ds-count")
                    .min_w_0()
                    .tooltip(crate::controls::tip(count_label.clone(), None))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .whitespace_nowrap()
                            .ellipsis()
                            .child(count_label),
                    ),
            )
            .child(div().flex_1())
            .child(
                div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_secondary).child("Boxes"))
                    .child(boxes_switch),
            )
            .child(size_seg)
            .child(fit_seg);
        if loading || (more && count == 0) {
            return div()
                .flex()
                .flex_col()
                .flex_1()
                .child(toolbar)
                .child(div().flex_1().child(skeleton(&t, "grid")))
                .into_any_element();
        }
        if count == 0 {
            // A dataset with no image at all is not a filter that matched nothing.
            let (icon, title, body) = if ds.splits.total() == 0 {
                let broken = ds.stats.as_ref().is_some_and(|s| s.broken_labels > 0);
                (
                    "icons/image.svg",
                    "No usable images",
                    if broken {
                        "FOLO found no image it can read with its labels here. Broken labels above says what's wrong."
                    } else {
                        "FOLO found no image in this dataset's splits."
                    },
                )
            } else {
                ("icons/list-filter.svg", "No images match", "Try another split or clear the class filter.")
            };
            return div()
                .flex()
                .flex_col()
                .flex_1()
                .child(toolbar)
                .child(panel(&t).flex_1().flex().items_center().justify_center().child(empty_state(
                    &t,
                    icon,
                    title,
                    body,
                    Tone::Neutral,
                )))
                .into_any_element();
        }
        let cols = self.grid_cols();
        let width = f32::from(self.grid_width.get());
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
                                file_name(&meta.file),
                                if named.is_empty() { "no boxes".into() } else { named.join(" · ") }
                            );
                            (if parts.is_empty() { "no boxes".to_string() } else { parts.join(" · ") }, all)
                        };
                        let dsid = ds_id.to_string();
                        let focused = this.fh(key.clone(), cx).is_focused(window);
                        let el = div()
                            .id(SharedString::from(key.clone()))
                            // Everything it holds, by class, on hover and keyboard focus (labels a
                            // tile can't fit, a tiny box's class included). The focus tip is the
                            // keyboard's: not after a click, and gone once the pointer moves on.
                            .tooltip(crate::controls::tip(all.clone(), Some("Enter opens")))
                            .when(focused && this.kb && !this.pointer_moved, |d| {
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
                                            .gap(px(8.))
                                            // The file's name, as Studio's grid shows it (a real
                                            // server's ids are list positions, "6", not names).
                                            .child(
                                                text(ty::SM, folo_ui::white())
                                                    .font_family(crate::theme::MONO)
                                                    .min_w_0()
                                                    .ellipsis()
                                                    .child(file_name(&meta.file).to_string()),
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

impl FoloApp {
    /// Columns of the dataset's image grid at its last laid-out width.
    pub(crate) fn grid_cols(&self) -> usize {
        let width = f32::from(self.grid_width.get());
        (((width + GAP) / (self.ds_tile + GAP)).floor() as usize).max(1)
    }
}

/// An image's file name without its folders ("images/train/000001.jpg" -> "000001.jpg"), as Studio's
/// grid names a tile; either slash, since a server on Windows may answer with backslashes.
fn file_name(file: &str) -> &str {
    file.rsplit(['/', '\\']).find(|s| !s.is_empty()).unwrap_or(file)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_tile_is_named_by_its_file_not_its_id() {
        assert_eq!(super::file_name("images/train/000000014007.jpg"), "000000014007.jpg");
        assert_eq!(super::file_name(r"images\val\a b.png"), "a b.png");
        assert_eq!(super::file_name("one.jpg"), "one.jpg");
        assert_eq!(super::file_name(""), "");
    }
}
