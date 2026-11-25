//! The image browser: one toolbar (the split, the class, the count; Boxes, the size, Fit / Fill)
//! and straight under it the virtualised grid of a dataset's images.

use std::rc::Rc;
use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, canvas, div, px, uniform_list,
};

use crate::api::{Dataset, ImageInfo, Split, SplitsExt};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
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
        // Keep the cached query synced even if a caller updated the input before subscription delivery.
        let current_query = self.ds_img_search.read(cx).text().to_string();
        if self.ds_img_query != current_query {
            self.ds_img_query = current_query;
        }
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
                this.ds_hover_tile = None;
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
        let picked = self.class_filter(&id);
        let current: SharedString =
            picked.map_or("All classes".into(), |c| ds.classes[c as usize].clone().into());
        // The 960 px window: a narrower picker and tighter gaps, so the count keeps its room.
        let narrow = window.viewport_size().width < px(1120.);
        let picker = div()
            .id("ds-class-pick")
            .w(px(if narrow { 112. } else { 128. }))
            .flex_none()
            .tooltip(crate::controls::tip("Show only images with a box of this class", None))
            .child(self.class_select(
                "ds-class",
                current,
                options,
                &classes,
                // The class stored for this dataset, even past its classes: then no option is the
                // choice, and "All classes" (shown) clears it by pointer or keys.
                self.ds_class.as_ref().filter(|(of, _)| *of == id).map_or(0, |(_, c)| *c as usize + 1),
                window,
                cx,
                Rc::new({
                    let id = id.clone();
                    move |this, i, _, cx| {
                        match i.checked_sub(1) {
                            Some(c) => {
                                this.ds_class = Some((id.clone(), c as u32));
                                // Empty images have no class: the two filters can't both apply.
                                if this.empty_only(&id) {
                                    this.ds_empty_of = None;
                                }
                            }
                            // "All classes" here clears this dataset's class, not another's.
                            None if this.ds_class.as_ref().is_some_and(|(of, _)| *of == id) => {
                                this.ds_class = None
                            }
                            None => {}
                        }
                        this.ds_hover_tile = None;
                        this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                        cx.notify();
                    }
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
                    this.ds_hover_tile = None;
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
            Some(Split::Train) => (ds.splits.train as usize, "train image"),
            Some(Split::Val) => (ds.splits.val as usize, "val image"),
            Some(Split::Test) => (ds.splits.test as usize, "test image"),
            None => (total, "image"),
        };
        // "1 image", "2 val images": the noun follows the count it closes.
        let of = |n: usize| if n == 1 { of.to_string() } else { format!("{of}s") };
        let loading = self.store.images.get(&id).is_none_or(|l| l.is_loading());
        // A large list arrives a page at a time: the grid shows what came, and the count says
        // how much of it is in (never "0 images" or a split's partial count as if final).
        let more = self.images_pending(&id);
        let failed_page = self.failed_image_page(&id).cloned();
        let count_label = if loading {
            let total = if self.ds_split.is_none() { ds.splits.total() } else { total as u64 };
            format!("Loading {} {}…", int(total), of(total as usize))
        } else if more {
            let loaded = self.store.images.get(&id).and_then(|l| l.ready()).map_or(0, |v| v.len());
            let all = ds.splits.total();
            let noun = if all == 1 { "image" } else { "images" };
            format!("Loading… {} of {} {noun}", int(loaded as u64), int(all))
        } else if failed_page.is_some() {
            let loaded = self.store.images.get(&id).and_then(|l| l.ready()).map_or(0, |v| v.len());
            let all = ds.splits.total();
            let noun = if all == 1 { "image" } else { "images" };
            format!("{} of {} {noun} loaded", int(loaded as u64), int(all))
        } else if count == total {
            format!("{} {}", int(total as u64), of(total))
        } else {
            format!("{} of {} {}", int(count as u64), int(total as u64), of(total))
        };
        let failed_banner = if let Some(failed) = failed_page.as_ref() {
            self.passing_stop("ds-page-retry", true, cx);
            let retry_id = id.clone();
            let msg = format!(
                "Couldn't load remaining images from offset {}: {}",
                int(failed.offset as u64),
                failed.error.message
            );
            let retry = self.button(
                "ds-page-retry",
                "Retry",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.retry_failed_image_page(&retry_id, cx)),
            );
            Some(
                div()
                    .mb(px(12.))
                    .px(px(12.))
                    .py(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.warning_muted)
                    .border_1()
                    .border_color(crate::theme::alpha(t.warning_solid, 0.35))
                    .child(icon("icons/circle-alert.svg", 14., t.warning_text))
                    .child(
                        div()
                            .id("ds-page-error")
                            .flex_1()
                            .min_w_0()
                            .tooltip(crate::controls::tip(msg.clone(), None))
                            .child(text(ty::SM, t.warning_text).ellipsis().child(msg)),
                    )
                    .child(retry)
                    .into_any_element(),
            )
        } else {
            // Keep the Retry button's handle in tab order as a non-stop after activation so the
            // next Tab continues from here instead of restarting at the top of the window.
            self.passing_stop("ds-page-retry", false, cx).map(|fh| div().track_focus(&fh).into_any_element())
        };
        // One single-row toolbar over the grid: filters/search/sort/reset on the left, display
        // options (Boxes, size, Fit / Fill) on the right; the count shrinks cleanly with full hover tip.
        let img_search = self.search_field(
            "ds-img-search",
            self.ds_img_search.clone(),
            if narrow { 104. } else { 124. },
            window,
            cx,
        );
        let sort_opts: Vec<(SharedString, Option<SharedString>)> =
            super::ImageSort::ALL.iter().map(|s| (s.label().into(), None)).collect();
        let sort_sel = super::ImageSort::ALL.iter().position(|s| *s == self.ds_img_sort).unwrap_or(0);
        let img_sort = div().w(px(if narrow { 96. } else { 104. })).flex_none().child(self.compact_select(
            "ds-img-sort",
            self.ds_img_sort.label().into(),
            sort_opts,
            sort_sel,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.ds_img_sort = super::ImageSort::ALL[i];
                this.ds_hover_tile = None;
                this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                cx.notify();
            }),
        ));
        let filters_active = self.dataset_image_filters_active(&id);
        let reset_stop = self.passing_stop("ds-img-reset", filters_active, cx);
        let reset_btn = if filters_active {
            let reset_id = id.clone();
            Some(
                self.button(
                    "ds-img-reset",
                    "Reset",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(move |this, _, cx| this.reset_dataset_image_filters(&reset_id, cx)),
                )
                .tooltip(crate::controls::tip("Reset split, class, search and sort", None))
                .into_any_element(),
            )
        } else {
            reset_stop.map(|fh| div().track_focus(&fh).into_any_element())
        };
        let toolbar = div()
            .h(px(tsize::CONTROL_LG))
            .mb(px(12.))
            .flex()
            .items_center()
            .gap(px(if narrow { 6. } else { 8. }))
            .child(seg)
            // Empty images have no class, so the two filters never both apply: while "Show them"
            // (Statistics) holds the empty images, its chip takes the class picker's place.
            .when(!self.empty_only(&id), |d| d.child(picker))
            .when(self.empty_only(&id), |d| {
                d.child(self.chip(
                    "ds-empty-chip",
                    "Empty images ×",
                    true,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_empty_of = None;
                        this.ds_hover_tile = None;
                        cx.notify();
                    }),
                ))
            })
            .child(img_search)
            .child(img_sort)
            .children(reset_btn)
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
                .when_some(failed_banner, |d, b| d.child(b))
                .child(div().flex_1().child(skeleton(&t, "grid")))
                .into_any_element();
        }
        if count == 0 {
            // A dataset with no image at all is not a filter that matched nothing.
            let (icon, title, body, show_reset) = if ds.splits.total() == 0 {
                let broken = ds.stats.as_ref().is_some_and(|s| s.broken_labels > 0);
                (
                    "icons/image.svg",
                    "No usable images",
                    if broken {
                        "FOLO found no image it can read with its labels here. Broken labels above says what's wrong."
                    } else {
                        "FOLO found no image in this dataset's splits."
                    },
                    false,
                )
            } else {
                (
                    "icons/list-filter.svg",
                    "No images match",
                    "Try another split, class, or filename search.",
                    true,
                )
            };
            let empty_reset = show_reset.then(|| {
                let reset_id = id.clone();
                self.button(
                    "ds-img-empty-reset",
                    "Reset filters",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(move |this, w, cx| {
                        this.reset_dataset_image_filters(&reset_id, cx);
                        w.focus(&folo_ui::Focusable::focus_handle(this.ds_img_search.read(cx), cx));
                    }),
                )
            });
            return div()
                .flex()
                .flex_col()
                .flex_1()
                .child(toolbar)
                .when_some(failed_banner, |d, b| d.child(b))
                .child(
                    panel(&t)
                        .flex_1()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(empty_state(&t, icon, title, body, Tone::Neutral).children(empty_reset)),
                )
                .into_any_element();
        }
        let cols = self.grid_cols();
        let width = f32::from(self.grid_width.get());
        let rows = count.div_ceil(cols);
        let cell_w = (width - GAP * (cols as f32 - 1.)) / cols as f32;
        let row_h = cell_w * 0.75 + GAP + 1.;
        let width_cell = self.grid_width.clone();
        let ds_id: SharedString = id.clone().into();
        // Pre-interned class names, shared across visible tiles and box chips without per-box allocs.
        let class_names: Arc<[SharedString]> = ds.classes.iter().cloned().map(SharedString::from).collect();
        let fit = if self.ds_fill { crate::viz::Fit::Cover } else { crate::viz::Fit::Contain };
        // A focused image tile names its keys (decided here: the list's rows render after the
        // page has handed its hint to the strip).
        if self.kb && self.focus.borrow().iter().any(|(k, f)| k.starts_with("tile-") && f.is_focused(window))
        {
            self.hint_keys("Arrows move between images · Enter opens one in the inspector");
        }
        let wheel_ds = ds_id.clone();
        let grid = uniform_list(
            "ds-grid",
            rows,
            cx.processor(move |this, range: std::ops::Range<usize>, window, cx| {
                let t = theme(cx);
                let app = cx.entity().downgrade();
                let (grid_bounds, scroll_y, item_h) = {
                    let state = this.grid_scroll.0.borrow();
                    let handle = &state.base_handle;
                    (handle.bounds(), handle.offset().y, state.last_item_height.unwrap_or(px(row_h)))
                };
                let mouse_pos = window.mouse_position();
                // Look up the filtered index slice and image vec once per visible range rather than
                // per tile, and clone only the visible rows' metadata.
                let first = range.start * cols;
                let visible: Vec<Option<(usize, ImageInfo)>> = {
                    let indices = this.filtered_indices(&ds_id);
                    let all_images = this.store.images.get(ds_id.as_ref()).and_then(|l| l.ready());
                    (first..range.end * cols)
                        .map(|i| {
                            let raw_idx = *indices.get(i)?;
                            let meta = all_images?.get(raw_idx)?.clone();
                            Some((raw_idx, meta))
                        })
                        .collect()
                };
                let mut out = Vec::with_capacity(range.len());
                for r in range {
                    let mut row = div().flex().gap(px(GAP)).h(px(row_h)).pb(px(GAP));
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let Some((raw_idx, meta)) = visible.get(idx - first).and_then(|m| m.as_ref()) else {
                            row = row.child(div().w(px(cell_w)).flex_none());
                            continue;
                        };
                        let raw_idx = *raw_idx;
                        let key = format!("tile-{}", meta.id);
                        let key_shared = SharedString::from(key.clone());
                        let img_id = SharedString::from(meta.id.clone());
                        let focused = this.fh(key.clone(), cx).is_focused(window);
                        let hovered = this.ds_hover_tile.as_deref().and_then(parse_hover_tile).is_some_and(
                            |(h_ds, _, h_id)| h_ds == ds_id.as_ref() && h_id == meta.id.as_str(),
                        ) && tile_contains_point(
                            grid_bounds,
                            scroll_y,
                            idx,
                            cols,
                            cell_w,
                            item_h,
                            mouse_pos,
                        );
                        let reveal = focused || hovered;
                        // Filtered, each tile at rest names its largest match (see `tile`).
                        let tile_el = this.tile_with_classes(
                            &ds_id,
                            meta,
                            Some(&class_names),
                            this.ds_boxes,
                            true,
                            fit,
                            reveal,
                            cx,
                        );
                        let dsid = ds_id.clone();
                        let el = div()
                            .id(key_shared)
                            // Everything it holds, by class, on hover and keyboard focus (labels a
                            // tile can't fit, a tiny box's class included). Built lazily on hover so
                            // idle tiles allocate no class-count strings. The focus tip is the
                            // keyboard's: not after a click, and gone once the pointer moves on.
                            .tooltip({
                                let app = app.clone();
                                let ds_id = ds_id.clone();
                                let img_id = img_id.clone();
                                let class_names = class_names.clone();
                                move |w, cx| {
                                    let text: SharedString = app
                                        .upgrade()
                                        .and_then(|entity| {
                                            let this = entity.read(cx);
                                            let imgs = this.store.images.get(ds_id.as_ref())?.ready()?;
                                            let meta = imgs
                                                .get(raw_idx)
                                                .filter(|m| m.id == img_id.as_ref())
                                                .or_else(|| imgs.iter().find(|m| m.id == img_id.as_ref()))?;
                                            Some(SharedString::from(tile_tip(meta, &class_names)))
                                        })
                                        .unwrap_or_default();
                                    crate::controls::tip(text, Some("Enter opens"))(w, cx)
                                }
                            })
                            .when(focused && this.kb && !this.pointer_moved, |d| {
                                let all = tile_tip(meta, &class_names);
                                d.child(crate::controls::focus_tip(&t, all, row_h - GAP))
                            })
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
                                let ds_id = ds_id.clone();
                                let mid = img_id;
                                cx.listener(move |this, h: &bool, _, cx| {
                                    if *h {
                                        this.ds_hover_tile = Some(format!("{ds_id}\0{idx}\0{mid}"));
                                        cx.notify();
                                    } else if this
                                        .ds_hover_tile
                                        .as_deref()
                                        .and_then(parse_hover_tile)
                                        .is_some_and(|(h_ds, _, h_id)| {
                                            h_ds == ds_id.as_ref() && h_id == mid.as_ref()
                                        })
                                    {
                                        this.ds_hover_tile = None;
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
                                    // Build the caption strip only while the tile is hovered or
                                    // focused (skipping hidden overlay divs + text layout on every
                                    // idle tile).
                                    .when(reveal, |d| {
                                        let what = tile_caption(meta, &class_names);
                                        d.child(
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
                                                    folo_ui::linear_color_stop(
                                                        folo_ui::hsla(0., 0., 0., 0.),
                                                        0.,
                                                    ),
                                                    folo_ui::linear_color_stop(
                                                        folo_ui::hsla(0., 0., 0., 0.7),
                                                        1.,
                                                    ),
                                                ))
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
                                        )
                                    }),
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
                            act(move |this, w, cx| this.open_inspector(dsid.to_string(), idx, w, cx)),
                        );
                        row = row.child(el);
                    }
                    out.push(row.into_any_element());
                }
                out
            }),
        )
        .track_scroll(self.grid_scroll.clone())
        // A scroll moves tiles under a still pointer (and can recycle the hovered one away), which
        // `on_hover` never hears. Keep `ds_hover_tile` in lockstep with `folo-ui`'s mounted
        // `hover_state`: clear it only once the hovered tile's row scrolls out of the visible range
        // (unmounting its element state), and otherwise notify so `tile_contains_point` hides or
        // restores the caption as the tile scrolls off or back under the pointer.
        .on_scroll_wheel(cx.listener({
            let ds_id = wheel_ds;
            move |this, _, _, cx| {
                let Some(hovered) = this.ds_hover_tile.as_deref() else { return };
                let Some((hover_ds, cached_idx, hover_img)) = parse_hover_tile(hovered) else {
                    this.ds_hover_tile = None;
                    cx.notify();
                    return;
                };
                if hover_ds != ds_id.as_ref() {
                    this.ds_hover_tile = None;
                    cx.notify();
                    return;
                }
                let (grid_bounds, new_y, item_h) = {
                    let state = this.grid_scroll.0.borrow();
                    let handle = &state.base_handle;
                    let max_y = handle.max_offset().height;
                    (
                        handle.bounds(),
                        handle.offset().y.clamp(-max_y, px(0.)),
                        state.last_item_height.unwrap_or(px(row_h)),
                    )
                };
                let stays_mounted = {
                    let indices = this.filtered_indices(&ds_id);
                    let all_images = this.store.images.get(ds_id.as_ref()).and_then(|l| l.ready());
                    all_images
                        .and_then(|imgs| {
                            indices
                                .get(cached_idx)
                                .filter(|&&raw| imgs.get(raw).is_some_and(|m| m.id == hover_img))
                                .map(|_| cached_idx)
                                .or_else(|| {
                                    indices
                                        .iter()
                                        .position(|&raw| imgs.get(raw).is_some_and(|m| m.id == hover_img))
                                })
                        })
                        .is_some_and(|idx| row_in_viewport(grid_bounds, new_y, idx / cols, item_h))
                };
                if !stays_mounted {
                    this.ds_hover_tile = None;
                }
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
            .when_some(failed_banner, |d, b| d.child(b))
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

/// Box counts per class, sorted most-frequent first with ties broken by class index.
fn tile_class_counts(meta: &ImageInfo) -> Vec<(u32, usize)> {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for b in &meta.boxes {
        match counts.iter_mut().find(|(c, _)| *c == b.class) {
            Some((_, k)) => *k += 1,
            None => counts.push((b.class, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts
}

fn format_class_count(class: u32, count: usize, class_names: &[SharedString]) -> String {
    match class_names.get(class as usize) {
        Some(name) => format!("{count} {name}"),
        None => format!("{count} class {class}"),
    }
}

/// "2 square · 1 triangle · +2": the two commonest classes, then the rest as a count, so the box
/// colours aren't the only way to tell what a hovered or focused tile holds.
fn tile_caption(meta: &ImageInfo, class_names: &[SharedString]) -> String {
    let counts = tile_class_counts(meta);
    if counts.is_empty() {
        return "no boxes".to_string();
    }
    let mut parts: Vec<String> =
        counts.iter().take(2).map(|&(c, k)| format_class_count(c, k, class_names)).collect();
    let rest: usize = counts.iter().skip(2).map(|(_, k)| k).sum();
    if rest > 0 {
        parts.push(format!("+{rest}"));
    }
    parts.join(" · ")
}

/// Full tooltip text for a tile: file name plus every class count it holds.
fn tile_tip(meta: &ImageInfo, class_names: &[SharedString]) -> String {
    let counts = tile_class_counts(meta);
    let summary = if counts.is_empty() {
        "no boxes".to_string()
    } else {
        counts.iter().map(|&(c, k)| format_class_count(c, k, class_names)).collect::<Vec<_>>().join(" · ")
    };
    format!("{} · {summary}", file_name(&meta.file))
}

/// Parses a dataset-scoped hover key `"{ds_id}\0{filtered_idx}\0{img_id}"`.
fn parse_hover_tile(value: &str) -> Option<(&str, usize, &str)> {
    let (ds_id, rest) = value.split_once('\0')?;
    let (idx, img_id) = rest.split_once('\0')?;
    Some((ds_id, idx.parse().ok()?, img_id))
}

/// Whether `row` falls in `UniformList`'s mounted visible range at `scroll_y` with measured `item_h`.
fn row_in_viewport(
    grid_bounds: folo_ui::Bounds<folo_ui::Pixels>,
    scroll_y: folo_ui::Pixels,
    row: usize,
    item_h: folo_ui::Pixels,
) -> bool {
    if grid_bounds.is_empty() || item_h <= px(0.) {
        return false;
    }
    let first = (-scroll_y / item_h).floor().max(0.) as usize;
    let last = ((-scroll_y + grid_bounds.size.height) / item_h).ceil().max(0.) as usize;
    (first..last).contains(&row)
}

/// Whether `pos` lies inside tile `idx`'s on-screen rect within the scrolled grid viewport, using
/// `UniformList`'s measured `item_h` so deep rows never drift from rounded device-pixel layout.
fn tile_contains_point(
    grid_bounds: folo_ui::Bounds<folo_ui::Pixels>,
    scroll_y: folo_ui::Pixels,
    idx: usize,
    cols: usize,
    cell_w: f32,
    item_h: folo_ui::Pixels,
    pos: folo_ui::Point<folo_ui::Pixels>,
) -> bool {
    if cols == 0 || grid_bounds.is_empty() || item_h <= px(0.) || !grid_bounds.contains(&pos) {
        return false;
    }
    let r = idx / cols;
    let c = idx % cols;
    let tile_h = (item_h - px(GAP)).max(px(0.));
    let tile_bounds = folo_ui::Bounds::new(
        folo_ui::point(
            grid_bounds.origin.x + px(c as f32 * (cell_w + GAP)),
            grid_bounds.origin.y + scroll_y + item_h * r as f32,
        ),
        folo_ui::size(px(cell_w), tile_h),
    );
    tile_bounds.contains(&pos)
}

/// An image's file name without its folders ("images/train/000001.jpg" -> "000001.jpg"), as Studio's
/// grid names a tile; either slash, since a server on Windows may answer with backslashes.
pub(crate) fn file_name(file: &str) -> &str {
    file.rsplit(['/', '\\']).find(|s| !s.is_empty()).unwrap_or(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::BBox;

    fn sample_image(file: &str, classes: &[u32]) -> ImageInfo {
        ImageInfo {
            id: "0".into(),
            file: file.into(),
            split: Split::Train,
            width: 640,
            height: 480,
            boxes: classes.iter().map(|&c| BBox { class: c, x1: 0., y1: 0., x2: 10., y2: 10. }).collect(),
        }
    }

    #[test]
    fn a_tile_is_named_by_its_file_not_its_id() {
        assert_eq!(super::file_name("images/train/000000014007.jpg"), "000000014007.jpg");
        assert_eq!(super::file_name(r"images\val\a b.png"), "a b.png");
        assert_eq!(super::file_name("one.jpg"), "one.jpg");
        assert_eq!(super::file_name(""), "");
    }

    #[test]
    fn tile_caption_and_tip_format_counts_and_unknown_classes() {
        let names: Vec<SharedString> = vec!["circle".into(), "square".into(), "triangle".into()];
        let empty = sample_image("images/train/empty.jpg", &[]);
        assert_eq!(tile_caption(&empty, &names), "no boxes");
        assert_eq!(tile_tip(&empty, &names), "empty.jpg · no boxes");

        let single = sample_image("images/train/one.jpg", &[0, 0]);
        assert_eq!(tile_caption(&single, &names), "2 circle");
        assert_eq!(tile_tip(&single, &names), "one.jpg · 2 circle");

        let pair = sample_image("images/train/two.jpg", &[2, 0, 2]);
        assert_eq!(tile_caption(&pair, &names), "2 triangle · 1 circle");
        assert_eq!(tile_tip(&pair, &names), "two.jpg · 2 triangle · 1 circle");

        let mixed = sample_image("images/val/mix.jpg", &[2, 1, 1, 0, 5]);
        // square: 2, circle: 1, triangle: 1, class 5: 1 (ties broken by class id).
        assert_eq!(tile_caption(&mixed, &names), "2 square · 1 circle · +2");
        assert_eq!(tile_tip(&mixed, &names), "mix.jpg · 2 square · 1 circle · 1 triangle · 1 class 5");
    }

    #[test]
    fn tile_contains_point_tracks_boundary_small_and_large_scrolls() {
        let grid = folo_ui::Bounds::new(folo_ui::point(px(20.), px(100.)), folo_ui::size(px(400.), px(300.)));
        let (cols, cell_w, item_h) = (2, 196., px(156.));
        // Tile 0 occupies x in [20, 216), y in [100, 248) at scroll_y = 0.
        let cursor = folo_ui::point(px(100.), px(180.));
        // Boundary scroll (scroll_y stays 0): still inside tile 0.
        assert!(tile_contains_point(grid, px(0.), 0, cols, cell_w, item_h, cursor));
        // Small scroll (-30 px): tile 0 spans y in [70, 218), cursor at 180 is still inside.
        assert!(tile_contains_point(grid, px(-30.), 0, cols, cell_w, item_h, cursor));
        // Larger scroll (-100 px): tile 0 spans y in [0, 148), cursor at 180 is now below it,
        // while row 0 remains mounted in the viewport so scrolling back restores it.
        assert!(!tile_contains_point(grid, px(-100.), 0, cols, cell_w, item_h, cursor));
        assert!(row_in_viewport(grid, px(-100.), 0, item_h));
        // Scrolling past row 0 unmounts it.
        assert!(!row_in_viewport(grid, px(-200.), 0, item_h));
        // Cursor outside the grid viewport never matches.
        assert!(!tile_contains_point(
            grid,
            px(-30.),
            0,
            cols,
            cell_w,
            item_h,
            folo_ui::point(px(100.), px(80.))
        ));
    }

    #[test]
    fn deep_rows_use_measured_item_height_without_rounding_drift() {
        let grid =
            folo_ui::Bounds::new(folo_ui::point(px(20.), px(100.)), folo_ui::size(px(1000.), px(400.)));
        let (cols, cell_w, measured_h) = (5, 193.6, px(154.));
        let row = 750;
        let idx = row * cols + 2;
        // UniformList places row 750 at y = grid_top + scroll_y + measured_h * 750.
        let scroll_y = -(measured_h * row as f32);
        let cursor = folo_ui::point(px(20. + 2. * (cell_w + GAP) + 50.), px(150.));
        assert!(tile_contains_point(grid, scroll_y, idx, cols, cell_w, measured_h, cursor));
        assert!(row_in_viewport(grid, scroll_y, row, measured_h));
        assert_eq!(parse_hover_tile(&format!("ds1\0{idx}\0img-42")), Some(("ds1", idx, "img-42")));
    }
}
