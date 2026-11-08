//! Datasets: the dataset index and a dataset's images (virtualised grid) and statistics.

use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, canvas, div, px, uniform_list,
};

use super::*;
use crate::api::{Dataset, ImageInfo, Split};
use crate::app::{DsTab, FoloApp, act};
use crate::charts::{HoverFn, histogram};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

const GAP: f32 = 8.;

impl FoloApp {
    pub fn datasets(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let list = self.store.datasets.ready().cloned();
        let subtitle = match &list {
            Some(l) => {
                format!(
                    "{} · {} images",
                    plural(l.len(), "dataset"),
                    int(l.iter().map(|d| d.splits.total()).sum())
                )
            }
            None => "Loading…".into(),
        };
        let add = self
            .button(
                "ds-add",
                "Add dataset",
                Btn::Primary,
                Some("icons/folder-open.svg"),
                window,
                cx,
                act(|this, _, cx| this.add_dataset_folder(cx)),
            )
            .into_any_element();
        let synth = self
            .button(
                "ds-synth",
                "Generate synthetic",
                Btn::Secondary,
                Some("icons/zap.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_synth(w, cx)),
            )
            .into_any_element();
        let header = page_header(&t, "Datasets", subtitle, vec![synth, add]);
        if let Some(e) = self.store.datasets.error().cloned() {
            let err = self.error_panel(&e, "ds-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }
        let body = match list {
            None => div().flex().gap(px(16.)).children((0..3).map(|i| div().flex_1().h(px(200.)).child(skeleton(&t, format!("dsc{i}"))))).into_any_element(),
            Some(l) if l.is_empty() => panel(&t)
                .child(empty_state(&t, "icons/images.svg", "No datasets yet", "Point FOLO at a YOLO or COCO folder, or generate a synthetic shapes dataset to try training end to end.", Tone::Neutral))
                .into_any_element(),
            Some(l) => {
                let cards: Vec<AnyElement> = l.iter().map(|d| self.dataset_card(d, window, cx)).collect();
                let mut rows = Vec::new();
                let mut it = cards.into_iter().peekable();
                while it.peek().is_some() {
                    // Equal thirds (basis 0, no min width), so a short last row matches the rows above.
                    let mut row: Vec<AnyElement> = it
                        .by_ref()
                        .take(3)
                        .map(|c| div().flex_grow().flex_basis(px(0.)).min_w_0().flex().child(c).into_any_element())
                        .collect();
                    while row.len() < 3 {
                        row.push(div().flex_grow().flex_basis(px(0.)).min_w_0().into_any_element());
                    }
                    rows.push(div().flex().gap(px(16.)).children(row));
                }
                div().flex().flex_col().gap(px(16.)).children(rows).into_any_element()
            }
        };
        self.page_frame(&t, header, body, cx)
    }

    /// Images of the open dataset after the split and class filters.
    /// Indices (into the dataset's image list) that pass the split and class filters. Cached
    /// until the dataset, its image list or a filter changes, so scrolling never rescans it.
    pub fn filtered_indices(&self, id: &str) -> Rc<Vec<usize>> {
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else { return Rc::new(Vec::new()) };
        let key = (
            id.to_string(),
            self.ds_split,
            self.ds_class,
            self.ds_empty_only,
            all.len(),
            all.as_ptr() as usize,
        );
        if let Some((k, v)) = self.ds_filter_cache.borrow().as_ref()
            && *k == key
        {
            return v.clone();
        }
        let v: Rc<Vec<usize>> = Rc::new(
            all.iter()
                .enumerate()
                .filter(|(_, m)| self.ds_split.is_none_or(|s| m.split == s))
                .filter(|(_, m)| self.ds_class.is_none_or(|c| m.boxes.iter().any(|b| b.class == c)))
                .filter(|(_, m)| !self.ds_empty_only || m.boxes.is_empty())
                .map(|(i, _)| i)
                .collect(),
        );
        *self.ds_filter_cache.borrow_mut() = Some((key, v.clone()));
        v
    }

    /// Boxes per class among a split's images (cached per dataset and split).
    pub fn split_class_counts(&self, id: &str, split: Split, classes: usize) -> Vec<u64> {
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else { return vec![0; classes] };
        let key = (id.to_string(), split, all.len(), all.as_ptr() as usize, classes);
        if let Some((k, v)) = self.ds_split_counts.borrow().as_ref()
            && *k == key
        {
            return v.clone();
        }
        let mut v = vec![0u64; classes];
        for m in all.iter().filter(|m| m.split == split) {
            for b in &m.boxes {
                if let Some(c) = v.get_mut(b.class as usize) {
                    *c += 1;
                }
            }
        }
        *self.ds_split_counts.borrow_mut() = Some((key, v.clone()));
        v
    }

    /// The `n`th filtered image of a dataset.
    pub fn filtered_image(&self, id: &str, n: usize) -> Option<&ImageInfo> {
        let i = *self.filtered_indices(id).get(n)?;
        self.store.images.get(id)?.ready()?.get(i)
    }

    pub fn dataset(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_dataset(id, cx);
        let ds = match self.store.dataset.get(id) {
            Some(Load::Ready(d)) => d.clone(),
            Some(Load::Failed(e)) => {
                let e = e.clone();
                let header = page_header(&t, "Dataset", "", vec![]);
                let err = self.error_panel(&e, "ds-detail-retry", window, cx);
                return self.page_frame(&t, header, err, cx);
            }
            _ => {
                let header = page_header(
                    &t,
                    self.store.dataset_summary(id).map_or("Dataset".to_string(), |d| d.name.clone()),
                    "Loading…",
                    vec![],
                );
                return self.page_frame(&t, header, panel(&t).child(loading_rows(&t, "ds", 6)), cx);
            }
        };
        let did = ds.id.clone();
        let train = self
            .button(
                "ds-train",
                "Train on this dataset",
                Btn::Primary,
                Some("icons/play.svg"),
                window,
                cx,
                act(move |this, w, cx| this.open_new_run(Some(&did), w, cx)),
            )
            .into_any_element();
        let path = ds.path.clone();
        let reveal = self
            .button(
                "ds-reveal",
                "Show in Explorer",
                Btn::Secondary,
                Some("icons/folder-open.svg"),
                window,
                cx,
                act(move |_, _, cx| cx.reveal_path(std::path::Path::new(&path))),
            )
            .into_any_element();
        let title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .child(text(ty::XL, t.text_primary).child(ds.name.clone()))
            .child(badge(&t, ds.format.label(), Tone::Neutral));
        let header = detail_header(
            &t,
            title,
            text(ty::SM, t.text_tertiary)
                .font_family(crate::theme::MONO)
                .truncate()
                .child(display_path(&ds.path)),
            vec![reveal.into_any_element(), train.into_any_element()],
        );

        let strip = self.dataset_strip_stats(&ds, window, cx);
        // A view switch, so underline tabs: the segmented control below is the split filter.
        let tab = |this: &Self,
                   key: &str,
                   label: &str,
                   on: bool,
                   which: DsTab,
                   window: &Window,
                   cx: &mut Context<Self>| {
            let el = div()
                .id(SharedString::from(key.to_string()))
                .h(px(32.))
                .flex()
                .items_center()
                .border_b_2()
                .border_color(if on { t.text_primary } else { gpui::transparent_black() })
                .cursor_pointer()
                .child(
                    div()
                        .id(SharedString::from(format!("{key}-in")))
                        .h(px(26.))
                        .px(px(6.))
                        .flex()
                        .items_center()
                        .rounded(px(radius::SM))
                        .hover(|s| s.bg(t.bg_hover))
                        .active(|s| s.bg(t.bg_active))
                        .child(
                            text(ty::BASE, if on { t.text_primary } else { t.text_secondary })
                                .font_weight(FontWeight::MEDIUM)
                                .child(label.to_string()),
                        ),
                );
            this.bind(
                el,
                key,
                radius::SM,
                window,
                cx,
                act(move |this, _, cx| {
                    this.ds_tab = which;
                    cx.notify();
                }),
            )
        };
        let on_images = self.ds_tab == DsTab::Images;
        let tabs = div()
            .flex()
            .gap(px(12.))
            .child(tab(self, "ds-tab-images", "Images", on_images, DsTab::Images, window, cx))
            .child(tab(self, "ds-tab-stats", "Statistics", !on_images, DsTab::Statistics, window, cx));
        let content = match self.ds_tab {
            DsTab::Images => self.image_browser(&ds, window, cx),
            DsTab::Statistics => self.dataset_stats(&ds, window, cx),
        };
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap(px(16.))
            .px(px(24.))
            .pt(px(20.))
            .pb(px(16.))
            .child(header)
            .child(strip)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(12.))
                    .child(tabs)
                    .children(self.ds_view_controls.take().filter(|_| self.ds_tab == DsTab::Images)),
            )
            .child(div().flex_1().min_h_0().flex().flex_col().child(content))
            .into_any_element()
    }

    fn dataset_strip_stats(&self, ds: &Dataset, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let total = ds.splits.total().max(1) as f32;
        let s = ds.stats.as_ref();
        let split_c = [
            t.text_tertiary,
            crate::theme::alpha(t.text_tertiary, 0.6),
            crate::theme::alpha(t.text_tertiary, 0.35),
        ];
        let split_bar = div()
            .flex()
            .h(px(4.))
            .w_full()
            .rounded_full()
            .overflow_hidden()
            .gap(px(2.))
            .child(div().h_full().w(gpui::relative(ds.splits.train as f32 / total)).bg(split_c[0]))
            .child(div().h_full().w(gpui::relative(ds.splits.val as f32 / total)).bg(split_c[1]))
            .when(ds.splits.test > 0, |d| d.child(div().h_full().flex_1().bg(split_c[2])));
        let legend = div().flex().gap(px(10.)).children(
            [
                ("train", ds.splits.train, split_c[0]),
                ("val", ds.splits.val, split_c[1]),
                ("test", ds.splits.test, split_c[2]),
            ]
            .into_iter()
            .filter(|x| x.1 > 0)
            .map(|(n, c, col)| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(5.))
                    .child(swatch(col, 6.))
                    .child(text(ty::XS, t.text_tertiary).tnum().child(format!("{n} {}", int(c))))
            }),
        );
        let tile = |content: gpui::Div| div().flex_1().min_w_0().px(px(16.)).py(px(12.)).child(content);
        let boxes = s.map_or(0, |s| s.boxes);
        let per = boxes as f32 / total;
        let broken = s.map_or(0, |s| s.broken_labels);
        let narrow_strip = window.viewport_size().width < px(1120.);
        panel(&t)
            .flex()
            .child(
                div().flex_none().w(px(if narrow_strip { 220. } else { 300. })).px(px(16.)).py(px(12.)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .child(caption(&t, "Images"))
                        .child(metric(&t, int(ds.splits.total()), "", ty::X2XL, ty::SM))
                        .child(split_bar)
                        .child(legend),
                ),
            )
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Boxes",
                int(boxes),
                "",
                Some(text(ty::XS, t.text_tertiary).tnum().child(format!("{per:.1} per image"))),
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Classes",
                ds.classes.len().to_string(),
                "",
                Some(text(ty::XS, t.text_tertiary).whitespace_nowrap().child(names_summary(&ds.classes, 24))),
            )))
            .child(vline(&t))
            .child({
                // Defects are actionable: this tile filters the grid to the images in question.
                let empty = s.map_or(0, |s| s.empty_images);
                let on = self.ds_empty_only;
                let el = div()
                    .id("ds-empty-tile")
                    .flex_1()
                    .min_w_0()
                    .px(px(16.))
                    .py(px(12.))
                    .when(empty > 0, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)))
                    .when(on, |d| d.bg(t.bg_selected))
                    .child(stat(
                        &t,
                        "Empty images",
                        int(empty),
                        "",
                        Some(if empty > 0 {
                            div()
                                .flex()
                                .items_center()
                                .gap(px(2.))
                                .child(text(ty::XS, if on { t.accent_text } else { t.text_secondary }).child(if on {
                                    "Showing them · clear"
                                } else {
                                    "Show them"
                                }))
                                .when(!on, |d| d.child(icon("icons/chevron-right.svg", 12., t.text_secondary)))
                        } else {
                            text(ty::XS, t.text_tertiary).child("every image has labels")
                        }),
                    ));
                if empty > 0 {
                    self.bind(
                        el,
                        "ds-empty-tile",
                        0.,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.ds_empty_only = !this.ds_empty_only;
                            this.ds_tab = DsTab::Images;
                            this.grid_scroll.scroll_to_item(0, gpui::ScrollStrategy::Top);
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                } else {
                    el.into_any_element()
                }
            })
            .child(vline(&t))
            .child({
                let el = div()
                    .id("ds-broken-tile")
                    .flex_1()
                    .min_w_0()
                    .px(px(16.))
                    .py(px(12.))
                    .when(broken > 0, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)))
                    .when(broken > 0, |d| {
                        d.tooltip(crate::controls::tip(
                            "These label files couldn't be read, so their images are skipped. The check command lists them.",
                            None,
                        ))
                    })
                    .child(if broken > 0 {
                        // Actionable: a warning, not one more inventory count.
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(6.))
                            .min_w_0()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(4.))
                                    .child(icon("icons/triangle-alert.svg", 12., t.warning_solid))
                                    .child(
                                        text(ty::XS, t.warning_text)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("Broken labels"),
                                    ),
                            )
                            .child(metric(&t, broken.to_string(), "", ty::X2XL, ty::SM))
                            .child(broken_detail(&t, narrow_strip))
                    } else {
                        stat(
                            &t,
                            "Broken labels",
                            broken.to_string(),
                            "",
                            Some(div().child(text(ty::XS, t.success_text).child("all labels valid"))),
                        )
                    });
                if broken > 0 {
                    let cmd = format!("folo dataset check \"{}\"", ds.path);
                    self.bind(
                        el,
                        "ds-broken-tile",
                        0.,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            cx.write_to_clipboard(gpui::ClipboardItem::new_string(cmd.clone()));
                            this.toast(
                                Tone::Neutral,
                                "Copied the check command",
                                Some(format!("Run `{cmd}` in a terminal to list the broken label files.").into()),
                                cx,
                            );
                        }),
                    )
                    .into_any_element()
                } else {
                    el.into_any_element()
                }
            })
            .into_any_element()
    }

    fn image_browser(&mut self, ds: &Dataset, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
                this.grid_scroll.scroll_to_item(0, gpui::ScrollStrategy::Top);
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
                self.chip(
                    &format!("ds-class-{i}"),
                    name.clone(),
                    on,
                    Some(t.class(i as u32)),
                    counts.get(i).copied(),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.ds_class = if this.ds_class == Some(i as u32) { None } else { Some(i as u32) };
                        this.grid_scroll.scroll_to_item(0, gpui::ScrollStrategy::Top);
                        cx.notify();
                    }),
                )
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
            .child(text(ty::XS, t.text_tertiary).mr(px(2.)).child("Boxes by class"))
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
                    .tooltip(crate::controls::tip("Thumbnail size", None))
                    .child(div().w(px(14.)).flex().justify_center().child(icon(
                        "icons/square.svg",
                        9.,
                        t.text_tertiary,
                    )))
                    .child(size_slider)
                    .child(icon("icons/square.svg", 14., t.text_tertiary)),
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
        let fit = if self.ds_fill { crate::viz::Fit::Cover } else { crate::viz::Fit::Contain };
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
                        let tile_el = this.tile(&ds_id, meta, this.ds_boxes, fit, cx);
                        let key = format!("tile-{}", meta.id);
                        let n = meta.boxes.len();
                        let dsid = ds_id.to_string();
                        let focused = this.fh(key.clone(), cx).is_focused(window);
                        let el = div()
                            .id(SharedString::from(key.clone()))
                            .group(SharedString::from(key.clone()))
                            .w(px(cell_w))
                            .h_full()
                            .flex_none()
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .border_1()
                            .border_color(gpui::transparent_black())
                            .hover(|s| s.border_color(t.line_strong))
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
                                            .bg(gpui::linear_gradient(
                                                180.,
                                                gpui::linear_color_stop(gpui::hsla(0., 0., 0., 0.), 0.),
                                                gpui::linear_color_stop(gpui::hsla(0., 0., 0., 0.7), 1.),
                                            ))
                                            .when(!focused, |d| d.invisible())
                                            .group_hover(SharedString::from(key.clone()), |s| s.visible())
                                            .child(
                                                text(ty::XS, gpui::white())
                                                    .font_family(crate::theme::MONO)
                                                    .child(meta.id.clone()),
                                            )
                                            .child(
                                                text(ty::XS, gpui::hsla(0., 0., 1., 0.75))
                                                    .tnum()
                                                    .child(format!("{n} boxes")),
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
                                gpui::ScrollStrategy::Top
                            } else {
                                gpui::ScrollStrategy::Bottom
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

    fn dataset_stats(&mut self, ds: &Dataset, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(s) = ds.stats.clone() else {
            return panel(&t)
                .child(empty_state(
                    &t,
                    "icons/chart-line.svg",
                    "No statistics",
                    "This dataset has not been scanned yet.",
                    Tone::Neutral,
                ))
                .into_any_element();
        };
        let max = s.class_counts.iter().copied().max().unwrap_or(1).max(1) as f32;
        let total: u64 = s.class_counts.iter().sum();
        let class_rows = div().flex().flex_col().gap(px(10.)).p(px(16.)).children(
            ds.classes.iter().enumerate().map(|(i, name)| {
                let c = s.class_counts.get(i).copied().unwrap_or(0);
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(
                        div()
                            .w(px(96.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(crate::ui::class_swatch(&t, t.class(i as u32), 8.))
                            .child(text(ty::SM, t.text_secondary).truncate().child(name.clone())),
                    )
                    .child(
                        div().flex_1().h(px(16.)).rounded(px(3.)).bg(t.bg_inset).child(
                            div()
                                .h_full()
                                .w(gpui::relative(c as f32 / max))
                                .rounded(px(3.))
                                .bg(crate::theme::alpha(t.class(i as u32), 0.85)),
                        ),
                    )
                    .child(text(ty::SM, t.text_primary).tnum().w(px(56.)).text_right().child(int(c)))
                    .child(
                        text(ty::XS, t.text_tertiary)
                            .tnum()
                            .w(px(40.))
                            .text_right()
                            .child(format!("{:.0}%", c as f32 / total.max(1) as f32 * 100.)),
                    )
            }),
        );
        let view = cx.entity().downgrade();
        let hover_fn = |key: &'static str| -> HoverFn {
            let view = view.clone();
            Rc::new(move |f, _, cx| {
                view.update(cx, |this, cx| {
                    match f {
                        Some(v) => this.hist_hover.insert(key, v),
                        None => this.hist_hover.remove(key),
                    };
                    cx.notify();
                })
                .ok();
            })
        };
        let size_labels: Vec<String> =
            s.box_size_hist.edges.windows(2).map(|w| format!("{}–{}%", pct(w[0]), pct(w[1]))).collect();
        let size_n = s.box_size_hist.counts.len();
        let per_n = s.boxes_per_image_hist.counts.len();
        let per_labels: Vec<String> =
            s.boxes_per_image_hist
                .edges
                .windows(2)
                .map(|w| {
                    if w[1] - w[0] <= 1.0 {
                        format!("{}", w[0])
                    } else {
                        format!("{}–{}", w[0], w[1] - 1.0)
                    }
                })
                .collect();
        let size_hist = histogram(
            &t,
            "hist-size",
            Arc::new(s.box_size_hist.counts.clone()),
            Arc::new(size_labels),
            t.chart_series[0],
            200.,
            self.hist_hover.get("size").copied(),
            Some(hover_fn("size")),
        );
        let per_hist = histogram(
            &t,
            "hist-per",
            Arc::new(s.boxes_per_image_hist.counts.clone()),
            Arc::new(per_labels),
            t.chart_series[0],
            200.,
            self.hist_hover.get("per").copied(),
            Some(hover_fn("per")),
        );
        div()
            .id("ds-stats")
            .flex_1()
            .overflow_y_scroll()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(
                        panel(&t)
                            .child(panel_header(&t, "Class balance").child(
                                text(ty::XS, t.text_tertiary).tnum().child(format!("{} boxes", int(total))),
                            ))
                            .child(class_rows),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(16.))
                            .child(
                                panel(&t)
                                    .flex_1()
                                    .min_w_0()
                                    .child(panel_header(&t, "Box size").child(
                                        text(ty::XS, t.text_tertiary).child("√(box area ÷ image area)"),
                                    ))
                                    .child(self.hist_group(
                                        "hist-size-kb",
                                        "size",
                                        size_n,
                                        div().p(px(16.)).child(size_hist),
                                        window,
                                        cx,
                                    )),
                            )
                            .child(
                                panel(&t)
                                    .flex_1()
                                    .min_w_0()
                                    .child(
                                        panel_header(&t, "Boxes per image")
                                            .child(text(ty::XS, t.text_tertiary).child("images")),
                                    )
                                    .child(self.hist_group(
                                        "hist-per-kb",
                                        "per",
                                        per_n,
                                        div().p(px(16.)).child(per_hist),
                                        window,
                                        cx,
                                    )),
                            ),
                    ),
            )
            .into_any_element()
    }
}

/// Class names that fit in about `budget` characters, then "+N" ("circle, square +3").
pub fn names_summary(names: &[String], budget: usize) -> String {
    let mut out = String::new();
    for (i, n) in names.iter().enumerate() {
        let sep = if out.is_empty() { "" } else { ", " };
        let rest = names.len() - i;
        let tail = if rest > 1 { 4 } else { 0 };
        if !out.is_empty() && out.len() + sep.len() + n.len() + tail > budget {
            return format!("{out} +{rest}");
        }
        out.push_str(sep);
        out.push_str(n);
    }
    out
}

/// A fraction as a short percentage: one decimal under 10 % when it matters ("2.5", "10").
fn pct(v: f64) -> String {
    let p = v * 100.0;
    if p >= 10.0 || (p - p.round()).abs() < 0.05 { format!("{}", p.round()) } else { format!("{p:.1}") }
}

impl FoloApp {
    /// Make a histogram one Tab stop: ←/→ select bins, Home/End jump, Esc clears; the hover readout
    /// follows the selection.
    fn hist_group(
        &self,
        key: &str,
        hover_key: &'static str,
        bins: usize,
        content: gpui::Div,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let n = bins.max(1);
        content
            .id(SharedString::from(key.to_string()))
            .track_focus(&fh)
            .relative()
            .on_action(cx.listener(move |this, _: &crate::app::Dismiss, _, cx| {
                if this.hist_hover.remove(hover_key).is_some() {
                    cx.notify();
                } else {
                    cx.propagate();
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let bin = this.hist_hover.get(hover_key).map(|f| ((f * n as f32) as usize).min(n - 1));
                let next = match ev.keystroke.key.as_str() {
                    "left" => Some(bin.map_or(0, |b| b.saturating_sub(1))),
                    "right" => Some(bin.map_or(0, |b| (b + 1).min(n - 1))),
                    "home" => Some(0),
                    "end" => Some(n - 1),
                    "escape" if bin.is_some() => None,
                    _ => return,
                };
                match next {
                    Some(b) => this.hist_hover.insert(hover_key, (b as f32 + 0.5) / n as f32),
                    None => this.hist_hover.remove(hover_key),
                };
                cx.stop_propagation();
                cx.notify();
            }))
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::LG)))
            .into_any_element()
    }
}

/// Under the broken-label count: the check command to copy, or that every label is valid.
fn broken_detail(t: &crate::theme::Theme, narrow: bool) -> gpui::Div {
    if narrow { copy_cmd(t, "Copy command") } else { copy_cmd(t, "Copy check command") }
}

fn copy_cmd(t: &crate::theme::Theme, label: &'static str) -> gpui::Div {
    div()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .child(text(ty::XS, t.text_secondary).whitespace_nowrap().child(label))
        .child(div().flex_none().child(icon("icons/copy.svg", 12., t.text_secondary)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn pct_labels() {
        assert_eq!(super::pct(0.025), "2.5");
        assert_eq!(super::pct(0.1), "10");
        assert_eq!(super::pct(0.0), "0");
    }

    use super::names_summary;

    #[test]
    fn summaries_fit_whole_names() {
        let n: Vec<String> =
            ["circle", "square", "triangle", "star", "hexagon"].iter().map(|s| s.to_string()).collect();
        assert_eq!(names_summary(&n, 24), "circle, square +3");
        assert_eq!(names_summary(&n[..3], 30), "circle, square, triangle");
        assert_eq!(names_summary(&n[..1], 2), "circle");
    }
}
