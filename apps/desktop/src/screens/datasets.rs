//! Datasets: the dataset index and a dataset's images (virtualised grid) and statistics.

use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, canvas, div, px, uniform_list,
};

use super::*;
use crate::api::{Dataset, ImageMeta, Split};
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
            Some(l) => format!(
                "{} datasets · {} images",
                l.len(),
                int(l.iter().map(|d| d.splits.total() as u64).sum())
            ),
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
                act(|this, _, cx| {
                    this.toast(
                        Tone::Info,
                        "Registering folders needs folo serve",
                        Some("Mock data has three built-in datasets.".into()),
                        cx,
                    );
                }),
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
                act(|this, _, cx| {
                    this.toast(
                        Tone::Info,
                        "Synthetic datasets need folo serve",
                        Some("Run `folo dataset synth` or start the server.".into()),
                        cx,
                    );
                }),
            )
            .into_any_element();
        let header = page_header(&t, "Datasets", subtitle, vec![synth, add]);
        if let Some(e) = self.store.datasets.error().cloned() {
            let err = self.error_panel(&e, "ds-retry", window, cx);
            return self.page_frame(&t, header, err);
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
                    let mut row: Vec<AnyElement> = it.by_ref().take(3).collect();
                    while row.len() < 3 {
                        row.push(div().flex_1().into_any_element());
                    }
                    rows.push(div().flex().gap(px(16.)).children(row));
                }
                div().flex().flex_col().gap(px(16.)).children(rows).into_any_element()
            }
        };
        self.page_frame(&t, header, body)
    }

    /// Images of the open dataset after the split and class filters.
    /// Indices (into the dataset's image list) that pass the split and class filters. Cached
    /// until the dataset, its image list or a filter changes, so scrolling never rescans it.
    pub fn filtered_indices(&self, id: &str) -> Rc<Vec<usize>> {
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else { return Rc::new(Vec::new()) };
        let key = (id.to_string(), self.ds_split, self.ds_class, all.len());
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
                .map(|(i, _)| i)
                .collect(),
        );
        *self.ds_filter_cache.borrow_mut() = Some((key, v.clone()));
        v
    }

    /// The `n`th filtered image of a dataset.
    pub fn filtered_image(&self, id: &str, n: usize) -> Option<&ImageMeta> {
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
                return self.page_frame(&t, header, err);
            }
            _ => {
                let header = page_header(
                    &t,
                    self.store.dataset_summary(id).map_or("Dataset".to_string(), |d| d.name.clone()),
                    "Loading…",
                    vec![],
                );
                return self.page_frame(&t, header, panel(&t).child(loading_rows(&t, "ds", 6)));
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
            .child(badge(&t, ds.format.to_uppercase(), Tone::Neutral));
        let header = div()
            .flex()
            .items_end()
            .justify_between()
            .gap(px(16.))
            .child(
                div().flex().flex_col().gap(px(4.)).min_w_0().child(title).child(
                    text(ty::SM, t.text_tertiary)
                        .font_family(crate::theme::MONO)
                        .truncate()
                        .child(ds.path.clone()),
                ),
            )
            .child(div().flex().gap(px(8.)).child(reveal).child(train));

        let strip = self.dataset_strip_stats(&ds, cx);
        let tabs = self.segmented(
            "ds-tabs",
            &["Images", "Statistics"],
            if self.ds_tab == DsTab::Images { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, _| this.ds_tab = if i == 0 { DsTab::Images } else { DsTab::Statistics }),
        );
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

    fn dataset_strip_stats(&self, ds: &Dataset, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let total = ds.splits.total().max(1) as f32;
        let s = ds.stats.as_ref();
        let split_bar = div()
            .flex()
            .h(px(4.))
            .w_full()
            .rounded_full()
            .overflow_hidden()
            .gap(px(2.))
            .child(div().h_full().w(gpui::relative(ds.splits.train as f32 / total)).bg(t.chart_series[0]))
            .child(div().h_full().w(gpui::relative(ds.splits.val as f32 / total)).bg(t.chart_series[1]))
            .when(ds.splits.test > 0, |d| d.child(div().h_full().flex_1().bg(t.chart_series[2])));
        let legend = div().flex().gap(px(10.)).children(
            [
                ("train", ds.splits.train, t.chart_series[0]),
                ("val", ds.splits.val, t.chart_series[1]),
                ("test", ds.splits.test, t.chart_series[2]),
            ]
            .into_iter()
            .filter(|x| x.1 > 0)
            .map(|(n, c, col)| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(5.))
                    .child(swatch(col, 6.))
                    .child(text(ty::XS, t.text_tertiary).tnum().child(format!("{n} {}", int(c as u64))))
            }),
        );
        let tile = |content: gpui::Div| div().flex_1().min_w_0().px(px(16.)).py(px(12.)).child(content);
        let boxes = s.map_or(0, |s| s.boxes);
        let per = boxes as f32 / total;
        let broken = s.map_or(0, |s| s.broken_labels);
        panel(&t)
            .flex()
            .child(
                div().flex_none().w(px(300.)).px(px(16.)).py(px(12.)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .child(caption(&t, "Images"))
                        .child(metric(&t, int(ds.splits.total() as u64), "", ty::X2XL, ty::SM))
                        .child(split_bar)
                        .child(legend),
                ),
            )
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Boxes",
                int(boxes as u64),
                "",
                Some(text(ty::XS, t.text_tertiary).tnum().child(format!("{per:.1} per image"))),
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Classes",
                ds.classes.len().to_string(),
                "",
                Some(text(ty::XS, t.text_tertiary).truncate().child(if ds.classes.len() > 3 {
                    format!("{} +{} more", ds.classes[..3].join(", "), ds.classes.len() - 3)
                } else {
                    ds.classes.join(", ")
                })),
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Empty images",
                int(s.map_or(0, |s| s.empty_images) as u64),
                "",
                Some(text(ty::XS, t.text_tertiary).child("no labels")),
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Broken labels",
                broken.to_string(),
                "",
                Some(if broken > 0 {
                    text(ty::XS, t.warning_text).child("out of image bounds")
                } else {
                    text(ty::XS, t.success_text).child("all labels valid")
                }),
            )))
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
        let counts = ds.stats.as_ref().map(|s| s.class_counts.clone()).unwrap_or_default();
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
        let count_label = if count == total {
            format!("{} images", int(total as u64))
        } else {
            format!("{} of {} images", int(count as u64), int(total as u64))
        };
        let toolbar = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .pb(px(12.))
            .child(seg)
            .child(div().w(px(1.)).h(px(20.)).bg(t.line_subtle))
            .child(div().flex().items_center().gap(px(6.)).flex_1().min_w_0().flex_wrap().children(chips))
            .child(text(ty::SM, t.text_tertiary).tnum().whitespace_nowrap().child(count_label));
        let view_controls = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_secondary).child("Boxes"))
                    .child(boxes_switch),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .w(px(140.))
                    .flex_none()
                    .child(icon("icons/zoom-out.svg", 14., t.text_tertiary))
                    .child(size_slider)
                    .child(icon("icons/zoom-in.svg", 14., t.text_tertiary)),
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
        let grid = uniform_list(
            "ds-grid",
            rows,
            cx.processor(move |this, range: std::ops::Range<usize>, window, cx| {
                let t = theme(cx);
                // Clone only the visible rows' metadata.
                let visible: Vec<Option<ImageMeta>> = (range.start * cols..range.end * cols)
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
                        let tile_el = this.tile(&ds_id, meta, this.ds_boxes, crate::viz::Fit::Contain, cx);
                        let key = format!("tile-{}", meta.id);
                        let n = meta.boxes.len();
                        let dsid = ds_id.to_string();
                        let el = div()
                            .id(SharedString::from(key.clone()))
                            .group(SharedString::from(key.clone()))
                            .w(px(cell_w))
                            .h_full()
                            .flex_none()
                            .rounded(px(radius::MD))
                            .overflow_hidden()
                            .bg(t.bg_inset)
                            .cursor_pointer()
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
                                    .invisible()
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
                            );
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

    fn dataset_stats(&mut self, ds: &Dataset, _window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
        let total: u32 = s.class_counts.iter().sum();
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
                            .child(swatch(t.class(i as u32), 8.))
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
                    .child(text(ty::SM, t.text_primary).tnum().w(px(56.)).text_right().child(int(c as u64)))
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
            s.box_size_hist.edges.windows(2).map(|w| format!("<{}%", (w[1] * 100.0).round())).collect();
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
            t.chart_series[1],
            200.,
            self.hist_hover.get("size").copied(),
            Some(hover_fn("size")),
        );
        let per_hist = histogram(
            &t,
            "hist-per",
            Arc::new(s.boxes_per_image_hist.counts.clone()),
            Arc::new(per_labels),
            t.chart_series[2],
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
                            .child(
                                panel_header(&t, "Class balance").child(
                                    text(ty::XS, t.text_tertiary)
                                        .tnum()
                                        .child(format!("{} boxes", int(total as u64))),
                                ),
                            )
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
                                    .child(
                                        panel_header(&t, "Box size").child(
                                            text(ty::XS, t.text_tertiary).child("share of image area"),
                                        ),
                                    )
                                    .child(div().p(px(16.)).child(size_hist)),
                            )
                            .child(
                                panel(&t)
                                    .flex_1()
                                    .min_w_0()
                                    .child(
                                        panel_header(&t, "Boxes per image")
                                            .child(text(ty::XS, t.text_tertiary).child("images")),
                                    )
                                    .child(div().p(px(16.)).child(per_hist)),
                            ),
                    ),
            )
            .into_any_element()
    }
}
