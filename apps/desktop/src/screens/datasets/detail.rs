//! A dataset's page, and its images after the split and class filters.

use std::rc::Rc;

use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, StatefulInteractiveElement, Styled,
    Window, div, px,
};

use super::health::{finding_count, health_checks};
use super::images::file_name;
use crate::api::SplitsExt;
use crate::api::{ImageInfo, Label, Split};
use crate::app::{DsTab, FoloApp, act};
use crate::controls::{Btn, Tab};
use crate::screens::{detail_header, loading_rows, page_header};
use crate::store::Load;
use crate::theme::{theme, ty};
use crate::ui::*;

/// How a dataset's Images grid is ordered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ImageSort {
    /// Original server order.
    #[default]
    Default,
    /// Filename A–Z (case-insensitive).
    NameAsc,
    /// Filename Z–A (case-insensitive).
    NameDesc,
    /// Most ground-truth boxes first.
    MostBoxes,
    /// Fewest ground-truth boxes first.
    FewestBoxes,
}

impl ImageSort {
    pub const ALL: [Self; 5] =
        [Self::Default, Self::NameAsc, Self::NameDesc, Self::MostBoxes, Self::FewestBoxes];

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Order",
            Self::NameAsc => "Filename A–Z",
            Self::NameDesc => "Filename Z–A",
            Self::MostBoxes => "Most boxes",
            Self::FewestBoxes => "Fewest boxes",
        }
    }
}

/// Cache key and result for filtered and sorted dataset image indices.
#[derive(Clone)]
pub struct DatasetFilterCache {
    dataset: String,
    split: Option<Split>,
    class: Option<u32>,
    empty_only: bool,
    query: String,
    sort: ImageSort,
    len: usize,
    ptr: usize,
    indices: Rc<Vec<usize>>,
}

/// Filters and sorts indices into `all` by split, class, empty-only, normalized filename/path query,
/// and `sort`.
pub fn filter_and_sort_images(
    all: &[ImageInfo],
    split: Option<Split>,
    class: Option<u32>,
    empty_only: bool,
    query: &str,
    sort: ImageSort,
) -> Vec<usize> {
    let q = query.trim().to_lowercase();
    let q_slash = q.contains('\\').then(|| q.replace('\\', "/"));
    let q_norm = q_slash.as_deref().unwrap_or(&q);
    let mut indices: Vec<usize> = all
        .iter()
        .enumerate()
        .filter(|(_, m)| split.is_none_or(|s| m.split == s))
        .filter(|(_, m)| class.is_none_or(|c| m.boxes.iter().any(|b| b.class == c)))
        .filter(|(_, m)| !empty_only || m.boxes.is_empty())
        .filter(|(_, m)| {
            if q.is_empty() {
                return true;
            }
            let file = m.file.to_lowercase();
            file.contains(&q)
                || ((q_slash.is_some() || file.contains('\\')) && file.replace('\\', "/").contains(q_norm))
        })
        .map(|(i, _)| i)
        .collect();
    if sort != ImageSort::Default {
        let cmp_name = |a: usize, b: usize| {
            let fa = file_name(&all[a].file);
            let fb = file_name(&all[b].file);
            fa.to_lowercase()
                .cmp(&fb.to_lowercase())
                .then_with(|| fa.cmp(fb))
                .then_with(|| all[a].file.cmp(&all[b].file))
                .then_with(|| a.cmp(&b))
        };
        indices.sort_by(|&a, &b| match sort {
            ImageSort::Default => a.cmp(&b),
            ImageSort::NameAsc => cmp_name(a, b),
            ImageSort::NameDesc => cmp_name(b, a),
            ImageSort::MostBoxes => all[b].boxes.len().cmp(&all[a].boxes.len()).then_with(|| cmp_name(a, b)),
            ImageSort::FewestBoxes => {
                all[a].boxes.len().cmp(&all[b].boxes.len()).then_with(|| cmp_name(a, b))
            }
        });
    }
    indices
}

impl FoloApp {
    /// A registered dataset its server can't read (folder moved away, images deleted): what is
    /// wrong in the server's words and the ways on: read it again once the folder is back, remove
    /// it from FOLO (its files stay), or the list. Not the connection panel: the server answered.
    fn dataset_problem_panel(
        &mut self,
        id: &str,
        label: &'static str,
        e: &crate::api::ApiError,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let again = id.to_string();
        let retry = self.button(
            "ds-problem-retry",
            "Retry",
            Btn::Primary,
            Some("icons/refresh-cw.svg"),
            window,
            cx,
            act(move |this, _, cx| {
                this.ds_recheck.insert(again.clone());
                cx.notify();
            }),
        );
        let target = crate::app::DeleteTarget::Dataset(id.to_string());
        let remove = self.button(
            "ds-problem-delete",
            "Remove from FOLO",
            Btn::Secondary,
            Some("icons/trash-2.svg"),
            window,
            cx,
            act(move |this, w, cx| this.ask_delete(target.clone(), w, cx)),
        );
        let all = self.button(
            "ds-problem-all",
            "All datasets",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.navigate(crate::app::Route::Datasets, w, cx)),
        );
        let hint = e.hint.clone().unwrap_or_else(|| {
            "Put the folder back (or fix it) and retry, or remove the dataset from FOLO.".to_string()
        });
        panel(&t)
            .child(
                empty_state(
                    &t,
                    "icons/circle-alert.svg",
                    label,
                    &format!("{} {hint}", e.message),
                    Tone::Warning,
                )
                .child(div().flex().gap(px(8.)).child(all).child(remove).child(retry)),
            )
            .into_any_element()
    }

    /// The class filter of dataset `id`: one set on that dataset, while its class is one of the
    /// dataset's classes as last read (a refreshed class list can leave it past them). Anywhere
    /// else none: another dataset's index means another class there.
    pub fn class_filter(&self, id: &str) -> Option<u32> {
        let (of, class) = self.ds_class.as_ref()?;
        let ds =
            self.store.dataset.get(id).and_then(|l| l.ready()).or_else(|| self.store.dataset_summary(id))?;
        (of == id && (*class as usize) < ds.classes.len()).then_some(*class)
    }

    /// Whether dataset `id`'s grid shows only its images without labels ("Show them").
    pub fn empty_only(&self, id: &str) -> bool {
        self.ds_empty_of.as_deref() == Some(id)
    }

    /// Whether any filter or non-default sort is active on dataset `id`'s Images tab.
    pub fn dataset_image_filters_active(&self, id: &str) -> bool {
        self.ds_split.is_some()
            || self.class_filter(id).is_some()
            || self.empty_only(id)
            || !self.ds_img_query.trim().is_empty()
            || self.ds_img_sort != ImageSort::Default
    }

    /// Resets split, class, empty-only, filename search, and sort on dataset `id`'s Images tab.
    pub fn reset_dataset_image_filters(&mut self, id: &str, cx: &mut Context<Self>) {
        self.ds_split = None;
        if self.ds_class.as_ref().is_some_and(|(of, _)| of == id) {
            self.ds_class = None;
        }
        if self.empty_only(id) {
            self.ds_empty_of = None;
        }
        self.ds_img_query.clear();
        self.ds_img_search.update(cx, |i, cx| i.set_text("", cx));
        self.ds_img_sort = ImageSort::Default;
        self.ds_hover_tile = None;
        self.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
        cx.notify();
    }

    /// Indices (into the dataset's image list) that pass the split, class, empty-only, and filename
    /// filters, ordered by `ds_img_sort`. Cached until the dataset, its image list, a filter, or the
    /// sort changes, so scrolling never rescans it.
    pub fn filtered_indices(&self, id: &str) -> Rc<Vec<usize>> {
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else { return Rc::new(Vec::new()) };
        let class = self.class_filter(id);
        let empty_only = self.empty_only(id);
        let len = all.len();
        let ptr = all.as_ptr() as usize;
        if let Some(c) = self.ds_filter_cache.borrow().as_ref()
            && c.dataset == id
            && c.split == self.ds_split
            && c.class == class
            && c.empty_only == empty_only
            && c.query == self.ds_img_query
            && c.sort == self.ds_img_sort
            && c.len == len
            && c.ptr == ptr
        {
            return c.indices.clone();
        }
        let v = Rc::new(filter_and_sort_images(
            all,
            self.ds_split,
            class,
            empty_only,
            &self.ds_img_query,
            self.ds_img_sort,
        ));
        *self.ds_filter_cache.borrow_mut() = Some(DatasetFilterCache {
            dataset: id.to_string(),
            split: self.ds_split,
            class,
            empty_only,
            query: self.ds_img_query.clone(),
            sort: self.ds_img_sort,
            len,
            ptr,
            indices: v.clone(),
        });
        v
    }

    /// Boxes per class among a split's images (cached per dataset and split).
    pub fn split_class_counts(&self, id: &str, split: Split, classes: usize) -> Vec<u64> {
        let Some(all) = self.store.images.get(id).and_then(|l| l.ready()) else { return vec![0; classes] };
        let len = all.len();
        let ptr = all.as_ptr() as usize;
        if let Some((k, v)) = self.ds_split_counts.borrow().as_ref()
            && (k.0.as_str(), k.1, k.2, k.3, k.4) == (id, split, len, ptr, classes)
        {
            return v.clone();
        }
        let key = (id.to_string(), split, len, ptr, classes);
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
                // The server answered: no such dataset (removed elsewhere, or a stale link). Not a
                // connection problem, so no Connection settings: the way back is the list.
                let err = if e.code == "dataset_not_found" {
                    let all = self.button(
                        "ds-missing-all",
                        "All datasets",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, w, cx| this.navigate(crate::app::Route::Datasets, w, cx)),
                    );
                    panel(&t)
                        .child(
                            empty_state(
                                &t,
                                "icons/circle-alert.svg",
                                "Dataset not found",
                                "It may have been removed.",
                                Tone::Neutral,
                            )
                            .child(all),
                        )
                        .into_any_element()
                } else if let Some(label) = crate::screens::dataset_problem(&e) {
                    self.dataset_problem_panel(id, label, &e, window, cx)
                } else {
                    self.error_panel(&e, "ds-detail-retry", window, cx)
                };
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
        let target = crate::app::DeleteTarget::Dataset(ds.id.clone());
        let delete = self
            .button(
                "ds-delete",
                "Remove from FOLO",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                window,
                cx,
                act(move |this, w, cx| this.ask_delete(target.clone(), w, cx)),
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
        // Short windows (the 620 px minimum): gaps tighten, so the first image row starts high.
        let short = window.viewport_size().height < px(700.);
        let title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            // A long name gives way (ellipsis, full name on hover) before the actions do; text
            // truncates against a known width, so it's capped by the window's.
            .child(
                div()
                    .id("ds-title")
                    .min_w_0()
                    .flex_shrink()
                    .tooltip(crate::controls::tip(ds.name.clone(), None))
                    .child(
                        text(ty::XL, t.text_primary)
                            .max_w(px(if window.viewport_size().width < px(1280.) { 300. } else { 560. }))
                            .ellipsis()
                            .child(ds.name.clone()),
                    ),
            )
            .child(div().flex_none().child(badge(&t, ds.format.label(), Tone::Neutral)));
        // One meta line: where it is (cut short in a narrow window, the full path on hover), its
        // splits, its classes and when it was added. Below 1280 px the splits leave it for the path:
        // the toolbar's split filter counts them.
        let now = self.store.api.now();
        let mut facts: Vec<String> = if window.viewport_size().width < px(1280.) {
            Vec::new()
        } else {
            [("train", ds.splits.train), ("val", ds.splits.val), ("test", ds.splits.test)]
                .into_iter()
                .filter(|(_, n)| *n > 0)
                .map(|(s, n)| format!("{} {s}", int(n)))
                .collect()
        };
        facts.push(counted(ds.classes.len() as u64, "class", "classes"));
        if let Some(c) = crate::api::time::parse(&ds.created) {
            facts.push(format!("added {}", crate::api::time::ago(c, now)));
        }
        let meta = div()
            .flex()
            .items_center()
            .gap(px(6.))
            .min_w_0()
            .child(
                div()
                    .id("ds-path")
                    .min_w_0()
                    .flex_shrink()
                    .tooltip(crate::controls::tip(display_path(&ds.path), None))
                    .child(
                        text(ty::SM, t.text_secondary)
                            .font_family(crate::theme::MONO)
                            .max_w(px(if window.viewport_size().width < px(1280.) { 220. } else { 420. }))
                            .ellipsis()
                            .child(display_path(&ds.path)),
                    ),
            )
            .child(
                text(ty::SM, t.text_secondary)
                    .tnum()
                    .flex_none()
                    .whitespace_nowrap()
                    .child(format!("· {}", facts.join(" · "))),
            );
        let header = detail_header(&t, title, meta, vec![delete, reveal, train]);

        // The check's counts feed the health summary: unreadable images and check warnings show
        // there even when every label parses.
        if ds.stats.is_some() {
            self.ensure_issues(&ds.id, cx);
        }
        // Each section with what it holds: the images, and how many findings the statistics have.
        let issues = self.store.issues.get(&ds.id).and_then(|l| l.ready()).cloned();
        let health = ds.stats.as_ref().map(|s| health_checks(&ds, s, issues.as_ref())).unwrap_or_default();
        let findings = finding_count(&health);
        let tabs = self.tabs(
            "ds-tab",
            vec![
                Tab::new("Images", Some(int(ds.splits.total()))),
                Tab::new(
                    "Statistics",
                    (findings > 0).then(|| format!("· {}", counted(findings as u64, "finding", "findings"))),
                ),
            ],
            usize::from(self.ds_tab == DsTab::Statistics),
            window,
            cx,
            Rc::new(|this, i, _, _| {
                this.ds_tab = if i == 0 { DsTab::Images } else { DsTab::Statistics };
                this.ds_hover_tile = None;
            }),
        );
        let content = match self.ds_tab {
            DsTab::Images => self.image_browser(&ds, window, cx),
            DsTab::Statistics => self.dataset_stats(&ds, &health, window, cx),
        };
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap(px(if short { 12. } else { 16. }))
            .px(px(self.gutter()))
            .pt(px(if short { 16. } else { 20. }))
            // In keyboard mode the key-hint strip (32 px) lies over the page's bottom: the grid
            // ends above it, so a focused bottom row is never under it.
            .pb(px(if self.kb { 48. } else { 16. }))
            .child(header)
            .child(div().flex().child(tabs))
            .child(div().flex_1().min_h_0().flex().flex_col().child(content))
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageSort, filter_and_sort_images};
    use crate::api::{BBox, ImageInfo, Split};

    fn img(id: &str, file: &str, split: Split, boxes: usize) -> ImageInfo {
        ImageInfo {
            id: id.into(),
            file: file.into(),
            split,
            width: 640,
            height: 480,
            boxes: (0..boxes).map(|_| BBox { class: 0, x1: 0., y1: 0., x2: 10., y2: 10. }).collect(),
        }
    }

    #[test]
    fn filters_by_normalized_filename_and_sorts_dataset_images() {
        let all = vec![
            img("0", r"images\train\zeta.jpg", Split::Train, 2),
            img("1", "images/val/Alpha.jpg", Split::Val, 5),
            img("2", "images/train/mid.jpg", Split::Train, 0),
        ];

        // Slash-normalized case-insensitive filename/path search.
        assert_eq!(
            filter_and_sort_images(&all, None, None, false, "train/ZETA", ImageSort::Default),
            vec![0]
        );
        assert_eq!(
            filter_and_sort_images(&all, None, None, false, r"val\alpha", ImageSort::Default),
            vec![1]
        );

        // Sort orders.
        assert_eq!(filter_and_sort_images(&all, None, None, false, "", ImageSort::Default), vec![0, 1, 2]);
        assert_eq!(filter_and_sort_images(&all, None, None, false, "", ImageSort::NameAsc), vec![1, 2, 0]);
        assert_eq!(filter_and_sort_images(&all, None, None, false, "", ImageSort::NameDesc), vec![0, 2, 1]);
        assert_eq!(filter_and_sort_images(&all, None, None, false, "", ImageSort::MostBoxes), vec![1, 0, 2]);
        assert_eq!(
            filter_and_sort_images(&all, None, None, false, "", ImageSort::FewestBoxes),
            vec![2, 0, 1]
        );
    }
}
