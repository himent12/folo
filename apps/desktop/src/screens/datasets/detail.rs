//! A dataset's page, and its images after the split and class filters.

use std::rc::Rc;

use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, StatefulInteractiveElement, Styled,
    Window, div, px,
};

use super::health::{finding_count, health_checks};
use crate::api::SplitsExt;
use crate::api::{ImageInfo, Label, Split};
use crate::app::{DsTab, FoloApp, act};
use crate::controls::{Btn, Tab};
use crate::screens::{detail_header, loading_rows, page_header};
use crate::store::Load;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
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

        if ds.stats.as_ref().is_some_and(|s| s.broken_labels > 0) {
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
            Rc::new(|this, i, _, _| this.ds_tab = if i == 0 { DsTab::Images } else { DsTab::Statistics }),
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
