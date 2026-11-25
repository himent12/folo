//! The dataset index, format/search filtering, and class names summarised to fit.

use std::rc::Rc;

use folo_ui::{AnyElement, Context, Focusable, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::{Dataset, DatasetFormat, Label, SplitsExt};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::{size as tsize, theme, ty};
use crate::ui::*;

/// Filters `list` by optional format and case-insensitive substring query across name, path,
/// format label, and id (matching Studio's Datasets filter).
pub fn filter_datasets<'a>(
    list: &'a [Dataset],
    format: Option<DatasetFormat>,
    query: &str,
) -> Vec<&'a Dataset> {
    let q = query.trim().to_lowercase();
    list.iter()
        .filter(|d| format.is_none_or(|f| d.format == f))
        .filter(|d| {
            q.is_empty()
                || d.name.to_lowercase().contains(&q)
                || d.path.to_lowercase().contains(&q)
                || d.format.label().to_lowercase().contains(&q)
                || d.id.to_lowercase().contains(&q)
        })
        .collect()
}

impl FoloApp {
    /// The Datasets header's facts, "2 datasets · 1,240 images": the images of every dataset its
    /// server can still read. One it can't (folder gone, images deleted) has only a last known
    /// count, which its card says; it is left out of the total, as in Studio (s1-datasets-6).
    pub(crate) fn datasets_subtitle(&self, l: &[Dataset]) -> String {
        let images = l.iter().filter(|d| !self.dataset_unreadable(&d.id)).map(|d| d.splits.total()).sum();
        format!("{} · {}", plural(l.len(), "dataset"), counted(images, "image", "images"))
    }

    pub fn datasets(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let list = self.store.datasets.ready().cloned();
        let subtitle = match &list {
            Some(l) => self.datasets_subtitle(l),
            // As Models and Benchmarks say: not "Loading…" over the error while it is asked again.
            None if self.store.datasets.error().is_some() => "The dataset list didn't load".into(),
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
            None => div()
                .flex()
                .gap(px(16.))
                .children((0..3).map(|i| div().flex_1().h(px(200.)).child(skeleton(&t, format!("dsc{i}")))))
                .into_any_element(),
            Some(l) if l.is_empty() => panel(&t)
                .child(empty_state(
                    &t,
                    "icons/images.svg",
                    "No datasets yet",
                    "Point FOLO at a YOLO or COCO folder, or generate a synthetic shapes dataset to try training end to end.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Some(l) => {
                let total = l.len();
                let formats = [None, Some(DatasetFormat::Yolo), Some(DatasetFormat::Coco)];
                let fmt_count = |f: Option<DatasetFormat>| {
                    l.iter().filter(|d| f.is_none_or(|want| d.format == want)).count()
                };
                let labels: Vec<String> = formats
                    .iter()
                    .map(|&f| {
                        let name = match f {
                            None => "All",
                            Some(DatasetFormat::Yolo) => "YOLO",
                            Some(DatasetFormat::Coco) => "COCO",
                            Some(DatasetFormat::Auto) => "Auto",
                        };
                        format!("{name}\u{a0}{}", fmt_count(f))
                    })
                    .collect();
                let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                let sel = formats.iter().position(|f| *f == self.ds_format_filter).unwrap_or(0);
                let seg = self.segmented(
                    "ds-format-filter",
                    &label_refs,
                    sel,
                    window,
                    cx,
                    Rc::new(move |this, i, _, cx| {
                        this.ds_format_filter = formats[i];
                        cx.notify();
                    }),
                );
                let query = self.ds_search.read(cx).text().to_string();
                let filtered: Vec<Dataset> =
                    filter_datasets(&l, self.ds_format_filter, &query).into_iter().cloned().collect();
                let is_filtered = self.ds_format_filter.is_some() || !query.trim().is_empty();
                let search = self.search_field("ds-search", self.ds_search.clone(), 240., window, cx);
                let mut toolbar = div()
                    .h(px(tsize::CONTROL_LG))
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(seg)
                    .child(div().flex_1());
                if is_filtered {
                    toolbar = toolbar.child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .whitespace_nowrap()
                            .child(format!("{} of {}", filtered.len(), plural(total, "dataset"))),
                    );
                }
                toolbar = toolbar.child(search);

                let content = if filtered.is_empty() {
                    let clear = self.button(
                        "ds-filter-clear",
                        "Clear filters",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, w, cx| {
                            this.ds_format_filter = None;
                            this.ds_search.update(cx, |i, cx| i.set_text("", cx));
                            w.focus(&this.ds_search.read(cx).focus_handle(cx));
                            cx.notify();
                        }),
                    );
                    panel(&t)
                        .child(
                            empty_state(
                                &t,
                                "icons/list-filter.svg",
                                "No datasets match",
                                "Try another format or clear the search filter.",
                                Tone::Neutral,
                            )
                            .child(clear),
                        )
                        .into_any_element()
                } else {
                    // Filter before constructing cards so hidden datasets don't kick off preview reads.
                    let cards: Vec<AnyElement> =
                        filtered.iter().map(|d| self.dataset_card(d, window, cx)).collect();
                    let per_row = card_columns(self.content_width(window));
                    let mut rows = Vec::new();
                    let mut it = cards.into_iter().peekable();
                    while it.peek().is_some() {
                        // Equal columns (basis 0, no min width), so a short last row matches the rows above.
                        let mut row: Vec<AnyElement> = it
                            .by_ref()
                            .take(per_row)
                            .map(|c| {
                                div().flex_grow().flex_basis(px(0.)).min_w_0().flex().child(c).into_any_element()
                            })
                            .collect();
                        while row.len() < per_row {
                            row.push(div().flex_grow().flex_basis(px(0.)).min_w_0().into_any_element());
                        }
                        rows.push(div().flex().gap(px(16.)).children(row));
                    }
                    div().flex().flex_col().gap(px(16.)).children(rows).into_any_element()
                };
                div().flex().flex_col().gap(px(16.)).child(toolbar).child(content).into_any_element()
            }
        };
        self.page_frame(&t, header, body, cx)
    }
}

/// How many dataset cards share a row `width` px wide: as many as stay at least 300 px wide with
/// 16 px between them (design §4 Datasets), at least one.
pub fn card_columns(width: f32) -> usize {
    (((width + 16.) / (300. + 16.)).floor() as usize).max(1)
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

#[cfg(test)]
mod tests {
    use super::{card_columns, filter_datasets, names_summary};
    use crate::api::{Dataset, DatasetFormat, Splits};

    fn sample_dataset(id: &str, name: &str, path: &str, format: DatasetFormat) -> Dataset {
        Dataset {
            id: id.into(),
            name: name.into(),
            path: path.into(),
            format,
            classes: vec!["a".into()],
            created: "2026-11-01T00:00:00Z".into(),
            splits: Splits { train: 10, val: 2, test: 0 },
            stats: None,
            revision: None,
        }
    }

    #[test]
    fn cards_stay_at_least_300_wide() {
        // 960 (content 872): two; 1280 (1000): three; 1680 (1392): four.
        assert_eq!([card_columns(872.), card_columns(1000.), card_columns(1392.)], [2, 3, 4]);
        assert_eq!(card_columns(200.), 1);
    }

    #[test]
    fn summaries_fit_whole_names() {
        let n: Vec<String> =
            ["circle", "square", "triangle", "star", "hexagon"].iter().map(|s| s.to_string()).collect();
        assert_eq!(names_summary(&n, 24), "circle, square +3");
        assert_eq!(names_summary(&n[..3], 30), "circle, square, triangle");
        assert_eq!(names_summary(&n[..1], 2), "circle");
    }

    #[test]
    fn filters_datasets_by_format_and_query() {
        let list = vec![
            sample_dataset("ds_1", "Shapes Hard", "D:/data/shapes-hard", DatasetFormat::Yolo),
            sample_dataset("ds_2", "COCO Val", "D:/datasets/coco2017", DatasetFormat::Coco),
            sample_dataset("ds_3", "BCCD Cells", "/mnt/blood/bccd", DatasetFormat::Yolo),
        ];
        assert_eq!(filter_datasets(&list, None, "").len(), 3);
        assert_eq!(
            filter_datasets(&list, Some(DatasetFormat::Yolo), "")
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ds_1", "ds_3"]
        );
        assert_eq!(
            filter_datasets(&list, Some(DatasetFormat::Coco), "")
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ds_2"]
        );
        // Case-insensitive name and path matching, trimmed whitespace.
        assert_eq!(
            filter_datasets(&list, None, "  BLOOD/BCCD ").iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            vec!["ds_3"]
        );
        assert_eq!(
            filter_datasets(&list, Some(DatasetFormat::Yolo), "shapes")
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ds_1"]
        );
        assert!(filter_datasets(&list, Some(DatasetFormat::Coco), "shapes").is_empty());
    }
}
