//! The dataset index, format/search filtering, and class names summarised to fit.

use std::cmp::Ordering;
use std::rc::Rc;

use folo_ui::{AnyElement, Context, Focusable, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::{Dataset, DatasetFormat, Label, SplitsExt};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::{size as tsize, theme, ty};
use crate::ui::*;

/// How the Datasets index is ordered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DatasetSort {
    #[default]
    Name,
    Images,
    Boxes,
    Classes,
}

impl DatasetSort {
    pub const ALL: [Self; 4] = [Self::Name, Self::Images, Self::Boxes, Self::Classes];

    pub fn label(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Images => "Images",
            Self::Boxes => "Boxes",
            Self::Classes => "Classes",
        }
    }

    /// Natural initial direction when picking a sort column: A–Z for Name, largest-first for counts.
    pub fn default_desc(self) -> bool {
        !matches!(self, Self::Name)
    }
}

/// Filters `list` by optional format and case-insensitive substring query across name, path
/// (normalizing `\` and `/`), format label, and id (matching Studio's Datasets filter).
pub fn filter_datasets<'a>(
    list: &'a [Dataset],
    format: Option<DatasetFormat>,
    query: &str,
) -> Vec<&'a Dataset> {
    let q = query.trim().to_lowercase();
    let q_slash = q.contains('\\').then(|| q.replace('\\', "/"));
    let q_norm = q_slash.as_deref().unwrap_or(&q);
    list.iter()
        .filter(|d| format.is_none_or(|f| d.format == f))
        .filter(|d| {
            if q.is_empty() {
                return true;
            }
            if d.name.to_lowercase().contains(&q)
                || d.format.label().to_lowercase().contains(&q)
                || d.id.to_lowercase().contains(&q)
            {
                return true;
            }
            let path = d.path.to_lowercase();
            path.contains(&q)
                || ((q_slash.is_some() || path.contains('\\')) && path.replace('\\', "/").contains(q_norm))
        })
        .collect()
}

/// Sorts `list` by `sort` and `desc`. Missing/unscanned box counts always sort LAST in both
/// ascending and descending order; ties break alphabetically by name, then id.
pub fn sort_datasets(list: &mut [Dataset], sort: DatasetSort, desc: bool) {
    let cmp_name = |a: &Dataset, b: &Dataset| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.id.cmp(&b.id))
    };
    list.sort_by(|a, b| {
        let primary = match sort {
            DatasetSort::Name => {
                let ord = cmp_name(a, b);
                return if desc { ord.reverse() } else { ord };
            }
            DatasetSort::Images => {
                let ord = a.splits.total().cmp(&b.splits.total());
                if desc { ord.reverse() } else { ord }
            }
            DatasetSort::Boxes => {
                match (a.stats.as_ref().map(|s| s.boxes), b.stats.as_ref().map(|s| s.boxes)) {
                    (Some(x), Some(y)) => {
                        let ord = x.cmp(&y);
                        if desc { ord.reverse() } else { ord }
                    }
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                }
            }
            DatasetSort::Classes => {
                let ord = a.classes.len().cmp(&b.classes.len());
                if desc { ord.reverse() } else { ord }
            }
        };
        primary.then_with(|| cmp_name(a, b))
    });
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
                let mut filtered: Vec<Dataset> =
                    filter_datasets(&l, self.ds_format_filter, &query).into_iter().cloned().collect();
                sort_datasets(&mut filtered, self.ds_sort, self.ds_sort_desc);
                let is_filtered = self.ds_format_filter.is_some() || !query.trim().is_empty();
                let sort_labels: Vec<&str> = DatasetSort::ALL.iter().map(|s| s.label()).collect();
                let sort_sel = DatasetSort::ALL.iter().position(|s| *s == self.ds_sort).unwrap_or(0);
                let sort_seg = self.segmented(
                    "ds-sort",
                    &sort_labels,
                    sort_sel,
                    window,
                    cx,
                    Rc::new(|this, i, _, cx| {
                        let picked = DatasetSort::ALL[i];
                        if this.ds_sort == picked {
                            this.ds_sort_desc = !this.ds_sort_desc;
                        } else {
                            this.ds_sort = picked;
                            this.ds_sort_desc = picked.default_desc();
                        }
                        cx.notify();
                    }),
                );
                let sort_dir = self.icon_button(
                    "ds-sort-dir",
                    if self.ds_sort_desc { "icons/chevron-down.svg" } else { "icons/chevron-up.svg" },
                    if self.ds_sort_desc {
                        "Descending · click for ascending"
                    } else {
                        "Ascending · click for descending"
                    },
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_sort_desc = !this.ds_sort_desc;
                        cx.notify();
                    }),
                );
                let sort_group = div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap(px(4.))
                    .child(sort_seg)
                    .child(sort_dir);
                let search = self.search_field("ds-search", self.ds_search.clone(), 220., window, cx);
                let mut toolbar = div()
                    .min_h(px(tsize::CONTROL_LG))
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_x(px(12.))
                    .gap_y(px(8.))
                    .child(seg)
                    .child(sort_group)
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
    use super::{DatasetSort, card_columns, filter_datasets, names_summary, sort_datasets};
    use crate::api::{Dataset, DatasetFormat, DatasetStats, Histogram, Splits};

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

    fn stats_with_boxes(boxes: u64) -> DatasetStats {
        DatasetStats {
            boxes,
            empty_images: 0,
            broken_labels: 0,
            class_counts: vec![boxes],
            box_size_hist: Histogram { edges: vec![0., 1.], counts: vec![boxes] },
            boxes_per_image_hist: Histogram { edges: vec![0., 1.], counts: vec![1] },
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
            sample_dataset("ds_1", "Shapes Hard", r"D:\data\shapes-hard", DatasetFormat::Yolo),
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
        // Case-insensitive name and path matching, trimmed whitespace, and slash normalization.
        assert_eq!(
            filter_datasets(&list, None, "  BLOOD/BCCD ").iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            vec!["ds_3"]
        );
        assert_eq!(
            filter_datasets(&list, None, r"blood\bccd").iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            vec!["ds_3"]
        );
        assert_eq!(
            filter_datasets(&list, None, "d:/data/shapes-hard")
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ds_1"]
        );
        assert_eq!(
            filter_datasets(&list, None, r"D:\datasets\coco2017")
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ds_2"]
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

    #[test]
    fn sorts_datasets_by_name_images_boxes_and_classes_with_missing_boxes_last() {
        let mut a = sample_dataset("ds_a", "beta", "/a", DatasetFormat::Yolo);
        a.splits = Splits { train: 50, val: 10, test: 0 };
        a.classes = vec!["c1".into(), "c2".into()];
        a.stats = Some(stats_with_boxes(400));

        let mut b = sample_dataset("ds_b", "Alpha", "/b", DatasetFormat::Coco);
        b.splits = Splits { train: 5, val: 5, test: 0 };
        b.classes = vec!["c1".into(), "c2".into(), "c3".into(), "c4".into()];
        b.stats = Some(stats_with_boxes(100));

        let mut c = sample_dataset("ds_c", "Gamma", "/c", DatasetFormat::Yolo);
        c.splits = Splits { train: 100, val: 20, test: 5 };
        c.classes = vec!["c1".into()];
        c.stats = None; // Unscanned: must sort LAST in both asc and desc Boxes sort.

        fn ids(l: &[Dataset]) -> Vec<&str> {
            l.iter().map(|d| d.id.as_str()).collect()
        }

        let mut list = vec![a.clone(), b.clone(), c.clone()];
        sort_datasets(&mut list, DatasetSort::Name, false);
        assert_eq!(ids(&list), vec!["ds_b", "ds_a", "ds_c"]);
        sort_datasets(&mut list, DatasetSort::Name, true);
        assert_eq!(ids(&list), vec!["ds_c", "ds_a", "ds_b"]);

        sort_datasets(&mut list, DatasetSort::Images, true);
        assert_eq!(ids(&list), vec!["ds_c", "ds_a", "ds_b"]);
        sort_datasets(&mut list, DatasetSort::Images, false);
        assert_eq!(ids(&list), vec!["ds_b", "ds_a", "ds_c"]);

        sort_datasets(&mut list, DatasetSort::Boxes, true);
        assert_eq!(ids(&list), vec!["ds_a", "ds_b", "ds_c"]);
        sort_datasets(&mut list, DatasetSort::Boxes, false);
        assert_eq!(ids(&list), vec!["ds_b", "ds_a", "ds_c"]);

        sort_datasets(&mut list, DatasetSort::Classes, true);
        assert_eq!(ids(&list), vec!["ds_b", "ds_a", "ds_c"]);
        sort_datasets(&mut list, DatasetSort::Classes, false);
        assert_eq!(ids(&list), vec!["ds_c", "ds_a", "ds_b"]);
    }
}
