//! Dataset Class-Balance & Annotation Quality Strip: computes exact max-to-min class imbalance
//! ratios, rarest-class and zero-box class indicators, and empty/unlabeled image stats strictly
//! from loaded `Dataset` / `DatasetStats`, with one-click filter jumps into the dataset gallery.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Dataset, DatasetStats, SplitsExt};
use crate::app::{DsTab, FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

/// Matches `health.rs`: ratio above 5.0× warns; 20.0× and above is severe skew.
const SKEW_WARN: f64 = 5.0;
const SKEW_SEVERE: f64 = 20.0;
/// Matches `health.rs`: classes with fewer than 100 boxes are underrepresented.
const UNDERREPRESENTED_BOXES: u64 = 100;

/// Severity bucket for class imbalance ratio.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BalanceSeverity {
    /// Only 0 or 1 present class: no ratio applies (or missing classes exist).
    SingleOrEmpty,
    /// Max/min ratio ≤ 5.0×.
    Balanced,
    /// 5.0× < ratio < 20.0×.
    Skewed,
    /// Ratio ≥ 20.0×, or one or more declared classes have zero boxes alongside populated classes.
    Severe,
}

/// Summary of one class (index, name, box count, share of total boxes).
#[derive(Clone, Debug, PartialEq)]
pub struct ClassCountSummary {
    pub index: u32,
    pub name: String,
    pub count: u64,
    pub share: f64,
}

/// Class-balance and annotation-quality metrics computed strictly from loaded `Dataset` + `DatasetStats`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassBalanceSummary {
    pub total_images: u64,
    pub total_boxes: u64,
    pub empty_images: u64,
    pub empty_share: f64,
    pub dominant: Option<ClassCountSummary>,
    pub rarest_present: Option<ClassCountSummary>,
    pub zero_classes: Vec<(u32, String)>,
    pub underrepresented_count: usize,
    /// Max-to-min ratio among classes with > 0 boxes (when at least 2 classes are present),
    /// rounded to one decimal place to match `health.rs`.
    pub imbalance_ratio: Option<f64>,
    pub severity: BalanceSeverity,
}

impl ClassBalanceSummary {
    /// Computes exact class-balance and annotation-quality metrics from `ds` and `s`.
    pub fn compute(ds: &Dataset, s: &DatasetStats) -> Self {
        let total_images = ds.splits.total();
        let empty_images = s.empty_images;
        let empty_share = if total_images > 0 { empty_images as f64 / total_images as f64 } else { 0.0 };

        let mut total_boxes = 0u64;
        let mut present: Vec<(u32, &str, u64)> = Vec::new();
        let mut zero_classes: Vec<(u32, String)> = Vec::new();
        for (i, name) in ds.classes.iter().enumerate() {
            let count = s.class_counts.get(i).copied().unwrap_or(0);
            total_boxes = total_boxes.saturating_add(count);
            if count == 0 {
                zero_classes.push((i as u32, name.clone()));
            } else {
                present.push((i as u32, name.as_str(), count));
            }
        }
        let denom = total_boxes.max(1) as f64;

        // Deterministic tie-breaking: lowest class index wins on equal counts.
        let dominant = present.iter().max_by(|a, b| a.2.cmp(&b.2).then_with(|| b.0.cmp(&a.0))).map(
            |&(index, name, count)| ClassCountSummary {
                index,
                name: name.to_string(),
                count,
                share: count as f64 / denom,
            },
        );
        let rarest_present = present.iter().min_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0))).map(
            |&(index, name, count)| ClassCountSummary {
                index,
                name: name.to_string(),
                count,
                share: count as f64 / denom,
            },
        );
        let underrepresented_count = present.iter().filter(|&&(_, _, c)| c < UNDERREPRESENTED_BOXES).count();

        let imbalance_ratio = match (&dominant, &rarest_present) {
            (Some(hi), Some(lo)) if present.len() >= 2 && lo.count > 0 => {
                Some(((hi.count as f64 / lo.count as f64) * 10.0).round() / 10.0)
            }
            _ => None,
        };
        let severity = if !zero_classes.is_empty() && !present.is_empty() {
            BalanceSeverity::Severe
        } else if let Some(ratio) = imbalance_ratio {
            if ratio >= SKEW_SEVERE {
                BalanceSeverity::Severe
            } else if ratio > SKEW_WARN {
                BalanceSeverity::Skewed
            } else {
                BalanceSeverity::Balanced
            }
        } else {
            BalanceSeverity::SingleOrEmpty
        };

        Self {
            total_images,
            total_boxes,
            empty_images,
            empty_share,
            dominant,
            rarest_present,
            zero_classes,
            underrepresented_count,
            imbalance_ratio,
            severity,
        }
    }

    /// Short badge label and tone strictly reflecting the numeric max/min imbalance ratio
    /// (zero-box classes are surfaced in their own Danger badge so a 1.0× ratio among present
    /// classes never reads as "1.0× severe skew").
    pub fn ratio_badge(&self) -> Option<(String, Tone)> {
        let r = self.imbalance_ratio?;
        if r >= SKEW_SEVERE {
            Some((format!("{r:.1}× severe skew"), Tone::Danger))
        } else if r > SKEW_WARN {
            Some((format!("{r:.1}× skew"), Tone::Warning))
        } else {
            Some((format!("{r:.1}× balanced"), Tone::Success))
        }
    }

    /// Separate Danger badge when one or more declared classes have zero boxes.
    pub fn unlabeled_classes_badge(&self) -> Option<(String, Tone)> {
        if self.zero_classes.is_empty() || self.dominant.is_none() {
            return None;
        }
        let n = self.zero_classes.len();
        Some((format!("{n} unlabeled {}", if n == 1 { "class" } else { "classes" }), Tone::Danger))
    }
}

fn fmt_share(frac: f64) -> String {
    let pct = frac * 100.0;
    if pct < 10.0 { format!("{pct:.1}%") } else { format!("{pct:.0}%") }
}

/// Truncates a class name for compact inline pills/badges (full name stays in the tooltip).
fn trunc_name(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() { format!("{head}…") } else { head }
}

impl FoloApp {
    /// Clears split and filename search restrictions for dataset `ds_id` when jumping from a
    /// dataset-wide statistic (rarest class / empty images) so the gallery always shows the
    /// advertised dataset-wide items rather than an empty intersection with stale filters.
    fn clear_hidden_gallery_filters(&mut self, ds_id: &str, cx: &mut Context<Self>) {
        self.ds_split = None;
        if self.ds_img_dataset.as_deref() == Some(ds_id) && !self.ds_img_query.is_empty() {
            self.ds_img_query.clear();
            self.ds_img_search.update(cx, |i, cx| i.set_text("", cx));
        }
    }

    /// Jumps straight to the Images tab filtered to `class_idx` on dataset `ds_id`.
    pub fn jump_dataset_class(
        &mut self,
        ds_id: String,
        class_idx: u32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.clear_hidden_gallery_filters(&ds_id, cx);
        self.ds_class = Some((ds_id.clone(), class_idx));
        if self.empty_only(&ds_id) {
            self.ds_empty_of = None;
        }
        self.ds_tab = DsTab::Images;
        self.ds_hover_tile = None;
        self.ds_scroll_anchor = None;
        self.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
        let fh = self.fh("ds-class", cx);
        window.focus(&fh);
        cx.notify();
    }

    /// Jumps straight to the Images tab filtered to empty (unlabeled) images on dataset `ds_id`.
    pub fn jump_dataset_empty(&mut self, ds_id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.clear_hidden_gallery_filters(&ds_id, cx);
        self.ds_empty_of = Some(ds_id.clone());
        if self.ds_class.as_ref().is_some_and(|(of, _)| *of == ds_id) {
            self.ds_class = None;
        }
        self.ds_tab = DsTab::Images;
        self.ds_hover_tile = None;
        self.ds_scroll_anchor = None;
        self.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
        let fh = self.fh("ds-empty-chip", cx);
        window.focus(&fh);
        cx.notify();
    }

    /// Compact inline quality & balance indicators placed on the right side of the Dataset Detail
    /// tab bar (zero extra vertical height, one-click filter jump to rarest class or empty images).
    pub(super) fn dataset_balance_quick_pills(
        &self,
        ds: &Dataset,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let s = ds.stats.as_ref()?;
        // Hide on narrower viewports so the Images / Statistics tabs keep full breathing room.
        if self.content_width(window) < 880. {
            return None;
        }
        let summary = ClassBalanceSummary::compute(ds, s);
        let t = theme(cx);
        let ratio = summary.ratio_badge();
        let unlabeled = summary.unlabeled_classes_badge();
        let rarest_pill =
            (ds.classes.len() >= 2).then_some(summary.rarest_present.as_ref()).flatten().map(|r| {
                let did = ds.id.clone();
                let idx = r.index;
                let active = self.ds_tab == DsTab::Images && self.class_filter(&ds.id) == Some(idx);
                let short_name = trunc_name(&r.name, 16);
                self.chip(
                    "ds-balance-rarest-pill",
                    format!("Rarest: {short_name}"),
                    active,
                    Some(idx),
                    Some(r.count),
                    window,
                    cx,
                    act(move |this, w, cx| {
                        if this.ds_tab == DsTab::Images && this.class_filter(&did) == Some(idx) {
                            this.ds_class = None;
                            this.ds_hover_tile = None;
                            this.ds_scroll_anchor = None;
                            this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                            cx.notify();
                        } else {
                            this.jump_dataset_class(did.clone(), idx, w, cx);
                        }
                    }),
                )
                .tooltip(crate::controls::tip(
                    format!(
                        "Rarest class: {} ({}, {} of boxes) · click to filter all splits",
                        r.name,
                        counted(r.count, "box", "boxes"),
                        fmt_share(r.share),
                    ),
                    None,
                ))
            });
        let empty_pill = (summary.empty_images > 0).then(|| {
            let did = ds.id.clone();
            let active = self.ds_tab == DsTab::Images && self.empty_only(&ds.id);
            self.chip(
                "ds-balance-empty-pill",
                "Empty",
                active,
                None,
                Some(summary.empty_images),
                window,
                cx,
                act(move |this, w, cx| {
                    if this.ds_tab == DsTab::Images && this.empty_only(&did) {
                        this.ds_empty_of = None;
                        this.ds_hover_tile = None;
                        this.ds_scroll_anchor = None;
                        this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                        cx.notify();
                    } else {
                        this.jump_dataset_empty(did.clone(), w, cx);
                    }
                }),
            )
            .tooltip(crate::controls::tip(
                format!(
                    "{} ({}) without boxes · click to inspect across all splits",
                    counted(summary.empty_images, "empty image", "empty images"),
                    fmt_share(summary.empty_share),
                ),
                None,
            ))
        });
        if ratio.is_none() && unlabeled.is_none() && rarest_pill.is_none() && empty_pill.is_none() {
            return None;
        }
        Some(
            div()
                .min_w_0()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_end()
                .gap(px(8.))
                .when_some(ratio, |d, (label, tone)| d.child(badge(&t, label, tone)))
                .when_some(unlabeled, |d, (label, tone)| d.child(badge(&t, label, tone)))
                .children(rarest_pill)
                .children(empty_pill)
                .into_any_element(),
        )
    }

    /// Full Class-Balance & Annotation Quality Strip embedded at the top of the Statistics tab's
    /// Class balance panel, with imbalance ratio, rarest-class indicator, empty/unlabeled image
    /// summary, and one-click filter jump actions into the gallery.
    pub(super) fn dataset_balance_strip(
        &self,
        t: &Theme,
        ds: &Dataset,
        s: &DatasetStats,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let summary = ClassBalanceSummary::compute(ds, s);
        let ratio_el = match summary.ratio_badge() {
            Some((label, tone)) => {
                let detail = match (&summary.dominant, &summary.rarest_present) {
                    (Some(hi), Some(lo)) if hi.index != lo.index => format!(
                        "{} ({}) vs {} ({})",
                        trunc_name(&hi.name, 18),
                        int(hi.count),
                        trunc_name(&lo.name, 18),
                        int(lo.count)
                    ),
                    _ => "max : min class boxes".to_string(),
                };
                div()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(8.))
                    .child(badge(t, label, tone))
                    .child(text(ty::SM, t.text_tertiary).min_w_0().tnum().child(detail))
            }
            None => div().child(text(ty::SM, t.text_tertiary).child("Single-class or unannotated dataset")),
        };

        let rarest_el =
            (ds.classes.len() >= 2).then_some(summary.rarest_present.as_ref()).flatten().map(|r| {
                let did = ds.id.clone();
                let idx = r.index;
                let btn = self.button(
                    "ds-balance-rarest",
                    "Inspect rarest",
                    Btn::Secondary,
                    Some("icons/list-filter.svg"),
                    window,
                    cx,
                    act(move |this, w, cx| this.jump_dataset_class(did.clone(), idx, w, cx)),
                );
                div()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_tertiary).child("Rarest"))
                    .child(class_swatch(t, r.index, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .max_w(px(180.))
                            .ellipsis()
                            .child(trunc_name(&r.name, 22)),
                    )
                    .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                        "{} ({})",
                        counted(r.count, "box", "boxes"),
                        fmt_share(r.share)
                    )))
                    .child(btn)
            });

        let (empty_label, empty_tone) = if summary.total_images == 0 {
            ("no images".to_string(), Tone::Neutral)
        } else if summary.empty_images > 0 {
            let tone = if summary.empty_share > 0.10 { Tone::Warning } else { Tone::Neutral };
            (
                format!(
                    "{} ({})",
                    counted(summary.empty_images, "empty image", "empty images"),
                    fmt_share(summary.empty_share)
                ),
                tone,
            )
        } else {
            ("0 empty images".to_string(), Tone::Success)
        };
        let empty_btn = (summary.empty_images > 0).then(|| {
            let did = ds.id.clone();
            self.button(
                "ds-balance-empty",
                "Inspect empty",
                Btn::Secondary,
                Some("icons/image.svg"),
                window,
                cx,
                act(move |this, w, cx| this.jump_dataset_empty(did.clone(), w, cx)),
            )
        });
        let empty_el = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(8.))
            .child(badge(t, empty_label, empty_tone))
            .children(empty_btn);

        let zero_note = (!summary.zero_classes.is_empty()).then(|| {
            let names: Vec<String> =
                summary.zero_classes.iter().take(2).map(|(_, n)| trunc_name(n, 16)).collect();
            let extra = summary.zero_classes.len().saturating_sub(2);
            let suffix = if extra > 0 { format!(" +{extra} more") } else { String::new() };
            let full_tip =
                summary.zero_classes.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>().join(", ");
            div()
                .id("ds-balance-zero-note")
                .tooltip(crate::controls::tip(format!("Classes with 0 boxes: {full_tip}"), None))
                .child(badge(t, format!("0 boxes: {}{suffix}", names.join(", ")), Tone::Danger))
        });

        let few_note = (summary.underrepresented_count > 0).then(|| {
            badge(
                t,
                format!(
                    "{} under {UNDERREPRESENTED_BOXES} boxes",
                    counted(summary.underrepresented_count as u64, "class", "classes")
                ),
                Tone::Warning,
            )
        });

        div()
            .px(px(16.))
            .py(px(10.))
            .bg(t.bg_inset)
            .border_b_1()
            .border_color(t.line_subtle)
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .child(
                div()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(10.))
                    .child(ratio_el)
                    .children(zero_note)
                    .children(few_note),
            )
            .child(
                div()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(12.))
                    .children(rarest_el)
                    .child(empty_el),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, mock::MockBackend};

    #[test]
    fn computes_imbalance_ratio_rarest_zero_classes_and_empty_share() {
        let api = MockBackend::new(4200, true);
        let mut ds = api.dataset("ds_3b91").unwrap();
        let mut s = ds.stats.clone().unwrap();

        // Balanced dataset (ratio 2.0x <= 5.0x).
        ds.classes = vec!["circle".into(), "square".into(), "triangle".into()];
        ds.splits.train = 80;
        ds.splits.val = 20;
        ds.splits.test = 0;
        s.class_counts = vec![400, 300, 200];
        s.empty_images = 5;
        let b = ClassBalanceSummary::compute(&ds, &s);
        assert_eq!(b.imbalance_ratio, Some(2.0));
        assert_eq!(b.severity, BalanceSeverity::Balanced);
        assert_eq!(b.dominant.as_ref().unwrap().name, "circle");
        assert_eq!(b.rarest_present.as_ref().unwrap().name, "triangle");
        assert_eq!(b.rarest_present.as_ref().unwrap().index, 2);
        assert_eq!(b.empty_images, 5);
        assert!((b.empty_share - 0.05).abs() < 1e-6);
        assert_eq!(b.underrepresented_count, 0);
        assert_eq!(b.ratio_badge(), Some(("2.0× balanced".into(), Tone::Success)));
        assert_eq!(b.unlabeled_classes_badge(), None);

        // Skewed dataset (ratio 10.0x -> Warning).
        s.class_counts = vec![500, 250, 50];
        let sk = ClassBalanceSummary::compute(&ds, &s);
        assert_eq!(sk.imbalance_ratio, Some(10.0));
        assert_eq!(sk.severity, BalanceSeverity::Skewed);
        assert_eq!(sk.underrepresented_count, 1);
        assert_eq!(sk.ratio_badge(), Some(("10.0× skew".into(), Tone::Warning)));

        // Severe skew (>= 20.0x -> Danger).
        s.class_counts = vec![1000, 200, 40];
        let sev = ClassBalanceSummary::compute(&ds, &s);
        assert_eq!(sev.imbalance_ratio, Some(25.0));
        assert_eq!(sev.severity, BalanceSeverity::Severe);
        assert_eq!(sev.ratio_badge(), Some(("25.0× severe skew".into(), Tone::Danger)));

        // Missing class (0 boxes) marks overall severity Severe, but ratio badge stays honest
        // ("1.0× balanced") while unlabeled_classes_badge surfaces the missing class in Danger.
        s.class_counts = vec![300, 300, 0];
        let zero = ClassBalanceSummary::compute(&ds, &s);
        assert_eq!(zero.zero_classes, vec![(2, "triangle".into())]);
        assert_eq!(zero.severity, BalanceSeverity::Severe);
        assert_eq!(zero.rarest_present.as_ref().unwrap().index, 0);
        assert_eq!(zero.ratio_badge(), Some(("1.0× balanced".into(), Tone::Success)));
        assert_eq!(zero.unlabeled_classes_badge(), Some(("1 unlabeled class".into(), Tone::Danger)));
    }

    #[test]
    fn single_class_and_empty_dataset_handle_without_panicking() {
        let api = MockBackend::new(4200, true);
        let mut ds = api.dataset("ds_3b91").unwrap();
        let mut s = ds.stats.clone().unwrap();
        ds.classes = vec!["only".into()];
        ds.splits.train = 0;
        ds.splits.val = 0;
        ds.splits.test = 0;
        s.class_counts = vec![0];
        s.empty_images = 0;
        let b = ClassBalanceSummary::compute(&ds, &s);
        assert_eq!(b.imbalance_ratio, None);
        assert_eq!(b.severity, BalanceSeverity::SingleOrEmpty);
        assert!(b.rarest_present.is_none());
        assert_eq!(b.empty_share, 0.0);
        assert_eq!(b.ratio_badge(), None);
        assert_eq!(b.unlabeled_classes_badge(), None);
    }

    #[test]
    fn trunc_name_bounds_long_class_names_cleanly() {
        assert_eq!(trunc_name("short", 16), "short");
        assert_eq!(trunc_name("a-very-long-class-name-for-testing", 10), "a-very-lon…");
    }
}
