//! Dataset Class-Balance & Annotation Quality Strip: computes exact max-to-min class imbalance
//! ratios, rarest-class and zero-box class indicators, and empty/unlabeled image stats strictly
//! from loaded `Dataset` / `DatasetStats`, with one-click filter jumps into the dataset gallery.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window,
    div, px,
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

    /// Short badge label and tone for the imbalance ratio.
    pub fn ratio_badge(&self) -> Option<(String, Tone)> {
        if let Some(r) = self.imbalance_ratio {
            let tone = match self.severity {
                BalanceSeverity::Severe => Tone::Danger,
                BalanceSeverity::Skewed => Tone::Warning,
                BalanceSeverity::Balanced => Tone::Success,
                BalanceSeverity::SingleOrEmpty => Tone::Neutral,
            };
            let suffix = match self.severity {
                BalanceSeverity::Severe => "severe skew",
                BalanceSeverity::Skewed => "skew",
                BalanceSeverity::Balanced => "balanced",
                BalanceSeverity::SingleOrEmpty => "ratio",
            };
            Some((format!("{r:.1}× {suffix}"), tone))
        } else if !self.zero_classes.is_empty() && self.dominant.is_some() {
            Some((
                format!(
                    "{} unlabeled {}",
                    self.zero_classes.len(),
                    if self.zero_classes.len() == 1 { "class" } else { "classes" }
                ),
                Tone::Danger,
            ))
        } else {
            None
        }
    }
}

fn fmt_share(frac: f64) -> String {
    let pct = frac * 100.0;
    if pct < 10.0 { format!("{pct:.1}%") } else { format!("{pct:.0}%") }
}

impl FoloApp {
    /// Jumps straight to the Images tab filtered to `class_idx` on dataset `ds_id`.
    pub fn jump_dataset_class(
        &mut self,
        ds_id: String,
        class_idx: u32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
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
        // Hide on very narrow viewports so the Images / Statistics tabs keep full breathing room.
        if self.content_width(window) < 820. {
            return None;
        }
        let summary = ClassBalanceSummary::compute(ds, s);
        let t = theme(cx);
        let ratio = summary.ratio_badge();
        let rarest_pill = summary.rarest_present.as_ref().map(|r| {
            let did = ds.id.clone();
            let idx = r.index;
            let active = self.ds_tab == DsTab::Images && self.class_filter(&ds.id) == Some(idx);
            self.chip(
                "ds-balance-rarest-pill",
                format!("Rarest: {}", r.name),
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
                    "Least-represented class ({}, {} of boxes) · click to filter gallery",
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
                    "{} ({}) without boxes · click to inspect in gallery",
                    counted(summary.empty_images, "empty image", "empty images"),
                    fmt_share(summary.empty_share),
                ),
                None,
            ))
        });
        if ratio.is_none() && rarest_pill.is_none() && empty_pill.is_none() {
            return None;
        }
        Some(
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .when_some(ratio, |d, (label, tone)| d.child(badge(&t, label, tone)))
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
                    (Some(hi), Some(lo)) if hi.index != lo.index => {
                        format!("{} ({}) vs {} ({})", hi.name, int(hi.count), lo.name, int(lo.count))
                    }
                    _ => "max : min class boxes".to_string(),
                };
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(badge(t, label, tone))
                    .child(text(ty::SM, t.text_tertiary).tnum().child(detail))
            }
            None => div().child(text(ty::SM, t.text_tertiary).child("Single-class or unannotated dataset")),
        };

        let rarest_el = summary.rarest_present.as_ref().map(|r| {
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
                .flex()
                .items_center()
                .gap(px(8.))
                .child(text(ty::SM, t.text_tertiary).child("Rarest"))
                .child(class_swatch(t, r.index, 8.))
                .child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).child(r.name.clone()))
                .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                    "{} ({})",
                    counted(r.count, "box", "boxes"),
                    fmt_share(r.share)
                )))
                .child(btn)
        });

        let empty_tone = if summary.empty_share > 0.10 {
            Tone::Warning
        } else if summary.empty_images > 0 {
            Tone::Neutral
        } else {
            Tone::Success
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
            .items_center()
            .gap(px(8.))
            .child(badge(
                t,
                if summary.empty_images > 0 {
                    format!(
                        "{} ({})",
                        counted(summary.empty_images, "empty image", "empty images"),
                        fmt_share(summary.empty_share)
                    )
                } else {
                    "0 empty images".to_string()
                },
                empty_tone,
            ))
            .children(empty_btn);

        let zero_note = (!summary.zero_classes.is_empty()).then(|| {
            let names: Vec<&str> = summary.zero_classes.iter().take(3).map(|(_, n)| n.as_str()).collect();
            let extra = summary.zero_classes.len().saturating_sub(3);
            let suffix = if extra > 0 { format!(" +{extra} more") } else { String::new() };
            badge(t, format!("0 boxes: {}{suffix}", names.join(", ")), Tone::Danger)
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
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(12.))
                    .child(ratio_el)
                    .children(zero_note)
                    .children(few_note),
            )
            .child(div().flex().flex_wrap().items_center().gap(px(12.)).children(rarest_el).child(empty_el))
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

        // Missing class (0 boxes) marks Severe even if present classes are equal.
        s.class_counts = vec![300, 300, 0];
        let zero = ClassBalanceSummary::compute(&ds, &s);
        assert_eq!(zero.zero_classes, vec![(2, "triangle".into())]);
        assert_eq!(zero.severity, BalanceSeverity::Severe);
        assert_eq!(zero.rarest_present.as_ref().unwrap().index, 0);
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
    }
}
