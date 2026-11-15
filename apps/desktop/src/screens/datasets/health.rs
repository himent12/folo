//! A dataset's health: what stands out in its statistics and in `folo dataset check`'s counts, most
//! severe first. The rules and thresholds are Studio's (`routes/dataset/health.ts`), so both apps
//! count the same findings: the Statistics tab's count and its one-line summary.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    div, px, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement, Styled,
};

use crate::api::{Dataset, DatasetStats, Histogram, IssueCounts, SplitsExt};
use crate::theme::{ty, Theme};
use crate::ui::*;

/// How much a finding matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Danger,
    Warning,
    /// Nothing stands out (only when there is no danger or warning).
    Ok,
    Info,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub severity: Severity,
    /// A few words with the unit named, for the one-line summary ("3 check warnings").
    pub short: String,
    /// What it means and what to do, for its tooltip.
    pub detail: &'static str,
}

/// Imbalance (largest / smallest class) above which a dataset is warned about.
const IMBALANCE_WARN: f64 = 5.;
/// Boxes under this share of the image side (sqrt of the area share) are tiny: about 32 px at 640.
const TINY_SIZE: f64 = 0.05;
/// Share of tiny boxes above which a dataset is warned about.
const TINY_WARN: f64 = 0.1;
/// Share of empty images above which a dataset is warned about.
const EMPTY_WARN: f64 = 0.1;
/// Boxes per class under which a class is under-represented.
const FEW_BOXES: u64 = 100;

/// Fraction of `h`'s mass in bins that lie entirely below `limit`.
fn share_below(h: &Histogram, limit: f64) -> f64 {
    let (mut below, mut total) = (0u64, 0u64);
    for (i, &c) in h.counts.iter().enumerate() {
        total += c;
        if h.edges.get(i + 1).is_some_and(|&e| e <= limit + 1e-12) {
            below += c;
        }
    }
    if total == 0 {
        0.
    } else {
        below as f64 / total as f64
    }
}

/// "35%", "0.8%": one decimal under 10 %.
fn share(x: f64) -> String {
    if x < 0.1 {
        format!("{:.1}%", x * 100.)
    } else {
        format!("{:.0}%", x * 100.)
    }
}

fn n_of(n: u64, one: &str, many: &str) -> String {
    format!("{} {}", int(n), if n == 1 { one } else { many })
}

/// The findings, most severe first; one `Ok` when nothing is a danger or a warning. `counts` (the
/// check's, when it has run) adds what the stats can't see: unreadable images, check warnings.
pub fn health_checks(d: &Dataset, s: &DatasetStats, counts: Option<&IssueCounts>) -> Vec<Finding> {
    let mut out = Vec::new();
    let images = d.splits.total();
    let mut push = |severity, short: String, detail| out.push(Finding { severity, short, detail });
    if s.broken_labels > 0 {
        push(
            Severity::Danger,
            n_of(s.broken_labels, "broken label file", "broken label files"),
            "These labels could not be parsed, so their images are skipped in training. Fix or delete them.",
        );
        let kind = |k: &str| counts.and_then(|c| c.by_kind.get(k).copied()).unwrap_or(0);
        let other = kind("corrupt_image") + kind("missing_image");
        if other > 0 {
            push(
                Severity::Danger,
                n_of(other, "unreadable image", "unreadable images"),
                "Images that cannot be read are left out of training. Fix or remove them.",
            );
        }
    } else if let Some(c) = counts.filter(|c| c.error > 0) {
        push(
            Severity::Danger,
            n_of(c.error, "file error", "file errors"),
            "Images that cannot be read and labels that cannot be parsed are left out of training.",
        );
    }
    if let Some(c) = counts.filter(|c| c.warning > 0) {
        push(
            Severity::Warning,
            n_of(c.warning, "check warning", "check warnings"),
            "Training still uses these files: boxes past the image edge, duplicate labels or images.",
        );
    }
    let missing = d
        .classes
        .iter()
        .enumerate()
        .filter(|(i, _)| s.class_counts.get(*i).copied().unwrap_or(0) == 0)
        .count();
    if missing > 0 {
        push(
            Severity::Danger,
            format!("{missing} {} without boxes", if missing == 1 { "class" } else { "classes" }),
            "The model can never learn a class it has not seen. Label some examples or remove the class.",
        );
    }
    let present: Vec<u64> = s.class_counts.iter().copied().filter(|&c| c > 0).collect();
    if present.len() >= 2 {
        let (hi, lo) = (*present.iter().max().unwrap_or(&1), *present.iter().min().unwrap_or(&1));
        // Compared as shown (one decimal), so "5.0×" never warns.
        let ratio = (hi as f64 / lo as f64 * 10.).round() / 10.;
        if ratio > IMBALANCE_WARN {
            push(
                Severity::Warning,
                format!("{ratio:.1}× class imbalance"),
                "Rare classes usually score lower; add examples or oversample them.",
            );
        }
    }
    let few = present.iter().filter(|&&c| c < FEW_BOXES).count();
    if few > 0 {
        push(
            Severity::Warning,
            format!("{few} {} under {FEW_BOXES} boxes", if few == 1 { "class" } else { "classes" }),
            "Aim for a few hundred boxes per class or more.",
        );
    }
    let tiny = share_below(&s.box_size_hist, TINY_SIZE);
    if tiny > TINY_WARN {
        push(
            Severity::Warning,
            format!("{} tiny boxes", share(tiny)),
            "They are under 5% of the image side (about 32 px at 640). Train at a larger image size or tile large images.",
        );
    }
    if d.splits.val == 0 {
        push(
            Severity::Warning,
            "no validation split".into(),
            "Training and evaluation use the training split instead. Move 10–20% of the images into a val split.",
        );
    }
    let notes = counts.map_or(0, |c| c.info);
    if images > 0 && s.empty_images > 0 {
        let e = s.empty_images as f64 / images as f64;
        push(
            if e > EMPTY_WARN { Severity::Warning } else { Severity::Info },
            format!("{} ({})", n_of(s.empty_images, "empty image", "empty images"), share(e)),
            if e > EMPTY_WARN {
                "Background images reduce false positives, but above ~10% they slow learning. Check that labels were not lost."
            } else {
                "Images without boxes act as background examples and reduce false positives. Up to ~10% is healthy."
            },
        );
    } else if notes > 0 {
        push(
            Severity::Info,
            n_of(notes, "check note", "check notes"),
            "Images without boxes: background examples.",
        );
    }
    if images > 0 && images < 200 {
        push(
            Severity::Info,
            format!("only {}", n_of(images, "image", "images")),
            "Expect overfitting; keep augmentation on and start from pretrained weights.",
        );
    }
    if !out.iter().any(|f| matches!(f.severity, Severity::Danger | Severity::Warning)) {
        out.insert(
            0,
            Finding {
                severity: Severity::Ok,
                short: "no problems found".into(),
                detail: "Labels parse, every class is represented and the class balance is within 5×.",
            },
        );
    }
    out.sort_by_key(|f| f.severity);
    out
}

/// How many findings there are (not the `Ok` one): the Statistics tab's count.
pub fn finding_count(items: &[Finding]) -> usize {
    items.iter().filter(|f| f.severity != Severity::Ok).count()
}

/// About how wide a finding's words are, at the summary's text size (an upper bound).
fn words_w(s: &str) -> f32 {
    let mut w = 0.;
    while crate::ui::may_overflow(s, ty::SM.size, w) {
        w += 4.;
    }
    w
}

/// How many findings fit on a summary line `width` wide (the panel's), most severe first; the rest
/// are counted in a "+N more" at its end.
pub fn findings_that_fit(items: &[Finding], width: f32) -> usize {
    // The panel's padding, "Health" and its gap; room kept for "+N more".
    let mut room = width - 32. - 64. - 72.;
    let mut n = 0;
    for f in items {
        // Its mark, the mark's gap, the words and the gap to the next.
        let need = 14. + 6. + words_w(&f.short) + 16.;
        if need > room {
            break;
        }
        room -= need;
        n += 1;
    }
    // The last one may take the room kept for "+N more" when nothing follows it.
    if n + 1 == items.len() && 14. + 6. + words_w(&items[n].short) <= room + 72. {
        n += 1;
    }
    n
}

/// The Statistics tab's one-line health summary: "Health", then each finding that fits with its
/// severity's mark (its explanation in its tooltip), then "+N more" for the rest (named in its
/// tooltip). The caller makes the line a Tab stop whose focus tip lists every finding in full.
pub fn health_line(t: &Theme, items: &[Finding], width: f32) -> folo_ui::Div {
    let shown = findings_that_fit(items, width);
    let rest: Vec<String> = items[shown..].iter().map(|f| f.short.clone()).collect();
    panel(t)
        .h(px(40.))
        .px(px(16.))
        .flex()
        .items_center()
        .gap(px(16.))
        .overflow_hidden()
        .child(
            text(ty::BASE, t.text_primary)
                .font_weight(folo_ui::FontWeight::MEDIUM)
                .flex_none()
                .child("Health"),
        )
        .children(items[..shown].iter().enumerate().map(|(i, f)| {
            let (mark, color, ink) = match f.severity {
                Severity::Danger => ("icons/circle-alert.svg", t.danger_solid, t.text_primary),
                Severity::Warning => ("icons/triangle-alert.svg", t.warning_solid, t.text_primary),
                Severity::Ok => ("icons/circle-check.svg", t.success_solid, t.text_secondary),
                Severity::Info => ("icons/info.svg", t.text_tertiary, t.text_secondary),
            };
            div()
                .id(SharedString::from(format!("ds-health-{i}")))
                .flex()
                .flex_none()
                .items_center()
                .gap(px(6.))
                .tooltip(crate::controls::tip(f.detail, None))
                .child(icon(mark, 14., color))
                .child(
                    text(ty::SM, ink)
                        .tnum()
                        .whitespace_nowrap()
                        .when(f.severity == Severity::Danger, |d| d.font_weight(folo_ui::FontWeight::MEDIUM))
                        .child(f.short.clone()),
                )
                .into_any_element()
        }))
        .when(!rest.is_empty(), |d| {
            d.child(
                div()
                    .id("ds-health-more")
                    .flex_none()
                    .tooltip(crate::controls::tip(rest.join(" · "), None))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .whitespace_nowrap()
                            .child(format!("+{} more", rest.len())),
                    ),
            )
        })
}

/// Every finding in full, for the summary's focus tip.
pub fn findings_text(items: &[Finding]) -> String {
    items.iter().map(|f| format!("{}: {}", f.short, f.detail)).collect::<Vec<_>>().join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mock::MockBackend;
    use crate::api::Backend;

    fn hist(edges: &[f64], counts: &[u64]) -> Histogram {
        Histogram { edges: edges.to_vec(), counts: counts.to_vec() }
    }

    #[test]
    fn findings_are_counted_as_studio_counts_them() {
        let api = MockBackend::new(4200, true);
        let mut d = api.dataset("ds_3b91").unwrap();
        let mut s = d.stats.clone().unwrap();
        s.broken_labels = 3;
        s.class_counts = vec![1000, 900, 100, 0, 950];
        s.box_size_hist = hist(&[0., 0.02, 0.05, 1.], &[30, 10, 60]);
        s.empty_images = 40;
        let counts = IssueCounts { error: 3, warning: 2, info: 0, by_kind: Default::default() };
        let items = health_checks(&d, &s, Some(&counts));
        let shorts: Vec<&str> = items.iter().map(|f| f.short.as_str()).collect();
        // Most severe first; the broken files, the check's warnings, the empty class, the imbalance
        // (1,000 / 100 = 10×), the tiny boxes (40 %).
        assert_eq!(items[0].short, "3 broken label files");
        assert!(
            shorts.contains(&"2 check warnings") && shorts.contains(&"1 class without boxes"),
            "{shorts:?}"
        );
        assert!(
            shorts.contains(&"10.0× class imbalance") && shorts.contains(&"40% tiny boxes"),
            "{shorts:?}"
        );
        assert!(items.windows(2).all(|w| w[0].severity <= w[1].severity));
        assert_eq!(finding_count(&items), items.len());

        // A narrow line keeps the most severe and counts the rest; a wide one shows them all.
        let fit = findings_that_fit(&items, 560.);
        assert!(fit >= 1 && fit < items.len(), "{fit} of {}", items.len());
        assert_eq!(findings_that_fit(&items, 4000.), items.len());

        // A clean dataset: one Ok, no findings counted.
        d.splits.val = d.splits.val.max(1);
        let clean = DatasetStats {
            broken_labels: 0,
            empty_images: 0,
            class_counts: d.classes.iter().map(|_| 500).collect(),
            box_size_hist: hist(&[0., 0.05, 1.], &[0, 100]),
            ..s
        };
        let items = health_checks(&d, &clean, None);
        assert_eq!(items[0].severity, Severity::Ok);
        assert_eq!(finding_count(&items), items.iter().filter(|f| f.severity == Severity::Info).count());
        // 5.0× exactly is not an imbalance.
        let mut five = clean.clone();
        five.class_counts = vec![500, 100];
        assert!(!health_checks(&d, &five, None).iter().any(|f| f.short.contains("imbalance")));
    }

    #[test]
    fn a_small_dataset_is_counted_in_the_singular_at_one() {
        let api = MockBackend::new(4200, true);
        let mut d = api.dataset("ds_3b91").unwrap();
        let s = d.stats.clone().unwrap();
        let small = |d: &Dataset| {
            health_checks(d, &s, None).into_iter().map(|f| f.short).find(|s| s.starts_with("only "))
        };
        for (train, val, said) in [(1, 0, Some("only 1 image")), (1, 1, Some("only 2 images")), (0, 0, None)]
        {
            (d.splits.train, d.splits.val, d.splits.test) = (train, val, 0);
            assert_eq!(small(&d).as_deref(), said, "{train} + {val}");
        }
    }
}
