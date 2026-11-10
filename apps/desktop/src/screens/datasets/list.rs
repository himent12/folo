//! The dataset index, and class names summarised to fit.

use gpui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::SplitsExt;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::theme;
use crate::ui::*;

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
