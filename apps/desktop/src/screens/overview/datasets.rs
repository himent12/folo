//! The datasets strip and its cards, and the panels' quiet links ("All datasets →").

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Label, SplitsExt};
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::cell2;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn dataset_strip(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-ds-all", "All datasets", Route::Datasets, window, cx);
        let datasets = self.store.datasets.ready().cloned().unwrap_or_default();
        let cards: Vec<AnyElement> =
            datasets.iter().take(3).map(|d| self.dataset_card(d, window, cx)).collect();
        // A list that failed says why, with Retry: not skeletons for good (a damaged registry on the
        // server answers 500 with how to fix it).
        let failed = self.store.datasets.error().cloned().map(|e| {
            let retry = self.button(
                "ov-ds-retry",
                "Retry",
                Btn::Primary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(|this, _, cx| this.retry_all(cx)),
            );
            let hint = e.hint.clone().unwrap_or_else(|| format!("Error code {}", e.code));
            empty_state(&t, "icons/circle-alert.svg", &e.message, &hint, Tone::Danger).child(retry)
        });
        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Datasets").child(all))
            .map(|d| {
                if let Some(failed) = failed {
                    d.child(failed)
                } else if cards.is_empty() && self.store.datasets.ready().is_some() {
                    d.child(empty_state(
                        &t,
                        "icons/images.svg",
                        "No datasets registered",
                        "Datasets you add or generate show up here with a preview of their images.",
                        Tone::Neutral,
                    ))
                } else if cards.is_empty() {
                    d.child(div().p(px(16.)).flex().gap(px(16.)).children(
                        (0..3).map(|i| div().flex_1().h(px(196.)).child(skeleton(&t, format!("ov-ds-{i}")))),
                    ))
                } else {
                    let n = cards.len();
                    d.child(
                        div()
                            .p(px(16.))
                            .flex()
                            .gap(px(16.))
                            .children(cards.into_iter().map(|c| {
                                div()
                                    .flex_grow()
                                    .flex_basis(px(0.))
                                    .min_w_0()
                                    .flex()
                                    .child(c)
                                    .into_any_element()
                            }))
                            .children((n..3).map(|_| div().flex_grow().flex_basis(px(0.)).min_w_0())),
                    )
                }
            })
            .into_any_element()
    }

    /// Dataset card: a 2×2 mosaic of its first images, name and counts, and its class balance along
    /// the bottom edge.
    pub fn dataset_card(
        &mut self,
        d: &crate::api::Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        self.ensure_dataset(&d.id, cx);
        let metas: Vec<crate::api::ImageInfo> = self
            .store
            .images
            .get(&d.id)
            .and_then(|l| l.ready())
            .map(|v| v.iter().filter(|m| !m.boxes.is_empty()).take(4).cloned().collect())
            .unwrap_or_default();
        let tiles: Vec<AnyElement> = (0..4)
            .map(|i| match metas.get(i) {
                Some(m) => {
                    // Outlines only: the card previews the data, its page names every box.
                    let tile = self.tile(&d.id, m, true, false, crate::viz::Fit::Cover, false, cx);
                    div().flex_1().h_full().overflow_hidden().child(tile).into_any_element()
                }
                None => div().flex_1().h_full().bg(t.bg_raised).into_any_element(),
            })
            .collect();
        let mut it = tiles.into_iter();
        let row1: Vec<AnyElement> = it.by_ref().take(2).collect();
        let row2: Vec<AnyElement> = it.collect();
        let id = d.id.clone();
        let key = format!("ds-card-{}", d.id);
        // Boxes per class, from the dataset's own record (the list has no stats).
        let counts = self
            .store
            .dataset
            .get(&d.id)
            .and_then(|l| l.ready())
            .and_then(|x| x.stats.as_ref())
            .map(|s| s.class_counts.clone());
        let (badge_label, badge_tone, problem) = self.dataset_card_badge(d);
        // The card in full: its name, size and every class with its boxes (the strip's breakdown), on
        // hover and on keyboard focus; what is wrong with it, when something is.
        let mut full = format!(
            "{} · {} · {}",
            d.name,
            counted(d.splits.total(), "image", "images"),
            counts.as_deref().map_or_else(|| d.classes.join(", "), |c| balance_text(&d.classes, c))
        );
        if let Some(why) = &problem {
            full = format!("{full} · {badge_label}: {why}");
        }
        let focused = self.ring(&self.fh(key.clone(), cx), window);
        let card = div()
            .id(SharedString::from(key.clone()))
            .tooltip(crate::controls::tip(full.clone(), None))
            // Under the card (its mosaic and footer are about 196 px tall), kept inside the window.
            .when(focused, |d| d.child(crate::controls::focus_tip(&t, full.clone(), 196.)))
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .rounded(px(radius::LG))
            .bg(t.bg_panel)
            .border_1()
            .border_color(t.line_default)
            .overflow_hidden()
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_strong))
            .active(|s| s.bg(crate::theme::over(t.bg_panel, t.bg_active)))
            .child(
                div()
                    .h(px(136.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .bg(t.bg_inset)
                    .child(div().flex_1().flex().gap(px(1.)).children(row1))
                    .child(div().flex_1().flex().gap(px(1.)).children(row2)),
            )
            .child(
                div()
                    .p(px(12.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .child(cell2(
                        &t,
                        d.name.clone(),
                        // Counts that never need cutting; the class names are the card's tip (hover
                        // and keyboard focus), in full.
                        format!(
                            "{} · {}",
                            counted(d.splits.total(), "image", "images"),
                            if d.classes.len() == 1 {
                                "1 class".to_string()
                            } else {
                                format!("{} classes", d.classes.len())
                            }
                        ),
                    ))
                    .child(badge(&t, badge_label, badge_tone)),
            )
            .child(class_strip(&t, counts.as_deref()));
        self.bind(
            card,
            &key,
            radius::LG,
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Dataset(id.clone()), w, cx)),
        )
        .into_any_element()
    }

    /// A dataset card's badge: its format, or what is wrong with it when its own read failed
    /// (folder gone, images deleted), as in Studio, with the server's reason for the card's tip.
    /// Without it the card looked like a healthy dataset whose previews are still loading.
    pub(crate) fn dataset_card_badge(&self, d: &crate::api::Dataset) -> (&'static str, Tone, Option<String>) {
        let failed = self.store.dataset.get(&d.id).and_then(|l| l.error());
        match failed.and_then(|e| dataset_problem(e).map(|label| (label, e.message.clone()))) {
            Some((label, why)) => (label, Tone::Warning, Some(why)),
            None => (d.format.label(), Tone::Neutral, None),
        }
    }

    /// A panel header's quiet link to where it leads ("All runs →"): secondary ink with the arrow
    /// after the words, ink and the hover fill under the pointer (Studio's small ghost button).
    pub fn quiet_link(
        &self,
        key: &str,
        label: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
        action: crate::app::Action,
    ) -> AnyElement {
        let t = theme(cx);
        let group = SharedString::from(format!("{key}-link"));
        let el = div()
            .id(SharedString::from(key.to_string()))
            .group(group.clone())
            .flex()
            .items_center()
            .gap(px(4.))
            .px(px(6.))
            .mr(px(-6.))
            .h(px(22.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .child(
                text(ty::SM, t.text_secondary)
                    .whitespace_nowrap()
                    .group_hover(group.clone(), |s| s.text_color(t.text_primary))
                    .child(label),
            )
            .child(
                icon("icons/arrow-right.svg", 12., t.text_secondary)
                    .group_hover(group, |s| s.text_color(t.text_primary)),
            );
        self.bind(el, key, radius::SM, window, cx, action).into_any_element()
    }

    /// The quiet link from a panel to the list it previews ("All datasets →").
    pub fn view_all(
        &self,
        key: &str,
        label: &'static str,
        route: Route,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        self.quiet_link(key, label, window, cx, act(move |this, w, cx| this.navigate(route.clone(), w, cx)))
    }
}

/// What a failed read of a listed dataset means for its card, in Studio's words
/// (web/src/routes/datasets/problem.ts): its folder is gone, or the folder no longer reads as a
/// dataset. Nothing for a failure that is not about the dataset (the server is unreachable, the
/// token is wrong, the dataset was just deleted).
pub(crate) fn dataset_problem(e: &crate::api::ApiError) -> Option<&'static str> {
    match e.code.as_str() {
        "path_not_found" => Some("Folder missing"),
        "server_unreachable" | "unauthorized" | "http_error" | "bad_response" | "not_supported"
        | "dataset_not_found" | "internal" => None,
        _ => Some("Can't read"),
    }
}

/// The class-balance strip along a dataset card's bottom edge: one segment per class with boxes,
/// as wide as its share, in the class's colour, 1 px of the card between segments (Studio's
/// `ClassStrip`). An empty track until the stats are in, and for a dataset without boxes.
fn class_strip(t: &crate::theme::Theme, counts: Option<&[u64]>) -> folo_ui::Div {
    let strip = div().h(px(STRIP_H)).w_full().flex().gap(px(1.)).bg(t.bg_active);
    match counts.filter(|c| c.iter().any(|&n| n > 0)) {
        None => strip,
        Some(c) => {
            strip.bg(t.bg_panel).children(c.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, &n)| {
                grow(div().h_full().flex_basis(px(0.)).min_w(px(1.)).bg(t.class(i as u32)), n)
            }))
        }
    }
}

/// The strip's height: a hairline you can read at a glance, flush with the card's edge.
const STRIP_H: f32 = 4.;

/// A segment's share of the strip: flex-grow by its box count.
fn grow(d: folo_ui::Div, n: u64) -> folo_ui::Div {
    let mut d = d;
    d.style().flex_grow = Some(n as f32);
    d
}

/// The breakdown in words: classes largest first, "circle 5,523 · square 5,533", the first ten and
/// the rest summed.
pub(crate) fn balance_text(classes: &[String], counts: &[u64]) -> String {
    let mut ranked: Vec<(usize, u64)> = counts.iter().copied().enumerate().filter(|(_, n)| *n > 0).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if ranked.is_empty() {
        return "no boxes yet".to_string();
    }
    let name = |i: usize| classes.get(i).cloned().unwrap_or_else(|| format!("class {i}"));
    let mut parts: Vec<String> =
        ranked.iter().take(10).map(|&(i, n)| format!("{} {}", name(i), int(n))).collect();
    let rest = &ranked[ranked.len().min(10)..];
    if !rest.is_empty() {
        let sum: u64 = rest.iter().map(|r| r.1).sum();
        parts.push(format!("{} more classes {}", rest.len(), int(sum)));
    }
    parts.join(" · ")
}

#[cfg(test)]
mod tests {
    use super::{balance_text, dataset_problem};
    use crate::api::ApiError;

    /// x1-parity-7: a dataset whose images were deleted (the server answers 400 invalid_dataset)
    /// reads "Can't read" on its card, as in Studio; a lost connection is not the dataset's fault.
    #[test]
    fn an_unreadable_dataset_is_named_on_its_card() {
        let e = |code: &str| ApiError::new(code, "x");
        assert_eq!(dataset_problem(&e("invalid_dataset")), Some("Can't read"));
        assert_eq!(dataset_problem(&e("unknown_format")), Some("Can't read"));
        assert_eq!(dataset_problem(&e("path_not_found")), Some("Folder missing"));
        assert_eq!(dataset_problem(&e("server_unreachable")), None);
        assert_eq!(dataset_problem(&e("unauthorized")), None);
    }

    #[test]
    fn the_balance_reads_largest_first_and_sums_past_ten() {
        let names: Vec<String> = ["circle", "square", "star"].iter().map(|s| s.to_string()).collect();
        assert_eq!(balance_text(&names, &[10, 1200, 0]), "square 1,200 · circle 10");
        let many: Vec<String> = (0..12).map(|i| format!("c{i}")).collect();
        let counts: Vec<u64> = (0..12).map(|i| 100 - i).collect();
        assert!(balance_text(&many, &counts).ends_with("· 2 more classes 179"));
        assert_eq!(balance_text(&names, &[0, 0, 0]), "no boxes yet");
    }
}
