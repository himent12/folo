//! The datasets strip and its cards, and the "View all" links.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Label, SplitsExt};
use crate::app::{FoloApp, Route, act};
use crate::screens::cell2;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn dataset_strip(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-ds-all", Route::Datasets, window, cx);
        let datasets = self.store.datasets.ready().cloned().unwrap_or_default();
        let cards: Vec<AnyElement> =
            datasets.iter().take(3).map(|d| self.dataset_card(d, window, cx)).collect();
        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Datasets").child(all))
            .map(|d| {
                if cards.is_empty() && self.store.datasets.ready().is_some() {
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

    /// Dataset card: a 2×2 mosaic of its first images, name and counts.
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
        // The card in full: its name, size and every class, on hover and on keyboard focus.
        let full = format!("{} · {} images · {}", d.name, int(d.splits.total()), d.classes.join(", "));
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
                            "{} images · {}",
                            int(d.splits.total()),
                            if d.classes.len() == 1 {
                                "1 class".to_string()
                            } else {
                                format!("{} classes", d.classes.len())
                            }
                        ),
                    ))
                    .child(badge(&t, d.format.label(), Tone::Neutral)),
            );
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

    pub fn view_all(&self, key: &str, route: Route, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let el = div()
            .id(SharedString::from(key.to_string()))
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
            .child(text(ty::SM, t.text_secondary).child("View all"))
            .child(icon("icons/chevron-right.svg", 12., t.text_tertiary));
        self.bind(
            el,
            key,
            radius::SM,
            window,
            cx,
            act(move |this, w, cx| this.navigate(route.clone(), w, cx)),
        )
        .into_any_element()
    }
}
