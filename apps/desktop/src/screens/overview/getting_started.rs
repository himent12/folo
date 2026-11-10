//! First run: what FOLO needs, in order, with the next step as the one primary action.

use gpui::prelude::FluentBuilder;
use gpui::{AnyElement, Context, FontWeight, IntoElement, ParentElement, Styled, Window, div, px};

use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// First run: what FOLO needs, in order, with the next step as the one primary action.
    pub(super) fn getting_started(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let add = self
            .button(
                "gs-add",
                "Add dataset",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    // The label promises the picker: land on Datasets with it open (the built-in demo
                    // can't register folders, and says so there instead).
                    this.navigate(Route::Datasets, w, cx);
                    this.add_dataset_folder(cx);
                }),
            )
            .into_any_element();
        // The app makes a sample itself (Datasets' generator), so a first run needn't leave for a
        // terminal to have something to train on.
        let synth = self
            .button(
                "gs-synth",
                "Generate synthetic",
                Btn::Secondary,
                Some("icons/zap.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    this.navigate(Route::Datasets, w, cx);
                    this.open_synth(w, cx);
                }),
            )
            .into_any_element();
        let step = |n: &str, title: &str, body: &str, active: bool| {
            div()
                // Fills its wrapper, so the three cards end at one height whichever wraps most; may
                // shrink below its text (which then wraps).
                .flex_1()
                .min_w_0()
                .flex()
                .gap(px(14.))
                .p(px(16.))
                .rounded(px(radius::LG))
                .border_1()
                .border_color(if active { t.line_strong } else { t.line_subtle })
                .when(active, |d| d.bg(t.bg_raised))
                .child(
                    div()
                        .size(px(24.))
                        .flex_none()
                        .rounded_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(if active { t.accent_fill() } else { t.bg_active })
                        .child(
                            text(ty::SM, if active { t.accent_on } else { t.text_tertiary })
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(n.to_string()),
                        ),
                )
                // Shrinks to its third of the row, so the body (one text run) wraps inside it.
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(
                            text(ty::BASE, if active { t.text_primary } else { t.text_secondary })
                                .font_weight(FontWeight::MEDIUM)
                                .child(title.to_string()),
                        )
                        .child(rich(&t, ty::SM, t.text_tertiary, body).text_left()),
                )
        };
        let steps = panel(&t)
            .p(px(16.))
            .flex()
            .flex_col()
            .gap(px(20.))
            .child(
                div().flex().items_center().gap(px(14.)).child(crate::app::logo(&t, 32.)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(text(ty::LG, t.text_primary).child("Get started with FOLO"))
                        .child(
                            text(ty::SM, t.text_secondary)
                                .child("Three steps from images to a model you can run."),
                        ),
                ),
            )
            .child(
                div()
                    // Children stretch to the row's height (the flex default).
                    .flex()
                    .gap(px(12.))
                    .child(div().flex().flex_1().flex_basis(px(0.)).min_w_0().child(step(
                        "1",
                        "Add a dataset",
                        "A YOLO or COCO folder.",
                        true,
                    )))
                    .child(div().flex().flex_1().flex_basis(px(0.)).min_w_0().child(step(
                        "2",
                        "Train a model",
                        "Watch loss and mAP live.",
                        false,
                    )))
                    .child(div().flex().flex_1().flex_basis(px(0.)).min_w_0().child(step(
                        "3",
                        "Try it",
                        "Open any image in the Playground.",
                        false,
                    ))),
            )
            .child(div().flex().gap(px(8.)).child(add).child(synth));
        // The same path from a terminal, for people who live there (docs/CLI.md).
        const CMDS: [&str; 3] =
            ["folo serve", "folo dataset synth shapes --images 1000", "folo train --data shapes --epochs 50"];
        let mut copy: Vec<Option<AnyElement>> = CMDS
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                Some(
                    self.icon_button(
                        &format!("gs-copy-{i}"),
                        "icons/copy.svg",
                        "Copy command",
                        window,
                        cx,
                        act(move |this, _, cx| {
                            cx.write_to_clipboard(gpui::ClipboardItem::new_string(c.to_string()));
                            this.toast(Tone::Neutral, "Copied", Some(format!("`{c}`").into()), cx);
                        }),
                    )
                    .into_any_element(),
                )
            })
            .collect();
        let mut cmd = |i: usize, what: &str| {
            div()
                .flex()
                .items_center()
                .gap(px(16.))
                .child(
                    div()
                        .w(px(420.))
                        .flex_none()
                        .flex()
                        .items_center()
                        .gap(px(4.))
                        .pl(px(10.))
                        .pr(px(2.))
                        .py(px(2.))
                        .rounded(px(radius::SM))
                        .bg(t.bg_inset)
                        .child(
                            text(ty::SM, t.text_primary)
                                .font_family(crate::theme::MONO)
                                .flex_1()
                                .min_w_0()
                                .ellipsis()
                                .child(CMDS[i]),
                        )
                        .children(copy[i].take()),
                )
                .child(text(ty::SM, t.text_tertiary).child(what.to_string()))
        };
        let terminal = panel(&t)
            .p(px(16.))
            .flex()
            .flex_col()
            .gap(px(10.))
            .child(text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).child("Or from a terminal"))
            // Honest about what reaches this app: the server and the datasets registered with it;
            // `folo train` runs on its own (runs/ in the working folder).
            .child(cmd(0, "start it, then Connect in Settings"))
            .child(cmd(1, "then Add dataset → choose ./shapes"))
            // Training from the terminal stands apart from the sequence above.
            .child(div().pt(px(6.)).child(caption(&t, "Independent CLI training")))
            .child(cmd(2, "trains here in the terminal; this run won't appear in Desktop"));
        // Capped like Settings, so a wide window doesn't stretch three short steps and the
        // commands' captions across 1,600 px of panel.
        div().flex().flex_col().gap(px(16.)).max_w(px(1040.)).child(steps).child(terminal).into_any_element()
    }
}
