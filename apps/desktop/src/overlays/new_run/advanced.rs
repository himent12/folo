//! The collapsed Advanced group of the New run dialog: end-to-end head, optimizer and learning
//! rate, mixed precision, and augmentation sliders.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{FoloApp, act};
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

use super::{nr_field, nr_label, nr_typed};

pub(super) struct AdvancedInputs {
    pub(super) short: bool,
    pub(super) opt: usize,
    pub(super) opt_unknown: bool,
    pub(super) auto_pick: Option<(&'static str, String)>,
    pub(super) opt_seg: AnyElement,
    pub(super) lr_box: AnyElement,
    pub(super) lr_slider: AnyElement,
    pub(super) lr_bad: bool,
    pub(super) amp: bool,
    pub(super) amp_why: Option<String>,
    pub(super) is_cloud: bool,
    pub(super) mosaic: f32,
    pub(super) mixup: f32,
    pub(super) fliplr: f32,
    pub(super) scale: f32,
    pub(super) mosaic_slider: AnyElement,
    pub(super) mixup_slider: AnyElement,
    pub(super) flip_slider: AnyElement,
    pub(super) scale_slider: AnyElement,
}

impl FoloApp {
    pub(super) fn nr_advanced_section(
        &mut self,
        t: Theme,
        inputs: AdvancedInputs,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Div {
        let AdvancedInputs {
            short,
            opt,
            opt_unknown,
            auto_pick,
            opt_seg,
            lr_box,
            lr_slider,
            lr_bad,
            amp,
            amp_why,
            is_cloud,
            mosaic,
            mixup,
            fliplr,
            scale,
            mosaic_slider,
            mixup_slider,
            flip_slider,
            scale_slider,
        } = inputs;
        // A slider row as the Playground rail writes one: name · slider · value, on one line.
        let valued = |name: &str, value: String, control: AnyElement| {
            div()
                .flex()
                .items_center()
                .gap(px(10.))
                .child(div().w(px(100.)).flex_none().child(nr_label(&t, name)))
                .child(div().flex_1().min_w_0().child(control))
                .child(
                    div()
                        .w(px(44.))
                        .flex_none()
                        .flex()
                        .justify_end()
                        .child(text(ty::SM, t.text_primary).tnum().child(value)),
                )
        };
        // Advanced's body: indented under the header's chevron, a rule down its left, so
        // what the disclosure encloses is visible; its parts are sub-heads, not sections.
        div()
            .ml(px(6.))
            .pl(px(18.))
            .border_l_1()
            .border_color(t.line_subtle)
            .flex()
            .flex_col()
            .gap(px(if short { 8. } else { 14. }))
            // The head: one-to-many with NMS (the default), or the end-to-end head that predicts
            // without it. It leads Advanced, as the model property among them.
            .child({
                let e2e = self.new_run.as_ref().is_some_and(|f| f.e2e);
                let switch = self
                    .switch(
                        "nr-e2e",
                        e2e,
                        window,
                        cx,
                        act(|this, _, cx| {
                            if let Some(f) = this.new_run.as_mut() {
                                f.e2e = !f.e2e;
                                cx.notify();
                            }
                        }),
                    )
                    .into_any_element();
                div()
                    // The left column's width (the 16 px gap halved off), so the switch sits by its
                    // label, not across a two-column dialog.
                    .w(folo_ui::relative(0.5))
                    .pr(px(8.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(16.))
                    .child(
                        // The label toggles too (the switch keeps the one Tab stop).
                        div()
                            .id("nr-e2e-label")
                            .cursor_pointer()
                            .on_click(cx.listener(|this, _, _, cx| {
                                if let Some(f) = this.new_run.as_mut() {
                                    f.e2e = !f.e2e;
                                    cx.notify();
                                }
                            }))
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(nr_label(&t, "End-to-end (no NMS)"))
                            .child(
                                text(ty::SM, t.text_tertiary)
                                    .whitespace_normal()
                                    .child("Predicts boxes directly, with no NMS step"),
                            ),
                    )
                    .child(switch)
            })
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(nr_field(
                        &t,
                        "Optimizer",
                        opt_seg,
                        Some(match (opt, &auto_pick) {
                            _ if opt_unknown => "Unknown optimizer (from a newer FOLO): pick one".to_string(),
                            (0, Some((kind, _))) => {
                                format!("Auto: {kind} for this run, at its own learning rate")
                            }
                            (0, None) => "Auto: MuSGD for every run, at its own learning rate".to_string(),
                            (1, _) => "SGD with Nesterov momentum 0.937".to_string(),
                            (2, _) => "AdamW with weight decay".to_string(),
                            _ => "MuSGD: SGD plus a Muon update on conv weights".to_string(),
                        }),
                    )))
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap(px(8.))
                            // Under Auto the engine sets the rate itself: the field shows that rate,
                            // read-only, instead of an editable value it would ignore.
                            .child(if opt == 0 {
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap(px(8.))
                                    .child(nr_label(&t, "Initial learning rate"))
                                    .child(
                                        div()
                                            .h(px(crate::theme::size::CONTROL_MD))
                                            .px(px(8.))
                                            .flex()
                                            .items_center()
                                            .rounded(px(radius::MD))
                                            // A value, not an input: an outline, no well.
                                            .border_1()
                                            .border_color(t.line_subtle)
                                            .child(crate::ui::sci(
                                                ty::SM,
                                                t.text_tertiary,
                                                match &auto_pick {
                                                    Some((_, lr)) => format!("Auto · {lr}"),
                                                    None => "Auto".to_string(),
                                                },
                                            )),
                                    )
                                    .into_any_element()
                            } else {
                                nr_typed(
                                    &t,
                                    "Initial learning rate",
                                    lr_box,
                                    lr_slider,
                                    lr_bad.then_some("Enter a number above 0 and below 10, e.g. 0.01 or 1e-3"),
                                )
                                .into_any_element()
                            })
                            // The schedule (short windows keep the Schedule above the footer).
                            .when(!short && !lr_bad, |d| {
                                d.child(text(ty::SM, t.text_tertiary).child(if opt == 0 {
                                    "Set by Auto; pick SGD or AdamW to choose your own"
                                } else {
                                    "3-epoch warmup, then linear decay to 1%"
                                }))
                            }),
                    ),
            )
            // Mixed precision: f16 on tensor cores. The server says whether its GPU can
            // (`capabilities.amp`); when it can't, the switch is off and inert with its reason. An older
            // server can't say, so the requirement is stated and a run's failure tells.
            .child({
                let blocked = amp_why;
                let off = blocked.is_some();
                let amp_on = amp && !off;
                // Blocked: a disabled switch (no hover, press or focus stop); the reason is beside it.
                let switch = if off {
                    self.switch_disabled("nr-amp", cx).into_any_element()
                } else {
                    self.switch(
                        "nr-amp",
                        amp_on,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            if let Some(f) = this.new_run.as_mut() {
                                f.amp = !f.amp;
                                cx.notify();
                            }
                        }),
                    )
                    .into_any_element()
                };
                div()
                    // The left column's width (the 16 px gap halved off), so the switch sits by its
                    // label, not across a two-column dialog.
                    .w(folo_ui::relative(0.5))
                    .pr(px(8.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(16.))
                    .child(
                        // The label toggles too when the switch can (it keeps the one Tab stop).
                        div()
                            .id("nr-amp-label")
                            .when(!off, |d| {
                                d.cursor_pointer().on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(f) = this.new_run.as_mut() {
                                        f.amp = !f.amp;
                                        cx.notify();
                                    }
                                }))
                            })
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(nr_label(&t, "Mixed precision"))
                            .child(
                                // The server's reason is plain text (a backtick in it is literal); only
                                // the stated need carries a code chip.
                                match blocked {
                                    Some(why) => text(ty::SM, t.text_tertiary).whitespace_normal().child(why),
                                    None if is_cloud => text(ty::SM, t.text_tertiary)
                                        .whitespace_normal()
                                        .child("Checked on the instance when it boots."),
                                    // The server confirmed it can; an older one can't say, so the need is stated.
                                    None if self.store.capabilities.is_some() => {
                                        text(ty::SM, t.text_tertiary).whitespace_normal().child(
                                            "f16 on this server's tensor cores: about 40% less GPU memory at 640 px. Not faster.",
                                        )
                                    }
                                    None => rich(
                                        &t,
                                        ty::SM,
                                        t.text_tertiary,
                                        "Trains in f16 on tensor cores; needs `folo serve --tensor-cores`",
                                    )
                                    .text_left(),
                                },
                            ),
                    )
                    .child(switch)
            })
            .child(text(ty::BASE, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Augmentation"))
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(valued("Mosaic", crate::ui::fmt_aug(mosaic), mosaic_slider)))
                    .child(div().flex_1().child(valued("Mixup", crate::ui::fmt_aug(mixup), mixup_slider))),
            )
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(valued(
                        "Horizontal flip",
                        crate::ui::fmt_aug(fliplr),
                        flip_slider,
                    )))
                    .child(div().flex_1().child(valued(
                        "Scale jitter",
                        crate::ui::fmt_scale(scale),
                        scale_slider,
                    ))),
            )
    }
}
