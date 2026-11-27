//! Inline error, remote-edit conflict, and unsaved-switch banners inside the Agent Labeling review
//! workspace.

use folo_ui::{AnyElement, Context, FontWeight, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::LabelImage;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn label_review_notices(
        &self,
        t: &Theme,
        jid: &str,
        active: &LabelImage,
        changed_elsewhere: bool,
        has_pending_switch: bool,
        locked: bool,
        busy: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let mut notices = Vec::new();
        if let Some(err) = &active.error {
            notices.push(
                div()
                    .mx(px(14.))
                    .mt(px(10.))
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.danger_muted)
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(div().flex_none().child(icon("icons/circle-alert.svg", 14., t.danger_solid)))
                    .child(
                        text(ty::SM, t.danger_text).flex_1().min_w_0().whitespace_normal().child(err.clone()),
                    )
                    .into_any_element(),
            );
        }
        if let Some(err) = &self.label.review_error {
            notices.push(
                div()
                    .mx(px(14.))
                    .mt(px(10.))
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.warning_muted)
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(div().flex_none().child(icon("icons/triangle-alert.svg", 14., t.warning_solid)))
                    .child(
                        text(ty::SM, t.warning_text)
                            .flex_1()
                            .min_w_0()
                            .whitespace_normal()
                            .child(err.message.clone()),
                    )
                    .into_any_element(),
            );
        }
        if changed_elsewhere {
            let (j_ov, i_ov) = (jid.to_string(), active.id.clone());
            let (j_ld, i_ld) = (jid.to_string(), active.id.clone());
            let overwrite_btn = if busy || locked {
                self.button_disabled(
                    "label-overwrite-edits",
                    "Overwrite server boxes",
                    Btn::Primary,
                    None,
                    if locked { "Agent is labeling this image" } else { "Saving…" },
                    window,
                    cx,
                )
            } else {
                self.button(
                    "label-overwrite-edits",
                    "Overwrite server boxes",
                    Btn::Primary,
                    Some("icons/check.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.save_label_edits_overwrite(&j_ov, &i_ov, cx)),
                )
            };
            let load_btn = if busy {
                self.button_disabled(
                    "label-load-server",
                    "Load server boxes",
                    Btn::Secondary,
                    None,
                    "Saving…",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "label-load-server",
                    "Load server boxes",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(move |this, _, cx| this.discard_label_edits(&j_ld, &i_ld, cx)),
                )
            };
            notices.push(
                div()
                    .mx(px(14.))
                    .mt(px(10.))
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.warning_muted)
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .flex_1()
                            .min_w(px(200.))
                            .max_w_full()
                            .child(div().flex_none().child(icon("icons/info.svg", 14., t.warning_solid)))
                            .child(
                                text(ty::SM, t.warning_text)
                                    .flex_1()
                                    .min_w_0()
                                    .whitespace_normal()
                                    .child("Boxes changed on the server while you edited. Choose whether to overwrite with your draft or load the latest server boxes."),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .flex_none()
                            .max_w_full()
                            .items_center()
                            .gap(px(6.))
                            .child(load_btn)
                            .child(overwrite_btn),
                    )
                    .into_any_element(),
            );
        }
        if has_pending_switch {
            let j_save = jid.to_string();
            let j_disc = jid.to_string();
            let save_cont = if busy || locked {
                self.button_disabled(
                    "label-sw-save",
                    "Save & continue",
                    Btn::Primary,
                    None,
                    if locked { "Agent is labeling this image" } else { "Saving…" },
                    window,
                    cx,
                )
            } else {
                self.button(
                    "label-sw-save",
                    "Save & continue",
                    Btn::Primary,
                    Some("icons/check.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.resolve_pending_label_switch(&j_save, true, cx)),
                )
            };
            let disc_cont = if busy {
                self.button_disabled("label-sw-discard", "Discard", Btn::Danger, None, "Saving…", window, cx)
            } else {
                self.button(
                    "label-sw-discard",
                    "Discard",
                    Btn::Danger,
                    None,
                    window,
                    cx,
                    act(move |this, _, cx| this.resolve_pending_label_switch(&j_disc, false, cx)),
                )
            };
            let stay_btn = self.button(
                "label-sw-stay",
                "Stay",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.label.editor.pending_switch = None;
                    cx.notify();
                }),
            );
            notices.push(
                div()
                    .mx(px(14.))
                    .mt(px(10.))
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.warning_muted)
                    .border_1()
                    .border_color(t.warning_solid)
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .flex_1()
                            .min_w(px(180.))
                            .max_w_full()
                            .child(div().flex_none().child(icon(
                                "icons/triangle-alert.svg",
                                14.,
                                t.warning_solid,
                            )))
                            .child(
                                text(ty::SM, t.warning_text)
                                    .font_weight(FontWeight::MEDIUM)
                                    .flex_1()
                                    .min_w_0()
                                    .whitespace_normal()
                                    .child(format!("Unsaved box edits on {}.", active.file)),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .flex_none()
                            .max_w_full()
                            .items_center()
                            .gap(px(6.))
                            .child(stay_btn)
                            .child(disc_cont)
                            .child(save_cont),
                    )
                    .into_any_element(),
            );
        }
        notices
    }
}
