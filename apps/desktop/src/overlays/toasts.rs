//! The toast stack in the window's corner.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{Action, FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn toasts_view(&mut self, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        type ToastRow = (u64, Tone, SharedString, Option<SharedString>, Option<(SharedString, Action)>);
        let toasts: Vec<ToastRow> = self
            .toasts
            .iter()
            .map(|x| (x.id, x.tone, x.title.clone(), x.body.clone(), x.action.clone()))
            .collect();
        // One card at a time: older notifications fold into a single row above it.
        self.toast_focus = toasts.iter().map(|x| x.0).find(|id| {
            [format!("toast-x-{id}"), format!("toast-act-{id}")]
                .iter()
                .any(|k| self.focus.borrow().get(k.as_str()).is_some_and(|f| f.is_focused(window)))
        });
        // Focus on the stack's own controls holds every toast (none expires under the keyboard).
        self.toast_stack_focus = ["toast-expand", "toast-dismiss-all"]
            .iter()
            .any(|k| self.focus.borrow().get(*k).is_some_and(|f| f.is_focused(window)));
        let older = toasts.len().saturating_sub(1);
        let expanded = self.toasts_expanded;
        let toasts: Vec<ToastRow> = toasts.into_iter().skip(if expanded { 0 } else { older }).collect();
        let text_btn =
            |this: &Self, key: &str, label: String, window: &Window, cx: &mut Context<Self>, a: Action| {
                let el = div()
                    .id(SharedString::from(key.to_string()))
                    .px(px(6.))
                    .h(px(22.))
                    .flex()
                    .items_center()
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .child(text(ty::SM, t.text_secondary).child(label));
                this.bind(el, key, radius::SM, window, cx, a)
            };
        let folded = (older > 0).then(|| {
            let show = text_btn(
                self,
                "toast-expand",
                if expanded {
                    "Show latest only".to_string()
                } else if older == 1 {
                    "Show 1 earlier".to_string()
                } else {
                    format!("Show {older} earlier")
                },
                window,
                cx,
                act(|this, _, cx| {
                    this.toasts_expanded = !this.toasts_expanded;
                    cx.notify();
                }),
            );
            let dismiss = text_btn(
                self,
                "toast-dismiss-all",
                "Dismiss all".to_string(),
                window,
                cx,
                act(|this, _, cx| {
                    this.toasts.clear();
                    this.toasts_expanded = false;
                    cx.notify();
                }),
            );
            (show.into_any_element(), dismiss.into_any_element())
        });
        // While folded, the earlier toasts' edges peek out behind the latest one: opaque cards (not
        // tints the page shows through), line.default so the toast's own top edge isn't doubled.
        let peeks = (older > 0 && !expanded).then(|| {
            let edge = |inset: f32, a: f32| {
                div()
                    .occlude()
                    .mx(px(inset))
                    .h(px(6.))
                    .rounded_t(px(radius::LG))
                    .bg(if a < 1. {
                        crate::theme::over(t.bg_raised, crate::theme::fade(t.bg_app, 1. - a))
                    } else {
                        t.bg_raised
                    })
                    .border_1()
                    .border_b_0()
                    .border_color(t.line_default)
            };
            div().flex().flex_col().when(older > 1, |d| d.child(edge(16., 0.6))).child(edge(8., 1.))
        });
        // The stack's controls live in the newest card's action row (no floating pill).
        let mut stack_ctrls = folded;
        let newest = toasts.len().saturating_sub(1);
        div()
            .absolute()
            .bottom(px(24.))
            .right(px(24.))
            .w(px(360.))
            .flex()
            .flex_col()
            .gap(px(if peeks.is_some() { 0. } else { 8. }))
            .children(peeks)
            .children(toasts.into_iter().enumerate().map(|(i, (id, tone, title, body, action))| {
                let ctrls = if i == newest { stack_ctrls.take() } else { None };
                let action_btn = action.map(|(label, a)| {
                    self.button(
                        &format!("toast-act-{id}"),
                        label,
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(move |this, w, cx| {
                            a(this, w, cx);
                            this.dismiss_toast(id, cx);
                        }),
                    )
                });
                let (_, _, dotc) = tone.colors(&t);
                let icon_path = match tone {
                    Tone::Success => "icons/circle-check.svg",
                    Tone::Danger => "icons/circle-x.svg",
                    Tone::Warning => "icons/triangle-alert.svg",
                    Tone::Accent => "icons/activity.svg",
                    _ => "icons/info.svg",
                };
                // Each toast fades in as it arrives (160 ms; none under Reduce motion).
                crate::ui::appear(
                    div()
                        .id(("toast", id as usize))
                        .occlude()
                        .on_hover(cx.listener(move |this, h: &bool, _, _| {
                            if *h {
                                this.toast_hover = Some(id);
                            } else if this.toast_hover == Some(id) {
                                this.toast_hover = None;
                            }
                        }))
                        .flex()
                        .flex_col()
                        .p(px(12.))
                        .rounded(px(radius::LG))
                        .bg(t.bg_overlay)
                        .border_1()
                        .border_color(t.line_strong)
                        .shadow(t.shadow_lg())
                        .child(
                            div()
                                .flex()
                                .items_start()
                                .gap(px(10.))
                                .child(div().pt(px(2.)).child(icon(
                                    icon_path,
                                    16.,
                                    if tone == Tone::Neutral { t.text_secondary } else { dotc },
                                )))
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .flex()
                                        .flex_col()
                                        .gap(px(2.))
                                        .child(
                                            text(ty::BASE, t.text_primary)
                                                .font_weight(FontWeight::MEDIUM)
                                                .child(title),
                                        )
                                        .when_some(body, |d, b| {
                                            d.child(text(ty::SM, t.text_secondary).child(b))
                                        }),
                                )
                                .child(
                                    self.icon_button(
                                        &format!("toast-x-{id}"),
                                        "icons/x.svg",
                                        "Dismiss",
                                        window,
                                        cx,
                                        act(move |this, _, cx| this.dismiss_toast(id, cx)),
                                    )
                                    .size(px(24.)),
                                ),
                        )
                        // One footer: the card's action under the text, the stack's controls ending at
                        // the card's padding, under the close button.
                        .when(action_btn.is_some() || ctrls.is_some(), |d| {
                            d.child(
                                div()
                                    .pt(px(8.))
                                    .pl(px(26.))
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .gap(px(8.))
                                    .child(div().flex().children(action_btn))
                                    .children(ctrls.map(|(show, dismiss)| {
                                        div().flex().items_center().gap(px(4.)).child(show).child(dismiss)
                                    })),
                            )
                        }),
                    &format!("toast-in-{id}"),
                )
            }))
            .into_any_element()
    }
}
