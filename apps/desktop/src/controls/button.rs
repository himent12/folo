//! Buttons: their kinds and colours, icon buttons and the zoom readout.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    Context, Div, FontWeight, Hsla, InteractiveElement, ParentElement, SharedString, Stateful,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::tooltip::{focus_tip, tip};
use crate::app::{Action, FoloApp, focus_ring};
use crate::theme::{Theme, radius, size as tsize, theme, ty};
use crate::ui::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Btn {
    Primary,
    Secondary,
    Danger,
    /// A dialog's destructive commit: the Danger tint with a danger edge, so it outweighs the safe
    /// choice beside it. (A solid red fill fails AA with white text in both themes, and the
    /// tokens have no on-danger ink.)
    DangerCommit,
}

impl FoloApp {
    pub fn button(
        &self,
        key: &str,
        label: impl Into<SharedString>,
        kind: Btn,
        icon_path: Option<&'static str>,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let (bg, fg, border, hover) = btn_colors(&t, kind);
        // Pressed is a real step past hover, not a fade.
        let pressed = match kind {
            Btn::Primary => crate::theme::over(t.accent_solid, folo_ui::hsla(0., 0., 0., 0.14)),
            Btn::Secondary => crate::theme::over(t.bg_raised, t.bg_active),
            Btn::Danger | Btn::DangerCommit => crate::theme::alpha(t.danger_solid, 0.28),
        };
        let label: SharedString = label.into();
        let el = div()
            .id(SharedString::from(key.to_string()))
            .h(px(tsize::CONTROL_MD))
            .px(px(if icon_path.is_some() { 10. } else { 12. }))
            .flex()
            .flex_none()
            .items_center()
            .justify_center()
            .gap(px(6.))
            .rounded(px(radius::MD))
            .bg(bg)
            .when_some(border, |d, b| d.border_1().border_color(b))
            .cursor_pointer()
            .hover(move |s| s.bg(hover))
            .active(move |s| s.bg(pressed))
            .when_some(icon_path, |d, p| d.child(icon(p, 14., fg)))
            .child(text(ty::BASE, fg).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label));
        let ring_color = if kind == Btn::Primary { t.text_primary } else { t.focus };
        self.bind_ring(el, key, radius::MD, ring_color, window, cx, action)
    }

    /// A button that can't be used right now: dimmed and inert, but a Tab stop, so keyboard users
    /// reach it and its reason shows on focus (and as its tooltip).
    pub fn button_disabled(
        &self,
        key: &str,
        label: impl Into<SharedString>,
        _kind: Btn,
        icon_path: Option<&'static str>,
        reason: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let t = theme(cx);
        // Disabled drops the kind's colour entirely (`_kind`), so it never reads as a dimmed hover.
        let (bg, fg, border) = (t.bg_inset, t.text_disabled, Some(t.line_default));
        let label: SharedString = label.into();
        let fh = self.fh(format!("{key}-disabled"), cx);
        let ring = self.ring(&fh, window);
        div()
            .id(SharedString::from(format!("{key}-disabled")))
            .h(px(tsize::CONTROL_MD))
            .px(px(if icon_path.is_some() { 10. } else { 12. }))
            .flex()
            .flex_none()
            .items_center()
            .justify_center()
            .gap(px(6.))
            .rounded(px(radius::MD))
            .bg(bg)
            .when_some(border, |d, b| d.border_1().border_color(b))
            .cursor_not_allowed()
            .tooltip(tip(reason, None))
            .when_some(icon_path, |d, p| d.child(icon(p, 14., fg)))
            .child(text(ty::BASE, fg).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label))
            // Focusable, so keyboard users reach it; focus shows the reason the tooltip gives on hover.
            .relative()
            .track_focus(&fh)
            .when(ring, |d| {
                d.child(focus_ring(&t, radius::MD)).child(
                    div()
                        .absolute()
                        .top(px(tsize::CONTROL_MD + 6.))
                        .right_0()
                        .px(px(8.))
                        .py(px(4.))
                        .rounded(px(radius::SM))
                        .bg(t.bg_overlay)
                        .border_1()
                        .border_color(t.line_default)
                        .shadow(t.shadow_lg())
                        .child(text(ty::SM, t.text_secondary).whitespace_nowrap().child(reason)),
                )
            })
    }

    pub fn icon_button(
        &self,
        key: &str,
        icon_path: &'static str,
        tooltip: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let el = div()
            .id(SharedString::from(key.to_string()))
            .size(px(tsize::CONTROL_MD))
            .flex()
            .flex_none()
            .items_center()
            .justify_center()
            .rounded(px(radius::MD))
            .cursor_pointer()
            .hover(move |s| s.bg(t.bg_hover))
            .active(move |s| s.bg(t.bg_active))
            .tooltip(tip(tooltip, None))
            .child(icon(icon_path, 16., t.text_secondary));
        // Keyboard focus shows the same text the pointer gets from the tooltip.
        let ring = self.ring(&self.fh(key.to_string(), cx), window);
        let el = el.when(ring, |d| d.child(focus_tip(&t, tooltip, tsize::CONTROL_MD)));
        self.bind(el, key, radius::MD, window, cx, action)
    }

    /// An image's zoom level: "2.4×" when zoomed, a quiet button that fits again; "Fit" when fitted,
    /// plain text (no hover, no Tab stop: there's nothing to do). One control instead of a readout
    /// beside a fit icon that read as full screen.
    pub fn zoom_readout(
        &self,
        key: &str,
        zoom: f32,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let fitted = zoom <= 1.0;
        if fitted {
            return div()
                .id(SharedString::from(key.to_string()))
                .h(px(tsize::CONTROL_MD))
                .w(px(48.))
                .flex()
                .flex_none()
                .items_center()
                .justify_center()
                .child(text(ty::SM, t.text_tertiary).child("Fit"));
        }
        let tooltip = "Fit to the view (0)";
        let el = div()
            .id(SharedString::from(key.to_string()))
            .h(px(tsize::CONTROL_MD))
            .w(px(48.))
            .flex()
            .flex_none()
            .items_center()
            .justify_center()
            .rounded(px(radius::MD))
            .cursor_pointer()
            .hover(move |s| s.bg(t.bg_hover))
            .active(move |s| s.bg(t.bg_active))
            .tooltip(tip(tooltip, None))
            .child(text(ty::SM, t.text_secondary).tnum().child(format!("{zoom:.1}×")));
        let ring = self.ring(&self.fh(key.to_string(), cx), window);
        let el = el.when(ring, |d| d.child(focus_tip(&t, tooltip, tsize::CONTROL_MD)));
        self.bind(el, key, radius::MD, window, cx, action)
    }
}

pub fn btn_colors(t: &Theme, kind: Btn) -> (Hsla, Hsla, Option<Hsla>, Hsla) {
    match kind {
        // Light surfaces: a neutral 1 px edge, not a lime one (design/README). Dark keeps the same
        // 1 px as a transparent edge, so a theme switch never changes a button's width.
        Btn::Primary => (
            t.accent_solid,
            t.accent_on,
            Some(if t.is_dark() { folo_ui::transparent_black() } else { t.line_default }),
            t.accent_solid_hover,
        ),
        Btn::Secondary => {
            (t.bg_raised, t.text_primary, Some(t.line_default), crate::theme::over(t.bg_raised, t.bg_hover))
        }
        Btn::Danger => (t.danger_muted, t.danger_text, None, crate::theme::alpha(t.danger_solid, 0.2)),
        Btn::DangerCommit => {
            (t.danger_muted, t.danger_text, Some(t.danger_solid), crate::theme::alpha(t.danger_solid, 0.2))
        }
    }
}
