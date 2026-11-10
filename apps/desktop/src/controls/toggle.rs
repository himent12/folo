//! Toggles: filter chips and switches.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnimationExt, Context, Div, InteractiveElement, IntoElement, ParentElement, SharedString, Stateful,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{Action, FoloApp};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// Filter chip (toggle).
    pub fn chip(
        &self,
        key: &str,
        label: impl Into<SharedString>,
        on: bool,
        swatch_class: Option<u32>,
        count: Option<u64>,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let el = div()
            .id(SharedString::from(key.to_string()))
            .h(px(tsize::CONTROL_SM))
            .px(px(8.))
            .flex()
            .flex_none()
            .items_center()
            .gap(px(6.))
            .rounded(px(radius::SM))
            .border_1()
            .cursor_pointer()
            // Selected chips answer the pointer too: the hover tint over the selected fill.
            // Pressed is a fill a step past hover, as on buttons (not a fade).
            .when(on, |d| {
                let hot = crate::theme::over(t.bg_selected, t.bg_hover);
                let down = crate::theme::over(t.bg_selected, t.bg_active);
                d.bg(t.bg_selected)
                    .border_color(t.accent_line)
                    .hover(move |s| s.bg(hot))
                    .active(move |s| s.bg(down))
            })
            .when(!on, |d| {
                d.border_color(t.line_control)
                    .hover(|s| s.bg(t.bg_hover).border_color(t.text_tertiary))
                    .active(|s| s.bg(t.bg_active))
            })
            // Selected: a check in the class colour takes the swatch's slot, so selection isn't
            // colour alone and nothing shifts.
            .when_some(swatch_class, |d, c| {
                d.child(if on {
                    div().size(px(12.)).flex().items_center().justify_center().child(icon(
                        "icons/check.svg",
                        12.,
                        t.class_on(c, crate::theme::Surface::Panel),
                    ))
                } else {
                    div()
                        .size(px(12.))
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(crate::ui::class_swatch(&t, c, 8.))
                })
            })
            .child(
                text(ty::SM, if on { t.text_primary } else { t.text_secondary })
                    .whitespace_nowrap()
                    .child(label.into()),
            )
            .when_some(count, |d, c| {
                d.child(
                    text(ty::SM, if on { t.text_secondary } else { t.text_tertiary }).tnum().child(int(c)),
                )
            });
        self.bind(el, key, radius::SM, window, cx, action)
    }

    pub fn switch(
        &self,
        key: &str,
        on: bool,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let t = theme(cx);
        // A 36 x 28 target around the 30 x 18 track: the track answers its wrapper's hover and press.
        let group = SharedString::from(format!("{key}-sw"));
        let track = div()
            .id(SharedString::from(format!("{key}-track")))
            .relative()
            .w(px(30.))
            .h(px(18.))
            .flex_none()
            .rounded_full()
            .bg(if on { t.accent_fill() } else { t.line_control })
            // Hover and pressed are fills a step apart, like every other control.
            .when(!on, |d| {
                d.group_hover(group.clone(), |s| s.bg(t.text_tertiary))
                    .group_active(group.clone(), |s| s.bg(t.text_secondary))
            })
            .when(on, |d| {
                let hot = crate::theme::over(t.accent_fill(), t.bg_hover);
                let down = crate::theme::over(t.accent_fill(), t.bg_active);
                d.group_hover(group.clone(), move |s| s.bg(hot))
                    .group_active(group.clone(), move |s| s.bg(down))
            })
            .child({
                // Dark: a dark knob on lime; light: white on olive. Off: a bright knob on grey.
                let knob = if on && t.is_dark() {
                    t.accent_on
                } else if t.is_dark() {
                    t.text_primary
                } else {
                    gpui::white()
                };
                let knob =
                    div().absolute().top(px(2.)).size(px(14.)).rounded_full().bg(knob).shadow(t.shadow_sm());
                let (from, to) = if on { (2., 14.) } else { (14., 2.) };
                // Animate only within the motion's time after a real change of value.
                let changed = {
                    let mut seen = self.switch_seen.borrow_mut();
                    let e = seen.entry(key.to_string()).or_insert((on, None));
                    if e.0 != on {
                        *e = (on, Some(std::time::Instant::now()));
                    }
                    e.1.is_some_and(|at| at.elapsed() < std::time::Duration::from_millis(150))
                };
                if reduce_motion() || !changed {
                    knob.left(px(to)).into_any_element()
                } else {
                    knob.with_animation(
                        SharedString::from(format!("{key}-knob-{on}")),
                        gpui::Animation::new(std::time::Duration::from_millis(120))
                            .with_easing(gpui::ease_out_quint()),
                        move |d, p| d.left(px(from + (to - from) * p)),
                    )
                    .into_any_element()
                }
            });
        let el = div()
            .id(SharedString::from(key.to_string()))
            .group(group)
            .w(px(36.))
            .h(px(28.))
            .flex_none()
            .flex()
            .items_center()
            .justify_center()
            .rounded(px(radius::SM))
            .cursor_pointer()
            .child(track);
        self.bind(el, key, radius::SM, window, cx, action)
    }

    /// A switch that can't be used now: off, an edged inset track and a tertiary knob, with no
    /// pointer, hover or focus stop. The caller says why in the text beside it.
    pub fn switch_disabled(&self, key: &str, cx: &mut Context<Self>) -> Stateful<Div> {
        let t = theme(cx);
        // The enabled switch's 36 x 28 footprint, so the row doesn't shift when it's blocked.
        div()
            .id(SharedString::from(format!("{key}-disabled")))
            .w(px(36.))
            .h(px(28.))
            .flex_none()
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .relative()
                    .w(px(30.))
                    .h(px(18.))
                    .rounded_full()
                    .bg(t.bg_inset)
                    .border_1()
                    .border_color(t.line_default)
                    .child(
                        // Inside the 1 px edge: 1 px in, so the knob sits where an enabled one does.
                        div()
                            .absolute()
                            .top(px(1.))
                            .left(px(1.))
                            .size(px(14.))
                            .rounded_full()
                            .bg(t.text_tertiary),
                    ),
            )
    }
}
