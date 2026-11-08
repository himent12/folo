//! Interactive controls. Each takes a stable key: it names the element, owns its focus handle (so
//! Tab reaches it and Enter/Space activate it) and shows the focus ring under keyboard use.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, AnyView, AppContext, Context, Div, FontWeight, Hsla, InteractiveElement, IntoElement,
    KeyDownEvent, MouseButton, MouseDownEvent, ParentElement, Render, SharedString, Stateful,
    StatefulInteractiveElement, Styled, Window, canvas, div, px, relative,
};

use crate::app::{Action, Drag, FoloApp, focus_ring};
use crate::theme::{Theme, radius, size as tsize, theme, ty};
use crate::ui::*;
use gpui::AnimationExt;

/// Called with the chosen option index.
pub type OnSelect = std::rc::Rc<dyn Fn(&mut FoloApp, usize, &mut Window, &mut Context<FoloApp>)>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Btn {
    Primary,
    Secondary,
    Ghost,
    Danger,
}

/// Hover tooltip: one short line on the overlay surface.
pub struct Tip(pub SharedString, pub Option<SharedString>);

impl Render for Tip {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        div().p(px(4.)).child(
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(8.))
                .py(px(5.))
                .rounded(px(radius::MD))
                .bg(t.bg_overlay)
                .border_1()
                .border_color(t.line_default)
                .shadow(t.shadow_md())
                .child(text(ty::SM, t.text_primary).child(self.0.clone()))
                .when_some(self.1.clone(), |d, k| d.child(text(ty::XS, t.text_tertiary).child(k))),
        )
    }
}

pub fn tip(
    text: impl Into<SharedString>,
    keys: Option<&'static str>,
) -> impl Fn(&mut Window, &mut gpui::App) -> AnyView + 'static {
    let text: SharedString = text.into();
    move |_, cx| cx.new(|_| Tip(text.clone(), keys.map(Into::into))).into()
}

impl FoloApp {
    /// Make `el` focusable and activatable by click, Enter and Space.
    pub fn bind(
        &self,
        el: Stateful<Div>,
        key: &str,
        radius_px: f32,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let t = theme(cx);
        let on_click = action.clone();
        el.track_focus(&fh)
            .relative()
            .on_click(cx.listener(move |this, _, window, cx| {
                on_click(this, window, cx);
                cx.notify();
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") && !ev.keystroke.modifiers.control {
                    action(this, window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, radius_px)))
    }

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
            .active(move |s| s.opacity(0.85))
            .when_some(icon_path, |d, p| d.child(icon(p, 14., fg)))
            .child(text(ty::BASE, fg).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label));
        self.bind(el, key, radius::MD, window, cx, action)
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
        self.bind(el, key, radius::MD, window, cx, action)
    }

    /// A segmented control. Arrow keys move the selection while it has focus.
    pub fn segmented(
        &self,
        key: &str,
        options: &[&str],
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Stateful<Div> {
        self.segmented_ext(key, options, selected, false, window, cx, on_select)
    }

    /// A segmented control that fills its container with equal-width segments (forms).
    pub fn segmented_fill(
        &self,
        key: &str,
        options: &[&str],
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Stateful<Div> {
        self.segmented_ext(key, options, selected, true, window, cx, on_select)
    }

    #[allow(clippy::too_many_arguments)]
    fn segmented_ext(
        &self,
        key: &str,
        options: &[&str],
        selected: usize,
        fill: bool,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let n = options.len();
        let keys = on_select.clone();
        div()
            .id(SharedString::from(key.to_string()))
            .track_focus(&fh)
            .relative()
            .h(px(tsize::CONTROL_MD))
            .p(px(2.))
            .flex()
            .when(fill, |d| d.w_full())
            .when(!fill, |d| d.flex_none())
            .items_center()
            .gap(px(2.))
            .rounded(px(radius::MD))
            .bg(t.bg_inset)
            .border_1()
            .border_color(t.line_default)
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                let next = match ev.keystroke.key.as_str() {
                    "left" | "up" => Some((selected + n - 1) % n),
                    "right" | "down" => Some((selected + 1) % n),
                    "home" => Some(0),
                    "end" => Some(n - 1),
                    _ => None,
                };
                if let Some(i) = next {
                    keys(this, i, window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, radius::MD)))
            .children(options.iter().enumerate().map(|(i, label)| {
                let on = i == selected;
                let f = on_select.clone();
                div()
                    .id(SharedString::from(format!("{key}-{i}")))
                    .h_full()
                    .px(px(10.))
                    .when(fill, |d| d.flex_1())
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .border_1()
                    .border_color(gpui::transparent_black())
                    .when(on, |d| d.bg(t.bg_raised).border_color(t.line_strong))
                    .when(!on, |d| d.hover(|s| s.bg(t.bg_hover)))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        f(this, i, window, cx);
                        cx.notify();
                    }))
                    .child(
                        text(ty::SM, if on { t.text_primary } else { t.text_secondary })
                            .font_weight(if on { FontWeight::MEDIUM } else { FontWeight::NORMAL })
                            .whitespace_nowrap()
                            .tnum()
                            .child(label.to_string()),
                    )
            }))
    }

    /// Filter chip (toggle).
    pub fn chip(
        &self,
        key: &str,
        label: impl Into<SharedString>,
        on: bool,
        swatch_color: Option<Hsla>,
        count: Option<u32>,
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
            .when(on, |d| d.bg(t.bg_selected).border_color(t.accent_line))
            .when(!on, |d| {
                d.border_color(t.line_default).hover(|s| s.bg(t.bg_hover).border_color(t.line_strong))
            })
            .when_some(swatch_color, |d, c| d.child(swatch(c, 8.)))
            .child(
                text(ty::SM, if on { t.text_primary } else { t.text_secondary })
                    .whitespace_nowrap()
                    .child(label.into()),
            )
            .when_some(count, |d, c| d.child(text(ty::XS, t.text_tertiary).tnum().child(int(c as u64))));
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
        let el = div()
            .id(SharedString::from(key.to_string()))
            .w(px(30.))
            .h(px(18.))
            .flex_none()
            .rounded_full()
            .cursor_pointer()
            .bg(if on { t.accent_fill() } else { t.line_strong })
            .when(!on, |d| d.hover(|s| s.bg(t.text_disabled)))
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
                if reduce_motion() {
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
        self.bind(el, key, 9., window, cx, action)
    }

    /// A horizontal slider over `[min, max]`. Drag, click, or use arrows / Home / End.
    #[allow(clippy::too_many_arguments)]
    pub fn slider(
        &self,
        key: &'static str,
        value: f32,
        min: f32,
        max: f32,
        step: f32,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let fh = self.fh(key, cx);
        let ring = self.ring(&fh, window);
        let dragging = self.drag.as_ref().is_some_and(|d| d.id == key);
        let f = ((value - min) / (max - min)).clamp(0.0, 1.0);
        let store = self.slider_bounds.clone();
        let fill_color = t.accent_fill();
        div()
            .id(key)
            .track_focus(&fh)
            .relative()
            .w_full()
            .h(px(20.))
            .cursor_pointer()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, window, cx| {
                    window.focus(&this.fh(key, cx));
                    this.drag = Some(Drag { id: key });
                    let b = this.slider_bounds.borrow().get(key).copied();
                    if let Some(b) = b {
                        let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
                        this.set_slider(key, frac, cx);
                    }
                    cx.notify();
                }),
            )
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let big = ev.keystroke.modifiers.shift;
                let s = step / (max - min) * if big { 10.0 } else { 1.0 };
                let next = match ev.keystroke.key.as_str() {
                    "left" | "down" => Some(f - s),
                    "right" | "up" => Some(f + s),
                    "home" => Some(0.0),
                    "end" => Some(1.0),
                    _ => None,
                };
                if let Some(n) = next {
                    this.set_slider(key, n.clamp(0.0, 1.0), cx);
                    if key == "play-iou" {
                        this.run_predict(cx);
                    }
                    cx.stop_propagation();
                }
            }))
            .child(
                canvas(
                    move |b, _, _| {
                        store.borrow_mut().insert(key, b);
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .left(px(7.))
                .right(px(7.))
                .top_0()
                .bottom_0(),
            )
            .child(
                div()
                    .absolute()
                    .left(px(7.))
                    .right(px(7.))
                    .top(px(8.5))
                    .h(px(3.))
                    .rounded_full()
                    .bg(t.line_strong)
                    .child(div().h_full().w(relative(f)).rounded_full().bg(fill_color)),
            )
            .child(
                div().absolute().left(px(7.)).right(px(7.)).top_0().bottom_0().child(
                    div()
                        .absolute()
                        .left(relative(f))
                        .ml(px(-7.))
                        .top(px(3.))
                        .size(px(14.))
                        .rounded_full()
                        .bg(if dragging { t.accent_fill() } else { gpui::white() })
                        .border_2()
                        .border_color(t.accent_fill())
                        .shadow(t.shadow_sm())
                        .when(ring, |d| d.child(focus_ring(&t, 7.))),
                ),
            )
    }

    /// Map a slider fraction to its value and store it.
    pub fn set_slider(&mut self, key: &str, frac: f32, cx: &mut Context<Self>) {
        let q = |v: f32, step: f32| (v / step).round() * step;
        match key {
            "play-conf" => self.play_conf = q(0.05 + frac * 0.90, 0.01),
            "play-iou" => self.play_iou = q(0.3 + frac * 0.65, 0.01),
            "ds-tile" => self.ds_tile = q(112. + frac * 160., 8.),
            _ => {
                if let Some(f) = self.new_run.as_mut() {
                    match key {
                        "nr-epochs" => f.epochs = q(10. + frac * 290., 5.),
                        "nr-lr0" => f.lr0 = q(0.001 + frac * 0.049, 0.001),
                        "nr-mosaic" => f.mosaic = q(frac, 0.05),
                        "nr-mixup" => f.mixup = q(frac * 0.5, 0.05),
                        "nr-fliplr" => f.fliplr = q(frac, 0.05),
                        "nr-close" => f.close_mosaic = q(frac * 30., 1.),
                        _ => {}
                    }
                }
            }
        }
        cx.notify();
    }

    /// Dropdown trigger that opens a listbox under itself.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        let t = theme(cx);
        let open = self.menu_open == Some(key);
        let fh = self.fh(key, cx);
        let ring = self.ring(&fh, window);
        let n = options.len();
        let empty = n == 0;
        let keys = on_select.clone();
        let trigger = div()
            .id(key)
            .track_focus(&fh)
            .relative()
            .h(px(tsize::CONTROL_LG))
            .w_full()
            .px(px(10.))
            .flex()
            .items_center()
            .gap(px(8.))
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if open { t.accent_line } else { t.line_default })
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_strong))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.menu_open = if this.menu_open == Some(key) { None } else { Some(key) };
                cx.notify();
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                match ev.keystroke.key.as_str() {
                    "enter" | "space" => {
                        this.menu_open = if this.menu_open == Some(key) { None } else { Some(key) };
                    }
                    "up" if n > 0 => keys(this, (selected + n - 1) % n, window, cx),
                    "down" if n > 0 => keys(this, (selected + 1) % n, window, cx),
                    _ => return,
                }
                cx.stop_propagation();
                cx.notify();
            }))
            .when(ring, |d| d.child(focus_ring(&t, radius::MD)))
            .child(text(ty::BASE, t.text_primary).flex_1().truncate().child(current))
            .child(icon(
                if open { "icons/chevron-up.svg" } else { "icons/chevron-down.svg" },
                14.,
                t.text_tertiary,
            ));
        let menu = (open && !empty).then(|| {
            gpui::deferred(
                div()
                    .absolute()
                    .top(px(tsize::CONTROL_LG + 4.))
                    .left_0()
                    .right_0()
                    .occlude()
                    .p(px(4.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .rounded(px(radius::LG))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    .children(options.into_iter().enumerate().map(|(i, (label, detail))| {
                        let f = on_select.clone();
                        let on = i == selected;
                        div()
                            .id(SharedString::from(format!("{key}-opt-{i}")))
                            .h(px(32.))
                            .px(px(8.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .rounded(px(radius::SM))
                            .cursor_pointer()
                            .when(on, |d| d.bg(t.bg_selected))
                            .hover(|s| s.bg(t.bg_hover))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                f(this, i, window, cx);
                                this.menu_open = None;
                                cx.notify();
                            }))
                            .child(text(ty::BASE, t.text_primary).flex_1().truncate().child(label))
                            .when_some(detail, |d, s| d.child(text(ty::XS, t.text_tertiary).tnum().child(s)))
                            .child(
                                div()
                                    .w(px(14.))
                                    .when(on, |d| d.child(icon("icons/check.svg", 14., t.accent_text))),
                            )
                    })),
            )
            .with_priority(2)
        });
        div().relative().w_full().child(trigger).children(menu).into_any_element()
    }
}

pub fn btn_colors(t: &Theme, kind: Btn) -> (Hsla, Hsla, Option<Hsla>, Hsla) {
    match kind {
        Btn::Primary => (t.accent_solid, t.accent_on, None, t.accent_solid_hover),
        Btn::Secondary => {
            (t.bg_raised, t.text_primary, Some(t.line_default), crate::theme::over(t.bg_raised, t.bg_hover))
        }
        Btn::Ghost => (gpui::transparent_black(), t.text_secondary, None, t.bg_hover),
        Btn::Danger => (t.danger_muted, t.danger_text, None, crate::theme::alpha(t.danger_solid, 0.2)),
    }
}
