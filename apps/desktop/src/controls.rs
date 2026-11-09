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

/// Which shared chart cursor a chart group drives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cursor {
    /// The linked cursor of the run dashboard and the Overview hero.
    Run,
    /// The PR curve on a model page.
    Pr,
}

/// Called with the chosen option index.
pub type OnSelect = std::rc::Rc<dyn Fn(&mut FoloApp, usize, &mut Window, &mut Context<FoloApp>)>;

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
                .child(text(ty::SM, t.text_primary).max_w(px(280.)).child(self.0.clone()))
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
        let t = theme(cx);
        self.bind_ring(el, key, radius_px, t.focus, window, cx, action)
    }

    /// [`Self::bind`] with a custom ring colour (accent-filled controls ring in the text colour,
    /// since a lime ring around a lime button reads as a thicker button, not focus).
    #[allow(clippy::too_many_arguments)]
    pub fn bind_ring(
        &self,
        el: Stateful<Div>,
        key: &str,
        radius_px: f32,
        ring_color: gpui::Hsla,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let t = theme(cx);
        let on_click = action.clone();
        let click_fh = fh.clone();
        el.track_focus(&fh)
            .relative()
            .on_click(cx.listener(move |this, _, window, cx| {
                // The clicked control takes focus first (keys then act on it, not on whatever had
                // focus before); the action may still move focus on, as dialogs do.
                window.focus(&click_fh);
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
            .when(ring, |d| d.child(focus_ring(&t, radius_px).border_color(ring_color)))
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
        // Pressed is a real step past hover, not a fade.
        let pressed = match kind {
            Btn::Primary => crate::theme::over(t.accent_solid, gpui::hsla(0., 0., 0., 0.14)),
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
                        .child(text(ty::XS, t.text_secondary).whitespace_nowrap().child(reason)),
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
                .child(text(ty::XS, t.text_tertiary).child("Fit"));
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
            .child(text(ty::XS, t.text_secondary).tnum().child(format!("{zoom:.1}×")));
        let ring = self.ring(&self.fh(key.to_string(), cx), window);
        let el = el.when(ring, |d| d.child(focus_tip(&t, tooltip, tsize::CONTROL_MD)));
        self.bind(el, key, radius::MD, window, cx, action)
    }

    fn cursor_mut(&mut self, c: Cursor) -> &mut Option<f32> {
        match c {
            Cursor::Run => &mut self.chart_hover,
            Cursor::Pr => &mut self.pr_hover,
        }
    }

    /// Make a group of charts one Tab stop: ←/→ step the shared cursor by `step` (a fraction of the
    /// x range), Shift steps ×10, Home/End jump, Esc clears. The readouts follow the cursor. `end` is
    /// where recorded data stops (a live run's latest epoch): the cursor starts there and never
    /// steps past it into the epochs still to come.
    #[allow(clippy::too_many_arguments)]
    pub fn chart_group(
        &self,
        key: &str,
        step: f32,
        end: f32,
        cursor: Cursor,
        content: Div,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        content
            .id(SharedString::from(key.to_string()))
            .track_focus(&fh)
            .relative()
            // Esc is bound globally to Dismiss (go back), and actions run before key handlers: with
            // a cursor showing, Esc clears it here instead; otherwise it passes on.
            .on_action(cx.listener(move |this, _: &crate::app::Dismiss, _, cx| {
                let cur = this.cursor_mut(cursor);
                if cur.is_some() {
                    *cur = None;
                    cx.notify();
                } else {
                    cx.propagate();
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let big = if ev.keystroke.modifiers.shift { 10. } else { 1. };
                let cur = this.cursor_mut(cursor);
                let end = end.clamp(0.0, 1.0);
                let at = cur.map_or(end, |c| c.min(end));
                let next = match ev.keystroke.key.as_str() {
                    // The first press inspects the latest point itself, then steps from there.
                    "left" if cur.is_none() => Some(end),
                    "left" => Some((at - step * big).max(0.0)),
                    "right" => Some((at + step * big).min(end)),
                    "home" => Some(0.0),
                    "end" => Some(end),
                    "escape" if cur.is_some() => None,
                    _ => return,
                };
                *cur = next;
                cx.stop_propagation();
                cx.notify();
            }))
            .when(ring, |d| {
                d.child(focus_ring(&t, radius::LG)).child(
                    div()
                        .absolute()
                        .top(gpui::relative(1.))
                        .right_0()
                        .mt(px(6.))
                        .px(px(8.))
                        .py(px(3.))
                        .rounded(px(radius::SM))
                        .bg(t.bg_overlay)
                        .border_1()
                        .border_color(t.line_default)
                        .child(
                            text(ty::XS, t.text_secondary)
                                .child("← → inspect · Shift ×10 · Home / End · Esc clears"),
                        ),
                )
            })
            .into_any_element()
    }

    /// An info glyph that explains a control: tooltip on hover, and a Tab stop that shows the same
    /// text in a popover while keyboard-focused.
    pub fn help_icon(
        &self,
        key: &str,
        help: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let shown = self.ring(&fh, window);
        let el = div()
            .id(SharedString::from(key.to_string()))
            .relative()
            .size(px(20.))
            .flex()
            .items_center()
            .justify_center()
            .rounded(px(radius::SM))
            .cursor_default()
            .hover(|s| s.bg(t.bg_hover))
            .tooltip(tip(help, None))
            .child(icon("icons/info.svg", 12., t.text_tertiary))
            .when(shown, |d| {
                // Anchored under the icon and kept 8 px inside the window (an icon near the right
                // edge would otherwise push it out), like `focus_tip`.
                d.child(
                    div().absolute().top(px(24.)).left(px(-8.)).child(
                        gpui::deferred(
                            gpui::anchored().snap_to_window_with_margin(px(8.)).child(
                                div()
                                    .w(px(260.))
                                    .p(px(8.))
                                    .rounded(px(radius::MD))
                                    .bg(t.bg_overlay)
                                    .border_1()
                                    .border_color(t.line_default)
                                    .shadow(t.shadow_md())
                                    .child(text(ty::SM, t.text_primary).child(help)),
                            ),
                        )
                        .with_priority(1),
                    ),
                )
            });
        self.bind(el, key, radius::SM, window, cx, std::rc::Rc::new(|_, _, _| {})).into_any_element()
    }

    /// A scroll thumb for a scroll container: drawn from its handle, draggable, and a click on
    /// the track jumps there. Place it after the container inside a relative parent.
    pub fn scrollbar(&self, handle: gpui::ScrollHandle, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let paint = handle.clone();
        let hot = self.scroll_drag.as_ref().is_some_and(|(h, _, _)| h.bounds() == handle.bounds());
        div()
            .absolute()
            .top_0()
            .bottom_0()
            .right_0()
            .w(px(12.))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &gpui::MouseDownEvent, _, cx| {
                    let Some(g) = crate::ui::ThumbGeo::of(&handle) else { return };
                    if ev.position.y < g.y || ev.position.y > g.y + g.thumb {
                        // Track click: centre the thumb on the pointer.
                        let travel = (g.h - g.thumb).max(px(1.));
                        let at = ((ev.position.y - g.top - g.thumb / 2.) / travel).clamp(0., 1.);
                        handle.set_offset(gpui::point(handle.offset().x, -(g.max * at)));
                    }
                    this.scroll_drag = Some((handle.clone(), ev.position.y, -handle.offset().y));
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
            .child(
                gpui::canvas(
                    |_, _, _| (),
                    move |_, _, window, _| crate::ui::paint_thumb(&t, &paint, hot, window),
                )
                .size_full(),
            )
            .into_any_element()
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
        self.segmented_ext(key, options, selected, false, &[], window, cx, on_select)
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
        self.segmented_ext(key, options, selected, true, &[], window, cx, on_select)
    }

    /// [`Self::segmented_fill`] with unavailable options: `disabled[i]` is why option `i` can't be
    /// chosen (shown as its tooltip); arrow keys skip it.
    #[allow(clippy::too_many_arguments)]
    pub fn segmented_fill_with(
        &self,
        key: &str,
        options: &[&str],
        selected: usize,
        disabled: &[Option<SharedString>],
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Stateful<Div> {
        self.segmented_ext(key, options, selected, true, disabled, window, cx, on_select)
    }

    #[allow(clippy::too_many_arguments)]
    fn segmented_ext(
        &self,
        key: &str,
        options: &[&str],
        selected: usize,
        fill: bool,
        disabled: &[Option<SharedString>],
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let n = options.len();
        let keys = on_select.clone();
        let off: Vec<Option<SharedString>> = (0..n).map(|i| disabled.get(i).cloned().flatten()).collect();
        let usable: Vec<usize> = (0..n).filter(|&i| off[i].is_none()).collect();
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
            // Its edge is what marks it, as on inputs and selects beside it: line.control.
            .border_color(t.line_control)
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                // Steps to the next option that can be chosen in that direction, wrapping (also from
                // a selection that is itself disabled).
                let ok = |i: &usize| usable.contains(i);
                let next = match ev.keystroke.key.as_str() {
                    "left" | "up" => (1..=n).map(|k| (selected + n - k) % n).find(ok),
                    "right" | "down" => (1..=n).map(|k| (selected + k) % n).find(ok),
                    "home" => usable.first().copied(),
                    "end" => usable.last().copied(),
                    _ => None,
                };
                // A navigation key is this control's even when the selection can't move (one usable
                // option): it mustn't fall through to the page (Playground pans a zoomed image).
                if matches!(ev.keystroke.key.as_str(), "left" | "up" | "right" | "down" | "home" | "end") {
                    cx.stop_propagation();
                }
                if let Some(i) = next.filter(|&i| i != selected) {
                    keys(this, i, window, cx);
                    cx.notify();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, radius::MD)))
            // Keyboard focus shows why the disabled options are off (their tooltips need a pointer).
            .when(ring && off.iter().any(Option::is_some), |d| {
                // Options off for the same reason share one line ("Local GPU, Local CPU: …").
                let mut why: Vec<(Vec<&str>, &str)> = Vec::new();
                for (label, w) in options.iter().zip(&off) {
                    let Some(w) = w.as_ref().map(|w| -> &str { w.as_ref() }) else { continue };
                    match why.iter_mut().find(|(_, r)| *r == w) {
                        Some((labels, _)) => labels.push(label),
                        None => why.push((vec![label], w)),
                    }
                }
                let why: Vec<String> =
                    why.into_iter().map(|(l, w)| format!("{}: {w}", l.join(", "))).collect();
                d.child(focus_tip(&t, why.join("\n"), tsize::CONTROL_MD))
            })
            .children(options.iter().enumerate().map(|(i, label)| {
                let on = i == selected;
                let f = on_select.clone();
                let group = key.to_string();
                let why = off[i].clone();
                let dead = why.is_some();
                div()
                    .id(SharedString::from(format!("{key}-{i}")))
                    .when_some(why, |d, why| d.tooltip(tip(why, None)))
                    .h_full()
                    .px(px(10.))
                    .when(fill, |d| d.flex_1())
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(px(radius::SM))
                    .when(!dead, |d| d.cursor_pointer())
                    .border_1()
                    .border_color(gpui::transparent_black())
                    .when(on, |d| d.bg(t.bg_raised).border_color(t.line_control))
                    .when(!on && !dead, |d| d.hover(|s| s.bg(t.bg_hover)))
                    // The selected option answers the pointer too: hover tint over its raised fill.
                    .when(on, |d| {
                        let hot = crate::theme::over(t.bg_raised, t.bg_hover);
                        d.hover(move |s| s.bg(hot))
                    })
                    .when(!dead, |d| d.active(|s| s.bg(t.bg_active)))
                    .when(!dead, |d| {
                        d.on_click(cx.listener(move |this, _, window, cx| {
                            window.focus(&this.fh(group.clone(), cx));
                            f(this, i, window, cx);
                            cx.notify();
                        }))
                    })
                    .child({
                        // A count after a no-break space ("All\u{a0}9", marked by the caller) is
                        // metadata: tertiary tabular digits after the name. "GPU 0" is a name.
                        let (name, count) = match label.split_once('\u{a0}') {
                            Some((n, c)) => (n, Some(c)),
                            _ => (*label, None),
                        };
                        div()
                            .flex()
                            .items_baseline()
                            .gap(px(6.))
                            .child(
                                text(
                                    ty::SM,
                                    if on {
                                        t.text_primary
                                    } else if dead {
                                        t.text_disabled
                                    } else {
                                        t.text_secondary
                                    },
                                )
                                // One weight either way: a Medium selected label is wider and nudged
                                // its neighbours; the raised fill and primary ink mark the selection.
                                .font_weight(FontWeight::NORMAL)
                                .whitespace_nowrap()
                                .tnum()
                                .child(name.to_string()),
                            )
                            .children(count.map(|c| {
                                text(ty::SM, t.text_tertiary).tnum().whitespace_nowrap().child(c.to_string())
                            }))
                    })
            }))
    }

    /// Filter chip (toggle).
    pub fn chip(
        &self,
        key: &str,
        label: impl Into<SharedString>,
        on: bool,
        swatch_color: Option<Hsla>,
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
            .when_some(swatch_color, |d, c| {
                d.child(if on {
                    div().size(px(12.)).flex().items_center().justify_center().child(icon(
                        "icons/check.svg",
                        12.,
                        c,
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
                    text(ty::XS, if on { t.text_secondary } else { t.text_tertiary }).tnum().child(int(c)),
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
        let dragging = self.drag.as_ref().is_some_and(|d| d.id == key);
        // Mid-drag the range holds still (a widened range would otherwise shrink with the value
        // under the pointer and the thumb run away from it).
        let (min, max) = match self.slider_ranges.borrow().get(key) {
            Some(&r) if dragging => r,
            _ => (min, max),
        };
        self.slider_ranges.borrow_mut().insert(key, (min, max));
        let fh = self.fh(key, cx);
        let ring = self.ring(&fh, window);
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
            // The track spans the full column (aligned with the fields above); the thumb's centre
            // travels 7 px inside it, so the fill is a 7 px lead plus its share of the travel.
            .child(
                div()
                    .absolute()
                    .left_0()
                    .right_0()
                    .top(px(8.5))
                    .h(px(3.))
                    .rounded_full()
                    // The rail is the control's only mark: line.control (3:1), not a separator line.
                    .bg(t.line_control)
                    // 8 px: a px under the proportional fill, so no seam shows where the two meet.
                    .child(
                        div().absolute().left_0().top_0().h_full().w(px(8.)).rounded_l_full().bg(fill_color),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(px(7.))
                            .right(px(7.))
                            .top_0()
                            .h_full()
                            .child(div().h_full().w(relative(f)).bg(fill_color)),
                    ),
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
                        .hover(|s| {
                            s.shadow(t.shadow_md()).bg(crate::theme::over(gpui::white(), t.accent_muted))
                        })
                        .when(ring, |d| d.child(focus_ring(&t, 7.))),
                ),
            )
    }

    /// Map a slider fraction to its value and store it.
    pub fn set_slider(&mut self, key: &str, frac: f32, cx: &mut Context<Self>) {
        let q = |v: f32, step: f32| (v / step).round() * step;
        // The range the slider was drawn with (it may have widened to a typed value), else its
        // preset.
        let range = self.slider_ranges.borrow().get(key).copied();
        let at = |lo: f32, hi: f32| {
            let (lo, hi) = range.unwrap_or((lo, hi));
            lo + frac * (hi - lo)
        };
        // Quantised, except the ends stay exact (a typed 1 epoch or 0.0001 learning rate is an
        // end of a widened range: rounding it to the step would make it invalid).
        let snap = |v: f32, step: f32| {
            if frac <= 0. || frac >= 1. { v } else { q(v, step) }
        };
        match key {
            "play-conf" => self.play_conf = q(at(0.05, 0.95), 0.01),
            "play-iou" => self.play_iou = q(at(0.3, 0.95), 0.01),
            "ds-tile" => self.ds_tile = q(at(112., 272.), 8.),
            _ => {
                if let Some(f) = self.new_run.as_mut() {
                    match key {
                        "nr-epochs" => {
                            f.epochs = snap(at(10., 300.), 5.).round().clamp(1., 10_000.);
                            f.epochs_invalid = false;
                            let (input, v) = (f.epochs_input.clone(), format!("{:.0}", f.epochs));
                            input.update(cx, |i, cx| i.set_text(v, cx));
                            let close = f.close_input.read(cx).text();
                            f.check_close(&close);
                        }
                        "nr-lr0" => {
                            // Above 0 and below 10, the range the field accepts.
                            f.lr0 = snap(at(0.001, 0.05), 0.001).clamp(1e-6, 9.9999);
                            let (input, v) = (f.lr_input.clone(), crate::app::format_lr(f.lr0));
                            input.update(cx, |i, cx| i.set_text(v, cx));
                        }
                        "nr-mosaic" => f.mosaic = q(at(0., 1.), 0.05),
                        "nr-mixup" => f.mixup = q(at(0., 0.5), 0.05),
                        "nr-fliplr" => f.fliplr = q(at(0., 1.), 0.05),
                        "nr-close" => {
                            f.close_mosaic = snap(at(0., 30.), 1.).round().max(0.);
                            let (input, v) = (f.close_input.clone(), format!("{:.0}", f.close_mosaic));
                            input.update(cx, |i, cx| i.set_text(v.clone(), cx));
                            f.check_close(&v);
                        }
                        "nr-scale" => f.scale = q(at(0., 0.9), 0.05),
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
        // While open, arrows move a highlight; Enter commits it, Esc closes without a change.
        let hl = if open { self.menu_hl.min(n.saturating_sub(1)) } else { selected };
        let menu_scroll = self.menu_scroll.clone();
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
            .border_color(if open { t.accent_line } else { t.line_control })
            .cursor_pointer()
            .hover(|s| s.border_color(t.text_tertiary))
            .active(|s| s.bg(t.bg_active))
            .when(open, |d| {
                let cell = self.menu_trigger.clone();
                // Where the trigger is decides where the menu fits; a move redraws once.
                d.child(
                    gpui::canvas(
                        move |b, window, _| {
                            if cell.get() != Some(b) {
                                cell.set(Some(b));
                                // refresh() is ignored mid-draw; this schedules the next frame.
                                window.request_animation_frame();
                            }
                        },
                        |_, _, _, _| {},
                    )
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full(),
                )
            })
            .on_click(cx.listener(move |this, _, window, cx| {
                // The clicked select takes focus, so the keys then work its menu.
                window.focus(&this.fh(key, cx));
                this.menu_open = if this.menu_open == Some(key) { None } else { Some(key) };
                this.menu_hl = selected;
                this.menu_scroll.scroll_to_item(selected);
                cx.notify();
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                let is_open = this.menu_open == Some(key);
                match ev.keystroke.key.as_str() {
                    _ if n == 0 => return,
                    "enter" | "space" if is_open => {
                        let pick = this.menu_hl.min(n - 1);
                        this.menu_open = None;
                        if pick != selected {
                            keys(this, pick, window, cx);
                        }
                    }
                    "enter" | "space" | "down" | "up" if !is_open => {
                        this.menu_open = Some(key);
                        this.menu_hl = selected;
                        this.menu_scroll.scroll_to_item(selected);
                    }
                    "down" => {
                        this.menu_hl = (this.menu_hl + 1).min(n - 1);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "up" => {
                        this.menu_hl = this.menu_hl.saturating_sub(1);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "home" if is_open => {
                        this.menu_hl = 0;
                        this.menu_scroll.scroll_to_item(0);
                    }
                    "end" if is_open => {
                        this.menu_hl = n - 1;
                        this.menu_scroll.scroll_to_item(n - 1);
                    }
                    "escape" if is_open => this.menu_open = None,
                    "tab" if is_open => {
                        this.menu_open = None;
                        return;
                    }
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
        // Below the trigger when it fits, else above; never taller than the room there (8 px margin).
        let want = (n as f32 * 33. + 8.).min(288.);
        let (up, max_h) = match self.menu_trigger.get() {
            Some(b) if open => {
                let vh = f32::from(window.viewport_size().height);
                let below = vh - f32::from(b.bottom()) - 12.;
                let above = f32::from(b.top()) - 12.;
                if below >= want || below >= above {
                    (false, below.min(288.))
                } else {
                    (true, above.min(288.))
                }
            }
            _ => (false, 288.),
        };
        let menu = (open && !empty).then(|| {
            gpui::deferred(
                div()
                    .absolute()
                    .when(!up, |d| d.top(px(tsize::CONTROL_LG + 4.)))
                    .when(up, |d| d.bottom(px(tsize::CONTROL_LG + 4.)))
                    .left_0()
                    .right_0()
                    .occlude()
                    .on_mouse_down_out(cx.listener(move |this, ev: &gpui::MouseDownEvent, _, cx| {
                        let on_trigger = this.menu_trigger.get().is_some_and(|b| b.contains(&ev.position));
                        if this.menu_open == Some(key) && !on_trigger {
                            this.menu_open = None;
                            cx.notify();
                        }
                    }))
                    .rounded(px(radius::LG))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    .child(
                        div()
                            .id(SharedString::from(format!("{key}-menu")))
                            .max_h(px(max_h.max(64.)))
                            .overflow_y_scroll()
                            .track_scroll(&menu_scroll)
                            .p(px(4.))
                            .flex()
                            .flex_col()
                            .gap(px(1.))
                            .children(options.into_iter().enumerate().map(|(i, (label, detail))| {
                                let f = on_select.clone();
                                let on = i == selected;
                                let lit = i == hl;
                                div()
                                    .id(SharedString::from(format!("{key}-opt-{i}")))
                                    .flex_none()
                                    .h(px(32.))
                                    .px(px(8.))
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .rounded(px(radius::SM))
                                    .cursor_pointer()
                                    .when(on, |d| d.bg(t.bg_selected))
                                    .when(lit && !on, |d| d.bg(t.bg_hover))
                                    .when(lit && on, |d| d.bg(crate::theme::over(t.bg_selected, t.bg_hover)))
                                    .hover(|s| s.bg(t.bg_hover))
                                    .active(|s| s.bg(t.bg_active))
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        f(this, i, window, cx);
                                        this.menu_open = None;
                                        cx.notify();
                                    }))
                                    .child(text(ty::BASE, t.text_primary).flex_1().truncate().child(label))
                                    .when_some(detail, |d, s| {
                                        d.child(text(ty::XS, t.text_tertiary).tnum().child(s))
                                    })
                                    .child(
                                        div().w(px(14.)).when(on, |d| {
                                            d.child(icon("icons/check.svg", 14., t.accent_text))
                                        }),
                                    )
                            })),
                    ),
            )
            .with_priority(2)
        });
        div().relative().w_full().child(trigger).children(menu).into_any_element()
    }
}

pub fn btn_colors(t: &Theme, kind: Btn) -> (Hsla, Hsla, Option<Hsla>, Hsla) {
    match kind {
        // Light surfaces: a neutral 1 px edge, not a lime one (design/README). Dark keeps the same
        // 1 px as a transparent edge, so a theme switch never changes a button's width.
        Btn::Primary => (
            t.accent_solid,
            t.accent_on,
            Some(if t.is_dark() { gpui::transparent_black() } else { t.line_default }),
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

/// A tooltip shown while a control has keyboard focus, below it and kept inside the window.
pub fn focus_tip(t: &Theme, text_: impl Into<SharedString>, below: f32) -> Div {
    let text_: SharedString = text_.into();
    // The box the tip is laid out in: it sizes to its text up to this width, then wraps (a narrow
    // control, like a rail icon, would otherwise wrap it word by word).
    div().absolute().top(px(below + 6.)).left_0().w(px(320.)).child(
        gpui::deferred(
            gpui::anchored().snap_to_window_with_margin(px(8.)).child(
                div()
                    .px(px(8.))
                    .py(px(4.))
                    .rounded(px(radius::SM))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    // Short tips stay on one line; long ones wrap at 320 px, not across the window.
                    .max_w(px(320.))
                    .child(text(ty::XS, t.text_secondary).whitespace_normal().child(text_)),
            ),
        )
        .with_priority(3),
    )
}
