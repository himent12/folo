//! Segmented controls.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    Context, Div, FontWeight, InteractiveElement, KeyDownEvent, ParentElement, SharedString, Stateful,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::select::OnSelect;
use super::tooltip::{focus_tip, tip};
use crate::app::{FoloApp, focus_ring};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

impl FoloApp {
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

    /// [`Self::segmented`] with unavailable options: `disabled[i]` is why option `i` can't be chosen
    /// (shown as its tooltip); arrow keys skip it.
    #[allow(clippy::too_many_arguments)]
    pub fn segmented_with(
        &self,
        key: &str,
        options: &[&str],
        selected: usize,
        disabled: &[Option<SharedString>],
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Stateful<Div> {
        self.segmented_ext(key, options, selected, false, disabled, window, cx, on_select)
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
                let m = &ev.keystroke.modifiers;
                if n == 0 || m.control || m.alt || m.platform {
                    return;
                }
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
                    .border_color(folo_ui::transparent_black())
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
}
