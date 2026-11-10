//! Selects: a dropdown trigger and its listbox.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::tooltip::tip;
use crate::app::{FoloApp, focus_ring};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

/// Called with the chosen option index.
pub type OnSelect = std::rc::Rc<dyn Fn(&mut FoloApp, usize, &mut Window, &mut Context<FoloApp>)>;

impl FoloApp {
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
        self.select_with(key, current, options, selected, &[], window, cx, on_select)
    }

    /// [`Self::select`] with options that can't be chosen now: `disabled[i]` says why option `i`
    /// can't (its tooltip); it shows dimmed, the pointer can't pick it and the arrows skip it.
    #[allow(clippy::too_many_arguments)]
    pub fn select_with(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        disabled: &[Option<SharedString>],
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
        let off: Vec<Option<SharedString>> = (0..n).map(|i| disabled.get(i).cloned().flatten()).collect();
        let usable: Vec<usize> = (0..n).filter(|&i| off[i].is_none()).collect();
        let (first, last) = (usable.first().copied().unwrap_or(0), usable.last().copied().unwrap_or(0));
        // The usable option after (or before) `i`, staying put at the ends.
        let step = move |i: usize, down: bool| {
            let next =
                if down { usable.iter().find(|&&u| u > i) } else { usable.iter().rev().find(|&&u| u < i) };
            next.copied().unwrap_or(i)
        };
        // While open, arrows move a highlight; Enter commits it, Esc closes without a change.
        let hl = if open { self.menu_hl.min(n.saturating_sub(1)) } else { selected };
        let menu_scroll = self.menu_scroll.clone();
        let dead: Vec<bool> = off.iter().map(Option::is_some).collect();
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
                        if pick != selected && !dead[pick] {
                            keys(this, pick, window, cx);
                        }
                    }
                    "enter" | "space" | "down" | "up" if !is_open => {
                        this.menu_open = Some(key);
                        this.menu_hl = selected;
                        this.menu_scroll.scroll_to_item(selected);
                    }
                    "down" => {
                        this.menu_hl = step(this.menu_hl.min(n - 1), true);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "up" => {
                        this.menu_hl = step(this.menu_hl.min(n - 1), false);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "home" if is_open => {
                        this.menu_hl = first;
                        this.menu_scroll.scroll_to_item(first);
                    }
                    "end" if is_open => {
                        this.menu_hl = last;
                        this.menu_scroll.scroll_to_item(last);
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
            .child(text(ty::BASE, t.text_primary).flex_1().ellipsis().child(current))
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
        let menu_chrome = if open && !empty {
            self.scroll_chrome(menu_scroll.clone(), t.bg_overlay, 0., radius::LG - 1., cx)
        } else {
            Vec::new()
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
                                let why = off[i].clone();
                                let dead = why.is_some();
                                div()
                                    .id(SharedString::from(format!("{key}-opt-{i}")))
                                    .flex_none()
                                    .h(px(32.))
                                    .px(px(8.))
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .rounded(px(radius::SM))
                                    .when_some(why, |d, why| d.tooltip(tip(why, None)))
                                    .when(on, |d| d.bg(t.bg_selected))
                                    .when(lit && !on, |d| d.bg(t.bg_hover))
                                    .when(lit && on, |d| d.bg(crate::theme::over(t.bg_selected, t.bg_hover)))
                                    .when(!dead, |d| {
                                        d.cursor_pointer()
                                            .hover(|s| s.bg(t.bg_hover))
                                            .active(|s| s.bg(t.bg_active))
                                            .on_click(cx.listener(move |this, _, window, cx| {
                                                f(this, i, window, cx);
                                                this.menu_open = None;
                                                cx.notify();
                                            }))
                                    })
                                    .child(
                                        text(ty::BASE, if dead { t.text_tertiary } else { t.text_primary })
                                            .flex_1()
                                            .ellipsis()
                                            .child(label),
                                    )
                                    .when_some(detail, |d, s| {
                                        d.child(text(ty::SM, t.text_tertiary).tnum().child(s))
                                    })
                                    .child(
                                        div().w(px(14.)).when(on, |d| {
                                            d.child(icon("icons/check.svg", 14., t.accent_text))
                                        }),
                                    )
                            })),
                    )
                    .children(menu_chrome),
            )
            .with_priority(2)
        });
        div().relative().w_full().child(trigger).children(menu).into_any_element()
    }
}
