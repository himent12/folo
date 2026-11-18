//! Selects: a dropdown trigger and its listbox.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, ClickEvent, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::tooltip::tip;
use crate::app::{FoloApp, focus_ring};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

/// The narrowest a menu opens: wide enough for an option's name and detail side by side, also under
/// a narrow trigger (a leaderboard's Run select is 220 px).
const MENU_MIN_W: f32 = 300.;

/// A menu's width and which edge of its trigger it hangs from: as wide as the trigger, at least
/// [`MENU_MIN_W`]; from the left edge, or the right one when the window has no room for it on the
/// right (`trigger`: its left and right edge, `viewport_w`: the window's width).
fn menu_box(trigger: (f32, f32), viewport_w: f32) -> (f32, bool) {
    let w = (trigger.1 - trigger.0).max(MENU_MIN_W);
    (w, trigger.0 + w > viewport_w - 8.)
}

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
        self.select_full(key, current, options, selected, disabled, &[], false, window, cx, on_select)
    }

    /// A compact (28 px) select of classes, for a toolbar: `classes[i]` is option `i`'s class (its
    /// swatch before the name; `None` for an option that is no class, "All classes"), and the trigger
    /// shows the chosen one's.
    #[allow(clippy::too_many_arguments)]
    pub fn class_select(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        classes: &[Option<u32>],
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        self.select_full(key, current, options, selected, &[], classes, true, window, cx, on_select)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_full(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        disabled: &[Option<SharedString>],
        classes: &[Option<u32>],
        compact: bool,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        let t = theme(cx);
        let trigger_h = if compact { tsize::CONTROL_MD } else { tsize::CONTROL_LG };
        let class_of = |i: usize| classes.get(i).copied().flatten();
        let trigger_class = class_of(selected);
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
            .h(px(trigger_h))
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
                    folo_ui::canvas(
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
            .on_click(cx.listener(move |this, ev: &ClickEvent, window, cx| {
                // Enter and Space work the menu on the press (below); the release's keyboard click
                // would open what they closed, or close what they opened.
                if ev.is_keyboard() {
                    return;
                }
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
            .children(trigger_class.map(|c| class_swatch(&t, c, 8.)))
            .child(
                text(if compact { ty::SM } else { ty::BASE }, t.text_primary)
                    .flex_1()
                    .ellipsis()
                    .child(current),
            )
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
        // Measured on the first open frame: until then, the trigger's own width.
        let wide = self.menu_trigger.get().filter(|_| open).map(|b| {
            menu_box((f32::from(b.left()), f32::from(b.right())), f32::from(window.viewport_size().width))
        });
        let menu_chrome = if open && !empty {
            self.scroll_chrome(menu_scroll.clone(), t.bg_overlay, 0., radius::LG - 1., cx)
        } else {
            Vec::new()
        };
        let menu = (open && !empty).then(|| {
            folo_ui::deferred(
                div()
                    .absolute()
                    .when(!up, |d| d.top(px(trigger_h + 4.)))
                    .when(up, |d| d.bottom(px(trigger_h + 4.)))
                    .map(|d| match wide {
                        Some((w, true)) => d.right_0().w(px(w)),
                        Some((w, false)) => d.left_0().w(px(w)),
                        None => d.left_0().right_0(),
                    })
                    .occlude()
                    .on_mouse_down_out(cx.listener(move |this, ev: &folo_ui::MouseDownEvent, _, cx| {
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
                                let has_detail = detail.is_some();
                                let class = class_of(i);
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
                                    .children(class.map(|c| class_swatch(&t, c, 8.)))
                                    // The name is what you choose by: it keeps its width (up to most
                                    // of the row) and the detail beside it is cut first.
                                    .child(
                                        text(ty::BASE, if dead { t.text_tertiary } else { t.text_primary })
                                            .map(|d| {
                                                if has_detail {
                                                    d.flex_none().max_w(relative(0.7))
                                                } else {
                                                    d.flex_1()
                                                }
                                            })
                                            .ellipsis()
                                            .child(label),
                                    )
                                    .when_some(detail, |d, s| {
                                        d.child(
                                            text(ty::SM, t.text_tertiary)
                                                .tnum()
                                                .flex_1()
                                                .min_w(px(0.))
                                                .text_right()
                                                .ellipsis()
                                                .child(s),
                                        )
                                    })
                                    .child(
                                        div().w(px(14.)).flex_none().when(on, |d| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_narrow_select_opens_a_wider_menu_toward_the_room() {
        // A 220 px select by the window's right edge: 300 px wide, hung from its right edge.
        assert_eq!(menu_box((1000., 1220.), 1280.), (300., true));
        // The same select further left: from its left edge.
        assert_eq!(menu_box((600., 820.), 1280.), (300., false));
        // A wide select keeps its own width.
        assert_eq!(menu_box((900., 1240.), 1280.), (340., false));
    }
}
