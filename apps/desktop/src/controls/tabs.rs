//! Underline tabs: a page's sections (a dataset's Images / Statistics, a benchmark's Leaderboard /
//! Plan / Loops), each with its count (design/UNIFIED.md §2, Segmented control and tabs). A filter or
//! a view option is a segmented control instead.

use folo_ui::{
    Context, Div, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::select::OnSelect;
use crate::app::{FoloApp, act};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

/// One tab: its label and what it holds, after the label in tertiary ("2,400", "· 3 findings").
pub struct Tab {
    pub label: SharedString,
    pub meta: Option<SharedString>,
}

impl Tab {
    pub fn new(label: impl Into<SharedString>, meta: Option<String>) -> Self {
        Tab { label: label.into(), meta: meta.map(Into::into) }
    }
}

impl FoloApp {
    /// A row of underline tabs, `selected` one marked by a 2 px underline in primary ink. Each tab is
    /// a Tab stop (`{key}-{i}`), and Enter or Space opens it, as a click does; ← / → (wrapping) and
    /// Home / End open the next, the previous, the first or the last tab and move focus with it.
    pub fn tabs(
        &self,
        key: &str,
        tabs: Vec<Tab>,
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> Div {
        let t = theme(cx);
        let n = tabs.len().max(1);
        let items: Vec<_> = tabs
            .into_iter()
            .enumerate()
            .map(|(i, tab)| {
                let on = i == selected;
                let id = format!("{key}-{i}");
                let el = div()
                    .id(SharedString::from(id.clone()))
                    .h(px(tsize::CONTROL_LG))
                    .flex()
                    .items_center()
                    .border_b_2()
                    .border_color(if on { t.text_primary } else { folo_ui::transparent_black() })
                    .cursor_pointer()
                    .child(
                        div()
                            .id(SharedString::from(format!("{id}-in")))
                            .h(px(26.))
                            .px(px(6.))
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .rounded(px(radius::SM))
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .child(
                                text(ty::BASE, if on { t.text_primary } else { t.text_secondary })
                                    .font_weight(FontWeight::MEDIUM)
                                    .whitespace_nowrap()
                                    .child(tab.label),
                            )
                            .children(tab.meta.map(|m| {
                                text(ty::BASE, t.text_tertiary).tnum().whitespace_nowrap().child(m)
                            })),
                    );
                let keys = on_select.clone();
                let tab_key = key.to_string();
                let el = el.on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                    let next = match ev.keystroke.key.as_str() {
                        "left" => (i + n - 1) % n,
                        "right" => (i + 1) % n,
                        "home" => 0,
                        "end" => n - 1,
                        _ => return,
                    };
                    cx.stop_propagation();
                    if next != i {
                        keys(this, next, window, cx);
                        let fh = this.fh(format!("{tab_key}-{next}"), cx);
                        window.focus(&fh);
                        cx.notify();
                    }
                }));
                let f = on_select.clone();
                self.bind(
                    el,
                    &id,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, w, cx| {
                        f(this, i, w, cx);
                        cx.notify();
                    }),
                )
                .into_any_element()
            })
            .collect();
        div().flex().gap(px(12.)).children(items)
    }
}

#[cfg(test)]
mod tests {
    use folo_ui::{HeadlessApp, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput, WindowHandle};

    use crate::app::tests::{drive, open};
    use crate::app::{FoloApp, Route};
    use crate::bench::BenchTab;

    fn press(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
        let keystroke = Keystroke::parse(key).unwrap();
        app.simulate_input(
            window.into(),
            PlatformInput::KeyDown(KeyDownEvent { keystroke: keystroke.clone(), is_held: false }),
        );
        app.run_until_idle();
        app.frame(window.into());
        app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
        app.run_until_idle();
        app.frame(window.into());
    }

    #[test]
    fn arrows_and_home_end_move_between_tabs_and_focus_follows() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Bench("bn_3c90".into()), w, cx));
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| w.focus(&this.fh("bench-tab-0", cx)));
        app.frame(window.into());
        press(&mut app, window, "right");
        drive(&mut app, window, |this, w, cx| {
            assert_eq!(this.bench.tab, BenchTab::Plan);
            assert!(this.fh("bench-tab-1", cx).is_focused(w), "focus moves with the selection");
        });
        press(&mut app, window, "end");
        assert_eq!(drive(&mut app, window, |this, _, _| this.bench.tab), BenchTab::Loops);
        // Wraps: right from the last is the first.
        press(&mut app, window, "right");
        assert_eq!(drive(&mut app, window, |this, _, _| this.bench.tab), BenchTab::Leaderboard);
        press(&mut app, window, "left");
        assert_eq!(drive(&mut app, window, |this, _, _| this.bench.tab), BenchTab::Loops);
        press(&mut app, window, "home");
        assert_eq!(drive(&mut app, window, |this, _, _| this.bench.tab), BenchTab::Leaderboard);
    }
}
