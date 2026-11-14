//! Underline tabs: a page's sections (a dataset's Images / Statistics, a benchmark's Leaderboard /
//! Plan / Loops), each with its count (design/UNIFIED.md §2, Segmented control and tabs). A filter or
//! a view option is a segmented control instead.

use folo_ui::{
    Context, Div, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
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
    /// a Tab stop (`{key}-{i}`), and Enter or Space opens it, as a click does.
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
