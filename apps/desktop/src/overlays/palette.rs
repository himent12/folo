//! The command palette: its items (pages, actions, datasets, runs, models) and fuzzy matching.

use crate::api::{Label, SplitsExt};
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, MouseMoveEvent, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::RunStatus;
use crate::app::{Action, FoloApp, Route, act};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

pub struct PaletteItem {
    pub group: &'static str,
    pub icon: &'static str,
    pub title: String,
    pub detail: String,
    pub keys: &'static [&'static str],
    pub action: Action,
}

/// Subsequence match score: higher is better; None if `q` is not a subsequence of `s`.
pub fn fuzzy(q: &str, s: &str) -> Option<i32> {
    if q.is_empty() {
        return Some(0);
    }
    let s_l = s.to_lowercase();
    let q_l = q.to_lowercase();
    if let Some(pos) = s_l.find(&q_l) {
        return Some(1000 - pos as i32 * 2 + if pos == 0 { 200 } else { 0 });
    }
    let mut score = 0;
    let mut last = None;
    let mut it = s_l.char_indices();
    for qc in q_l.chars() {
        loop {
            let (i, c) = it.next()?;
            if c == qc {
                score += if last.is_some_and(|l: usize| l + c.len_utf8() == i) { 8 } else { 1 };
                last = Some(i);
                break;
            }
        }
    }
    Some(score)
}

impl FoloApp {
    fn palette_items(&self) -> Vec<PaletteItem> {
        let mut items = Vec::new();
        let nav: [(&str, &'static str, Route, &'static [&'static str]); 6] = [
            ("Overview", "icons/layout-dashboard.svg", Route::Overview, &["Ctrl", "1"]),
            ("Datasets", "icons/images.svg", Route::Datasets, &["Ctrl", "2"]),
            ("Runs", "icons/activity.svg", Route::Runs, &["Ctrl", "3"]),
            ("Models", "icons/box.svg", Route::Models, &["Ctrl", "4"]),
            ("Playground", "icons/scan-eye.svg", Route::Playground, &["Ctrl", "5"]),
            ("Settings", "icons/settings.svg", Route::Settings, &["Ctrl", ","]),
        ];
        items.push(PaletteItem {
            group: "Actions",
            icon: "icons/plus.svg",
            title: "New training run".into(),
            detail: String::new(),
            keys: &["Ctrl", "N"],
            action: act(|this, w, cx| this.open_new_run(None, w, cx)),
        });
        items.push(PaletteItem {
            group: "Actions",
            icon: "icons/image-up.svg",
            title: "Open image in Playground".into(),
            detail: String::new(),
            keys: &["Ctrl", "O"],
            action: act(|this, _, cx| this.pick_image(cx)),
        });
        let dark = self.mode == crate::theme::Mode::Dark;
        items.push(PaletteItem {
            group: "Actions",
            icon: if dark { "icons/sun.svg" } else { "icons/moon.svg" },
            title: if dark { "Switch to light theme".into() } else { "Switch to dark theme".into() },
            detail: String::new(),
            keys: &["Ctrl", "Shift", "L"],
            action: act(|this, _, cx| this.toggle_theme(cx)),
        });
        items.push(PaletteItem {
            group: "Actions",
            icon: "icons/maximize.svg",
            title: "Toggle full screen".into(),
            detail: String::new(),
            keys: &["F11"],
            action: act(|_, w, _| w.toggle_fullscreen()),
        });
        if let Some(live) = self.store.live_run() {
            let id = live.id.clone();
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/stop.svg",
                title: format!("Stop {}", live.name),
                detail: format!("epoch {} / {}", live.epoch, live.epochs),
                keys: &[],
                action: act(move |this, w, cx| {
                    this.ask_stop(id.clone(), w, cx);
                    cx.notify();
                }),
            });
        }
        for (name, icon_path, route, keys) in nav {
            items.push(PaletteItem {
                group: "Go to",
                icon: icon_path,
                title: name.into(),
                detail: String::new(),
                keys,
                action: act(move |this, w, cx| this.navigate(route.clone(), w, cx)),
            });
        }
        if let Some(runs) = self.store.runs.ready() {
            for r in runs {
                let id = r.id.clone();
                let detail = match r.status {
                    RunStatus::Running => format!("running · epoch {} / {}", r.epoch, r.epochs),
                    _ => format!("{} · {}", status_label(r).to_lowercase(), r.config.arch.as_str()),
                };
                items.push(PaletteItem {
                    group: "Runs",
                    icon: "icons/activity.svg",
                    title: r.name.clone(),
                    detail,
                    keys: &[],
                    action: act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
                });
            }
        }
        if let Some(models) = self.store.models.ready() {
            for m in models {
                let id = m.id.clone();
                let detail = m.metrics.as_ref().map_or(m.arch.as_str().to_string(), |x| {
                    format!("{} · mAP50-95 {:.3}", m.arch.as_str(), x.map50_95)
                });
                items.push(PaletteItem {
                    group: "Models",
                    icon: "icons/box.svg",
                    title: m.name.clone(),
                    detail,
                    keys: &[],
                    action: act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx)),
                });
            }
        }
        if let Some(ds) = self.store.datasets.ready() {
            for d in ds {
                let id = d.id.clone();
                items.push(PaletteItem {
                    group: "Datasets",
                    icon: "icons/images.svg",
                    title: d.name.clone(),
                    detail: format!("{} images · {}", int(d.splits.total()), d.format.label()),
                    keys: &[],
                    action: act(move |this, w, cx| this.navigate(Route::Dataset(id.clone()), w, cx)),
                });
            }
        }
        items
    }

    fn filtered_palette(&self) -> Vec<PaletteItem> {
        let q = self.palette.as_ref().map(|p| p.query.trim().to_string()).unwrap_or_default();
        let mut scored: Vec<(i32, usize, PaletteItem)> = self
            .palette_items()
            .into_iter()
            .enumerate()
            .filter_map(|(i, it)| fuzzy(&q, &format!("{} {}", it.title, it.group)).map(|s| (s, i, it)))
            .collect();
        if !q.is_empty() {
            scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        }
        scored.into_iter().map(|(_, _, it)| it).take(40).collect()
    }

    pub(super) fn palette_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let items = self.filtered_palette();
        let n = items.len();
        let p = self.palette.as_ref().unwrap();
        // With no query the list is every item (commands, datasets, runs, models); with one, matches.
        let searching = !p.query.trim().is_empty();
        let selected = p.selected.min(n.saturating_sub(1));
        let input = p.input.clone();
        let actions: Vec<Action> = items.iter().map(|i| i.action.clone()).collect();
        let scroll = p.scroll.clone();
        let scroll_bar = p.scroll.clone();
        let mut list = div()
            .id("palette-list")
            .max_h(px(400.))
            .overflow_y_scroll()
            .track_scroll(&scroll)
            .flex()
            .flex_col()
            .p(px(6.));
        // Child index of each result row (group headings are children too).
        let mut row_child: Vec<usize> = Vec::new();
        let mut children = 0usize;
        let mut last_group = "";
        for (i, it) in items.into_iter().enumerate() {
            if it.group != last_group {
                last_group = it.group;
                children += 1;
                list = list.child(
                    div()
                        .flex_none()
                        .px(px(10.))
                        .pt(px(if i == 0 { 4. } else { 10. }))
                        .pb(px(4.))
                        .child(overline(&t, it.group)),
                );
            }
            let on = i == selected;
            let action = it.action.clone();
            row_child.push(children);
            children += 1;
            list = list.child(
                div()
                    .id(("pal", i))
                    .flex_none()
                    .h(px(36.))
                    .px(px(10.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .rounded(px(radius::MD))
                    .cursor_pointer()
                    .when(on, |d| d.bg(t.bg_active))
                    // Pressed reads stronger than the pointer-selected row.
                    .active(|s| s.bg(crate::theme::over(t.bg_active, t.bg_hover)))
                    .on_mouse_move(cx.listener(move |this, _: &MouseMoveEvent, _, cx| {
                        if let Some(p) = this.palette.as_mut()
                            && p.selected != i
                        {
                            p.selected = i;
                            cx.notify();
                        }
                    }))
                    .on_click(cx.listener(move |this, _, w, cx| {
                        this.close_palette(w, cx);
                        action(this, w, cx);
                    }))
                    // Stopping a run is destructive: danger text here as on its button.
                    .child(icon(
                        it.icon,
                        16.,
                        if it.icon == "icons/stop.svg" {
                            t.danger_text
                        } else if on {
                            t.text_primary
                        } else {
                            t.text_tertiary
                        },
                    ))
                    .child(
                        text(
                            ty::BASE,
                            if it.icon == "icons/stop.svg" { t.danger_text } else { t.text_primary },
                        )
                        // Shrinks (a 100-character run name ends in "…"), after the detail does.
                        .flex_shrink()
                        .min_w(px(80.))
                        .max_w(px(360.))
                        .ellipsis()
                        .child(it.title),
                    )
                    .child(
                        text(ty::SM, if on { t.text_secondary } else { t.text_tertiary })
                            .flex_1()
                            .min_w_0()
                            .ellipsis()
                            .child(it.detail),
                    )
                    .when(!it.keys.is_empty(), |d| d.child(shortcut(&t, it.keys)))
                    .when(on && it.keys.is_empty(), |d| {
                        d.child(icon("icons/arrow-right.svg", 14., t.text_tertiary))
                    }),
            );
        }
        if n == 0 {
            list =
                list.child(div().p(px(24.)).flex().justify_center().child(
                    text(ty::SM, t.text_tertiary).child("No matches. Try a run, model or dataset name."),
                ));
        }
        let panel_el =
            div().id("palette").absolute().top(px(88.)).left_0().right_0().flex().justify_center().child(
                div()
                    .id("palette-box")
                    .w(px(640.))
                    .occlude()
                    .flex()
                    .flex_col()
                    .rounded(px(radius::XL))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    .overflow_hidden()
                    .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, w, cx| {
                        let Some(p) = this.palette.as_mut() else { return };
                        match ev.keystroke.key.as_str() {
                            "down" | "up" => {
                                p.selected = if ev.keystroke.key == "down" {
                                    (p.selected + 1).min(n.saturating_sub(1))
                                } else {
                                    p.selected.saturating_sub(1)
                                };
                                // Reveal the row; for the first row of a group, its heading too.
                                if let Some(&ix) = row_child.get(p.selected) {
                                    let ix = if ix > 0
                                        && (p.selected == 0 || row_child[p.selected - 1] + 1 != ix)
                                    {
                                        ix - 1
                                    } else {
                                        ix
                                    };
                                    p.scroll.scroll_to_item(ix);
                                }
                            }
                            "enter" => {
                                let sel = p.selected.min(n.saturating_sub(1));
                                if let Some(a) = actions.get(sel).cloned() {
                                    this.close_palette(w, cx);
                                    a(this, w, cx);
                                }
                            }
                            _ => return,
                        }
                        cx.stop_propagation();
                        cx.notify();
                    }))
                    .child(
                        div()
                            .h(px(52.))
                            .px(px(16.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .text_size(px(ty::MD.size))
                            .child(icon("icons/search.svg", 16., t.text_tertiary))
                            .child(input)
                            .child(kbd(&t, "Esc")),
                    )
                    .child(div().relative().child(list).children(self.scroll_chrome(
                        scroll_bar,
                        t.bg_overlay,
                        0.,
                        0.,
                        cx,
                    )))
                    .child(
                        div()
                            .h(px(36.))
                            .px(px(14.))
                            .flex()
                            .items_center()
                            .gap(px(14.))
                            .border_t_1()
                            .border_color(t.line_subtle)
                            .bg(t.bg_raised)
                            // The frame's 12 px corners, less its 1 px edge: a square fill would show past them.
                            .rounded_b(px(radius::XL - 1.))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(shortcut(&t, &["↑", "↓"]))
                                    .child(text(ty::SM, t.text_tertiary).child("navigate")),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(kbd(&t, "Enter"))
                                    .child(text(ty::SM, t.text_tertiary).child("open")),
                            )
                            .child(div().flex_1())
                            .child(text(ty::SM, t.text_tertiary).tnum().child(if searching {
                                format!("{n} results")
                            } else {
                                format!("{n} items")
                            })),
                    ),
            );
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(self.scrim(&t, "palette-scrim", window, cx, act(|this, w, cx| this.close_palette(w, cx))))
            .child(crate::ui::appear(panel_el, "palette-in"))
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_ranks_prefix_and_substring() {
        assert!(fuzzy("run", "Runs").unwrap() > fuzzy("run", "shapes-n-run").unwrap());
        assert!(fuzzy("shn", "shapes-n").is_some());
        assert!(fuzzy("zzz", "shapes").is_none());
        assert_eq!(fuzzy("", "anything"), Some(0));
    }
}
