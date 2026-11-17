//! The command palette: its items (pages, actions, datasets, runs, models) and fuzzy matching.

use crate::api::{Label, SplitsExt};
use folo_ui::prelude::FluentBuilder;
use folo_ui::{
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

/// The items matching `q`, at most 40: the 40 best matches, each group kept together under one heading
/// (the group of the best match first, best first within a group), so a group never shows twice. No
/// query: every item as listed.
pub fn rank(q: &str, items: Vec<PaletteItem>) -> Vec<PaletteItem> {
    let mut scored: Vec<(i32, usize, PaletteItem)> = items
        .into_iter()
        .enumerate()
        .filter_map(|(i, it)| fuzzy(q, &format!("{} {}", it.title, it.group)).map(|s| (s, i, it)))
        .collect();
    if !q.is_empty() {
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        // Cut before grouping: grouping first let a big group of weak matches push a strong match
        // of a later group out of the list.
        scored.truncate(40);
        let mut groups: Vec<&'static str> = Vec::new();
        for (_, _, it) in &scored {
            if !groups.contains(&it.group) {
                groups.push(it.group);
            }
        }
        // Stable: within a group the order by score stays.
        scored.sort_by_key(|(_, _, it)| groups.iter().position(|g| *g == it.group));
    }
    scored.into_iter().map(|(_, _, it)| it).take(40).collect()
}

impl FoloApp {
    /// Every command; `shown` is the theme on screen (its command says so).
    fn palette_items(&self, shown: crate::theme::ThemeId) -> Vec<PaletteItem> {
        let mut items = Vec::new();
        let nav: [(&str, &'static str, Route, &'static [&'static str]); 7] = [
            ("Overview", "icons/layout-dashboard.svg", Route::Overview, &["Ctrl", "1"]),
            ("Datasets", "icons/images.svg", Route::Datasets, &["Ctrl", "2"]),
            ("Runs", "icons/activity.svg", Route::Runs, &["Ctrl", "3"]),
            ("Models", "icons/box.svg", Route::Models, &["Ctrl", "4"]),
            ("Benchmarks", "icons/target.svg", Route::Benches, &["Ctrl", "6"]),
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
        // The open page's run or dataset can be deleted from here too (a run once it isn't active).
        let page_delete =
            match &self.route {
                Route::Run(id) => self.store.run(id).filter(|r| !r.status.is_active()).map(|r| {
                    (r.name.clone(), crate::app::DeleteTarget::Run(id.clone()), "run and its models")
                }),
                Route::Dataset(id) => self.store.dataset_summary(id).map(|d| {
                    (d.name.clone(), crate::app::DeleteTarget::Dataset(id.clone()), "dataset; its files stay")
                }),
                _ => None,
            };
        if let Some((name, target, detail)) = page_delete {
            let title = match target {
                crate::app::DeleteTarget::Run(_) => format!("Delete {name}"),
                crate::app::DeleteTarget::Dataset(_) => format!("Remove {name} from FOLO"),
            };
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/trash-2.svg",
                title,
                detail: detail.into(),
                keys: &[],
                action: act(move |this, w, cx| this.ask_delete(target.clone(), w, cx)),
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
                    detail: format!(
                        "{} · {}",
                        counted(d.splits.total(), "image", "images"),
                        d.format.label()
                    ),
                    keys: &[],
                    action: act(move |this, w, cx| this.navigate(Route::Dataset(id.clone()), w, cx)),
                });
            }
        }
        // One command per theme (custom ones too), as Settings' selects pick them: the theme becomes
        // its mode's, and the mode switches to it unless it is System. Last, so the lists above
        // keep their place; "theme" finds them.
        for e in self.themes.entries() {
            let id = e.id.clone();
            let dark = e.mode == crate::theme::Mode::Dark;
            let mut detail = if dark { "dark theme" } else { "light theme" }.to_string();
            if e.custom {
                detail.insert_str(0, "custom ");
            }
            if shown.as_str() == e.id {
                detail.push_str(" · current");
            }
            items.push(PaletteItem {
                group: "Themes",
                icon: if dark { "icons/moon.svg" } else { "icons/sun.svg" },
                title: format!("Theme: {}", e.name),
                detail,
                keys: &[],
                action: act(move |this, _, cx| this.pick_theme(&id, cx)),
            });
        }
        items
    }

    fn filtered_palette(&self, shown: crate::theme::ThemeId) -> Vec<PaletteItem> {
        let q = self.palette.as_ref().map(|p| p.query.trim().to_string()).unwrap_or_default();
        rank(&q, self.palette_items(shown))
    }

    pub(super) fn palette_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let items = self.filtered_palette(t.id);
        let n = items.len();
        let rows: Vec<(&'static str, String)> = items.iter().map(|it| (it.group, it.title.clone())).collect();
        if let Some(p) = self.palette.as_mut()
            && p.shown != rows
        {
            if let Some(was) = p.shown.get(p.selected)
                && let Some(now) = rows.iter().position(|r| r == was)
            {
                p.selected = now;
            }
            p.shown = rows;
        }
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
                                counted(n as u64, "result", "results")
                            } else {
                                counted(n as u64, "item", "items")
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

    fn item(group: &'static str, title: &str) -> PaletteItem {
        PaletteItem {
            group,
            icon: "",
            title: title.into(),
            detail: String::new(),
            keys: &[],
            action: act(|_, _, _| {}),
        }
    }

    /// "run" matches a run by its group's name ("shapes-n Runs") better than "New training run": the
    /// Runs group showed twice, split around Actions (stress d5-window-input-6).
    #[test]
    fn a_group_shows_once_with_its_best_match_first() {
        let items = vec![
            item("Actions", "New training run"),
            item("Go to", "Runs"),
            item("Runs", "shapes-n"),
            item("Runs", "shapes-mini-s"),
        ];
        let ranked: Vec<(&str, String)> =
            rank("run", items).into_iter().map(|i| (i.group, i.title)).collect();
        let groups: Vec<&str> = ranked.iter().map(|r| r.0).collect();
        assert_eq!(groups, ["Go to", "Runs", "Runs", "Actions"], "{ranked:?}");
        assert_eq!(ranked[1].1, "shapes-n", "within a group, best first");
        // No query: the items as listed.
        let all = rank("", vec![item("Actions", "a"), item("Runs", "b"), item("Actions", "c")]);
        assert_eq!(all.iter().map(|i| i.title.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
    }

    /// The 40 shown are the 40 best before they are grouped: with 40 runs of long names that match
    /// "run" weakly (through their group), grouping first pushed "New training run" out of the list
    /// (round 1 review of stress d5-window-input-6).
    #[test]
    fn the_cut_keeps_the_best_matches_of_every_group() {
        let mut items = vec![item("Actions", "New training run"), item("Go to", "Runs")];
        items.push(item("Runs", "a"));
        for n in 0..39 {
            items.push(item("Runs", &format!("experiment-with-a-very-long-name-{n:03}")));
        }
        assert!(fuzzy("run", "a Runs") > fuzzy("run", "New training run Actions"));
        assert!(
            fuzzy("run", "New training run Actions")
                > fuzzy("run", "experiment-with-a-very-long-name-000 Runs")
        );
        let ranked = rank("run", items);
        assert_eq!(ranked.len(), 40);
        let titles: Vec<&str> = ranked.iter().map(|i| i.title.as_str()).collect();
        assert!(titles.contains(&"New training run"), "{titles:?}");
        let groups: Vec<&str> = ranked.iter().map(|i| i.group).collect();
        assert_eq!(&groups[..3], ["Go to", "Runs", "Runs"], "grouped, the best group first");
        assert_eq!(groups.last(), Some(&"Actions"));
    }
}

#[cfg(test)]
mod theme_commands {
    use crate::app::tests::{drive, open};
    use crate::theme::{ModePref, theme};

    /// "Theme: <name>" for every theme, custom ones too, picking it as Settings does; the toggle stays.
    #[test]
    fn every_theme_has_a_command() {
        let (mut app, window) = open();
        let titles = drive(&mut app, window, |this, _, cx| {
            let custom = crate::theme::custom::parse(
                r#"{"folo_theme": 1, "id": "sepia-2", "name": "Sepia", "mode": "light", "base": "paper"}"#,
            )
            .unwrap();
            this.themes = crate::theme::Catalog::with_custom(vec![custom]);
            let items = super::rank("theme", this.palette_items(theme(cx).id));
            items.into_iter().map(|i| (i.title, i.detail)).collect::<Vec<_>>()
        });
        let names: Vec<&str> = titles.iter().filter_map(|(t, _)| t.strip_prefix("Theme: ")).collect();
        assert_eq!(
            names,
            [
                "Dark",
                "Midnight",
                "Dim",
                "High contrast dark",
                "Light",
                "Paper",
                "High contrast light",
                "Sepia"
            ]
        );
        assert!(titles.iter().any(|(t, _)| t == "Switch to light theme"), "the toggle stays: {titles:?}");
        assert!(titles.iter().any(|(t, d)| t == "Theme: Dark" && d.ends_with("· current")));
        assert!(titles.iter().any(|(t, d)| t == "Theme: Sepia" && d == "custom light theme"));
        // Running one: Sepia's mode with it in the light slot.
        drive(&mut app, window, |this, w, cx| {
            let items = this.palette_items(theme(cx).id);
            let sepia = items.into_iter().find(|i| i.title == "Theme: Sepia").unwrap();
            (sepia.action)(this, w, cx);
        });
        let now = drive(&mut app, window, |this, _, cx| {
            (theme(cx).id.as_str().to_string(), this.theme_prefs.mode, this.theme_prefs.light.clone())
        });
        assert_eq!(now, ("sepia-2".into(), ModePref::Light, "sepia-2".into()));
    }
}
