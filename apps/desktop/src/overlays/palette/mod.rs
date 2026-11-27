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
/// (the group of the best match first, best first within a group), so a group never shows twice.
/// Every `title`/`group` match strictly precedes fallback matches in `detail` (detail-only matches
/// whose group already appeared in primary matches are grouped under `"Details"`). No query: every
/// item as listed.
pub fn rank(q: &str, items: Vec<PaletteItem>) -> Vec<PaletteItem> {
    let mut scored: Vec<(bool, i32, usize, PaletteItem)> = items
        .into_iter()
        .enumerate()
        .filter_map(|(i, it)| {
            if let Some(s) = fuzzy(q, &format!("{} {}", it.title, it.group)) {
                Some((true, s, i, it))
            } else if !it.detail.is_empty() {
                fuzzy(q, &it.detail).map(|s| (false, s, i, it))
            } else {
                None
            }
        })
        .collect();
    if !q.is_empty() {
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        // Cut before grouping: grouping first let a big group of weak matches push a strong match
        // of a later group out of the list.
        scored.truncate(40);
        let mut primary_groups: Vec<&'static str> = Vec::new();
        for (primary, _, _, it) in &scored {
            if *primary && !primary_groups.contains(&it.group) {
                primary_groups.push(it.group);
            }
        }
        // Keep detail-only matches strictly below every primary match without repeating a heading.
        for (primary, _, _, it) in &mut scored {
            if !*primary && primary_groups.contains(&it.group) {
                it.group = "Details";
            }
        }
        let mut groups: Vec<&'static str> = primary_groups;
        for (_, _, _, it) in &scored {
            if !groups.contains(&it.group) {
                groups.push(it.group);
            }
        }
        // Stable: within a group the order by (primary, score) stays.
        scored.sort_by_key(|(_, _, _, it)| groups.iter().position(|g| *g == it.group));
    }
    let limit = if q.is_empty() { 48 } else { 40 };
    scored.into_iter().map(|(_, _, _, it)| it).take(limit).collect()
}

/// Rows skipped by PageUp / PageDown inside the palette result list.
const PALETTE_PAGE_STEP: usize = 8;

/// Scrolls the palette list to keep the selected row in view. When moving upward (or jumping to row
/// 0), reveals a group's heading above its first row; when moving downward, targets the row itself
/// so `FirstVisible` never stops at the heading and leaves the selected row below the viewport.
fn reveal_palette_item(p: &mut crate::app::Palette, row_child: &[usize], moving_down: bool) {
    if let Some(&ix) = row_child.get(p.selected) {
        let first_in_group = ix > 0 && (p.selected == 0 || row_child[p.selected - 1] + 1 != ix);
        let target = if !moving_down && first_in_group { ix - 1 } else { ix };
        p.scroll.scroll_to_item(target);
    }
}

impl FoloApp {
    /// Every command; `shown` is the theme on screen (its command says so).
    fn palette_items(&self, shown: crate::theme::ThemeId) -> Vec<PaletteItem> {
        let mut items = Vec::new();
        let nav: [(&str, &'static str, Route, &'static [&'static str]); 8] = [
            ("Overview", "icons/layout-dashboard.svg", Route::Overview, &["Ctrl", "1"]),
            ("Datasets", "icons/images.svg", Route::Datasets, &["Ctrl", "2"]),
            ("Runs", "icons/activity.svg", Route::Runs, &["Ctrl", "3"]),
            ("Models", "icons/box.svg", Route::Models, &["Ctrl", "4"]),
            ("Benchmarks", "icons/target.svg", Route::Benches, &["Ctrl", "5"]),
            ("Playground", "icons/scan-eye.svg", Route::Playground, &["Ctrl", "6"]),
            ("Labeling", "icons/sparkles.svg", Route::LabelJobs, &["Ctrl", "7"]),
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
        if self.store.runs.ready().is_some_and(|l| l.len() >= 2) {
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/git-compare-arrows.svg",
                title: "Compare runs".into(),
                detail: "Overlay loss & mAP curves".into(),
                keys: &[],
                action: act(|this, w, cx| this.open_run_compare(None, w, cx)),
            });
        }
        if self.store.models.ready().is_some_and(|l| l.len() >= 2) {
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/git-compare-arrows.svg",
                title: "Compare models".into(),
                detail: "Side-by-side metrics, per-class AP & export".into(),
                keys: &[],
                action: act(|this, w, cx| this.open_model_compare(None, w, cx)),
            });
        }
        if self.label_on() {
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/sparkles.svg",
                title: "New labeling job".into(),
                detail: "Agent labeling".into(),
                keys: &[],
                action: act(|this, w, cx| {
                    this.navigate(Route::LabelJobs, w, cx);
                    this.open_new_label_job(w, cx);
                }),
            });
        }
        items.push(PaletteItem {
            group: "Actions",
            icon: "icons/image-up.svg",
            title: "Open images in Playground".into(),
            detail: "Single image or multi-file batch".into(),
            keys: &["Ctrl", "O"],
            action: act(|this, w, cx| this.pick_image(w, cx)),
        });
        items.push(PaletteItem {
            group: "Actions",
            icon: "icons/folder-open.svg",
            title: "Open folder in Playground".into(),
            detail: "Run a model across a folder of images".into(),
            keys: &[],
            action: act(|this, w, cx| this.pick_play_folder(w, cx)),
        });
        if self.play_sample_dataset().is_some() {
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/images.svg",
                title: if self.server_mode == crate::app::ServerMode::Mock {
                    "Load demo batch in Playground".into()
                } else {
                    "Load validation batch in Playground".into()
                },
                detail: "Sample images for batch filmstrip & export".into(),
                keys: &[],
                action: act(|this, w, cx| this.load_play_demo_batch(w, cx)),
            });
        }
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
        if !self.narrow {
            let collapsed = self.sidebar_collapsed;
            items.push(PaletteItem {
                group: "Actions",
                icon: if collapsed { "icons/chevron-right.svg" } else { "icons/chevron-left.svg" },
                title: if collapsed { "Expand sidebar".into() } else { "Collapse sidebar".into() },
                detail: String::new(),
                keys: &["Ctrl", "B"],
                action: act(|this, w, cx| {
                    if w.viewport_size().width >= px(1120.) {
                        this.sidebar_collapsed = !this.sidebar_collapsed;
                        cx.notify();
                    }
                }),
            });
        }
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
        // Page-scoped export and CLI quick-copy actions on Run detail and Playground Compare.
        if let Route::Run(id) = &self.route
            && let Some(r) = self.store.run(id)
        {
            if let Some(d) = self.store.run_data.get(id).and_then(|l| l.ready())
                && !d.metrics.epochs.is_empty()
            {
                let rid_csv = id.clone();
                items.push(PaletteItem {
                    group: "Actions",
                    icon: "icons/download.svg",
                    title: "Export run metrics (CSV)".into(),
                    detail: format!("{} · {} epochs", r.name, d.metrics.epochs.len()),
                    keys: &[],
                    action: act(move |this, _, cx| this.export_run_metrics_csv(&rid_csv, cx)),
                });
                if d.metrics.epochs.last().is_some_and(|e| !e.val.per_class.is_empty()) {
                    let rid_pc = id.clone();
                    items.push(PaletteItem {
                        group: "Actions",
                        icon: "icons/download.svg",
                        title: "Export run per-class AP (CSV)".into(),
                        detail: r.name.clone(),
                        keys: &[],
                        action: act(move |this, _, cx| this.export_run_per_class_csv(&rid_pc, cx)),
                    });
                }
                let rid_json = id.clone();
                items.push(PaletteItem {
                    group: "Actions",
                    icon: "icons/file-text.svg",
                    title: "Export run metrics (JSON)".into(),
                    detail: format!("{} · config & epochs", r.name),
                    keys: &[],
                    action: act(move |this, _, cx| this.export_run_metrics_json(&rid_json, cx)),
                });
            }
            if r.model_id.is_some() {
                let rid_val = id.clone();
                items.push(PaletteItem {
                    group: "Actions",
                    icon: "icons/copy.svg",
                    title: "Copy checkpoint val CLI".into(),
                    detail: "folo val for this run's checkpoint".into(),
                    keys: &[],
                    action: act(move |this, _, cx| {
                        this.copy_run_checkpoint_cli(
                            &rid_val,
                            crate::screens::runs::CheckpointCliMode::Val,
                            cx,
                        )
                    }),
                });
                let rid_pred = id.clone();
                items.push(PaletteItem {
                    group: "Actions",
                    icon: "icons/copy.svg",
                    title: "Copy checkpoint predict CLI".into(),
                    detail: "folo predict for this run's checkpoint".into(),
                    keys: &[],
                    action: act(move |this, _, cx| {
                        this.copy_run_checkpoint_cli(
                            &rid_pred,
                            crate::screens::runs::CheckpointCliMode::Predict,
                            cx,
                        )
                    }),
                });
            }
        }
        if matches!(self.route, Route::Playground) && self.play_compare.enabled && self.has_play_export_json()
        {
            items.push(PaletteItem {
                group: "Actions",
                icon: "icons/file-text.svg",
                title: "Export A/B compare JSON".into(),
                detail: "Per-class diff (Both / Only A / Only B) and detections".into(),
                keys: &[],
                action: act(|this, _, cx| this.export_play_json(cx)),
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
                let arch = self.store.run_arch(r);
                let detail = match r.status {
                    RunStatus::Running => format!("running · epoch {} / {}", r.epoch, r.epochs),
                    _ => format!("{} · {}", status_label(r).to_lowercase(), arch.as_str()),
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
                let arch = self.store.model_arch(m);
                let detail = m.metrics.as_ref().map_or(arch.as_str().to_string(), |x| {
                    format!("{} · mAP50-95 {:.3}", arch.as_str(), x.map50_95)
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
        if let Some(benches) = self.bench.list.ready() {
            for b in benches {
                let id = b.id.clone();
                let mut detail = counted(b.images, "image", "images");
                if let Some(best) = b.best.as_ref().filter(|x| x.score.is_finite()) {
                    detail.push_str(&format!(" · {} {:.1}", best.name, best.score));
                }
                items.push(PaletteItem {
                    group: "Benchmarks",
                    icon: "icons/target.svg",
                    title: b.name.clone(),
                    detail,
                    keys: &[],
                    action: act(move |this, w, cx| this.navigate(Route::Bench(id.clone()), w, cx)),
                });
            }
        }
        if let Some(jobs) = self.label.jobs.ready() {
            for j in jobs {
                let id = j.id.clone();
                let detail = format!(
                    "{} / {} labeled · {}",
                    int(j.counts.labeled),
                    int(j.counts.images),
                    j.classes.join(", ")
                );
                items.push(PaletteItem {
                    group: "Labeling",
                    icon: "icons/sparkles.svg",
                    title: j.name.clone(),
                    detail,
                    keys: &[],
                    action: act(move |this, w, cx| this.navigate(Route::LabelJob(id.clone()), w, cx)),
                });
            }
        }
        // One command per theme (custom ones too), as Settings' selects pick them: the theme becomes
        // its mode's, and the mode switches to it unless it is System. Last, and for a query only
        // (`filtered_palette`): "theme" finds them.
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
        let mut items = self.palette_items(shown);
        // The theme commands are offered for a query only, as in Studio (design/THEMES.md): seven or
        // more of them would crowd the empty palette.
        if q.is_empty() {
            items.retain(|it| it.group != "Themes");
        }
        rank(&q, items)
    }

    pub(super) fn palette_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        // Trigger benchmark & labeling loading if `/system` reported features after the palette opened.
        self.ensure_benches(cx);
        self.ensure_label_jobs(cx);
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
            list = list.child(div().p(px(24.)).flex().justify_center().child(
                text(ty::SM, t.text_tertiary).child("No matches. Try a run, model, dataset or benchmark."),
            ));
        }
        let row_child_home = row_child.clone();
        let row_child_end = row_child.clone();
        let panel_el =
            div().id("palette").absolute().top(px(88.)).left_0().right_0().flex().justify_center().child(
                div()
                    .id("palette-box")
                    .w(px(640.))
                    .max_w(folo_ui::relative(0.94))
                    .occlude()
                    .capture_any_mouse_down(Self::guard_presses(cx))
                    // Empty-query Home/End propagate out of `TextInput::home`/`end` so they jump the list
                    // even when `input::bind_keys` is active.
                    .on_action(cx.listener(move |this, _: &crate::input::Home, _, cx| {
                        let Some(p) = this.palette.as_mut() else { return };
                        if n > 0 {
                            p.selected = 0;
                            reveal_palette_item(p, &row_child_home, false);
                        }
                        cx.notify();
                    }))
                    .on_action(cx.listener(move |this, _: &crate::input::End, _, cx| {
                        let Some(p) = this.palette.as_mut() else { return };
                        if n > 0 {
                            let prev = p.selected.min(n - 1);
                            p.selected = n - 1;
                            reveal_palette_item(p, &row_child_end, p.selected > prev);
                        }
                        cx.notify();
                    }))
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
                        let m = ev.keystroke.modifiers;
                        match ev.keystroke.key.as_str() {
                            "down" | "up" | "pagedown" | "pageup" if !m.alt && !m.platform => {
                                if n == 0 {
                                    return;
                                }
                                let max_ix = n - 1;
                                let cur = p.selected.min(max_ix);
                                p.selected = match ev.keystroke.key.as_str() {
                                    "down" => (cur + 1).min(max_ix),
                                    "up" => cur.saturating_sub(1),
                                    "pagedown" => cur.saturating_add(PALETTE_PAGE_STEP).min(max_ix),
                                    _ => cur.saturating_sub(PALETTE_PAGE_STEP),
                                };
                                reveal_palette_item(p, &row_child, p.selected > cur);
                            }
                            "home" | "end"
                                if (m.control && !m.alt && !m.shift && !m.platform)
                                    || (p.query.is_empty()
                                        && !m.control
                                        && !m.alt
                                        && !m.shift
                                        && !m.platform) =>
                            {
                                if n > 0 {
                                    let prev = p.selected.min(n - 1);
                                    p.selected = if ev.keystroke.key == "home" { 0 } else { n - 1 };
                                    reveal_palette_item(p, &row_child, p.selected > prev);
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
mod tests;
