//! Overlays drawn above the page: command palette, dialogs, the image inspector and toasts.

use std::rc::Rc;

#[allow(unused_imports)]
use crate::api::{Label, RunExt, SplitsExt};
use gpui::Focusable;
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, MouseButton,
    MouseDownEvent, MouseMoveEvent, ParentElement, ScrollWheelEvent, SharedString,
    StatefulInteractiveElement, Styled, Window, canvas, div, px,
};

use crate::api::RunStatus;
use crate::app::{ARCHS, Action, BATCHES, DEVICES, FoloApp, IMGSZ, Route, act};

/// The optimizer segments' labels (the helper under them repeats the chosen one).
const OPT_LABELS: [&str; 4] = ["Auto", "SGD", "AdamW", "MuSGD"];
use crate::controls::Btn;
use crate::store::{ImgKind, Load};
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, image_rect, paint_image_with_boxes, to_image};

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

/// A reference estimate's duration, as precise as it is: whole minutes ("≈ 10 min", "≈ 1 h 25 min"),
/// seconds only under a minute.
fn rough_duration(secs: f64) -> String {
    let m = (secs / 60.).round() as u64;
    match m {
        0 => format!("{:.0} s", secs.max(1.)),
        1..60 => format!("{m} min"),
        _ => format!("{} h {} min", m / 60, m % 60),
    }
}

/// An estimate's range, as rough as its ends: whole minutes under an hour ("7–15 min"), else
/// hours to a tenth ("1.2–2.0 h").
fn rough_range(lo: f64, hi: f64) -> String {
    let (lo, hi) = (lo.min(hi).max(0.), lo.max(hi));
    if hi < 60. {
        format!("{:.0}–{:.0} s", lo.max(1.), hi.max(1.))
    } else if hi < 3600. {
        format!("{}–{} min", (lo / 60.).round().max(1.) as u64, (hi / 60.).round().max(1.) as u64)
    } else {
        format!("{:.1}–{:.1} h", lo / 3600., hi / 3600.)
    }
}

/// Rough training cost of a configuration on this machine: (seconds, VRAM GiB).
pub fn estimate(arch: usize, epochs: f32, batch: u32, imgsz: u32, images: u32) -> (f64, f32) {
    const GFLOPS: [f32; 5] = [6.6, 21.5, 48.0, 104.0, 165.0];
    const ACT_MB: [f32; 5] = [45., 80., 140., 210., 300.];
    let scale = (imgsz as f32 / 640.).powi(2);
    let ips = 12_900. / (GFLOPS[arch] * scale);
    let secs = epochs as f64 * images as f64 / ips as f64 * 1.08;
    let vram = 0.8 + batch as f32 * scale * ACT_MB[arch] / 1024.;
    (secs, vram)
}

/// A dialog's footer: its actions on the raised band under a hairline.
fn dialog_footer(t: &Theme) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(8.))
        .px(px(20.))
        .py(px(14.))
        .border_t_1()
        .border_color(t.line_subtle)
        .bg(t.bg_raised)
        // The frame's 12 px corners, less its 1 px edge: a square fill would show past them.
        .rounded_b(px(radius::XL - 1.))
}

/// A refusal inside a dialog: the server's message, and its hint (what to change) under it.
fn error_note(t: &Theme, message: String, hint: Option<String>) -> Div {
    div()
        .flex()
        .items_start()
        .gap(px(8.))
        .p(px(10.))
        .rounded(px(radius::MD))
        .bg(t.danger_muted)
        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(2.))
                .min_w_0()
                .child(text(ty::SM, t.danger_text).whitespace_normal().child(message))
                .when_some(hint, |d, h| d.child(text(ty::SM, t.text_secondary).whitespace_normal().child(h))),
        )
}

impl FoloApp {
    pub fn overlays(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out = Vec::new();
        if self.inspector.is_some() {
            out.push(self.inspector_view(window, cx));
        }
        if self.new_run.is_some() {
            out.push(self.new_run_dialog(window, cx));
        }
        if self.synth.is_some() {
            out.push(self.synth_dialog(window, cx));
        }
        if let Some(id) = self.confirm_stop.clone() {
            out.push(self.stop_dialog(&id, window, cx));
        }
        if self.cloud.connect.is_some() {
            out.push(self.connect_dialog(window, cx));
        }
        if self.cloud.disconnect.is_some() {
            out.push(self.disconnect_dialog(window, cx));
        }
        if self.cloud.terminate.is_some() {
            out.push(self.terminate_dialog(window, cx));
        }
        if self.palette.is_some() {
            out.push(self.palette_view(window, cx));
        }
        if !self.toasts.is_empty() {
            out.push(self.toasts_view(window, cx));
        }
        out
    }

    fn scrim(
        &self,
        t: &Theme,
        id: &'static str,
        cx: &mut Context<Self>,
        on_click: Action,
    ) -> gpui::Stateful<Div> {
        div().id(id).absolute().top_0().left_0().size_full().bg(t.bg_scrim).occlude().on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, w, cx| {
                on_click(this, w, cx);
                cx.notify();
            }),
        )
    }

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

    fn palette_view(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
            .child(self.scrim(&t, "palette-scrim", cx, act(|this, w, cx| this.close_palette(w, cx))))
            .child(crate::ui::appear(panel_el, "palette-in"))
            .into_any_element()
    }

    fn dialog_frame(
        &self,
        t: &Theme,
        id: &'static str,
        width: f32,
        title: &str,
        subtitle: Option<String>,
        body: impl IntoElement,
        footer: impl IntoElement,
        window: &Window,
        cx: &mut Context<Self>,
        on_close: Action,
    ) -> AnyElement {
        let close_btn =
            self.icon_button("dialog-close", "icons/x.svg", "Close (Esc)", window, cx, on_close.clone());
        // From the last layout, as the run log's follow state is.
        // Scrolled: a hairline under the fixed header, so rows passing beneath it read as scrolled
        // away, not cut.
        let scrolled = self.dialog_scroll.offset().y < px(-1.);
        crate::ui::appear(
            div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .child(self.scrim(t, "dialog-scrim", cx, on_close))
                .child(
                    div()
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            div()
                                .id(id)
                                .w(px(width))
                                .max_h(gpui::relative(0.92))
                                .occlude()
                                .flex()
                                .flex_col()
                                .rounded(px(radius::XL))
                                .bg(t.bg_overlay)
                                .border_1()
                                .border_color(t.line_default)
                                .shadow(t.shadow_lg())
                                .overflow_hidden()
                                .child(
                                    div()
                                        .flex()
                                        .items_start()
                                        .justify_between()
                                        .px(px(20.))
                                        .pt(px(18.))
                                        .pb(px(14.))
                                        .child(
                                            div()
                                                .flex()
                                                .flex_col()
                                                .gap(px(4.))
                                                .child(
                                                    text(ty::XL, t.text_primary)
                                                        .font_weight(FontWeight::SEMIBOLD)
                                                        .child(title.to_string()),
                                                )
                                                .when_some(subtitle, |d, s| {
                                                    d.child(text(ty::SM, t.text_secondary).child(s))
                                                }),
                                        )
                                        .child(close_btn),
                                )
                                .child(
                                    div()
                                        .relative()
                                        .flex_1()
                                        .min_h_0()
                                        .flex()
                                        .flex_col()
                                        .border_t_1()
                                        .border_color(if scrolled {
                                            t.line_subtle
                                        } else {
                                            gpui::transparent_black()
                                        })
                                        .child(
                                            div()
                                                .id("dialog-body")
                                                .flex_1()
                                                .min_h_0()
                                                .overflow_y_scroll()
                                                .track_scroll(&self.dialog_scroll)
                                                .child(body),
                                        )
                                        // Fades only where more of the form lies (a row half under
                                        // the header reads as scrolled away; the ends show undimmed).
                                        .children(self.scroll_chrome(
                                            self.dialog_scroll.clone(),
                                            t.bg_overlay,
                                            0.,
                                            0.,
                                            cx,
                                        )),
                                )
                                .child(footer),
                        ),
                ),
            &format!("{id}-in"),
        )
    }

    /// New run's Run on (servers with `cloud`): this machine or a cloud connection; for a
    /// connection, its GPU offers and Stop after, the most the run can cost said beside it. With no
    /// connection, the way to make one.
    fn nr_compute_rows(
        &mut self,
        cloud: Option<&crate::api::CloudConnection>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        use crate::cloud::ComputeField;
        let t = theme(cx);
        let Some(f) = self.new_run.as_ref() else { return div().into_any_element() };
        let (offer_id, hours, hours_input, hours_bad, compute_error) =
            (f.offer.clone(), f.max_hours, f.hours_input.clone(), f.hours_invalid, f.compute_error.clone());
        let error_on =
            |c: ComputeField| compute_error.as_ref().filter(|(k, _)| *k == c).map(|(_, m)| m.clone());
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        let danger = |s: String| text(ty::SM, t.danger_text).whitespace_normal().child(s);
        let quiet = |s: String| text(ty::SM, t.text_tertiary).whitespace_normal().child(s);
        let connections = self.cloud.connections.ready().cloned().unwrap_or_default();
        let here = self
            .store
            .system
            .ready()
            .and_then(|s| s.gpus.first())
            .map_or("this PC".to_string(), |g| crate::ui::short_gpu(&g.name));
        let mut options = vec![(SharedString::from("This machine"), Some(SharedString::from(here)))];
        let mut disabled: Vec<Option<SharedString>> = vec![None];
        for c in &connections {
            let failed = c.status == crate::api::CloudConnectionStatus::Error;
            options.push((
                c.name.clone().into(),
                Some(if failed {
                    "check failed".into()
                } else {
                    c.account.clone().unwrap_or_default().into()
                }),
            ));
            disabled.push(
                failed.then(|| c.error.clone().unwrap_or_else(|| "Its last check failed".into()).into()),
            );
        }
        let selected = cloud.and_then(|c| connections.iter().position(|x| x.id == c.id)).map_or(0, |i| i + 1);
        let ids: Vec<String> = connections.iter().map(|c| c.id.clone()).collect();
        let run_on = self.select_with(
            "nr-run-on",
            cloud.map_or("This machine".into(), |c| c.name.clone().into()),
            options,
            selected,
            &disabled,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                if let Some(f) = this.new_run.as_mut() {
                    f.run_on = i.checked_sub(1).and_then(|i| ids.get(i).cloned());
                    f.offer = None;
                    f.want_offer = None;
                    f.compute_error = None;
                }
                this.sync_compute(cx);
            }),
        );
        let provisions = cloud.is_some_and(|c| {
            self.cloud
                .providers
                .ready()
                .is_some_and(|ps| ps.iter().any(|p| p.id == c.provider && p.provisions))
        });
        let run_on_help = match cloud {
            None => "Trains here, on the device chosen under Model".to_string(),
            Some(c) if provisions => {
                format!("A GPU rented from {} for this run, by the hour", self.provider_name(&c.provider))
            }
            Some(_) => "That server trains it, in its own queue".to_string(),
        };
        let left = div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(label("Run on"))
            .child(run_on)
            .child(match error_on(ComputeField::RunOn) {
                Some(e) => danger(e),
                None => quiet(run_on_help),
            });
        let Some(conn) = cloud else {
            // No connection yet: the way to one, in Settings › Cloud (the form waits as a draft).
            let right = self.cloud.connections.ready().is_some_and(|l| l.is_empty()).then(|| {
                let link = div()
                    .id("nr-cloud-settings")
                    .mt(px(24.))
                    .px(px(4.))
                    .mx(px(-4.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(4.))
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .whitespace_normal()
                                    .child("Connect a cloud in Settings to train on a rented GPU."),
                            )
                            .child(icon("icons/chevron-right.svg", 12., t.text_secondary)),
                    );
                self.bind(
                    link,
                    "nr-cloud-settings",
                    radius::SM,
                    window,
                    cx,
                    act(|this, w, cx| {
                        this.close_new_run(w, cx);
                        this.navigate(Route::Settings, w, cx);
                    }),
                )
            });
            return div()
                .flex()
                .gap(px(16.))
                .child(left)
                .child(div().flex_1().min_w_0().children(right))
                .into_any_element();
        };
        // The GPU: the connection's offers, each with its price; none in stock can't be picked.
        let offers = self.cloud.offers.get(&conn.id).cloned().unwrap_or(Load::Loading);
        let offer =
            offers.ready().and_then(|l| l.offers.iter().find(|o| Some(&o.id) == offer_id.as_ref()).cloned());
        let gpu: AnyElement = match &offers {
            Load::Ready(list) => {
                let opts = list
                    .offers
                    .iter()
                    .map(|o| {
                        let price = fmt_price(o.price_per_hour);
                        let detail = match o.availability {
                            crate::api::CloudAvailability::None => format!("none available now · {price}"),
                            crate::api::CloudAvailability::Low => format!("low stock · {price}"),
                            _ => price,
                        };
                        (SharedString::from(offer_line(o)), Some(SharedString::from(detail)))
                    })
                    .collect();
                let off: Vec<Option<SharedString>> = list
                    .offers
                    .iter()
                    .map(|o| {
                        (o.availability == crate::api::CloudAvailability::None)
                            .then(|| "None available now: launching would be refused".into())
                    })
                    .collect();
                let picked =
                    offer.as_ref().and_then(|o| list.offers.iter().position(|x| x.id == o.id)).unwrap_or(0);
                let ids: Vec<String> = list.offers.iter().map(|o| o.id.clone()).collect();
                self.select_with(
                    "nr-gpu",
                    offer.as_ref().map_or("Choose a GPU".into(), |o| {
                        format!("{} · {}", offer_line(o), fmt_price(o.price_per_hour)).into()
                    }),
                    opts,
                    picked,
                    &off,
                    window,
                    cx,
                    Rc::new(move |this, i, _, _| {
                        if let Some(f) = this.new_run.as_mut() {
                            f.offer = ids.get(i).cloned();
                            f.compute_error.take_if(|(c, _)| *c == ComputeField::Gpu);
                        }
                    }),
                )
            }
            Load::Failed(e) => div()
                .h(px(crate::theme::size::CONTROL_LG))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(8.))
                .child(text(ty::SM, t.danger_text).flex_1().min_w_0().ellipsis().child(e.message.clone()))
                .child({
                    let id = conn.id.clone();
                    self.button(
                        "nr-offers-retry",
                        "Retry",
                        Btn::Secondary,
                        Some("icons/refresh-cw.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.ensure_offers(&id, true, cx)),
                    )
                })
                .into_any_element(),
            // Shaped like the select it becomes, inert.
            _ => div()
                .h(px(crate::theme::size::CONTROL_LG))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_inset)
                .border_1()
                .border_color(t.line_default)
                .child(text(ty::BASE, t.text_tertiary).child("Loading offers…"))
                .into_any_element(),
        };
        let gpu_help = match (error_on(ComputeField::Gpu), &offer) {
            (Some(e), _) => danger(e),
            (None, Some(o)) if o.price_per_hour.is_none() => quiet("Not billed: a machine of yours".into()),
            (None, Some(o)) if o.availability == crate::api::CloudAvailability::Low => {
                quiet("Low stock: the provider may refuse to start it".into())
            }
            _ => quiet("Billed by the hour while its instance exists".into()),
        };
        let right = div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(label("GPU"))
            .child(gpu)
            .child(gpu_help);
        // Stop after: the run's time limit, and so the most it can cost.
        let fh = hours_input.read(cx).focus_handle(cx);
        let ring = self.ring(&fh, window);
        let focused = fh.is_focused(window);
        let bad = hours_bad || error_on(ComputeField::StopAfter).is_some();
        let hours_box = div()
            .relative()
            .w(px(84.))
            .h(px(crate::theme::size::CONTROL_MD))
            .px(px(8.))
            .flex()
            .flex_none()
            .items_center()
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if bad {
                t.danger_solid
            } else if focused {
                t.accent_line
            } else {
                t.line_control
            })
            .text_size(px(ty::SM.size))
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .child(hours_input);
        let price = offer.as_ref().and_then(|o| o.price_per_hour);
        let limit = match (provisions, price) {
            (true, Some(p)) => format!(
                "The run is stopped and the instance terminated after this long at the latest. It cannot cost more than {}.",
                fmt_cost(p * hours as f64)
            ),
            (true, None) => {
                "The run is stopped and the instance terminated after this long at the latest.".to_string()
            }
            (false, _) => "The run is stopped after this long at the latest.".to_string(),
        };
        let stop_after = div().flex().flex_col().gap(px(8.)).child(label("Stop after")).child(
            div()
                .flex()
                .items_center()
                .gap(px(12.))
                .child(
                    div()
                        .flex()
                        .flex_none()
                        .items_center()
                        .gap(px(6.))
                        .child(hours_box)
                        .child(text(ty::SM, t.text_secondary).child("hours")),
                )
                .child(match error_on(ComputeField::StopAfter) {
                    Some(e) => danger(e).flex_1().min_w_0(),
                    None if hours_bad => danger("Enter hours from 0.25 to 720".into()).flex_1().min_w_0(),
                    None => quiet(limit).flex_1().min_w_0(),
                }),
        );
        div()
            .flex()
            .flex_col()
            .gap(px(14.))
            .child(div().flex().gap(px(16.)).child(left).child(right))
            .child(stop_after)
            .into_any_element()
    }

    fn new_run_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        // The compute choice stays whole: offers loading, one preselected, Stop after following.
        self.sync_compute(cx);
        let cloud_on = self.cloud_on();
        let cloud = self
            .new_run
            .as_ref()
            .and_then(|f| f.run_on.clone())
            .filter(|_| cloud_on)
            .and_then(|id| self.connection(&id).cloned());
        let cloud_offer = cloud.as_ref().and_then(|c| {
            let id = self.new_run.as_ref()?.offer.clone()?;
            self.cloud.offers.get(&c.id)?.ready()?.offers.iter().find(|o| o.id == id).cloned()
        });
        let compute_rows = cloud_on.then(|| self.nr_compute_rows(cloud.as_ref(), window, cx));
        let datasets = self.store.datasets.ready().cloned().unwrap_or_default();
        let f = self.new_run.as_ref().unwrap();
        let (ds_i, arch, epochs, batch, imgsz, device, opt, lr0, mosaic, mixup, fliplr) = (
            f.dataset,
            f.arch,
            f.epochs,
            f.batch,
            f.imgsz,
            f.device,
            f.optimizer,
            f.lr0,
            f.mosaic,
            f.mixup,
            f.fliplr,
        );
        let name_input = f.name.clone();
        let error = f.error.clone();
        let submitting = f.submitting;
        let ds = datasets.get(ds_i).cloned();
        // Advanced: what differs from a fresh form's defaults, so a collapsed group still says
        // what it will do. Open while one of its fields needs fixing.
        let adv_summary = {
            let mut changed: Vec<String> = Vec::new();
            if f.e2e {
                changed.push("end-to-end".into());
            }
            if opt != 0 {
                changed.push(format!("{} at {}", OPT_LABELS[opt], crate::ui::fmt_lr(lr0 as f64)));
            }
            if f.amp {
                changed.push("mixed precision".into());
            }
            let aug = mosaic != 1.0 || mixup != 0.0 || fliplr != 0.5 || f.scale != 0.5;
            if aug {
                changed.push("custom augmentation".into());
            }
            if changed.is_empty() { "Defaults".to_string() } else { changed.join(" · ") }
        };
        // An invalid rate (only with a chosen optimizer: Auto ignores the field) keeps it open.
        let adv_open = f.advanced || (f.lr_invalid && opt != 0);
        // A Mixed precision switch that held focus when it became blocked (the device changed by
        // key, or the server's answer arrived) has no focus stop any more: focus moves to the
        // control before it, so keys keep working. Before the controls are built, so that one
        // draws its ring this frame.
        // A cloud run's instance checks mixed precision when it boots: nothing here gates it.
        let amp_why = if cloud.is_some() { None } else { self.amp_blocked(DEVICES[device]) };
        if amp_why.is_some() && self.focus.borrow().get("nr-amp").is_some_and(|f| f.is_focused(window)) {
            // The control before it in Schedule: the learning rate, or the optimizer when Auto
            // sets the rate (the field is read-only then).
            let prev = if opt == 0 { self.fh("nr-opt", cx) } else { f.lr_input.read(cx).focus_handle(cx) };
            window.focus(&prev);
        }
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        let field = |name: &str, control: AnyElement, help: Option<String>| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .child(label(name))
                .child(control)
                .when_some(help, |d, h| d.child(text(ty::SM, t.text_tertiary).tnum().child(h)))
        };
        let name_fh = name_input.read(cx).focus_handle(cx);
        let name_ring = self.ring(&name_fh, window);
        let name_focused = name_fh.is_focused(window);
        let name_box = div()
            .relative()
            .h(px(32.))
            .px(px(10.))
            .flex()
            .items_center()
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if name_focused { t.accent_line } else { t.line_control })
            .text_size(px(ty::BASE.size))
            .when(name_ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .child(name_input)
            .into_any_element();
        let ds_select = self.select(
            "nr-dataset",
            ds.as_ref().map_or("No datasets".into(), |d| SharedString::from(d.name.clone())),
            datasets
                .iter()
                .map(|d| {
                    (
                        SharedString::from(d.name.clone()),
                        Some(SharedString::from(format!("{} images", int(d.splits.total())))),
                    )
                })
                .collect(),
            ds_i,
            window,
            cx,
            Rc::new(|this, i, _, _| {
                if let Some(f) = this.new_run.as_mut() {
                    f.dataset = i;
                }
            }),
        );
        let seg = |this: &mut Self,
                   key: &str,
                   opts: &[&str],
                   sel: usize,
                   window: &mut Window,
                   cx: &mut Context<Self>,
                   set: fn(&mut crate::app::NewRunForm, usize)| {
            this.segmented_fill(
                key,
                opts,
                sel,
                window,
                cx,
                Rc::new(move |this, i, _, _| {
                    if let Some(f) = this.new_run.as_mut() {
                        set(f, i);
                    }
                }),
            )
            .into_any_element()
        };
        // Geist Mono, so "l" can't read as "1" or "|".
        let arch_seg = div()
            .w_full()
            .font_family(crate::theme::MONO)
            .child(seg(self, "nr-arch", &["n", "s", "m", "l", "x"], arch, window, cx, |f, i| f.arch = i))
            .into_any_element();
        let batch_labels: Vec<String> = BATCHES.iter().map(|b| b.to_string()).collect();
        let batch_refs: Vec<&str> = batch_labels.iter().map(String::as_str).collect();
        let batch_seg = seg(self, "nr-batch", &batch_refs, batch, window, cx, |f, i| f.batch = i);
        let size_labels: Vec<String> = IMGSZ.iter().map(|b| b.to_string()).collect();
        let size_refs: Vec<&str> = size_labels.iter().map(String::as_str).collect();
        let size_seg = seg(self, "nr-imgsz", &size_refs, imgsz, window, cx, |f, i| f.imgsz = i);
        let device_seg =
            seg(self, "nr-device", &["Auto", "GPU 0", "CPU"], device, window, cx, |f, i| f.device = i);
        let opt_seg = seg(self, "nr-opt", &OPT_LABELS, opt, window, cx, |f, i| f.optimizer = i);
        // A typed value outside a slider's preset range widens it (rather than pinning the thumb at
        // an end, where one arrow press would jump to the end value).
        let epochs_slider = self
            .slider("nr-epochs", epochs, 10f32.min(epochs), 300f32.max(epochs), 5., window, cx)
            .into_any_element();
        let lr_slider = self
            .slider("nr-lr0", lr0, 0.001f32.min(lr0), 0.05f32.max(lr0), 0.001, window, cx)
            .into_any_element();
        let mosaic_slider = self.slider("nr-mosaic", mosaic, 0., 1., 0.05, window, cx).into_any_element();
        let mixup_slider = self.slider("nr-mixup", mixup, 0., 0.5, 0.05, window, cx).into_any_element();
        let scale = self.new_run.as_ref().map_or(0.5, |f| f.scale);
        let scale_slider = self.slider("nr-scale", scale, 0., 0.9, 0.05, window, cx).into_any_element();
        let note = self.new_run.as_ref().and_then(|f| f.note.clone());
        let flip_slider = self.slider("nr-fliplr", fliplr, 0., 1., 0.05, window, cx).into_any_element();
        let close_mosaic = self.new_run.as_ref().map_or(10., |f| f.close_mosaic);
        let amp = self.new_run.as_ref().is_some_and(|f| f.amp);
        let close_slider = self
            .slider("nr-close", close_mosaic, 0., 30f32.max(close_mosaic), 1., window, cx)
            .into_any_element();
        const PARAMS: [&str; 5] = [
            "2.63 M params · 6.6 GFLOPs",
            "9.42 M params · 21.5 GFLOPs",
            "20.1 M params · 48.0 GFLOPs",
            "43.7 M params · 104 GFLOPs",
            "68.2 M params · 165 GFLOPs",
        ];
        // A slider row as the Playground rail writes one: name · slider · value, on one line.
        let valued = |name: &str, value: String, control: AnyElement| {
            div()
                .flex()
                .items_center()
                .gap(px(10.))
                .child(div().w(px(100.)).flex_none().child(label(name)))
                .child(div().flex_1().min_w_0().child(control))
                .child(
                    div()
                        .w(px(44.))
                        .flex_none()
                        .flex()
                        .justify_end()
                        .child(text(ty::SM, t.text_primary).tnum().child(value)),
                )
        };
        // Short windows (the 620 px minimum): tighter rhythm and no restating helpers in the Model
        // rows, so the Schedule's controls are in view without scrolling.
        let short = window.viewport_size().height < px(700.);
        // What Auto resolves to for this dataset, batch and length (the engine's own rule, docs/MODEL.md §8.2:
        // MuSGD at 0.01 past 10,000 optimizer steps of a 64-image nominal batch, else AdamW at its fitted rate), so
        // the form can say which, and that the rate field is then unused. Written as every rate is (1.11×10⁻³).
        let auto_pick = ds.as_ref().filter(|_| opt == 0).map(|d| {
            let a = folo_nn::optim::auto_optimizer(
                d.splits.train as usize,
                BATCHES[batch] as usize,
                64,
                epochs.round().max(1.) as usize,
                d.classes.len(),
            );
            let kind = match a.kind {
                folo_nn::optim::OptimKind::Sgd => "SGD",
                folo_nn::optim::OptimKind::AdamW => "AdamW",
                folo_nn::optim::OptimKind::MuSgd => "MuSGD",
            };
            (kind, crate::ui::fmt_lr(a.lr0))
        });
        // The adapter behind "GPU 0", named as the other screens name it.
        let gpu_name =
            self.store.system.ready().and_then(|s| s.gpus.first()).map(|g| crate::ui::short_gpu(&g.name));
        // Section headings sit above a hairline and lead their group (stronger than field labels).
        let section = |title: &str| {
            div()
                .mt(px(if short { 0. } else { 6. }))
                .pt(px(if short { 8. } else { 14. }))
                .border_t_1()
                .border_color(t.line_subtle)
                // Two size steps above the 12 px field labels, not only a weight step.
                .child(
                    text(ty::MD, t.text_primary).font_weight(FontWeight::SEMIBOLD).child(title.to_string()),
                )
        };
        let (epochs_input, lr_input, epochs_bad, lr_bad) = {
            let f = self.new_run.as_ref().unwrap();
            (f.epochs_input.clone(), f.lr_input.clone(), f.epochs_invalid, f.lr_invalid && f.optimizer != 0)
        };
        let (close_input, close_bad) = {
            let f = self.new_run.as_ref().unwrap();
            (f.close_input.clone(), f.close_invalid)
        };
        let number_box = |this: &Self,
                          input: gpui::Entity<crate::input::TextInput>,
                          bad: bool,
                          window: &Window,
                          cx: &mut Context<Self>| {
            let fh = input.read(cx).focus_handle(cx);
            let ring = this.ring(&fh, window);
            let focused = fh.is_focused(window);
            div()
                .relative()
                .w(px(84.))
                // As tall as the segmented controls beside it, so the two columns' helpers align.
                .h(px(crate::theme::size::CONTROL_MD))
                .px(px(8.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if bad {
                    t.danger_solid
                } else if focused {
                    t.accent_line
                } else {
                    t.line_control
                })
                .font_family(crate::theme::SANS)
                .text_size(px(ty::SM.size))
                .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(input)
                .into_any_element()
        };
        let epochs_box = number_box(self, epochs_input, epochs_bad, window, cx);
        let lr_box = number_box(self, lr_input, lr_bad, window, cx);
        let close_box = number_box(self, close_input, close_bad, window, cx);
        let typed = |name: &str, field: AnyElement, control: AnyElement, bad: Option<&str>| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .child(label(name))
                // Slider and exact value on one row.
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(12.))
                        .child(div().flex_1().min_w_0().child(control))
                        .child(field),
                )
                .when_some(bad, |d, msg| d.child(text(ty::SM, t.danger_text).child(msg.to_string())))
        };
        // Short windows drop the helper rows; the two that explain a choice stay one hover or
        // focus away.
        let batch_help =
            short.then(|| self.help_icon("nr-help-batch", "Images per training step.", window, cx));
        let device_help = short.then(|| {
            self.help_icon(
                "nr-help-device",
                "Auto uses the GPU when one is available, else the CPU.",
                window,
                cx,
            )
        });
        let size_help = short.then(|| {
            self.help_icon("nr-help-size", "Pixels; images are letterboxed to a square.", window, cx)
        });
        let labelled = |name: &str, help: AnyElement, control: AnyElement| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                // As tall as a plain label row (the icon overhangs), so a field with help lines up with
                // its neighbour without.
                .child(div().h(px(16.)).flex().items_center().gap(px(4.)).child(label(name)).child(help))
                .child(control)
                .into_any_element()
        };
        // A cloud run trains on what its instance has: a fact, not a choice (`device` goes as auto).
        let device_fact = cloud.as_ref().map(|c| {
            let instance = self
                .cloud
                .providers
                .ready()
                .is_some_and(|ps| ps.iter().any(|p| p.id == c.provider && p.provisions));
            let fact = div()
                .h(px(crate::theme::size::CONTROL_MD))
                .px(px(8.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                // A value, not an input: an outline, no well (as Auto's learning rate).
                .border_1()
                .border_color(t.line_subtle)
                .child(text(ty::SM, t.text_secondary).child(if instance {
                    "The instance's GPU"
                } else {
                    "The server's device"
                }))
                .into_any_element();
            field("Device", fact, (!short).then(|| "The worker picks it (sent as auto)".to_string()))
                .into_any_element()
        });
        let body = div()
            .flex()
            .flex_col()
            .gap(px(if short { 8. } else { 14. }))
            .px(px(20.))
            // Room past the last control (the bottom fade shows only while more of the form lies
            // below, so at the end every control reads undimmed).
            .pb(px(36.))
            // Errors lead the form, where a failed submit is seen (the dialog scrolls to the top).
            // The server's hint says what to change (device_not_found lists the adapters there
            // are; gpu_out_of_memory names batch, imgsz and arch).
            .when_some(error.clone(), |d, e| d.child(error_note(&t, e.message, e.hint)))
            .child(
                div().flex().gap(px(16.)).child(div().flex_1().child(field("Name", name_box, None))).child(
                    div().flex_1().child({
                        let summary = ds.as_ref().map(|d| {
                            format!(
                                "{} classes · train {} · val {}",
                                d.classes.len(),
                                int(d.splits.train),
                                int(d.splits.val)
                            )
                        });
                        // The summary rides on the label row (no band under Name beside it,
                        // and a line saved toward keeping the Schedule above the footer).
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(8.))
                            .child(
                                div()
                                    .flex()
                                    .items_baseline()
                                    .justify_between()
                                    .gap(px(8.))
                                    .child(label("Dataset"))
                                    .children(
                                        summary.map(|s| {
                                            text(ty::SM, t.text_tertiary).tnum().ellipsis().child(s)
                                        }),
                                    ),
                            )
                            .child(ds_select)
                    }),
                ),
            )
            // Where it trains, before what (the Device below follows from it).
            .when_some(compute_rows, |d, rows| d.child(section("Compute")).child(rows))
            .child(section("Model"))
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(field(
                        "Architecture",
                        arch_seg,
                        // The letter's size named, so the five choices read without knowing FOLO.
                        Some(format!(
                            "{}, {} · {}",
                            ARCHS[arch].as_str(),
                            ["nano", "small", "medium", "large", "extra large"][arch],
                            PARAMS[arch]
                        )),
                    )))
                    .child(
                        div().flex_1().child(match size_help {
                            Some(h) => labelled("Image size", h, size_seg),
                            None => field(
                                "Image size",
                                size_seg,
                                Some("Pixels; images are letterboxed to a square".into()),
                            )
                            .into_any_element(),
                        }),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(match batch_help {
                        Some(h) => labelled("Batch size", h, batch_seg),
                        None => {
                            field("Batch size", batch_seg, Some("Images per step".into())).into_any_element()
                        }
                    }))
                    .child(
                        div().flex_1().child(match (device_fact, device_help) {
                            (Some(fact), _) => fact,
                            (None, Some(h)) => labelled("Device", h, device_seg),
                            (None, None) => field(
                                "Device",
                                device_seg,
                                Some(match (DEVICES[device], &gpu_name) {
                                    ("auto", Some(g)) => format!("{g} when available, else CPU"),
                                    ("auto", None) => "GPU when available, else CPU".into(),
                                    ("cpu", _) => "Trains on the CPU (much slower)".into(),
                                    (_, Some(g)) => format!("GPU 0: {g}"),
                                    (d, None) => d.to_string(),
                                }),
                            )
                            .into_any_element(),
                        }),
                    ),
            )
            // Basic first: how long it trains. Everything a first run can leave alone sits under
            // Advanced, collapsed, with a summary of what differs from the defaults.
            .child(section("Schedule"))
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    // A help line on both halves, so the pair ends level (what an epoch is here, as
                    // the right half says which epochs drop mosaic).
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap(px(8.))
                            .child(typed(
                                "Epochs",
                                epochs_box,
                                epochs_slider,
                                epochs_bad.then_some("Enter a whole number from 1 to 10,000"),
                            ))
                            .child(text(ty::SM, t.text_tertiary).child(match ds.as_ref() {
                                Some(d) => format!("One epoch = one pass over {} training images", int(d.splits.train)),
                                None => "One epoch = one pass over the training images".to_string(),
                            })),
                    )
                    .child(
                        div().flex_1().child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(8.))
                                .child(typed(
                                    "Final epochs without mosaic",
                                    close_box,
                                    close_slider,
                                    close_bad.then_some("Enter a whole number up to the epoch count"),
                                ))
                                // Which epochs those are (the label already says what they do).
                                .child(text(ty::SM, t.text_tertiary).child({
                                    // As the engine runs it: capped at half the run.
                                    let last = epochs.round().max(1.) as u32;
                                    let asked = close_mosaic.round() as u32;
                                    let n = crate::api::capped_close_mosaic(asked, last);
                                    let capped = if n < asked { " (capped at half the run)" } else { "" };
                                    match n {
                                        // Mosaic off (Advanced): every epoch is on whole images.
                                        _ if mosaic <= 0. => "Mosaic is off: every epoch trains on whole images".to_string(),
                                        0 if asked == 0 => "Off: mosaic all the way through".to_string(),
                                        0 => "Too short a run: mosaic all the way through".to_string(),
                                        1 => format!("Epoch {last} trains on whole images{capped}"),
                                        _ => format!("Epochs {}–{last} train on whole images{capped}", last - n + 1),
                                    }
                                })),
                        ),
                    ),
            )
            .child({
                let open = adv_open;
                let hdr = div()
                    .id("nr-adv")
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .py(px(4.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .pt(px(14.))
                    .cursor_pointer()
                    .child(icon(
                        if open { "icons/chevron-down.svg" } else { "icons/chevron-right.svg" },
                        14.,
                        t.text_secondary,
                    ))
                    .child(text(ty::MD, t.text_primary).font_weight(FontWeight::SEMIBOLD).child("Advanced"))
                    .child(text(ty::SM, t.text_tertiary).min_w_0().ellipsis().child(adv_summary.clone()));
                self.bind(
                    hdr,
                    "nr-adv",
                    radius::SM,
                    window,
                    cx,
                    act(|this, _, cx| {
                        // A field in the group that needs fixing keeps it open.
                        if let Some(f) = this.new_run.as_mut()
                            && !(f.lr_invalid && f.optimizer != 0)
                        {
                            f.advanced = !f.advanced;
                            cx.notify();
                        }
                    }),
                )
            })
            .when(adv_open, |d| {
                d.child(
                    // Advanced's body: indented under the header's chevron, a rule down its left, so
                    // what the disclosure encloses is visible; its parts are sub-heads, not sections.
                    div()
                        .ml(px(6.))
                        .pl(px(18.))
                        .border_l_1()
                        .border_color(t.line_subtle)
                        .flex()
                        .flex_col()
                        .gap(px(if short { 8. } else { 14. }))
                // The head: one-to-many with NMS (the default), or the end-to-end head that predicts
                // without it. It leads Advanced, as the model property among them.
                .child({
                    let e2e = self.new_run.as_ref().is_some_and(|f| f.e2e);
                    let switch = self
                        .switch(
                            "nr-e2e",
                            e2e,
                            window,
                            cx,
                            act(|this, _, cx| {
                                if let Some(f) = this.new_run.as_mut() {
                                    f.e2e = !f.e2e;
                                    cx.notify();
                                }
                            }),
                        )
                        .into_any_element();
                    div()
                        // The left column's width (the 16 px gap halved off), so the switch sits by its
                        // label, not across a two-column dialog.
                        .w(gpui::relative(0.5))
                        .pr(px(8.))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap(px(16.))
                        .child(
                            // The label toggles too (the switch keeps the one Tab stop).
                            div()
                                .id("nr-e2e-label")
                                .cursor_pointer()
                                .on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(f) = this.new_run.as_mut() {
                                        f.e2e = !f.e2e;
                                        cx.notify();
                                    }
                                }))
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .child(label("End-to-end (no NMS)"))
                                .child(
                                    text(ty::SM, t.text_tertiary)
                                        .whitespace_normal()
                                        .child("Predicts boxes directly, with no NMS step"),
                                ),
                        )
                        .child(switch)
                })
                .child(
                    div()
                        .flex()
                        .gap(px(16.))
                        .child(div().flex_1().child(field(
                            "Optimizer",
                            opt_seg,
                            Some(match (opt, &auto_pick) {
                                (0, Some((kind, _))) => {
                                    format!("Auto: {kind} for this run, at its own learning rate")
                                }
                                (0, None) => "Auto: MuSGD for long runs, else AdamW".to_string(),
                                (1, _) => "SGD with Nesterov momentum 0.937".to_string(),
                                _ => "AdamW with weight decay".to_string(),
                            }),
                        )))
                        .child(
                            div()
                                .flex_1()
                                .flex()
                                .flex_col()
                                .gap(px(8.))
                                // Under Auto the engine sets the rate itself: the field shows that rate,
                                // read-only, instead of an editable value it would ignore.
                                .child(if opt == 0 {
                                    div()
                                        .flex()
                                        .flex_col()
                                        .gap(px(8.))
                                        .child(label("Initial learning rate"))
                                        .child(
                                            div()
                                                .h(px(crate::theme::size::CONTROL_MD))
                                                .px(px(8.))
                                                .flex()
                                                .items_center()
                                                .rounded(px(radius::MD))
                                                // A value, not an input: an outline, no well.
                                                .border_1()
                                                .border_color(t.line_subtle)
                                                .child(crate::ui::sci(
                                                    ty::SM,
                                                    t.text_tertiary,
                                                    match &auto_pick {
                                                        Some((_, lr)) => format!("Auto · {lr}"),
                                                        None => "Auto".to_string(),
                                                    },
                                                )),
                                        )
                                        .into_any_element()
                                } else {
                                    typed(
                                        "Initial learning rate",
                                        lr_box,
                                        lr_slider,
                                        lr_bad
                                            .then_some("Enter a number above 0 and below 10, e.g. 0.01 or 1e-3"),
                                    )
                                    .into_any_element()
                                })
                                // The schedule (short windows keep the Schedule above the footer).
                                .when(!short && !lr_bad, |d| {
                                    d.child(text(ty::SM, t.text_tertiary).child(if opt == 0 {
                                        "Set by Auto; pick SGD or AdamW to choose your own"
                                    } else {
                                        "3-epoch warmup, then linear decay to 1%"
                                    }))
                                }),
                        ),
                )
                // Mixed precision: f16 on tensor cores. The server says whether its GPU can
                // (`capabilities.amp`); when it can't, the switch is off and inert with its reason. An older
                // server can't say, so the requirement is stated and a run's failure tells.
                .child({
                    let blocked = amp_why.clone();
                    let off = blocked.is_some();
                    let amp_on = amp && !off;
                    // Blocked: a disabled switch (no hover, press or focus stop); the reason is beside it.
                    let switch = if off {
                        self.switch_disabled("nr-amp", cx).into_any_element()
                    } else {
                        self.switch(
                            "nr-amp",
                            amp_on,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                if let Some(f) = this.new_run.as_mut() {
                                    f.amp = !f.amp;
                                    cx.notify();
                                }
                            }),
                        )
                        .into_any_element()
                    };
                    div()
                        // The left column's width (the 16 px gap halved off), so the switch sits by its
                        // label, not across a two-column dialog.
                        .w(gpui::relative(0.5))
                        .pr(px(8.))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap(px(16.))
                        .child(
                            // The label toggles too when the switch can (it keeps the one Tab stop).
                            div()
                                .id("nr-amp-label")
                                .when(!off, |d| {
                                    d.cursor_pointer().on_click(cx.listener(|this, _, _, cx| {
                                        if let Some(f) = this.new_run.as_mut() {
                                            f.amp = !f.amp;
                                            cx.notify();
                                        }
                                    }))
                                })
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .child(label("Mixed precision"))
                                .child(
                                    // The server's reason is plain text (a backtick in it is literal); only
                                    // the stated need carries a code chip.
                                    match blocked {
                                        Some(why) => text(ty::SM, t.text_tertiary).whitespace_normal().child(why),
                                        None if cloud.is_some() => text(ty::SM, t.text_tertiary)
                                            .whitespace_normal()
                                            .child("Checked on the instance when it boots."),
                                        // The server confirmed it can; an older one can't say, so the need is stated.
                                        None if self.store.capabilities.is_some() => {
                                            text(ty::SM, t.text_tertiary).whitespace_normal().child(
                                                "f16 on this server's tensor cores: about 40% less GPU memory at 640 px. Not faster.",
                                            )
                                        }
                                        None => rich(
                                            &t,
                                            ty::SM,
                                            t.text_tertiary,
                                            "Trains in f16 on tensor cores; needs `folo serve --tensor-cores`",
                                        )
                                        .text_left(),
                                    },
                                ),
                        )
                        .child(switch)
                })
                .child(text(ty::BASE, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Augmentation"))
                .child(
                    div()
                        .flex()
                        .gap(px(16.))
                        .child(div().flex_1().child(valued("Mosaic", crate::ui::fmt_aug(mosaic), mosaic_slider)))
                        .child(div().flex_1().child(valued("Mixup", crate::ui::fmt_aug(mixup), mixup_slider))),
                )
                .child(
                    div()
                        .flex()
                        .gap(px(16.))
                        .child(div().flex_1().child(valued(
                            "Horizontal flip",
                            crate::ui::fmt_aug(fliplr),
                            flip_slider,
                        )))
                        .child(div().flex_1().child(valued(
                            "Scale jitter",
                            crate::ui::fmt_scale(scale),
                            scale_slider,
                        ))),
                )
                )
            })
            .when_some(note, |d, n| {
                d.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .p(px(10.))
                        .rounded(px(radius::MD))
                        .bg(t.info_muted)
                        .child(icon("icons/info.svg", 14., t.info_solid))
                        .child(text(ty::SM, t.text_primary).child(n)),
                )
            });
        self.ensure_estimate(cx);
        let images = ds.as_ref().map_or(0, |d| d.splits.train);
        let cpu = DEVICES[device] == "cpu" && cloud.is_none();
        let compute_blocked = self.compute_blocked();
        // What the footer can honestly say. With the server's estimator: its answer for this
        // config; while a changed config is out, the last answer marked as updating (if it was for
        // the same kind of device); nothing for an unparsed value or a rejected config. Servers
        // without one: the app's own reference figures.
        enum Est {
            Server(Box<crate::api::RunEstimate>, bool),
            Pending,
            Invalid,
            Rejected(String),
            Local,
        }
        // The last answer stands in only for the same kind of device (CPU, GPU, a cloud GPU).
        let same_kind = |e: &crate::api::RunEstimate| {
            (e.device == "cpu") == cpu && (e.device == "cloud") == cloud.is_some()
        };
        let est = if !self.store.features.iter().any(|f| f == "run_estimate") {
            Est::Local
        } else if epochs_bad
            || lr_bad
            || close_bad
            || self.new_run.as_ref().is_some_and(|f| f.hours_invalid && cloud.is_some())
        {
            Est::Invalid
        } else if compute_blocked.is_some() {
            Est::Pending
        } else {
            match &self.nr_estimate {
                Some((_, crate::store::Load::Ready(e))) => Est::Server(Box::new(e.clone()), false),
                Some((_, crate::store::Load::Failed(e))) if e.code == "not_supported" => Est::Local,
                Some((_, crate::store::Load::Failed(e))) => Est::Rejected(e.message.clone()),
                _ => match &self.nr_estimate_last {
                    Some(e) if same_kind(e) => Est::Server(Box::new(e.clone()), true),
                    _ => Est::Pending,
                },
            }
        };
        let (local_secs, local_vram) = estimate(arch, epochs, BATCHES[batch], IMGSZ[imgsz], images as u32);
        // Mixed precision trains in about 0.58x the memory (folo-n batch 16: 2022 vs 3474 MiB,
        // docs/BENCHMARKS.md) and no faster at 640 px (the epoch is bound elsewhere), so only the
        // memory moves.
        let f16 = amp && amp_why.is_none();
        let local_vram = if f16 { local_vram * 0.58 } else { local_vram };
        let gib = (1u64 << 30) as f32;
        // The device the estimate is for: the server's resolved one (Auto can land on the CPU).
        let on_cpu = match &est {
            Est::Server(e, _) => e.device == "cpu",
            _ => cpu,
        };
        let vram = match &est {
            Est::Server(e, _) => e.est_peak_vram_bytes.map(|b| b as f32 / gib),
            _ => (!on_cpu).then_some(local_vram),
        };
        let gpu = self.store.system.ready().and_then(|s| s.gpus.first().cloned());
        // 0 = not reported (wgpu can't on every platform): make no claim about fitting.
        // GiB, like the estimate and every size the app shows. A cloud run's is its offer's.
        let gpu_gb = match (&cloud, &cloud_offer) {
            (Some(_), Some(o)) => o.vram_bytes as f32 / gib,
            (Some(_), None) => 0.0,
            _ => gpu.as_ref().map_or(0.0, |g| g.memory_bytes as f32 / gib),
        };
        let on_gpu = match &cloud_offer {
            Some(o) if cloud.is_some() => format!("on the {}", o.gpu),
            _ => "on this GPU".to_string(),
        };
        // An answer being refreshed is never a warning: it's marked as updating instead.
        let fits = !matches!(est, Est::Server(_, false) | Est::Local)
            || on_cpu
            || gpu_gb <= 0.0
            || vram.is_none_or(|v| v <= gpu_gb * 0.97);
        let place = if on_cpu { "CPU" } else { "GPU" };
        let (headline, basis, note): (String, String, Option<String>) = match &est {
            Est::Invalid => ("No estimate while a value is invalid".into(), String::new(), None),
            Est::Pending => ("Estimating…".into(), String::new(), None),
            Est::Rejected(msg) => (
                "No estimate for this config".into(),
                "The server couldn't estimate it".into(),
                Some(msg.clone()),
            ),
            Est::Local if cpu => {
                ("CPU selected · no time estimate".into(), "Expect many times slower than a GPU".into(), None)
            }
            _ if !fits => (
                // No adapter name here: it can be long enough to reach the buttons.
                format!("≈ {:.0} GiB needed, {gpu_gb:.0} GiB {on_gpu}", vram.unwrap_or(0.)),
                if cloud.is_some() {
                    "Pick a larger GPU, or lower the batch or image size"
                } else {
                    "Lower the batch or image size"
                }
                .into(),
                None,
            ),
            Est::Server(e, updating) => {
                let range = rough_range(e.est_total_s_low, e.est_total_s_high);
                let headline = match vram {
                    Some(v) => {
                        format!("≈ {} ({range}) · ≈ {v:.1} GiB GPU memory", rough_duration(e.est_total_s))
                    }
                    None => format!("≈ {} ({range}) on the CPU", rough_duration(e.est_total_s)),
                };
                // What the figures are, said plainly: measured here or reference rates, and how far
                // to trust them (the server's own confidence).
                let confidence = match e.confidence {
                    crate::api::Confidence::High => "high",
                    crate::api::Confidence::Medium => "medium",
                    crate::api::Confidence::Low => "low",
                };
                let basis = if *updating {
                    "Updating for the new settings…".to_string()
                } else {
                    match e.source {
                        crate::api::ThroughputSource::Measured => {
                            format!("Measured on this {place} · {confidence} confidence")
                        }
                        // A cloud GPU is never measured here: the time is the reference machine's.
                        crate::api::ThroughputSource::Reference if e.device == "cloud" => {
                            format!("The reference machine's rates · {confidence} confidence")
                        }
                        crate::api::ThroughputSource::Reference => {
                            format!("Reference rates, not measured here · {confidence} confidence")
                        }
                    }
                };
                // How the server derived it, in its own sentence, on hover.
                (headline, basis, Some(e.basis.note.clone()))
            }
            _ => (
                format!("≈ {} · ≈ {local_vram:.1} GiB GPU memory", rough_duration(local_secs)),
                if f16 {
                    "Reference estimate in f16, not measured on this GPU".into()
                } else {
                    "Reference estimate, not measured on this GPU".into()
                },
                None,
            ),
        };
        let estimate_line = div()
            .id("nr-estimate")
            .when_some(note, |d, n| d.tooltip(crate::controls::tip(n, None)))
            .flex()
            .items_center()
            .gap(px(8.))
            .child(icon(
                if fits { "icons/timer.svg" } else { "icons/triangle-alert.svg" },
                14.,
                if fits { t.text_tertiary } else { t.warning_solid },
            ))
            .min_w_0()
            .child(
                // Time and memory lead (short enough for the minimum width); what they are sits below.
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .child(
                        text(ty::SM, if fits { t.text_secondary } else { t.warning_text })
                            .tnum()
                            .whitespace_nowrap()
                            .child(headline),
                    )
                    .when(!basis.is_empty(), |d| {
                        d.child(text(ty::SM, t.text_tertiary).whitespace_nowrap().child(basis))
                    }),
            );
        let cancel = self.button(
            "nr-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_new_run(w, cx)),
        );
        let busy_with = self
            .store
            .runs
            .ready()
            .and_then(|rs| rs.iter().find(|r| r.status.is_active() && r.status != RunStatus::Queued))
            .map(|r| r.name.clone())
            // A cloud run doesn't wait in this machine's queue.
            .filter(|_| cloud.is_none());
        let busy = busy_with.is_some();
        let (go, going) =
            if busy { ("Queue training", "Queuing…") } else { ("Start training", "Starting…") };
        // Queueing waits (a clock); starting plays.
        let go_icon = if busy { "icons/clock.svg" } else { "icons/play.svg" };
        let start = if submitting {
            self.button_disabled(
                "nr-start",
                going,
                Btn::Primary,
                Some(go_icon),
                "Creating the run…",
                window,
                cx,
            )
        } else if epochs_bad || lr_bad || close_bad {
            self.button_disabled(
                "nr-start",
                go,
                Btn::Primary,
                Some(go_icon),
                "Fix the highlighted fields first",
                window,
                cx,
            )
        } else if let Some(why) = compute_blocked {
            self.button_disabled("nr-start", go, Btn::Primary, Some(go_icon), why, window, cx)
        } else {
            self.button(
                "nr-start",
                go,
                Btn::Primary,
                Some(go_icon),
                window,
                cx,
                act(|this, w, cx| this.submit_new_run(w, cx)),
            )
        };
        // A cloud run's money, next to the button that spends it: where it runs, what training
        // should cost, and the most it can.
        let runs_on = cloud.as_ref().map(|c| {
            let provider = self.provider_name(&c.provider);
            cloud_offer.as_ref().map_or(provider.clone(), |o| format!("{provider} · {}", o.gpu))
        });
        let cost_facts = runs_on.clone().map(|on| {
            let price = cloud_offer.as_ref().and_then(|o| o.price_per_hour);
            let hours = self.new_run.as_ref().map_or(folo_api::DEFAULT_MAX_HOURS, |f| f.max_hours);
            let est_cost = match (&est, price) {
                (_, None) => "not billed".to_string(),
                (Est::Server(e, _), Some(_)) => {
                    e.cost.as_ref().map_or("—".to_string(), |c| fmt_cost_range(c.est_usd_low, c.est_usd_high))
                }
                _ => "…".to_string(),
            };
            let limit = match price {
                Some(p) => format!("{} at {}", fmt_cost(p * hours as f64), fmt_hours(hours)),
                None => fmt_hours(hours),
            };
            let fact = |name: &str, value: String| {
                div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap(px(6.))
                    .child(text(ty::SM, t.text_tertiary).whitespace_nowrap().child(name.to_string()))
                    .child(
                        text(ty::SM, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .tnum()
                            .whitespace_nowrap()
                            .child(value),
                    )
            };
            div()
                .flex()
                .items_center()
                .gap(px(24.))
                .pb(px(10.))
                .mb(px(2.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(fact("Runs on", on))
                .child(
                    fact("Est. cost", est_cost).child(
                        // Its 28 px target overhangs a 20 px box (the row's texts keep their widths).
                        div().size(px(20.)).flex_none().flex().items_center().justify_center().child(self.help_icon(
                            "nr-help-cost",
                            "Training only, from the time estimate: booting the instance and uploading the data come on top.",
                            window,
                            cx,
                        )),
                    ),
                )
                .child(fact("Limit", limit))
        });
        let footer = div()
            .flex()
            .flex_col()
            .gap(px(10.))
            .px(px(20.))
            .py(px(14.))
            .border_t_1()
            .border_color(t.line_subtle)
            .bg(t.bg_raised)
            // The frame's 12 px corners, less its 1 px edge: a square fill would show past them.
            .rounded_b(px(radius::XL - 1.))
            .children(cost_facts)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(12.))
                    .child(estimate_line.flex_1())
                    .child(
                        div()
                            .flex()
                            .flex_none()
                            .items_center()
                            .gap(px(8.))
                            .child(cancel)
                            .child(start)
                            // Ctrl Enter triggers the button it follows.
                            .child(shortcut(&t, &["Ctrl", "Enter"])),
                    ),
            );
        self.dialog_frame(
            &t,
            "new-run",
            720.,
            "New training run",
            Some(match (&busy_with, &runs_on) {
                (_, Some(on)) => format!("Trains on {on} and streams metrics into the run dashboard."),
                (Some(n), None) => {
                    format!("{n} is training; this run queues and starts when it finishes.")
                }
                (None, None) => "Trains on this machine and streams metrics into the run dashboard.".into(),
            }),
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_new_run(w, cx)),
        )
    }

    /// Generate a synthetic shapes dataset on folo serve: name, size, image size and classes.
    fn synth_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let f = self.synth.as_ref().unwrap();
        let (name, images, imgsz, classes, submitting, error) =
            (f.name.clone(), f.images.clone(), f.imgsz, f.classes, f.submitting, f.error.clone());
        let images_ok = crate::app::parse_images(&images.read(cx).text()).is_some();
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        let input_box = |this: &Self,
                         input: gpui::Entity<crate::input::TextInput>,
                         bad: bool,
                         w: f32,
                         window: &Window,
                         cx: &mut Context<Self>| {
            let fh = input.read(cx).focus_handle(cx);
            let ring = this.ring(&fh, window);
            let focused = fh.is_focused(window);
            div()
                .relative()
                .w(px(w))
                .h(px(32.))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if bad {
                    t.danger_solid
                } else if focused {
                    t.accent_line
                } else {
                    t.line_control
                })
                .text_size(px(ty::BASE.size))
                .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(input)
                .into_any_element()
        };
        let row = |name: &str, control: AnyElement, help: &str| {
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(label(name))
                        .child(text(ty::SM, t.text_tertiary).child(help.to_string())),
                )
                .child(control)
        };
        let name_box = input_box(self, name, false, 200., window, cx);
        let images_box = input_box(self, images, !images_ok, 120., window, cx);
        let size_labels: Vec<String> = crate::app::SYNTH_IMGSZ.iter().map(|s| format!("{s} px")).collect();
        let size_refs: Vec<&str> = size_labels.iter().map(String::as_str).collect();
        let size_seg = self.segmented(
            "syn-size",
            &size_refs,
            imgsz,
            window,
            cx,
            Rc::new(|this, i, _, _| {
                if let Some(f) = this.synth.as_mut() {
                    f.imgsz = i;
                }
            }),
        );
        let class_seg = self.segmented(
            "syn-classes",
            &["3 shapes", "5 shapes", "8 shapes"],
            classes,
            window,
            cx,
            Rc::new(|this, i, _, _| {
                if let Some(f) = this.synth.as_mut() {
                    f.classes = i;
                }
            }),
        );
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(row("Name", name_box, "Folder under the server's data dir"))
            .child(row("Images", images_box, "Split 80 / 20 into train and val"))
            .child(row("Image size", size_seg.into_any_element(), "Square JPEGs"))
            .child(row(
                "Classes",
                class_seg.into_any_element(),
                crate::app::SYNTH_CLASSES[classes].join(", ").as_str(),
            ))
            .when_some(error, |d, e| {
                d.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
                        .child(text(ty::SM, t.danger_text).child(e)),
                )
            });
        let cancel = self.button(
            "syn-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_synth(w, cx)),
        );
        let go = if submitting {
            self.button_disabled(
                "syn-go",
                "Generating…",
                Btn::Primary,
                Some("icons/zap.svg"),
                "Writing images…",
                window,
                cx,
            )
        } else {
            self.button(
                "syn-go",
                "Generate",
                Btn::Primary,
                Some("icons/zap.svg"),
                window,
                cx,
                act(|this, w, cx| this.submit_synth(w, cx)),
            )
        };
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "synth-dialog",
            520.,
            "Generate a synthetic dataset",
            Some("Procedural shapes with exact labels: a quick way to try training end to end.".into()),
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_synth(w, cx)),
        )
    }

    fn stop_dialog(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let run = self.store.run(id).cloned();
        let name = run.as_ref().map_or(id.to_string(), |r| r.name.clone());
        // The checkpoint kept is the last validated epoch's: name it from the history when it's
        // loaded (Run.epoch alone can't tell "epoch 12 finished" from "epoch 12 under way").
        let finished = self
            .store
            .run_data
            .get(id)
            .and_then(|d| d.ready())
            .map(|d| d.metrics.epochs.last().map(|e| e.epoch));
        let rid = id.to_string();
        let cancel = self.button(
            "stop-cancel",
            "Keep training",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| {
                this.confirm_stop = None;
                this.restore_focus(w);
                cx.notify();
            }),
        );
        let stop = self.button(
            "stop-confirm",
            "Stop run",
            Btn::DangerCommit,
            Some("icons/stop.svg"),
            window,
            cx,
            act(move |this, w, cx| {
                this.restore_focus(w);
                this.stop_run(rid.clone(), cx)
            }),
        );
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .child(
                text(ty::BASE, t.text_secondary)
                    .w_full()
                    .min_w_0()
                    .whitespace_normal()
                    // The epoch in progress is discarded; last.folo is the last finished epoch's.
                    .child(match finished {
                        Some(Some(e)) => format!("Stops after the current batch and keeps the epoch {e} checkpoint for export or resume. Progress after epoch {e} is lost."),
                        Some(None) => "Stops after the current batch. No epoch has finished yet, so there is no checkpoint to keep.".to_string(),
                        None => "Stops after the current batch and keeps the last finished epoch's checkpoint for export or resume. Progress after it is lost.".to_string(),
                    }),
            );
        let footer = dialog_footer(&t).justify_end().child(cancel).child(stop);
        self.dialog_frame(
            &t,
            "stop-dialog",
            440.,
            &format!("Stop {name}?"),
            None,
            body,
            footer,
            window,
            cx,
            // X, scrim and Esc dismiss like Keep training: focus goes back to what opened it.
            act(|this, w, cx| {
                this.confirm_stop = None;
                this.restore_focus(w);
                cx.notify();
            }),
        )
    }

    /// Connect a cloud: the provider, its fields as the server lists them (secrets masked), and an
    /// optional name. The server checks the values with the provider before it keeps them; what it
    /// refuses shows on the field it is about.
    fn connect_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let providers = self.cloud.providers.ready().cloned().unwrap_or_default();
        let Some(f) = self.cloud.connect.as_ref() else { return div().into_any_element() };
        let (chosen, inputs, name, submitting, error) =
            (f.provider, f.fields.clone(), f.name.clone(), f.submitting, f.error.clone());
        let Some(p) = providers.get(chosen).cloned() else { return div().into_any_element() };
        let names: Vec<&str> = providers.iter().map(|p| p.name.as_str()).collect();
        let provider_seg = self.segmented_fill(
            "cc-provider",
            &names,
            chosen,
            window,
            cx,
            Rc::new(|this, i, _, cx| this.set_connect_provider(i, cx)),
        );
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        // A field's box; Enter moves to the next field, and in the last one connects.
        let last = inputs.len();
        let input_box = |this: &Self,
                         input: gpui::Entity<crate::input::TextInput>,
                         index: usize,
                         bad: bool,
                         mono: bool,
                         window: &Window,
                         cx: &mut Context<Self>| {
            let fh = input.read(cx).focus_handle(cx);
            let ring = this.ring(&fh, window);
            let focused = fh.is_focused(window);
            div()
                .relative()
                .w_full()
                .h(px(32.))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if bad {
                    t.danger_solid
                } else if focused {
                    t.accent_line
                } else {
                    t.line_control
                })
                .text_size(px(ty::BASE.size))
                .when(mono, |d| d.font_family(crate::theme::MONO).text_size(px(ty::SM.size)))
                .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, w, cx| {
                    if ev.keystroke.key != "enter" || ev.keystroke.modifiers.control {
                        return;
                    }
                    let next = this
                        .cloud
                        .connect
                        .as_ref()
                        .and_then(|f| f.fields.get(index + 1).or((index + 1 == last).then_some(&f.name)));
                    match next {
                        Some(next) => w.focus(&next.read(cx).focus_handle(cx)),
                        None => this.submit_connect(w, cx),
                    }
                    cx.stop_propagation();
                }))
                .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(input)
                .into_any_element()
        };
        let field_error = |i: usize| match &error {
            Some(crate::cloud::ConnectError::Field(j, m)) if *j == i => Some(m.clone()),
            _ => None,
        };
        let first_secret = p.fields.iter().position(|f| f.secret);
        let mut rows: Vec<AnyElement> = Vec::new();
        for (i, (field, input)) in p.fields.iter().zip(inputs).enumerate() {
            let err = field_error(i);
            // Keys and addresses in Geist Mono, where every character counts.
            let mono = field.secret || field.key.ends_with("url");
            let control = input_box(self, input, i, err.is_some(), mono, window, cx);
            // Where the provider's console makes the credential, beside the first secret's help.
            let docs = p.docs_url.clone().filter(|_| Some(i) == first_secret).map(|url| {
                let link = div()
                    .id("cc-docs")
                    .flex_none()
                    .px(px(4.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(4.))
                            .child(
                                text(ty::SM, t.text_secondary).whitespace_nowrap().child("Where to find it"),
                            )
                            .child(icon("icons/external-link.svg", 12., t.text_secondary)),
                    );
                self.bind(link, "cc-docs", radius::SM, window, cx, act(move |_, _, cx| cx.open_url(&url)))
                    .tooltip(crate::controls::tip(p.docs_url.clone().unwrap_or_default(), None))
            });
            rows.push(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(label(&field.label))
                            .when(!field.required, |d| {
                                d.child(text(ty::SM, t.text_tertiary).child("Optional"))
                            }),
                    )
                    .child(control)
                    .when_some(err, |d, e| d.child(text(ty::SM, t.danger_text).whitespace_normal().child(e)))
                    .when(field.help.is_some() || docs.is_some(), |d| {
                        d.child(
                            div()
                                .flex()
                                .items_start()
                                .justify_between()
                                .gap(px(12.))
                                // Providers write commands and flags as `code`.
                                .child(
                                    rich(
                                        &t,
                                        ty::SM,
                                        t.text_tertiary,
                                        field.help.as_deref().unwrap_or_default(),
                                    )
                                    .text_left()
                                    .flex_1()
                                    .min_w_0(),
                                )
                                .children(docs),
                        )
                    })
                    .into_any_element(),
            );
        }
        let name_box = input_box(self, name, last, false, false, window, cx);
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(16.))
            .when_some(
                match &error {
                    Some(crate::cloud::ConnectError::Dialog(e)) => Some(e.clone()),
                    _ => None,
                },
                |d, e| d.child(error_note(&t, e.message, e.hint)),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(label("Provider"))
                    .child(provider_seg)
                    .when(!p.about.is_empty(), |d| {
                        d.child(rich(&t, ty::SM, t.text_tertiary, &p.about).text_left())
                    }),
            )
            .children(rows)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(label("Name"))
                            .child(text(ty::SM, t.text_tertiary).child("Optional")),
                    )
                    .child(name_box),
            );
        let cancel = self.button(
            "cc-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_connect(w, cx)),
        );
        let go = if submitting {
            self.button_disabled(
                "cc-go",
                "Connecting…",
                Btn::Primary,
                Some("icons/plug.svg"),
                "Checking the credentials with the provider…",
                window,
                cx,
            )
        } else {
            self.button(
                "cc-go",
                "Connect",
                Btn::Primary,
                Some("icons/plug.svg"),
                window,
                cx,
                act(|this, w, cx| this.submit_connect(w, cx)),
            )
        };
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "cloud-connect",
            520.,
            "Connect a cloud",
            Some("FOLO checks these with the provider, then keeps them on the server.".into()),
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_connect(w, cx)),
        )
    }

    /// Disconnect a connection: the server forgets its credentials; runs and models stay. A
    /// connection still in use (`409 connection_in_use`) says so here.
    fn disconnect_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(c) = self.cloud.disconnect.as_ref() else { return div().into_any_element() };
        let (busy, error) = (c.busy, c.error.clone());
        let conn = self.connection(&c.id).cloned();
        let name = conn.as_ref().map_or(c.id.clone(), |c| c.name.clone());
        // What it forgets, as the provider calls it ("the API key", "the token").
        let secret = conn
            .as_ref()
            .and_then(|c| self.cloud.providers.ready()?.iter().find(|p| p.id == c.provider))
            .and_then(|p| p.fields.iter().find(|f| f.secret))
            .map_or("credentials".to_string(), |f| crate::cloud::in_sentence(&f.label));
        let cancel = self.button(
            "cd-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        );
        let go = if busy {
            self.button_disabled(
                "cd-confirm",
                "Disconnecting…",
                Btn::DangerCommit,
                None,
                "Asking the server…",
                window,
                cx,
            )
        } else {
            self.button(
                "cd-confirm",
                "Disconnect",
                Btn::DangerCommit,
                None,
                window,
                cx,
                act(|this, _, cx| this.disconnect(cx)),
            )
        };
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(12.))
            .child(
                text(ty::BASE, t.text_secondary)
                    .whitespace_normal()
                    .child(format!("FOLO forgets the {secret}. Runs and models stay.")),
            )
            .when_some(error, |d, e| d.child(error_note(&t, e.message, e.hint)));
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "cloud-disconnect",
            440.,
            &format!("Disconnect {name}?"),
            None,
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        )
    }

    /// Terminate an instance by hand: it is shut down now and its run fails, keeping the
    /// checkpoints already copied here.
    fn terminate_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(c) = self.cloud.terminate.as_ref() else { return div().into_any_element() };
        let (busy, error) = (c.busy, c.error.clone());
        let inst = self.cloud.instances.ready().and_then(|l| l.iter().find(|i| i.id == c.id)).cloned();
        let what = inst.as_ref().map_or("instance".to_string(), |i| {
            format!("{} at {}", i.offer.gpu, self.provider_name(&i.provider))
        });
        let run = inst
            .as_ref()
            .and_then(|i| i.run_id.as_deref())
            .map(|r| self.store.run(r).map_or(r.to_string(), |run| run.name.clone()));
        let sentence = match run {
            Some(r) => format!(
                "The {what} is shut down now. Its run `{r}` fails; the checkpoints already copied to this machine stay."
            ),
            None => format!("The {what} is shut down now."),
        };
        let cancel = self.button(
            "ct-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        );
        let go = if busy {
            self.button_disabled(
                "ct-confirm",
                "Terminating…",
                Btn::DangerCommit,
                Some("icons/stop.svg"),
                "Asking the provider…",
                window,
                cx,
            )
        } else {
            self.button(
                "ct-confirm",
                "Terminate",
                Btn::DangerCommit,
                Some("icons/stop.svg"),
                window,
                cx,
                act(|this, _, cx| this.terminate(cx)),
            )
        };
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(12.))
            .child(rich(&t, ty::BASE, t.text_secondary, &sentence).text_left())
            .when_some(error, |d, e| d.child(error_note(&t, e.message, e.hint)));
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "cloud-terminate",
            440.,
            "Terminate this instance?",
            None,
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        )
    }

    fn toasts_view(&mut self, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        type ToastRow = (u64, Tone, SharedString, Option<SharedString>, Option<(SharedString, Action)>);
        let toasts: Vec<ToastRow> = self
            .toasts
            .iter()
            .map(|x| (x.id, x.tone, x.title.clone(), x.body.clone(), x.action.clone()))
            .collect();
        // One card at a time: older notifications fold into a single row above it.
        self.toast_focus = toasts.iter().map(|x| x.0).find(|id| {
            [format!("toast-x-{id}"), format!("toast-act-{id}")]
                .iter()
                .any(|k| self.focus.borrow().get(k.as_str()).is_some_and(|f| f.is_focused(window)))
        });
        // Focus on the stack's own controls holds every toast (none expires under the keyboard).
        self.toast_stack_focus = ["toast-expand", "toast-dismiss-all"]
            .iter()
            .any(|k| self.focus.borrow().get(*k).is_some_and(|f| f.is_focused(window)));
        let older = toasts.len().saturating_sub(1);
        let expanded = self.toasts_expanded;
        let toasts: Vec<ToastRow> = toasts.into_iter().skip(if expanded { 0 } else { older }).collect();
        let text_btn =
            |this: &Self, key: &str, label: String, window: &Window, cx: &mut Context<Self>, a: Action| {
                let el = div()
                    .id(SharedString::from(key.to_string()))
                    .px(px(6.))
                    .h(px(22.))
                    .flex()
                    .items_center()
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .child(text(ty::SM, t.text_secondary).child(label));
                this.bind(el, key, radius::SM, window, cx, a)
            };
        let folded = (older > 0).then(|| {
            let show = text_btn(
                self,
                "toast-expand",
                if expanded {
                    "Show latest only".to_string()
                } else if older == 1 {
                    "Show 1 earlier".to_string()
                } else {
                    format!("Show {older} earlier")
                },
                window,
                cx,
                act(|this, _, cx| {
                    this.toasts_expanded = !this.toasts_expanded;
                    cx.notify();
                }),
            );
            let dismiss = text_btn(
                self,
                "toast-dismiss-all",
                "Dismiss all".to_string(),
                window,
                cx,
                act(|this, _, cx| {
                    this.toasts.clear();
                    this.toasts_expanded = false;
                    cx.notify();
                }),
            );
            (show.into_any_element(), dismiss.into_any_element())
        });
        // While folded, the earlier toasts' edges peek out behind the latest one: opaque cards (not
        // tints the page shows through), line.default so the toast's own top edge isn't doubled.
        let peeks = (older > 0 && !expanded).then(|| {
            let edge = |inset: f32, a: f32| {
                div()
                    .occlude()
                    .mx(px(inset))
                    .h(px(6.))
                    .rounded_t(px(radius::LG))
                    .bg(if a < 1. {
                        crate::theme::over(t.bg_raised, crate::theme::fade(t.bg_app, 1. - a))
                    } else {
                        t.bg_raised
                    })
                    .border_1()
                    .border_b_0()
                    .border_color(t.line_default)
            };
            div().flex().flex_col().when(older > 1, |d| d.child(edge(16., 0.6))).child(edge(8., 1.))
        });
        // The stack's controls live in the newest card's action row (no floating pill).
        let mut stack_ctrls = folded;
        let newest = toasts.len().saturating_sub(1);
        div()
            .absolute()
            .bottom(px(24.))
            .right(px(24.))
            .w(px(360.))
            .flex()
            .flex_col()
            .gap(px(if peeks.is_some() { 0. } else { 8. }))
            .children(peeks)
            .children(toasts.into_iter().enumerate().map(|(i, (id, tone, title, body, action))| {
                let ctrls = if i == newest { stack_ctrls.take() } else { None };
                let action_btn = action.map(|(label, a)| {
                    self.button(
                        &format!("toast-act-{id}"),
                        label,
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(move |this, w, cx| {
                            a(this, w, cx);
                            this.dismiss_toast(id, cx);
                        }),
                    )
                });
                let (_, _, dotc) = tone.colors(&t);
                let icon_path = match tone {
                    Tone::Success => "icons/circle-check.svg",
                    Tone::Danger => "icons/circle-x.svg",
                    Tone::Warning => "icons/triangle-alert.svg",
                    Tone::Accent => "icons/activity.svg",
                    _ => "icons/info.svg",
                };
                // Each toast fades in as it arrives (160 ms; none under Reduce motion).
                crate::ui::appear(
                    div()
                        .id(("toast", id as usize))
                        .occlude()
                        .on_hover(cx.listener(move |this, h: &bool, _, _| {
                            if *h {
                                this.toast_hover = Some(id);
                            } else if this.toast_hover == Some(id) {
                                this.toast_hover = None;
                            }
                        }))
                        .flex()
                        .flex_col()
                        .p(px(12.))
                        .rounded(px(radius::LG))
                        .bg(t.bg_overlay)
                        .border_1()
                        .border_color(t.line_strong)
                        .shadow(t.shadow_lg())
                        .child(
                            div()
                                .flex()
                                .items_start()
                                .gap(px(10.))
                                .child(div().pt(px(2.)).child(icon(
                                    icon_path,
                                    16.,
                                    if tone == Tone::Neutral { t.text_secondary } else { dotc },
                                )))
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .flex()
                                        .flex_col()
                                        .gap(px(2.))
                                        .child(
                                            text(ty::BASE, t.text_primary)
                                                .font_weight(FontWeight::MEDIUM)
                                                .child(title),
                                        )
                                        .when_some(body, |d, b| {
                                            d.child(text(ty::SM, t.text_secondary).child(b))
                                        }),
                                )
                                .child(
                                    self.icon_button(
                                        &format!("toast-x-{id}"),
                                        "icons/x.svg",
                                        "Dismiss",
                                        window,
                                        cx,
                                        act(move |this, _, cx| this.dismiss_toast(id, cx)),
                                    )
                                    .size(px(24.)),
                                ),
                        )
                        // One footer: the card's action under the text, the stack's controls ending at
                        // the card's padding, under the close button.
                        .when(action_btn.is_some() || ctrls.is_some(), |d| {
                            d.child(
                                div()
                                    .pt(px(8.))
                                    .pl(px(26.))
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .gap(px(8.))
                                    .child(div().flex().children(action_btn))
                                    .children(ctrls.map(|(show, dismiss)| {
                                        div().flex().items_center().gap(px(4.)).child(show).child(dismiss)
                                    })),
                            )
                        }),
                    &format!("toast-in-{id}"),
                )
            }))
            .into_any_element()
    }

    /// The inspector was asked for an image by id (`Inspector::want`): once the list is in, its
    /// place in the filtered list. An image the dataset page's filters hide clears them (the
    /// inspector walks that list); one the dataset doesn't have opens the first image.
    fn land_inspector(&mut self) {
        let Some((ds, want)) =
            self.inspector.as_ref().and_then(|i| Some((i.dataset.clone(), i.want.clone()?)))
        else {
            return;
        };
        let Some(all) = self.store.images.get(&ds).and_then(|l| l.ready()) else { return };
        let in_all = all.iter().position(|m| m.id == want);
        let shown = self.filtered_indices(&ds).iter().position(|&i| all[i].id == want);
        let index = match (shown, in_all) {
            (Some(i), _) => i,
            (None, Some(i)) => {
                self.ds_split = None;
                self.ds_class = None;
                self.ds_empty_only = false;
                // Unfiltered, the list is every image in order.
                i
            }
            (None, None) => 0,
        };
        if let Some(i) = self.inspector.as_mut() {
            i.index = index;
            i.want = None;
        }
    }

    fn inspector_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.land_inspector();
        let insp = self.inspector.as_ref().unwrap();
        let ds_id = insp.dataset.clone();
        let n = self.filtered_indices(&ds_id).len();
        if n == 0 {
            // Still loading the image list (or the filter emptied it): keep the scrim up.
            if self.store.images.get(&ds_id).is_some_and(|l| !l.is_loading()) {
                self.inspector = None;
            }
            return div().absolute().top_0().left_0().size_full().bg(t.bg_scrim).into_any_element();
        }
        let index = insp.index.min(n - 1);
        let (zoom, pan, hover) = (insp.zoom, insp.pan, insp.hover_box);
        let Some(meta) = self.filtered_image(&ds_id, index).cloned() else { return div().into_any_element() };
        // A keyboard-focused label row highlights its box like the pointer does; while the keyboard
        // is in use it wins over a pointer hover left behind on the image.
        // Opened from a run's samples: this image's predictions at that epoch, if it is one of them.
        let sample_ctx = self.inspector.as_ref().and_then(|i| i.sample.clone());
        let sample_set =
            sample_ctx.as_ref().and_then(|(run, _)| self.store.samples.get(run)?.ready().cloned());
        let sample_preds = sample_set.as_ref().and_then(|s| {
            let smp = s.samples.iter().find(|x| x.image_id == meta.id)?;
            Some((s.epoch, smp.pred.clone()))
        });
        let show_preds = sample_ctx.as_ref().is_some_and(|(_, v)| *v != crate::app::SampleView::Truth);
        // Shown predictions (score ≥ 0.30, as the run page draws them). Box indices: the labels
        // first (0..n), then these (n..), for hover, focus and the list rows alike.
        let preds: Option<Vec<crate::api::ScoredBox>> = sample_preds
            .as_ref()
            .filter(|_| show_preds)
            .map(|(_, ps)| ps.iter().filter(|p| p.score >= 0.3).cloned().collect());
        let n_labels = meta.boxes.len();
        let focused =
            (0..n_labels).find(|i| self.fh(format!("insp-box-{i}"), cx).is_focused(window)).or_else(|| {
                (0..preds.as_ref().map_or(0, Vec::len))
                    .find(|j| self.fh(format!("insp-pred-{j}"), cx).is_focused(window))
                    .map(|j| n_labels + j)
            });
        let hover = if self.kb { focused.or(hover) } else { hover.or(focused) };
        let classes = self.store.dataset_summary(&ds_id).map(|d| d.classes.clone()).unwrap_or_default();
        let img = self
            .image(&ds_id, &meta.id, ImgKind::Full, cx)
            .or_else(|| self.image(&ds_id, &meta.id, ImgKind::Thumb, cx));
        let dims = (meta.width as f32, meta.height as f32);
        let mut boxes: Vec<DrawBox> = meta
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| DrawBox {
                label: classes.get(b.class as usize).cloned().map(Into::into),
                highlight: hover == Some(i),
                dim: hover.is_some_and(|h| h != i),
                // With predictions over them, the labels are drawn as the run page draws ground
                // truth: dashed, no chip (the list beside names them).
                truth: preds.is_some() && hover != Some(i),
                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
            })
            .collect();
        if let Some(ps) = &preds {
            // After the labels, so a label row's index is still its box's.
            boxes.extend(ps.iter().enumerate().map(|(j, p)| DrawBox {
                label: classes.get(p.class as usize).cloned().map(Into::into),
                score: Some(p.score),
                highlight: hover == Some(n_labels + j),
                dim: hover.is_some_and(|h| h != n_labels + j),
                ..DrawBox::new(p.class, p.x1, p.y1, p.x2, p.y2)
            }));
        }
        // What the image shows, said beside it: the run, the epoch, and how each kind is drawn.
        let sample_line: Option<(String, String)> = sample_ctx.as_ref().map(|(run, _)| {
            let name = self.store.run(run).map_or(run.clone(), |r| r.name.clone());
            match (&sample_preds, show_preds) {
                (Some((epoch, _)), true) => (
                    format!("{name} · epoch {epoch}"),
                    "Predictions ≥ 0.30 solid · labels dashed".to_string(),
                ),
                (Some((epoch, _)), false) => {
                    (format!("{name} · epoch {epoch}"), "Labels (the run page's view)".to_string())
                }
                (None, _) => (name, "Not one of this run's samples: labels only".to_string()),
            }
        });
        let bounds_cell = self.insp_view.clone();
        // A pan pushed past the edge is stored clamped, so reversing it responds at once.
        if let (Some(b), Some(i)) = (bounds_cell.get(), self.inspector.as_mut()) {
            i.pan = crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, i.zoom, i.pan);
        }
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let down_cell = bounds_cell.clone();
        let paint_boxes = boxes.clone();
        let hit_boxes = boxes.clone();
        let tt = t;
        let img2 = img.clone();
        let stage = div()
            .id("insp-stage")
            .relative()
            .flex_1()
            .min_w_0()
            .bg(t.bg_inset)
            .cursor(if zoom > 1.0 { gpui::CursorStyle::OpenHand } else { gpui::CursorStyle::Arrow })
            .on_scroll_wheel(cx.listener(move |this, ev: &ScrollWheelEvent, w, cx| {
                let Some(b) = wheel_cell.get() else { return };
                let Some(insp) = this.inspector.as_mut() else { return };
                let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                let factor = (1.0 + dy * 0.0025).clamp(0.5, 2.0);
                let new_zoom = (insp.zoom * factor).clamp(1.0, 12.0);
                // Zoom about the cursor: keep the image point under it fixed.
                let rect = image_rect(b, dims.0, dims.1, Fit::Contain, insp.zoom, insp.pan);
                let (px_, py_) = (
                    f32::from(ev.position.x - rect.left()) / f32::from(rect.size.width),
                    f32::from(ev.position.y - rect.top()) / f32::from(rect.size.height),
                );
                let new_rect = image_rect(b, dims.0, dims.1, Fit::Contain, new_zoom, (0., 0.));
                let want_x = f32::from(ev.position.x) - px_ * f32::from(new_rect.size.width);
                let want_y = f32::from(ev.position.y) - py_ * f32::from(new_rect.size.height);
                insp.pan = if new_zoom <= 1.0 {
                    (0., 0.)
                } else {
                    (want_x - f32::from(new_rect.left()), want_y - f32::from(new_rect.top()))
                };
                insp.zoom = new_zoom;
                cx.notify();
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, _, cx| {
                    let _ = down_cell.get();
                    if let Some(insp) = this.inspector.as_mut() {
                        insp.drag_from = Some((ev.position, insp.pan));
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if let Some(insp) = this.inspector.as_mut() {
                        insp.drag_from = None;
                        cx.notify();
                    }
                }),
            )
            .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                let Some(insp) = this.inspector.as_mut() else { return };
                if let Some((from, pan0)) = insp.drag_from
                    && ev.pressed_button == Some(MouseButton::Left)
                {
                    if insp.zoom > 1.0 {
                        insp.pan = (
                            pan0.0 + f32::from(ev.position.x - from.x),
                            pan0.1 + f32::from(ev.position.y - from.y),
                        );
                        cx.notify();
                    }
                    return;
                }
                let Some(b) = bounds_cell.get() else { return };
                let (x, y) = to_image(b, dims, Fit::Contain, insp.zoom, insp.pan, ev.position);
                let hit = hit_box(&hit_boxes, x, y);
                if hit != insp.hover_box {
                    insp.hover_box = hit;
                    // The pointer moved over the image: hover leads again (not a keyboard
                    // focus left on a label row).
                    this.kb = false;
                    cx.notify();
                }
            }))
            .child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, cx| {
                        let inner = gpui::Bounds::new(
                            b.origin + gpui::point(px(24.), px(24.)),
                            // 24 px on every side (the zoom controls live in the header now).
                            b.size - gpui::size(px(48.), px(48.)),
                        );
                        paint_cell.set(Some(inner));
                        window.with_content_mask(Some(gpui::ContentMask { bounds: b }), |window| {
                            paint_image_with_boxes(
                                &tt,
                                window,
                                cx,
                                inner,
                                img2.as_ref(),
                                dims,
                                &paint_boxes,
                                Fit::Contain,
                                zoom,
                                pan,
                                true,
                                false,
                                0.,
                            );
                        });
                    },
                )
                .size_full(),
            );
        // By class, then largest first, like the PR legend; indices stay the image's.
        let mut box_order: Vec<(usize, &crate::api::BBox)> = meta.boxes.iter().enumerate().collect();
        box_order.sort_by(|(_, a), (_, b)| {
            let area = |x: &crate::api::BBox| (x.x2 - x.x1) * (x.y2 - x.y1);
            a.class.cmp(&b.class).then(area(b).total_cmp(&area(a)))
        });
        let insp_chrome = self.scroll_chrome(self.insp_list_scroll.clone(), t.bg_overlay, 0., 0., cx);
        let mut box_rows: Vec<AnyElement> = box_order
            .into_iter()
            .map(|(i, b)| {
                let hot = hover == Some(i);
                let key = format!("insp-box-{i}");
                // The class column is 70 px: a name that may not fit shows in full on hover and on
                // keyboard focus (short ones need neither).
                let full_name = classes.get(b.class as usize).cloned().unwrap_or_default();
                let long = crate::ui::may_overflow(&full_name, ty::SM.size, 70.);
                let row = self
                    .disclose(
                        div().id(SharedString::from(key.clone())),
                        &key,
                        full_name,
                        long,
                        30.,
                        false,
                        window,
                        cx,
                    )
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(&t)))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            let next = if *h { Some(i) } else { None };
                            if insp.hover_box != next {
                                insp.hover_box = next;
                                cx.notify();
                            }
                        }
                    }))
                    .child(crate::ui::class_swatch(&t, b.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .ellipsis()
                            .child(classes.get(b.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(
                        text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                            .tnum()
                            .flex_1()
                            .text_right()
                            .child(format!(
                                "{:.0}, {:.0} · {:.0}×{:.0}",
                                b.x1,
                                b.y1,
                                b.x2 - b.x1,
                                b.y2 - b.y1
                            )),
                    );
                // Focusable: Tab through labels to highlight each box from the keyboard.
                self.bind(
                    row,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            insp.hover_box = if insp.hover_box == Some(i) { None } else { Some(i) };
                        }
                        cx.notify();
                    }),
                )
                .into_any_element()
            })
            .collect();
        // The epoch's predictions, after the labels: score and box, each a Tab stop that
        // highlights its box (as a label row does).
        if let Some(ps) = &preds {
            box_rows.push(
                div()
                    .px(px(8.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(&t, "Predictions"))
                    .child(text(ty::SM, t.text_tertiary).child("score · x, y · w×h"))
                    .into_any_element(),
            );
            for (j, p) in ps.iter().enumerate() {
                let idx = n_labels + j;
                let hot = hover == Some(idx);
                let key = format!("insp-pred-{j}");
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(&t)))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            let next = if *h { Some(idx) } else { None };
                            if insp.hover_box != next {
                                insp.hover_box = next;
                                cx.notify();
                            }
                        }
                    }))
                    .child(crate::ui::class_swatch(&t, p.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .ellipsis()
                            .child(classes.get(p.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(text(ty::SM, t.text_primary).tnum().w(px(32.)).child(format!("{:.2}", p.score)))
                    .child(
                        text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                            .tnum()
                            .flex_1()
                            .text_right()
                            .child(format!(
                                "{:.0}, {:.0} · {:.0}×{:.0}",
                                p.x1,
                                p.y1,
                                p.x2 - p.x1,
                                p.y2 - p.y1
                            )),
                    );
                box_rows.push(
                    self.bind(
                        row,
                        &key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            if let Some(insp) = this.inspector.as_mut() {
                                insp.hover_box = if insp.hover_box == Some(idx) { None } else { Some(idx) };
                            }
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                );
            }
        }
        let prev = self.icon_button(
            "insp-prev",
            "icons/chevron-left.svg",
            "Previous (←)",
            window,
            cx,
            act(|this, _, cx| this.step_inspector(-1, cx)),
        );
        let next = self.icon_button(
            "insp-next",
            "icons/chevron-right.svg",
            "Next (→)",
            window,
            cx,
            act(|this, _, cx| this.step_inspector(1, cx)),
        );
        let close = self.icon_button(
            "insp-close",
            "icons/x.svg",
            "Close (Esc)",
            window,
            cx,
            act(|this, w, cx| this.close_inspector(w, cx)),
        );
        let side = div()
            .w(px(300.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(t.bg_overlay)
            .border_l_1()
            .border_color(t.line_subtle)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .p(px(16.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child({
                        // The panel's 268 px hold ~33 monospace glyphs (0.6 em each); a longer
                        // path ends in "…" and shows in full on hover.
                        let long = meta.file.chars().count() as f32 * 0.6 * ty::BASE.size > 268.;
                        let el = div().id("insp-file");
                        self.disclose(el, "insp-file", meta.file.clone(), long, 24., true, window, cx).child(
                            text(ty::BASE, t.text_primary)
                                .font_family(crate::theme::MONO)
                                .ellipsis()
                                .child(meta.file.clone()),
                        )
                    })
                    .child(
                        div()
                            .flex()
                            .gap(px(6.))
                            .child(badge(&t, meta.split.label(), Tone::Neutral))
                            .child(badge(&t, format!("{} × {}", meta.width, meta.height), Tone::Neutral))
                            .child(badge(
                                &t,
                                match &preds {
                                    Some(_) => format!("{} labels", meta.boxes.len()),
                                    None => format!("{} boxes", meta.boxes.len()),
                                },
                                Tone::Neutral,
                            ))
                            .when_some(preds.as_ref().map(Vec::len), |d, m| {
                                d.child(badge(
                                    &t,
                                    format!("{m} {}", if m == 1 { "prediction" } else { "predictions" }),
                                    Tone::Neutral,
                                ))
                            }),
                    )
                    .when_some(sample_line, |d, (run, how)| {
                        d.child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .child(text(ty::SM, t.text_secondary).ellipsis().child(run))
                                .child(text(ty::SM, t.text_tertiary).child(how)),
                        )
                    }),
            )
            .child(
                div()
                    .px(px(16.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(&t, "Labels"))
                    .child(text(ty::SM, t.text_tertiary).child("top-left x, y · w×h (px)")),
            )
            .child(
                div()
                    .relative()
                    .flex_shrink()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("insp-boxes")
                            // As tall as its rows (up to the room left), so the key hints sit right
                            // under it.
                            .flex_shrink()
                            .min_h_0()
                            .overflow_y_scroll()
                            .track_scroll(&self.insp_list_scroll)
                            .px(px(8.))
                            .children(box_rows)
                            .when(meta.boxes.is_empty(), |d| {
                                d.child(div().p(px(8.)).child(
                                    text(ty::SM, t.text_tertiary).child("No labels: a background image."),
                                ))
                            }),
                    )
                    .children(insp_chrome),
            )
            .child(
                // One action per row on a fixed key column, so the keys line up and nothing wraps.
                {
                    let row = |keys: AnyElement, what: &'static str| {
                        div()
                            .flex()
                            .items_center()
                            .child(
                                div()
                                    .w(px(96.))
                                    .flex_none()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap(px(4.))
                                    .child(keys),
                            )
                            // At 12 px the longest hint needs two lines in the 300 px panel: it
                            // wraps inside its column rather than running past the edge.
                            .child(
                                text(ty::SM, t.text_tertiary)
                                    .flex_1()
                                    .min_w_0()
                                    .whitespace_normal()
                                    .child(what),
                            )
                    };
                    div()
                        .p(px(12.))
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .border_t_1()
                        .border_color(t.line_subtle)
                        // Both pairs step the same way: one row, the pairs stacked in the key column.
                        .child(row(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(4.))
                                .child(shortcut(&t, &["←", "→"]))
                                .child(shortcut(&t, &["PgUp", "PgDn"]))
                                .into_any_element(),
                            "previous / next",
                        ))
                        .child(row(
                            div()
                                .flex()
                                .gap(px(4.))
                                .child(kbd(&t, "Scroll"))
                                .child(shortcut(&t, &["+", "−"]))
                                .into_any_element(),
                            "zoom",
                        ))
                        .child(row(kbd(&t, "0").into_any_element(), "fit"))
                        .child(row(kbd(&t, "Drag").into_any_element(), "pan (arrows too, when zoomed)"))
                },
            );
        // Fitted by any path (the readout, 0, −, another image), focus left on the readout moves
        // to its neighbour.
        if zoom <= 1.0 && self.fh("insp-fit", cx).is_focused(window) {
            self.focus_key("insp-zoom-in", window, cx);
        }
        // Zoom in the top bar, right-aligned before Close, as Playground keeps it in its header.
        let zoom_ctrls = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(self.icon_button(
                "insp-zoom-out",
                "icons/zoom-out.svg",
                "Zoom out",
                window,
                cx,
                act(|this, _, cx| this.zoom_by(1.0 / 1.4, cx)),
            ))
            .child(self.zoom_readout(
                "insp-fit",
                zoom,
                window,
                cx,
                act(|this, _, cx| {
                    if let Some(i) = this.inspector.as_mut() {
                        i.zoom = 1.0;
                        i.pan = (0., 0.);
                    }
                    cx.notify();
                }),
            ))
            .child(self.icon_button(
                "insp-zoom-in",
                "icons/zoom-in.svg",
                "Zoom in",
                window,
                cx,
                act(|this, _, cx| this.zoom_by(1.4, cx)),
            ));
        let header = div()
            .h(px(48.))
            .flex_none()
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(8.))
            .bg(t.bg_overlay)
            .border_b_1()
            .border_color(t.line_subtle)
            .child(prev)
            .child(next)
            .child(text(ty::SM, t.text_secondary).tnum().child(
                match sample_set.as_ref().and_then(|s| {
                    let at = s.samples.iter().position(|x| x.image_id == meta.id)?;
                    Some((at, s.samples.len(), s.epoch))
                }) {
                    // Opened from a run's samples: ← / → walk those.
                    Some((at, len, epoch)) => format!("Sample {} of {len} · epoch {epoch}", at + 1),
                    None => format!("{} of {}", int(index as u64 + 1), int(n as u64)),
                },
            ))
            .child(div().flex_1())
            .child(zoom_ctrls)
            .child(div().w(px(1.)).h(px(16.)).bg(t.line_subtle))
            .child(close);
        // Fades in like the dialogs and palette (160 ms; none under Reduce motion).
        crate::ui::appear(
            div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .child(self.scrim(&t, "insp-scrim", cx, act(|this, w, cx| this.close_inspector(w, cx))))
                .child(
                    div()
                        .id("inspector")
                        .track_focus(&self.inspector_focus)
                        .absolute()
                        .top(px(24.))
                        .left(px(24.))
                        .right(px(24.))
                        .bottom(px(24.))
                        .occlude()
                        .flex()
                        .flex_col()
                        .rounded(px(radius::XL))
                        .border_1()
                        .border_color(t.line_default)
                        .shadow(t.shadow_lg())
                        .overflow_hidden()
                        .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _, cx| {
                            // Zoomed in, arrows pan (Shift: further); otherwise ←/→ browse. Page Up/Down
                            // always browse.
                            let zoomed = this.inspector.as_ref().is_some_and(|i| i.zoom > 1.0);
                            let d = if ev.keystroke.modifiers.shift { 128. } else { 32. };
                            let pan = |this: &mut Self, dx: f32, dy: f32| {
                                if let Some(i) = this.inspector.as_mut() {
                                    i.pan = (i.pan.0 + dx, i.pan.1 + dy);
                                }
                            };
                            match ev.keystroke.key.as_str() {
                                "left" if zoomed => pan(this, d, 0.),
                                "right" if zoomed => pan(this, -d, 0.),
                                "up" if zoomed => pan(this, 0., d),
                                "down" if zoomed => pan(this, 0., -d),
                                "left" | "pageup" => this.step_inspector(-1, cx),
                                "right" | "pagedown" => this.step_inspector(1, cx),
                                "+" | "=" => this.zoom_by(1.4, cx),
                                "-" => this.zoom_by(1.0 / 1.4, cx),
                                "0" => {
                                    if let Some(i) = this.inspector.as_mut() {
                                        i.zoom = 1.0;
                                        i.pan = (0., 0.);
                                    }
                                }
                                _ => return,
                            }
                            cx.stop_propagation();
                            cx.notify();
                        }))
                        .child(header)
                        .child(div().flex_1().min_h_0().flex().child(stage).child(side)),
                ),
            "inspector-in",
        )
    }

    pub fn step_inspector(&mut self, d: i32, cx: &mut Context<Self>) {
        // Opened from a run's samples: the previous / next sample (the inspector lands on it by id).
        let next_sample = self.inspector.as_ref().and_then(|i| {
            let (run, _) = i.sample.as_ref()?;
            let set = self.store.samples.get(run)?.ready()?;
            let cur = self.filtered_image(&i.dataset, i.index).map(|m| m.id.clone());
            let at = cur.and_then(|c| set.samples.iter().position(|x| x.image_id == c))?;
            let len = set.samples.len() as i32;
            Some(set.samples[(at as i32 + d).rem_euclid(len) as usize].image_id.clone())
        });
        if let (Some(id), Some(i)) = (next_sample, self.inspector.as_mut()) {
            i.want = Some(id);
            i.zoom = 1.0;
            i.pan = (0., 0.);
            i.hover_box = None;
            cx.notify();
            return;
        }
        let n = self.inspector.as_ref().map_or(0, |i| self.filtered_indices(&i.dataset).len());
        if let Some(i) = self.inspector.as_mut()
            && n > 0
        {
            i.index = ((i.index as i32 + d).rem_euclid(n as i32)) as usize;
            i.zoom = 1.0;
            i.pan = (0., 0.);
            i.hover_box = None;
            cx.notify();
        }
    }

    pub fn zoom_by(&mut self, f: f32, cx: &mut Context<Self>) {
        if let Some(i) = self.inspector.as_mut() {
            (i.zoom, i.pan) = crate::viz::zoom_step(i.zoom, i.pan, f);
            cx.notify();
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rough_durations() {
        assert_eq!(super::rough_duration(576.), "10 min");
        assert_eq!(super::rough_duration(25.), "25 s");
        assert_eq!(super::rough_duration(5100.), "1 h 25 min");
    }

    use super::*;

    #[test]
    fn fuzzy_ranks_prefix_and_substring() {
        assert!(fuzzy("run", "Runs").unwrap() > fuzzy("run", "shapes-n-run").unwrap());
        assert!(fuzzy("shn", "shapes-n").is_some());
        assert!(fuzzy("zzz", "shapes").is_none());
        assert_eq!(fuzzy("", "anything"), Some(0));
    }

    #[test]
    fn estimates_flag_oversized_configs() {
        let (_, small) = estimate(2, 100., 32, 640, 3200);
        assert!(small < 8.0);
        let (_, huge) = estimate(3, 100., 64, 1024, 3200);
        assert!(huge > 16.0, "the failed fixture run should not fit");
        let (s1, _) = estimate(0, 100., 16, 640, 800);
        let (s2, _) = estimate(0, 200., 16, 640, 800);
        assert!((s2 / s1 - 2.0).abs() < 1e-6);
    }
}
