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
const OPT_LABELS: [&str; 3] = ["Auto", "SGD", "AdamW"];
use crate::controls::Btn;
use crate::store::ImgKind;
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
                    s => format!("{} · {}", s.label().to_lowercase(), r.config.arch.as_str()),
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
                        .truncate()
                        .child(it.title),
                    )
                    .child(
                        text(ty::SM, if on { t.text_secondary } else { t.text_tertiary })
                            .flex_1()
                            .min_w_0()
                            .truncate()
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
                    .child(div().relative().child(list).child(self.scrollbar(scroll_bar, cx)).child(
                        div().absolute().bottom_0().left_0().right_0().h(px(16.)).bg(gpui::linear_gradient(
                            0.,
                            gpui::linear_color_stop(t.bg_overlay, 0.),
                            gpui::linear_color_stop(crate::theme::alpha(t.bg_overlay, 0.), 1.),
                        )),
                    ))
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
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(shortcut(&t, &["↑", "↓"]))
                                    .child(text(ty::XS, t.text_tertiary).child("navigate")),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(kbd(&t, "Enter"))
                                    .child(text(ty::XS, t.text_tertiary).child("open")),
                            )
                            .child(div().flex_1())
                            .child(text(ty::XS, t.text_tertiary).tnum().child(if searching {
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
        let more_below = {
            let s = &self.dialog_scroll;
            s.max_offset().height + s.offset().y > px(1.)
        };
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
                                        .child(
                                            div()
                                                .id("dialog-body")
                                                .flex_1()
                                                .min_h_0()
                                                .overflow_y_scroll()
                                                .track_scroll(&self.dialog_scroll)
                                                .child(body),
                                        )
                                        .child(self.scrollbar(self.dialog_scroll.clone(), cx))
                                        // Only while more of the form lies below: at the end, the last
                                        // row's controls show undimmed.
                                        .when(more_below, |d| {
                                            d.child(
                                                div().absolute().bottom_0().left_0().right_0().h(px(16.)).bg(
                                                    gpui::linear_gradient(
                                                        0.,
                                                        gpui::linear_color_stop(t.bg_overlay, 0.),
                                                        gpui::linear_color_stop(
                                                            crate::theme::alpha(t.bg_overlay, 0.),
                                                            1.,
                                                        ),
                                                    ),
                                                ),
                                            )
                                        }),
                                )
                                .child(footer),
                        ),
                ),
            &format!("{id}-in"),
        )
    }

    fn new_run_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
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
                changed.push(format!("{} at {lr0}", OPT_LABELS[opt]));
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
        if self.amp_blocked(DEVICES[device]).is_some()
            && self.focus.borrow().get("nr-amp").is_some_and(|f| f.is_focused(window))
        {
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
                .when_some(help, |d, h| d.child(text(ty::XS, t.text_tertiary).tnum().child(h)))
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
        let valued = |name: &str, value: String, control: AnyElement| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .child(label(name))
                        .child(text(ty::SM, t.text_primary).tnum().child(value)),
                )
                .child(control)
        };
        // Short windows (the 620 px minimum): tighter rhythm and no restating helpers in the Model
        // rows, so the Schedule's controls are in view without scrolling.
        let short = window.viewport_size().height < px(700.);
        // What Auto resolves to for this dataset, batch and length (the engine's rule: SGD at 0.01
        // past 10,000 optimizer steps of a 64-image nominal batch, else AdamW at 0.01 / (4 + classes) to six decimals),
        // so the form can say which, and that the rate field is then unused.
        let auto_pick = ds.as_ref().filter(|_| opt == 0).map(|d| {
            let steps = (d.splits.train as usize).div_ceil(BATCHES[batch].max(64) as usize)
                * epochs.round().max(1.) as usize;
            if steps > 10_000 {
                ("SGD", "0.01".to_string())
            } else {
                let lr = (0.01 / (4. + d.classes.len() as f64) * 1e6).round() / 1e6;
                // Its six decimals, written as the run dashboard writes rates (1.11×10⁻³).
                let p = lr.log10().floor() as i32;
                ("AdamW", format!("{:.2}×10{}", lr * 10f64.powi(-p), crate::charts::superscript(p)))
            }
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
                .when_some(bad, |d, msg| d.child(text(ty::XS, t.danger_text).child(msg.to_string())))
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
        let body = div()
            .flex()
            .flex_col()
            .gap(px(if short { 8. } else { 14. }))
            .px(px(20.))
            // 16 px past the last control for the footer's fade, so it never dims a control.
            .pb(px(36.))
            // Errors lead the form, where a failed submit is seen (the dialog scrolls to the top).
            .when_some(error.clone(), |d, e| {
                d.child(
                    div()
                        .flex()
                        .items_start()
                        .gap(px(8.))
                        .p(px(10.))
                        .rounded(px(radius::MD))
                        .bg(t.danger_muted)
                        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
                        // The server's hint says what to change (device_not_found lists the
                        // adapters there are; gpu_out_of_memory names batch, imgsz and arch).
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .min_w_0()
                                .child(text(ty::SM, t.danger_text).whitespace_normal().child(e.message))
                                .when_some(e.hint, |d, h| {
                                    d.child(text(ty::XS, t.text_secondary).whitespace_normal().child(h))
                                }),
                        ),
                )
            })
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
                                            text(ty::XS, t.text_tertiary).tnum().truncate().child(s)
                                        }),
                                    ),
                            )
                            .child(ds_select)
                    }),
                ),
            )
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
                        div().flex_1().child(match device_help {
                            Some(h) => labelled("Device", h, device_seg),
                            None => field(
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
                            .child(text(ty::XS, t.text_tertiary).child(match ds.as_ref() {
                                Some(d) => format!("Each one pass over {} training images", int(d.splits.train)),
                                None => "Each one pass over the training images".to_string(),
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
                                .child(text(ty::XS, t.text_tertiary).child({
                                    let last = epochs.round().max(1.) as u32;
                                    let n = (close_mosaic.round() as u32).min(last);
                                    match n {
                                        0 => "Off: mosaic all the way through".to_string(),
                                        1 => format!("Epoch {last} trains on whole images"),
                                        _ => format!("Epochs {}–{last} train on whole images", last - n + 1),
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
                    .child(text(ty::XS, t.text_tertiary).min_w_0().truncate().child(adv_summary.clone()));
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
                d
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
                                    text(ty::XS, t.text_tertiary)
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
                                (0, None) => "Auto: SGD for long runs, else AdamW".to_string(),
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
                                                .child(text(ty::SM, t.text_tertiary).tnum().child(
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
                                    d.child(text(ty::XS, t.text_tertiary).child(if opt == 0 {
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
                    let blocked = self.amp_blocked(DEVICES[device]);
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
                                        Some(why) => text(ty::XS, t.text_tertiary).whitespace_normal().child(why),
                                        // The server confirmed it can; an older one can't say, so the need is stated.
                                        None if self.store.capabilities.is_some() => {
                                            text(ty::XS, t.text_tertiary).whitespace_normal().child(
                                                "Trains in f16 on this server's tensor cores: about 40% less GPU memory (at 640 px, not faster)",
                                            )
                                        }
                                        None => rich(
                                            &t,
                                            ty::XS,
                                            t.text_tertiary,
                                            "Trains in f16 on tensor cores; needs `folo serve --tensor-cores`",
                                        )
                                        .text_left(),
                                    },
                                ),
                        )
                        .child(switch)
                })
                .child(section("Augmentation"))
                .child(
                    div()
                        .flex()
                        .gap(px(16.))
                        .child(div().flex_1().child(valued("Mosaic", format!("{mosaic:.2}"), mosaic_slider)))
                        .child(div().flex_1().child(valued("Mixup", format!("{mixup:.2}"), mixup_slider))),
                )
                .child(
                    div()
                        .flex()
                        .gap(px(16.))
                        .child(div().flex_1().child(valued(
                            "Horizontal flip",
                            format!("{fliplr:.2}"),
                            flip_slider,
                        )))
                        .child(div().flex_1().child(valued(
                            "Scale jitter",
                            format!("±{scale:.2}"),
                            scale_slider,
                        ))),
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
        let images = ds.as_ref().map_or(0, |d| d.splits.train);
        let (secs, vram) = estimate(arch, epochs, BATCHES[batch], IMGSZ[imgsz], images as u32);
        // Mixed precision trains in about 0.58x the memory (folo-n batch 16: 2022 vs 3474 MiB,
        // docs/BENCHMARKS.md) and no faster at 640 px (the epoch is bound elsewhere), so only the
        // memory moves.
        let f16 = amp && self.amp_blocked(DEVICES[device]).is_none();
        let vram = if f16 { vram * 0.58 } else { vram };
        let cpu = DEVICES[device] == "cpu";
        let gpu = self.store.system.ready().and_then(|s| s.gpus.first().cloned());
        // 0 = not reported (wgpu can't on every platform): make no claim about fitting.
        // GiB, like the estimate (MiB of activations / 1024) and every size the app shows.
        let gpu_gb = gpu.as_ref().map_or(0.0, |g| g.memory_bytes as f32 / (1u64 << 30) as f32);
        let fits = cpu || gpu_gb <= 0.0 || vram <= gpu_gb * 0.97;
        let estimate_line = div()
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
                            .child(if cpu {
                                "CPU selected · no time estimate".to_string()
                            } else if fits {
                                format!("≈ {} · ≈ {vram:.1} GiB GPU memory", rough_duration(secs))
                            } else {
                                // No adapter name here: it can be long enough to reach the buttons.
                                format!("≈ {vram:.0} GiB needed, {gpu_gb:.0} GiB on this GPU")
                            }),
                    )
                    .child(text(ty::XS, t.text_tertiary).whitespace_nowrap().child(if cpu {
                        "Expect many times slower than a GPU"
                    } else if fits && f16 {
                        "Reference estimate in f16, not measured on this GPU"
                    } else if fits {
                        "Reference estimate, not measured on this GPU"
                    } else {
                        "Lower the batch or image size"
                    })),
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
            .map(|r| r.name.clone());
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
        let footer = div()
            .flex()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .px(px(20.))
            .py(px(14.))
            .border_t_1()
            .border_color(t.line_subtle)
            .bg(t.bg_raised)
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
            );
        self.dialog_frame(
            &t,
            "new-run",
            720.,
            "New training run",
            Some(match &busy_with {
                Some(n) => {
                    format!("{n} is training; this run queues and starts when it finishes.")
                }
                None => "Trains on this machine and streams metrics into the run dashboard.".into(),
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
                        .child(text(ty::XS, t.text_tertiary).child(help.to_string())),
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
        let footer = div()
            .flex()
            .justify_end()
            .gap(px(8.))
            .px(px(20.))
            .py(px(14.))
            .border_t_1()
            .border_color(t.line_subtle)
            .bg(t.bg_raised)
            .child(cancel)
            .child(go);
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
        let footer = div()
            .flex()
            .justify_end()
            .gap(px(8.))
            .px(px(20.))
            .py(px(14.))
            .border_t_1()
            .border_color(t.line_subtle)
            .bg(t.bg_raised)
            .child(cancel)
            .child(stop);
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
                    .child(text(ty::XS, t.text_secondary).child(label));
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

    fn inspector_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
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
        let focused = (0..meta.boxes.len()).find(|i| self.fh(format!("insp-box-{i}"), cx).is_focused(window));
        let hover = if self.kb { focused.or(hover) } else { hover.or(focused) };
        let classes = self.store.dataset_summary(&ds_id).map(|d| d.classes.clone()).unwrap_or_default();
        let img = self
            .image(&ds_id, &meta.id, ImgKind::Full, cx)
            .or_else(|| self.image(&ds_id, &meta.id, ImgKind::Thumb, cx));
        let dims = (meta.width as f32, meta.height as f32);
        let boxes: Vec<DrawBox> = meta
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| DrawBox {
                label: classes.get(b.class as usize).cloned().map(Into::into),
                highlight: hover == Some(i),
                dim: hover.is_some_and(|h| h != i),
                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
            })
            .collect();
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
        let box_rows: Vec<AnyElement> = box_order
            .into_iter()
            .map(|(i, b)| {
                let hot = hover == Some(i);
                let key = format!("insp-box-{i}");
                // The class column is 70 px: a name that may not fit shows in full on hover and on
                // keyboard focus (short ones need neither).
                let full_name = classes.get(b.class as usize).cloned().unwrap_or_default();
                let long = crate::ui::may_overflow(&full_name, ty::SM.size, 70.);
                let focused = long && self.ring(&self.fh(key.clone(), cx), window);
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .when(long, |d| d.tooltip(crate::controls::tip(full_name.clone(), None)))
                    .when(focused, |d| d.child(crate::controls::focus_tip(&t, full_name.clone(), 30.)))
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
                            .truncate()
                            .child(classes.get(b.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(
                        text(ty::XS, if hot { t.text_secondary } else { t.text_tertiary })
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
                    .child(
                        text(ty::BASE, t.text_primary)
                            .font_family(crate::theme::MONO)
                            .truncate()
                            .child(meta.file.clone()),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(6.))
                            .child(badge(&t, meta.split.label(), Tone::Neutral))
                            .child(badge(&t, format!("{} × {}", meta.width, meta.height), Tone::Neutral))
                            .child(badge(&t, format!("{} boxes", meta.boxes.len()), Tone::Neutral)),
                    ),
            )
            .child(
                div()
                    .px(px(16.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(&t, "Labels"))
                    .child(text(ty::XS, t.text_tertiary).child("top-left x, y · w×h (px)")),
            )
            .child(
                div()
                    .id("insp-boxes")
                    // As tall as its rows (up to the room left), so the key hints sit right under it.
                    .flex_shrink()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.insp_list_scroll)
                    .px(px(8.))
                    .children(box_rows)
                    .when(meta.boxes.is_empty(), |d| {
                        d.child(
                            div()
                                .p(px(8.))
                                .child(text(ty::SM, t.text_tertiary).child("No labels: a background image.")),
                        )
                    }),
            )
            .child(
                // One action per row on a fixed key column, so the keys line up and nothing wraps.
                {
                    let row = |keys: AnyElement, what: &'static str| {
                        div()
                            .flex()
                            .items_center()
                            .child(div().w(px(96.)).flex_none().flex().items_center().gap(px(4.)).child(keys))
                            .child(text(ty::XS, t.text_tertiary).child(what))
                    };
                    div()
                        .p(px(12.))
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .border_t_1()
                        .border_color(t.line_subtle)
                        .child(row(
                            shortcut(&t, &["←", "→"]).into_any_element(),
                            "previous / next · pan when zoomed",
                        ))
                        .child(row(shortcut(&t, &["PgUp", "PgDn"]).into_any_element(), "previous / next"))
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
                        .child(row(kbd(&t, "Drag").into_any_element(), "pan"))
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
            .child(text(ty::SM, t.text_secondary).tnum().child(format!(
                "{} of {}",
                int(index as u64 + 1),
                int(n as u64)
            )))
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
