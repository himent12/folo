//! Overlays drawn above the page: command palette, dialogs, the image inspector and toasts.

use std::rc::Rc;

use gpui::Focusable;
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, MouseButton,
    MouseDownEvent, MouseMoveEvent, ParentElement, ScrollWheelEvent, SharedString,
    StatefulInteractiveElement, Styled, Window, canvas, div, px,
};

use crate::api::RunStatus;
use crate::app::{ARCHS, Action, BATCHES, DEVICES, FoloApp, IMGSZ, OPTIMIZERS, Route, act};
use crate::controls::Btn;
use crate::store::ImgKind;
use crate::theme::{Theme, alpha, radius, theme, ty};
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

/// Rough training cost of a configuration on this machine: (seconds, VRAM GB).
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
                icon: "icons/square.svg",
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
                    s => format!("{} · {}", s.label().to_lowercase(), r.config.arch),
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
                let detail = m
                    .metrics
                    .as_ref()
                    .map_or(m.arch.clone(), |x| format!("{} · mAP {:.3}", m.arch, x.map50_95));
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
                    detail: format!("{} images · {}", int(d.splits.total() as u64), d.format),
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
        let selected = p.selected.min(n.saturating_sub(1));
        let input = p.input.clone();
        let actions: Vec<Action> = items.iter().map(|i| i.action.clone()).collect();
        let scroll = p.scroll.clone();
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
                    .h(px(36.))
                    .px(px(10.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .rounded(px(radius::MD))
                    .cursor_pointer()
                    .when(on, |d| d.bg(t.bg_active))
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
                    .child(icon(it.icon, 16., if on { t.text_primary } else { t.text_tertiary }))
                    .child(text(ty::BASE, t.text_primary).flex_none().child(it.title))
                    .child(text(ty::SM, t.text_tertiary).flex_1().min_w_0().truncate().child(it.detail))
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
                    .child(list)
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
                            .child(text(ty::XS, t.text_tertiary).tnum().child(format!("{n} results"))),
                    ),
            );
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(self.scrim(&t, "palette-scrim", cx, act(|this, w, cx| this.close_palette(w, cx))))
            .child(panel_el)
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
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(self.scrim(t, "dialog-scrim", cx, on_close))
            .child(
                div().absolute().top_0().left_0().size_full().flex().items_center().justify_center().child(
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
                                        .child(text(ty::LG, t.text_primary).child(title.to_string()))
                                        .when_some(subtitle, |d, s| {
                                            d.child(text(ty::SM, t.text_secondary).child(s))
                                        }),
                                )
                                .child(close_btn),
                        )
                        .child(div().id("dialog-body").flex_1().min_h_0().overflow_y_scroll().child(body))
                        .child(footer),
                ),
            )
            .into_any_element()
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
            .border_color(if name_focused { t.accent_line } else { t.line_default })
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
                        Some(SharedString::from(format!("{} images", int(d.splits.total() as u64)))),
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
        let arch_seg = seg(self, "nr-arch", &["n", "s", "m", "l", "x"], arch, window, cx, |f, i| f.arch = i);
        let batch_labels: Vec<String> = BATCHES.iter().map(|b| b.to_string()).collect();
        let batch_refs: Vec<&str> = batch_labels.iter().map(String::as_str).collect();
        let batch_seg = seg(self, "nr-batch", &batch_refs, batch, window, cx, |f, i| f.batch = i);
        let size_labels: Vec<String> = IMGSZ.iter().map(|b| b.to_string()).collect();
        let size_refs: Vec<&str> = size_labels.iter().map(String::as_str).collect();
        let size_seg = seg(self, "nr-imgsz", &size_refs, imgsz, window, cx, |f, i| f.imgsz = i);
        let device_seg =
            seg(self, "nr-device", &["Auto", "GPU 0", "CPU"], device, window, cx, |f, i| f.device = i);
        let opt_seg = seg(self, "nr-opt", &["Auto", "SGD", "AdamW"], opt, window, cx, |f, i| f.optimizer = i);
        let epochs_slider = self.slider("nr-epochs", epochs, 10., 300., 5., window, cx).into_any_element();
        let lr_slider = self.slider("nr-lr0", lr0, 0.001, 0.05, 0.001, window, cx).into_any_element();
        let mosaic_slider = self.slider("nr-mosaic", mosaic, 0., 1., 0.05, window, cx).into_any_element();
        let mixup_slider = self.slider("nr-mixup", mixup, 0., 0.5, 0.05, window, cx).into_any_element();
        let scale = self.new_run.as_ref().map_or(0.5, |f| f.scale);
        let scale_slider = self.slider("nr-scale", scale, 0., 0.9, 0.05, window, cx).into_any_element();
        let note = self.new_run.as_ref().and_then(|f| f.note.clone());
        let flip_slider = self.slider("nr-fliplr", fliplr, 0., 1., 0.05, window, cx).into_any_element();
        let close_mosaic = self.new_run.as_ref().map_or(10., |f| f.close_mosaic);
        let close_slider = self.slider("nr-close", close_mosaic, 0., 30., 1., window, cx).into_any_element();
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
        let section = |title: &str| div().pt(px(4.)).child(overline(&t, title));
        let body = div()
            .flex()
            .flex_col()
            .gap(px(18.))
            .px(px(20.))
            .pb(px(20.))
            .child(
                div().flex().gap(px(16.)).child(div().flex_1().child(field("Name", name_box, None))).child(
                    div().flex_1().child(field(
                        "Dataset",
                        ds_select,
                        ds.as_ref().map(|d| {
                            format!(
                                "{} classes · train {} · val {}",
                                d.classes.len(),
                                int(d.splits.train as u64),
                                int(d.splits.val as u64)
                            )
                        }),
                    )),
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
                        Some(format!("{} · {}", ARCHS[arch], PARAMS[arch])),
                    )))
                    .child(div().flex_1().child(field(
                        "Image size",
                        size_seg,
                        Some("pixels · images are letterboxed to a square".into()),
                    ))),
            )
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(field(
                        "Batch size",
                        batch_seg,
                        Some("images per step".into()),
                    )))
                    .child(div().flex_1().child(field(
                        "Device",
                        device_seg,
                        Some(if DEVICES[device] == "auto" {
                            "GPU when available, else CPU".into()
                        } else {
                            DEVICES[device].to_string()
                        }),
                    ))),
            )
            .child(section("Schedule"))
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(valued("Epochs", format!("{epochs:.0}"), epochs_slider)))
                    .child(div().flex_1().child(valued(
                        "Initial learning rate",
                        format!("{lr0:.3}"),
                        lr_slider,
                    ))),
            )
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().child(field(
                        "Optimizer",
                        opt_seg,
                        Some(format!("{} · warmup 3 epochs, then cosine decay", OPTIMIZERS[opt])),
                    )))
                    .child(
                        div().flex_1().child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(8.))
                                .child(valued(
                                    "Close mosaic",
                                    if close_mosaic < 0.5 {
                                        "off".to_string()
                                    } else {
                                        format!("last {close_mosaic:.0} epochs")
                                    },
                                    close_slider,
                                ))
                                .child(
                                    text(ty::XS, t.text_tertiary)
                                        .child("Train the final epochs on plain images"),
                                ),
                        ),
                    ),
            )
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
            })
            .when_some(error.clone(), |d, e| {
                d.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .p(px(10.))
                        .rounded(px(radius::MD))
                        .bg(t.danger_muted)
                        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
                        .child(text(ty::SM, t.danger_text).child(e.message)),
                )
            });
        let images = ds.as_ref().map_or(0, |d| d.splits.train);
        let (secs, vram) = estimate(arch, epochs, BATCHES[batch], IMGSZ[imgsz], images);
        let fits = vram <= 15.5 || DEVICES[device] == "cpu";
        let estimate_line = div()
            .flex()
            .items_center()
            .gap(px(8.))
            .child(icon(
                if fits { "icons/timer.svg" } else { "icons/triangle-alert.svg" },
                14.,
                if fits { t.text_tertiary } else { t.warning_solid },
            ))
            .child(text(ty::SM, if fits { t.text_secondary } else { t.warning_text }).tnum().child(
                if fits {
                    format!(
                        "≈ {} on RTX 4080 SUPER · ≈ {vram:.1} GB of GPU memory",
                        crate::api::time::duration(secs)
                    )
                } else {
                    format!(
                        "≈ {vram:.0} GB of GPU memory needed, 16 GB available · lower the batch or image size"
                    )
                },
            ));
        let cancel = self.button(
            "nr-cancel",
            "Cancel",
            Btn::Ghost,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_new_run(w, cx)),
        );
        let start = self.button(
            "nr-start",
            if submitting { "Starting…" } else { "Start training" },
            Btn::Primary,
            Some("icons/play.svg"),
            window,
            cx,
            act(|this, w, cx| this.submit_new_run(w, cx)),
        );
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
            .child(estimate_line)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(cancel)
                    .child(start)
                    .child(shortcut(&t, &["Ctrl", "Enter"])),
            );
        self.dialog_frame(
            &t,
            "new-run",
            720.,
            "New training run",
            Some("Trains on this machine and streams metrics into the run dashboard.".into()),
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_new_run(w, cx)),
        )
    }

    fn stop_dialog(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let run = self.store.run(id).cloned();
        let name = run.as_ref().map_or(id.to_string(), |r| r.name.clone());
        let epoch = run.as_ref().map_or(0, |r| r.epoch);
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
            Btn::Danger,
            Some("icons/square.svg"),
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
                    .child(format!("Training stops after this step. The epoch {epoch} checkpoint is kept, so you can export it or train on from it.")),
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
            act(|this, _, cx| {
                this.confirm_stop = None;
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
        div()
            .absolute()
            .bottom(px(20.))
            .right(px(24.))
            .w(px(380.))
            .flex()
            .flex_col()
            .gap(px(8.))
            .children(toasts.into_iter().map(|(id, tone, title, body, action)| {
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
                div()
                    .id(("toast", id as usize))
                    .occlude()
                    .flex()
                    .items_start()
                    .gap(px(10.))
                    .p(px(12.))
                    .rounded(px(radius::LG))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
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
                                text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).child(title),
                            )
                            .when_some(body, |d, b| d.child(text(ty::SM, t.text_secondary).child(b)))
                            .when_some(action_btn, |d, b| d.child(div().pt(px(8.)).flex().child(b))),
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
        // A keyboard-focused label row highlights its box like the pointer does.
        let hover = hover.or_else(|| {
            (0..meta.boxes.len()).find(|i| self.fh(format!("insp-box-{i}"), cx).is_focused(window))
        });
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
        let bounds_cell: Rc<std::cell::Cell<Option<gpui::Bounds<gpui::Pixels>>>> =
            Rc::new(std::cell::Cell::new(None));
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let down_cell = bounds_cell.clone();
        let paint_boxes = boxes.clone();
        let hit_boxes = boxes.clone();
        let tt = t;
        let img2 = img.clone();
        let stage =
            div()
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
                        cx.notify();
                    }
                }))
                .child(
                    canvas(
                        |_, _, _| (),
                        move |b, _, window, cx| {
                            let inner = gpui::Bounds::new(
                                b.origin + gpui::point(px(24.), px(24.)),
                                b.size - gpui::size(px(48.), px(24. + 72.)),
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
                )
                .child(
                    // Zoom controls.
                    div()
                        .absolute()
                        .bottom(px(16.))
                        .left(px(16.))
                        .flex()
                        .items_center()
                        .gap(px(2.))
                        .p(px(3.))
                        .rounded(px(radius::MD))
                        .bg(alpha(t.bg_overlay, 0.92))
                        .border_1()
                        .border_color(t.line_default)
                        .child(self.icon_button(
                            "insp-zoom-out",
                            "icons/zoom-out.svg",
                            "Zoom out",
                            window,
                            cx,
                            act(|this, _, cx| this.zoom_by(1.0 / 1.4, cx)),
                        ))
                        .child(
                            text(ty::XS, t.text_secondary).tnum().w(px(44.)).text_center().child(
                                if zoom <= 1.0 { "Fit".to_string() } else { format!("{:.1}×", zoom) },
                            ),
                        )
                        .child(self.icon_button(
                            "insp-zoom-in",
                            "icons/zoom-in.svg",
                            "Zoom in",
                            window,
                            cx,
                            act(|this, _, cx| this.zoom_by(1.4, cx)),
                        ))
                        .child(div().w(px(1.)).h(px(16.)).bg(t.line_subtle))
                        .child(self.icon_button(
                            "insp-fit",
                            "icons/maximize.svg",
                            "Fit (0)",
                            window,
                            cx,
                            act(|this, _, cx| {
                                if let Some(i) = this.inspector.as_mut() {
                                    i.zoom = 1.0;
                                    i.pan = (0., 0.);
                                }
                                cx.notify();
                            }),
                        )),
                );
        let box_rows: Vec<AnyElement> = meta
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let hot = hover == Some(i);
                let key = format!("insp-box-{i}");
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .when(hot, |d| d.bg(t.bg_active))
                    .hover(|s| s.bg(t.bg_hover))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            let next = if *h { Some(i) } else { None };
                            if insp.hover_box != next {
                                insp.hover_box = next;
                                cx.notify();
                            }
                        }
                    }))
                    .child(swatch(t.class(b.class), 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .truncate()
                            .child(classes.get(b.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(
                        text(ty::XS, t.text_tertiary)
                            .font_family(crate::theme::MONO)
                            .flex_1()
                            .text_right()
                            .child(format!("{:.0},{:.0} {:.0}×{:.0}", b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)),
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
            .child(div().px(px(16.)).pt(px(12.)).pb(px(4.)).child(overline(&t, "Labels")))
            .child(
                div()
                    .id("insp-boxes")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
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
                div()
                    .p(px(12.))
                    .flex()
                    .flex_wrap()
                    .gap(px(10.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(5.))
                            .child(shortcut(&t, &["←", "→"]))
                            .child(text(ty::XS, t.text_tertiary).child("browse")),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(5.))
                            .child(kbd(&t, "Scroll"))
                            .child(text(ty::XS, t.text_tertiary).child("zoom")),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(5.))
                            .child(kbd(&t, "Drag"))
                            .child(text(ty::XS, t.text_tertiary).child("pan")),
                    ),
            );
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
            .child(text(ty::SM, t.text_tertiary).font_family(crate::theme::MONO).child(meta.id.clone()))
            .child(close);
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
                        match ev.keystroke.key.as_str() {
                            "left" => this.step_inspector(-1, cx),
                            "right" => this.step_inspector(1, cx),
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
            )
            .into_any_element()
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
