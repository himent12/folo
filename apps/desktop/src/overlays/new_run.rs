//! The New run dialog: name and dataset, where it trains (`new_run_compute`), the model, the
//! schedule and the Advanced group. Its footer and frame are `new_run_footer`'s.

use std::rc::Rc;

use crate::api::SplitsExt;
use folo_ui::Focusable;
use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{ARCHS, BATCHES, DEVICES, FoloApp, IMGSZ, act};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

use super::error_note;
use super::new_run_footer::NewRunSummary;

/// The optimizer segments' labels (the helper under them repeats the chosen one).
const OPT_LABELS: [&str; 4] = ["Auto", "SGD", "AdamW", "MuSGD"];

impl FoloApp {
    pub(super) fn new_run_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
        let opt_unknown = f.optimizer_unknown;
        let ds = datasets.get(ds_i).cloned();
        // Advanced: what differs from a fresh form's defaults, so a collapsed group still says
        // what it will do. Open while one of its fields needs fixing.
        let adv_summary = {
            let mut changed: Vec<String> = Vec::new();
            if f.e2e {
                changed.push("end-to-end".into());
            }
            if opt_unknown {
                changed.push("unknown optimizer".into());
            } else if opt != 0 {
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
        let adv_open = f.advanced || (f.lr_invalid && opt != 0) || opt_unknown;
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
            Rc::new(|this, i, _, cx| {
                if let Some(f) = this.new_run.as_mut() {
                    f.dataset = i;
                    this.resuggest_run_name(cx);
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
            .child(self.segmented_fill(
                "nr-arch",
                &["n", "s", "m", "l", "x"],
                arch,
                window,
                cx,
                Rc::new(|this, i, _, cx| {
                    if let Some(f) = this.new_run.as_mut() {
                        f.arch = i;
                        // The suggested name carries the architecture's letter.
                        this.resuggest_run_name(cx);
                    }
                }),
            ))
            .into_any_element();
        let batch_labels: Vec<String> = BATCHES.iter().map(|b| b.to_string()).collect();
        let batch_refs: Vec<&str> = batch_labels.iter().map(String::as_str).collect();
        let batch_seg = seg(self, "nr-batch", &batch_refs, batch, window, cx, |f, i| f.batch = i);
        let size_labels: Vec<String> = IMGSZ.iter().map(|b| b.to_string()).collect();
        let size_refs: Vec<&str> = size_labels.iter().map(String::as_str).collect();
        let size_seg = seg(self, "nr-imgsz", &size_refs, imgsz, window, cx, |f, i| f.imgsz = i);
        let device_seg =
            seg(self, "nr-device", &["Auto", "GPU 0", "CPU"], device, window, cx, |f, i| f.device = i);
        // An unknown prefilled optimizer selects no segment.
        let opt_sel = if opt_unknown { OPT_LABELS.len() } else { opt };
        let opt_seg = seg(self, "nr-opt", &OPT_LABELS, opt_sel, window, cx, |f, i| f.pick_optimizer(i));
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
                          input: folo_ui::Entity<crate::input::TextInput>,
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
                            && !f.optimizer_unknown
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
                        .w(folo_ui::relative(0.5))
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
                                _ if opt_unknown => "Unknown optimizer (from a newer FOLO): pick one".to_string(),
                                (0, Some((kind, _))) => {
                                    format!("Auto: {kind} for this run, at its own learning rate")
                                }
                                (0, None) => "Auto: MuSGD for long runs, else AdamW".to_string(),
                                (1, _) => "SGD with Nesterov momentum 0.937".to_string(),
                                (2, _) => "AdamW with weight decay".to_string(),
                                _ => "MuSGD: SGD plus a Muon update on conv weights".to_string(),
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
                        .w(folo_ui::relative(0.5))
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
        let form = NewRunSummary {
            ds,
            cloud,
            cloud_offer,
            arch,
            epochs,
            batch,
            imgsz,
            device,
            amp,
            amp_why,
            epochs_bad,
            lr_bad,
            close_bad,
            submitting,
            opt_unknown,
        };
        self.frame_new_run(t, body, form, window, cx)
    }
}
