//! A model's Details panel: the facts about it in three columns (Studio's `DetailsList.tsx`): what it
//! is, how it was trained and stored, where it came from, what it was evaluated on, its id, and its
//! architecture's measured and reference latency.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, StatefulInteractiveElement, Styled,
    Window, div, px,
};

use crate::api::{InferThroughput, Model, ModelEval, ModelSource, Run, RunStatus, ThroughputSource, time};
use crate::app::{FoloApp, Route, act};
use crate::store::Load;
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;

/// Maximum classes previewed inline before collapsing behind a "Show all" toggle.
pub(super) const CLASS_PREVIEW_CAP: usize = 10;

/// Formats the model's classes for the compact Details row (first [`CLASS_PREVIEW_CAP`] plus
/// `+{more} more` when collapsed; count alone when expanded into the full-width band below).
pub(super) fn classes_summary(classes: &[String], expanded: bool) -> String {
    if classes.is_empty() {
        return "0".to_string();
    }
    let more = classes.len().saturating_sub(CLASS_PREVIEW_CAP);
    if expanded && more > 0 {
        return format!("{} classes", int(classes.len() as u64));
    }
    let shown: Vec<&str> = classes.iter().take(CLASS_PREVIEW_CAP).map(String::as_str).collect();
    format!(
        "{} · {}{}",
        int(classes.len() as u64),
        shown.join(", "),
        if more > 0 { format!(" +{} more", int(more as u64)) } else { String::new() }
    )
}

/// The epochs a run finished, None when not known (Studio's `finishedEpochs`,
/// web/src/routes/model/weightsEpoch.ts). A `completed` or `stopped` run's `epoch` is the server's
/// count, brought up to its `weights/last.folo` after a crash between that checkpoint and its
/// history line, so it is the authority (a completed run may also have stopped early, patience).
/// Any other run's `epoch` may be one in progress or the one it failed in: its recorded history's
/// highest epoch (`history_top`, None while not loaded, `Some(0)` for an empty one) counts, but
/// only when `epoch` is not past it.
pub(super) fn finished_epochs(status: RunStatus, epoch: u32, history_top: Option<u32>) -> Option<u32> {
    if matches!(status, RunStatus::Completed | RunStatus::Stopped) {
        return (epoch > 0).then_some(epoch);
    }
    history_top.filter(|&n| n > 0 && epoch <= n)
}

/// The Details' "Trained for": the epochs the model's run finished, of the planned ones when it
/// ended early (`trained_with.epochs` is the planned count): "50 epochs at 640 px", "1 of 50
/// epochs at 320 px"; while that is not known, only the plan, labelled so: "50 epochs planned at
/// 320 px" (s4-models-1, as Studio's `trainedFor`).
pub(super) fn trained_for(m: &Model, run: Option<&Run>, done: Option<u32>) -> Option<String> {
    let t = m.trained_with?;
    let planned = t.epochs;
    let unit = if planned == 1 { "epoch" } else { "epochs" };
    let ran = done.filter(|_| run.is_some_and(|r| m.run_id.as_deref() == Some(r.id.as_str())));
    Some(match ran {
        None => format!("{} {unit} planned at {} px", int(planned as u64), t.imgsz),
        Some(n) if n < planned => {
            format!("{} of {} {unit} at {} px", int(n as u64), int(planned as u64), t.imgsz)
        }
        Some(_) => format!("{} {unit} at {} px", int(planned as u64), t.imgsz),
    })
}

/// Which epoch a run model's weights are from (docs/API.md, Models): `<run>-best` the run's best
/// epoch, `<run>-last` its last finished one ("last finished epoch, 2 of 50" for a run that ended
/// before its planned end; `done` is [`finished_epochs`]); when they are the same epoch, each says
/// so. None without the run, or when the server didn't say which epoch was best.
pub(super) fn weights_epoch(m: &Model, run: Option<&Run>, done: Option<u32>) -> Option<String> {
    let run = run.filter(|r| m.run_id.as_deref() == Some(r.id.as_str()))?;
    let best = run.best.as_ref().map(|b| b.epoch);
    if m.id.ends_with("-last") {
        let Some(done) = done else { return Some("last finished epoch".into()) };
        let same = if best == Some(done) { ", same weights as best" } else { "" };
        let which = if done >= run.epochs { "last epoch" } else { "last finished epoch" };
        return Some(format!("{which}, {done} of {}{same}", run.epochs));
    }
    let best = best?;
    let same = if done == Some(best) { ", same weights as last" } else { "" };
    Some(format!("best epoch {best} of {}{same}", run.epochs))
}

/// This server's measured end-to-end latency for `arch` at `imgsz` on its default device (a row of
/// its own adapter, the one with the most samples), as Studio picks it.
fn measured_latency<'a>(t: &'a crate::api::ThroughputTable, m: &Model) -> Option<&'a InferThroughput> {
    t.inference
        .iter()
        .filter(|r| {
            r.source == ThroughputSource::Measured
                && r.arch == m.arch
                && r.imgsz == m.imgsz
                && r.device == t.device
                && r.measured_on == t.device_name
        })
        .max_by_key(|r| r.samples)
}

/// The reference latency for the architecture (docs/BENCHMARKS.md, as the table carries it): at
/// 640 px, f16 first.
fn reference_latency<'a>(t: &'a crate::api::ThroughputTable, m: &Model) -> Option<&'a InferThroughput> {
    t.inference
        .iter()
        .filter(|r| r.source == ThroughputSource::Reference && r.arch == m.arch && r.imgsz == 640)
        .min_by_key(|r| r.precision != crate::api::InferPrecision::F16)
}

/// "5.90 ms" / "12.3 ms": two decimals under 10 ms.
fn ms(x: f64) -> String {
    if x < 10. { format!("{x:.2} ms") } else { format!("{x:.1} ms") }
}

/// A fact's value: wraps under its key's line rather than cut.
fn value(t: &Theme, s: impl Into<String>) -> folo_ui::Div {
    text(ty::SM, t.text_primary).w_full().tnum().text_right().whitespace_normal().child(s.into())
}

/// A fact: its key, and its value right-aligned in the rest of the line (a width it wraps in).
fn fact_row(t: &Theme, key: &str, v: AnyElement) -> folo_ui::Div {
    div()
        .flex()
        .items_start()
        .gap(px(12.))
        .py(px(5.))
        .child(text(ty::SM, t.text_tertiary).flex_none().child(key.to_string()))
        .child(div().flex_1().min_w_0().flex().justify_end().child(v))
}

/// A fact with a quieter second part ("9.42 M" then "(9,420,000)").
fn value_with(t: &Theme, s: impl Into<String>, faint: impl Into<String>) -> folo_ui::Div {
    div()
        .w_full()
        .flex()
        .flex_wrap()
        .justify_end()
        .gap_x(px(4.))
        .child(text(ty::SM, t.text_primary).tnum().child(s.into()))
        .child(text(ty::SM, t.text_tertiary).tnum().whitespace_normal().child(faint.into()))
}

impl FoloApp {
    /// A compact clickable link inside the Details panel, navigating to a related run, dataset or
    /// parent model.
    fn details_link(
        &self,
        t: &Theme,
        key: &'static str,
        label: String,
        route: Route,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let el = div()
            .id(key)
            .max_w_full()
            .min_w_0()
            .px(px(4.))
            .mx(px(-4.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .flex()
            .items_center()
            .gap(px(3.))
            .child(text(ty::SM, t.text_primary).min_w_0().ellipsis().child(label))
            .child(icon("icons/arrow-up-right.svg", 11., t.text_tertiary));
        self.bind(
            el,
            key,
            radius::SM,
            window,
            cx,
            act(move |this, w, cx| this.navigate(route.clone(), w, cx)),
        )
        .into_any_element()
    }

    pub(super) fn details_panel(
        &mut self,
        m: &Model,
        ev: Option<&ModelEval>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // The architecture's latency comes from the throughput table (Settings › Throughput's).
        if matches!(self.throughput, Load::Idle) {
            self.load_throughput(cx);
        }
        let run = m.run_id.as_ref().and_then(|r| self.store.run(r)).cloned();
        // How many epochs its run finished: a run that ended (completed, stopped) says it; any
        // other one's recorded history does (fetched for it).
        let done = run.as_ref().and_then(|r| {
            let ended = matches!(r.status, RunStatus::Completed | RunStatus::Stopped);
            if !ended {
                self.ensure_run_data(&r.id, cx);
            }
            let top = self
                .store
                .run_data
                .get(&r.id)
                .and_then(|l| l.ready())
                .map(|d| d.metrics.epochs.iter().map(|e| e.epoch).max().unwrap_or(0));
            finished_epochs(r.status, r.epoch, top)
        });
        let mut facts: Vec<(&str, AnyElement)> = Vec::new();
        let mut fact = |k: &'static str, v: AnyElement| facts.push((k, v));
        let model_arch = self.store.model_arch(m);
        fact(
            "Architecture",
            div()
                .flex()
                .items_center()
                .justify_end()
                .gap(px(6.))
                .child(value(&t, model_arch.as_str()))
                .when(model_arch.is_v2(), |d| d.child(badge(&t, "v2", Tone::Accent)))
                .into_any_element(),
        );
        fact("Input size", value(&t, format!("{} × {} px", m.imgsz, m.imgsz)).into_any_element());
        let (p, pu) = count_si(m.params);
        fact(
            "Parameters",
            value_with(&t, format!("{p} {pu}"), format!("({})", int(m.params))).into_any_element(),
        );
        fact("Compute", value(&t, format!("{:.1} GFLOPs", m.flops_g)).into_any_element());
        // The e2e tag's help moved here from beside the name: the head is a fact of the model.
        let e2e = self.model_e2e(m);
        fact(
            "Head",
            div()
                .w_full()
                .flex()
                .items_center()
                .justify_end()
                .gap(px(4.))
                .child(
                    text(ty::SM, t.text_primary)
                        .flex_1()
                        .min_w_0()
                        .text_right()
                        .whitespace_normal()
                        .child(if e2e { "end-to-end (no NMS)" } else { "one-to-many + NMS" }),
                )
                .when(e2e, |d| d.child(self.help_icon("model-head-help", FoloApp::E2E_HELP, window, cx)))
                .into_any_element(),
        );
        // How it was trained, from the file's training record (none for exports and imports).
        if let Some(w) = m.trained_with {
            fact(
                "Trained in",
                value(&t, if w.amp { "mixed precision (f16)" } else { "f32" }).into_any_element(),
            );
            if let Some(s) = trained_for(m, run.as_ref(), done) {
                fact("Trained for", value(&t, s).into_any_element());
            }
        }
        if let Some(w) = weights_epoch(m, run.as_ref(), done) {
            fact("Weights from", value(&t, w).into_any_element());
        }
        if let Some(s) = model_storage(m) {
            fact("Stored as", value(&t, s).into_any_element());
        }
        let (b, bu) = bytes(m.bytes);
        // A training checkpoint also holds the optimizer state: say why it is bigger than an export.
        let checkpoint = m.trained_with.is_some() && m.fused != Some(true);
        fact(
            "File size",
            if checkpoint {
                value_with(&t, format!("{b} {bu}"), "training checkpoint").into_any_element()
            } else {
                value(&t, format!("{b} {bu} · .folo")).into_any_element()
            },
        );
        let more = m.classes.len().saturating_sub(CLASS_PREVIEW_CAP);
        let classes_expanded = more > 0 && self.model_classes_expanded.contains(&m.id);
        let mut class_actions: Vec<AnyElement> = Vec::new();
        if more > 0 {
            let mid = m.id.clone();
            let toggle_label = if classes_expanded {
                "Show fewer".to_string()
            } else {
                format!("Show all {}", int(m.classes.len() as u64))
            };
            let el = div()
                .id("model-classes-toggle")
                .px(px(4.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .active(|s| s.bg(t.bg_active))
                .child(inline_link(&t, toggle_label));
            class_actions.push(
                self.bind(
                    el,
                    "model-classes-toggle",
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        if !this.model_classes_expanded.remove(&mid) {
                            this.model_classes_expanded.insert(mid.clone());
                        }
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        if !m.classes.is_empty() {
            let classes_text = m.classes.join(", ");
            let class_count = m.classes.len();
            let el = div()
                .id("model-classes-copy")
                .px(px(4.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .active(|s| s.bg(t.bg_active))
                .tooltip(crate::controls::tip("Copy all class names", None))
                .flex()
                .items_center()
                .gap(px(4.))
                .child(icon("icons/copy.svg", 12., t.text_secondary))
                .child(text(ty::SM, t.text_secondary).child("Copy"));
            class_actions.push(
                self.bind(
                    el,
                    "model-classes-copy",
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(classes_text.clone()));
                        this.toast(
                            Tone::Neutral,
                            format!("Copied {}", counted(class_count as u64, "class", "classes")),
                            None,
                            cx,
                        );
                    }),
                )
                .into_any_element(),
            );
        }
        fact(
            "Classes",
            div()
                .w_full()
                .flex()
                .flex_col()
                .items_end()
                .gap(px(4.))
                .child(value(&t, classes_summary(&m.classes, classes_expanded)))
                .when(!class_actions.is_empty(), |d| {
                    d.child(div().flex().items_center().justify_end().gap(px(6.)).children(class_actions))
                })
                .into_any_element(),
        );
        let ds = run.as_ref().map(|r| self.store.run_dataset_name(r));
        let source_val: AnyElement = match (&run, m.source) {
            (Some(r), _) => {
                let run_link = self.details_link(
                    &t,
                    "model-details-run-link",
                    r.name.clone(),
                    Route::Run(r.id.clone()),
                    window,
                    cx,
                );
                match ds.filter(|d| !d.is_empty()) {
                    Some(d) => {
                        let ds_el: AnyElement = if self.store.dataset_summary(&r.config.dataset_id).is_some()
                        {
                            self.details_link(
                                &t,
                                "model-details-train-ds-link",
                                d,
                                Route::Dataset(r.config.dataset_id.clone()),
                                window,
                                cx,
                            )
                        } else {
                            text(ty::SM, t.text_tertiary).whitespace_normal().child(d).into_any_element()
                        };
                        div()
                            .w_full()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_end()
                            .gap_x(px(4.))
                            .child(run_link)
                            .child(text(ty::SM, t.text_tertiary).child("on"))
                            .child(ds_el)
                            .into_any_element()
                    }
                    None => div().w_full().flex().justify_end().child(run_link).into_any_element(),
                }
            }
            (None, Some(ModelSource::Published)) => match &m.about {
                Some(a) => value_with(&t, "Published weights", a.clone()).into_any_element(),
                None => value(&t, "Published weights").into_any_element(),
            },
            (None, _) if m.run_id.is_some() => {
                value(&t, m.run_id.clone().unwrap_or_default()).into_any_element()
            }
            _ => value(&t, "Imported .folo file").into_any_element(),
        };
        fact("Source", source_val);
        // What the curves, the class table and the matrix were measured on.
        let evaluated = ev
            .and_then(|e| {
                e.dataset_name.clone().zip(e.split.map(|s| s.as_str())).map(|(d, s)| {
                    let id = e
                        .dataset_id
                        .as_deref()
                        .filter(|id| self.store.dataset_summary(id).is_some())
                        .map(str::to_string);
                    (id, d, s, Some(e.images as u64))
                })
            })
            .or_else(|| {
                run.as_ref().and_then(|r| self.store.dataset_summary(&r.config.dataset_id)).map(|d| {
                    (
                        Some(d.id.clone()),
                        d.name.clone(),
                        if d.splits.val == 0 { "train" } else { "val" },
                        ev.map(|e| e.images as u64),
                    )
                })
            });
        if let Some((ds_id, d, s, n)) = evaluated {
            let label = split_label(&d, s, n);
            let el = match ds_id {
                Some(id) => div()
                    .w_full()
                    .flex()
                    .justify_end()
                    .child(self.details_link(
                        &t,
                        "model-details-eval-ds-link",
                        label,
                        Route::Dataset(id),
                        window,
                        cx,
                    ))
                    .into_any_element(),
                None => value(&t, label).into_any_element(),
            };
            fact("Evaluated on", el);
        }
        // What the run started from: a model by name (clickable when still present), or nothing.
        if let Some(r) = &run {
            let el = match r.config.weights.as_deref() {
                None => value(&t, "from scratch (random weights)").into_any_element(),
                Some(w) => match self.store.model(w) {
                    Some(p) if p.id != m.id => div()
                        .w_full()
                        .flex()
                        .justify_end()
                        .child(self.details_link(
                            &t,
                            "model-details-parent-link",
                            p.name.clone(),
                            Route::Model(p.id.clone()),
                            window,
                            cx,
                        ))
                        .into_any_element(),
                    Some(p) => value(&t, p.name.clone()).into_any_element(),
                    None => value(&t, w.to_string()).into_any_element(),
                },
            };
            fact("Started from", el);
        }
        fact(
            "Model id",
            text(ty::SM, t.text_secondary)
                .font_family(crate::theme::MONO)
                .child(m.id.clone())
                .into_any_element(),
        );
        let created = time::parse(&m.created).map_or(m.created.clone(), time::local_datetime);
        fact("Created", value(&t, created).into_any_element());
        // The architecture's number, never this model's own timing: measured on this server where it
        // has predicted at the model's size, and the reference (docs/BENCHMARKS.md).
        if let Some(tp) = self.throughput.ready() {
            if let Some(h) = measured_latency(tp, m) {
                fact(
                    "Measured latency",
                    value_with(
                        &t,
                        ms(h.latency_ms),
                        format!(
                            "{}–{} · {} predictions · {} · this server",
                            ms(h.latency_ms_low).trim_end_matches(" ms"),
                            ms(h.latency_ms_high),
                            int(h.samples),
                            if h.precision == crate::api::InferPrecision::F16 { "f16" } else { "f32" }
                        ),
                    )
                    .into_any_element(),
                );
            }
            fact(
                "Reference latency",
                match reference_latency(tp, m) {
                    Some(r) => value_with(
                        &t,
                        ms(r.latency_ms),
                        format!("{} · 640 px · {}", model_arch.as_str(), r.measured_on),
                    )
                    .into_any_element(),
                    None => text(ty::SM, t.text_tertiary)
                        .child(format!("not measured for {} yet", model_arch.as_str()))
                        .into_any_element(),
                },
            );
        }
        // Three columns, filled in order, top to bottom.
        let cols = if self.content_width(window) < 760. { 2 } else { 3 };
        let per = facts.len().div_ceil(cols);
        let mut columns = Vec::new();
        let mut it = facts.into_iter().peekable();
        while it.peek().is_some() {
            columns.push(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .children(it.by_ref().take(per).map(|(k, v)| fact_row(&t, k, v))),
            );
        }
        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Details"))
            .child(div().flex().gap(px(32.)).px(px(16.)).py(px(8.)).children(columns))
            .when(classes_expanded, |d| {
                d.child(
                    div()
                        .border_t_1()
                        .border_color(t.line_subtle)
                        .px(px(16.))
                        .py(px(12.))
                        .flex()
                        .flex_col()
                        .gap(px(8.))
                        .child(
                            text(ty::SM, t.text_tertiary)
                                .child(format!("All {} classes", int(m.classes.len() as u64))),
                        )
                        .child(div().flex().flex_wrap().gap(px(6.)).children(
                            m.classes.iter().enumerate().map(|(i, name)| {
                                div()
                                    .max_w_full()
                                    .min_w_0()
                                    .px(px(8.))
                                    .py(px(2.))
                                    .rounded(px(radius::SM))
                                    .bg(t.bg_inset)
                                    .flex()
                                    .items_center()
                                    .gap(px(5.))
                                    .child(crate::ui::class_swatch(&t, i as u32, 6.))
                                    .child(
                                        text(ty::SM, t.text_tertiary).flex_none().tnum().child(i.to_string()),
                                    )
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .min_w_0()
                                            .max_w_full()
                                            .whitespace_normal()
                                            .child(name.clone()),
                                    )
                            }),
                        )),
                )
            })
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::{finished_epochs, trained_for, weights_epoch};
    use crate::api::{Backend, Model, Run, RunStatus, mock::MockBackend};

    /// Studio's fixtures (web/src/routes/model/weightsEpoch.test.ts): run `run_a` planned for
    /// `epochs`, best at `best`; its best and last models.
    fn fixtures(best: Option<u32>, epochs: u32) -> (Run, Model, Model) {
        let api = MockBackend::new(4200, true);
        let mut run = api.runs().unwrap().into_iter().find(|r| r.id == "run_52e8").unwrap();
        let mut m =
            api.models().unwrap().into_iter().find(|m| m.run_id.as_deref() == Some("run_52e8")).unwrap();
        run.id = "run_a".into();
        run.epochs = epochs;
        match best {
            Some(b) => run.best.as_mut().unwrap().epoch = b,
            None => run.best = None,
        }
        m.run_id = Some("run_a".into());
        m.id = "m_a".into();
        m.trained_with = Some(folo_api::TrainedWith { amp: false, imgsz: 320, epochs });
        let mut last = m.clone();
        last.id = "m_a-last".into();
        (run, m, last)
    }

    fn we(m: &Model, run: Option<&Run>, done: Option<u32>) -> Option<String> {
        weights_epoch(m, run, done)
    }

    #[test]
    fn names_the_epoch_the_weights_are_from_and_the_same_weights_when_best_is_last() {
        let (r, best, last) = fixtures(Some(1), 5);
        assert_eq!(we(&best, Some(&r), Some(5)).as_deref(), Some("best epoch 1 of 5"));
        assert_eq!(we(&last, Some(&r), Some(5)).as_deref(), Some("last epoch, 5 of 5"));
        let (r, best, last) = fixtures(Some(5), 5);
        assert_eq!(we(&best, Some(&r), Some(5)).as_deref(), Some("best epoch 5 of 5, same weights as last"));
        assert_eq!(we(&last, Some(&r), Some(5)).as_deref(), Some("last epoch, 5 of 5, same weights as best"));
    }

    /// s4-models-1: a run stopped after epoch 1 of 50 said only "last finished epoch"; a completed
    /// run that stopped early (patience) is not at its planned end either.
    #[test]
    fn numbers_the_last_finished_epoch_of_a_run_that_ended_early() {
        let (r, best, last) = fixtures(Some(1), 50);
        assert_eq!(
            we(&last, Some(&r), Some(1)).as_deref(),
            Some("last finished epoch, 1 of 50, same weights as best")
        );
        assert_eq!(we(&last, Some(&r), Some(2)).as_deref(), Some("last finished epoch, 2 of 50"));
        assert_eq!(we(&best, Some(&r), Some(2)).as_deref(), Some("best epoch 1 of 50"));
        let (r, best, last) = fixtures(Some(1), 200);
        assert_eq!(we(&last, Some(&r), Some(101)).as_deref(), Some("last finished epoch, 101 of 200"));
        assert_eq!(we(&best, Some(&r), Some(101)).as_deref(), Some("best epoch 1 of 200"));
    }

    #[test]
    fn says_nothing_it_does_not_know() {
        let (r, best, _) = fixtures(Some(1), 5);
        assert_eq!(we(&best, None, Some(5)), None);
        let mut other = r.clone();
        other.id = "run_b".into();
        assert_eq!(we(&best, Some(&other), Some(5)), None);
        let (r, best, last) = fixtures(None, 50);
        assert_eq!(we(&best, Some(&r), Some(5)), None);
        assert_eq!(we(&last, Some(&r), None).as_deref(), Some("last finished epoch"));
        let (r, best, _) = fixtures(Some(5), 5);
        assert_eq!(we(&best, Some(&r), None).as_deref(), Some("best epoch 5 of 5"));
    }

    #[test]
    fn a_completed_or_stopped_runs_epoch_counts_over_a_history_that_lags_it() {
        use RunStatus::*;
        assert_eq!(finished_epochs(Stopped, 3, Some(2)), Some(3));
        assert_eq!(finished_epochs(Stopped, 3, None), Some(3));
        assert_eq!(finished_epochs(Completed, 101, Some(101)), Some(101));
        assert_eq!(finished_epochs(Stopped, 0, Some(0)), None);
    }

    #[test]
    fn any_other_run_counts_its_history_when_its_epoch_is_not_past_it() {
        use RunStatus::*;
        assert_eq!(finished_epochs(Failed, 2, Some(2)), Some(2));
        assert_eq!(finished_epochs(Queued, 3, Some(3)), Some(3));
        assert_eq!(finished_epochs(Running, 2, Some(2)), Some(2));
        assert_eq!(finished_epochs(Failed, 3, Some(2)), None);
        assert_eq!(finished_epochs(Queued, 3, Some(2)), None);
        assert_eq!(finished_epochs(Running, 3, Some(2)), None);
        assert_eq!(finished_epochs(Failed, 2, None), None);
        assert_eq!(finished_epochs(Queued, 0, Some(0)), None);
    }

    /// s4-models-1: "Trained for 50 epochs" for a run stopped after one; `trained_with.epochs` is
    /// the planned count.
    #[test]
    fn trained_for_says_how_many_of_the_planned_epochs_ran() {
        let (r, m, _) = fixtures(Some(1), 50);
        assert_eq!(trained_for(&m, Some(&r), Some(1)).as_deref(), Some("1 of 50 epochs at 320 px"));
        assert_eq!(trained_for(&m, Some(&r), Some(2)).as_deref(), Some("2 of 50 epochs at 320 px"));
        assert_eq!(trained_for(&m, Some(&r), Some(50)).as_deref(), Some("50 epochs at 320 px"));
        let (r, m, _) = fixtures(Some(1), 200);
        assert_eq!(trained_for(&m, Some(&r), Some(101)).as_deref(), Some("101 of 200 epochs at 320 px"));
        let (r, m, _) = fixtures(Some(1), 1);
        assert_eq!(trained_for(&m, Some(&r), Some(1)).as_deref(), Some("1 epoch at 320 px"));
    }

    #[test]
    fn trained_for_labels_the_plan_while_what_ran_is_not_known() {
        let (r, mut m, _) = fixtures(Some(1), 50);
        assert_eq!(trained_for(&m, Some(&r), None).as_deref(), Some("50 epochs planned at 320 px"));
        assert_eq!(trained_for(&m, None, None).as_deref(), Some("50 epochs planned at 320 px"));
        m.run_id = Some("run_b".into());
        assert_eq!(trained_for(&m, Some(&r), Some(1)).as_deref(), Some("50 epochs planned at 320 px"));
        m.run_id = None;
        assert_eq!(trained_for(&m, None, None).as_deref(), Some("50 epochs planned at 320 px"));
        m.trained_with = None;
        assert_eq!(trained_for(&m, None, None), None);
    }
}
