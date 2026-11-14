//! A model's Details panel: the facts about it in three columns (Studio's `DetailsList.tsx`): what it
//! is, how it was trained and stored, where it came from, what it was evaluated on, its id, and its
//! architecture's measured and reference latency.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::{InferThroughput, Model, ModelEval, ModelSource, Run, RunStatus, ThroughputSource, time};
use crate::app::FoloApp;
use crate::store::Load;
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

/// Which epoch a run model's weights are from (docs/API.md, Models): `<run>-best` the run's best
/// epoch, `<run>-last` its last finished one; when they are the same epoch, each says so. None
/// without the run, or when the server didn't say which epoch was best.
pub(super) fn weights_epoch(m: &Model, run: Option<&Run>) -> Option<String> {
    let run = run.filter(|r| m.run_id.as_deref() == Some(r.id.as_str()))?;
    let done = run.status == RunStatus::Completed;
    let best = run.best.as_ref().map(|b| b.epoch);
    if m.id.ends_with("-last") {
        if !done {
            return Some("last finished epoch".into());
        }
        let same = if best == Some(run.epochs) { ", same weights as best" } else { "" };
        return Some(format!("last epoch, {} of {}{same}", run.epochs, run.epochs));
    }
    let best = best?;
    let same = if done && best == run.epochs { ", same weights as last" } else { "" };
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
        let mut facts: Vec<(&str, AnyElement)> = Vec::new();
        let mut fact = |k: &'static str, v: AnyElement| facts.push((k, v));
        fact("Architecture", value(&t, m.arch.as_str()).into_any_element());
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
            fact(
                "Trained for",
                value(
                    &t,
                    format!(
                        "{} {} at {} px",
                        int(w.epochs as u64),
                        if w.epochs == 1 { "epoch" } else { "epochs" },
                        w.imgsz
                    ),
                )
                .into_any_element(),
            );
        }
        if let Some(w) = weights_epoch(m, run.as_ref()) {
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
        let shown: Vec<String> = m.classes.iter().take(10).cloned().collect();
        let more = m.classes.len().saturating_sub(10);
        fact(
            "Classes",
            value(
                &t,
                format!(
                    "{} · {}{}",
                    m.classes.len(),
                    shown.join(", "),
                    if more > 0 { format!(" +{more} more") } else { String::new() }
                ),
            )
            .into_any_element(),
        );
        let ds = run.as_ref().map(|r| self.store.run_dataset_name(r));
        fact(
            "Source",
            match (&run, m.source) {
                (Some(r), _) => match ds.filter(|d| !d.is_empty()) {
                    Some(d) => value_with(&t, r.name.clone(), format!("on {d}")).into_any_element(),
                    None => value(&t, r.name.clone()).into_any_element(),
                },
                (None, Some(ModelSource::Published)) => match &m.about {
                    Some(a) => value_with(&t, "Published weights", a.clone()).into_any_element(),
                    None => value(&t, "Published weights").into_any_element(),
                },
                (None, _) if m.run_id.is_some() => {
                    value(&t, m.run_id.clone().unwrap_or_default()).into_any_element()
                }
                _ => value(&t, "Imported .folo file").into_any_element(),
            },
        );
        // What the curves, the class table and the matrix were measured on.
        let evaluated = ev
            .and_then(|e| {
                e.dataset_name
                    .clone()
                    .zip(e.split.map(|s| s.as_str()))
                    .map(|(d, s)| (d, s, Some(e.images as u64)))
            })
            .or_else(|| {
                run.as_ref().and_then(|r| self.store.dataset_summary(&r.config.dataset_id)).map(|d| {
                    (
                        d.name.clone(),
                        if d.splits.val == 0 { "train" } else { "val" },
                        ev.map(|e| e.images as u64),
                    )
                })
            });
        if let Some((d, s, n)) = evaluated {
            fact("Evaluated on", value(&t, split_label(&d, s, n)).into_any_element());
        }
        // What the run started from: a model by name, or nothing (from scratch).
        if let Some(r) = &run {
            let from = r.config.weights.as_deref().map_or("from scratch (random weights)".to_string(), |w| {
                self.store.model(w).map_or(w.to_string(), |p| p.name.clone())
            });
            fact("Started from", value(&t, from).into_any_element());
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
                        format!("{} · 640 px · {}", m.arch.as_str(), r.measured_on),
                    )
                    .into_any_element(),
                    None => text(ty::SM, t.text_tertiary)
                        .child(format!("not measured for {} yet", m.arch.as_str()))
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
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::weights_epoch;
    use crate::api::{Backend, RunStatus, mock::MockBackend};

    #[test]
    fn says_which_epoch_the_weights_are_from() {
        let api = MockBackend::new(4200, true);
        let mut run = api.runs().unwrap().into_iter().find(|r| r.id == "run_52e8").unwrap();
        let mut m =
            api.models().unwrap().into_iter().find(|m| m.run_id.as_deref() == Some("run_52e8")).unwrap();
        m.id = "m_52e8-best".into();
        run.status = RunStatus::Completed;
        run.epochs = 5;
        run.best.as_mut().unwrap().epoch = 1;
        assert_eq!(weights_epoch(&m, Some(&run)).as_deref(), Some("best epoch 1 of 5"));
        run.best.as_mut().unwrap().epoch = 5;
        assert_eq!(weights_epoch(&m, Some(&run)).as_deref(), Some("best epoch 5 of 5, same weights as last"));
        m.id = "m_52e8-last".into();
        assert_eq!(
            weights_epoch(&m, Some(&run)).as_deref(),
            Some("last epoch, 5 of 5, same weights as best")
        );
        run.status = RunStatus::Stopped;
        assert_eq!(weights_epoch(&m, Some(&run)).as_deref(), Some("last finished epoch"));
        assert_eq!(weights_epoch(&m, None), None);
        m.run_id = Some("run_other".into());
        assert_eq!(weights_epoch(&m, Some(&run)), None);
    }
}
