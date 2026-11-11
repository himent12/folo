//! The New run dialog's footer: the time and memory estimate (the server's, or the app's own
//! reference figures), a cloud run's cost, Cancel and Start; and the dialog's frame.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, ParentElement, StatefulInteractiveElement,
    Styled, Window, div, px,
};

use crate::api::RunStatus;
use crate::app::{BATCHES, DEVICES, FoloApp, IMGSZ, act};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

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

/// A server estimate as the footer writes it, its point and range ("10 min", "7–15 min"): the point
/// never outside the range, nor in a finer unit than the range's ends (a server's point below its
/// low end read "≈ 5 s (1–3 min)").
fn estimate_text(point: f64, lo: f64, hi: f64) -> (String, String) {
    let (lo, hi) = (lo.min(hi).max(0.), lo.max(hi));
    let p = point.clamp(lo, hi);
    let point = if hi >= 60. && p < 60. { "1 min".to_string() } else { rough_duration(p) };
    (point, rough_range(lo, hi))
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

/// What `new_run_dialog` read from the form and the footer needs: the choices that decide the estimate and the
/// cost, and what keeps Start disabled.
pub(super) struct NewRunSummary {
    pub(super) ds: Option<crate::api::Dataset>,
    pub(super) cloud: Option<crate::api::CloudConnection>,
    pub(super) cloud_offer: Option<crate::api::CloudOffer>,
    pub(super) arch: usize,
    pub(super) epochs: f32,
    pub(super) batch: usize,
    pub(super) imgsz: usize,
    pub(super) device: usize,
    pub(super) amp: bool,
    pub(super) amp_why: Option<String>,
    pub(super) epochs_bad: bool,
    pub(super) lr_bad: bool,
    pub(super) close_bad: bool,
    pub(super) submitting: bool,
    pub(super) opt_unknown: bool,
}

impl FoloApp {
    /// The New run dialog around its `body`: the footer (the estimate, a cloud run's cost, Cancel and
    /// Start) and the frame. `new_run_dialog` hands over what it read from the form.
    pub(super) fn frame_new_run(
        &mut self,
        t: Theme,
        body: Div,
        form: NewRunSummary,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let NewRunSummary {
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
        } = form;
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
                let (point, range) = estimate_text(e.est_total_s, e.est_total_s_low, e.est_total_s_high);
                let headline = match vram {
                    Some(v) => format!("≈ {point} ({range}) · ≈ {v:.1} GiB GPU memory"),
                    None => format!("≈ {point} ({range}) on the CPU"),
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
        } else if opt_unknown {
            self.button_disabled(
                "nr-start",
                go,
                Btn::Primary,
                Some(go_icon),
                "Pick an optimizer first",
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rough_durations() {
        assert_eq!(super::rough_duration(576.), "10 min");
        assert_eq!(super::rough_duration(25.), "25 s");
        assert_eq!(super::rough_duration(5100.), "1 h 25 min");
    }

    #[test]
    fn a_point_estimate_stays_inside_its_range() {
        assert_eq!(estimate_text(5., 50., 170.), ("1 min".to_string(), "1–3 min".to_string()));
        assert_eq!(estimate_text(900., 420., 600.), ("10 min".to_string(), "7–10 min".to_string()));
        assert_eq!(estimate_text(576., 420., 900.), ("10 min".to_string(), "7–15 min".to_string()));
        assert_eq!(estimate_text(20., 10., 40.), ("20 s".to_string(), "10–40 s".to_string()));
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
