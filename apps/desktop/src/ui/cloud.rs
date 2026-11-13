//! Cloud compute as text: prices, costs, hours, offers, transfers and phases.

use super::format::{bytes, int, with_article};
use crate::api::{CloudOffer, CloudPhase, Label, RunCloud, TransferProgress};

// ---------- cloud compute ----------

/// A price per hour, one way everywhere: "$0.69/h"; "—" for compute that isn't billed.
pub fn fmt_price(per_hour: Option<f64>) -> String {
    match per_hour {
        Some(p) => format!("${p:.2}/h"),
        None => "—".to_string(),
    }
}

/// Money spent or to be spent: "$0.42", "<$0.01" under a cent, whole dollars from $100 ("$1,280").
pub fn fmt_cost(usd: f64) -> String {
    let usd = usd.max(0.);
    if usd > 0. && usd < 0.01 {
        "<$0.01".to_string()
    } else if usd >= 100. {
        format!("${}", int(usd.round() as u64))
    } else {
        format!("${usd:.2}")
    }
}

/// A cost range, "$0.12–$0.93"; one figure when both ends read the same.
pub fn fmt_cost_range(lo: f64, hi: f64) -> String {
    let (lo, hi) = (fmt_cost(lo.min(hi)), fmt_cost(lo.max(hi)));
    if lo == hi { lo } else { format!("{lo}–{hi}") }
}

/// Hours as typed in Stop after: "12", "0.5", "0.25" (no trailing zeros).
pub fn hours_value(h: f32) -> String {
    let s = format!("{h:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Hours as a limit reads: "12 h", "0.5 h".
pub fn fmt_hours(h: f32) -> String {
    format!("{} h", hours_value(h))
}

/// An offer in one line: "RTX 4090 · 24.0 GiB · Secure Cloud" (memory and qualifier when known).
pub fn offer_line(o: &CloudOffer) -> String {
    let mut parts =
        vec![if o.gpu_count > 1 { format!("{}× {}", o.gpu_count, o.gpu) } else { o.gpu.clone() }];
    if o.vram_bytes > 0 {
        let (v, u) = bytes(o.vram_bytes);
        parts.push(format!("{v} {u}"));
    }
    parts.extend(o.detail.clone());
    parts.join(" · ")
}

/// A transfer's progress: "1.1 of 4.5 GiB" (one unit when both sides share it), or what has moved
/// when the total is unknown.
pub fn fmt_transfer(p: &TransferProgress) -> String {
    let (done, done_u) = bytes(p.done_bytes);
    if p.total_bytes == 0 {
        return format!("{done} {done_u}");
    }
    let (total, total_u) = bytes(p.total_bytes);
    if done_u == total_u {
        format!("{done} of {total} {total_u}")
    } else {
        format!("{done} {done_u} of {total} {total_u}")
    }
}

/// The fraction of a transfer done, when its total is known.
pub fn transfer_frac(p: &TransferProgress) -> Option<f32> {
    (p.total_bytes > 0).then(|| (p.done_bytes as f64 / p.total_bytes as f64).clamp(0., 1.) as f32)
}

/// A cloud run's phase in a few words: "Training", "Uploading data · 25%" while a transfer with a
/// known size runs.
pub fn phase_text(c: &RunCloud) -> String {
    match c.progress.as_ref().and_then(transfer_frac) {
        Some(f) => format!("{} · {:.0}%", c.phase.label(), f * 100.),
        None => c.phase.label().to_string(),
    }
}

/// What a cloud run that hasn't started training is doing, as a sentence-case title:
/// "Provisioning an RTX 4090 at RunPod", "Booting the instance", "Uploading data · 25%".
pub fn cloud_wait_title(c: &RunCloud, provider: &str) -> String {
    match c.phase {
        CloudPhase::Provisioning => format!("Provisioning {} at {provider}", with_article(&c.offer.gpu)),
        CloudPhase::Booting => "Booting the instance".to_string(),
        _ => phase_text(c),
    }
}

#[cfg(test)]
mod cloud_format_tests {
    use super::*;
    use crate::api::CloudAvailability;
    use crate::api::Run;
    use crate::ui::format::waiting_text;
    use crate::ui::status::{Tone, status_label, status_tone};

    fn offer(gpu: &str, vram: u64, detail: Option<&str>) -> CloudOffer {
        CloudOffer {
            id: "o".into(),
            gpu: gpu.into(),
            gpu_count: 1,
            vram_bytes: vram,
            price_per_hour: Some(0.69),
            region: None,
            detail: detail.map(Into::into),
            availability: CloudAvailability::High,
        }
    }

    #[test]
    fn money_reads_one_way() {
        assert_eq!(fmt_price(Some(0.69)), "$0.69/h");
        assert_eq!(fmt_price(Some(2.5)), "$2.50/h");
        assert_eq!(fmt_price(None), "—");
        assert_eq!(fmt_cost(0.42), "$0.42");
        assert_eq!(fmt_cost(0.0), "$0.00");
        assert_eq!(fmt_cost(0.004), "<$0.01");
        assert_eq!(fmt_cost(99.994), "$99.99");
        assert_eq!(fmt_cost(1280.4), "$1,280");
        assert_eq!(fmt_cost(-1.0), "$0.00");
        assert_eq!(fmt_cost_range(0.12, 0.93), "$0.12–$0.93");
        assert_eq!(fmt_cost_range(0.93, 0.12), "$0.12–$0.93");
        assert_eq!(fmt_cost_range(0.001, 0.004), "<$0.01");
    }

    #[test]
    fn hours_and_offers() {
        assert_eq!(
            (fmt_hours(12.0), fmt_hours(0.5), fmt_hours(0.25)),
            ("12 h".into(), "0.5 h".into(), "0.25 h".into())
        );
        assert_eq!(hours_value(720.0), "720");
        assert_eq!(
            offer_line(&offer("RTX 4090", 25_769_803_776, Some("Secure Cloud"))),
            "RTX 4090 · 24.0 GiB · Secure Cloud"
        );
        assert_eq!(offer_line(&offer("RTX 4080 SUPER", 0, None)), "RTX 4080 SUPER");
        assert_eq!(offer_line(&CloudOffer { gpu_count: 2, ..offer("A100", 0, None) }), "2× A100");
        assert_eq!(with_article("RTX 4090"), "an RTX 4090");
        assert_eq!(with_article("H100 SXM"), "an H100 SXM");
        assert_eq!(with_article("T4"), "a T4");
    }

    #[test]
    fn phases_and_transfers() {
        let gib = 1u64 << 30;
        let p = TransferProgress { done_bytes: gib + gib / 5, total_bytes: 4 * gib + gib * 4 / 5 };
        assert_eq!(fmt_transfer(&p), "1.2 of 4.8 GiB");
        assert_eq!(
            fmt_transfer(&TransferProgress { done_bytes: 300 << 20, total_bytes: 2 * gib }),
            "300.0 MiB of 2.0 GiB"
        );
        assert_eq!(fmt_transfer(&TransferProgress { done_bytes: 300 << 20, total_bytes: 0 }), "300.0 MiB");
        let mut c = RunCloud {
            connection_id: "cc".into(),
            provider: "runpod".into(),
            offer: offer("RTX 4090", 0, None),
            instance_id: None,
            phase: CloudPhase::Provisioning,
            progress: None,
            max_hours: 12.,
            deadline: None,
            cost_usd: None,
            synced: None,
            ended_in: None,
            unanswered_since: None,
        };
        assert_eq!(cloud_wait_title(&c, "RunPod"), "Provisioning an RTX 4090 at RunPod");
        c.phase = CloudPhase::Booting;
        assert_eq!(cloud_wait_title(&c, "RunPod"), "Booting the instance");
        c.phase = CloudPhase::Syncing;
        c.progress = Some(TransferProgress { done_bytes: 1, total_bytes: 4 });
        assert_eq!(cloud_wait_title(&c, "RunPod"), "Uploading data · 25%");
        c.progress = Some(TransferProgress { done_bytes: 1, total_bytes: 0 });
        assert_eq!(phase_text(&c), "Uploading data");
    }

    #[test]
    fn a_cloud_run_handing_over_reads_its_phase_not_queued() {
        use crate::api::{Backend, RunStatus};
        let api = crate::api::mock::MockBackend::with_cloud_runs(0, true);
        let mut r = api.run("run_c1d0").unwrap();
        assert_eq!(r.status, RunStatus::Queued);
        assert_eq!((status_label(&r), waiting_text(&r).as_str()), ("Uploading data", "1.2 of 4.8 GiB"));
        let phase = |r: &mut Run, p: CloudPhase| {
            if let Some(c) = r.cloud.as_mut() {
                c.phase = p;
                c.progress = None;
            }
        };
        phase(&mut r, CloudPhase::Provisioning);
        assert_eq!((status_label(&r), waiting_text(&r).as_str()), ("Provisioning", "–"));
        phase(&mut r, CloudPhase::Booting);
        assert_eq!(status_label(&r), "Booting");
        // It still looks like a waiting run.
        assert_eq!(status_tone(r.status), Tone::Neutral);
        // Training (even before the status catches up), a newer server's phase, or not queued any
        // more: the status says it.
        for p in [CloudPhase::Training, CloudPhase::Unknown] {
            phase(&mut r, p);
            assert_eq!((status_label(&r), waiting_text(&r).as_str()), ("Queued", "Waiting to start"));
        }
        phase(&mut r, CloudPhase::Syncing);
        r.status = RunStatus::Stopping;
        assert_eq!(status_label(&r), "Stopping");
        // A run on this machine reads its status and its place in the queue.
        let local = api.run("run_b7e2").unwrap();
        assert_eq!((status_label(&local), waiting_text(&local).as_str()), ("Queued", "Next to start"));
        assert_eq!(status_label(&api.run("run_c9f1").unwrap()), "Completed");
    }
}
