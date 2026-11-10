//! A cloud run's worker: the FOLO server it trains on, reached with the public API (docs/API.md) through
//! `folo-client`. The worker check of docs/CLOUD.md, and the conversions of what the worker serves into a local run
//! folder's files (the history in the trainer's order, samples in the trainer's format).

use std::time::Duration;

use folo_api::{RunEvent, RunMetrics, RunSamples, SystemResponse};
use folo_client::Client;
use serde_json::{Value, json};

/// Time to reach a worker, and for one answer: every request is short, because provider proxies cut connections
/// after 100 seconds.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(90);

/// A client of the worker at `url` with its `token`.
pub(crate) fn client(url: &str, token: &str) -> Client {
    Client::with_timeouts(url, CONNECT_TIMEOUT, REQUEST_TIMEOUT).token(token)
}

/// Why the worker described by `system` (running FOLO `version`) cannot train the run, if it cannot: it lacks a
/// feature the control plane needs, it has no GPU while `gpu` is required (a provisioned instance), or the run
/// needs mixed precision (`amp`) it does not have. The message names the check and what the worker reported.
pub(crate) fn check(system: &SystemResponse, version: &str, gpu: bool, amp: bool) -> Result<(), String> {
    let missing: Vec<&str> = folo_cloud::remote::REQUIRED_FEATURES
        .into_iter()
        .filter(|f| !system.features.iter().any(|g| g == f))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "the worker runs FOLO {version}, which cannot train cloud runs (it lacks {}); it needs FOLO {} or later",
            missing.join(", "),
            env!("CARGO_PKG_VERSION")
        ));
    }
    let device = system.capabilities.as_ref().map_or("cpu", |c| c.device.as_str());
    if gpu && !device.starts_with("gpu") {
        let adapters: Vec<String> = system
            .system
            .gpus
            .iter()
            .map(|g| format!("{} ({:?})", g.name, g.backend).to_lowercase())
            .collect();
        let reported = if adapters.is_empty() {
            "it reported no GPU adapter".to_string()
        } else {
            format!("the adapters it reported: {}", adapters.join(", "))
        };
        return Err(format!(
            "the instance has no GPU FOLO can use, so it would train on the CPU at GPU prices ({reported}); the \
             provider's image may lack the Vulkan driver"
        ));
    }
    if amp && let Some(c) = system.capabilities.as_ref().filter(|c| !c.amp.available) {
        return Err(format!(
            "the run uses mixed precision (amp), which the worker cannot train: {}",
            c.amp.reason.as_deref().unwrap_or("not available")
        ));
    }
    Ok(())
}

/// The new history of one poll, in the order a trainer emits it: each epoch's steps, then the epoch.
pub(crate) fn in_order(m: RunMetrics) -> Vec<RunEvent> {
    let mut events: Vec<(u32, u8, u64, RunEvent)> = m
        .steps
        .into_iter()
        .map(|s| (s.epoch, 0, s.step, RunEvent::Step(s)))
        .chain(m.epochs.into_iter().map(|e| (e.epoch, 1, 0, RunEvent::Epoch(e))))
        .collect();
    events.sort_by_key(|e| (e.0, e.1, e.2));
    events.into_iter().map(|e| e.3).collect()
}

/// The worker's samples of an epoch as the trainer writes `samples/epoch_NNN.json`: with the dataset's `classes`
/// and each image's position in the validation split (`index`). Image ids are the same on both sides (the dataset
/// travels in its own order), so the local run serves them as `val_offset + index`, its own ids.
pub(crate) fn samples_file(s: &RunSamples, classes: &[String], val_offset: u64) -> Value {
    let samples: Vec<Value> = s
        .samples
        .iter()
        .map(|x| {
            let index = x.image_id.parse::<u64>().unwrap_or(0).saturating_sub(val_offset);
            let truth: Vec<Value> = x
                .truth
                .iter()
                .map(|b| json!({ "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class": b.class }))
                .collect();
            let pred: Vec<Value> = x
                .pred
                .iter()
                .map(|b| json!({ "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class": b.class, "score": b.score }))
                .collect();
            json!({ "index": index, "file": null, "width": x.width, "height": x.height, "truth": truth, "pred": pred })
        })
        .collect();
    json!({ "epoch": s.epoch, "classes": classes, "samples": samples })
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_api::{AmpSupport, Capabilities, GpuBackend, GpuInfo, SystemInfo, TensorCoreSupport};

    fn system(features: &[&str], device: &str, amp: bool) -> SystemResponse {
        let cpu = folo_api::CpuInfo { name: "c".into(), threads: 1, util: 0.0 };
        let memory = folo_api::MemoryInfo { used_bytes: 0, total_bytes: 0 };
        let gpus = vec![GpuInfo {
            ordinal: 0,
            name: "llvmpipe".into(),
            backend: GpuBackend::Vulkan,
            memory_bytes: 0,
            util: None,
        }];
        SystemResponse {
            system: SystemInfo { cpu, memory, gpus },
            data_dir: None,
            features: features.iter().map(|f| f.to_string()).collect(),
            capabilities: Some(Capabilities {
                device: device.into(),
                shader_f16: false,
                tensor_cores: TensorCoreSupport { available: false, enabled: false, reason: None },
                amp: AmpSupport { available: amp, reason: (!amp).then(|| "no tensor cores".to_string()) },
            }),
            gpu_pool: None,
        }
    }

    #[test]
    fn the_worker_check_names_what_failed() {
        let all = ["uploads", "run_cursors", "run_checkpoint"];
        assert_eq!(check(&system(&all, "gpu:0", true), "0.1.0", true, true), Ok(()));
        let old = check(&system(&["uploads"], "gpu:0", true), "0.0.9", true, false).unwrap_err();
        assert!(old.contains("FOLO 0.0.9") && old.contains("run_cursors, run_checkpoint"), "{old}");
        let cpu = check(&system(&all, "cpu", false), "0.1.0", true, false).unwrap_err();
        assert!(cpu.contains("no GPU") && cpu.contains("llvmpipe (vulkan)"), "{cpu}");
        assert_eq!(
            check(&system(&all, "cpu", false), "0.1.0", false, false),
            Ok(()),
            "a remote server as it is"
        );
        let amp = check(&system(&all, "gpu:0", false), "0.1.0", true, true).unwrap_err();
        assert!(amp.contains("mixed precision") && amp.contains("no tensor cores"), "{amp}");
    }
}
