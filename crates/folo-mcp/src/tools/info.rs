//! `folo_info`: version, devices, data dir and model scales.

use folo_api::Arch;

use super::dispatch::join;
use super::results::{Output, out};
use crate::errors::Error;
use crate::server::Inner;
use crate::types::*;

impl Inner {
    pub(super) async fn folo_info(&self, _: NoArgs) -> Result<Output, Error> {
        let info = self.info().await?;
        let gpus = info.devices.gpus.len();
        let amp = match &info.capabilities {
            Some(c) if c.amp.available => ", amp available",
            Some(_) => ", no amp",
            None => "",
        };
        let summary = format!(
            "FOLO {} ({} mode{}), {} GPU{}, {} CPU threads{amp}",
            info.version,
            info.mode,
            if info.read_only { ", read-only" } else { "" },
            gpus,
            if gpus == 1 { "" } else { "s" },
            info.devices.cpu.threads
        );
        out(&info, summary)
    }

    /// The `folo_info` record.
    pub async fn info(&self) -> Result<FoloInfo, Error> {
        let (cpu, gpus, data_dir, capabilities) = match &self.proxy {
            Some(p) => {
                let p = p.clone();
                let sys = tokio::task::spawn_blocking(move || p.system()).await.map_err(join)??;
                let (data_dir, capabilities, sys) = (sys.data_dir, sys.capabilities, sys.system);
                let cpu = CpuDevice {
                    name: sys.cpu.name,
                    threads: sys.cpu.threads,
                    memory_bytes: sys.memory.total_bytes,
                };
                // The server's data dir: runs, models and exports live there, not under this process's default.
                (cpu, sys.gpus, data_dir, capabilities)
            }
            None => {
                let store = self.store.clone();
                tokio::task::spawn_blocking(move || {
                    let host = folo_server::system::host_info();
                    let cpu = CpuDevice {
                        name: host.cpu_name,
                        threads: host.cpu_threads,
                        memory_bytes: host.memory_total_bytes,
                    };
                    // Capabilities first, as `/api/system` does: they create the GPU context, and `gpus()` (cached
                    // on first use) puts the context's adapter at ordinal 0 only once it exists.
                    let caps = store.capabilities();
                    (cpu, store.gpus(), caps)
                })
                .await
                .map(|(cpu, gpus, caps)| (cpu, gpus, Some(folo_api::json::path(&self.paths.data_dir)), caps))
                .map_err(join)?
            }
        };
        Ok(FoloInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            mode: if self.proxy.is_some() {
                "proxy"
            } else if self.mock {
                "mock"
            } else {
                "embedded"
            }
            .to_string(),
            server: self.proxy.as_ref().map(|p| p.base().to_string()),
            data_dir,
            read_only: self.opts.read_only,
            confine: self.paths.confine,
            devices: Devices { cpu, gpus },
            capabilities,
            scales: SCALES
                .iter()
                .map(|&(arch, params, gflops_640, gflops_320)| ScaleInfo {
                    arch,
                    params,
                    gflops_640,
                    gflops_320,
                })
                .collect(),
        })
    }
}

/// Model scales at 80 classes (docs/MODEL.md, "Parameter and FLOP totals").
const SCALES: [(Arch, u64, f64, f64); 5] = [
    (Arch::N, 2_624_064, 6.54, 1.62),
    (Arch::S, 9_458_736, 21.59, 5.37),
    (Arch::M, 20_114_672, 68.10, 17.00),
    (Arch::L, 25_372_144, 87.15, 21.74),
    (Arch::X, 56_966_160, 195.27, 48.75),
];
