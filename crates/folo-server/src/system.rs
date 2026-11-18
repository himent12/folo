//! Host CPU and memory information (shared with `folo doctor`).

use std::sync::Mutex;

use folo_api::{CpuInfo, MemoryInfo};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Static facts about the host.
#[derive(Clone, Debug, serde::Serialize)]
pub struct HostInfo {
    /// OS name and version, e.g. `Windows 11 (26300)`.
    pub os: String,
    /// CPU architecture, e.g. `x86_64`.
    pub arch: String,
    /// CPU name.
    pub cpu_name: String,
    /// Logical threads.
    pub cpu_threads: u32,
    /// Total physical memory in bytes.
    pub memory_total_bytes: u64,
    /// Available memory in bytes.
    pub memory_available_bytes: u64,
}

/// What runs with `device: auto` can use here (f16, tensor cores, `amp`): `/api/system`'s `capabilities`, which
/// `folo doctor` reports too. Creates the GPU context on first use, as a run would.
pub fn capabilities() -> folo_api::Capabilities {
    crate::engine::capabilities()
}

/// The devices FOLO can run on, for error hints: `cpu, gpu:0 (NVIDIA GeForce RTX 4080 SUPER, vulkan)`. GPUs
/// count once their backend is registered (folo-gpu's `init()`, which `auto` and `gpu:<n>` devices trigger).
pub fn available_devices() -> String {
    let gpus = crate::engine::gpus();
    folo_core::backend::available_devices()
        .into_iter()
        .map(|d| match d {
            folo_core::Device::Gpu(i) => match gpus.iter().find(|g| g.ordinal as usize == i) {
                Some(g) => format!(
                    "{d} ({}, {})",
                    g.name,
                    serde_json::to_value(g.backend)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_string))
                        .unwrap_or_default()
                ),
                None => d.to_string(),
            },
            d => d.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One-shot host information.
pub fn host_info() -> HostInfo {
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing())
            .with_memory(MemoryRefreshKind::everything()),
    );
    let os = match (System::long_os_version(), System::kernel_version()) {
        (Some(v), Some(k)) => format!("{v} (kernel {k})"),
        (Some(v), None) => v,
        _ => std::env::consts::OS.to_string(),
    };
    HostInfo {
        os,
        arch: std::env::consts::ARCH.to_string(),
        cpu_name: cpu_name(&sys),
        cpu_threads: sys.cpus().len().max(1) as u32,
        memory_total_bytes: sys.total_memory(),
        memory_available_bytes: sys.available_memory(),
    }
}

fn cpu_name(sys: &System) -> String {
    sys.cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Keeps a [`System`] between calls so CPU utilisation is measured over the interval since the last call.
///
/// Setting up the CPU usage counters takes Windows about 150 ms, which `folo serve` spent before it said it was
/// listening (and FOLO Desktop waited on). They are set up on a thread of their own instead, started where the
/// probe is made; the first [`SystemProbe::sample`] waits for them if they are not ready yet.
pub struct SystemProbe {
    sys: Mutex<Probe>,
}

/// The probe's [`System`], or the thread still setting it up.
enum Probe {
    Starting(std::thread::JoinHandle<System>),
    Ready(System),
}

fn new_system() -> System {
    System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    )
}

impl Default for SystemProbe {
    fn default() -> Self {
        let probe = match std::thread::Builder::new().name("folo-system-probe".into()).spawn(new_system) {
            Ok(thread) => Probe::Starting(thread),
            Err(_) => Probe::Ready(new_system()),
        };
        SystemProbe { sys: Mutex::new(probe) }
    }
}

impl SystemProbe {
    /// Current CPU and memory.
    pub fn sample(&self) -> (CpuInfo, MemoryInfo) {
        let mut probe = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        if let Probe::Starting(_) = &*probe {
            // Swapped out for a moment to take the thread by value; a set-up that panicked is done again here.
            let Probe::Starting(thread) = std::mem::replace(&mut *probe, Probe::Ready(System::new())) else {
                unreachable!("matched above")
            };
            *probe = Probe::Ready(thread.join().unwrap_or_else(|_| new_system()));
        }
        let Probe::Ready(sys) = &mut *probe else { unreachable!("made ready above") };
        sys.refresh_cpu_usage();
        sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        let util = (sys.global_cpu_usage() / 100.0).clamp(0.0, 1.0);
        let cpu = CpuInfo {
            name: cpu_name(sys),
            threads: sys.cpus().len().max(1) as u32,
            util: (util * 100.0).round() / 100.0,
        };
        let total = sys.total_memory();
        let mem = MemoryInfo { used_bytes: total.saturating_sub(sys.available_memory()), total_bytes: total };
        (cpu, mem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_samples_once_its_counters_are_set_up_off_the_caller() {
        let probe = SystemProbe::default();
        for _ in 0..2 {
            let (cpu, mem) = probe.sample();
            assert!(cpu.threads >= 1);
            assert!((0.0..=1.0).contains(&cpu.util), "{}", cpu.util);
            assert!(mem.total_bytes > 0 && mem.used_bytes <= mem.total_bytes);
        }
    }
}
