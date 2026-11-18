//! Host CPU and memory information (shared with `folo doctor`).

use std::sync::Mutex;
use std::time::{Duration, Instant};

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
/// probe is made. The first reading needs an interval behind it: sysinfo measures CPU use between two refreshes,
/// and one made right after the set-up reads 100 %. So [`SystemProbe::warm_up`] (and the first
/// [`SystemProbe::sample`], which does it) waits for the set-up and then out the rest of
/// [`sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`] from when it finished; `/api/system` does that beside creating the GPU
/// device, which takes longer.
pub struct SystemProbe {
    sys: Mutex<Probe>,
    /// [`SystemProbe::warm_up`] calls so far (`/api/system`'s preparations).
    #[cfg(test)]
    pub(crate) warm_ups: std::sync::atomic::AtomicUsize,
}

/// The probe's [`System`], or the thread still setting it up (which returns it with when it finished).
enum Probe {
    Starting(std::thread::JoinHandle<(System, Instant)>),
    /// `measured_from`: when the counters last read (set up or refreshed), the start of the next reading's interval;
    /// `first`: no reading made yet.
    Ready {
        sys: System,
        measured_from: Instant,
        first: bool,
    },
}

fn new_system() -> (System, Instant) {
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    (sys, Instant::now())
}

fn ready((sys, measured_from): (System, Instant)) -> Probe {
    Probe::Ready { sys, measured_from, first: true }
}

impl Default for SystemProbe {
    fn default() -> Self {
        let probe = match std::thread::Builder::new().name("folo-system-probe".into()).spawn(new_system) {
            Ok(thread) => Probe::Starting(thread),
            Err(_) => ready(new_system()),
        };
        SystemProbe {
            sys: Mutex::new(probe),
            #[cfg(test)]
            warm_ups: Default::default(),
        }
    }
}

impl SystemProbe {
    /// Makes the next [`SystemProbe::sample`] quick and its CPU reading real: waits for the counters' set-up and,
    /// before the first reading, out the rest of [`sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`] since it. Blocks.
    pub fn warm_up(&self) {
        #[cfg(test)]
        self.warm_ups.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut probe = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        warm_up(&mut probe);
    }

    /// Current CPU and memory.
    pub fn sample(&self) -> (CpuInfo, MemoryInfo) {
        self.sample_over().0
    }

    /// [`SystemProbe::sample`] and the interval its CPU reading was measured over.
    fn sample_over(&self) -> ((CpuInfo, MemoryInfo), Duration) {
        let mut probe = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        warm_up(&mut probe);
        let Probe::Ready { sys, measured_from, first } = &mut *probe else {
            unreachable!("made ready above")
        };
        let interval = measured_from.elapsed();
        sys.refresh_cpu_usage();
        sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        *measured_from = Instant::now();
        *first = false;
        let util = (sys.global_cpu_usage() / 100.0).clamp(0.0, 1.0);
        let cpu = CpuInfo {
            name: cpu_name(sys),
            threads: sys.cpus().len().max(1) as u32,
            util: (util * 100.0).round() / 100.0,
        };
        let total = sys.total_memory();
        let mem = MemoryInfo { used_bytes: total.saturating_sub(sys.available_memory()), total_bytes: total };
        ((cpu, mem), interval)
    }
}

/// [`SystemProbe::warm_up`] on the locked probe.
fn warm_up(probe: &mut Probe) {
    if let Probe::Starting(_) = probe {
        // Swapped out for a moment to take the thread by value; a set-up that panicked is done again here.
        let placeholder = Probe::Ready { sys: System::new(), measured_from: Instant::now(), first: true };
        let Probe::Starting(thread) = std::mem::replace(probe, placeholder) else {
            unreachable!("matched above")
        };
        *probe = ready(thread.join().unwrap_or_else(|_| new_system()));
    }
    if let Probe::Ready { measured_from, first: true, .. } = probe {
        let left = sysinfo::MINIMUM_CPU_UPDATE_INTERVAL.saturating_sub(measured_from.elapsed());
        if !left.is_zero() {
            std::thread::sleep(left);
        }
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

    /// A reading taken right after the counters were set up has no interval behind it: sysinfo reports 100 %
    /// then (Overview's "CPU 100%" on every cold launch). The first one waits for one.
    #[test]
    fn the_first_reading_is_measured_over_sysinfos_minimum_interval() {
        let probe = SystemProbe::default();
        let (_, first) = probe.sample_over();
        assert!(first >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL, "{first:?}");
        // Warmed up first (as `/api/system` does beside the GPU device), the reading itself is quick.
        let probe = SystemProbe::default();
        probe.warm_up();
        let t = Instant::now();
        let (_, first) = probe.sample_over();
        assert!(first >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL, "{first:?}");
        assert!(t.elapsed() < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL, "{:?}", t.elapsed());
    }
}
