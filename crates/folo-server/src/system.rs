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

/// CPU SIMD ISA availability and the active CPU int8 (`q8`) micro-kernel tier on this host.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CpuSimd {
    /// x86-64 AVX2 instruction set detected.
    pub avx2: bool,
    /// x86-64 AVX-VNNI (256-bit VEX `vpdpbusd`) detected.
    pub avx_vnni: bool,
    /// x86-64 AVX-512 VNNI (`avx512vnni`) detected.
    pub avx512_vnni: bool,
    /// ARM64 NEON instruction set detected.
    pub neon: bool,
    /// True when the active int8 micro-kernel uses hardware VNNI (`vpdpbusd` via AVX-512 VNNI or AVX-VNNI).
    pub int8_fast: bool,
    /// Human-readable active CPU int8 micro-kernel tier (`"AVX-512 VNNI (512-bit)"`, `"AVX-VNNI"`,
    /// `"AVX-512 VNNI"`, `"AVX2"`, `"Wasm SIMD128"`, or `"portable"`).
    pub int8_kernel: String,
}

/// Detects CPU SIMD ISA capabilities and the active `folo_core::cpu::q8` int8 kernel tier on this machine.
pub fn cpu_simd() -> CpuSimd {
    #[cfg(target_arch = "x86_64")]
    let (avx2, avx_vnni, avx512_vnni, neon) = (
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("avxvnni"),
        std::is_x86_feature_detected!("avx512vnni"),
        false,
    );
    #[cfg(target_arch = "aarch64")]
    let (avx2, avx_vnni, avx512_vnni, neon) =
        (false, false, false, std::arch::is_aarch64_feature_detected!("neon"));
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let (avx2, avx_vnni, avx512_vnni, neon) = (false, false, false, false);
    CpuSimd {
        avx2,
        avx_vnni,
        avx512_vnni,
        neon,
        int8_fast: folo_core::cpu::q8::uses_vnni(),
        int8_kernel: folo_core::cpu::q8::kernel_name().to_string(),
    }
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

/// Keeps a [`System`] between calls so CPU utilisation is measured over the interval since the last reading.
///
/// Setting up the CPU usage counters takes Windows about 150 ms, which `folo serve` spent before it said it was
/// listening (and FOLO Desktop waited on). They are set up on a thread of their own instead, started where the
/// probe is made. Every reading needs an interval behind it: sysinfo measures CPU use between two refreshes, and one
/// made less than [`sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`] after the last refresh (or the set-up) can read 100 %.
/// So [`SystemProbe::warm_up`] (and the first [`SystemProbe::sample`], which does it) waits for the set-up and then
/// out the rest of that interval from when it finished; `/api/system` does that beside creating the GPU device, which
/// takes longer. A later sample sooner than that interval after the last reading gives that reading's CPU use again
/// (a burst of requests); memory is read every time.
pub struct SystemProbe {
    sys: Mutex<Probe>,
    /// Sets up a [`System`] and says when it finished: [`new_system`], or a stand-in in tests.
    setup: fn() -> (System, Instant),
    /// [`SystemProbe::warm_up`] calls so far (`/api/system`'s preparations).
    #[cfg(test)]
    pub(crate) warm_ups: std::sync::atomic::AtomicUsize,
    /// Times a call waited out the interval before the first reading.
    #[cfg(test)]
    waits: std::sync::atomic::AtomicUsize,
    /// Closed by a test to keep [`SystemProbe::warm_up`] from finishing: it passes it once counted.
    #[cfg(test)]
    pub(crate) gate: Gate,
}

/// A gate a test closes to hold blocking jobs where they pass it, until it is opened again.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct Gate {
    closed: Mutex<bool>,
    opened: std::sync::Condvar,
}

#[cfg(test)]
impl Gate {
    /// Closes the gate until the guard is dropped (also when the test fails, so no job stays held).
    pub(crate) fn close(&self) -> impl Drop + '_ {
        struct Open<'a>(&'a Gate);
        impl Drop for Open<'_> {
            fn drop(&mut self) {
                *self.0.closed.lock().unwrap_or_else(|e| e.into_inner()) = false;
                self.0.opened.notify_all();
            }
        }
        *self.closed.lock().unwrap() = true;
        Open(self)
    }

    /// Waits while the gate is closed.
    fn pass(&self) {
        let mut closed = self.closed.lock().unwrap_or_else(|e| e.into_inner());
        while *closed {
            closed = self.opened.wait(closed).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// The probe's [`System`], the thread still setting it up, or neither.
enum Probe {
    /// Not set up: the set-up thread and the retry after it both panicked. The next call sets up again.
    Unset,
    /// The set-up thread, which returns the [`System`] with when it finished.
    Starting(std::thread::JoinHandle<(System, Instant)>),
    /// `measured_from`: when the CPU counters last read (set up or refreshed), the start of the next reading's
    /// interval; `last`: the last reading, `None` before the first. Boxed: a `System` is large on Linux.
    Ready { sys: Box<System>, measured_from: Instant, last: Option<CpuReading> },
}

/// One reading of CPU use (0 to 1) and the interval it was measured over.
#[derive(Clone, Copy)]
struct CpuReading {
    util: f32,
    interval: Duration,
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
    Probe::Ready { sys: Box::new(sys), measured_from, last: None }
}

impl Default for SystemProbe {
    fn default() -> Self {
        SystemProbe::with_setup(new_system)
    }
}

impl SystemProbe {
    /// A probe whose [`System`] `setup` makes, on a thread of its own (here, if the thread can't be started).
    fn with_setup(setup: fn() -> (System, Instant)) -> Self {
        let probe = match std::thread::Builder::new().name("folo-system-probe".into()).spawn(setup) {
            Ok(thread) => Probe::Starting(thread),
            Err(_) => ready(setup()),
        };
        SystemProbe {
            sys: Mutex::new(probe),
            setup,
            #[cfg(test)]
            warm_ups: Default::default(),
            #[cfg(test)]
            waits: Default::default(),
            #[cfg(test)]
            gate: Default::default(),
        }
    }

    /// Makes the next [`SystemProbe::sample`] quick and its CPU reading real: waits for the counters' set-up and,
    /// before the first reading, out the rest of [`sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`] since it. Blocks.
    pub fn warm_up(&self) {
        #[cfg(test)]
        {
            self.warm_ups.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.gate.pass();
        }
        let mut probe = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        self.ready_to_read(&mut probe);
    }

    /// Current CPU and memory.
    pub fn sample(&self) -> (CpuInfo, MemoryInfo) {
        self.sample_over().0
    }

    /// [`SystemProbe::sample`] and the interval its CPU reading was measured over.
    fn sample_over(&self) -> ((CpuInfo, MemoryInfo), Duration) {
        let mut probe = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        self.ready_to_read(&mut probe);
        let Probe::Ready { sys, measured_from, last } = &mut *probe else { unreachable!("made ready above") };
        let since = measured_from.elapsed();
        let reading = match *last {
            // Too soon for a reading of its own: the last one again.
            Some(last) if since < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL => last,
            _ => {
                sys.refresh_cpu_usage();
                *measured_from = Instant::now();
                let util = (sys.global_cpu_usage() / 100.0).clamp(0.0, 1.0);
                *last.insert(CpuReading { util, interval: since })
            }
        };
        sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        let cpu = CpuInfo {
            name: cpu_name(sys),
            threads: sys.cpus().len().max(1) as u32,
            util: (reading.util * 100.0).round() / 100.0,
        };
        let total = sys.total_memory();
        let mem = MemoryInfo { used_bytes: total.saturating_sub(sys.available_memory()), total_bytes: total };
        ((cpu, mem), reading.interval)
    }

    /// Sets the locked probe up if it isn't (waiting for the set-up thread; again here if that panicked or no
    /// set-up finished) and, before the first reading, waits out the rest of the interval since the set-up.
    fn ready_to_read(&self, probe: &mut Probe) {
        // `Unset` while the set-up runs: if it panics, the next call sets up again instead of reading nothing.
        *probe = match std::mem::replace(probe, Probe::Unset) {
            Probe::Starting(thread) => ready(thread.join().unwrap_or_else(|_| (self.setup)())),
            Probe::Unset => ready((self.setup)()),
            set_up => set_up,
        };
        if let Probe::Ready { measured_from, last: None, .. } = probe {
            let left = sysinfo::MINIMUM_CPU_UPDATE_INTERVAL.saturating_sub(measured_from.elapsed());
            if !left.is_zero() {
                #[cfg(test)]
                self.waits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                std::thread::sleep(left);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        // Warmed up first (as `/api/system` does beside the GPU device), the reading itself waits no more.
        let probe = SystemProbe::default();
        probe.warm_up();
        let waited = probe.waits.load(Ordering::Relaxed);
        let (_, first) = probe.sample_over();
        assert!(first >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL, "{first:?}");
        assert_eq!(probe.waits.load(Ordering::Relaxed), waited, "waited again");
    }

    /// Readings asked closer together than sysinfo's interval (a burst of `/api/system` calls) are never
    /// measured over less: such a refresh can read 100 %. They give the last reading again.
    #[test]
    fn readings_closer_than_sysinfos_interval_give_the_last_one_again() {
        let probe = SystemProbe::default();
        let ((first, _), first_over) = probe.sample_over();
        for _ in 0..10 {
            let ((cpu, mem), over) = probe.sample_over();
            assert!(over >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL, "{over:?}");
            assert!(mem.total_bytes > 0);
            // The first reading again, unless this machine stalled long enough for a new one.
            if over == first_over {
                assert_eq!(cpu.util, first.util);
            }
        }
    }

    /// A set-up that panics on its thread and again in the retry leaves the probe not set up (not an empty
    /// `System` taken for one): the next call sets up again and reads real numbers.
    #[test]
    fn a_set_up_that_panicked_twice_is_done_again_by_the_next_call() {
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn fails_twice() -> (System, Instant) {
            if CALLS.fetch_add(1, Ordering::SeqCst) < 2 {
                panic!("set-up failed (on purpose)");
            }
            new_system()
        }
        let probe = SystemProbe::with_setup(fails_twice);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe.sample())).is_err());
        assert_eq!(CALLS.load(Ordering::SeqCst), 2);
        let (cpu, mem) = probe.sample();
        assert_eq!(CALLS.load(Ordering::SeqCst), 3, "set up again");
        assert!(cpu.threads >= 1 && mem.total_bytes > 0);
    }
}
