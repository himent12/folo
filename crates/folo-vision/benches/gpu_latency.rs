//! Where the time of a synchronised batch-1 forward goes, call by call. Run with
//! `cargo bench -p folo-vision --bench gpu_latency` (`FOLO_BENCH_SCALES` (default `n`), `FOLO_BENCH_PRECISION`
//! (`f32,f16tc`; `f16tc` needs `FOLO_GPU_TENSOR_CORES=1`), `FOLO_BENCH_CALLS` (default 1000), `FOLO_BENCH_GAP_MS`
//! (time the thread sleeps after each call, default 0), `FOLO_BENCH_SPINS` (wait spins in microseconds to compare in
//! one process, interleaved in blocks of 50 calls, for example `0,300,2000`; default: the context's),
//! `FOLO_BENCH_RAMP=1` (trace the recovery after two idle seconds first), `FOLO_BENCH_ECORES` (see below)).
//!
//! The fused model at 640×640, input on the GPU, every call synchronised (the `gpu` bench's forward b1 latency row).
//! Per call: host time encoding the forward, submitting the rest of the batch (`flush`), and waiting for the GPU
//! (`sync`), and the logical processor the thread ran on right before and right after the wait, and at the start of
//! the call. The processor number is `rdtscp`'s `IA32_TSC_AUX`, which Windows and Linux set to it (that is the OS's
//! choice, not the processor's; other systems may store something else there); without `rdtscp` it is not recorded.
//! `FOLO_BENCH_ECORES` (for example `16-27`, the E-cores of an i7-14700KF under Windows; unset by default) names the
//! processor numbers to treat as E-cores, which splits the report into calls that touched one and the others: the
//! numbering is the machine's and the OS's, so the bench does not guess it. Meanwhile `nvidia-smi` samples the GPU
//! clocks, P-state and utilisation every 50 ms. Printed: percentiles of each phase, the slowest decile split the same
//! way, processor changes, and the clocks seen.

use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_nn::Module;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{Precision, f16_unavailable};

/// Whether this processor has `rdtscp` (CPUID extended leaf `0x8000_0001`, EDX bit 27).
#[cfg(target_arch = "x86_64")]
fn has_rdtscp() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        use core::arch::x86_64::__cpuid;
        // SAFETY: `cpuid` exists on every x86-64 processor; leaves above the maximum are not queried.
        #[allow(unused_unsafe)]
        unsafe {
            __cpuid(0x8000_0000).eax >= 0x8000_0001 && __cpuid(0x8000_0001).edx & (1 << 27) != 0
        }
    })
}

/// The logical processor this thread runs on: `rdtscp`'s `IA32_TSC_AUX`, which Windows and Linux set to the processor
/// number (an OS convention); `u32::MAX` without `rdtscp` or off x86-64.
fn cpu() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_rdtscp() {
            return u32::MAX;
        }
        let mut aux = 0;
        // SAFETY: CPUID reported `rdtscp` above.
        unsafe { core::arch::x86_64::__rdtscp(&mut aux) };
        aux & 0xfff
    }
    #[cfg(not(target_arch = "x86_64"))]
    u32::MAX
}

/// One call's phases in milliseconds and the processors it ran on.
struct Call {
    enc: f64,
    sub: f64,
    wait: f64,
    /// At the start of the call, right before `synchronize` and right after it.
    cpus: [u32; 3],
}

impl Call {
    fn total(&self) -> f64 {
        self.enc + self.sub + self.wait
    }
}

/// The `p` quantile of `v` (sorted in place); NaN for no values.
fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

/// `nvidia-smi` sampling clocks every 50 ms until finished or dropped (a drop, also in a panic, stops it). A thread
/// drains its output as it comes, noting when each line arrived (`nvidia-smi` writes a line per sample), so a long
/// run cannot fill the pipe and stall the sampling.
struct Clocks {
    child: Child,
    lines: Option<JoinHandle<Vec<(Instant, String)>>>,
}

/// One `nvidia-smi` sample.
struct Sample {
    at: Instant,
    gr: f64,
    mem: f64,
    state: String,
    util: f64,
}

impl Clocks {
    fn start() -> Option<Self> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=clocks.gr,clocks.mem,pstate,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "50",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let out = child.stdout.take()?;
        let lines = std::thread::spawn(move || {
            BufReader::new(out).lines().map_while(Result::ok).map(|l| (Instant::now(), l)).collect()
        });
        Some(Self { child, lines: Some(lines) })
    }

    /// Stops sampling and returns the samples.
    fn finish(mut self) -> Vec<Sample> {
        self.stop();
        let lines = self.lines.take().and_then(|h| h.join().ok()).unwrap_or_default();
        lines
            .into_iter()
            .filter_map(|(at, line)| {
                let f: Vec<&str> = line.split(',').map(str::trim).collect();
                if f.len() < 4 {
                    return None;
                }
                Some(Sample {
                    at,
                    gr: f[0].parse().ok()?,
                    mem: f[1].parse().ok()?,
                    state: f[2].to_string(),
                    util: f[3].parse().ok()?,
                })
            })
            .collect()
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Clocks {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Graphics clock percentiles, memory clock, utilisation and P-states of `samples`.
fn summary(samples: &[&Sample]) -> String {
    if samples.is_empty() {
        return "n/a".into();
    }
    let col = |f: fn(&Sample) -> f64| samples.iter().map(|s| f(s)).collect::<Vec<f64>>();
    let (mut gr, mut mem, mut util) = (col(|s| s.gr), col(|s| s.mem), col(|s| s.util));
    let mut states = std::collections::BTreeMap::<&str, usize>::new();
    for s in samples {
        *states.entry(s.state.as_str()).or_default() += 1;
    }
    format!(
        "{} samples: graphics clock p10 {:.0} / p50 {:.0} / max {:.0} MHz, memory clock p10 {:.0} / p50 {:.0} MHz, \
         utilisation p50 {:.0}%, P-states {states:?}",
        samples.len(),
        pct(&mut gr, 0.1),
        pct(&mut gr, 0.5),
        pct(&mut gr, 1.0),
        pct(&mut mem, 0.1),
        pct(&mut mem, 0.5),
        pct(&mut util, 0.5),
    )
}

/// Summary of every sample, or why there is none.
fn clocks_line(clocks: Option<Clocks>) -> String {
    match clocks {
        Some(c) => summary(&c.finish().iter().collect::<Vec<_>>()),
        None => "nvidia-smi not available".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    scale: &str,
    dev: Device,
    ctx: &GpuContext,
    prec: Precision,
    tc: bool,
    calls: usize,
    gap: Duration,
    spins: &[Duration],
    spin0: Duration,
    e_cores: &[u32],
) {
    ctx.set_tensor_cores(tc);
    // The context's own spin for the warm-up and the ramp, not the previous run's last setting.
    ctx.set_wait_spin(spin0);
    let label = if tc { format!("{prec}, tensor cores") } else { prec.to_string() };
    let mut model = FoloModel::new(ModelConfig::new(scale, 80).expect("config")).expect("model");
    model.to_device(dev).expect("to_device");
    model.fuse().expect("fuse");
    model.to_dtype(prec.dtype()).expect("to_dtype");
    // Recorded forwards, as inside a `Predictor` (a model on its own has them off).
    model.set_forward_plans(true);
    let _g = no_grad();
    let x = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(0))
        .to_device(dev)
        .expect("upload")
        .to_dtype(prec.dtype())
        .expect("dtype");
    let forward = || drop(black_box(model.forward(&x).expect("forward")));
    for _ in 0..25 {
        forward();
        synchronize(dev).expect("sync");
    }
    if std::env::var("FOLO_BENCH_RAMP").is_ok_and(|v| v != "0" && !v.is_empty()) {
        ramp(dev, &forward, &label, scale);
    }
    // One setting per block of calls, cycling, so every setting sees the same stretches of machine load; cycle `c`
    // starts with setting `c % n`, so no setting always goes first.
    let settings: Vec<Option<Duration>> =
        if spins.is_empty() { vec![None] } else { spins.iter().copied().map(Some).collect() };
    const BLOCK: usize = 50;
    let n = settings.len();
    let clocks = Clocks::start();
    let mut logs: Vec<Vec<Call>> = settings.iter().map(|_| Vec::with_capacity(calls)).collect();
    let mut i = 0;
    while logs.iter().any(|l| l.len() < calls) {
        let block = i / BLOCK;
        let k = (block % n + block / n) % n;
        i += 1;
        if logs[k].len() >= calls {
            continue;
        }
        if let Some(spin) = settings[k] {
            ctx.set_wait_spin(spin);
        }
        if !gap.is_zero() {
            std::thread::sleep(gap);
        }
        let c0 = cpu();
        let t0 = Instant::now();
        forward();
        let t1 = Instant::now();
        ctx.flush();
        let t2 = Instant::now();
        let c1 = cpu();
        synchronize(dev).expect("sync");
        let c2 = cpu();
        let t3 = Instant::now();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        logs[k].push(Call { enc: ms(t0, t1), sub: ms(t1, t2), wait: ms(t2, t3), cpus: [c0, c1, c2] });
    }
    let clocks = clocks_line(clocks);
    println!(
        "\nfolo-{scale} {label} b1, {calls} synchronised calls per setting, {} ms sleep after each call",
        gap.as_millis()
    );
    for (setting, log) in settings.iter().zip(&logs) {
        match setting {
            Some(s) => println!(" wait spin {} us:", s.as_micros()),
            None => println!(" wait spin {} us (the context's):", ctx.wait_spin().as_micros()),
        }
        report(log, e_cores);
    }
    println!("  GPU: {clocks}");
}

/// After two seconds idle (the GPU drops its clocks), synchronised calls back to back for two seconds: the median
/// latency of every 50 ms of them, and the median graphics clock `nvidia-smi` read in each 50 ms (by when its sample
/// arrived), which shows how long the clocks take to ramp up again.
fn ramp(dev: Device, forward: &dyn Fn(), label: &str, scale: &str) {
    const BUCKETS: usize = 40;
    let clocks = Clocks::start();
    std::thread::sleep(Duration::from_secs(2));
    let start = Instant::now();
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); BUCKETS];
    while start.elapsed() < Duration::from_secs(2) {
        let t = Instant::now();
        forward();
        synchronize(dev).expect("sync");
        let at = (t - start).as_millis() as usize / 50;
        buckets[at.min(BUCKETS - 1)].push(t.elapsed().as_secs_f64() * 1e3);
    }
    let end = Instant::now();
    let cells = |b: &mut [Vec<f64>], digits: usize| {
        b.iter_mut()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .map(|(i, b)| format!("{}ms:{:.*}", i * 50, digits, pct(b, 0.5)))
            .collect::<Vec<String>>()
            .join(" ")
    };
    println!("\nfolo-{scale} {label} b1 after 2 s idle, median per 50 ms: {}", cells(&mut buckets, 2));
    let Some(clocks) = clocks else {
        println!("  GPU: nvidia-smi not available");
        return;
    };
    let samples = clocks.finish();
    let idle: Vec<&Sample> = samples.iter().filter(|s| s.at < start).collect();
    let busy: Vec<&Sample> = samples.iter().filter(|s| s.at >= start && s.at <= end).collect();
    let mut gr: Vec<Vec<f64>> = vec![Vec::new(); BUCKETS];
    for s in &busy {
        gr[((s.at - start).as_millis() as usize / 50).min(BUCKETS - 1)].push(s.gr);
    }
    println!("  graphics clock (MHz) per 50 ms: {}", cells(&mut gr, 0));
    println!("  GPU while idle: {}", summary(&idle));
    println!("  GPU during the calls: {}", summary(&busy));
}

/// Percentiles of every phase over all calls, the slowest decile, calls touching an E-core and the others (only with
/// `e_cores` named), processor changes.
fn report(log: &[Call], e_cores: &[u32]) {
    let col = |f: &dyn Fn(&Call) -> f64, calls: &[&Call]| calls.iter().map(|c| f(c)).collect::<Vec<f64>>();
    let line = |name: &str, calls: &[&Call]| {
        let mut parts = Vec::new();
        for (phase, f) in [
            ("total", &Call::total as &dyn Fn(&Call) -> f64),
            ("encode", &|c: &Call| c.enc),
            ("submit", &|c: &Call| c.sub),
            ("wait", &|c: &Call| c.wait),
        ] {
            let mut v = col(f, calls);
            parts.push(format!(
                "{phase} p10 {:.2} p50 {:.2} p90 {:.2} p99 {:.2}",
                pct(&mut v, 0.1),
                pct(&mut v, 0.5),
                pct(&mut v, 0.9),
                pct(&mut v, 0.99)
            ));
        }
        println!("  {name} ({} calls): {}", calls.len(), parts.join(" | "));
    };
    let all: Vec<&Call> = log.iter().collect();
    line("all", &all);
    let mut totals: Vec<f64> = log.iter().map(Call::total).collect();
    let p90 = pct(&mut totals, 0.9);
    let slow: Vec<&Call> = log.iter().filter(|c| c.total() >= p90).collect();
    line("slowest decile", &slow);
    if !e_cores.is_empty() {
        let e_core = |c: &&Call| c.cpus.iter().any(|p| e_cores.contains(p));
        let on_e: Vec<&Call> = log.iter().filter(e_core).collect();
        let on_p: Vec<&Call> = log.iter().filter(|c| !e_core(c)).collect();
        line("touching a processor of FOLO_BENCH_ECORES", &on_e);
        line("the other processors only", &on_p);
    }
    if log.iter().any(|c| c.cpus.contains(&u32::MAX)) {
        println!("  processor numbers: not available (no rdtscp)");
        return;
    }
    let wait = log.iter().filter(|c| c.cpus[1] != c.cpus[2]).count();
    let call = log.iter().filter(|c| c.cpus[0] != c.cpus[2]).count();
    println!(
        "  calls whose thread changed processor across the wait (right before / after synchronize): {wait}; \
         between the start of the call and the end of the wait: {call}"
    );
}

/// `FOLO_BENCH_ECORES`: processor numbers and ranges, for example `16-27` or `8,10,12-15`.
fn parse_cores(s: &str) -> Vec<u32> {
    let num = |v: &str| {
        v.trim().parse::<u32>().expect("FOLO_BENCH_ECORES: processor numbers and ranges, e.g. 16-27")
    };
    let mut cores = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => cores.extend(num(a)..=num(b)),
            None => cores.push(num(part)),
        }
    }
    cores
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to measure");
            return;
        }
    };
    let ctx = GpuContext::shared().expect("context");
    let s = ctx.summary();
    println!("folo-vision GPU latency on {} ({}, driver {})", s.name, s.backend, s.driver);
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let calls: usize = var("FOLO_BENCH_CALLS", "1000")
        .parse()
        .ok()
        .filter(|&n| n > 0)
        .expect("FOLO_BENCH_CALLS: a number of calls per setting, at least 1");
    let gap = Duration::from_millis(var("FOLO_BENCH_GAP_MS", "0").parse().expect("FOLO_BENCH_GAP_MS"));
    // Wait spins to compare, interleaved in blocks of calls; none given: the context's own.
    let spins: Vec<Duration> = var("FOLO_BENCH_SPINS", "")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| Duration::from_micros(s.parse().expect("FOLO_BENCH_SPINS: microseconds")))
        .collect();
    let e_cores = parse_cores(&var("FOLO_BENCH_ECORES", ""));
    let spin0 = ctx.wait_spin();
    let tc = ctx.tensor_cores();
    for scale in var("FOLO_BENCH_SCALES", "n").split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for p in var("FOLO_BENCH_PRECISION", "f32,f16tc").split(',').map(str::trim).filter(|s| !s.is_empty())
        {
            let (prec, on) = match p {
                "f16tc" => (Precision::F16, true),
                p => (p.parse().expect("FOLO_BENCH_PRECISION: f32, f16 and / or f16tc"), false),
            };
            if prec == Precision::F16
                && let Some(why) = f16_unavailable(dev)
            {
                println!("\nfolo-{scale} f16: skipped ({why})");
                continue;
            }
            if on && !ctx.caps().coop_f16 {
                println!("\nfolo-{scale} f16 on tensor cores: skipped ({})", ctx.tensor_cores_status());
                continue;
            }
            run(scale, dev, &ctx, prec, on, calls, gap, &spins, spin0, &e_cores);
        }
    }
    ctx.set_tensor_cores(tc);
    ctx.set_wait_spin(spin0);
}
