//! Where the time of a synchronised batch-1 forward goes, call by call. Run with
//! `cargo bench -p folo-vision --bench gpu_latency` (`FOLO_BENCH_SCALES` (default `n`), `FOLO_BENCH_PRECISION`
//! (`f32,f16tc`; `f16tc` needs `FOLO_GPU_TENSOR_CORES=1`), `FOLO_BENCH_CALLS` (default 1000), `FOLO_BENCH_GAP_MS`
//! (idle time between calls, default 0), `FOLO_BENCH_SPINS` (wait spins in microseconds to compare in one process,
//! interleaved in blocks of 50 calls, for example `0,300,2000`; default: the context's)).
//!
//! The fused model at 640×640, input on the GPU, every call synchronised (the `gpu` bench's forward b1 latency row).
//! Per call: host time encoding the forward, submitting the rest of the batch (`flush`), and waiting for the GPU
//! (`sync`), and the logical processor the thread ran on before and after the wait (`rdtscp`'s processor number;
//! on an i7-14700KF 0–15 are the P-cores' hyperthreads, 16–27 the E-cores). Meanwhile `nvidia-smi` samples the GPU
//! clocks, P-state and utilisation every 50 ms. Printed: percentiles of each phase, the slowest decile split the
//! same way, the calls on E-cores, and the clocks seen.

use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_nn::Module;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{Precision, f16_unavailable};

/// The logical processor this thread runs on (`IA32_TSC_AUX`, which Windows and Linux set to it), or `u32::MAX`.
fn cpu() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut aux = 0;
        // SAFETY: `rdtscp` has no preconditions on x86-64 processors made since 2008.
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
    cpus: [u32; 2],
}

impl Call {
    fn total(&self) -> f64 {
        self.enc + self.sub + self.wait
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

/// `nvidia-smi` sampling clocks every 50 ms until dropped.
struct Clocks(std::process::Child);

impl Clocks {
    fn start() -> Option<Self> {
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=clocks.gr,clocks.mem,pstate,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "50",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()
            .map(Self)
    }

    /// Stops sampling and summarises: graphics clock percentiles, P-states, utilisation.
    fn finish(mut self) -> String {
        let _ = self.0.kill();
        let Some(out) = self.0.stdout.take() else { return "n/a".into() };
        let mut gr = Vec::new();
        let mut mem = Vec::new();
        let mut states = std::collections::BTreeMap::<String, usize>::new();
        let mut util = Vec::new();
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 4 {
                continue;
            }
            if let (Ok(g), Ok(m), Ok(u)) = (f[0].parse::<f64>(), f[1].parse::<f64>(), f[3].parse::<f64>()) {
                gr.push(g);
                mem.push(m);
                util.push(u);
                *states.entry(f[2].to_string()).or_default() += 1;
            }
        }
        let _ = self.0.wait();
        if gr.is_empty() {
            return "n/a".into();
        }
        let n = gr.len();
        format!(
            "{n} samples: graphics clock p10 {:.0} / p50 {:.0} / max {:.0} MHz, memory clock p10 {:.0} / p50 {:.0} MHz, \
             utilisation p50 {:.0}%, P-states {states:?}",
            pct(&mut gr, 0.1),
            pct(&mut gr, 0.5),
            pct(&mut gr, 1.0),
            pct(&mut mem, 0.1),
            pct(&mut mem, 0.5),
            pct(&mut util, 0.5),
        )
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
) {
    ctx.set_tensor_cores(tc);
    let label = if tc { format!("{prec}, tensor cores") } else { prec.to_string() };
    let mut model = FoloModel::new(ModelConfig::new(scale, 80).expect("config")).expect("model");
    model.to_device(dev).expect("to_device");
    model.fuse().expect("fuse");
    model.to_dtype(prec.dtype()).expect("to_dtype");
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
    // One setting per block of calls, cycling, so every setting sees the same stretches of machine load.
    let settings: Vec<Option<Duration>> =
        if spins.is_empty() { vec![None] } else { spins.iter().copied().map(Some).collect() };
    const BLOCK: usize = 50;
    let clocks = Clocks::start();
    let mut logs: Vec<Vec<Call>> = settings.iter().map(|_| Vec::with_capacity(calls)).collect();
    let mut i = 0;
    while logs.iter().any(|l| l.len() < calls) {
        let k = (i / BLOCK) % settings.len();
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
        synchronize(dev).expect("sync");
        let t3 = Instant::now();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        logs[k].push(Call { enc: ms(t0, t1), sub: ms(t1, t2), wait: ms(t2, t3), cpus: [c0, cpu()] });
    }
    let clocks = clocks.map(Clocks::finish).unwrap_or_else(|| "nvidia-smi not available".into());
    println!(
        "\nfolo-{scale} {label} b1, {calls} synchronised calls per setting, {} ms idle between calls",
        gap.as_millis()
    );
    for (setting, log) in settings.iter().zip(&logs) {
        match setting {
            Some(s) => println!(" wait spin {} us:", s.as_micros()),
            None => println!(" wait spin {} us (the context's):", ctx.wait_spin().as_micros()),
        }
        report(log);
    }
    println!("  GPU: {clocks}");
}

/// After two seconds idle (the GPU drops its clocks), synchronised calls back to back for two seconds: the median
/// latency of every 50 ms of them, which shows how long the clocks take to ramp up again.
fn ramp(dev: Device, forward: &dyn Fn(), label: &str, scale: &str) {
    std::thread::sleep(Duration::from_secs(2));
    let start = Instant::now();
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); 40];
    while start.elapsed() < Duration::from_secs(2) {
        let t = Instant::now();
        forward();
        synchronize(dev).expect("sync");
        let at = (t - start).as_millis() as usize / 50;
        buckets[at.min(39)].push(t.elapsed().as_secs_f64() * 1e3);
    }
    let cells: Vec<String> = buckets
        .iter_mut()
        .enumerate()
        .filter(|(_, b)| !b.is_empty())
        .map(|(i, b)| format!("{}ms:{:.2}", i * 50, pct(b, 0.5)))
        .collect();
    println!("\nfolo-{scale} {label} b1 after 2 s idle, median per 50 ms: {}", cells.join(" "));
}

/// Percentiles of every phase over all calls, the slowest decile, E-core and P-core calls, processor changes.
fn report(log: &[Call]) {
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
    let e_core = |c: &&Call| c.cpus.iter().any(|&p| (16..28).contains(&p));
    let on_e: Vec<&Call> = log.iter().filter(e_core).collect();
    let on_p: Vec<&Call> = log.iter().filter(|c| !e_core(c)).collect();
    if !on_e.is_empty() && !on_p.is_empty() {
        line("touching an E-core", &on_e);
        line("P-cores only", &on_p);
    }
    let moved = log.iter().filter(|c| c.cpus[0] != c.cpus[1]).count();
    println!("  calls whose thread changed processor across the wait: {moved}");
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
    let calls: usize = var("FOLO_BENCH_CALLS", "1000").parse().expect("FOLO_BENCH_CALLS");
    let gap = Duration::from_millis(var("FOLO_BENCH_GAP_MS", "0").parse().expect("FOLO_BENCH_GAP_MS"));
    // Wait spins to compare, interleaved in blocks of calls; none given: the context's own.
    let spins: Vec<Duration> = var("FOLO_BENCH_SPINS", "")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| Duration::from_micros(s.parse().expect("FOLO_BENCH_SPINS: microseconds")))
        .collect();
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
            run(scale, dev, &ctx, prec, on, calls, gap, &spins);
        }
    }
    ctx.set_tensor_cores(tc);
    ctx.set_wait_spin(spin0);
}
