//! folo-core on the GPU backend vs the CPU backend: a YOLOv8n-shaped network (Conv-BN-SiLU, C2f blocks,
//! SPPF, PAN neck with upsample + concat, decoupled 3-scale head) written with plain folo-core tensor ops.
//!
//! Measures, at 640x640 and batch 1 / 16:
//! - `infer`: inference forward with BatchNorm folded into the conv bias (conv + bias, SiLU), under `no_grad`;
//! - `train fwd`: training forward with BatchNorm from batch statistics (composite ops), recording the graph;
//! - `train fwd+bwd`: the same plus `backward()` of a scalar loss over the head outputs.
//!
//! Every timed iteration ends with `synchronize`, so times are wall-clock GPU latency. Run with
//! `cargo bench -p folo-gpu --bench backend`; `FOLO_BENCH_CPU=0` skips the CPU rows, `FOLO_BENCH_ITERS=n`
//! sets the GPU iteration count (default 20).

use std::time::Instant;

use folo_core::rng::Rng;
use folo_core::{Device, Result, Tensor, no_grad};

struct ConvBn {
    w: Tensor,
    gamma: Tensor,
    beta: Tensor,
    /// Folded inference bias (`beta` here: running mean 0, variance 1).
    bias: Tensor,
    stride: usize,
    pad: usize,
}

impl ConvBn {
    fn new(c1: usize, c2: usize, k: usize, s: usize, rng: &mut Rng) -> Self {
        let fan = (c1 * k * k) as f32;
        ConvBn {
            w: Tensor::randn([c2, c1, k, k], 0.0, (2.0 / fan).sqrt(), rng),
            gamma: Tensor::ones([1, c2, 1, 1]),
            beta: Tensor::zeros([1, c2, 1, 1]),
            bias: Tensor::randn([c2], 0.0, 0.01, rng),
            stride: s,
            pad: k / 2,
        }
    }

    fn to(&self, d: Device, train: bool) -> Self {
        let mv = |t: &Tensor| {
            let t = t.to_device(d).unwrap().detach();
            if train { t.requires_grad_() } else { t }
        };
        ConvBn {
            w: mv(&self.w),
            gamma: mv(&self.gamma),
            beta: mv(&self.beta),
            bias: mv(&self.bias),
            stride: self.stride,
            pad: self.pad,
        }
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![&self.w, &self.gamma, &self.beta]
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        count_macs(x, &self.w, self.stride);
        if !train {
            return x.conv2d(&self.w, Some(&self.bias), self.stride, self.pad, 1, 1)?.silu();
        }
        let y = x.conv2d(&self.w, None, self.stride, self.pad, 1, 1)?;
        let mean = y.mean([0, 2, 3], true)?;
        let d = y.sub(&mean)?;
        let var = d.square()?.mean([0, 2, 3], true)?;
        d.mul(&var.add_scalar(1e-3)?.rsqrt()?)?.mul(&self.gamma)?.add(&self.beta)?.silu()
    }
}

/// A plain conv with bias (the head's final 1x1).
struct Plain {
    w: Tensor,
    b: Tensor,
}

impl Plain {
    fn new(c1: usize, c2: usize, rng: &mut Rng) -> Self {
        Plain { w: Tensor::randn([c2, c1, 1, 1], 0.0, (1.0 / c1 as f32).sqrt(), rng), b: Tensor::zeros([c2]) }
    }
    fn to(&self, d: Device, train: bool) -> Self {
        let mv = |t: &Tensor| {
            let t = t.to_device(d).unwrap().detach();
            if train { t.requires_grad_() } else { t }
        };
        Plain { w: mv(&self.w), b: mv(&self.b) }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        count_macs(x, &self.w, 1);
        x.conv2d(&self.w, Some(&self.b), 1, 0, 1, 1)
    }
}

struct C2f {
    cv1: ConvBn,
    cv2: ConvBn,
    m: Vec<(ConvBn, ConvBn)>,
    shortcut: bool,
}

impl C2f {
    fn new(c1: usize, c2: usize, n: usize, shortcut: bool, rng: &mut Rng) -> Self {
        let c = c2 / 2;
        C2f {
            cv1: ConvBn::new(c1, 2 * c, 1, 1, rng),
            cv2: ConvBn::new((2 + n) * c, c2, 1, 1, rng),
            m: (0..n).map(|_| (ConvBn::new(c, c, 3, 1, rng), ConvBn::new(c, c, 3, 1, rng))).collect(),
            shortcut,
        }
    }
    fn to(&self, d: Device, train: bool) -> Self {
        C2f {
            cv1: self.cv1.to(d, train),
            cv2: self.cv2.to(d, train),
            m: self.m.iter().map(|(a, b)| (a.to(d, train), b.to(d, train))).collect(),
            shortcut: self.shortcut,
        }
    }
    fn params(&self) -> Vec<&Tensor> {
        let mut p = self.cv1.params();
        p.extend(self.cv2.params());
        for (a, b) in &self.m {
            p.extend(a.params());
            p.extend(b.params());
        }
        p
    }
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let y = self.cv1.forward(x, train)?;
        let mut ys = y.chunk(2, 1)?;
        for (a, b) in &self.m {
            let last = ys.last().expect("two chunks").clone();
            let z = b.forward(&a.forward(&last, train)?, train)?;
            ys.push(if self.shortcut { z.add(&last)? } else { z });
        }
        self.cv2.forward(&Tensor::cat(&ys, 1)?, train)
    }
}

struct Sppf {
    cv1: ConvBn,
    cv2: ConvBn,
}

impl Sppf {
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.cv1.forward(x, train)?;
        let y1 = x.max_pool2d(5, 1, 2)?;
        let y2 = y1.max_pool2d(5, 1, 2)?;
        let y3 = y2.max_pool2d(5, 1, 2)?;
        self.cv2.forward(&Tensor::cat(&[&x, &y1, &y2, &y3], 1)?, train)
    }
}

struct Head {
    reg: (ConvBn, ConvBn, Plain),
    cls: (ConvBn, ConvBn, Plain),
}

impl Head {
    fn new(c: usize, rng: &mut Rng) -> Self {
        Head {
            reg: (ConvBn::new(c, 64, 3, 1, rng), ConvBn::new(64, 64, 3, 1, rng), Plain::new(64, 64, rng)),
            cls: (ConvBn::new(c, 80, 3, 1, rng), ConvBn::new(80, 80, 3, 1, rng), Plain::new(80, 80, rng)),
        }
    }
    fn to(&self, d: Device, train: bool) -> Self {
        let t = |b: &(ConvBn, ConvBn, Plain)| (b.0.to(d, train), b.1.to(d, train), b.2.to(d, train));
        Head { reg: t(&self.reg), cls: t(&self.cls) }
    }
    fn params(&self) -> Vec<&Tensor> {
        let mut p = Vec::new();
        for b in [&self.reg, &self.cls] {
            p.extend(b.0.params());
            p.extend(b.1.params());
            p.extend([&b.2.w, &b.2.b]);
        }
        p
    }
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let f = |b: &(ConvBn, ConvBn, Plain)| b.2.forward(&b.1.forward(&b.0.forward(x, train)?, train)?);
        Tensor::cat(&[f(&self.reg)?, f(&self.cls)?], 1)
    }
}

struct Net {
    stem: Vec<ConvBn>,
    c2f: Vec<C2f>,
    downs: Vec<ConvBn>,
    sppf: Sppf,
    neck: Vec<C2f>,
    neck_down: Vec<ConvBn>,
    heads: Vec<Head>,
}

impl Net {
    fn new(seed: u64) -> Self {
        let rng = &mut Rng::seed(seed);
        Net {
            stem: vec![ConvBn::new(3, 16, 3, 2, rng), ConvBn::new(16, 32, 3, 2, rng)],
            c2f: vec![
                C2f::new(32, 32, 1, true, rng),
                C2f::new(64, 64, 2, true, rng),
                C2f::new(128, 128, 2, true, rng),
                C2f::new(256, 256, 1, true, rng),
            ],
            downs: vec![
                ConvBn::new(32, 64, 3, 2, rng),
                ConvBn::new(64, 128, 3, 2, rng),
                ConvBn::new(128, 256, 3, 2, rng),
            ],
            sppf: Sppf { cv1: ConvBn::new(256, 128, 1, 1, rng), cv2: ConvBn::new(512, 256, 1, 1, rng) },
            neck: vec![
                C2f::new(384, 128, 1, false, rng),
                C2f::new(192, 64, 1, false, rng),
                C2f::new(192, 128, 1, false, rng),
                C2f::new(384, 256, 1, false, rng),
            ],
            neck_down: vec![ConvBn::new(64, 64, 3, 2, rng), ConvBn::new(128, 128, 3, 2, rng)],
            heads: vec![Head::new(64, rng), Head::new(128, rng), Head::new(256, rng)],
        }
    }

    fn to(&self, d: Device, train: bool) -> Self {
        Net {
            stem: self.stem.iter().map(|c| c.to(d, train)).collect(),
            c2f: self.c2f.iter().map(|c| c.to(d, train)).collect(),
            downs: self.downs.iter().map(|c| c.to(d, train)).collect(),
            sppf: Sppf { cv1: self.sppf.cv1.to(d, train), cv2: self.sppf.cv2.to(d, train) },
            neck: self.neck.iter().map(|c| c.to(d, train)).collect(),
            neck_down: self.neck_down.iter().map(|c| c.to(d, train)).collect(),
            heads: self.heads.iter().map(|h| h.to(d, train)).collect(),
        }
    }

    fn params(&self) -> Vec<&Tensor> {
        let mut p: Vec<&Tensor> = Vec::new();
        self.stem.iter().for_each(|c| p.extend(c.params()));
        self.c2f.iter().for_each(|c| p.extend(c.params()));
        self.downs.iter().for_each(|c| p.extend(c.params()));
        p.extend(self.sppf.cv1.params());
        p.extend(self.sppf.cv2.params());
        self.neck.iter().for_each(|c| p.extend(c.params()));
        self.neck_down.iter().for_each(|c| p.extend(c.params()));
        self.heads.iter().for_each(|h| p.extend(h.params()));
        p
    }

    /// The three head outputs `[N, 144, H/s, W/s]` for strides 8, 16, 32.
    fn forward(&self, x: &Tensor, train: bool) -> Result<Vec<Tensor>> {
        let x = self.stem[1].forward(&self.stem[0].forward(x, train)?, train)?;
        let x = self.c2f[0].forward(&x, train)?;
        let p3 = self.c2f[1].forward(&self.downs[0].forward(&x, train)?, train)?;
        let p4 = self.c2f[2].forward(&self.downs[1].forward(&p3, train)?, train)?;
        let p5 =
            self.sppf.forward(&self.c2f[3].forward(&self.downs[2].forward(&p4, train)?, train)?, train)?;
        let n4 = self.neck[0].forward(&Tensor::cat(&[p5.upsample_nearest2d(2)?, p4], 1)?, train)?;
        let n3 = self.neck[1].forward(&Tensor::cat(&[n4.upsample_nearest2d(2)?, p3], 1)?, train)?;
        let o4 =
            self.neck[2].forward(&Tensor::cat(&[self.neck_down[0].forward(&n3, train)?, n4], 1)?, train)?;
        let o5 =
            self.neck[3].forward(&Tensor::cat(&[self.neck_down[1].forward(&o4, train)?, p5], 1)?, train)?;
        Ok(vec![
            self.heads[0].forward(&n3, train)?,
            self.heads[1].forward(&o4, train)?,
            self.heads[2].forward(&o5, train)?,
        ])
    }

    /// Convolution GFLOPs (2 x MACs) of one 640x640 image: a forward at 64x64 on the CPU, scaled by 100
    /// (every feature map scales with the input area).
    fn gflops(&self) -> f64 {
        let x = Tensor::zeros([1, 3, 64, 64]);
        let _g = no_grad();
        MACS.with(|m| m.set(0));
        COUNT.with(|c| c.set(true));
        let _ = self.forward(&x, false);
        COUNT.with(|c| c.set(false));
        2.0 * MACS.with(|m| m.get()) as f64 * 100.0 / 1e9
    }
}

thread_local! {
    static MACS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static COUNT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn count_macs(x: &Tensor, w: &Tensor, stride: usize) {
    if COUNT.with(|c| c.get()) {
        let (h, wd) = (x.dims()[2].div_ceil(stride), x.dims()[3].div_ceil(stride));
        let macs = (h * wd * w.numel()) as u64;
        MACS.with(|m| m.set(m.get() + macs));
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time<F: FnMut()>(dev: Device, warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
        folo_core::backend::synchronize(dev).unwrap();
    }
    let mut ts = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        folo_core::backend::synchronize(dev).unwrap();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(ts)
}

fn bench(dev: Device, net: &Net, batch: usize, iters: usize) -> (f64, f64, f64) {
    let x = Tensor::rand_uniform([batch, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(1)).to_device(dev).unwrap();
    let infer = net.to(dev, false);
    let train = net.to(dev, true);
    let warm = if dev.is_gpu() { 10 } else { 1 };
    let t_inf = time(dev, warm, iters, || {
        let _g = no_grad();
        let out = infer.forward(&x, false).unwrap();
        drop(out);
    });
    let t_fwd = time(dev, warm, iters, || {
        let out = train.forward(&x, true).unwrap();
        drop(out);
    });
    let params = train.params();
    let t_bwd = time(dev, warm, iters, || {
        let out = train.forward(&x, true).unwrap();
        let mut loss = out[0].square().unwrap().mean_all().unwrap();
        for o in &out[1..] {
            loss = loss.add(&o.square().unwrap().mean_all().unwrap()).unwrap();
        }
        let g = loss.backward().unwrap();
        assert!(params.iter().all(|p| g.get(p).is_some()));
    });
    (t_inf, t_fwd, t_bwd)
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    println!("adapter: {}", ctx.summary());
    let iters: usize = std::env::var("FOLO_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let cpu = std::env::var("FOLO_BENCH_CPU").map(|v| v != "0").unwrap_or(true);
    let net = Net::new(0);
    let nparams: usize = net.params().iter().map(|p| p.numel()).sum();
    println!(
        "yolov8n-shaped net: {:.2} M params (BN gamma/beta included), {:.1} conv GFLOPs per 640x640 image",
        nparams as f64 / 1e6,
        net.gflops()
    );

    // Dispatches and submissions per pass (batch 1).
    {
        let x = Tensor::zeros([1, 3, 640, 640]).to_device(dev).unwrap();
        let (inf, tr) = (net.to(dev, false), net.to(dev, true));
        for (name, f) in [
            ("infer", Box::new(|| drop(inf.forward(&x, false).unwrap())) as Box<dyn Fn()>),
            (
                "train fwd+bwd",
                Box::new(|| {
                    let out = tr.forward(&x, true).unwrap();
                    let l =
                        out.iter().map(|o| o.sum_all().unwrap()).reduce(|a, b| a.add(&b).unwrap()).unwrap();
                    l.backward().unwrap();
                }),
            ),
        ] {
            let _g = (name == "infer").then(no_grad);
            f();
            folo_core::backend::synchronize(dev).unwrap();
            let before = ctx.work_stats();
            f();
            folo_core::backend::synchronize(dev).unwrap();
            let after = ctx.work_stats();
            println!(
                "{name}: {} dispatches, {} submissions per pass",
                after.dispatches - before.dispatches,
                after.submissions - before.submissions
            );
        }
    }

    if std::env::var("FOLO_BENCH_PROFILE").is_ok() {
        let x = Tensor::zeros([1, 3, 640, 640]).to_device(dev).unwrap();
        let inf = net.to(dev, false);
        let _g = no_grad();
        for _ in 0..5 {
            drop(inf.forward(&x, false).unwrap());
            folo_core::backend::synchronize(dev).unwrap();
        }
        for _ in 0..5 {
            let t0 = Instant::now();
            let out = inf.forward(&x, false).unwrap();
            let t1 = t0.elapsed().as_secs_f64() * 1e3;
            ctx.flush();
            let t2 = t0.elapsed().as_secs_f64() * 1e3;
            folo_core::backend::synchronize(dev).unwrap();
            let t3 = t0.elapsed().as_secs_f64() * 1e3;
            drop(out);
            let t4 = Instant::now();
            folo_core::backend::synchronize(dev).unwrap();
            let t5 = t4.elapsed().as_secs_f64() * 1e3;
            println!(
                "encode {t1:.2} ms, flush {t2:.2}, sync {t3:.2}; empty sync {t5:.3} ms; pool {:?}",
                ctx.pool_stats()
            );
        }
    }
    println!("| batch | device | infer (BN folded) ms | train fwd ms | train fwd+bwd ms |");
    println!("|---|---|---|---|---|");
    for batch in [1usize, 16] {
        let (a, b, c) = bench(dev, &net, batch, iters);
        println!("| {batch} | GPU (wgpu) | {a:.2} | {b:.2} | {c:.2} |");
        if cpu {
            let n = if batch == 1 { 5 } else { 2 };
            let (a, b, c) = bench(Device::Cpu, &net, batch, n);
            println!("| {batch} | CPU | {a:.1} | {b:.1} | {c:.1} |");
        }
    }
    println!("pool: {:?}", ctx.pool_stats());
}
