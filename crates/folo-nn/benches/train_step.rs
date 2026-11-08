//! One CPU training step of a folo-n-sized network: `cargo bench -p folo-nn --bench train_step`.
//!
//! The network follows docs/MODEL.md's folo-n layer table (Conv-BN-SiLU stem and downsamples, C3k2 stages with
//! Bottleneck / C3k blocks, SPPF, the PAN-FPN neck with upsample + concat, and a YOLO11-style decoupled head with
//! depthwise convs, nc = 80) minus the C2PSA attention block. It runs forward, backward and an SGD step at
//! batch 8, 320 px, and prints the best and median times of each phase.
//!
//! `cargo bench -p folo-nn --bench train_step -- --steps N` runs N timed steps (default 5); `--loop` runs
//! forever, for attaching a sampling profiler (`samply record`).

use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Grads, Tensor};
use folo_nn::optim::{Optimizer, ParamGroup, Sgd};
use folo_nn::{Activation, Concat, ConvBnAct, Forward, MaxPool2d, Module, Upsample};

fn conv(c1: usize, c2: usize, k: usize, s: usize, rng: &mut Rng) -> ConvBnAct {
    ConvBnAct::new(c1, c2, k, s, 1, Activation::Silu, rng)
}

fn dwconv(c: usize, k: usize, rng: &mut Rng) -> ConvBnAct {
    ConvBnAct::new(c, c, k, 1, c, Activation::Silu, rng)
}

/// docs/MODEL.md §2.2, always residual.
struct Bottleneck {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
}

impl Bottleneck {
    fn new(c: usize, e: f64, rng: &mut Rng) -> Self {
        let h = (c as f64 * e) as usize;
        Bottleneck { cv1: conv(c, h, 3, 1, rng), cv2: conv(h, c, 3, 1, rng) }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = self.cv2.forward(&self.cv1.forward(x).unwrap()).unwrap();
        x.add(&y).unwrap()
    }
}

/// docs/MODEL.md §2.3.
struct C3k {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    cv3: ConvBnAct,
    blocks: Vec<Bottleneck>,
}

impl C3k {
    fn new(c1: usize, c2: usize, rng: &mut Rng) -> Self {
        let h = c2 / 2;
        C3k {
            cv1: conv(c1, h, 1, 1, rng),
            cv2: conv(c1, h, 1, 1, rng),
            cv3: conv(2 * h, c2, 1, 1, rng),
            blocks: (0..2).map(|_| Bottleneck::new(h, 1.0, rng)).collect(),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut a = self.cv1.forward(x).unwrap();
        for b in &self.blocks {
            a = b.forward(&a);
        }
        let b = self.cv2.forward(x).unwrap();
        self.cv3.forward(&Concat::channels().forward(&[a, b]).unwrap()).unwrap()
    }
}

enum Block {
    Bottleneck(Bottleneck),
    C3k(C3k),
}

/// docs/MODEL.md §2.4 with n = 1.
struct C3k2 {
    c: usize,
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    block: Block,
}

impl C3k2 {
    fn new(c1: usize, c2: usize, c3k: bool, e: f64, rng: &mut Rng) -> Self {
        let c = (c2 as f64 * e) as usize;
        let cv1 = conv(c1, 2 * c, 1, 1, rng);
        let block = if c3k {
            Block::C3k(C3k::new(c, c, rng))
        } else {
            Block::Bottleneck(Bottleneck::new(c, 0.5, rng))
        };
        let cv2 = conv(3 * c, c2, 1, 1, rng);
        C3k2 { c, cv1, cv2, block }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = self.cv1.forward(x).unwrap();
        let parts = y.split_sizes(&[self.c, self.c], 1).unwrap();
        let y2 = match &self.block {
            Block::Bottleneck(b) => b.forward(&parts[1]),
            Block::C3k(b) => b.forward(&parts[1]),
        };
        self.cv2.forward(&Concat::channels().forward(&[&parts[0], &parts[1], &y2]).unwrap()).unwrap()
    }
}

/// docs/MODEL.md §2.5, k = 5.
struct Sppf {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
}

impl Sppf {
    fn new(c1: usize, c2: usize, rng: &mut Rng) -> Self {
        Sppf { cv1: conv(c1, c1 / 2, 1, 1, rng), cv2: conv(2 * c1, c2, 1, 1, rng) }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let pool = MaxPool2d::new(5, 1, 2);
        let x = self.cv1.forward(x).unwrap();
        let y1 = pool.forward(&x).unwrap();
        let y2 = pool.forward(&y1).unwrap();
        let y3 = pool.forward(&y2).unwrap();
        self.cv2.forward(&Concat::channels().forward(&[x, y1, y2, y3]).unwrap()).unwrap()
    }
}

/// One level of the YOLO11 decoupled head: box branch (2 x Conv3x3 + 1x1 to 4 * reg_max) and class branch
/// (DWConv, Conv1x1, DWConv, Conv1x1, 1x1 to nc).
struct HeadLevel {
    bx: [ConvBnAct; 2],
    bx_out: folo_nn::Conv2d,
    cls: [ConvBnAct; 4],
    cls_out: folo_nn::Conv2d,
}

impl HeadLevel {
    fn new(c: usize, c_box: usize, c_cls: usize, nc: usize, rng: &mut Rng) -> Self {
        use folo_nn::{Conv2d, Conv2dConfig};
        HeadLevel {
            bx: [conv(c, c_box, 3, 1, rng), conv(c_box, c_box, 3, 1, rng)],
            bx_out: Conv2d::new(Conv2dConfig::new(c_box, 64, 1), rng),
            cls: [
                dwconv(c, 3, rng),
                conv(c, c_cls, 1, 1, rng),
                dwconv(c_cls, 3, rng),
                conv(c_cls, c_cls, 1, 1, rng),
            ],
            cls_out: Conv2d::new(Conv2dConfig::new(c_cls, nc, 1), rng),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let b = self.bx[1].forward(&self.bx[0].forward(x).unwrap()).unwrap();
        let b = self.bx_out.forward(&b).unwrap();
        let mut c = x.clone();
        for l in &self.cls {
            c = l.forward(&c).unwrap();
        }
        let c = self.cls_out.forward(&c).unwrap();
        Tensor::cat(&[b, c], 1).unwrap()
    }
}

struct Net {
    stem: ConvBnAct,
    down2: ConvBnAct,
    stage2: C3k2,
    down3: ConvBnAct,
    stage3: C3k2,
    down4: ConvBnAct,
    stage4: C3k2,
    down5: ConvBnAct,
    stage5: C3k2,
    sppf: Sppf,
    td4: C3k2,
    out3: C3k2,
    down3n: ConvBnAct,
    out4: C3k2,
    down4n: ConvBnAct,
    out5: C3k2,
    head: [HeadLevel; 3],
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        Net {
            stem: conv(3, 16, 3, 2, rng),
            down2: conv(16, 32, 3, 2, rng),
            stage2: C3k2::new(32, 64, false, 0.25, rng),
            down3: conv(64, 64, 3, 2, rng),
            stage3: C3k2::new(64, 128, false, 0.25, rng),
            down4: conv(128, 128, 3, 2, rng),
            stage4: C3k2::new(128, 128, true, 0.5, rng),
            down5: conv(128, 256, 3, 2, rng),
            stage5: C3k2::new(256, 256, true, 0.5, rng),
            sppf: Sppf::new(256, 256, rng),
            td4: C3k2::new(384, 128, false, 0.5, rng),
            out3: C3k2::new(256, 64, false, 0.5, rng),
            down3n: conv(64, 64, 3, 2, rng),
            out4: C3k2::new(192, 128, false, 0.5, rng),
            down4n: conv(128, 128, 3, 2, rng),
            out5: C3k2::new(384, 256, true, 0.5, rng),
            head: [
                HeadLevel::new(64, 64, 80, 80, rng),
                HeadLevel::new(128, 64, 80, 80, rng),
                HeadLevel::new(256, 64, 80, 80, rng),
            ],
        }
    }

    fn forward(&self, x: &Tensor) -> [Tensor; 3] {
        let cat = Concat::channels();
        let up = Upsample::nearest(2);
        let x = self.down2.forward(&self.stem.forward(x).unwrap()).unwrap();
        let x = self.stage2.forward(&x);
        let p3 = self.stage3.forward(&self.down3.forward(&x).unwrap());
        let p4 = self.stage4.forward(&self.down4.forward(&p3).unwrap());
        let p5 = self.sppf.forward(&self.stage5.forward(&self.down5.forward(&p4).unwrap()));
        let t4 = self.td4.forward(&cat.forward(&[up.forward(&p5).unwrap(), p4]).unwrap());
        let o3 = self.out3.forward(&cat.forward(&[up.forward(&t4).unwrap(), p3]).unwrap());
        let o4 = self.out4.forward(&cat.forward(&[self.down3n.forward(&o3).unwrap(), t4]).unwrap());
        let o5 = self.out5.forward(&cat.forward(&[self.down4n.forward(&o4).unwrap(), p5]).unwrap());
        [self.head[0].forward(&o3), self.head[1].forward(&o4), self.head[2].forward(&o5)]
    }

    fn parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        let mut push = |m: &dyn Module| {
            let i = out.len();
            out.extend(m.named_parameters().into_iter().map(|(name, t)| (format!("m{i}.{name}"), t)));
        };
        let mut convs: Vec<&dyn Module> = vec![
            &self.stem,
            &self.down2,
            &self.down3,
            &self.down4,
            &self.down5,
            &self.down3n,
            &self.down4n,
            &self.sppf.cv1,
            &self.sppf.cv2,
        ];
        for s in [
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
            &self.td4,
            &self.out3,
            &self.out4,
            &self.out5,
        ] {
            convs.push(&s.cv1);
            convs.push(&s.cv2);
            match &s.block {
                Block::Bottleneck(b) => convs.extend([&b.cv1 as &dyn Module, &b.cv2]),
                Block::C3k(c) => {
                    convs.extend([&c.cv1 as &dyn Module, &c.cv2, &c.cv3]);
                    for b in &c.blocks {
                        convs.extend([&b.cv1 as &dyn Module, &b.cv2]);
                    }
                }
            }
        }
        for h in &self.head {
            convs.extend(h.bx.iter().map(|m| m as &dyn Module));
            convs.extend(h.cls.iter().map(|m| m as &dyn Module));
            convs.push(&h.bx_out);
            convs.push(&h.cls_out);
        }
        for m in convs {
            push(m);
        }
        out
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn summary(name: &str, mut v: Vec<Duration>) {
    v.sort();
    println!("  {name:<9} best {:8.1} ms   median {:8.1} ms", ms(v[0]), ms(v[v.len() / 2]));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let forever = args.iter().any(|a| a == "--loop");
    let steps = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5usize);
    let (batch, size) = (8, 320);

    let mut rng = Rng::seed(0);
    let net = Net::new(&mut rng);
    let params = net.parameters();
    let numel: usize = params.iter().map(|(_, t)| t.numel()).sum();
    let group = ParamGroup::new("all", params).lr(1e-3).momentum(0.9).weight_decay(5e-4).nesterov(true);
    let mut sgd = Sgd::new(vec![group]).unwrap();
    let x = Tensor::rand_uniform([batch, 3, size, size], 0.0, 1.0, &mut rng);
    println!(
        "train step: folo-n-like conv stack, {:.2}M parameters, batch {batch}, {size} px, {} rayon threads",
        numel as f64 / 1e6,
        rayon::current_num_threads()
    );

    let mut step = |grads_out: &mut Option<Grads>| -> (Duration, Duration, Duration) {
        let t0 = Instant::now();
        let outs = net.forward(&x);
        let mut loss = outs[0].square().unwrap().mean_all().unwrap();
        for o in &outs[1..] {
            loss = loss.add(&o.square().unwrap().mean_all().unwrap()).unwrap();
        }
        let t1 = Instant::now();
        let grads = loss.backward().unwrap();
        drop((outs, loss));
        let t2 = Instant::now();
        sgd.step(&grads).unwrap();
        let t3 = Instant::now();
        *grads_out = Some(grads);
        (t1 - t0, t2 - t1, t3 - t2)
    };

    let mut keep = None;
    step(&mut keep); // warm-up: buffer cache, momentum buffers
    keep = None;
    if forever {
        loop {
            step(&mut keep);
            keep = None;
        }
    }
    let (mut f, mut b, mut o, mut tot) = (vec![], vec![], vec![], vec![]);
    for _ in 0..steps {
        let (tf, tb, to) = step(&mut keep);
        keep = None;
        f.push(tf);
        b.push(tb);
        o.push(to);
        tot.push(tf + tb + to);
    }
    summary("forward", f);
    summary("backward", b);
    summary("sgd", o);
    summary("total", tot);
}
