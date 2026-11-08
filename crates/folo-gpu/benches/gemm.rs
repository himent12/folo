//! GEMM throughput (GFLOPS) at square sizes, plus transposed, batched and f16 variants.
//!
//! `cargo bench -p folo-gpu --bench gemm` (set `FOLO_GPU_ADAPTER` to pick an adapter). With
//! `FOLO_GPU_TENSOR_CORES=1` on an adapter that has them, the f16 rows also run on tensor cores (and
//! `FOLO_GEMM_TUNE=1` then times every tensor-core tile config).

use std::time::Instant;

use folo_gpu::kernels::coop::{self, CoopConfig};
use folo_gpu::kernels::gemm::{self, GemmDesc, TileConfig};
use folo_gpu::{DType, GpuContext};

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// Which kernel a bench row runs.
#[derive(Clone, Copy)]
enum Path {
    /// `gemm::gemm` (tensor cores for f16 when the context has them on).
    Auto,
    /// The regular kernel with this tile config.
    Tiles(TileConfig),
    /// The tensor-core kernel, explicitly (a picked or the given config).
    Coop(Option<CoopConfig>),
}

fn bench(ctx: &GpuContext, d: &GemmDesc, cfg: Option<TileConfig>) -> f64 {
    bench_path(ctx, d, cfg.map_or(Path::Auto, Path::Tiles))
}

fn bench_path(ctx: &GpuContext, d: &GemmDesc, path: Path) -> f64 {
    let (ar, ac) = if d.trans_a { (d.k, d.m) } else { (d.m, d.k) };
    let (br, bc) = if d.trans_b { (d.n, d.k) } else { (d.k, d.n) };
    let batch = d.batch.0 * d.batch.1;
    let es = d.dtype.size();
    let to_bytes = |v: Vec<f32>| -> Vec<u8> {
        match d.dtype {
            DType::F32 => bytemuck::cast_slice(&v).to_vec(),
            DType::F16 => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        }
    };
    let a = ctx.upload(&to_bytes(rand_vec(ar * ac * batch, 1))).unwrap();
    let b = ctx.upload(&to_bytes(rand_vec(br * bc * batch, 2))).unwrap();
    let c = ctx.alloc((d.m * d.n * batch * es) as u64).unwrap();
    let run = |n: usize| {
        for i in 0..n {
            // Submit in small groups so the GPU runs while the next dispatches are encoded.
            if i % 8 == 7 {
                ctx.flush();
            }
            match path {
                Path::Tiles(cfg) => {
                    gemm::gemm_with_config(ctx, d, a.raw(), b.raw(), None, c.raw(), cfg).unwrap()
                }
                Path::Auto => gemm::gemm(ctx, d, a.raw(), b.raw(), None, c.raw()).unwrap(),
                Path::Coop(cfg) => {
                    coop::gemm(ctx, d, d.dtype, a.raw(), b.raw(), None, c.raw(), cfg).unwrap();
                }
            }
        }
        ctx.sync().unwrap();
    };
    run(3);
    let t = Instant::now();
    run(1);
    let once = t.elapsed().as_secs_f64();
    let iters = ((0.15 / once.max(1e-6)) as usize).clamp(3, 200);
    // Best of three timed runs: the GPU is shared and boost clocks vary.
    let secs = (0..3)
        .map(|_| {
            let t = Instant::now();
            run(iters);
            t.elapsed().as_secs_f64() / iters as f64
        })
        .fold(f64::INFINITY, f64::min);
    d.flops() / secs / 1e9
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping GEMM bench: {e}");
            return;
        }
    };
    println!("adapter: {}", ctx.summary());
    if std::env::var("FOLO_GEMM_TUNE").is_ok() {
        let cfgs = [
            TileConfig { bm: 128, bn: 128, bk: 16, tm: 8, tn: 8 },
            TileConfig { bm: 128, bn: 128, bk: 8, tm: 8, tn: 8 },
            TileConfig { bm: 64, bn: 128, bk: 16, tm: 4, tn: 8 },
            TileConfig { bm: 128, bn: 256, bk: 16, tm: 8, tn: 16 },
            TileConfig { bm: 256, bn: 128, bk: 16, tm: 16, tn: 8 },
            TileConfig { bm: 256, bn: 128, bk: 8, tm: 16, tn: 8 },
            TileConfig { bm: 64, bn: 64, bk: 16, tm: 4, tn: 4 },
            TileConfig { bm: 64, bn: 64, bk: 32, tm: 4, tn: 4 },
            TileConfig { bm: 64, bn: 32, bk: 16, tm: 4, tn: 4 },
            TileConfig { bm: 32, bn: 32, bk: 16, tm: 4, tn: 4 },
        ];
        for &n in &[512usize, 768, 1024, 2048, 4096] {
            let d = GemmDesc::new(n, n, n, false, false, 1);
            for cfg in cfgs {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bench(&ctx, &d, Some(cfg))));
                match r {
                    Ok(g) => println!("{n:>5} {cfg:?} {g:>8.0}"),
                    Err(_) => println!("{n:>5} {cfg:?} failed"),
                }
            }
        }
        return;
    }
    let only_sq = std::env::args().any(|a| a == "--square");
    println!("{:<34} {:>10}", "case", "GFLOPS");
    for &n in &[512usize, 1024, 2048, 4096] {
        let d = GemmDesc::new(n, n, n, false, false, 1);
        println!("{:<34} {:>10.0}", format!("sgemm {n}x{n}x{n}"), bench(&ctx, &d, None));
        if std::env::var("FOLO_GEMM_CFG").is_ok() {
            for (name, cfg) in [("large", TileConfig::LARGE), ("small", TileConfig::SMALL)] {
                println!("{:<34} {:>10.0}", format!("  {name} tiles"), bench(&ctx, &d, Some(cfg)));
            }
        }
    }
    if only_sq {
        return;
    }
    for (ta, tb) in [(true, false), (false, true), (true, true)] {
        let d = GemmDesc::new(2048, 2048, 2048, ta, tb, 1);
        let name = format!("sgemm 2048 {}{}", if ta { "A^T" } else { "A" }, if tb { "·B^T" } else { "·B" });
        println!("{:<34} {:>10.0}", name, bench(&ctx, &d, None));
    }
    let d = GemmDesc::new(1000, 1000, 1000, false, false, 1);
    println!("{:<34} {:>10.0}", "sgemm 1000 (unaligned)", bench(&ctx, &d, None));
    let d = GemmDesc::new(256, 256, 256, false, false, 64);
    println!("{:<34} {:>10.0}", "batched 64 x 256^3", bench(&ctx, &d, None));
    if ctx.caps().shader_f16 {
        let tc = ctx.tensor_cores();
        ctx.set_tensor_cores(false);
        for &n in &[1024usize, 2048, 4096] {
            let mut d = GemmDesc::new(n, n, n, false, false, 1);
            d.dtype = DType::F16;
            println!("{:<34} {:>10.0}", format!("hgemm (f16 io, f32 acc) {n}"), bench(&ctx, &d, None));
        }
        ctx.set_tensor_cores(tc);
    }
    if !ctx.caps().coop_f16 {
        println!("tensor cores: {}", ctx.tensor_cores_status());
        return;
    }
    for &n in &[512usize, 1024, 2048, 4096] {
        let mut d = GemmDesc::new(n, n, n, false, false, 1);
        d.dtype = DType::F16;
        println!("{:<34} {:>10.0}", format!("tensor-core hgemm {n}"), bench_path(&ctx, &d, Path::Coop(None)));
        if std::env::var("FOLO_GEMM_TUNE").is_ok() {
            for cfg in [CoopConfig::C128X128, CoopConfig::C64X128, CoopConfig::C64X64, CoopConfig::C32X128] {
                let g = bench_path(&ctx, &d, Path::Coop(Some(cfg)));
                println!("{:<34} {:>10.0}", format!("  {}x{}x{}", cfg.bm, cfg.bn, cfg.bk), g);
            }
        }
    }
    for (ta, tb) in [(true, false), (false, true), (true, true)] {
        let mut d = GemmDesc::new(2048, 2048, 2048, ta, tb, 1);
        d.dtype = DType::F16;
        let name =
            format!("tensor-core 2048 {}{}", if ta { "A^T" } else { "A" }, if tb { "·B^T" } else { "·B" });
        println!("{:<34} {:>10.0}", name, bench_path(&ctx, &d, Path::Coop(None)));
    }
}
