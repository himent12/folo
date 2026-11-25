//! Fused CPU training kernel for [`super::BatchNorm2d`] (identity, SiLU, and ReLU activations).

use folo_core::Tensor;
use rayon::prelude::*;

use crate::Result;

/// The activations [`train_cpu_raw`] applies itself.
pub(super) const ACT_NONE: u8 = 0;
pub(super) const ACT_SILU: u8 = 1;
pub(super) const ACT_RELU: u8 = 2;

/// Activation `ACT` of `z`, exactly as the CPU's `Tensor::silu` / `Tensor::relu` compute it.
#[inline(always)]
fn act_fwd<const ACT: u8>(z: f32) -> f32 {
    match ACT {
        ACT_SILU => folo_core::cpu::silu(z),
        ACT_RELU => z.max(0.0),
        _ => z,
    }
}

/// `g · act'(z)`, exactly as the CPU backward of `Tensor::silu` / `Tensor::relu` computes it.
#[inline(always)]
fn act_bwd<const ACT: u8>(z: f32, g: f32) -> f32 {
    match ACT {
        ACT_SILU => folo_core::cpu::silu_backward(z, g),
        ACT_RELU => {
            if z > 0.0 {
                g
            } else {
                0.0
            }
        }
        _ => g,
    }
}

/// Train-mode BatchNorm followed by activation `ACT` on the CPU: per-channel statistics in `f64` (one read of
/// x, [`plane_stats`]), then the output `act(z)`, `z = (x − mean) · inv · γ + β`, with a hand-written backward:
/// `dx = γ · inv / n · (n · g' − Σg' − x̂ · Σ(g' · x̂))`, `dγ = Σ(g' · x̂)`, `dβ = Σg'`, where
/// `g' = g · act'(z)` (`z` recomputed from `x`), `x̂ = (x − mean) · inv` and `inv = 1 / sqrt(var + eps)`.
/// The activation and its gradient run inside the BatchNorm passes, bit for bit the BatchNorm followed by the
/// separate activation op, without that op's passes over memory or a stored `z`. Returns the output and the
/// batch mean and biased variance as host vectors (`len == C`).
pub(super) fn train_cpu_raw<const ACT: u8>(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f64,
) -> Result<(Tensor, Vec<f32>, Vec<f32>)> {
    let (nb, c, h, w) = x.dims4()?;
    let hw = h * w;
    let n = (nb * hw) as f64;
    let g = gamma.to_vec::<f32>()?;
    let bv = beta.to_vec::<f32>()?;
    // Channel-parallel fuses the two passes per channel while its `nb` planes stay hot in L1/L2; only use it
    // when there are enough channels to load-balance all threads and one channel fits comfortably in cache.
    const CHANNEL_CACHE_FLOATS: usize = 32_768;
    let channel_par =
        hw > 0 && c >= 4 * rayon::current_num_threads() && nb.saturating_mul(hw) <= CHANNEL_CACHE_FLOATS;
    // One read of x: each plane's sum and moments (`plane_stats`), combined per channel in image order: the
    // sums added (the mean is `Σ / n`), the moments merged with Chan's formula (the variance is `M2 / n`).
    // Deterministic for any thread count. When there are enough channels to saturate the thread pool, each
    // channel runs stats + normalise back-to-back so its `nb` planes stay hot in L2/L3; otherwise planes are
    // the parallel unit so a narrow layer still uses every thread.
    let (y, mean, inv, var, scale) = x.with_f32_slice(|xs| {
        // Every element is written below.
        let mut y = folo_core::cpu::take_buffer(xs.len());
        if channel_par {
            let ch_planes = split_planes_by_channel(&mut y, nb, c, hw);
            let ch_stats: Vec<(f32, f32, f32, f32)> = ch_planes
                .into_par_iter()
                .enumerate()
                .map(|(ch, mut planes)| {
                    let (g_ch, b_ch) = (g[ch], bv[ch]);
                    with_avx2(|| {
                        let mut sum = 0.0f64;
                        let mut mom = Moments::default();
                        for b in 0..nb {
                            let base = (b * c + ch) * hw;
                            let (ps, pm) = plane_stats(&xs[base..base + hw]);
                            sum += ps;
                            mom = Moments::merge(mom, pm);
                        }
                        let v = mom.m2 / n;
                        let inv_ch = (1.0 / (v + eps).sqrt()) as f32;
                        let mean_ch = (sum / n) as f32;
                        let scale_ch = inv_ch * g_ch;
                        for (b, o) in planes.iter_mut().enumerate() {
                            let base = (b * c + ch) * hw;
                            for (dst, &val) in o.iter_mut().zip(&xs[base..base + hw]) {
                                *dst = act_fwd::<ACT>((val - mean_ch) * scale_ch + b_ch);
                            }
                        }
                        (mean_ch, inv_ch, v as f32, scale_ch)
                    })
                })
                .collect();
            let mut mean = Vec::with_capacity(c);
            let mut inv = Vec::with_capacity(c);
            let mut var = Vec::with_capacity(c);
            let mut scale = Vec::with_capacity(c);
            for (m, i, v, s) in ch_stats {
                mean.push(m);
                inv.push(i);
                var.push(v);
                scale.push(s);
            }
            (y, mean, inv, var, scale)
        } else {
            let planes: Vec<(f64, Moments)> = if hw == 0 {
                vec![(0.0, Moments::default()); nb * c]
            } else {
                xs.par_chunks(hw).map(|p| with_avx2(|| plane_stats(p))).collect()
            };
            let stats: Vec<(f64, Moments)> = (0..c)
                .map(|ch| {
                    (0..nb)
                        .map(|b| planes[b * c + ch])
                        .fold((0.0, Moments::default()), |a, p| (a.0 + p.0, Moments::merge(a.1, p.1)))
                })
                .collect();
            let var_f64: Vec<f64> = stats.iter().map(|s| s.1.m2 / n).collect();
            let inv: Vec<f32> = var_f64.iter().map(|v| (1.0 / (v + eps).sqrt()) as f32).collect();
            let mean: Vec<f32> = stats.iter().map(|s| (s.0 / n) as f32).collect();
            let var: Vec<f32> = var_f64.iter().map(|&v| v as f32).collect();
            let scale: Vec<f32> = inv.iter().zip(&g).map(|(i, g)| i * g).collect();
            if hw > 0 {
                y.par_chunks_mut(hw).zip(xs.par_chunks(hw)).enumerate().for_each(|(p, (o, xp))| {
                    let ch = p % c;
                    let (m, s, b) = (mean[ch], scale[ch], bv[ch]);
                    with_avx2(|| {
                        for (o, &v) in o.iter_mut().zip(xp) {
                            *o = act_fwd::<ACT>((v - m) * s + b);
                        }
                    });
                });
            }
            (y, mean, inv, var, scale)
        }
    })?;
    let y = Tensor::from_vec(y, x.shape().clone())?;
    let needs_grad =
        folo_core::is_grad_enabled() && (x.requires_grad() || gamma.requires_grad() || beta.requires_grad());
    if !needs_grad {
        return Ok((y, mean, var));
    }
    let mean_out = mean.clone();
    let xd = x.detach();
    let shape = x.shape().clone();
    let name = match ACT {
        ACT_SILU => "batch_norm2d_silu",
        ACT_RELU => "batch_norm2d_relu",
        _ => "batch_norm2d",
    };
    // Per-channel sums of per-plane partial sums, planes added in image order.
    let per_channel = move |plane: &[(f64, f64)]| -> Vec<(f64, f64)> {
        (0..c)
            .map(|ch| (0..nb).map(|b| plane[b * c + ch]).fold((0.0, 0.0), |a, v| (a.0 + v.0, a.1 + v.1)))
            .collect()
    };
    let y = Tensor::custom_op(name, &[x, gamma, beta], y, move |gy| {
        let gy = gy.contiguous()?;
        let (dx, dg, db) = gy.with_f32_slice(|gs| {
            xd.with_f32_slice(|xs| {
                // Every element is written below. With an activation, the first pass computes each plane's
                // `g' = g · act'(z)` and reduces it in L1-resident tiles (`bwd_act_and_sum`); the second pass
                // turns it into dx in place. When `channel_par` holds, both passes run back-to-back per
                // channel so `xs` and `dx`/`gs` stay hot in L2/L3 across the two passes.
                let mut dx = folo_core::cpu::take_buffer(xs.len());
                let sums: Vec<(f64, f64)> = if channel_par {
                    let ch_planes = split_planes_by_channel(&mut dx, nb, c, hw);
                    ch_planes
                        .into_par_iter()
                        .enumerate()
                        .map(|(ch, mut planes)| {
                            let (m, i, s, b_val) = (mean[ch], inv[ch], scale[ch], bv[ch]);
                            let k = g[ch] * i / n as f32;
                            with_avx2(|| {
                                let (mut sum0, mut sum1) = (0.0f64, 0.0f64);
                                for (b, o) in planes.iter_mut().enumerate() {
                                    let base = (b * c + ch) * hw;
                                    let gp = &gs[base..base + hw];
                                    let xp = &xs[base..base + hw];
                                    let (p0, p1) = if ACT == ACT_NONE {
                                        sum2_f64(gp, xp, |gv, v| {
                                            (gv as f64, gv as f64 * ((v - m) * i) as f64)
                                        })
                                    } else {
                                        bwd_act_and_sum::<ACT>(o, gp, xp, m, i, s, b_val)
                                    };
                                    sum0 += p0;
                                    sum1 += p1;
                                }
                                let (sg, sgx) = (sum0 as f32, sum1 as f32);
                                for (b, o) in planes.iter_mut().enumerate() {
                                    let base = (b * c + ch) * hw;
                                    bwd_write_dx::<ACT>(
                                        o,
                                        &gs[base..base + hw],
                                        &xs[base..base + hw],
                                        m,
                                        i,
                                        k,
                                        n as f32,
                                        sg,
                                        sgx,
                                    );
                                }
                                (sum0, sum1)
                            })
                        })
                        .collect()
                } else {
                    let planes: Vec<(f64, f64)> = if hw == 0 {
                        vec![(0.0, 0.0); nb * c]
                    } else {
                        dx.par_chunks_mut(hw)
                            .zip(gs.par_chunks(hw))
                            .zip(xs.par_chunks(hw))
                            .enumerate()
                            .map(|(p, ((o, gp), xp))| {
                                let ch = p % c;
                                let (m, i, s, b_val) = (mean[ch], inv[ch], scale[ch], bv[ch]);
                                with_avx2(|| {
                                    if ACT == ACT_NONE {
                                        sum2_f64(gp, xp, |gv, v| {
                                            (gv as f64, gv as f64 * ((v - m) * i) as f64)
                                        })
                                    } else {
                                        bwd_act_and_sum::<ACT>(o, gp, xp, m, i, s, b_val)
                                    }
                                })
                            })
                            .collect()
                    };
                    let sums = per_channel(&planes);
                    if hw > 0 {
                        dx.par_chunks_mut(hw).enumerate().for_each(|(p, o)| {
                            let ch = p % c;
                            let (m, i) = (mean[ch], inv[ch]);
                            let k = g[ch] * i / n as f32;
                            let (sg, sgx) = (sums[ch].0 as f32, sums[ch].1 as f32);
                            let r = p * hw..(p + 1) * hw;
                            with_avx2(|| {
                                bwd_write_dx::<ACT>(o, &gs[r.clone()], &xs[r], m, i, k, n as f32, sg, sgx);
                            });
                        });
                    }
                    sums
                };
                let dg: Vec<f32> = sums.iter().map(|s| s.1 as f32).collect();
                let db: Vec<f32> = sums.iter().map(|s| s.0 as f32).collect();
                (dx, dg, db)
            })
        })??;
        Ok(vec![
            Some(Tensor::from_vec(dx, shape.clone())?),
            Some(Tensor::from_vec(dg, [c])?),
            Some(Tensor::from_vec(db, [c])?),
        ])
    })?;
    Ok((y, mean_out, var))
}

#[cfg(test)]
pub(super) fn train_cpu<const ACT: u8>(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let c = x.dims4()?.1;
    let (y, mean, var) = train_cpu_raw::<ACT>(x, gamma, beta, eps)?;
    Ok((y, Tensor::from_vec(mean, [c])?, Tensor::from_vec(var, [c])?))
}

/// `f()` compiled for AVX2 when the CPU has it (detected at run time; the build targets baseline x86-64, whose
/// SSE2 loops are half as wide), else as is. Bit for bit the same result either way: the same IEEE operations
/// lane by lane, and Rust never fuses a multiply and an add on its own (FMA is not enabled here).
#[inline(always)]
fn with_avx2<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn run<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        #[cfg(test)]
        let fallback = FORCE_FALLBACK.load(std::sync::atomic::Ordering::SeqCst);
        #[cfg(not(test))]
        let fallback = false;
        if !fallback && std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (detected above).
            return unsafe { run(f) };
        }
    }
    f()
}

/// Test-only: [`with_avx2`] runs its closure as is, so a test can compare the two builds of the loops.
#[cfg(test)]
pub(super) static FORCE_FALLBACK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Count, mean and sum of squared deviations (`M2`) of some values, in `f64`, as [`plane_stats`] makes them
/// and [`Moments::merge`] combines them. The mean here only serves the merge; the reported mean is a sum.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// Chan's parallel combination of two sets' moments: `M2 = M2a + M2b + d² · na · nb / n` with `d` the
    /// difference of the means, a sum of non-negative terms (no cancellation as in `Σv² − n·mean²`).
    fn merge(a: Moments, b: Moments) -> Moments {
        if b.n == 0.0 {
            return a;
        }
        if a.n == 0.0 {
            return b;
        }
        let n = a.n + b.n;
        let d = b.mean - a.mean;
        Moments { n, mean: a.mean + d * (b.n / n), m2: a.m2 + b.m2 + d * d * (a.n * b.n / n) }
    }
}

/// Floats per block of [`plane_stats`]: a multiple of 8 (the lanes of the plane sum), and 1 KiB, so a block
/// is still in L1 for its second loop.
const STAT_BLOCK: usize = 256;

/// One plane's sum and moments in one read, a block of [`STAT_BLOCK`] floats at a time, in `f64`.
///
/// - The **sum** is exactly the two-pass code's: 8 independent accumulators across the whole plane (each
///   block feeds them in turn, so every lane sees the same values in the same order), then the tail. Summed per
///   channel in image order and divided by `n`, it is bit for bit the old mean.
/// - **M2**: each block's own mean, then `Σ(v − mean_b)²` over the block while it is in L1, and the blocks
///   merged with Chan's formula. Every term is a square, so M2 cannot go negative, and a NaN or an infinity
///   in the plane makes it NaN (as the two-pass `Σ(v − mean)²` does).
#[inline]
fn plane_stats(p: &[f32]) -> (f64, Moments) {
    let mut acc = [0f64; 8];
    let mut m = Moments::default();
    let mut tail: &[f32] = &[];
    for blk in p.chunks(STAT_BLOCK) {
        let (chunks, rest) = blk.as_chunks::<8>();
        let mut bs = [0f64; 8];
        for ch in chunks {
            for i in 0..8 {
                acc[i] += ch[i] as f64;
                bs[i] += ch[i] as f64;
            }
        }
        let bn = blk.len() as f64;
        let mean = (bs.iter().sum::<f64>() + rest.iter().map(|&v| v as f64).sum::<f64>()) / bn;
        let mut sq = [0f64; 8];
        for ch in chunks {
            for i in 0..8 {
                let d = ch[i] as f64 - mean;
                sq[i] += d * d;
            }
        }
        let m2 =
            sq.iter().sum::<f64>() + rest.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>();
        m = Moments::merge(m, Moments { n: bn, mean, m2 });
        // Only the last block can have a rest (the blocks are whole lanes): the plane's tail.
        tail = rest;
    }
    let tail: f64 = tail.iter().map(|&v| v as f64).sum();
    (acc.iter().sum::<f64>() + tail, m)
}

/// Split a flat `[N, C, H, W]` buffer (`hw = H·W > 0`) into per-channel lists of `N` disjoint plane slices,
/// ordered by batch index `0..nb`.
fn split_planes_by_channel(buf: &mut [f32], nb: usize, c: usize, hw: usize) -> Vec<Vec<&mut [f32]>> {
    let mut by_ch: Vec<Vec<&mut [f32]>> = (0..c).map(|_| Vec::with_capacity(nb)).collect();
    for (p, plane) in buf.chunks_mut(hw).enumerate() {
        by_ch[p % c].push(plane);
    }
    by_ch
}

/// Floats per L1-resident backward tile: a multiple of 8 (AVX2 width) and 4 (`sum2_f64` lanes), 1 KiB per
/// slice so `o`, `gp` and `xp` (3 KiB total) stay in L1d between `act_bwd` and the reduction.
const BWD_BLOCK: usize = 256;
const _: () = assert!(BWD_BLOCK.is_multiple_of(4));

/// Compute `o = g · act'(z)` and accumulate `(Σ g', Σ g' · x̂)` in `f64` in [`BWD_BLOCK`]-float L1 tiles.
/// Because `BWD_BLOCK` is a multiple of 4, every 4-lane accumulator sees the same elements in the same order
/// as an unblocked pass followed by [`sum2_f64`], keeping the result bit-for-bit identical.
#[inline]
fn bwd_act_and_sum<const ACT: u8>(
    o: &mut [f32],
    gp: &[f32],
    xp: &[f32],
    m: f32,
    i: f32,
    s: f32,
    b: f32,
) -> (f64, f64) {
    let (mut s0, mut s1) = ([0f64; 4], [0f64; 4]);
    let mut tail_o = [0.0f32; 3];
    let mut tail_x = [0.0f32; 3];
    let mut tail_len = 0usize;
    for ((ob, gpb), xpb) in o.chunks_mut(BWD_BLOCK).zip(gp.chunks(BWD_BLOCK)).zip(xp.chunks(BWD_BLOCK)) {
        for ((dst, &gv), &v) in ob.iter_mut().zip(gpb).zip(xpb) {
            *dst = act_bwd::<ACT>((v - m) * s + b, gv);
        }
        let (co, ro) = ob.as_chunks::<4>();
        let (cx, rx) = xpb.as_chunks::<4>();
        for (g4, x4) in co.iter().zip(cx) {
            for k in 0..4 {
                let g = g4[k] as f64;
                s0[k] += g;
                s1[k] += g * ((x4[k] - m) * i) as f64;
            }
        }
        if !ro.is_empty() {
            tail_len = ro.len();
            tail_o[..tail_len].copy_from_slice(ro);
            tail_x[..tail_len].copy_from_slice(rx);
        }
    }
    let (mut t0, mut t1) = (s0.iter().sum::<f64>(), s1.iter().sum::<f64>());
    for k in 0..tail_len {
        let g = tail_o[k] as f64;
        t0 += g;
        t1 += g * ((tail_x[k] - m) * i) as f64;
    }
    (t0, t1)
}

/// Write `dx = k · (n · g' − Σg' − x̂ · Σ(g' · x̂))` for one plane (`g'` is `gp` when `ACT == ACT_NONE`,
/// else already stored in `o`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn bwd_write_dx<const ACT: u8>(
    o: &mut [f32],
    gp: &[f32],
    xp: &[f32],
    m: f32,
    i: f32,
    k: f32,
    n: f32,
    sg: f32,
    sgx: f32,
) {
    if ACT == ACT_NONE {
        for ((dst, &gv), &v) in o.iter_mut().zip(gp).zip(xp) {
            *dst = k * (n * gv - sg - (v - m) * i * sgx);
        }
    } else {
        for (dst, &v) in o.iter_mut().zip(xp) {
            *dst = k * (n * *dst - sg - (v - m) * i * sgx);
        }
    }
}

/// The two sums `(sum f(a, b).0, sum f(a, b).1)` in `f64` with 4 independent accumulator pairs.
#[inline]
fn sum2_f64(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> (f64, f64)) -> (f64, f64) {
    let (mut s0, mut s1) = ([0f64; 4], [0f64; 4]);
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..4 {
            let (p, q) = f(x[i], y[i]);
            s0[i] += p;
            s1[i] += q;
        }
    }
    let (mut t0, mut t1) = (s0.iter().sum::<f64>(), s1.iter().sum::<f64>());
    for (&x, &y) in ra.iter().zip(rb) {
        let (p, q) = f(x, y);
        t0 += p;
        t1 += q;
    }
    (t0, t1)
}
