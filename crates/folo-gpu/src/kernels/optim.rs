//! Fused in-place optimizer updates (SGD, AdamW, and `p = a·p + b·g` for EMAs): one dispatch per parameter
//! instead of the ten-odd elementwise ops and copies of the tensor-op formulation.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

static OPTIM: KernelDef = KernelDef {
    name: "optim",
    storage: &[Access::ReadWrite, Access::Read, Access::ReadWrite, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/optim.wgsl")),
};

static UNSCALE: KernelDef = KernelDef {
    name: "unscale_check",
    storage: &[Access::ReadWrite, Access::ReadWrite],
    source: Source::Static(
        "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> FLAG: array<atomic<u32>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let v = X[i] * pf(1u);\n\
         \x20 X[i] = v;\n\
         \x20 // Not finite: all exponent bits set (inf or NaN).\n\
         \x20 if ((bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u) { atomicOr(&FLAG[0], 1u); }\n\
         }\n",
    ),
};

/// The loss scaler's unscale and overflow check: `x *= factor` over `n` f32 elements in place, and
/// `flag[0] |= 1` (a `u32`, never cleared here) if any result is inf or NaN. One dispatch.
pub fn unscale_check(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    n: usize,
    factor: f32,
    flag: &wgpu::Buffer,
) -> Result<()> {
    check_buf(x, n, DType::F32, "unscale_check input")?;
    check_buf(flag, 1, DType::F32, "unscale_check flag")?;
    if n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&UNSCALE, Spec::new())?;
    let p = Params::new().uz(n)?.f(factor);
    ctx.dispatch(&k, &[x, flag], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Tensors one multi-tensor dispatch updates ([`optimizer_step_many`], [`unscale_check_many`]).
pub const MANY: usize = 16;

/// The multi-tensor kernels' binding layouts: per tensor `PARAM` (read-write), `G` (read) and `NS` state buffers
/// (read-write) for the updates (`kind` = `NS` = 0..=2), or one read-write `X` per tensor and the flag last for the
/// unscale (`kind` 3). `j` tensors. Built once per layout and kept for the process (at most `4 · MANY`).
fn many_def(kind: u32, j: usize) -> &'static KernelDef {
    static DEFS: OnceLock<Mutex<HashMap<(u32, usize), &'static KernelDef>>> = OnceLock::new();
    let mut defs = DEFS.get_or_init(Default::default).lock().expect("many defs lock");
    defs.entry((kind, j)).or_insert_with(|| {
        let storage: Vec<Access> = if kind == 3 {
            vec![Access::ReadWrite; j + 1]
        } else {
            (0..j)
                .flat_map(|_| {
                    [Access::ReadWrite, Access::Read].into_iter().chain((0..kind).map(|_| Access::ReadWrite))
                })
                .collect()
        };
        let (name, source) = if kind == 3 {
            ("unscale_check_many", Source::Generated(gen_unscale_many))
        } else {
            ("optim_many", Source::Generated(gen_optim_many))
        };
        Box::leak(Box::new(KernelDef { name, storage: Box::leak(storage.into_boxed_slice()), source }))
    })
}

/// Params of the multi-tensor kernels: the update's own words from 0 (as `shaders/optim.wgsl`), then the first
/// workgroup of tensor `j` at `16 + j` and its element count at `32 + j`.
fn many_params(mut p: Params, ns: &[usize]) -> Result<(Params, u64)> {
    let mut start = 0u64;
    let mut starts = Vec::with_capacity(ns.len());
    for &n in ns {
        starts.push(
            u32::try_from(start).map_err(|_| crate::error::GpuError::Limit("too many elements".into()))?,
        );
        start += (n as u64).div_ceil(256);
    }
    p = p.at(16);
    for s in starts {
        p = p.u(s);
    }
    p = p.at(32);
    for &n in ns {
        p = p.uz(n)?;
    }
    Ok((p, start))
}

/// The multi-tensor kernels' prologue: tensor `j` of this workgroup (the last whose first workgroup is at or
/// before it) and the element `i` of it this invocation takes.
fn many_prologue(s: &mut String, j: usize) {
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let w = linear_id(wid, nwg, 0u, 256u) / 256u;\n\
         \x20 var j = 0u;\n",
    );
    for t in 1..j {
        let _ = writeln!(s, "  if (w >= pu({}u)) {{ j = {t}u; }}", 16 + t);
    }
    s.push_str("  let i = (w - pu(16u + j)) * 256u + lid;\n  switch j {\n");
}

fn gen_optim_many(spec: &Spec) -> String {
    let (j, ns, mode) = (spec.get_u32("J", 1) as usize, spec.get_u32("NS", 0), spec.get_u32("MODE", 0));
    let mut s = String::new();
    let mut b = 1;
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({b}) var<storage, read_write> P{t}: array<f32>;");
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> G{t}: array<f32>;", b + 1);
        b += 2;
        for k in 0..ns {
            let _ = writeln!(s, "@group(0) @binding({b}) var<storage, read_write> S{k}_{t}: array<f32>;");
            b += 1;
        }
    }
    many_prologue(&mut s, j);
    // The update of `shaders/optim.wgsl`, op for op.
    let body = match mode {
        0 => {
            "      var d = g;\n\
             \x20     if (FLAG) { d = g + pf(3u) * p; }\n\
             \x20     p = p + (-pf(1u)) * d;\n"
        }
        1 => {
            "      var d = g;\n\
             \x20     if (FLAG) { d = g + pf(3u) * p; }\n\
             \x20     let mu = pf(2u);\n\
             \x20     let b = mu * S0_T[i] + d;\n\
             \x20     S0_T[i] = b;\n\
             \x20     if (NESTEROV) { d = d + mu * b; } else { d = b; }\n\
             \x20     p = p + (-pf(1u)) * d;\n"
        }
        2 => {
            "      if (FLAG) { p = p * pf(1u); }\n\
             \x20     let m = pf(2u) * S0_T[i] + pf(3u) * g;\n\
             \x20     let v = pf(4u) * S1_T[i] + pf(5u) * (g * g);\n\
             \x20     S0_T[i] = m;\n\
             \x20     S1_T[i] = v;\n\
             \x20     let denom = sqrt(v) / pf(7u) + pf(8u);\n\
             \x20     p = p + (-pf(6u)) * (m / denom);\n"
        }
        _ => "      p = p * pf(1u) + g * pf(2u);\n",
    };
    for t in 0..j {
        let _ = writeln!(
            s,
            "  case {t}u: {{\n    if (i < pu({}u)) {{\n      let g = G{t}[i];\n      var p = P{t}[i];\n{}      P{t}[i] = p;\n    }}\n  }}",
            32 + t,
            body.replace("S0_T", &format!("S0_{t}")).replace("S1_T", &format!("S1_{t}"))
        );
    }
    s.push_str("  default: {}\n  }\n}\n");
    s
}

fn gen_unscale_many(spec: &Spec) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> X{t}: array<f32>;", t + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> FLAG: array<atomic<u32>>;", j + 1);
    many_prologue(&mut s, j);
    for t in 0..j {
        let _ = writeln!(
            s,
            "  case {t}u: {{\n    if (i < pu({}u)) {{\n      let v = X{t}[i] * pf(1u);\n      X{t}[i] = v;\n\
             \x20     if ((bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u) {{ atomicOr(&FLAG[0], 1u); }}\n    }}\n  }}",
            32 + t
        );
    }
    s.push_str("  default: {}\n  }\n}\n");
    s
}

/// One tensor of [`optimizer_step_many`]: the arguments of [`optimizer_step`].
#[derive(Clone, Copy, Debug)]
pub struct ManyItem<'a> {
    /// Parameter.
    pub p: &'a wgpu::Buffer,
    /// Gradient.
    pub g: &'a wgpu::Buffer,
    /// First state buffer (momentum, `exp_avg`).
    pub s0: Option<&'a wgpu::Buffer>,
    /// Second state buffer (`exp_avg_sq`).
    pub s1: Option<&'a wgpu::Buffer>,
    /// Elements.
    pub n: usize,
}

/// [`optimizer_step`] for many tensors with the same update: one dispatch per [`MANY`] tensors (each workgroup
/// updates 256 elements of one of them), the same arithmetic per element. The tensors must all have the same
/// state buffers (none, `s0`, or both); a device that cannot bind that many buffers at once gets one dispatch
/// per tensor.
pub fn optimizer_step_many(ctx: &GpuContext, u: &OptimUpdate, items: &[ManyItem<'_>]) -> Result<()> {
    let Some(first) = items.first() else { return Ok(()) };
    let ns = usize::from(first.s0.is_some()) + usize::from(first.s1.is_some());
    if items.iter().any(|it| {
        usize::from(it.s0.is_some()) + usize::from(it.s1.is_some()) != ns
            || it.s1.is_some() && it.s0.is_none()
    }) {
        return Err(crate::error::shape_err("optimizer_step_many: tensors with different state buffers"));
    }
    let per = 2 + ns;
    let j_max = MANY.min(ctx.caps().max_storage_buffers as usize / per);
    if j_max < 2 {
        for it in items {
            optimizer_step(ctx, u, it.p, it.g, it.s0, it.s1, it.n)?;
        }
        return Ok(());
    }
    for it in items {
        check_buf(it.p, it.n, DType::F32, "optimizer param")?;
        check_buf(it.g, it.n, DType::F32, "optimizer grad")?;
        for s in [it.s0, it.s1].into_iter().flatten() {
            check_buf(s, it.n, DType::F32, "optimizer state")?;
        }
    }
    let (mode, flag, nesterov, base) = update_params(u, ns > 0)?;
    if mode == 2 && ns != 2 {
        return Err(crate::error::shape_err("optimizer_step: AdamW needs exp_avg and exp_avg_sq"));
    }
    let live: Vec<&ManyItem<'_>> = items.iter().filter(|it| it.n > 0).collect();
    for chunk in live.chunks(j_max) {
        let spec = Spec::new()
            .u32("MODE", mode)
            .bool("FLAG", flag)
            .bool("NESTEROV", nesterov)
            .u32("J", chunk.len() as u32)
            .u32("NS", ns as u32);
        let k = ctx.kernel(many_def(ns as u32, chunk.len()), spec)?;
        let ns_: Vec<usize> = chunk.iter().map(|it| it.n).collect();
        let (p, groups) = many_params(base.clone(), &ns_)?;
        let mut bufs: Vec<&wgpu::Buffer> = Vec::with_capacity(chunk.len() * per);
        for it in chunk {
            bufs.push(it.p);
            bufs.push(it.g);
            bufs.extend(it.s0);
            bufs.extend(it.s1);
        }
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
    }
    Ok(())
}

/// [`unscale_check`] for many tensors: one dispatch per [`MANY`] tensors, one shared flag.
pub fn unscale_check_many(
    ctx: &GpuContext,
    xs: &[(&wgpu::Buffer, usize)],
    factor: f32,
    flag: &wgpu::Buffer,
) -> Result<()> {
    let j_max = MANY.min((ctx.caps().max_storage_buffers as usize).saturating_sub(1));
    if j_max < 2 {
        for &(x, n) in xs {
            unscale_check(ctx, x, n, factor, flag)?;
        }
        return Ok(());
    }
    check_buf(flag, 1, DType::F32, "unscale_check flag")?;
    for &(x, n) in xs {
        check_buf(x, n, DType::F32, "unscale_check input")?;
    }
    let live: Vec<&(&wgpu::Buffer, usize)> = xs.iter().filter(|x| x.1 > 0).collect();
    for chunk in live.chunks(j_max) {
        let k = ctx.kernel(many_def(3, chunk.len()), Spec::new().u32("J", chunk.len() as u32))?;
        let ns: Vec<usize> = chunk.iter().map(|x| x.1).collect();
        let (p, groups) = many_params(Params::new().u(0).f(factor), &ns)?;
        let mut bufs: Vec<&wgpu::Buffer> = chunk.iter().map(|x| x.0).collect();
        bufs.push(flag);
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
    }
    Ok(())
}

/// The update's kernel spec (`MODE`, `FLAG`, `NESTEROV`) and parameter words (`shaders/optim.wgsl`), without the
/// element count in word 0.
fn update_params(u: &OptimUpdate, momentum_buffer: bool) -> Result<(u32, bool, bool, Params)> {
    Ok(match *u {
        OptimUpdate::Sgd { lr, momentum, weight_decay, nesterov } => {
            let mode = if momentum_buffer { 1 } else { 0 };
            (mode, weight_decay != 0.0, nesterov, Params::new().u(0).f(lr).f(momentum).f(weight_decay))
        }
        OptimUpdate::AdamW {
            decay,
            beta1,
            one_minus_beta1,
            beta2,
            one_minus_beta2,
            step_size,
            bc2_sqrt,
            eps,
        } => {
            let p = Params::new()
                .u(0)
                .f(decay.unwrap_or(1.0))
                .f(beta1)
                .f(one_minus_beta1)
                .f(beta2)
                .f(one_minus_beta2)
                .f(step_size)
                .f(bc2_sqrt)
                .f(eps);
            (2, decay.is_some(), false, p)
        }
        OptimUpdate::Axpby { a, b } => (3, false, false, Params::new().u(0).f(a).f(b)),
    })
}

/// One update, with its hyper-parameters (the formulas are in `shaders/optim.wgsl` and folo-core's
/// `OptimStep`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimUpdate {
    /// SGD; `momentum` is used with a momentum buffer.
    Sgd {
        /// Learning rate.
        lr: f32,
        /// Momentum.
        momentum: f32,
        /// Coupled weight decay (`0` for none).
        weight_decay: f32,
        /// Nesterov momentum.
        nesterov: bool,
    },
    /// AdamW with precomputed bias corrections.
    AdamW {
        /// `1 − lr · wd`, if decayed.
        decay: Option<f32>,
        /// β1.
        beta1: f32,
        /// `1 − β1`.
        one_minus_beta1: f32,
        /// β2.
        beta2: f32,
        /// `1 − β2`.
        one_minus_beta2: f32,
        /// `lr / (1 − β1^t)`.
        step_size: f32,
        /// `sqrt(1 − β2^t)`.
        bc2_sqrt: f32,
        /// ε.
        eps: f32,
    },
    /// `p = a·p + b·g`.
    Axpby {
        /// Weight of `p`.
        a: f32,
        /// Weight of `g`.
        b: f32,
    },
}

/// Applies `u` to the `n` f32 elements of `p` with gradient `g` and the state buffers it needs: SGD with
/// momentum takes `s0` (the momentum buffer), AdamW takes `s0` / `s1` (`exp_avg`, `exp_avg_sq`).
pub fn optimizer_step(
    ctx: &GpuContext,
    u: &OptimUpdate,
    p: &wgpu::Buffer,
    g: &wgpu::Buffer,
    s0: Option<&wgpu::Buffer>,
    s1: Option<&wgpu::Buffer>,
    n: usize,
) -> Result<()> {
    check_buf(p, n, DType::F32, "optimizer param")?;
    check_buf(g, n, DType::F32, "optimizer grad")?;
    for (s, what) in [(s0, "optimizer state 0"), (s1, "optimizer state 1")] {
        if let Some(s) = s {
            check_buf(s, n, DType::F32, what)?;
        }
    }
    if n == 0 {
        return Ok(());
    }
    let (mode, flag, nesterov, p_) = match *u {
        OptimUpdate::Sgd { lr, momentum, weight_decay, nesterov } => {
            let mode = if s0.is_some() { 1 } else { 0 };
            (mode, weight_decay != 0.0, nesterov, Params::new().uz(n)?.f(lr).f(momentum).f(weight_decay))
        }
        OptimUpdate::AdamW {
            decay,
            beta1,
            one_minus_beta1,
            beta2,
            one_minus_beta2,
            step_size,
            bc2_sqrt,
            eps,
        } => {
            if s0.is_none() || s1.is_none() {
                return Err(crate::error::shape_err("optimizer_step: AdamW needs exp_avg and exp_avg_sq"));
            }
            let p = Params::new()
                .uz(n)?
                .f(decay.unwrap_or(1.0))
                .f(beta1)
                .f(one_minus_beta1)
                .f(beta2)
                .f(one_minus_beta2)
                .f(step_size)
                .f(bc2_sqrt)
                .f(eps);
            (2, decay.is_some(), false, p)
        }
        OptimUpdate::Axpby { a, b } => (3, false, false, Params::new().uz(n)?.f(a).f(b)),
    };
    let k =
        ctx.kernel(&OPTIM, Spec::new().u32("MODE", mode).bool("FLAG", flag).bool("NESTEROV", nesterov))?;
    let (d0, d1) = (s0.unwrap_or(ctx.dummy()), s1.unwrap_or(ctx.dummy()));
    ctx.dispatch(&k, &[p, g, d0, d1], &p_, ctx.grid_1d(n as u64, 256)?)
}
