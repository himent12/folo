//! Fused in-place optimizer updates (SGD, AdamW, and `p = a·p + b·g` for EMAs): one dispatch per parameter
//! instead of the ten-odd elementwise ops and copies of the tensor-op formulation.

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
