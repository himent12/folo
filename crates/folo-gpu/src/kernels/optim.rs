//! Fused in-place optimizer updates (SGD, AdamW, and `p = a·p + b·g` for EMAs): one dispatch per parameter
//! instead of the ten-odd elementwise ops and copies of the tensor-op formulation.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

use super::{check_buf, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

fn gen_optim(spec: &Spec) -> String {
    let ty = if spec.get_bool("VEC4") { "vec4<f32>" } else { "f32" };
    format!("alias V = {ty};\n{}", include_str!("../shaders/optim.wgsl"))
}

static OPTIM: KernelDef = KernelDef {
    name: "optim",
    storage: &[Access::ReadWrite, Access::Read, Access::ReadWrite, Access::ReadWrite],
    source: Source::Generated(gen_optim),
};

fn nonfinite_expr(vec4: bool) -> &'static str {
    if vec4 {
        "any((bitcast<vec4<u32>>(v) & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u))"
    } else {
        "(bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u"
    }
}

fn gen_unscale(spec: &Spec) -> String {
    let vec4 = spec.get_bool("VEC4");
    let ty = if vec4 { "vec4<f32>" } else { "f32" };
    let nonfinite = nonfinite_expr(vec4);
    if spec.is_subgroups() {
        format!(
            "@group(0) @binding(1) var<storage, read_write> X: array<{ty}>;\n\
             @group(0) @binding(2) var<storage, read_write> FLAG: array<atomic<u32>>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {{\n\
             \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
             \x20 var bad = false;\n\
             \x20 if (i < pu(0u)) {{\n\
             \x20   let v = X[i] * pf(1u);\n\
             \x20   X[i] = v;\n\
             \x20   bad = {nonfinite};\n\
             \x20 }}\n\
             \x20 if (subgroupAny(bad) && lane == 0u) {{ atomicOr(&FLAG[0], 1u); }}\n\
             }}\n"
        )
    } else {
        format!(
            "@group(0) @binding(1) var<storage, read_write> X: array<{ty}>;\n\
             @group(0) @binding(2) var<storage, read_write> FLAG: array<atomic<u32>>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {{\n\
             \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
             \x20 if (i >= pu(0u)) {{ return; }}\n\
             \x20 let v = X[i] * pf(1u);\n\
             \x20 X[i] = v;\n\
             \x20 // Not finite: all exponent bits set (inf or NaN).\n\
             \x20 if ({nonfinite}) {{ atomicOr(&FLAG[0], 1u); }}\n\
             }}\n"
        )
    }
}

static UNSCALE: KernelDef = KernelDef {
    name: "unscale_check",
    storage: &[Access::ReadWrite, Access::ReadWrite],
    source: Source::Generated(gen_unscale),
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
    let vec4 = n.is_multiple_of(4);
    let units = if vec4 { n / 4 } else { n };
    let k = ctx.kernel(&UNSCALE, Spec::new().bool("VEC4", vec4).subgroups(use_subgroups(ctx)))?;
    let p = Params::new().uz(units)?.f(factor);
    ctx.dispatch(&k, &[x, flag], &p, ctx.grid_1d(units as u64, 256)?)
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
        } else if kind >= 4 {
            // 4: abs-max partials (X read, PART), 5: scaled-square partials (X read, H, PART), 6: two-factor scale
            // (X read-write, F1, F2).
            let (x, tail) = match kind {
                4 => (Access::Read, vec![Access::ReadWrite]),
                5 => (Access::Read, vec![Access::Read, Access::ReadWrite]),
                _ => (Access::ReadWrite, vec![Access::Read, Access::Read]),
            };
            std::iter::repeat_n(x, j).chain(tail).collect()
        } else {
            (0..j)
                .flat_map(|_| {
                    [Access::ReadWrite, Access::Read].into_iter().chain((0..kind).map(|_| Access::ReadWrite))
                })
                .collect()
        };
        let (name, source) = if kind == 3 {
            ("unscale_check_many", Source::Generated(gen_unscale_many))
        } else if kind == 6 {
            ("scale_many", Source::Generated(gen_scale_many))
        } else if kind >= 4 {
            ("norm_partials_many", Source::Generated(gen_norm_partials_many))
        } else {
            ("optim_many", Source::Generated(gen_optim_many))
        };
        Box::leak(Box::new(KernelDef { name, storage: Box::leak(storage.into_boxed_slice()), source }))
    })
}

/// The multi-tensor kernel recorded as `name` with `spec` (the precompile registry, `kernels::find_def`): the
/// [`many_def`] its call site picked, read back from the constants that call site sets. `None` for other names
/// and for constants no call site sets.
pub(crate) fn many_def_for(name: &str, spec: &Spec) -> Option<&'static KernelDef> {
    let j = spec.get_u32("J", 0) as usize;
    if !(1..=MANY).contains(&j) {
        return None;
    }
    let kind = match name {
        "optim_many" => Some(spec.get_u32("NS", 3)).filter(|&ns| ns <= 2)?,
        "unscale_check_many" => 3,
        "norm_partials_many" => 4 + spec.get_u32("OP", 0).min(1),
        "scale_many" => 6,
        _ => return None,
    };
    Some(many_def(kind, j))
}

/// Params of the multi-tensor kernels: the update's own words from 0 (as `shaders/optim.wgsl`), then the first
/// workgroup of tensor `j` at `16 + j` and its unit count at `32 + j` (`wg` units per workgroup).
fn many_params_wg(mut p: Params, ns: &[usize], wg: u64) -> Result<(Params, u64)> {
    let mut start = 0u64;
    let mut starts = Vec::with_capacity(ns.len());
    for &n in ns {
        starts.push(
            u32::try_from(start).map_err(|_| crate::error::GpuError::Limit("too many elements".into()))?,
        );
        start += (n as u64).div_ceil(wg);
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

fn many_params(p: Params, ns: &[usize]) -> Result<(Params, u64)> {
    many_params_wg(p, ns, 256)
}

/// The multi-tensor kernels' prologue: tensor `j` of this workgroup (the last whose first workgroup is at or
/// before it) and the element (or vec4) `i` of it this invocation takes.
fn many_prologue(s: &mut String, j: usize, with_lane: bool) {
    if with_lane {
        s.push_str(
            "@compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
             \x20 let w = wid.y * nwg.x + wid.x;\n\
             \x20 var j = 0u;\n",
        );
    } else {
        s.push_str(
            "@compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {\n\
             \x20 let w = wid.y * nwg.x + wid.x;\n\
             \x20 var j = 0u;\n",
        );
    }
    for t in 1..j {
        let _ = writeln!(s, "  if (w >= pu({}u)) {{ j = {t}u; }}", 16 + t);
    }
    if with_lane {
        s.push_str("  let i = (w - pu(16u + j)) * 256u + lid;\n  var bad = false;\n  switch j {\n");
    } else {
        s.push_str("  let i = (w - pu(16u + j)) * 256u + lid;\n  switch j {\n");
    }
}

fn gen_optim_many(spec: &Spec) -> String {
    let (j, ns, mode) = (spec.get_u32("J", 1) as usize, spec.get_u32("NS", 0), spec.get_u32("MODE", 0));
    let ty = if spec.get_bool("VEC4") { "vec4<f32>" } else { "f32" };
    let mut s = format!("alias V = {ty};\n");
    let mut b = 1;
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({b}) var<storage, read_write> P{t}: array<V>;");
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> G{t}: array<V>;", b + 1);
        b += 2;
        for k in 0..ns {
            let _ = writeln!(s, "@group(0) @binding({b}) var<storage, read_write> S{k}_{t}: array<V>;");
            b += 1;
        }
    }
    many_prologue(&mut s, j, false);
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
             \x20     let denom = sqrt(v) / pf(7u) + V(pf(8u));\n\
             \x20     p = p + (-pf(6u)) * (m / denom);\n"
        }
        4 => {
            "      var decay = V(0.0);\n\
             \x20     if (FLAG) { decay = pf(3u) * p; }\n\
             \x20     p = p - pf(1u) * g - decay;\n"
        }
        5 => {
            "      let mu = pf(2u);\n\
             \x20     let b = mu * S0_T[i] + g;\n\
             \x20     S0_T[i] = b;\n\
             \x20     var u = b;\n\
             \x20     if (NESTEROV) { u = g + mu * b; }\n\
             \x20     var decay = V(0.0);\n\
             \x20     if (FLAG) { decay = pf(3u) * p; }\n\
             \x20     p = p - pf(1u) * u - decay;\n"
        }
        6 => {
            "      var decay = V(0.0);\n\
             \x20     if (FLAG) { decay = select(V(0.0), pf(3u) * p, g * p > V(0.0)); }\n\
             \x20     p = p - pf(1u) * g - decay;\n"
        }
        7 => {
            "      let mu = pf(2u);\n\
             \x20     let b = mu * S0_T[i] + g;\n\
             \x20     S0_T[i] = b;\n\
             \x20     var u = b;\n\
             \x20     if (NESTEROV) { u = g + mu * b; }\n\
             \x20     var decay = V(0.0);\n\
             \x20     if (FLAG) { decay = select(V(0.0), pf(3u) * p, u * p > V(0.0)); }\n\
             \x20     p = p - pf(1u) * u - decay;\n"
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
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let ty = if vec4 { "vec4<f32>" } else { "f32" };
    let nonfinite = nonfinite_expr(vec4);
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> X{t}: array<{ty}>;", t + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> FLAG: array<atomic<u32>>;", j + 1);
    many_prologue(&mut s, j, sg);
    for t in 0..j {
        if sg {
            let _ = writeln!(
                s,
                "  case {t}u: {{\n    if (i < pu({}u)) {{\n      let v = X{t}[i] * pf(1u);\n      X{t}[i] = v;\n\
                 \x20     bad = {nonfinite};\n    }}\n  }}",
                32 + t
            );
        } else {
            let _ = writeln!(
                s,
                "  case {t}u: {{\n    if (i < pu({}u)) {{\n      let v = X{t}[i] * pf(1u);\n      X{t}[i] = v;\n\
                 \x20     if ({nonfinite}) {{ atomicOr(&FLAG[0], 1u); }}\n    }}\n  }}",
                32 + t
            );
        }
    }
    if sg {
        s.push_str(
            "  default: {}\n  }\n  if (subgroupAny(bad) && lane == 0u) { atomicOr(&FLAG[0], 1u); }\n}\n",
        );
    } else {
        s.push_str("  default: {}\n  }\n}\n");
    }
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
    let (vec4_items, scalar_items): (Vec<&ManyItem<'_>>, Vec<&ManyItem<'_>>) =
        items.iter().filter(|it| it.n > 0).partition(|it| it.n.is_multiple_of(4));
    for (vec4, group) in [(true, &vec4_items), (false, &scalar_items)] {
        for chunk in group.chunks(j_max) {
            let spec = Spec::new()
                .u32("MODE", mode)
                .bool("FLAG", flag)
                .bool("NESTEROV", nesterov)
                .bool("VEC4", vec4)
                .u32("J", chunk.len() as u32)
                .u32("NS", ns as u32);
            let k = ctx.kernel(many_def(ns as u32, chunk.len()), spec)?;
            let units: Vec<usize> = chunk.iter().map(|it| if vec4 { it.n / 4 } else { it.n }).collect();
            let (p, groups) = many_params(base.clone(), &units)?;
            let mut bufs: Vec<&wgpu::Buffer> = Vec::with_capacity(chunk.len() * per);
            for it in chunk {
                bufs.push(it.p);
                bufs.push(it.g);
                bufs.extend(it.s0);
                bufs.extend(it.s1);
            }
            ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
        }
    }
    Ok(())
}

/// What [`norm_partials_many`] reduces, per workgroup of 256 elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormPartial {
    /// The largest `|x|`; NaN propagates, as in the backend's `max`.
    AbsMax,
    /// `Σ t²` with `t = (x · h) · h` (`h` a one-element buffer), each term rounded as the tensor ops
    /// `x.mul(h).mul(h).square()` round it.
    ScaledSquares,
}

/// The multi-tensor reductions' prologue: this workgroup's index `w` (none past the last), its tensor `j` and
/// unit index `i` (`wg` = 64 vec4s or 256 scalars = 256 elements per workgroup). Params: 1 = first partial,
/// 2 = workgroups in this dispatch.
fn norm_prologue(s: &mut String, j: usize, sg: bool, vec4: bool) {
    let wg = if vec4 { 64u32 } else { 256u32 };
    let red_len = if sg { if vec4 { 2 } else { 8 } } else { wg };
    let _ = writeln!(s, "var<workgroup> red: array<f32, {red_len}>;");
    if sg {
        let _ = write!(
            s,
            "@compute @workgroup_size({wg})\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {{\n\
             \x20 let w = wid.y * nwg.x + wid.x;\n\
             \x20 if (w >= pu(2u)) {{ return; }}\n\
             \x20 var j = 0u;\n"
        );
    } else {
        let _ = write!(
            s,
            "@compute @workgroup_size({wg})\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {{\n\
             \x20 let w = wid.y * nwg.x + wid.x;\n\
             \x20 if (w >= pu(2u)) {{ return; }}\n\
             \x20 var j = 0u;\n"
        );
    }
    for t in 1..j {
        let _ = writeln!(s, "  if (w >= pu({}u)) {{ j = {t}u; }}", 16 + t);
    }
    let _ = writeln!(s, "  let i = (w - pu(16u + j)) * {wg}u + lid;\n  var v = 0.0;\n  switch j {{");
}

fn gen_norm_partials_many(spec: &Spec) -> String {
    let (j, squares, sg, vec4) = (
        spec.get_u32("J", 1) as usize,
        spec.get_u32("OP", 0) == 1,
        spec.is_subgroups(),
        spec.get_bool("VEC4"),
    );
    let ty = if vec4 { "vec4<f32>" } else { "f32" };
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> X{t}: array<{ty}>;", t + 1);
    }
    let mut b = j + 1;
    if squares {
        let _ = writeln!(s, "@group(0) @binding({b}) var<storage, read> H: array<f32>;");
        b += 1;
    }
    let _ = writeln!(s, "@group(0) @binding({b}) var<storage, read_write> PART: array<f32>;");
    // Max propagates NaN explicitly (WGSL's builtin may drop it), as the backend's reductions do.
    s.push_str(
        "fn nmax(a: f32, b: f32) -> f32 {\n\
         \x20 if ((bitcast<u32>(a) & 0x7fffffffu) > 0x7f800000u) { return a; }\n\
         \x20 if ((bitcast<u32>(b) & 0x7fffffffu) > 0x7f800000u) { return b; }\n\
         \x20 return max(a, b);\n}\n",
    );
    norm_prologue(&mut s, j, sg, vec4);
    // `^ pu(0u)` (always 0) acts as an opaque bit-exact barrier so GPU compilers cannot reassociate
    // `(x * h) * h` into `x * (h * h)`, which underflows `h * h` to 0 near `f32::MAX`.
    let term = match (vec4, squares) {
        (true, true) => {
            "let h0 = H[0]; let s = bitcast<vec4<f32>>(bitcast<vec4<u32>>(x * h0) ^ vec4<u32>(pu(0u))); let t = s * h0; let q = t * t; v = (q.x + q.y) + (q.z + q.w);"
        }
        (true, false) => "let a = abs(x); v = nmax(nmax(a.x, a.y), nmax(a.z, a.w));",
        (false, true) => {
            "let h0 = H[0]; let s = bitcast<f32>(bitcast<u32>(x * h0) ^ pu(0u)); let t = s * h0; v = t * t;"
        }
        (false, false) => "v = abs(x);",
    };
    for t in 0..j {
        let _ = writeln!(s, "  case {t}u: {{ if (i < pu({}u)) {{ let x = X{t}[i]; {term} }} }}", 32 + t);
    }
    if sg {
        let sg_max = if vec4 { 1u32 } else { 7u32 };
        if squares {
            let merge = if vec4 {
                "red[0] + red[1]"
            } else {
                "((red[0] + red[1]) + (red[2] + red[3])) + ((red[4] + red[5]) + (red[6] + red[7]))"
            };
            let _ = write!(
                s,
                "  default: {{}}\n  }}\n\
                 \x20 let r = subgroupAdd(v);\n\
                 \x20 let sg = min(sgid, {sg_max}u);\n\
                 \x20 if (lane == 0u) {{ red[sg] = r; }}\n\
                 \x20 workgroupBarrier();\n\
                 \x20 if (lid == 0u) {{\n\
                 \x20   PART[pu(1u) + w] = {merge};\n\
                 \x20 }}\n}}\n"
            );
        } else {
            let merge = if vec4 {
                "nmax(red[0], red[1])"
            } else {
                "nmax(nmax(nmax(red[0], red[1]), nmax(red[2], red[3])), nmax(nmax(red[4], red[5]), nmax(red[6], red[7])))"
            };
            let _ = write!(
                s,
                "  default: {{}}\n  }}\n\
                 \x20 let nan_v = (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;\n\
                 \x20 let mask = subgroupBallot(nan_v).x;\n\
                 \x20 let nan_lane = select(0u, firstTrailingBit(mask), mask != 0u);\n\
                 \x20 let r = select(subgroupMax(select(v, 0.0, nan_v)), subgroupShuffle(v, nan_lane), mask != 0u);\n\
                 \x20 let sg = min(sgid, {sg_max}u);\n\
                 \x20 if (lane == 0u) {{ red[sg] = r; }}\n\
                 \x20 workgroupBarrier();\n\
                 \x20 if (lid == 0u) {{\n\
                 \x20   PART[pu(1u) + w] = {merge};\n\
                 \x20 }}\n}}\n"
            );
        }
    } else {
        let s_init = if vec4 { 32u32 } else { 128u32 };
        let op = if squares { "red[lid] + red[lid + s]" } else { "nmax(red[lid], red[lid + s])" };
        let _ = writeln!(
            s,
            "  default: {{}}\n  }}\n  red[lid] = v;\n  workgroupBarrier();\n\
             \x20 for (var s = {s_init}u; s > 0u; s >>= 1u) {{\n\
             \x20   if (lid < s) {{ red[lid] = {op}; }}\n\
             \x20   workgroupBarrier();\n  }}\n\
             \x20 if (lid == 0u) {{ PART[pu(1u) + w] = red[0]; }}\n}}"
        );
    }
    s
}

fn gen_scale_many(spec: &Spec) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let vec4 = spec.get_bool("VEC4");
    let ty = if vec4 { "vec4<f32>" } else { "f32" };
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> X{t}: array<{ty}>;", t + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> F1: array<f32>;", j + 1);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> F2: array<f32>;", j + 2);
    many_prologue(&mut s, j, false);
    // `^ pu(0u)` (always 0) prevents GPU compilers from reassociating `(x * f1) * f2` into `x * (f1 * f2)`,
    // preserving the exact two roundings (and avoiding `f1 * f2` underflow on extreme clip scales).
    let mul = if vec4 {
        "let y = bitcast<vec4<f32>>(bitcast<vec4<u32>>(X_T[i] * F1[0]) ^ vec4<u32>(pu(0u))); X_T[i] = y * F2[0];"
    } else {
        "let y = bitcast<f32>(bitcast<u32>(X_T[i] * F1[0]) ^ pu(0u)); X_T[i] = y * F2[0];"
    };
    for t in 0..j {
        // Two roundings, as `x *= f1` then `x *= f2`.
        let _ = writeln!(
            s,
            "  case {t}u: {{ if (i < pu({}u)) {{ {} }} }}",
            32 + t,
            mul.replace("X_T", &format!("X{t}"))
        );
    }
    s.push_str("  default: {}\n  }\n}\n");
    s
}

/// The partial count [`norm_partials_many`] writes for tensors of `ns` elements: one per 256 elements of each.
pub fn norm_partials_len(ns: &[usize]) -> usize {
    ns.iter().map(|n| n.div_ceil(256)).sum()
}

/// One partial per workgroup of 256 elements of each tensor in `xs` (in order; [`norm_partials_len`] of them)
/// into `part`, [`MANY`] tensors per dispatch: the largest `|x|` or the sum of `((x · h) · h)²` of the
/// workgroup's elements, reduced in a fixed tree, so the result is deterministic. `h` (one f32) is required for
/// [`NormPartial::ScaledSquares`]. The global maximum or sum is a reduction over `part`.
pub fn norm_partials_many(
    ctx: &GpuContext,
    op: NormPartial,
    xs: &[(&wgpu::Buffer, usize)],
    h: Option<&wgpu::Buffer>,
    part: &wgpu::Buffer,
) -> Result<()> {
    let squares = op == NormPartial::ScaledSquares;
    let h = match (squares, h) {
        (true, Some(h)) => {
            check_buf(h, 1, DType::F32, "norm partials scale")?;
            Some(h)
        }
        (true, None) => return Err(crate::error::shape_err("norm_partials_many: squares need a scale")),
        (false, _) => None,
    };
    let ns: Vec<usize> = xs.iter().map(|x| x.1).collect();
    check_buf(part, norm_partials_len(&ns).max(1), DType::F32, "norm partials output")?;
    for &(x, n) in xs {
        check_buf(x, n, DType::F32, "norm partials input")?;
    }
    let extra = 1 + usize::from(squares);
    let j_max = MANY.min((ctx.caps().max_storage_buffers as usize).saturating_sub(extra)).max(1);
    let live: Vec<&(&wgpu::Buffer, usize)> = xs.iter().filter(|x| x.1 > 0).collect();
    let mut first = 0usize;
    let sg = use_subgroups(ctx);
    for chunk in live.chunks(j_max) {
        let vec4 = chunk.iter().all(|x| x.1.is_multiple_of(4));
        let spec = Spec::new()
            .u32("J", chunk.len() as u32)
            .u32("OP", u32::from(squares))
            .bool("VEC4", vec4)
            .subgroups(sg);
        let k = ctx.kernel(many_def(if squares { 5 } else { 4 }, chunk.len()), spec)?;
        let ns: Vec<usize> = chunk.iter().map(|x| x.1).collect();
        let groups = norm_partials_len(&ns);
        let (units, wg) = if vec4 { (ns.iter().map(|&n| n / 4).collect(), 64u32) } else { (ns, 256u32) };
        let (p, _) = many_params_wg(Params::new().u(0).uz(first)?.uz(groups)?, &units, wg as u64)?;
        let mut bufs: Vec<&wgpu::Buffer> = chunk.iter().map(|x| x.0).collect();
        bufs.extend(h);
        bufs.push(part);
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups as u64 * wg as u64, wg)?)?;
        first += groups;
    }
    Ok(())
}

/// `x = (x · f1) · f2` in place for every tensor in `xs` (`f1`, `f2` one f32 each, on the device), [`MANY`] per
/// dispatch: the arithmetic and roundings of `x *= f1` followed by `x *= f2`.
pub fn scale_many(
    ctx: &GpuContext,
    xs: &[(&wgpu::Buffer, usize)],
    f1: &wgpu::Buffer,
    f2: &wgpu::Buffer,
) -> Result<()> {
    check_buf(f1, 1, DType::F32, "scale factor")?;
    check_buf(f2, 1, DType::F32, "scale factor")?;
    for &(x, n) in xs {
        check_buf(x, n, DType::F32, "scale input")?;
    }
    let j_max = MANY.min((ctx.caps().max_storage_buffers as usize).saturating_sub(2)).max(1);
    let (vec4_xs, scalar_xs): (Vec<_>, Vec<_>) =
        xs.iter().copied().filter(|x| x.1 > 0).partition(|x| x.1.is_multiple_of(4));
    for (vec4, group) in [(true, &vec4_xs), (false, &scalar_xs)] {
        for chunk in group.chunks(j_max) {
            let k = ctx.kernel(
                many_def(6, chunk.len()),
                Spec::new().u32("J", chunk.len() as u32).bool("VEC4", vec4),
            )?;
            let units: Vec<usize> = chunk.iter().map(|x| if vec4 { x.1 / 4 } else { x.1 }).collect();
            let (p, groups) = many_params(Params::new().u(0), &units)?;
            let mut bufs: Vec<&wgpu::Buffer> = chunk.iter().map(|x| x.0).collect();
            bufs.push(f1);
            bufs.push(f2);
            ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
        }
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
    let (vec4_xs, scalar_xs): (Vec<_>, Vec<_>) =
        xs.iter().copied().filter(|x| x.1 > 0).partition(|x| x.1.is_multiple_of(4));
    let sg = use_subgroups(ctx);
    for (vec4, group) in [(true, &vec4_xs), (false, &scalar_xs)] {
        for chunk in group.chunks(j_max) {
            let k = ctx.kernel(
                many_def(3, chunk.len()),
                Spec::new().u32("J", chunk.len() as u32).bool("VEC4", vec4).subgroups(sg),
            )?;
            let units: Vec<usize> = chunk.iter().map(|x| if vec4 { x.1 / 4 } else { x.1 }).collect();
            let (p, groups) = many_params(Params::new().u(0).f(factor), &units)?;
            let mut bufs: Vec<&wgpu::Buffer> = chunk.iter().map(|x| x.0).collect();
            bufs.push(flag);
            ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
        }
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
        OptimUpdate::SgdDecoupled { lr, momentum, wd_step, nesterov, cautious } => {
            let base = if cautious { 6 } else { 4 };
            let mode = base + u32::from(momentum_buffer);
            (mode, wd_step != 0.0, nesterov, Params::new().u(0).f(lr).f(momentum).f(wd_step))
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
    /// Decoupled or Cautious-Decoupled SGD (P3).
    SgdDecoupled {
        /// Learning rate for `u`.
        lr: f32,
        /// Momentum.
        momentum: f32,
        /// Pre-multiplied decoupled decay `lr_unscaled · wd`.
        wd_step: f32,
        /// Nesterov momentum.
        nesterov: bool,
        /// Cautious mask (`u · p > 0`).
        cautious: bool,
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
    let vec4 = n.is_multiple_of(4);
    let units = if vec4 { n / 4 } else { n };
    let (mode, flag, nesterov, p_) = match *u {
        OptimUpdate::Sgd { lr, momentum, weight_decay, nesterov } => {
            let mode = if s0.is_some() { 1 } else { 0 };
            (mode, weight_decay != 0.0, nesterov, Params::new().uz(units)?.f(lr).f(momentum).f(weight_decay))
        }
        OptimUpdate::SgdDecoupled { lr, momentum, wd_step, nesterov, cautious } => {
            let base = if cautious { 6 } else { 4 };
            let mode = base + u32::from(s0.is_some());
            (mode, wd_step != 0.0, nesterov, Params::new().uz(units)?.f(lr).f(momentum).f(wd_step))
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
                .uz(units)?
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
        OptimUpdate::Axpby { a, b } => (3, false, false, Params::new().uz(units)?.f(a).f(b)),
    };
    let k = ctx.kernel(
        &OPTIM,
        Spec::new().u32("MODE", mode).bool("FLAG", flag).bool("NESTEROV", nesterov).bool("VEC4", vec4),
    )?;
    let (d0, d1) = (s0.unwrap_or(ctx.dummy()), s1.unwrap_or(ctx.dummy()));
    ctx.dispatch(&k, &[p, g, d0, d1], &p_, ctx.grid_1d(units as u64, 256)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&OPTIM, &UNSCALE];
