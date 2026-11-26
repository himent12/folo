//! Arg-reductions (`argmax` / `argmin`) over dense f32 `[outer, r, inner]`:
//! - Contiguous rows (`inner == 1`, `r >= 32`): 256-thread workgroup-parallel `vec4` + 1-barrier 32-wide
//!   subgroup reduction (or portable shared-memory tree fallback), with 2-pass split-K when few rows have long
//!   `r`.
//! - General (`inner > 1` or tiny `r`): one coalesced thread per output column (`shaders/arg_reduce.wgsl`).

use super::split_plan;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::{check_buf, prod, use_subgroups};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

pub(super) static ARG: KernelDef = KernelDef {
    name: "arg_reduce",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../../shaders/arg_reduce.wgsl")),
};

fn gen_arg_rows(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 0);
    let vec4 = mode < 2 && spec.get_bool("VEC4");
    let sg = spec.is_subgroups();
    let x_ty = match (mode, vec4) {
        (2, _) => "u32",
        (_, true) => "vec4<f32>",
        _ => "f32",
    };
    let scan = match (mode, vec4) {
        (0, true) => {
            "  if (valid) {\n\
             \x20   let r = pu(1u); let base = row * r;\n\
             \x20   for (var i = lid; i < r; i += 256u) {\n\
             \x20     let v = X[base + i]; let b = i * 4u;\n\
             \x20     if (arg_better(v.x, b, bv, bi)) { bv = v.x; bi = b; }\n\
             \x20     if (arg_better(v.y, b + 1u, bv, bi)) { bv = v.y; bi = b + 1u; }\n\
             \x20     if (arg_better(v.z, b + 2u, bv, bi)) { bv = v.z; bi = b + 2u; }\n\
             \x20     if (arg_better(v.w, b + 3u, bv, bi)) { bv = v.w; bi = b + 3u; }\n\
             \x20   }\n  }\n"
        }
        (0, false) => {
            "  if (valid) {\n\
             \x20   let r = pu(1u); let base = row * r;\n\
             \x20   for (var i = lid; i < r; i += 256u) {\n\
             \x20     let v = X[base + i];\n\
             \x20     if (arg_better(v, i, bv, bi)) { bv = v; bi = i; }\n\
             \x20   }\n  }\n"
        }
        (1, true) => {
            "  if (valid) {\n\
             \x20   let r = pu(1u); let splits = pu(2u); let chunk = pu(3u);\n\
             \x20   let o = row / splits; let k = row % splits;\n\
             \x20   let start = k * chunk; let end = min(start + chunk, r); let base = o * r;\n\
             \x20   for (var i = start + lid; i < end; i += 256u) {\n\
             \x20     let v = X[base + i]; let b = i * 4u;\n\
             \x20     if (arg_better(v.x, b, bv, bi)) { bv = v.x; bi = b; }\n\
             \x20     if (arg_better(v.y, b + 1u, bv, bi)) { bv = v.y; bi = b + 1u; }\n\
             \x20     if (arg_better(v.z, b + 2u, bv, bi)) { bv = v.z; bi = b + 2u; }\n\
             \x20     if (arg_better(v.w, b + 3u, bv, bi)) { bv = v.w; bi = b + 3u; }\n\
             \x20   }\n  }\n"
        }
        (1, false) => {
            "  if (valid) {\n\
             \x20   let r = pu(1u); let splits = pu(2u); let chunk = pu(3u);\n\
             \x20   let o = row / splits; let k = row % splits;\n\
             \x20   let start = k * chunk; let end = min(start + chunk, r); let base = o * r;\n\
             \x20   for (var i = start + lid; i < end; i += 256u) {\n\
             \x20     let v = X[base + i];\n\
             \x20     if (arg_better(v, i, bv, bi)) { bv = v; bi = i; }\n\
             \x20   }\n  }\n"
        }
        _ => {
            "  if (valid) {\n\
             \x20   let splits = pu(1u); let base = row * splits;\n\
             \x20   for (var i = lid; i < splits; i += 256u) {\n\
             \x20     let o = 2u * (base + i); let v = bitcast<f32>(X[o]); let idx = X[o + 1u];\n\
             \x20     if (arg_better(v, idx, bv, bi)) { bv = v; bi = idx; }\n\
             \x20   }\n  }\n"
        }
    };
    let reduce = if sg {
        "var<workgroup> sh_v: array<f32, 8>;\nvar<workgroup> sh_i: array<u32, 8>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32,\n\
         \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
         \x20 let row = wid.y * nwg.x + wid.x; let valid = row < pu(0u);\n\
         \x20 var bv = 0.0; var bi = ARG_EMPTY;\n"
    } else {
        "var<workgroup> sh_v: array<f32, 256>;\nvar<workgroup> sh_i: array<u32, 256>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let row = wid.y * nwg.x + wid.x; let valid = row < pu(0u);\n\
         \x20 var bv = 0.0; var bi = ARG_EMPTY;\n"
    };
    let finish = if sg {
        "  for (var off = 16u; off > 0u; off = off >> 1u) {\n\
         \x20   let ov = subgroupShuffleDown(bv, off); let oi = subgroupShuffleDown(bi, off);\n\
         \x20   if (lane < off && arg_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }\n\
         \x20 }\n\
         \x20 let sg = min(sgid, 7u);\n\
         \x20 if (lane == 0u) { sh_v[sg] = bv; sh_i[sg] = bi; }\n\
         \x20 workgroupBarrier();\n\
         \x20 if (lid == 0u && valid) {\n\
         \x20   var fv = sh_v[0]; var fi = sh_i[0];\n\
         \x20   for (var s = 1u; s < 8u; s += 1u) {\n\
         \x20     if (arg_better(sh_v[s], sh_i[s], fv, fi)) { fv = sh_v[s]; fi = sh_i[s]; }\n\
         \x20   }\n\
         \x20   arg_write(row, fv, fi);\n\
         \x20 }\n}\n"
    } else {
        "  sh_v[lid] = bv; sh_i[lid] = bi; workgroupBarrier();\n\
         \x20 for (var s = 128u; s > 0u; s = s >> 1u) {\n\
         \x20   if (lid < s && arg_better(sh_v[lid + s], sh_i[lid + s], sh_v[lid], sh_i[lid])) {\n\
         \x20     sh_v[lid] = sh_v[lid + s]; sh_i[lid] = sh_i[lid + s];\n\
         \x20   }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 if (lid == 0u && valid) { arg_write(row, sh_v[0], sh_i[0]); }\n}\n"
    };
    format!(
        "@group(0) @binding(1) var<storage, read> X: array<{x_ty}>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<u32>;\n\
         {}{reduce}{scan}{finish}",
        include_str!("../../shaders/arg_reduce_rows.wgsl")
    )
}

pub(super) static ARG_ROWS: KernelDef = KernelDef {
    name: "arg_reduce_rows",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_arg_rows),
};

/// Index of the first maximum (`max`) or minimum over the middle axis of dense f32 `[outer, r, inner]`, as
/// `i64` (two words per element) into `y` (`[outer, inner]`). NaN counts as the extreme (the first NaN wins).
pub fn arg_reduce(
    ctx: &GpuContext,
    max: bool,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, DType::F32, "arg_reduce input")?;
    check_buf(y, prod(&[outer, inner, 2])?, DType::F32, "arg_reduce output")?;
    let n = outer * inner;
    if n == 0 {
        return Ok(());
    }
    if r == 0 {
        return Err(shape_err("arg_reduce over an empty dimension"));
    }
    let use_rows = inner == 1
        && r < u32::MAX as usize
        && (r >= 256 || (!r.is_multiple_of(4) && r >= 128) || (outer < 256 && r >= 64));
    if use_rows {
        let sg = use_subgroups(ctx);
        let vec4 = r.is_multiple_of(4);
        let (splits, chunk) = split_plan(outer, r, 1);
        if splits > 1 {
            let outer_s = prod(&[outer, splits])?;
            let part = ctx.alloc(DType::F32.bytes(prod(&[outer_s, 2])?))?;
            let (r_u, chunk_u) = if vec4 { (r / 4, chunk / 4) } else { (r, chunk) };
            let k1 = ctx.kernel(
                &ARG_ROWS,
                Spec::new().bool("MAX", max).u32("MODE", 1).bool("VEC4", vec4).subgroups(sg),
            )?;
            let p1 = Params::new().uz(outer_s)?.uz(r_u)?.uz(splits)?.uz(chunk_u)?;
            ctx.dispatch(&k1, &[x, part.raw()], &p1, ctx.grid_1d(outer_s as u64 * 256, 256)?)?;
            let k2 = ctx.kernel(
                &ARG_ROWS,
                Spec::new().bool("MAX", max).u32("MODE", 2).bool("VEC4", false).subgroups(sg),
            )?;
            let p2 = Params::new().uz(outer)?.uz(splits)?;
            return ctx.dispatch(&k2, &[part.raw(), y], &p2, ctx.grid_1d(outer as u64 * 256, 256)?);
        }
        let r_u = if vec4 { r / 4 } else { r };
        let k = ctx.kernel(
            &ARG_ROWS,
            Spec::new().bool("MAX", max).u32("MODE", 0).bool("VEC4", vec4).subgroups(sg),
        )?;
        let p = Params::new().uz(outer)?.uz(r_u)?;
        return ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(outer as u64 * 256, 256)?);
    }
    let k = ctx.kernel(&ARG, Spec::new().bool("MAX", max))?;
    let p = Params::new().uz(n)?.uz(r)?.uz(inner)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_rows_portable_emits_no_subgroup_tokens() {
        for mode in [0u32, 1, 2] {
            let spec =
                Spec::new().bool("MAX", true).u32("MODE", mode).bool("VEC4", mode < 2).subgroups(false);
            let wgsl = crate::pipeline::render(&ARG_ROWS, &spec);
            assert!(
                !wgsl.contains("subgroup"),
                "portable arg_reduce_rows mode={mode} emitted subgroup token:\n{wgsl}"
            );
        }
    }
}
