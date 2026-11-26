//! The WGSL generators of the direct depthwise kernels: the forward (four adjacent outputs of a row per
//! invocation, the taps unrolled) and the input and weight gradients.

use std::fmt::Write as _;

use crate::kernels::UNARY_FN;
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// Outputs per invocation of the depthwise kernel (adjacent along a row).
pub(super) const DW_VX: u32 = 4;
/// Tiled depthwise output tile height (`16` rows per workgroup).
pub(super) const DW_TILE_H: u32 = 16;
/// Tiled depthwise output tile width (`4` threads × `4` outputs = `16` columns per workgroup).
pub(super) const DW_TILE_W: u32 = 16;
/// Minimum and maximum kernel side eligible for the shared-memory tiled depthwise forward kernel.
pub(super) const DW_TILED_MIN_K: u32 = 2;
pub(super) const DW_TILED_MAX_K: u32 = 7;

fn gen_depthwise_tiled(spec: &Spec) -> String {
    let g = |name: &str| spec.get_u32(name, 1);
    let (kh, kw) = (g("KH"), g("KW"));
    let vec4_out = spec.get_bool("VEC4_OUT");
    let sh_h = DW_TILE_H + kh - 1;
    let sh_w = DW_TILE_W + kw - 1;
    let sh_stride = if sh_w.is_multiple_of(4) { sh_w + 1 } else { sh_w };
    let sh_elems = sh_h * sh_w;
    let sh_total = sh_h * sh_stride;
    let y_ty = if vec4_out { "vec4<T>" } else { "T" };
    let span = DW_VX + kw - 1;
    let mut s = UNARY_FN.to_string();
    let _ = writeln!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
         @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
         @group(0) @binding(3) var<storage, read> BIAS: array<T>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<{y_ty}>;\n\
         var<workgroup> sh: array<f32, {sh_total}>;\n\
         @compute @workgroup_size(4, 16)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_id) tid: vec3<u32>) {{\n\
         \x20 let lid = tid.y * 4u + tid.x;\n\
         \x20 let g = wid.y * nwg.x + wid.x;\n\
         \x20 let total_tiles = pu(0u);\n\
         \x20 let h = i32(pu(1u)); let w = i32(pu(2u)); let oh = pu(3u); let ow = pu(4u); let cout = pu(5u);\n\
         \x20 let tiles_x = pu(7u); let tiles_y = pu(8u);\n\
         \x20 let valid_tile = g < total_tiles;\n\
         \x20 let tx_tile = g % tiles_x; let t = g / tiles_x; let ty_tile = t % tiles_y; let nc = t / tiles_y;\n\
         \x20 let co = nc % cout; let n = nc / cout;\n\
         \x20 let cin = cout / MULT;\n\
         \x20 let base = (n * cin + co / MULT) * u32(h) * u32(w);\n\
         \x20 let tile_x = tx_tile * {DW_TILE_W}u; let tile_y = ty_tile * {DW_TILE_H}u;\n\
         \x20 let iy0 = i32(tile_y) - PH; let ix0 = i32(tile_x) - PW;\n\
         \x20 for (var idx = lid; idx < {sh_elems}u; idx += 64u) {{\n\
         \x20   let sy = idx / {sh_w}u;\n\
         \x20   let sx = idx - sy * {sh_w}u;\n\
         \x20   let iy = iy0 + i32(sy);\n\
         \x20   let ix = ix0 + i32(sx);\n\
         \x20   var v = 0.0;\n\
         \x20   if (valid_tile && iy >= 0 && iy < h && ix >= 0 && ix < w) {{\n\
         \x20     v = f32(X[base + u32(iy) * u32(w) + u32(ix)]);\n\
         \x20   }}\n\
         \x20   sh[sy * {sh_stride}u + sx] = v;\n\
         \x20 }}\n\
         \x20 workgroupBarrier();\n\
         \x20 if (!valid_tile) {{ return; }}\n\
         \x20 let oy = tile_y + tid.y;\n\
         \x20 let ox0 = tile_x + tid.x * 4u;\n\
         \x20 if (oy >= oh || ox0 >= ow) {{ return; }}\n\
         \x20 let wbase = co * {kh}u * {kw}u;"
    );
    for t in 0..kh * kw {
        let _ = writeln!(s, "  let w{t} = f32(Wt[wbase + {t}u]);");
    }
    s.push_str(
        "  var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;\n\
         \x20 let sx0 = tid.x * 4u;\n\
         \x20 let ix_base = i32(ox0) - PW;\n",
    );
    for j in 0..span {
        let _ = writeln!(s, "  let vx{j} = ix_base + {j} >= 0 && ix_base + {j} < w;");
    }
    for ky in 0..kh {
        let _ = writeln!(
            s,
            "  {{ let iy = i32(oy) - PH + {ky};\n\
             \x20   if (iy >= 0 && iy < h) {{\n\
             \x20     let row_off = (tid.y + {ky}u) * {sh_stride}u + sx0;"
        );
        for j in 0..span {
            let _ = writeln!(s, "      let x{j} = sh[row_off + {j}u];");
        }
        for v in 0..DW_VX {
            for kx in 0..kw {
                let j = v + kx;
                let wt = ky * kw + kx;
                let _ = writeln!(s, "      acc{v} = select(acc{v}, fma(x{j}, w{wt}, acc{v}), vx{j});");
            }
        }
        s.push_str("    }\n  }\n");
    }
    s.push_str("  var b = 0.0;\n  if (HAS_BIAS) { b = f32(BIAS[co]); }\n  let alpha = pf(6u);\n");
    if vec4_out {
        s.push_str(
            "  let v = vec4<f32>(\n\
             \x20   unary1(ACT, acc0 + b, alpha),\n\
             \x20   unary1(ACT, acc1 + b, alpha),\n\
             \x20   unary1(ACT, acc2 + b, alpha),\n\
             \x20   unary1(ACT, acc3 + b, alpha),\n\
             \x20 );\n\
             \x20 Y[(nc * oh + oy) * (ow / 4u) + (ox0 / 4u)] = vec4<T>(v);\n}\n",
        );
    } else {
        s.push_str(
            "  let out = (nc * oh + oy) * ow + ox0;\n\
             \x20 Y[out] = T(unary1(ACT, acc0 + b, alpha));\n\
             \x20 if (ox0 + 1u < ow) { Y[out + 1u] = T(unary1(ACT, acc1 + b, alpha)); }\n\
             \x20 if (ox0 + 2u < ow) { Y[out + 2u] = T(unary1(ACT, acc2 + b, alpha)); }\n\
             \x20 if (ox0 + 3u < ow) { Y[out + 3u] = T(unary1(ACT, acc3 + b, alpha)); }\n}\n",
        );
    }
    s
}

fn gen_depthwise(spec: &Spec) -> String {
    if spec.get_bool("TILED") {
        return gen_depthwise_tiled(spec);
    }
    let g = |name: &str| spec.get_u32(name, 1);
    let (kh, kw, sw, dh, dw) = (g("KH"), g("KW"), g("SW"), g("DH"), g("DW"));
    let vec4_out = spec.get_bool("VEC4_OUT");
    // Input columns the DW_VX outputs of a row read: (DW_VX - 1) · SW + (KW - 1) · DW + 1.
    let span = (DW_VX - 1) * sw + (kw - 1) * dw + 1;
    let mut taps = String::from(
        "fn taps(acc: ptr<function, array<f32, DW_VX>>, base: u32, wbase: u32, iy0: i32, ix0: i32, h: i32,          w: i32) {
",
    );
    for t in 0..kh * kw {
        let _ = writeln!(taps, "    let w{t} = f32(Wt[wbase + {t}u]);");
    }
    for ky in 0..kh {
        let _ = writeln!(
            taps,
            "    {{
        let iy = iy0 + {};",
            ky * dh
        );
        taps.push_str(
            "        if (iy >= 0 && iy < h) {
            let row = base + u32(iy) * u32(w);
",
        );
        for j in 0..span {
            let _ = writeln!(taps, "            let x{j} = ld(row, ix0 + {j}, w);");
        }
        for v in 0..DW_VX {
            for kx in 0..kw {
                let _ = writeln!(
                    taps,
                    "            (*acc)[{v}] = fma(x{}, w{}, (*acc)[{v}]);",
                    v * sw + kx * dw,
                    ky * kw + kx
                );
            }
        }
        taps.push_str(
            "        }
    }
",
        );
    }
    taps.push_str(
        "}
",
    );
    // Constant indices only, so `acc` stays in registers.
    taps.push_str(
        "fn store(acc: ptr<function, array<f32, DW_VX>>, out: u32, ox0: u32, ow: u32, b: f32) {
",
    );
    if vec4_out {
        taps.push_str(
            "    let alpha = pf(6u);\n\
             \x20   let v = vec4<f32>(\n\
             \x20       unary1(ACT, (*acc)[0] + b, alpha),\n\
             \x20       unary1(ACT, (*acc)[1] + b, alpha),\n\
             \x20       unary1(ACT, (*acc)[2] + b, alpha),\n\
             \x20       unary1(ACT, (*acc)[3] + b, alpha),\n\
             \x20   );\n\
             \x20   Y[out / 4u] = vec4<T>(v);\n",
        );
    } else {
        for v in 0..DW_VX {
            let _ = writeln!(
                taps,
                "    if (ox0 + {v}u < ow) {{ Y[out + {v}u] = T(unary1(ACT, (*acc)[{v}] + b, pf(6u))); }}"
            );
        }
    }
    taps.push_str(
        "}
",
    );
    let y_ty = if vec4_out { "vec4<T>" } else { "T" };
    let body = include_str!("../../shaders/depthwise.wgsl").replace(
        "var<storage, read_write> Y: array<T>;",
        &format!("var<storage, read_write> Y: array<{y_ty}>;"),
    );
    format!(
        "const DW_VX: u32 = {DW_VX}u;
{UNARY_FN}{taps}{body}"
    )
}

pub(super) static DEPTHWISE: KernelDef = KernelDef {
    name: "conv2d_depthwise",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_depthwise),
};

/// Shared-memory 16×16 tiled stride-1 depthwise input gradient (`MODE = 0, TILED = true`):
/// loads each output channel's `(16 + KH - 1) × (16 + KW - 1)` gradient halo once into workgroup memory and
/// writes 4 adjacent input gradients per thread (`vec4<T>` when `VEC4_OUT` is true).
/// Params: 0 = total_tiles, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_in, 6 = tiles_x, 7 = tiles_y.
fn gen_depthwise_grad_tiled(spec: &Spec) -> String {
    let g = |name: &str| spec.get_u32(name, 1);
    let (kh, kw) = (g("KH"), g("KW"));
    let vec4_out = spec.get_bool("VEC4_OUT");
    let sh_h = DW_TILE_H + kh - 1;
    let sh_w = DW_TILE_W + kw - 1;
    let sh_stride = if sh_w.is_multiple_of(4) { sh_w + 1 } else { sh_w };
    let sh_elems = sh_h * sh_w;
    let sh_total = sh_h * sh_stride;
    let gx_ty = if vec4_out { "vec4<T>" } else { "T" };
    let span = DW_VX + kw - 1;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
         @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
         @group(0) @binding(3) var<storage, read_write> GX: array<{gx_ty}>;\n\
         var<workgroup> sh: array<f32, {sh_total}>;\n\
         @compute @workgroup_size(4, 16)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_id) tid: vec3<u32>) {{\n\
         \x20 let lid = tid.y * 4u + tid.x;\n\
         \x20 let g = wid.y * nwg.x + wid.x;\n\
         \x20 let total_tiles = pu(0u);\n\
         \x20 let h = pu(1u); let w = pu(2u); let oh = i32(pu(3u)); let ow = i32(pu(4u)); let cin = pu(5u);\n\
         \x20 let tiles_x = pu(6u); let tiles_y = pu(7u);\n\
         \x20 let valid_tile = g < total_tiles;\n\
         \x20 let tx_tile = g % tiles_x; let t = g / tiles_x; let ty_tile = t % tiles_y; let nc = t / tiles_y;\n\
         \x20 let ci = nc % cin; let n = nc / cin;\n\
         \x20 let tile_x = tx_tile * {DW_TILE_W}u; let tile_y = ty_tile * {DW_TILE_H}u;\n\
         \x20 let oy0 = i32(tile_y) + PH - i32(KH - 1u); let ox0 = i32(tile_x) + PW - i32(KW - 1u);\n\
         \x20 let iy = tile_y + tid.y;\n\
         \x20 let ix0 = tile_x + tid.x * 4u;\n\
         \x20 let ox_base = i32(ix0) + PW - i32(KW - 1u);\n\
         \x20 var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;\n\
         \x20 for (var m = 0u; m < MULT; m++) {{\n\
         \x20   let co = ci * MULT + m;\n\
         \x20   let gbase = (n * cin * MULT + co) * u32(oh) * u32(ow);\n\
         \x20   for (var idx = lid; idx < {sh_elems}u; idx += 64u) {{\n\
         \x20     let sy = idx / {sh_w}u;\n\
         \x20     let sx = idx - sy * {sh_w}u;\n\
         \x20     let oy = oy0 + i32(sy);\n\
         \x20     let ox = ox0 + i32(sx);\n\
         \x20     var v = 0.0;\n\
         \x20     if (valid_tile && oy >= 0 && oy < oh && ox >= 0 && ox < ow) {{\n\
         \x20       v = f32(G[gbase + u32(oy) * u32(ow) + u32(ox)]);\n\
         \x20     }}\n\
         \x20     sh[sy * {sh_stride}u + sx] = v;\n\
         \x20   }}\n\
         \x20   workgroupBarrier();\n\
         \x20   if (valid_tile && iy < h && ix0 < w) {{\n\
         \x20     let wbase = co * {kh}u * {kw}u;"
    );
    for t in 0..kh * kw {
        let _ = writeln!(s, "      let w{t} = f32(Wt[wbase + {t}u]);");
    }
    s.push_str("      let sx0 = tid.x * 4u;\n");
    for j in 0..span {
        let _ = writeln!(s, "      let vg{j} = ox_base + {j} >= 0 && ox_base + {j} < ow;");
    }
    for ky in 0..kh {
        let sy_off = kh - 1 - ky;
        let _ = writeln!(
            s,
            "      {{ let oy = i32(iy) + PH - {ky};\n\
             \x20       if (oy >= 0 && oy < oh) {{\n\
             \x20         let row_off = (tid.y + {sy_off}u) * {sh_stride}u + sx0;"
        );
        for j in 0..span {
            let _ = writeln!(s, "          let g{j} = sh[row_off + {j}u];");
        }
        for v in 0..DW_VX {
            for kx in 0..kw {
                let gi = v + kw - 1 - kx;
                let wt = ky * kw + kx;
                let _ = writeln!(s, "          acc{v} = select(acc{v}, fma(g{gi}, w{wt}, acc{v}), vg{gi});");
            }
        }
        s.push_str("        }\n      }\n");
    }
    s.push_str(
        "    }\n\
         \x20   if (m + 1u < MULT) { workgroupBarrier(); }\n\
         \x20 }\n\
         \x20 if (!valid_tile || iy >= h || ix0 >= w) { return; }\n",
    );
    if vec4_out {
        s.push_str(
            "  GX[(nc * h + iy) * (w / 4u) + (ix0 / 4u)] = vec4<T>(T(acc0), T(acc1), T(acc2), T(acc3));\n}\n",
        );
    } else {
        s.push_str(
            "  let out = (nc * h + iy) * w + ix0;\n\
             \x20 GX[out] = T(acc0);\n\
             \x20 if (ix0 + 1u < w) { GX[out + 1u] = T(acc1); }\n\
             \x20 if (ix0 + 2u < w) { GX[out + 2u] = T(acc2); }\n\
             \x20 if (ix0 + 3u < w) { GX[out + 3u] = T(acc3); }\n}\n",
        );
    }
    s
}

/// Phase-unrolled stride-2 dilation-1 depthwise input gradient (`MODE = 0, S2 = true`):
/// each thread computes a 2×2 input block `(2·by + py, 2·bx + px)` by loading the shared `G` phase window once
/// and emitting only the compile-time in-phase taps (zero modulo/divide branches). Writes `vec2<T>` per row when
/// `W % 2 == 0` (`VEC2_OUT`).
/// Params: 0 = invocations (`N·C_in·ceil(H/2)·ceil(W/2)`), 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_in,
/// 6 = `ceil(H/2)`, 7 = `ceil(W/2)`.
fn gen_depthwise_grad_s2(spec: &Spec) -> String {
    let (kh, kw) = (spec.get_u32("KH", 1), spec.get_u32("KW", 1));
    let (ph, pw) = (spec.get_u32("PH", 0) as i32, spec.get_u32("PW", 0) as i32);
    let vec2_out = spec.get_bool("VEC2_OUT");
    let gx_ty = if vec2_out { "vec2<T>" } else { "T" };

    // Collect the exact (dy, dx) G offsets relative to (by, bx) touched by each of the 4 input phases.
    let mut dys = Vec::new();
    let mut dxs = Vec::new();
    let mut taps_for = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for py in 0..2i32 {
        for px in 0..2i32 {
            let pid = (py * 2 + px) as usize;
            for ky in 0..kh {
                let ny = py + ph - ky as i32;
                if (ny & 1) != 0 {
                    continue;
                }
                let dy = ny >> 1;
                if !dys.contains(&dy) {
                    dys.push(dy);
                }
                for kx in 0..kw {
                    let nx = px + pw - kx as i32;
                    if (nx & 1) != 0 {
                        continue;
                    }
                    let dx = nx >> 1;
                    if !dxs.contains(&dx) {
                        dxs.push(dx);
                    }
                    taps_for[pid].push((dy, dx, ky * kw + kx));
                }
            }
        }
    }
    dys.sort_unstable();
    dxs.sort_unstable();

    let mut s = String::new();
    let _ = writeln!(
        s,
        "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
         @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
         @group(0) @binding(3) var<storage, read_write> GX: array<{gx_ty}>;\n\
         fn ld_g(row: u32, ox: i32, ow: i32) -> f32 {{\n\
         \x20 let v = f32(G[row + u32(clamp(ox, 0, ow - 1))]);\n\
         \x20 return select(0.0, v, ox >= 0 && ox < ow);\n\
         }}\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {{\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) {{ return; }}\n\
         \x20 let h = pu(1u); let w = pu(2u); let oh = i32(pu(3u)); let ow = i32(pu(4u)); let cin = pu(5u);\n\
         \x20 let bh = pu(6u); let bw = pu(7u);\n\
         \x20 let bx = i % bw; let t = i / bw; let by = t % bh; let nc = t / bh; let ci = nc % cin; let n = nc / cin;\n\
         \x20 var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;\n\
         \x20 for (var m = 0u; m < MULT; m++) {{\n\
         \x20   let co = ci * MULT + m;\n\
         \x20   let gbase = (n * cin * MULT + co) * u32(oh) * u32(ow);\n\
         \x20   let wbase = co * {kh}u * {kw}u;"
    );
    for t in 0..kh * kw {
        let _ = writeln!(s, "    let w{t} = f32(Wt[wbase + {t}u]);");
    }
    for (xi, &dx) in dxs.iter().enumerate() {
        let _ = writeln!(s, "    let ox{xi} = i32(bx) + ({dx}); let vx{xi} = ox{xi} >= 0 && ox{xi} < ow;");
    }
    for (yi, &dy) in dys.iter().enumerate() {
        let _ = writeln!(
            s,
            "    {{ let oy = i32(by) + ({dy}); if (oy >= 0 && oy < oh) {{ let row = gbase + u32(oy) * u32(ow);"
        );
        for (xi, &dx) in dxs.iter().enumerate() {
            // Only load (dy, dx) if at least one phase tap uses it.
            if taps_for.iter().any(|ts| ts.iter().any(|&(ty, tx, _)| ty == dy && tx == dx)) {
                let _ = writeln!(s, "      let g{yi}_{xi} = ld_g(row, ox{xi}, ow);");
            }
        }
        for (pid, ts) in taps_for.iter().enumerate() {
            for &(ty, tx, wt) in ts {
                if ty == dy {
                    let xi = dxs.iter().position(|&d| d == tx).expect("present");
                    let _ =
                        writeln!(s, "      a{pid} = select(a{pid}, fma(g{yi}_{xi}, w{wt}, a{pid}), vx{xi});");
                }
            }
        }
        s.push_str("    } }\n");
    }
    s.push_str("  }\n  let iy0 = by * 2u; let ix0 = bx * 2u;\n");
    if vec2_out {
        s.push_str(
            "  let w2 = w / 2u;\n\
             \x20 GX[(nc * h + iy0) * w2 + bx] = vec2<T>(T(a0), T(a1));\n\
             \x20 if (iy0 + 1u < h) { GX[(nc * h + iy0 + 1u) * w2 + bx] = vec2<T>(T(a2), T(a3)); }\n}\n",
        );
    } else {
        s.push_str(
            "  let r0 = (nc * h + iy0) * w + ix0;\n\
             \x20 GX[r0] = T(a0);\n\
             \x20 if (ix0 + 1u < w) { GX[r0 + 1u] = T(a1); }\n\
             \x20 if (iy0 + 1u < h) {\n\
             \x20   let r1 = r0 + w;\n\
             \x20   GX[r1] = T(a2);\n\
             \x20   if (ix0 + 1u < w) { GX[r1 + 1u] = T(a3); }\n\
             \x20 }\n}\n",
        );
    }
    s
}

/// Direct depthwise kernels: input gradient (`MODE` 0: 16×16 tiled stride-1 `TILED`, phase-unrolled stride-2
/// `S2`, 4-px unrolled `VX4`, or general loop fallback) and weight-gradient partials (`MODE` 1: one workgroup
/// per output channel and pixel chunk, `VEC4` 4-column input reuse when `SW == 1, DW == 1, OW % 4 == 0`,
/// 1-barrier 32-wide subgroup reduction or batched `vec4` portable tree fallback).
fn gen_depthwise_grad(spec: &Spec) -> String {
    let (kh, kw) = (spec.get_u32("KH", 1), spec.get_u32("KW", 1));
    let mut s = String::new();
    if spec.get_u32("MODE", 0) == 0 {
        if spec.get_bool("TILED") {
            return gen_depthwise_grad_tiled(spec);
        }
        if spec.get_bool("S2") {
            return gen_depthwise_grad_s2(spec);
        }
        if spec.get_bool("VX4") {
            let vec4_out = spec.get_bool("VEC4_OUT");
            let gx_ty = if vec4_out { "vec4<T>" } else { "T" };
            // Params: 0 = invocations (N·C_in·H·QW), 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_in, 6 = QW.
            let span = DW_VX + kw.saturating_sub(1);
            let _ = write!(
                s,
                "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
                 @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
                 @group(0) @binding(3) var<storage, read_write> GX: array<{gx_ty}>;\n\
                 fn ld_g(row: u32, ox: i32, ow: i32) -> f32 {{\n\
                 \x20 let v = f32(G[row + u32(clamp(ox, 0, ow - 1))]);\n\
                 \x20 return select(0.0, v, ox >= 0 && ox < ow);\n\
                 }}\n\
                 @compute @workgroup_size(256)\n\
                 fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 \x20       @builtin(local_invocation_index) lid: u32) {{\n\
                 \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
                 \x20 if (i >= pu(0u)) {{ return; }}\n\
                 \x20 let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u); let cin = pu(5u); let qw = pu(6u);\n\
                 \x20 let q = i % qw; let t = i / qw; let iy = t % h; let nc = t / h; let ci = nc % cin; let n = nc / cin;\n\
                 \x20 let ix0 = q * 4u;\n\
                 \x20 let ox_base = i32(ix0) + PW - i32(KW - 1u);\n\
                 \x20 var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;\n\
                 \x20 for (var m = 0u; m < MULT; m++) {{\n\
                 \x20   let co = ci * MULT + m;\n\
                 \x20   let gbase = (n * cin * MULT + co) * oh * ow;\n\
                 \x20   let wbase = co * KH * KW;\n"
            );
            for t in 0..kh * kw {
                let _ = writeln!(s, "    let w{t} = f32(Wt[wbase + {t}u]);");
            }
            for ky in 0..kh {
                let _ = writeln!(
                    s,
                    "    {{ let oy = i32(iy) + PH - {ky}; if (oy >= 0 && oy < i32(oh)) {{ let row = gbase + u32(oy) * ow;"
                );
                for j in 0..span {
                    let _ = writeln!(
                        s,
                        "      let ox{j} = ox_base + {j}; let vg{j} = ox{j} >= 0 && ox{j} < i32(ow); let g{j} = ld_g(row, ox{j}, i32(ow));"
                    );
                }
                for v in 0..DW_VX {
                    for kx in 0..kw {
                        let gi = v + kw - 1 - kx;
                        let wt = ky * kw + kx;
                        let _ =
                            writeln!(s, "      acc{v} = select(acc{v}, fma(g{gi}, w{wt}, acc{v}), vg{gi});");
                    }
                }
                s.push_str("    } }\n");
            }
            if vec4_out {
                s.push_str("  }\n  GX[i] = vec4<T>(T(acc0), T(acc1), T(acc2), T(acc3));\n}\n");
            } else {
                s.push_str(
                    "  }\n\
                     \x20 let out_base = (nc * h + iy) * w + ix0;\n\
                     \x20 GX[out_base] = T(acc0);\n\
                     \x20 if (ix0 + 1u < w) { GX[out_base + 1u] = T(acc1); }\n\
                     \x20 if (ix0 + 2u < w) { GX[out_base + 2u] = T(acc2); }\n\
                     \x20 if (ix0 + 3u < w) { GX[out_base + 3u] = T(acc3); }\n}\n",
                );
            }
            return s;
        }
        // Params: 0 = input elements, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_in.
        s.push_str(
            "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
             @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
             @group(0) @binding(3) var<storage, read_write> GX: array<T>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {\n\
             \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
             \x20 if (i >= pu(0u)) { return; }\n\
             \x20 let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u); let cin = pu(5u);\n\
             \x20 let ix = i % w; let t = i / w; let iy = t % h; let nc = t / h; let ci = nc % cin; let n = nc / cin;\n\
             \x20 var acc = 0.0;\n\
             \x20 for (var m = 0u; m < MULT; m++) {\n\
             \x20   let co = ci * MULT + m;\n\
             \x20   let gbase = (n * cin * MULT + co) * oh * ow;\n\
             \x20   let wbase = co * KH * KW;\n\
             \x20   for (var ky = 0u; ky < KH; ky++) {\n\
             \x20     let th = i32(iy) + PH - i32(ky * DH);\n\
             \x20     if (th < 0) { continue; }\n\
             \x20     let uh = u32(th);\n\
             \x20     if (uh % SH != 0u || uh / SH >= oh) { continue; }\n\
             \x20     for (var kx = 0u; kx < KW; kx++) {\n\
             \x20       let tw = i32(ix) + PW - i32(kx * DW);\n\
             \x20       if (tw < 0) { continue; }\n\
             \x20       let uw = u32(tw);\n\
             \x20       if (uw % SW != 0u || uw / SW >= ow) { continue; }\n\
             \x20       acc = fma(f32(G[gbase + (uh / SH) * ow + uw / SW]), f32(Wt[wbase + ky * KW + kx]), acc);\n\
             \x20     }\n\
             \x20   }\n\
             \x20 }\n\
             \x20 GX[i] = T(acc);\n}\n",
        );
        return s;
    }
    // Params: 0 = C_out, 1 = chunks S, 2 = units per chunk, 3 = total units (N·OH·OW_u), 4 = OH·OW_u,
    // 5 = OW_u (OW / 4 when VEC4 else OW), 6 = H, 7 = W. Output: partial[s][co][tap].
    let vec4 = spec.get_bool("VEC4");
    let sg = spec.is_subgroups();
    let taps = kh * kw;
    let g_ty = if vec4 { "vec4<T>" } else { "T" };
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> G: array<{g_ty}>;");
    s.push_str(
        "@group(0) @binding(2) var<storage, read> X: array<T>;\n\
         @group(0) @binding(3) var<storage, read_write> OUT: array<f32>;\n",
    );
    if sg {
        let _ = writeln!(s, "var<workgroup> red: array<f32, {}>;", taps.max(1) * 8);
    } else {
        s.push_str("var<workgroup> red: array<vec4<f32>, 256>;\n");
    }
    if vec4 {
        s.push_str(
            "fn ld_x(row: u32, ix: i32, w: i32) -> f32 {\n\
             \x20 let v = f32(X[row + u32(clamp(ix, 0, w - 1))]);\n\
             \x20 return select(0.0, v, ix >= 0 && ix < w);\n\
             }\n",
        );
    }
    s.push_str("@compute @workgroup_size(256)\n");
    if sg {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_invocation_id) s_lane: u32, @builtin(subgroup_id) sg_id: u32) {\n",
        );
    } else {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {\n",
        );
    }
    s.push_str(
        "  let g = wid.y * nwg.x + wid.x;\n\
         \x20 let cout = pu(0u); let chunks = pu(1u);\n\
         \x20 let co = g / chunks; let sidx = g % chunks;\n\
         \x20 let valid = co < cout;\n\
         \x20 let ps = pu(2u); let np = pu(3u); let ohow = pu(4u); let ow = pu(5u); let h = pu(6u); let w = pu(7u);\n",
    );
    for t in 0..taps {
        let _ = writeln!(s, "  var acc{t} = 0.0;");
    }
    // Uniform loop bounds so Tint/WebGPU uniformity holds at the subgroup/workgroup reduction below.
    s.push_str(
        "  let p0 = sidx * ps;\n\
         \x20 if (valid && p0 < np) {\n\
         \x20   let rem = min(ps, np - p0);\n\
         \x20   let n_iter = rem / 256u + select(0u, 1u, (rem % 256u) != 0u);\n\
         \x20   let cin = cout / MULT; let ci = co / MULT;\n\
         \x20   for (var it = 0u; it < n_iter; it += 1u) {\n\
         \x20     let off = it * 256u + lid;\n\
         \x20     if (off < rem) {\n\
         \x20       let p = p0 + off;\n\
         \x20       let n = p / ohow; let r = p % ohow;\n\
         \x20       let base = (n * cin + ci) * h * w;\n",
    );
    if vec4 {
        let span = 3 + kw;
        s.push_str(
            "        let gv = vec4<f32>(G[(n * cout + co) * ohow + r]);\n\
             \x20       let iy0 = i32((r / ow) * SH) - PH; let ix0 = i32((r % ow) * 4u) - PW;\n",
        );
        for ky in 0..kh {
            let _ = writeln!(
                s,
                "        {{ let iy = iy0 + {}; if (iy >= 0 && iy < i32(h)) {{ let row = base + u32(iy) * w;",
                ky as i64 * spec.get_u32("DH", 1) as i64
            );
            for j in 0..span {
                let _ = writeln!(
                    s,
                    "          let ix{j} = ix0 + {j}; let vx{j} = ix{j} >= 0 && ix{j} < i32(w); let x{j} = ld_x(row, ix{j}, i32(w));"
                );
            }
            for kx in 0..kw {
                let t = ky * kw + kx;
                let _ = writeln!(
                    s,
                    "          acc{t} = select(acc{t}, fma(gv.x, x{kx}, acc{t}), vx{kx});\n\
                     \x20         acc{t} = select(acc{t}, fma(gv.y, x{k1}, acc{t}), vx{k1});\n\
                     \x20         acc{t} = select(acc{t}, fma(gv.z, x{k2}, acc{t}), vx{k2});\n\
                     \x20         acc{t} = select(acc{t}, fma(gv.w, x{k3}, acc{t}), vx{k3});",
                    k1 = kx + 1,
                    k2 = kx + 2,
                    k3 = kx + 3
                );
            }
            s.push_str("        } }\n");
        }
    } else {
        s.push_str(
            "        let gv = f32(G[(n * cout + co) * ohow + r]);\n\
             \x20       let iy0 = i32((r / ow) * SH) - PH; let ix0 = i32((r % ow) * SW) - PW;\n",
        );
        for ky in 0..kh {
            let _ = writeln!(
                s,
                "        {{ let iy = iy0 + {}; if (iy >= 0 && iy < i32(h)) {{ let row = base + u32(iy) * w;",
                ky as i64 * spec.get_u32("DH", 1) as i64
            );
            for kx in 0..kw {
                let t = ky * kw + kx;
                let _ = writeln!(
                    s,
                    "          {{ let ix = ix0 + {}; if (ix >= 0 && ix < i32(w)) {{ acc{t} = fma(gv, f32(X[row + u32(ix)]), acc{t}); }} }}",
                    kx as i64 * spec.get_u32("DW", 1) as i64
                );
            }
            s.push_str("        } }\n");
        }
    }
    s.push_str("      }\n    }\n  }\n");
    if sg {
        s.push_str("  let sg = min(sg_id, 7u);\n");
        for t in 0..taps {
            let _ = writeln!(
                s,
                "  let sg{t} = subgroupAdd(acc{t}); if (s_lane == 0u) {{ red[{base}u + sg] = sg{t}; }}",
                base = t * 8
            );
        }
        let _ = writeln!(
            s,
            "  workgroupBarrier();\n\
             \x20 if (lid < {taps}u && valid) {{\n\
             \x20   let off = lid * 8u;\n\
             \x20   OUT[(sidx * cout + co) * {taps}u + lid] = ((red[off] + red[off + 1u]) + (red[off + 2u] + red[off + 3u])) + ((red[off + 4u] + red[off + 5u]) + (red[off + 6u] + red[off + 7u]));\n\
             \x20 }}"
        );
    } else {
        const COMPS: [&str; 4] = ["x", "y", "z", "w"];
        for t0 in (0..taps).step_by(4) {
            let lane = |k: u32| {
                if t0 + k < taps { format!("acc{}", t0 + k) } else { "0.0".into() }
            };
            let _ = writeln!(
                s,
                "  red[lid] = vec4<f32>({}, {}, {}, {});\n  workgroupBarrier();\n\
                 \x20 for (var s = 128u; s > 0u; s >>= 1u) {{ if (lid < s) {{ red[lid] += red[lid + s]; }} workgroupBarrier(); }}\n\
                 \x20 if (lid == 0u && valid) {{ let sum = red[0];",
                lane(0),
                lane(1),
                lane(2),
                lane(3)
            );
            for (k, comp) in COMPS.iter().enumerate() {
                let t = t0 + k as u32;
                if t < taps {
                    let _ = writeln!(s, "    OUT[(sidx * cout + co) * {taps}u + {t}u] = sum.{comp};");
                }
            }
            s.push_str("  }\n");
            if t0 + 4 < taps {
                s.push_str("  workgroupBarrier();\n");
            }
        }
    }
    s.push_str("}\n");
    s
}

/// Largest `KH · KW` of a [`DEPTHWISE`] kernel a kernel list may ask for (`kernels::bounds`): the forward unrolls
/// every tap, so its source grows with the kernel. 32×32 covers the largest depthwise kernels in use (31×31);
/// a larger one still compiles when a run asks for it.
const MAX_LISTED_TAPS: u64 = 1024;
/// Largest number of unrolled tap statements ([`gen_depthwise`]: `KH · (span + DW_VX · KW)`) a list may ask for.
const MAX_LISTED_LINES: u64 = 1 << 16;

/// Whether a kernel list may ask for [`DEPTHWISE`] with `spec` (`kernels::bounds`): non-zero sizes, at most
/// [`MAX_LISTED_TAPS`] taps and [`MAX_LISTED_LINES`] unrolled statements; tiled kernels additionally require
/// stride 1, dilation 1, and `2..=7` kernel sides.
pub(super) fn forward_fits(spec: &Spec) -> bool {
    let g = |name: &str| u64::from(spec.get_u32(name, 1));
    let (kh, kw, sh, sw, dh, dw) = (g("KH"), g("KW"), g("SH"), g("SW"), g("DH"), g("DW"));
    if [kh, kw, sh, sw, dh, dw].contains(&0) {
        return false;
    }
    if spec.get_bool("TILED") {
        let ok_k = |k: u64| (u64::from(DW_TILED_MIN_K)..=u64::from(DW_TILED_MAX_K)).contains(&k);
        return ok_k(kh) && ok_k(kw) && sh == 1 && sw == 1 && dh == 1 && dw == 1;
    }
    let span = (u64::from(DW_VX) - 1) * sw + (kw - 1) * dw + 1;
    kh * kw <= MAX_LISTED_TAPS && kh * (span + u64::from(DW_VX) * kw) <= MAX_LISTED_LINES
}

/// Whether a kernel list may ask for [`DEPTHWISE_GRAD`] with `spec`: `MODE` 0 (`TILED` stride-1 `2..=7`,
/// `S2` stride-2 within `max_taps`, unrolled `VX4` within `max_taps`, or general loop) or `MODE` 1 (every tap
/// unrolled) within `max_taps`.
pub(super) fn grad_fits(spec: &Spec, max_taps: usize) -> bool {
    let kh = u64::from(spec.get_u32("KH", 1));
    let kw = u64::from(spec.get_u32("KW", 1));
    let taps = kh * kw;
    let in_range = (1..=max_taps as u64).contains(&taps);
    match spec.get_u32("MODE", 0) {
        0 => {
            if spec.get_bool("TILED") {
                let ok_k = |k: u64| (u64::from(DW_TILED_MIN_K)..=u64::from(DW_TILED_MAX_K)).contains(&k);
                let (sh, sw, dh, dw) = (
                    spec.get_u32("SH", 1),
                    spec.get_u32("SW", 1),
                    spec.get_u32("DH", 1),
                    spec.get_u32("DW", 1),
                );
                return ok_k(kh) && ok_k(kw) && sh == 1 && sw == 1 && dh == 1 && dw == 1;
            }
            if spec.get_bool("S2") {
                return in_range
                    && spec.get_u32("SH", 2) == 2
                    && spec.get_u32("SW", 2) == 2
                    && spec.get_u32("DH", 1) == 1
                    && spec.get_u32("DW", 1) == 1;
            }
            !spec.get_bool("VX4") || in_range
        }
        1 => in_range,
        _ => false,
    }
}

pub(super) static DEPTHWISE_GRAD: KernelDef = KernelDef {
    name: "conv2d_depthwise_grad",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_depthwise_grad),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depthwise_dw_portable_emits_no_subgroup_tokens() {
        let spec = Spec::new().u32("KH", 3).u32("KW", 3).u32("MODE", 1).bool("VEC4", true).subgroups(false);
        let wgsl = crate::pipeline::render(&DEPTHWISE_GRAD, &spec);
        assert!(!wgsl.contains("subgroup"), "portable depthwise dW emitted subgroup token:\n{wgsl}");
    }
}
