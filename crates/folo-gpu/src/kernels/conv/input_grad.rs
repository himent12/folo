//! The input gradient ([`conv2d_backward_input`]): a transposed conv as implicit GEMM (`CONV_MODE` 2), a fused
//! 2×2 sub-pixel conv (`CONV_MODE` 1 with `SUBPIXEL`) for 3×3 stride-2 convs, per-stride-phase decomposition
//! ([`Phase`]) for other strided convs, a GEMM over the transposed weight for 1×1 convs, and the direct depthwise
//! kernel.

use std::fmt::Write as _;

use super::depthwise::DEPTHWISE_GRAD;
use super::{Conv2dDesc, check_grad_desc, conv_spec, coop_grad};
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::kernels::gemm::{self, BatchStrides, GEMM, GemmDesc, base_spec, gemm_params};
use crate::kernels::{check_buf, coop, prod};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// WGSL of the stride-1 `CONV_MODE` 2 tap mask setup (shared by `gemm` and `coop/shader`): builds `vm` (bit
/// `kh·KW + kw` set when the tap lands inside `(OH, OW)`) and `base_var` (the top-left output-gradient element of
/// the window).
pub(crate) fn dx_tap_mask_setup(kh_n: u32, kw_n: u32, dh_n: u32, dw_n: u32, base_var: &str) -> String {
    let mut s = String::from("  var vm = 0u;\n  if (cjok) {\n");
    for kh in 0..kh_n {
        for kw in 0..kw_n {
            let _ = writeln!(
                s,
                "    if (u32(cih - {}i) < OH && u32(ciw - {}i) < OW) {{ vm |= {}u; }}",
                kh * dh_n,
                kw * dw_n,
                1u32 << (kh * kw_n + kw)
            );
        }
    }
    let _ = writeln!(
        s,
        "  }}\n  let {base_var} = offB + bitcast<u32>(cih * i32(OW) + ciw);\n\
         \x20 let OHW = OH * OW; let DHOW = DH * OW;"
    );
    s
}

/// WGSL of the stride-phase `CONV_MODE` 2 tap-mask setup (`PTH·PTW <= 32`, shared by `gemm` and `coop/shader`):
/// builds `vm` (bit `th·PTW + tw` set when `(ca − th, cb − tw)` lies inside `(OH, OW)`) and `base_var`.
pub(crate) fn phase_tap_mask_setup(pth: u32, ptw: u32, base_var: &str) -> String {
    let mut s = String::from("  var vm = 0u;\n  if (cjok) {\n");
    for th in 0..pth {
        for tw in 0..ptw {
            let _ = writeln!(
                s,
                "    if (u32(ca - {th}i) < OH && u32(cb - {tw}i) < OW) {{ vm |= {}u; }}",
                1u32 << (th * ptw + tw)
            );
        }
    }
    let _ =
        writeln!(s, "  }}\n  let {base_var} = offB + bitcast<u32>(ca * i32(OW) + cb);\n  let OHW = OH * OW;");
    s
}

/// Packs a 3×3 stride-2 conv weight `[C_out, C_in/groups, 3, 3]` into the 2×2 sub-pixel conv weight
/// `[groups, 4·C_in/groups, 4·C_out/groups]` where output row `ci·4 + qy·2 + qx` and reduction column
/// `co·4 + ty·2 + tx` holds `W[g, co, ci, ky, kx]` for `ky = qy + off_h − 2·ty`, `kx = qx + off_w − 2·tx`
/// (zero when out of `0..3`), with `off_h = PH − 2·⌊(PH − 1) / 2⌋ ∈ {1, 2}`.
/// Params: 0 = threads (`16 · C_in · C_out/groups`), 1 = `C_in/groups`, 2 = `C_out/groups`, 3 = `off_h`,
/// 4 = `off_w`.
fn gen_subpixel_pack(_: &Spec) -> String {
    "@group(0) @binding(1) var<storage, read> W: array<T>;\n\
     @group(0) @binding(2) var<storage, read_write> PK: array<T>;\n\
     @compute @workgroup_size(256)\n\
     fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
     \x20       @builtin(local_invocation_index) lid: u32) {\n\
     \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
     \x20 if (i >= pu(0u)) { return; }\n\
     \x20 let cin_g = pu(1u); let cout_g = pu(2u);\n\
     \x20 let t = i & 3u; var q = i >> 2u;\n\
     \x20 let co = q % cout_g; q = q / cout_g;\n\
     \x20 let sp = q & 3u; q = q >> 2u;\n\
     \x20 let ci = q % cin_g; let g = q / cin_g;\n\
     \x20 let ty = t >> 1u; let tx = t & 1u;\n\
     \x20 let qy = sp >> 1u; let qx = sp & 1u;\n\
     \x20 let ky = i32(qy) + i32(pu(3u)) - i32(ty * 2u);\n\
     \x20 let kx = i32(qx) + i32(pu(4u)) - i32(tx * 2u);\n\
     \x20 var v = T(0.0);\n\
     \x20 if (ky >= 0 && ky < 3 && kx >= 0 && kx < 3) {\n\
     \x20   v = W[(((g * cout_g + co) * cin_g + ci) * 3u + u32(ky)) * 3u + u32(kx)];\n\
     \x20 }\n\
     \x20 PK[i] = v;\n}\n"
        .into()
}

pub(super) static SUBPIXEL_PACK: KernelDef = KernelDef {
    name: "conv2d_subpixel_pack",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_subpixel_pack),
};

/// Fused 2×2 pixel-shuffle epilogue for the regular GEMM generator (`CONV_MODE` 1 with `SUBPIXEL`): rows
/// `ci·4 + (qy·2 + qx)` of cell `j = (u, v)` scatter directly into `dX[ci, 2·u + qy, 2·v + qx]`, as aligned
/// `vec4` stores when `XW % 4 == 0` (`vc`) or bounds-checked scalars otherwise. Params: 27 = `UW`, 28 = `XH`,
/// 29 = `XW`.
pub(crate) fn subpixel_epilogue_gemm(tm: u32, tn: u32, wgx: u32, wgy: u32, vc: bool, c_f16: bool) -> String {
    let mut s = String::from("  let XH = pu(28u); let XW = pu(29u); let cin = M >> 2u;\n");
    for i in 0..tm / 4 {
        let _ = writeln!(s, "  {{\n    let ci = (m0 >> 2u) + {}u + ty;", i * wgy);
        s.push_str("    if (ci < cin) {\n      let ch_off = offC + ci * ldc;\n");
        for j in 0..tn / 4 {
            let _ = writeln!(s, "      {{\n        let cell0 = n0 + ({}u + tx) * 4u;", j * wgx);
            let (a0, a1, b0, b1) = (
                format!("acc_{}_{j}", i * 4),
                format!("acc_{}_{j}", i * 4 + 1),
                format!("acc_{}_{j}", i * 4 + 2),
                format!("acc_{}_{j}", i * 4 + 3),
            );
            if vc {
                let cast = |v: &str| if c_f16 { format!("vec4<f16>({v})") } else { v.to_string() };
                let _ = writeln!(
                    s,
                    "        let u0 = cell0 / OW; let vx0 = cell0 - u0 * OW;\n\
                     \x20       var u1 = u0; var vx1 = vx0 + 2u;\n\
                     \x20       if (vx1 >= OW) {{ vx1 -= OW; u1 += 1u; }}\n\
                     \x20       let p0 = vec4<f32>({a0}.x, {a1}.x, {a0}.y, {a1}.y);\n\
                     \x20       let p1 = vec4<f32>({a0}.z, {a1}.z, {a0}.w, {a1}.w);\n\
                     \x20       let p2 = vec4<f32>({b0}.x, {b1}.x, {b0}.y, {b1}.y);\n\
                     \x20       let p3 = vec4<f32>({b0}.z, {b1}.z, {b0}.w, {b1}.w);\n\
                     \x20       if (u0 * 2u < XH) {{ C[(ch_off + (u0 * 2u) * XW + vx0 * 2u) / 4u] = {}; }}\n\
                     \x20       if (u1 * 2u < XH) {{ C[(ch_off + (u1 * 2u) * XW + vx1 * 2u) / 4u] = {}; }}\n\
                     \x20       if (u0 * 2u + 1u < XH) {{ C[(ch_off + (u0 * 2u + 1u) * XW + vx0 * 2u) / 4u] = {}; }}\n\
                     \x20       if (u1 * 2u + 1u < XH) {{ C[(ch_off + (u1 * 2u + 1u) * XW + vx1 * 2u) / 4u] = {}; }}",
                    cast("p0"),
                    cast("p1"),
                    cast("p2"),
                    cast("p3"),
                );
            } else {
                let cast = |e: &str| if c_f16 { format!("f16({e})") } else { e.to_string() };
                s.push_str("        var su = cell0 / OW; var sv = cell0 - su * OW;\n");
                let comps = ["x", "y", "z", "w"];
                for (c, comp) in comps.iter().enumerate() {
                    let _ = writeln!(
                        s,
                        "        {{ let iy0 = su * 2u; let ix0 = sv * 2u;\n\
                         \x20         if (iy0 < XH) {{\n\
                         \x20           if (ix0 < XW) {{ C[ch_off + iy0 * XW + ix0] = {}; }}\n\
                         \x20           if (ix0 + 1u < XW) {{ C[ch_off + iy0 * XW + ix0 + 1u] = {}; }}\n\
                         \x20         }}\n\
                         \x20         if (iy0 + 1u < XH) {{\n\
                         \x20           if (ix0 < XW) {{ C[ch_off + (iy0 + 1u) * XW + ix0] = {}; }}\n\
                         \x20           if (ix0 + 1u < XW) {{ C[ch_off + (iy0 + 1u) * XW + ix0 + 1u] = {}; }}\n\
                         \x20         }} }}",
                        cast(&format!("{a0}.{comp}")),
                        cast(&format!("{a1}.{comp}")),
                        cast(&format!("{b0}.{comp}")),
                        cast(&format!("{b1}.{comp}")),
                    );
                    if c < 3 {
                        s.push_str("        sv += 1u; if (sv == OW) { sv = 0u; su += 1u; }\n");
                    }
                }
            }
            s.push_str("      }\n");
        }
        s.push_str("    }\n  }\n");
    }
    s
}

/// Fused 2×2 pixel-shuffle epilogue for the tensor-core generator (`CONV_MODE` 1 with `SUBPIXEL`): each 16×16
/// fragment in `Cs` holds 4 channels × 4 sub-pixels × 16 cells, so the 32 lanes interleave `qx = 0` and `qx = 1`
/// directly out of `Cs` into coalesced `vec4` stores (when `XW % 4 == 0`) or bounds-checked scalars.
pub(crate) fn subpixel_epilogue_coop(fm: u32, fnn: u32, vc: bool) -> String {
    let mut s = String::from(
        "  let XH = pu(28u); let XW = pu(29u); let cin = M >> 2u;\n\
         \x20 let s_row = (lane & 31u) >> 2u; let s_c4 = s_row >> 1u; let s_qy = s_row & 1u;\n\
         \x20 let s_j0 = (lane & 3u) * 4u;\n\
         \x20 let cbase = sg * 256u;\n",
    );
    for i in 0..fm {
        for j in 0..fnn {
            let _ = writeln!(s, "  coopStoreT(acc{i}_{j}, &Cs[cbase], 16u);");
            s.push_str("  workgroupBarrier();\n  {\n");
            let _ = writeln!(
                s,
                "    let ci = ((m0 + wm * {}u + {}u) >> 2u) + s_c4;\n\
                 \x20   let cell0 = n0 + wn * {}u + {}u + s_j0;\n\
                 \x20   let r0 = cbase + s_row * 32u + s_j0; let r1 = r0 + 16u;\n\
                 \x20   let v0 = vec4<f32>(Cs[r0], Cs[r1], Cs[r0 + 1u], Cs[r1 + 1u]);\n\
                 \x20   let v1 = vec4<f32>(Cs[r0 + 2u], Cs[r1 + 2u], Cs[r0 + 3u], Cs[r1 + 3u]);\n\
                 \x20   if (ci < cin) {{\n\
                 \x20     let ch_off = offC + ci * ldc;",
                fm * 16,
                i * 16,
                fnn * 16,
                j * 16,
            );
            if vc {
                s.push_str(
                    "      let u0 = cell0 / OW; let vx0 = cell0 - u0 * OW;\n\
                     \x20     let iy0 = u0 * 2u + s_qy;\n\
                     \x20     if (iy0 < XH) { MC[(ch_off + iy0 * XW + vx0 * 2u) / 4u] = vec4<CT>(v0); }\n\
                     \x20     var u1 = u0; var vx1 = vx0 + 2u;\n\
                     \x20     if (vx1 >= OW) { vx1 -= OW; u1 += 1u; }\n\
                     \x20     let iy1 = u1 * 2u + s_qy;\n\
                     \x20     if (iy1 < XH) { MC[(ch_off + iy1 * XW + vx1 * 2u) / 4u] = vec4<CT>(v1); }\n",
                );
            } else {
                s.push_str("      var su = cell0 / OW; var sv = cell0 - su * OW;\n");
                let pairs = [("v0.x", "v0.y"), ("v0.z", "v0.w"), ("v1.x", "v1.y"), ("v1.z", "v1.w")];
                for (c, (e0, e1)) in pairs.iter().enumerate() {
                    let _ = writeln!(
                        s,
                        "      {{ let iy = su * 2u + s_qy; let ix = sv * 2u;\n\
                         \x20       if (iy < XH) {{\n\
                         \x20         if (ix < XW) {{ MC[ch_off + iy * XW + ix] = CT({e0}); }}\n\
                         \x20         if (ix + 1u < XW) {{ MC[ch_off + iy * XW + ix + 1u] = CT({e1}); }}\n\
                         \x20       }} }}"
                    );
                    if c < 3 {
                        s.push_str("      sv += 1u; if (sv == OW) { sv = 0u; su += 1u; }\n");
                    }
                }
            }
            s.push_str("    }\n  }\n  workgroupBarrier();\n");
        }
    }
    s
}

/// WGSL of the stride-phase input gradient's output mapping: phase column `j = (a, b)` is input pixel
/// `(IH0 + a·SH, IW0 + b·SW)` of the plane. An epilogue walks a run of consecutive columns with a cursor
/// (`ppix_at` once, then `ppix_next`), so it divides once per run rather than per element. Params: 25 = `W`,
/// 28 = `IH0`, 29 = `IW0`, 33 = the phase's width.
pub(crate) const PHASE_PIXEL: &str = "struct PPix { a: u32, b: u32 }\n\
    fn ppix_at(j: u32) -> PPix { return PPix(j / pu(33u), j % pu(33u)); }\n\
    fn ppix_off(p: PPix) -> u32 { return (pu(28u) + SH * p.a) * pu(25u) + pu(29u) + SW * p.b; }\n\
    fn ppix_next(p: ptr<function, PPix>) {\n\
    \x20 (*p).b += 1u;\n\
    \x20 if ((*p).b == pu(33u)) { (*p).b = 0u; (*p).a += 1u; }\n}\n";

/// Packs a conv weight `[C_out, C_in/groups, KH, KW]` for the stride-phase input gradient: phase
/// `ph = (rh, rw)` with taps (`rh < min(SH, KH)`, `rw < min(SW, KW)`, row-major) gets the block
/// `[groups, C_in/groups, C_out/groups, th, tw]` of the taps `kh = rh + SH·th`, `kw = rw + SW·tw` at offset
/// `ph · BMAX` (a row-major `A` with `K = C_out/groups · th · tw` per group).
/// Params: 0 = threads (`phases · C_in · C_out/groups · TMAX`), 1 = groups, 2 = `C_in/groups`, 3 = `C_out/groups`,
/// 4 = `BMAX`, 5 = phase columns (`min(SW, KW)`).
fn gen_phase_pack(_: &Spec) -> String {
    "@group(0) @binding(1) var<storage, read> W: array<T>;\n\
     @group(0) @binding(2) var<storage, read_write> PK: array<T>;\n\
     @compute @workgroup_size(256)\n\
     fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
     \x20       @builtin(local_invocation_index) lid: u32) {\n\
     \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
     \x20 if (i >= pu(0u)) { return; }\n\
     \x20 let cin_g = pu(2u); let cout_g = pu(3u);\n\
     \x20 let t = i % TMAX; var q = i / TMAX;\n\
     \x20 let co = q % cout_g; q = q / cout_g;\n\
     \x20 let ci = q % cin_g; q = q / cin_g;\n\
     \x20 let g = q % pu(1u); let ph = q / pu(1u);\n\
     \x20 let rh = ph / pu(5u); let rw = ph % pu(5u);\n\
     \x20 if (rh >= KH || rw >= KW) { return; }\n\
     \x20 let nh = (KH - rh + SH - 1u) / SH; let nw = (KW - rw + SW - 1u) / SW;\n\
     \x20 if (t >= nh * nw) { return; }\n\
     \x20 let kh = rh + SH * (t / nw); let kw = rw + SW * (t % nw);\n\
     \x20 PK[ph * pu(4u) + ((g * cin_g + ci) * cout_g + co) * nh * nw + t] =\n\
     \x20   W[(((g * cout_g + co) * cin_g + ci) * KH + kh) * KW + kw];\n}\n"
        .into()
}

pub(super) static PHASE_PACK: KernelDef = KernelDef {
    name: "conv2d_phase_pack",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_phase_pack),
};

/// One stride phase of a strided conv's input gradient: the input pixels `(ih, iw)` with
/// `(ih + PH) % SH == rh` and `(iw + PW) % SW == rw`, which only the taps `kh ≡ rh`, `kw ≡ rw` (mod the stride) read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Phase {
    /// Taps per axis (`th`, `tw`): `kh = rh + SH·t` for `t < th`.
    pub taps: (usize, usize),
    /// First input row and column of the phase.
    pub first: (usize, usize),
    /// Output pixel of phase pixel `(0, 0)` and tap `(0, 0)`: phase pixel `(a, b)` and tap `(t, u)` read output
    /// pixel `(base.0 + a − t, base.1 + b − u)`.
    pub base: (usize, usize),
    /// Phase rows and columns.
    pub size: (usize, usize),
}

impl Phase {
    /// The phases of `d` (dilation 1) that have taps, `rh < min(SH, KH)` and `rw < min(SW, KW)`, row-major in
    /// `(rh, rw)`. The other phases' pixels get no gradient at all.
    fn with_taps(d: &Conv2dDesc) -> Vec<Phase> {
        let axis = |r: usize, s: usize, p: usize, k: usize, len: usize| {
            let taps = if r < k { (k - r).div_ceil(s) } else { 0 };
            // The smallest i >= 0 with (i + p) % s == r.
            let first = (r + s - p % s) % s;
            let size = if first < len { (len - first).div_ceil(s) } else { 0 };
            (taps, first, (first + p - r) / s, size)
        };
        let (sh, sw) = d.win.stride;
        let (nh, nw) = (sh.min(d.win.kernel.0), sw.min(d.win.kernel.1));
        let mut out = Vec::with_capacity(nh * nw);
        for rh in 0..nh {
            for rw in 0..nw {
                let h = axis(rh, sh, d.win.padding.0, d.win.kernel.0, d.h);
                let w = axis(rw, sw, d.win.padding.1, d.win.kernel.1, d.w);
                out.push(Phase { taps: (h.0, w.0), first: (h.1, w.1), base: (h.2, w.2), size: (h.3, w.3) });
            }
        }
        out
    }

    /// The phase's spec constants (the tap counts).
    pub(crate) fn spec(&self, spec: Spec) -> Spec {
        spec.bool("PHASE", true).u32("PTH", self.taps.0.max(1) as u32).u32("PTW", self.taps.1.max(1) as u32)
    }

    /// The parameters after the conv sizes (24..27): 28 = first row, 29 = first column, 30 / 31 = base, 32 / 33 =
    /// phase rows / columns.
    pub(crate) fn params(&self, p: Params) -> Result<Params> {
        p.at(28)
            .uz(self.first.0)?
            .uz(self.first.1)?
            .uz(self.base.0)?
            .uz(self.base.1)?
            .uz(self.size.0)?
            .uz(self.size.1)
    }
}

/// Whether the input gradient of `d` takes the fused 2×2 sub-pixel conv reformulation: 3×3, stride 2, dilation
/// 1, not depthwise, always on tensor cores and up to `C_in/groups <= 32` on the regular f32/f16 kernels (above
/// which wider f32 layers are strictly compute-bound and prefer 1×-FLOP tap-masked stride phases).
fn subpixel_eligible(d: &Conv2dDesc, coop: bool) -> bool {
    let cin_g = d.c_in / d.groups;
    d.win.kernel == (3, 3)
        && d.win.stride == (2, 2)
        && d.win.dilation == (1, 1)
        && d.win.padding.0 <= i32::MAX as usize
        && d.win.padding.1 <= i32::MAX as usize
        && !d.is_depthwise()
        && (coop || cin_g <= 32)
}

/// The input gradient of a 3×3 stride-2 dilation-1 conv as ONE 2×2 stride-1 sub-pixel conv (`C_out → 4·C_in`)
/// over `grad`, with the 2×2 pixel shuffle fused directly into the GEMM epilogue (`CONV_MODE` 1 + `SUBPIXEL`).
fn conv2d_backward_input_subpixel(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    grad: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    gx: &wgpu::Buffer,
    coop: bool,
) -> Result<()> {
    let (oh, ow) = d.output()?;
    let dt = d.dtype;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let (uh, uw) = (d.h.div_ceil(2), d.w.div_ceil(2));
    let bh = (d.win.padding.0 as i32 - 1).div_euclid(2);
    let bw = (d.win.padding.1 as i32 - 1).div_euclid(2);
    let off_h = (d.win.padding.0 as i32 - 2 * bh) as u32;
    let off_w = (d.win.padding.1 as i32 - 2 * bw) as u32;
    let (m, n_cells, k_dim) = (4 * cin_g, uh * uw, 4 * cout_g);
    let threads = prod(&[d.groups, m, k_dim])?;
    let packed = ctx.alloc(dt.bytes(threads))?;
    let pack_k = ctx.kernel(&SUBPIXEL_PACK, Spec::new().f16(dt == DType::F16))?;
    let pack_p = Params::new().uz(threads)?.uz(cin_g)?.uz(cout_g)?.u(off_h).u(off_w);
    ctx.dispatch(&pack_k, &[weight, packed.raw()], &pack_p, ctx.grid_1d(threads as u64, 256)?)?;
    let (hw_in, hw_out) = (d.h * d.w, oh * ow);
    let mut g = GemmDesc::new(m, n_cells, k_dim, false, false, d.n);
    g.batch = (d.n, d.groups);
    g.lda = k_dim;
    g.ldb = hw_out;
    g.ldc = hw_in;
    g.a = BatchStrides { offset: 0, outer: 0, inner: m * k_dim };
    g.b = BatchStrides { offset: 0, outer: d.c_out * hw_out, inner: cout_g * hw_out };
    g.c = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
    g.dtype = dt;
    if coop
        && coop::encode_conv_backward_input_subpixel(
            ctx,
            d,
            &g,
            (oh, ow, uh, uw, bh, bw),
            packed.raw(),
            grad,
            gx,
        )?
    {
        return Ok(());
    }
    let cfg = gemm::pick_config_conv(ctx, &g);
    gemm::check_config(ctx, cfg)?;
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", true)
        .bool("VEC_B", false)
        .bool("VEC_C", d.w.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1)
        .bool("SUBPIXEL", true)
        .u32("KH", 2)
        .u32("KW", 2)
        .u32("SH", 1)
        .u32("SW", 1)
        .u32("DH", 1)
        .u32("DW", 1)
        .i32("PH", -bh)
        .i32("PW", -bw);
    let kernel = ctx.kernel(&GEMM, spec)?;
    let p = gemm_params(&g)?.at(24).uz(oh)?.uz(ow)?.uz(uh)?.uz(uw)?.uz(d.h)?.uz(d.w)?;
    let groups = [
        n_cells.div_ceil(cfg.bn as usize) as u32,
        m.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.n * d.groups).unwrap_or(u32::MAX),
    ];
    ctx.dispatch(&kernel, &[packed.raw(), grad, ctx.dummy(), gx], &p, groups)
}

/// Whether the input gradient of `d` takes the stride-phase decomposition: strided, dilation 1, not depthwise.
fn phased(d: &Conv2dDesc) -> bool {
    d.win.stride != (1, 1) && d.win.dilation == (1, 1) && !d.is_depthwise()
}

/// The input gradient of a strided conv (dilation 1) by stride phase: a transposed conv reads, for an input pixel,
/// only the taps congruent to its phase (at 3×3 stride 2: 1, 2, 2 or 4 of the 9), so each phase is a dense
/// implicit GEMM with `K = C_out/groups · th · tw` and none of the multiply-adds by the zeros a stride leaves.
/// The weight is packed per phase once ([`gen_phase_pack`]), then one GEMM per phase writes its pixels (a strided
/// subset of `gx`; the phases tile it).
fn conv2d_backward_input_phased(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    grad: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    gx: &wgpu::Buffer,
    coop: bool,
) -> Result<()> {
    let (oh, ow) = d.output()?;
    let dt = d.dtype;
    let (kh, kw) = d.win.kernel;
    let (sh, sw) = d.win.stride;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let (hw_in, hw_out) = (d.h * d.w, oh * ow);
    let tmax = kh.div_ceil(sh) * kw.div_ceil(sw);
    let bmax = prod(&[d.c_in, cout_g, tmax])?;
    // Only the phases with taps are packed and run (at most KH·KW of them, whatever the stride); a kernel smaller
    // than the stride leaves the other phases' pixels zero.
    let phases = Phase::with_taps(d);
    if kh < sh || kw < sw {
        ctx.clear(gx, dt.bytes(prod(&[d.n, d.c_in, d.h, d.w])?));
    }
    let threads = prod(&[phases.len(), bmax])?;
    let packed = ctx.alloc(dt.bytes(threads))?;
    let k = ctx.kernel(&PHASE_PACK, d.win.spec().u32("TMAX", tmax as u32).f16(dt == DType::F16))?;
    let p = Params::new().uz(threads)?.uz(d.groups)?.uz(cin_g)?.uz(cout_g)?.uz(bmax)?.uz(sw.min(kw))?;
    ctx.dispatch(&k, &[weight, packed.raw()], &p, ctx.grid_1d(threads as u64, 256)?)?;
    for (i, ph) in phases.iter().enumerate() {
        let n_ph = ph.size.0 * ph.size.1;
        if n_ph == 0 {
            continue;
        }
        let k_dim = cout_g * ph.taps.0 * ph.taps.1;
        let mut g = GemmDesc::new(cin_g, n_ph, k_dim, false, false, d.n);
        g.batch = (d.n, d.groups);
        g.lda = k_dim;
        g.ldb = hw_out;
        g.ldc = hw_in;
        g.a = BatchStrides { offset: i * bmax, outer: 0, inner: cin_g * k_dim };
        g.b = BatchStrides { offset: 0, outer: d.c_out * hw_out, inner: cout_g * hw_out };
        g.c = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
        g.dtype = dt;
        let vec_a = k_dim.is_multiple_of(4) && bmax.is_multiple_of(4);
        if coop && coop::encode_conv_backward_input_phase(ctx, d, &g, ph, vec_a, packed.raw(), grad, gx)? {
            continue;
        }
        let cfg = gemm::pick_config_conv(ctx, &g);
        gemm::check_config(ctx, cfg)?;
        let check = !g.m.is_multiple_of(cfg.bm as usize)
            || !g.n.is_multiple_of(cfg.bn as usize)
            || !g.k.is_multiple_of(cfg.bk as usize)
            || k_dim == 0;
        let spec = base_spec(&g, cfg)
            .bool("VEC_A", vec_a)
            .bool("VEC_B", false)
            .bool("VEC_C", false)
            .bool("TRANS_A", false)
            .bool("TRANS_B", false)
            .bool("CHECK", check)
            .u32("CONV_MODE", 2);
        let kernel = ctx.kernel(&GEMM, ph.spec(conv_spec(spec, d)))?;
        let p = ph.params(gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?)?;
        let groups = [
            n_ph.div_ceil(cfg.bn as usize) as u32,
            cin_g.div_ceil(cfg.bm as usize) as u32,
            u32::try_from(d.n * d.groups).unwrap_or(u32::MAX),
        ];
        ctx.dispatch(&kernel, &[packed.raw(), grad, ctx.dummy(), gx], &p, groups)?;
    }
    Ok(())
}

/// Gradient of [`conv2d`](crate::kernels::conv::conv2d) with respect to its input. `d` describes the forward problem (its bias and
/// activation are ignored). `grad`: `[N, C_out, OH, OW]`, `weight`: `[C_out, C_in/groups, KH, KW]`,
/// `gx`: `[N, C_in, H, W]` (fully written), all stored as `d.dtype`: f32, or f16 with f32 accumulation (on
/// tensor cores when the context has them on, as a transposed-conv implicit GEMM or, for 1×1 convs, a GEMM over
/// the transposed weight; otherwise the regular kernels).
pub fn conv2d_backward_input(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    grad: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    gx: &wgpu::Buffer,
) -> Result<()> {
    let (oh, ow) = check_grad_desc(ctx, d)?;
    let dt = d.dtype;
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let total = prod(&[d.n, d.c_in, d.h, d.w])?;
    check_buf(grad, prod(&[d.n, d.c_out, oh, ow])?, dt, "conv2d_backward_input grad")?;
    check_buf(weight, prod(&[d.c_out, cin_g, kh, kw])?, dt, "conv2d_backward_input weight")?;
    check_buf(gx, total, dt, "conv2d_backward_input output")?;
    if total == 0 {
        return Ok(());
    }
    if d.c_out * oh * ow == 0 {
        ctx.clear(gx, dt.bytes(total));
        return Ok(());
    }
    let (hw_in, hw_out, khkw) = (d.h * d.w, oh * ow, kh * kw);
    if d.is_depthwise() {
        let spec = d.win.spec().u32("MULT", (d.c_out / d.c_in) as u32).u32("MODE", 0).f16(dt == DType::F16);
        let k = ctx.kernel(&DEPTHWISE_GRAD, spec)?;
        let p = Params::new().uz(total)?.uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?.uz(d.c_in)?;
        return ctx.dispatch(&k, &[grad, weight, gx], &p, ctx.grid_1d(total as u64, 256)?);
    }
    let coop = coop_grad(ctx, d);
    if subpixel_eligible(d, coop) {
        return conv2d_backward_input_subpixel(ctx, d, grad, weight, gx, coop);
    }
    if phased(d) {
        return conv2d_backward_input_phased(ctx, d, grad, weight, gx, coop);
    }
    if d.is_pointwise() && !coop {
        // gx[n, g] = W[g]^T · G[n, g].
        let mut g = GemmDesc::new(cin_g, hw_in, cout_g, true, false, d.n);
        g.batch = (d.n, d.groups);
        g.lda = cin_g;
        g.ldb = hw_in;
        g.ldc = hw_in;
        g.a = BatchStrides { offset: 0, outer: 0, inner: cout_g * cin_g };
        g.b = BatchStrides { offset: 0, outer: d.c_out * hw_in, inner: cout_g * hw_in };
        g.c = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
        g.dtype = dt;
        return gemm::gemm(ctx, &g, weight, grad, None, gx);
    }
    // A[g][ci][(co, kh, kw)] = W[g * cout_g + co][ci][kh, kw].
    let wt = ctx.alloc(dt.bytes(prod(&[d.c_out, cin_g, khkw])?))?;
    crate::kernels::layout::permute(
        ctx,
        weight,
        wt.raw(),
        &[d.groups, cout_g, cin_g, khkw],
        &[0, 2, 1, 3],
        dt,
    )?;
    let k_dim = cout_g * khkw;
    let mut g = GemmDesc::new(cin_g, hw_in, k_dim, false, false, d.n);
    g.batch = (d.n, d.groups);
    g.lda = k_dim;
    g.ldb = hw_out;
    g.ldc = hw_in;
    g.a = BatchStrides { offset: 0, outer: 0, inner: cin_g * k_dim };
    g.b = BatchStrides { offset: 0, outer: d.c_out * hw_out, inner: cout_g * hw_out };
    g.c = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
    g.dtype = dt;
    if coop {
        // A 1×1 conv (stride 1, no padding) has the same view as a plain GEMM over the transposed weight.
        let done = if d.is_pointwise() {
            coop::encode_gemm(ctx, &g, DType::F16, wt.raw(), grad, ctx.dummy(), gx, None)?
        } else {
            coop::encode_conv_backward_input(ctx, d, &g, wt.raw(), grad, gx)?
        };
        if done {
            return Ok(());
        }
        // The tensor-core kernel failed to build (tensor cores are off now): the regular kernel below.
    }
    let cfg = gemm::pick_config_conv(ctx, &g);
    gemm::check_config(ctx, cfg)?;
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", k_dim.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", hw_in.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 2);
    let kernel = ctx.kernel(&GEMM, conv_spec(spec, d))?;
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = [
        hw_in.div_ceil(cfg.bn as usize) as u32,
        cin_g.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.n * d.groups).unwrap_or(u32::MAX),
    ];
    ctx.dispatch(&kernel, &[wt.raw(), grad, ctx.dummy(), gx], &p, groups)
}
