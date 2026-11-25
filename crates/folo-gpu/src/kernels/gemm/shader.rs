//! WGSL generator for the tiled GEMM and implicit-GEMM convolution kernels.

use std::fmt::Write as _;

use crate::kernels::{UNARY_FN, conv, gemm_cat};
use crate::pipeline::Spec;

fn wgsl_ty(f16: bool, vec: bool) -> &'static str {
    match (f16, vec) {
        (false, false) => "f32",
        (false, true) => "vec4<f32>",
        (true, false) => "f16",
        (true, true) => "vec4<f16>",
    }
}

/// How one operand tile is fetched: `inner` units along the contiguous direction per `outer` line.
struct Plan {
    vec: bool,
    /// Units (vec4 or scalar) along the contiguous direction of the tile.
    inner: u32,
    /// Units per thread.
    count: u32,
    /// Outer lines advanced per unit step.
    step: u32,
}

fn plan(units_inner: u32, total_units: u32, threads: u32, vec: bool) -> Plan {
    Plan { vec, inner: units_inner, count: total_units / threads, step: threads / units_inner }
}

/// Decoded specialisation options shared by the code-emission helpers.
struct GenCfg {
    bm: u32,
    bn: u32,
    bk: u32,
    tm: u32,
    tn: u32,
    wgx: u32,
    wgy: u32,
    ta: bool,
    tb: bool,
    vc: bool,
    check: bool,
    mode: u32,
    phase: bool,
    subpixel: bool,
    per_img: bool,
    split: bool,
    tap_mask: bool,
    dx_tap_mask: bool,
    phase_tap_mask: bool,
    sites: bool,
    bias: u32,
    cat: u32,
    c_f16: bool,
    as_vec: bool,
    bs_vec: bool,
    pa: Plan,
    pb: Plan,
}

pub(crate) fn gen_gemm(spec: &Spec) -> String {
    let bm = spec.get_u32("BM", 128);
    let bn = spec.get_u32("BN", 128);
    let bk = spec.get_u32("BK", 16);
    let tm = spec.get_u32("TM", 8);
    let tn = spec.get_u32("TN", 8);
    let (va, vb, vc) = (spec.get_bool("VEC_A"), spec.get_bool("VEC_B"), spec.get_bool("VEC_C"));
    let (ta, tb) = (spec.get_bool("TRANS_A"), spec.get_bool("TRANS_B"));
    let check = spec.get_bool("CHECK");
    // 0 = plain GEMM, 1 = conv forward, 2 = conv input gradient, 3 = conv weight gradient.
    let mode = spec.get_u32("CONV_MODE", 0);
    let conv_mode = mode != 0;
    // Input gradient of one stride phase (`conv`'s stride-phase decomposition).
    let phase = mode == 2 && spec.get_bool("PHASE");
    // Fused 2×2 sub-pixel stride-2 input gradient (`CONV_MODE` 1 with 2×2 pixel-shuffle epilogue).
    let subpixel = mode == 1 && spec.get_bool("SUBPIXEL");
    // Per-image weight gradient (`CONV_MODE` 3 with `b1 = n`, `K = OH·OW`: drops `/ OHOW` and `% OHOW`).
    let per_img = mode == 3 && spec.get_bool("PER_IMG");
    let split = spec.get_bool("SPLIT");
    // Forward, stride-1 dX, and stride-phase dX windows of at most 32 taps keep a per-column tap mask (`vm`).
    let (kh_n, kw_n) = (spec.get_u32("KH", 1), spec.get_u32("KW", 1));
    let (sh_n, sw_n) = (spec.get_u32("SH", 1), spec.get_u32("SW", 1));
    let (dh_n, dw_n) = (spec.get_u32("DH", 1), spec.get_u32("DW", 1));
    let (pth_n, ptw_n) = (spec.get_u32("PTH", 1), spec.get_u32("PTW", 1));
    let taps = kh_n.saturating_mul(kw_n);
    let ptaps = pth_n.saturating_mul(ptw_n);
    let tap_mask = mode == 1 && taps > 0 && taps <= 32;
    let dx_tap_mask = mode == 2 && !phase && sh_n == 1 && sw_n == 1 && taps > 0 && taps <= 32;
    let phase_tap_mask = phase && ptaps > 0 && ptaps <= 32;
    let sites = mode == 1 && spec.get_bool("SITES");
    let bias = spec.get_u32("BIAS_KIND", 0);
    // `CAT` > 1: `B` is that many row blocks in separate buffers (`gemm_cat`; plain GEMM, `B` not transposed).
    let cat = spec.get_u32("CAT", 1);
    let f16 = spec.is_f16();
    // `OUT_F32`: f16 operands but an f32 `C` (the weight gradient's f32 partial slabs / f32 result).
    let c_f16 = f16 && !spec.get_bool("OUT_F32");
    let wgx = bn / tn;
    let wgy = bm / tm;
    let threads = wgx * wgy;
    let (bm4, bn4) = (bm / 4, bn / 4);

    // Operand tile fetch plans (see the module docs for the layouts).
    let pa = match (va, ta) {
        (true, false) => plan(bk / 4, bm * bk / 4, threads, true),
        (true, true) => plan(bm4, bm * bk / 4, threads, true),
        (false, false) => plan(bk, bm * bk, threads, false),
        (false, true) => plan(bm, bm * bk, threads, false),
    };
    let pb = match (vb, tb) {
        (true, false) => plan(bn4, bn * bk / 4, threads, true),
        (true, true) => plan(bk / 4, bn * bk / 4, threads, true),
        (false, false) => plan(bn, bn * bk, threads, false),
        (false, true) => plan(bk, bn * bk, threads, false),
    };
    // Workgroup tiles use `vec4<f32>` only when one invocation writes all four lanes; otherwise scalar `f32`
    // avoids WGSL multi-invocation lane writes to the same `vec4` before the barrier. `WG_PAD` adds one vec4
    // (4 floats, `2 · BK · 16` bytes) per `Bs` row on `CONV_MODE` 3 so its stores stay free of bank conflicts.
    let (as_vec, bs_vec) = (pa.vec && ta, mode == 0 && pb.vec && !tb);
    let wg_pad = mode == 3 && spec.get_bool("WG_PAD");

    let g = GenCfg {
        bm,
        bn,
        bk,
        tm,
        tn,
        wgx,
        wgy,
        ta,
        tb,
        vc,
        check,
        mode,
        phase,
        subpixel,
        per_img,
        split,
        tap_mask,
        dx_tap_mask,
        phase_tap_mask,
        sites,
        bias,
        cat,
        c_f16,
        as_vec,
        bs_vec,
        pa,
        pb,
    };

    let mut s = String::with_capacity(32 * 1024);
    s.push_str(UNARY_FN);
    if phase {
        s.push_str(conv::PHASE_PIXEL);
    }
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> A: array<{}>;", wgsl_ty(f16, va));
    let _ =
        writeln!(s, "@group(0) @binding(2) var<storage, read> B: array<{}>;", wgsl_ty(f16, vb && !conv_mode));
    s.push_str("@group(0) @binding(3) var<storage, read> BIAS: array<T>;\n");
    let _ = writeln!(s, "@group(0) @binding(4) var<storage, read_write> C: array<{}>;", wgsl_ty(c_f16, vc));
    if cat > 1 {
        s.push_str(&gemm_cat::cat_source(cat, "B", wgsl_ty(f16, vb), vb));
    }
    if sites {
        s.push_str(conv::SITES_BINDING);
    }
    let (at, aw) = if as_vec { ("vec4<f32>", bm4) } else { ("f32", bm) };
    let (bt, bw) = if bs_vec { ("vec4<f32>", bn4) } else { ("f32", bn + 4 * u32::from(wg_pad)) };
    let _ = writeln!(s, "var<workgroup> As: array<array<{at}, {aw}>, {}>;", 2 * bk);
    let _ = writeln!(s, "var<workgroup> Bs: array<array<{bt}, {bw}>, {}>;", 2 * bk);
    let _ = writeln!(s, "@compute @workgroup_size({wgx}, {wgy}, 1)");
    s.push_str(
        "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {\n",
    );
    let _ = writeln!(s, "  let tx = lid3.x; let ty = lid3.y; let tid = ty * {wgx}u + tx;");
    // Runs of 16 pixels where the tile is that deep (two 32-byte sectors per column and load).
    let wb = (mode == 3).then(|| conv::wgrad_b_units(threads, bk, bn, spec.get_u32("WG_RUN", bk.min(16))));
    if let Some(wb) = &wb {
        let _ = writeln!(s, "  let wl = tid % {0}u; let wq = tid / {0}u;", wb.run);
    }
    emit_prologue(&mut s, &g, spec, wb.as_ref(), (kh_n, kw_n, dh_n, dw_n, pth_n, ptw_n));

    let load = emit_loads(&g, wb.as_ref());
    let store0 = emit_stores(&g, wb.as_ref(), 0);
    let store1 = emit_stores(&g, wb.as_ref(), 1);
    let compute0 = emit_compute(&g, 0);
    let compute1 = emit_compute(&g, 1);

    // --- main loop: two K tiles per iteration so every workgroup-memory buffer index is a constant ----------
    // Guarded by `nt > 0u` (workgroup-uniform) so K == 0 or empty split chunks never touch A/B or synchronize.
    // `workgroupBarrier()` sits inside `if (more)` so the final tile skips a redundant trailing barrier.
    s.push_str("  if (nt > 0u) {\n");
    let _ = writeln!(s, "    {{\n      let k0 = kbeg;\n{load}    }}");
    let _ = writeln!(s, "    {{\n{store0}    }}");
    s.push_str("    workgroupBarrier();\n");
    s.push_str("    for (var t = 0u; t < nt; t += 2u) {\n");
    let _ = writeln!(
        s,
        "     {{\n      let more = t + 1u < nt;\n      if (more) {{\n      let k0 = kbeg + (t + 1u) * BK;\n{load}      }}\n{compute0}\
         \x20     if (more) {{\n{store1}        workgroupBarrier();\n      }}\n     }}"
    );
    let _ = writeln!(
        s,
        "     if (t + 1u < nt) {{\n      let more = t + 2u < nt;\n      if (more) {{\n      let k0 = kbeg + (t + 2u) * BK;\n{load}      }}\n{compute1}\
         \x20     if (more) {{\n{store0}        workgroupBarrier();\n      }}\n     }}"
    );
    s.push_str("    }\n  }\n\n");

    emit_epilogue(&mut s, &g);
    s.push_str("}\n");
    s
}

fn emit_prologue(
    s: &mut String,
    g: &GenCfg,
    spec: &Spec,
    wb: Option<&conv::WgradPlan>,
    (kh_n, kw_n, dh_n, dw_n, pth_n, ptw_n): (u32, u32, u32, u32, u32, u32),
) {
    s.push_str(
        "  let M = pu(0u); let N = pu(1u);\n\
         let lda = pu(3u); let ldb = pu(4u); let ldc = pu(5u);\n",
    );
    if g.split {
        // wid.z = batch index * S + chunk; chunk `sidx` covers k in [kbeg, kbeg + KS) and writes its own C slab.
        s.push_str(
            "  let S = pu(28u); let zz = wid.z / S; let sidx = wid.z % S;\n\
             let b1 = zz / pu(6u); let b2 = zz % pu(6u);\n\
             let kbeg = sidx * pu(29u); let K = min(pu(2u), kbeg + pu(29u));\n\
             let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u) + sidx * pu(30u);\n",
        );
    } else {
        s.push_str(
            "  let b1 = wid.z / pu(6u); let b2 = wid.z % pu(6u);\n\
             let kbeg = 0u; let K = pu(2u);\n\
             let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u);\n",
        );
    }
    s.push_str(
        "  let offA = pu(8u) + b1 * pu(9u) + b2 * pu(10u);\n\
         let offB = pu(11u) + b1 * pu(12u) + b2 * pu(13u);\n\
         let offBias = pu(18u) + b1 * pu(19u) + b2 * pu(20u);\n",
    );
    let _ = writeln!(s, "  let m0 = wid.y * {}u; let n0 = wid.x * {}u;", g.bm, g.bn);
    if g.sites {
        s.push_str(conv::SITES_LIVE);
        s.push_str(conv::SITES_RETIRE);
    }
    match g.mode {
        // Forward gather: B[k][j] = X[ci][oh*SH - PH + kh*DH][ow*SW - PW + kw*DW].
        1 => s.push_str("  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u);\n"),
        // Input gradient: B[k = (co, kh, kw)][j = (ih, iw)] = G[co][(ih + PH - kh*DH) / SH][(iw + PW - kw*DW) / SW]
        // where the divisions are exact and in range, else 0.
        2 => {
            s.push_str("  let IH = pu(24u); let IW = pu(25u); let OH = pu(26u); let OW = pu(27u);\n");
            if g.phase {
                s.push_str("  let WR = pu(33u);\n");
            }
        }
        // Weight gradient: k = (n, oh, ow) (or `(oh, ow)` per image when `PER_IMG`); column = (ci, kh, kw).
        3 => {
            s.push_str(&conv::weight_grad_dims(spec));
            if !g.per_img {
                s.push_str("  let SAN = pu(32u); let SBN = pu(33u);\n");
            }
        }
        _ => {}
    }
    // Per-thread tile coordinates for the loads.
    let _ = writeln!(s, "  let ai = tid % {}u; let ao = tid / {}u;", g.pa.inner, g.pa.inner);
    let _ = writeln!(s, "  let bi = tid % {}u; let bo = tid / {}u;", g.pb.inner, g.pb.inner);
    // One B column per thread for the whole K loop, so its index math is done once.
    match g.mode {
        1 => {
            s.push_str(&conv::forward_column("bi", g.sites));
            if g.tap_mask {
                s.push_str(&conv::fwd_tap_mask_setup(kh_n, kw_n, dh_n, dw_n, "cbase"));
            }
        }
        2 if g.phase => {
            s.push_str(
                "  let cj = n0 + bi; let cjok = cj < N;\n\
                 let ca = pi(30u) + i32(cj / WR); let cb = pi(31u) + i32(cj % WR);\n",
            );
            if g.phase_tap_mask {
                s.push_str(&conv::phase_tap_mask_setup(pth_n, ptw_n, "pbase"));
            }
        }
        2 => {
            s.push_str(
                "  let cj = n0 + bi; let cjok = cj < N;\n\
                 let cih = i32(cj / IW) + PH; let ciw = i32(cj % IW) + PW;\n",
            );
            if g.dx_tap_mask {
                s.push_str(&conv::dx_tap_mask_setup(kh_n, kw_n, dh_n, dw_n, "cbase"));
            }
        }
        3 => {
            if let Some(wb) = wb {
                s.push_str(&conv::wgrad_b_setup(wb, g.per_img));
            }
        }
        _ => {}
    }
    let _ = writeln!(
        s,
        "  var ra: array<vec4<f32>, {}>;",
        g.pa.count.div_ceil(if g.pa.vec { 1 } else { 4 }).max(1)
    );
    let _ = writeln!(
        s,
        "  var rb: array<vec4<f32>, {}>;",
        g.pb.count.div_ceil(if g.pb.vec { 1 } else { 4 }).max(1)
    );
    for i in 0..g.tm {
        for j in 0..g.tn / 4 {
            let _ = writeln!(s, "  var acc_{i}_{j} = vec4<f32>(0.0);");
        }
    }
    for i in (0..g.tm / 4).filter(|_| !g.as_vec) {
        let _ = writeln!(s, "  let ac{i} = (ty + {}u) * 4u;", i * g.wgy);
    }
    for j in (0..g.tn / 4).filter(|_| !g.bs_vec) {
        let _ = writeln!(s, "  let bc{j} = (tx + {}u) * 4u;", j * g.wgx);
    }
    s.push_str("  let nt = select(0u, (K - kbeg + BK - 1u) / BK, kbeg < K);\n");
}

/// Emits the global → register loads for one `K` tile at `k0`.
fn emit_loads(g: &GenCfg, wb: Option<&conv::WgradPlan>) -> String {
    let mut load = String::new();
    for u in 0..g.pa.count {
        let o = format!("(ao + {}u)", u * g.pa.step);
        let (reg, comp) = if g.pa.vec {
            (format!("ra[{u}]"), String::new())
        } else {
            (format!("ra[{}]", u / 4), format!("[{}]", u % 4))
        };
        let (idx, cond) = match (g.pa.vec, g.ta) {
            // Weight gradient: k = (n, pixel) folds the batch into K; `PER_IMG` takes the linear path below.
            (true, false) if g.mode == 3 && !g.per_img => (
                format!(
                    "(offA + ((k0 + ai * 4u) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + ai * 4u) % OHOW) / 4u"
                ),
                format!("m0 + {o} < M && k0 + ai * 4u < K"),
            ),
            (false, false) if g.mode == 3 && !g.per_img => (
                format!("offA + ((k0 + ai) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + ai) % OHOW"),
                format!("m0 + {o} < M && k0 + ai < K"),
            ),
            (true, false) => (
                format!("(offA / 4u) + (m0 + {o}) * (lda / 4u) + ((k0 >> 2u) + ai)"),
                format!("m0 + {o} < M && k0 + ai * 4u < K"),
            ),
            (true, true) => (
                format!("(offA / 4u) + (k0 + {o}) * (lda / 4u) + ((m0 >> 2u) + ai)"),
                format!("k0 + {o} < K && m0 + ai * 4u < M"),
            ),
            (false, false) => {
                (format!("offA + (m0 + {o}) * lda + k0 + ai"), format!("m0 + {o} < M && k0 + ai < K"))
            }
            (false, true) => {
                (format!("offA + (k0 + {o}) * lda + m0 + ai"), format!("k0 + {o} < K && m0 + ai < M"))
            }
        };
        let val = if g.pa.vec { format!("vec4<f32>(A[{idx}])") } else { format!("f32(A[{idx}])") };
        let zero = if g.pa.vec { "vec4<f32>(0.0)" } else { "0.0" };
        if g.check {
            let _ =
                writeln!(load, "    if ({cond}) {{ {reg}{comp} = {val}; }} else {{ {reg}{comp} = {zero}; }}");
        } else {
            let _ = writeln!(load, "    {reg}{comp} = {val};");
        }
    }
    if let Some(wb) = wb {
        let reg = |u: usize| format!("rb[{}][{}]", u / 4, u % 4);
        load.push_str(&conv::wgrad_b_load(wb, "B", reg, "f32", "0.0", g.per_img, g.check));
    }
    for u in 0..g.pb.count {
        let o = format!("(bo + {}u)", u * g.pb.step);
        let (reg, comp) = if g.pb.vec {
            (format!("rb[{u}]"), String::new())
        } else {
            (format!("rb[{}]", u / 4), format!("[{}]", u % 4))
        };
        let dst = format!("{reg}{comp}");
        match g.mode {
            1 if g.tap_mask => {
                load.push_str(&conv::fwd_tap_mask_load("B", "cbase", &dst, "f32", "0.0", &o, g.check));
                continue;
            }
            1 => {
                // Unit (k = k0 + o, j = cj): gather from the image.
                let _ = writeln!(
                    load,
                    "    {{ let k = k0 + {o}; let ci = k / (KH * KW); let r = k % (KH * KW);\n\
                     \x20     let ih = cih0 + i32((r / KW) * DH); let iw = ciw0 + i32((r % KW) * DW);\n\
                     \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                     \x20       {dst} = f32(B[offB + (ci * IH + u32(ih)) * IW + u32(iw)]);\n\
                     \x20     }} else {{ {dst} = 0.0; }} }}"
                );
                continue;
            }
            2 if g.phase_tap_mask => {
                load.push_str(&conv::phase_tap_mask_load("B", "pbase", &dst, "f32", "0.0", &o, g.check));
                continue;
            }
            2 if g.phase => {
                // Unit (k = (co, th, tw), j = phase pixel (a, b)): output pixel (BH + a - th, BW + b - tw).
                let _ = writeln!(
                    load,
                    "    {{ let k = k0 + {o}; let co = k / (PTH * PTW); let r = k % (PTH * PTW);\n\
                     \x20     let oh = ca - i32(r / PTW); let ow = cb - i32(r % PTW);\n\
                     \x20     var v = 0.0;\n\
                     \x20     if (cjok && k < K && oh >= 0 && ow >= 0 && oh < i32(OH) && ow < i32(OW)) {{\n\
                     \x20       v = f32(B[offB + (co * OH + u32(oh)) * OW + u32(ow)]);\n\
                     \x20     }}\n\
                     \x20     {dst} = v; }}"
                );
                continue;
            }
            2 if g.dx_tap_mask => {
                load.push_str(&conv::dx_tap_mask_load("B", "cbase", &dst, "f32", "0.0", &o, g.check));
                continue;
            }
            2 => {
                // Unit (k = (co, kh, kw), j = input pixel): the output pixel that read it, if any.
                let _ = writeln!(
                    load,
                    "    {{ let k = k0 + {o}; let co = k / (KH * KW); let r = k % (KH * KW);\n\
                     \x20     let th = cih - i32((r / KW) * DH); let tw = ciw - i32((r % KW) * DW);\n\
                     \x20     var v = 0.0;\n\
                     \x20     if (cjok && k < K && th >= 0 && tw >= 0) {{\n\
                     \x20       let uh = u32(th); let uw = u32(tw);\n\
                     \x20       if (uh % SH == 0u && uw % SW == 0u && uh / SH < OH && uw / SW < OW) {{\n\
                     \x20         v = f32(B[offB + (co * OH + uh / SH) * OW + uw / SW]);\n\
                     \x20       }}\n\
                     \x20     }}\n\
                     \x20     {reg}{comp} = v; }}"
                );
                continue;
            }
            // Loaded above (`wgrad_b_load`).
            3 => continue,
            _ => {}
        }
        let (idx, cond) = match (g.pb.vec, g.tb) {
            (true, false) => (
                format!("(offB / 4u) + (k0 + {o}) * (ldb / 4u) + ((n0 >> 2u) + bi)"),
                format!("k0 + {o} < K && n0 + bi * 4u < N"),
            ),
            (true, true) => (
                format!("(offB / 4u) + (n0 + {o}) * (ldb / 4u) + ((k0 >> 2u) + bi)"),
                format!("n0 + {o} < N && k0 + bi * 4u < K"),
            ),
            (false, false) => {
                (format!("offB + (k0 + {o}) * ldb + n0 + bi"), format!("k0 + {o} < K && n0 + bi < N"))
            }
            (false, true) => {
                (format!("offB + (n0 + {o}) * ldb + k0 + bi"), format!("n0 + {o} < N && k0 + bi < K"))
            }
        };
        let fetch = match (g.cat > 1, g.pb.vec) {
            (true, true) => format!("cat_b(k0 + {o}, (n0 >> 2u) + bi, b1)"),
            (true, false) => format!("cat_b(k0 + {o}, n0 + bi, b1)"),
            (false, _) => format!("B[{idx}]"),
        };
        let val = if g.pb.vec { format!("vec4<f32>({fetch})") } else { format!("f32({fetch})") };
        let zero = if g.pb.vec { "vec4<f32>(0.0)" } else { "0.0" };
        if g.check {
            let _ =
                writeln!(load, "    if ({cond}) {{ {reg}{comp} = {val}; }} else {{ {reg}{comp} = {zero}; }}");
        } else {
            let _ = writeln!(load, "    {reg}{comp} = {val};");
        }
    }
    load
}

/// Emits register → workgroup-memory stores for double-buffer half `sb`.
///
/// Tiles are 2-D arrays (`As[k][m/4]` / `As[k][m]`): with the buffer index and `k` constant and the column
/// index loop-invariant, the index clamps naga adds for workgroup arrays fold or hoist out of the main loop.
fn emit_stores(g: &GenCfg, wb: Option<&conv::WgradPlan>, sb: u32) -> String {
    let mut st = String::new();
    let k0s = sb * g.bk;
    for u in 0..g.pa.count {
        let o = format!("(ao + {}u)", u * g.pa.step);
        let (x, y) = (u / 4, u % 4);
        match (g.pa.vec, g.ta) {
            (true, false) => {
                for c in 0..4 {
                    let _ = writeln!(st, "    As[{k0s}u + ai * 4u + {c}u][{o}] = ra[{u}][{c}];");
                }
            }
            (true, true) => {
                let _ = writeln!(st, "    As[{k0s}u + {o}][ai] = ra[{u}];");
            }
            (false, false) => {
                let _ = writeln!(st, "    As[{k0s}u + ai][{o}] = ra[{x}][{y}];");
            }
            (false, true) => {
                let _ = writeln!(st, "    As[{k0s}u + {o}][ai] = ra[{x}][{y}];");
            }
        }
    }
    for (u, w) in wb.iter().flat_map(|wb| wb.units.iter().map(move |w| (wb.run, w))).enumerate() {
        let row = format!("{k0s}u + {} * {}u + wl", w.1.kq, w.0);
        let _ = writeln!(st, "    Bs[{row}][{}] = rb[{}][{}];", w.1.jj, u / 4, u % 4);
    }
    for u in (0..g.pb.count).filter(|_| g.mode != 3) {
        let o = format!("(bo + {}u)", u * g.pb.step);
        let (x, y) = (u / 4, u % 4);
        match (g.pb.vec, g.tb) {
            (true, false) if g.bs_vec => {
                let _ = writeln!(st, "    Bs[{k0s}u + {o}][bi] = rb[{u}];");
            }
            (true, false) => {
                for c in 0..4 {
                    let _ = writeln!(st, "    Bs[{k0s}u + {o}][bi * 4u + {c}u] = rb[{u}][{c}];");
                }
            }
            (true, true) => {
                for c in 0..4 {
                    let _ = writeln!(st, "    Bs[{k0s}u + bi * 4u + {c}u][{o}] = rb[{u}][{c}];");
                }
            }
            (false, false) => {
                let _ = writeln!(st, "    Bs[{k0s}u + {o}][bi] = rb[{x}][{y}];");
            }
            (false, true) => {
                let _ = writeln!(st, "    Bs[{k0s}u + bi][{o}] = rb[{x}][{y}];");
            }
        }
    }
    st
}

/// Emits the unrolled inner `BK` FMA step on double-buffer half `sb`.
///
/// The k loop is unrolled here rather than left to the driver: naga guards every WGSL loop with an
/// iteration counter (infinite-loop protection), which blocks the driver's unrolling and costs 20-70%.
fn emit_compute(g: &GenCfg, sb: u32) -> String {
    let read4 = |a: &str, r: u32, col: &str| {
        format!(
            "vec4<f32>({a}[{r}u][{col}], {a}[{r}u][{col} + 1u], {a}[{r}u][{col} + 2u], {a}[{r}u][{col} + 3u])"
        )
    };
    let mut c = String::new();
    for kk in 0..g.bk {
        let r = sb * g.bk + kk;
        c.push_str("    {\n");
        for i in 0..g.tm / 4 {
            let v = if g.as_vec {
                format!("As[{r}u][ty + {}u]", i * g.wgy)
            } else {
                read4("As", r, &format!("ac{i}"))
            };
            let _ = writeln!(c, "      let a{i} = {v};");
        }
        for j in 0..g.tn / 4 {
            let v = if g.bs_vec {
                format!("Bs[{r}u][tx + {}u]", j * g.wgx)
            } else {
                read4("Bs", r, &format!("bc{j}"))
            };
            let _ = writeln!(c, "      let b{j} = {v};");
        }
        for i in 0..g.tm / 4 {
            for q in 0..4 {
                for j in 0..g.tn / 4 {
                    let acc = format!("acc_{}_{j}", i * 4 + q);
                    let _ = writeln!(c, "      {acc} = fma(vec4<f32>(a{i}[{q}]), b{j}, {acc});");
                }
            }
        }
        c.push_str("    }\n");
    }
    c
}

/// Emits the alpha / bias / activation / store epilogue.
fn emit_epilogue(s: &mut String, g: &GenCfg) {
    if g.subpixel {
        s.push_str(&conv::subpixel_epilogue_gemm(g.tm, g.tn, g.wgx, g.wgy, g.vc, g.c_f16));
        return;
    }
    s.push_str("  let alpha = pf(7u); let act_alpha = pf(17u);\n");
    for i in 0..g.tm / 4 {
        for r in 0..4 {
            let row_i = i * 4 + r;
            let _ = writeln!(s, "  {{\n    let row = m0 + ({}u + ty) * 4u + {r}u;", i * g.wgy);
            s.push_str("    if (row < M) {\n");
            for j in 0..g.tn / 4 {
                let _ = writeln!(s, "    {{\n      let col = n0 + ({}u + tx) * 4u;", j * g.wgx);
                let _ = writeln!(s, "      var v = acc_{row_i}_{j} * alpha;");
                match g.bias {
                    1 => s.push_str("      v += vec4<f32>(f32(BIAS[offBias + row]));\n"),
                    2 => s.push_str(
                        "      v += vec4<f32>(f32(BIAS[offBias + min(col, N - 1u)]), f32(BIAS[offBias + min(col + 1u, N - 1u)]),\n\
                         \x20                f32(BIAS[offBias + min(col + 2u, N - 1u)]), f32(BIAS[offBias + min(col + 3u, N - 1u)]));\n",
                    ),
                    _ => {}
                }
                s.push_str("      v = unary4(ACT, v, act_alpha);\n");
                if g.vc {
                    let cast = if g.c_f16 { "vec4<f16>(v)" } else { "v" };
                    let _ =
                        writeln!(s, "      if (col < N) {{ C[(offC + row * ldc + col) / 4u] = {cast}; }}");
                } else {
                    if g.phase {
                        s.push_str("      var pp = ppix_at(col);\n");
                    }
                    for c in 0..4 {
                        let cast = if g.c_f16 { format!("f16(v[{c}])") } else { format!("v[{c}]") };
                        let at = if g.phase { "ppix_off(pp)".to_string() } else { format!("col + {c}u") };
                        let _ = writeln!(
                            s,
                            "      if (col + {c}u < N) {{ C[offC + row * ldc + {at}] = {cast}; }}"
                        );
                        if g.phase && c < 3 {
                            s.push_str("      ppix_next(&pp);\n");
                        }
                    }
                }
                s.push_str("    }\n");
            }
            s.push_str("    }\n  }\n");
        }
    }
}
