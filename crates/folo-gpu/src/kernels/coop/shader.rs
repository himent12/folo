//! The WGSL generator of the tensor-core GEMM / implicit-GEMM conv kernel ([`COOP`]): tile-load plans, the
//! double-buffered K loop on cooperative matrices and the fragment-by-fragment epilogue.

use std::fmt::Write as _;

use super::{CoopConfig, FRAG, PAD};
use crate::kernels::UNARY_FN;
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// One operand-tile load plan: element `u` of this invocation covers inner unit `tid % inner` of line
/// `tid / inner + u · (threads / inner)`; `guard` is set when fewer than `threads` units exist.
struct Lines {
    threads: u32,
    inner: u32,
    count: u32,
    guard: Option<u32>,
}

fn lines(threads: u32, inner: u32, units: u32) -> Lines {
    Lines { threads, inner, count: units.div_ceil(threads), guard: (units < threads).then_some(units) }
}

impl Lines {
    fn outer(&self, u: u32) -> String {
        format!("(tid / {}u + {}u)", self.inner, u * (self.threads / self.inner))
    }

    fn inner_expr(&self) -> String {
        format!("(tid % {}u)", self.inner)
    }

    /// Wraps `body` in the "fewer than `threads` units" guard.
    fn guarded(&self, body: String) -> String {
        match self.guard {
            Some(n) => format!("    if (tid < {n}u) {{\n{body}    }}\n"),
            None => body,
        }
    }
}

pub(crate) fn gen_coop(spec: &Spec) -> String {
    let cfg = CoopConfig {
        bm: spec.get_u32("BM", 128),
        bn: spec.get_u32("BN", 128),
        bk: spec.get_u32("BK", 32),
        wgm: spec.get_u32("WGM", 2),
        wgn: spec.get_u32("WGN", 4),
    };
    assert!(cfg.valid(), "invalid tensor-core config {cfg:?}");
    let CoopConfig { bm, bn, bk, wgm: _, wgn } = cfg;
    let (threads, subgroups) = (cfg.threads(), cfg.subgroups());
    let (fm, fnn) = (cfg.fm(), cfg.fnn());
    let (sa, sb) = (bk + PAD, bn + PAD);
    let (va, vb, vc) = (spec.get_bool("VEC_A"), spec.get_bool("VEC_B"), spec.get_bool("VEC_C"));
    let (ta, tb) = (spec.get_bool("TRANS_A"), spec.get_bool("TRANS_B"));
    let check = spec.get_bool("CHECK");
    // 0 = plain GEMM, 1 = conv forward, 2 = conv input gradient, 3 = conv weight gradient (as `crate::kernels::gemm`).
    let mode = spec.get_u32("CONV_MODE", 0);
    let conv = mode != 0;
    // Input gradient of one stride phase (`crate::kernels::conv`'s stride-phase decomposition).
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
    // The forward conv at listed output pixels (`crate::kernels::conv::conv2d_sites`).
    let sites = mode == 1 && spec.get_bool("SITES");
    let bias = spec.get_u32("BIAS_KIND", 0);
    // `CAT` > 1: `B` is that many row blocks in separate buffers (`crate::kernels::gemm_cat`).
    let cat = spec.get_u32("CAT", 1);
    let ct = if spec.get_bool("OUT_F16") { "f16" } else { "f32" };
    let (a_len, b_len) = (bm * sa, bk * sb);

    // Condition 2 of `experimental_opt_in`: every fragment lies inside its workgroup array. The largest
    // subgroup index is `subgroups - 1` (clamped in the shader), so these are the furthest fragments.
    let a_last = ((cfg.wgm - 1) * fm * 16 + (fm - 1) * 16) * sa + (bk - 16);
    assert!(a_last + 15 * sa + 16 <= a_len, "A fragment out of bounds");
    let b_last = (bk - 16) * sb + (wgn - 1) * fnn * 16 + (fnn - 1) * 16;
    assert!(b_last + 15 * sb + 16 <= b_len, "B fragment out of bounds");
    assert!((subgroups - 1) * FRAG + 15 * 16 + 16 <= subgroups * FRAG, "scratch fragment out of bounds");

    let vty = |v: bool, scalar: &str| if v { format!("vec4<{scalar}>") } else { scalar.to_string() };
    let mut s = String::new();
    s.push_str(UNARY_FN);
    if phase {
        s.push_str(crate::kernels::conv::PHASE_PIXEL);
    }
    let _ = writeln!(s, "alias CT = {ct};");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> MA: array<{}>;", vty(va, "f16"));
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read> MB: array<{}>;", vty(vb && !conv, "f16"));
    s.push_str("@group(0) @binding(3) var<storage, read> MBIAS: array<CT>;\n");
    let _ = writeln!(s, "@group(0) @binding(4) var<storage, read_write> MC: array<{}>;", vty(vc, "CT"));
    if cat > 1 {
        s.push_str(&crate::kernels::gemm_cat::cat_source(cat, "MB", &vty(vb, "f16"), vb));
    }
    if sites {
        s.push_str(crate::kernels::conv::SITES_BINDING);
    }
    for b in 0..2 {
        let _ = writeln!(s, "var<workgroup> As{b}: array<f16, {a_len}>;");
        let _ = writeln!(s, "var<workgroup> Bs{b}: array<f16, {b_len}>;");
    }
    let _ = writeln!(s, "var<workgroup> Cs: array<f32, {}>;", subgroups * FRAG);
    let _ = writeln!(s, "@compute @workgroup_size({threads})");
    s.push_str(
        "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32,\n\
         \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
         \x20 let M = pu(0u); let N = pu(1u);\n\
         \x20 let lda = pu(3u); let ldb = pu(4u); let ldc = pu(5u);\n",
    );
    if split {
        s.push_str(
            "  let S = pu(28u); let zz = wid.z / S; let sidx = wid.z % S;\n\
             \x20 let b1 = zz / pu(6u); let b2 = zz % pu(6u);\n\
             \x20 let kbeg = sidx * pu(29u); let K = min(pu(2u), kbeg + pu(29u));\n\
             \x20 let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u) + sidx * pu(30u);\n",
        );
    } else {
        s.push_str(
            "  let b1 = wid.z / pu(6u); let b2 = wid.z % pu(6u);\n\
             \x20 let kbeg = 0u; let K = pu(2u);\n\
             \x20 let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u);\n",
        );
    }
    s.push_str(
        "  let offA = pu(8u) + b1 * pu(9u) + b2 * pu(10u);\n\
         \x20 let offB = pu(11u) + b1 * pu(12u) + b2 * pu(13u);\n\
         \x20 let offBias = pu(18u) + b1 * pu(19u) + b2 * pu(20u);\n",
    );
    let _ = writeln!(s, "  let m0 = wid.y * {bm}u; let n0 = wid.x * {bn}u;");
    if sites {
        s.push_str(crate::kernels::conv::SITES_LIVE);
    }
    let _ =
        writeln!(s, "  let sg = min(sgid, {}u); let wm = sg / {wgn}u; let wn = sg % {wgn}u;", subgroups - 1);

    // --- tile-load plans ------------------------------------------------------------------------------------
    let pa = match (va, ta) {
        (true, false) => lines(threads, bk / 4, bm * bk / 4),
        (false, false) => lines(threads, bk, bm * bk),
        (true, true) => lines(threads, bm / 4, bm * bk / 4),
        (false, true) => lines(threads, bm, bm * bk),
    };
    let pb = if conv {
        lines(threads, bn, bk * bn)
    } else {
        match (vb, tb) {
            (true, false) => lines(threads, bn / 4, bn * bk / 4),
            (false, false) => lines(threads, bn, bn * bk),
            (true, true) => lines(threads, bk / 4, bn * bk / 4),
            (false, true) => lines(threads, bk, bn * bk),
        }
    };
    let wb =
        (mode == 3).then(|| crate::kernels::conv::wgrad_b_units(threads, bk, bn, spec.get_u32("WG_RUN", 8)));
    // One B column per invocation for the whole K loop, so its index math is done once.
    match mode {
        // Forward: column = output pixel; B[k = (ci, kh, kw)][j] = X[ci][oh·SH − PH + kh·DH][ow·SW − PW + kw·DW].
        1 => {
            s.push_str("  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u);\n");
            s.push_str(&crate::kernels::conv::forward_column(&pb.inner_expr(), sites));
            if tap_mask {
                s.push_str(&crate::kernels::conv::fwd_tap_mask_setup(kh_n, kw_n, dh_n, dw_n, "wbase"));
            }
        }
        // Input gradient: column = input pixel; B[k = (co, kh, kw)][j] =
        // G[co][(ih + PH − kh·DH) / SH][(iw + PW − kw·DW) / SW] where both divisions are exact and in range, else 0.
        2 if phase => {
            let _ = writeln!(
                s,
                "  let OH = pu(26u); let OW = pu(27u); let WR = pu(33u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let ca = pi(30u) + i32(cj / WR); let cb = pi(31u) + i32(cj % WR);",
                pb.inner_expr()
            );
            if phase_tap_mask {
                s.push_str(&crate::kernels::conv::phase_tap_mask_setup(pth_n, ptw_n, "pbase"));
            }
        }
        2 => {
            let _ = writeln!(
                s,
                "  let IW = pu(25u); let OH = pu(26u); let OW = pu(27u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let cih = i32(cj / IW) + PH; let ciw = i32(cj % IW) + PW;",
                pb.inner_expr()
            );
            if dx_tap_mask {
                s.push_str(&crate::kernels::conv::dx_tap_mask_setup(kh_n, kw_n, dh_n, dw_n, "wbase"));
            }
        }
        // Weight gradient: k = (n, oh, ow) (or `(oh, ow)` per image when `PER_IMG`), column = (ci, kh, kw).
        3 => {
            s.push_str(&crate::kernels::conv::weight_grad_dims(spec));
            if !per_img {
                s.push_str("  let SAN = pu(32u); let SBN = pu(33u);\n");
            }
            let _ = writeln!(s, "  let wl = tid % {0}u; let wq = tid / {0}u;", spec.get_u32("WG_RUN", 8));
            if let Some(wb) = &wb {
                s.push_str(&crate::kernels::conv::wgrad_b_setup(wb, per_img));
            }
        }
        _ => {}
    }
    let a_reg = if va { "vec4<f16>" } else { "f16" };
    let b_reg = if vb && !conv { "vec4<f16>" } else { "f16" };
    for u in 0..pa.count {
        let _ = writeln!(s, "  var ra{u} = {a_reg}(0.0);");
    }
    for u in 0..pb.count {
        let _ = writeln!(s, "  var rb{u} = {b_reg}(0.0);");
    }
    for i in 0..fm {
        for j in 0..fnn {
            let _ = writeln!(s, "  var acc{i}_{j} = coop_mat16x16<f32, C>();");
        }
    }
    if sites {
        let _ = writeln!(s, "  let nt = select(0u, (K - kbeg + {bk}u - 1u) / {bk}u, n0 < live);");
    } else {
        let _ = writeln!(s, "  let nt = (K - kbeg + {bk}u - 1u) / {bk}u;");
    }

    // --- global → registers (ordinary bounds-checked loads) --------------------------------------------------
    let mut load = String::new();
    for u in 0..pa.count {
        let (o, i) = (pa.outer(u), pa.inner_expr());
        let (idx, cond) = match (va, ta) {
            (true, false) if mode == 3 && !per_img => (
                format!(
                    "(offA + ((k0 + {i} * 4u) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + {i} * 4u) % OHOW) / 4u"
                ),
                format!("m0 + {o} < M && k0 + {i} * 4u < K"),
            ),
            (false, false) if mode == 3 && !per_img => (
                format!("offA + ((k0 + {i}) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + {i}) % OHOW"),
                format!("m0 + {o} < M && k0 + {i} < K"),
            ),
            (true, false) => (
                format!("offA / 4u + (m0 + {o}) * (lda / 4u) + k0 / 4u + {i}"),
                format!("m0 + {o} < M && k0 + {i} * 4u < K"),
            ),
            (false, false) => {
                (format!("offA + (m0 + {o}) * lda + k0 + {i}"), format!("m0 + {o} < M && k0 + {i} < K"))
            }
            (true, true) => (
                format!("offA / 4u + (k0 + {o}) * (lda / 4u) + m0 / 4u + {i}"),
                format!("k0 + {o} < K && m0 + {i} * 4u < M"),
            ),
            (false, true) => {
                (format!("offA + (k0 + {o}) * lda + m0 + {i}"), format!("k0 + {o} < K && m0 + {i} < M"))
            }
        };
        let body = if check {
            format!("    if ({cond}) {{ ra{u} = MA[{idx}]; }} else {{ ra{u} = {a_reg}(0.0); }}\n")
        } else {
            format!("    ra{u} = MA[{idx}];\n")
        };
        load.push_str(&pa.guarded(body));
    }
    if let Some(wb) = &wb {
        load.push_str(&crate::kernels::conv::wgrad_b_load(
            wb,
            "MB",
            |u| format!("rb{u}"),
            "",
            "f16(0.0)",
            per_img,
            check,
        ));
    }
    for u in (0..pb.count).filter(|_| mode != 3) {
        let (o, i) = (pb.outer(u), pb.inner_expr());
        let body = if mode == 1 && tap_mask {
            let _ = i;
            crate::kernels::conv::fwd_tap_mask_load(
                "MB",
                "wbase",
                &format!("rb{u}"),
                "",
                "f16(0.0)",
                &o,
                check,
            )
        } else if mode == 1 {
            let _ = i;
            format!(
                "    {{ let k = k0 + {o}; let ci = k / (KH * KW); let r = k % (KH * KW);\n\
                 \x20     let ih = cih0 + i32((r / KW) * DH); let iw = ciw0 + i32((r % KW) * DW);\n\
                 \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                 \x20       rb{u} = MB[offB + (ci * IH + u32(ih)) * IW + u32(iw)];\n\
                 \x20     }} else {{ rb{u} = f16(0.0); }} }}\n"
            )
        } else if phase_tap_mask {
            let _ = i;
            crate::kernels::conv::phase_tap_mask_load(
                "MB",
                "pbase",
                &format!("rb{u}"),
                "",
                "f16(0.0)",
                &o,
                check,
            )
        } else if phase {
            format!(
                "    {{ let k = k0 + {o}; let co = k / (PTH * PTW); let r = k % (PTH * PTW);\n\
                 \x20     let oh = ca - i32(r / PTW); let ow = cb - i32(r % PTW);\n\
                 \x20     var v = f16(0.0);\n\
                 \x20     if (cjok && k < K && oh >= 0 && ow >= 0 && oh < i32(OH) && ow < i32(OW)) {{\n\
                 \x20       v = MB[offB + (co * OH + u32(oh)) * OW + u32(ow)];\n\
                 \x20     }}\n\
                 \x20     rb{u} = v; }}\n"
            )
        } else if dx_tap_mask {
            let _ = i;
            crate::kernels::conv::dx_tap_mask_load(
                "MB",
                "wbase",
                &format!("rb{u}"),
                "",
                "f16(0.0)",
                &o,
                check,
            )
        } else if mode == 2 {
            format!(
                "    {{ let k = k0 + {o}; let co = k / (KH * KW); let r = k % (KH * KW);\n\
                 \x20     let th = cih - i32((r / KW) * DH); let tw = ciw - i32((r % KW) * DW);\n\
                 \x20     var v = f16(0.0);\n\
                 \x20     if (cjok && k < K && th >= 0 && tw >= 0) {{\n\
                 \x20       let uh = u32(th); let uw = u32(tw);\n\
                 \x20       if (uh % SH == 0u && uw % SW == 0u && uh / SH < OH && uw / SW < OW) {{\n\
                 \x20         v = MB[offB + (co * OH + uh / SH) * OW + uw / SW];\n\
                 \x20       }}\n\
                 \x20     }}\n\
                 \x20     rb{u} = v; }}\n"
            )
        } else {
            let (idx, cond) = match (vb, tb) {
                (true, false) => (
                    format!("offB / 4u + (k0 + {o}) * (ldb / 4u) + n0 / 4u + {i}"),
                    format!("k0 + {o} < K && n0 + {i} * 4u < N"),
                ),
                (false, false) => {
                    (format!("offB + (k0 + {o}) * ldb + n0 + {i}"), format!("k0 + {o} < K && n0 + {i} < N"))
                }
                (true, true) => (
                    format!("offB / 4u + (n0 + {o}) * (ldb / 4u) + k0 / 4u + {i}"),
                    format!("n0 + {o} < N && k0 + {i} * 4u < K"),
                ),
                (false, true) => {
                    (format!("offB + (n0 + {o}) * ldb + k0 + {i}"), format!("n0 + {o} < N && k0 + {i} < K"))
                }
            };
            let fetch = match (cat > 1, vb) {
                (true, true) => format!("cat_b(k0 + {o}, n0 / 4u + {i}, b1)"),
                (true, false) => format!("cat_b(k0 + {o}, n0 + {i}, b1)"),
                (false, _) => format!("MB[{idx}]"),
            };
            if check {
                format!("    if ({cond}) {{ rb{u} = {fetch}; }} else {{ rb{u} = {b_reg}(0.0); }}\n")
            } else {
                format!("    rb{u} = {fetch};\n")
            }
        };
        load.push_str(&pb.guarded(body));
    }

    // --- registers → workgroup tiles (row-major `As[m][k]`, `Bs[k][n]`) --------------------------------------
    let store = |buf: u32| -> String {
        let mut st = String::new();
        for u in 0..pa.count {
            let (o, i) = (pa.outer(u), pa.inner_expr());
            let mut body = String::new();
            match (va, ta) {
                (true, false) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    As{buf}[{o} * {sa}u + {i} * 4u + {c}u] = ra{u}[{c}];");
                    }
                }
                (false, false) => {
                    let _ = writeln!(body, "    As{buf}[{o} * {sa}u + {i}] = ra{u};");
                }
                (true, true) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    As{buf}[({i} * 4u + {c}u) * {sa}u + {o}] = ra{u}[{c}];");
                    }
                }
                (false, true) => {
                    let _ = writeln!(body, "    As{buf}[{i} * {sa}u + {o}] = ra{u};");
                }
            }
            st.push_str(&pa.guarded(body));
        }
        for (u, w) in wb.iter().flat_map(|wb| wb.units.iter().map(move |w| (wb.run, w))).enumerate() {
            let (run, w) = w;
            let row = format!("({} * {run}u + wl)", w.kq);
            let _ = writeln!(st, "    Bs{buf}[{row} * {sb}u + {}] = rb{u};", w.jj);
        }
        for u in (0..pb.count).filter(|_| mode != 3) {
            let (o, i) = (pb.outer(u), pb.inner_expr());
            let mut body = String::new();
            match (vb && !conv, tb && !conv) {
                (true, false) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    Bs{buf}[{o} * {sb}u + {i} * 4u + {c}u] = rb{u}[{c}];");
                    }
                }
                (false, false) => {
                    let _ = writeln!(body, "    Bs{buf}[{o} * {sb}u + {i}] = rb{u};");
                }
                (true, true) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    Bs{buf}[({i} * 4u + {c}u) * {sb}u + {o}] = rb{u}[{c}];");
                    }
                }
                (false, true) => {
                    let _ = writeln!(body, "    Bs{buf}[{i} * {sb}u + {o}] = rb{u};");
                }
            }
            st.push_str(&pb.guarded(body));
        }
        st
    };

    // --- tensor-core multiply-accumulate on buffer `buf` (uniform control flow) ------------------------------
    let compute = |buf: u32| -> String {
        let mut c = String::new();
        for kk in 0..bk / 16 {
            c.push_str("    {\n");
            for i in 0..fm {
                let _ = writeln!(
                    c,
                    "      let fa{i} = coopLoadT<coop_mat16x16<f16, A>>(&As{buf}[(wm * {}u + {}u) * {sa}u + {}u], {sa}u);",
                    fm * 16,
                    i * 16,
                    kk * 16
                );
            }
            for j in 0..fnn {
                let _ = writeln!(
                    c,
                    "      let fb{j} = coopLoadT<coop_mat16x16<f16, B>>(&Bs{buf}[{}u + wn * {}u + {}u], {sb}u);",
                    kk * 16 * sb,
                    fnn * 16,
                    j * 16
                );
            }
            for i in 0..fm {
                for j in 0..fnn {
                    let _ = writeln!(c, "      acc{i}_{j} = coopMultiplyAdd(fa{i}, fb{j}, acc{i}_{j});");
                }
            }
            c.push_str("    }\n");
        }
        c
    };

    // --- main loop: two K tiles per iteration so the tile buffers are named statically -----------------------
    let _ = writeln!(s, "  {{\n    let k0 = kbeg;\n{load}  }}");
    let _ = writeln!(s, "  {{\n{}  }}", store(0));
    s.push_str("  workgroupBarrier();\n  var t = 0u;\n  loop {\n");
    for (cur, next) in [(0, 1), (1, 0)] {
        let _ = writeln!(
            s,
            "   if (t >= nt) {{ break; }}\n   {{\n    let more = t + 1u < nt;\n    if (more) {{\n    let k0 = kbeg + (t + 1u) * {bk}u;\n{load}    }}\n{}\
             \x20   if (more) {{\n{}    }}\n    workgroupBarrier();\n   }}\n   t += 1u;",
            compute(cur),
            store(next)
        );
    }
    s.push_str("  }\n\n");

    // --- epilogue: one fragment at a time through the subgroup's scratch slice ---------------------------------
    if subpixel {
        s.push_str(&crate::kernels::conv::subpixel_epilogue_coop(fm, fnn, vc));
        s.push_str("}\n");
        return s;
    }
    s.push_str(
        "  let alpha = pf(7u); let act_alpha = pf(17u);\n\
         \x20 let er = (lane & 31u) >> 1u; let ec = (lane & 1u) * 8u;\n",
    );
    let _ = writeln!(s, "  let cbase = sg * {FRAG}u;");
    for i in 0..fm {
        for j in 0..fnn {
            let last = i + 1 == fm && j + 1 == fnn;
            let _ = writeln!(s, "  coopStoreT(acc{i}_{j}, &Cs[cbase], 16u);");
            s.push_str("  subgroupBarrier();\n  {\n");
            let _ = writeln!(
                s,
                "    let row = m0 + wm * {}u + {}u + er; let col = n0 + wn * {}u + {}u + ec;",
                fm * 16,
                i * 16,
                fnn * 16,
                j * 16
            );
            s.push_str(
                "    let e = cbase + er * 16u + ec;\n\
                 \x20   var v0 = vec4<f32>(Cs[e], Cs[e + 1u], Cs[e + 2u], Cs[e + 3u]) * alpha;\n\
                 \x20   var v1 = vec4<f32>(Cs[e + 4u], Cs[e + 5u], Cs[e + 6u], Cs[e + 7u]) * alpha;\n",
            );
            if check {
                s.push_str("    if (row < M) {\n");
            } else {
                s.push_str("    {\n");
            }
            match bias {
                1 => s.push_str(
                    "      let bv = vec4<f32>(f32(MBIAS[offBias + row]));\n      v0 += bv; v1 += bv;\n",
                ),
                2 => {
                    for (v, base) in [("v0", 0), ("v1", 4)] {
                        let terms: Vec<String> = (0..4)
                            .map(|c| format!("f32(MBIAS[offBias + min(col + {}u, N - 1u)])", base + c))
                            .collect();
                        let _ = writeln!(s, "      {v} += vec4<f32>({});", terms.join(", "));
                    }
                }
                _ => {}
            }
            s.push_str("      v0 = unary4(ACT, v0, act_alpha); v1 = unary4(ACT, v1, act_alpha);\n");
            let at = "offC + row * ldc + col";
            if vc {
                if check {
                    let _ = writeln!(s, "      if (col < N) {{ MC[({at}) / 4u] = vec4<CT>(v0); }}");
                    let _ = writeln!(s, "      if (col + 4u < N) {{ MC[({at}) / 4u + 1u] = vec4<CT>(v1); }}");
                } else {
                    let _ = writeln!(s, "      MC[({at}) / 4u] = vec4<CT>(v0);");
                    let _ = writeln!(s, "      MC[({at}) / 4u + 1u] = vec4<CT>(v1);");
                }
            } else {
                if phase {
                    s.push_str("      var pp = ppix_at(col);\n");
                }
                for c in 0..8 {
                    let (v, x) = if c < 4 { ("v0", c) } else { ("v1", c - 4) };
                    let at = if phase {
                        "offC + row * ldc + ppix_off(pp)".to_string()
                    } else {
                        format!("{at} + {c}u")
                    };
                    if check {
                        let _ = writeln!(s, "      if (col + {c}u < N) {{ MC[{at}] = CT({v}[{x}]); }}");
                    } else {
                        let _ = writeln!(s, "      MC[{at}] = CT({v}[{x}]);");
                    }
                    if phase && c < 7 {
                        s.push_str("      ppix_next(&pp);\n");
                    }
                }
            }
            if last {
                s.push_str("    }\n  }\n");
            } else {
                s.push_str("    }\n  }\n  subgroupBarrier();\n");
            }
        }
    }
    s.push_str("}\n");
    s
}

pub(crate) static COOP: KernelDef = KernelDef {
    name: "coop_gemm",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_coop),
};

/// [`COOP`] as the forward conv at listed output pixels (`CONV_MODE` 1 with `SITES`; binding 5 is the list).
pub(crate) static COOP_SITES: KernelDef = KernelDef {
    name: "coop_gemm_sites",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Generated(gen_coop),
};

/// [`COOP`] with `B` in up to [`crate::kernels::gemm_cat::MAX_CAT_PARTS`] buffers (`CAT`).
pub(crate) static COOP_CAT: KernelDef = KernelDef {
    name: "coop_gemm_cat",
    storage: &[
        Access::Read,
        Access::Read,
        Access::Read,
        Access::ReadWrite,
        Access::Read,
        Access::Read,
        Access::Read,
    ],
    source: Source::Generated(gen_coop),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epilogue_uses_subgroup_barriers_without_trailing_barrier() {
        for cfg in [
            CoopConfig::C32X32,
            CoopConfig::C64X64,
            CoopConfig::C64X128,
            CoopConfig::C128X128,
            CoopConfig::C32X128,
            CoopConfig::C16X256,
        ] {
            let frags = (cfg.fm() * cfg.fnn()) as usize;
            for subpixel in [false, true] {
                let mut spec = Spec::new()
                    .u32("BM", cfg.bm)
                    .u32("BN", cfg.bn)
                    .u32("BK", cfg.bk)
                    .u32("WGM", cfg.wgm)
                    .u32("WGN", cfg.wgn)
                    .u32("CONV_MODE", u32::from(subpixel))
                    .bool("SUBPIXEL", subpixel);
                if subpixel {
                    spec = spec.u32("KH", 2).u32("KW", 2);
                }
                let src = gen_coop(&spec);
                let epilogue = src.split_once("let cbase = sg * ").expect("epilogue marker").1;
                assert_eq!(
                    epilogue.matches("workgroupBarrier()").count(),
                    0,
                    "{cfg:?} subpixel={subpixel}: unexpected workgroupBarrier in epilogue"
                );
                assert_eq!(
                    epilogue.matches("subgroupBarrier()").count(),
                    2 * frags - 1,
                    "{cfg:?} subpixel={subpixel}: expected 2*frags-1 subgroupBarriers"
                );
            }
        }
    }
}
