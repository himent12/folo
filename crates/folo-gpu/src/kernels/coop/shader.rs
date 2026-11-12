//! The WGSL generator of the tensor-core GEMM / implicit-GEMM conv kernel ([`COOP`]): tile-load plans, the
//! double-buffered K loop on cooperative matrices and the fragment-by-fragment epilogue.

use std::fmt::Write as _;

use super::{CoopConfig, FRAG, PAD, SUBGROUPS, THREADS};
use crate::kernels::UNARY_FN;
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// One operand-tile load plan: element `u` of this invocation covers inner unit `tid % inner` of line
/// `tid / inner + u · (256 / inner)`; `guard` is set when fewer than 256 units exist.
struct Lines {
    inner: u32,
    count: u32,
    guard: Option<u32>,
}

fn lines(inner: u32, units: u32) -> Lines {
    Lines { inner, count: units.div_ceil(THREADS), guard: (units < THREADS).then_some(units) }
}

impl Lines {
    fn outer(&self, u: u32) -> String {
        format!("(tid / {}u + {}u)", self.inner, u * (THREADS / self.inner))
    }

    fn inner_expr(&self) -> String {
        format!("(tid % {}u)", self.inner)
    }

    /// Wraps `body` in the "fewer than 256 units" guard.
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
    let split = spec.get_bool("SPLIT");
    let bias = spec.get_u32("BIAS_KIND", 0);
    // `CAT` > 1: `B` is that many row blocks in separate buffers (`crate::kernels::gemm_cat`).
    let cat = spec.get_u32("CAT", 1);
    let ct = if spec.get_bool("OUT_F16") { "f16" } else { "f32" };
    let (a_len, b_len) = (bm * sa, bk * sb);

    // Condition 2 of `experimental_opt_in`: every fragment lies inside its workgroup array. The largest
    // subgroup index is 7 (clamped in the shader), so these are the furthest fragments.
    let a_last = ((cfg.wgm - 1) * fm * 16 + (fm - 1) * 16) * sa + (bk - 16);
    assert!(a_last + 15 * sa + 16 <= a_len, "A fragment out of bounds");
    let b_last = (bk - 16) * sb + (wgn - 1) * fnn * 16 + (fnn - 1) * 16;
    assert!(b_last + 15 * sb + 16 <= b_len, "B fragment out of bounds");
    const { assert!((SUBGROUPS - 1) * FRAG + 15 * 16 + 16 <= SUBGROUPS * FRAG, "scratch fragment out of bounds") };

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
    for b in 0..2 {
        let _ = writeln!(s, "var<workgroup> As{b}: array<f16, {a_len}>;");
        let _ = writeln!(s, "var<workgroup> Bs{b}: array<f16, {b_len}>;");
    }
    let _ = writeln!(s, "var<workgroup> Cs: array<f32, {}>;", SUBGROUPS * FRAG);
    let _ = writeln!(s, "@compute @workgroup_size({THREADS})");
    s.push_str(
        "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32,\n\
         \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
         \x20 let M = pu(0u); let N = pu(1u);\n\
         \x20 let lda = pu(3u); let ldb = pu(4u); let ldc = pu(5u);\n",
    );
    if split {
        // wid.z = batch index * S + chunk; chunk `sidx` covers k in [kbeg, kbeg + KS) and writes its own C slab.
        // All of it derives from `wid` and uniforms, so the K loop stays workgroup-uniform.
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
    // The clamp keeps the fragment offsets in bounds whatever the driver reports (safety condition 2).
    let _ =
        writeln!(s, "  let sg = min(sgid, {}u); let wm = sg / {wgn}u; let wn = sg % {wgn}u;", SUBGROUPS - 1);

    // --- tile-load plans ------------------------------------------------------------------------------------
    let pa = match (va, ta) {
        (true, false) => lines(bk / 4, bm * bk / 4),
        (false, false) => lines(bk, bm * bk),
        (true, true) => lines(bm / 4, bm * bk / 4),
        (false, true) => lines(bm, bm * bk),
    };
    let pb = if conv {
        lines(bn, bk * bn)
    } else {
        match (vb, tb) {
            (true, false) => lines(bn / 4, bn * bk / 4),
            (false, false) => lines(bn, bn * bk),
            (true, true) => lines(bk / 4, bn * bk / 4),
            (false, true) => lines(bk, bn * bk),
        }
    };
    // One B column per invocation for the whole K loop, so its index math is done once.
    match mode {
        // Forward: column = output pixel; B[k = (ci, kh, kw)][j] = X[ci][oh·SH − PH + kh·DH][ow·SW − PW + kw·DW].
        1 => {
            let _ = writeln!(
                s,
                "  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let coh = cj / OW; let cow = cj % OW;\n\
                 \x20 let cih0 = i32(coh * SH) - PH; let ciw0 = i32(cow * SW) - PW;",
                pb.inner_expr()
            );
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
        }
        2 => {
            let _ = writeln!(
                s,
                "  let IW = pu(25u); let OH = pu(26u); let OW = pu(27u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let cih = i32(cj / IW) + PH; let ciw = i32(cj % IW) + PW;",
                pb.inner_expr()
            );
        }
        // Weight gradient: k = (n, oh, ow) runs over the whole batch, column = (ci, kh, kw);
        // A[m][k] = G[n][m][oh, ow], B[k][j] = X[n][ci][oh·SH − PH + kh·DH][ow·SW − PW + kw·DW].
        3 => {
            let _ = writeln!(
                s,
                "{}  let SAN = pu(32u); let SBN = pu(33u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let cci = cj / (KH * KW); let cr = cj % (KH * KW);\n\
                 \x20 let cdh = i32((cr / KW) * DH) - PH; let cdw = i32((cr % KW) * DW) - PW;",
                crate::kernels::conv::weight_grad_dims(spec),
                pb.inner_expr()
            );
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
    let _ = writeln!(s, "  let nt = (K - kbeg + {bk}u - 1u) / {bk}u;");

    // --- global → registers (ordinary bounds-checked loads) --------------------------------------------------
    let mut load = String::new();
    for u in 0..pa.count {
        let (o, i) = (pa.outer(u), pa.inner_expr());
        let (idx, cond) = match (va, ta) {
            // Weight gradient: k = (n, pixel) folds the batch into K and the gradient rows are `OHOW` long (a
            // multiple of 4 for the vec4 form, so four consecutive k stay in one image).
            (true, false) if mode == 3 => (
                format!(
                    "(offA + ((k0 + {i} * 4u) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + {i} * 4u) % OHOW) / 4u"
                ),
                format!("m0 + {o} < M && k0 + {i} * 4u < K"),
            ),
            (false, false) if mode == 3 => (
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
    for u in 0..pb.count {
        let (o, i) = (pb.outer(u), pb.inner_expr());
        let body = if mode == 1 {
            let _ = i;
            format!(
                "    {{ let k = k0 + {o}; let ci = k / (KH * KW); let r = k % (KH * KW);\n\
                 \x20     let ih = cih0 + i32((r / KW) * DH); let iw = ciw0 + i32((r % KW) * DW);\n\
                 \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                 \x20       rb{u} = MB[offB + (ci * IH + u32(ih)) * IW + u32(iw)];\n\
                 \x20     }} else {{ rb{u} = f16(0.0); }} }}\n"
            )
        } else if phase {
            // k = (co, th, tw), column = phase pixel (a, b): output pixel (BH + a - th, BW + b - tw).
            format!(
                "    {{ let k = k0 + {o}; let co = k / (PTH * PTW); let r = k % (PTH * PTW);\n\
                 \x20     let oh = ca - i32(r / PTW); let ow = cb - i32(r % PTW);\n\
                 \x20     var v = f16(0.0);\n\
                 \x20     if (cjok && k < K && oh >= 0 && ow >= 0 && oh < i32(OH) && ow < i32(OW)) {{\n\
                 \x20       v = MB[offB + (co * OH + u32(oh)) * OW + u32(ow)];\n\
                 \x20     }}\n\
                 \x20     rb{u} = v; }}\n"
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
        } else if mode == 3 {
            format!(
                "    {{ let k = k0 + {o}; let nn = k / OHOW; let r = k % OHOW;\n\
                 \x20     let ih = i32((r / OW) * SH) + cdh; let iw = i32((r % OW) * SW) + cdw;\n\
                 \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                 \x20       rb{u} = MB[offB + nn * SBN + (cci * IH + u32(ih)) * IW + u32(iw)];\n\
                 \x20     }} else {{ rb{u} = f16(0.0); }} }}\n"
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
        for u in 0..pb.count {
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
    // The tensor-core ops must be in uniform control flow, and naga treats anything derived from the loop
    // counter as non-uniform. A `break` does not taint the statements after it, so each half starts with a
    // `break` check instead of being wrapped in an `if`.
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
    s.push_str(
        "  let alpha = pf(7u); let act_alpha = pf(17u);\n\
         \x20 let er = (lane & 31u) >> 1u; let ec = (lane & 1u) * 8u;\n",
    );
    let _ = writeln!(s, "  let cbase = sg * {FRAG}u;");
    for i in 0..fm {
        for j in 0..fnn {
            let _ = writeln!(s, "  coopStoreT(acc{i}_{j}, &Cs[cbase], 16u);");
            s.push_str("  workgroupBarrier();\n  {\n");
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
            s.push_str("    }\n  }\n  workgroupBarrier();\n");
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
