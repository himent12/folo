// Detection loss, box terms (docs/MODEL.md §7.4) for the assigner's positives, one thread per pick slot:
// pd = anchor ∓ Σ_i i · softmax(box logits) (grid units), L_box = t · (1 − CIoU(pd, gt / stride)) and the DFL
// cross-entropy at the two bins around each side's clamped target distance, weighted t / 4. Slots that are not
// positive, or have t = 0, contribute nothing.
// MODE 0 writes OUT[4s] = box term, OUT[4s + 1] = Σ_sides (w_l · log p_j + w_r · log p_{j+1}) · t / 4 (the DFL CE
// term before its sign), and OUT[4s + 2..=3] = exponent-split `(mant, exp)` of Σ_sides wt · L_ot (unscaled by
// dfl_ot_weight so det_combine reduces and normalises by tss before multiplying by dfl_ot_weight · dfl_gain).
// MODE 1 writes the gradient of the weighted loss into OUT = d box_raw ([B, 4R, A], zeroed first; each anchor has at
// most one positive, so writes do not collide): g = GRAD[0], with tss = STATS[7], S_box = STATS[8], S_dfl = STATS[10],
// and B = STATS[11] from the combine kernel. α of the CIoU is a constant, as in the CPU loss.
// Spec: R, K, MODE. Params: 0 = slots, 1 = A, 2 = G, 3 = gt box word offset, 4 = top clamp (R − 1 − 0.01),
// 5 = nwd_weight (0 = pure CIoU), 6 = box_loss_kind (0 = CIoU, 1 = R-PIoU), 7 = rpiou_beta, 8 = rpiou_kappa,
// 9 = rpiou_delta, 10 = dfl_ot_weight, 11 = box_gain, 12 = dfl_gain.
@group(0) @binding(1) var<storage, read> BOX: array<f32>;
@group(0) @binding(2) var<storage, read> META: array<u32>;
@group(0) @binding(3) var<storage, read> PICK: array<u32>;
@group(0) @binding(4) var<storage, read> SLOT: array<u32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(6) var<storage, read> STATS: array<f32>;
@group(0) @binding(7) var<storage, read> GRAD: array<f32>;

// Exact 2^-24 guard factor for comparing huge opposite-sign logits (`m_e * OT_SHIFT - m_r * OT_SHIFT`) without
// forming intermediate `±inf` in `m_e - m_r`.
const OT_SHIFT: f32 = 5.960464477539063e-8; // 2^-24

// Overflow/underflow-safe `exp(a - b)` for `a <= b` when adding into a `>= 1.0` sum (avoids `-inf` in `a - b`).
fn safe_exp_diff(a: f32, b: f32) -> f32 {
    if (a * OT_SHIFT - b * OT_SHIFT <= -100.0 * OT_SHIFT) { return 0.0; }
    return fexp(a - b);
}

// Exponent-split `exp(a - b)` for `a <= b` down to `-700.0` (avoids `-inf` in `a - b`).
fn exp_diff_split(a: f32, b: f32) -> vec2<f32> {
    if (a * OT_SHIFT - b * OT_SHIFT <= -700.0 * OT_SHIFT) { return vec2<f32>(0.0, 0.0); }
    return exp_split_neg(a - b);
}

// Exponent-split `softplus((m_e - m_r) + (log_ge1(a_e) - log_ge1(a_r)))` without f32 overflow, small-excluded-mass
// cancellation, or subnormal underflow down to `delta = -700`.
fn ot_softplus_split(m_e: f32, a_e: f32, m_r: f32, a_r: f32) -> vec2<f32> {
    let dm_s = m_e * OT_SHIFT - m_r * OT_SHIFT;
    let dl = log_ge1(a_e) - log_ge1(a_r);
    if (dm_s > 100.0 * OT_SHIFT) {
        let n = norm_f32(dm_s + dl * OT_SHIFT);
        return vec2<f32>(n.x, n.y + 24.0);
    }
    if (dm_s < -700.0 * OT_SHIFT) {
        return vec2<f32>(0.0, 0.0);
    }
    let delta = (m_e - m_r) + dl;
    if (delta >= 0.0) {
        return norm_f32(delta + log1p_unit(safe_exp_neg(-delta)));
    }
    if (delta > -20.0) {
        return norm_f32(log1p_unit(fexp(delta)));
    }
    // For delta <= -20, ln(1 + e^delta) == e^delta to > 28 bits of precision.
    return exp_split_neg(delta);
}

// Exponent-split `sigmoid((m_e - m_r) + (log_ge1(a_e) - log_ge1(a_r)))` down to `delta = -700`.
fn ot_sigmoid_split(m_e: f32, a_e: f32, m_r: f32, a_r: f32) -> vec2<f32> {
    let dm_s = m_e * OT_SHIFT - m_r * OT_SHIFT;
    if (dm_s > 100.0 * OT_SHIFT) { return vec2<f32>(1.0, 0.0); }
    if (dm_s < -700.0 * OT_SHIFT) { return vec2<f32>(0.0, 0.0); }
    let delta = (m_e - m_r) + (log_ge1(a_e) - log_ge1(a_r));
    if (delta >= 0.0) {
        return norm_f32(1.0 / (1.0 + safe_exp_neg(-delta)));
    }
    if (delta > -20.0) {
        let e = fexp(delta);
        return norm_f32(e / (1.0 + e));
    }
    return exp_split_neg(delta);
}

// Exponent-split R-PIoU edge derivative `sign_s * (u / denom) * inv1^2 * (q_m * 2^e_q) * g_half` as normalized `(m, e)`.
// Kept in split form until multiplied by the DFL softmax Jacobian so extreme `B * box_gain * kappa / tss` never
// overflows an intermediate coordinate gradient when the final logit gradient fits in f32.
fn rpiou_edge_split(sign_s: f32, u: f32, denom: f32, in1: vec2<f32>, q_m: f32, e_q: i32, gn: vec2<f32>) -> vec2<f32> {
    if (u == 0.0 || gn.x == 0.0) { return vec2<f32>(0.0, 0.0); }
    let un = norm_f32(u / denom);
    let qn = norm_f32(q_m);
    let base = mul_split(mul_split(un, mul_split(in1, in1)), vec2<f32>(qn.x, qn.y + f32(e_q)));
    let res = mul_split(base, gn);
    return vec2<f32>(sign_s * res.x, res.y);
}

// Gradient share of `a` in min(a, b) / max(a, b): folo-core gives ties half to each side.
fn w_min(a: f32, b: f32) -> f32 { return select(select(0.0, 0.5, a == b), 1.0, a < b); }
fn w_max(a: f32, b: f32) -> f32 { return select(select(0.0, 0.5, a == b), 1.0, a > b); }
// Overflow/underflow-safe exp(d) for d <= 0 (d = -inf maps to 0.0 without NaN).
fn safe_exp_neg(d: f32) -> f32 { return select(0.0, fexp(d), d > -100.0); }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let s = linear_id(wid, nwg, lid, 64u);
    if (s >= pu(0u)) { return; }
    let t = bitcast<f32>(SLOT[2u * s]);
    let pos = SLOT[2u * s + 1u] == 1u;
    if (MODE == 0u) {
        OUT[4u * s] = 0.0;
        OUT[4u * s + 1u] = 0.0;
        OUT[4u * s + 2u] = 0.0;
        OUT[4u * s + 3u] = 0.0;
    }
    if (!pos || t == 0.0) { return; }
    let na = pu(1u);
    let ng = pu(2u);
    let gi = s / K;
    let b = gi / ng;
    let a = PICK[3u * s];
    let ax = bitcast<f32>(META[3u * a]);
    let ay = bitcast<f32>(META[3u * a + 1u]);
    let inv = 1.0 / bitcast<f32>(META[3u * a + 2u]);
    let o = pu(3u) + 4u * gi;
    let tb = vec4<f32>(bitcast<f32>(META[o]), bitcast<f32>(META[o + 1u]), bitcast<f32>(META[o + 2u]),
                       bitcast<f32>(META[o + 3u])) * inv;
    let base = b * 4u * R * na + a;
    let wt = t * 0.25;
    let hi = pf(4u);
    let ltrb = vec4<f32>(ax - tb.x, ay - tb.y, tb.z - ax, tb.w - ay);
    let box_kind = pu(6u);
    let rpiou_beta = pf(7u);
    let rpiou_kappa = pf(8u);
    let rpiou_delta = pf(9u);
    let dfl_ot_w = pf(10u);
    let box_gain = pf(11u);
    let dfl_gain = pf(12u);

    // Softmax per side (p) and log-softmax (lp), kept for the backward.
    var p: array<f32, 4u * R>;
    var lp: array<f32, 4u * R>;
    var d: vec4<f32>;
    var dfl = 0.0;
    var ot_sum = vec2<f32>(0.0, 0.0);
    var jl: array<u32, 4>;
    var wl: vec4<f32>;
    var wr: vec4<f32>;
    for (var side = 0u; side < 4u; side++) {
        var m = -3.402823466e+38;
        for (var k = 0u; k < R; k++) {
            let x = BOX[base + (side * R + k) * na];
            p[side * R + k] = x;
            m = max(m, x);
        }
        var z = 0.0;
        for (var k = 0u; k < R; k++) {
            let x = p[side * R + k] - m;
            lp[side * R + k] = x;
            let e = fexp(x);
            p[side * R + k] = e;
            z = z + e;
        }
        let lz = log_ge1(z);
        var dist = 0.0;
        for (var k = 0u; k < R; k++) {
            let q = p[side * R + k] / z;
            p[side * R + k] = q;
            lp[side * R + k] = lp[side * R + k] - lz;
            dist = dist + q * f32(k);
        }
        d[side] = dist;
        // The DFL target bins (assign::bbox2dist_clamped and dfl_bins).
        let dt = clamp(ltrb[side], 0.0, hi);
        let j = floor(max(dt, 0.0));
        let l = j + 1.0 - dt;
        let ju = u32(j);
        jl[side] = ju;
        wl[side] = l * wt;
        wr[side] = (1.0 - l) * wt;
        dfl = dfl + lp[side * R + ju] * wl[side] + lp[side * R + ju + 1u] * wr[side];
        if (MODE == 0u && dfl_ot_w > 0.0 && R > 2u) {
            var pref_m: array<f32, R>;
            var pref_a: array<f32, R>;
            var pm = BOX[base + (side * R) * na];
            var pa = 1.0;
            pref_m[0] = pm;
            pref_a[0] = pa;
            for (var k = 1u; k + 1u < R; k++) {
                let x = BOX[base + (side * R + k) * na];
                if (x > pm) {
                    pa = pa * safe_exp_diff(pm, x) + 1.0;
                    pm = x;
                } else {
                    pa = pa + safe_exp_diff(x, pm);
                }
                pref_m[k] = pm;
                pref_a[k] = pa;
            }
            var l_tail = vec2<f32>(0.0, 0.0);
            var sm = BOX[base + (side * R + (R - 1u)) * na];
            var sa = 1.0;
            for (var step = 0u; step + 1u < R; step++) {
                let k = (R - 2u) - step;
                if (step > 0u) {
                    let x = BOX[base + (side * R + k + 1u) * na];
                    if (x > sm) {
                        sa = sa * safe_exp_diff(sm, x) + 1.0;
                        sm = x;
                    } else {
                        sa = sa + safe_exp_diff(x, sm);
                    }
                }
                if (k < ju) {
                    l_tail = add_split(l_tail, ot_softplus_split(pref_m[k], pref_a[k], sm, sa));
                } else if (k > ju) {
                    l_tail = add_split(l_tail, ot_softplus_split(sm, sa, pref_m[k], pref_a[k]));
                }
            }
            ot_sum = add_split(ot_sum, mul_split(norm_f32(wt), l_tail));
        }
    }
    let pd = vec4<f32>(ax - d[0], ay - d[1], ax + d[2], ay + d[3]);
    let nwd_w = pf(5u);
    // tb is in stride units (inv = 1 / stride); convert GT size to pixels for lambda and C = 12.8px to stride units.
    let lam = nwd_lambda((tb.z - tb.x) / inv, (tb.w - tb.y) / inv, nwd_w);
    let c_s = 12.8 * inv;
    if (MODE == 0u) {
        var l_base = 0.0;
        if (box_kind == 1u) {
            l_base = rpiou_loss_f(pd, tb, rpiou_beta, rpiou_kappa, rpiou_delta);
        } else {
            l_base = 1.0 - ciou_eps(pd, tb);
        }
        var l_box = l_base;
        if (lam > 0.0) {
            l_box = (1.0 - lam) * l_base + lam * (1.0 - nwd_box(pd, tb, c_s));
        }
        OUT[4u * s] = l_box * t;
        OUT[4u * s + 1u] = dfl;
        OUT[4u * s + 2u] = ot_sum.x;
        OUT[4u * s + 3u] = ot_sum.y;
        return;
    }

    // Backward: R-PIoU or CIoU (§7.2, α constant), then through the expectation.
    let g = GRAD[0];
    let tss = STATS[7];
    let nb = STATS[11];
    let g_base = t * STATS[8] * g * (1.0 - lam);
    let g_base_sp = div_split(
        mul_split(
            mul_split(mul_split(norm_f32(t), norm_f32(nb)), mul_split(norm_f32(box_gain), norm_f32(g))),
            norm_f32(1.0 - lam),
        ),
        norm_f32(tss),
    );
    var gx1 = 0.0;
    var gy1 = 0.0;
    var gx2 = 0.0;
    var gy2 = 0.0;
    var fiou_d = vec4<f32>(0.0);
    var edge_m = vec4<f32>(0.0);
    var edge_e = vec4<f32>(0.0);
    if (box_kind == 1u) {
        let eps = 1e-7;
        let w1 = max(pd.z - pd.x, 0.0);
        let h1 = max(pd.w - pd.y, 0.0);
        let w2 = max(tb.z - tb.x, 0.0);
        let h2 = max(tb.w - tb.y, 0.0);
        let ix = min(pd.z, tb.z) - max(pd.x, tb.x);
        let iy = min(pd.w, tb.w) - max(pd.y, tb.y);
        let iw = max(ix, 0.0);
        let ih = max(iy, 0.0);
        let inter = iw * ih;
        let uni = w1 * h1 + w2 * h2 - inter + eps;
        let iou = inter / uni;
        // Keep focal-IoU derivatives unscaled by `g_base_sp` until multiplied by the DFL softmax Jacobian so extreme
        // `B * box_gain / tss` values cannot overflow before the final logit gradient.
        let g_u = -(1.0 - rpiou_beta + 2.0 * rpiou_beta * iou);
        let g_inter = g_u * (1.0 / uni + inter / (uni * uni));
        let g_uni = -g_u * inter / (uni * uni);
        let g_w1 = select(0.0, g_uni * h1, pd.z > pd.x);
        let g_h1 = select(0.0, g_uni * w1, pd.w > pd.y);
        let g_ix = select(0.0, g_inter * ih, ix > 0.0);
        let g_iy = select(0.0, g_inter * iw, iy > 0.0);
        fiou_d = vec4<f32>(
            g_w1 + g_ix * w_max(pd.x, tb.x),
            g_h1 + g_iy * w_max(pd.y, tb.y),
            g_w1 + g_ix * w_min(pd.z, tb.z),
            g_h1 + g_iy * w_min(pd.w, tb.w),
        );
        let rkn = norm_f32(sqrt(rpiou_kappa));
        let sx = max(w2, rpiou_delta);
        let sy = max(h2, rpiou_delta);
        let sxn = norm_f32(sx);
        let syn = norm_f32(sy);
        let wx = w2 / sx;
        let dx_f = rpiou_delta / sx;
        let denom_x = wx * wx + dx_f * dx_f;
        let hy = h2 / sy;
        let dy_f = rpiou_delta / sy;
        let denom_y = hy * hy + dy_f * dy_f;
        let q_xm = rkn.x / sxn.x;
        let e_qx = i32(rkn.y) - i32(sxn.y);
        let q_ym = rkn.x / syn.x;
        let e_qy = i32(rkn.y) - i32(syn.y);
        let ux1 = scaled_edge_u(pd.x - tb.x, q_xm, e_qx);
        let ux2 = scaled_edge_u(pd.z - tb.z, q_xm, e_qx);
        let uy1 = scaled_edge_u(pd.y - tb.y, q_ym, e_qy);
        let uy2 = scaled_edge_u(pd.w - tb.w, q_ym, e_qy);
        let ke2_raw = 0.25 * ((ux1 * ux1 + ux2 * ux2) / denom_x + (uy1 * uy1 + uy2 * uy2) / denom_y);
        if (is_finite_f(ke2_raw) && ke2_raw < 1e16) {
            let in1 = norm_f32(1.0 / (1.0 + ke2_raw));
            let gn = select(vec2<f32>(0.0, 0.0), vec2<f32>(g_base_sp.x, g_base_sp.y - 1.0), g_base_sp.x != 0.0);
            let s0 = rpiou_edge_split(-1.0, ux1, denom_x, in1, q_xm, e_qx, gn);
            let s1 = rpiou_edge_split(-1.0, uy1, denom_y, in1, q_ym, e_qy, gn);
            let s2 = rpiou_edge_split(1.0, ux2, denom_x, in1, q_xm, e_qx, gn);
            let s3 = rpiou_edge_split(1.0, uy2, denom_y, in1, q_ym, e_qy, gn);
            edge_m = vec4<f32>(s0.x, s1.x, s2.x, s3.x);
            edge_e = vec4<f32>(s0.y, s1.y, s2.y, s3.y);
        }
    } else {
        let eps = 1e-7;
        let w1 = pd.z - pd.x;
        let h1 = pd.w - pd.y + eps;
        let w2 = tb.z - tb.x;
        let h2 = tb.w - tb.y + eps;
        let ix = min(pd.z, tb.z) - max(pd.x, tb.x);
        let iy = min(pd.w, tb.w) - max(pd.y, tb.y);
        let iw = max(ix, 0.0);
        let ih = max(iy, 0.0);
        let inter = iw * ih;
        let uni = w1 * h1 + w2 * h2 - inter + eps;
        let iou = inter / uni;
        let cw = max(pd.z, tb.z) - min(pd.x, tb.x);
        let ch = max(pd.w, tb.w) - min(pd.y, tb.y);
        let c2 = cw * cw + ch * ch + eps;
        let dx = tb.x + tb.z - pd.x - pd.z;
        let dy = tb.y + tb.w - pd.y - pd.w;
        let rho2 = (dx * dx + dy * dy) * 0.25;
        let r1 = w1 / h1;
        let dd = atan_wh(w2, h2) - atan_wh(w1, h1);
        let v = dd * dd * V_K;
        let alpha = v / (v - iou + (1.0 + eps));

        let gc = -g_base;
        let g_rho2 = -gc / c2;
        let g_c2 = gc * rho2 / (c2 * c2);
        let g_v = -gc * alpha;
        let g_inter = gc * (1.0 / uni + inter / (uni * uni));
        let g_uni = -gc * inter / (uni * uni);
        let g_r1 = -g_v * V_K * 2.0 * dd / (1.0 + r1 * r1);
        let g_w1 = g_uni * h1 + g_r1 / h1;
        let g_h1 = g_uni * w1 - g_r1 * r1 / h1;
        let g_ix = select(0.0, g_inter * ih, ix > 0.0);
        let g_iy = select(0.0, g_inter * iw, iy > 0.0);
        let g_cw = g_c2 * 2.0 * cw;
        let g_ch = g_c2 * 2.0 * ch;
        let g_dx = g_rho2 * 0.5 * dx;
        let g_dy = g_rho2 * 0.5 * dy;
        gx1 = -g_w1 - g_ix * w_max(pd.x, tb.x) - g_cw * w_min(pd.x, tb.x) - g_dx;
        gy1 = -g_h1 - g_iy * w_max(pd.y, tb.y) - g_ch * w_min(pd.y, tb.y) - g_dy;
        gx2 = g_w1 + g_ix * w_min(pd.z, tb.z) + g_cw * w_max(pd.z, tb.z) - g_dx;
        gy2 = g_h1 + g_iy * w_min(pd.w, tb.w) + g_ch * w_max(pd.w, tb.w) - g_dy;
    }
    if (lam > 0.0) {
        let dcx = 0.5 * (pd.x + pd.z - tb.x - tb.z);
        let dcy = 0.5 * (pd.y + pd.w - tb.y - tb.w);
        let dw = (pd.z - pd.x) - (tb.z - tb.x);
        let dh = (pd.w - pd.y) - (tb.w - tb.y);
        let dist = sqrt(dcx * dcx + dcy * dcy + 0.25 * (dw * dw + dh * dh) + 1e-6);
        let nwd = fexp(-dist / c_s);
        let gn = t * STATS[8] * g * lam * nwd / (2.0 * c_s * dist);
        gx1 = gx1 + gn * (dcx - 0.5 * dw);
        gy1 = gy1 + gn * (dcy - 0.5 * dh);
        gx2 = gx2 + gn * (dcx + 0.5 * dw);
        gy2 = gy2 + gn * (dcy + 0.5 * dh);
    }
    let g_d = vec4<f32>(-gx1, -gy1, gx2, gy2);
    // dL/dlogit for L = −S_dfl · Σ (w_l · lp_j + w_r · lp_{j+1}) + S_dfl · OT_tail.
    let sd = STATS[10] * g;
    let use_ot = dfl_ot_w > 0.0 && R > 2u;
    let sd_sp = div_split(
        mul_split(mul_split(norm_f32(nb), norm_f32(dfl_gain)), norm_f32(g)),
        norm_f32(tss),
    );
    let ot_scale = mul_split(mul_split(sd_sp, norm_f32(dfl_ot_w)), norm_f32(wt));
    for (var side = 0u; side < 4u; side++) {
        let wsum = wl[side] + wr[side];
        let ju = jl[side];
        var m_ret: array<f32, R>;
        var ret_sp: array<vec2<f32>, R>;
        var m_exc: array<f32, R>;
        var exc_sp: array<vec2<f32>, R>;
        if (use_ot) {
            // Prefix `0..=k` in `pref_m/pref_a` (left-tail excluded for k < ju, right-tail retained for k > ju).
            var pref_m: array<f32, R>;
            var pref_a: array<f32, R>;
            var pm = BOX[base + (side * R) * na];
            var pa = 1.0;
            pref_m[0] = pm;
            pref_a[0] = pa;
            for (var k = 1u; k + 1u < R; k++) {
                let x = BOX[base + (side * R + k) * na];
                if (x > pm) {
                    pa = pa * safe_exp_diff(pm, x) + 1.0;
                    pm = x;
                } else {
                    pa = pa + safe_exp_diff(x, pm);
                }
                pref_m[k] = pm;
                pref_a[k] = pa;
            }
            // Suffix `(k+1)..R` (left-tail retained for k < ju, right-tail excluded for k > ju).
            var sm = BOX[base + (side * R + (R - 1u)) * na];
            var sa = 1.0;
            for (var step = 0u; step + 1u < R; step++) {
                let k = (R - 2u) - step;
                if (step > 0u) {
                    let x = BOX[base + (side * R + k + 1u) * na];
                    if (x > sm) {
                        sa = sa * safe_exp_diff(sm, x) + 1.0;
                        sm = x;
                    } else {
                        sa = sa + safe_exp_diff(x, sm);
                    }
                }
                if (k < ju) {
                    let sig = ot_sigmoid_split(pref_m[k], pref_a[k], sm, sa);
                    m_exc[k] = pref_m[k];
                    exc_sp[k] = div_split(sig, norm_f32(pref_a[k]));
                    m_ret[k] = sm;
                    ret_sp[k] = div_split(sig, norm_f32(sa));
                } else if (k > ju) {
                    let sig = ot_sigmoid_split(sm, sa, pref_m[k], pref_a[k]);
                    m_ret[k] = pref_m[k];
                    ret_sp[k] = div_split(sig, norm_f32(pref_a[k]));
                    m_exc[k] = sm;
                    exc_sp[k] = div_split(sig, norm_f32(sa));
                }
            }
            if (ju > 0u) {
                // Left-tail retained: prefix sum 0..ju; left-tail excluded: suffix sum ju-1 down to 0.
                for (var r = 1u; r < ju; r++) {
                    ret_sp[r] = add_split(mul_split(ret_sp[r - 1u], exp_diff_split(m_ret[r], m_ret[r - 1u])), ret_sp[r]);
                }
                var r = ju - 1u;
                while (r > 0u) {
                    r = r - 1u;
                    exc_sp[r] = add_split(mul_split(exc_sp[r + 1u], exp_diff_split(m_exc[r], m_exc[r + 1u])), exc_sp[r]);
                }
            }
            if (ju + 2u < R) {
                // Right-tail retained: suffix sum R-2 down to ju+1; right-tail excluded: prefix sum ju+1..R-1.
                var r = R - 2u;
                while (r > ju + 1u) {
                    r = r - 1u;
                    ret_sp[r] = add_split(mul_split(ret_sp[r + 1u], exp_diff_split(m_ret[r], m_ret[r + 1u])), ret_sp[r]);
                }
                for (var t = ju + 2u; t + 1u < R; t++) {
                    exc_sp[t] = add_split(mul_split(exc_sp[t - 1u], exp_diff_split(m_exc[t], m_exc[t - 1u])), exc_sp[t]);
                }
            }
        }
        for (var k = 0u; k < R; k++) {
            let q = p[side * R + k];
            let jac = q * (f32(k) - d[side]);
            var gk = g_d[side] * jac + sd * wsum * q;
            if (fiou_d[side] != 0.0 && jac != 0.0 && g_base_sp.x != 0.0) {
                gk = gk + split_to_f32(mul_split(mul_split(norm_f32(fiou_d[side]), norm_f32(jac)), g_base_sp));
            }
            if (edge_m[side] != 0.0 && jac != 0.0) {
                gk = gk + split_to_f32(mul_split(vec2<f32>(edge_m[side], edge_e[side]), norm_f32(jac)));
            }
            if (k == ju) { gk = gk - sd * wl[side]; }
            if (k == ju + 1u) { gk = gk - sd * wr[side]; }
            if (use_ot) {
                let xk = BOX[base + (side * R + k) * na];
                var d_ot = vec2<f32>(0.0, 0.0);
                if (ju > 0u) {
                    if (k < ju) {
                        d_ot = add_split(d_ot, mul_split(exp_diff_split(xk, m_exc[k]), exc_sp[k]));
                    }
                    if (k > 0u) {
                        let ti = min(k - 1u, ju - 1u);
                        let term = mul_split(exp_diff_split(xk, m_ret[ti]), ret_sp[ti]);
                        d_ot = add_split(d_ot, vec2<f32>(-term.x, term.y));
                    }
                }
                if (ju + 2u < R) {
                    if (k + 1u < R) {
                        let ti = max(k, ju + 1u);
                        let term = mul_split(exp_diff_split(xk, m_ret[ti]), ret_sp[ti]);
                        d_ot = add_split(d_ot, vec2<f32>(-term.x, term.y));
                    }
                    if (k > ju + 1u) {
                        let ti = min(k - 1u, R - 2u);
                        d_ot = add_split(d_ot, mul_split(exp_diff_split(xk, m_exc[ti]), exc_sp[ti]));
                    }
                }
                gk = gk + split_to_f32(mul_split(d_ot, ot_scale));
            }
            OUT[base + (side * R + k) * na] = gk;
        }
    }
}
