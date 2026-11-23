// Detection loss, box terms (docs/MODEL.md §7.4) for the assigner's positives, one thread per pick slot:
// pd = anchor ∓ Σ_i i · softmax(box logits) (grid units), L_box = t · (1 − CIoU(pd, gt / stride)) and the DFL
// cross-entropy at the two bins around each side's clamped target distance, weighted t / 4. Slots that are not
// positive, or have t = 0, contribute nothing.
// MODE 0 writes OUT[3s] = box term, OUT[3s + 1] = Σ_sides (w_l · log p_j + w_r · log p_{j+1}) · t / 4 (the DFL CE
// term before its sign), and OUT[3s + 2] = Σ_sides wt · L_ot (unscaled by dfl_ot_weight so det_combine normalises
// by tss before multiplying). MODE 1 writes the gradient of the weighted loss into OUT = d box_raw ([B, 4R, A], zeroed
// first; each anchor has at most one positive, so writes do not collide): g = GRAD[0], with S_box = STATS[8] and
// S_dfl = STATS[10] from the combine kernel. α of the CIoU is a constant, as in the CPU loss.
// Spec: R, K, MODE. Params: 0 = slots, 1 = A, 2 = G, 3 = gt box word offset, 4 = top clamp (R − 1 − 0.01),
// 5 = nwd_weight (0 = pure CIoU), 6 = box_loss_kind (0 = CIoU, 1 = R-PIoU), 7 = rpiou_beta, 8 = rpiou_kappa,
// 9 = rpiou_delta, 10 = dfl_ot_weight.
@group(0) @binding(1) var<storage, read> BOX: array<f32>;
@group(0) @binding(2) var<storage, read> META: array<u32>;
@group(0) @binding(3) var<storage, read> PICK: array<u32>;
@group(0) @binding(4) var<storage, read> SLOT: array<u32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(6) var<storage, read> STATS: array<f32>;
@group(0) @binding(7) var<storage, read> GRAD: array<f32>;

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
        OUT[3u * s] = 0.0;
        OUT[3u * s + 1u] = 0.0;
        OUT[3u * s + 2u] = 0.0;
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

    // Softmax per side (p) and log-softmax (lp), kept for the backward.
    var p: array<f32, 4u * R>;
    var lp: array<f32, 4u * R>;
    var d: vec4<f32>;
    var dfl = 0.0;
    var ot_sum = 0.0;
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
            var l_tail = 0.0;
            if (ju > 0u) {
                var ms = BOX[base + (side * R + (R - 1u)) * na];
                var s_acc = 1.0;
                for (var step = 0u; step + 1u < R; step++) {
                    let k = (R - 2u) - step;
                    if (step > 0u) {
                        let x = BOX[base + (side * R + k + 1u) * na];
                        if (x > ms) {
                            s_acc = s_acc * safe_exp_neg(ms - x) + 1.0;
                            ms = x;
                        } else {
                            s_acc = s_acc + safe_exp_neg(x - ms);
                        }
                    }
                    if (k < ju) { l_tail = l_tail + max((m - ms) + (lz - log_ge1(s_acc)), 0.0); }
                }
            }
            if (ju + 2u < R) {
                var mf = BOX[base + (side * R) * na];
                var f_acc = 1.0;
                for (var k = 1u; k + 1u < R; k++) {
                    let x = BOX[base + (side * R + k) * na];
                    if (x > mf) {
                        f_acc = f_acc * safe_exp_neg(mf - x) + 1.0;
                        mf = x;
                    } else {
                        f_acc = f_acc + safe_exp_neg(x - mf);
                    }
                    if (k > ju) { l_tail = l_tail + max((m - mf) + (lz - log_ge1(f_acc)), 0.0); }
                }
            }
            ot_sum = ot_sum + wt * l_tail;
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
        OUT[3u * s] = l_box * t;
        OUT[3u * s + 1u] = dfl;
        OUT[3u * s + 2u] = ot_sum;
        return;
    }

    // Backward: R-PIoU or CIoU (§7.2, α constant), then through the expectation.
    let g = GRAD[0];
    let g_base = t * STATS[8] * g * (1.0 - lam);
    var gx1 = 0.0;
    var gy1 = 0.0;
    var gx2 = 0.0;
    var gy2 = 0.0;
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
        let g_u = -g_base * (1.0 - rpiou_beta + 2.0 * rpiou_beta * iou);
        let g_inter = g_u * (1.0 / uni + inter / (uni * uni));
        let g_uni = -g_u * inter / (uni * uni);
        let g_w1 = select(0.0, g_uni * h1, pd.z > pd.x);
        let g_h1 = select(0.0, g_uni * w1, pd.w > pd.y);
        let g_ix = select(0.0, g_inter * ih, ix > 0.0);
        let g_iy = select(0.0, g_inter * iw, iy > 0.0);
        let rk = sqrt(rpiou_kappa);
        let sx = max(w2, rpiou_delta);
        let sy = max(h2, rpiou_delta);
        let wx = w2 / sx;
        let dx_f = rpiou_delta / sx;
        let denom_x = wx * wx + dx_f * dx_f;
        let hy = h2 / sy;
        let dy_f = rpiou_delta / sy;
        let denom_y = hy * hy + dy_f * dy_f;
        let qx = rk / sx;
        let qy = rk / sy;
        let dx1 = pd.x - tb.x;
        let dx2 = pd.z - tb.z;
        let dy1 = pd.y - tb.y;
        let dy2 = pd.w - tb.w;
        let ux1 = select(0.0, dx1 * qx, dx1 != 0.0);
        let ux2 = select(0.0, dx2 * qx, dx2 != 0.0);
        let uy1 = select(0.0, dy1 * qy, dy1 != 0.0);
        let uy2 = select(0.0, dy2 * qy, dy2 != 0.0);
        let ke2_raw = 0.25 * ((ux1 * ux1 + ux2 * ux2) / denom_x + (uy1 * uy1 + uy2 * uy2) / denom_y);
        var ex1 = 0.0;
        var ey1 = 0.0;
        var ex2 = 0.0;
        var ey2 = 0.0;
        if (is_finite_f(ke2_raw) && ke2_raw < 1e16) {
            let inv1 = 1.0 / (1.0 + ke2_raw);
            let g_half = g_base * 0.5;
            let qx_i = qx * inv1;
            let qy_i = qy * inv1;
            if (abs(g_half) <= 1.0) {
                if (dx1 != 0.0) { ex1 = (((ux1 / denom_x) * g_half) * inv1) * qx_i; }
                if (dy1 != 0.0) { ey1 = (((uy1 / denom_y) * g_half) * inv1) * qy_i; }
                if (dx2 != 0.0) { ex2 = (((ux2 / denom_x) * g_half) * inv1) * qx_i; }
                if (dy2 != 0.0) { ey2 = (((uy2 / denom_y) * g_half) * inv1) * qy_i; }
            } else {
                if (dx1 != 0.0) { ex1 = (((ux1 / denom_x) * inv1) * qx_i) * g_half; }
                if (dy1 != 0.0) { ey1 = (((uy1 / denom_y) * inv1) * qy_i) * g_half; }
                if (dx2 != 0.0) { ex2 = (((ux2 / denom_x) * inv1) * qx_i) * g_half; }
                if (dy2 != 0.0) { ey2 = (((uy2 / denom_y) * inv1) * qy_i) * g_half; }
            }
        }
        gx1 = -g_w1 - g_ix * w_max(pd.x, tb.x) + ex1;
        gy1 = -g_h1 - g_iy * w_max(pd.y, tb.y) + ey1;
        gx2 = g_w1 + g_ix * w_min(pd.z, tb.z) + ex2;
        gy2 = g_h1 + g_iy * w_min(pd.w, tb.w) + ey2;
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
    let r_m2 = f32(R - 2u);
    for (var side = 0u; side < 4u; side++) {
        let wsum = wl[side] + wr[side];
        let ju = jl[side];
        var ms_arr: array<f32, R>;
        var bs_mant: array<f32, R>;
        var mf_arr: array<f32, R>;
        var ap_mant: array<f32, R>;
        if (use_ot) {
            if (ju > 0u) {
                var ms = BOX[base + (side * R + (R - 1u)) * na];
                var s_acc = 1.0;
                for (var step = 0u; step + 1u < R; step++) {
                    let k = (R - 2u) - step;
                    if (step > 0u) {
                        let x = BOX[base + (side * R + k + 1u) * na];
                        if (x > ms) {
                            s_acc = s_acc * safe_exp_neg(ms - x) + 1.0;
                            ms = x;
                        } else {
                            s_acc = s_acc + safe_exp_neg(x - ms);
                        }
                    }
                    if (k < ju) {
                        ms_arr[k] = ms;
                        bs_mant[k] = 1.0 / s_acc;
                    }
                }
                for (var r = 1u; r < ju; r++) {
                    bs_mant[r] = bs_mant[r - 1u] * safe_exp_neg(ms_arr[r] - ms_arr[r - 1u]) + bs_mant[r];
                }
            }
            if (ju + 2u < R) {
                var mf = BOX[base + (side * R) * na];
                var f_acc = 1.0;
                for (var k = 1u; k + 1u < R; k++) {
                    let x = BOX[base + (side * R + k) * na];
                    if (x > mf) {
                        f_acc = f_acc * safe_exp_neg(mf - x) + 1.0;
                        mf = x;
                    } else {
                        f_acc = f_acc + safe_exp_neg(x - mf);
                    }
                    if (k > ju) {
                        mf_arr[k] = mf;
                        ap_mant[k] = 1.0 / f_acc;
                    }
                }
                var r = R - 2u;
                while (r > ju + 1u) {
                    r = r - 1u;
                    ap_mant[r] = ap_mant[r + 1u] * safe_exp_neg(mf_arr[r] - mf_arr[r + 1u]) + ap_mant[r];
                }
            }
        }
        for (var k = 0u; k < R; k++) {
            let q = p[side * R + k];
            var gk = g_d[side] * q * (f32(k) - d[side]) + sd * wsum * q;
            if (k == ju) { gk = gk - sd * wl[side]; }
            if (k == ju + 1u) { gk = gk - sd * wr[side]; }
            if (use_ot) {
                let xk = BOX[base + (side * R + k) * na];
                var bm = 0.0;
                if (k > 0u && ju > 0u) {
                    let ti = min(k - 1u, ju - 1u);
                    bm = safe_exp_neg(xk - ms_arr[ti]) * bs_mant[ti];
                }
                var am = 0.0;
                if (k + 1u < R && ju + 2u < R) {
                    let ti = max(k, ju + 1u);
                    am = safe_exp_neg(xk - mf_arr[ti]) * ap_mant[ti];
                }
                let u_ot = wt * (r_m2 * q - bm - am);
                if (abs(sd) <= 1.0) {
                    gk = gk + (u_ot * sd) * dfl_ot_w;
                } else {
                    gk = gk + (u_ot * dfl_ot_w) * sd;
                }
            }
            OUT[base + (side * R + k) * na] = gk;
        }
    }
}
