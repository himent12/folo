// Detection loss, box terms (docs/MODEL.md §7.4) for the assigner's positives, one thread per pick slot:
// pd = anchor ∓ Σ_i i · softmax(box logits) (grid units), L_box = t · (1 − CIoU(pd, gt / stride)) and the DFL
// cross-entropy at the two bins around each side's clamped target distance, weighted t / 4. Slots that are not
// positive, or have t = 0, contribute nothing.
// MODE 0 writes OUT[2s] = box term, OUT[2s + 1] = Σ_sides (w_l · log p_j + w_r · log p_{j+1}) · t / 4 (the DFL term
// before its sign). MODE 1 writes the gradient of the weighted loss into OUT = d box_raw ([B, 4R, A], zeroed
// first; each anchor has at most one positive, so writes do not collide): g = GRAD[0], with S_box = STATS[8] and
// S_dfl = STATS[10] from the combine kernel. α of the CIoU is a constant, as in the CPU loss.
// Spec: R, K, MODE. Params: 0 = slots, 1 = A, 2 = G, 3 = gt box word offset, 4 = top clamp (R − 1 − 0.01),
// 5 = nwd_weight (0 = pure CIoU).
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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let s = linear_id(wid, nwg, lid, 64u);
    if (s >= pu(0u)) { return; }
    let t = bitcast<f32>(SLOT[2u * s]);
    let pos = SLOT[2u * s + 1u] == 1u;
    if (MODE == 0u) {
        OUT[2u * s] = 0.0;
        OUT[2u * s + 1u] = 0.0;
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

    // Softmax per side (p) and log-softmax (lp), kept for the backward.
    var p: array<f32, 4u * R>;
    var lp: array<f32, 4u * R>;
    var d: vec4<f32>;
    var dfl = 0.0;
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
    }
    let pd = vec4<f32>(ax - d[0], ay - d[1], ax + d[2], ay + d[3]);
    let nwd_w = pf(5u);
    // tb is in stride units (inv = 1 / stride); convert GT size to pixels for lambda and C = 12.8px to stride units.
    let lam = nwd_lambda((tb.z - tb.x) / inv, (tb.w - tb.y) / inv, nwd_w);
    let c_s = 12.8 * inv;
    if (MODE == 0u) {
        let l_ciou = 1.0 - ciou_eps(pd, tb);
        let l_box = select(l_ciou, (1.0 - lam) * l_ciou + lam * (1.0 - nwd_box(pd, tb, c_s)), lam > 0.0);
        OUT[2u * s] = l_box * t;
        OUT[2u * s + 1u] = dfl;
        return;
    }

    // Backward: the CIoU of §7.2 differentiated by hand (α constant), then through the expectation.
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

    let g = GRAD[0];
    // dL/dCIoU for L = S_box · t · (1 − lam) · (1 − CIoU).
    let gc = -t * STATS[8] * g * (1.0 - lam);
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
    var gx1 = -g_w1 - g_ix * w_max(pd.x, tb.x) - g_cw * w_min(pd.x, tb.x) - g_dx;
    var gy1 = -g_h1 - g_iy * w_max(pd.y, tb.y) - g_ch * w_min(pd.y, tb.y) - g_dy;
    var gx2 = g_w1 + g_ix * w_min(pd.z, tb.z) + g_cw * w_max(pd.z, tb.z) - g_dx;
    var gy2 = g_h1 + g_iy * w_min(pd.w, tb.w) + g_ch * w_max(pd.w, tb.w) - g_dy;
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
    // dL/dlogit for L = −S_dfl · Σ (w_l · lp_j + w_r · lp_{j+1}).
    let sd = STATS[10] * g;
    for (var side = 0u; side < 4u; side++) {
        let wsum = wl[side] + wr[side];
        for (var k = 0u; k < R; k++) {
            let q = p[side * R + k];
            var gk = g_d[side] * q * (f32(k) - d[side]) + sd * wsum * q;
            if (k == jl[side]) { gk = gk - sd * wl[side]; }
            if (k == jl[side] + 1u) { gk = gk - sd * wr[side]; }
            OUT[base + (side * R + k) * na] = gk;
        }
    }
}
