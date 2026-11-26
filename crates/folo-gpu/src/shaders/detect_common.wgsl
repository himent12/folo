// Shared by the detection kernels (task-aligned assigner, detection loss, and candidate decode, kernels/detect.rs).
//
// The transcendental helpers are written out in basic arithmetic so every adapter computes them the same way and
// close to folo-core's CPU kernels: `fexp` uses a degree-5 minimax polynomial with FMA and split-shift scaling,
// `log1p_unit` uses a 4-term Remez minimax on [0, 1] (WGSL's `log` only promises an absolute error of 2^-21 near 1,
// which would be a relative error of several percent for the `log(1 + e^-|x|)` of a background logit), and `atan_wh`
// is a branch-free 1-division 3-region Remez approximation.

const DET_NONE: u32 = 0xffffffffu;

fn is_finite_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn fexp(x: f32) -> f32 {
    let xc = clamp(x, -104.0, 89.0);
    let z = fma(xc, 1.442695041, 12582912.0);
    let n = z - 12582912.0;
    let r = fma(n, 2.1219444e-4, fma(n, -0.6933594, xc));
    var p = 8.290312e-3;
    p = fma(p, r, 4.1897953e-2);
    p = fma(p, r, 1.6667636e-1);
    p = fma(p, r, 4.999915e-1);
    p = fma(p, r, 9.999997e-1);
    let y = fma(p, r, 1.0);
    let ni = bitcast<i32>(z) - 0x4b400000;
    let n1 = ni >> 1u;
    let n2 = ni - n1;
    let s1 = bitcast<f32>(u32(n1 + 127) << 23u);
    let s2 = bitcast<f32>(u32(n2 + 127) << 23u);
    return (y * s1) * s2;
}

// ln(1 + e) for e in [0, 1]: 4-term Remez minimax on z^2 with z = e / (2 + e) <= 1/3 (max rel err 1.03e-8).
fn log1p_unit(e: f32) -> f32 {
    let z = e / (2.0 + e);
    let z2 = z * z;
    var s = 1.3893566e-1;
    s = fma(s, z2, 1.402164e-1);
    s = fma(s, z2, 2.0009655e-1);
    s = fma(s, z2, 3.3333224e-1);
    let u = 2.0 * z;
    return fma(u * z2, s, u);
}

// softplus(x) = max(x, 0) + ln(1 + e^-|x|), as folo-core computes it.
fn softplus_f(x: f32) -> f32 {
    return max(x, 0.0) + log1p_unit(fexp(-abs(x)));
}

// Stable sigmoid matching IEEE f32 underflow-to-zero semantics on extreme negative logits (`<= -88.72284`).
fn sigmoid_f(x: f32) -> f32 {
    if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) { return x; }
    if (x <= -88.72284) { return 0.0; }
    let e = fexp(-abs(x));
    let r = 1.0 / (1.0 + e);
    return select(e * r, r, x >= 0.0);
}

// Branch-free 1-division 3-region Remez atan(w / h), kept finite if a malformed box makes h == 0.
fn atan_wh(w: f32, h: f32) -> f32 {
    let aw = abs(w);
    let ah = abs(h);
    let gt_hi = aw > 2.414213562 * ah;
    let gt_lo = aw > 0.414213562 * ah;
    let y0 = select(select(0.0, 0.785398163, gt_lo), 1.570796327, gt_hi);
    let num = select(select(aw, aw - ah, gt_lo), -ah, gt_hi);
    let den = select(select(ah, aw + ah, gt_lo), aw, gt_hi);
    let x = select(num / den, 0.0, den == 0.0);
    let z = x * x;
    var p = 8.05374449538e-2;
    p = fma(p, z, -1.38776856032e-1);
    p = fma(p, z, 1.99777106478e-1);
    p = fma(p, z, -3.33329491539e-1);
    let y = y0 + fma(p * z, x, x);
    return select(y, -y, (w < 0.0) != (h < 0.0));
}

// docs/MODEL.md §7.2 CIoU of xyxy boxes a and b, in the evaluation order of folo-vision's boxes::ciou_eps.
// V_K is 4 / pi^2 as the CPU computes it in f32 (declared by kernels/detect.rs).
fn ciou_eps(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let eps = 1e-7;
    let w1 = a.z - a.x;
    let h1 = a.w - a.y + eps;
    let w2 = b.z - b.x;
    let h2 = b.w - b.y + eps;
    let inter = max(min(a.z, b.z) - max(a.x, b.x), 0.0) * max(min(a.w, b.w) - max(a.y, b.y), 0.0);
    let uni = w1 * h1 + w2 * h2 - inter + eps;
    let iou = inter / uni;
    let cw = max(a.z, b.z) - min(a.x, b.x);
    let ch = max(a.w, b.w) - min(a.y, b.y);
    let c2 = cw * cw + ch * ch + eps;
    let dx = b.x + b.z - a.x - a.z;
    let dy = b.y + b.w - a.y - a.w;
    let rho2 = (dx * dx + dy * dy) / 4.0;
    let d = atan_wh(w2, h2) - atan_wh(w1, h1);
    let v = V_K * d * d;
    let alpha = v / (v - iou + (1.0 + eps));
    return iou - (rho2 / c2 + v * alpha);
}

// ln(z) for a normal z >= 1: exponent times ln 2 plus log1p_unit of the mantissa (accurate where WGSL's `log`
// only promises an absolute error near 1).
fn log_ge1(z: f32) -> f32 {
    let bits = bitcast<u32>(z);
    let k = i32((bits >> 23u) & 0xffu) - 127;
    let f = bitcast<f32>((bits & 0x007fffffu) | 0x3f800000u);
    return f32(k) * 0.6931471805599453 + log1p_unit(f - 1.0);
}

// x^y for x >= 0, y >= 0, safe at x = 0 (WGSL leaves pow(0, y) undefined).
fn pow_pos(x: f32, y: f32) -> f32 {
    if (y == 0.0) { return 1.0; }
    if (x <= 0.0) { return 0.0; }
    return pow(x, y);
}

// Plain IoU of xyxy boxes a and b (with eps = 1e-7 in denominator).
fn iou_plain(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let eps = 1e-7;
    let w1 = max(a.z - a.x, 0.0);
    let h1 = max(a.w - a.y, 0.0);
    let w2 = max(b.z - b.x, 0.0);
    let h2 = max(b.w - b.y, 0.0);
    let inter = max(min(a.z, b.z) - max(a.x, b.x), 0.0) * max(min(a.w, b.w) - max(a.y, b.y), 0.0);
    return inter / (w1 * h1 + w2 * h2 - inter + eps);
}

// Normalized Wasserstein Distance (NWD) between xyxy boxes a and b with scale constant c (in boxes' units).
fn nwd_box(a: vec4<f32>, b: vec4<f32>, c: f32) -> f32 {
    let dcx = 0.5 * (a.x + a.z - b.x - b.z);
    let dcy = 0.5 * (a.y + a.w - b.y - b.w);
    let dw = (a.z - a.x) - (b.z - b.x);
    let dh = (a.w - a.y) - (b.w - b.y);
    let w2_sq = dcx * dcx + dcy * dcy + 0.25 * (dw * dw + dh * dh);
    return fexp(-sqrt(w2_sq + 1e-6) / c);
}

// Scale-adaptive NWD blending weight lambda(s_g) = weight * exp(-(s_g / 24)^2), with s_g^2 = gt_w_px * gt_h_px.
fn nwd_lambda(gt_w_px: f32, gt_h_px: f32, weight: f32) -> f32 {
    if (weight <= 0.0) { return 0.0; }
    let area = max(gt_w_px, 0.0) * max(gt_h_px, 0.0);
    return weight * fexp(-area / (24.0 * 24.0));
}

// Split finite `x` into `(m, e)` with `|m| in [1, 2)` and `x = m * 2^e` (`0` maps to `(0, 0)`).
// Uses pure integer bit manipulation (no float arithmetic on subnormals) so it is exact even on FTZ GPUs.
fn norm_f32(x: f32) -> vec2<f32> {
    let bits = bitcast<u32>(x);
    let exp_bits = (bits >> 23u) & 0xffu;
    let mant_bits = bits & 0x007fffffu;
    let sign_bit = bits & 0x80000000u;
    if (exp_bits == 0u) {
        if (mant_bits == 0u) { return vec2<f32>(0.0, 0.0); }
        let shift = countLeadingZeros(mant_bits) - 8u;
        let m = bitcast<f32>(sign_bit | 0x3f800000u | ((mant_bits << shift) & 0x007fffffu));
        return vec2<f32>(m, f32(-126 - i32(shift)));
    }
    let m = bitcast<f32>(sign_bit | 0x3f800000u | mant_bits);
    return vec2<f32>(m, f32(i32(exp_bits) - 127));
}

// Exact power-of-two scaling `x * 2^e` across the full exponent range without intermediate overflow/underflow.
fn mul_pow2(x: f32, e: i32) -> f32 {
    if (x == 0.0) { return 0.0; }
    var v = x;
    var r = e;
    if (r > 127) { v = v * bitcast<f32>(0x7f000000u); r = r - 127; }
    if (r > 127) { v = v * bitcast<f32>(0x7f000000u); r = r - 127; }
    if (r < -126) { v = v * bitcast<f32>(0x00800000u); r = r + 126; }
    if (r < -126) { v = v * bitcast<f32>(0x00800000u); r = r + 126; }
    let ec = clamp(r, -126, 127);
    return v * bitcast<f32>(u32(ec + 127) << 23u);
}

// Compute `exp(x)` for `x <= 0` in exponent-split form `(m, e)` with `m in [1, 2)` and `exp(x) = m * 2^e`
// (`(0, 0)` when `x <= -700.0`). Keeps full 24-bit precision across `[-700, 0]` without subnormal underflow.
fn exp_split_neg(x: f32) -> vec2<f32> {
    if (x <= -700.0) { return vec2<f32>(0.0, 0.0); }
    let n = round(x * 1.442695041);
    let r = x - n * 0.6933594 - n * -2.1219444e-4;
    var p = 1.9875691e-4;
    p = p * r + 1.3981999e-3;
    p = p * r + 8.333452e-3;
    p = p * r + 4.1665796e-2;
    p = p * r + 1.6666665e-1;
    p = p * r + 0.5;
    let y = p * r * r + r + 1.0;
    let yn = norm_f32(y);
    return vec2<f32>(yn.x, yn.y + n);
}

// Add two normalized exponent-split values `a = a.x * 2^a.y` and `b = b.x * 2^b.y` (`x == 0` encodes `0`).
fn add_split(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (a.x == 0.0) { return b; }
    if (b.x == 0.0) { return a; }
    var hi = a;
    var lo = b;
    if (b.y > a.y) {
        hi = b;
        lo = a;
    }
    let diff = i32(hi.y) - i32(lo.y);
    if (diff > 26) { return hi; }
    let s = hi.x + mul_pow2(lo.x, -diff);
    if (s == 0.0) { return vec2<f32>(0.0, 0.0); }
    let sn = norm_f32(s);
    return vec2<f32>(sn.x, sn.y + hi.y);
}

// Multiply two normalized exponent-split values.
fn mul_split(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (a.x == 0.0 || b.x == 0.0) { return vec2<f32>(0.0, 0.0); }
    let pn = norm_f32(a.x * b.x);
    return vec2<f32>(pn.x, pn.y + a.y + b.y);
}

// Divide normalized exponent-split `a` by non-zero `b`.
fn div_split(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (a.x == 0.0) { return vec2<f32>(0.0, 0.0); }
    let qn = norm_f32(a.x / b.x);
    return vec2<f32>(qn.x, qn.y + a.y - b.y);
}

// Materialize a normalized exponent-split value back to `f32`.
fn split_to_f32(a: vec2<f32>) -> f32 {
    if (a.x == 0.0 || a.y < -160.0) { return 0.0; }
    if (a.y > 160.0) { return mul_pow2(a.x, 200); }
    return mul_pow2(a.x, i32(a.y));
}

// Compute `dx * q_m * 2^e_q` without forming an overflowing or underflowing `q_m * 2^e_q`.
fn scaled_edge_u(dx: f32, q_m: f32, e_q: i32) -> f32 {
    if (dx == 0.0) { return 0.0; }
    let dn = norm_f32(dx);
    return mul_pow2(dn.x * q_m, i32(dn.y) + e_q);
}

// Rational-Focal Edge-Decoupled IoU loss (R-PIoU) between predicted xyxy `a` and GT xyxy `b` (grid units).
// Uses exponent-split scaling `q = sqrt(kappa) / max(w, delta0)` so extreme `delta0`, `w2`, and `kappa` never
// prematurely overflow or underflow.
fn rpiou_loss_f(a: vec4<f32>, b: vec4<f32>, beta: f32, kappa: f32, delta0: f32) -> f32 {
    let eps = 1e-7;
    let w1 = max(a.z - a.x, 0.0);
    let h1 = max(a.w - a.y, 0.0);
    let w2 = max(b.z - b.x, 0.0);
    let h2 = max(b.w - b.y, 0.0);
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let inter = max(min(a.z, b.z) - max(a.x, b.x), 0.0) * max(min(a.w, b.w) - max(a.y, b.y), 0.0);
    let uni = area1 + area2 - inter + eps;
    let iou = inter / uni;
    let one_minus_iou = (max(area1 - inter, 0.0) + max(area2 - inter, 0.0) + eps) / uni;
    let focal_iou = one_minus_iou * (1.0 + beta * iou);
    let rkn = norm_f32(sqrt(kappa));
    let sx = max(w2, delta0);
    let sy = max(h2, delta0);
    let sxn = norm_f32(sx);
    let syn = norm_f32(sy);
    let wx = w2 / sx;
    let dx_f = delta0 / sx;
    let denom_x = wx * wx + dx_f * dx_f;
    let hy = h2 / sy;
    let dy_f = delta0 / sy;
    let denom_y = hy * hy + dy_f * dy_f;
    let q_xm = rkn.x / sxn.x;
    let e_qx = i32(rkn.y) - i32(sxn.y);
    let q_ym = rkn.x / syn.x;
    let e_qy = i32(rkn.y) - i32(syn.y);
    let ux1 = scaled_edge_u(a.x - b.x, q_xm, e_qx);
    let ux2 = scaled_edge_u(a.z - b.z, q_xm, e_qx);
    let uy1 = scaled_edge_u(a.y - b.y, q_ym, e_qy);
    let uy2 = scaled_edge_u(a.w - b.w, q_ym, e_qy);
    let ke2_raw = 0.25 * ((ux1 * ux1 + ux2 * ux2) / denom_x + (uy1 * uy1 + uy2 * uy2) / denom_y);
    let ke2 = select(1e16, min(ke2_raw, 1e16), is_finite_f(ke2_raw));
    let inv1 = 1.0 / (1.0 + ke2);
    let edge_pen = select(1.0, ke2 * inv1, is_finite_f(ke2_raw));
    return focal_iou + edge_pen;
}

// Overflow-safe `(q(u), t(u))` where `q(u) = (1 + u^2)^(-1/2)` and `t(u) = u * q(u)`.
fn inv_sqrt_1p_sq(u: f32) -> vec2<f32> {
    let au = abs(u);
    if (au > 1e4) {
        return vec2<f32>(1.0 / au, select(-1.0, 1.0, u >= 0.0));
    }
    let q = 1.0 / sqrt(1.0 + u * u);
    return vec2<f32>(q, u * q);
}

// Scale-normalized Algebraic Edge IoU loss (S-AEIoU) between predicted xyxy `a` and GT xyxy `b` (grid units).
// Purely algebraic: zero `atan`, zero `exp`, zero `norm_f32`.
fn saeiou_loss_f(a: vec4<f32>, b: vec4<f32>, beta: f32, kappa: f32, delta0: f32) -> f32 {
    let eps_s = 1e-7;
    let w1 = max(a.z - a.x, 0.0);
    let h1 = max(a.w - a.y, 0.0);
    let w2 = max(b.z - b.x, 0.0);
    let h2 = max(b.w - b.y, 0.0);
    let mx = max(w1, w2);
    let my = max(h1, h2);
    let inv_sx = select(1.0, 1.0 / mx, mx > 1e6);
    let inv_sy = select(1.0, 1.0 / my, my > 1e6);
    let w1s = w1 * inv_sx;
    let w2s = w2 * inv_sx;
    let iws = max(min(a.z, b.z) - max(a.x, b.x), 0.0) * inv_sx;
    let h1s = h1 * inv_sy;
    let h2s = h2 * inv_sy;
    let ihs = max(min(a.w, b.w) - max(a.y, b.y), 0.0) * inv_sy;
    let area1 = w1s * h1s;
    let area2 = w2s * h2s;
    let inter = iws * ihs;
    let eps_eff = max(eps_s * inv_sx * inv_sy, 1e-30);
    let uni = area1 + area2 - inter + eps_eff;
    let iou = inter / uni;
    let deficit = max(area1 - inter, 0.0) + max(area2 - inter, 0.0) + eps_eff;
    let focal_iou = (deficit / uni) * (1.0 + beta * iou);
    let m_d = max(max(w2, h2), delta0);
    let sc_d = select(1.0, 1e-20, m_d > 1e18);
    let sx_n = w2 * sc_d + delta0 * sc_d;
    let sy_n = h2 * sc_d + delta0 * sc_d;
    let inv_sum_n = 1.0 / (sx_n + sy_n);
    let wx = 2.0 * sx_n * inv_sum_n;
    let wy = 2.0 * sy_n * inv_sum_n;
    let sc_x = select(1.0, 1e-20, max(w2, delta0) > 1e18);
    let sc_y = select(1.0, 1e-20, max(h2, delta0) > 1e18);
    let sx_x = w2 * sc_x + delta0 * sc_x;
    let sy_y = h2 * sc_y + delta0 * sc_y;
    let rk = sqrt(kappa);
    let qt_x1 = inv_sqrt_1p_sq(rk * ((a.x * sc_x - b.x * sc_x) / sx_x));
    let qt_x2 = inv_sqrt_1p_sq(rk * ((a.z * sc_x - b.z * sc_x) / sx_x));
    let qt_y1 = inv_sqrt_1p_sq(rk * ((a.y * sc_y - b.y * sc_y) / sy_y));
    let qt_y2 = inv_sqrt_1p_sq(rk * ((a.w * sc_y - b.w * sc_y) / sy_y));
    let phi_x1 = (qt_x1.y * qt_x1.y) / (1.0 + qt_x1.x);
    let phi_x2 = (qt_x2.y * qt_x2.y) / (1.0 + qt_x2.x);
    let phi_y1 = (qt_y1.y * qt_y1.y) / (1.0 + qt_y1.x);
    let phi_y2 = (qt_y2.y * qt_y2.y) / (1.0 + qt_y2.x);
    let edge_pen = 0.25 * (wx * (phi_x1 + phi_x2) + wy * (phi_y1 + phi_y2));
    return focal_iou + edge_pen;
}
