// Shared by the detection kernels (task-aligned assigner and detection loss, kernels/detect.rs).
//
// The transcendental helpers are written out in basic arithmetic so every adapter computes them the same way and
// close to folo-core's CPU kernels: `fexp` is folo-core's `cpu::fast_exp` (Cephes `expf`, about 1 ulp), `log1p_unit`
// is accurate to a few ulp on [0, 1] (WGSL's `log` only promises an absolute error of 2^-21 near 1, which would
// be a relative error of several percent for the `log(1 + e^-|x|)` of a background logit), and `atan_f` is
// Cephes `atanf`.

const DET_NONE: u32 = 0xffffffffu;

fn is_finite_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn fexp(x: f32) -> f32 {
    let xc = clamp(x, -104.0, 89.0);
    let n = round(xc * 1.442695041);
    let r = xc - n * 0.6933594 - n * -2.1219444e-4;
    var p = 1.9875691e-4;
    p = p * r + 1.3981999e-3;
    p = p * r + 8.333452e-3;
    p = p * r + 4.1665796e-2;
    p = p * r + 1.6666665e-1;
    p = p * r + 0.5;
    let y = p * r * r + r + 1.0;
    let ni = i32(n);
    let n1 = ni >> 1u;
    let n2 = ni - n1;
    let s1 = bitcast<f32>(u32(n1 + 127) << 23u);
    let s2 = bitcast<f32>(u32(n2 + 127) << 23u);
    return y * s1 * s2;
}

// ln(1 + e) for e in [0, 1]: 2 atanh(z) with z = e / (2 + e) <= 1/3, eight terms of the odd series.
fn log1p_unit(e: f32) -> f32 {
    let z = e / (2.0 + e);
    let z2 = z * z;
    var s = 1.0 / 15.0;
    s = s * z2 + 1.0 / 13.0;
    s = s * z2 + 1.0 / 11.0;
    s = s * z2 + 1.0 / 9.0;
    s = s * z2 + 1.0 / 7.0;
    s = s * z2 + 1.0 / 5.0;
    s = s * z2 + 1.0 / 3.0;
    s = s * z2 + 1.0;
    return 2.0 * z * s;
}

// softplus(x) = max(x, 0) + ln(1 + e^-|x|), as folo-core computes it.
fn softplus_f(x: f32) -> f32 {
    return max(x, 0.0) + log1p_unit(fexp(-abs(x)));
}

// Stable sigmoid, as folo-core computes it.
fn sigmoid_f(x: f32) -> f32 {
    let e = fexp(-abs(x));
    let r = 1.0 / (1.0 + e);
    return select(e * r, r, x >= 0.0);
}

fn atan_f(xx: f32) -> f32 {
    var x = abs(xx);
    var y = 0.0;
    if (x > 2.414213562373095) {
        y = 1.5707963267948966;
        x = -1.0 / x;
    } else if (x > 0.4142135623730950) {
        y = 0.7853981633974483;
        x = (x - 1.0) / (x + 1.0);
    }
    let z = x * x;
    y = y + (((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z - 3.33329491539e-1) * z * x + x;
    return select(y, -y, xx < 0.0);
}

// atan(w / h), kept finite if a malformed box makes h exactly 0 (as boxes::ciou_eps).
fn atan_wh(w: f32, h: f32) -> f32 {
    if (h != 0.0) { return atan_f(w / h); }
    return atan2(w, h);
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

// Rational-Focal Edge-Decoupled IoU loss (R-PIoU) between predicted xyxy `a` and GT xyxy `b` (grid units).
// Uses scaled coordinates `q = sqrt(kappa) / max(w, delta0)` so extreme `delta0` / `kappa` never overflow or
// underflow `delta0^2`.
fn rpiou_loss_f(a: vec4<f32>, b: vec4<f32>, beta: f32, kappa: f32, delta0: f32) -> f32 {
    let eps = 1e-7;
    let w1 = max(a.z - a.x, 0.0);
    let h1 = max(a.w - a.y, 0.0);
    let w2 = max(b.z - b.x, 0.0);
    let h2 = max(b.w - b.y, 0.0);
    let inter = max(min(a.z, b.z) - max(a.x, b.x), 0.0) * max(min(a.w, b.w) - max(a.y, b.y), 0.0);
    let uni = w1 * h1 + w2 * h2 - inter + eps;
    let iou = inter / uni;
    let focal_iou = (1.0 - iou) * (1.0 + beta * iou);
    let rk = sqrt(kappa);
    let sx = max(w2, delta0);
    let sy = max(h2, delta0);
    let wx = w2 / sx;
    let dx_f = delta0 / sx;
    let denom_x = wx * wx + dx_f * dx_f;
    let hy = h2 / sy;
    let dy_f = delta0 / sy;
    let denom_y = hy * hy + dy_f * dy_f;
    let qx = rk / sx;
    let qy = rk / sy;
    let dx1 = a.x - b.x;
    let dx2 = a.z - b.z;
    let dy1 = a.y - b.y;
    let dy2 = a.w - b.w;
    let ux1 = select(0.0, dx1 * qx, dx1 != 0.0);
    let ux2 = select(0.0, dx2 * qx, dx2 != 0.0);
    let uy1 = select(0.0, dy1 * qy, dy1 != 0.0);
    let uy2 = select(0.0, dy2 * qy, dy2 != 0.0);
    let ke2_raw = 0.25 * ((ux1 * ux1 + ux2 * ux2) / denom_x + (uy1 * uy1 + uy2 * uy2) / denom_y);
    let ke2 = select(1e16, min(ke2_raw, 1e16), is_finite_f(ke2_raw));
    let inv1 = 1.0 / (1.0 + ke2);
    let edge_pen = select(1.0, ke2 * inv1, is_finite_f(ke2_raw));
    return focal_iou + edge_pen;
}
