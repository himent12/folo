// Forward fused `h = x + res; y = norm(h, w, b, eps)` for LayerNorm (`!RMS`) and RMSNorm (`RMS`) over dense `[outer, d]`.
// `residual_norm.rs` prepends `vec_prelude(_f16)`, `param_f32_prelude`, `CACHE_ROW`, `LANE_HELPERS`, reduction helpers, and `main`.
// Params: 0 = `rows`, 1 = `units`, 2 = `inv_d`, 3 = `eps`, 4 = `sqrt(eps)`.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> RES: array<E>;
@group(0) @binding(3) var<storage, read> W: array<PF>;
@group(0) @binding(4) var<storage, read> B: array<PF>;
@group(0) @binding(5) var<storage, read_write> H: array<E>;
@group(0) @binding(6) var<storage, read_write> Y: array<E>;
@group(0) @binding(7) var<storage, read_write> MEAN: array<f32>;
@group(0) @binding(8) var<storage, read_write> RSTD: array<f32>;

fn apply_affine(xh: vec4<f32>, idx: u32) -> E {
    var v = xh;
    if (HAS_W) { v *= PF4(W[idx]); }
    if (HAS_B) { v += PF4(B[idx]); }
    return OUT(v);
}

fn res_norm_fwd_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let units = pu(1u);
    let inv_d = pf(2u);
    let eps = pf(3u);
    let eps_sqrt = pf(4u);
    let valid = row < rows;
    let base = row * units;

    if (CACHE_ROW) {
        var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
        let i0 = lid; let i1 = lid + 256u; let i2 = lid + 512u; let i3 = lid + 768u;
        let h0 = valid && i0 < units; let h1 = valid && i1 < units;
        let h2 = valid && i2 < units; let h3 = valid && i3 < units;
        if (h0) { v0 = V4(X[base + i0]) + V4(RES[base + i0]); H[base + i0] = OUT(v0); v0 = V4(OUT(v0)); }
        if (h1) { v1 = V4(X[base + i1]) + V4(RES[base + i1]); H[base + i1] = OUT(v1); v1 = V4(OUT(v1)); }
        if (h2) { v2 = V4(X[base + i2]) + V4(RES[base + i2]); H[base + i2] = OUT(v2); v2 = V4(OUT(v2)); }
        if (h3) { v3 = V4(X[base + i3]) + V4(RES[base + i3]); H[base + i3] = OUT(v3); v3 = V4(OUT(v3)); }

        var mean = 0.0;
        if (!RMS) {
            var s = 0.0; var am = 0.0;
            if (h0) { s += lane_sum(v0); am = max(am, lane_amax(v0)); }
            if (h1) { s += lane_sum(v1); am = max(am, lane_amax(v1)); }
            if (h2) { s += lane_sum(v2); am = max(am, lane_amax(v2)); }
            if (h3) { s += lane_sum(v3); am = max(am, lane_amax(v3)); }
            let sm = reduce_sum_max(lid, sgid, lane, s, am);
            mean = sm.x * inv_d;
            let raw_nonfinite = !is_finite1(sm.x) && sm.y > 0.0 && is_finite1(sm.y);
            if (raw_nonfinite) {
                var ss = 0.0;
                if (h0) { ss += lane_sum(ldexp(v0, vec4<i32>(-32))); }
                if (h1) { ss += lane_sum(ldexp(v1, vec4<i32>(-32))); }
                if (h2) { ss += lane_sum(ldexp(v2, vec4<i32>(-32))); }
                if (h3) { ss += lane_sum(ldexp(v3, vec4<i32>(-32))); }
                mean = ldexp(reduce_sum1(lid, sgid, lane, ss) * inv_d, 32i);
            }
            if (h0) { v0 -= vec4<f32>(mean); }
            if (h1) { v1 -= vec4<f32>(mean); }
            if (h2) { v2 -= vec4<f32>(mean); }
            if (h3) { v3 -= vec4<f32>(mean); }
        }

        var sq = 0.0; var dam = 0.0;
        if (h0) { sq += lane_sum(v0 * v0); dam = max(dam, lane_amax(v0)); }
        if (h1) { sq += lane_sum(v1 * v1); dam = max(dam, lane_amax(v1)); }
        if (h2) { sq += lane_sum(v2 * v2); dam = max(dam, lane_amax(v2)); }
        if (h3) { sq += lane_sum(v3 * v3); dam = max(dam, lane_amax(v3)); }
        let sq_m = reduce_sum_max(lid, sgid, lane, sq, dam);
        let ms = sq_m.x * inv_d;
        let sq_nan = sq_m.x != sq_m.x;
        let need_scale = !sq_nan && (is_inf1(sq_m.x) || (ms < 1e-30 && eps < 1e-30))
            && sq_m.y > 0.0 && is_finite1(sq_m.y);
        var rstd = inverseSqrt(ms + eps);
        var shift_s = 0i; var inv_adj = 1.0; var inv_rms_s = rstd;
        if (need_scale) {
            let scale = max(sq_m.y, eps_sqrt);
            shift_s = select(select(0i, 64i, scale < 1e-18), -64i, scale > 1e18);
            inv_adj = 1.0 / ldexp(scale, shift_s);
            let sv = vec4<i32>(shift_s);
            var usq = 0.0;
            if (h0) { let u = ldexp(v0, sv) * inv_adj; usq += lane_sum(u * u); }
            if (h1) { let u = ldexp(v1, sv) * inv_adj; usq += lane_sum(u * u); }
            if (h2) { let u = ldexp(v2, sv) * inv_adj; usq += lane_sum(u * u); }
            if (h3) { let u = ldexp(v3, sv) * inv_adj; usq += lane_sum(u * u); }
            let usq_tot = reduce_sum1(lid, sgid, lane, usq);
            let es = ldexp(eps_sqrt, shift_s) * inv_adj;
            inv_rms_s = inverseSqrt(usq_tot * inv_d + es * es);
            rstd = ldexp(inv_rms_s * inv_adj, shift_s);
        }
        if (!valid) { return; }
        if (lid == 0u) {
            if (!RMS) { MEAN[row] = mean; }
            RSTD[row] = rstd;
        }
        if (need_scale) {
            let sv = vec4<i32>(shift_s);
            if (h0) { Y[base + i0] = apply_affine((ldexp(v0, sv) * inv_adj) * inv_rms_s, i0); }
            if (h1) { Y[base + i1] = apply_affine((ldexp(v1, sv) * inv_adj) * inv_rms_s, i1); }
            if (h2) { Y[base + i2] = apply_affine((ldexp(v2, sv) * inv_adj) * inv_rms_s, i2); }
            if (h3) { Y[base + i3] = apply_affine((ldexp(v3, sv) * inv_adj) * inv_rms_s, i3); }
        } else {
            if (h0) { Y[base + i0] = apply_affine(v0 * rstd, i0); }
            if (h1) { Y[base + i1] = apply_affine(v1 * rstd, i1); }
            if (h2) { Y[base + i2] = apply_affine(v2 * rstd, i2); }
            if (h3) { Y[base + i3] = apply_affine(v3 * rstd, i3); }
        }
        return;
    }

    var mean = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            H[base + i] = OUT(V4(X[base + i]) + V4(RES[base + i]));
        }
    }
    if (!RMS) {
        var s = 0.0; var am = 0.0;
        if (valid) {
            for (var i = lid; i < units; i += 256u) {
                let v = V4(H[base + i]);
                s += lane_sum(v);
                am = max(am, lane_amax(v));
            }
        }
        let sm = reduce_sum_max(lid, sgid, lane, s, am);
        mean = sm.x * inv_d;
        let raw_nonfinite = !is_finite1(sm.x) && sm.y > 0.0 && is_finite1(sm.y);
        if (raw_nonfinite) {
            var ss = 0.0;
            if (valid) {
                for (var i = lid; i < units; i += 256u) {
                    ss += lane_sum(ldexp(V4(H[base + i]), vec4<i32>(-32)));
                }
            }
            mean = ldexp(reduce_sum1(lid, sgid, lane, ss) * inv_d, 32i);
        }
    }
    var sq = 0.0; var dam = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            var v = V4(H[base + i]);
            if (!RMS) { v -= vec4<f32>(mean); }
            sq += lane_sum(v * v);
            dam = max(dam, lane_amax(v));
        }
    }
    let sq_m = reduce_sum_max(lid, sgid, lane, sq, dam);
    let ms = sq_m.x * inv_d;
    let sq_nan = sq_m.x != sq_m.x;
    let need_scale = !sq_nan && (is_inf1(sq_m.x) || (ms < 1e-30 && eps < 1e-30))
        && sq_m.y > 0.0 && is_finite1(sq_m.y);
    var rstd = inverseSqrt(ms + eps);
    var shift_s = 0i; var inv_adj = 1.0; var inv_rms_s = rstd;
    if (need_scale) {
        let scale = max(sq_m.y, eps_sqrt);
        shift_s = select(select(0i, 64i, scale < 1e-18), -64i, scale > 1e18);
        inv_adj = 1.0 / ldexp(scale, shift_s);
        let sv = vec4<i32>(shift_s);
        var usq = 0.0;
        if (valid) {
            for (var i = lid; i < units; i += 256u) {
                var v = V4(H[base + i]);
                if (!RMS) { v -= vec4<f32>(mean); }
                let u = ldexp(v, sv) * inv_adj;
                usq += lane_sum(u * u);
            }
        }
        let usq_tot = reduce_sum1(lid, sgid, lane, usq);
        let es = ldexp(eps_sqrt, shift_s) * inv_adj;
        inv_rms_s = inverseSqrt(usq_tot * inv_d + es * es);
        rstd = ldexp(inv_rms_s * inv_adj, shift_s);
    }
    if (!valid) { return; }
    if (lid == 0u) {
        if (!RMS) { MEAN[row] = mean; }
        RSTD[row] = rstd;
    }
    let sv = vec4<i32>(shift_s);
    for (var i = lid; i < units; i += 256u) {
        var v = V4(H[base + i]);
        if (!RMS) { v -= vec4<f32>(mean); }
        let xh = select(v * rstd, (ldexp(v, sv) * inv_adj) * inv_rms_s, need_scale);
        Y[base + i] = apply_affine(xh, i);
    }
}
