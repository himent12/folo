// Backward `dh = dx_norm + grad_res` (optional `grad_res`), plus per-row `mu_xh` for exact `dweight`.
// `residual_norm.rs` prepends `vec_prelude(_f16)`, `param_f32_prelude`, `CACHE_ROW`, `LANE_HELPERS`, reduction helpers, and `main`.
// Params: 0 = `rows`, 1 = `units`, 2 = `inv_d`.
@group(0) @binding(1) var<storage, read> H: array<E>;
@group(0) @binding(2) var<storage, read> GY: array<E>;
@group(0) @binding(3) var<storage, read> GRES: array<E>;
@group(0) @binding(4) var<storage, read> W: array<PF>;
@group(0) @binding(5) var<storage, read> MEAN: array<f32>;
@group(0) @binding(6) var<storage, read> RSTD: array<f32>;
@group(0) @binding(7) var<storage, read_write> DH: array<E>;
@group(0) @binding(8) var<storage, read_write> MU_XH: array<f32>;

fn add_gres(dx: vec4<f32>, idx: u32) -> E {
    if (HAS_GRES) { return OUT(dx + V4(GRES[idx])); }
    return OUT(dx);
}

fn res_norm_bwd_dx_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let units = pu(1u);
    let inv_d = pf(2u);
    let valid = row < rows;
    let base = row * units;
    var mean = 0.0; var rstd = 0.0;
    if (valid) {
        if (!RMS) { mean = MEAN[row]; }
        rstd = RSTD[row];
    }

    if (CACHE_ROW) {
        var xh0 = vec4<f32>(0.0); var gw0 = vec4<f32>(0.0);
        var xh1 = vec4<f32>(0.0); var gw1 = vec4<f32>(0.0);
        var xh2 = vec4<f32>(0.0); var gw2 = vec4<f32>(0.0);
        var xh3 = vec4<f32>(0.0); var gw3 = vec4<f32>(0.0);
        let i0 = lid; let i1 = lid + 256u; let i2 = lid + 512u; let i3 = lid + 768u;
        let h0 = valid && i0 < units; let h1 = valid && i1 < units;
        let h2 = valid && i2 < units; let h3 = valid && i3 < units;
        var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        if (h0) {
            xh0 = norm_xh(V4(H[base + i0]), mean, rstd);
            gw0 = V4(GY[base + i0]); if (HAS_W) { gw0 *= PF4(W[i0]); }
            s2 += lane_sum(gw0 * xh0);
            if (!RMS) { s1 += lane_sum(gw0); s3 += lane_sum(xh0); }
        }
        if (h1) {
            xh1 = norm_xh(V4(H[base + i1]), mean, rstd);
            gw1 = V4(GY[base + i1]); if (HAS_W) { gw1 *= PF4(W[i1]); }
            s2 += lane_sum(gw1 * xh1);
            if (!RMS) { s1 += lane_sum(gw1); s3 += lane_sum(xh1); }
        }
        if (h2) {
            xh2 = norm_xh(V4(H[base + i2]), mean, rstd);
            gw2 = V4(GY[base + i2]); if (HAS_W) { gw2 *= PF4(W[i2]); }
            s2 += lane_sum(gw2 * xh2);
            if (!RMS) { s1 += lane_sum(gw2); s3 += lane_sum(xh2); }
        }
        if (h3) {
            xh3 = norm_xh(V4(H[base + i3]), mean, rstd);
            gw3 = V4(GY[base + i3]); if (HAS_W) { gw3 *= PF4(W[i3]); }
            s2 += lane_sum(gw3 * xh3);
            if (!RMS) { s1 += lane_sum(gw3); s3 += lane_sum(xh3); }
        }
        if (RMS) {
            let c_d = reduce_sum1(lid, sgid, lane, s2) * inv_d;
            if (!valid) { return; }
            if (lid == 0u) { MU_XH[row] = 0.0; }
            if (h0) { DH[base + i0] = add_gres(rstd * (gw0 - xh0 * c_d), base + i0); }
            if (h1) { DH[base + i1] = add_gres(rstd * (gw1 - xh1 * c_d), base + i1); }
            if (h2) { DH[base + i2] = add_gres(rstd * (gw2 - xh2 * c_d), base + i2); }
            if (h3) { DH[base + i3] = add_gres(rstd * (gw3 - xh3 * c_d), base + i3); }
        } else {
            let sums = reduce_sum3(lid, sgid, lane, s1, s2, s3);
            if (!valid) { return; }
            let mu_xh = sums.z * inv_d;
            if (lid == 0u) { MU_XH[row] = mu_xh; }
            let c1 = sums.x * inv_d;
            let c2 = (sums.y - mu_xh * sums.x) * inv_d;
            let c1v = vec4<f32>(c1);
            let muv = vec4<f32>(mu_xh);
            if (h0) { DH[base + i0] = add_gres(rstd * (gw0 - c1v - (xh0 - muv) * c2), base + i0); }
            if (h1) { DH[base + i1] = add_gres(rstd * (gw1 - c1v - (xh1 - muv) * c2), base + i1); }
            if (h2) { DH[base + i2] = add_gres(rstd * (gw2 - c1v - (xh2 - muv) * c2), base + i2); }
            if (h3) { DH[base + i3] = add_gres(rstd * (gw3 - c1v - (xh3 - muv) * c2), base + i3); }
        }
        return;
    }

    var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            let xh = norm_xh(V4(H[base + i]), mean, rstd);
            var gw = V4(GY[base + i]); if (HAS_W) { gw *= PF4(W[i]); }
            s2 += lane_sum(gw * xh);
            if (!RMS) { s1 += lane_sum(gw); s3 += lane_sum(xh); }
        }
    }
    if (RMS) {
        let c_d = reduce_sum1(lid, sgid, lane, s2) * inv_d;
        if (!valid) { return; }
        if (lid == 0u) { MU_XH[row] = 0.0; }
        for (var i = lid; i < units; i += 256u) {
            let xh = norm_xh(V4(H[base + i]), mean, rstd);
            var gw = V4(GY[base + i]); if (HAS_W) { gw *= PF4(W[i]); }
            DH[base + i] = add_gres(rstd * (gw - xh * c_d), base + i);
        }
    } else {
        let sums = reduce_sum3(lid, sgid, lane, s1, s2, s3);
        if (!valid) { return; }
        let mu_xh = sums.z * inv_d;
        if (lid == 0u) { MU_XH[row] = mu_xh; }
        let c1 = sums.x * inv_d;
        let c2 = (sums.y - mu_xh * sums.x) * inv_d;
        let c1v = vec4<f32>(c1);
        let muv = vec4<f32>(mu_xh);
        for (var i = lid; i < units; i += 256u) {
            let xh = norm_xh(V4(H[base + i]), mean, rstd);
            var gw = V4(GY[base + i]); if (HAS_W) { gw *= PF4(W[i]); }
            DH[base + i] = add_gres(rstd * (gw - c1v - (xh - muv) * c2), base + i);
        }
    }
}
