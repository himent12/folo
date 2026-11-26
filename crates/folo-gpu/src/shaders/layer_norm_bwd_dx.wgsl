// Backward `dx` kernel for LayerNorm (`!RMS`) and RMSNorm (`RMS`) over dense `[outer, d]` (`F32` or `F16` activations, `F32` params/stats).
// `layer_norm.rs` prepends `vec_prelude` / `vec_prelude_f16`, `param_f32_prelude`, `CACHE_ROW`, `LANE_HELPERS`, reduction helpers, and `@compute fn main`.
// Params: 0 = rows (`outer`), 1 = units (`d / 4` when `VEC4` else `d`), 2 = `inv_d`.
// Bindings: 1 = X, 2 = G, 3 = W, 4 = MEAN, 5 = RSTD, 6 = DX.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> W: array<PF>;
@group(0) @binding(4) var<storage, read> MEAN: array<f32>;
@group(0) @binding(5) var<storage, read> RSTD: array<f32>;
@group(0) @binding(6) var<storage, read_write> DX: array<E>;

fn norm_bwd_dx_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let units = pu(1u);
    let inv_d = pf(2u);
    let valid = row < rows;
    let base = row * units;
    var mean = 0.0;
    var rstd = 0.0;
    if (valid) {
        if (!RMS) { mean = MEAN[row]; }
        rstd = RSTD[row];
    }

    if (CACHE_ROW) {
        var xh0 = vec4<f32>(0.0); var gw0 = vec4<f32>(0.0);
        var xh1 = vec4<f32>(0.0); var gw1 = vec4<f32>(0.0);
        var xh2 = vec4<f32>(0.0); var gw2 = vec4<f32>(0.0);
        var xh3 = vec4<f32>(0.0); var gw3 = vec4<f32>(0.0);
        let i0 = lid;
        let i1 = lid + 256u;
        let i2 = lid + 512u;
        let i3 = lid + 768u;
        let h0 = valid && i0 < units;
        let h1 = valid && i1 < units;
        let h2 = valid && i2 < units;
        let h3 = valid && i3 < units;
        var s1 = 0.0;
        var s2 = 0.0;
        var s3 = 0.0;
        if (h0) {
            xh0 = norm_xh(V4(X[base + i0]), mean, rstd);
            gw0 = V4(G[base + i0]);
            if (HAS_W) { gw0 *= PF4(W[i0]); }
            s2 += lane_sum(gw0 * xh0);
            if (!RMS) { s1 += lane_sum(gw0); s3 += lane_sum(xh0); }
        }
        if (h1) {
            xh1 = norm_xh(V4(X[base + i1]), mean, rstd);
            gw1 = V4(G[base + i1]);
            if (HAS_W) { gw1 *= PF4(W[i1]); }
            s2 += lane_sum(gw1 * xh1);
            if (!RMS) { s1 += lane_sum(gw1); s3 += lane_sum(xh1); }
        }
        if (h2) {
            xh2 = norm_xh(V4(X[base + i2]), mean, rstd);
            gw2 = V4(G[base + i2]);
            if (HAS_W) { gw2 *= PF4(W[i2]); }
            s2 += lane_sum(gw2 * xh2);
            if (!RMS) { s1 += lane_sum(gw2); s3 += lane_sum(xh2); }
        }
        if (h3) {
            xh3 = norm_xh(V4(X[base + i3]), mean, rstd);
            gw3 = V4(G[base + i3]);
            if (HAS_W) { gw3 *= PF4(W[i3]); }
            s2 += lane_sum(gw3 * xh3);
            if (!RMS) { s1 += lane_sum(gw3); s3 += lane_sum(xh3); }
        }
        if (RMS) {
            let c_d = reduce_sum1(lid, sgid, lane, s2) * inv_d;
            if (!valid) { return; }
            if (h0) { DX[base + i0] = OUT(rstd * (gw0 - xh0 * c_d)); }
            if (h1) { DX[base + i1] = OUT(rstd * (gw1 - xh1 * c_d)); }
            if (h2) { DX[base + i2] = OUT(rstd * (gw2 - xh2 * c_d)); }
            if (h3) { DX[base + i3] = OUT(rstd * (gw3 - xh3 * c_d)); }
        } else {
            let sums = reduce_sum3(lid, sgid, lane, s1, s2, s3);
            if (!valid) { return; }
            let c2_d = sums.y * inv_d;
            let c1_adj = sums.x * inv_d - (sums.z * inv_d) * c2_d;
            if (h0) { DX[base + i0] = OUT(rstd * (gw0 - (vec4<f32>(c1_adj) + xh0 * c2_d))); }
            if (h1) { DX[base + i1] = OUT(rstd * (gw1 - (vec4<f32>(c1_adj) + xh1 * c2_d))); }
            if (h2) { DX[base + i2] = OUT(rstd * (gw2 - (vec4<f32>(c1_adj) + xh2 * c2_d))); }
            if (h3) { DX[base + i3] = OUT(rstd * (gw3 - (vec4<f32>(c1_adj) + xh3 * c2_d))); }
        }
        return;
    }

    var s1 = 0.0;
    var s2 = 0.0;
    var s3 = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            let xh = norm_xh(V4(X[base + i]), mean, rstd);
            var gw = V4(G[base + i]);
            if (HAS_W) { gw *= PF4(W[i]); }
            s2 += lane_sum(gw * xh);
            if (!RMS) { s1 += lane_sum(gw); s3 += lane_sum(xh); }
        }
    }
    if (RMS) {
        let c_d = reduce_sum1(lid, sgid, lane, s2) * inv_d;
        if (!valid) { return; }
        for (var i = lid; i < units; i += 256u) {
            let xh = norm_xh(V4(X[base + i]), 0.0, rstd);
            var gw = V4(G[base + i]);
            if (HAS_W) { gw *= PF4(W[i]); }
            DX[base + i] = OUT(rstd * (gw - xh * c_d));
        }
    } else {
        let sums = reduce_sum3(lid, sgid, lane, s1, s2, s3);
        if (!valid) { return; }
        let c2_d = sums.y * inv_d;
        let c1_adj = sums.x * inv_d - (sums.z * inv_d) * c2_d;
        for (var i = lid; i < units; i += 256u) {
            let xh = norm_xh(V4(X[base + i]), mean, rstd);
            var gw = V4(G[base + i]);
            if (HAS_W) { gw *= PF4(W[i]); }
            DX[base + i] = OUT(rstd * (gw - (vec4<f32>(c1_adj) + xh * c2_d)));
        }
    }
}
