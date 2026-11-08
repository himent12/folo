// Pass 2 of train-mode BatchNorm: merge the chunk partials of each channel, write the mean, invstd, the
// apply coefficients K = (scale, shift) and, with RUNNING, the running-statistic update.
// Grid: (C). Params: 0 = C, 1 = chunks, 2 = eps, 3 = 1 - momentum, 4 = momentum, 5 = momentum · n / (n - 1).
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> MEAN: array<f32>;
@group(0) @binding(5) var<storage, read_write> INVSTD: array<f32>;
@group(0) @binding(6) var<storage, read_write> K: array<f32>;
@group(0) @binding(7) var<storage, read_write> RM: array<f32>;
@group(0) @binding(8) var<storage, read_write> RV: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x;
    let chunks = pu(1u);
    var cnt = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    for (var k = lid; k < chunks; k += 256u) {
        let o = (c * chunks + k) * 3u;
        let nb = PART[o];
        if (nb > 0.0) {
            let n = cnt + nb;
            let d = PART[o + 1u] - mean;
            mean += d * (nb / n);
            m2 += PART[o + 2u] + d * d * (cnt * nb / n);
            cnt = n;
        }
    }
    wn[lid] = cnt;
    wm[lid] = mean;
    wq[lid] = m2;
    workgroupBarrier();
    bn_tree_merge(lid);
    if (lid == 0u) {
        let m = wm[0];
        let var_b = max(wq[0] / wn[0], 0.0);
        let inv = 1.0 / sqrt(var_b + pf(2u));
        MEAN[c] = m;
        INVSTD[c] = inv;
        let scale = inv * W[c];
        K[2u * c] = scale;
        K[2u * c + 1u] = B[c] - m * scale;
        if (RUNNING) {
            RM[c] = RM[c] * pf(3u) + pf(4u) * m;
            RV[c] = RV[c] * pf(3u) + pf(5u) * var_b;
        }
    }
}
