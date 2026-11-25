// Pass 2 of train-mode BatchNorm: merge the chunk partials of each channel, write the mean, invstd, the
// apply coefficients K = (mean, invstd, w, b) and, with RUNNING, the running-statistic update.
// `norm.rs` appends either the 1-barrier 32-wide pairwise Chan subgroupShuffleDown reduction or the portable
// 8-barrier shared-memory tree fallback.
// Grid: (C). Params: 0 = C, 1 = chunks, 2 = eps, 3 = 1 - momentum, 4 = momentum, 5 = momentum · n / (n - 1).
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> MEAN: array<f32>;
@group(0) @binding(5) var<storage, read_write> INVSTD: array<f32>;
@group(0) @binding(6) var<storage, read_write> K: array<f32>;
@group(0) @binding(7) var<storage, read_write> RM: array<f32>;
@group(0) @binding(8) var<storage, read_write> RV: array<f32>;

fn bn_finalize_thread(c: u32, lid: u32) -> vec3<f32> {
    let chunks = pu(1u);
    var cnt = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    for (var k = lid; k < chunks; k += 256u) {
        let o = (c * chunks + k) * 3u;
        let nb = PART[o];
        if (cnt == 0.0) {
            // The first non-empty partial is copied (see bn_tree_merge).
            cnt = nb;
            mean = PART[o + 1u];
            m2 = PART[o + 2u];
        } else if (nb > 0.0) {
            let n = cnt + nb;
            let d = PART[o + 1u] - mean;
            let delta = d * (nb / n);
            mean += delta;
            m2 += PART[o + 2u] + d * (delta * cnt);
            cnt = n;
        }
    }
    return vec3<f32>(cnt, mean, m2);
}

fn bn_finalize_write(c: u32, n: f32, m: f32, q: f32) {
    let var_b = max(q / n, 0.0);
    let inv = 1.0 / sqrt(var_b + pf(2u));
    MEAN[c] = m;
    INVSTD[c] = inv;
    K[4u * c] = m;
    K[4u * c + 1u] = inv;
    K[4u * c + 2u] = W[c];
    K[4u * c + 3u] = B[c];
    if (RUNNING) {
        RM[c] = RM[c] * pf(3u) + pf(4u) * m;
        RV[c] = RV[c] * pf(3u) + pf(5u) * var_b;
    }
}
