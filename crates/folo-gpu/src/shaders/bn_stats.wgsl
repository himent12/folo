// Pass 1 of train-mode BatchNorm: per-thread vectorized Welford accumulation over NCHW X (E is f32/f16 or
// vec4<f32>/vec4<f16>; statistics are f32). Using 4-lane Welford keeps 1 vec4 load and 1 scalar reciprocal per 4
// elements while avoiding both `sum2` overflow on extreme values (`1e19`) and first-element outlier cancellation;
// `norm.rs` appends either the 1-barrier 32-wide subgroup Chan merge (when `SUBGROUPS` is true) or the portable
// 8-barrier shared-memory `bn_tree_merge` fallback.
// Grid: (C, chunks). Params: 0 = C, 1 = hw_units, 2 = N·hw_units, 3 = chunk length, 4 = chunks,
// 5 = 256 / hw_units, 6 = 256 % hw_units; when STRIDED: 7 = x_off, 8 = x_n_stride (E units). PART is [C, chunks, 3].
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;

fn bn_stats_thread(c: u32, chunk: u32, lid: u32) -> vec3<f32> {
    let channels = pu(0u);
    let hw = pu(1u);
    let start = chunk * pu(3u);
    let end = min(start + pu(3u), pu(2u));
    let step_n = pu(5u);
    let step_i = pu(6u);
    let x_ns = select(channels * hw, pu(8u), IS_STRIDED);
    let x_step = step_n * x_ns + step_i;
    let x_gap = x_ns - hw;
    var k = 0.0;
    var m4 = vec4<f32>(0.0);
    var q4 = vec4<f32>(0.0);
    var e = start + lid;
    let n0 = e / hw;
    var wi = e - n0 * hw;
    var idx = select(0u, pu(7u), IS_STRIDED) + n0 * x_ns + c * hw + wi;
    loop {
        if (e >= end) { break; }
        let x = V4(X[idx]);
        k += 1.0;
        let inv_k = 1.0 / k;
        if (VEC) {
            let d = x - m4;
            m4 += d * inv_k;
            q4 += d * (x - m4);
        } else {
            let d = x.x - m4.x;
            m4.x += d * inv_k;
            q4.x += d * (x.x - m4.x);
        }
        e += 256u;
        wi += step_i;
        idx += x_step;
        if (wi >= hw) {
            wi -= hw;
            idx += x_gap;
        }
    }
    if (k == 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    if (!VEC) {
        return vec3<f32>(k, m4.x, q4.x);
    }
    // Exact equal-count Chan merge of the 4 vector lanes (each has count `k`).
    let d01 = m4.y - m4.x;
    let m01 = m4.x + 0.5 * d01;
    let q01 = q4.x + q4.y + (0.5 * k * d01) * d01;
    let d23 = m4.w - m4.z;
    let m23 = m4.z + 0.5 * d23;
    let q23 = q4.z + q4.w + (0.5 * k * d23) * d23;
    let d_all = m23 - m01;
    let mean = m01 + 0.5 * d_all;
    let m2 = q01 + q23 + (k * d_all) * d_all;
    return vec3<f32>(4.0 * k, mean, m2);
}
