// Pass 1 of the BatchNorm backward: per-thread partial sums of dz and dz · x̂, where
// x̂ = (x - mean) · invstd, z = x̂ · w + b and dz = g · act'(z); X and G are E (f32/f16 or vec4<f32>/vec4<f16>),
// the sums f32. Evaluates bn_act_grad on 4 real lanes when VEC4 is enabled (and scalar bn_act_grad_f32 on
// odd hw), eliminating the 4x redundant exp() computation. `norm.rs` appends either the 1-barrier 32-wide
// subgroupAdd reduction or the portable 8-barrier shared-memory tree fallback.
// Grid: (C, chunks). Params: 0..6 as bn_stats.wgsl; when STRIDED: 7 = x_off, 8 = x_n_stride, 9 = g_off,
// 10 = g_n_stride (all in E units). PART is [C, chunks, 2].
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> INVSTD: array<f32>;
@group(0) @binding(5) var<storage, read> W: array<f32>;
@group(0) @binding(6) var<storage, read> B: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

fn bn_bwd_reduce_thread(c: u32, chunk: u32, lid: u32) -> vec2<f32> {
    let channels = pu(0u);
    let hw = pu(1u);
    let start = chunk * pu(3u);
    let end = min(start + pu(3u), pu(2u));
    let m = MEAN[c];
    let inv = INVSTD[c];
    let wc = W[c];
    let bc = B[c];
    let step_n = pu(5u);
    let step_i = pu(6u);
    let ch_off = c * hw;
    let x_ns = select(channels * hw, pu(8u), IS_STRIDED);
    let g_ns = select(channels * hw, pu(10u), IS_STRIDED);
    let x_step = step_n * x_ns + step_i;
    let g_step = step_n * g_ns + step_i;
    let x_gap = x_ns - hw;
    let g_gap = g_ns - hw;
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var e = start + lid;
    let n0 = e / hw;
    var wi = e - n0 * hw;
    var x_idx = select(0u, pu(7u), IS_STRIDED) + n0 * x_ns + ch_off + wi;
    var g_idx = select(0u, pu(9u), IS_STRIDED) + n0 * g_ns + ch_off + wi;
    if (VEC) {
        loop {
            if (e + 256u >= end) { break; }
            let xi0 = x_idx;
            let gi0 = g_idx;
            e += 256u;
            wi += step_i;
            x_idx += x_step;
            if (IS_STRIDED) { g_idx += g_step; }
            if (wi >= hw) {
                wi -= hw;
                x_idx += x_gap;
                if (IS_STRIDED) { g_idx += g_gap; }
            }
            let xi1 = x_idx;
            let gi1 = select(x_idx, g_idx, IS_STRIDED);
            e += 256u;
            wi += step_i;
            x_idx += x_step;
            if (IS_STRIDED) { g_idx += g_step; }
            if (wi >= hw) {
                wi -= hw;
                x_idx += x_gap;
                if (IS_STRIDED) { g_idx += g_gap; }
            }
            let xv0 = V4(X[xi0]);
            let gv0 = V4(G[select(xi0, gi0, IS_STRIDED)]);
            let xv1 = V4(X[xi1]);
            let gv1 = V4(G[gi1]);
            let xh0 = (xv0 - vec4<f32>(m)) * inv;
            var dz0 = gv0;
            if (ACT != 0u) { dz0 *= bn_act_grad(xh0 * wc + vec4<f32>(bc)); }
            let xh1 = (xv1 - vec4<f32>(m)) * inv;
            var dz1 = gv1;
            if (ACT != 0u) { dz1 *= bn_act_grad(xh1 * wc + vec4<f32>(bc)); }
            a1 += dz0 + dz1;
            a2 += dz0 * xh0 + dz1 * xh1;
        }
        if (e < end) {
            let xh = (V4(X[x_idx]) - vec4<f32>(m)) * inv;
            var dz = V4(G[select(x_idx, g_idx, IS_STRIDED)]);
            if (ACT != 0u) { dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }
            a1 += dz;
            a2 += dz * xh;
        }
    } else {
        loop {
            if (e >= end) { break; }
            let xh = (V4(X[x_idx]).x - m) * inv;
            var dz = V4(G[select(x_idx, g_idx, IS_STRIDED)]).x;
            if (ACT != 0u) { dz *= bn_act_grad_f32(xh * wc + bc); }
            a1.x += dz;
            a2.x += dz * xh;
            e += 256u;
            wi += step_i;
            x_idx += x_step;
            if (IS_STRIDED) { g_idx += g_step; }
            if (wi >= hw) {
                wi -= hw;
                x_idx += x_gap;
                if (IS_STRIDED) { g_idx += g_gap; }
            }
        }
    }
    return vec2<f32>(
        select(a1.x, (a1.x + a1.y) + (a1.z + a1.w), VEC),
        select(a2.x, (a2.x + a2.y) + (a2.z + a2.w), VEC),
    );
}
