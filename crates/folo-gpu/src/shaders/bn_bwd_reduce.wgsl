// Pass 1 of the BatchNorm backward: per-thread partial sums of dz and dz · x̂, where
// x̂ = (x - mean) · invstd, z = x̂ · w + b and dz = g · act'(z); X and G are E (f32/f16 or vec4<f32>/vec4<f16>),
// the sums f32. Evaluates bn_act_grad on 4 real lanes when VEC4 is enabled (and scalar bn_act_grad_f32 on
// odd hw), eliminating the 4x redundant exp() computation. `norm.rs` appends either the 1-barrier 32-wide
// subgroupAdd reduction or the portable 8-barrier shared-memory tree fallback.
// Grid: (chunks, C). Params as bn_stats.wgsl. PART is [C, chunks, 2].
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
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var w = bn_walk_start(start, lid, hw);
    loop {
        if (w.e >= end) { break; }
        let idx = (w.n * channels + c) * hw + w.i;
        if (VEC) {
            let xh = (V4(X[idx]) - vec4<f32>(m)) * inv;
            var dz = V4(G[idx]);
            if (ACT != 0u) {
                dz *= bn_act_grad(xh * wc + vec4<f32>(bc));
            }
            a1 += dz;
            a2 += dz * xh;
        } else {
            let xh = (V4(X[idx]).x - m) * inv;
            var dz = V4(G[idx]).x;
            if (ACT != 0u) {
                dz *= bn_act_grad_f32(xh * wc + bc);
            }
            a1.x += dz;
            a2.x += dz * xh;
        }
        bn_walk_next(&w, hw);
    }
    return vec2<f32>(
        select(a1.x, (a1.x + a1.y) + (a1.z + a1.w), VEC),
        select(a2.x, (a2.x + a2.y) + (a2.z + a2.w), VEC),
    );
}
