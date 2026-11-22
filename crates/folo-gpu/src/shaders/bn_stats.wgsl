// Pass 1 of train-mode BatchNorm: per-thread block-shifted 2-moment accumulation over NCHW X (E is f32/f16
// or vec4<f32>/vec4<f16>; statistics are f32). Zero divisions in the hot loop and zero catastrophic cancellation;
// `norm.rs` appends either the 1-barrier 32-wide subgroup Chan merge (when `SUBGROUPS` is true) or the portable
// 8-barrier shared-memory `bn_tree_merge` fallback.
// Grid: (chunks, C). Params: 0 = C, 1 = hw_units, 2 = N·hw_units, 3 = chunk length, 4 = chunks,
// 5 = 256 / hw_units, 6 = 256 % hw_units. PART is [C, chunks, 3].
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;

fn bn_stats_thread(c: u32, chunk: u32, lid: u32) -> vec3<f32> {
    let channels = pu(0u);
    let hw = pu(1u);
    let start = chunk * pu(3u);
    let end = min(start + pu(3u), pu(2u));
    var has_v0 = false;
    var v0 = 0.0;
    var cnt = 0.0;
    var s1 = vec4<f32>(0.0);
    var s2 = vec4<f32>(0.0);
    var w = bn_walk_start(start, lid, hw);
    loop {
        if (w.e >= end) { break; }
        let idx = (w.n * channels + c) * hw + w.i;
        let x = V4(X[idx]);
        if (!has_v0) {
            v0 = x.x;
            has_v0 = true;
        }
        if (VEC) {
            let dv = x - vec4<f32>(v0);
            s1 += dv;
            s2 += dv * dv;
            cnt += 4.0;
        } else {
            let dv = x.x - v0;
            s1.x += dv;
            s2.x += dv * dv;
            cnt += 1.0;
        }
        bn_walk_next(&w, hw);
    }
    var mean = 0.0;
    var m2 = 0.0;
    if (cnt > 0.0) {
        let sum1 = select(s1.x, (s1.x + s1.y) + (s1.z + s1.w), VEC);
        let sum2 = select(s2.x, (s2.x + s2.y) + (s2.z + s2.w), VEC);
        let off = sum1 / cnt;
        mean = v0 + off;
        m2 = max(sum2 - sum1 * off, 0.0);
    }
    return vec3<f32>(cnt, mean, m2);
}
