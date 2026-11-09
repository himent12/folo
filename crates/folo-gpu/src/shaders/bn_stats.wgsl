// Pass 1 of train-mode BatchNorm: per (channel, chunk) Welford partials (count, mean, M2) of NCHW X (f32 or f16,
// T; the statistics are f32).
// Grid: (chunks, C). Params: 0 = C, 1 = hw, 2 = N·hw, 3 = chunk length, 4 = chunks, 5 = 256 / hw,
// 6 = 256 % hw. PART is [C, chunks, 3].
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.y;
    let chunk = wid.x;
    let channels = pu(0u);
    let hw = pu(1u);
    let start = chunk * pu(3u);
    let end = min(start + pu(3u), pu(2u));
    var cnt = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    var w = bn_walk_start(start, lid, hw);
    loop {
        if (w.e >= end) { break; }
        let v = f32(X[(w.n * channels + c) * hw + w.i]);
        cnt += 1.0;
        let d = v - mean;
        mean += d / cnt;
        m2 += d * (v - mean);
        bn_walk_next(&w, hw);
    }
    wn[lid] = cnt;
    wm[lid] = mean;
    wq[lid] = m2;
    workgroupBarrier();
    bn_tree_merge(lid);
    if (lid == 0u) {
        let o = (c * pu(4u) + chunk) * 3u;
        PART[o] = wn[0];
        PART[o + 1u] = wm[0];
        PART[o + 2u] = wq[0];
    }
}
