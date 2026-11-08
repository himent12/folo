// Pass 1 of the BatchNorm backward: per (channel, chunk) partial sums of dz and dz · x̂, where
// x̂ = (x - mean) · invstd, z = x̂ · w + b and dz = g · act'(z). Grid: (chunks, C). Params as bn_stats.wgsl.
// PART is [C, chunks, 2].
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> INVSTD: array<f32>;
@group(0) @binding(5) var<storage, read> W: array<f32>;
@group(0) @binding(6) var<storage, read> B: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> s1: array<f32, 256>;
var<workgroup> s2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.y;
    let chunk = wid.x;
    let channels = pu(0u);
    let hw = pu(1u);
    let start = chunk * pu(3u);
    let end = min(start + pu(3u), pu(2u));
    let m = MEAN[c];
    let inv = INVSTD[c];
    let wc = W[c];
    let bc = B[c];
    var a1 = 0.0;
    var a2 = 0.0;
    var w = bn_walk_start(start, lid, hw);
    loop {
        if (w.e >= end) { break; }
        let idx = (w.n * channels + c) * hw + w.i;
        let xh = (X[idx] - m) * inv;
        var dz = G[idx];
        if (ACT != 0u) {
            dz *= bn_act_grad(vec4<f32>(xh * wc + bc)).x;
        }
        a1 += dz;
        a2 += dz * xh;
        bn_walk_next(&w, hw);
    }
    s1[lid] = a1;
    s2[lid] = a2;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            s1[lid] += s1[lid + s];
            s2[lid] += s2[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let o = (c * pu(4u) + chunk) * 2u;
        PART[o] = s1[0];
        PART[o + 1u] = s2[0];
    }
}
