// Softmax / log-softmax over the middle axis of [outer, D, inner] with inner > 1: one thread per (outer, inner)
// column, so loads are coalesced across columns. A non-finite column maximum is replaced by 0, as on the CPU.
// Spec: LOG. Params: 0 = columns (outer * inner), 1 = D, 2 = inner, 3 = -inf. XT / YT are the storage types of
// X and Y (f32 or f16); the arithmetic is f32.
@group(0) @binding(1) var<storage, read> X: array<XT>;
@group(0) @binding(2) var<storage, read_write> Y: array<YT>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = linear_id(wid, nwg, lid, 256u);
    if (t >= pu(0u)) { return; }
    let d = pu(1u);
    let inner = pu(2u);
    let base = (t / inner) * d * inner + t % inner;
    var m = pf(3u);
    for (var i = 0u; i < d; i++) { m = max(m, f32(X[base + i * inner])); }
    if ((bitcast<u32>(m) & 0x7fffffffu) >= 0x7f800000u) { m = 0.0; }
    var s = 0.0;
    for (var i = 0u; i < d; i++) { s += exp(f32(X[base + i * inner]) - m); }
    if (LOG) {
        let ls = log(s);
        for (var i = 0u; i < d; i++) { let o = base + i * inner; Y[o] = YT(f32(X[o]) - m - ls); }
    } else {
        let inv = 1.0 / s;
        for (var i = 0u; i < d; i++) { let o = base + i * inner; Y[o] = YT(exp(f32(X[o]) - m) * inv); }
    }
}
