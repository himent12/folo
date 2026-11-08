// Index of the first maximum (MAX) or minimum over the middle axis of [outer, R, inner], written as i64 (low
// word, then 0). NaN beats everything and the first NaN wins, as on the CPU. Params: 0 = outputs, 1 = R, 2 = inner.
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<u32>;

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = linear_id(wid, nwg, lid, 256u);
    if (t >= pu(0u)) { return; }
    let r = pu(1u);
    let inner = pu(2u);
    let base = (t / inner) * r * inner + t % inner;
    var best = X[base];
    var bi = 0u;
    for (var i = 1u; i < r; i++) {
        if (isnan1(best)) { break; }
        let v = X[base + i * inner];
        var better = isnan1(v);
        if (MAX) { better = better || v > best; } else { better = better || v < best; }
        if (better) { best = v; bi = i; }
    }
    Y[2u * t] = bi;
    Y[2u * t + 1u] = 0u;
}
