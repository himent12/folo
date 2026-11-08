// f32 <-> f16 conversion on packed words (works without shader-f16). Spec: TO_F16.
// Params: 0 = element count. One invocation handles two elements (one u32 of f16 pairs).
@group(0) @binding(1) var<storage, read> X: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    let n = pu(0u);
    if (2u * i >= n) { return; }
    let two = 2u * i + 1u < n;
    if (TO_F16) {
        let a = bitcast<f32>(X[2u * i]);
        let b = select(0.0, bitcast<f32>(X[2u * i + 1u]), two);
        Y[i] = pack2x16float(vec2<f32>(a, b));
    } else {
        let v = unpack2x16float(X[i]);
        Y[2u * i] = bitcast<u32>(v.x);
        if (two) { Y[2u * i + 1u] = bitcast<u32>(v.y); }
    }
}
