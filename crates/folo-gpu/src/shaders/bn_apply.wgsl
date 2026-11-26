// Pass 3 of train-mode BatchNorm: Y = act(((X - mean) · invstd) · w + b) with K[c] = (mean, invstd, w, b) per
// channel. Centring first keeps inputs with a large mean and a small spread exact (the expanded
// x · scale + shift form cancels catastrophically), and the backward recomputes z with the same expression.
// Spec: VEC4 (E is vec4 when hw % 4 == 0, so the four lanes share a channel). Params: 0 = element count
// (E units), 1 = C, 2 = hw_units.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let c = (i / pu(2u)) % pu(1u);
    let kv = K[c];
    if (VEC) {
        let xh = (V4(X[i]) - kv.x) * kv.y;
        Y[i] = OUT(bn_act(xh * kv.z + kv.w));
    } else {
        let xh = (V4(X[i]).x - kv.x) * kv.y;
        Y[i] = OUT(vec4<f32>(bn_act_f32(xh * kv.z + kv.w)));
    }
}
