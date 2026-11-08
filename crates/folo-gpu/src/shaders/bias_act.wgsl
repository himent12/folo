// y = act(x + bias[(i / inner) % channels]). Spec: OP (activation as a unary op code), HAS_BIAS, VEC.
// Params: 0 = element count (E units), 1 = alpha, 2 = channels, 3 = inner.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> BIAS: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    var v = V4(X[i]);
    if (HAS_BIAS) {
        let c = pu(2u);
        let inner = pu(3u);
        if (VEC) {
            let base = i * 4u;
            v += vec4<f32>(BIAS[(base / inner) % c], BIAS[((base + 1u) / inner) % c],
                           BIAS[((base + 2u) / inner) % c], BIAS[((base + 3u) / inner) % c]);
        } else {
            v += vec4<f32>(BIAS[(i / inner) % c]);
        }
    }
    Y[i] = OUT(unary4(OP, v, pf(1u)));
}
