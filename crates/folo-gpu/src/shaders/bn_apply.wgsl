// Pass 3 of train-mode BatchNorm: Y = act(X · scale[c] + shift[c]) with K = (scale, shift) per channel.
// Spec: VEC4 (E is vec4 when hw % 4 == 0, so the four lanes share a channel). Params: 0 = element count
// (E units), 1 = C, 2 = hw.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    var lanes = 1u;
    if (VEC) { lanes = 4u; }
    let c = ((i * lanes) / pu(2u)) % pu(1u);
    Y[i] = OUT(bn_act(V4(X[i]) * K[2u * c] + K[2u * c + 1u]));
}
