// Pass 3 of the BatchNorm backward: DX = scale · dz + alpha · x + beta with dz = g · act'(x · scale + shift)
// and K = (scale, shift, alpha, beta) per channel. Spec: VEC4. Params: 0 = element count (E units), 1 = C,
// 2 = hw.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> DX: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    var lanes = 1u;
    if (VEC) { lanes = 4u; }
    let c = ((i * lanes) / pu(2u)) % pu(1u);
    let x = V4(X[i]);
    let scale = K[4u * c];
    var dz = V4(G[i]);
    if (ACT != 0u) {
        dz *= bn_act_grad(x * scale + K[4u * c + 1u]);
    }
    DX[i] = OUT(scale * dz + K[4u * c + 2u] * x + K[4u * c + 3u]);
}
