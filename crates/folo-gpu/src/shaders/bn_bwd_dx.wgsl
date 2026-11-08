// Pass 3 of the BatchNorm backward: DX = w · invstd · (dz − db / n − x̂ · dw / n) with x̂ = (x − mean) · invstd and
// dz = g · act'(x̂ · w + b), the same pre-activation expression as the forward and the reduction pass.
// K = (mean, invstd, w, b, db / n, dw / n) per channel. Spec: VEC4. Params: 0 = element count (E units), 1 = C,
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
    let k = 6u * c;
    let inv = K[k + 1u];
    let w = K[k + 2u];
    let xh = (V4(X[i]) - K[k]) * inv;
    var dz = V4(G[i]);
    if (ACT != 0u) {
        dz *= bn_act_grad(xh * w + K[k + 3u]);
    }
    DX[i] = OUT((w * inv) * (dz - K[k + 4u] - xh * K[k + 5u]));
}
