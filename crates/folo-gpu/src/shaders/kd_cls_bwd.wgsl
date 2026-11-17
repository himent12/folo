// Distillation, gradient of the class term: dx = g · S · (sigmoid(x) − sigmoid(y)), with g the upstream gradient
// of the total (GRAD[0]) and S = B · cls_gain / N.
// Params: 0 = elements (B·NC·A), 1 = S (f32).
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> TC: array<TW>;
@group(0) @binding(3) var<storage, read> GRAD: array<f32>;
@group(0) @binding(4) var<storage, read_write> DX: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    DX[i] = GRAD[0] * pf(1u) * (sigmoid_f(X[i]) - sigmoid_f(tv(TC[ti(i)], i)));
}
