// Distillation, gradient of the class term: dx = g · S · w · (sigmoid(x) − p_t), with g = GRAD[0],
// S = B · cls_gain / N, and w = |sigmoid(x) − p_t|^gamma when bckd_gamma > 0 (1 otherwise).
// Spec: TF16, HAS_GT. Params: 0 = elements (B·NC·A), 1 = S (f32), 2 = A, 3 = NC, 4 = B·A, 5 = bckd_gamma.
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> TC: array<TW>;
@group(0) @binding(3) var<storage, read> GRAD: array<f32>;
@group(0) @binding(4) var<storage, read_write> DX: array<f32>;
@group(0) @binding(5) var<storage, read> GT_DENSE: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let ps = sigmoid_f(X[i]);
    var pt = sigmoid_f(tv(TC[ti(i)], i));
    if (HAS_GT) {
        let na = pu(2u);
        let row = i / na;
        let a = i - row * na;
        let b = row / pu(3u);
        let c = row - b * pu(3u);
        let cell = b * na + a;
        if (GT_DENSE[cell] == c + 1u) {
            let t_gt = bitcast<f32>(GT_DENSE[pu(4u) + cell]);
            if (t_gt > pt) { pt = t_gt; }
        }
    }
    var diff = ps - pt;
    let gamma = pf(5u);
    if (gamma > 0.0) {
        diff = pow_pos(abs(diff), gamma) * diff;
    }
    DX[i] = GRAD[0] * pf(1u) * diff;
}
