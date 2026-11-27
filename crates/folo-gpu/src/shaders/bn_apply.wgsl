// Pass 3 of train-mode BatchNorm: Y = act(((X - mean) · invstd) · w + b) with K[c] = (mean, invstd, w, b) per
// channel. Centring first keeps inputs with a large mean and a small spread exact (the expanded
// x · scale + shift form cancels catastrophically), and the backward recomputes z with the same expression.
// Spec: VEC4, GRID3D, STRIDED. Params: 0 = element count (E units), 1 = C, 2 = hw_units; when STRIDED:
// 3 = x_off, 4 = x_n_stride (E units). When GRID3D, grid is (ceil(hw_units / 512), C, N), 2 units/thread.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let channels = pu(1u);
    let hw = pu(2u);
    if (USE_GRID3D) {
        let j0 = wid.x * 512u + lid;
        if (j0 >= hw) { return; }
        let c = wid.y;
        let n = wid.z;
        let kv = K[c];
        let ch_j0 = c * hw + j0;
        let y_i0 = n * channels * hw + ch_j0;
        let x_i0 = select(y_i0, pu(3u) + n * pu(4u) + ch_j0, IS_STRIDED);
        let j1 = j0 + 256u;
        if (j1 < hw) {
            let x_i1 = x_i0 + 256u;
            let y_i1 = y_i0 + 256u;
            if (VEC) {
                let v0 = V4(X[x_i0]);
                let v1 = V4(X[x_i1]);
                Y[y_i0] = OUT(bn_act(((v0 - kv.x) * kv.y) * kv.z + kv.w));
                Y[y_i1] = OUT(bn_act(((v1 - kv.x) * kv.y) * kv.z + kv.w));
            } else {
                let v0 = V4(X[x_i0]).x;
                let v1 = V4(X[x_i1]).x;
                Y[y_i0] = OUT(vec4<f32>(bn_act_f32(((v0 - kv.x) * kv.y) * kv.z + kv.w)));
                Y[y_i1] = OUT(vec4<f32>(bn_act_f32(((v1 - kv.x) * kv.y) * kv.z + kv.w)));
            }
        } else {
            if (VEC) {
                let xh = (V4(X[x_i0]) - kv.x) * kv.y;
                Y[y_i0] = OUT(bn_act(xh * kv.z + kv.w));
            } else {
                let xh = (V4(X[x_i0]).x - kv.x) * kv.y;
                Y[y_i0] = OUT(vec4<f32>(bn_act_f32(xh * kv.z + kv.w)));
            }
        }
        return;
    }
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let p = i / hw;
    let j = i - p * hw;
    let n = p / channels;
    let c = p - n * channels;
    let x_i = select(i, pu(3u) + n * pu(4u) + c * hw + j, IS_STRIDED);
    let kv = K[c];
    if (VEC) {
        let xh = (V4(X[x_i]) - kv.x) * kv.y;
        Y[i] = OUT(bn_act(xh * kv.z + kv.w));
    } else {
        let xh = (V4(X[x_i]).x - kv.x) * kv.y;
        Y[i] = OUT(vec4<f32>(bn_act_f32(xh * kv.z + kv.w)));
    }
}
