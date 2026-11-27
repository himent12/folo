// Pass 3 of the BatchNorm backward: DX = w · invstd · (dz − db / n − x̂ · dw / n) with x̂ = (x − mean) · invstd and
// dz = g · act'(x̂ · w + b), the same pre-activation expression as the forward and the reduction pass.
// K = (mean, invstd, w, b, db / n, dw / n) per channel. Spec: VEC4, GRID3D, STRIDED.
// Params: 0 = element count (E units), 1 = C, 2 = hw_units; when STRIDED: 3 = x_off, 4 = x_n_stride, 5 = g_off,
// 6 = g_n_stride (all in E units). When GRID3D, grid is (ceil(hw_units / 512), C, N) for division-free indexing,
// uniform per-workgroup K loads, and 2 units/thread ILP.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> DX: array<E>;

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
        let k = 6u * c;
        let m = K[k];
        let inv = K[k + 1u];
        let w = K[k + 2u];
        let b = K[k + 3u];
        let db_n = K[k + 4u];
        let dw_n = K[k + 5u];
        let sc = w * inv;
        let ch_j0 = c * hw + j0;
        let dx_i0 = n * channels * hw + ch_j0;
        let x_i0 = select(dx_i0, pu(3u) + n * pu(4u) + ch_j0, IS_STRIDED);
        let g_i0 = select(dx_i0, pu(5u) + n * pu(6u) + ch_j0, IS_STRIDED);
        let j1 = j0 + 256u;
        if (j1 < hw) {
            let x_i1 = x_i0 + 256u;
            let g_i1 = g_i0 + 256u;
            let dx_i1 = dx_i0 + 256u;
            if (VEC) {
                let xv0 = V4(X[x_i0]);
                let gv0 = V4(G[g_i0]);
                let xv1 = V4(X[x_i1]);
                let gv1 = V4(G[g_i1]);
                let xh0 = (xv0 - m) * inv;
                var dz0 = gv0;
                if (ACT != 0u) { dz0 *= bn_act_grad(xh0 * w + b); }
                let xh1 = (xv1 - m) * inv;
                var dz1 = gv1;
                if (ACT != 0u) { dz1 *= bn_act_grad(xh1 * w + b); }
                DX[dx_i0] = OUT(sc * (dz0 - db_n - xh0 * dw_n));
                DX[dx_i1] = OUT(sc * (dz1 - db_n - xh1 * dw_n));
            } else {
                let xv0 = V4(X[x_i0]).x;
                let gv0 = V4(G[g_i0]).x;
                let xv1 = V4(X[x_i1]).x;
                let gv1 = V4(G[g_i1]).x;
                let xh0 = (xv0 - m) * inv;
                var dz0 = gv0;
                if (ACT != 0u) { dz0 *= bn_act_grad_f32(xh0 * w + b); }
                let xh1 = (xv1 - m) * inv;
                var dz1 = gv1;
                if (ACT != 0u) { dz1 *= bn_act_grad_f32(xh1 * w + b); }
                DX[dx_i0] = OUT(vec4<f32>(sc * (dz0 - db_n - xh0 * dw_n)));
                DX[dx_i1] = OUT(vec4<f32>(sc * (dz1 - db_n - xh1 * dw_n)));
            }
        } else {
            if (VEC) {
                let xh = (V4(X[x_i0]) - m) * inv;
                var dz = V4(G[g_i0]);
                if (ACT != 0u) { dz *= bn_act_grad(xh * w + b); }
                DX[dx_i0] = OUT(sc * (dz - db_n - xh * dw_n));
            } else {
                let xh = (V4(X[x_i0]).x - m) * inv;
                var dz = V4(G[g_i0]).x;
                if (ACT != 0u) { dz *= bn_act_grad_f32(xh * w + b); }
                DX[dx_i0] = OUT(vec4<f32>(sc * (dz - db_n - xh * dw_n)));
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
    let ch_j = c * hw + j;
    let x_i = select(i, pu(3u) + n * pu(4u) + ch_j, IS_STRIDED);
    let g_i = select(i, pu(5u) + n * pu(6u) + ch_j, IS_STRIDED);
    let k = 6u * c;
    let inv = K[k + 1u];
    let w = K[k + 2u];
    if (VEC) {
        let xh = (V4(X[x_i]) - K[k]) * inv;
        var dz = V4(G[g_i]);
        if (ACT != 0u) {
            dz *= bn_act_grad(xh * w + K[k + 3u]);
        }
        DX[i] = OUT((w * inv) * (dz - K[k + 4u] - xh * K[k + 5u]));
    } else {
        let xh = (V4(X[x_i]).x - K[k]) * inv;
        var dz = V4(G[g_i]).x;
        if (ACT != 0u) {
            dz *= bn_act_grad_f32(xh * w + K[k + 3u]);
        }
        DX[i] = OUT(vec4<f32>((w * inv) * (dz - K[k + 4u] - xh * K[k + 5u])));
    }
}
