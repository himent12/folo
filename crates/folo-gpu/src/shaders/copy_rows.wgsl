// Row-blocked copy, used for concat: y[o * dst_row + dst_off + j] = x[o * row + j] for j < row.
// Spec: VEC (E = vec4 units). Params (in E units): 0 = total units, 1 = row, 2 = dst_row, 3 = dst_off.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let row = pu(1u);
    let o = i / row;
    let j = i - o * row;
    Y[o * pu(2u) + pu(3u) + j] = X[i];
}
