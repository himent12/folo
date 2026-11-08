// Generic strided copy (permute / make-contiguous): out[i] = x[offset(i)], where the coordinates of i over
// the output dims are dotted with per-dim input strides. Spec: RANK. Params: 0 = elements, 4..12 = out dims,
// 12..20 = input stride per out dim (innermost last).
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    var rem = i;
    var src = 0u;
    for (var d = 0u; d < RANK; d++) {
        let k = RANK - 1u - d;
        let dim = pu(4u + k);
        src += (rem % dim) * pu(12u + k);
        rem = rem / dim;
    }
    Y[i] = X[src];
}
