// y = op(x), elementwise. Spec: OP (op code), VEC (process vec4<T> elements).
// Params: 0 = element count (in E units), 1 = alpha.
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    Y[i] = OUT(unary4(OP, V4(X[i]), pf(1u)));
}
