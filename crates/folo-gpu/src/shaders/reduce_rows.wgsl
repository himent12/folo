// Reduce the contiguous last axis: out[row] = reduce(x[row * R .. row * R + R]). One workgroup per row.
// Spec: OP (0 sum, 1 max, 2 min). Params: 0 = rows, 1 = R, 2 = scale applied to the result (1/R for mean),
// 3 = identity.
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

var<workgroup> partial: array<f32, 256>;

// The reduction identity (0, -inf or +inf) comes from the host: WGSL constants cannot be infinite.
fn identity() -> f32 { return pf(3u); }

fn combine(a: f32, b: f32) -> f32 {
    switch OP {
        case 1u: { return max(a, b); }
        case 2u: { return min(a, b); }
        default: { return a + b; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let r = pu(1u);
    var acc = identity();
    if (row < rows) {
        let base = row * r;
        for (var i = lid; i < r; i += 256u) {
            acc = combine(acc, X[base + i]);
        }
    }
    partial[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            partial[lid] = combine(partial[lid], partial[lid + s]);
        }
        workgroupBarrier();
    }
    if (lid == 0u && row < rows) {
        Y[row] = partial[0] * pf(2u);
    }
}
