// Softmax / log-softmax over the contiguous last axis, one workgroup per row.
// Spec: LOG (log_softmax). Params: 0 = rows, 1 = row length.
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

var<workgroup> partial: array<f32, 256>;

fn reduce_max(lid: u32, v: f32) -> f32 {
    partial[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { partial[lid] = max(partial[lid], partial[lid + s]); }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();
    return r;
}

fn reduce_sum(lid: u32, v: f32) -> f32 {
    partial[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { partial[lid] = partial[lid] + partial[lid + s]; }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let n = pu(1u);
    // `row` is uniform across the workgroup, so every invocation reaches the barriers.
    let valid = row < rows;
    let base = row * n;
    var m = -3.4028235e38;
    if (valid) {
        for (var i = lid; i < n; i += 256u) { m = max(m, X[base + i]); }
    }
    m = reduce_max(lid, m);
    var s = 0.0;
    if (valid) {
        for (var i = lid; i < n; i += 256u) { s += exp(X[base + i] - m); }
    }
    s = reduce_sum(lid, s);
    if (!valid) { return; }
    if (LOG) {
        let ls = log(s);
        for (var i = lid; i < n; i += 256u) { Y[base + i] = X[base + i] - m - ls; }
    } else {
        let inv = 1.0 / s;
        for (var i = lid; i < n; i += 256u) { Y[base + i] = exp(X[base + i] - m) * inv; }
    }
}
