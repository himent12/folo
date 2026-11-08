// Top-k per row, one thread per row: k selection passes, each taking the best element strictly after the
// previous pick in the output order (value by total order, descending when LARGEST, then index ascending).
// Spec: LARGEST. Params: 0 = rows, 1 = row length D, 2 = k. Indices are written as i64 (low word, then 0).
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> I: array<u32>;

// Monotone key: the unsigned order of keys is the f32::total_cmp order of values.
fn key(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    return select(b ^ 0x80000000u, ~b, (b >> 31u) == 1u);
}

// True if (ka, ia) comes before (kb, ib) in the output order.
fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    if (ka != kb) {
        if (LARGEST) { return ka > kb; }
        return ka < kb;
    }
    return ia < ib;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = linear_id(wid, nwg, lid, 64u);
    if (row >= pu(0u)) { return; }
    let d = pu(1u);
    let k = pu(2u);
    let base = row * d;
    var pk = 0u;
    var pi = 0u;
    for (var j = 0u; j < k; j++) {
        var found = false;
        var bk = 0u;
        var bi = 0u;
        for (var i = 0u; i < d; i++) {
            let ki = key(X[base + i]);
            // Candidates come strictly after the previous pick.
            if (j > 0u && !before(pk, pi, ki, i)) { continue; }
            if (!found || before(ki, i, bk, bi)) { found = true; bk = ki; bi = i; }
        }
        pk = bk;
        pi = bi;
        V[row * k + j] = X[base + bi];
        I[2u * (row * k + j)] = bi;
        I[2u * (row * k + j) + 1u] = 0u;
    }
}
