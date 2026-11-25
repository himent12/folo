// Softmax / log-softmax over the contiguous last axis, one workgroup per row.
// Single-pass online softmax (Milakov-Gimelshein) caches up to 4 elements per thread in registers when
// `CACHE_ROW` is true (n <= 1024, 1 global read of X); longer rows use 2-pass online softmax (2 global reads
// instead of 3) with a single workgroup reduction.
// Spec: LOG (log_softmax), CACHE (n <= 1024). Params: 0 = rows, 1 = row length, 2 = -inf.
// XT / YT are the storage types of X and Y (f32 or f16); the arithmetic is f32.
@group(0) @binding(1) var<storage, read> X: array<XT>;
@group(0) @binding(2) var<storage, read_write> Y: array<YT>;

var<workgroup> partial: array<vec2<f32>, 256>;

fn online_step(st: vec2<f32>, x: f32, neg_inf: f32) -> vec2<f32> {
    let m = st.x;
    var d = st.y;
    if (x > m) {
        return vec2<f32>(x, d * exp(m - x) + 1.0);
    }
    if (x == m) {
        if (x != neg_inf) { d += 1.0; }
        return vec2<f32>(m, d);
    }
    // Either x < m (finite <= 0 or -inf) or x is NaN (propagates NaN into d).
    return vec2<f32>(m, d + exp(x - m));
}

fn online_merge(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (a.x > b.x) {
        return vec2<f32>(a.x, a.y + b.y * exp(b.x - a.x));
    }
    if (b.x > a.x) {
        return vec2<f32>(b.x, b.y + a.y * exp(a.x - b.x));
    }
    return vec2<f32>(a.x, a.y + b.y);
}

fn reduce_online(lid: u32, v: vec2<f32>) -> vec2<f32> {
    partial[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { partial[lid] = online_merge(partial[lid], partial[lid + s]); }
        workgroupBarrier();
    }
    return partial[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let n = pu(1u);
    let neg_inf = pf(2u);
    // `row` is uniform across the workgroup, so every invocation reaches the barriers.
    let valid = row < rows;
    let base = row * n;
    var st = vec2<f32>(neg_inf, 0.0);
    if (CACHE_ROW) {
        var v0 = 0.0;
        var v1 = 0.0;
        var v2 = 0.0;
        var v3 = 0.0;
        let i0 = lid;
        let i1 = lid + 256u;
        let i2 = lid + 512u;
        let i3 = lid + 768u;
        if (valid) {
            if (i0 < n) { v0 = f32(X[base + i0]); st = online_step(st, v0, neg_inf); }
            if (i1 < n) { v1 = f32(X[base + i1]); st = online_step(st, v1, neg_inf); }
            if (i2 < n) { v2 = f32(X[base + i2]); st = online_step(st, v2, neg_inf); }
            if (i3 < n) { v3 = f32(X[base + i3]); st = online_step(st, v3, neg_inf); }
        }
        st = reduce_online(lid, st);
        if (!valid) { return; }
        var m = st.x;
        var s = st.y;
        // A non-finite row maximum (+inf or -inf) is replaced by 0, as on the CPU: for +inf, adding +inf to
        // `s` turns finite sums into +inf while preserving NaN; for -inf, `s` is already 0.0 (or NaN).
        if ((bitcast<u32>(m) & 0x7fffffffu) >= 0x7f800000u) {
            if (m > 0.0) { s = s + m; }
            m = 0.0;
        }
        if (LOG) {
            let ls = log(s);
            if (i0 < n) { Y[base + i0] = YT(v0 - m - ls); }
            if (i1 < n) { Y[base + i1] = YT(v1 - m - ls); }
            if (i2 < n) { Y[base + i2] = YT(v2 - m - ls); }
            if (i3 < n) { Y[base + i3] = YT(v3 - m - ls); }
        } else {
            let inv = 1.0 / s;
            if (i0 < n) { Y[base + i0] = YT(exp(v0 - m) * inv); }
            if (i1 < n) { Y[base + i1] = YT(exp(v1 - m) * inv); }
            if (i2 < n) { Y[base + i2] = YT(exp(v2 - m) * inv); }
            if (i3 < n) { Y[base + i3] = YT(exp(v3 - m) * inv); }
        }
        return;
    }
    if (valid) {
        for (var i = lid; i < n; i += 256u) {
            st = online_step(st, f32(X[base + i]), neg_inf);
        }
    }
    st = reduce_online(lid, st);
    if (!valid) { return; }
    var m = st.x;
    var s = st.y;
    if ((bitcast<u32>(m) & 0x7fffffffu) >= 0x7f800000u) {
        if (m > 0.0) { s = s + m; }
        m = 0.0;
    }
    if (LOG) {
        let ls = log(s);
        for (var i = lid; i < n; i += 256u) { Y[base + i] = YT(f32(X[base + i]) - m - ls); }
    } else {
        let inv = 1.0 / s;
        for (var i = lid; i < n; i += 256u) { Y[base + i] = YT(exp(f32(X[base + i]) - m) * inv); }
    }
}
