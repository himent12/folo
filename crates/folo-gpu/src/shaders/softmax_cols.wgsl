// Softmax / log-softmax over the middle axis of [outer, D, inner] with inner > 1: one thread per (outer, inner)
// column, so loads are coalesced across columns. When `SMALL_COL` is true (D <= 32, e.g. DFL reg_max=16), the
// column is cached in a 32-element local array (1 global read of X and 1 exp per element); larger D uses 2-pass
// online softmax (2 global reads instead of 3). A non-finite column maximum is replaced by 0, as on the CPU.
// Spec: LOG, SMALL (D <= 32). Params: 0 = columns (outer * inner), 1 = D, 2 = inner, 3 = -inf.
// XT / YT are the storage types of X and Y (f32 or f16); the arithmetic is f32.
@group(0) @binding(1) var<storage, read> X: array<XT>;
@group(0) @binding(2) var<storage, read_write> Y: array<YT>;

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = linear_id(wid, nwg, lid, 256u);
    if (t >= pu(0u)) { return; }
    let d = pu(1u);
    let inner = pu(2u);
    let neg_inf = pf(3u);
    let base = (t / inner) * d * inner + t % inner;
    if (SMALL_COL) {
        var vals: array<f32, 32>;
        var m = neg_inf;
        for (var i = 0u; i < d; i++) {
            let x = f32(X[base + i * inner]);
            vals[i] = x;
            m = max(m, x);
        }
        if ((bitcast<u32>(m) & 0x7fffffffu) >= 0x7f800000u) { m = 0.0; }
        var s = 0.0;
        if (LOG) {
            for (var i = 0u; i < d; i++) {
                let shifted = vals[i] - m;
                vals[i] = shifted;
                s += exp(shifted);
            }
            let ls = log(s);
            for (var i = 0u; i < d; i++) {
                Y[base + i * inner] = YT(vals[i] - ls);
            }
        } else {
            for (var i = 0u; i < d; i++) {
                let e = exp(vals[i] - m);
                vals[i] = e;
                s += e;
            }
            let inv = 1.0 / s;
            for (var i = 0u; i < d; i++) {
                Y[base + i * inner] = YT(vals[i] * inv);
            }
        }
        return;
    }
    var st = vec2<f32>(neg_inf, 0.0);
    for (var i = 0u; i < d; i++) {
        st = online_step(st, f32(X[base + i * inner]), neg_inf);
    }
    var m = st.x;
    var s = st.y;
    if ((bitcast<u32>(m) & 0x7fffffffu) >= 0x7f800000u) {
        if (m > 0.0) { s = s + m; }
        m = 0.0;
    }
    if (LOG) {
        let ls = log(s);
        for (var i = 0u; i < d; i++) { let o = base + i * inner; Y[o] = YT(f32(X[o]) - m - ls); }
    } else {
        let inv = 1.0 / s;
        for (var i = 0u; i < d; i++) { let o = base + i * inner; Y[o] = YT(exp(f32(X[o]) - m) * inv); }
    }
}
