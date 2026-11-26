// Detection-head decode with candidate compaction, one thread per (image, anchor). The anchor's best class is
// the first maximum of its NC sigmoid scores; when that score is > conf, its box is decoded (DFL: softmax
// expectation over RM bins per side, ltrb around the anchor point, times the stride) and a record
// (anchor, class, score, x1, y1, x2, y2) is appended to the image's region of OUT: word 0 of the region
// counts the records (zeroed by the host), the records follow in arrival order.
// Params: 0 = B * A, 1 = A, 2 = NC, 3 = RM, 4 = conf (f32). T is the storage type of BOX and CLS (f32 or f16);
// the decode is f32.
@group(0) @binding(1) var<storage, read> BOX: array<T>;
@group(0) @binding(2) var<storage, read> CLS: array<T>;
@group(0) @binding(3) var<storage, read> ANC: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let na = pu(1u);
    let nc = pu(2u);
    let rm = pu(3u);
    let b = i / na;
    let a = i % na;
    let cb = b * nc * na + a;
    var best = 0u;
    var bx = f32(CLS[cb]);
    // Bit-test c = 0 for NaN: compilers may lower `if (x > bx)` to `maxNum` (which drops a NaN in `bx`,
    // notably for f16), whereas evaluating `sigmoid_f(c = 0)` keeps `bs = NaN` and drops the anchor.
    if ((bitcast<u32>(bx) & 0x7fffffffu) > 0x7f800000u) { return; }
    var prev_bx = bx;
    for (var c = 1u; c < nc; c++) {
        let x = f32(CLS[cb + c * na]);
        if (x > bx) {
            prev_bx = bx;
            best = c;
            bx = x;
        }
    }
    let bs = sigmoid_f(bx);
    if (!(bs > pf(4u))) { return; }
    // Sigmoid is non-decreasing but can tie in f32 (saturation to 1.0 / 0.0 or rounding); `prev_bx` is the
    // max non-NaN logit in `0..best`, so rescan `0..best` only if an earlier class could tie `bs`.
    if (best > 0u && sigmoid_f(prev_bx) >= bs) {
        for (var c = 0u; c < best; c++) {
            if (sigmoid_f(f32(CLS[cb + c * na])) >= bs) {
                best = c;
                break;
            }
        }
    }
    var d = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    if (R > 0u) {
        var v: array<f32, R_CAP>;
        for (var side = 0u; side < 4u; side++) {
            let base = (b * 4u + side) * R * na + a;
            var m = f32(BOX[base]);
            var has_nan = (bitcast<u32>(m) & 0x7fffffffu) > 0x7f800000u;
            v[0] = m;
            for (var k = 1u; k < R; k++) {
                let x = f32(BOX[base + k * na]);
                has_nan = has_nan || ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u);
                v[k] = x;
                m = max(m, x);
            }
            let bad = has_nan || !is_finite_f(m);
            var z = 0.0;
            var e = 0.0;
            for (var k = 0u; k < R; k++) {
                let p = fexp(v[k] - m);
                z += p;
                e = fma(p, f32(k), e);
            }
            d[side] = select(e / z, bitcast<f32>(0x7fc00000u), bad);
        }
    } else {
        for (var side = 0u; side < 4u; side++) {
            let base = (b * 4u + side) * rm * na + a;
            var m = f32(BOX[base]);
            var has_nan = (bitcast<u32>(m) & 0x7fffffffu) > 0x7f800000u;
            for (var k = 1u; k < rm; k++) {
                let x = f32(BOX[base + k * na]);
                has_nan = has_nan || ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u);
                m = max(m, x);
            }
            let bad = has_nan || !is_finite_f(m);
            var z = 0.0;
            var e = 0.0;
            for (var k = 0u; k < rm; k++) {
                let p = fexp(f32(BOX[base + k * na]) - m);
                z += p;
                e = fma(p, f32(k), e);
            }
            d[side] = select(e / z, bitcast<f32>(0x7fc00000u), bad);
        }
    }
    let ax = ANC[a];
    let ay = ANC[na + a];
    let st = ANC[2u * na + a];
    let region = b * (1u + 7u * na);
    let r = region + 1u + atomicAdd(&OUT[region], 1u) * 7u;
    atomicStore(&OUT[r], a);
    atomicStore(&OUT[r + 1u], best);
    atomicStore(&OUT[r + 2u], bitcast<u32>(bs));
    atomicStore(&OUT[r + 3u], bitcast<u32>((ax - d[0]) * st));
    atomicStore(&OUT[r + 4u], bitcast<u32>((ay - d[1]) * st));
    atomicStore(&OUT[r + 5u], bitcast<u32>((ax + d[2]) * st));
    atomicStore(&OUT[r + 6u], bitcast<u32>((ay + d[3]) * st));
}
