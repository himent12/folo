// The box decode of one detection-head level's candidates, one thread per (image, slot): slot s below the
// image's count (and CAP) is pixel SITES[b][1 + s] with class and score RECS[b][s]; its box is decoded from
// RAW [B, 4 * RM, CAP] (column s) exactly as detect.wgsl decodes it (DFL: softmax expectation over RM bins per
// side, ltrb around the anchor point (x + 0.5, y + 0.5), times the stride), and a record (anchor, class, score,
// x1, y1, x2, y2) is appended to the image's region of OUT: word 0 counts the records, word 1 collects one bit per
// level whose candidates did not fit (zeroed by the host), the records follow in arrival order.
// Params: 0 = B * CAP, 1 = CAP, 2 = RM, 3 = W, 4 = stride (f32), 5 = the level's first anchor, 6 = OUT words per
// image, 7 = the level's bit. T is the storage type of RAW (f32 or f16); the decode is f32.
@group(0) @binding(1) var<storage, read> RAW: array<T>;
@group(0) @binding(2) var<storage, read> SITES: array<u32>;
@group(0) @binding(3) var<storage, read> RECS: array<u32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let cap = pu(1u);
    let rm = pu(2u);
    let b = i / cap;
    let s = i % cap;
    let row = b * (cap + 1u);
    let n = SITES[row];
    let region = b * pu(6u);
    if (s == 0u && n > cap) {
        atomicOr(&OUT[region + 1u], pu(7u));
    }
    if (s >= min(n, cap)) { return; }
    let j = SITES[row + 1u + s];
    var d = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    if (R > 0u) {
        var v: array<f32, R_CAP>;
        for (var side = 0u; side < 4u; side++) {
            let base = (b * 4u + side) * R * cap + s;
            var m = f32(RAW[base]);
            v[0] = m;
            for (var k = 1u; k < R; k++) {
                let x = f32(RAW[base + k * cap]);
                v[k] = x;
                m = max(m, x);
            }
            var z = 0.0;
            var e = 0.0;
            for (var k = 0u; k < R; k++) {
                let p = fexp(v[k] - m);
                z += p;
                e = fma(p, f32(k), e);
            }
            d[side] = e / z;
        }
    } else {
        for (var side = 0u; side < 4u; side++) {
            let base = (b * 4u + side) * rm * cap + s;
            var m = f32(RAW[base]);
            for (var k = 1u; k < rm; k++) {
                m = max(m, f32(RAW[base + k * cap]));
            }
            var z = 0.0;
            var e = 0.0;
            for (var k = 0u; k < rm; k++) {
                let p = fexp(f32(RAW[base + k * cap]) - m);
                z += p;
                e = fma(p, f32(k), e);
            }
            d[side] = e / z;
        }
    }
    let w = pu(3u);
    let ax = f32(j % w) + 0.5;
    let ay = f32(j / w) + 0.5;
    let st = pf(4u);
    let r = region + 2u + atomicAdd(&OUT[region], 1u) * 7u;
    atomicStore(&OUT[r], pu(5u) + j);
    atomicStore(&OUT[r + 1u], RECS[(b * cap + s) * 2u]);
    atomicStore(&OUT[r + 2u], RECS[(b * cap + s) * 2u + 1u]);
    atomicStore(&OUT[r + 3u], bitcast<u32>((ax - d[0]) * st));
    atomicStore(&OUT[r + 4u], bitcast<u32>((ay - d[1]) * st));
    atomicStore(&OUT[r + 5u], bitcast<u32>((ax + d[2]) * st));
    atomicStore(&OUT[r + 6u], bitcast<u32>((ay + d[3]) * st));
}
