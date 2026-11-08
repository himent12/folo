// Task-aligned assigner, step 0 (docs/MODEL.md §7.1): the DFL decode of every anchor to a box in input pixels,
// one thread per (image, anchor), in the order of folo-vision's host decode (max-subtracted softmax, then
// `(anchor ∓ d) · stride`). Also raises MISC[0] bit 0 if any box logit is not finite.
// Spec: R (bins per side). Params: 0 = B·A, 1 = A.
// BOX: [B, 4R, A]; META: anchors (x, y in grid units, stride) at word 0; PD: [B, A] xyxy.
@group(0) @binding(1) var<storage, read> BOX: array<f32>;
@group(0) @binding(2) var<storage, read> META: array<u32>;
@group(0) @binding(3) var<storage, read_write> PD: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> MISC: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let na = pu(1u);
    let b = i / na;
    let a = i - b * na;
    let base = b * 4u * R * na + a;
    var finite = true;
    var d: array<f32, 4>;
    for (var side = 0u; side < 4u; side++) {
        var v: array<f32, R>;
        var m = -3.402823466e+38;
        for (var k = 0u; k < R; k++) {
            let x = BOX[base + (side * R + k) * na];
            v[k] = x;
            finite = finite && is_finite_f(x);
            m = max(m, x);
        }
        var z = 0.0;
        var e = 0.0;
        for (var k = 0u; k < R; k++) {
            let p = fexp(v[k] - m);
            z = z + p;
            e = e + p * f32(k);
        }
        d[side] = e / z;
    }
    let ax = bitcast<f32>(META[3u * a]);
    let ay = bitcast<f32>(META[3u * a + 1u]);
    let s = bitcast<f32>(META[3u * a + 2u]);
    PD[i] = vec4<f32>((ax - d[0]) * s, (ay - d[1]) * s, (ax + d[2]) * s, (ay + d[3]) * s);
    if (!finite) { atomicOr(&MISC[0], 1u); }
}
