// GT-Verified DFL KD anchor weights (positive-only teacher + student DFL decode, one thread per pick slot):
// For each positive pick slot with t > 0, decodes ONLY that anchor's student and teacher boxes in stride units,
// computes plain IoU against the slot's GT box, and writes VER_W[b * A + a] = iou_t - iou_s when
// (iou_t > iou_s + 0.02 && iou_t > 0.50), else 0.0. VER_W is zeroed by the caller; each anchor has at most one
// positive pick, so writes do not collide.
// Spec: TF16, R, K. Params: 0 = slots, 1 = A, 2 = G, 3 = gt box word offset.
@group(0) @binding(1) var<storage, read> S_BOX: array<f32>;
@group(0) @binding(2) var<storage, read> T_BOX: array<TW>;
@group(0) @binding(3) var<storage, read> META: array<u32>;
@group(0) @binding(4) var<storage, read> PICK: array<u32>;
@group(0) @binding(5) var<storage, read> SLOT: array<u32>;
@group(0) @binding(6) var<storage, read_write> VER_W: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let s = linear_id(wid, nwg, lid, 64u);
    if (s >= pu(0u)) { return; }
    let t = bitcast<f32>(SLOT[2u * s]);
    let pos = SLOT[2u * s + 1u] == 1u;
    if (!pos || t == 0.0) { return; }
    let na = pu(1u);
    let ng = pu(2u);
    let gi = s / K;
    let b = gi / ng;
    let a = PICK[3u * s];
    let ax = bitcast<f32>(META[3u * a]);
    let ay = bitcast<f32>(META[3u * a + 1u]);
    let inv = 1.0 / bitcast<f32>(META[3u * a + 2u]);
    let o = pu(3u) + 4u * gi;
    let tb = vec4<f32>(bitcast<f32>(META[o]), bitcast<f32>(META[o + 1u]), bitcast<f32>(META[o + 2u]),
                       bitcast<f32>(META[o + 3u])) * inv;
    let base = b * 4u * R * na + a;
    var ds: vec4<f32>;
    var dt: vec4<f32>;
    for (var side = 0u; side < 4u; side++) {
        var ms = -3.402823466e+38;
        var mt = -3.402823466e+38;
        for (var k = 0u; k < R; k++) {
            let idx = base + (side * R + k) * na;
            ms = max(ms, S_BOX[idx]);
            mt = max(mt, tv(T_BOX[ti(idx)], idx));
        }
        var zs = 0.0;
        var es = 0.0;
        var zt = 0.0;
        var et = 0.0;
        for (var k = 0u; k < R; k++) {
            let idx = base + (side * R + k) * na;
            let ps = fexp(S_BOX[idx] - ms);
            let pt = fexp(tv(T_BOX[ti(idx)], idx) - mt);
            zs = zs + ps;
            es = es + ps * f32(k);
            zt = zt + pt;
            et = et + pt * f32(k);
        }
        ds[side] = es / zs;
        dt[side] = et / zt;
    }
    let pd_s = vec4<f32>(ax - ds[0], ay - ds[1], ax + ds[2], ay + ds[3]);
    let pd_t = vec4<f32>(ax - dt[0], ay - dt[1], ax + dt[2], ay + dt[3]);
    let iou_s = iou_plain(pd_s, tb);
    let iou_t = iou_plain(pd_t, tb);
    if (iou_t > iou_s + 0.02 && iou_t > 0.50) {
        VER_W[b * na + a] = iou_t - iou_s;
    }
}
