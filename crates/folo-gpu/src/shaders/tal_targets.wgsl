// Task-aligned assigner, step 5 (docs/MODEL.md §7.3): one workgroup per (gt g, image b), one thread per pick.
// A pick is a final positive iff the gt owns its anchor after step 4; over those, m_max and u_max give the soft
// target t = m · u_max / (m_max + eps). Writes per pick slot SLOT = (t bits or 0, 1 if positive) and, for each
// positive, the dense per-anchor map DENSE = [label + 1 | t bits] (zeroed before: 0 means background).
// Spec: K (<= 64). Params: 0 = A, 1 = G, 2 = B·A, 3 = label word offset, 4 = eps.
@group(0) @binding(1) var<storage, read> PICK: array<u32>;
@group(0) @binding(2) var<storage, read> OWN: array<u32>;
@group(0) @binding(3) var<storage, read> META: array<u32>;
@group(0) @binding(4) var<storage, read_write> SLOT: array<u32>;
@group(0) @binding(5) var<storage, read_write> DENSE: array<u32>;

var<workgroup> MM: array<f32, 64>;
var<workgroup> UM: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let b = wid.y;
    let na = pu(0u);
    let ng = pu(1u);
    let gi = b * ng + g;
    let s = gi * K + lid;
    var pos = false;
    var a = 0u;
    var m = 0.0;
    var u = 0.0;
    if (lid < K) {
        a = PICK[3u * s];
        if (a != DET_NONE) {
            m = bitcast<f32>(PICK[3u * s + 1u]);
            u = bitcast<f32>(PICK[3u * s + 2u]);
            pos = OWN[pu(2u) + b * na + a] == ~g;
        }
    }
    MM[lid] = select(0.0, m, pos);
    UM[lid] = select(0.0, u, pos);
    workgroupBarrier();
    for (var step = 32u; step > 0u; step = step >> 1u) {
        if (lid < step) {
            MM[lid] = max(MM[lid], MM[lid + step]);
            UM[lid] = max(UM[lid], UM[lid + step]);
        }
        workgroupBarrier();
    }
    if (lid < K) {
        var t = 0.0;
        if (pos) {
            t = m * UM[0] / (MM[0] + pf(4u));
            let cell = b * na + a;
            DENSE[cell] = META[pu(3u) + gi] + 1u;
            DENSE[pu(2u) + cell] = bitcast<u32>(t);
        }
        SLOT[2u * s] = bitcast<u32>(t);
        SLOT[2u * s + 1u] = select(0u, 1u, pos);
    }
}
