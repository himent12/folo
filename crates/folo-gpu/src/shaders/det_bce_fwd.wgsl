// Detection loss, classification term (docs/MODEL.md §7.4): Σ BCE(cls_raw, T) with T read from the assigner's
// dense map (T = t at the assigned class of a positive anchor, 0 elsewhere), each element formed whole
// (max(x, 0) − x·T + ln(1 + e^-|x|)) so confident positives do not cancel against the dense sum. One workgroup
// per (image·class row, chunk of 1024 anchors) writes its partial sum (a fixed tree, so the result does not depend
// on scheduling). Raises MISC[0] bit 0 if a logit is not finite.
// Params: 0 = A, 1 = NC, 2 = chunks per row, 3 = B·A.
@group(0) @binding(1) var<storage, read> CLS: array<f32>;
@group(0) @binding(2) var<storage, read> DENSE: array<u32>;
@group(0) @binding(3) var<storage, read_write> PART: array<f32>;
@group(0) @binding(4) var<storage, read_write> MISC: array<atomic<u32>>;

var<workgroup> ACC: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let na = pu(0u);
    let nc = pu(1u);
    let row = wid.y;
    let b = row / nc;
    let c = row - b * nc;
    var acc = 0.0;
    var finite = true;
    for (var e = 0u; e < 4u; e++) {
        let a = wid.x * 1024u + e * 256u + lid;
        if (a < na) {
            let x = CLS[row * na + a];
            finite = finite && is_finite_f(x);
            let cell = b * na + a;
            let t = select(0.0, bitcast<f32>(DENSE[pu(3u) + cell]), DENSE[cell] == c + 1u);
            acc = acc + (softplus_f(x) - x * t);
        }
    }
    if (!finite) { atomicOr(&MISC[0], 1u); }
    ACC[lid] = acc;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step = step >> 1u) {
        if (lid < step) { ACC[lid] = ACC[lid] + ACC[lid + step]; }
        workgroupBarrier();
    }
    if (lid == 0u) { PART[row * pu(2u) + wid.x] = ACC[0]; }
}
