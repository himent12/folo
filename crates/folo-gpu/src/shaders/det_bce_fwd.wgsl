// Detection loss, classification term (docs/MODEL.md §7.4): Σ BCE(cls_raw, T) with T read from the assigner's
// dense map (T = t at the assigned class of a positive anchor, 0 elsewhere), each element formed whole
// (max(x, 0) − x·T + ln(1 + e^-|x|)) so confident positives do not cancel against the dense sum. One workgroup
// per (image·class row, chunk of 1024 anchors) writes its partial sum (a fixed tree, so the result does not depend
// on scheduling). Raises MISC[0] bit 0 if a logit is not finite.
// `kernels/detect.rs` appends either the 1-barrier 32-wide subgroup reduction or the portable 8-barrier fallback.
// Params: 0 = A, 1 = NC, 2 = chunks per row, 3 = B·A, 4 = mal_gamma (0 = plain BCE).
@group(0) @binding(1) var<storage, read> CLS: array<f32>;
@group(0) @binding(2) var<storage, read> DENSE: array<u32>;
@group(0) @binding(3) var<storage, read_write> PART: array<f32>;
@group(0) @binding(4) var<storage, read_write> MISC: array<atomic<u32>>;

fn det_bce_thread(wid: vec3<u32>, lid: u32) -> vec2<f32> {
    let na = pu(0u);
    let nc = pu(1u);
    let gamma = pf(4u);
    let row = wid.y;
    let b = row / nc;
    let c = row - b * nc;
    var acc = 0.0;
    var bad = false;
    for (var e = 0u; e < 4u; e++) {
        let a = wid.x * 1024u + e * 256u + lid;
        if (a < na) {
            let x = CLS[row * na + a];
            bad = bad || !is_finite_f(x);
            let cell = b * na + a;
            let t = select(0.0, bitcast<f32>(DENSE[pu(3u) + cell]), DENSE[cell] == c + 1u);
            let sp = softplus_f(x);
            var elem = sp - x * t;
            if (gamma > 0.0 && t == 0.0) {
                elem = pow_pos(sigmoid_f(x), gamma) * sp;
            }
            acc = acc + elem;
        }
    }
    return vec2<f32>(acc, select(0.0, 1.0, bad));
}
