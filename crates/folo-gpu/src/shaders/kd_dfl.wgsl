// Distillation, DFL term (docs/MODEL.md §8.9), one thread per (image, anchor): q = max_c sigmoid(y_c) of the
// teacher's class logits, then per side the teacher's s~ = softmax(z_T / T) and the student's softmax(z / T) over
// the R bins. MODE 0 (forward): Σ_side Σ_r s~_r · (ln s~_r − log_softmax(z / T)_r), times q, summed per workgroup
// of 256 anchors in a fixed tree into PART (one partial per workgroup). Both log-softmaxes stay in the max-shifted
// domain (the teacher's (z − max z) / T − ln Σ, the maximum subtracted before dividing by T; the student's
// z / T − max (z / T) − ln Σ, as the tensor path), and logarithms are taken of the softmax sums (>= 1) only, not of
// rounded probabilities. A teacher bin of −inf has s~ = 0 and adds nothing (the tensor path's limit). Raises
// MISC[0] bit 0 for the inputs that make the tensor path's term non-finite: a student box value whose z or z / T
// is not finite, a teacher box value of NaN or +inf or whose z_T / T is +inf, or a side whose teacher bins all
// have z_T / T = −inf. MODE 1 (backward): DZ = g · S · q · (softmax(z / T) − s~), with g = GRAD[0] and
// S = B · dfl_gain · T / (4 N).
// Params: 0 = A, 1 = NC, 2 = B·A, 3 = T, 4 = S (MODE 1).
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read> TZ: array<TW>;
@group(0) @binding(3) var<storage, read> TC: array<TW>;
@group(0) @binding(4) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(5) var<storage, read_write> MISC: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> GRAD: array<f32>;

var<workgroup> ACC: array<f32, 256>;

fn anchor(i: u32) -> f32 {
    let na = pu(0u);
    let nc = pu(1u);
    let temp = pf(3u);
    let b = i / na;
    let a = i - b * na;
    var m = tv(TC[ti(b * nc * na + a)], b * nc * na + a);
    for (var c = 1u; c < nc; c++) {
        let k = (b * nc + c) * na + a;
        m = max(m, tv(TC[ti(k)], k));
    }
    let q = sigmoid_f(m);
    var acc = 0.0;
    var bad = false;
    var u: array<f32, R>;
    var v: array<f32, R>;
    var et: array<f32, R>;
    for (var side = 0u; side < 4u; side++) {
        let base = (b * 4u + side) * R * na + a;
        // The teacher: u = (z_T − max z_T) / T, the subtraction first, so the division rounds only the small
        // differences (not logits near 2^24). The student: v = z / T − max (z / T), the tensor path's own order, so
        // it overflows exactly where the tensor path does. Which bins count is decided from the raw values and
        // z / T: a teacher bin of −inf, or whose z_T / T is −inf, has s~ = 0 and adds nothing; a student bin whose
        // z or z / T is not finite, a teacher bin of NaN or +inf, or one whose z_T / T is +inf, makes the term
        // non-finite, as does a side whose teacher bins all have s~ = 0.
        var mt = -3.40282347e38;
        var ms = -3.40282347e38;
        var live = 0u;
        for (var r = 0u; r < R; r++) {
            let k = base + r * na;
            let zt = tv(TZ[ti(k)], k);
            let zs = Z[k];
            let vs = zs / temp;
            bad = bad || !is_finite_f(zs) || !is_finite_f(vs) || is_nan_f(zt) || is_pos_inf_f(zt);
            if (is_finite_f(zt)) {
                let qt = zt / temp;
                bad = bad || is_pos_inf_f(qt);
                if (is_finite_f(qt)) {
                    live = live | (1u << r);
                    mt = max(mt, zt);
                }
            }
            u[r] = zt;
            v[r] = vs;
            ms = max(ms, vs);
        }
        bad = bad || live == 0u;
        var st = 0.0;
        var ss = 0.0;
        for (var r = 0u; r < R; r++) {
            let d = (u[r] - mt) / temp;
            let fin = (live & (1u << r)) != 0u && is_finite_f(d);
            u[r] = select(0.0, d, fin);
            v[r] = v[r] - ms;
            et[r] = select(0.0, fexp(u[r]), fin);
            st = st + et[r];
            ss = ss + fexp(v[r]);
        }
        if (MODE == 0u) {
            let lt = log_ge1(st);
            let ls = log_ge1(ss);
            var side_sum = 0.0;
            for (var r = 0u; r < R; r++) {
                if (et[r] > 0.0) {
                    side_sum = side_sum + et[r] / st * ((u[r] - lt) - (v[r] - ls));
                }
            }
            acc = acc + side_sum;
        } else {
            let s = GRAD[0] * pf(4u) * q;
            for (var r = 0u; r < R; r++) {
                OUT[base + r * na] = s * (fexp(v[r]) / ss - et[r] / st);
            }
        }
    }
    if (MODE == 0u && bad) { atomicOr(&MISC[0], 1u); }
    return q * acc;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    var acc = 0.0;
    if (i < pu(2u)) { acc = anchor(i); }
    if (MODE == 0u) {
        ACC[lid] = acc;
        workgroupBarrier();
        for (var step = 128u; step > 0u; step = step >> 1u) {
            if (lid < step) { ACC[lid] = ACC[lid] + ACC[lid + step]; }
            workgroupBarrier();
        }
        if (lid == 0u) { OUT[wid.y * nwg.x + wid.x] = ACC[0]; }
    }
}
