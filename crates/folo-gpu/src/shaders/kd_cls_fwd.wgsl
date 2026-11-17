// Distillation, class term (docs/MODEL.md §8.9): Σ over elements of softplus(x) − x·p~ − H(p~), the Bernoulli KL
// of the student's logit x against the teacher's probability p~ = sigmoid(y), formed per element before any sum.
// H(p~) = p~·softplus(−y) + (1 − p~)·softplus(y) (−ln p~ and −ln(1 − p~) without a logarithm of a rounded
// probability), 0 for a teacher logit of ±inf (p~ exactly 0 or 1, the tensor path's limit). One workgroup per
// (image·class row, chunk of 1024 anchors), numbered row-major and folded into x / y, writes its partial sum (a
// fixed tree, so the result does not depend on scheduling). Raises MISC[0] bit 0 if a student logit is not finite
// or a teacher logit is NaN (the inputs that make the tensor path's term non-finite).
// Params: 0 = A, 1 = chunks per row, 2 = workgroups (rows · chunks).
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> TC: array<TW>;
@group(0) @binding(3) var<storage, read_write> PART: array<f32>;
@group(0) @binding(4) var<storage, read_write> MISC: array<atomic<u32>>;

var<workgroup> ACC: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let na = pu(0u);
    let chunks = pu(1u);
    let w = wid.y * nwg.x + wid.x;
    let live = w < pu(2u);
    let row = w / chunks;
    let chunk = w - row * chunks;
    var acc = 0.0;
    var bad = false;
    for (var e = 0u; e < 4u; e++) {
        let a = chunk * 1024u + e * 256u + lid;
        if (live && a < na) {
            let i = row * na + a;
            let x = X[i];
            let y = tv(TC[ti(i)], i);
            bad = bad || !is_finite_f(x) || is_nan_f(y);
            let p = sigmoid_f(y);
            let h = select(0.0, p * softplus_f(-y) + (1.0 - p) * softplus_f(y), is_finite_f(y));
            acc = acc + (softplus_f(x) - x * p - h);
        }
    }
    if (bad) { atomicOr(&MISC[0], 1u); }
    ACC[lid] = acc;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step = step >> 1u) {
        if (lid < step) { ACC[lid] = ACC[lid] + ACC[lid + step]; }
        workgroupBarrier();
    }
    if (lid == 0u && live) { PART[w] = ACC[0]; }
}
