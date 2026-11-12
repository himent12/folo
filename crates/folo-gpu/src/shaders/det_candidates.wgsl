// Multi-label candidates of decoded detections, one thread per (image, anchor). DEC is the decoded head
// channel-major, [B, 4 + NC, A] f32 (x1, y1, x2, y2, then the NC class scores); every (anchor, class) whose score
// is > conf becomes a record (anchor, class, score bits) in the image's region of OUT: word 0 counts the pairs
// (zeroed by the host; it keeps counting past the capacity, so the host sees an overflow), then up to CAP records
// in arrival order. A thread counts its pairs first and reserves them with one atomic add.
// Params: 0 = B * A, 1 = A, 2 = NC, 3 = CAP, 4 = conf (f32).
@group(0) @binding(1) var<storage, read> DEC: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let na = pu(1u);
    let nc = pu(2u);
    let cap = pu(3u);
    let conf = pf(4u);
    let b = i / na;
    let a = i % na;
    let base = (b * (4u + nc) + 4u) * na + a;
    var n = 0u;
    for (var c = 0u; c < nc; c++) {
        if (DEC[base + c * na] > conf) { n++; }
    }
    if (n == 0u) { return; }
    let region = b * (1u + 3u * cap);
    var at = atomicAdd(&OUT[region], n);
    for (var c = 0u; c < nc; c++) {
        let s = DEC[base + c * na];
        if (s > conf) {
            if (at < cap) {
                let r = region + 1u + 3u * at;
                atomicStore(&OUT[r], a);
                atomicStore(&OUT[r + 1u], c);
                atomicStore(&OUT[r + 2u], bitcast<u32>(s));
            }
            at++;
        }
    }
}
