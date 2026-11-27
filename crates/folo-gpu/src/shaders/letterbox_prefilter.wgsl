// Box-prefilter pass for letterbox_rgb8 when fx > 1 || fy > 1: reduces each fx x fy block of the
// interleaved u8 RGB source ONCE into a packed u32 (R | (G << 8) | (B << 16)) so the bilinear letterbox pass
// reads 4 aligned u32 words per output pixel instead of recomputing 4 full box averages per pixel.
// Params: 0 = pn (pw * ph), 1 = pw, 2 = sw, 3 = sh, 4 = fx, 5 = fy, 6 = src_offset (bytes).
@group(0) @binding(1) var<storage, read> SRC: array<u32>;
@group(0) @binding(2) var<storage, read_write> PRE: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let idx = linear_id(wid, nwg, lid, 256u);
    if (idx >= pu(0u)) { return; }
    let pw = pu(1u);
    let sw = pu(2u);
    let sh = pu(3u);
    let fx = pu(4u);
    let fy = pu(5u);
    let src_off = pu(6u);
    let px = idx % pw;
    let py = idx / pw;
    let x0 = px * fx;
    let x1 = min(x0 + fx, sw);
    let y0 = py * fy;
    let y1 = min(y0 + fy, sh);
    var s = vec3<u32>(0u);
    for (var y = y0; y < y1; y++) {
        var byte_off = src_off + (y * sw + x0) * 3u;
        var wi = byte_off >> 2u;
        var w0 = SRC[wi];
        // Each pixel consumes 3 bytes: phase 0 stays in w0 (next phase 3); phase 1 finishes w0 (loads next
        // word only if another pixel follows); phases 2 and 3 span w0..w1 and carry w1 forward as the next w0.
        for (var x = x0; x < x1; x++) {
            let phase = byte_off & 3u;
            if (phase == 0u) {
                s += vec3<u32>(w0 & 0xffu, (w0 >> 8u) & 0xffu, (w0 >> 16u) & 0xffu);
            } else if (phase == 1u) {
                s += vec3<u32>((w0 >> 8u) & 0xffu, (w0 >> 16u) & 0xffu, w0 >> 24u);
                if (x + 1u < x1) {
                    wi += 1u;
                    w0 = SRC[wi];
                }
            } else {
                wi += 1u;
                let w1 = SRC[wi];
                if (phase == 2u) {
                    s += vec3<u32>((w0 >> 16u) & 0xffu, w0 >> 24u, w1 & 0xffu);
                } else {
                    s += vec3<u32>(w0 >> 24u, w1 & 0xffu, (w1 >> 8u) & 0xffu);
                }
                w0 = w1;
            }
            byte_off += 3u;
        }
    }
    let n = (x1 - x0) * (y1 - y0);
    let r = (s + vec3<u32>(n >> 1u)) / vec3<u32>(n);
    PRE[idx] = r.x | (r.y << 8u) | (r.z << 16u);
}
