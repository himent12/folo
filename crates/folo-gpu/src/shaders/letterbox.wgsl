// Letterbox one u8 RGB image into an f32 [3, OH, OW] slice of the output, sampling exactly as folo-data's
// resize (an integer box prefilter, then 11-bit fixed-point bilinear taps; see kernels/vision.rs).
// Params: 0 = OW, 1 = OH, 2 = source width, 3 = source height, 4 = pad x, 5 = pad y, 6 = resized width,
// 7 = resized height, 8 = fx, 9 = fy, 10 = pad value, 11 = scale (f32), 12 = output offset (elements),
// 13 = taps offset (words).
@group(0) @binding(1) var<storage, read> SRC: array<u32>;
@group(0) @binding(2) var<storage, read> TAPS: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

fn byte_at(i: u32) -> u32 {
    return (SRC[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

// Channel c of prefiltered pixel (px, py): the rounded mean of its fx x fy block, clipped to the image (the
// plain pixel without a prefilter). The host guarantees fx * fy * 255 fits in u32.
fn src_px(px: u32, py: u32, c: u32) -> u32 {
    let w = pu(2u);
    let h = pu(3u);
    let fx = pu(8u);
    let fy = pu(9u);
    if (fx == 1u && fy == 1u) {
        return byte_at((py * w + px) * 3u + c);
    }
    let x0 = px * fx;
    let x1 = min(x0 + fx, w);
    let y0 = py * fy;
    let y1 = min(y0 + fy, h);
    var s = 0u;
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            s += byte_at((y * w + x) * 3u + c);
        }
    }
    let n = (x1 - x0) * (y1 - y0);
    return (s + n / 2u) / n;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let ow = pu(0u);
    let oh = pu(1u);
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= ow * oh) { return; }
    let x = i % ow;
    let y = i / ow;
    let plane = ow * oh;
    let base = pu(12u) + i;
    let scale = pf(11u);
    let px = pu(4u);
    let py = pu(5u);
    let rw = pu(6u);
    let rh = pu(7u);
    if (x < px || x >= px + rw || y < py || y >= py + rh) {
        let v = f32(pu(10u)) * scale;
        Y[base] = v;
        Y[base + plane] = v;
        Y[base + 2u * plane] = v;
        return;
    }
    let dx = x - px;
    let dy = y - py;
    let tx = pu(13u);
    let a = TAPS[tx + dx];
    let b = TAPS[tx + rw + dx];
    let wx = TAPS[tx + 2u * rw + dx];
    let ty = tx + 3u * rw;
    let ya = TAPS[ty + dy];
    let yb = TAPS[ty + rh + dy];
    let wy = TAPS[ty + 2u * rh + dy];
    let iwx = 2048u - wx;
    let iwy = 2048u - wy;
    for (var c = 0u; c < 3u; c++) {
        let t = src_px(a, ya, c) * iwx + src_px(b, ya, c) * wx;
        let u = src_px(a, yb, c) * iwx + src_px(b, yb, c) * wx;
        let v = (t * iwy + u * wy + (1u << 21u)) >> 22u;
        Y[base + c * plane] = f32(v) * scale;
    }
}
