// Letterbox one u8 RGB image into a [3, OH, OW] slice of the output, sampling exactly as folo-data's
// resize (an integer box prefilter, then 11-bit fixed-point bilinear taps; see kernels/vision.rs).
// Spec: PREFILTER (bool: fx > 1 || fy > 1), f16 (T is f16 or f32).
// Params: 0 = OW, 1 = OH, 2 = source width, 3 = source height, 4 = pad x, 5 = pad y, 6 = resized width,
// 7 = resized height, 8 = fx, 9 = fy, 10 = pad value, 11 = scale (f32), 12 = output offset (elements),
// 13 = taps offset (words), 14 = source offset (bytes). T is the output type (f32, or f16 rounded once from the
// f32 value).
@group(0) @binding(1) var<storage, read> SRC: array<u32>;
@group(0) @binding(2) var<storage, read> TAPS: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<T>;

// Reads three contiguous RGB bytes starting at `byte_off` (relative to source offset `pu(14u)`), touching only
// the 1 or 2 storage words that actually contain those bytes.
fn rgb_at(byte_off: u32) -> vec3<u32> {
    let i = byte_off + pu(14u);
    let wi = i >> 2u;
    let w0 = SRC[wi];
    let s = (i & 3u) * 8u;
    if (s == 0u) {
        return vec3<u32>(w0 & 0xffu, (w0 >> 8u) & 0xffu, (w0 >> 16u) & 0xffu);
    }
    if (s == 8u) {
        return vec3<u32>((w0 >> 8u) & 0xffu, (w0 >> 16u) & 0xffu, w0 >> 24u);
    }
    let w1 = SRC[wi + 1u];
    if (s == 16u) {
        return vec3<u32>((w0 >> 16u) & 0xffu, w0 >> 24u, w1 & 0xffu);
    }
    return vec3<u32>(w0 >> 24u, w1 & 0xffu, (w1 >> 8u) & 0xffu);
}

// All three RGB channels of prefiltered pixel (px, py) in a single pass: the rounded mean of its fx x fy block,
// clipped to the image (the plain pixel without a prefilter). The host guarantees fx * fy * 255 + (fx * fy) / 2
// fits in u32.
fn src_px3(px: u32, py: u32) -> vec3<u32> {
    let w = pu(2u);
    if (!PREFILTER) {
        return rgb_at((py * w + px) * 3u);
    }
    let h = pu(3u);
    let fx = pu(8u);
    let fy = pu(9u);
    let x0 = px * fx;
    let x1 = min(x0 + fx, w);
    let y0 = py * fy;
    let y1 = min(y0 + fy, h);
    var s = vec3<u32>(0u);
    for (var y = y0; y < y1; y++) {
        var row_off = (y * w + x0) * 3u;
        for (var x = x0; x < x1; x++) {
            s += rgb_at(row_off);
            row_off += 3u;
        }
    }
    let n = (x1 - x0) * (y1 - y0);
    return (s + vec3<u32>(n / 2u)) / vec3<u32>(n);
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
        let v = T(f32(pu(10u)) * scale);
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
    let p00 = src_px3(a, ya);
    let p10 = src_px3(b, ya);
    let p01 = src_px3(a, yb);
    let p11 = src_px3(b, yb);
    let t = p00 * iwx + p10 * wx;
    let u = p01 * iwx + p11 * wx;
    let v = (t * iwy + u * wy + vec3<u32>(1u << 21u)) >> vec3<u32>(22u);
    Y[base] = T(f32(v.x) * scale);
    Y[base + plane] = T(f32(v.y) * scale);
    Y[base + 2u * plane] = T(f32(v.z) * scale);
}
