// Letterbox one u8 RGB image (or packed u32 prefiltered image when PACKED_SRC = true) into a [3, OH, OW]
// slice of the output, sampling exactly as folo-data's resize (an integer box prefilter, then 11-bit
// fixed-point bilinear taps; see kernels/vision.rs).
// Spec: PREFILTER (bool: 1-pass box prefilter), PACKED_SRC (bool: read packed u32 [R,G,B,0] from Pass 1),
// VEC4 (bool: 4-pixel coalesced vec4<T> stores when OW % 4 == 0 && out_offset % 4 == 0), f16 (T is f16 or f32).
// Params: 0 = OW, 1 = OH, 2 = source width (pw when PACKED_SRC), 3 = source height (ph when PACKED_SRC),
// 4 = pad x, 5 = pad y, 6 = resized width, 7 = resized height, 8 = fx, 9 = fy, 10 = pad value,
// 11 = scale (f32), 12 = output offset (elements when !VEC4, vec4 units when VEC4),
// 13 = taps offset (words), 14 = source offset (bytes; 0 when PACKED_SRC).

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

// All three RGB channels of prefiltered pixel (px, py) in a single pass.
fn src_px3(px: u32, py: u32) -> vec3<u32> {
    let w = pu(2u);
    if (PACKED_SRC) {
        let p = SRC[py * w + px];
        return vec3<u32>(p & 0xffu, (p >> 8u) & 0xffu, (p >> 16u) & 0xffu);
    }
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

// Exact 11-bit fixed-point bilinear sample at horizontal tap `dx`, given row-uniform vertical tap
// `(ya, yb, wy)`. Bit-exact with `((p00*iwx + p10*wx)*iwy + (p01*iwx + p11*wx)*wy + (1 << 21)) >> 22`.
// Branches only on row-uniform `wy == 0` (and `wx == 0` inside that identity-row branch) to avoid warp
// divergence on fractional resizes while still cutting 75% of loads on exact identity/integer resizes.
fn sample_bilinear(dx: u32, rw: u32, tx: u32, ya: u32, yb: u32, wy: u32, iwy: u32) -> vec3<u32> {
    let a = TAPS[tx + dx];
    let wx = TAPS[tx + 2u * rw + dx];
    if (wy == 0u) {
        let p00 = src_px3(a, ya);
        if (wx == 0u) {
            return p00;
        }
        let b = TAPS[tx + rw + dx];
        let t = p00 * (2048u - wx) + src_px3(b, ya) * wx;
        return (t + vec3<u32>(1024u)) >> vec3<u32>(11u);
    }
    let b = TAPS[tx + rw + dx];
    let iwx = 2048u - wx;
    let t = src_px3(a, ya) * iwx + src_px3(b, ya) * wx;
    let u = src_px3(a, yb) * iwx + src_px3(b, yb) * wx;
    return (t * iwy + u * wy + vec3<u32>(1u << 21u)) >> vec3<u32>(22u);
}
