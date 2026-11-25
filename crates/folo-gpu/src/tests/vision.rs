//! Vision pre-processing and packed `u8` upload/normalize kernel tests against CPU references.

use super::{Rng, assert_close, gpu};
use crate::DType;
use crate::kernels::strided::{self, Elem};
use crate::kernels::vision::{
    LetterboxDesc, letterbox_rgb8, rgb8_hwc_to_chw_norm, u8_to_float_scale, upload_rgb8_hwc_to_chw_norm,
    upload_u8_scaled,
};

fn read_out(ctx: &crate::GpuContext, buf: &wgpu::Buffer, elems: usize, dtype: DType) -> Vec<f32> {
    let raw = ctx.read(buf, 0, (elems * dtype.size()) as u64).unwrap();
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(&raw).to_vec(),
        DType::F16 => {
            bytemuck::cast_slice::<u8, u16>(&raw).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
        }
    }
}

fn assert_vision_parity(got: &[f32], want: &[f32], dtype: DType, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    match dtype {
        DType::F32 => assert_close(got, want, 0.0, 0.0, ctx),
        // WGSL f32 -> f16 narrowing permits either adjacent representable binary16 value (<= 1 ULP).
        DType::F16 => {
            for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                let gb = half::f16::from_f32(g).to_bits();
                let wb = half::f16::from_f32(w).to_bits();
                let same_zero = (gb & 0x7fff) == 0 && (wb & 0x7fff) == 0;
                let same_sign = (gb & 0x8000) == (wb & 0x8000);
                assert!(
                    same_zero || (same_sign && gb.abs_diff(wb) <= 1),
                    "{ctx}[{i}]: f16 mismatch got={g} (0x{gb:04x}) want={w} (0x{wb:04x})"
                );
            }
        }
    }
}

fn cpu_scale(bytes: &[u8], scale: f32, dtype: DType) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            let v = b as f32 * scale;
            match dtype {
                DType::F32 => v,
                DType::F16 => half::f16::from_f32(v).to_f32(),
            }
        })
        .collect()
}

fn cpu_hwc_to_chw(rgb: &[u8], batch: usize, h: usize, w: usize, scale: f32, dtype: DType) -> Vec<f32> {
    let hw = h * w;
    let mut out = vec![0.0f32; batch * 3 * hw];
    for b in 0..batch {
        for p in 0..hw {
            let s = (b * hw + p) * 3;
            for c in 0..3 {
                let v = rgb[s + c] as f32 * scale;
                out[b * 3 * hw + c * hw + p] = match dtype {
                    DType::F32 => v,
                    DType::F16 => half::f16::from_f32(v).to_f32(),
                };
            }
        }
    }
    out
}

#[test]
fn u8_to_float_scale_parity_and_slack_safety() {
    let ctx = gpu!();
    let scale = 1.0 / 255.0;
    let mut dtypes = vec![DType::F32];
    if ctx.caps().shader_f16 {
        dtypes.push(DType::F16);
    }

    // Exercise every byte value 0..=255 plus VEC4, aligned-src odd-n/odd-out, and unaligned-src phases.
    let mut src_bytes = vec![0u8; 320];
    for (i, b) in src_bytes.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    let src_buf = ctx.upload(&src_bytes).unwrap();

    for dtype in dtypes {
        let sentinel = match dtype {
            DType::F32 => -17.25f32,
            DType::F16 => half::f16::from_f32(-17.25).to_f32(),
        };
        let cases = [
            // (n, src_offset, out_offset)
            (256usize, 0usize, 0usize), // VEC4, all 256 byte values
            (256, 4, 4),                // VEC4, non-zero 4-aligned offsets
            (0, 0, 0),                  // empty
            (1, 0, 0),                  // aligned src, odd n
            (3, 4, 0),                  // aligned src, odd n = 3
            (5, 0, 1),                  // aligned src, odd n and odd out_offset
            (257, 4, 3),                // aligned src, >1 workgroup element count + odd out
            (13, 1, 0),                 // unaligned src phase 1
            (17, 2, 2),                 // unaligned src phase 2
            (259, 3, 5),                // unaligned src phase 3, >256 elems
        ];
        for (n, src_off, out_off) in cases {
            let total_out = out_off + n + 7;
            let init_f32 = vec![sentinel; total_out];
            let out_buf = match dtype {
                DType::F32 => ctx.upload(bytemuck::cast_slice(&init_f32)).unwrap(),
                DType::F16 => {
                    let h: Vec<u16> = init_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
                    ctx.upload(bytemuck::cast_slice(&h)).unwrap()
                }
            };
            u8_to_float_scale(&ctx, src_buf.raw(), out_buf.raw(), n, src_off, out_off, scale, dtype).unwrap();
            let got = read_out(&ctx, out_buf.raw(), total_out, dtype);
            let want = cpu_scale(&src_bytes[src_off..src_off + n], scale, dtype);
            assert_vision_parity(
                &got[out_off..out_off + n],
                &want,
                dtype,
                &format!("u8_to_float_scale {dtype:?} n={n} src={src_off} out={out_off}"),
            );
            assert!(
                got[..out_off].iter().all(|&v| v == sentinel),
                "prefix slack clobbered for {dtype:?} n={n} src={src_off} out={out_off}"
            );
            assert!(
                got[out_off + n..].iter().all(|&v| v == sentinel),
                "suffix slack clobbered for {dtype:?} n={n} src={src_off} out={out_off}"
            );
        }

        // End-to-end staging upload + GPU normalize (aligned, odd, and empty).
        for n in [0usize, 1, 3, 15, 256, 261] {
            let buf = upload_u8_scaled(&ctx, &src_bytes[..n], scale, dtype).unwrap();
            if n > 0 {
                let got = read_out(&ctx, buf.raw(), n, dtype);
                let want = cpu_scale(&src_bytes[..n], scale, dtype);
                assert_vision_parity(&got, &want, dtype, &format!("upload_u8_scaled {dtype:?} n={n}"));
            }
        }
    }

    // Validation errors: too-small buffers reject cleanly before dispatch (pool MIN_CLASS is 256 bytes).
    let tiny_src = ctx.alloc(4).unwrap();
    let tiny_dst = ctx.alloc(4).unwrap();
    let out_ok = ctx.alloc(4096).unwrap();
    assert!(u8_to_float_scale(&ctx, tiny_src.raw(), tiny_dst.raw(), 128, 0, 0, scale, DType::F32).is_err());
    assert!(u8_to_float_scale(&ctx, tiny_src.raw(), out_ok.raw(), 512, 0, 0, scale, DType::F32).is_err());
    assert!(ctx.kernels_without_registry().is_empty());
}

#[test]
fn rgb8_hwc_to_chw_norm_parity_and_slack_safety() {
    let ctx = gpu!();
    let scale = 1.0 / 255.0;
    let mut rng = Rng::new(0x5eed_cafe_1234);
    let mut dtypes = vec![DType::F32];
    if ctx.caps().shader_f16 {
        dtypes.push(DType::F16);
    }

    // (batch, h, w, src_off, out_off) covering VEC4 fast path, odd H*W, batch > 1, and every alignment phase.
    let cases = [
        (1usize, 4usize, 4usize, 0usize, 0usize), // VEC4 single image
        (3, 8, 6, 4, 8),                          // VEC4 batch > 1, non-zero aligned offsets
        (2, 16, 17, 0, 0),                        // VEC4 with odd W (H*W = 272, multiple of 4)
        (1, 1, 1, 0, 0),                          // scalar 1x1
        (2, 5, 7, 0, 0),                          // scalar odd H*W = 35, batch > 1
        (2, 3, 5, 1, 3),                          // scalar unaligned src phase 1, odd out
        (2, 4, 4, 2, 0),                          // unaligned src phase 2 forces scalar
        (2, 4, 4, 3, 1),                          // unaligned src phase 3 + odd out
        (0, 4, 4, 0, 0),                          // empty batch
    ];

    for dtype in dtypes {
        let sentinel = match dtype {
            DType::F32 => -11.5f32,
            DType::F16 => half::f16::from_f32(-11.5).to_f32(),
        };
        for (b, h, w, src_off, out_off) in cases {
            let total = b * h * w * 3;
            let mut src_bytes = vec![0u8; src_off + total + 5];
            for byte in &mut src_bytes {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let src_buf = ctx.upload(&src_bytes).unwrap();
            let total_out = out_off + total + 6;
            let init_f32 = vec![sentinel; total_out];
            let out_buf = match dtype {
                DType::F32 => ctx.upload(bytemuck::cast_slice(&init_f32)).unwrap(),
                DType::F16 => {
                    let h16: Vec<u16> = init_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
                    ctx.upload(bytemuck::cast_slice(&h16)).unwrap()
                }
            };
            rgb8_hwc_to_chw_norm(&ctx, src_buf.raw(), out_buf.raw(), b, h, w, src_off, out_off, scale, dtype)
                .unwrap();
            let got = read_out(&ctx, out_buf.raw(), total_out, dtype);
            let want = cpu_hwc_to_chw(&src_bytes[src_off..src_off + total], b, h, w, scale, dtype);
            assert_vision_parity(
                &got[out_off..out_off + total],
                &want,
                dtype,
                &format!("rgb8_hwc_to_chw_norm {dtype:?} [{b},{h},{w}] src={src_off} out={out_off}"),
            );
            assert!(got[..out_off].iter().all(|&v| v == sentinel));
            assert!(got[out_off + total..].iter().all(|&v| v == sentinel));
        }

        // End-to-end staging upload + HWC->CHW normalize (both VEC4 and scalar odd H*W).
        for (b, h, w) in [(2usize, 8usize, 8usize), (2, 5, 7), (0, 4, 4)] {
            let rgb: Vec<u8> = (0..(b * h * w * 3)).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let buf = upload_rgb8_hwc_to_chw_norm(&ctx, &rgb, b, h, w, scale, dtype).unwrap();
            if b * h * w > 0 {
                let got = read_out(&ctx, buf.raw(), rgb.len(), dtype);
                let want = cpu_hwc_to_chw(&rgb, b, h, w, scale, dtype);
                assert_vision_parity(&got, &want, dtype, &format!("upload_rgb8 {dtype:?} [{b},{h},{w}]"));
            }
        }
    }

    assert!(upload_rgb8_hwc_to_chw_norm(&ctx, &[0u8; 5], 1, 2, 2, scale, DType::F32).is_err());
    assert!(ctx.kernels_without_registry().is_empty());
}

#[test]
fn letterbox_prefilter_specialized_and_plain_parity() {
    let ctx = gpu!();
    let scale = 1.0 / 255.0;
    let [sw, sh] = [10usize, 8usize];
    let src: Vec<u8> = (0..(sw * sh * 3)).map(|i| ((i * 53 + 7) % 256) as u8).collect();
    let src_buf = ctx.upload(&src).unwrap();

    // Test both PREFILTER=false (fx=1, fy=1) and PREFILTER=true (fx=2, fy=2) against exact CPU integer formula.
    for (fx, fy, rw, rh, ow, oh, px, py) in
        [(1usize, 1usize, 6usize, 4usize, 8usize, 6usize, 1usize, 1usize), (2, 2, 4, 3, 6, 5, 1, 1)]
    {
        let pw = sw.div_ceil(fx);
        let ph = sh.div_ceil(fy);
        let x0: Vec<u32> = (0..rw).map(|x| ((x * pw) / rw).min(pw - 1) as u32).collect();
        let x1: Vec<u32> = x0.iter().map(|&a| (a + 1).min(pw as u32 - 1)).collect();
        let wx: Vec<u32> = (0..rw).map(|x| ((x * 733) % 2049) as u32).collect();
        let y0: Vec<u32> = (0..rh).map(|y| ((y * ph) / rh).min(ph - 1) as u32).collect();
        let y1: Vec<u32> = y0.iter().map(|&a| (a + 1).min(ph as u32 - 1)).collect();
        let wy: Vec<u32> = (0..rh).map(|y| ((y * 911) % 2049) as u32).collect();
        let mut taps: Vec<u32> = Vec::new();
        taps.extend(&x0);
        taps.extend(&x1);
        taps.extend(&wx);
        taps.extend(&y0);
        taps.extend(&y1);
        taps.extend(&wy);
        let taps_buf = ctx.upload(bytemuck::cast_slice(&taps)).unwrap();
        let pad_value = 114u8;

        let sample_px = |gx: usize, gy: usize| -> [u32; 3] {
            if fx == 1 && fy == 1 {
                let off = (gy * sw + gx) * 3;
                return [src[off] as u32, src[off + 1] as u32, src[off + 2] as u32];
            }
            let bx0 = gx * fx;
            let bx1 = (bx0 + fx).min(sw);
            let by0 = gy * fy;
            let by1 = (by0 + fy).min(sh);
            let mut s = [0u32; 3];
            for y in by0..by1 {
                for x in bx0..bx1 {
                    let off = (y * sw + x) * 3;
                    for c in 0..3 {
                        s[c] += src[off + c] as u32;
                    }
                }
            }
            let n = ((bx1 - bx0) * (by1 - by0)) as u32;
            [(s[0] + n / 2) / n, (s[1] + n / 2) / n, (s[2] + n / 2) / n]
        };

        let mut want = vec![0.0f32; 3 * ow * oh];
        let plane = ow * oh;
        for y in 0..oh {
            for x in 0..ow {
                let idx = y * ow + x;
                if x < px || x >= px + rw || y < py || y >= py + rh {
                    let v = pad_value as f32 * scale;
                    want[idx] = v;
                    want[idx + plane] = v;
                    want[idx + 2 * plane] = v;
                } else {
                    let (dx, dy) = (x - px, y - py);
                    let p00 = sample_px(x0[dx] as usize, y0[dy] as usize);
                    let p10 = sample_px(x1[dx] as usize, y0[dy] as usize);
                    let p01 = sample_px(x0[dx] as usize, y1[dy] as usize);
                    let p11 = sample_px(x1[dx] as usize, y1[dy] as usize);
                    let (iwx, iwy) = (2048 - wx[dx], 2048 - wy[dy]);
                    for c in 0..3 {
                        let t = p00[c] * iwx + p10[c] * wx[dx];
                        let u = p01[c] * iwx + p11[c] * wx[dx];
                        let v = (t * iwy + u * wy[dy] + (1 << 21)) >> 22;
                        want[idx + c * plane] = v as f32 * scale;
                    }
                }
            }
        }

        let out_buf = ctx.alloc(DType::F32.bytes(3 * plane)).unwrap();
        let desc = LetterboxDesc {
            out: [ow, oh],
            src: [sw, sh],
            src_offset: 0,
            pad: [px, py],
            resized: [rw, rh],
            prefilter: [fx, fy],
            pad_value,
            scale,
            out_offset: 0,
            taps: 0,
        };
        letterbox_rgb8(&ctx, src_buf.raw(), taps_buf.raw(), out_buf.raw(), &desc, DType::F32).unwrap();
        let got = read_out(&ctx, out_buf.raw(), 3 * plane, DType::F32);
        assert_close(&got, &want, 0.0, 0.0, &format!("letterbox prefilter=({fx},{fy})"));
    }

    // Overflowing offsets that would wrap u32 on wasm32 must be rejected before dispatch.
    let small_taps = ctx.alloc(256).unwrap();
    let small_out = ctx.alloc(256).unwrap();
    let bad_base = LetterboxDesc {
        out: [2, 2],
        src: [2, 2],
        src_offset: 0,
        pad: [0, 0],
        resized: [2, 2],
        prefilter: [1, 1],
        pad_value: 114,
        scale,
        out_offset: 0,
        taps: 0,
    };
    let mut bad_src = bad_base;
    bad_src.src_offset = (u32::MAX - 7) as usize;
    assert!(
        letterbox_rgb8(&ctx, src_buf.raw(), small_taps.raw(), small_out.raw(), &bad_src, DType::F32).is_err()
    );
    let mut bad_taps = bad_base;
    bad_taps.taps = (u32::MAX - 4) as usize;
    assert!(
        letterbox_rgb8(&ctx, src_buf.raw(), small_taps.raw(), small_out.raw(), &bad_taps, DType::F32)
            .is_err()
    );
    let mut bad_out = bad_base;
    bad_out.out_offset = (u32::MAX - 4) as usize;
    assert!(
        letterbox_rgb8(&ctx, src_buf.raw(), small_taps.raw(), small_out.raw(), &bad_out, DType::F32).is_err()
    );
    assert!(ctx.kernels_without_registry().is_empty());
}

#[test]
fn strided_cast_u8_to_float_vec4_and_odd_exact() {
    let ctx = gpu!();
    let bytes: Vec<u8> = (0u8..=255).collect();
    let src_buf = ctx.upload(&bytes).unwrap();

    for (to, elem) in [(DType::F32, Elem::F32), (DType::F16, Elem::F16)] {
        // 4-aligned offset 0 and offset 4 (hits `cast_u8_vec4`), plus odd length & odd offset (fallback).
        for (offset, n) in [(0usize, 256usize), (4, 252), (0, 255), (1, 17), (3, 16)] {
            let dst_buf = ctx.alloc(elem.buffer_bytes(n).unwrap().max(4)).unwrap();
            strided::cast(&ctx, src_buf.raw(), Elem::U8, offset, dst_buf.raw(), elem, n).unwrap();
            let raw = ctx.read(dst_buf.raw(), 0, (n * elem.bytes()) as u64).unwrap();
            match to {
                DType::F32 => {
                    let got = bytemuck::cast_slice::<u8, f32>(&raw);
                    let want: Vec<f32> = bytes[offset..offset + n].iter().map(|&b| b as f32).collect();
                    assert_eq!(got, want.as_slice(), "u8 -> f32 off={offset} n={n}");
                }
                DType::F16 => {
                    let got = bytemuck::cast_slice::<u8, u16>(&raw);
                    let want: Vec<u16> = bytes[offset..offset + n]
                        .iter()
                        .map(|&b| half::f16::from_f32(b as f32).to_bits())
                        .collect();
                    assert_eq!(got, want.as_slice(), "u8 -> f16 off={offset} n={n}");
                }
            }
        }
    }
    assert!(ctx.kernels_without_registry().is_empty());
}
