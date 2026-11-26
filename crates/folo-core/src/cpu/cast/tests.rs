use super::*;
use crate::Shape;

#[test]
fn f16_to_f32_exhaustive_matches_half() {
    let all_f16: Vec<f16> = (0u16..=u16::MAX).map(f16::from_bits).collect();
    let mut got = vec![0f32; all_f16.len()];
    f16_to_f32_slice(&all_f16, &mut got);
    for (&h, &v) in all_f16.iter().zip(&got) {
        let want = h.to_f32();
        if want.is_nan() {
            assert!(v.is_nan(), "f16 {:#06x}: expected NaN, got {v}", h.to_bits());
        } else {
            assert_eq!(
                v.to_bits(),
                want.to_bits(),
                "f16 {:#06x}: got {:#010x}, want {:#010x}",
                h.to_bits(),
                v.to_bits(),
                want.to_bits()
            );
        }
    }
    // Also test ragged tails (lengths 0..17) and parallel cast (> PAR_MIN).
    for len in 0..17 {
        let mut tail = vec![0f32; len];
        f16_to_f32_slice(&all_f16[..len], &mut tail);
        for i in 0..len {
            assert_eq!(tail[i].to_bits(), all_f16[i].to_f32().to_bits());
        }
    }
    let storage = CpuStorage::F16(all_f16.clone());
    let l = Layout::contiguous(Shape::new(&[all_f16.len()]));
    let CpuStorage::F32(par_out) = cast(&storage, &l, DType::F32) else {
        panic!("expected F32");
    };
    assert_eq!(par_out.len(), got.len());
    for (a, b) in par_out.iter().zip(&got) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn f32_to_f16_matches_half_across_ties_subnormals_and_extremes() {
    let mut xs: Vec<f32> = (0u16..=u16::MAX)
        .filter_map(|b| {
            let h = f16::from_bits(b);
            (!h.is_nan()).then(|| h.to_f32())
        })
        .collect();
    // Exact halfway ties-to-even (normal and subnormal), underflow, overflow, signed zeros, NaNs.
    xs.extend([
        0.0,
        -0.0,
        f32::from_bits(0x3300_0000), // half of min f16 subnormal (2^-25 -> rounds to even 0)
        f32::from_bits(0x3300_0001), // just above half of min f16 subnormal -> min subnormal
        f32::from_bits(0x3380_0000), // min f16 subnormal (2^-24)
        f32::from_bits(0x33c0_0000), // 1.5 * 2^-24 -> rounds to even 2 * 2^-24
        f32::from_bits(0x3880_0000), // min f16 normal (2^-14)
        1.0 + f32::exp2(-11.0),      // exact halfway above even mantissa -> 1.0
        1.0 + f32::exp2(-10.0) + f32::exp2(-11.0), // exact halfway above odd mantissa -> rounds up
        65504.0,                     // max finite f16
        65519.996,                   // rounds down to 65504.0
        65520.0,                     // exact tie -> overflows to +inf
        -65520.0,                    // overflows to -inf
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0x7f80_0001), // sNaN
    ]);
    // Non-multiple of 8 length exercising tail path + parallel threshold.
    while xs.len() % 8 != 5 {
        xs.push(0.125);
    }
    let storage = CpuStorage::F32(xs.clone());
    let l = Layout::contiguous(Shape::new(&[xs.len()]));
    let CpuStorage::F16(got) = cast(&storage, &l, DType::F16) else {
        panic!("expected F16");
    };
    for (&x, &h) in xs.iter().zip(&got) {
        let want = f16::from_f32(x);
        if want.is_nan() {
            assert!(h.is_nan(), "f32 {x} ({:#010x}): expected NaN, got {:#06x}", x.to_bits(), h.to_bits());
        } else {
            assert_eq!(
                h.to_bits(),
                want.to_bits(),
                "f32 {x} ({:#010x}): got {:#06x}, want {:#06x}",
                x.to_bits(),
                h.to_bits(),
                want.to_bits()
            );
        }
    }
}

#[test]
fn cast_handles_empty_offset_and_non_contiguous_views() {
    let empty = Layout::new(Shape::new(&[0, 4]), vec![4, 1].into(), 10);
    let CpuStorage::F16(v) = cast(&CpuStorage::F32(vec![]), &empty, DType::F16) else {
        panic!("expected F16");
    };
    assert!(v.is_empty());

    // Transposed F32 -> F16 and I64 -> F32 (single-rounding above 2^53).
    let src: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
    let lt = Layout::contiguous(Shape::new(&[4, 6])).transpose(0, 1);
    let CpuStorage::F16(h) = cast(&CpuStorage::F32(src.clone()), &lt, DType::F16) else {
        panic!("expected F16");
    };
    for c in 0..6 {
        for r in 0..4 {
            assert_eq!(h[c * 4 + r].to_bits(), f16::from_f32(src[r * 6 + c]).to_bits());
        }
    }

    let big: i64 = (1i64 << 53) + 1;
    let l1 = Layout::contiguous(Shape::new(&[1]));
    let CpuStorage::F32(out) = cast(&CpuStorage::I64(vec![big]), &l1, DType::F32) else {
        panic!("expected F32");
    };
    assert_eq!(out[0].to_bits(), (big as f32).to_bits());
}

#[test]
fn u8_f32_u32_avx2_casts_match_scalar_bitwise() {
    // 1. U8 -> F32: all 256 byte values, odd lengths 0..=65, and parallel > PAR_MIN.
    let bytes: Vec<u8> = (0u8..=255).collect();
    for len in [0usize, 1, 7, 8, 9, 15, 31, 32, 33, 65, 256] {
        let mut got = vec![0f32; len];
        u8_to_f32_slice(&bytes[..len], &mut got);
        for (i, (&g, &b)) in got.iter().zip(&bytes[..len]).enumerate() {
            assert_eq!(g.to_bits(), (b as f32).to_bits(), "u8->f32 len={len} at {i}");
        }
    }
    let big_u8: Vec<u8> = (0..PAR_MIN + 37).map(|i| (i * 31) as u8).collect();
    let CpuStorage::F32(got_big) =
        cast(&CpuStorage::U8(big_u8.clone()), &Layout::contiguous(Shape::new(&[big_u8.len()])), DType::F32)
    else {
        panic!("expected F32");
    };
    for (&g, &b) in got_big.iter().zip(&big_u8) {
        assert_eq!(g.to_bits(), (b as f32).to_bits());
    }

    // 2. F32 -> U8: NaNs, ±Inf, ±0.0, negatives, fractional truncation, saturation, tails, strided, parallel.
    let mut f_vals = vec![
        0.0f32,
        -0.0,
        -0.1,
        -1.0,
        -1000.0,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0x7f80_0001),
        0.0001,
        0.5,
        0.9999,
        1.0,
        1.9999,
        127.5,
        128.5,
        254.0,
        254.9999,
        255.0,
        255.0001,
        255.9,
        256.0,
        1e20,
        f32::MAX,
        f32::INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    for i in 0..45 {
        f_vals.push(i as f32 * 6.25 - 12.3);
    }
    for len in [0usize, 1, 7, 8, 9, 15, 31, 32, 33, 41, f_vals.len()] {
        let mut got = vec![0u8; len];
        f32_to_u8_slice(&f_vals[..len], &mut got);
        for (i, (&g, &x)) in got.iter().zip(&f_vals[..len]).enumerate() {
            assert_eq!(g, x as u8, "f32->u8 len={len} at {i} x={x}");
        }
    }
    let big_f: Vec<f32> = (0..PAR_MIN + 41).map(|i| (i % 300) as f32 - 20.5).collect();
    let CpuStorage::U8(got_u8) =
        cast(&CpuStorage::F32(big_f.clone()), &Layout::contiguous(Shape::new(&[big_f.len()])), DType::U8)
    else {
        panic!("expected U8");
    };
    for (&g, &x) in got_u8.iter().zip(&big_f) {
        assert_eq!(g, x as u8);
    }
    let lt = Layout::contiguous(Shape::new(&[4, 9])).transpose(0, 1);
    let CpuStorage::U8(got_t) = cast(&CpuStorage::F32(f_vals[..36].to_vec()), &lt, DType::U8) else {
        panic!("expected U8");
    };
    for c in 0..9 {
        for r in 0..4 {
            assert_eq!(got_t[c * 4 + r], f_vals[r * 9 + c] as u8);
        }
    }

    // 3. U32 -> F32: full 32-bit range boundaries including exact halfway ties-to-even, tails, and parallel.
    let mut u_vals = vec![
        0u32,
        1,
        0xFFFF,
        0x1_0000,
        0x1_0001,
        (1u32 << 24) - 1,
        1u32 << 24,
        (1u32 << 24) + 1,
        (1u32 << 24) + 2,
        (1u32 << 24) + 3,
        0x7FFF_FF7F,
        0x7FFF_FF80,
        0x7FFF_FFBF,
        0x7FFF_FFC0,
        0x7FFF_FFFF,
        0x8000_0000,
        0x8000_0001,
        0x8000_0080,
        0x8000_0081,
        0x8000_0180,
        0xFFFF_FF7F,
        0xFFFF_FF80,
        0xFFFF_FF81,
        u32::MAX,
    ];
    let mut rng = crate::rng::Rng::seed(12345);
    for _ in 0..50 {
        u_vals.push((rng.next_u64() >> 17) as u32);
    }
    for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 33, u_vals.len()] {
        let mut got = vec![0f32; len];
        u32_to_f32_slice(&u_vals[..len], &mut got);
        for (i, (&g, &u)) in got.iter().zip(&u_vals[..len]).enumerate() {
            assert_eq!(g.to_bits(), (u as f32).to_bits(), "u32->f32 len={len} at {i} u={u:#010x}");
        }
    }
    let big_u32: Vec<u32> = (0..PAR_MIN + 29).map(|i| (i as u32).wrapping_mul(2_654_435_761)).collect();
    let CpuStorage::F32(got_uf) = cast(
        &CpuStorage::U32(big_u32.clone()),
        &Layout::contiguous(Shape::new(&[big_u32.len()])),
        DType::F32,
    ) else {
        panic!("expected F32");
    };
    for (&g, &u) in got_uf.iter().zip(&big_u32) {
        assert_eq!(g.to_bits(), (u as f32).to_bits());
    }
}

/// Verify the exact branchless integer `f16 <-> f32` algorithm used by `f16x4_to_f32x4_wasm` and
/// `f32x4_to_f16x4_wasm` natively across all 65,536 `f16` bit patterns and edge `f32` values, plus ragged
/// lengths around the 32-wide F16C unroll boundary.
#[test]
fn branchless_f16_f32_algorithm_and_32wide_tails_are_bit_exact() {
    fn branchless_f16_to_f32(h: u16) -> f32 {
        let h32 = h as u32;
        let sign = (h32 & 0x8000) << 16;
        let exp = (h32 >> 10) & 0x1F;
        let mant = h32 & 0x03FF;
        let norm = sign | ((exp + 112) << 23) | (mant << 13);
        let sub = sign | ((mant as f32) * f32::from_bits(0x3380_0000)).to_bits();
        let inf_nan = sign | 0x7F80_0000 | (mant << 13) | if mant != 0 { 0x0040_0000 } else { 0 };
        let bits = if exp == 31 {
            inf_nan
        } else if exp == 0 {
            sub
        } else {
            norm
        };
        f32::from_bits(bits)
    }

    fn branchless_f32_to_f16(x: f32) -> f16 {
        let u = x.to_bits();
        let sign = (u >> 16) & 0x8000;
        let abs_u = u & 0x7FFF_FFFF;
        let sub_h = (f32::from_bits(abs_u) + 0.5f32).to_bits().wrapping_sub(0x3F00_0000);
        let lsb = (abs_u >> 13) & 1;
        let norm_h = abs_u.wrapping_sub(0x3800_0000).wrapping_add(0x0FFF).wrapping_add(lsb) >> 13;
        let nan_h = 0x7E00 | ((abs_u >> 13) & 0x03FF);
        let mag = if abs_u > 0x7F80_0000 {
            nan_h
        } else if abs_u >= 0x4780_0000 {
            0x7C00
        } else if abs_u < 0x3880_0000 {
            sub_h
        } else {
            norm_h
        };
        f16::from_bits((sign | mag) as u16)
    }

    // Exhaustive 65,536 f16 -> f32 check.
    for b in 0u16..=u16::MAX {
        let h = f16::from_bits(b);
        let got = branchless_f16_to_f32(b);
        let want = h.to_f32();
        if want.is_nan() {
            assert!(got.is_nan(), "f16 {b:#06x}: expected NaN, got {got}");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "f16 {b:#06x}");
        }
    }
    // Exhaustive round-trip + ties, subnormals, subnormal-to-normal carry, overflow, NaNs.
    let mut probe: Vec<f32> = (0u16..=u16::MAX).map(|b| f16::from_bits(b).to_f32()).collect();
    probe.extend([
        0.0,
        -0.0,
        f32::from_bits(0x3300_0000),
        f32::from_bits(0x3300_0001),
        f32::from_bits(0x3380_0000),
        f32::from_bits(0x33C0_0000),
        f32::from_bits(0x387F_EFFF), // just below subnormal->normal halfway
        f32::from_bits(0x387F_F000), // exact halfway max subnormal -> min normal (carries to 0x0400)
        f32::from_bits(0x3880_0000), // min normal
        1.0 + f32::exp2(-11.0),
        1.0 + f32::exp2(-10.0) + f32::exp2(-11.0),
        65504.0,
        65519.996,
        65520.0,
        -65520.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0x7f80_0001),
    ]);
    for &x in &probe {
        let got = branchless_f32_to_f16(x);
        let want = f16::from_f32(x);
        if want.is_nan() {
            assert!(got.is_nan(), "f32 {:#010x}: expected NaN", x.to_bits());
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "f32 {x} ({:#010x})", x.to_bits());
        }
    }
    // Check 32-wide unroll boundary tails for direct-pointer F16C paths.
    for len in [0usize, 1, 7, 8, 9, 15, 24, 31, 32, 33, 39, 40, 41, 63, 64, 65] {
        let xs: Vec<f32> = (0..len).map(|i| (i as f32 - 17.5) * 0.375).collect();
        let mut hs = vec![f16::ZERO; len];
        f32_to_f16_slice(&xs, &mut hs);
        for (i, (&x, &h)) in xs.iter().zip(&hs).enumerate() {
            assert_eq!(h.to_bits(), f16::from_f32(x).to_bits(), "f32->f16 len={len} at {i}");
        }
        let mut back = vec![0f32; len];
        f16_to_f32_slice(&hs, &mut back);
        for (i, (&h, &v)) in hs.iter().zip(&back).enumerate() {
            assert_eq!(v.to_bits(), h.to_f32().to_bits(), "f16->f32 len={len} at {i}");
        }
    }
}
