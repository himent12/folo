//! Wasm SIMD128 (`f32x4`) and portable scalar row-wise softmax and batched axis-aligned box-IoU kernels.

/// Numerically stable row-wise softmax over `rows × cols` logits `src` into `dst`.
///
/// For each row, subtracts the row maximum before exponentiating and dividing by the row sum. Uses Wasm
/// SIMD128 (`f32x4`) 4-lane max reduction, minimax `exp`, sum, and scaling when compiled with `+simd128`.
/// Rows containing `NaN` propagate `NaN`; rows whose maximum is non-finite (`±∞`) fall back cleanly to
/// standard IEEE semantics (`exp(x - 0)` or `NaN`).
///
/// # Panics
/// If `src.len() != rows * cols` or `dst.len() != rows * cols`.
pub fn softmax_rows(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    let total = rows.checked_mul(cols).expect("softmax_rows: rows * cols overflow");
    assert_eq!(src.len(), total, "softmax_rows: src length mismatch");
    assert_eq!(dst.len(), total, "softmax_rows: dst length mismatch");
    if total == 0 {
        return;
    }
    for r in 0..rows {
        let s = &src[r * cols..(r + 1) * cols];
        let d = &mut dst[r * cols..(r + 1) * cols];
        softmax_row(s, d);
    }
}

fn softmax_row(src: &[f32], dst: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd::softmax_row(src, dst);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    softmax_row_scalar(src, dst);
}

#[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128"), allow(dead_code))]
fn softmax_row_scalar(src: &[f32], dst: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    let mut any_nan = false;
    for &v in src {
        if v.is_nan() {
            any_nan = true;
        } else if v > m {
            m = v;
        }
    }
    if any_nan {
        dst.fill(f32::NAN);
        return;
    }
    if !m.is_finite() {
        m = 0.0;
    }
    let mut sum = 0.0f32;
    for (d, &v) in dst.iter_mut().zip(src) {
        let e = (v - m).exp();
        *d = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for d in dst.iter_mut() {
        *d *= inv;
    }
}

/// Scalar axis-aligned IoU between two `[x1, y1, x2, y2]` boxes.
///
/// Returns `0.0` for disjoint, degenerate (`x2 <= x1` or `y2 <= y1`), zero-area, or non-finite boxes.
#[inline]
fn box_iou_scalar(a: [f32; 4], b: [f32; 4]) -> f32 {
    if !(a[0].is_finite()
        && a[1].is_finite()
        && a[2].is_finite()
        && a[3].is_finite()
        && b[0].is_finite()
        && b[1].is_finite()
        && b[2].is_finite()
        && b[3].is_finite())
    {
        return 0.0;
    }
    let aw = a[2] - a[0];
    let ah = a[3] - a[1];
    let bw = b[2] - b[0];
    let bh = b[3] - b[1];
    if !(aw > 0.0 && ah > 0.0 && bw > 0.0 && bh > 0.0) {
        return 0.0;
    }
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = aw * ah + bw * bh - inter;
    if union > 0.0 && union.is_finite() { inter / union } else { 0.0 }
}

/// Computes the `[boxes_a.len(), boxes_b.len()]` pairwise axis-aligned IoU matrix for `[x1, y1, x2, y2]` boxes.
///
/// Pre-transposes `boxes_b` into contiguous SoA arrays once and evaluates 4 candidate boxes per iteration via
/// Wasm SIMD128 (`f32x4`) when compiled with `+simd128`. Degenerate, inverted, zero-area, or non-finite boxes
/// yield `0.0`.
///
/// # Panics
/// If `out.len() != boxes_a.len() * boxes_b.len()`.
pub fn box_iou_matrix(boxes_a: &[[f32; 4]], boxes_b: &[[f32; 4]], out: &mut [f32]) {
    let n = boxes_a.len();
    let m = boxes_b.len();
    let total = n.checked_mul(m).expect("box_iou_matrix: n * m overflow");
    assert_eq!(out.len(), total, "box_iou_matrix: output length mismatch");
    if total == 0 {
        return;
    }

    // Precompute SoA layout for `boxes_b` (padded to 4-lane multiple for SIMD).
    let padded = m.div_ceil(4) * 4;
    let mut bx1 = vec![0.0f32; padded];
    let mut by1 = vec![0.0f32; padded];
    let mut bx2 = vec![0.0f32; padded];
    let mut by2 = vec![0.0f32; padded];
    let mut area_b = vec![0.0f32; padded];
    for (j, &b) in boxes_b.iter().enumerate() {
        let valid = b[0].is_finite() && b[1].is_finite() && b[2].is_finite() && b[3].is_finite();
        let bw = b[2] - b[0];
        let bh = b[3] - b[1];
        if valid && bw > 0.0 && bh > 0.0 {
            bx1[j] = b[0];
            by1[j] = b[1];
            bx2[j] = b[2];
            by2[j] = b[3];
            area_b[j] = bw * bh;
        }
    }

    for (i, &a) in boxes_a.iter().enumerate() {
        let row = &mut out[i * m..(i + 1) * m];
        let valid_a = a[0].is_finite() && a[1].is_finite() && a[2].is_finite() && a[3].is_finite();
        let aw = a[2] - a[0];
        let ah = a[3] - a[1];
        if !(valid_a && aw > 0.0 && ah > 0.0) {
            row.fill(0.0);
            continue;
        }
        let area_a = aw * ah;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            wasm_simd::box_iou_soa_row(a, area_a, &bx1, &by1, &bx2, &by2, &area_b, row);
            continue;
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        for j in 0..m {
            let ab = area_b[j];
            if ab <= 0.0 {
                row[j] = 0.0;
                continue;
            }
            let iw = (a[2].min(bx2[j]) - a[0].max(bx1[j])).max(0.0);
            let ih = (a[3].min(by2[j]) - a[1].max(by1[j])).max(0.0);
            let inter = iw * ih;
            let union = area_a + ab - inter;
            row[j] = if union > 0.0 && union.is_finite() { inter / union } else { 0.0 };
        }
    }
}

/// Zero-allocation one-to-many IoU between `query` and `boxes` (useful inside NMS suppression loops).
///
/// On Wasm SIMD128, transposes 4 AoS `[x1, y1, x2, y2]` boxes in registers via `u32x4_shuffle` without
/// allocating scratch buffers.
///
/// # Panics
/// If `out.len() != boxes.len()`.
pub fn box_iou_one_to_many(query: [f32; 4], boxes: &[[f32; 4]], out: &mut [f32]) {
    assert_eq!(out.len(), boxes.len(), "box_iou_one_to_many: output length mismatch");
    if boxes.is_empty() {
        return;
    }
    let valid_q =
        query[0].is_finite() && query[1].is_finite() && query[2].is_finite() && query[3].is_finite();
    let qw = query[2] - query[0];
    let qh = query[3] - query[1];
    if !(valid_q && qw > 0.0 && qh > 0.0) {
        out.fill(0.0);
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd::box_iou_one_to_many(query, qw * qh, boxes, out);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for (slot, &b) in out.iter_mut().zip(boxes) {
        *slot = box_iou_scalar(query, b);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use core::arch::wasm32::*;

    use super::box_iou_scalar;

    /// 4-lane minimax `exp(x)` accurate to `< 1e-6` relative error on `[-88, 88]`, flushing underflow to `0.0`.
    #[inline(always)]
    fn fast_exp4(x: v128) -> v128 {
        let xc = f32x4_pmin(f32x4_splat(88.722_84), f32x4_pmax(f32x4_splat(-104.0), x));
        let magic = f32x4_splat(12_582_912.0);
        let n = f32x4_sub(f32x4_add(f32x4_mul(xc, f32x4_splat(core::f32::consts::LOG2_E)), magic), magic);
        let r = f32x4_sub(
            f32x4_sub(xc, f32x4_mul(n, f32x4_splat(0.693_359_4))),
            f32x4_mul(n, f32x4_splat(-2.121_944_4e-4)),
        );
        let mut p = f32x4_splat(1.987_569_1e-4);
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(1.398_199_9e-3));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(8.333_452e-3));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(4.166_579_6e-2));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(1.666_666_5e-1));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(0.5));
        let y = f32x4_add(f32x4_add(f32x4_mul(f32x4_mul(p, r), r), r), f32x4_splat(1.0));
        let ni = i32x4_trunc_sat_f32x4(n);
        let n1 = i32x4_shr(ni, 1);
        let n2 = i32x4_sub(ni, n1);
        let bias = i32x4_splat(127);
        let s1 = i32x4_shl(i32x4_add(n1, bias), 23);
        let s2 = i32x4_shl(i32x4_add(n2, bias), 23);
        let v = f32x4_mul(f32x4_mul(y, s1), s2);
        v128_bitselect(x, v, f32x4_ne(x, x))
    }

    pub(super) fn softmax_row(src: &[f32], dst: &mut [f32]) {
        let mut mv = f32x4_splat(f32::NEG_INFINITY);
        let mut nan_mask = i32x4_splat(0);
        let (sc, sr) = src.as_chunks::<4>();
        for c in sc {
            // SAFETY: `c` is 4 floats.
            unsafe {
                let v = v128_load(c.as_ptr() as *const v128);
                mv = f32x4_pmax(mv, v);
                nan_mask = v128_or(nan_mask, f32x4_ne(v, v));
            }
        }
        let mut lanes = [0.0f32; 4];
        // SAFETY: `lanes` holds 4 floats.
        unsafe { v128_store(lanes.as_mut_ptr() as *mut v128, mv) };
        let mut m = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut any_nan = v128_any_true(nan_mask);
        for &v in sr {
            if v.is_nan() {
                any_nan = true;
            } else if v > m {
                m = v;
            }
        }
        if any_nan {
            dst.fill(f32::NAN);
            return;
        }
        if !m.is_finite() {
            m = 0.0;
        }

        let vm = f32x4_splat(m);
        let mut vsum = f32x4_splat(0.0);
        let (dc, dr) = dst.as_chunks_mut::<4>();
        for (s, d) in sc.iter().zip(dc.iter_mut()) {
            // SAFETY: `s` and `d` are 4 floats.
            unsafe {
                let e = fast_exp4(f32x4_sub(v128_load(s.as_ptr() as *const v128), vm));
                v128_store(d.as_mut_ptr() as *mut v128, e);
                vsum = f32x4_add(vsum, e);
            }
        }
        // SAFETY: `lanes` holds 4 floats.
        unsafe { v128_store(lanes.as_mut_ptr() as *mut v128, vsum) };
        let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for (d, &v) in dr.iter_mut().zip(sr) {
            let e = (v - m).exp();
            *d = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let vinv = f32x4_splat(inv);
        for d in dc.iter_mut() {
            // SAFETY: `d` is 4 floats.
            unsafe {
                let p = d.as_mut_ptr() as *mut v128;
                v128_store(p, f32x4_mul(v128_load(p), vinv));
            }
        }
        for d in dr {
            *d *= inv;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn box_iou_soa_row(
        a: [f32; 4],
        area_a: f32,
        bx1: &[f32],
        by1: &[f32],
        bx2: &[f32],
        by2: &[f32],
        area_b: &[f32],
        out: &mut [f32],
    ) {
        let m = out.len();
        let ax1 = f32x4_splat(a[0]);
        let ay1 = f32x4_splat(a[1]);
        let ax2 = f32x4_splat(a[2]);
        let ay2 = f32x4_splat(a[3]);
        let vaa = f32x4_splat(area_a);
        let zero = f32x4_splat(0.0);
        let inf = f32x4_splat(f32::INFINITY);

        let chunks = m / 4;
        for c in 0..chunks {
            let k = c * 4;
            // SAFETY: `k + 4 <= m <= bx1.len(), by1.len(), bx2.len(), by2.len(), area_b.len(), out.len()`.
            unsafe {
                let x1 = v128_load(bx1.as_ptr().add(k) as *const v128);
                let y1 = v128_load(by1.as_ptr().add(k) as *const v128);
                let x2 = v128_load(bx2.as_ptr().add(k) as *const v128);
                let y2 = v128_load(by2.as_ptr().add(k) as *const v128);
                let ab = v128_load(area_b.as_ptr().add(k) as *const v128);

                let iw = f32x4_pmax(zero, f32x4_sub(f32x4_pmin(ax2, x2), f32x4_pmax(ax1, x1)));
                let ih = f32x4_pmax(zero, f32x4_sub(f32x4_pmin(ay2, y2), f32x4_pmax(ay1, y1)));
                let inter = f32x4_mul(iw, ih);
                let union = f32x4_sub(f32x4_add(vaa, ab), inter);
                let valid =
                    v128_and(f32x4_gt(ab, zero), v128_and(f32x4_gt(union, zero), f32x4_lt(union, inf)));
                let iou = v128_bitselect(f32x4_div(inter, union), zero, valid);
                v128_store(out.as_mut_ptr().add(k) as *mut v128, iou);
            }
        }
        for j in chunks * 4..m {
            let ab = area_b[j];
            if ab <= 0.0 {
                out[j] = 0.0;
                continue;
            }
            let iw = (a[2].min(bx2[j]) - a[0].max(bx1[j])).max(0.0);
            let ih = (a[3].min(by2[j]) - a[1].max(by1[j])).max(0.0);
            let inter = iw * ih;
            let union = area_a + ab - inter;
            out[j] = if union > 0.0 && union.is_finite() { inter / union } else { 0.0 };
        }
    }

    pub(super) fn box_iou_one_to_many(query: [f32; 4], area_q: f32, boxes: &[[f32; 4]], out: &mut [f32]) {
        let qx1 = f32x4_splat(query[0]);
        let qy1 = f32x4_splat(query[1]);
        let qx2 = f32x4_splat(query[2]);
        let qy2 = f32x4_splat(query[3]);
        let vaq = f32x4_splat(area_q);
        let zero = f32x4_splat(0.0);
        let inf = f32x4_splat(f32::INFINITY);

        let chunks = boxes.len() / 4;
        for c in 0..chunks {
            let k = c * 4;
            // SAFETY: `k + 4 <= boxes.len() == out.len()`, each `[f32; 4]` is 16 bytes.
            unsafe {
                let p = boxes.as_ptr().add(k) as *const v128;
                let b0 = v128_load(p);
                let b1 = v128_load(p.add(1));
                let b2 = v128_load(p.add(2));
                let b3 = v128_load(p.add(3));
                // 4x4 transpose from AoS `[x1, y1, x2, y2]` to SoA `x1, y1, x2, y2`.
                let t0 = u32x4_shuffle::<0, 4, 1, 5>(b0, b1);
                let t1 = u32x4_shuffle::<0, 4, 1, 5>(b2, b3);
                let t2 = u32x4_shuffle::<2, 6, 3, 7>(b0, b1);
                let t3 = u32x4_shuffle::<2, 6, 3, 7>(b2, b3);
                let x1 = u32x4_shuffle::<0, 1, 4, 5>(t0, t1);
                let y1 = u32x4_shuffle::<2, 3, 6, 7>(t0, t1);
                let x2 = u32x4_shuffle::<0, 1, 4, 5>(t2, t3);
                let y2 = u32x4_shuffle::<2, 3, 6, 7>(t2, t3);

                let finite = v128_and(
                    v128_and(f32x4_lt(f32x4_abs(x1), inf), f32x4_lt(f32x4_abs(y1), inf)),
                    v128_and(f32x4_lt(f32x4_abs(x2), inf), f32x4_lt(f32x4_abs(y2), inf)),
                );
                let bw = f32x4_sub(x2, x1);
                let bh = f32x4_sub(y2, y1);
                let pos = v128_and(f32x4_gt(bw, zero), f32x4_gt(bh, zero));
                let ab = f32x4_mul(bw, bh);

                let iw = f32x4_pmax(zero, f32x4_sub(f32x4_pmin(qx2, x2), f32x4_pmax(qx1, x1)));
                let ih = f32x4_pmax(zero, f32x4_sub(f32x4_pmin(qy2, y2), f32x4_pmax(qy1, y1)));
                let inter = f32x4_mul(iw, ih);
                let union = f32x4_sub(f32x4_add(vaq, ab), inter);
                let ok =
                    v128_and(v128_and(finite, pos), v128_and(f32x4_gt(union, zero), f32x4_lt(union, inf)));
                let iou = v128_bitselect(f32x4_div(inter, union), zero, ok);
                v128_store(out.as_mut_ptr().add(k) as *mut v128, iou);
            }
        }
        for i in chunks * 4..boxes.len() {
            out[i] = box_iou_scalar(query, boxes[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_stability_accuracy_and_edges() {
        for &cols in &[0usize, 1, 2, 3, 4, 5, 7, 16, 33] {
            let rows = 3;
            let src: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    ((i * 19 + 5) % 41) as f32 * 0.4 - 8.0 + if i / cols.max(1) == 1 { 1000.0 } else { 0.0 }
                })
                .collect();
            let mut got = vec![0.0f32; rows * cols];
            softmax_rows(&src, rows, cols, &mut got);
            if cols == 0 {
                continue;
            }
            for r in 0..rows {
                let slice = &got[r * cols..(r + 1) * cols];
                let sum: f32 = slice.iter().sum();
                assert!((sum - 1.0).abs() <= 1e-5, "cols={cols} row={r}: sum={sum}");
                let mut ref_row = vec![0.0f32; cols];
                softmax_row_scalar(&src[r * cols..(r + 1) * cols], &mut ref_row);
                for (g, w) in slice.iter().zip(&ref_row) {
                    assert!((g - w).abs() <= 1e-5, "cols={cols} row={r}: {g} vs {w}");
                }
            }
        }

        // Shift invariance and NaN propagation.
        let base = [1.0f32, -2.5, 0.3, 4.2, -0.7];
        let shifted: Vec<f32> = base.iter().map(|&v| v - 500.0).collect();
        let (mut a, mut b) = ([0.0f32; 5], [0.0f32; 5]);
        softmax_rows(&base, 1, 5, &mut a);
        softmax_rows(&shifted, 1, 5, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 1e-6);
        }

        let with_nan = [1.0f32, f32::NAN, 2.0, 3.0, 4.0];
        let mut nan_out = [0.0f32; 5];
        softmax_rows(&with_nan, 1, 5, &mut nan_out);
        assert!(nan_out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn box_iou_matrix_and_one_to_many_cover_overlaps_and_degenerate_boxes() {
        let a = [
            [0.0f32, 0.0, 10.0, 10.0], // normal 10x10
            [5.0, 5.0, 15.0, 15.0],    // overlaps a[0] by 5x5 = 25 / (100+100-25) = 25/175
            [10.0, 10.0, 10.0, 20.0],  // zero width
            [5.0, 5.0, 2.0, 8.0],      // inverted x
            [f32::NAN, 0.0, 5.0, 5.0], // non-finite
        ];
        let b = [
            [0.0f32, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
            [20.0, 20.0, 30.0, 30.0],       // disjoint
            [2.0, 2.0, 8.0, 8.0],           // nested inside a[0] -> 36 / 100
            [0.0, 0.0, 0.0, 0.0],           // degenerate zero-area
            [0.0, 0.0, f32::INFINITY, 5.0], // non-finite
        ];
        let mut mat = vec![-1.0f32; a.len() * b.len()];
        box_iou_matrix(&a, &b, &mut mat);

        for (i, &ba) in a.iter().enumerate() {
            let mut row_o2m = vec![-1.0f32; b.len()];
            box_iou_one_to_many(ba, &b, &mut row_o2m);
            for (j, &bb) in b.iter().enumerate() {
                let want = box_iou_scalar(ba, bb);
                let got_m = mat[i * b.len() + j];
                let got_o = row_o2m[j];
                assert!((got_m - want).abs() <= 1e-6, "matrix[{i},{j}]: {got_m} vs {want}");
                assert!((got_o - want).abs() <= 1e-6, "o2m[{i},{j}]: {got_o} vs {want}");
            }
        }
        assert!((mat[0] - 1.0).abs() < 1e-6);
        assert!((mat[1] - 25.0 / 175.0).abs() < 1e-6);
        assert_eq!(mat[2], 0.0);
        assert!((mat[3] - 0.36).abs() < 1e-6);
        assert_eq!(mat[4], 0.0);
        assert_eq!(mat[5], 0.0);
        // Degenerate / NaN rows are all zeros.
        assert!(mat[2 * b.len()..].iter().all(|&v| v == 0.0));

        // Empty slices.
        let mut empty = Vec::new();
        box_iou_matrix(&[], &b, &mut empty);
        box_iou_one_to_many(a[0], &[], &mut empty);
    }
}
