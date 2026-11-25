//! Single-pass contiguous concatenation along one axis.

use smallvec::SmallVec;

use super::super::{CHUNK, PAR_MIN, par_chunks};

/// Concatenate contiguous `srcs` (each `outer * piece_lens[p]` elements) into contiguous `dst`
/// (`outer * sum(piece_lens)` elements) in a single sequential pass over `dst`.
pub(crate) fn cat_contiguous_slices<T: Copy + Send + Sync>(
    dst: &mut [T],
    srcs: &[&[T]],
    piece_lens: &[usize],
    outer: usize,
) {
    assert_eq!(srcs.len(), piece_lens.len(), "cat_contiguous_slices: count mismatch");
    if dst.is_empty() || outer == 0 {
        return;
    }
    let mut ends: SmallVec<[usize; 8]> = SmallVec::with_capacity(piece_lens.len());
    let mut dst_row = 0usize;
    for (&src, &len) in srcs.iter().zip(piece_lens) {
        let expected = outer.checked_mul(len).expect("cat_contiguous_slices: piece size overflow");
        assert_eq!(src.len(), expected, "cat_contiguous_slices: source length mismatch");
        dst_row = dst_row.checked_add(len).expect("cat_contiguous_slices: row length overflow");
        ends.push(dst_row);
    }
    assert_eq!(
        dst.len(),
        outer.checked_mul(dst_row).expect("cat_contiguous_slices: output size overflow"),
        "cat_contiguous_slices: destination length mismatch"
    );
    if dst_row == 0 {
        return;
    }

    let copy_rows = |start_row: usize, out_slice: &mut [T]| {
        for (local_r, row_dst) in out_slice.chunks_exact_mut(dst_row).enumerate() {
            let r = start_row + local_r;
            let mut pos = 0;
            for (&src, &len) in srcs.iter().zip(piece_lens) {
                if len > 0 {
                    let s0 = r * len;
                    row_dst[pos..pos + len].copy_from_slice(&src[s0..s0 + len]);
                    pos += len;
                }
            }
        }
    };

    if dst.len() < PAR_MIN {
        copy_rows(0, dst);
    } else if dst_row < CHUNK {
        let rows_per_task = (CHUNK / dst_row).max(1);
        par_chunks(dst, rows_per_task * dst_row, &|t, chunk| copy_rows(t * rows_per_task, chunk));
    } else {
        // Large row (includes `outer == 1` and small-batch NCHW channel cat): one parallel pass over `CHUNK`
        // slices of `dst`, mapping each slice to the exact contiguous sub-ranges of `srcs`.
        par_chunks(dst, CHUNK, &|c, chunk| {
            let mut global_pos = c * CHUNK;
            let mut rem = chunk;
            while !rem.is_empty() {
                let r = global_pos / dst_row;
                let col = global_pos % dst_row;
                let p = ends.partition_point(|&e| e <= col);
                let piece_start = if p == 0 { 0 } else { ends[p - 1] };
                let in_piece = col - piece_start;
                let take = rem.len().min(ends[p] - col);
                let s0 = r * piece_lens[p] + in_piece;
                rem[..take].copy_from_slice(&srcs[p][s0..s0 + take]);
                rem = &mut rem[take..];
                global_pos += take;
            }
        });
    }
}
