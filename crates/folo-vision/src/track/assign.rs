//! Sparse-pruned exact rectangular Hungarian (Kuhn-Munkres) linear assignment for IoU track association.

use crate::boxes::{IouKind, Xyxy, iou as box_iou};
use crate::nms::Detection;

use super::Track;

/// Sentinel cost for disallowed track-detection pairs in the Hungarian assignment matrix.
const DISALLOWED_COST: f64 = 1e6;

/// Exact minimum-cost linear assignment (Kuhn-Munkres / Hungarian algorithm) for rectangular `n_rows × n_cols`
/// matrices. Returns matched `(row, col)` pairs sorted by ascending `row`.
pub(crate) fn hungarian(n_rows: usize, n_cols: usize, costs: &[f64]) -> Vec<(usize, usize)> {
    if n_rows == 0 || n_cols == 0 {
        return Vec::new();
    }
    if n_rows > n_cols {
        let mut transposed = vec![0.0f64; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                transposed[c * n_rows + r] = costs[r * n_cols + c];
            }
        }
        let mut pairs: Vec<(usize, usize)> =
            hungarian(n_cols, n_rows, &transposed).into_iter().map(|(c, r)| (r, c)).collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        return pairs;
    }

    let mut u = vec![0.0f64; n_rows + 1];
    let mut v = vec![0.0f64; n_cols + 1];
    let mut p = vec![0usize; n_cols + 1];
    let mut way = vec![0usize; n_cols + 1];
    let mut minv = vec![0.0f64; n_cols + 1];
    let mut used = vec![false; n_cols + 1];

    for i in 1..=n_rows {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(f64::INFINITY);
        used.fill(false);
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            let row_off = (i0 - 1) * n_cols;
            for j in 1..=n_cols {
                if !used[j] {
                    let cur = costs[row_off + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n_cols {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs = Vec::with_capacity(n_rows);
    for (j, &row) in p.iter().enumerate().skip(1) {
        if row != 0 {
            pairs.push((row - 1, j - 1));
        }
    }
    pairs.sort_unstable_by_key(|&(r, _)| r);
    pairs
}

/// Associate `tracks` with `dets` by optimal IoU matching (`IoU >= min_iou`, class-matched when `class_aware`).
///
/// Prunes isolated tracks and detections with zero legal edges before building the Hungarian matrix so crowded
/// buffers with sparse viable overlaps avoid `O(N^3)` dense solves. Returns `(matched_pairs,
/// unmatched_track_positions, unmatched_det_positions)`.
pub(crate) fn associate(
    tracks: &[&Track],
    dets: &[&Detection],
    min_iou: f32,
    class_aware: bool,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let nt = tracks.len();
    let nd = dets.len();
    if nt == 0 || nd == 0 {
        return (Vec::new(), (0..nt).collect(), (0..nd).collect());
    }

    let track_boxes: Vec<Xyxy> = tracks.iter().map(|t| t.kalman.bbox()).collect();
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut row_has_edge = vec![false; nt];
    let mut col_has_edge = vec![false; nd];

    for (ti, (track, tb)) in tracks.iter().zip(&track_boxes).enumerate() {
        for (di, det) in dets.iter().enumerate() {
            if class_aware && track.class_id != det.class {
                continue;
            }
            let iou = box_iou(tb, &det.bbox, IouKind::Iou);
            if iou.is_finite() && iou >= min_iou {
                edges.push((ti, di, (1.0 - iou as f64).clamp(0.0, 1.0)));
                row_has_edge[ti] = true;
                col_has_edge[di] = true;
            }
        }
    }

    if edges.is_empty() {
        return (Vec::new(), (0..nt).collect(), (0..nd).collect());
    }

    // Compact active rows/cols into dense sub-indices before calling Hungarian.
    let mut active_rows = Vec::new();
    let mut row_to_compact = vec![usize::MAX; nt];
    for (ti, &has) in row_has_edge.iter().enumerate() {
        if has {
            row_to_compact[ti] = active_rows.len();
            active_rows.push(ti);
        }
    }
    let mut active_cols = Vec::new();
    let mut col_to_compact = vec![usize::MAX; nd];
    for (di, &has) in col_has_edge.iter().enumerate() {
        if has {
            col_to_compact[di] = active_cols.len();
            active_cols.push(di);
        }
    }

    let r_act = active_rows.len();
    let c_act = active_cols.len();
    let mut compact_costs = vec![DISALLOWED_COST; r_act * c_act];
    for &(ti, di, cost) in &edges {
        compact_costs[row_to_compact[ti] * c_act + col_to_compact[di]] = cost;
    }

    let assigned = hungarian(r_act, c_act, &compact_costs);
    let mut matched = Vec::with_capacity(assigned.len());
    let mut t_used = vec![false; nt];
    let mut d_used = vec![false; nd];
    for (rc, cc) in assigned {
        if compact_costs[rc * c_act + cc] < DISALLOWED_COST * 0.5 {
            let ti = active_rows[rc];
            let di = active_cols[cc];
            matched.push((ti, di));
            t_used[ti] = true;
            d_used[di] = true;
        }
    }
    let unmatched_t = (0..nt).filter(|&i| !t_used[i]).collect();
    let unmatched_d = (0..nd).filter(|&i| !d_used[i]).collect();
    (matched, unmatched_t, unmatched_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::kalman::KalmanBox;
    use crate::track::{Track, TrackState};

    #[test]
    fn hungarian_matches_bruteforce_optimum_on_rectangular_matrices() {
        fn brute_best(r: usize, c: usize, costs: &[f64]) -> f64 {
            fn dfs(
                row: usize,
                r: usize,
                c: usize,
                costs: &[f64],
                used: &mut [bool],
                acc: f64,
                best: &mut f64,
            ) {
                if row == r {
                    if acc < *best {
                        *best = acc;
                    }
                    return;
                }
                for col in 0..c {
                    if !used[col] {
                        used[col] = true;
                        dfs(row + 1, r, c, costs, used, acc + costs[row * c + col], best);
                        used[col] = false;
                    }
                }
            }
            let mut best = f64::INFINITY;
            if r <= c {
                let mut used = vec![false; c];
                dfs(0, r, c, costs, &mut used, 0.0, &mut best);
            } else {
                let mut t = vec![0.0; r * c];
                for i in 0..r {
                    for j in 0..c {
                        t[j * r + i] = costs[i * c + j];
                    }
                }
                let mut used = vec![false; r];
                dfs(0, c, r, &t, &mut used, 0.0, &mut best);
            }
            best
        }

        let mut seed = 0x517c_c1b7_2722_0a95u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for &(r, c) in &[(1, 1), (2, 3), (3, 2), (4, 4), (3, 5), (5, 3)] {
            for _ in 0..25 {
                let costs: Vec<f64> =
                    (0..r * c).map(|_| if next() < 0.25 { DISALLOWED_COST } else { next() }).collect();
                let got = hungarian(r, c, &costs);
                assert_eq!(got.len(), r.min(c));
                let got_cost: f64 = got.iter().map(|&(ri, ci)| costs[ri * c + ci]).sum();
                let want_cost = brute_best(r, c, &costs);
                assert!((got_cost - want_cost).abs() < 1e-9, "{r}x{c}: got {got_cost} vs {want_cost}");
            }
        }
    }

    #[test]
    fn sparse_pruning_matches_large_coasting_pool_in_sub_millisecond_time() {
        // 400 isolated buffered tracks + 2 matching tracks against 100 detections (only 2 viable pairs).
        let mut tracks: Vec<Track> = (0..400u64)
            .map(|i| {
                let x = (i as f32) * 100.0;
                Track {
                    id: i + 1,
                    kalman: KalmanBox::from_bbox(Xyxy::new(x, 0.0, x + 20.0, 20.0)),
                    class_id: 0,
                    score: 0.9,
                    age: 5,
                    hits: 3,
                    time_since_update: 2,
                    state: TrackState::Lost,
                }
            })
            .collect();
        tracks[17].kalman = KalmanBox::from_bbox(Xyxy::new(10.0, 500.0, 60.0, 550.0));
        tracks[250].kalman = KalmanBox::from_bbox(Xyxy::new(200.0, 500.0, 250.0, 550.0));

        let mut dets: Vec<Detection> = (0..100)
            .map(|i| {
                let x = (i as f32) * 100.0;
                Detection { bbox: Xyxy::new(x, 1000.0, x + 20.0, 1020.0), score: 0.8, class: 0 }
            })
            .collect();
        dets[5] = Detection { bbox: Xyxy::new(202.0, 502.0, 252.0, 552.0), score: 0.85, class: 0 };
        dets[77] = Detection { bbox: Xyxy::new(12.0, 502.0, 62.0, 552.0), score: 0.88, class: 0 };

        let t_refs: Vec<&Track> = tracks.iter().collect();
        let d_refs: Vec<&Detection> = dets.iter().collect();
        let (matched, unm_t, unm_d) = associate(&t_refs, &d_refs, 0.5, true);
        assert_eq!(matched, vec![(17, 77), (250, 5)]);
        assert_eq!(unm_t.len(), 398);
        assert_eq!(unm_d.len(), 98);
    }
}
