// Welford partials (count, mean, M2) per thread, merged across the workgroup.
var<workgroup> wn: array<f32, 256>;
var<workgroup> wm: array<f32, 256>;
var<workgroup> wq: array<f32, 256>;

// Chan et al.'s pairwise merge of Welford partials (count, mean, M2) held in workgroup arrays.
fn bn_tree_merge(lid: u32) {
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            let na = wn[lid];
            let nb = wn[lid + s];
            if (na == 0.0) {
                // Copy instead of merging into nothing: `d · d · 0` overflows to NaN for huge means.
                wn[lid] = nb;
                wm[lid] = wm[lid + s];
                wq[lid] = wq[lid + s];
            } else if (nb > 0.0) {
                let n = na + nb;
                let d = wm[lid + s] - wm[lid];
                let delta = d * (nb / n);
                wm[lid] += delta;
                wq[lid] += wq[lid + s] + d * (delta * na);
                wn[lid] = n;
            }
        }
        workgroupBarrier();
    }
}
