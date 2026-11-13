// The class pass of one detection-head level, one thread per (image, pixel): the pixel's best class is the first
// maximum of its NC sigmoid scores (as detect.wgsl); when that score is > conf, the pixel is a candidate and
// takes the next slot of its image's site list SITES (word 0 counts every candidate, the pixels follow) and the
// same slot of RECS (class, score bits); a candidate past CAP is only counted. Every pixel of a slotted
// candidate's 3×3 window (inside the map) is added once to the image's FIELD list (word 0 counts, the pixels
// follow): MARK, cleared by the host, flags the pixels already added. SITES and FIELD are cleared by the host.
// Params: 0 = B * HW, 1 = HW, 2 = NC, 3 = conf (f32), 4 = W, 5 = H, 6 = CAP, 7 = FCAP.
@group(0) @binding(1) var<storage, read> CLS: array<T>;
@group(0) @binding(2) var<storage, read_write> SITES: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> RECS: array<u32>;
@group(0) @binding(4) var<storage, read_write> FIELD: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> MARK: array<atomic<u32>>;

fn sig(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let hw = pu(1u);
    let nc = pu(2u);
    let b = i / hw;
    let j = i % hw;
    let cb = b * nc * hw + j;
    var best = 0u;
    var bs = sig(f32(CLS[cb]));
    for (var c = 1u; c < nc; c++) {
        let s = sig(f32(CLS[cb + c * hw]));
        if (s > bs) {
            best = c;
            bs = s;
        }
    }
    if (!(bs > pf(3u))) { return; }
    let cap = pu(6u);
    let slot = atomicAdd(&SITES[b * (cap + 1u)], 1u);
    if (slot >= cap) { return; }
    atomicStore(&SITES[b * (cap + 1u) + 1u + slot], j);
    RECS[(b * cap + slot) * 2u] = best;
    RECS[(b * cap + slot) * 2u + 1u] = bitcast<u32>(bs);
    let w = pu(4u);
    let h = pu(5u);
    let fcap = pu(7u);
    let y = j / w;
    let x = j % w;
    for (var yy = max(y, 1u) - 1u; yy < min(y + 2u, h); yy++) {
        for (var xx = max(x, 1u) - 1u; xx < min(x + 2u, w); xx++) {
            let p = yy * w + xx;
            if (atomicOr(&MARK[b * hw + p], 1u) == 0u) {
                let f = atomicAdd(&FIELD[b * (fcap + 1u)], 1u);
                if (f < fcap) {
                    atomicStore(&FIELD[b * (fcap + 1u) + 1u + f], p);
                }
            }
        }
    }
}
