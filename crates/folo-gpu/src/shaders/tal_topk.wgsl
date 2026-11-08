// Task-aligned assigner, steps 1-3 (docs/MODEL.md §7.3): one workgroup per (gt g, image b). Every anchor inside
// the gt's enlarged candidate box gets u = max(CIoU(gt, pred), 0) and m = score^alpha · u^beta; the gt keeps its
// top K candidates by (m descending, anchor ascending), the order of folo_vision::assign.
//
// Each thread scans a strided share of the anchors and keeps its own sorted top K in workgroup memory; K rounds of
// a tournament (a tree reduction over the threads' list heads) then merge them. Slots past the number of
// candidates, and every slot of a padding gt (g >= the image's gt count), are written as DET_NONE.
//
// Spec: K, WG (threads), ALPHA_SQRT (alpha = 0.5), BETA_INT (beta as a small integer, 0 = use pow).
// Params: 0 = A, 1 = NC, 2 = G, 3 = gt box word offset, 4 = label offset, 5 = count offset, 6 = eps,
// 7 = min box size, 8 = alpha, 9 = beta.
// PICK: [B, G, K] × (anchor, m bits, u bits).
@group(0) @binding(1) var<storage, read> PD: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> CLS: array<f32>;
@group(0) @binding(3) var<storage, read> META: array<u32>;
@group(0) @binding(4) var<storage, read_write> PICK: array<u32>;

var<workgroup> LM: array<f32, WG * K>;
var<workgroup> LA: array<u32, WG * K>;
var<workgroup> LU: array<f32, WG * K>;
var<workgroup> LEN: array<u32, WG>;
var<workgroup> HEAD: array<u32, WG>;
var<workgroup> BEST: array<u32, WG>;

fn pow_alpha(x: f32) -> f32 {
    if (ALPHA_SQRT) { return sqrt(x); }
    if (x == 0.0) { return 0.0; }
    return pow(x, pf(8u));
}

fn pow_beta(x: f32) -> f32 {
    if (BETA_INT == 0u) {
        if (x == 0.0) { return 0.0; }
        return pow(x, pf(9u));
    }
    var r = 1.0;
    var base = x;
    var e = BETA_INT;
    while (e > 0u) {
        if ((e & 1u) == 1u) { r = r * base; }
        base = base * base;
        e = e >> 1u;
    }
    return r;
}

// Whether list entry i ranks before entry j.
fn before(i: u32, j: u32) -> bool {
    if (LM[i] != LM[j]) { return LM[i] > LM[j]; }
    return LA[i] < LA[j];
}

// Thread t's current list head, or DET_NONE.
fn head_of(t: u32) -> u32 {
    if (t == DET_NONE || HEAD[t] >= LEN[t]) { return DET_NONE; }
    return t * K + HEAD[t];
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let b = wid.y;
    let na = pu(0u);
    let nc = pu(1u);
    let ng = pu(2u);
    let gi = b * ng + g;
    let real = g < META[pu(5u) + b];
    let o = pu(3u) + 4u * gi;
    let gt = vec4<f32>(bitcast<f32>(META[o]), bitcast<f32>(META[o + 1u]), bitcast<f32>(META[o + 2u]),
                       bitcast<f32>(META[o + 3u]));
    let label = META[pu(4u) + gi];
    let eps = pf(6u);
    let ms = pf(7u);
    // §7.3 step 1: every side shorter than the minimum is widened to it about the centre.
    var cb = gt;
    if (gt.z - gt.x < ms) {
        let c = (gt.x + gt.z) * 0.5;
        cb.x = c - ms * 0.5;
        cb.z = c + ms * 0.5;
    }
    if (gt.w - gt.y < ms) {
        let c = (gt.y + gt.w) * 0.5;
        cb.y = c - ms * 0.5;
        cb.w = c + ms * 0.5;
    }
    let seg = lid * K;
    var len = 0u;
    if (real) {
        let cls_row = (b * nc + label) * na;
        for (var a = lid; a < na; a += WG) {
            let s = bitcast<f32>(META[3u * a + 2u]);
            let px = bitcast<f32>(META[3u * a]) * s;
            let py = bitcast<f32>(META[3u * a + 1u]) * s;
            if (min(min(px - cb.x, py - cb.y), min(cb.z - px, cb.w - py)) > eps) {
                let c = ciou_eps(gt, PD[b * na + a]);
                let u = select(0.0, c, c > 0.0);
                let score = 1.0 / (1.0 + fexp(-CLS[cls_row + a]));
                let m = pow_alpha(score) * pow_beta(u);
                // Anchors arrive in increasing order, so an equal m ranks after every entry already listed.
                var pos = len;
                if (len < K) {
                    len = len + 1u;
                } else if (m > LM[seg + K - 1u]) {
                    pos = K - 1u;
                } else {
                    continue;
                }
                while (pos > 0u && m > LM[seg + pos - 1u]) {
                    LM[seg + pos] = LM[seg + pos - 1u];
                    LA[seg + pos] = LA[seg + pos - 1u];
                    LU[seg + pos] = LU[seg + pos - 1u];
                    pos = pos - 1u;
                }
                LM[seg + pos] = m;
                LA[seg + pos] = a;
                LU[seg + pos] = u;
            }
        }
    }
    LEN[lid] = len;
    HEAD[lid] = 0u;
    workgroupBarrier();
    let out = gi * K;
    for (var r = 0u; r < K; r++) {
        BEST[lid] = select(DET_NONE, lid, HEAD[lid] < LEN[lid]);
        workgroupBarrier();
        for (var step = WG / 2u; step > 0u; step = step >> 1u) {
            if (lid < step) {
                let x = BEST[lid];
                let y = BEST[lid + step];
                let hx = head_of(x);
                let hy = head_of(y);
                if (hx == DET_NONE || (hy != DET_NONE && before(hy, hx))) { BEST[lid] = y; }
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            let w = BEST[0];
            let h = head_of(w);
            let p = 3u * (out + r);
            if (h == DET_NONE) {
                PICK[p] = DET_NONE;
                PICK[p + 1u] = 0u;
                PICK[p + 2u] = 0u;
            } else {
                PICK[p] = LA[h];
                PICK[p + 1u] = bitcast<u32>(LM[h]);
                PICK[p + 2u] = bitcast<u32>(LU[h]);
                HEAD[w] = HEAD[w] + 1u;
            }
        }
        workgroupBarrier();
    }
}
