// Detection loss, final reduction (docs/MODEL.md §7.4), one workgroup: Σ of the BCE partials, of the per-slot box
// and DFL terms and of the positives' t (the target score sum), each in a fixed order, then
// tss = max(Σ t, 1), the reported items (box_gain · L_box, cls_gain · L_cls, dfl_gain · L_dfl) and
// total = B · (box + cls + dfl). A non-finite head output (MISC[0]) makes every result NaN.
// STATS: 0 total, 1 box, 2 cls, 3 dfl, 4 positives, 5 Σ t, 6 non-finite flag, 8..10 the backward scales
// B · gain / tss for box, cls, dfl. TOTAL[0] = total (the loss tensor's storage).
// Params: 0 = BCE partials, 1 = slots, 2 = B (f32), 3 = box gain, 4 = cls gain, 5 = dfl gain, 6 = dfl_ot_weight.
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> TERMS: array<f32>;
@group(0) @binding(3) var<storage, read> SLOT: array<u32>;
@group(0) @binding(4) var<storage, read> MISC: array<u32>;
@group(0) @binding(5) var<storage, read_write> STATS: array<f32>;
@group(0) @binding(6) var<storage, read_write> TOTAL: array<f32>;

var<workgroup> S: array<vec4<f32>, 256>;
var<workgroup> NF: array<f32, 256>;
var<workgroup> OT: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    let dfl_ot_w = pf(6u);
    var acc = vec4<f32>(0.0);
    var nfg = 0.0;
    var ot_acc = 0.0;
    for (var i = lid; i < pu(0u); i += 256u) { acc.x = acc.x + PART[i]; }
    for (var s = lid; s < pu(1u); s += 256u) {
        acc.y = acc.y + TERMS[3u * s];
        acc.z = acc.z + TERMS[3u * s + 1u];
        if (dfl_ot_w > 0.0) { ot_acc = ot_acc + TERMS[3u * s + 2u]; }
        if (SLOT[2u * s + 1u] == 1u) {
            acc.w = acc.w + bitcast<f32>(SLOT[2u * s]);
            nfg = nfg + 1.0;
        }
    }
    S[lid] = acc;
    NF[lid] = nfg;
    OT[lid] = ot_acc;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step = step >> 1u) {
        if (lid < step) {
            S[lid] = S[lid] + S[lid + step];
            NF[lid] = NF[lid] + NF[lid + step];
            if (dfl_ot_w > 0.0) { OT[lid] = OT[lid] + OT[lid + step]; }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let r = S[0];
        let tss = max(r.w, 1.0);
        let nb = pf(2u);
        var wb = r.y / tss * pf(3u);
        var wc = r.x / tss * pf(4u);
        let dfl_g = pf(5u);
        var wd = -r.z / tss * dfl_g;
        if (dfl_ot_w > 0.0) {
            let ot_norm = OT[0] / tss;
            if (abs(dfl_g) <= 1.0) {
                wd = wd + (ot_norm * dfl_g) * dfl_ot_w;
            } else {
                wd = wd + (ot_norm * dfl_ot_w) * dfl_g;
            }
        }
        var total = (wb + wc + wd) * nb;
        let bad = MISC[0] != 0u;
        if (bad) {
            let nan = bitcast<f32>(0x7fc00000u);
            wb = nan;
            wc = nan;
            wd = nan;
            total = nan;
        }
        STATS[0] = total;
        STATS[1] = wb;
        STATS[2] = wc;
        STATS[3] = wd;
        STATS[4] = NF[0];
        STATS[5] = r.w;
        STATS[6] = select(0.0, 1.0, bad);
        STATS[7] = 0.0;
        STATS[8] = nb * pf(3u) / tss;
        STATS[9] = nb * pf(4u) / tss;
        STATS[10] = nb * pf(5u) / tss;
        STATS[11] = 0.0;
        TOTAL[0] = total;
    }
}
