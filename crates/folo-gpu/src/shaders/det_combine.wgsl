// Detection loss, final reduction (docs/MODEL.md §7.4), one workgroup: Σ of the BCE partials, of the per-slot box
// and DFL terms and of the positives' t (the target score sum), each in a fixed order, then
// tss = max(Σ t, 1), the reported items (box_gain · L_box, cls_gain · L_cls, dfl_gain · L_dfl) and
// total = B · (box + cls + dfl). A non-finite head output (MISC[0]) makes every result NaN.
// STATS: 0 total, 1 box, 2 cls, 3 dfl, 4 positives, 5 Σ t, 6 non-finite flag, 7 tss, 8..10 the backward scales
// (B / tss) · gain for box, cls, dfl, 11 B. TOTAL[0] = total (the loss tensor's storage).
// Params: 0 = BCE partials, 1 = slots, 2 = B (f32), 3 = box gain, 4 = cls gain, 5 = dfl gain, 6 = dfl_ot_weight.
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> TERMS: array<f32>;
@group(0) @binding(3) var<storage, read> SLOT: array<u32>;
@group(0) @binding(4) var<storage, read> MISC: array<u32>;
@group(0) @binding(5) var<storage, read_write> STATS: array<f32>;
@group(0) @binding(6) var<storage, read_write> TOTAL: array<f32>;

var<workgroup> S: array<vec4<f32>, 256>;
var<workgroup> NF: array<f32, 256>;
var<workgroup> OT: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    let dfl_ot_w = pf(6u);
    var acc = vec4<f32>(0.0);
    var nfg = 0.0;
    var ot_acc = vec2<f32>(0.0, 0.0);
    for (var i = lid; i < pu(0u); i += 256u) { acc.x = acc.x + PART[i]; }
    for (var s = lid; s < pu(1u); s += 256u) {
        acc.y = acc.y + TERMS[4u * s];
        acc.z = acc.z + TERMS[4u * s + 1u];
        if (dfl_ot_w > 0.0) {
            ot_acc = add_split(ot_acc, vec2<f32>(TERMS[4u * s + 2u], TERMS[4u * s + 3u]));
        }
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
            if (dfl_ot_w > 0.0) { OT[lid] = add_split(OT[lid], OT[lid + step]); }
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
        if (dfl_ot_w > 0.0 && dfl_g != 0.0 && OT[0].x > 0.0) {
            let ot_norm = div_split(OT[0], norm_f32(tss));
            let ot_scaled = mul_split(mul_split(ot_norm, norm_f32(dfl_ot_w)), norm_f32(dfl_g));
            wd = wd + split_to_f32(ot_scaled);
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
        STATS[7] = tss;
        STATS[8] = (nb / tss) * pf(3u);
        STATS[9] = (nb / tss) * pf(4u);
        STATS[10] = (nb / tss) * pf(5u);
        STATS[11] = nb;
        TOTAL[0] = total;
    }
}
