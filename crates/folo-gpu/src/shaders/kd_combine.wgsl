// Distillation, final reduction, one workgroup: Σ of the class partials and of the DFL partials, each in a fixed
// order, then the reported terms cls_gain · L_kc = Σ_cls · pf(2) and dfl_gain · L_kd = Σ_dfl · pf(3) and the total
// B · (both). A non-finite input (MISC[0]) makes every result NaN.
// STATS: 0 cls_gain · L_kc, 1 dfl_gain · L_kd, 2 non-finite flag, 3 0. TOTAL[0] = total (the loss tensor's storage).
// Params: 0 = class partials (PC), 1 = DFL partials (PD), 2 = cls_gain / N,
// 3 = dfl_gain · T² / (4 N), 4 = B (f32).
@group(0) @binding(1) var<storage, read> PC: array<f32>;
@group(0) @binding(2) var<storage, read> PD: array<f32>;
@group(0) @binding(3) var<storage, read> MISC: array<u32>;
@group(0) @binding(4) var<storage, read_write> STATS: array<f32>;
@group(0) @binding(5) var<storage, read_write> TOTAL: array<f32>;

var<workgroup> S: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    var acc = vec2<f32>(0.0);
    for (var i = lid; i < pu(0u); i += 256u) { acc.x = acc.x + PC[i]; }
    for (var i = lid; i < pu(1u); i += 256u) { acc.y = acc.y + PD[i]; }
    S[lid] = acc;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step = step >> 1u) {
        if (lid < step) { S[lid] = S[lid] + S[lid + step]; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        var c = S[0].x * pf(2u);
        var d = S[0].y * pf(3u);
        var total = (c + d) * pf(4u);
        let bad = MISC[0] != 0u;
        if (bad) {
            let nan = bitcast<f32>(0x7fc00000u);
            c = nan;
            d = nan;
            total = nan;
        }
        STATS[0] = c;
        STATS[1] = d;
        STATS[2] = select(0.0, 1.0, bad);
        STATS[3] = 0.0;
        TOTAL[0] = total;
    }
}
