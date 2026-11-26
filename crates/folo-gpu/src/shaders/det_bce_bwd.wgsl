// Detection loss, gradient of the classification term: d/dx = g · S_cls · dL_cls/dx, with g the upstream
// gradient of the total (GRAD[0]) and S_cls = B · cls_gain / tss from the combine kernel (STATS[9]).
// Supports BCE/MAL (`cls_kind = 0`), VFL (`1`), and QFL (`2`).
// Params: 0 = A, 1 = NC, 2 = B·NC·A, 3 = B·A, 4 = mal_gamma (0 = plain BCE),
// 5 = cls_loss_kind, 6 = cls_param0 (vfl_alpha / qfl_beta), 7 = cls_param1 (vfl_gamma).
@group(0) @binding(1) var<storage, read> CLS: array<f32>;
@group(0) @binding(2) var<storage, read> DENSE: array<u32>;
@group(0) @binding(3) var<storage, read> STATS: array<f32>;
@group(0) @binding(4) var<storage, read> GRAD: array<f32>;
@group(0) @binding(5) var<storage, read_write> DX: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(2u)) { return; }
    let na = pu(0u);
    let row = i / na;
    let a = i - row * na;
    let b = row / pu(1u);
    let c = row - b * pu(1u);
    let cell = b * na + a;
    let t = select(0.0, bitcast<f32>(DENSE[pu(3u) + cell]), DENSE[cell] == c + 1u);
    let x = CLS[i];
    let p = sigmoid_f(x);
    let gamma = pf(4u);
    let cls_kind = pu(5u);
    let p0 = pf(6u);
    let p1 = pf(7u);
    var g_x = 0.0;
    if (cls_kind == 0u) {
        g_x = p - t;
        if (gamma > 0.0 && t == 0.0) {
            let q = sigmoid_f(-x);
            let sp = softplus_f(x);
            g_x = select(0.0, pow_pos(p, gamma) * (gamma * q * sp + p), p > 0.0);
        }
    } else if (cls_kind == 1u) {
        if (t == 0.0) {
            if (p1 == 0.0) {
                g_x = p0 * p;
            } else if (p > 0.0) {
                let q = sigmoid_f(-x);
                let sp = softplus_f(x);
                g_x = p0 * pow_pos(p, p1) * (p1 * q * sp + p);
            }
        } else {
            g_x = t * (p - t);
        }
    } else {
        if (t == 0.0) {
            if (p0 == 0.0) {
                g_x = p;
            } else if (p > 0.0) {
                let q = sigmoid_f(-x);
                let sp = softplus_f(x);
                g_x = pow_pos(p, p0) * (p0 * q * sp + p);
            }
        } else if (p0 == 0.0) {
            g_x = p - t;
        } else {
            let diff = p - t;
            let d = abs(diff);
            let bce = softplus_f(x) - x * t;
            if (d == 0.0 && is_finite_f(bce)) {
                g_x = 0.0;
            } else {
                let q = sigmoid_f(-x);
                let pow_b = select(select(pow_pos(d, p0), d, p0 == 1.0), d * d, p0 == 2.0);
                let d_pow = select(
                    select(p0 * pow_pos(d, p0 - 1.0) * sign(diff), sign(diff), p0 == 1.0),
                    2.0 * diff,
                    p0 == 2.0,
                );
                g_x = pow_b * diff + d_pow * p * q * bce;
            }
        }
    }
    DX[i] = GRAD[0] * STATS[9] * g_x;
}
