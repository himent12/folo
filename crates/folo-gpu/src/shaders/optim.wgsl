// In-place optimizer updates, one thread per element, in the exact op order of folo-nn's CPU loops.
// Spec: MODE (0 SGD, 1 SGD with momentum, 2 AdamW, 3 p = a·p + b·g, 4/5 decoupled SGD no-mom/mom,
// 6/7 cautious decoupled SGD no-mom/mom), FLAG (SGD: weight decay, AdamW: decoupled decay), NESTEROV.
// Params: 0 = n, then
//   SGD (0,1):     1 = lr, 2 = momentum, 3 = coupled weight decay
//   SGD (4..7):    1 = lr, 2 = momentum, 3 = decoupled wd_step (unscaled_lr * wd)
//   AdamW (2):     1 = decay, 2 = β1, 3 = 1 − β1, 4 = β2, 5 = 1 − β2, 6 = step size, 7 = sqrt(1 − β2^t), 8 = ε
//   axpby (3):     1 = a, 2 = b
@group(0) @binding(1) var<storage, read_write> PARAM: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> S0: array<f32>;
@group(0) @binding(4) var<storage, read_write> S1: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let g = G[i];
    var p = PARAM[i];
    switch MODE {
        case 0u, 1u: {
            var d = g;
            if (FLAG) { d = g + pf(3u) * p; }
            if (MODE == 1u) {
                let mu = pf(2u);
                let b = mu * S0[i] + d;
                S0[i] = b;
                if (NESTEROV) { d = d + mu * b; } else { d = b; }
            }
            p = p + (-pf(1u)) * d;
        }
        case 2u: {
            if (FLAG) { p = p * pf(1u); }
            let m = pf(2u) * S0[i] + pf(3u) * g;
            let v = pf(4u) * S1[i] + pf(5u) * (g * g);
            S0[i] = m;
            S1[i] = v;
            let denom = sqrt(v) / pf(7u) + pf(8u);
            p = p + (-pf(6u)) * (m / denom);
        }
        case 4u, 5u, 6u, 7u: {
            var u = g;
            if (MODE == 5u || MODE == 7u) {
                let mu = pf(2u);
                let b = mu * S0[i] + g;
                S0[i] = b;
                if (NESTEROV) { u = g + mu * b; } else { u = b; }
            }
            var decay = 0.0;
            if (FLAG) {
                if (MODE >= 6u) {
                    if (u * p > 0.0) { decay = pf(3u) * p; }
                } else {
                    decay = pf(3u) * p;
                }
            }
            p = p - pf(1u) * u - decay;
        }
        default: {
            p = p * pf(1u) + g * pf(2u);
        }
    }
    PARAM[i] = p;
}
