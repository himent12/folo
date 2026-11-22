// Detection loss, gradient of the classification term: d/dx = g · S_cls · (sigmoid(x) − T), with g the upstream
// gradient of the total (GRAD[0]) and S_cls = B · cls_gain / tss from the combine kernel (STATS[9]).
// Params: 0 = A, 1 = NC, 2 = B·NC·A, 3 = B·A, 4 = mal_gamma (0 = plain BCE).
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
    var g_x = p - t;
    if (gamma > 0.0 && t == 0.0) {
        let q = sigmoid_f(-x);
        let sp = softplus_f(x);
        g_x = pow_pos(p, gamma) * (gamma * q * sp + p);
    }
    DX[i] = GRAD[0] * STATS[9] * g_x;
}
