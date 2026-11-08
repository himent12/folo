// Direct depthwise conv2d (groups == C_in, C_out = mult * C_in) with fused bias + activation, NCHW.
// Spec: KH, KW, SH, SW, PH, PW, DH, DW, MULT, HAS_BIAS, ACT. Params: 0 = output elements, 1 = H, 2 = W,
// 3 = OH, 4 = OW, 5 = C_out, 6 = act alpha.
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read> Wt: array<T>;
@group(0) @binding(3) var<storage, read> BIAS: array<T>;
@group(0) @binding(4) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = i32(pu(1u)); let w = i32(pu(2u)); let oh = pu(3u); let ow = pu(4u); let cout = pu(5u);
    let ox = i % ow;
    let t = i / ow;
    let oy = t % oh;
    let nc = t / oh;
    let co = nc % cout;
    let n = nc / cout;
    let cin = cout / MULT;
    let base = (n * cin + co / MULT) * u32(h) * u32(w);
    let wbase = co * KH * KW;
    let iy0 = i32(oy * SH) - PH;
    let ix0 = i32(ox * SW) - PW;
    var acc = 0.0;
    for (var ky = 0u; ky < KH; ky++) {
        let iy = iy0 + i32(ky * DH);
        if (iy < 0 || iy >= h) { continue; }
        for (var kx = 0u; kx < KW; kx++) {
            let ix = ix0 + i32(kx * DW);
            if (ix < 0 || ix >= w) { continue; }
            acc = fma(f32(X[base + u32(iy * w + ix)]), f32(Wt[wbase + ky * KW + kx]), acc);
        }
    }
    if (HAS_BIAS) { acc += f32(BIAS[co]); }
    Y[i] = T(unary1(ACT, acc, pf(6u)));
}
