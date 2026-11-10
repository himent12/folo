// Direct depthwise conv2d (groups == C_in, C_out = mult * C_in) with fused bias + activation, NCHW. Each
// invocation computes DW_VX horizontally adjacent outputs of one row, so the input columns they share are loaded
// once; the generator (kernels/conv.rs) unrolls the taps into `taps(...)`, which loads the weights and adds every
// input row of the window into `acc`, and the bias, activation and bounds-checked stores into `store(...)`. Spec: KH, KW, SH, SW, PH, PW, DH, DW, MULT, HAS_BIAS, ACT. Params:
// 0 = invocations (C_out planes · OH · QW), 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_out, 6 = act alpha,
// 7 = QW (= ceil(OW / DW_VX)).
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read> Wt: array<T>;
@group(0) @binding(3) var<storage, read> BIAS: array<T>;
@group(0) @binding(4) var<storage, read_write> Y: array<T>;

// Input column ix of row `row` (the element index of its first column), zero outside the image.
fn ld(row: u32, ix: i32, w: i32) -> f32 {
    let v = f32(X[row + u32(clamp(ix, 0, w - 1))]);
    return select(0.0, v, ix >= 0 && ix < w);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = i32(pu(1u)); let w = i32(pu(2u)); let oh = pu(3u); let ow = pu(4u); let cout = pu(5u);
    let qw = pu(7u);
    let q = i % qw;
    let t = i / qw;
    let oy = t % oh;
    let nc = t / oh;
    let co = nc % cout;
    let n = nc / cout;
    let cin = cout / MULT;
    let base = (n * cin + co / MULT) * u32(h) * u32(w);
    let wbase = co * KH * KW;
    let ox0 = q * DW_VX;
    let iy0 = i32(oy * SH) - PH;
    let ix0 = i32(ox0 * SW) - PW;
    var acc = array<f32, DW_VX>();
    taps(&acc, base, wbase, iy0, ix0, h, w);
    var b = 0.0;
    if (HAS_BIAS) { b = f32(BIAS[co]); }
    store(&acc, nc * oh * ow + oy * ow + ox0, ox0, ow, b);
}
