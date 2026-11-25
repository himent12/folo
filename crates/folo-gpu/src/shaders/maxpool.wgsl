// max_pool2d on NCHW planes (padding counts as -inf, floor output size; NaN wins, matching CPU and
// pool_grad.rs). Spec: KH, KW, SH, SW, PH, PW, DH, DW.
// Params: 0 = output elements, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = -inf.
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u);
    let ox = i % ow;
    let t = i / ow;
    let oy = t % oh;
    let p = t / oh;
    let base = p * h * w;
    let iy0 = i32(oy * SH) - PH;
    let ix0 = i32(ox * SW) - PW;
    var m = pf(5u);
    var has_nan = false;
    for (var ky = 0u; ky < KH; ky++) {
        let iy = iy0 + i32(ky * DH);
        if (iy < 0 || iy >= i32(h)) { continue; }
        let row = base + u32(iy) * w;
        for (var kx = 0u; kx < KW; kx++) {
            let ix = ix0 + i32(kx * DW);
            if (ix < 0 || ix >= i32(w)) { continue; }
            let v = f32(X[row + u32(ix)]);
            if (!has_nan) {
                if (isnan1(v)) {
                    m = v;
                    has_nan = true;
                } else if (v > m) {
                    m = v;
                }
            }
        }
    }
    Y[i] = T(m);
}
