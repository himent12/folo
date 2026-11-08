// max_pool2d on NCHW planes (padding counts as -inf, floor output size). Spec: KH, KW, SH, SW, PH, PW, DH, DW.
// Params: 0 = output elements, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = -inf.
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = i32(pu(1u)); let w = i32(pu(2u)); let oh = pu(3u); let ow = pu(4u);
    let ox = i % ow;
    let t = i / ow;
    let oy = t % oh;
    let p = t / oh;
    let base = p * u32(h) * u32(w);
    let iy0 = i32(oy * SH) - PH;
    let ix0 = i32(ox * SW) - PW;
    var m = T(pf(5u));
    for (var ky = 0u; ky < KH; ky++) {
        let iy = iy0 + i32(ky * DH);
        if (iy < 0 || iy >= h) { continue; }
        for (var kx = 0u; kx < KW; kx++) {
            let ix = ix0 + i32(kx * DW);
            if (ix < 0 || ix >= w) { continue; }
            m = max(m, X[base + u32(iy * w + ix)]);
        }
    }
    Y[i] = m;
}
