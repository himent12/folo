// Nearest-neighbour upsampling of NCHW planes: y[p, oh, ow] = x[p, oh * H / OH, ow * W / OW].
// Params: 0 = output elements, 1 = H, 2 = W, 3 = OH, 4 = OW.
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u);
    let x = i % ow;
    let t = i / ow;
    let y = t % oh;
    let p = t / oh;
    Y[i] = X[(p * h + (y * h) / oh) * w + (x * w) / ow];
}
