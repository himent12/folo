// Batched 2-D transpose through a padded 32x32 workgroup tile: x [B, R, C] -> y [B, C, R].
// Params: 0 = R, 1 = C. Dispatch: x = ceil(C / 32), y = ceil(R / 32), z = B.
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;

var<workgroup> tile: array<T, 1056>; // 32 x 33

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = pu(0u);
    let c = pu(1u);
    let base = wid.z * r * c;
    let c0 = wid.x * 32u;
    let r0 = wid.y * 32u;
    for (var j = lid.y; j < 32u; j += 8u) {
        let row = r0 + j;
        let col = c0 + lid.x;
        if (row < r && col < c) {
            tile[j * 33u + lid.x] = X[base + row * c + col];
        }
    }
    workgroupBarrier();
    for (var j = lid.y; j < 32u; j += 8u) {
        let col = c0 + j;
        let row = r0 + lid.x;
        if (row < r && col < c) {
            Y[base + col * r + row] = tile[lid.x * 33u + j];
        }
    }
}
