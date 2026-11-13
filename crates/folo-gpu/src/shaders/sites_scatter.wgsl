// Writes a conv's values at listed sites back into a full map, one thread per (image, channel, slot):
// DST[b][c][SITES[b][1 + s]] = SRC[b][c][s] for the slots s below the image's count (and CAP). Other elements of
// DST are left as they are.
// Params: 0 = B * C * CAP, 1 = CAP, 2 = C, 3 = HW.
@group(0) @binding(1) var<storage, read> SRC: array<T>;
@group(0) @binding(2) var<storage, read> SITES: array<u32>;
@group(0) @binding(3) var<storage, read_write> DST: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let cap = pu(1u);
    let c = pu(2u);
    let s = i % cap;
    let bc = i / cap;
    let b = bc / c;
    let row = b * (cap + 1u);
    if (s >= min(SITES[row], cap)) { return; }
    DST[bc * pu(3u) + SITES[row + 1u + s]] = SRC[i];
}
