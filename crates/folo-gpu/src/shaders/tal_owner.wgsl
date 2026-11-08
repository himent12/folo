// Task-aligned assigner, step 4 (docs/MODEL.md §7.3): an anchor picked by several gts goes to the one with the
// largest u, ties to the smaller gt index. Two passes over the picks, one thread each, on a zeroed OWN = [2, B, A]:
// MODE 0 takes the largest u per anchor (u >= +0, so its bits order like the value), MODE 1 the smallest gt index
// among the picks with that u (stored complemented so a plain atomicMax over a zeroed buffer finds it).
// Spec: MODE, K. Params: 0 = B·G·K, 1 = A, 2 = G, 3 = B·A.
@group(0) @binding(1) var<storage, read> PICK: array<u32>;
@group(0) @binding(2) var<storage, read_write> OWN: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let s = linear_id(wid, nwg, lid, 256u);
    if (s >= pu(0u)) { return; }
    let a = PICK[3u * s];
    if (a == DET_NONE) { return; }
    let gi = s / K;
    let ng = pu(2u);
    let b = gi / ng;
    let g = gi - b * ng;
    let cell = b * pu(1u) + a;
    let u = PICK[3u * s + 2u];
    if (MODE == 0u) {
        atomicMax(&OWN[cell], u);
    } else if (atomicLoad(&OWN[cell]) == u) {
        atomicMax(&OWN[pu(3u) + cell], ~g);
    }
}
