// Reduce the middle axis of [outer, R, inner] with inner > 1: out[o, i] = reduce_r x[o, r, i].
// Workgroup = 64 columns x 4 lanes over R. Spec: OP. Params: 0 = outer, 1 = R, 2 = scale, 3 = inner,
// 4 = column chunks per outer row (ceil(inner / 64)), 5 = identity.
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

var<workgroup> partial: array<f32, 256>;

// The reduction identity (0, -inf or +inf) comes from the host: WGSL constants cannot be infinite.
fn identity() -> f32 { return pf(5u); }

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

// Max / min propagate NaN explicitly (WGSL's builtins may drop it), as folo-core's CPU reductions do.
fn combine(a: f32, b: f32) -> f32 {
    switch OP {
        case 1u: { if (isnan1(a)) { return a; } if (isnan1(b)) { return b; } return max(a, b); }
        case 2u: { if (isnan1(a)) { return a; } if (isnan1(b)) { return b; } return min(a, b); }
        default: { return a + b; }
    }
}

@compute @workgroup_size(64, 4)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.y * nwg.x + wid.x;
    let outer = pu(0u);
    let r = pu(1u);
    let inner = pu(3u);
    let chunks = pu(4u);
    let o = g / chunks;
    let col = (g % chunks) * 64u + lid.x;
    var acc = identity();
    if (o < outer && col < inner) {
        let base = o * r * inner + col;
        for (var i = lid.y; i < r; i += 4u) {
            acc = combine(acc, X[base + i * inner]);
        }
    }
    let slot = lid.y * 64u + lid.x;
    partial[slot] = acc;
    workgroupBarrier();
    if (lid.y == 0u && o < outer && col < inner) {
        let v = combine(combine(partial[lid.x], partial[64u + lid.x]),
                        combine(partial[128u + lid.x], partial[192u + lid.x]));
        Y[o * inner + col] = v * pf(2u);
    }
}
