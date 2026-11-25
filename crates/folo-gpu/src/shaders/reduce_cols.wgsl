// Reduce the middle axis of [outer, R, inner] with inner > 1: out[o, i] = reduce_r x[o, r, i].
// When `SPLIT_K` is true (pass 1 of `reduce_split`), `pu(0u) = outer * splits` and each workgroup reduces
// chunk `k = os % splits` (up to `pu(7u)` elements, clamped at `R`) of outer slice `o = os / splits`, writing
// `[outer, splits, inner]`.
// Workgroup = 64 columns x 4 lanes over R. Spec: OP, SPLIT (optional). Params: 0 = outer (or outer * splits),
// 1 = R, 2 = scale, 3 = inner, 4 = column chunks per outer row (ceil(inner / 64)), 5 = identity,
// [6 = splits, 7 = chunk when SPLIT]. XT / YT are the storage types of X and Y (f32 or f16); arithmetic is f32.
@group(0) @binding(1) var<storage, read> X: array<XT>;
@group(0) @binding(2) var<storage, read_write> Y: array<YT>;

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
    let outer_s = pu(0u);
    let r = pu(1u);
    let inner = pu(3u);
    let chunks = pu(4u);
    let os = g / chunks;
    let col = (g % chunks) * 64u + lid.x;
    var acc = identity();
    if (os < outer_s && col < inner) {
        if (SPLIT_K) {
            let splits = pu(6u);
            let chunk = pu(7u);
            let o = os / splits;
            let k = os % splits;
            let start = k * chunk;
            let len = min(chunk, r - start);
            let base = (o * r + start) * inner + col;
            for (var j = lid.y; j < len; j += 4u) {
                acc = combine(acc, f32(X[base + j * inner]));
            }
        } else {
            let base = os * r * inner + col;
            for (var i = lid.y; i < r; i += 4u) {
                acc = combine(acc, f32(X[base + i * inner]));
            }
        }
    }
    let slot = lid.y * 64u + lid.x;
    partial[slot] = acc;
    workgroupBarrier();
    if (lid.y == 0u && os < outer_s && col < inner) {
        let v = combine(combine(partial[lid.x], partial[64u + lid.x]),
                        combine(partial[128u + lid.x], partial[192u + lid.x]));
        Y[os * inner + col] = YT(v * pf(2u));
    }
}
