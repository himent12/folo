// Reduce the middle axis of [outer, R, inner] with inner > 1: out[o, i] = reduce_r x[o, r, i].
// When `SPLIT_K` is true (pass 1 of `reduce_split`), `pu(0u) = outer * splits` and each workgroup reduces
// chunk `k = os % splits` (up to `pu(7u)` elements, clamped at `R`) of outer slice `o = os / splits`, writing
// `[outer, splits, inner]`.
// `reduce.rs` appends `X` / `Y` bindings and `main` (`vec4<XT>` / `vec4<YT>` when `VEC4=true`, else scalar).
// Workgroup = 64 columns x 4 lanes over R. Spec: OP, SPLIT (optional), VEC4 (optional).
// Params: 0 = outer (or outer * splits), 1 = R, 2 = scale, 3 = inner (elements when scalar, vec4 units when VEC4),
// 4 = column chunks per outer row (ceil(inner_units / 64)), 5 = identity, [6 = splits, 7 = chunk when SPLIT].
// XT / YT are the storage types of X and Y (f32 or f16); arithmetic is f32.

// The reduction identity (0, -inf or +inf) comes from the host: WGSL constants cannot be infinite.
fn identity() -> f32 { return pf(5u); }

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

fn isnan4(v: vec4<f32>) -> vec4<bool> {
    return (bitcast<vec4<u32>>(v) & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
}

// Max / min propagate NaN explicitly (WGSL's builtins may drop it), as folo-core's CPU reductions do.
fn combine(a: f32, b: f32) -> f32 {
    switch OP {
        case 1u: { if (isnan1(a)) { return a; } if (isnan1(b)) { return b; } return max(a, b); }
        case 2u: { if (isnan1(a)) { return a; } if (isnan1(b)) { return b; } return min(a, b); }
        default: { return a + b; }
    }
}

fn combine4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    switch OP {
        case 1u: { return select(select(max(a, b), b, isnan4(b)), a, isnan4(a)); }
        case 2u: { return select(select(min(a, b), b, isnan4(b)), a, isnan4(a)); }
        default: { return a + b; }
    }
}
