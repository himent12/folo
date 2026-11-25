// Reduce the contiguous last axis: out[row] = reduce(x[row * R .. row * R + R]). One workgroup per row.
// When `SPLIT_K` is true (pass 1 of `reduce_split`), `rows = outer * splits`, and workgroup `row` reduces
// chunk `row % splits` (up to `pu(5u)` elements, clamped at `R`) of input row `row / splits`.
// `reduce.rs` appends either the 1-barrier 32-wide subgroup reduction or the portable 8-barrier fallback.
// Spec: OP (0 sum, 1 max, 2 min), SPLIT (optional). Params: 0 = rows, 1 = R, 2 = scale applied to the result
// (1/R for mean), 3 = identity, [4 = splits, 5 = chunk when SPLIT]. XT / YT are the storage types of X and Y
// (f32 or f16); the arithmetic is f32.
@group(0) @binding(1) var<storage, read> X: array<XT>;
@group(0) @binding(2) var<storage, read_write> Y: array<YT>;

// The reduction identity (0, -inf or +inf) comes from the host: WGSL constants cannot be infinite.
fn identity() -> f32 { return pf(3u); }

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

// Max / min propagate NaN explicitly (WGSL's builtins may drop it), as folo-core's CPU reductions do.
fn combine(a: f32, b: f32) -> f32 {
    switch OP {
        case 1u: { if (isnan1(a)) { return a; } if (isnan1(b)) { return b; } return max(a, b); }
        case 2u: { if (isnan1(a)) { return a; } if (isnan1(b)) { return b; } return min(a, b); }
        default: { return a + b; }
    }
}

fn reduce_rows_thread(row: u32, rows: u32, r: u32, lid: u32) -> f32 {
    var acc = identity();
    if (row < rows) {
        if (SPLIT_K) {
            let splits = pu(4u);
            let chunk = pu(5u);
            let o = row / splits;
            let k = row % splits;
            let start = k * chunk;
            let len = min(chunk, r - start);
            let base = o * r + start;
            for (var j = lid; j < len; j += 256u) {
                acc = combine(acc, f32(X[base + j]));
            }
        } else {
            let base = row * r;
            for (var i = lid; i < r; i += 256u) {
                acc = combine(acc, f32(X[base + i]));
            }
        }
    }
    return acc;
}
