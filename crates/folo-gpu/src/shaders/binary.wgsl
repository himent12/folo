// out = a (op) b with NumPy broadcasting. Spec: OP (binary op code), RANK (dims after coalescing, 0 = same
// shape fast path), VEC (vec4 elements, only for RANK == 0).
// Params: 0 = out elements (E units), 4..12 = out dims, 12..20 = a strides, 20..28 = b strides
// (innermost dimension last; broadcast dims have stride 0), 28 = NaN, 29 = +inf (pow edge cases).
@group(0) @binding(1) var<storage, read> A: array<E>;
@group(0) @binding(2) var<storage, read> B: array<E>;
@group(0) @binding(3) var<storage, read_write> O: array<E>;

fn pow_np(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {
    // pow() is undefined for x < 0 in WGSL; follow C/NumPy powf for integer exponents of negative bases.
    let ax = abs(x);
    var r = exp2(y * log2(ax));
    let yi = round(y);
    let is_int = yi == y;
    let odd = is_int & (abs(yi - 2.0 * floor(yi * 0.5)) == vec4<f32>(1.0));
    r = select(r, -r, (x < vec4<f32>(0.0)) & odd);
    r = select(r, vec4<f32>(pf(28u)), (x < vec4<f32>(0.0)) & !is_int);
    // 0^y: 0 for y > 0, 1 for y == 0, inf for y < 0; x^0 == 1 always.
    r = select(r, select(select(vec4<f32>(0.0), vec4<f32>(pf(29u)), y < vec4<f32>(0.0)),
                          vec4<f32>(1.0), y == vec4<f32>(0.0)), ax == vec4<f32>(0.0));
    r = select(r, vec4<f32>(1.0), y == vec4<f32>(0.0));
    return r;
}

fn binop(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    switch OP {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return max(a, b); }
        case 5u: { return min(a, b); }
        case 6u: { return pow_np(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    if (RANK == 0u) {
        O[i] = OUT(binop(V4(A[i]), V4(B[i])));
        return;
    }
    var rem = i;
    var ia = 0u;
    var ib = 0u;
    for (var d = 0u; d < RANK; d++) {
        let k = RANK - 1u - d;
        let dim = pu(4u + k);
        let c = rem % dim;
        rem = rem / dim;
        ia += c * pu(12u + k);
        ib += c * pu(20u + k);
    }
    O[i] = OUT(binop(V4(A[ia]), V4(B[ib])));
}
