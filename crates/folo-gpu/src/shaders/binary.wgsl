// out = a (op) b with NumPy broadcasting. Spec: OP (binary op code), RANK (dims after coalescing, 0 = linear
// fast path), VEC4 (vec4 output units, with optional scalar-broadcast A/B via BCAST bits 0/1).
// Params: 0 = out elements (E units), 4..12 = out dims, 12..20 = a strides, 20..28 = b strides
// (innermost dimension last; broadcast dims have stride 0), 28 = NaN, 29 = +inf (pow edge cases).
@group(0) @binding(1) var<storage, read> A: array<TA>;
@group(0) @binding(2) var<storage, read> B: array<TB>;
@group(0) @binding(3) var<storage, read_write> O: array<E>;

// C `powf` for one lane. NaN, infinities and signed zeros are classified by bit pattern (WGSL builtins
// are allowed to assume finite inputs), and every special case of C99 Annex F is handled explicitly.
fn pow1(x: f32, y: f32) -> f32 {
    let bx = bitcast<u32>(x);
    let ax = bx & 0x7fffffffu;
    let ay = bitcast<u32>(y) & 0x7fffffffu;
    let nan = pf(28u);
    let inf = pf(29u);
    // x^±0 = 1 and 1^y = 1, even for NaN.
    if (ay == 0u || bx == 0x3f800000u) { return 1.0; }
    if (ax > 0x7f800000u || ay > 0x7f800000u) { return nan; }
    if (ay == 0x7f800000u) {
        // y = ±inf: (-1)^±inf = 1; |x| > 1 grows towards +inf, |x| < 1 shrinks towards +inf for y = -inf.
        if (ax == 0x3f800000u) { return 1.0; }
        return select(0.0, inf, (ax > 0x3f800000u) == (y > 0.0));
    }
    let yi = round(y);
    let is_int = yi == y;
    let odd = is_int && abs(yi - 2.0 * floor(yi * 0.5)) == 1.0;
    let negate = (bx >> 31u) == 1u && odd;
    if (ax == 0u || ax == 0x7f800000u) {
        // x = ±0 or ±inf: the magnitude is inf exactly when (x is 0) == (y < 0); odd integer y keeps x's sign.
        let r = select(0.0, inf, (ax == 0u) == (y < 0.0));
        return select(r, -r, negate);
    }
    if ((bx >> 31u) == 1u && !is_int) { return nan; }
    let r = exp2(y * log2(bitcast<f32>(ax)));
    return select(r, -r, negate);
}

fn pow_np(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(pow1(x.x, y.x), pow1(x.y, y.y), pow1(x.z, y.z), pow1(x.w, y.w));
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
        O[i] = OUT(binop(ld_a(select(i, 0u, BCAST_A)), ld_b(select(i, 0u, BCAST_B))));
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
    O[i] = OUT(binop(ld_a(ia), ld_b(ib)));
}
