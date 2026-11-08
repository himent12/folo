// Elementwise unary functions on vec4<f32>, selected by an op code (folded when the code is a constant).
// Codes match `UnaryOp::code` in kernels/elementwise.rs.
fn tanh_safe(x: vec4<f32>) -> vec4<f32> {
    // Saturate exactly for |x| >= 9 (tanh(9) rounds to 1 in f32): some backends overflow for |x| > ~44 and
    // others (HLSL) return 1 - 1e-8 at 15, which leaks into GELU of large negative inputs.
    return select(tanh(clamp(x, vec4<f32>(-9.0), vec4<f32>(9.0))), sign(x), abs(x) >= vec4<f32>(9.0));
}

fn unary4(op: u32, x: vec4<f32>, a: f32) -> vec4<f32> {
    let zero = vec4<f32>(0.0);
    let one = vec4<f32>(1.0);
    switch op {
        case 0u: { return x; }
        case 1u: { return exp(x); }
        case 2u: { return log(x); }
        case 3u: { return sqrt(x); }
        case 4u: { return inverseSqrt(x); }
        case 5u: { return -x; }
        case 6u: { return abs(x); }
        case 7u: { return one / (one + exp(-x)); }
        case 8u: { return x / (one + exp(-x)); }
        case 9u: { return max(x, zero); }
        case 10u: { return tanh_safe(x); }
        case 11u: {
            let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            return 0.5 * x * (one + tanh_safe(inner));
        }
        case 12u: { return select(x * a, x, x >= zero); }
        case 13u: { return x * clamp(x + 3.0, zero, vec4<f32>(6.0)) / 6.0; }
        case 14u: { return clamp(x / 6.0 + 0.5, zero, one); }
        case 15u: { return x * x; }
        case 16u: { return one / x; }
        case 17u: { return sin(x); }
        case 18u: { return cos(x); }
        case 19u: { return floor(x); }
        case 20u: { return ceil(x); }
        case 21u: { return clamp(x, zero, vec4<f32>(6.0)); }
        default: { return x; }
    }
}

fn unary1(op: u32, x: f32, a: f32) -> f32 {
    return unary4(op, vec4<f32>(x), a).x;
}
