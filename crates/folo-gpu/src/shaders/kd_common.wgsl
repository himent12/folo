// Shared by the distillation kernels (kernels/distill.rs), after detect_common.wgsl. `TW` (u32 or f32) and `TF16`
// are declared by the generator: the teacher's outputs are f16 (two per u32 word, the f16 teacher of a
// mixed-precision run) or f32. `tv(T[ti(i)], i)` is element `i` of such an array as f32 (exact either way).

fn ti(i: u32) -> u32 {
    return select(i, i >> 1u, TF16);
}

fn tv(w: TW, i: u32) -> f32 {
    if (TF16) { return f16_bits_to_f32((bitcast<u32>(w) >> (16u * (i & 1u))) & 0xffffu); }
    return bitcast<f32>(w);
}

// An f16 value from its bits, exactly, infinities and NaNs included (decoded by hand: `unpack2x16float` does not
// carry them through on every backend, e.g. DX12).
fn f16_bits_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1fu;
    let m = h & 0x03ffu;
    if (e == 31u) { return bitcast<f32>(sign | 0x7f800000u | (m << 13u)); }
    if (e == 0u) { return select(1.0, -1.0, sign != 0u) * f32(m) * 5.9604645e-8; }
    return bitcast<f32>(sign | ((e + 112u) << 23u) | (m << 13u));
}

fn is_nan_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn is_pos_inf_f(x: f32) -> bool {
    return bitcast<u32>(x) == 0x7f800000u;
}
