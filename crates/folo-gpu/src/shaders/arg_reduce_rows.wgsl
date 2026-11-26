// Workgroup-parallel argmax (`MAX = true`) / argmin (`MAX = false`) along the contiguous last axis (`inner == 1`).
// Spec: MAX (bool), MODE (0 = single-pass, 1 = split-K pass 1 to PART, 2 = split-K pass 2 merge from PART),
// VEC4 (bool, MODE 0/1 when R % 4 == 0).
// Params:
//   MODE 0: 0 = rows, 1 = r_units (R / 4 if VEC4 else R)
//   MODE 1: 0 = rows * splits, 1 = r_units, 2 = splits, 3 = chunk_units
//   MODE 2: 0 = rows, 1 = splits
// Exact CPU semantics: NaN beats every finite/inf value (first/smallest index NaN wins); ties broken by smallest index.
const ARG_EMPTY: u32 = 0xffffffffu;

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

fn arg_better(va: f32, ia: u32, vb: f32, ib: u32) -> bool {
    if (ia == ARG_EMPTY) { return false; }
    if (ib == ARG_EMPTY) { return true; }
    let na = isnan1(va);
    let nb = isnan1(vb);
    if (na && nb) { return ia < ib; }
    if (na) { return true; }
    if (nb) { return false; }
    if (MAX) {
        return va > vb || (va == vb && ia < ib);
    }
    return va < vb || (va == vb && ia < ib);
}

fn arg_write(row: u32, bv: f32, bi: u32) {
    if (MODE == 1u) {
        Y[2u * row] = bitcast<u32>(bv);
        Y[2u * row + 1u] = bi;
    } else {
        Y[2u * row] = bi;
        Y[2u * row + 1u] = 0u;
    }
}
