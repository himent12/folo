// Workgroup-parallel top-k per row (one 256-thread workgroup per row):
// - When `BITONIC` is true (D <= 512), loads the row once into workgroup memory and runs an in-place
//   Batcher bitonic sort (1 global read of X, coalesced write of top k).
// - When `BITONIC` is false (D > 512), runs k workgroup-parallel selection passes with coalesced stride-256
//   scans and either a 2-barrier 32-wide subgroup reduction or a portable 8-barrier tree reduction.
// Exact `f32::total_cmp` order (descending when LARGEST, then index ascending; -0 and +0 tie by index).
// Spec: LARGEST, BITONIC (optional). Params: 0 = rows, 1 = row length D, 2 = k.
// Indices are written as i64 (low word, then 0). Values are read and written as raw u32 bits so NaN payloads
// and signed zeros are preserved bit-for-bit.
@group(0) @binding(1) var<storage, read> X: array<u32>;
@group(0) @binding(2) var<storage, read_write> V: array<u32>;
@group(0) @binding(3) var<storage, read_write> I: array<u32>;

const TOPK_EMPTY: u32 = 0xffffffffu;

// Monotone key: the unsigned order of keys is the f32::total_cmp order of values (given as bits), except that -0
// has +0's key: they are equal, so they tie by index, as on the CPU.
fn key(bits: u32) -> u32 {
    let b = select(bits, 0u, bits == 0x80000000u);
    return select(b ^ 0x80000000u, ~b, (b >> 31u) == 1u);
}

// True if real element (ka, ia) comes before real element (kb, ib) in the output order.
fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    if (ka != kb) {
        if (LARGEST) { return ka > kb; }
        return ka < kb;
    }
    return ia < ib;
}

// True if slot (ka, ia) ranks strictly ahead of slot (kb, ib), treating TOPK_EMPTY as trailing padding.
fn better_slot(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    if (ia == TOPK_EMPTY) { return false; }
    if (ib == TOPK_EMPTY) { return true; }
    return before(ka, ia, kb, ib);
}
