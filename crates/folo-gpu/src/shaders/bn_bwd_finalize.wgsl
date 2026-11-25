// Pass 2 of the BatchNorm backward: per channel, db = Σdz and dw = Σdz·x̂, and the dx coefficients
// K = (mean, invstd, w, b, db / n, dw / n) for dx = w · invstd · (dz − db / n − x̂ · dw / n).
// `norm.rs` appends either the 1-barrier 32-wide subgroupAdd reduction or the portable 8-barrier fallback.
// Grid: (C). Params: 0 = C, 1 = chunks, 2 = 1 / (N·hw).
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> MEAN: array<f32>;
@group(0) @binding(3) var<storage, read> INVSTD: array<f32>;
@group(0) @binding(4) var<storage, read> W: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> DW: array<f32>;
@group(0) @binding(7) var<storage, read_write> DB: array<f32>;
@group(0) @binding(8) var<storage, read_write> K: array<f32>;

fn bn_bwd_finalize_thread(c: u32, lid: u32) -> vec2<f32> {
    let chunks = pu(1u);
    var a1 = 0.0;
    var a2 = 0.0;
    for (var k = lid; k < chunks; k += 256u) {
        a1 += PART[(c * chunks + k) * 2u];
        a2 += PART[(c * chunks + k) * 2u + 1u];
    }
    return vec2<f32>(a1, a2);
}

fn bn_bwd_finalize_write(c: u32, db: f32, dw: f32) {
    DB[c] = db;
    DW[c] = dw;
    K[6u * c] = MEAN[c];
    K[6u * c + 1u] = INVSTD[c];
    K[6u * c + 2u] = W[c];
    K[6u * c + 3u] = B[c];
    K[6u * c + 4u] = db * pf(2u);
    K[6u * c + 5u] = dw * pf(2u);
}
