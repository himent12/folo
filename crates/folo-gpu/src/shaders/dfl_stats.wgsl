// Fused single-pass per-anchor DFL statistics: reads BOX [B, 4 * R, A] (T = f32 or f16) once into registers
// per side and writes OUT [B, ch, A] f32 where ch = (MOMENTS ? 8 : 0) + (ENT ? 1 : 0):
//   - MOMENTS: channels 0..4 = mean (l, t, r, b), channels 4..8 = second moment E[k^2] (l, t, r, b)
//   - ENT:     channel (MOMENTS ? 8 : 0) = mean 4-side Shannon entropy (nats, >= 0)
// Spec: f16 (T = f16/f32), R (u32, reg_max in 1..=32), MOMENTS (bool), ENT (bool).
// Params: 0 = B * A, 1 = A.
@group(0) @binding(1) var<storage, read> BOX: array<T>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let na = pu(1u);
    let b = i / na;
    let a = i % na;
    let ch = select(0u, 8u, MOMENTS) + select(0u, 1u, ENT);
    let nan_f = bitcast<f32>(0x7fc00000u);
    var ent_sum = 0.0;
    var ent_bad = false;
    var v: array<f32, R>;
    for (var side = 0u; side < 4u; side++) {
        let base = (b * 4u + side) * R * na + a;
        var m = f32(BOX[base]);
        v[0] = m;
        var has_nan = (bitcast<u32>(m) & 0x7fffffffu) > 0x7f800000u;
        for (var k = 1u; k < R; k++) {
            let x = f32(BOX[base + k * na]);
            v[k] = x;
            has_nan = has_nan || ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u);
            m = max(m, x);
        }
        if (has_nan || !is_finite_f(m)) {
            if (MOMENTS) {
                OUT[(b * ch + side) * na + a] = nan_f;
                OUT[(b * ch + 4u + side) * na + a] = nan_f;
            }
            if (ENT) {
                ent_bad = true;
            }
            continue;
        }
        var z = 0.0;
        var e = 0.0;
        var e2 = 0.0;
        var su = 0.0;
        for (var k = 0u; k < R; k++) {
            let u = v[k] - m;
            let p = fexp(u);
            z += p;
            if (MOMENTS) {
                let kf = f32(k);
                e = fma(p, kf, e);
                e2 = fma(p, kf * kf, e2);
            }
            if (ENT) {
                if (p > 0.0) {
                    su = fma(p, u, su);
                }
            }
        }
        let inv_z = 1.0 / z;
        if (MOMENTS) {
            OUT[(b * ch + side) * na + a] = e * inv_z;
            OUT[(b * ch + 4u + side) * na + a] = e2 * inv_z;
        }
        if (ENT) {
            let side_ent = log_ge1(z) - su * inv_z;
            if (is_finite_f(side_ent)) {
                ent_sum += max(side_ent, 0.0);
            } else {
                ent_bad = true;
            }
        }
    }
    if (ENT) {
        let ent_ch = select(0u, 8u, MOMENTS);
        OUT[(b * ch + ent_ch) * na + a] = select(0.25 * ent_sum, nan_f, ent_bad);
    }
}
