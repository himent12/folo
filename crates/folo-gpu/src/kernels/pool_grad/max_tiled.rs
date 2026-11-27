//! Stride-1 shared-memory tiled and whole-plane `vec4` shaders for [`super::max_pool2d_indices`] and
//! [`super::max_pool2d_backward`].

use std::fmt::Write as _;

use crate::kernels::pooling::{TILED_MAX_K, is_same_vec4_k};
use crate::pipeline::{Spec, SpecVal};

const VI_HELPERS: &str = r#"
struct VI { v: f32, i: i32 }
fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn isnan4(v: vec4<f32>) -> vec4<bool> {
    return (bitcast<vec4<u32>>(v) & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
}
fn pick_vi(a: VI, b: VI) -> VI {
    if (b.i < 0) { return a; }
    if (a.i < 0) { return b; }
    let a_nan = isnan1(a.v);
    let take_b = !a_nan && (isnan1(b.v) || b.v > a.v);
    return VI(select(a.v, b.v, take_b), select(a.i, b.i, take_b));
}
fn take_vi4(av: vec4<f32>, ai: vec4<i32>, bv: vec4<f32>, bi: vec4<i32>) -> vec4<bool> {
    let b_valid = bi >= vec4<i32>(0);
    let a_invalid = ai < vec4<i32>(0);
    let a_nan = isnan4(av);
    let b_nan = isnan4(bv);
    return b_valid & (a_invalid | (!a_nan & (b_nan | (bv > av))));
}
struct VI4 { v: vec4<f32>, i: vec4<i32> }
"#;

/// Emits `horiz_vi_vec4(left, mid, right, base_idx, has_left, has_right)` for `KW ∈ {3, 5, 7}` (`pw = KW / 2`),
/// using strictly associative left-to-right shared-subexpression trees so ties and NaNs match row-major scan.
fn emit_horiz_vi_vec4(kw: u32) -> &'static str {
    match kw {
        3 => {
            r#"
fn horiz_vi_vec4(left: vec4<f32>, mid: vec4<f32>, right: vec4<f32>, b: i32, hl: bool, hr: bool) -> VI4 {
    let a0 = VI(left.w, select(-1, b - 1, hl));
    let a1 = VI(mid.x, b);
    let a2 = VI(mid.y, b + 1);
    let a3 = VI(mid.z, b + 2);
    let a4 = VI(mid.w, b + 3);
    let a5 = VI(right.x, select(-1, b + 4, hr));
    let m12 = pick_vi(a1, a2);
    let m23 = pick_vi(a2, a3);
    let m34 = pick_vi(a3, a4);
    let r0 = pick_vi(a0, m12);
    let r1 = pick_vi(m12, a3);
    let r2 = pick_vi(m23, a4);
    let r3 = pick_vi(m34, a5);
    return VI4(vec4<f32>(r0.v, r1.v, r2.v, r3.v), vec4<i32>(r0.i, r1.i, r2.i, r3.i));
}
"#
        }
        5 => {
            r#"
fn horiz_vi_vec4(left: vec4<f32>, mid: vec4<f32>, right: vec4<f32>, b: i32, hl: bool, hr: bool) -> VI4 {
    let a0 = VI(left.z, select(-1, b - 2, hl));
    let a1 = VI(left.w, select(-1, b - 1, hl));
    let a2 = VI(mid.x, b);
    let a3 = VI(mid.y, b + 1);
    let a4 = VI(mid.z, b + 2);
    let a5 = VI(mid.w, b + 3);
    let a6 = VI(right.x, select(-1, b + 4, hr));
    let a7 = VI(right.y, select(-1, b + 5, hr));
    let m23 = pick_vi(a2, a3);
    let m34 = pick_vi(a3, a4);
    let m234 = pick_vi(m23, a4);
    let m345 = pick_vi(m34, a5);
    let r0 = pick_vi(pick_vi(a0, a1), m234);
    let r1 = pick_vi(pick_vi(a1, a2), m345);
    let r2 = pick_vi(m234, pick_vi(a5, a6));
    let r3 = pick_vi(m345, pick_vi(a6, a7));
    return VI4(vec4<f32>(r0.v, r1.v, r2.v, r3.v), vec4<i32>(r0.i, r1.i, r2.i, r3.i));
}
"#
        }
        7 => {
            r#"
fn horiz_vi_vec4(left: vec4<f32>, mid: vec4<f32>, right: vec4<f32>, b: i32, hl: bool, hr: bool) -> VI4 {
    let a0 = VI(left.y, select(-1, b - 3, hl));
    let a1 = VI(left.z, select(-1, b - 2, hl));
    let a2 = VI(left.w, select(-1, b - 1, hl));
    let a3 = VI(mid.x, b);
    let a4 = VI(mid.y, b + 1);
    let a5 = VI(mid.z, b + 2);
    let a6 = VI(mid.w, b + 3);
    let a7 = VI(right.x, select(-1, b + 4, hr));
    let a8 = VI(right.y, select(-1, b + 5, hr));
    let a9 = VI(right.z, select(-1, b + 6, hr));
    let m34 = pick_vi(a3, a4);
    let m56 = pick_vi(a5, a6);
    let m3456 = pick_vi(m34, m56);
    let m23456 = pick_vi(a2, m3456);
    let m34567 = pick_vi(m3456, a7);
    let r0 = pick_vi(pick_vi(a0, a1), m23456);
    let r1 = pick_vi(a1, pick_vi(m23456, a7));
    let r2 = pick_vi(a2, pick_vi(m34567, a8));
    let r3 = pick_vi(m34567, pick_vi(a8, a9));
    return VI4(vec4<f32>(r0.v, r1.v, r2.v, r3.v), vec4<i32>(r0.i, r1.i, r2.i, r3.i));
}
"#
        }
        _ => unreachable!("is_same_vec4_k only allows 3, 5, 7"),
    }
}

/// Whole-plane `vec4` stride-1 same-padded `max_pool2d_indices` (`PLANE_CAP ∈ {256, 512}`, 128 or 256
/// threads/plane): coalesced `vec4<T>` loads/stores and coalesced `vec4<u32>` `i64` index stores.
/// Spec: `MODE = 0`, `PLANE_VEC4 = true`, `PLANE_CAP ∈ {256, 512}`, `KH ∈ {3, 5, 7}`, `KW ∈ {3, 5, 7}`, `f16` (optional).
/// Params: 0 = planes, 1 = H, 2 = W/4, 3 = H*(W/4), 4 = -inf.
pub(super) fn gen_pool_idx_plane_vec4(kh: u32, kw: u32, cap: u32) -> String {
    let ph = kh / 2;
    let wg = if cap > 256 { 256u32 } else { 128u32 };
    let mut s = String::new();
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
         @group(0) @binding(3) var<storage, read_write> YI: array<vec4<u32>>;\n\
         var<workgroup> sh_in: array<vec4<f32>, {cap}u>;\n\
         var<workgroup> sh_h: array<vec4<f32>, {cap}u>;\n\
         var<workgroup> sh_hi: array<vec4<i32>, {cap}u>;\n"
    );
    s.push_str(VI_HELPERS);
    s.push_str(emit_horiz_vi_vec4(kw));
    let _ = write!(
        s,
        r#"
@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {{
    let p = wid.y * nwg.x + wid.x;
    let valid = p < pu(0u);
    let h = pu(1u); let w4 = pu(2u); let hw4 = pu(3u);
    let neg_inf = vec4<f32>(pf(4u));
    let base4 = p * hw4;

    if (valid) {{
        for (var u = lid; u < hw4; u += {wg}u) {{
            sh_in[u] = vec4<f32>(X[base4 + u]);
        }}
    }}
    workgroupBarrier();

    if (valid) {{
        for (var u = lid; u < hw4; u += {wg}u) {{
            let y = u / w4;
            let x4 = u - y * w4;
            let hl = x4 > 0u;
            let hr = x4 + 1u < w4;
            var left = neg_inf;
            if (hl) {{ left = sh_in[u - 1u]; }}
            var right = neg_inf;
            if (hr) {{ right = sh_in[u + 1u]; }}
            let r = horiz_vi_vec4(left, sh_in[u], right, i32(u << 2u), hl, hr);
            sh_h[u] = r.v;
            sh_hi[u] = r.i;
        }}
    }}
    workgroupBarrier();

    if (!valid) {{ return; }}
    for (var u = lid; u < hw4; u += {wg}u) {{
        let y = u / w4;
        let x4 = u - y * w4;
        var best = neg_inf;
        var bi = vec4<i32>(-1);
        for (var ky = 0u; ky < {kh}u; ky++) {{
            let iy = i32(y) + i32(ky) - {ph};
            if (iy >= 0 && iy < i32(h)) {{
                let off = u32(iy) * w4 + x4;
                let rv = sh_h[off];
                let ri = sh_hi[off];
                let take = take_vi4(best, bi, rv, ri);
                best = select(best, rv, take);
                bi = select(bi, ri, take);
            }}
        }}
        let out4 = base4 + u;
        Y[out4] = vec4<T>(best);
        let bi_u = vec4<u32>(max(bi, vec4<i32>(0)));
        YI[2u * out4] = vec4<u32>(bi_u.x, 0u, bi_u.y, 0u);
        YI[2u * out4 + 1u] = vec4<u32>(bi_u.z, 0u, bi_u.w, 0u);
    }}
}}
"#
    );
    s
}

/// Shared-memory `32×16` (`8×16` threads of `vec4`) separable stride-1 same-padded `max_pool2d_indices`.
/// Spec: `MODE = 0`, `TILED_VEC4 = true`, `KH ∈ {3, 5, 7}`, `KW ∈ {3, 5, 7}`, `f16` (optional).
/// Params: 0 = total_tiles, 1 = H, 2 = W/4, 3 = -inf, 4 = tiles_x, 5 = tiles_y.
pub(super) fn gen_pool_idx_tiled_vec4(kh: u32, kw: u32) -> String {
    let ph = kh / 2;
    let load_h = 16 + kh - 1;
    let load_n = load_h * 10;
    let h_n = load_h * 8;
    let mut s = String::new();
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
         @group(0) @binding(3) var<storage, read_write> YI: array<vec4<u32>>;\n\
         var<workgroup> sh_in: array<vec4<f32>, {load_n}u>;\n\
         var<workgroup> sh_h: array<vec4<f32>, {h_n}u>;\n\
         var<workgroup> sh_hi: array<vec4<i32>, {h_n}u>;\n"
    );
    s.push_str(VI_HELPERS);
    s.push_str(emit_horiz_vi_vec4(kw));
    let _ = write!(
        s,
        r#"
@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tile_id = wid.y * nwg.x + wid.x;
    let valid = tile_id < pu(0u);
    let h = pu(1u); let w4 = pu(2u);
    let neg_inf = vec4<f32>(pf(3u));
    let tiles_x = pu(4u); let tiles_y = pu(5u);
    let tx = tile_id % tiles_x;
    let tmp = tile_id / tiles_x;
    let ty = tmp % tiles_y;
    let p = tmp / tiles_y;
    let tx4_0 = tx * 8u;
    let ty0 = ty * 16u;
    let base4 = p * h * w4;
    let tid = lid.y * 8u + lid.x;

    if (valid) {{
        for (var idx = tid; idx < {load_n}u; idx += 128u) {{
            let ly = idx / 10u;
            let lx = idx - ly * 10u;
            let iy = i32(ty0 + ly) - {ph};
            let ix4 = i32(tx4_0 + lx) - 1;
            if (iy >= 0 && iy < i32(h) && ix4 >= 0 && ix4 < i32(w4)) {{
                sh_in[idx] = vec4<f32>(X[base4 + u32(iy) * w4 + u32(ix4)]);
            }} else {{
                sh_in[idx] = neg_inf;
            }}
        }}
    }}
    workgroupBarrier();

    if (valid) {{
        for (var idx = tid; idx < {h_n}u; idx += 128u) {{
            let ly = idx >> 3u;
            let lx = idx & 7u;
            let iy = i32(ty0 + ly) - {ph};
            let ix4 = tx4_0 + lx;
            if (iy >= 0 && iy < i32(h) && ix4 < w4) {{
                let row = ly * 10u + lx;
                let hl = ix4 > 0u;
                let hr = ix4 + 1u < w4;
                let b = i32((u32(iy) * w4 + ix4) << 2u);
                let r = horiz_vi_vec4(sh_in[row], sh_in[row + 1u], sh_in[row + 2u], b, hl, hr);
                sh_h[idx] = r.v;
                sh_hi[idx] = r.i;
            }} else {{
                sh_h[idx] = neg_inf;
                sh_hi[idx] = vec4<i32>(-1);
            }}
        }}
    }}
    workgroupBarrier();

    if (!valid) {{ return; }}
    let ox4 = tx4_0 + lid.x;
    let oy = ty0 + lid.y;
    if (oy < h && ox4 < w4) {{
        let col = lid.y * 8u + lid.x;
        var best = sh_h[col];
        var bi = sh_hi[col];
        for (var ky = 1u; ky < {kh}u; ky++) {{
            let off = (lid.y + ky) * 8u + lid.x;
            let rv = sh_h[off];
            let ri = sh_hi[off];
            let take = take_vi4(best, bi, rv, ri);
            best = select(best, rv, take);
            bi = select(bi, ri, take);
        }}
        let out4 = base4 + oy * w4 + ox4;
        Y[out4] = vec4<T>(best);
        let bi_u = vec4<u32>(max(bi, vec4<i32>(0)));
        YI[2u * out4] = vec4<u32>(bi_u.x, 0u, bi_u.y, 0u);
        YI[2u * out4 + 1u] = vec4<u32>(bi_u.z, 0u, bi_u.w, 0u);
    }}
}}
"#
    );
    s
}

/// Emits unrolled per-`dx` gather checks for 4 horizontally adjacent input pixels `(own0..own0+3)`, reading
/// each shared `(sh_idx, sh_g)` pair once per `(dy, dx)` in exact row-major order.
fn emit_bw_dx_loop(kw: u32, idx_expr: &str, check_bounds: bool) -> String {
    let pw = kw / 2;
    let mut s = String::new();
    for dx in 0..kw + 3 {
        if check_bounds {
            let _ = writeln!(s, "            {{ let ox = i32(x0) + {dx} - {pw};");
            s.push_str("              if (ox >= 0 && ox < i32(w)) {\n");
            let _ = writeln!(s, "                let pos = {idx_expr};");
        } else {
            let _ = writeln!(s, "            {{ let pos = {idx_expr} + {dx}u;");
        }
        s.push_str("                let id = sh_idx[pos]; let gv = sh_g[pos];\n");
        for c in 0..4u32 {
            if dx >= c && dx < c + kw {
                let own = if c == 0 { "own0".to_string() } else { format!("own0 + {c}u") };
                let _ = writeln!(s, "                if (id == {own}) {{ a{c} += gv; }}");
            }
        }
        if check_bounds {
            s.push_str("              }\n            }\n");
        } else {
            s.push_str("            }\n");
        }
    }
    s
}

/// Whole-plane `vec4` stride-1 same-padded `max_pool2d_backward` (`PLANE_CAP ∈ {256, 512}`, 128 or 256
/// threads/plane): loads the plane's upstream gradients (`vec4<T>`) and argmax indices (`vec4<u32>`) once per
/// plane and gathers 4 adjacent input pixels per thread into a coalesced `vec4<T>` store.
/// Spec: `MODE = 1`, `PLANE_VEC4 = true`, `PLANE_CAP ∈ {256, 512}`, `KH ∈ {3, 5, 7}`, `KW ∈ {3, 5, 7}`, `f16` (optional).
/// Params: 0 = planes, 1 = H, 2 = W/4, 3 = H*(W/4).
pub(super) fn gen_pool_bw_idx_plane_vec4(kh: u32, kw: u32, cap: u32) -> String {
    let ph = kh / 2;
    let wg = if cap > 256 { 256u32 } else { 128u32 };
    let cap_sc = cap * 4;
    let dx_body = emit_bw_dx_loop(kw, "row + u32(ox)", true);
    format!(
        r#"
@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> IDX: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<T>>;

var<workgroup> sh_idx: array<u32, {cap_sc}u>;
var<workgroup> sh_g: array<f32, {cap_sc}u>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {{
    let p = wid.y * nwg.x + wid.x;
    let valid = p < pu(0u);
    let h = pu(1u); let w4 = pu(2u); let hw4 = pu(3u);
    let w = w4 << 2u;
    let base4 = p * hw4;

    if (valid) {{
        for (var u = lid; u < hw4; u += {wg}u) {{
            let o4 = base4 + u;
            let g4 = vec4<f32>(X[o4]);
            let p01 = IDX[2u * o4];
            let p23 = IDX[2u * o4 + 1u];
            let s = u << 2u;
            sh_g[s] = g4.x; sh_g[s + 1u] = g4.y; sh_g[s + 2u] = g4.z; sh_g[s + 3u] = g4.w;
            sh_idx[s] = p01.x; sh_idx[s + 1u] = p01.z; sh_idx[s + 2u] = p23.x; sh_idx[s + 3u] = p23.z;
        }}
    }}
    workgroupBarrier();

    if (!valid) {{ return; }}
    for (var u = lid; u < hw4; u += {wg}u) {{
        let y = u / w4;
        let x4 = u - y * w4;
        let x0 = x4 << 2u;
        let own0 = u << 2u;
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        for (var dy = 0u; dy < {kh}u; dy++) {{
            let oy = i32(y) + i32(dy) - {ph};
            if (oy >= 0 && oy < i32(h)) {{
                let row = u32(oy) * w;
{dx_body}            }}
        }}
        Y[base4 + u] = vec4<T>(vec4<f32>(a0, a1, a2, a3));
    }}
}}
"#
    )
}

/// Shared-memory `32×16` (`8×16` threads of `vec4`) stride-1 same-padded `max_pool2d_backward`.
/// Spec: `MODE = 1`, `TILED_VEC4 = true`, `KH ∈ {3, 5, 7}`, `KW ∈ {3, 5, 7}`, `f16` (optional).
/// Params: 0 = total_tiles, 1 = H, 2 = W/4, 3 = tiles_x, 4 = tiles_y.
pub(super) fn gen_pool_bw_idx_tiled_vec4(kh: u32, kw: u32) -> String {
    let ph = kh / 2;
    let pw = kw / 2;
    let load_h = 16 + kh - 1;
    let load_v4 = load_h * 10;
    let load_sc = load_v4 * 4;
    let dx_body = emit_bw_dx_loop(kw, "row", false);
    format!(
        r#"
@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> IDX: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<T>>;

var<workgroup> sh_idx: array<u32, {load_sc}u>;
var<workgroup> sh_g: array<f32, {load_sc}u>;

@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tile_id = wid.y * nwg.x + wid.x;
    let valid = tile_id < pu(0u);
    let h = pu(1u); let w4 = pu(2u);
    let tiles_x = pu(3u); let tiles_y = pu(4u);
    let tx = tile_id % tiles_x;
    let tmp = tile_id / tiles_x;
    let ty = tmp % tiles_y;
    let p = tmp / tiles_y;
    let tx4_0 = tx * 8u;
    let ty0 = ty * 16u;
    let base4 = p * h * w4;
    let tid = lid.y * 8u + lid.x;

    if (valid) {{
        for (var idx = tid; idx < {load_v4}u; idx += 128u) {{
            let ly = idx / 10u;
            let lx = idx - ly * 10u;
            let oy = i32(ty0 + ly) - {ph};
            let ox4 = i32(tx4_0 + lx) - 1;
            let s = idx << 2u;
            if (oy >= 0 && oy < i32(h) && ox4 >= 0 && ox4 < i32(w4)) {{
                let o4 = base4 + u32(oy) * w4 + u32(ox4);
                let g4 = vec4<f32>(X[o4]);
                let p01 = IDX[2u * o4];
                let p23 = IDX[2u * o4 + 1u];
                sh_g[s] = g4.x; sh_g[s + 1u] = g4.y; sh_g[s + 2u] = g4.z; sh_g[s + 3u] = g4.w;
                sh_idx[s] = p01.x; sh_idx[s + 1u] = p01.z; sh_idx[s + 2u] = p23.x; sh_idx[s + 3u] = p23.z;
            }} else {{
                sh_g[s] = 0.0; sh_g[s + 1u] = 0.0; sh_g[s + 2u] = 0.0; sh_g[s + 3u] = 0.0;
                sh_idx[s] = 0xffffffffu; sh_idx[s + 1u] = 0xffffffffu;
                sh_idx[s + 2u] = 0xffffffffu; sh_idx[s + 3u] = 0xffffffffu;
            }}
        }}
    }}
    workgroupBarrier();

    if (!valid) {{ return; }}
    let ix4 = tx4_0 + lid.x;
    let iy = ty0 + lid.y;
    if (iy < h && ix4 < w4) {{
        let own0 = (iy * w4 + ix4) << 2u;
        let col0 = (4u - {pw}u) + (lid.x << 2u);
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        for (var dy = 0u; dy < {kh}u; dy++) {{
            let row = (lid.y + dy) * 40u + col0;
{dx_body}        }}
        Y[base4 + iy * w4 + ix4] = vec4<T>(vec4<f32>(a0, a1, a2, a3));
    }}
}}
"#
    )
}

/// Shared-memory separable `16×16` tiled stride-1 `max_pool2d_indices`: bit-exact row-major tie-breaking
/// (leftmost column, topmost row) and first-NaN preservation.
/// Spec: `MODE = 0`, `TILED = true`, `KH` (`2..=7`), `KW` (`2..=7`), `f16` (optional).
/// Params: 0 = total_tiles, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = -inf, 6 = PH (i32), 7 = PW (i32),
/// 8 = tiles_x, 9 = tiles_y.
pub(super) const POOL_IDX_TILED_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;
@group(0) @binding(3) var<storage, read_write> YI: array<u32>;

const LOAD_H: u32 = 16u + KH - 1u;
const LOAD_W: u32 = 16u + KW - 1u;
const LOAD_N: u32 = LOAD_H * LOAD_W;
const H_N: u32 = LOAD_H * 16u;

var<workgroup> sh_in: array<f32, LOAD_N>;
var<workgroup> sh_in_i: array<i32, LOAD_N>;
var<workgroup> sh_h: array<f32, H_N>;
var<workgroup> sh_h_i: array<i32, H_N>;

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_id = wid.y * nwg.x + wid.x;
    let valid = tile_id < pu(0u);
    let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u);
    let neg_inf = pf(5u);
    let ph = i32(pu(6u)); let pw = i32(pu(7u));
    let tiles_x = pu(8u); let tiles_y = pu(9u);
    let tx = tile_id % tiles_x;
    let tmp = tile_id / tiles_x;
    let ty = tmp % tiles_y;
    let p = tmp / tiles_y;
    let tx0 = tx * 16u;
    let ty0 = ty * 16u;
    let base = p * h * w;
    let tid = lid.y * 16u + lid.x;

    if (valid) {
        for (var idx = tid; idx < LOAD_N; idx += 256u) {
            let ly = idx / LOAD_W;
            let lx = idx % LOAD_W;
            let iy = i32(ty0 + ly) - ph;
            let ix = i32(tx0 + lx) - pw;
            if (iy >= 0 && iy < i32(h) && ix >= 0 && ix < i32(w)) {
                let at = iy * i32(w) + ix;
                sh_in[idx] = f32(X[base + u32(at)]);
                sh_in_i[idx] = at;
            } else {
                sh_in[idx] = neg_inf;
                sh_in_i[idx] = -1;
            }
        }
    }
    workgroupBarrier();

    if (valid) {
        for (var idx = tid; idx < H_N; idx += 256u) {
            let ly = idx / 16u;
            let lx = idx % 16u;
            let row = ly * LOAD_W + lx;
            var best = sh_in[row];
            var bi = sh_in_i[row];
            var has_nan = bi >= 0 && isnan1(best);
            for (var kx = 1u; kx < KW; kx++) {
                let vi = sh_in_i[row + kx];
                if (vi < 0) { continue; }
                let v = sh_in[row + kx];
                if (bi < 0 || (!has_nan && (isnan1(v) || v > best))) {
                    best = v;
                    bi = vi;
                    has_nan = isnan1(v);
                }
            }
            sh_h[idx] = best;
            sh_h_i[idx] = bi;
        }
    }
    workgroupBarrier();

    if (!valid) { return; }
    let ox = tx0 + lid.x;
    let oy = ty0 + lid.y;
    if (oy < oh && ox < ow) {
        let col = lid.y * 16u + lid.x;
        var best = sh_h[col];
        var bi = sh_h_i[col];
        var has_nan = bi >= 0 && isnan1(best);
        for (var ky = 1u; ky < KH; ky++) {
            let off = (lid.y + ky) * 16u + lid.x;
            let vi = sh_h_i[off];
            if (vi < 0) { continue; }
            let v = sh_h[off];
            if (bi < 0 || (!has_nan && (isnan1(v) || v > best))) {
                best = v;
                bi = vi;
                has_nan = isnan1(v);
            }
        }
        let out_i = (p * oh + oy) * ow + ox;
        Y[out_i] = T(best);
        YI[2u * out_i] = u32(max(bi, 0));
        YI[2u * out_i + 1u] = 0u;
    }
}
"#;

/// Shared-memory `16×16` tiled stride-1 `max_pool2d_backward`: loads the `(16 + KH - 1) × (16 + KW - 1)`
/// output-gradient and argmax-index halo once per workgroup and accumulates in exact row-major `(oy, ox)` order.
/// Spec: `MODE = 1`, `TILED = true`, `KH` (`2..=7`), `KW` (`2..=7`), `f16` (optional).
/// Params: 0 = total_tiles, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = PH (i32), 6 = PW (i32), 7 = tiles_x, 8 = tiles_y.
pub(super) const POOL_BW_IDX_TILED_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read> IDX: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<T>;

const LOAD_H: u32 = 16u + KH - 1u;
const LOAD_W: u32 = 16u + KW - 1u;
const LOAD_N: u32 = LOAD_H * LOAD_W;

var<workgroup> sh_idx: array<u32, LOAD_N>;
var<workgroup> sh_g: array<f32, LOAD_N>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_id = wid.y * nwg.x + wid.x;
    let valid = tile_id < pu(0u);
    let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u);
    let ph = i32(pu(5u)); let pw = i32(pu(6u));
    let tiles_x = pu(7u); let tiles_y = pu(8u);
    let tx = tile_id % tiles_x;
    let tmp = tile_id / tiles_x;
    let ty = tmp % tiles_y;
    let p = tmp / tiles_y;
    let tx0 = tx * 16u;
    let ty0 = ty * 16u;
    let gbase = p * oh * ow;
    let tid = lid.y * 16u + lid.x;
    let oy_base = i32(ty0) + ph - i32(KH - 1u);
    let ox_base = i32(tx0) + pw - i32(KW - 1u);

    if (valid) {
        for (var idx = tid; idx < LOAD_N; idx += 256u) {
            let ly = idx / LOAD_W;
            let lx = idx % LOAD_W;
            let oy = oy_base + i32(ly);
            let ox = ox_base + i32(lx);
            if (oy >= 0 && oy < i32(oh) && ox >= 0 && ox < i32(ow)) {
                let o = gbase + u32(oy) * ow + u32(ox);
                sh_idx[idx] = IDX[2u * o];
                sh_g[idx] = f32(X[o]);
            } else {
                sh_idx[idx] = 0xffffffffu;
                sh_g[idx] = 0.0;
            }
        }
    }
    workgroupBarrier();

    if (!valid) { return; }
    let ix = tx0 + lid.x;
    let iy = ty0 + lid.y;
    if (iy < h && ix < w) {
        let own = iy * w + ix;
        var acc = 0.0;
        for (var dy = 0u; dy < KH; dy++) {
            let row = (lid.y + dy) * LOAD_W + lid.x;
            for (var dx = 0u; dx < KW; dx++) {
                if (sh_idx[row + dx] == own) {
                    acc += sh_g[row + dx];
                }
            }
        }
        Y[(p * h + iy) * w + ix] = T(acc);
    }
}
"#;

/// Bounds check for precompiling `"pool"`, `"pool_idx"`, and `"pool_bw_idx"` from a kernel list
/// (`kernels/bounds.rs`).
pub(crate) fn pool_tiled_fits(name: &str, spec: &Spec) -> bool {
    let plane_mode = spec.get_bool("PLANE_VEC4");
    let tiled_vec4_mode = spec.get_bool("TILED_VEC4");
    let tiled_mode = spec.get_bool("TILED");
    if !plane_mode && !tiled_vec4_mode && !tiled_mode {
        return true;
    }
    let expected_mode = match name {
        "pool_idx" => 0,
        "pool_bw_idx" => 1,
        _ => return false,
    };
    if !matches!(spec.get("MODE"), Some(SpecVal::U32(m)) if m == expected_mode) {
        return false;
    }
    if plane_mode {
        return matches!(spec.get("PLANE_CAP"), Some(SpecVal::U32(256 | 512)))
            && matches!(spec.get("KH"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize))
            && matches!(spec.get("KW"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize));
    }
    if tiled_vec4_mode {
        return matches!(spec.get("KH"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize))
            && matches!(spec.get("KW"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize));
    }
    matches!(spec.get("KH"), Some(SpecVal::U32(k)) if (2..=TILED_MAX_K).contains(&(k as usize)))
        && matches!(spec.get("KW"), Some(SpecVal::U32(k)) if (2..=TILED_MAX_K).contains(&(k as usize)))
}
