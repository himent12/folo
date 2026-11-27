//! 2-D max pooling (NCHW).

use std::fmt::Write as _;

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec, SpecVal};

/// Window geometry shared by pooling and convolution: kernel, stride, padding and dilation per axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window2d {
    /// Kernel size `(h, w)`.
    pub kernel: (usize, usize),
    /// Stride `(h, w)`.
    pub stride: (usize, usize),
    /// Zero padding `(h, w)` on each side.
    pub padding: (usize, usize),
    /// Dilation `(h, w)`.
    pub dilation: (usize, usize),
}

impl Window2d {
    /// Square window with the given kernel, stride and padding, dilation 1.
    pub fn square(kernel: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel: (kernel, kernel),
            stride: (stride, stride),
            padding: (padding, padding),
            dilation: (1, 1),
        }
    }

    /// Output size for an `h × w` input (floor mode, as PyTorch with `ceil_mode = false`).
    pub fn output(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let axis = |n: usize, k: usize, s: usize, p: usize, d: usize| -> Result<usize> {
            if k == 0 || s == 0 || d == 0 {
                return Err(shape_err(format!(
                    "window {self:?}: kernel, stride and dilation must be non-zero"
                )));
            }
            let span = d.checked_mul(k - 1).and_then(|v| v.checked_add(1));
            let padded = p.checked_mul(2).and_then(|v| v.checked_add(n));
            let (Some(span), Some(padded)) = (span, padded) else {
                return Err(shape_err(format!("window {self:?} overflows for a {h}x{w} input")));
            };
            if padded < span {
                return Err(shape_err(format!("window {self:?} does not fit a {h}x{w} input")));
            }
            Ok((padded - span) / s + 1)
        };
        Ok((
            axis(h, self.kernel.0, self.stride.0, self.padding.0, self.dilation.0)?,
            axis(w, self.kernel.1, self.stride.1, self.padding.1, self.dilation.1)?,
        ))
    }

    pub(crate) fn spec(&self) -> Spec {
        Spec::new()
            .u32("KH", self.kernel.0 as u32)
            .u32("KW", self.kernel.1 as u32)
            .u32("SH", self.stride.0 as u32)
            .u32("SW", self.stride.1 as u32)
            .i32("PH", self.padding.0 as i32)
            .i32("PW", self.padding.1 as i32)
            .u32("DH", self.dilation.0 as u32)
            .u32("DW", self.dilation.1 as u32)
    }

    pub(crate) fn check_limits(&self) -> Result<()> {
        let all = [self.kernel, self.stride, self.padding, self.dilation];
        if all.iter().any(|&(a, b)| a > 1 << 16 || b > 1 << 16) {
            return Err(shape_err(format!("window {self:?} is implausibly large")));
        }
        Ok(())
    }
}

/// Maximum kernel dimension supported by the separable `16×16` shared-memory tiled stride-1 max-pool path.
pub(crate) const TILED_MAX_K: usize = 7;

/// Maximum `vec4` units (`h * (w / 4)`) per plane for the single-workgroup whole-plane `vec4` stride-1
/// max-pool path (256 units = 128-thread WG covering `20×20` SPPF through `32×32`; 512 units = 256-thread WG
/// covering `40×40` SPPF at 1280px when the device's workgroup storage holds the tile).
pub(crate) const PLANE_VEC4_SMALL_CAP: usize = 256;
pub(crate) const PLANE_VEC4_LARGE_CAP: usize = 512;

/// Odd symmetric kernel widths supported by the `vec4` same-padded stride-1 max-pool kernels (`3×3 p1`,
/// `5×5 p2`, `7×7 p3`, where `ow == w` implies `2 * pad == k - 1` and halo fits in 1 `vec4` per side).
pub(crate) fn is_same_vec4_k(k: usize) -> bool {
    matches!(k, 3 | 5 | 7)
}

/// Selects `(plane_cap, workgroup_size)` for whole-plane `vec4` pooling when `hw4` fits in workgroup storage
/// (`smem_per_unit` bytes per `vec4` slot).
pub(crate) fn plane_vec4_plan(ctx: &GpuContext, hw4: usize, smem_per_unit: u32) -> Option<(u32, u32)> {
    let (cap, wg) = if hw4 <= PLANE_VEC4_SMALL_CAP {
        (PLANE_VEC4_SMALL_CAP as u32, 128u32)
    } else if hw4 <= PLANE_VEC4_LARGE_CAP {
        (PLANE_VEC4_LARGE_CAP as u32, 256u32)
    } else {
        return None;
    };
    (cap.saturating_mul(smem_per_unit) <= ctx.caps().max_workgroup_storage).then_some((cap, wg))
}

/// Shared NaN-propagating, signed-zero-preserving scalar and `vec4` max helpers.
const MAX_NAN_HELPERS: &str = r#"
fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn max_nan(a: f32, b: f32) -> f32 {
    return select(select(a, b, b > a || isnan1(b)), a, isnan1(a));
}
fn isnan4(v: vec4<f32>) -> vec4<bool> {
    return (bitcast<vec4<u32>>(v) & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
}
fn max_nan4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(select(a, b, (b > a) | isnan4(b)), a, isnan4(a));
}
"#;

/// Emits `horiz_max_vec4(left, mid, right)` using left-to-right associative shared-subexpression trees for
/// `KW ∈ {3, 5, 7}` (`pw = KW / 2`), preserving exact NaN propagation and signed-zero order.
fn emit_horiz_max_vec4(kw: u32) -> &'static str {
    match kw {
        3 => {
            r#"
fn horiz_max_vec4(left: vec4<f32>, mid: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    let m12 = max_nan(mid.x, mid.y);
    let m23 = max_nan(mid.y, mid.z);
    let m34 = max_nan(mid.z, mid.w);
    return vec4<f32>(
        max_nan(left.w, m12),
        max_nan(m12, mid.z),
        max_nan(m23, mid.w),
        max_nan(m34, right.x),
    );
}
"#
        }
        5 => {
            r#"
fn horiz_max_vec4(left: vec4<f32>, mid: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    let m23 = max_nan(mid.x, mid.y);
    let m34 = max_nan(mid.y, mid.z);
    let m234 = max_nan(m23, mid.z);
    let m345 = max_nan(m34, mid.w);
    return vec4<f32>(
        max_nan(max_nan(left.z, left.w), m234),
        max_nan(max_nan(left.w, mid.x), m345),
        max_nan(m234, max_nan(mid.w, right.x)),
        max_nan(m345, max_nan(right.x, right.y)),
    );
}
"#
        }
        7 => {
            r#"
fn horiz_max_vec4(left: vec4<f32>, mid: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    let m34 = max_nan(mid.x, mid.y);
    let m56 = max_nan(mid.z, mid.w);
    let m3456 = max_nan(m34, m56);
    let m23456 = max_nan(left.w, m3456);
    let m34567 = max_nan(m3456, right.x);
    return vec4<f32>(
        max_nan(max_nan(left.y, left.z), m23456),
        max_nan(left.z, max_nan(m23456, right.x)),
        max_nan(left.w, max_nan(m34567, right.y)),
        max_nan(m34567, max_nan(right.y, right.z)),
    );
}
"#
        }
        _ => unreachable!("is_same_vec4_k only allows 3, 5, 7"),
    }
}

/// Whole-plane `vec4` stride-1 same-padded `max_pool2d`: one workgroup (128 threads for `PLANE_CAP = 256`,
/// 256 threads for `PLANE_CAP = 512`) loads the entire plane once with coalesced `vec4<T>` reads, computes
/// horizontal `KW`-max into `sh_h`, then vertical `KH`-max with vectorized `max_nan4` and coalesced `vec4<T>`
/// stores.
/// Spec: `PLANE_VEC4 = true`, `PLANE_CAP ∈ {256, 512}`, `KH ∈ {3, 5, 7}`, `KW ∈ {3, 5, 7}`, `f16` (optional).
/// Params: 0 = planes, 1 = H, 2 = W/4, 3 = H*(W/4), 4 = -inf.
fn gen_maxpool_plane_vec4(kh: u32, kw: u32, cap: u32) -> String {
    let ph = kh / 2;
    let wg = if cap > 256 { 256u32 } else { 128u32 };
    let mut s = String::new();
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
         var<workgroup> sh_in: array<vec4<f32>, {cap}u>;\n\
         var<workgroup> sh_h: array<vec4<f32>, {cap}u>;\n"
    );
    s.push_str(MAX_NAN_HELPERS);
    s.push_str(emit_horiz_max_vec4(kw));
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
            var left = neg_inf;
            if (x4 > 0u) {{ left = sh_in[u - 1u]; }}
            var right = neg_inf;
            if (x4 + 1u < w4) {{ right = sh_in[u + 1u]; }}
            sh_h[u] = horiz_max_vec4(left, sh_in[u], right);
        }}
    }}
    workgroupBarrier();

    if (!valid) {{ return; }}
    for (var u = lid; u < hw4; u += {wg}u) {{
        let y = u / w4;
        let x4 = u - y * w4;
        var m = neg_inf;
        for (var ky = 0u; ky < {kh}u; ky++) {{
            let iy = i32(y) + i32(ky) - {ph};
            if (iy >= 0 && iy < i32(h)) {{
                m = max_nan4(m, sh_h[u32(iy) * w4 + x4]);
            }}
        }}
        Y[base4 + u] = vec4<T>(m);
    }}
}}
"#
    );
    s
}

/// Shared-memory `32×16` (`8×16` threads of `vec4`) separable stride-1 same-padded `max_pool2d`: coalesced
/// `vec4<T>` tile loads (`LOAD_H × 10`), branchless horizontal shared-subexpression reduction into `sh_h`,
/// and vectorized vertical `max_nan4` with coalesced `vec4<T>` stores.
/// Spec: `TILED_VEC4 = true`, `KH ∈ {3, 5, 7}`, `KW ∈ {3, 5, 7}`, `f16` (optional).
/// Params: 0 = total_tiles, 1 = H, 2 = W/4, 3 = -inf, 4 = tiles_x, 5 = tiles_y.
fn gen_maxpool_tiled_vec4(kh: u32, kw: u32) -> String {
    let ph = kh / 2;
    let load_h = 16 + kh - 1;
    let load_n = load_h * 10;
    let h_n = load_h * 8;
    let mut s = String::new();
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
         var<workgroup> sh_in: array<vec4<f32>, {load_n}u>;\n\
         var<workgroup> sh_h: array<vec4<f32>, {h_n}u>;\n"
    );
    s.push_str(MAX_NAN_HELPERS);
    s.push_str(emit_horiz_max_vec4(kw));
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
            let row = ly * 10u + lx;
            sh_h[idx] = horiz_max_vec4(sh_in[row], sh_in[row + 1u], sh_in[row + 2u]);
        }}
    }}
    workgroupBarrier();

    if (!valid) {{ return; }}
    let ox4 = tx4_0 + lid.x;
    let oy = ty0 + lid.y;
    if (oy < h && ox4 < w4) {{
        var m = sh_h[lid.y * 8u + lid.x];
        for (var ky = 1u; ky < {kh}u; ky++) {{
            m = max_nan4(m, sh_h[(lid.y + ky) * 8u + lid.x]);
        }}
        Y[base4 + oy * w4 + ox4] = vec4<T>(m);
    }}
}}
"#
    );
    s
}

/// Shared-memory separable `16×16` tiled stride-1 `max_pool2d` (values only, NaN-propagating).
/// Spec: `TILED = true`, `KH` (`2..=7`), `KW` (`2..=7`), `f16` (optional).
/// Params: 0 = total_tiles, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = -inf, 6 = PH (i32), 7 = PW (i32),
/// 8 = tiles_x, 9 = tiles_y.
const MAXPOOL_TILED_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> Y: array<T>;

const LOAD_H: u32 = 16u + KH - 1u;
const LOAD_W: u32 = 16u + KW - 1u;
const LOAD_N: u32 = LOAD_H * LOAD_W;
const H_N: u32 = LOAD_H * 16u;

var<workgroup> sh_in: array<f32, LOAD_N>;
var<workgroup> sh_h: array<f32, H_N>;

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn max_nan(a: f32, b: f32) -> f32 {
    return select(select(a, b, b > a || isnan1(b)), a, isnan1(a));
}

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
                sh_in[idx] = f32(X[base + u32(iy) * w + u32(ix)]);
            } else {
                sh_in[idx] = neg_inf;
            }
        }
    }
    workgroupBarrier();

    if (valid) {
        for (var idx = tid; idx < H_N; idx += 256u) {
            let ly = idx / 16u;
            let lx = idx % 16u;
            let row = ly * LOAD_W + lx;
            var m = sh_in[row];
            for (var kx = 1u; kx < KW; kx++) {
                m = max_nan(m, sh_in[row + kx]);
            }
            sh_h[idx] = m;
        }
    }
    workgroupBarrier();

    if (!valid) { return; }
    let ox = tx0 + lid.x;
    let oy = ty0 + lid.y;
    if (oy < oh && ox < ow) {
        var m = sh_h[lid.y * 16u + lid.x];
        for (var ky = 1u; ky < KH; ky++) {
            m = max_nan(m, sh_h[(lid.y + ky) * 16u + lid.x]);
        }
        Y[(p * oh + oy) * ow + ox] = T(m);
    }
}
"#;

/// Vectorized 2×2 stride-2 pad-0 `max_pool2d` (values only, NaN-propagating, signed-zero preserving) for
/// exact 2× input dims (`h == 2 * oh`, `w == 2 * ow`) and even `ow`: each thread reads two `vec4<T>`s (8 input
/// pixels across consecutive rows) and writes one `vec2<T>`.
/// Params: 0 = output `vec2<T>` units (`planes * oh * (ow / 2)`), 1 = `ow / 2`.
const MAXPOOL_2X2_S2_VEC4_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec2<T>>;

fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn max_nan(a: f32, b: f32) -> f32 {
    return select(select(a, b, b > a || isnan1(b)), a, isnan1(a));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let w2 = pu(1u);
    let x2 = i % w2;
    let t = i / w2;
    let in0 = t * (2u * w2) + x2;
    let r0 = vec4<f32>(X[in0]);
    let r1 = vec4<f32>(X[in0 + w2]);
    let a = max_nan(max_nan(r0.x, r0.y), max_nan(r1.x, r1.y));
    let b = max_nan(max_nan(r0.z, r0.w), max_nan(r1.z, r1.w));
    Y[i] = vec2<T>(T(a), T(b));
}
"#;

fn gen_maxpool(spec: &Spec) -> String {
    if spec.get_bool("S2_VEC4") {
        MAXPOOL_2X2_S2_VEC4_SRC.into()
    } else if spec.get_bool("PLANE_VEC4") {
        gen_maxpool_plane_vec4(
            spec.get_u32("KH", 5),
            spec.get_u32("KW", 5),
            spec.get_u32("PLANE_CAP", PLANE_VEC4_SMALL_CAP as u32),
        )
    } else if spec.get_bool("TILED_VEC4") {
        gen_maxpool_tiled_vec4(spec.get_u32("KH", 5), spec.get_u32("KW", 5))
    } else if spec.get_bool("TILED") {
        MAXPOOL_TILED_SRC.into()
    } else {
        include_str!("../shaders/maxpool.wgsl").into()
    }
}

/// Bounds check for precompiling `"max_pool2d"` from a kernel list (`kernels/bounds.rs`).
pub(crate) fn maxpool_fits(spec: &Spec) -> bool {
    if spec.get_bool("PLANE_VEC4") {
        let cap_ok = matches!(spec.get("PLANE_CAP"), Some(SpecVal::U32(256 | 512)));
        return cap_ok
            && matches!(spec.get("KH"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize))
            && matches!(spec.get("KW"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize));
    }
    if spec.get_bool("TILED_VEC4") {
        return matches!(spec.get("KH"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize))
            && matches!(spec.get("KW"), Some(SpecVal::U32(k)) if is_same_vec4_k(k as usize));
    }
    if !spec.get_bool("TILED") {
        return true;
    }
    matches!(spec.get("KH"), Some(SpecVal::U32(k)) if (2..=TILED_MAX_K).contains(&(k as usize)))
        && matches!(spec.get("KW"), Some(SpecVal::U32(k)) if (2..=TILED_MAX_K).contains(&(k as usize)))
}

static MAXPOOL: KernelDef = KernelDef {
    name: "max_pool2d",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_maxpool),
};

/// `max_pool2d` over `planes = N·C` planes of `h × w`; padded positions are ignored (−∞) and NaN wins
/// (matching the CPU reference and [`super::pool_grad::max_pool2d_indices`]).
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    planes: usize,
    h: usize,
    w: usize,
    win: &Window2d,
    dtype: DType,
) -> Result<(usize, usize)> {
    check_dtype(ctx, dtype, "max_pool2d")?;
    win.check_limits()?;
    if win.padding.0 * 2 > win.kernel.0 || win.padding.1 * 2 > win.kernel.1 {
        return Err(shape_err(format!("max_pool2d: padding must be at most half the kernel, got {win:?}")));
    }
    let (oh, ow) = win.output(h, w)?;
    let in_elems = prod(&[planes, h, w])?;
    let n = prod(&[planes, oh, ow])?;
    check_buf(x, in_elems, dtype, "max_pool2d input")?;
    check_buf(y, n, dtype, "max_pool2d output")?;
    if n == 0 {
        return Ok((oh, ow));
    }
    if in_elems > u32::MAX as usize || n > u32::MAX as usize {
        return Err(GpuError::Limit("max_pool2d: tensor exceeds u32 indexing".into()));
    }
    if h > i32::MAX as usize - 2 * win.padding.0 || w > i32::MAX as usize - 2 * win.padding.1 {
        return Err(GpuError::Limit("max_pool2d: plane dimensions exceed i32 indexing".into()));
    }
    let half = dtype == DType::F16;
    if *win == Window2d::square(2, 2, 0) && h == 2 * oh && w == 2 * ow && ow.is_multiple_of(2) {
        let w2 = ow / 2;
        let units = planes * oh * w2;
        let k = ctx.kernel(&MAXPOOL, Spec::new().bool("S2_VEC4", true).f16(half))?;
        let p = Params::new().uz(units)?.uz(w2)?;
        ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(units as u64, 256)?)?;
        return Ok((oh, ow));
    }
    let same_vec4 = win.stride == (1, 1)
        && win.dilation == (1, 1)
        && is_same_vec4_k(win.kernel.0)
        && is_same_vec4_k(win.kernel.1)
        && oh == h
        && ow == w
        && w.is_multiple_of(4)
        && h <= i32::MAX as usize - 64
        && w <= i32::MAX as usize - 64;
    if same_vec4 {
        let w4 = w / 4;
        let hw4 = h * w4;
        if let Some((cap, wg)) = plane_vec4_plan(ctx, hw4, 32)
            && planes <= u32::MAX as usize
            && let Some(threads) = (planes as u64).checked_mul(wg as u64)
            && let Ok(grid) = ctx.grid_1d(threads, wg)
        {
            let spec = Spec::new()
                .bool("PLANE_VEC4", true)
                .u32("PLANE_CAP", cap)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&MAXPOOL, spec)?;
            let p = Params::new().uz(planes)?.uz(h)?.uz(w4)?.uz(hw4)?.f(f32::NEG_INFINITY);
            ctx.dispatch(&k, &[x, y], &p, grid)?;
            return Ok((oh, ow));
        }
        let (tiles_x, tiles_y) = (w4.div_ceil(8), h.div_ceil(16));
        if let Ok(total_tiles) = prod(&[planes, tiles_y, tiles_x])
            && total_tiles <= u32::MAX as usize
            && let Some(threads) = (total_tiles as u64).checked_mul(128)
            && let Ok(grid) = ctx.grid_1d(threads, 128)
        {
            let spec = Spec::new()
                .bool("TILED_VEC4", true)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&MAXPOOL, spec)?;
            let p = Params::new()
                .uz(total_tiles)?
                .uz(h)?
                .uz(w4)?
                .f(f32::NEG_INFINITY)
                .uz(tiles_x)?
                .uz(tiles_y)?;
            ctx.dispatch(&k, &[x, y], &p, grid)?;
            return Ok((oh, ow));
        }
    }
    let tiled_geom = win.stride == (1, 1)
        && win.dilation == (1, 1)
        && (2..=TILED_MAX_K).contains(&win.kernel.0)
        && (2..=TILED_MAX_K).contains(&win.kernel.1)
        && h <= i32::MAX as usize - 64
        && w <= i32::MAX as usize - 64;
    if tiled_geom {
        let (tiles_x, tiles_y) = (ow.div_ceil(16), oh.div_ceil(16));
        if let Ok(total_tiles) = prod(&[planes, tiles_y, tiles_x])
            && total_tiles <= u32::MAX as usize
            && let Some(threads) = (total_tiles as u64).checked_mul(256)
            && let Ok(grid) = ctx.grid_1d(threads, 256)
        {
            let spec = Spec::new()
                .bool("TILED", true)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&MAXPOOL, spec)?;
            let p = Params::new()
                .uz(total_tiles)?
                .uz(h)?
                .uz(w)?
                .uz(oh)?
                .uz(ow)?
                .f(f32::NEG_INFINITY)
                .uz(win.padding.0)?
                .uz(win.padding.1)?
                .uz(tiles_x)?
                .uz(tiles_y)?;
            ctx.dispatch(&k, &[x, y], &p, grid)?;
            return Ok((oh, ow));
        }
    }
    let k = ctx.kernel(&MAXPOOL, win.spec().f16(half))?;
    let p = Params::new().uz(n)?.uz(h)?.uz(w)?.uz(oh)?.uz(ow)?.f(f32::NEG_INFINITY);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)?;
    Ok((oh, ow))
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&MAXPOOL];
