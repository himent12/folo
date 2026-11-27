//! Pooling and upsampling kernels with the exact semantics of folo-core's CPU backend, and their gradients:
//! max pooling with argmax indices, average and adaptive average pooling, nearest-upsampling's backward.
//!
//! All gradients are gathers (one thread per input element sums the outputs that read it), so they are
//! deterministic and need no float atomics. Planes are dense NCHW; max pooling, average pooling, their
//! gradients, and upsample backward run natively in f32 or f16 (accumulated/compared in f32), while adaptive
//! average pooling runs in f32. Windows have no dilation; padding is at most half the kernel, so every window
//! holds at least one real element.

mod max_tiled;

use super::pooling::{PLANE_VEC4_SMALL_CAP, TILED_MAX_K, is_same_vec4_k, plane_vec4_plan};
use super::reduce::{self, ReduceOp};
use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
pub(crate) use max_tiled::pool_tiled_fits;
use max_tiled::{
    POOL_BW_IDX_TILED_SRC, POOL_IDX_TILED_SRC, gen_pool_bw_idx_plane_vec4, gen_pool_bw_idx_tiled_vec4,
    gen_pool_idx_plane_vec4, gen_pool_idx_tiled_vec4,
};

/// Pooling window: kernel, stride and padding per axis `(h, w)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool2d {
    /// Kernel `(h, w)`.
    pub kernel: (usize, usize),
    /// Stride `(h, w)`.
    pub stride: (usize, usize),
    /// Padding `(h, w)` on both sides.
    pub padding: (usize, usize),
}

impl Pool2d {
    /// Output size for an `h × w` plane (floor mode).
    pub fn output(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let axis = |i: usize, k: usize, s: usize, p: usize| {
            if k == 0 || s == 0 || 2 * p > k {
                return Err(shape_err(format!("pooling window {self:?} is invalid")));
            }
            (i + 2 * p)
                .checked_sub(k)
                .map(|v| v / s + 1)
                .ok_or_else(|| shape_err(format!("pooling window {self:?} does not fit a {h}x{w} plane")))
        };
        Ok((
            axis(h, self.kernel.0, self.stride.0, self.padding.0)?,
            axis(w, self.kernel.1, self.stride.1, self.padding.1)?,
        ))
    }
}

const COMMON: &str = r#"
fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
// Output rows / cols whose window covers input coordinate `i` (window o covers [o*s - p, o*s - p + k)).
fn first_out(i: u32, k: u32, s: u32, p: u32) -> u32 {
    let num = i32(i + p) - i32(k) + 1;
    if (num <= 0) { return 0u; }
    return (u32(num) + s - 1u) / s;
}
fn last_out(i: u32, s: u32, p: u32, n: u32) -> i32 { return min(i32((i + p) / s), i32(n) - 1); }
// Adaptive windows (PyTorch's rule): output o covers [o*in/out, ceil((o+1)*in/out)).
fn astart(o: u32, inp: u32, outp: u32) -> u32 { return (o * inp) / outp; }
fn aend(o: u32, inp: u32, outp: u32) -> u32 { return ((o + 1u) * inp + outp - 1u) / outp; }
"#;

/// Vectorized 2×2 stride-2 reduction shared by `upsample_nearest2d_backward(2, 2)` (`scale = 1.0`) and
/// `avg_pool2d(2, 2, 0)` (`scale = 0.25`) for even output widths and exact 2× input heights (`h == 2 * oh`):
/// each thread reads two `vec4<T>`s (8 elements across consecutive rows) and writes one `vec2<T>`.
/// Params: 0 = output `vec2<T>` units (`planes * oh * (ow / 2)`), 1 = `ow / 2`, 2 = `scale` (f32).
const POOL_2X2_S2_VEC4_SRC: &str = "\
@group(0) @binding(1) var<storage, read> X: array<vec4<T>>;\n\
@group(0) @binding(2) var<storage, read_write> Y: array<vec2<T>>;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
        @builtin(local_invocation_index) lid: u32) {\n\
    let i = linear_id(wid, nwg, lid, 256u);\n\
    if (i >= pu(0u)) { return; }\n\
    let w2 = pu(1u);\n\
    let x2 = i % w2;\n\
    let t = i / w2;\n\
    let in0 = t * (2u * w2) + x2;\n\
    let r0 = vec4<f32>(X[in0]);\n\
    let r1 = vec4<f32>(X[in0 + w2]);\n\
    let s = pf(2u);\n\
    let a = ((((0.0 + r0.x) + r0.y) + r1.x) + r1.y) * s;\n\
    let b = ((((0.0 + r0.z) + r0.w) + r1.z) + r1.w) * s;\n\
    Y[i] = vec2<T>(T(a), T(b));\n\
}\n";

/// Vectorized 2×2 stride-2 `avg_pool2d_backward` for exact 2× input dims (`h == 2 * oh`, `w == 2 * ow`) and
/// even `ow`: reads one `vec2<T>` of upstream gradients and writes two `vec4<T>`s scaled by `0.25`.
/// Params: 0 = grad `vec2<T>` units (`planes * oh * (ow / 2)`), 1 = `ow / 2`.
const AVG_2X2_S2_BWD_VEC4_SRC: &str = "\
@group(0) @binding(1) var<storage, read> X: array<vec2<T>>;\n\
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
        @builtin(local_invocation_index) lid: u32) {\n\
    let i = linear_id(wid, nwg, lid, 256u);\n\
    if (i >= pu(0u)) { return; }\n\
    let w2 = pu(1u);\n\
    let x2 = i % w2;\n\
    let t = i / w2;\n\
    let v = vec2<f32>(X[i]) * 0.25;\n\
    let vx = T(0.0 + v.x);\n\
    let vy = T(0.0 + v.y);\n\
    let v4 = vec4<T>(vx, vx, vy, vy);\n\
    let out0 = t * (2u * w2) + x2;\n\
    Y[out0] = v4;\n\
    Y[out0 + w2] = v4;\n\
}\n";

/// Global average pooling (`1×1`) backward fill: broadcasts `0.0 + X[p] / f32(H * W)` across each plane,
/// writing 16-byte `vec4<f32>` chunks when `H * W` is a multiple of 4.
/// Spec: `VEC4` (bool). Params: 0 = output units, 1 = plane units (`H*W/4` or `H*W`), 2 = `f32(H*W)`.
fn gen_adaptive_gap_bwd(spec: &Spec) -> String {
    if spec.get_bool("VEC4") {
        "@group(0) @binding(1) var<storage, read> X: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {\n\
             let i = linear_id(wid, nwg, lid, 256u);\n\
             if (i >= pu(0u)) { return; }\n\
             let p = i / pu(1u);\n\
             Y[i] = vec4<f32>(0.0 + X[p] / pf(2u));\n\
         }\n"
        .into()
    } else {
        "@group(0) @binding(1) var<storage, read> X: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<f32>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {\n\
             let i = linear_id(wid, nwg, lid, 256u);\n\
             if (i >= pu(0u)) { return; }\n\
             let p = i / pu(1u);\n\
             Y[i] = 0.0 + X[p] / pf(2u);\n\
         }\n"
        .into()
    }
}

/// Spec: MODE (0 max forward, 1 max backward, 2 avg forward, 3 avg backward, 4 adaptive forward, 5 adaptive
/// backward, 6 upsample backward). Params: 0 = threads, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = KH, 6 = KW, 7 = SH,
/// 8 = SW, 9 = PH, 10 = PW, 11 = -inf. Upsampling uses KH / KW as the scale factors.
fn gen_pool(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 0);
    if spec.get_bool("PLANE_VEC4") {
        let (kh, kw) = (spec.get_u32("KH", 5), spec.get_u32("KW", 5));
        let cap = spec.get_u32("PLANE_CAP", PLANE_VEC4_SMALL_CAP as u32);
        match mode {
            0 => return gen_pool_idx_plane_vec4(kh, kw, cap),
            1 => return gen_pool_bw_idx_plane_vec4(kh, kw, cap),
            _ => {}
        }
    }
    if spec.get_bool("TILED_VEC4") {
        let (kh, kw) = (spec.get_u32("KH", 5), spec.get_u32("KW", 5));
        match mode {
            0 => return gen_pool_idx_tiled_vec4(kh, kw),
            1 => return gen_pool_bw_idx_tiled_vec4(kh, kw),
            _ => {}
        }
    }
    if spec.get_bool("TILED") {
        match mode {
            0 => return POOL_IDX_TILED_SRC.into(),
            1 => return POOL_BW_IDX_TILED_SRC.into(),
            _ => {}
        }
    }
    let mut s = String::new();
    match mode {
        0 => s.push_str(
            "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
             @group(0) @binding(2) var<storage, read_write> Y: array<T>;\n\
             @group(0) @binding(3) var<storage, read_write> YI: array<u32>;\n",
        ),
        1 => s.push_str(
            "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
             @group(0) @binding(2) var<storage, read> IDX: array<u32>;\n\
             @group(0) @binding(3) var<storage, read_write> Y: array<T>;\n",
        ),
        // Avg pool forward/backward and upsample backward: f32 or f16 (T), summed in f32.
        2 | 3 | 6 => s.push_str(
            "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
             @group(0) @binding(2) var<storage, read_write> Y: array<T>;\n",
        ),
        _ => s.push_str(
            "@group(0) @binding(1) var<storage, read> X: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> Y: array<f32>;\n",
        ),
    }
    s.push_str(COMMON);
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u);\n\
         \x20 let kh = pu(5u); let kw = pu(6u); let sh = pu(7u); let sw = pu(8u); let ph = pu(9u); let pw = pu(10u);\n",
    );
    let body = match mode {
        // Forward modes: one thread per output.
        0 | 2 | 4 => {
            let head = "  let ox = i % ow; let t = i / ow; let oy = t % oh; let p = t / oh;\n\
                        \x20 let base = p * h * w;\n";
            let rest = match mode {
                0 => {
                    "  let y0 = i32(oy * sh) - i32(ph); let x0 = i32(ox * sw) - i32(pw);\n\
                      \x20 var best = pf(11u); var bi = -1;\n\
                      \x20 for (var ky = 0u; ky < kh; ky++) {\n\
                      \x20   let yy = y0 + i32(ky);\n\
                      \x20   if (yy < 0 || yy >= i32(h)) { continue; }\n\
                      \x20   for (var kx = 0u; kx < kw; kx++) {\n\
                      \x20     let xx = x0 + i32(kx);\n\
                      \x20     if (xx < 0 || xx >= i32(w)) { continue; }\n\
                      \x20     let at = yy * i32(w) + xx;\n\
                      \x20     let v = f32(X[base + u32(at)]);\n\
                      \x20     if (v > best || bi < 0 || (isnan1(v) && !isnan1(best))) { best = v; bi = at; }\n\
                      \x20   }\n\
                      \x20 }\n\
                      \x20 Y[i] = T(best);\n\
                      \x20 YI[2u * i] = u32(max(bi, 0)); YI[2u * i + 1u] = 0u;\n"
                }
                2 => {
                    "  let y0 = i32(oy * sh) - i32(ph); let x0 = i32(ox * sw) - i32(pw);\n\
                      \x20 var acc = 0.0;\n\
                      \x20 for (var ky = 0u; ky < kh; ky++) {\n\
                      \x20   let yy = y0 + i32(ky);\n\
                      \x20   if (yy < 0 || yy >= i32(h)) { continue; }\n\
                      \x20   for (var kx = 0u; kx < kw; kx++) {\n\
                      \x20     let xx = x0 + i32(kx);\n\
                      \x20     if (xx < 0 || xx >= i32(w)) { continue; }\n\
                      \x20     acc += f32(X[base + u32(yy) * w + u32(xx)]);\n\
                      \x20   }\n\
                      \x20 }\n\
                      \x20 Y[i] = T(acc * (1.0 / f32(kh * kw)));\n"
                }
                _ => {
                    "  let ya = astart(oy, h, oh); let yb = aend(oy, h, oh);\n\
                      \x20 let xa = astart(ox, w, ow); let xb = aend(ox, w, ow);\n\
                      \x20 var acc = 0.0;\n\
                      \x20 for (var yy = ya; yy < yb; yy++) {\n\
                      \x20   for (var xx = xa; xx < xb; xx++) { acc += X[base + yy * w + xx]; }\n\
                      \x20 }\n\
                      \x20 Y[i] = acc / f32((yb - ya) * (xb - xa));\n"
                }
            };
            format!("{head}{rest}")
        }
        // Backward modes: one thread per input element; X is the output gradient.
        _ => {
            let head = "  let ix = i % w; let t = i / w; let iy = t % h; let p = t / h;\n\
                        \x20 let gbase = p * oh * ow;\n\
                        \x20 var acc = 0.0;\n";
            let rest = match mode {
                1 => {
                    "  let own = iy * w + ix;\n\
                      \x20 let yb = last_out(iy, sh, ph, oh); let xb = last_out(ix, sw, pw, ow);\n\
                      \x20 for (var oy = i32(first_out(iy, kh, sh, ph)); oy <= yb; oy++) {\n\
                      \x20   for (var ox = i32(first_out(ix, kw, sw, pw)); ox <= xb; ox++) {\n\
                      \x20     let o = gbase + u32(oy) * ow + u32(ox);\n\
                      \x20     if (IDX[2u * o] == own) { acc += f32(X[o]); }\n\
                      \x20   }\n\
                      \x20 }\n"
                }
                3 => {
                    // Scale each term before summing, as the CPU does (summing first can overflow).
                    "  let yb = last_out(iy, sh, ph, oh); let xb = last_out(ix, sw, pw, ow);\n\
                      \x20 let inv = 1.0 / f32(kh * kw);\n\
                      \x20 for (var oy = i32(first_out(iy, kh, sh, ph)); oy <= yb; oy++) {\n\
                      \x20   for (var ox = i32(first_out(ix, kw, sw, pw)); ox <= xb; ox++) {\n\
                      \x20     acc += f32(X[gbase + u32(oy) * ow + u32(ox)]) * inv;\n\
                      \x20   }\n\
                      \x20 }\n"
                }
                5 => {
                    "  var ya = (iy * oh) / h; var yb = ya;\n\
                      \x20 loop { if (ya == 0u || aend(ya - 1u, h, oh) <= iy) { break; } ya -= 1u; }\n\
                      \x20 loop { if (yb + 1u >= oh || astart(yb + 1u, h, oh) > iy) { break; } yb += 1u; }\n\
                      \x20 var xa = (ix * ow) / w; var xb = xa;\n\
                      \x20 loop { if (xa == 0u || aend(xa - 1u, w, ow) <= ix) { break; } xa -= 1u; }\n\
                      \x20 loop { if (xb + 1u >= ow || astart(xb + 1u, w, ow) > ix) { break; } xb += 1u; }\n\
                      \x20 for (var oy = ya; oy <= yb; oy++) {\n\
                      \x20   let ny = aend(oy, h, oh) - astart(oy, h, oh);\n\
                      \x20   for (var ox = xa; ox <= xb; ox++) {\n\
                      \x20     let nx = aend(ox, w, ow) - astart(ox, w, ow);\n\
                      \x20     acc += X[gbase + oy * ow + ox] / f32(ny * nx);\n\
                      \x20   }\n\
                      \x20 }\n"
                }
                _ => {
                    "  for (var a = 0u; a < kh; a++) {\n\
                      \x20   for (var b = 0u; b < kw; b++) { acc += f32(X[gbase + (iy * kh + a) * ow + ix * kw + b]); }\n\
                      \x20 }\n"
                }
            };
            format!("{head}{rest}  Y[i] = T(acc);\n")
        }
    };
    s.push_str(&body);
    s.push_str("}\n");
    s
}

static POOL_IDX: KernelDef = KernelDef {
    name: "pool_idx",
    storage: &[Access::Read, Access::ReadWrite, Access::ReadWrite],
    source: Source::Generated(gen_pool),
};
static POOL_BW_IDX: KernelDef = KernelDef {
    name: "pool_bw_idx",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_pool),
};
static POOL: KernelDef = KernelDef {
    name: "pool",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_pool),
};
static POOL_2X2_S2_VEC4: KernelDef = KernelDef {
    name: "pool_2x2_s2_vec4",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(POOL_2X2_S2_VEC4_SRC),
};
static AVG_2X2_S2_BWD_VEC4: KernelDef = KernelDef {
    name: "avg_2x2_s2_bwd_vec4",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(AVG_2X2_S2_BWD_VEC4_SRC),
};
static ADAPTIVE_GAP_BWD: KernelDef = KernelDef {
    name: "adaptive_gap_bwd",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_adaptive_gap_bwd),
};

#[allow(clippy::too_many_arguments)]
fn launch(
    ctx: &GpuContext,
    def: &'static KernelDef,
    mode: u32,
    bufs: &[&wgpu::Buffer],
    threads: usize,
    (h, w, oh, ow): (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    half: bool,
) -> Result<()> {
    if threads == 0 {
        return Ok(());
    }
    let k = ctx.kernel(def, Spec::new().u32("MODE", mode).f16(half))?;
    let p = Params::new()
        .uz(threads)?
        .uz(h)?
        .uz(w)?
        .uz(oh)?
        .uz(ow)?
        .uz(kernel.0)?
        .uz(kernel.1)?
        .uz(stride.0)?
        .uz(stride.1)?
        .uz(pad.0)?
        .uz(pad.1)?
        .f(f32::NEG_INFINITY);
    ctx.dispatch(&k, bufs, &p, ctx.grid_1d(threads as u64, 256)?)
}

/// Computes the 16×16 tile grid for `planes × dim_y × dim_x`, returning `(total_tiles, tiles_x, tiles_y, grid)`
/// when the dispatch fits `u32` and `ctx.grid_1d`.
fn tile_grid_16(
    ctx: &GpuContext,
    planes: usize,
    dim_y: usize,
    dim_x: usize,
) -> Option<(usize, usize, usize, [u32; 3])> {
    let (tiles_x, tiles_y) = (dim_x.div_ceil(16), dim_y.div_ceil(16));
    let total_tiles = prod(&[planes, tiles_y, tiles_x]).ok()?;
    if total_tiles > u32::MAX as usize {
        return None;
    }
    let threads = (total_tiles as u64).checked_mul(256)?;
    let grid = ctx.grid_1d(threads, 256).ok()?;
    Some((total_tiles, tiles_x, tiles_y, grid))
}

/// Computes the `8×16` `vec4` tile grid (`32×16` pixels per 128-thread workgroup) for `planes × h × w4`.
fn tile_grid_vec4(
    ctx: &GpuContext,
    planes: usize,
    h: usize,
    w4: usize,
) -> Option<(usize, usize, usize, [u32; 3])> {
    let (tiles_x, tiles_y) = (w4.div_ceil(8), h.div_ceil(16));
    let total_tiles = prod(&[planes, tiles_y, tiles_x]).ok()?;
    if total_tiles > u32::MAX as usize {
        return None;
    }
    let threads = (total_tiles as u64).checked_mul(128)?;
    let grid = ctx.grid_1d(threads, 128).ok()?;
    Some((total_tiles, tiles_x, tiles_y, grid))
}

/// Max pooling of `planes` dense `h × w` planes of `dtype` (f32 or f16) into `y`, with the flat in-plane index
/// `h * W + w` of each chosen element written as `i64` to `indices` (first maximum in window order wins; NaN
/// wins). Returns `(OH, OW)`.
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_indices(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    win: &Pool2d,
    dtype: DType,
) -> Result<(usize, usize)> {
    check_dtype(ctx, dtype, "max_pool2d")?;
    let (oh, ow) = win.output(h, w)?;
    let in_elems = prod(&[planes, h, w])?;
    let n = prod(&[planes, oh, ow])?;
    check_buf(x, in_elems, dtype, "max_pool2d input")?;
    check_buf(y, n, dtype, "max_pool2d output")?;
    check_buf(indices, prod(&[n, 2])?, DType::F32, "max_pool2d indices")?;
    if n == 0 {
        return Ok((oh, ow));
    }
    if in_elems > u32::MAX as usize || n > i32::MAX as usize {
        return Err(GpuError::Limit("max_pool2d: tensor exceeds u32 indexing".into()));
    }
    if h.checked_mul(w).is_none_or(|hw| hw > i32::MAX as usize) {
        return Err(GpuError::Limit("max_pool2d: plane size exceeds i32 indexing".into()));
    }
    let half = dtype == DType::F16;
    if win.stride == (1, 1)
        && is_same_vec4_k(win.kernel.0)
        && is_same_vec4_k(win.kernel.1)
        && oh == h
        && ow == w
        && w.is_multiple_of(4)
        && h <= i32::MAX as usize - 64
        && w <= i32::MAX as usize - 64
    {
        let w4 = w / 4;
        let hw4 = h * w4;
        if let Some((cap, wg)) = plane_vec4_plan(ctx, hw4, 48)
            && planes <= u32::MAX as usize
            && let Some(threads) = (planes as u64).checked_mul(wg as u64)
            && let Ok(grid) = ctx.grid_1d(threads, wg)
        {
            let spec = Spec::new()
                .u32("MODE", 0)
                .bool("PLANE_VEC4", true)
                .u32("PLANE_CAP", cap)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&POOL_IDX, spec)?;
            let p = Params::new().uz(planes)?.uz(h)?.uz(w4)?.uz(hw4)?.f(f32::NEG_INFINITY);
            ctx.dispatch(&k, &[x, y, indices], &p, grid)?;
            return Ok((oh, ow));
        }
        if let Some((total_tiles, tiles_x, tiles_y, grid)) = tile_grid_vec4(ctx, planes, h, w4) {
            let spec = Spec::new()
                .u32("MODE", 0)
                .bool("TILED_VEC4", true)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&POOL_IDX, spec)?;
            let p = Params::new()
                .uz(total_tiles)?
                .uz(h)?
                .uz(w4)?
                .f(f32::NEG_INFINITY)
                .uz(tiles_x)?
                .uz(tiles_y)?;
            ctx.dispatch(&k, &[x, y, indices], &p, grid)?;
            return Ok((oh, ow));
        }
    }
    if win.stride == (1, 1)
        && (2..=TILED_MAX_K).contains(&win.kernel.0)
        && (2..=TILED_MAX_K).contains(&win.kernel.1)
        && h <= i32::MAX as usize - 64
        && w <= i32::MAX as usize - 64
        && let Some((total_tiles, tiles_x, tiles_y, grid)) = tile_grid_16(ctx, planes, oh, ow)
    {
        let spec = Spec::new()
            .u32("MODE", 0)
            .bool("TILED", true)
            .u32("KH", win.kernel.0 as u32)
            .u32("KW", win.kernel.1 as u32)
            .f16(half);
        let k = ctx.kernel(&POOL_IDX, spec)?;
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
        ctx.dispatch(&k, &[x, y, indices], &p, grid)?;
        return Ok((oh, ow));
    }
    let dims = (h, w, oh, ow);
    launch(ctx, &POOL_IDX, 0, &[x, y, indices], n, dims, win.kernel, win.stride, win.padding, half)?;
    Ok((oh, ow))
}

/// Gradient of [`max_pool2d_indices`]: `gx[p, i] = Σ g[p, o]` over the outputs `o` whose index is `i`. `g` and
/// `gx` are stored as `dtype` (f32 or f16; the sum is f32).
pub fn max_pool2d_backward(
    ctx: &GpuContext,
    g: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    gx: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    win: &Pool2d,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "max_pool2d_backward")?;
    let (oh, ow) = win.output(h, w)?;
    let n = prod(&[planes, oh, ow])?;
    check_buf(g, n, dtype, "max_pool2d_backward grad")?;
    check_buf(indices, prod(&[n, 2])?, DType::F32, "max_pool2d_backward indices")?;
    let total = prod(&[planes, h, w])?;
    check_buf(gx, total, dtype, "max_pool2d_backward output")?;
    if total == 0 {
        return Ok(());
    }
    if total > u32::MAX as usize || n > i32::MAX as usize {
        return Err(GpuError::Limit("max_pool2d_backward: tensor exceeds u32 indexing".into()));
    }
    if h.checked_mul(w).is_none_or(|hw| hw > i32::MAX as usize) {
        return Err(GpuError::Limit("max_pool2d_backward: plane size exceeds i32 indexing".into()));
    }
    let half = dtype == DType::F16;
    if win.stride == (1, 1)
        && is_same_vec4_k(win.kernel.0)
        && is_same_vec4_k(win.kernel.1)
        && oh == h
        && ow == w
        && w.is_multiple_of(4)
        && h <= i32::MAX as usize - 64
        && w <= i32::MAX as usize - 64
    {
        let w4 = w / 4;
        let hw4 = h * w4;
        if let Some((cap, wg)) = plane_vec4_plan(ctx, hw4, 32)
            && planes <= u32::MAX as usize
            && let Some(threads) = (planes as u64).checked_mul(wg as u64)
            && let Ok(grid) = ctx.grid_1d(threads, wg)
        {
            let spec = Spec::new()
                .u32("MODE", 1)
                .bool("PLANE_VEC4", true)
                .u32("PLANE_CAP", cap)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&POOL_BW_IDX, spec)?;
            let p = Params::new().uz(planes)?.uz(h)?.uz(w4)?.uz(hw4)?;
            return ctx.dispatch(&k, &[g, indices, gx], &p, grid);
        }
        if let Some((total_tiles, tiles_x, tiles_y, grid)) = tile_grid_vec4(ctx, planes, h, w4) {
            let spec = Spec::new()
                .u32("MODE", 1)
                .bool("TILED_VEC4", true)
                .u32("KH", win.kernel.0 as u32)
                .u32("KW", win.kernel.1 as u32)
                .f16(half);
            let k = ctx.kernel(&POOL_BW_IDX, spec)?;
            let p = Params::new().uz(total_tiles)?.uz(h)?.uz(w4)?.uz(tiles_x)?.uz(tiles_y)?;
            return ctx.dispatch(&k, &[g, indices, gx], &p, grid);
        }
    }
    if win.stride == (1, 1)
        && (2..=TILED_MAX_K).contains(&win.kernel.0)
        && (2..=TILED_MAX_K).contains(&win.kernel.1)
        && h <= i32::MAX as usize - 64
        && w <= i32::MAX as usize - 64
        && let Some((total_tiles, tiles_x, tiles_y, grid)) = tile_grid_16(ctx, planes, h, w)
    {
        let spec = Spec::new()
            .u32("MODE", 1)
            .bool("TILED", true)
            .u32("KH", win.kernel.0 as u32)
            .u32("KW", win.kernel.1 as u32)
            .f16(half);
        let k = ctx.kernel(&POOL_BW_IDX, spec)?;
        let p = Params::new()
            .uz(total_tiles)?
            .uz(h)?
            .uz(w)?
            .uz(oh)?
            .uz(ow)?
            .uz(win.padding.0)?
            .uz(win.padding.1)?
            .uz(tiles_x)?
            .uz(tiles_y)?;
        return ctx.dispatch(&k, &[g, indices, gx], &p, grid);
    }
    launch(
        ctx,
        &POOL_BW_IDX,
        1,
        &[g, indices, gx],
        total,
        (h, w, oh, ow),
        win.kernel,
        win.stride,
        win.padding,
        half,
    )
}

/// Average pooling (padding zeros count toward the divisor `KH · KW`), stored as `dtype` (f32 or f16;
/// accumulated in f32). Returns `(OH, OW)`.
pub fn avg_pool2d(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    win: &Pool2d,
    dtype: DType,
) -> Result<(usize, usize)> {
    check_dtype(ctx, dtype, "avg_pool2d")?;
    let (oh, ow) = win.output(h, w)?;
    let in_elems = prod(&[planes, h, w])?;
    let n = prod(&[planes, oh, ow])?;
    check_buf(x, in_elems, dtype, "avg_pool2d input")?;
    check_buf(y, n, dtype, "avg_pool2d output")?;
    if n == 0 {
        return Ok((oh, ow));
    }
    let half = dtype == DType::F16;
    if win.kernel == (2, 2)
        && win.stride == (2, 2)
        && win.padding == (0, 0)
        && h.is_multiple_of(2)
        && w.is_multiple_of(4)
        && in_elems <= u32::MAX as usize
    {
        let w2 = ow / 2;
        let units = planes * oh * w2;
        let k = ctx.kernel(&POOL_2X2_S2_VEC4, Spec::new().f16(half))?;
        let p = Params::new().uz(units)?.uz(w2)?.f(0.25);
        ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(units as u64, 256)?)?;
        return Ok((oh, ow));
    }
    launch(ctx, &POOL, 2, &[x, y], n, (h, w, oh, ow), win.kernel, win.stride, win.padding, half)?;
    Ok((oh, ow))
}

/// Gradient of [`avg_pool2d`] into `gx` (`planes × h × w`), stored as `dtype` (f32 or f16; accumulated in f32).
pub fn avg_pool2d_backward(
    ctx: &GpuContext,
    g: &wgpu::Buffer,
    gx: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    win: &Pool2d,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "avg_pool2d_backward")?;
    let (oh, ow) = win.output(h, w)?;
    check_buf(g, prod(&[planes, oh, ow])?, dtype, "avg_pool2d_backward grad")?;
    let total = prod(&[planes, h, w])?;
    check_buf(gx, total, dtype, "avg_pool2d_backward output")?;
    if total == 0 {
        return Ok(());
    }
    let half = dtype == DType::F16;
    if win.kernel == (2, 2)
        && win.stride == (2, 2)
        && win.padding == (0, 0)
        && h == 2 * oh
        && w == 2 * ow
        && ow.is_multiple_of(2)
        && total <= u32::MAX as usize
    {
        let w2 = ow / 2;
        let units = planes * oh * w2;
        let k = ctx.kernel(&AVG_2X2_S2_BWD_VEC4, Spec::new().f16(half))?;
        let p = Params::new().uz(units)?.uz(w2)?;
        return ctx.dispatch(&k, &[g, gx], &p, ctx.grid_1d(units as u64, 256)?);
    }
    launch(ctx, &POOL, 3, &[g, gx], total, (h, w, oh, ow), win.kernel, win.stride, win.padding, half)
}

/// The shader computes adaptive window bounds as `(o + 1) · in + out - 1` in `u32`; reject sizes where that wraps.
fn check_adaptive((h, w): (usize, usize), (oh, ow): (usize, usize)) -> Result<()> {
    let fits = |i: usize, o: usize| (o as u64).saturating_mul(i as u64 + 1) <= u32::MAX as u64;
    if !fits(h, oh) || !fits(w, ow) {
        return Err(shape_err(format!(
            "adaptive_avg_pool2d: {h}x{w} -> {oh}x{ow} overflows the window arithmetic (input · output must stay below 2^32)"
        )));
    }
    Ok(())
}

/// Adaptive average pooling of `planes × h × w` to `oh × ow` (PyTorch's window rule). Fails when
/// `(output + 1) · input` would exceed `u32` on either axis.
pub fn adaptive_avg_pool2d(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    (oh, ow): (usize, usize),
) -> Result<()> {
    if h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Err(shape_err("adaptive_avg_pool2d: empty plane"));
    }
    check_adaptive((h, w), (oh, ow))?;
    let in_elems = prod(&[planes, h, w])?;
    let n = prod(&[planes, oh, ow])?;
    check_buf(x, in_elems, DType::F32, "adaptive_avg_pool2d input")?;
    check_buf(y, n, DType::F32, "adaptive_avg_pool2d output")?;
    if n == 0 {
        return Ok(());
    }
    if oh == 1 && ow == 1 && in_elems <= u32::MAX as usize {
        return reduce::reduce_split(ctx, ReduceOp::Mean, x, y, planes, h * w, 1, DType::F32);
    }
    launch(ctx, &POOL, 4, &[x, y], n, (h, w, oh, ow), (1, 1), (1, 1), (0, 0), false)
}

/// Gradient of [`adaptive_avg_pool2d`] into `gx` (`planes × h × w`).
pub fn adaptive_avg_pool2d_backward(
    ctx: &GpuContext,
    g: &wgpu::Buffer,
    gx: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    (oh, ow): (usize, usize),
) -> Result<()> {
    if h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Err(shape_err("adaptive_avg_pool2d_backward: empty plane"));
    }
    check_adaptive((h, w), (oh, ow))?;
    check_buf(g, prod(&[planes, oh, ow])?, DType::F32, "adaptive_avg_pool2d_backward grad")?;
    let total = prod(&[planes, h, w])?;
    check_buf(gx, total, DType::F32, "adaptive_avg_pool2d_backward output")?;
    if total == 0 {
        return Ok(());
    }
    if oh == 1 && ow == 1 && total <= u32::MAX as usize {
        let hw = h * w;
        let vec4 = hw.is_multiple_of(4);
        let units_hw = if vec4 { hw / 4 } else { hw };
        let units = planes * units_hw;
        let k = ctx.kernel(&ADAPTIVE_GAP_BWD, Spec::new().bool("VEC4", vec4))?;
        let p = Params::new().uz(units)?.uz(units_hw)?.f(hw as f32);
        return ctx.dispatch(&k, &[g, gx], &p, ctx.grid_1d(units as u64, 256)?);
    }
    launch(ctx, &POOL, 5, &[g, gx], total, (h, w, oh, ow), (1, 1), (1, 1), (0, 0), false)
}

/// Gradient of nearest upsampling by integer factors `(sh, sw)`: each input element sums its `sh × sw` block
/// of `g` (`planes × h·sh × w·sw`) into `gx` (`planes × h × w`), both stored as `dtype` (f32 or f16; the sum
/// is f32).
pub fn upsample_nearest2d_backward(
    ctx: &GpuContext,
    g: &wgpu::Buffer,
    gx: &wgpu::Buffer,
    (planes, h, w): (usize, usize, usize),
    (sh, sw): (usize, usize),
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "upsample_nearest2d_backward")?;
    let (oh, ow) = (prod(&[h, sh])?, prod(&[w, sw])?);
    let in_elems = prod(&[planes, oh, ow])?;
    check_buf(g, in_elems, dtype, "upsample_backward grad")?;
    let total = prod(&[planes, h, w])?;
    check_buf(gx, total, dtype, "upsample_backward output")?;
    if total == 0 {
        return Ok(());
    }
    let half = dtype == DType::F16;
    if sh == 2 && sw == 2 && w.is_multiple_of(2) && in_elems <= u32::MAX as usize {
        let w2 = w / 2;
        let units = planes * h * w2;
        let k = ctx.kernel(&POOL_2X2_S2_VEC4, Spec::new().f16(half))?;
        let p = Params::new().uz(units)?.uz(w2)?.f(1.0);
        return ctx.dispatch(&k, &[g, gx], &p, ctx.grid_1d(units as u64, 256)?);
    }
    launch(ctx, &POOL, 6, &[g, gx], total, (h, w, oh, ow), (sh, sw), (1, 1), (0, 0), half)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] =
    &[&POOL_IDX, &POOL_BW_IDX, &POOL, &POOL_2X2_S2_VEC4, &AVG_2X2_S2_BWD_VEC4, &ADAPTIVE_GAP_BWD];
