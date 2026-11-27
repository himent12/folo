//! Detector pre- and post-processing: letterboxing `u8` RGB images into the network input (f32, or f16 for
//! half-precision inference), fused packed-`u32` `u8 -> f32/f16` scaling and RGB HWC -> CHW normalization, and
//! decoding the detection head (f32 or f16, decoded in f32) into a compact list of above-threshold candidates, so
//! that prediction and training upload raw `u8` pixels and download only what NMS needs.

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
use crate::pool::GpuBuffer;

static LETTERBOX_PREFILTER: KernelDef = KernelDef {
    name: "letterbox_prefilter",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/letterbox_prefilter.wgsl")),
};
static LETTERBOX: KernelDef = KernelDef {
    name: "letterbox",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_letterbox),
};

const LETTERBOX_MAIN_SCALAR: &str = r#"
@group(0) @binding(1) var<storage, read> SRC: array<u32>;
@group(0) @binding(2) var<storage, read> TAPS: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let ow = pu(0u);
    let oh = pu(1u);
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= ow * oh) { return; }
    let x = i % ow;
    let y = i / ow;
    let plane = ow * oh;
    let base = pu(12u) + i;
    let scale = pf(11u);
    let px = pu(4u);
    let py = pu(5u);
    let rw = pu(6u);
    let rh = pu(7u);
    if (x < px || x >= px + rw || y < py || y >= py + rh) {
        let v = T(f32(pu(10u)) * scale);
        Y[base] = v;
        Y[base + plane] = v;
        Y[base + 2u * plane] = v;
        return;
    }
    let dx = x - px;
    let dy = y - py;
    let tx = pu(13u);
    let ty = tx + 3u * rw;
    let ya = TAPS[ty + dy];
    let yb = TAPS[ty + rh + dy];
    let wy = TAPS[ty + 2u * rh + dy];
    let v = sample_bilinear(dx, rw, tx, ya, yb, wy, 2048u - wy);
    Y[base] = T(f32(v.x) * scale);
    Y[base + plane] = T(f32(v.y) * scale);
    Y[base + 2u * plane] = T(f32(v.z) * scale);
}
"#;

const LETTERBOX_MAIN_VEC4: &str = r#"
@group(0) @binding(1) var<storage, read> SRC: array<u32>;
@group(0) @binding(2) var<storage, read> TAPS: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<T>>;

fn sample_lane(x: u32, px: u32, rw: u32, tx: u32, ya: u32, yb: u32, wy: u32, iwy: u32, pad_v: f32, scale: f32) -> vec3<f32> {
    if (x < px || x >= px + rw) {
        return vec3<f32>(pad_v);
    }
    let v = sample_bilinear(x - px, rw, tx, ya, yb, wy, iwy);
    return vec3<f32>(v) * scale;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let ow4 = pu(0u) >> 2u;
    let oh = pu(1u);
    let plane4 = ow4 * oh;
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= plane4) { return; }
    let x4 = i % ow4;
    let y = i / ow4;
    let x0 = x4 << 2u;
    let base4 = pu(12u) + i;
    let scale = pf(11u);
    let pad_v = f32(pu(10u)) * scale;
    let px = pu(4u);
    let py = pu(5u);
    let rw = pu(6u);
    let rh = pu(7u);
    if (y < py || y >= py + rh || x0 + 4u <= px || x0 >= px + rw) {
        let pv = vec4<T>(T(pad_v));
        Y[base4] = pv;
        Y[base4 + plane4] = pv;
        Y[base4 + 2u * plane4] = pv;
        return;
    }
    let dy = y - py;
    let tx = pu(13u);
    let ty = tx + 3u * rw;
    let ya = TAPS[ty + dy];
    let yb = TAPS[ty + rh + dy];
    let wy = TAPS[ty + 2u * rh + dy];
    let iwy = 2048u - wy;
    let c0 = sample_lane(x0, px, rw, tx, ya, yb, wy, iwy, pad_v, scale);
    let c1 = sample_lane(x0 + 1u, px, rw, tx, ya, yb, wy, iwy, pad_v, scale);
    let c2 = sample_lane(x0 + 2u, px, rw, tx, ya, yb, wy, iwy, pad_v, scale);
    let c3 = sample_lane(x0 + 3u, px, rw, tx, ya, yb, wy, iwy, pad_v, scale);
    Y[base4] = vec4<T>(vec4<f32>(c0.x, c1.x, c2.x, c3.x));
    Y[base4 + plane4] = vec4<T>(vec4<f32>(c0.y, c1.y, c2.y, c3.y));
    Y[base4 + 2u * plane4] = vec4<T>(vec4<f32>(c0.z, c1.z, c2.z, c3.z));
}
"#;

fn gen_letterbox(spec: &Spec) -> String {
    let main = if spec.get_bool("VEC4") { LETTERBOX_MAIN_VEC4 } else { LETTERBOX_MAIN_SCALAR };
    format!("{main}{}", include_str!("../shaders/letterbox.wgsl"))
}
static DETECT: KernelDef = KernelDef {
    name: "detect",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(|spec| {
        let vk = super::detect::v_k();
        let r_cap = spec.get_u32("R", 0).max(1);
        format!(
            "const V_K: f32 = {vk:?}f;\nconst R_CAP: u32 = {r_cap}u;\n{}{}",
            include_str!("../shaders/detect_common.wgsl"),
            include_str!("../shaders/detect.wgsl")
        )
    }),
};
static U8_SCALE: KernelDef = KernelDef {
    name: "u8_to_float_scale",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_u8_scale),
};
static RGB8_CHW_NORM: KernelDef = KernelDef {
    name: "rgb8_hwc_to_chw_norm",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_rgb8_chw_norm),
};

/// Fused packed-`u32` `u8 -> T` (`f32` or `f16`) scaling: `Y[out_off + i] = T(f32(src_byte[src_off + i]) * scale)`.
///
/// Three modes controlled by `Spec`:
/// - `VEC4=true`: 4-aligned source, output, and length (`SRC: array<u32>`, `Y: array<vec4<T>>`), 1 packed load +
///   1 vector store per 4 elements.
/// - `VEC4=false, ALIGNED_SRC=true`: 4-aligned source with odd length or output offset; still unpacks 1 `u32`
///   word per 4 elements and bounds-checks scalar writes so buffer slack is never touched.
/// - `VEC4=false, ALIGNED_SRC=false`: arbitrary byte source offset and length, 1 element per thread.
fn gen_u8_scale(spec: &Spec) -> String {
    if spec.get_bool("VEC4") {
        "@group(0) @binding(1) var<storage, read> SRC: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let w = SRC[pu(2u) + i];\n\
         \x20 let v = vec4<f32>(\n\
         \x20   f32(w & 0xffu),\n\
         \x20   f32((w >> 8u) & 0xffu),\n\
         \x20   f32((w >> 16u) & 0xffu),\n\
         \x20   f32(w >> 24u),\n\
         \x20 ) * pf(1u);\n\
         \x20 Y[pu(3u) + i] = vec4<T>(v);\n\
         }\n"
        .into()
    } else if spec.get_bool("ALIGNED_SRC") {
        "@group(0) @binding(1) var<storage, read> SRC: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<T>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(4u)) { return; }\n\
         \x20 let base = i * 4u;\n\
         \x20 let rem = pu(0u) - base;\n\
         \x20 let w = SRC[pu(2u) + i];\n\
         \x20 let scale = pf(1u);\n\
         \x20 let o = pu(3u) + base;\n\
         \x20 Y[o] = T(f32(w & 0xffu) * scale);\n\
         \x20 if (rem > 1u) { Y[o + 1u] = T(f32((w >> 8u) & 0xffu) * scale); }\n\
         \x20 if (rem > 2u) { Y[o + 2u] = T(f32((w >> 16u) & 0xffu) * scale); }\n\
         \x20 if (rem > 3u) { Y[o + 3u] = T(f32(w >> 24u) * scale); }\n\
         }\n"
        .into()
    } else {
        "@group(0) @binding(1) var<storage, read> SRC: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<T>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let byte_off = pu(2u) + i;\n\
         \x20 let b = (SRC[byte_off >> 2u] >> ((byte_off & 3u) * 8u)) & 0xffu;\n\
         \x20 Y[pu(3u) + i] = T(f32(b) * pf(1u));\n\
         }\n"
        .into()
    }
}

/// Fused packed-`u32` interleaved RGB `u8` `[B, H, W, 3]` -> planar NCHW `[B, 3, H, W]` `T` (`f32` or `f16`)
/// multiplied by `scale` (`1 / 255` for network input).
///
/// - `VEC4=true` (`(H * W) % 4 == 0`, 4-aligned source and output offsets): reads 3 packed `u32` words (4 RGB
///   pixels = 12 bytes) per thread and writes 3 coalesced `vec4<T>` planes (R, G, B).
/// - `VEC4=false`: arbitrary `H * W` and byte/element offsets, 1 pixel per thread touching only the 1 or 2
///   source words containing its 3 RGB bytes.
fn gen_rgb8_chw_norm(spec: &Spec) -> String {
    if spec.get_bool("VEC4") {
        "@group(0) @binding(1) var<storage, read> SRC: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<T>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let plane4 = pu(1u);\n\
         \x20 let img = i / plane4;\n\
         \x20 let p4 = i - img * plane4;\n\
         \x20 let sw = pu(3u) + i * 3u;\n\
         \x20 let w0 = SRC[sw];\n\
         \x20 let w1 = SRC[sw + 1u];\n\
         \x20 let w2 = SRC[sw + 2u];\n\
         \x20 let scale = pf(2u);\n\
         \x20 let r = vec4<f32>(\n\
         \x20   f32(w0 & 0xffu),\n\
         \x20   f32(w0 >> 24u),\n\
         \x20   f32((w1 >> 16u) & 0xffu),\n\
         \x20   f32((w2 >> 8u) & 0xffu),\n\
         \x20 ) * scale;\n\
         \x20 let g = vec4<f32>(\n\
         \x20   f32((w0 >> 8u) & 0xffu),\n\
         \x20   f32(w1 & 0xffu),\n\
         \x20   f32(w1 >> 24u),\n\
         \x20   f32((w2 >> 16u) & 0xffu),\n\
         \x20 ) * scale;\n\
         \x20 let b = vec4<f32>(\n\
         \x20   f32((w0 >> 16u) & 0xffu),\n\
         \x20   f32((w1 >> 8u) & 0xffu),\n\
         \x20   f32(w2 & 0xffu),\n\
         \x20   f32(w2 >> 24u),\n\
         \x20 ) * scale;\n\
         \x20 let ob = pu(4u) + img * 3u * plane4 + p4;\n\
         \x20 Y[ob] = vec4<T>(r);\n\
         \x20 Y[ob + plane4] = vec4<T>(g);\n\
         \x20 Y[ob + 2u * plane4] = vec4<T>(b);\n\
         }\n"
        .into()
    } else {
        "@group(0) @binding(1) var<storage, read> SRC: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<T>;\n\
         fn rgb_at(i: u32) -> vec3<u32> {\n\
         \x20 let wi = i >> 2u;\n\
         \x20 let w0 = SRC[wi];\n\
         \x20 let s = (i & 3u) * 8u;\n\
         \x20 if (s == 0u) { return vec3<u32>(w0 & 0xffu, (w0 >> 8u) & 0xffu, (w0 >> 16u) & 0xffu); }\n\
         \x20 if (s == 8u) { return vec3<u32>((w0 >> 8u) & 0xffu, (w0 >> 16u) & 0xffu, w0 >> 24u); }\n\
         \x20 let w1 = SRC[wi + 1u];\n\
         \x20 if (s == 16u) { return vec3<u32>((w0 >> 16u) & 0xffu, w0 >> 24u, w1 & 0xffu); }\n\
         \x20 return vec3<u32>(w0 >> 24u, w1 & 0xffu, (w1 >> 8u) & 0xffu);\n\
         }\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let hw = pu(1u);\n\
         \x20 let img = i / hw;\n\
         \x20 let p = i - img * hw;\n\
         \x20 let rgb = rgb_at(pu(3u) + i * 3u);\n\
         \x20 let scale = pf(2u);\n\
         \x20 let ob = pu(4u) + img * 3u * hw + p;\n\
         \x20 Y[ob] = T(f32(rgb.x) * scale);\n\
         \x20 Y[ob + hw] = T(f32(rgb.y) * scale);\n\
         \x20 Y[ob + 2u * hw] = T(f32(rgb.z) * scale);\n\
         }\n"
        .into()
    }
}

/// One image of [`letterbox_rgb8`]: where it goes on the canvas and how it is sampled (folo-data's
/// `ResizePlan`: box prefilter `prefilter`, then the bilinear taps at word `taps` of the taps buffer: `x0, x1,
/// wx` with `resized[0]` entries each, then `y0, y1, wy` with `resized[1]` each, weights in 1/2048).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterboxDesc {
    /// Canvas `[width, height]`.
    pub out: [usize; 2],
    /// Source `[width, height]`; the source is `height * width * 3` bytes, row-major RGB, from byte
    /// `src_offset` of its buffer.
    pub src: [usize; 2],
    /// Byte offset of the source image in its buffer (several images can share one upload).
    pub src_offset: usize,
    /// Top-left corner `[x, y]` of the resized image on the canvas.
    pub pad: [usize; 2],
    /// Resized `[width, height]`.
    pub resized: [usize; 2],
    /// Box-prefilter factors `[fx, fy]`.
    pub prefilter: [usize; 2],
    /// Grey level of the padding.
    pub pad_value: u8,
    /// Factor applied to every output value (`1 / 255` for network input).
    pub scale: f32,
    /// Element offset of this image's `[3, OH, OW]` block in the output.
    pub out_offset: usize,
    /// Word offset of this image's taps.
    pub taps: usize,
}

/// Letterbox one image (see [`LetterboxDesc`]) into `y` of `dtype`. In f32 bit-exact with folo-data's CPU
/// resize followed by `value as f32 * scale`; in f16 converted from that f32 value (within 1 binary16 ULP under
/// WGSL's floating-point conversion rules).
pub fn letterbox_rgb8(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    taps: &wgpu::Buffer,
    y: &wgpu::Buffer,
    d: &LetterboxDesc,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "letterbox")?;
    let [ow, oh] = d.out;
    let [sw, sh] = d.src;
    let [rw, rh] = d.resized;
    let [fx, fy] = d.prefilter;
    let pad_x_end = d.pad[0].checked_add(rw);
    let pad_y_end = d.pad[1].checked_add(rh);
    if [ow, oh, sw, sh, rw, rh, fx, fy].contains(&0)
        || pad_x_end.is_none_or(|e| e > ow)
        || pad_y_end.is_none_or(|e| e > oh)
    {
        return Err(shape_err(format!("letterbox: invalid geometry {d:?}")));
    }
    // The prefilter sums a block (plus the rounding term) in u32; the plain bilinear sums stay below 2^31.
    if !prefilter_fits(fx, fy) {
        return Err(GpuError::Limit(format!("letterbox: a {fx}x{fy} prefilter block is too large")));
    }
    // Prefiltered taps must index inside the prefiltered image (checked on the host when built).
    // All shader byte/word/element addresses are u32, so reject any extent that overflows usize or u32.
    let src_bytes = prod(&[sw, sh, 3])?;
    let src_end = d
        .src_offset
        .checked_add(src_bytes)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("letterbox: source range exceeds u32".into()))?;
    let taps_words = rw
        .checked_add(rh)
        .and_then(|s| s.checked_mul(3))
        .and_then(|w| d.taps.checked_add(w))
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("letterbox: taps range exceeds u32".into()))?;
    let n = prod(&[ow, oh])?;
    let out_elems = prod(&[n, 3])?;
    let out_end = d
        .out_offset
        .checked_add(out_elems)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("letterbox: output range exceeds u32".into()))?;
    check_buf(src, src_end.div_ceil(4), DType::F32, "letterbox source")?;
    check_buf(taps, taps_words, DType::F32, "letterbox taps")?;
    check_buf(y, out_end, dtype, "letterbox output")?;
    let prefilter = fx > 1 || fy > 1;
    let pw = sw.div_ceil(fx);
    let ph = sh.div_ceil(fy);
    let pn = pw.checked_mul(ph);
    // Run the 2-pass box prefilter when blocks have >= 4 source pixels (so avoiding 4-corner recomputation
    // outweighs the extra dispatch and packed intermediate traffic), when 2-pass computes fewer cells than
    // 4 * rw * rh, and when the compact packed-u32 intermediate fits within the storage binding limit.
    let use_two_pass = prefilter
        && fx.saturating_mul(fy) >= 4
        && pn.is_some_and(|cells| {
            cells <= u32::MAX as usize
                && (cells as u64).saturating_mul(4) <= ctx.caps().max_storage_buffer_binding_size
                && rw.saturating_mul(rh).saturating_mul(4) > cells
        });
    let pre_buf = if use_two_pass {
        let cells = pn.expect("checked above");
        let buf = ctx.alloc((cells as u64) * 4)?;
        let kp = ctx.kernel(&LETTERBOX_PREFILTER, Spec::new())?;
        let pp = Params::new().uz(cells)?.uz(pw)?.uz(sw)?.uz(sh)?.uz(fx)?.uz(fy)?.uz(d.src_offset)?;
        ctx.dispatch(&kp, &[src, buf.raw()], &pp, ctx.grid_1d(cells as u64, 256)?)?;
        Some(buf)
    } else {
        None
    };
    // Use 4-pixel-per-thread coalesced `vec4<T>` stores when reading aligned packed-u32 prefiltered pixels or
    // same-size non-prefiltered frames (low register pressure); fractional resizes on raw RGB bytes keep 1
    // pixel per thread so unaligned `rgb_at` loads maximize occupancy.
    let vec4 = ow.is_multiple_of(4)
        && d.out_offset.is_multiple_of(4)
        && (use_two_pass || (!prefilter && rw == sw && rh == sh));
    let (sample_src, sample_w, sample_h, sample_src_off) = match &pre_buf {
        Some(buf) => (buf.raw(), pw, ph, 0usize),
        None => (src, sw, sh, d.src_offset),
    };
    let k = ctx.kernel(
        &LETTERBOX,
        Spec::new()
            .bool("PREFILTER", prefilter && !use_two_pass)
            .bool("PACKED_SRC", use_two_pass)
            .bool("VEC4", vec4)
            .f16(dtype == DType::F16),
    )?;
    let (out_off_param, threads) = if vec4 { (d.out_offset / 4, n / 4) } else { (d.out_offset, n) };
    let p = Params::new()
        .uz(ow)?
        .uz(oh)?
        .uz(sample_w)?
        .uz(sample_h)?
        .uz(d.pad[0])?
        .uz(d.pad[1])?
        .uz(rw)?
        .uz(rh)?
        .uz(fx)?
        .uz(fy)?
        .u(d.pad_value as u32)
        .f(d.scale)
        .uz(out_off_param)?
        .uz(d.taps)?
        .uz(sample_src_off)?;
    ctx.dispatch(&k, &[sample_src, taps, y], &p, ctx.grid_1d(threads as u64, 256)?)
}

/// Whether the shader can sum and round an `fx x fy` prefilter block in u32: `255 · n + n / 2` for
/// `n = fx · fy` (the CPU sums in u64).
pub fn prefilter_fits(fx: usize, fy: usize) -> bool {
    (fx as u64)
        .checked_mul(fy as u64)
        .and_then(|n| n.checked_mul(255).and_then(|s| s.checked_add(n / 2)))
        .is_some_and(|v| v <= u32::MAX as u64)
}

/// Words per candidate record of [`detect_candidates`].
pub const RECORD_WORDS: usize = 7;

/// Decode `box_raw` (`[B, 4·rm, A]`), `cls_raw` (`[B, nc, A]`, both of `dtype`: f32 or f16, decoded in f32)
/// and `anchors` (f32 `[3, A]`: x, y in grid units, stride) into `out` (u32 `[B, 1 + 7·A]`): per image a
/// count, then that many records `(anchor, class, score, x1, y1, x2, y2)` in arrival order, one per anchor
/// whose best class score is `> conf`. `out` is cleared first.
#[allow(clippy::too_many_arguments)]
pub fn detect_candidates(
    ctx: &GpuContext,
    box_raw: &wgpu::Buffer,
    cls_raw: &wgpu::Buffer,
    anchors: &wgpu::Buffer,
    out: &wgpu::Buffer,
    batch: usize,
    anchors_n: usize,
    nc: usize,
    rm: usize,
    conf: f32,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "detect_candidates")?;
    if anchors_n == 0 || nc == 0 || rm == 0 {
        return Err(shape_err("detect_candidates: empty head"));
    }
    let ba = batch.checked_mul(anchors_n).ok_or_else(|| shape_err("detect_candidates: too many anchors"))?;
    check_buf(box_raw, ba * 4 * rm, dtype, "detect box_raw")?;
    check_buf(cls_raw, ba * nc, dtype, "detect cls_raw")?;
    check_buf(anchors, 3 * anchors_n, DType::F32, "detect anchors")?;
    let words = batch * (1 + RECORD_WORDS * anchors_n);
    check_buf(out, words, DType::F32, "detect output")?;
    // Every index the shader forms must fit in u32.
    if (words as u64).max((ba * nc.max(4 * rm)) as u64) > u32::MAX as u64 {
        return Err(GpuError::Limit("detect_candidates: head too large".into()));
    }
    if batch == 0 {
        return Ok(());
    }
    ctx.clear(out, 4 * words as u64);
    let r_spec = if rm <= super::detect::MAX_REG_MAX { rm as u32 } else { 0 };
    let k = ctx.kernel(&DETECT, Spec::new().f16(dtype == DType::F16).u32("R", r_spec))?;
    let p = Params::new().uz(ba)?.uz(anchors_n)?.uz(nc)?.uz(rm)?.f(conf);
    ctx.dispatch(&k, &[box_raw, cls_raw, anchors, out], &p, ctx.grid_1d(ba as u64, 256)?)
}

/// Converts `n` packed `u8` bytes from `src` (starting at byte `src_offset`) into `y` (starting at element
/// `out_offset`) as `dtype` (`F32` or `F16`), multiplying each value by `scale` (`1.0 / 255.0` for image
/// normalization). In f32 bit-exact with `byte as f32 * scale`; in f16 converted from that f32 value (within 1
/// binary16 ULP under WGSL's floating-point conversion rules).
#[allow(clippy::too_many_arguments)]
pub fn u8_to_float_scale(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
    src_offset: usize,
    out_offset: usize,
    scale: f32,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "u8_to_float_scale")?;
    let src_end = src_offset
        .checked_add(n)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("u8_to_float_scale: source range exceeds u32".into()))?;
    let out_end = out_offset
        .checked_add(n)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("u8_to_float_scale: output range exceeds u32".into()))?;
    check_buf(src, src_end.div_ceil(4), DType::F32, "u8_to_float_scale source")?;
    check_buf(y, out_end, dtype, "u8_to_float_scale output")?;
    if n == 0 {
        return Ok(());
    }
    let half = dtype == DType::F16;
    if src_offset.is_multiple_of(4) && out_offset.is_multiple_of(4) && n.is_multiple_of(4) {
        let units = n / 4;
        let k = ctx.kernel(&U8_SCALE, Spec::new().bool("VEC4", true).f16(half))?;
        let p = Params::new().uz(units)?.f(scale).uz(src_offset / 4)?.uz(out_offset / 4)?;
        ctx.dispatch(&k, &[src, y], &p, ctx.grid_1d(units as u64, 256)?)
    } else if src_offset.is_multiple_of(4) {
        let words = n.div_ceil(4);
        let k = ctx.kernel(&U8_SCALE, Spec::new().bool("VEC4", false).bool("ALIGNED_SRC", true).f16(half))?;
        let p = Params::new().uz(n)?.f(scale).uz(src_offset / 4)?.uz(out_offset)?.uz(words)?;
        ctx.dispatch(&k, &[src, y], &p, ctx.grid_1d(words as u64, 256)?)
    } else {
        let k =
            ctx.kernel(&U8_SCALE, Spec::new().bool("VEC4", false).bool("ALIGNED_SRC", false).f16(half))?;
        let p = Params::new().uz(n)?.f(scale).uz(src_offset)?.uz(out_offset)?;
        ctx.dispatch(&k, &[src, y], &p, ctx.grid_1d(n as u64, 256)?)
    }
}

/// Converts `batch` interleaved RGB `u8` images (`[batch, height, width, 3]` at byte `src_offset` in `src`)
/// into planar `[batch, 3, height, width]` of `dtype` (`F32` or `F16`) at element `out_offset` in `y`,
/// multiplying each value by `scale`. Uses a 3-word / 4-pixel packed `vec4` path when `(height * width) % 4 == 0`
/// and both offsets are 4-aligned. In f32 bit-exact with `byte as f32 * scale`; in f16 within 1 binary16 ULP
/// under WGSL's floating-point conversion rules.
#[allow(clippy::too_many_arguments)]
pub fn rgb8_hwc_to_chw_norm(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    y: &wgpu::Buffer,
    batch: usize,
    height: usize,
    width: usize,
    src_offset: usize,
    out_offset: usize,
    scale: f32,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "rgb8_hwc_to_chw_norm")?;
    let hw = prod(&[height, width])?;
    let pixels = prod(&[batch, hw])?;
    let total = prod(&[pixels, 3])?;
    let src_end = src_offset
        .checked_add(total)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("rgb8_hwc_to_chw_norm: source range exceeds u32".into()))?;
    let out_end = out_offset
        .checked_add(total)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| GpuError::Limit("rgb8_hwc_to_chw_norm: output range exceeds u32".into()))?;
    check_buf(src, src_end.div_ceil(4), DType::F32, "rgb8_hwc_to_chw_norm source")?;
    check_buf(y, out_end, dtype, "rgb8_hwc_to_chw_norm output")?;
    if pixels == 0 {
        return Ok(());
    }
    let half = dtype == DType::F16;
    if hw.is_multiple_of(4) && src_offset.is_multiple_of(4) && out_offset.is_multiple_of(4) {
        let plane4 = hw / 4;
        let total4 = batch * plane4;
        let k = ctx.kernel(&RGB8_CHW_NORM, Spec::new().bool("VEC4", true).f16(half))?;
        let p = Params::new().uz(total4)?.uz(plane4)?.f(scale).uz(src_offset / 4)?.uz(out_offset / 4)?;
        ctx.dispatch(&k, &[src, y], &p, ctx.grid_1d(total4 as u64, 256)?)
    } else {
        let k = ctx.kernel(&RGB8_CHW_NORM, Spec::new().bool("VEC4", false).f16(half))?;
        let p = Params::new().uz(pixels)?.uz(hw)?.f(scale).uz(src_offset)?.uz(out_offset)?;
        ctx.dispatch(&k, &[src, y], &p, ctx.grid_1d(pixels as u64, 256)?)
    }
}

/// Uploads `bytes` via a reused host-visible staging buffer (4× smaller transfer than uploading `f32`) and
/// converts them on the GPU with [`u8_to_float_scale`] into a new pooled `dtype` (`F32` or `F16`) buffer.
pub fn upload_u8_scaled(ctx: &GpuContext, bytes: &[u8], scale: f32, dtype: DType) -> Result<GpuBuffer> {
    check_dtype(ctx, dtype, "upload_u8_scaled")?;
    let n = bytes.len();
    let out = ctx.alloc(dtype.bytes(n).max(4))?;
    if n == 0 {
        return Ok(out);
    }
    let (raw_u8, ()) = ctx.upload_with(n as u64, |mut dst| dst.copy_from_slice(bytes))?;
    u8_to_float_scale(ctx, raw_u8.raw(), out.raw(), n, 0, 0, scale, dtype)?;
    Ok(out)
}

/// Uploads `batch` interleaved RGB `u8` images (`rgb.len() == batch * height * width * 3`) via a reused staging
/// buffer and converts them on the GPU with [`rgb8_hwc_to_chw_norm`] into a pooled `[batch, 3, height, width]`
/// `dtype` (`F32` or `F16`) buffer.
pub fn upload_rgb8_hwc_to_chw_norm(
    ctx: &GpuContext,
    rgb: &[u8],
    batch: usize,
    height: usize,
    width: usize,
    scale: f32,
    dtype: DType,
) -> Result<GpuBuffer> {
    check_dtype(ctx, dtype, "upload_rgb8_hwc_to_chw_norm")?;
    let expected = prod(&[batch, height, width, 3])?;
    if rgb.len() != expected {
        return Err(shape_err(format!(
            "upload_rgb8_hwc_to_chw_norm: got {} bytes for [{batch}, {height}, {width}, 3] (expected {expected})",
            rgb.len()
        )));
    }
    let out = ctx.alloc(dtype.bytes(expected).max(4))?;
    if expected == 0 {
        return Ok(out);
    }
    let (raw_u8, ()) = ctx.upload_with(expected as u64, |mut dst| dst.copy_from_slice(rgb))?;
    rgb8_hwc_to_chw_norm(ctx, raw_u8.raw(), out.raw(), batch, height, width, 0, 0, scale, dtype)?;
    Ok(out)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] =
    &[&LETTERBOX_PREFILTER, &LETTERBOX, &DETECT, &U8_SCALE, &RGB8_CHW_NORM];

#[cfg(test)]
mod tests {
    use super::prefilter_fits;

    #[test]
    fn prefilter_guard_counts_the_rounding_term() {
        assert!(prefilter_fits(1, 1));
        // 4100² · 255 + 4100² / 2 = 4 294 955 000 fits; 4101² · 255 fits but adding the rounding term wraps.
        assert!(prefilter_fits(4100, 4100));
        assert!((4101u64 * 4101 * 255) <= u32::MAX as u64);
        assert!(!prefilter_fits(4101, 4101));
        assert!(!prefilter_fits(usize::MAX, 2));
    }
}
