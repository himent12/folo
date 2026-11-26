//! Fast paths for [`super`]: branchless integer `vec4` `f16` <-> `f32` cast, packed-`u32` `u8 -> f32/f16`
//! cast, 16-byte `vec4<u32>` strided gather / `index_select` / scatter / fill, and shared-memory padded 32×32
//! batched 2-D transpose.

use std::fmt::Write as _;

use super::{Elem, Indices, Strided, check_rank, coalesce, coords, index_at_fn, push_dims};
use crate::context::GpuContext;
use crate::error::Result;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

const RW: &[Access] = &[Access::Read, Access::ReadWrite];

/// Branchless integer `vec4` `f16` <-> `f32` cast: handles four elements per thread with a 16 B <-> 8 B
/// vector fast path for normal values and `±0.0`, and exact IEEE-754 RNE / subnormal / `±inf` / `half`-crate
/// NaN-payload handling on rare lanes. Works identically on all backends (Vulkan, DX12, Metal, WebGPU) without
/// relying on driver rounding modes, denorm preservation, or `shader-f16`.
/// Spec: `TO_F16` (bool). Params: 0 = vec4 units (`n / 4`), 1 = source vec4 offset (`offset / 4`).
fn gen_cast_f16_vec4(spec: &Spec) -> String {
    if spec.get_bool("TO_F16") {
        "@group(0) @binding(1) var<storage, read> X: array<vec4<u32>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec2<u32>>;\n\
         fn f16bits_u32(b: u32) -> u32 {\n\
         \x20 let sgn = (b >> 16u) & 0x8000u;\n\
         \x20 let a = b & 0x7fffffffu;\n\
         \x20 if (a > 0x7f800000u) { return sgn | 0x7e00u | ((a >> 13u) & 0x3ffu); }\n\
         \x20 if (a >= 0x477ff000u) { return sgn | 0x7c00u; }\n\
         \x20 let e = i32(a >> 23u) - 127;\n\
         \x20 if (e >= -14) {\n\
         \x20   var out = (u32(e + 15) << 10u) | ((a >> 13u) & 0x3ffu);\n\
         \x20   let rem = a & 0x1fffu;\n\
         \x20   if (rem > 0x1000u || (rem == 0x1000u && (out & 1u) == 1u)) { out += 1u; }\n\
         \x20   return sgn | out;\n\
         \x20 }\n\
         \x20 if (e < -25) { return sgn; }\n\
         \x20 let m = (a & 0x7fffffu) | 0x800000u;\n\
         \x20 let s = u32(-e - 1);\n\
         \x20 var out = m >> s;\n\
         \x20 let rem = m & ((1u << s) - 1u);\n\
         \x20 let half = 1u << (s - 1u);\n\
         \x20 if (rem > half || (rem == half && (out & 1u) == 1u)) { out += 1u; }\n\
         \x20 return sgn | out;\n\
         }\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let bx = X[pu(1u) + i];\n\
         \x20 let ax = bx & vec4<u32>(0x7fffffffu);\n\
         \x20 let is_zero = ax == vec4<u32>(0u);\n\
         \x20 let is_norm = (ax >= vec4<u32>(0x38800000u)) & (ax < vec4<u32>(0x477ff000u));\n\
         \x20 if (all(is_norm | is_zero)) {\n\
         \x20   let sgn = (bx >> vec4<u32>(16u)) & vec4<u32>(0x8000u);\n\
         \x20   let norm = ((ax - vec4<u32>(0x38000000u)) + vec4<u32>(0x0fffu) + ((ax >> vec4<u32>(13u)) & vec4<u32>(1u))) >> vec4<u32>(13u);\n\
         \x20   let h = sgn | select(norm, vec4<u32>(0u), is_zero);\n\
         \x20   Y[i] = vec2<u32>(h.x | (h.y << 16u), h.z | (h.w << 16u));\n\
         \x20   return;\n\
         \x20 }\n\
         \x20 Y[i] = vec2<u32>(\n\
         \x20   f16bits_u32(bx.x) | (f16bits_u32(bx.y) << 16u),\n\
         \x20   f16bits_u32(bx.z) | (f16bits_u32(bx.w) << 16u),\n\
         \x20 );\n\
         }\n"
        .into()
    } else {
        "@group(0) @binding(1) var<storage, read> X: array<vec2<u32>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<u32>>;\n\
         fn f16_to_f32_bits(h: u32) -> u32 {\n\
         \x20 let sgn = (h & 0x8000u) << 16u;\n\
         \x20 let a = h & 0x7fffu;\n\
         \x20 if (a == 0u) { return sgn; }\n\
         \x20 if (a >= 0x7c00u) {\n\
         \x20   if (a == 0x7c00u) { return sgn | 0x7f800000u; }\n\
         \x20   return sgn | 0x7fc00000u | ((a & 0x3ffu) << 13u);\n\
         \x20 }\n\
         \x20 if (a < 0x0400u) {\n\
         \x20   let sh = countLeadingZeros(a) - 21u;\n\
         \x20   return sgn | ((113u - sh) << 23u) | (((a << sh) & 0x3ffu) << 13u);\n\
         \x20 }\n\
         \x20 return sgn | ((a << 13u) + 0x38000000u);\n\
         }\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let w = X[pu(1u) + i];\n\
         \x20 let h = vec4<u32>(w.x & 0xffffu, w.x >> 16u, w.y & 0xffffu, w.y >> 16u);\n\
         \x20 let a = h & vec4<u32>(0x7fffu);\n\
         \x20 let is_zero = a == vec4<u32>(0u);\n\
         \x20 let is_norm = (a >= vec4<u32>(0x0400u)) & (a < vec4<u32>(0x7c00u));\n\
         \x20 if (all(is_norm | is_zero)) {\n\
         \x20   let sgn = (h & vec4<u32>(0x8000u)) << vec4<u32>(16u);\n\
         \x20   let norm = (a << vec4<u32>(13u)) + vec4<u32>(0x38000000u);\n\
         \x20   Y[i] = sgn | select(norm, vec4<u32>(0u), is_zero);\n\
         \x20   return;\n\
         \x20 }\n\
         \x20 Y[i] = vec4<u32>(\n\
         \x20   f16_to_f32_bits(h.x), f16_to_f32_bits(h.y), f16_to_f32_bits(h.z), f16_to_f32_bits(h.w),\n\
         \x20 );\n\
         }\n"
        .into()
    }
}

/// 16-byte `vec4<u32>` fill: `Y[t] = vec4<u32>(p0, p1, p0, p1)`. Params: 0 = vec4 count, 1 = p0, 2 = p1.
const FILL_VEC4_SRC: &str = "@group(0) @binding(1) var<storage, read_write> Y: array<vec4<u32>>;\n\
     @compute @workgroup_size(256)\n\
     fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
     \x20       @builtin(local_invocation_index) lid: u32) {\n\
     \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
     \x20 if (t >= pu(0u)) { return; }\n\
     \x20 let p0 = pu(1u); let p1 = pu(2u);\n\
     \x20 Y[t] = vec4<u32>(p0, p1, p0, p1);\n}\n";

/// 16-byte `vec4<u32>` gather (MODE 0 strided, MODE 1 `index_select`). Spec: MODE, RANK, IDX64.
/// Params: 0 = vec4 units; MODE 0: 1 = source vec4 offset, 4.. dims, 12.. strides;
/// MODE 1: 1 = indexed dim size, 2 = inner vec4 units, 3 = index count.
fn gen_move_vec4(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 0);
    let rank = spec.get_u32("RANK", 0);
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> X: array<vec4<u32>>;\n\
         @group(0) @binding(2) var<storage, read> IDX: array<u32>;\n\
         @group(0) @binding(3) var<storage, read_write> Y: array<vec4<u32>>;\n\
         const BAD: u32 = 0xffffffffu;\n",
    );
    if mode == 1 {
        s.push_str(index_at_fn(spec.get_bool("IDX64")));
    }
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (t >= pu(0u)) { return; }\n",
    );
    if mode == 0 {
        s.push_str("  var src = pu(1u);\n");
        s.push_str(&coords(rank, 4, &[12], &["src"], "t"));
        s.push_str("  Y[t] = X[src];\n}\n");
    } else {
        s.push_str(
            "  let size = pu(1u); let inner = pu(2u); let l = pu(3u);\n\
             \x20 let j = t % inner; let q = t / inner; let i = q % l; let o = q / l;\n\
             \x20 let v = index_at(i, size);\n\
             \x20 if (v == BAD) { Y[t] = vec4<u32>(0u); } else { Y[t] = X[(o * size + v) * inner + j]; }\n}\n",
        );
    }
    s
}

/// 16-byte `vec4<u32>` strided scatter `Y[dst(e)] = X[src(e)]`. Spec: RANK.
/// Params: 0 = vec4 units, 1 = source vec4 offset, 2 = destination vec4 offset, 4.. dims, 12.. src strides, 20.. dst strides.
fn gen_scatter_vec4(spec: &Spec) -> String {
    let rank = spec.get_u32("RANK", 0);
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> X: array<vec4<u32>>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<u32>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let e = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (e >= pu(0u)) { return; }\n\
         \x20 var src = pu(1u); var dst = pu(2u);\n",
    );
    s.push_str(&coords(rank, 4, &[12, 20], &["src", "dst"], "e"));
    s.push_str("  Y[dst] = X[src];\n}\n");
    s
}

/// Shared-memory 32×32 padded batched 2-D strided transpose: reads `[B, R, C]` at `(offset, batch_stride,
/// row_stride, 1)` and writes dense `[B, C, R]`.
/// Spec: `EB` (`2`, `4`, or `8`), plus `f16(true)` for `EB == 2`.
/// Params: 0 = R, 1 = C, 2 = source element offset, 3 = source batch stride, 4 = source row stride.
/// Dispatch: `x = ceil(C / 32)`, `y = ceil(R / 32)`, `z = B`.
fn gen_strided_transpose(spec: &Spec) -> String {
    let eb = spec.get_u32("EB", 4);
    let ty = match eb {
        2 => "T",
        8 => "vec2<u32>",
        _ => "u32",
    };
    let mut s = String::new();
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> X: array<{ty}>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> Y: array<{ty}>;");
    let _ = writeln!(s, "var<workgroup> tile: array<{ty}, 1056>;");
    s.push_str(
        "@compute @workgroup_size(32, 8)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         \x20 let r = pu(0u); let c = pu(1u);\n\
         \x20 let base_in = pu(2u) + wid.z * pu(3u);\n\
         \x20 let row_st = pu(4u);\n\
         \x20 let base_out = wid.z * c * r;\n\
         \x20 let c0 = wid.x * 32u; let r0 = wid.y * 32u;\n\
         \x20 for (var k = 0u; k < 4u; k += 1u) {\n\
         \x20   let j = lid.y + k * 8u;\n\
         \x20   let row = r0 + j; let col = c0 + lid.x;\n\
         \x20   if (row < r && col < c) { tile[j * 33u + lid.x] = X[base_in + row * row_st + col]; }\n\
         \x20 }\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var k = 0u; k < 4u; k += 1u) {\n\
         \x20   let j = lid.y + k * 8u;\n\
         \x20   let col = c0 + j; let row = r0 + lid.x;\n\
         \x20   if (row < r && col < c) { Y[base_out + col * r + row] = tile[lid.x * 33u + j]; }\n\
         \x20 }\n\
         }\n",
    );
    s
}

/// Packed-`u32` `vec4` `u8 -> f32` or `u8 -> f16` cast: reads 1 `u32` (4 packed bytes) per thread and writes
/// `vec4<f32>` (16 B) or packed `vec2<u32>` of 4 IEEE binary16 lanes (8 B, via `pack2x16float`, exact for every
/// integer `0..=255` without requiring `shader-f16`).
/// Spec: `TO_F16` (bool). Params: 0 = vec4 units (`n / 4`), 1 = source word offset (`offset / 4`).
fn gen_cast_u8_vec4(spec: &Spec) -> String {
    if spec.get_bool("TO_F16") {
        "@group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec2<u32>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let w = X[pu(1u) + i];\n\
         \x20 let v = vec4<f32>(f32(w & 0xffu), f32((w >> 8u) & 0xffu), f32((w >> 16u) & 0xffu), f32(w >> 24u));\n\
         \x20 Y[i] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));\n\
         }\n"
            .into()
    } else {
        "@group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let w = X[pu(1u) + i];\n\
         \x20 Y[i] = vec4<f32>(f32(w & 0xffu), f32((w >> 8u) & 0xffu), f32((w >> 16u) & 0xffu), f32(w >> 24u));\n\
         }\n"
            .into()
    }
}

static CAST_F16_VEC4: KernelDef =
    KernelDef { name: "cast_f16_vec4", storage: RW, source: Source::Generated(gen_cast_f16_vec4) };
static CAST_U8_VEC4: KernelDef =
    KernelDef { name: "cast_u8_vec4", storage: RW, source: Source::Generated(gen_cast_u8_vec4) };
static FILL_VEC4: KernelDef =
    KernelDef { name: "fill_vec4", storage: &[Access::ReadWrite], source: Source::Static(FILL_VEC4_SRC) };
static MOVE_VEC4: KernelDef = KernelDef {
    name: "move_vec4",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_move_vec4),
};
static SCATTER_VEC4: KernelDef =
    KernelDef { name: "scatter_vec4", storage: RW, source: Source::Generated(gen_scatter_vec4) };
static STRIDED_TRANSPOSE: KernelDef =
    KernelDef { name: "strided_transpose", storage: RW, source: Source::Generated(gen_strided_transpose) };

/// 16-byte `vec4<u32>` `gather` for 4-byte elements (`F32`/`U32`) with `inner % 4 == 0`:
/// loads 4 indices at once (`vec4<u32>` for `U32`, two `vec4<u32>`s for `I64`), gathers 4 elements from `X`,
/// and writes one coalesced `vec4<u32>` to `Y`.
/// Spec: `IDX64` (bool). Params: 0 = `vec4` units (`n / 4`), 1 = `d`, 2 = `inner / 4`, 3 = `di`.
fn gen_gather_vec4(spec: &Spec) -> String {
    let idx64 = spec.get_bool("IDX64");
    let load_idx = if idx64 {
        "  let a = IDX[2u * t];\n\
         \x20 let b = IDX[2u * t + 1u];\n\
         \x20 let i0 = select(BAD, a.x, a.y == 0u && a.x < size);\n\
         \x20 let i1 = select(BAD, a.z, a.w == 0u && a.z < size);\n\
         \x20 let i2 = select(BAD, b.x, b.y == 0u && b.x < size);\n\
         \x20 let i3 = select(BAD, b.z, b.w == 0u && b.z < size);\n"
    } else {
        "  let iv = IDX[t];\n\
         \x20 let i0 = select(BAD, iv.x, iv.x < size);\n\
         \x20 let i1 = select(BAD, iv.y, iv.y < size);\n\
         \x20 let i2 = select(BAD, iv.z, iv.z < size);\n\
         \x20 let i3 = select(BAD, iv.w, iv.w < size);\n"
    };
    format!(
        "@group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         @group(0) @binding(2) var<storage, read> IDX: array<vec4<u32>>;\n\
         @group(0) @binding(3) var<storage, read_write> Y: array<vec4<u32>>;\n\
         const BAD: u32 = 0xffffffffu;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {{\n\
         \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (t >= pu(0u)) {{ return; }}\n\
         \x20 let size = pu(1u); let inner4 = pu(2u); let di = pu(3u);\n\
         \x20 let j4 = t % inner4; let o = (t / inner4) / di;\n\
         \x20 let inner = inner4 * 4u;\n\
         \x20 let base = o * size * inner + j4 * 4u;\n\
         {load_idx}\
         \x20 var v0 = 0u; if (i0 != BAD) {{ v0 = X[base + i0 * inner]; }}\n\
         \x20 var v1 = 0u; if (i1 != BAD) {{ v1 = X[base + i1 * inner + 1u]; }}\n\
         \x20 var v2 = 0u; if (i2 != BAD) {{ v2 = X[base + i2 * inner + 2u]; }}\n\
         \x20 var v3 = 0u; if (i3 != BAD) {{ v3 = X[base + i3 * inner + 3u]; }}\n\
         \x20 Y[t] = vec4<u32>(v0, v1, v2, v3);\n\
         }}\n"
    )
}

/// Vectorized `index_add` for `F32` with `inner % 4 == 0`: each thread loads one `vec4<f32>` from `SRC`,
/// resolves `index_at(i, size)` once for all 4 inner lanes, and performs 4 consecutive atomic float adds.
/// Spec: `IDX64` (bool). Params: 0 = `vec4` units (`n / 4`), 1 = `d`, 2 = `inner / 4`, 3 = `idx.len`.
fn gen_index_add_vec4(spec: &Spec) -> String {
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> SRC: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> IDX: array<u32>;\n\
         @group(0) @binding(3) var<storage, read_write> Y: array<atomic<u32>>;\n\
         const BAD: u32 = 0xffffffffu;\n",
    );
    s.push_str(index_at_fn(spec.get_bool("IDX64")));
    s.push_str(
        "fn add(i: u32, v: f32) {\n\
         \x20 var old = atomicLoad(&Y[i]);\n\
         \x20 loop {\n\
         \x20   let r = atomicCompareExchangeWeak(&Y[i], old, bitcast<u32>(bitcast<f32>(old) + v));\n\
         \x20   if (r.exchanged) { break; }\n\
         \x20   old = r.old_value;\n\
         \x20 }\n}\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (t >= pu(0u)) { return; }\n\
         \x20 let size = pu(1u); let inner4 = pu(2u); let l = pu(3u);\n\
         \x20 let j4 = t % inner4; let q = t / inner4; let i = q % l; let o = q / l;\n\
         \x20 let v = index_at(i, size);\n\
         \x20 if (v == BAD) { return; }\n\
         \x20 let s4 = SRC[t];\n\
         \x20 let dst = ((o * size + v) * inner4 + j4) * 4u;\n\
         \x20 add(dst, s4.x);\n\
         \x20 add(dst + 1u, s4.y);\n\
         \x20 add(dst + 2u, s4.z);\n\
         \x20 add(dst + 3u, s4.w);\n\
         }\n",
    );
    s
}

static GATHER_VEC4: KernelDef = KernelDef {
    name: "gather_vec4",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_gather_vec4),
};
static INDEX_ADD_VEC4: KernelDef = KernelDef {
    name: "index_add_vec4",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_index_add_vec4),
};

pub(super) static FAST_DEFS: &[&KernelDef] = &[
    &CAST_F16_VEC4,
    &FILL_VEC4,
    &MOVE_VEC4,
    &SCATTER_VEC4,
    &STRIDED_TRANSPOSE,
    &CAST_U8_VEC4,
    &GATHER_VEC4,
    &INDEX_ADD_VEC4,
];

/// Elements per 16-byte `vec4<u32>`.
pub(super) const fn elems_per_vec4(elem: Elem) -> usize {
    16 / elem.bytes()
}

pub(super) struct Vec4Plan {
    dims: Vec<usize>,
    strides: Vec<Vec<usize>>,
    offsets: Vec<usize>,
}

/// Rewrites `dims` / `strides` / `offsets` into 16-byte `vec4<u32>` units when the innermost coalesced dim is
/// contiguous (`stride == 1`) and 16-byte aligned in every operand (innermost length, offsets and outer strides
/// are all multiples of `16 / elem.bytes()`). Drops a trailing dim that collapses to 1 via re-coalescing.
pub(super) fn as_vec4_words(
    elem: Elem,
    dims: &[usize],
    strides: &[&[usize]],
    offsets: &[usize],
) -> Option<Vec4Plan> {
    let per = elems_per_vec4(elem);
    let last = dims.len().checked_sub(1)?;
    let aligned = dims[last].is_multiple_of(per)
        && strides.iter().zip(offsets).all(|(st, &off)| {
            st[last] == 1 && off.is_multiple_of(per) && st[..last].iter().all(|s| s.is_multiple_of(per))
        });
    if !aligned {
        return None;
    }
    let mut d = dims.to_vec();
    d[last] /= per;
    let scaled_st: Vec<Vec<usize>> = strides
        .iter()
        .map(|st| st.iter().enumerate().map(|(i, &x)| if i == last { 1 } else { x / per }).collect())
        .collect();
    let s_refs: Vec<&[usize]> = scaled_st.iter().map(Vec::as_slice).collect();
    let (cd, cs) = coalesce(&d, &s_refs);
    let offs = offsets.iter().map(|&o| o / per).collect();
    Some(Vec4Plan { dims: cd, strides: cs, offsets: offs })
}

/// Attempts the branchless integer `vec4` `f16` <-> `f32` cast fast path; returns `Ok(true)` if dispatched.
pub(super) fn try_cast_f16_vec4(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    from: Elem,
    offset: usize,
    dst: &wgpu::Buffer,
    to: Elem,
    n: usize,
) -> Result<bool> {
    let to_f16 = match (from, to) {
        (Elem::F32, Elem::F16) => true,
        (Elem::F16, Elem::F32) => false,
        _ => return Ok(false),
    };
    if !n.is_multiple_of(4) || !offset.is_multiple_of(4) {
        return Ok(false);
    }
    let units = n / 4;
    let k = ctx.kernel(&CAST_F16_VEC4, Spec::new().bool("TO_F16", to_f16))?;
    let p = Params::new().uz(units)?.uz(offset / 4)?;
    ctx.dispatch(&k, &[src, dst], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the packed-`u32` `vec4` `u8 -> f32/f16` cast fast path; returns `Ok(true)` if dispatched.
pub(super) fn try_cast_u8_to_float_vec4(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    from: Elem,
    offset: usize,
    dst: &wgpu::Buffer,
    to: Elem,
    n: usize,
) -> Result<bool> {
    let to_f16 = match (from, to) {
        (Elem::U8, Elem::F32) => false,
        (Elem::U8, Elem::F16) => true,
        _ => return Ok(false),
    };
    if !n.is_multiple_of(4) || !offset.is_multiple_of(4) {
        return Ok(false);
    }
    let units = n / 4;
    let k = ctx.kernel(&CAST_U8_VEC4, Spec::new().bool("TO_F16", to_f16))?;
    let p = Params::new().uz(units)?.uz(offset / 4)?;
    ctx.dispatch(&k, &[src, dst], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the 16-byte `vec4<u32>` fill fast path; returns `Ok(true)` if dispatched.
pub(super) fn try_fill_vec4(
    ctx: &GpuContext,
    dst: &wgpu::Buffer,
    words: usize,
    p0: u32,
    p1: u32,
) -> Result<bool> {
    if !words.is_multiple_of(4) {
        return Ok(false);
    }
    let units = words / 4;
    let k = ctx.kernel(&FILL_VEC4, Spec::new())?;
    let p = Params::new().uz(units)?.u(p0).u(p1);
    ctx.dispatch(&k, &[dst], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the 16-byte `vec4<u32>` strided gather fast path; returns `Ok(true)` if dispatched.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gather_vec4(
    ctx: &GpuContext,
    elem: Elem,
    src: &wgpu::Buffer,
    dims: &[usize],
    st: &[usize],
    offset: usize,
    dst: &wgpu::Buffer,
    n: usize,
) -> Result<bool> {
    let Some(plan) = as_vec4_words(elem, dims, &[st], &[offset]) else {
        return Ok(false);
    };
    check_rank("gather_strided", plan.dims.len())?;
    let units = n / elems_per_vec4(elem);
    let k = ctx.kernel(
        &MOVE_VEC4,
        Spec::new().u32("MODE", 0).u32("RANK", plan.dims.len() as u32).bool("IDX64", false),
    )?;
    let p = Params::new().uz(units)?.uz(plan.offsets[0])?;
    let p = push_dims(p, 4, &plan.dims, 1)?;
    let p = push_dims(p, 12, &plan.strides[0], 0)?;
    ctx.dispatch(&k, &[src, ctx.dummy(), dst], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the 16-byte `vec4<u32>` `index_select` fast path; returns `Ok(true)` if dispatched.
pub(super) fn try_index_select_vec4(
    ctx: &GpuContext,
    elem: Elem,
    x: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    out: &wgpu::Buffer,
    n: usize,
) -> Result<bool> {
    let per = elems_per_vec4(elem);
    if d == 0 || !inner.is_multiple_of(per) {
        return Ok(false);
    }
    let inner_v = inner / per;
    let units = n / per;
    let _ = outer;
    let k = ctx.kernel(&MOVE_VEC4, Spec::new().u32("MODE", 1).u32("RANK", 0).bool("IDX64", idx.i64))?;
    let p = Params::new().uz(units)?.uz(d)?.uz(inner_v)?.uz(idx.len)?;
    ctx.dispatch(&k, &[x, idx.buf, out], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the 16-byte `vec4<u32>` `gather` fast path for 4-byte elements (`F32`/`U32`) with `inner % 4 == 0`;
/// returns `Ok(true)` if dispatched.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gather_op_vec4(
    ctx: &GpuContext,
    elem: Elem,
    x: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    di: usize,
    out: &wgpu::Buffer,
    n: usize,
) -> Result<bool> {
    if elem.bytes() != 4 || d == 0 || !inner.is_multiple_of(4) {
        return Ok(false);
    }
    let in_elems = super::prod(&[outer, d, inner])?;
    if in_elems > u32::MAX as usize || n > u32::MAX as usize {
        return Ok(false);
    }
    let inner4 = inner / 4;
    let units = n / 4;
    let k = ctx.kernel(&GATHER_VEC4, Spec::new().bool("IDX64", idx.i64))?;
    let p = Params::new().uz(units)?.uz(d)?.uz(inner4)?.uz(di)?;
    ctx.dispatch(&k, &[x, idx.buf, out], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the `vec4<f32>` `index_add` fast path for `inner % 4 == 0` (hoisting `index_at` across 4 lanes);
/// returns `Ok(true)` if dispatched.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_index_add_vec4(
    ctx: &GpuContext,
    out: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    src: &wgpu::Buffer,
    n: usize,
) -> Result<bool> {
    if d == 0 || !inner.is_multiple_of(4) {
        return Ok(false);
    }
    let out_elems = super::prod(&[outer, d, inner])?;
    if out_elems > u32::MAX as usize || n > u32::MAX as usize {
        return Ok(false);
    }
    let inner4 = inner / 4;
    let units = n / 4;
    let k = ctx.kernel(&INDEX_ADD_VEC4, Spec::new().bool("IDX64", idx.i64))?;
    let p = Params::new().uz(units)?.uz(d)?.uz(inner4)?.uz(idx.len)?;
    ctx.dispatch(&k, &[src, idx.buf, out], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the 16-byte `vec4<u32>` strided scatter fast path; returns `Ok(true)` if dispatched.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_scatter_vec4(
    ctx: &GpuContext,
    elem: Elem,
    src: &wgpu::Buffer,
    dims: &[usize],
    ss: &[usize],
    ds: &[usize],
    src_off: usize,
    dst: &wgpu::Buffer,
    dst_off: usize,
    n: usize,
) -> Result<bool> {
    let Some(plan) = as_vec4_words(elem, dims, &[ss, ds], &[src_off, dst_off]) else {
        return Ok(false);
    };
    check_rank("scatter_strided", plan.dims.len())?;
    let units = n / elems_per_vec4(elem);
    let k = ctx.kernel(&SCATTER_VEC4, Spec::new().u32("RANK", plan.dims.len() as u32))?;
    let p = Params::new().uz(units)?.uz(plan.offsets[0])?.uz(plan.offsets[1])?;
    let p = push_dims(p, 4, &plan.dims, 1)?;
    let p = push_dims(p, 12, &plan.strides[0], 0)?;
    let p = push_dims(p, 20, &plan.strides[1], 0)?;
    ctx.dispatch(&k, &[src, dst], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(true)
}

/// Attempts the shared-memory 32×32 padded tiled transpose fast path for `[B, R, C] -> [B, C, R]` strided
/// gathers (`F32`, `U32`, `I64`, and `F16` when `shader-f16` is available); returns `Ok(true)` if dispatched.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_tiled_transpose(
    ctx: &GpuContext,
    elem: Elem,
    src: &wgpu::Buffer,
    view: &Strided,
    dims: &[usize],
    st: &[usize],
    dst: &wgpu::Buffer,
    n: usize,
) -> Result<bool> {
    let eb = elem.bytes() as u32;
    match elem {
        Elem::F32 | Elem::U32 | Elem::I64 => {}
        Elem::F16 if ctx.caps().shader_f16 => {}
        _ => return Ok(false),
    }
    let (b, c, r, b_stride, row_stride) = match dims.len() {
        2 if st[0] == 1 => (1, dims[0], dims[1], 0, st[1]),
        3 if st[1] == 1 => (dims[0], dims[1], dims[2], st[0], st[2]),
        _ => return Ok(false),
    };
    let max_wg = ctx.caps().max_workgroups_per_dim as usize;
    if c.div_ceil(32) > max_wg || r.div_ceil(32) > max_wg || b > max_wg {
        return Ok(false);
    }
    // The shader computes `base_in`, `base_out`, and element indices in `u32`.
    if n > u32::MAX as usize || view.extent()? > u32::MAX as usize {
        return Ok(false);
    }
    let k = ctx.kernel(&STRIDED_TRANSPOSE, Spec::new().u32("EB", eb).f16(elem == Elem::F16))?;
    let p = Params::new().uz(r)?.uz(c)?.uz(view.offset)?.uz(b_stride)?.uz(row_stride)?;
    let groups = [c.div_ceil(32) as u32, r.div_ceil(32) as u32, b as u32];
    ctx.dispatch(&k, &[src, dst], &p, groups)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_vec4_words_aligns_and_collapses() {
        // [2, 3, 8] strided f32 (outer stride 32 != 24) -> [2, 3, 2] vec4<u32> -> coalesces inner [3, 2] to [2, 6]
        let p = as_vec4_words(Elem::F32, &[2, 3, 8], &[&[32, 8, 1]], &[4]).unwrap();
        assert_eq!((p.dims, p.strides, p.offsets), (vec![2, 6], vec![vec![8, 1]], vec![1]));

        // [3, 8] f16 (8 f16s = 16 bytes) collapses trailing 1 to [3]
        let p = as_vec4_words(Elem::F16, &[3, 8], &[&[16, 1], &[8, 1]], &[8, 0]).unwrap();
        assert_eq!((p.dims, p.strides, p.offsets), (vec![3], vec![vec![2], vec![1]], vec![1, 0]));

        // [2, 4] i64 (2 i64s = 16 bytes) -> [2, 2]
        let p = as_vec4_words(Elem::I64, &[2, 4], &[&[6, 1]], &[2]).unwrap();
        assert_eq!((p.dims, p.strides, p.offsets), (vec![2, 2], vec![vec![3, 1]], vec![1]));

        // Unaligned offset or outer stride returns None
        assert!(as_vec4_words(Elem::F32, &[2, 8], &[&[8, 1]], &[2]).is_none());
        assert!(as_vec4_words(Elem::F32, &[2, 8], &[&[10, 1]], &[0]).is_none());
        assert!(as_vec4_words(Elem::F16, &[2, 4], &[&[4, 1]], &[0]).is_none());
    }
}
