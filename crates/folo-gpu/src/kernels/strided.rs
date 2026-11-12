//! Data movement for every folo-core dtype: strided gathers (make-contiguous, `index_select`, `gather`),
//! strided scatters (`copy_into`), fills, casts between all element types, and atomic scatter-adds.
//!
//! Buffers are untyped here: elements are 1, 2, 4 or 8 bytes ([`Elem::bytes`]) packed little-endian into
//! 32-bit words, so `u8` and `f16` elements share words and `i64` elements span two. Gathers compute one output
//! word per thread (assembling sub-word elements), so they never race. Scatters of sub-word elements touch
//! shared words and use `atomicAnd` / `atomicOr` on just their own bits. Strided views are `(offset, dims,
//! strides)` in elements and may broadcast (stride 0); they are coalesced before dispatch.

use std::fmt::Write as _;

use super::{prod, shape_numel};
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::elementwise::MAX_RANK;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Element types of untyped buffers (folo-core's dtypes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    /// 32-bit float.
    F32,
    /// IEEE binary16.
    F16,
    /// Byte.
    U8,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit signed integer (two words, little-endian).
    I64,
}

impl Elem {
    /// Bytes per element.
    pub const fn bytes(self) -> usize {
        match self {
            Elem::F32 | Elem::U32 => 4,
            Elem::F16 => 2,
            Elem::U8 => 1,
            Elem::I64 => 8,
        }
    }

    /// Buffer bytes for `n` elements, rounded up to whole words.
    pub fn buffer_bytes(self, n: usize) -> Result<u64> {
        n.checked_mul(self.bytes())
            .map(|b| (b as u64).next_multiple_of(4))
            .ok_or_else(|| GpuError::Limit(format!("{n} elements of {self:?} overflow")))
    }

    fn code(self) -> u32 {
        match self {
            Elem::F32 => 0,
            Elem::F16 => 1,
            Elem::U8 => 2,
            Elem::U32 => 3,
            Elem::I64 => 4,
        }
    }

    fn from_code(c: u32) -> Elem {
        match c {
            0 => Elem::F32,
            1 => Elem::F16,
            2 => Elem::U8,
            3 => Elem::U32,
            _ => Elem::I64,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Elem::F32 | Elem::F16)
    }
}

/// A strided view of a buffer, in elements.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Strided {
    /// Element offset of the first element.
    pub offset: usize,
    /// Logical dims (row-major order).
    pub dims: Vec<usize>,
    /// Stride per dim; 0 broadcasts.
    pub strides: Vec<usize>,
}

impl Strided {
    /// A dense row-major view of `dims` at offset 0.
    pub fn dense(dims: &[usize]) -> Self {
        let mut strides = vec![1usize; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        Strided { offset: 0, dims: dims.to_vec(), strides }
    }

    /// One past the largest element index the view touches (0 when empty).
    pub fn extent(&self) -> Result<usize> {
        if self.dims.contains(&0) {
            return Ok(0);
        }
        let mut last = self.offset;
        for (&d, &s) in self.dims.iter().zip(&self.strides) {
            last = (d - 1)
                .checked_mul(s)
                .and_then(|v| last.checked_add(v))
                .ok_or_else(|| GpuError::Limit(format!("strided view {self:?} overflows")))?;
        }
        Ok(last + 1)
    }
}

/// Coalesces a logical shape shared by several strided operands: size-1 dims are dropped and neighbours that
/// are contiguous in *every* operand are merged. Returns the dims and one stride list per operand, outermost
/// first.
pub fn coalesce(dims: &[usize], strides: &[&[usize]]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut d: Vec<usize> = Vec::new();
    let mut s: Vec<Vec<usize>> = vec![Vec::new(); strides.len()];
    for i in 0..dims.len() {
        if dims[i] == 1 {
            continue;
        }
        if let Some(&last) = d.last()
            && strides.iter().zip(&s).all(|(st, acc)| acc.last() == Some(&(st[i] * dims[i])))
        {
            let n = d.len() - 1;
            d[n] = last * dims[i];
            for (st, acc) in strides.iter().zip(s.iter_mut()) {
                acc[n] = st[i];
            }
            continue;
        }
        d.push(dims[i]);
        for (st, acc) in strides.iter().zip(s.iter_mut()) {
            acc.push(st[i]);
        }
    }
    (d, s)
}

/// The same copy over whole words: when the innermost coalesced run of a sub-word view (f16, u8) is contiguous in
/// every operand and word-aligned (its length, every offset and every outer stride a multiple of the elements per
/// word), the views read as `u32` views with that run shortened accordingly. A sub-word gather then moves one word
/// per element instead of assembling it, and a sub-word scatter stores words instead of two atomics per element
/// (concatenating f16 tensors). Returns one word view per operand.
fn as_words(elem: Elem, dims: &[usize], strides: &[&[usize]], offsets: &[usize]) -> Option<Vec<Strided>> {
    let per = match elem {
        Elem::F16 => 2,
        Elem::U8 => 4,
        _ => return None,
    };
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
    let view = |st: &[usize], off: usize| Strided {
        offset: off / per,
        dims: d.clone(),
        strides: st.iter().enumerate().map(|(i, &x)| if i == last { 1 } else { x / per }).collect(),
    };
    Some(strides.iter().zip(offsets).map(|(st, &off)| view(st, off)).collect())
}

// ----- WGSL generators ------------------------------------------------------------------------------------------

/// `fn rd(s: u32) -> u32`: element `s` (1, 2 or 4 bytes) of the word array `X`, in the low bits.
fn rd_fn(eb: u32) -> &'static str {
    match eb {
        1 => "fn rd(s: u32) -> u32 { return (X[s >> 2u] >> ((s & 3u) * 8u)) & 0xffu; }\n",
        2 => "fn rd(s: u32) -> u32 { return (X[s >> 1u] >> ((s & 1u) * 16u)) & 0xffffu; }\n",
        _ => "fn rd(s: u32) -> u32 { return X[s]; }\n",
    }
}

/// Code that adds `coord * stride` for each of the `rank` dims (params at `dims_at`, strides at each of
/// `strides_at`) of element `e` to the variables `vars`.
fn coords(rank: u32, dims_at: u32, strides_at: &[u32], vars: &[&str], e: &str) -> String {
    let mut s = format!("    var rem = {e};\n");
    for d in 0..rank {
        let k = rank - 1 - d;
        if k == 0 {
            s.push_str("    { let c = rem;\n");
        } else {
            let _ = writeln!(s, "    {{ let dim = pu({}u); let c = rem % dim; rem = rem / dim;", dims_at + k);
        }
        for (at, v) in strides_at.iter().zip(vars) {
            let _ = writeln!(s, "      {v} += c * pu({}u);", at + k);
        }
        s.push_str("    }\n");
    }
    s
}

const IDX_FNS: &str = "const BAD: u32 = 0xffffffffu;\n";

fn index_at_fn(idx64: bool) -> &'static str {
    if idx64 {
        "fn index_at(e: u32, size: u32) -> u32 {\n\
         \x20 let lo = IDX[2u * e]; let hi = IDX[2u * e + 1u];\n\
         \x20 if (hi != 0u || lo >= size) { return BAD; }\n\
         \x20 return lo;\n}\n"
    } else {
        "fn index_at(e: u32, size: u32) -> u32 {\n\
         \x20 let v = IDX[e];\n\
         \x20 if (v >= size) { return BAD; }\n\
         \x20 return v;\n}\n"
    }
}

/// Gathers: MODE 0 strided, 1 index_select, 2 gather. Spec: EB, MODE, RANK, IDX64.
/// Params: 0 = output elements; MODE 0: 1 = source offset, 4.. dims, 12.. strides; MODE 1/2: 1 = size of the
/// indexed dim, 2 = inner, 3 = index count (MODE 1) or indexed-dim size of the output (MODE 2).
fn gen_move(spec: &Spec) -> String {
    let eb = spec.get_u32("EB", 4);
    let mode = spec.get_u32("MODE", 0);
    let rank = spec.get_u32("RANK", 0);
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         @group(0) @binding(2) var<storage, read> IDX: array<u32>;\n\
         @group(0) @binding(3) var<storage, read_write> Y: array<u32>;\n",
    );
    s.push_str(IDX_FNS);
    s.push_str(index_at_fn(spec.get_bool("IDX64")));
    s.push_str(rd_fn(eb));
    s.push_str("fn src_of(e: u32) -> u32 {\n");
    match mode {
        0 => {
            s.push_str("    var src = pu(1u);\n");
            s.push_str(&coords(rank, 4, &[12], &["src"], "e"));
            s.push_str("    return src;\n");
        }
        1 => s.push_str(
            "    let size = pu(1u); let inner = pu(2u); let l = pu(3u);\n\
             \x20   let j = e % inner; let t = e / inner; let i = t % l; let o = t / l;\n\
             \x20   let v = index_at(i, size);\n\
             \x20   if (v == BAD) { return BAD; }\n\
             \x20   return (o * size + v) * inner + j;\n",
        ),
        _ => s.push_str(
            "    let size = pu(1u); let inner = pu(2u); let di = pu(3u);\n\
             \x20   let j = e % inner; let o = e / inner / di;\n\
             \x20   let v = index_at(e, size);\n\
             \x20   if (v == BAD) { return BAD; }\n\
             \x20   return (o * size + v) * inner + j;\n",
        ),
    }
    s.push_str("}\n");
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
         \x20 let n = pu(0u);\n",
    );
    match eb {
        4 => s.push_str(
            "  if (t >= n) { return; }\n\
             \x20 let src = src_of(t);\n\
             \x20 if (src == BAD) { Y[t] = 0u; } else { Y[t] = X[src]; }\n",
        ),
        8 => s.push_str(
            "  if (t >= n) { return; }\n\
             \x20 let src = src_of(t);\n\
             \x20 if (src == BAD) { Y[2u * t] = 0u; Y[2u * t + 1u] = 0u; }\n\
             \x20 else { Y[2u * t] = X[2u * src]; Y[2u * t + 1u] = X[2u * src + 1u]; }\n",
        ),
        _ => {
            let per = 4 / eb;
            let _ = writeln!(s, "  let e0 = t * {per}u;\n  if (e0 >= n) {{ return; }}\n  var w = 0u;");
            for q in 0..per {
                let _ = writeln!(
                    s,
                    "  if (e0 + {q}u < n) {{ let src = src_of(e0 + {q}u); \
                     if (src != BAD) {{ w |= rd(src) << {}u; }} }}",
                    q * eb * 8
                );
            }
            s.push_str("  Y[t] = w;\n");
        }
    }
    s.push_str("}\n");
    s
}

/// Strided scatter `Y[dst(e)] = X[src(e)]`. Spec: EB, RANK. Params: 0 = elements, 1 = source offset,
/// 2 = destination offset, 4.. dims, 12.. source strides, 20.. destination strides.
fn gen_scatter(spec: &Spec) -> String {
    let eb = spec.get_u32("EB", 4);
    let rank = spec.get_u32("RANK", 0);
    let mut s = String::from("@group(0) @binding(1) var<storage, read> X: array<u32>;\n");
    if eb < 4 {
        s.push_str("@group(0) @binding(2) var<storage, read_write> Y: array<atomic<u32>>;\n");
    } else {
        s.push_str("@group(0) @binding(2) var<storage, read_write> Y: array<u32>;\n");
    }
    s.push_str(rd_fn(eb));
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let e = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (e >= pu(0u)) { return; }\n\
         \x20 var src = pu(1u); var dst = pu(2u);\n",
    );
    s.push_str(&coords(rank, 4, &[12, 20], &["src", "dst"], "e"));
    match eb {
        4 => s.push_str("  Y[dst] = X[src];\n"),
        8 => s.push_str("  Y[2u * dst] = X[2u * src]; Y[2u * dst + 1u] = X[2u * src + 1u];\n"),
        _ => {
            let per = 4 / eb;
            let mask = if eb == 1 { "0xffu" } else { "0xffffu" };
            let _ = writeln!(
                s,
                "  let sh = (dst % {per}u) * {}u; let w = dst / {per}u;\n\
                 \x20 atomicAnd(&Y[w], ~({mask} << sh));\n\
                 \x20 atomicOr(&Y[w], rd(src) << sh);",
                eb * 8
            );
        }
    }
    s.push_str("}\n");
    s
}

/// `Y[t] = t even ? p0 : p1` over `n` words. Params: 0 = words, 1 = p0, 2 = p1.
const FILL_SRC: &str = "@group(0) @binding(1) var<storage, read_write> Y: array<u32>;\n\
     @compute @workgroup_size(256)\n\
     fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
     \x20       @builtin(local_invocation_index) lid: u32) {\n\
     \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
     \x20 if (t >= pu(0u)) { return; }\n\
     \x20 Y[t] = select(pu(1u), pu(2u), (t & 1u) == 1u);\n}\n";

/// Conversions shared by casts.
const CAST_FNS: &str = r#"
fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
// f32 to binary16 bits, rounding to nearest even in integer arithmetic (WGSL leaves `pack2x16float`'s rounding
// to the implementation), with the `half` crate's NaN encoding.
fn f16bits(v: f32) -> u32 {
  let b = bitcast<u32>(v);
  let sgn = (b >> 16u) & 0x8000u;
  let a = b & 0x7fffffffu;
  if (a > 0x7f800000u) { return sgn | 0x7e00u | ((a >> 13u) & 0x3ffu); }
  // |v| >= 65520 rounds to infinity.
  if (a >= 0x477ff000u) { return sgn | 0x7c00u; }
  let e = i32(a >> 23u) - 127;
  if (e >= -14) {
    // Normal: drop 13 mantissa bits; a carry into the exponent is the correct rounding.
    var out = (u32(e + 15) << 10u) | ((a >> 13u) & 0x3ffu);
    let rem = a & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u) == 1u)) { out += 1u; }
    return sgn | out;
  }
  if (e < -25) { return sgn; }
  // Subnormal: value / 2^-24 = m >> s with the implicit bit restored.
  let m = (a & 0x7fffffu) | 0x800000u;
  let s = u32(-e - 1);
  var out = m >> s;
  let rem = m & ((1u << s) - 1u);
  let half = 1u << (s - 1u);
  if (rem > half || (rem == half && (out & 1u) == 1u)) { out += 1u; }
  return sgn | out;
}
// binary16 bits to f32; NaNs keep sign and payload with the quiet bit set (as the `half` crate does).
fn f16tof32(h: u32) -> f32 {
  if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0u) {
    return bitcast<f32>(((h & 0x8000u) << 16u) | 0x7fc00000u | ((h & 0x3ffu) << 13u));
  }
  return unpack2x16float(h).x;
}
// 64-bit unsigned helpers on (lo, hi) pairs; shift amounts are 0..63.
fn shl64(v: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return v; }
  if (s >= 32u) { return vec2<u32>(0u, v.x << (s - 32u)); }
  return vec2<u32>(v.x << s, (v.y << s) | (v.x >> (32u - s)));
}
fn shr64(v: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return v; }
  if (s >= 32u) { return vec2<u32>(v.y >> (s - 32u), 0u); }
  return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}
// `v` rounded to its top `n` significant bits, to nearest even.
fn round_bits(v: vec2<u32>, n: u32) -> vec2<u32> {
  var len = 32u - countLeadingZeros(v.x);
  if (v.y != 0u) { len = 64u - countLeadingZeros(v.y); }
  if (len <= n) { return v; }
  let s = len - n;
  var q = shr64(v, s);
  let back = shl64(q, s);
  // r = v - back (< 2^s), compared with half = 2^(s - 1).
  let rlo = v.x - back.x;
  let rhi = v.y - back.y - select(0u, 1u, v.x < back.x);
  let half = shl64(vec2<u32>(1u, 0u), s - 1u);
  let gt = rhi > half.y || (rhi == half.y && rlo > half.x);
  let eq = rhi == half.y && rlo == half.x;
  if (gt || (eq && (q.x & 1u) == 1u)) {
    q.x += 1u;
    if (q.x == 0u) { q.y += 1u; }
  }
  return shl64(q, s);
}
// Integer (i64 bits) to f32 as the CPU does it: round once, straight to `n` bits (24 for f32, 11 for the f16
// path). The result has at most `n` significant bits, so both halves convert exactly.
fn int2f(v: vec2<u32>, n: u32) -> f32 {
  let neg = (v.y >> 31u) == 1u;
  var m = v;
  if (neg) {
    m.x = ~v.x + 1u;
    m.y = ~v.y + select(0u, 1u, m.x == 0u);
  }
  let r = round_bits(m, n);
  let f = f32(r.y) * 4294967296.0 + f32(r.x);
  return select(f, -f, neg);
}
// Float to integer: truncate toward zero and saturate; NaN becomes 0 (Rust `as` semantics).
fn f2u8(v: f32) -> u32 { if (isnan1(v) || v <= 0.0) { return 0u; } if (v >= 255.0) { return 255u; } return u32(v); }
fn f2u32(v: f32) -> u32 {
  if (isnan1(v) || v <= 0.0) { return 0u; }
  if (v >= 4294967296.0) { return 0xffffffffu; }
  return u32(v);
}
fn f2i64(v: f32) -> vec2<u32> {
  if (isnan1(v)) { return vec2<u32>(0u, 0u); }
  if (v >= 9.223372e18) { return vec2<u32>(0xffffffffu, 0x7fffffffu); }
  if (v <= -9.223372e18) { return vec2<u32>(0u, 0x80000000u); }
  if (abs(v) < 2147483648.0) {
    let i = i32(v);
    return vec2<u32>(bitcast<u32>(i), select(0u, 0xffffffffu, i < 0));
  }
  // |v| >= 2^31 is an integer multiple of 2^8, so both halves are exact.
  let hi = floor(v / 4294967296.0);
  return vec2<u32>(u32(v - hi * 4294967296.0), bitcast<u32>(i32(hi)));
}
// Integer (as i64 bits) to integer: saturate.
fn i2u8(v: vec2<u32>) -> u32 { if (v.y != 0u) { return select(255u, 0u, (v.y >> 31u) == 1u); } return min(v.x, 255u); }
fn i2u32(v: vec2<u32>) -> u32 { if (v.y != 0u) { return select(0xffffffffu, 0u, (v.y >> 31u) == 1u); } return v.x; }
"#;

/// Cast between element types. Spec: SRC, DST (element codes). Params: 0 = elements, 1 = source offset.
fn gen_cast(spec: &Spec) -> String {
    let src = Elem::from_code(spec.get_u32("SRC", 0));
    let dst = Elem::from_code(spec.get_u32("DST", 0));
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<u32>;\n",
    );
    s.push_str(CAST_FNS);
    s.push_str(rd_fn(if src == Elem::U8 { 1 } else { 4 }));
    let load_f = match src {
        Elem::F32 => "bitcast<f32>(X[e])",
        Elem::F16 => "f16tof32((X[e >> 1u] >> ((e & 1u) * 16u)) & 0xffffu)",
        // Integer sources convert through `int2f` in `conv`.
        Elem::U8 | Elem::U32 | Elem::I64 => "0.0",
    };
    let load_i = match src {
        Elem::U8 => "vec2<u32>(rd(e), 0u)",
        Elem::U32 => "vec2<u32>(X[e], 0u)",
        Elem::I64 => "vec2<u32>(X[2u * e], X[2u * e + 1u])",
        _ => "vec2<u32>(0u, 0u)",
    };
    let _ = writeln!(s, "fn load_f(e: u32) -> f32 {{ return {load_f}; }}");
    let _ = writeln!(s, "fn load_i(e: u32) -> vec2<u32> {{ return {load_i}; }}");
    // `conv(e)` returns the destination bits of source element `e` in the low bits (two words for i64).
    let conv = match (dst, src.is_float()) {
        (Elem::F32, true) => "vec2<u32>(bitcast<u32>(load_f(e)), 0u)",
        (Elem::F32, false) => "vec2<u32>(bitcast<u32>(int2f(load_i(e), 24u)), 0u)",
        (Elem::F16, true) => "vec2<u32>(f16bits(load_f(e)), 0u)",
        (Elem::F16, false) => "vec2<u32>(f16bits(int2f(load_i(e), 11u)), 0u)",
        (Elem::U8, true) => "vec2<u32>(f2u8(load_f(e)), 0u)",
        (Elem::U8, false) => "vec2<u32>(i2u8(load_i(e)), 0u)",
        (Elem::U32, true) => "vec2<u32>(f2u32(load_f(e)), 0u)",
        (Elem::U32, false) => "vec2<u32>(i2u32(load_i(e)), 0u)",
        (Elem::I64, true) => "f2i64(load_f(e))",
        (Elem::I64, false) => "load_i(e)",
    };
    let _ = writeln!(s, "fn conv(e: u32) -> vec2<u32> {{ return {conv}; }}");
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
         \x20 let n = pu(0u); let off = pu(1u);\n",
    );
    match dst.bytes() {
        4 => s.push_str("  if (t >= n) { return; }\n  Y[t] = conv(off + t).x;\n"),
        8 => s.push_str(
            "  if (t >= n) { return; }\n  let v = conv(off + t);\n  Y[2u * t] = v.x; Y[2u * t + 1u] = v.y;\n",
        ),
        eb => {
            let per = 4 / eb;
            let _ = writeln!(s, "  let e0 = t * {per}u;\n  if (e0 >= n) {{ return; }}\n  var w = 0u;");
            for q in 0..per {
                let _ = writeln!(
                    s,
                    "  if (e0 + {q}u < n) {{ w |= conv(off + e0 + {q}u).x << {}u; }}",
                    q * eb * 8
                );
            }
            s.push_str("  Y[t] = w;\n");
        }
    }
    s.push_str("}\n");
    s
}

/// Atomic float scatter-add: MODE 0 index_add, 1 scatter_add. Spec: MODE, IDX64.
/// Params: 0 = source elements, 1 = size of the indexed dim of `Y`, 2 = inner, 3 = index count (MODE 0) or
/// indexed-dim size of the source (MODE 1).
fn gen_scatter_add(spec: &Spec) -> String {
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> SRC: array<f32>;\n\
         @group(0) @binding(2) var<storage, read> IDX: array<u32>;\n\
         @group(0) @binding(3) var<storage, read_write> Y: array<atomic<u32>>;\n",
    );
    s.push_str(IDX_FNS);
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
         \x20 let e = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (e >= pu(0u)) { return; }\n\
         \x20 let size = pu(1u); let inner = pu(2u); let l = pu(3u);\n\
         \x20 let j = e % inner; let t = e / inner; let o = t / l;\n",
    );
    if spec.get_u32("MODE", 0) == 0 {
        s.push_str("  let v = index_at(t % l, size);\n");
    } else {
        s.push_str("  let v = index_at(e, size);\n");
    }
    s.push_str("  if (v != BAD) { add((o * size + v) * inner + j, SRC[e]); }\n}\n");
    s
}

static MOVE: KernelDef = KernelDef {
    name: "move",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_move),
};
static SCATTER: KernelDef = KernelDef {
    name: "scatter",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_scatter),
};
static FILL: KernelDef =
    KernelDef { name: "fill", storage: &[Access::ReadWrite], source: Source::Static(FILL_SRC) };
static CAST: KernelDef = KernelDef {
    name: "cast_any",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_cast),
};
static SCATTER_ADD: KernelDef = KernelDef {
    name: "scatter_add",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_scatter_add),
};

fn eb_of(e: Elem) -> u32 {
    e.bytes() as u32
}

/// Threads for an output of `n` elements of `e` (one per element, or one per word for sub-word elements).
fn threads(e: Elem, n: usize) -> u64 {
    match e.bytes() {
        b if b >= 4 => n as u64,
        b => n.div_ceil(4 / b) as u64,
    }
}

fn push_dims(mut p: Params, at: usize, v: &[usize], fill: usize) -> Result<Params> {
    p = p.at(at);
    for i in 0..MAX_RANK {
        p = p.uz(v.get(i).copied().unwrap_or(fill))?;
    }
    Ok(p)
}

fn check_rank(op: &str, rank: usize) -> Result<()> {
    if rank > MAX_RANK {
        return Err(shape_err(format!("{op}: rank {rank} after coalescing exceeds {MAX_RANK}")));
    }
    Ok(())
}

/// Writes the elements of `src` read through `view` densely (row-major) into `dst`.
pub fn gather_strided(
    ctx: &GpuContext,
    elem: Elem,
    src: &wgpu::Buffer,
    view: &Strided,
    dst: &wgpu::Buffer,
) -> Result<()> {
    let n = shape_numel(&view.dims)?;
    check_buf_elems(src, view.extent()?, elem, "gather source")?;
    check_buf_elems(dst, n, elem, "gather destination")?;
    if n == 0 {
        return Ok(());
    }
    let (dims, st) = coalesce(&view.dims, &[&view.strides]);
    let st = &st[0];
    // Contiguous source: a buffer copy when the bytes line up with words.
    if dims.len() <= 1 && st.first().is_none_or(|&s| s == 1) {
        let (off, len) = (view.offset * elem.bytes(), n * elem.bytes());
        if off.is_multiple_of(4) && (len.is_multiple_of(4) || elem.bytes() >= 4) {
            ctx.copy(src, off as u64, dst, 0, (len as u64).next_multiple_of(4));
            return Ok(());
        }
    }
    if let Some(w) = as_words(elem, &dims, &[st.as_slice()], &[view.offset]) {
        return gather_strided(ctx, Elem::U32, src, &w[0], dst);
    }
    check_rank("gather_strided", dims.len())?;
    let spec =
        Spec::new().u32("EB", eb_of(elem)).u32("MODE", 0).u32("RANK", dims.len() as u32).bool("IDX64", false);
    let k = ctx.kernel(&MOVE, spec)?;
    let p = Params::new().uz(n)?.uz(view.offset)?;
    let p = push_dims(p, 4, &dims, 1)?;
    let p = push_dims(p, 12, st, 0)?;
    ctx.dispatch(&k, &[src, ctx.dummy(), dst], &p, ctx.grid_1d(threads(elem, n), 256)?)
}

/// Writes `src` (read through `src_view`) into `dst` through `dst_view` (same dims; `dst_view` must not
/// overlap itself).
pub fn scatter_strided(
    ctx: &GpuContext,
    elem: Elem,
    src: &wgpu::Buffer,
    src_view: &Strided,
    dst: &wgpu::Buffer,
    dst_view: &Strided,
) -> Result<()> {
    if src_view.dims != dst_view.dims {
        return Err(shape_err(format!("scatter: dims {:?} vs {:?}", src_view.dims, dst_view.dims)));
    }
    let n = shape_numel(&src_view.dims)?;
    check_buf_elems(src, src_view.extent()?, elem, "scatter source")?;
    check_buf_elems(dst, dst_view.extent()?, elem, "scatter destination")?;
    if n == 0 {
        return Ok(());
    }
    let (dims, st) = coalesce(&src_view.dims, &[&src_view.strides, &dst_view.strides]);
    let (ss, ds) = (&st[0], &st[1]);
    if dims.len() <= 1 && ss.first().is_none_or(|&s| s == 1) && ds.first().is_none_or(|&s| s == 1) {
        let (so, dof, len) =
            (src_view.offset * elem.bytes(), dst_view.offset * elem.bytes(), n * elem.bytes());
        if so.is_multiple_of(4) && dof.is_multiple_of(4) && len.is_multiple_of(4) {
            ctx.copy(src, so as u64, dst, dof as u64, len as u64);
            return Ok(());
        }
    }
    if let Some(w) =
        as_words(elem, &dims, &[ss.as_slice(), ds.as_slice()], &[src_view.offset, dst_view.offset])
    {
        return scatter_strided(ctx, Elem::U32, src, &w[0], dst, &w[1]);
    }
    check_rank("scatter_strided", dims.len())?;
    let k = ctx.kernel(&SCATTER, Spec::new().u32("EB", eb_of(elem)).u32("RANK", dims.len() as u32))?;
    let p = Params::new().uz(n)?.uz(src_view.offset)?.uz(dst_view.offset)?;
    let p = push_dims(p, 4, &dims, 1)?;
    let p = push_dims(p, 12, ss, 0)?;
    let p = push_dims(p, 20, ds, 0)?;
    ctx.dispatch(&k, &[src, dst], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Fills the first `words` words of `dst` with the repeating pair `(p0, p1)` (pass the same word twice for
/// elements of four bytes or fewer).
pub fn fill_words(ctx: &GpuContext, dst: &wgpu::Buffer, words: usize, p0: u32, p1: u32) -> Result<()> {
    check_buf_elems(dst, words, Elem::U32, "fill")?;
    if words == 0 {
        return Ok(());
    }
    if p0 == 0 && p1 == 0 {
        ctx.clear(dst, words as u64 * 4);
        return Ok(());
    }
    let k = ctx.kernel(&FILL, Spec::new())?;
    let p = Params::new().uz(words)?.u(p0).u(p1);
    ctx.dispatch(&k, &[dst], &p, ctx.grid_1d(words as u64, 256)?)
}

/// Converts `n` dense elements of `src` starting at element `offset` from `from` to `to` into `dst`.
/// Floats to integers truncate toward zero and saturate (NaN becomes 0); integers saturate.
#[allow(clippy::too_many_arguments)]
pub fn cast(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    from: Elem,
    offset: usize,
    dst: &wgpu::Buffer,
    to: Elem,
    n: usize,
) -> Result<()> {
    check_buf_elems(
        src,
        offset.checked_add(n).ok_or_else(|| shape_err("cast overflows"))?,
        from,
        "cast input",
    )?;
    check_buf_elems(dst, n, to, "cast output")?;
    if n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&CAST, Spec::new().u32("SRC", from.code()).u32("DST", to.code()))?;
    let p = Params::new().uz(n)?.uz(offset)?;
    ctx.dispatch(&k, &[src, dst], &p, ctx.grid_1d(threads(to, n), 256)?)
}

/// Index tensor operand: a dense buffer of `I64` (two words per index) or `U32` indices.
#[derive(Clone, Copy, Debug)]
pub struct Indices<'a> {
    /// The buffer.
    pub buf: &'a wgpu::Buffer,
    /// `true` for `I64`, `false` for `U32`.
    pub i64: bool,
    /// Number of indices.
    pub len: usize,
}

impl Indices<'_> {
    fn check(&self, what: &str) -> Result<()> {
        check_buf_elems(self.buf, self.len, if self.i64 { Elem::I64 } else { Elem::U32 }, what)
    }
}

/// `out[o, i, j] = x[o, idx[i], j]` for dense `x` of `[outer, d, inner]`; out-of-range indices give zeros.
pub fn index_select(
    ctx: &GpuContext,
    elem: Elem,
    x: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    out: &wgpu::Buffer,
) -> Result<()> {
    let n = prod(&[outer, idx.len, inner])?;
    check_buf_elems(x, prod(&[outer, d, inner])?, elem, "index_select input")?;
    check_buf_elems(out, n, elem, "index_select output")?;
    idx.check("index_select indices")?;
    if n == 0 {
        return Ok(());
    }
    let spec = Spec::new().u32("EB", eb_of(elem)).u32("MODE", 1).u32("RANK", 0).bool("IDX64", idx.i64);
    let k = ctx.kernel(&MOVE, spec)?;
    let p = Params::new().uz(n)?.uz(d)?.uz(inner)?.uz(idx.len)?;
    ctx.dispatch(&k, &[x, idx.buf, out], &p, ctx.grid_1d(threads(elem, n), 256)?)
}

/// `out[o, i, j] = x[o, idx[o, i, j], j]` for dense `x` of `[outer, d, inner]` and `idx` of
/// `[outer, di, inner]`; out-of-range indices give zeros.
pub fn gather(
    ctx: &GpuContext,
    elem: Elem,
    x: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    di: usize,
    out: &wgpu::Buffer,
) -> Result<()> {
    let n = prod(&[outer, di, inner])?;
    if idx.len != n {
        return Err(shape_err(format!("gather: {} indices for {n} outputs", idx.len)));
    }
    check_buf_elems(x, prod(&[outer, d, inner])?, elem, "gather input")?;
    check_buf_elems(out, n, elem, "gather output")?;
    idx.check("gather indices")?;
    if n == 0 {
        return Ok(());
    }
    let spec = Spec::new().u32("EB", eb_of(elem)).u32("MODE", 2).u32("RANK", 0).bool("IDX64", idx.i64);
    let k = ctx.kernel(&MOVE, spec)?;
    let p = Params::new().uz(n)?.uz(d)?.uz(inner)?.uz(di)?;
    ctx.dispatch(&k, &[x, idx.buf, out], &p, ctx.grid_1d(threads(elem, n), 256)?)
}

/// `out[o, idx[i], j] += src[o, i, j]` (`index_add`) for `out` of `[outer, d, inner]` and `src` of
/// `[outer, idx.len, inner]`, f32, with atomics (summation order is unspecified). Out-of-range indices are
/// skipped. `out` must already hold the base values.
pub fn index_add(
    ctx: &GpuContext,
    out: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    src: &wgpu::Buffer,
) -> Result<()> {
    scatter_add_impl(ctx, 0, out, (outer, d, inner), idx, idx.len, src)
}

/// `out[o, idx[o, i, j], j] += src[o, i, j]` (`scatter_add`) for `out` of `[outer, d, inner]` and `idx`,
/// `src` of `[outer, di, inner]`, f32, with atomics. Out-of-range indices are skipped.
pub fn scatter_add(
    ctx: &GpuContext,
    out: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    di: usize,
    src: &wgpu::Buffer,
) -> Result<()> {
    scatter_add_impl(ctx, 1, out, (outer, d, inner), idx, di, src)
}

fn scatter_add_impl(
    ctx: &GpuContext,
    mode: u32,
    out: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    idx: Indices<'_>,
    l: usize,
    src: &wgpu::Buffer,
) -> Result<()> {
    let n = prod(&[outer, l, inner])?;
    if mode == 1 && idx.len != n {
        return Err(shape_err(format!("scatter_add: {} indices for {n} sources", idx.len)));
    }
    check_buf_elems(out, prod(&[outer, d, inner])?, Elem::F32, "scatter_add output")?;
    check_buf_elems(src, n, Elem::F32, "scatter_add source")?;
    idx.check("scatter_add indices")?;
    if n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&SCATTER_ADD, Spec::new().u32("MODE", mode).bool("IDX64", idx.i64))?;
    let p = Params::new().uz(n)?.uz(d)?.uz(inner)?.uz(l)?;
    ctx.dispatch(&k, &[src, idx.buf, out], &p, ctx.grid_1d(n as u64, 256)?)
}

fn check_buf_elems(buf: &wgpu::Buffer, elems: usize, elem: Elem, what: &str) -> Result<()> {
    let need = elem.buffer_bytes(elems)?;
    if buf.size() < need {
        return Err(shape_err(format!(
            "{what}: buffer of {} bytes is smaller than {need} bytes",
            buf.size()
        )));
    }
    Ok(())
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&MOVE, &SCATTER, &FILL, &CAST, &SCATTER_ADD];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_multi() {
        // A [2, 3, 4] dense view and a column slice [2, 3, 4] of [2, 3, 8].
        let (d, s) = coalesce(&[2, 3, 4], &[&[12, 4, 1], &[24, 8, 1]]);
        assert_eq!(d, vec![6, 4]);
        assert_eq!(s, vec![vec![4, 1], vec![8, 1]]);
        let (d, s) = coalesce(&[2, 1, 4], &[&[4, 4, 1]]);
        assert_eq!((d, s), (vec![8], vec![vec![1]]));
        // Broadcast dims stay separate unless every operand is contiguous across them.
        let (d, s) = coalesce(&[3, 4], &[&[4, 1], &[0, 1]]);
        assert_eq!((d, s), (vec![3, 4], vec![vec![4, 1], vec![0, 1]]));
    }

    #[test]
    fn strided_extent() {
        assert_eq!(Strided { offset: 2, dims: vec![2, 3], strides: vec![5, 1] }.extent().unwrap(), 10);
        assert_eq!(Strided { offset: 2, dims: vec![0, 3], strides: vec![5, 1] }.extent().unwrap(), 0);
        assert_eq!(Strided::dense(&[2, 3]).strides, vec![3, 1]);
    }
}
