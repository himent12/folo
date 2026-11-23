//! WGSL shader definitions and generators for [`super`] (`kernels::muon`): multi-tensor gather / apply,
//! Frobenius norm & divide, step-0 Schatten-4 spectral norm (`muon_s4`), and the grouped `vec4` GEMM
//! (`muon_gemm`) with upper-triangle symmetry, diagonal-tile reuse, XOR-swizzled bank-conflict-free shared
//! memory, and optional double buffering.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

use super::{MANY, STRIDE};
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// The WGSL that finds this invocation's matrix: the last whose `field` (a running start) is at or before `w`.
fn find_matrix(field: usize) -> String {
    format!(
        "fn find_matrix(w: u32) -> u32 {{\n\
         \x20 var lo = 0u; var hi = pu(0u);\n\
         \x20 loop {{\n\
         \x20   if (lo + 1u >= hi) {{ break; }}\n\
         \x20   let mid = (lo + hi) / 2u;\n\
         \x20   if (TBL[mid * {STRIDE}u + {field}u] <= w) {{ lo = mid; }} else {{ hi = mid; }}\n\
         \x20 }}\n\
         \x20 return lo;\n}}\n"
    )
}

/// The index in the padded oriented matrix (`r × align4(c)`) of element `e` of the `rows × cols` original.
const ORIENT: &str = "fn oriented_pad(e: u32, r: u32, c: u32, ld_c: u32, trans: bool) -> u32 {\n\
     \x20 if (trans) { return (e % r) * ld_c + e / r; }\n\
     \x20 return (e / c) * ld_c + e % c;\n}\n";

/// The multi-tensor kernels' binding layouts, by name and matrix count, kept for the process: `muon_gather`
/// (`G`, `M` per matrix, then `X`, `PART`, `TBL`), `muon_apply` (`P` per matrix, then `X`, `TBL`).
pub(super) fn many_def(name: &'static str, j: usize) -> &'static KernelDef {
    static DEFS: OnceLock<Mutex<HashMap<(&'static str, usize), &'static KernelDef>>> = OnceLock::new();
    let mut defs = DEFS.get_or_init(Default::default).lock().expect("muon defs lock");
    defs.entry((name, j)).or_insert_with(|| {
        let (storage, source): (Vec<Access>, fn(&Spec) -> String) = if name == "muon_gather" {
            let s = std::iter::repeat_n(Access::Read, 2 * j).chain([
                Access::ReadWrite,
                Access::ReadWrite,
                Access::Read,
            ]);
            (s.collect(), gen_gather)
        } else {
            (
                std::iter::repeat_n(Access::ReadWrite, j).chain([Access::Read, Access::Read]).collect(),
                gen_apply,
            )
        };
        let storage = Box::leak(storage.into_boxed_slice());
        Box::leak(Box::new(KernelDef { name, storage, source: Source::Generated(source) }))
    })
}

/// The per-layout kernel recorded as `name` with `spec` (the precompile registry, `kernels::find_def`): the
/// [`many_def`] its call site picked, read back from `J`. `None` for other names and for a `J` no call site sets.
pub(crate) fn many_def_for(name: &str, spec: &Spec) -> Option<&'static KernelDef> {
    let j = spec.get_u32("J", 0) as usize;
    let name = ["muon_gather", "muon_apply"].into_iter().find(|n| *n == name)?;
    (1..=MANY).contains(&j).then(|| many_def(name, j))
}

/// The multi-tensor prologue: chunk-local matrix `t` of this workgroup, its element `e` and the table offset `q`
/// of its matrix (`pu(0)` is the chunk's first matrix).
fn many_prologue(s: &mut String, j: usize) {
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let w = wid.y * nwg.x + wid.x;\n\
         \x20 if (w >= pu(3u)) { return; }\n\
         \x20 var t = 0u;\n",
    );
    for t in 1..j {
        let _ = writeln!(s, "  if (w >= pu({}u)) {{ t = {t}u; }}", 16 + t);
    }
    let _ = writeln!(
        s,
        "  let lw = w - pu(16u + t); let e = lw * 256u + lid; let n = pu(32u + t); let ok = e < n;\n\
         \x20 let q = (pu(0u) + t) * {STRIDE}u;"
    );
}

fn gen_gather(spec: &Spec) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> G{t}: array<f32>;", 2 * t + 1);
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> M{t}: array<f32>;", 2 * t + 2);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> X: array<f32>;", 2 * j + 1);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> PART: array<f32>;", 2 * j + 2);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> TBL: array<u32>;", 2 * j + 3);
    s.push_str("var<workgroup> red: array<f32, 256>;\nvar<workgroup> big_flag: atomic<u32>;\n");
    s.push_str(ORIENT);
    many_prologue(&mut s, j);
    s.push_str(
        "  if (lid == 0u) { atomicStore(&big_flag, 0u); }\n\
         \x20 workgroupBarrier();\n\
         \x20 let r = TBL[q]; let c = TBL[q + 1u]; let ld_c = (c + 3u) & 0xfffffffcu;\n\
         \x20 var g = 0.0; var m = 0.0;\n  if (ok) {\n  switch t {\n",
    );
    for t in 0..j {
        let _ = writeln!(s, "  case {t}u: {{ g = G{t}[e]; m = M{t}[e]; }}");
    }
    // `u = (g · (1 − μ)) · 1 + m · μ`; trailing elements `n .. r * ld_c` zero-fill align4 pad columns.
    // Downscale by exact 2^-80 (`0x17800000`) when any element exceeds 2^48 (`0x57800000`) so `u * u` never
    // overflows f32 under WGSL finite-math semantics (bit-exact no-op on normal gradients).
    s.push_str(
        "  default: {}\n  }\n  }\n\
         \x20 var u = m;\n\
         \x20 if (NESTEROV) { u = (g * pf(2u)) * 1.0 + m * pf(1u); }\n\
         \x20 if (ok) {\n\
         \x20   X[TBL[q + 3u] + oriented_pad(e, r, c, ld_c, TBL[q + 2u] != 0u)] = u;\n\
         \x20   if (pu(4u) != 0u && abs(u) > bitcast<f32>(0x57800000u)) { atomicStore(&big_flag, 1u); }\n\
         \x20 } else if (e < r * ld_c) {\n\
         \x20   let pe = e - n; let pad = ld_c - c;\n\
         \x20   X[TBL[q + 3u] + (pe / pad) * ld_c + c + pe % pad] = 0.0;\n\
         \x20 }\n\
         \x20 workgroupBarrier();\n\
         \x20 let is_big = atomicLoad(&big_flag) != 0u;\n\
         \x20 let u_s = select(u, u * bitcast<f32>(0x17800000u), is_big);\n\
         \x20 red[lid] = select(0.0, u_s * u_s, ok);\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var h = 128u; h > 0u; h >>= 1u) {\n\
         \x20   if (lid < h) { red[lid] += red[lid + h]; }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 if (lid == 0u && e < r * ld_c) {\n\
         \x20   let pidx = (TBL[q + 5u] + lw) * 2u;\n\
         \x20   PART[pidx] = red[0];\n\
         \x20   PART[pidx + 1u] = select(0.0, 1.0, is_big);\n\
         \x20 }\n}\n",
    );
    s
}

fn gen_apply(spec: &Spec) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> P{t}: array<f32>;", t + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> X: array<f32>;", j + 1);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> TBL: array<u32>;", j + 2);
    s.push_str(ORIENT);
    many_prologue(&mut s, j);
    let _ = writeln!(
        s,
        "  if (!ok) {{ return; }}\n\
         \x20 let r = TBL[q]; let c = TBL[q + 1u]; let ld_c = (c + 3u) & 0xfffffffcu;\n\
         \x20 let sc = bitcast<f32>(TBL[pu(2u) * {STRIDE}u + pu(0u) + t]);\n\
         \x20 var o = X[TBL[q + 3u] + oriented_pad(e, r, c, ld_c, TBL[q + 2u] != 0u)];\n\
         \x20 if (sc != 1.0) {{ o = o * sc; }}\n\
         \x20 switch t {{"
    );
    for t in 0..j {
        let _ = writeln!(s, "  case {t}u: {{ P{t}[e] = P{t}[e] * 1.0 + o * pf(1u); }}");
    }
    s.push_str("  default: {}\n  }\n}\n");
    s
}

/// One workgroup per matrix: `D[i] = ‖U · sc‖_F + eps · sc` and `D[count + i] = sc` (`1.0` for normal inputs,
/// `2^-80` when any element exceeded `2^48`) over the matrix's `(partial, is_big)` pairs in a fixed order.
pub(super) static NORM: KernelDef = KernelDef {
    name: "muon_norm",
    storage: &[Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Static(
        "@group(0) @binding(1) var<storage, read> PART: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> D: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> TBL: array<u32>;\n\
         var<workgroup> red: array<f32, 256>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = wid.y * nwg.x + wid.x;\n\
         \x20 let count = pu(0u);\n\
         \x20 if (i >= count) { return; }\n\
         \x20 let q = i * 8u;\n\
         \x20 let first = TBL[q + 5u];\n\
         \x20 let ld_c = (TBL[q + 1u] + 3u) & 0xfffffffcu;\n\
         \x20 let n = (TBL[q] * ld_c + 255u) / 256u;\n\
         \x20 var any_big = 0.0;\n\
         \x20 for (var e = lid; e < n; e += 256u) { any_big = max(any_big, PART[(first + e) * 2u + 1u]); }\n\
         \x20 red[lid] = any_big;\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var h = 128u; h > 0u; h >>= 1u) {\n\
         \x20   if (lid < h) { red[lid] = max(red[lid], red[lid + h]); }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 let big = red[0] != 0.0;\n\
         \x20 workgroupBarrier();\n\
         \x20 var acc = 0.0;\n\
         \x20 for (var e = lid; e < n; e += 256u) {\n\
         \x20   let pv = PART[(first + e) * 2u];\n\
         \x20   let pb = PART[(first + e) * 2u + 1u] != 0.0;\n\
         \x20   if (big && !pb) {\n\
         \x20     let s = sqrt(pv) * bitcast<f32>(0x17800000u);\n\
         \x20     acc += s * s;\n\
         \x20   } else {\n\
         \x20     acc += pv;\n\
         \x20   }\n\
         \x20 }\n\
         \x20 red[lid] = acc;\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var h = 128u; h > 0u; h >>= 1u) {\n\
         \x20   if (lid < h) { red[lid] += red[lid + h]; }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 if (lid == 0u) {\n\
         \x20   let sc = select(1.0, bitcast<f32>(0x17800000u), big);\n\
         \x20   D[i] = sqrt(red[0]) + pf(1u) * sc;\n\
         \x20   D[count + i] = sc;\n\
         \x20 }\n}\n",
    ),
};

pub(super) static DIVIDE: KernelDef = KernelDef {
    name: "muon_divide",
    storage: &[Access::ReadWrite, Access::Read, Access::Read],
    source: Source::Generated(gen_divide),
};

/// `X = (X * sc) / D[i]` over every matrix's padded elements (256 per workgroup, as the gather's partials).
fn gen_divide(_: &Spec) -> String {
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n\
         @group(0) @binding(2) var<storage, read> D: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> TBL: array<u32>;\n",
    );
    s.push_str(&find_matrix(5));
    let _ = writeln!(
        s,
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {{\n\
         \x20 let w = wid.y * nwg.x + wid.x;\n\
         \x20 if (w >= pu(1u)) {{ return; }}\n\
         \x20 let i = find_matrix(w);\n\
         \x20 let q = i * {STRIDE}u;\n\
         \x20 let ld_c = (TBL[q + 1u] + 3u) & 0xfffffffcu;\n\
         \x20 let e = (w - TBL[q + 5u]) * 256u + lid;\n\
         \x20 if (e >= TBL[q] * ld_c) {{ return; }}\n\
         \x20 let at = TBL[q + 3u] + e;\n\
         \x20 let sc = D[pu(0u) + i];\n\
         \x20 var v = X[at];\n\
         \x20 if (sc != 1.0) {{ v = v * sc; }}\n\
         \x20 X[at] = v / D[i];\n}}"
    );
    s
}

/// One workgroup per matrix: `1 / (‖A₀‖_F^(1/2) + (eps · sc) / D[i])` into `TBL[count * 9 + i]`, over `A₀` as
/// `vec4<f32>`.
pub(super) static S4_NORM: KernelDef = KernelDef {
    name: "muon_s4",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(
        "@group(0) @binding(1) var<storage, read> A4: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> D: array<f32>;\n\
         @group(0) @binding(3) var<storage, read_write> TBL: array<u32>;\n\
         var<workgroup> red: array<f32, 256>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = wid.y * nwg.x + wid.x;\n\
         \x20 let count = pu(0u);\n\
         \x20 if (i >= count) { return; }\n\
         \x20 let q = i * 8u;\n\
         \x20 let r = TBL[q];\n\
         \x20 let ld_r = (r + 3u) & 0xfffffffcu;\n\
         \x20 let base4 = TBL[q + 4u] >> 2u;\n\
         \x20 let n4 = r * (ld_r >> 2u);\n\
         \x20 var acc = 0.0;\n\
         \x20 for (var e = lid; e < n4; e += 256u) {\n\
         \x20   let v = A4[base4 + e];\n\
         \x20   acc += dot(v, v);\n\
         \x20 }\n\
         \x20 red[lid] = acc;\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var h = 128u; h > 0u; h >>= 1u) {\n\
         \x20   if (lid < h) { red[lid] += red[lid + h]; }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 if (lid == 0u) {\n\
         \x20   let alpha = sqrt(sqrt(red[0])) + (pf(1u) * D[count + i]) / D[i];\n\
         \x20   let inv_a = select(1.0 / alpha, 0.0, alpha == 0.0);\n\
         \x20   TBL[count * 9u + i] = bitcast<u32>(inv_a);\n\
         \x20 }\n}\n",
    ),
};

pub(super) static GROUPED: KernelDef = KernelDef {
    name: "muon_gemm",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Generated(gen_grouped),
};

/// This module's fixed kernels, for the precompile registry (`kernels::find_def`; the gather and the update are
/// built per layout, [`many_def_for`]).
pub(crate) static DEFS: &[&KernelDef] = &[&NORM, &DIVIDE, &S4_NORM, &GROUPED];

/// The grouped `vec4` GEMM of one odd-quintic stage (`MODE` 0: `A = X·Xᵀ`, 1: `Bm = b·A + c·(A·A)`, 2:
/// `X' = a·X + Bm·X`). Supports square (`BM == BN`) symmetric upper-triangle tiles for `MODE` 0/1 (skipping `Bs`
/// on diagonal tiles and writing register-transposed lower-triangle `vec4` blocks on off-diagonal tiles) and
/// rectangular `BM × BN` tiles for `MODE` 2, with bank-conflict-free XOR-swizzled shared memory and optional
/// double buffering (`DBUF`).
fn gen_grouped(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 0);
    let bm = spec.get_u32("BM", 64);
    let bn = spec.get_u32("BN", 64);
    let tm = spec.get_u32("TM", 4);
    let tn = spec.get_u32("TN", 4);
    let depth = spec.get_u32("DEPTH", 16);
    let dbuf = spec.get_bool("DBUF");
    let s4 = spec.get_bool("S4");
    let wgx = bn / tn;
    let wgy = bm / tm;
    let threads = wgx * wgy;
    let (bn4, d4, tm4, tn4) = (bn / 4, depth / 4, tm / 4, tn / 4);
    let sh_rows = if dbuf { 2 * depth } else { depth };

    let pa_count = (bm * d4) / threads;
    let pa_step = threads / d4;
    let pb_count = (depth * bn4) / threads;
    let pb_step = threads / bn4;
    // Bank-conflict-free XOR column swizzle: shift left 1 when d4 == 4 (8 rows/warp), shift 0 when d4 == 8.
    let swz_shift: u32 = if d4 == 4 { 1 } else { 0 };
    let swz_expr = if swz_shift == 1 { "(pk4 << 1u)" } else { "pk4" };

    let mut s = String::new();
    s.push_str(
        "@group(0) @binding(1) var<storage, read> SP: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> SQ: array<vec4<f32>>;\n\
         @group(0) @binding(3) var<storage, read> SR: array<vec4<f32>>;\n\
         @group(0) @binding(4) var<storage, read_write> SC: array<vec4<f32>>;\n\
         @group(0) @binding(5) var<storage, read> TBL: array<u32>;\n",
    );
    // Scalar workgroup storage for transposed tiles (As always, Bs for MODE 0/1) avoids WGSL multi-invocation
    // vec4 component read-modify-write races while keeping the exact same swizzled layout and byte footprint.
    let _ = writeln!(s, "var<workgroup> As: array<array<f32, {bm}>, {sh_rows}>;");
    if mode == 2 {
        let _ = writeln!(s, "var<workgroup> Bs: array<array<vec4<f32>, {bn4}>, {sh_rows}>;");
    } else {
        let _ = writeln!(s, "var<workgroup> Bs: array<array<f32, {bn}>, {sh_rows}>;");
    }
    s.push_str("var<workgroup> prob: array<u32, 13>;\n");
    s.push_str(&find_matrix(if mode == 2 { 7 } else { 6 }));
    let _ = writeln!(s, "@compute @workgroup_size({wgx}, {wgy}, 1)");
    let _ = writeln!(
        s,
        "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) tid: u32, @builtin(local_invocation_id) lid3: vec3<u32>) {{\n\
         \x20 let w = wid.y * nwg.x + wid.x;\n\
         \x20 if (w >= pu(4u)) {{ return; }}\n\
         \x20 if (tid == 0u) {{\n\
         \x20   let count = pu(0u);\n\
         \x20   let i = find_matrix(w);\n\
         \x20   let q = i * {STRIDE}u;\n\
         \x20   let r = TBL[q]; let c = TBL[q + 1u]; let ox = TBL[q + 3u]; let oa = TBL[q + 4u];\n\
         \x20   let ld_r = (r + 3u) & 0xfffffffcu; let ld_c = (c + 3u) & 0xfffffffcu;\n\
         \x20   let inv_a = TBL[count * 9u + i];"
    );
    match mode {
        0 | 1 => {
            let _ = writeln!(
                s,
                "    let tr = (r + {}u) / {bm}u;\n\
                 \x20   let t = w - TBL[q + 6u];\n\
                 \x20   var tm_i = 0u; var rem = t;\n\
                 \x20   loop {{ let row_len = tr - tm_i; if (rem < row_len) {{ break; }} rem -= row_len; tm_i += 1u; }}\n\
                 \x20   let tn_i = tm_i + rem;\n\
                 \x20   let diag = select(0u, 1u, tm_i == tn_i);",
                bm - 1
            );
            if mode == 0 {
                let _ = writeln!(
                    s,
                    "    prob = array<u32, 13>(r, r, c, ox, ld_c, ox, ld_c, oa, ld_r, tm_i * {bm}u, tn_i * {bn}u, diag, inv_a);"
                );
            } else {
                let _ = writeln!(
                    s,
                    "    prob = array<u32, 13>(r, r, r, oa, ld_r, oa, ld_r, oa, ld_r, tm_i * {bm}u, tn_i * {bn}u, diag, inv_a);"
                );
            }
        }
        _ => {
            let _ = writeln!(
                s,
                "    let tc = (c + {}u) / {bn}u;\n\
                 \x20   let t = w - TBL[q + 7u];\n\
                 \x20   prob = array<u32, 13>(r, c, r, oa, ld_r, ox, ld_c, ox, ld_c, (t / tc) * {bm}u, (t % tc) * {bn}u, 0u, inv_a);",
                bn - 1
            );
        }
    }
    s.push_str(
        "  }\n\
         \x20 let pr = workgroupUniformLoad(&prob);\n\
         \x20 let M = pr[0]; let N = pr[1]; let K = pr[2];\n\
         \x20 let offP = pr[3]; let ldP = pr[4]; let offQ = pr[5]; let ldQ = pr[6];\n\
         \x20 let offC = pr[7]; let ldC = pr[8]; let m0 = pr[9]; let n0 = pr[10];\n",
    );
    if mode != 2 {
        s.push_str("  let diag = pr[11];\n");
    }
    s.push_str("  let tx = lid3.x; let ty = lid3.y;\n");
    let _ = writeln!(s, "  let pk4 = tid % {d4}u; let pm0 = tid / {d4}u;");
    if mode == 2 {
        let _ = writeln!(s, "  let qn4 = tid % {bn4}u; let qk0 = tid / {bn4}u;");
    }
    let _ = writeln!(s, "  var ra: array<vec4<f32>, {pa_count}>;");
    let rb_len = if mode == 2 { pb_count } else { pa_count };
    let _ = writeln!(s, "  var rb: array<vec4<f32>, {rb_len}>;");
    for bi in 0..tm4 {
        for r in 0..4 {
            for bj in 0..tn4 {
                let _ = writeln!(s, "  var acc_{bi}_{r}_{bj} = vec4<f32>(0.0);");
            }
        }
    }
    let _ = writeln!(s, "  let nt = (K + {}u) / {depth}u;", depth - 1);

    let mut load = String::new();
    for u in 0..pa_count {
        let _ = writeln!(
            load,
            "    {{ let m = m0 + pm0 + {}u; let k = k0 + pk4 * 4u;\n\
             \x20     if (m < M && k < K) {{ ra[{u}] = SP[(offP + m * ldP + k) >> 2u]; }} else {{ ra[{u}] = vec4<f32>(0.0); }} }}",
            u * pa_step
        );
    }
    if mode == 2 {
        for u in 0..pb_count {
            let _ = writeln!(
                load,
                "    {{ let k = k0 + qk0 + {}u; let n = n0 + qn4 * 4u;\n\
                 \x20     if (k < K && n < N) {{ rb[{u}] = SQ[(offQ + k * ldQ + n) >> 2u]; }} else {{ rb[{u}] = vec4<f32>(0.0); }} }}",
                u * pb_step
            );
        }
    } else {
        load.push_str("    if (diag == 0u) {\n");
        for u in 0..pa_count {
            let _ = writeln!(
                load,
                "      {{ let n = n0 + pm0 + {}u; let k = k0 + pk4 * 4u;\n\
                 \x20       if (n < N && k < K) {{ rb[{u}] = SQ[(offQ + n * ldQ + k) >> 2u]; }} else {{ rb[{u}] = vec4<f32>(0.0); }} }}",
                u * pa_step
            );
        }
        load.push_str("    }\n");
    }

    let store = |sb: u32| -> String {
        let mut st = String::new();
        let base = sb * depth;
        for u in 0..pa_count {
            let _ = writeln!(
                st,
                "    {{ let mm = pm0 + {}u; let idx = (((mm >> 2u) ^ {swz_expr}) << 2u) | (mm & 3u);",
                u * pa_step
            );
            for c in 0..4 {
                let _ = writeln!(st, "      As[{base}u + pk4 * 4u + {c}u][idx] = ra[{u}][{c}];");
            }
            st.push_str("    }\n");
        }
        if mode == 2 {
            for u in 0..pb_count {
                let _ = writeln!(st, "    Bs[{base}u + qk0 + {}u][qn4] = rb[{u}];", u * pb_step);
            }
        } else {
            st.push_str("    if (diag == 0u) {\n");
            for u in 0..pa_count {
                let _ = writeln!(
                    st,
                    "      {{ let nn = pm0 + {}u; let idx = (((nn >> 2u) ^ {swz_expr}) << 2u) | (nn & 3u);",
                    u * pa_step
                );
                for c in 0..4 {
                    let _ = writeln!(st, "        Bs[{base}u + pk4 * 4u + {c}u][idx] = rb[{u}][{c}];");
                }
                st.push_str("      }\n");
            }
            st.push_str("    }\n");
        }
        st
    };

    let emit_fmas = |c: &mut String, sb: u32, use_a_for_b: bool| {
        for kk in 0..depth {
            let row = sb * depth + kk;
            let swz = (kk >> 2) << swz_shift;
            c.push_str("    {\n");
            for bi in 0..tm4 {
                let _ = writeln!(
                    c,
                    "      let ab{bi} = (({}u + ty) ^ {swz}u) << 2u;\n\
                     \x20     let a{bi} = vec4<f32>(As[{row}u][ab{bi}], As[{row}u][ab{bi} + 1u], As[{row}u][ab{bi} + 2u], As[{row}u][ab{bi} + 3u]);",
                    bi * wgy
                );
            }
            for bj in 0..tn4 {
                if use_a_for_b {
                    let _ = writeln!(
                        c,
                        "      let bb{bj} = (({}u + tx) ^ {swz}u) << 2u;\n\
                         \x20     let b{bj} = vec4<f32>(As[{row}u][bb{bj}], As[{row}u][bb{bj} + 1u], As[{row}u][bb{bj} + 2u], As[{row}u][bb{bj} + 3u]);",
                        bj * wgx
                    );
                } else if mode == 2 {
                    let _ = writeln!(c, "      let b{bj} = Bs[{row}u][{}u + tx];", bj * wgx);
                } else {
                    let _ = writeln!(
                        c,
                        "      let bb{bj} = (({}u + tx) ^ {swz}u) << 2u;\n\
                         \x20     let b{bj} = vec4<f32>(Bs[{row}u][bb{bj}], Bs[{row}u][bb{bj} + 1u], Bs[{row}u][bb{bj} + 2u], Bs[{row}u][bb{bj} + 3u]);",
                        bj * wgx
                    );
                }
            }
            for bi in 0..tm4 {
                for r in 0..4 {
                    for bj in 0..tn4 {
                        let acc = format!("acc_{bi}_{r}_{bj}");
                        let _ = writeln!(c, "      {acc} = fma(vec4<f32>(a{bi}[{r}]), b{bj}, {acc});");
                    }
                }
            }
            c.push_str("    }\n");
        }
    };
    let compute = |sb: u32| -> String {
        let mut c = String::new();
        if mode == 2 {
            emit_fmas(&mut c, sb, false);
        } else {
            c.push_str("    if (diag != 0u) {\n");
            emit_fmas(&mut c, sb, true);
            c.push_str("    } else {\n");
            emit_fmas(&mut c, sb, false);
            c.push_str("    }\n");
        }
        c
    };

    if dbuf {
        let _ = writeln!(s, "  {{\n    let k0 = 0u;\n{load}  }}");
        let _ = writeln!(s, "  {{\n{}  }}", store(0));
        s.push_str("  workgroupBarrier();\n");
        s.push_str("  for (var t = 0u; t < nt; t += 2u) {\n");
        let _ = writeln!(
            s,
            "   {{\n    let more = t + 1u < nt;\n    if (more) {{\n      let k0 = (t + 1u) * {depth}u;\n{load}    }}\n{}\
             \x20   if (more) {{\n{}    }}\n    workgroupBarrier();\n   }}",
            compute(0),
            store(1)
        );
        let _ = writeln!(
            s,
            "   if (t + 1u < nt) {{\n    let more = t + 2u < nt;\n    if (more) {{\n      let k0 = (t + 2u) * {depth}u;\n{load}    }}\n{}\
             \x20   if (more) {{\n{}    }}\n    workgroupBarrier();\n   }}",
            compute(1),
            store(0)
        );
        s.push_str("  }\n");
    } else {
        s.push_str("  for (var t = 0u; t < nt; t += 1u) {\n");
        let _ = writeln!(s, "    let k0 = t * {depth}u;\n{load}{}", store(0));
        s.push_str("    workgroupBarrier();\n");
        s.push_str(&compute(0));
        s.push_str("    workgroupBarrier();\n  }\n");
    }

    if mode == 1 {
        if s4 {
            s.push_str(
                "  let ia = bitcast<f32>(pr[12]); let ia2 = ia * ia;\n\
                 \x20 let bb = pf(2u) * ia2; let cc = pf(3u) * (ia2 * ia2);\n",
            );
        } else {
            s.push_str("  let bb = pf(2u); let cc = pf(3u);\n");
        }
    } else if mode == 2 {
        if s4 {
            s.push_str("  let ia = bitcast<f32>(pr[12]); let a_s4 = pf(1u) * ia;\n");
        } else {
            s.push_str("  let a = pf(1u);\n");
        }
    }
    for bi in 0..tm4 {
        let _ = writeln!(s, "  {{ let m_base = m0 + ({}u + ty) * 4u;", bi * wgy);
        for bj in 0..tn4 {
            let _ = writeln!(s, "    {{ let n_base = n0 + ({}u + tx) * 4u; if (n_base < ldC) {{", bj * wgx);
            s.push_str(
                "      let col_ok = vec4<bool>(n_base < N, n_base + 1u < N, n_base + 2u < N, n_base + 3u < N);\n",
            );
            for r in 0..4 {
                let acc = format!("acc_{bi}_{r}_{bj}");
                let expr = match mode {
                    0 => acc,
                    1 => format!("SR[idx] * bb + {acc} * cc"),
                    _ if s4 => format!("SR[idx] * a_s4 + {acc} * ia"),
                    _ => format!("SR[idx] * a + {acc}"),
                };
                let _ = writeln!(
                    s,
                    "      var v{r} = vec4<f32>(0.0);\n\
                     \x20     if (m_base + {r}u < M) {{\n\
                     \x20       let idx = (offC + (m_base + {r}u) * ldC + n_base) >> 2u;\n\
                     \x20       v{r} = {expr};\n\
                     \x20       SC[idx] = select(vec4<f32>(0.0), v{r}, col_ok);\n\
                     \x20     }}"
                );
            }
            if mode != 2 {
                s.push_str("      if (diag == 0u) {\n");
                for c in 0..4 {
                    let _ = writeln!(
                        s,
                        "        if (n_base + {c}u < N) {{\n\
                         \x20         SC[(offC + (n_base + {c}u) * ldC + m_base) >> 2u] = vec4<f32>(v0[{c}], v1[{c}], v2[{c}], v3[{c}]);\n\
                         \x20       }}"
                    );
                }
                s.push_str("      }\n");
            }
            s.push_str("    } }\n");
        }
        s.push_str("  }\n");
    }
    s.push_str("}\n");
    s
}
