//! The WGSL generators of the direct depthwise kernels: the forward (four adjacent outputs of a row per
//! invocation, the taps unrolled) and the input and weight gradients.

use std::fmt::Write as _;

use crate::kernels::UNARY_FN;
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// Outputs per invocation of the depthwise kernel (adjacent along a row).
pub(super) const DW_VX: u32 = 4;

fn gen_depthwise(spec: &Spec) -> String {
    let g = |name: &str| spec.get_u32(name, 1);
    let (kh, kw, sw, dh, dw) = (g("KH"), g("KW"), g("SW"), g("DH"), g("DW"));
    // Input columns the DW_VX outputs of a row read: (DW_VX - 1) · SW + (KW - 1) · DW + 1.
    let span = (DW_VX - 1) * sw + (kw - 1) * dw + 1;
    let mut taps = String::from(
        "fn taps(acc: ptr<function, array<f32, DW_VX>>, base: u32, wbase: u32, iy0: i32, ix0: i32, h: i32,          w: i32) {
",
    );
    for t in 0..kh * kw {
        let _ = writeln!(taps, "    let w{t} = f32(Wt[wbase + {t}u]);");
    }
    for ky in 0..kh {
        let _ = writeln!(
            taps,
            "    {{
        let iy = iy0 + {};",
            ky * dh
        );
        taps.push_str(
            "        if (iy >= 0 && iy < h) {
            let row = base + u32(iy) * u32(w);
",
        );
        for j in 0..span {
            let _ = writeln!(taps, "            let x{j} = ld(row, ix0 + {j}, w);");
        }
        for v in 0..DW_VX {
            for kx in 0..kw {
                let _ = writeln!(
                    taps,
                    "            (*acc)[{v}] = fma(x{}, w{}, (*acc)[{v}]);",
                    v * sw + kx * dw,
                    ky * kw + kx
                );
            }
        }
        taps.push_str(
            "        }
    }
",
        );
    }
    taps.push_str(
        "}
",
    );
    // Constant indices only, so `acc` stays in registers.
    taps.push_str(
        "fn store(acc: ptr<function, array<f32, DW_VX>>, out: u32, ox0: u32, ow: u32, b: f32) {
",
    );
    for v in 0..DW_VX {
        let _ = writeln!(
            taps,
            "    if (ox0 + {v}u < ow) {{ Y[out + {v}u] = T(unary1(ACT, (*acc)[{v}] + b, pf(6u))); }}"
        );
    }
    taps.push_str(
        "}
",
    );
    format!(
        "const DW_VX: u32 = {DW_VX}u;
{UNARY_FN}{taps}{}",
        include_str!("../../shaders/depthwise.wgsl")
    )
}

pub(super) static DEPTHWISE: KernelDef = KernelDef {
    name: "conv2d_depthwise",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_depthwise),
};

/// Direct depthwise kernels: input gradient (one thread per input element) and weight-gradient partials
/// (one workgroup per output channel and pixel chunk, the taps unrolled into registers).
fn gen_depthwise_grad(spec: &Spec) -> String {
    let (kh, kw) = (spec.get_u32("KH", 1), spec.get_u32("KW", 1));
    let mut s = String::new();
    if spec.get_u32("MODE", 0) == 0 {
        // Params: 0 = input elements, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_in.
        s.push_str(
            "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
             @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
             @group(0) @binding(3) var<storage, read_write> GX: array<T>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {\n\
             \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
             \x20 if (i >= pu(0u)) { return; }\n\
             \x20 let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u); let cin = pu(5u);\n\
             \x20 let ix = i % w; let t = i / w; let iy = t % h; let nc = t / h; let ci = nc % cin; let n = nc / cin;\n\
             \x20 var acc = 0.0;\n\
             \x20 for (var m = 0u; m < MULT; m++) {\n\
             \x20   let co = ci * MULT + m;\n\
             \x20   let gbase = (n * cin * MULT + co) * oh * ow;\n\
             \x20   let wbase = co * KH * KW;\n\
             \x20   for (var ky = 0u; ky < KH; ky++) {\n\
             \x20     let th = i32(iy) + PH - i32(ky * DH);\n\
             \x20     if (th < 0) { continue; }\n\
             \x20     let uh = u32(th);\n\
             \x20     if (uh % SH != 0u || uh / SH >= oh) { continue; }\n\
             \x20     for (var kx = 0u; kx < KW; kx++) {\n\
             \x20       let tw = i32(ix) + PW - i32(kx * DW);\n\
             \x20       if (tw < 0) { continue; }\n\
             \x20       let uw = u32(tw);\n\
             \x20       if (uw % SW != 0u || uw / SW >= ow) { continue; }\n\
             \x20       acc = fma(f32(G[gbase + (uh / SH) * ow + uw / SW]), f32(Wt[wbase + ky * KW + kx]), acc);\n\
             \x20     }\n\
             \x20   }\n\
             \x20 }\n\
             \x20 GX[i] = T(acc);\n}\n",
        );
        return s;
    }
    // Params: 0 = C_out, 1 = chunks S, 2 = pixels per chunk, 3 = N·OH·OW, 4 = OH·OW, 5 = OW, 6 = H, 7 = W.
    // Output: partial[s][co][tap].
    s.push_str(
        "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
         @group(0) @binding(2) var<storage, read> X: array<T>;\n\
         @group(0) @binding(3) var<storage, read_write> OUT: array<f32>;\n\
         var<workgroup> red: array<f32, 256>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let g = wid.y * nwg.x + wid.x;\n\
         \x20 let cout = pu(0u); let chunks = pu(1u);\n\
         \x20 let co = g / chunks; let sidx = g % chunks;\n\
         \x20 let valid = co < cout;\n\
         \x20 let ps = pu(2u); let np = pu(3u); let ohow = pu(4u); let ow = pu(5u); let h = pu(6u); let w = pu(7u);\n",
    );
    for t in 0..kh * kw {
        s.push_str(&format!("  var acc{t} = 0.0;\n"));
    }
    s.push_str(
        "  if (valid) {\n\
         \x20   let p0 = sidx * ps; let p1 = min(p0 + ps, np);\n\
         \x20   let cin = cout / MULT; let ci = co / MULT;\n\
         \x20   for (var p = p0 + lid; p < p1; p += 256u) {\n\
         \x20     let n = p / ohow; let r = p % ohow;\n\
         \x20     let gv = f32(G[(n * cout + co) * ohow + r]);\n\
         \x20     let base = (n * cin + ci) * h * w;\n\
         \x20     let iy0 = i32((r / ow) * SH) - PH; let ix0 = i32((r % ow) * SW) - PW;\n",
    );
    for ky in 0..kh {
        s.push_str(&format!(
            "      {{ let iy = iy0 + {}; if (iy >= 0 && iy < i32(h)) {{ let row = base + u32(iy) * w;\n",
            ky as i64 * spec.get_u32("DH", 1) as i64
        ));
        for kx in 0..kw {
            let t = ky * kw + kx;
            s.push_str(&format!(
                "        {{ let ix = ix0 + {}; if (ix >= 0 && ix < i32(w)) {{ acc{t} = fma(gv, f32(X[row + u32(ix)]), acc{t}); }} }}\n",
                kx as i64 * spec.get_u32("DW", 1) as i64
            ));
        }
        s.push_str("      } }\n");
    }
    s.push_str("    }\n  }\n");
    for t in 0..kh * kw {
        s.push_str(&format!(
            "  red[lid] = acc{t};\n  workgroupBarrier();\n\
             \x20 for (var s = 128u; s > 0u; s >>= 1u) {{ if (lid < s) {{ red[lid] += red[lid + s]; }} workgroupBarrier(); }}\n\
             \x20 if (lid == 0u && valid) {{ OUT[(sidx * cout + co) * KH * KW + {t}u] = red[0]; }}\n\
             \x20 workgroupBarrier();\n"
        ));
    }
    s.push_str("}\n");
    s
}

/// Largest `KH · KW` of a [`DEPTHWISE`] kernel a kernel list may ask for (`kernels::bounds`): the forward unrolls
/// every tap, so its source grows with the kernel. 32×32 covers the largest depthwise kernels in use (31×31);
/// a larger one still compiles when a run asks for it.
const MAX_LISTED_TAPS: u64 = 1024;
/// Largest number of unrolled tap statements ([`gen_depthwise`]: `KH · (span + DW_VX · KW)`) a list may ask for.
const MAX_LISTED_LINES: u64 = 1 << 16;

/// Whether a kernel list may ask for [`DEPTHWISE`] with `spec` (`kernels::bounds`): non-zero sizes, at most
/// [`MAX_LISTED_TAPS`] taps and [`MAX_LISTED_LINES`] unrolled statements.
pub(super) fn forward_fits(spec: &Spec) -> bool {
    let g = |name: &str| u64::from(spec.get_u32(name, 1));
    let (kh, kw, sw, dh, dw) = (g("KH"), g("KW"), g("SW"), g("DH"), g("DW"));
    if [kh, kw, sw, dh, dw].contains(&0) {
        return false;
    }
    let span = (u64::from(DW_VX) - 1) * sw + (kw - 1) * dw + 1;
    kh * kw <= MAX_LISTED_TAPS && kh * (span + u64::from(DW_VX) * kw) <= MAX_LISTED_LINES
}

/// Whether a kernel list may ask for [`DEPTHWISE_GRAD`] with `spec`: `MODE` 0 (loops, any kernel) or `MODE` 1
/// (every tap unrolled) with the taps its call site allows (`max_taps`).
pub(super) fn grad_fits(spec: &Spec, max_taps: usize) -> bool {
    let taps = spec.get_u32("KH", 1) as usize * spec.get_u32("KW", 1) as usize;
    match spec.get_u32("MODE", 0) {
        0 => true,
        1 => (1..=max_taps).contains(&taps),
        _ => false,
    }
}

pub(super) static DEPTHWISE_GRAD: KernelDef = KernelDef {
    name: "conv2d_depthwise_grad",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_depthwise_grad),
};
