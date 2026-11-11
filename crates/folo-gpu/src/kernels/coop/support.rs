//! Whether a device can run the tensor-core kernels: the adapter checks, wgpu's experimental opt-in (with the
//! safety conditions FOLO keeps) and the start-up self-test against a CPU reference.

use super::matmul::encode_gemm;
#[cfg(doc)]
use super::shader::gen_coop;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::kernels::gemm::{GemmBias, GemmDesc};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Whether `adapter` offers cooperative matrices with the configuration the kernels use: 16×16×16, f16 `A` and
/// `B`, f32 accumulator.
pub(crate) fn adapter_supports(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && adapter.cooperative_matrix_properties().iter().any(|p| {
            p.m_size == 16
                && p.n_size == 16
                && p.k_size == 16
                && p.ab_type == wgpu::CooperativeScalarType::F16
                && p.cr_type == wgpu::CooperativeScalarType::F32
        })
}

/// Why the tensor-core kernels cannot run on `adapter` with the device `features` FOLO is about to request, or
/// `None` when they can be tried.
pub(crate) fn unavailable_reason(adapter: &wgpu::Adapter, features: wgpu::Features) -> Option<String> {
    let info = adapter.get_info();
    if !features.contains(wgpu::Features::SHADER_F16) {
        return Some("needs shader-f16 (missing, or FOLO_GPU_NO_F16 is set)".into());
    }
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return Some(format!("{} ({:?}) has no cooperative matrices in wgpu", info.name, info.backend));
    }
    if !adapter_supports(adapter) {
        return Some(format!("{} has no 16x16x16 f16 -> f32 cooperative matrix configuration", info.name));
    }
    if !adapter.features().contains(wgpu::Features::SUBGROUP) {
        return Some(format!("{} has no subgroup support", info.name));
    }
    if info.subgroup_min_size != 32 || info.subgroup_max_size != 32 {
        return Some(format!(
            "the kernels assume 32-wide subgroups; {} has {}..{}",
            info.name, info.subgroup_min_size, info.subgroup_max_size
        ));
    }
    None
}

/// Why the tensor-core kernels cannot run on `adapter` at all (no `shader-f16`, no 16×16×16 f16 → f32
/// cooperative matrices, no 32-wide subgroups), or `None` when it qualifies: a context created on it with
/// [`crate::GpuOptions::tensor_cores`] then has them unless the device refuses or the self-test fails (which tests
/// treat as a bug, not a skip).
pub fn adapter_unavailable_reason(adapter: &wgpu::Adapter) -> Option<String> {
    unavailable_reason(adapter, adapter.features() & wgpu::Features::SHADER_F16)
}

/// Device features the tensor-core kernels need on top of FOLO's usual ones.
pub(crate) const FEATURES: wgpu::Features =
    wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX.union(wgpu::Features::SUBGROUP);

/// wgpu's opt-in for `EXPERIMENTAL_*` features. This is the one place FOLO enables them.
///
/// # Safety conditions (why this is sound for FOLO)
///
/// `ExperimentalFeatures::enabled` is `unsafe` because wgpu's experimental features are not fully validated:
/// misusing them can be undefined behaviour on the GPU (out-of-bounds memory access, driver faults) even from
/// safe code. FOLO uses exactly one of them, `EXPERIMENTAL_COOPERATIVE_MATRIX`, and only under these
/// conditions, which together keep every cooperative-matrix access in bounds and well defined:
///
/// 1. The device is created with it only when the user opted in (`FOLO_GPU_TENSOR_CORES=1`, `--tensor-cores`,
///    or `GpuOptions::tensor_cores`) and the adapter reports the exact 16×16×16 f16 × f16 → f32 configuration
///    with 32-wide subgroups ([`unavailable_reason`]).
/// 2. Only the kernels generated in this module (and the self-test probe) contain
///    `enable wgpu_cooperative_matrix;`, and they only `coopLoadT` / `coopStoreT` through **workgroup** arrays of
///    compile-time size. Every fragment's offset is a compile-time constant plus a subgroup index clamped to
///    `0..8`, and its stride is a constant; the generator asserts that all 16 rows of every fragment lie inside
///    the array ([`gen_coop`]). No cooperative load or store touches a storage buffer, so the missing bounds
///    checks on cooperative accesses cannot reach outside a binding: global memory is read and written only by
///    ordinary WGSL loads and stores, which wgpu bounds-checks (robust buffer access on Vulkan).
/// 3. Every `coopLoadT`, `coopMultiplyAdd` and `coopStoreT` runs in workgroup-uniform control flow: the trip
///    counts and conditions around them depend only on uniform parameters, never on the invocation, as
///    subgroup-scope operations require. Workgroups are 256 invocations, a multiple of the subgroup size, and
///    which fragments a subgroup owns comes from `@builtin(subgroup_id)`, so it is uniform within the subgroup.
/// 4. A device must pass [`self_test`] (the generated kernels against a CPU reference, plus a probe) before the
///    kernels are used, so a driver that mis-compiles them is caught at start-up; on failure the experimental
///    device is dropped and the context is rebuilt on a plain one.
/// 5. No other code gets the experimental device through a safe API: `GpuContext::device()` is `None` on such a
///    context, and the raw device is only reachable through the `unsafe` `GpuContext::device_unchecked`, whose
///    contract is conditions 2 and 3.
pub(crate) fn experimental_opt_in() -> wgpu::ExperimentalFeatures {
    // SAFETY: the only experimental feature FOLO requests is `EXPERIMENTAL_COOPERATIVE_MATRIX`, and it is used
    // only by this module's kernels under conditions 1-4 above.
    unsafe { wgpu::ExperimentalFeatures::enabled() }
}

static PROBE: KernelDef = KernelDef {
    name: "coop_probe",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(
        "@group(0) @binding(1) var<storage, read> GA: array<f16>;\n\
         @group(0) @binding(2) var<storage, read> GB: array<f16>;\n\
         @group(0) @binding(3) var<storage, read_write> GC: array<f32>;\n\
         var<workgroup> sa: array<f16, 256>;\n\
         var<workgroup> sb: array<f16, 256>;\n\
         var<workgroup> sc: array<f32, 256>;\n\
         @compute @workgroup_size(32)\n\
         fn main(@builtin(local_invocation_index) lid: u32) {\n\
           for (var i = lid; i < 256u; i += 32u) { sa[i] = GA[i]; sb[i] = GB[i]; }\n\
           workgroupBarrier();\n\
           let a = coopLoadT<coop_mat16x16<f16, A>>(&sa[0], 16u);\n\
           let b = coopLoadT<coop_mat16x16<f16, B>>(&sb[0], 16u);\n\
           var c = coop_mat16x16<f32, C>();\n\
           c = coopMultiplyAdd(a, b, c);\n\
           coopStoreT(c, &sc[0], 16u);\n\
           workgroupBarrier();\n\
           for (var i = lid; i < 256u; i += 32u) { GC[i] = sc[i]; }\n\
         }\n",
    ),
};

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
}

/// Runs the tensor-core kernels against a CPU reference: a single-fragment probe, then a ragged GEMM through the
/// generated kernel with a row bias and SiLU, f16 and f32 output. Called while creating a context that has
/// cooperative matrices, before they are used. Values are small multiples of powers of two, so the f16 inputs
/// and every product are exact and only the f32 summation order and the output rounding differ.
pub(crate) fn self_test(ctx: &GpuContext) -> Result<()> {
    let a: Vec<f32> = (0..256).map(|i| ((i * 7 % 13) as f32 - 6.0) / 4.0).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0).collect();
    let ga = ctx.upload(&to_f16_bytes(&a))?;
    let gb = ctx.upload(&to_f16_bytes(&b))?;
    let gc = ctx.alloc(256 * 4)?;
    let k = ctx.kernel(&PROBE, Spec::new().f16(true).coop(true))?;
    ctx.dispatch(&k, &[ga.raw(), gb.raw(), gc.raw()], &Params::new(), [1, 1, 1])?;
    let got: Vec<f32> = bytemuck::pod_collect_to_vec(&ctx.read(gc.raw(), 0, 256 * 4)?);
    for i in 0..16 {
        for j in 0..16 {
            let want: f32 = (0..16).map(|k| a[i * 16 + k] * b[k * 16 + j]).sum();
            let g = got[i * 16 + j];
            if (g - want).abs() > 1e-3 * (1.0 + want.abs()) {
                return Err(GpuError::Wgpu(format!("probe C[{i}][{j}] = {g}, expected {want}")));
            }
        }
    }

    // A ragged GEMM through the generated kernel: every tile edge, the bias and the activation.
    let (m, n, kd) = (70usize, 90usize, 75usize);
    let a: Vec<f32> = (0..m * kd).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
    let b: Vec<f32> = (0..kd * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 16.0).collect();
    let bias: Vec<f32> = (0..m).map(|i| (i % 5) as f32 / 4.0 - 0.5).collect();
    let mut d = GemmDesc::new(m, n, kd, false, false, 1);
    d.dtype = DType::F16;
    d.bias = GemmBias::Row;
    d.act = crate::UnaryOp::Silu;
    let (ga, gb) = (ctx.upload(&to_f16_bytes(&a))?, ctx.upload(&to_f16_bytes(&b))?);
    for out in [DType::F32, DType::F16] {
        let gbias = if out == DType::F16 {
            ctx.upload(&to_f16_bytes(&bias))?
        } else {
            ctx.upload(bytemuck::cast_slice(&bias))?
        };
        let gc = ctx.alloc(out.bytes(m * n))?;
        let built = encode_gemm(ctx, &d, out, ga.raw(), gb.raw(), gbias.raw(), gc.raw(), None)?;
        if !built {
            // `build` recorded the compiler's message in the status.
            return Err(GpuError::Wgpu(ctx.tensor_cores_status()));
        }
        let bytes = ctx.read(gc.raw(), 0, out.bytes(m * n))?;
        let got: Vec<f32> = match out {
            DType::F16 => {
                bytes.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect()
            }
            _ => bytemuck::pod_collect_to_vec(&bytes),
        };
        for i in 0..m {
            for j in 0..n {
                let z: f32 = (0..kd).map(|k| a[i * kd + k] * b[k * n + j]).sum::<f32>() + bias[i];
                let want = z / (1.0 + (-z).exp());
                let g = got[i * n + j];
                // f16 output: one rounding (2^-11 relative); f32: summation order and the SiLU's `exp`.
                let tol = if out == DType::F16 { 2e-3 } else { 1e-4 };
                let close = (g - want).abs() <= tol * (1.0 + want.abs()); // false for NaN
                if !close {
                    return Err(GpuError::Wgpu(format!("{out:?} GEMM C[{i}][{j}] = {g}, expected {want}")));
                }
            }
        }
    }
    Ok(())
}
