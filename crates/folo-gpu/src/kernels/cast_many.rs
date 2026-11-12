//! Many f32 → f16 casts in one dispatch ([`cast_f16_many`]): autocast's cached f16 weight copies, refreshed after
//! an optimizer step for every weight at once instead of one cast per conv in the next forward.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Tensors one dispatch casts.
const MANY: usize = super::optim::MANY;

/// The binding layout for `j` tensors (`X` read, `Y` read-write per tensor), kept for the process.
fn def(j: usize) -> &'static KernelDef {
    static DEFS: OnceLock<Mutex<HashMap<usize, &'static KernelDef>>> = OnceLock::new();
    let mut defs = DEFS.get_or_init(Default::default).lock().expect("cast defs lock");
    defs.entry(j).or_insert_with(|| {
        let storage: Vec<Access> = (0..j).flat_map(|_| [Access::Read, Access::ReadWrite]).collect();
        let storage = Box::leak(storage.into_boxed_slice());
        Box::leak(Box::new(KernelDef {
            name: "cast_f16_many",
            storage,
            source: Source::Generated(gen_cast_many),
        }))
    })
}

fn gen_cast_many(spec: &Spec) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let mut s = String::from(super::strided::CAST_FNS);
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> X{t}: array<u32>;", 2 * t + 1);
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> Y{t}: array<u32>;", 2 * t + 2);
    }
    // Each invocation writes one word: two f16 (the second zero past an odd end), rounded as `cast_any` rounds.
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let w = linear_id(wid, nwg, 0u, 256u) / 256u;\n\
         \x20 var t = 0u;\n",
    );
    for t in 1..j {
        let _ = writeln!(s, "  if (w >= pu({}u)) {{ t = {t}u; }}", 16 + t);
    }
    s.push_str(
        "  let i = (w - pu(16u + t)) * 256u + lid;\n\
         \x20 let n = pu(32u + t);\n\
         \x20 if (2u * i >= n) { return; }\n\
         \x20 let two = 2u * i + 1u < n;\n\
         \x20 switch t {\n",
    );
    for t in 0..j {
        let _ = writeln!(
            s,
            "  case {t}u: {{ let a = f16bits(bitcast<f32>(X{t}[2u * i]));\n\
             \x20   var b = 0u; if (two) {{ b = f16bits(bitcast<f32>(X{t}[2u * i + 1u])); }}\n\
             \x20   Y{t}[i] = a | (b << 16u); }}"
        );
    }
    s.push_str("  default: {}\n  }\n}\n");
    s
}

/// `ys[i] = f16(xs[i])` for `(x, y, n)` triples (`x` f32, `y` f16, `n` elements), [`MANY`] per dispatch, each
/// value rounded as [`crate::kernels::strided::cast`] rounds it (to nearest even, the `half` crate's NaNs).
pub fn cast_f16_many(ctx: &GpuContext, items: &[(&wgpu::Buffer, &wgpu::Buffer, usize)]) -> Result<()> {
    for &(x, y, n) in items {
        check_buf(x, n, DType::F32, "cast_f16_many input")?;
        check_buf(y, n, DType::F16, "cast_f16_many output")?;
    }
    let live: Vec<_> = items.iter().filter(|it| it.2 > 0).collect();
    let j_max = MANY.min(ctx.caps().max_storage_buffers as usize / 2).max(1);
    for chunk in live.chunks(j_max) {
        let k = ctx.kernel(def(chunk.len()), Spec::new().u32("J", chunk.len() as u32))?;
        let mut p = Params::new().at(16);
        let mut start = 0u64;
        for it in chunk {
            p = p.u(start as u32);
            start += (it.2 as u64).div_ceil(2).div_ceil(256);
        }
        p = p.at(32);
        for it in chunk {
            p = p.uz(it.2)?;
        }
        let bufs: Vec<&wgpu::Buffer> = chunk.iter().flat_map(|it| [it.0, it.1]).collect();
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(start * 256, 256)?)?;
    }
    Ok(())
}
