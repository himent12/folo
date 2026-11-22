//! Kernel definitions, specialisation constants, the per-dispatch parameter block and pipeline compilation.
//!
//! A kernel is a [`KernelDef`]: a name, the access mode of each storage binding and a WGSL source (static or
//! generated). A [`Spec`] is a list of specialisation constants; it is rendered as WGSL `const` declarations
//! in front of the source, so constants can size workgroup arrays, pick code paths and fold away bounds
//! checks. The pipeline cache in [`crate::GpuContext`] is keyed by `(kernel name, spec)`.
//!
//! Every kernel sees the same bind group layout shape: `@binding(0)` is a 256-byte uniform parameter block
//! ([`Params`], read with `pu(i)` / `pf(i)` / `pi(i)` from [`PRELUDE`], bound at a dynamic offset) and
//! `@binding(1..)` are the storage buffers in [`KernelDef::storage`] order.

use crate::error::{GpuError, Result};

/// Bytes in one parameter block (64 words).
pub(crate) const UNIFORM_SLOT: u64 = 256;
const PARAM_WORDS: usize = (UNIFORM_SLOT / 4) as usize;

/// WGSL shared by every kernel: the parameter block and helpers.
pub(crate) const PRELUDE: &str = r#"
struct Params { v: array<vec4<u32>, 16> }
@group(0) @binding(0) var<uniform> P: Params;
fn pu(i: u32) -> u32 { return P.v[i / 4u][i % 4u]; }
fn pi(i: u32) -> i32 { return bitcast<i32>(P.v[i / 4u][i % 4u]); }
fn pf(i: u32) -> f32 { return bitcast<f32>(P.v[i / 4u][i % 4u]); }
// Linear invocation id for a 1-D launch folded into x/y (see GpuContext::grid_1d).
fn linear_id(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, wg: u32) -> u32 {
    return (wid.y * nwg.x + wid.x) * wg + lid;
}
"#;

/// Access mode of a storage binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// `var<storage, read>`
    Read,
    /// `var<storage, read_write>`
    ReadWrite,
}

/// Where a kernel's WGSL comes from.
pub(crate) enum Source {
    /// A fixed source; specialisation constants are prepended.
    Static(&'static str),
    /// A generator that receives the spec (constants are still prepended).
    Generated(fn(&Spec) -> String),
}

/// A kernel: name, storage bindings and source.
pub(crate) struct KernelDef {
    pub name: &'static str,
    pub storage: &'static [Access],
    pub source: Source,
}

/// A specialisation constant value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SpecVal {
    U32(u32),
    I32(i32),
    Bool(bool),
    /// An `f32` stored by bit pattern so the key stays hashable.
    F32(u32),
}

/// Specialisation constants plus the f16 (`enable f16;`), cooperative-matrix and subgroup switches.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    vals: Vec<(&'static str, SpecVal)>,
    f16: bool,
    coop: bool,
    subgroups: bool,
}

impl Spec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(mut self, name: &'static str, v: u32) -> Self {
        self.vals.push((name, SpecVal::U32(v)));
        self
    }

    pub fn i32(mut self, name: &'static str, v: i32) -> Self {
        self.vals.push((name, SpecVal::I32(v)));
        self
    }

    pub fn bool(mut self, name: &'static str, v: bool) -> Self {
        self.vals.push((name, SpecVal::Bool(v)));
        self
    }

    #[allow(dead_code)]
    pub fn f32(mut self, name: &'static str, v: f32) -> Self {
        self.vals.push((name, SpecVal::F32(v.to_bits())));
        self
    }

    /// Enables `shader-f16` (`enable f16;`) and defines `T` as `f16`; otherwise `T` is `f32`.
    pub fn f16(mut self, on: bool) -> Self {
        self.f16 = on;
        self
    }

    /// Enables wgpu's cooperative matrices (`enable wgpu_cooperative_matrix;`) for the tensor-core kernels.
    /// Only valid on a device created with them (see `kernels::coop`).
    pub fn coop(mut self, on: bool) -> Self {
        self.coop = on;
        self
    }

    /// Enables WGSL subgroup built-ins (`enable subgroups;`). Only valid when [`crate::Capabilities::subgroups`]
    /// is true on the device.
    #[allow(dead_code)]
    pub fn subgroups(mut self, on: bool) -> Self {
        self.subgroups = on;
        self
    }

    /// Looks up a constant (for generators).
    pub fn get(&self, name: &str) -> Option<SpecVal> {
        self.vals.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }

    /// Integer constant or `default`.
    pub fn get_u32(&self, name: &str, default: u32) -> u32 {
        match self.get(name) {
            Some(SpecVal::U32(v)) => v,
            Some(SpecVal::I32(v)) => v as u32,
            _ => default,
        }
    }

    /// Boolean constant or `false`.
    pub fn get_bool(&self, name: &str) -> bool {
        matches!(self.get(name), Some(SpecVal::Bool(true)))
    }

    /// Whether the f16 variant is requested.
    pub fn is_f16(&self) -> bool {
        self.f16
    }

    /// `NAME=value` pairs (and `f16` / `coop`) on one line, for profiles and logs.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = self
            .vals
            .iter()
            .map(|(name, v)| match v {
                SpecVal::U32(x) => format!("{name}={x}"),
                SpecVal::I32(x) => format!("{name}={x}"),
                SpecVal::Bool(x) => format!("{name}={}", u8::from(*x)),
                SpecVal::F32(b) => format!("{name}={}", f32::from_bits(*b)),
            })
            .collect();
        if self.f16 {
            parts.push("f16".into());
        }
        if self.coop {
            parts.push("coop".into());
        }
        if self.subgroups {
            parts.push("subgroups".into());
        }
        parts.join(" ")
    }

    /// The switches and constants as fields of a kernel list line (`crate::kernel_list`): `-`, or a combination of
    /// `h` (f16), `c` (coop), `s` (subgroups), then ` NAME=u5`, ` NAME=i-3`, ` NAME=b1` or ` NAME=f<bits>` per
    /// constant.
    pub(crate) fn encode_fields(&self, out: &mut String) {
        use std::fmt::Write as _;
        if !self.f16 && !self.coop && !self.subgroups {
            out.push('-');
        } else {
            if self.f16 {
                out.push('h');
            }
            if self.coop {
                out.push('c');
            }
            if self.subgroups {
                out.push('s');
            }
        }
        for (name, v) in &self.vals {
            let _ = match v {
                SpecVal::U32(x) => write!(out, " {name}=u{x}"),
                SpecVal::I32(x) => write!(out, " {name}=i{x}"),
                SpecVal::Bool(x) => write!(out, " {name}=b{}", u8::from(*x)),
                SpecVal::F32(b) => write!(out, " {name}=f{b}"),
            };
        }
    }

    /// The spec [`Spec::encode_fields`] wrote as `fields`; `None` for anything malformed (a constant name that is
    /// not a WGSL identifier, a bad value, an integer constant larger in magnitude than `max_int`). `intern` gives
    /// each name its `'static` form (or refuses it).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn decode_fields<'a>(
        mut fields: impl Iterator<Item = &'a str>,
        max_int: u32,
        mut intern: impl FnMut(&str) -> Option<&'static str>,
    ) -> Option<Self> {
        let flag_str = fields.next()?;
        let (mut f16, mut coop, mut subgroups) = (false, false, false);
        if flag_str != "-" {
            if flag_str.is_empty() {
                return None;
            }
            for b in flag_str.bytes() {
                match b {
                    b'h' if !f16 => f16 = true,
                    b'c' if !coop => coop = true,
                    b's' if !subgroups => subgroups = true,
                    _ => return None,
                }
            }
        }
        let mut vals = Vec::new();
        for f in fields {
            let (name, v) = f.split_once('=')?;
            let ident = name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
            if !ident || !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
                return None;
            }
            let (t, x) = v.split_at_checked(1)?;
            let val = match t {
                "u" => SpecVal::U32(x.parse().ok().filter(|&v: &u32| v <= max_int)?),
                "i" => SpecVal::I32(x.parse().ok().filter(|&v: &i32| v.unsigned_abs() <= max_int)?),
                "b" => SpecVal::Bool(match x {
                    "0" => false,
                    "1" => true,
                    _ => return None,
                }),
                "f" => SpecVal::F32(x.parse().ok()?),
                _ => return None,
            };
            vals.push((intern(name)?, val));
        }
        Some(Self { vals, f16, coop, subgroups })
    }

    /// Whether the cooperative-matrix variant is requested.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn is_coop(&self) -> bool {
        self.coop
    }

    /// Whether the subgroup variant is requested.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn is_subgroups(&self) -> bool {
        self.subgroups
    }

    fn header(&self) -> String {
        let mut s = String::new();
        // Directives first: WGSL rejects `enable` after any declaration.
        // Native Naga 30 enables subgroup built-ins automatically via `wgpu::Features::SUBGROUP` and rejects an
        // explicit `enable subgroups;` directive; on Browser WebGPU (`wasm32`), Dawn/Tint/WebKit require it.
        #[cfg(target_arch = "wasm32")]
        if self.subgroups {
            s.push_str("enable subgroups;\n");
        }
        if self.coop {
            s.push_str("enable wgpu_cooperative_matrix;\n");
        }
        if self.f16 {
            s.push_str("enable f16;\nalias T = f16;\n");
        } else {
            s.push_str("alias T = f32;\n");
        }
        for (name, v) in &self.vals {
            let line = match v {
                SpecVal::U32(x) => format!("const {name}: u32 = {x}u;\n"),
                SpecVal::I32(x) => format!("const {name}: i32 = {x}i;\n"),
                SpecVal::Bool(x) => format!("const {name}: bool = {x};\n"),
                SpecVal::F32(b) => format!("const {name}: f32 = bitcast<f32>({b}u);\n"),
            };
            s.push_str(&line);
        }
        s
    }
}

/// Pipeline cache key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct KernelKey {
    pub name: &'static str,
    pub spec: Spec,
}

/// A compiled pipeline and its bind group layout.
pub(crate) struct Kernel {
    pub name: &'static str,
    /// The specialisation constants, for profiles ([`Spec::describe`]).
    pub spec: String,
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    pub storage: &'static [Access],
    /// The context's recording generation this kernel was last offered to the kernel lists being recorded at
    /// (`GpuContext::precompile`): a use at another generation records it in the newer recordings.
    pub recorded: std::sync::atomic::AtomicU64,
}

impl Clone for Kernel {
    fn clone(&self) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        Self {
            name: self.name,
            spec: self.spec.clone(),
            pipeline: self.pipeline.clone(),
            layout: self.layout.clone(),
            storage: self.storage,
            recorded: AtomicU64::new(self.recorded.load(Ordering::Relaxed)),
        }
    }
}

/// Full WGSL for `def` specialised with `spec` (exposed for tests and debugging).
pub(crate) fn render(def: &KernelDef, spec: &Spec) -> String {
    let body = match &def.source {
        Source::Static(s) => (*s).to_string(),
        Source::Generated(g) => g(spec),
    };
    format!("{}{}{}", spec.header(), PRELUDE, body)
}

/// The opt-in compile timer: with `FOLO_GPU_TRACE_COMPILE=1` (read once), every [`Kernel::compile`] adds its
/// `create_shader_module` time (WGSL parsing and validation) and its total time to process-wide totals, printed to
/// stderr every [`trace::EVERY`] pipelines and by [`crate::save_pipeline_caches`] (docs/GPU.md, Pipeline cache).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod trace {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    /// The environment variable that turns the timer on.
    pub const ENV: &str = "FOLO_GPU_TRACE_COMPILE";
    /// Pipelines between two reports.
    pub const EVERY: u64 = 25;
    static ON: OnceLock<bool> = OnceLock::new();
    static COUNT: AtomicU64 = AtomicU64::new(0);
    static MODULE_NS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    /// First uses of a kernel in a context, by how the kernel was got, and the time the requesting threads spent.
    static FIRST_READY: AtomicU64 = AtomicU64::new(0);
    static FIRST_WAITED: AtomicU64 = AtomicU64::new(0);
    static FIRST_COMPILED: AtomicU64 = AtomicU64::new(0);
    static FIRST_NS: AtomicU64 = AtomicU64::new(0);

    /// How the first use of a kernel in a context got its pipeline.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FirstUse {
        /// A precompile worker had finished it.
        Ready,
        /// A precompile worker was compiling it; the requesting thread waited.
        Waited,
        /// The requesting thread compiled it.
        Compiled,
    }

    /// The start of a compile, when the timer is on.
    pub fn start() -> Option<Instant> {
        ON.get_or_init(|| std::env::var(ENV).is_ok_and(|v| v == "1")).then(Instant::now)
    }

    /// Adds one compile that started at `t0` and spent `module` in `create_shader_module`.
    pub fn add(t0: Instant, module: Duration) {
        let total = t0.elapsed();
        MODULE_NS.fetch_add(module.as_nanos() as u64, Ordering::Relaxed);
        TOTAL_NS.fetch_add(total.as_nanos() as u64, Ordering::Relaxed);
        if (COUNT.fetch_add(1, Ordering::Relaxed) + 1).is_multiple_of(EVERY) {
            report();
        }
    }

    /// Adds one first use that started at `t0` on the requesting thread.
    pub fn first_use(t0: Instant, how: FirstUse) {
        FIRST_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match how {
            FirstUse::Ready => &FIRST_READY,
            FirstUse::Waited => &FIRST_WAITED,
            FirstUse::Compiled => &FIRST_COMPILED,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// Prints the totals so far, when the timer is on and has counted a compile.
    pub fn report() {
        let n = COUNT.load(Ordering::Relaxed);
        if ON.get().copied().unwrap_or(false) && n > 0 {
            let s = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 * 1e-9;
            eprintln!(
                "folo-gpu: compiled {n} pipelines, module {:.3} s, total {:.3} s",
                s(&MODULE_NS),
                s(&TOTAL_NS)
            );
            let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
            let (r, w, c) = (n(&FIRST_READY), n(&FIRST_WAITED), n(&FIRST_COMPILED));
            eprintln!(
                "folo-gpu: {} first uses, {:.3} s on the requesting threads ({r} precompiled, {w} waited for a \
                 precompile worker, {c} compiled on demand)",
                r + w + c,
                s(&FIRST_NS)
            );
        }
    }
}

impl Kernel {
    /// Compiles `def` specialised with `spec`, through the persistent pipeline cache when there is one.
    pub fn compile(
        device: &wgpu::Device,
        def: &'static KernelDef,
        spec: &Spec,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = trace::start();
        let src = render(def, spec);
        // Natively, compile errors are caught in an error scope. In the browser the scope's result arrives
        // asynchronously and nothing may block for it, so errors go to the device's uncaptured-error sink
        // and surface at the next readback instead.
        #[cfg(not(target_arch = "wasm32"))]
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(def.name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        #[cfg(not(target_arch = "wasm32"))]
        let module_time = t0.map(|t| t.elapsed());
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                // The slot in the uniform chunk, so one bind group serves every dispatch on the same buffers.
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(UNIFORM_SLOT),
            },
            count: None,
        }];
        for (i, a) in def.storage.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 + 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: *a == Access::Read },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(def.name),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(def.name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(def.name),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                // Kernels initialise the workgroup memory they read.
                zero_initialize_workgroup_memory: false,
            },
            cache,
        });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(GpuError::Wgpu(format!("compiling kernel {} ({spec:?}): {err}", def.name)));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(t0), Some(m)) = (t0, module_time) {
            trace::add(t0, m);
        }
        Ok(Self {
            name: def.name,
            spec: spec.describe(),
            pipeline,
            layout,
            storage: def.storage,
            recorded: Default::default(),
        })
    }
}

/// The per-dispatch parameter block: up to 64 32-bit words, read in WGSL with `pu(i)`, `pi(i)`, `pf(i)`.
#[derive(Clone)]
pub(crate) struct Params {
    words: [u32; PARAM_WORDS],
    len: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self { words: [0; PARAM_WORDS], len: 0 }
    }
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u(mut self, v: u32) -> Self {
        self.words[self.len] = v;
        self.len += 1;
        self
    }

    /// Pushes a `usize`, failing if it does not fit in 32 bits.
    pub fn uz(self, v: usize) -> Result<Self> {
        let v = u32::try_from(v)
            .map_err(|_| GpuError::Limit(format!("{v} does not fit in a u32 kernel parameter")))?;
        Ok(self.u(v))
    }

    pub fn f(self, v: f32) -> Self {
        self.u(v.to_bits())
    }

    /// Pads to word `at` (so array-like parameters can start at a fixed index).
    pub fn at(mut self, at: usize) -> Self {
        assert!(at >= self.len && at <= PARAM_WORDS);
        self.len = at;
        self
    }

    pub fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }
}
