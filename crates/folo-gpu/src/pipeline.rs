//! Kernel definitions, specialisation constants, the per-dispatch parameter block and pipeline compilation.
//!
//! A kernel is a [`KernelDef`]: a name, the access mode of each storage binding and a WGSL source (static or
//! generated). A [`Spec`] is a list of specialisation constants; it is rendered as WGSL `const` declarations
//! in front of the source, so constants can size workgroup arrays, pick code paths and fold away bounds
//! checks. The pipeline cache in [`crate::GpuContext`] is keyed by `(kernel name, spec)`.
//!
//! Every kernel sees the same bind group layout shape: `@binding(0)` is a 256-byte uniform parameter block
//! ([`Params`], read with `pu(i)` / `pf(i)` / `pi(i)` from [`PRELUDE`]) and `@binding(1..)` are the storage
//! buffers in [`KernelDef::storage`] order.

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

/// Specialisation constants plus the f16 switch (`enable f16;`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    vals: Vec<(&'static str, SpecVal)>,
    f16: bool,
    coop: bool,
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
        parts.join(" ")
    }

    fn header(&self) -> String {
        let mut s = String::new();
        // Directives first: WGSL rejects `enable` after any declaration.
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
#[derive(Clone)]
pub(crate) struct Kernel {
    pub name: &'static str,
    /// The specialisation constants, for profiles ([`Spec::describe`]).
    pub spec: String,
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    pub storage: &'static [Access],
}

/// Full WGSL for `def` specialised with `spec` (exposed for tests and debugging).
pub(crate) fn render(def: &KernelDef, spec: &Spec) -> String {
    let body = match &def.source {
        Source::Static(s) => (*s).to_string(),
        Source::Generated(g) => g(spec),
    };
    format!("{}{}{}", spec.header(), PRELUDE, body)
}

impl Kernel {
    /// Compiles `def` specialised with `spec`, through the persistent pipeline cache when there is one.
    pub fn compile(
        device: &wgpu::Device,
        def: &'static KernelDef,
        spec: &Spec,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Result<Self> {
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
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
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
        Ok(Self { name: def.name, spec: spec.describe(), pipeline, layout, storage: def.storage })
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
