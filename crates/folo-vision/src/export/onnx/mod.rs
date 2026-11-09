//! ONNX export of a fused [`FoloModel`] (`folo export --format onnx`, docs/CLI.md).
//!
//! The graph is opset 17 (IR version 8) in the default domain, written with the small protobuf encoder in
//! [`proto`]; no ONNX library is involved. It mirrors [`FoloModel::forward`] op for op:
//!
//! | FOLO | ONNX |
//! |---|---|
//! | fused `Conv` (conv + bias) | `Conv` (`kernel_shape`, `strides`, `pads`, `dilations`, `group`) |
//! | SiLU | `Sigmoid` then `Mul` (opset 17 has no SiLU op; runtimes fuse the pair) |
//! | ReLU / LeakyReLU / Sigmoid / Tanh / Hardswish | `Relu` / `LeakyRelu` / `Sigmoid` / `Tanh` / `HardSwish` |
//! | SPPF max pool | `MaxPool` (padding counts as −∞ in ONNX too) |
//! | Upsample ×2 nearest | `Resize` (`nearest`, `asymmetric`, `floor`), scales `[1, 1, 2, 2]` |
//! | `cat` / `split` / residual add | `Concat` / `Split` / `Add` |
//! | Attention | `Reshape`, `Split`, `Transpose`, `MatMul`, `Mul` (scale), `Softmax`, `Shape` |
//! | head flatten | `Reshape` to `[B, C, −1]`, `Concat` along the anchor axis |
//! | `--decode` (MODEL.md §6.4) | `Reshape`, `Softmax`, `Mul` + `ReduceSum` (DFL), `Split`, `Sub` / `Add` / `Mul` (anchors), `Sigmoid`, `Concat`, `Transpose` |
//! | `--decode`, `o2o` head (§6.6 top-k) | `ReduceMax`, `TopK` (twice), `GatherND` (`batch_dims = 1`), `Div` / `Mod`, `Cast`, `Unsqueeze`, `Concat` |
//!
//! The input is `images: [B, 3, H, W]`, RGB in `[0, 1]` (docs/MODEL.md conventions). Outputs are either the raw
//! head (`box_raw: [B, 4·reg_max, A]`, `cls_raw: [B, nc, A]`, MODEL.md §6.3) or, with
//! [`OnnxOptions::decode`], `detections: [B, A, 4 + nc]` (`x1, y1, x2, y2` in input pixels, then the class
//! scores after the sigmoid; NMS is left to the caller). With [`OnnxOptions::dynamic`] the batch, height and
//! width are symbolic (`batch`, `height`, `width`; H and W must be multiples of 32 at run time) and the decode
//! computes its anchor grid in the graph; otherwise everything is fixed and the anchors are constants.
//!
//! A model whose [`FoloModel::head_mode`] is `o2o` (a `head = "o2o"` model, docs/MODEL.md §6.6) exports its
//! one-to-one branch only (the one-to-many tensors are dropped), and its decoded output is end to end, with no
//! NMS needed: `detections: [B, K, 6]` with `K = min(max_det, A)`, rows `x1, y1, x2, y2, score, class` sorted
//! by score (the top-k selection of §6.6: the `K` anchors with the highest max-class score, then the `K` best
//! (anchor, class) pairs among them). The caller applies its confidence threshold. These rows are `F32` in an
//! [`OnnxOptions::half`] graph too (class ids above 2048 have no exact `F16`).
//!
//! Every model tensor becomes an initializer under its `.folo` name (`backbone.stem.conv.weight`, ...), in `F32`
//! or, with [`OnnxOptions::half`], `F16` (then the input, outputs and every activation are `F16` too). Anything
//! the exporter cannot map (an unfused BatchNorm, a tensor no node consumed) is an [`Error::Unsupported`]
//! naming it. The class names and FOLO metadata go into `metadata_props` (see [`OnnxExport::metadata`]).

pub mod proto;

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use folo_core::{DType, Tensor, f16};
use folo_nn::io::{ModelFile, Purpose};
use folo_nn::{Activation, Conv2d, ConvBnAct, Module};

use crate::model::blocks::{Attention, Bottleneck, C2Psa, C3k, C3k2, C3k2Block, PsaBlock, Sppf};
use crate::model::head::LEVELS;
use crate::model::{FoloModel, HeadMode};
use crate::{Error, Result};
use proto::{AttrValue, DataType, Dim, Msg};

/// The ONNX opset the exporter targets.
pub const OPSET: i64 = 17;
/// The IR version that goes with [`OPSET`].
pub const IR_VERSION: i64 = 8;
/// The input's name.
pub const INPUT: &str = "images";

/// What to export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxOptions {
    /// Input height in pixels (a positive multiple of 32). With `dynamic`, only recorded in the metadata.
    pub height: u32,
    /// Input width in pixels (a positive multiple of 32). With `dynamic`, only recorded in the metadata.
    pub width: u32,
    /// The fixed batch size (ignored with `dynamic`).
    pub batch: usize,
    /// Symbolic batch, height and width.
    pub dynamic: bool,
    /// `F16` weights, input, activations and outputs.
    pub half: bool,
    /// Decode inside the graph: one `detections: [B, A, 4 + nc]` output instead of the raw head (`[B, K, 6]`
    /// end-to-end detections for an `o2o` head).
    pub decode: bool,
    /// `o2o` head with `decode`: the detections per image, `K = min(max_det, A)` (300 by default).
    pub max_det: usize,
    /// [`export_file`]: the head branch to export, `None` for the file's ([`FoloModel::head_mode`]).
    /// [`export`] exports the model's own mode.
    pub head: Option<HeadMode>,
    /// Extra `metadata_props` (for example the source file's `folo.metrics`); the exporter's own keys win.
    pub metadata: BTreeMap<String, String>,
}

impl OnnxOptions {
    /// A square `imgsz × imgsz`, batch 1, `F32`, raw-output export.
    pub fn new(imgsz: u32) -> Self {
        OnnxOptions {
            height: imgsz,
            width: imgsz,
            batch: 1,
            dynamic: false,
            half: false,
            decode: false,
            max_det: 300,
            head: None,
            metadata: BTreeMap::new(),
        }
    }

    fn validate(&self, stride: u32) -> Result<()> {
        let bad = |s: String| Err(Error::Config(s));
        for (what, v) in [("height", self.height), ("width", self.width)] {
            if v == 0 || !v.is_multiple_of(stride) {
                return bad(format!("ONNX export: {what} {v} must be a positive multiple of {stride}"));
            }
        }
        if self.batch == 0 {
            return bad("ONNX export: batch must be at least 1".into());
        }
        if self.max_det == 0 {
            return bad("ONNX export: max_det must be at least 1".into());
        }
        Ok(())
    }
}

/// A graph input or output: its name, element type and shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxValue {
    /// The value's name.
    pub name: String,
    /// `float` or `float16`.
    pub dtype: &'static str,
    /// The shape; symbolic dimensions are named.
    pub shape: Vec<Dim>,
}

/// An encoded ONNX model and what it contains.
pub struct OnnxExport {
    /// The serialized `ModelProto`.
    pub bytes: Vec<u8>,
    /// The graph inputs (`images`).
    pub inputs: Vec<OnnxValue>,
    /// The graph outputs (`box_raw`, `cls_raw`, or `detections`).
    pub outputs: Vec<OnnxValue>,
    /// The op type of every node, in graph order.
    pub ops: Vec<String>,
    /// The number of initializers (model tensors plus constants).
    pub initializers: usize,
    /// The `metadata_props` written.
    pub metadata: BTreeMap<String, String>,
}

impl OnnxExport {
    /// Write the model to `path`, replacing it atomically: the bytes go to a fresh temporary file next to it
    /// (`<path>.<n>.tmp`, never an existing file) that is then renamed over `path`.
    pub fn write_to(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(format!(".{}", std::process::id()));
        let (tmp, f) = (0u32..)
            .find_map(|n| {
                let mut nm = name.clone();
                nm.push(format!(".{n}.tmp"));
                let p = path.with_file_name(nm);
                match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
                    Ok(f) => Some(Ok((p, f))),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                    Err(e) => Some(Err(Error::io(&p, e))),
                }
            })
            .expect("some temporary name is free")?;
        // `write_closed` takes the file and closes it before the rename, which Windows needs.
        let done = write_closed(f, &self.bytes);
        if let Err(e) = done.and_then(|()| std::fs::rename(&tmp, path)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(Error::io(path, e));
        }
        Ok(())
    }
}

/// Write `bytes` to `f`, flush them to disk and close the file.
fn write_closed(mut f: std::fs::File, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(bytes)?;
    f.sync_all()
}

impl FoloModel {
    /// Export this model as an ONNX graph (see [`crate::export::onnx`]). The model must be fused
    /// ([`FoloModel::fuse`]); weights are read from any device.
    pub fn to_onnx(&self, opts: &OnnxOptions) -> Result<OnnxExport> {
        export(self, opts)
    }
}

/// `folo export --format onnx`: build the inference model a `.folo` file describes (its EMA weights when present,
/// docs/FORMAT.md loading rules) on the CPU, fuse it in `F32` and export it.
pub fn export_file(file: &ModelFile, opts: &OnnxOptions) -> Result<OnnxExport> {
    let (mut model, _) = FoloModel::from_file(file, Purpose::Inference)?;
    if let Some(h) = opts.head {
        model.set_head_mode(h).map_err(|e| Error::Config(format!("ONNX export: {e}")))?;
    }
    model.fuse()?;
    export(&model, opts)
}

/// Build the ONNX model of a fused `model`.
pub fn export(model: &FoloModel, opts: &OnnxOptions) -> Result<OnnxExport> {
    if !model.is_fused() {
        return Err(Error::Unsupported(
            "ONNX export needs a fused model (Conv+BN folded): call FoloModel::fuse first".into(),
        ));
    }
    let cfg = model.config();
    let e2e = model.head_mode() == HeadMode::O2o;
    let stride = *model.strides().iter().max().expect("three strides");
    opts.validate(stride)?;

    let elem = if opts.half { DataType::Float16 } else { DataType::Float };
    let mut g = Graph::new(elem);
    let (bdim, hdim, wdim) = if opts.dynamic {
        (Dim::Param("batch".into()), Dim::Param("height".into()), Dim::Param("width".into()))
    } else {
        (Dim::Value(opts.batch), Dim::Value(opts.height as usize), Dim::Value(opts.width as usize))
    };
    let input = OnnxValue {
        name: INPUT.into(),
        dtype: elem.name(),
        shape: vec![bdim.clone(), Dim::Value(cfg.in_channels), hdim, wdim],
    };

    let feats = g.body(model, INPUT)?;
    let (box_raw, cls_raw) = g.head(model, &feats, !opts.decode)?;
    let (reg_max, nc) = (model.head.reg_max, model.num_classes());
    let a = if opts.dynamic {
        Dim::Param("anchors".into())
    } else {
        Dim::Value(model.anchors(opts.height as usize, opts.width as usize).len())
    };
    let outputs = if opts.decode && e2e {
        let k = match &a {
            Dim::Value(a) => Dim::Value(opts.max_det.min(*a)),
            Dim::Param(_) => Dim::Param("max_det".into()),
        };
        let all = g.decode(model, opts, &feats, &box_raw, &cls_raw)?;
        g.topk(&all, nc, opts);
        // Always f32: an f16 class id is exact only up to 2048.
        let dtype = DataType::Float.name();
        vec![OnnxValue { name: "detections".into(), dtype, shape: vec![bdim, k, Dim::Value(6)] }]
    } else if opts.decode {
        let all = g.decode(model, opts, &feats, &box_raw, &cls_raw)?;
        g.transpose("detections", &all, &[0, 2, 1]);
        vec![OnnxValue {
            name: "detections".into(),
            dtype: elem.name(),
            shape: vec![bdim, a, Dim::Value(4 + nc)],
        }]
    } else {
        vec![
            OnnxValue {
                name: box_raw,
                dtype: elem.name(),
                shape: vec![bdim.clone(), Dim::Value(4 * reg_max), a.clone()],
            },
            OnnxValue { name: cls_raw, dtype: elem.name(), shape: vec![bdim, Dim::Value(nc), a] },
        ]
    };

    // Every model tensor must have been consumed by exactly the node that maps it, except the head branch that
    // does not predict.
    let dropped = |n: &str| {
        if e2e {
            n.starts_with("head.box.") || n.starts_with("head.cls.")
        } else {
            n.starts_with("head.o2o.")
        }
    };
    for t in model.named_tensors() {
        if !g.used.contains(&t.name) && !dropped(&t.name) {
            return Err(Error::Unsupported(format!("ONNX export: tensor `{}` is not mapped", t.name)));
        }
    }

    let metadata = metadata(model, opts, &outputs)?;
    let ops = g.nodes.iter().map(|n| n.op.to_string()).collect();
    let initializers = g.inits.len();
    let bytes = g.finish(&input, &outputs, &metadata);
    Ok(OnnxExport { bytes, inputs: vec![input], outputs, ops, initializers, metadata })
}

/// The `metadata_props`: the model's `folo.*` weights metadata (`folo.dtype` the export's, `folo.fused`
/// `"true"`), `names` (`{"0": "<class 0>", ...}`) and `folo.onnx.*`: `outputs` (`"raw"`, `"decoded"`, or
/// `"end2end"` for a decoded `o2o` head), `head` (`"o2m"` or `"o2o"`, the exported branch), `max_det` (end to
/// end only), `imgsz` (`[H, W]`), `batch` (a number, or `"dynamic"`), `dynamic`, `opset`, `strides`, `reg_max`.
fn metadata(
    model: &FoloModel,
    opts: &OnnxOptions,
    outputs: &[OnnxValue],
) -> Result<BTreeMap<String, String>> {
    let mut m = opts.metadata.clone();
    let mut meta = model.metadata()?;
    meta.dtype = if opts.half { folo_nn::io::FloatDType::F16 } else { folo_nn::io::FloatDType::F32 };
    meta.fused = true;
    m.extend(meta.to_map());
    let names: serde_json::Map<String, serde_json::Value> =
        model.classes().iter().enumerate().map(|(i, c)| (i.to_string(), c.clone().into())).collect();
    m.insert("names".into(), serde_json::Value::Object(names).to_string());
    let s = |v: serde_json::Value| v.to_string();
    let e2e = model.head_mode() == HeadMode::O2o;
    let kind = match (opts.decode, e2e) {
        (false, _) => "raw",
        (true, false) => "decoded",
        (true, true) => "end2end",
    };
    m.insert("folo.onnx.outputs".into(), kind.into());
    m.insert("folo.onnx.head".into(), model.head_mode().as_str().into());
    if opts.decode && e2e {
        m.insert("folo.onnx.max_det".into(), opts.max_det.to_string());
    }
    m.insert("folo.onnx.output_names".into(), s(outputs.iter().map(|o| o.name.clone()).collect()));
    m.insert("folo.onnx.imgsz".into(), s(serde_json::json!([opts.height, opts.width])));
    m.insert("folo.onnx.batch".into(), if opts.dynamic { "dynamic".into() } else { opts.batch.to_string() });
    m.insert("folo.onnx.dynamic".into(), opts.dynamic.to_string());
    m.insert("folo.onnx.opset".into(), OPSET.to_string());
    m.insert("folo.onnx.strides".into(), s(serde_json::json!(model.strides())));
    m.insert("folo.onnx.reg_max".into(), model.head.reg_max.to_string());
    Ok(m)
}

struct Node {
    op: &'static str,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<(String, AttrValue)>,
}

/// Attribute list shorthand.
fn attrs<const N: usize>(a: [(&str, AttrValue); N]) -> Vec<(String, AttrValue)> {
    a.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn ints(v: &[i64]) -> AttrValue {
    AttrValue::Ints(v.to_vec())
}

/// The graph under construction.
struct Graph {
    /// The floating element type of weights and activations.
    elem: DataType,
    nodes: Vec<Node>,
    /// Encoded initializers.
    inits: Vec<Msg>,
    /// Every value name (node outputs, initializers, the input), kept unique.
    names: HashSet<String>,
    /// The model tensors consumed so far.
    used: HashSet<String>,
    /// Shared int64 constants, by value.
    consts: BTreeMap<Vec<i64>, String>,
}

impl Graph {
    fn new(elem: DataType) -> Self {
        let mut names = HashSet::new();
        names.insert(INPUT.to_string());
        Graph { elem, nodes: vec![], inits: vec![], names, used: HashSet::new(), consts: BTreeMap::new() }
    }

    /// A fresh value name based on `base`.
    fn fresh(&mut self, base: &str) -> String {
        let mut name = base.to_string();
        let mut n = 1;
        while !self.names.insert(name.clone()) {
            n += 1;
            name = format!("{base}_{n}");
        }
        name
    }

    /// Add a node with `n_out` outputs named after `name`.
    fn node_n(
        &mut self,
        op: &'static str,
        name: &str,
        inputs: &[&str],
        attrs: Vec<(String, AttrValue)>,
        n_out: usize,
    ) -> Vec<String> {
        let outputs: Vec<String> = if n_out == 1 {
            vec![self.fresh(name)]
        } else {
            (0..n_out).map(|i| self.fresh(&format!("{name}:{i}"))).collect()
        };
        let node_name = format!("{name}/{op}");
        self.nodes.push(Node {
            op,
            name: node_name,
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.clone(),
            attrs,
        });
        outputs
    }

    fn node(
        &mut self,
        op: &'static str,
        name: &str,
        inputs: &[&str],
        attrs: Vec<(String, AttrValue)>,
    ) -> String {
        self.node_n(op, name, inputs, attrs, 1).remove(0)
    }

    /// An initializer of the graph's floating type from f32 values.
    fn float_init(&mut self, name: &str, dims: &[usize], data: &[f32]) -> String {
        let name = self.fresh(name);
        let raw: Vec<u8> = match self.elem {
            DataType::Float16 => {
                data.iter().flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes()).collect()
            }
            _ => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        self.inits.push(proto::tensor(&name, self.elem, dims, &raw));
        name
    }

    /// An `F32` initializer, whatever the graph's type (`Resize` scales are always float).
    fn f32_init(&mut self, name: &str, dims: &[usize], data: &[f32]) -> String {
        let name = self.fresh(name);
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.inits.push(proto::tensor(&name, DataType::Float, dims, &raw));
        name
    }

    /// A shared `int64` 1-D constant (`scalar` for a 0-d one).
    fn i64_const(&mut self, values: &[i64], scalar: bool) -> String {
        let mut key = values.to_vec();
        key.push(scalar as i64); // scalars and 1-element vectors are distinct constants
        if let Some(n) = self.consts.get(&key) {
            return n.clone();
        }
        let tag = values.iter().map(|v| v.to_string().replace('-', "m")).collect::<Vec<_>>().join("_");
        let name = self.fresh(&format!("const.i64.{tag}"));
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dims: &[usize] = if scalar { &[] } else { &[values.len()] };
        self.inits.push(proto::tensor(&name, DataType::Int64, dims, &raw));
        self.consts.insert(key, name.clone());
        name
    }

    /// A model tensor as an initializer under its `.folo` name.
    fn weight(&mut self, name: &str, t: &Tensor) -> Result<String> {
        if !self.used.insert(name.to_string()) {
            return Err(Error::Unsupported(format!("ONNX export: tensor `{name}` mapped twice")));
        }
        let data = t.to_dtype(DType::F32)?.to_vec::<f32>()?;
        if self.names.contains(name) {
            return Err(Error::Unsupported(format!("ONNX export: name `{name}` clashes with a graph value")));
        }
        Ok(self.float_init(name, t.dims(), &data))
    }

    // ----- layers ---------------------------------------------------------------------------------------

    /// A bare `Conv2d` with optional bias (`p` is its prefix: `p.weight`, `p.bias`).
    fn conv2d(&mut self, p: &str, c: &Conv2d, x: &str) -> Result<String> {
        let cfg = c.config();
        let w = self.weight(&format!("{p}.weight"), &c.weight)?;
        let mut inputs = vec![x.to_string(), w];
        if let Some(b) = &c.bias {
            inputs.push(self.weight(&format!("{p}.bias"), b)?);
        }
        let (k, s, pd, d) = (cfg.kernel as i64, cfg.stride as i64, cfg.padding as i64, cfg.dilation as i64);
        let a = attrs([
            ("kernel_shape", ints(&[k, k])),
            ("strides", ints(&[s, s])),
            ("pads", ints(&[pd, pd, pd, pd])),
            ("dilations", ints(&[d, d])),
            ("group", AttrValue::Int(cfg.groups as i64)),
        ]);
        let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
        Ok(self.node("Conv", p, &refs, a))
    }

    /// A fused `Conv` block (`p.conv.*`) and its activation.
    fn conv(&mut self, p: &str, c: &ConvBnAct, x: &str) -> Result<String> {
        if c.bn.is_some() {
            return Err(Error::Unsupported(format!(
                "ONNX export: `{p}` still has its BatchNorm (export a fused model)"
            )));
        }
        let y = self.conv2d(&format!("{p}.conv"), &c.conv, x)?;
        self.act(p, c.act, &y)
    }

    fn act(&mut self, p: &str, act: Activation, x: &str) -> Result<String> {
        Ok(match act {
            Activation::Identity => x.to_string(),
            Activation::Silu => {
                let s = self.node("Sigmoid", &format!("{p}.sigmoid"), &[x], vec![]);
                self.node("Mul", &format!("{p}.silu"), &[x, &s], vec![])
            }
            Activation::Relu => self.node("Relu", &format!("{p}.relu"), &[x], vec![]),
            Activation::LeakyRelu(a) => self.node(
                "LeakyRelu",
                &format!("{p}.leaky_relu"),
                &[x],
                attrs([("alpha", AttrValue::Float(a))]),
            ),
            Activation::Sigmoid => self.node("Sigmoid", &format!("{p}.sigmoid"), &[x], vec![]),
            Activation::Tanh => self.node("Tanh", &format!("{p}.tanh"), &[x], vec![]),
            Activation::Hardswish => self.node("HardSwish", &format!("{p}.hardswish"), &[x], vec![]),
        })
    }

    fn concat(&mut self, name: &str, xs: &[&str], axis: i64) -> String {
        self.node("Concat", name, xs, attrs([("axis", AttrValue::Int(axis))]))
    }

    fn split(&mut self, name: &str, x: &str, sizes: &[i64], axis: i64) -> Vec<String> {
        let s = self.i64_const(sizes, false);
        self.node_n("Split", name, &[x, &s], attrs([("axis", AttrValue::Int(axis))]), sizes.len())
    }

    fn reshape(&mut self, name: &str, x: &str, shape: &[i64]) -> String {
        let s = self.i64_const(shape, false);
        self.node("Reshape", name, &[x, &s], vec![])
    }

    fn transpose(&mut self, name: &str, x: &str, perm: &[i64]) -> String {
        self.node("Transpose", name, &[x], attrs([("perm", ints(perm))]))
    }

    fn add(&mut self, name: &str, a: &str, b: &str) -> String {
        self.node("Add", name, &[a, b], vec![])
    }

    fn bottleneck(&mut self, p: &str, b: &Bottleneck, x: &str) -> Result<String> {
        let y = self.conv(&format!("{p}.cv1"), &b.cv1, x)?;
        let y = self.conv(&format!("{p}.cv2"), &b.cv2, &y)?;
        Ok(if b.residual { self.add(&format!("{p}.add"), x, &y) } else { y })
    }

    fn c3k(&mut self, p: &str, c: &C3k, x: &str) -> Result<String> {
        let mut a = self.conv(&format!("{p}.cv1"), &c.cv1, x)?;
        for (j, b) in c.blocks.iter().enumerate() {
            a = self.bottleneck(&format!("{p}.block{j}"), b, &a)?;
        }
        let b = self.conv(&format!("{p}.cv2"), &c.cv2, x)?;
        let cat = self.concat(&format!("{p}.cat"), &[&a, &b], 1);
        self.conv(&format!("{p}.cv3"), &c.cv3, &cat)
    }

    fn c3k2(&mut self, p: &str, c: &C3k2, x: &str) -> Result<String> {
        let y = self.conv(&format!("{p}.cv1"), &c.cv1, x)?;
        let half = (out_channels(&c.cv1) / 2) as i64;
        let mut parts = self.split(&format!("{p}.split"), &y, &[half, half], 1);
        for (i, b) in c.blocks.iter().enumerate() {
            let prev = parts.last().expect("two halves").clone();
            let q = format!("{p}.block{i}");
            let next = match b {
                C3k2Block::Bottleneck(b) => self.bottleneck(&q, b, &prev)?,
                C3k2Block::C3k(b) => self.c3k(&q, b, &prev)?,
            };
            parts.push(next);
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        let cat = self.concat(&format!("{p}.cat"), &refs, 1);
        self.conv(&format!("{p}.cv2"), &c.cv2, &cat)
    }

    fn sppf(&mut self, p: &str, s: &Sppf, x: &str) -> Result<String> {
        let y0 = self.conv(&format!("{p}.cv1"), &s.cv1, x)?;
        let (k, pad) = (s.k as i64, (s.k / 2) as i64);
        let a = || {
            attrs([("kernel_shape", ints(&[k, k])), ("strides", ints(&[1, 1])), ("pads", ints(&[pad; 4]))])
        };
        let y1 = self.node("MaxPool", &format!("{p}.pool1"), &[&y0], a());
        let y2 = self.node("MaxPool", &format!("{p}.pool2"), &[&y1], a());
        let y3 = self.node("MaxPool", &format!("{p}.pool3"), &[&y2], a());
        let cat = self.concat(&format!("{p}.cat"), &[&y0, &y1, &y2, &y3], 1);
        self.conv(&format!("{p}.cv2"), &s.cv2, &cat)
    }

    fn attention(&mut self, p: &str, at: &Attention, x: &str) -> Result<String> {
        let (heads, kd, hd) = (at.heads as i64, at.key_dim as i64, at.head_dim as i64);
        let qkv = self.conv(&format!("{p}.qkv"), &at.qkv, x)?;
        let qkv = self.reshape(&format!("{p}.qkv_heads"), &qkv, &[0, heads, 2 * kd + hd, -1]);
        let [q, k, v]: [String; 3] =
            self.split(&format!("{p}.qkv_split"), &qkv, &[kd, kd, hd], 2).try_into().expect("three parts");
        // A = softmax_j(scale · qᵀk): [B, heads, N, N].
        let qt = self.transpose(&format!("{p}.q_t"), &q, &[0, 1, 3, 2]);
        let logits = self.node("MatMul", &format!("{p}.qk"), &[&qt, &k], vec![]);
        let scale = self.float_init(&format!("{p}.scale"), &[], &[(kd as f64).powf(-0.5) as f32]);
        let logits = self.node("Mul", &format!("{p}.qk_scaled"), &[&logits, &scale], vec![]);
        let attn =
            self.node("Softmax", &format!("{p}.softmax"), &[&logits], attrs([("axis", AttrValue::Int(-1))]));
        // o = v · Aᵀ: [B, heads, hd, N], viewed as the input's [B, C, H, W].
        let at_t = self.transpose(&format!("{p}.attn_t"), &attn, &[0, 1, 3, 2]);
        let o = self.node("MatMul", &format!("{p}.av"), &[&v, &at_t], vec![]);
        let shape = self.node("Shape", &format!("{p}.shape"), &[x], vec![]);
        let o = self.node("Reshape", &format!("{p}.o"), &[&o, &shape], vec![]);
        let v = self.node("Reshape", &format!("{p}.v"), &[&v, &shape], vec![]);
        let pe = self.conv(&format!("{p}.pe"), &at.pe, &v)?;
        let sum = self.add(&format!("{p}.o_pe"), &o, &pe);
        self.conv(&format!("{p}.proj"), &at.proj, &sum)
    }

    fn psa_block(&mut self, p: &str, b: &PsaBlock, x: &str) -> Result<String> {
        let a = self.attention(&format!("{p}.attn"), &b.attn, x)?;
        let x = self.add(&format!("{p}.attn_add"), x, &a);
        let y = self.conv(&format!("{p}.ffn1"), &b.ffn1, &x)?;
        let y = self.conv(&format!("{p}.ffn2"), &b.ffn2, &y)?;
        Ok(self.add(&format!("{p}.ffn_add"), &x, &y))
    }

    fn c2psa(&mut self, p: &str, c: &C2Psa, x: &str) -> Result<String> {
        let y = self.conv(&format!("{p}.cv1"), &c.cv1, x)?;
        let half = (out_channels(&c.cv1) / 2) as i64;
        let [a, mut b]: [String; 2] =
            self.split(&format!("{p}.split"), &y, &[half, half], 1).try_into().expect("two halves");
        for (i, blk) in c.blocks.iter().enumerate() {
            b = self.psa_block(&format!("{p}.block{i}"), blk, &b)?;
        }
        let cat = self.concat(&format!("{p}.cat"), &[&a, &b], 1);
        self.conv(&format!("{p}.cv2"), &c.cv2, &cat)
    }

    fn upsample(&mut self, name: &str, x: &str) -> String {
        let scales = self.f32_init("const.f32.upsample_scales", &[4], &[1.0, 1.0, 2.0, 2.0]);
        let a = attrs([
            ("mode", AttrValue::Str("nearest".into())),
            ("coordinate_transformation_mode", AttrValue::Str("asymmetric".into())),
            ("nearest_mode", AttrValue::Str("floor".into())),
        ]);
        self.node("Resize", name, &[x, "", &scales], a)
    }

    /// Layers 0–22 (MODEL.md §4): the P3, P4 and P5 features.
    fn body(&mut self, m: &FoloModel, x: &str) -> Result<[String; 3]> {
        let (b, n) = (&m.backbone, &m.neck);
        let y = self.conv("backbone.stem", &b.stem, x)?;
        let y = self.conv("backbone.down2", &b.down2, &y)?;
        let y = self.c3k2("backbone.stage2", &b.stage2, &y)?;
        let y = self.conv("backbone.down3", &b.down3, &y)?;
        let x4 = self.c3k2("backbone.stage3", &b.stage3, &y)?;
        let y = self.conv("backbone.down4", &b.down4, &x4)?;
        let x6 = self.c3k2("backbone.stage4", &b.stage4, &y)?;
        let y = self.conv("backbone.down5", &b.down5, &x6)?;
        let y = self.c3k2("backbone.stage5", &b.stage5, &y)?;
        let y = self.sppf("backbone.sppf", &b.sppf, &y)?;
        let x10 = self.c2psa("backbone.psa", &b.psa, &y)?;
        let y = self.upsample("neck.up11", &x10);
        let y = self.concat("neck.cat12", &[&y, &x6], 1);
        let x13 = self.c3k2("neck.td4", &n.td4, &y)?;
        let y = self.upsample("neck.up14", &x13);
        let y = self.concat("neck.cat15", &[&y, &x4], 1);
        let x16 = self.c3k2("neck.out3", &n.out3, &y)?;
        let y = self.conv("neck.down3", &n.down3, &x16)?;
        let y = self.concat("neck.cat18", &[&y, &x13], 1);
        let x19 = self.c3k2("neck.out4", &n.out4, &y)?;
        let y = self.conv("neck.down4", &n.down4, &x19)?;
        let y = self.concat("neck.cat21", &[&y, &x10], 1);
        let x22 = self.c3k2("neck.out5", &n.out5, &y)?;
        Ok([x16, x19, x22])
    }

    /// Layer 23: the raw head outputs `(box_raw, cls_raw)` of the predicting branch ([`FoloModel::head_mode`]),
    /// named so when they are graph outputs.
    fn head(&mut self, m: &FoloModel, feats: &[String; 3], are_outputs: bool) -> Result<(String, String)> {
        let h = &m.head;
        let (r4, nc) = ((4 * h.reg_max) as i64, h.num_classes as i64);
        let (levels, root) = match (&h.o2o, m.head_mode()) {
            (Some(o2o), HeadMode::O2o) => (o2o, "head.o2o"),
            _ => (&h.levels, "head"),
        };
        let (mut boxes, mut classes) = (vec![], vec![]);
        for ((l, name), x) in levels.iter().zip(LEVELS).zip(feats) {
            let p = format!("{root}.box.{name}");
            let bb = &l.box_branch;
            let y = self.conv(&format!("{p}.conv0"), &bb.conv0, x)?;
            let y = self.conv(&format!("{p}.conv1"), &bb.conv1, &y)?;
            let y = self.conv2d(&format!("{p}.out"), &bb.out, &y)?;
            boxes.push(self.reshape(&format!("{p}.flat"), &y, &[0, r4, -1]));
            let p = format!("{root}.cls.{name}");
            let cb = &l.cls_branch;
            let y = self.conv(&format!("{p}.dw0"), &cb.dw0, x)?;
            let y = self.conv(&format!("{p}.pw0"), &cb.pw0, &y)?;
            let y = self.conv(&format!("{p}.dw1"), &cb.dw1, &y)?;
            let y = self.conv(&format!("{p}.pw1"), &cb.pw1, &y)?;
            let y = self.conv2d(&format!("{p}.out"), &cb.out, &y)?;
            classes.push(self.reshape(&format!("{p}.flat"), &y, &[0, nc, -1]));
        }
        let names = if are_outputs { ["box_raw", "cls_raw"] } else { ["head.box_raw", "head.cls_raw"] };
        let br: Vec<&str> = boxes.iter().map(String::as_str).collect();
        let cr: Vec<&str> = classes.iter().map(String::as_str).collect();
        let box_raw = self.concat(names[0], &br, 2);
        let cls_raw = self.concat(names[1], &cr, 2);
        Ok((box_raw, cls_raw))
    }

    /// The anchor points `[2, A]` (x row, then y row, in grid units) and strides `[1, A]`: constants for a
    /// fixed size, computed from the feature-map shapes with `dynamic`.
    fn anchors(&mut self, m: &FoloModel, opts: &OnnxOptions, feats: &[String; 3]) -> (String, String) {
        if !opts.dynamic {
            let an = m.anchors(opts.height as usize, opts.width as usize);
            let mut pts: Vec<f32> = an.points.iter().map(|p| p[0]).collect();
            pts.extend(an.points.iter().map(|p| p[1]));
            let a = an.len();
            let points = self.float_init("decode.anchors", &[2, a], &pts);
            let strides = self.float_init("decode.strides", &[1, a], &an.strides);
            return (points, strides);
        }
        let (zero, one) = (self.i64_const(&[0], true), self.i64_const(&[1], true));
        let (two, four) = (self.i64_const(&[2], false), self.i64_const(&[4], false));
        let to = AttrValue::Int(self.elem.code());
        let half = self.float_init("decode.half", &[], &[0.5]);
        let (mut pts, mut strides) = (vec![], vec![]);
        for ((x, name), s) in feats.iter().zip(LEVELS).zip(m.strides()) {
            let p = format!("decode.{name}");
            let shape = self.node("Shape", &format!("{p}.shape"), &[x], vec![]);
            let hw = self.node("Slice", &format!("{p}.hw"), &[&shape, &two, &four], vec![]);
            let [h, w]: [String; 2] = [2i64, 3].map(|i| {
                let idx = self.i64_const(&[i], true);
                self.node(
                    "Gather",
                    &format!("{p}.dim{i}"),
                    &[&shape, &idx],
                    attrs([("axis", AttrValue::Int(0))]),
                )
            });
            // Cell centres along each axis: (0.5, 1.5, …).
            let mut centres = |len: &str, axis: &str| {
                let r = self.node("Range", &format!("{p}.range_{axis}"), &[&zero, len, &one], vec![]);
                let r =
                    self.node("Cast", &format!("{p}.range_{axis}_f"), &[&r], vec![("to".into(), to.clone())]);
                self.node("Add", &format!("{p}.centres_{axis}"), &[&r, &half], vec![])
            };
            let (cx, cy) = (centres(&w, "x"), centres(&h, "y"));
            let ax0 = self.i64_const(&[0], false);
            let ax1 = self.i64_const(&[1], false);
            let cx = self.node("Unsqueeze", &format!("{p}.cx_row"), &[&cx, &ax0], vec![]); // [1, W]
            let cy = self.node("Unsqueeze", &format!("{p}.cy_col"), &[&cy, &ax1], vec![]); // [H, 1]
            let gx = self.node("Expand", &format!("{p}.grid_x"), &[&cx, &hw], vec![]);
            let gy = self.node("Expand", &format!("{p}.grid_y"), &[&cy, &hw], vec![]);
            let gx = self.reshape(&format!("{p}.flat_x"), &gx, &[1, -1]);
            let gy = self.reshape(&format!("{p}.flat_y"), &gy, &[1, -1]);
            pts.push(self.concat(&format!("{p}.points"), &[&gx, &gy], 0));
            let sv = self.float_init(&format!("{p}.stride"), &[1, 1], &[s as f32]);
            let n = self.node("Shape", &format!("{p}.count"), &[&gx], vec![]);
            strides.push(self.node("Expand", &format!("{p}.strides"), &[&sv, &n], vec![]));
        }
        let pr: Vec<&str> = pts.iter().map(String::as_str).collect();
        let sr: Vec<&str> = strides.iter().map(String::as_str).collect();
        (self.concat("decode.anchors", &pr, 1), self.concat("decode.strides", &sr, 1))
    }

    /// MODEL.md §6.4 in the graph: the decoded `[B, 4 + nc, A]` (boxes in input pixels, then class scores).
    fn decode(
        &mut self,
        m: &FoloModel,
        opts: &OnnxOptions,
        feats: &[String; 3],
        box_raw: &str,
        cls_raw: &str,
    ) -> Result<String> {
        let r = m.head.reg_max;
        let logits = self.reshape("decode.dfl_logits", box_raw, &[0, 4, r as i64, -1]); // [B, 4, R, A]
        let prob =
            self.node("Softmax", "decode.dfl_softmax", &[&logits], attrs([("axis", AttrValue::Int(2))]));
        let bins: Vec<f32> = (0..r).map(|i| i as f32).collect();
        let proj = self.float_init("decode.dfl_bins", &[1, 1, r, 1], &bins);
        let weighted = self.node("Mul", "decode.dfl_weighted", &[&prob, &proj], vec![]);
        let axis = self.i64_const(&[2], false);
        let dist = self.node(
            "ReduceSum",
            "decode.dfl_dist",
            &[&weighted, &axis],
            attrs([("keepdims", AttrValue::Int(0))]),
        ); // [B, 4, A]
        let [lt, rb]: [String; 2] =
            self.split("decode.ltrb", &dist, &[2, 2], 1).try_into().expect("two halves");
        let (points, strides) = self.anchors(m, opts, feats);
        let x1y1 = self.node("Sub", "decode.x1y1_cells", &[&points, &lt], vec![]);
        let x2y2 = self.node("Add", "decode.x2y2_cells", &[&points, &rb], vec![]);
        let x1y1 = self.node("Mul", "decode.x1y1", &[&x1y1, &strides], vec![]);
        let x2y2 = self.node("Mul", "decode.x2y2", &[&x2y2, &strides], vec![]);
        let boxes = self.concat("decode.boxes", &[&x1y1, &x2y2], 1); // [B, 4, A]
        let scores = self.node("Sigmoid", "decode.scores", &[cls_raw], vec![]);
        Ok(self.concat("decode.channels", &[&boxes, &scores], 1)) // [B, 4 + nc, A]
    }

    /// MODEL.md §6.6's selection in the graph, from the decoded `all: [B, 4 + nc, A]`: the end-to-end
    /// `detections: [B, K, 6]` output (`x1, y1, x2, y2, score, class`, by descending score). `TopK` puts equal
    /// values in index order, the tie rule of [`crate::nms::topk_detections`]. An `F16` graph selects in `F32`
    /// (the CPU `TopK` and `ReduceMax` of onnxruntime take no `float16`) and its rows stay `F32`, so class ids above
    /// 2048 stay exact.
    fn topk(&mut self, all: &str, nc: usize, opts: &OnnxOptions) {
        let half = self.elem == DataType::Float16;
        let to_f32 = || vec![("to".to_string(), AttrValue::Int(DataType::Float.code()))];
        let dets = self.transpose("e2e.anchor_major", all, &[0, 2, 1]); // [B, A, 4 + nc]
        let dets = if half { self.node("Cast", "e2e.f32", &[&dets], to_f32()) } else { dets };
        let [_, scores]: [String; 2] =
            self.split("e2e.split_anchors", &dets, &[4, nc as i64], 2).try_into().expect("two parts");
        let best = self.node(
            "ReduceMax",
            "e2e.best_score",
            &[&scores],
            attrs([("axes", ints(&[2])), ("keepdims", AttrValue::Int(0))]),
        ); // [B, A]
        // K = min(max_det, A), as a 1-element int64 tensor.
        let max_det = self.i64_const(&[opts.max_det as i64], false);
        let shape = self.node("Shape", "e2e.best_shape", &[&best], vec![]);
        let (one, two) = (self.i64_const(&[1], false), self.i64_const(&[2], false));
        let count = self.node("Slice", "e2e.anchor_count", &[&shape, &one, &two], vec![]);
        let k = self.node("Min", "e2e.k", &[&count, &max_det], vec![]);
        let top = |g: &mut Self, name: &str, x: &str| -> [String; 2] {
            let a = attrs([
                ("axis", AttrValue::Int(-1)),
                ("largest", AttrValue::Int(1)),
                ("sorted", AttrValue::Int(1)),
            ]);
            g.node_n("TopK", name, &[x, &k], a, 2).try_into().expect("values and indices")
        };
        let gather = |g: &mut Self, name: &str, data: &str, idx: &str| -> String {
            let ax = g.i64_const(&[2], false);
            let idx = g.node("Unsqueeze", &format!("{name}.index"), &[idx, &ax], vec![]); // [B, K, 1]
            g.node("GatherND", name, &[data, &idx], attrs([("batch_dims", AttrValue::Int(1))]))
        };
        let [_, anchors] = top(self, "e2e.top_anchors", &best); // [B, K]
        let sel = gather(self, "e2e.selected", &dets, &anchors); // [B, K, 4 + nc]
        let [sel_boxes, sel_scores]: [String; 2] =
            self.split("e2e.split_selected", &sel, &[4, nc as i64], 2).try_into().expect("two parts");
        let flat = self.reshape("e2e.pairs", &sel_scores, &[0, -1]); // [B, K · nc]
        let [score, pair] = top(self, "e2e.top_pairs", &flat); // [B, K]
        let ncs = self.i64_const(&[nc as i64], true);
        let slot = self.node("Div", "e2e.pair_anchor", &[&pair, &ncs], vec![]);
        let class = self.node("Mod", "e2e.pair_class", &[&pair, &ncs], vec![]);
        let class = self.node("Cast", "e2e.class_f32", &[&class], to_f32());
        let boxes = gather(self, "e2e.boxes", &sel_boxes, &slot); // [B, K, 4]
        let ax = self.i64_const(&[2], false);
        let score = self.node("Unsqueeze", "e2e.score_col", &[&score, &ax], vec![]);
        let class = self.node("Unsqueeze", "e2e.class_col", &[&class, &ax], vec![]);
        self.concat("detections", &[&boxes, &score, &class], 2);
    }

    /// Encode the `ModelProto`.
    fn finish(
        self,
        input: &OnnxValue,
        outputs: &[OnnxValue],
        metadata: &BTreeMap<String, String>,
    ) -> Vec<u8> {
        let vi = |v: &OnnxValue, elem| proto::value_info(&v.name, elem, &v.shape);
        let mut graph = Msg::default();
        for n in &self.nodes {
            graph.msg(1, &proto::node(n.op, &n.name, &n.inputs, &n.outputs, &n.attrs));
        }
        graph.string(2, "folo");
        // Each initializer is dropped once copied, so the weights are held about twice at most, not three times.
        let total: usize = self.inits.iter().map(|t| t.0.len() + 10).sum();
        graph.0.reserve(total);
        for t in self.inits {
            graph.msg(5, &t);
        }
        graph.msg(11, &vi(input, self.elem));
        for o in outputs {
            // End-to-end rows are `F32` in any graph.
            let elem = if o.dtype == DataType::Float.name() { DataType::Float } else { self.elem };
            graph.msg(12, &vi(o, elem));
        }

        let mut model = Msg::default();
        model
            .int(1, IR_VERSION)
            .string(2, "folo")
            .string(3, env!("CARGO_PKG_VERSION"))
            .string(4, "ai.folo")
            .int(5, 1)
            .string(6, "FOLO detector exported by `folo export --format onnx` (docs/CLI.md)");
        let mut opset = Msg::default();
        opset.string(1, "").int(2, OPSET);
        model.msg(8, &opset);
        for (k, v) in metadata {
            let mut e = Msg::default();
            e.string(1, k).string(2, v);
            model.msg(14, &e);
        }
        // The graph last, so its (large) body is copied once.
        model.msg_header(7, graph.0.len());
        let mut bytes = model.0;
        bytes.reserve_exact(graph.0.len());
        bytes.extend_from_slice(&graph.0);
        bytes
    }
}

fn out_channels(c: &ConvBnAct) -> usize {
    c.conv.weight.dims()[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    fn fused(scale: &str, nc: usize) -> FoloModel {
        let mut m = FoloModel::new(ModelConfig::new(scale, nc).unwrap()).unwrap();
        m.fuse().unwrap();
        m
    }

    #[test]
    fn needs_a_fused_model() {
        let m = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
        let e = m.to_onnx(&OnnxOptions::new(64)).err().unwrap();
        assert!(e.to_string().contains("fused"), "{e}");
    }

    #[test]
    fn rejects_bad_sizes() {
        let m = fused("n", 2);
        for (h, w, b) in [(0, 64, 1), (64, 100, 1), (64, 64, 0)] {
            let opts = OnnxOptions { height: h, width: w, batch: b, ..OnnxOptions::new(64) };
            assert!(matches!(m.to_onnx(&opts), Err(Error::Config(_))), "{h}x{w} b{b}");
        }
    }

    #[test]
    fn raw_graph_shapes_names_and_metadata() {
        let mut m = fused("n", 3);
        m.set_classes(vec!["a".into(), "b".into(), "c".into()]).unwrap();
        let x = m.to_onnx(&OnnxOptions { batch: 2, ..OnnxOptions::new(64) }).unwrap();
        assert_eq!(x.inputs[0].shape, [2, 3, 64, 64].map(Dim::Value));
        let a = 8 * 8 + 4 * 4 + 2 * 2;
        assert_eq!(x.outputs[0].name, "box_raw");
        assert_eq!(x.outputs[0].shape, [2, 64, a].map(Dim::Value));
        assert_eq!(x.outputs[1].name, "cls_raw");
        assert_eq!(x.outputs[1].shape, [2, 3, a].map(Dim::Value));
        assert_eq!(x.metadata["names"], r#"{"0":"a","1":"b","2":"c"}"#);
        assert_eq!(x.metadata["folo.classes"], r#"["a","b","c"]"#);
        assert_eq!(x.metadata["folo.onnx.outputs"], "raw");
        assert_eq!(x.metadata["folo.fused"], "true");
        // Every op is one the module docs list, and SiLU is Sigmoid * Mul.
        let known = [
            "Conv",
            "Sigmoid",
            "Mul",
            "MaxPool",
            "Resize",
            "Concat",
            "Split",
            "Add",
            "Reshape",
            "Transpose",
            "MatMul",
            "Softmax",
            "Shape",
        ];
        assert!(x.ops.iter().all(|o| known.contains(&o.as_str())), "{:?}", x.ops);
        assert_eq!(x.ops.iter().filter(|o| *o == "Conv").count(), conv_count(&m));
        // One initializer per model tensor, plus a few constants.
        assert!(x.initializers >= m.named_tensors().len());
        assert!(x.bytes.len() > 4 * m.num_parameters());
    }

    fn conv_count(m: &FoloModel) -> usize {
        m.named_tensors().iter().filter(|t| t.name.ends_with(".weight")).count()
    }

    #[test]
    fn half_is_about_half_the_size() {
        let m = fused("n", 2);
        let f = m.to_onnx(&OnnxOptions::new(64)).unwrap();
        let h = m.to_onnx(&OnnxOptions { half: true, ..OnnxOptions::new(64) }).unwrap();
        assert_eq!(h.outputs[0].dtype, "float16");
        assert_eq!(h.metadata["folo.dtype"], "F16");
        let ratio = h.bytes.len() as f64 / f.bytes.len() as f64;
        assert!((0.45..0.6).contains(&ratio), "{ratio}");
    }

    #[test]
    fn decoded_and_dynamic_outputs() {
        let m = fused("n", 4);
        let x = m.to_onnx(&OnnxOptions { decode: true, ..OnnxOptions::new(96) }).unwrap();
        assert_eq!(x.outputs.len(), 1);
        assert_eq!(x.outputs[0].name, "detections");
        assert_eq!(x.outputs[0].shape, [1, 144 + 36 + 9, 8].map(Dim::Value));
        assert!(!x.ops.contains(&"Range".to_string()), "fixed sizes use constant anchors");
        let d = m.to_onnx(&OnnxOptions { decode: true, dynamic: true, ..OnnxOptions::new(96) }).unwrap();
        let p = |s: &str| Dim::Param(s.into());
        assert_eq!(d.inputs[0].shape, [p("batch"), Dim::Value(3), p("height"), p("width")]);
        assert_eq!(d.outputs[0].shape, [p("batch"), p("anchors"), Dim::Value(8)]);
        assert!(d.ops.contains(&"Range".to_string()));
        assert_eq!(d.metadata["folo.onnx.batch"], "dynamic");
    }

    /// An `o2o` model exports one branch (the other's tensors are not initializers), and its decoded output is the
    /// end-to-end `[B, K, 6]`, `F32` even in a half graph (an `F16` class id is exact only up to 2048).
    #[test]
    fn o2o_exports_one_branch_and_end_to_end_rows() {
        let mut cfg = ModelConfig::new("n", 4).unwrap();
        cfg.head = "o2o".into();
        let mut m = FoloModel::new(cfg).unwrap();
        m.fuse().unwrap();
        let o2m = fused("n", 4).to_onnx(&OnnxOptions::new(96)).unwrap();
        let raw = m.to_onnx(&OnnxOptions::new(96)).unwrap();
        assert_eq!(raw.initializers, o2m.initializers, "one head branch, as many tensors as an o2m model");
        assert_eq!(
            (raw.metadata["folo.onnx.head"].as_str(), raw.metadata["folo.onnx.outputs"].as_str()),
            ("o2o", "raw")
        );
        let a = 144 + 36 + 9;
        for (half, max_det, k) in [(false, 300, a), (true, 300, a), (false, 20, 20)] {
            let x = m.to_onnx(&OnnxOptions { decode: true, half, max_det, ..OnnxOptions::new(96) }).unwrap();
            assert_eq!(x.outputs.len(), 1);
            let o = &x.outputs[0];
            assert_eq!(
                (o.name.as_str(), o.dtype, &o.shape),
                ("detections", "float", &[1, k, 6].map(Dim::Value).to_vec())
            );
            assert_eq!(x.metadata["folo.onnx.outputs"], "end2end");
            assert_eq!(x.metadata["folo.onnx.max_det"], max_det.to_string());
            assert_eq!(x.ops.iter().filter(|op| *op == "TopK").count(), 2);
        }
        let d = m.to_onnx(&OnnxOptions { decode: true, dynamic: true, ..OnnxOptions::new(96) }).unwrap();
        assert_eq!(
            d.outputs[0].shape,
            [Dim::Param("batch".into()), Dim::Param("max_det".into()), Dim::Value(6)]
        );
        // `head: o2m` on the model exports its one-to-many branch and the usual decode.
        m.set_head_mode(HeadMode::O2m).unwrap();
        let x = m.to_onnx(&OnnxOptions { decode: true, ..OnnxOptions::new(96) }).unwrap();
        assert_eq!(
            (x.outputs[0].shape[2].clone(), x.metadata["folo.onnx.head"].as_str()),
            (Dim::Value(8), "o2m")
        );
        assert!(OnnxOptions { max_det: 0, ..OnnxOptions::new(96) }.validate(32).is_err());
    }

    #[test]
    fn unfused_block_is_reported_by_name() {
        let mut m = fused("n", 2);
        // Swap one fused conv for an unfused block: the exporter names it instead of writing a wrong graph.
        let mut rng = folo_core::rng::Rng::seed(0);
        m.backbone.down2 = ConvBnAct::new(16, 32, 3, 2, 1, Activation::Silu, &mut rng);
        let e = m.to_onnx(&OnnxOptions::new(64)).err().unwrap().to_string();
        assert!(e.contains("backbone.down2") && e.contains("BatchNorm"), "{e}");
    }
}
