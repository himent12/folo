//! ONNX export of a fused [`FoloModel`] or [`crate::model::v2::FoloV2Model`] (`folo export --format onnx`,
//! docs/CLI.md).
//!
//! The graph is opset 17 (IR version 8) in the default domain, written with the small protobuf encoder in
//! [`proto`]; no ONNX library is involved. It mirrors [`FoloModel::forward`] and
//! [`crate::model::v2::FoloV2Model::forward`] op for op:
//!
//! | FOLO | ONNX |
//! |---|---|
//! | fused `Conv` / `WsQARepConv3x3` | `Conv` (`kernel_shape`, `strides`, `pads`, `dilations`, `group`) |
//! | SiLU | `Sigmoid` then `Mul` (opset 17 has no SiLU op; runtimes fuse the pair) |
//! | ReLU / LeakyReLU / Sigmoid / Tanh / Hardswish | `Relu` / `LeakyRelu` / `Sigmoid` / `Tanh` / `HardSwish` |
//! | SPPF max pool / `PeakSuppress` 3×3 max | `MaxPool` (padding counts as −∞ in ONNX too) |
//! | Upsample ×2 nearest | `Resize` (`nearest`, `asymmetric`, `floor`), scales `[1, 1, 2, 2]` |
//! | `space_to_depth_2x2` (v2) | `SpaceToDepth` (`blocksize = 2`) |
//! | `cat` / `split` / `narrow` / residual add | `Concat` / `Split` / `Slice` / `Add` |
//! | Attention (v1) / `HyperContext` (v2) | `Reshape`, `Split`, `Transpose`, `MatMul`, `Mul`, `Softmax`, `ReduceMean`, `Shape` |
//! | `FullPadGate` (v2) | `MatMul`, `Add`, `Tanh`, `Reshape`, `Mul`, `Cast` |
//! | `PeakSuppress` (v2 `o2o`) | `Conv`, `Sigmoid`, `Softplus`, `MaxPool`, `Sub`, `Mul` |
//! | head flatten | `Reshape` to `[B, C, −1]`, `Concat` along the anchor axis |
//! | `--decode` (MODEL.md §6.4, optional v2 `iou_out`) | `Reshape`, `Softmax`, `Mul` + `ReduceSum` (DFL), `Split`, `Sub` / `Add` / `Mul` (anchors), `Sigmoid`, `Sqrt` (v2 `iou_head`), `Concat`, `Transpose` |
//! | `--decode`, `o2o` head (§6.6 top-k) | `ReduceMax`, `TopK` (twice), `GatherND` (`batch_dims = 1`), `Div` / `Mod`, `Cast`, `Unsqueeze`, `Concat` |
//!
//! The input is `images: [B, 3, H, W]`, RGB in `[0, 1]` (docs/MODEL.md conventions). Outputs are either the raw
//! head (`box_raw: [B, 4·reg_max, A]`, `cls_raw: [B, nc, A]`, plus `iou_raw: [B, 1, A]` for a v2 model with
//! `iou_head = true`, MODEL.md §6.3 / §11.4) or, with [`OnnxOptions::decode`], `detections: [B, A, 4 + nc]`
//! (`x1, y1, x2, y2` in input pixels, then the class scores after the sigmoid — fused with `sqrt(p_cls · p_iou)`
//! when `iou_head = true`; NMS is left to the caller). Dynamic dimensions can be toggled together via
//! [`OnnxOptions::dynamic`] or independently via [`DynamicAxes`] (`batch` and `hw`, using
//! [`export_with_axes`] / [`export_v2_with_axes`]): enabling `batch` emits symbolic `"batch"`, and enabling `hw`
//! emits symbolic `"height"`, `"width"` (and `"anchors"` / `"max_det"`) and computes the anchor grid dynamically
//! in the graph; otherwise spatial dimensions are fixed and anchors are constants.
//!
//! A model whose head mode is `o2o` (docs/MODEL.md §6.6 / §11.4) exports its one-to-one branch only (unused
//! branch tensors are dropped), and its decoded output is end to end, with no NMS needed:
//! `detections: [B, K, 6]` with `K = min(max_det, A)`, rows `x1, y1, x2, y2, score, class` sorted by score
//! (the top-k selection of §6.6: the `K` anchors with the highest max-class score, then the `K` best
//! (anchor, class) pairs among them). The caller applies its confidence threshold. These rows are `F32` in an
//! [`OnnxOptions::half`] graph too (class ids above 2048 have no exact `F16`).
//!
//! Every model tensor becomes an initializer under its `.folo` name (`backbone.stem.conv.weight`, ...), in `F32`
//! or, with [`OnnxOptions::half`], `F16` (then the input, outputs and every activation are `F16` too, with
//! v2 `HyperContext` / `FullPadGate` / `PeakSuppress` stability casts matching `FoloV2Model::forward`). Anything
//! the exporter cannot map (an unfused BatchNorm or `WsQARepConv3x3`, a tensor no node consumed) is an
//! [`Error::Unsupported`] naming it. The class names and FOLO metadata go into `metadata_props` (see
//! [`OnnxExport::metadata`]).
//!
//! Files: [`proto`] (the protobuf encoder), `graph.rs` (the graph under construction and the layer-level helpers),
//! `blocks.rs` (v1 blocks as nodes), `v2.rs` (FOLO v2-Iso blocks, head and export), and `decode.rs` (`--decode`
//! in the graph).

pub mod proto;

mod blocks;
mod decode;
mod graph;
mod v2;

use std::collections::BTreeMap;
use std::path::Path;

use folo_nn::Module;
use folo_nn::io::{ModelFile, Purpose};

use crate::model::{FoloModel, HeadMode};
use crate::{Error, Result};
use graph::Graph;
use proto::{DataType, Dim};
pub use v2::{export_v2, export_v2_with_axes};

/// The ONNX opset the exporter targets.
pub const OPSET: i64 = 17;
/// The IR version that goes with [`OPSET`].
pub const IR_VERSION: i64 = 8;
/// The input's name.
pub const INPUT: &str = "images";

/// Independent control over symbolic batch and spatial (`height`, `width`) ONNX dimensions.
///
/// Use with [`export_with_axes`], [`export_file_with_axes`], [`FoloModel::to_onnx_with_axes`],
/// [`export_v2_with_axes`], or [`crate::model::v2::FoloV2Model::to_onnx_with_axes`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DynamicAxes {
    /// Emit symbolic `"batch"` dimension instead of a fixed [`OnnxOptions::batch`] size.
    pub batch: bool,
    /// Emit symbolic `"height"`, `"width"` (and `"anchors"` / `"max_det"`) dimensions and compute the anchor
    /// grid dynamically in the graph (`H` and `W` must be positive multiples of 32 at run time).
    pub hw: bool,
}

impl DynamicAxes {
    /// Fixed batch, height, and width.
    pub const STATIC: Self = DynamicAxes { batch: false, hw: false };
    /// Dynamic batch only; fixed height and width (constant anchor grid).
    pub const BATCH: Self = DynamicAxes { batch: true, hw: false };
    /// Dynamic height and width only; fixed batch size.
    pub const HW: Self = DynamicAxes { batch: false, hw: true };
    /// Dynamic batch, height, and width (equivalent to `OnnxOptions { dynamic: true, .. }`).
    pub const ALL: Self = DynamicAxes { batch: true, hw: true };

    /// Construct from explicit `(batch, hw)` flags.
    pub const fn new(batch: bool, hw: bool) -> Self {
        DynamicAxes { batch, hw }
    }

    /// Map [`OnnxOptions::dynamic`] (`true` → [`Self::ALL`], `false` → [`Self::STATIC`]).
    pub const fn from_dynamic(dynamic: bool) -> Self {
        DynamicAxes { batch: dynamic, hw: dynamic }
    }

    /// True if either batch or spatial dimensions are symbolic.
    pub const fn is_any(self) -> bool {
        self.batch || self.hw
    }

    pub(super) fn input_dims(self, opts: &OnnxOptions) -> (Dim, Dim, Dim) {
        let bdim = if self.batch { Dim::Param("batch".into()) } else { Dim::Value(opts.batch) };
        let (hdim, wdim) = if self.hw {
            (Dim::Param("height".into()), Dim::Param("width".into()))
        } else {
            (Dim::Value(opts.height as usize), Dim::Value(opts.width as usize))
        };
        (bdim, hdim, wdim)
    }
}

/// What to export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxOptions {
    /// Input height in pixels (a positive multiple of 32). With dynamic `hw`, only recorded in metadata.
    pub height: u32,
    /// Input width in pixels (a positive multiple of 32). With dynamic `hw`, only recorded in metadata.
    pub width: u32,
    /// The fixed batch size (ignored with dynamic `batch`).
    pub batch: usize,
    /// Symbolic batch, height and width (`DynamicAxes::ALL`). For independent control over batch vs spatial
    /// dimensions, pass [`DynamicAxes`] to [`export_with_axes`] / [`export_v2_with_axes`].
    pub dynamic: bool,
    /// `F16` weights, input, activations and outputs.
    pub half: bool,
    /// Decode inside the graph: one `detections: [B, A, 4 + nc]` output instead of the raw head (`[B, K, 6]`
    /// end-to-end detections for an `o2o` head).
    pub decode: bool,
    /// `o2o` head with `decode`: the detections per image, `K = min(max_det, A)` (300 by default).
    pub max_det: usize,
    /// [`export_file`]: the head branch to export, `None` for the file's ([`FoloModel::head_mode`]).
    /// [`export`] and [`export_v2`] export the model's own mode.
    pub head: Option<HeadMode>,
    /// Extra `metadata_props` (for example the source file's `folo.metrics`); exporter-owned keys always win
    /// (conditional keys that do not apply to the exported graph are cleared; [`export_v2`] preserves caller-supplied
    /// `names` / `folo.classes` when present because [`crate::model::v2::FoloV2Model`] does not store class names).
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

    /// Resolve [`DynamicAxes`] implied by `self.dynamic`.
    pub const fn dynamic_axes(&self) -> DynamicAxes {
        DynamicAxes::from_dynamic(self.dynamic)
    }

    pub(super) fn validate(&self, stride: u32) -> Result<()> {
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
        if let Err(e) = done.and_then(|()| folo_nn::io::rename_over(&tmp, path)) {
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
    /// Export this model as an ONNX graph (see [`crate::export::onnx`]), deriving dynamic axes from
    /// `opts.dynamic`. The model must be fused ([`FoloModel::fuse`]); weights are read from any device.
    pub fn to_onnx(&self, opts: &OnnxOptions) -> Result<OnnxExport> {
        export(self, opts)
    }

    /// Export this model as an ONNX graph with independent [`DynamicAxes`] (`batch` and `hw`), overriding
    /// `opts.dynamic`.
    pub fn to_onnx_with_axes(&self, opts: &OnnxOptions, axes: DynamicAxes) -> Result<OnnxExport> {
        export_with_axes(self, opts, axes)
    }
}

/// `folo export --format onnx`: build the inference model a `.folo` file describes (its EMA weights when present,
/// docs/FORMAT.md loading rules) on the CPU, fuse it in `F32` and export it.
pub fn export_file(file: &ModelFile, opts: &OnnxOptions) -> Result<OnnxExport> {
    export_file_with_axes(file, opts, opts.dynamic_axes())
}

/// [`export_file`] with independent [`DynamicAxes`] (`batch` and `hw`), overriding `opts.dynamic`.
pub fn export_file_with_axes(file: &ModelFile, opts: &OnnxOptions, axes: DynamicAxes) -> Result<OnnxExport> {
    let (mut model, _) = FoloModel::from_file(file, Purpose::Inference)?;
    if let Some(h) = opts.head {
        model.set_head_mode(h).map_err(|e| Error::Config(format!("ONNX export: {e}")))?;
    }
    model.fuse()?;
    export_with_axes(&model, opts, axes)
}

/// Build the ONNX model of a fused `model`, deriving dynamic axes from `opts.dynamic`.
pub fn export(model: &FoloModel, opts: &OnnxOptions) -> Result<OnnxExport> {
    export_with_axes(model, opts, opts.dynamic_axes())
}

/// Build the ONNX model of a fused `model` with independent [`DynamicAxes`] (`batch` and `hw`), overriding
/// `opts.dynamic`.
pub fn export_with_axes(model: &FoloModel, opts: &OnnxOptions, axes: DynamicAxes) -> Result<OnnxExport> {
    if !model.is_fused() {
        return Err(Error::Unsupported(
            "ONNX export needs a fused model (Conv+BN folded): call FoloModel::fuse first".into(),
        ));
    }
    let cfg = model.config();
    let e2e = model.head_mode() == HeadMode::O2o;
    let strides = model.strides();
    let stride = *strides.iter().max().expect("three strides");
    opts.validate(stride)?;

    let elem = if opts.half { DataType::Float16 } else { DataType::Float };
    let mut g = Graph::new(elem);
    let (bdim, hdim, wdim) = axes.input_dims(opts);
    let input = OnnxValue {
        name: INPUT.into(),
        dtype: elem.name(),
        shape: vec![bdim.clone(), Dim::Value(cfg.in_channels), hdim, wdim],
    };

    let feats = g.body(model, INPUT)?;
    let (box_raw, cls_raw) = g.head(model, &feats, !opts.decode)?;
    let (reg_max, nc) = (model.head.reg_max, model.num_classes());
    let a = if axes.hw {
        Dim::Param("anchors".into())
    } else {
        Dim::Value(model.anchors(opts.height as usize, opts.width as usize).len())
    };
    let outputs = if opts.decode && e2e {
        let k = match &a {
            Dim::Value(a) => Dim::Value(opts.max_det.min(*a)),
            Dim::Param(_) => Dim::Param("max_det".into()),
        };
        let all = g.decode(&strides, reg_max, opts, axes, &feats, &box_raw, &cls_raw, None, true)?;
        g.topk(&all, nc, opts, true);
        // Always f32: an f16 class id is exact only up to 2048.
        let dtype = DataType::Float.name();
        vec![OnnxValue { name: "detections".into(), dtype, shape: vec![bdim, k, Dim::Value(6)] }]
    } else if opts.decode {
        let all = g.decode(&strides, reg_max, opts, axes, &feats, &box_raw, &cls_raw, None, false)?;
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

    let metadata = metadata(model, opts, axes, &outputs)?;
    let ops = g.nodes.iter().map(|n| n.op.to_string()).collect();
    let initializers = g.inits.len();
    let bytes = g.finish(&input, &outputs, &metadata);
    Ok(OnnxExport { bytes, inputs: vec![input], outputs, ops, initializers, metadata })
}

/// The `metadata_props`: the model's `folo.*` weights metadata (`folo.dtype` the export's, `folo.fused`
/// `"true"`), `names` (`{"0": "<class 0>", ...}`) and `folo.onnx.*`: `outputs` (`"raw"`, `"decoded"`, or
/// `"end2end"` for a decoded `o2o` head), `head` (`"o2m"` or `"o2o"`, the exported branch), `max_det` (end to
/// end only), `imgsz` (`[H, W]`), `batch` (a number, or `"dynamic"`), `dynamic`, `dynamic_batch`, `dynamic_hw`,
/// `opset`, `strides`, `reg_max`.
fn metadata(
    model: &FoloModel,
    opts: &OnnxOptions,
    axes: DynamicAxes,
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
    m.retain(|k, _| !k.starts_with("folo.v2."));
    m.remove("folo.onnx.max_det");
    m.remove("folo.onnx.iou_head");
    m.remove("folo.onnx.iou_fusion_alpha");
    m.insert("folo.onnx.outputs".into(), kind.into());
    m.insert("folo.onnx.head".into(), model.head_mode().as_str().into());
    if opts.decode && e2e {
        m.insert("folo.onnx.max_det".into(), opts.max_det.to_string());
    }
    m.insert("folo.onnx.output_names".into(), s(outputs.iter().map(|o| o.name.clone()).collect()));
    m.insert("folo.onnx.imgsz".into(), s(serde_json::json!([opts.height, opts.width])));
    m.insert("folo.onnx.batch".into(), if axes.batch { "dynamic".into() } else { opts.batch.to_string() });
    m.insert("folo.onnx.dynamic".into(), axes.is_any().to_string());
    m.insert("folo.onnx.dynamic_batch".into(), axes.batch.to_string());
    m.insert("folo.onnx.dynamic_hw".into(), axes.hw.to_string());
    m.insert("folo.onnx.opset".into(), OPSET.to_string());
    m.insert("folo.onnx.strides".into(), s(serde_json::json!(model.strides())));
    m.insert("folo.onnx.reg_max".into(), model.head.reg_max.to_string());
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use folo_nn::{Activation, ConvBnAct};

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
