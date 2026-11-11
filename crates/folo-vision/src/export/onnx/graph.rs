//! The graph under construction: value names, nodes, initializers and shared constants, the layer-level
//! helpers (`Conv`, activations, `Concat`, `Split`, `Reshape`, ...) and the final `ModelProto` encoding.

use std::collections::{BTreeMap, HashSet};

use folo_core::{DType, Tensor, f16};
use folo_nn::{Activation, Conv2d, ConvBnAct};

use super::proto::{self, AttrValue, DataType, Msg};
use super::{INPUT, IR_VERSION, OPSET, OnnxValue};
use crate::{Error, Result};

pub(super) struct Node {
    pub(super) op: &'static str,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<(String, AttrValue)>,
}

/// Attribute list shorthand.
pub(super) fn attrs<const N: usize>(a: [(&str, AttrValue); N]) -> Vec<(String, AttrValue)> {
    a.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

pub(super) fn ints(v: &[i64]) -> AttrValue {
    AttrValue::Ints(v.to_vec())
}

/// The graph under construction.
pub(super) struct Graph {
    /// The floating element type of weights and activations.
    pub(super) elem: DataType,
    pub(super) nodes: Vec<Node>,
    /// Encoded initializers.
    pub(super) inits: Vec<Msg>,
    /// Every value name (node outputs, initializers, the input), kept unique.
    names: HashSet<String>,
    /// The model tensors consumed so far.
    pub(super) used: HashSet<String>,
    /// Shared int64 constants, by value.
    consts: BTreeMap<Vec<i64>, String>,
}

impl Graph {
    pub(super) fn new(elem: DataType) -> Self {
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
    pub(super) fn node_n(
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

    pub(super) fn node(
        &mut self,
        op: &'static str,
        name: &str,
        inputs: &[&str],
        attrs: Vec<(String, AttrValue)>,
    ) -> String {
        self.node_n(op, name, inputs, attrs, 1).remove(0)
    }

    /// An initializer of the graph's floating type from f32 values.
    pub(super) fn float_init(&mut self, name: &str, dims: &[usize], data: &[f32]) -> String {
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
    pub(super) fn f32_init(&mut self, name: &str, dims: &[usize], data: &[f32]) -> String {
        let name = self.fresh(name);
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.inits.push(proto::tensor(&name, DataType::Float, dims, &raw));
        name
    }

    /// A shared `int64` 1-D constant (`scalar` for a 0-d one).
    pub(super) fn i64_const(&mut self, values: &[i64], scalar: bool) -> String {
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
    pub(super) fn conv2d(&mut self, p: &str, c: &Conv2d, x: &str) -> Result<String> {
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
    pub(super) fn conv(&mut self, p: &str, c: &ConvBnAct, x: &str) -> Result<String> {
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

    pub(super) fn concat(&mut self, name: &str, xs: &[&str], axis: i64) -> String {
        self.node("Concat", name, xs, attrs([("axis", AttrValue::Int(axis))]))
    }

    pub(super) fn split(&mut self, name: &str, x: &str, sizes: &[i64], axis: i64) -> Vec<String> {
        let s = self.i64_const(sizes, false);
        self.node_n("Split", name, &[x, &s], attrs([("axis", AttrValue::Int(axis))]), sizes.len())
    }

    pub(super) fn reshape(&mut self, name: &str, x: &str, shape: &[i64]) -> String {
        let s = self.i64_const(shape, false);
        self.node("Reshape", name, &[x, &s], vec![])
    }

    pub(super) fn transpose(&mut self, name: &str, x: &str, perm: &[i64]) -> String {
        self.node("Transpose", name, &[x], attrs([("perm", ints(perm))]))
    }

    pub(super) fn add(&mut self, name: &str, a: &str, b: &str) -> String {
        self.node("Add", name, &[a, b], vec![])
    }

    /// Encode the `ModelProto`.
    pub(super) fn finish(
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
