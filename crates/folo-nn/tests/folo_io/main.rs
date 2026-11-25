//! `.folo` files: compatibility both ways with safetensors, loading rules, export, and checkpoint resume.

mod checkpoint;
mod format;
mod musgd_resume;

use std::collections::BTreeMap;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::io::*;
use folo_nn::*;
use folo_test_support::TestFile;
use serde_json::Value;

pub(crate) fn tmp(name: &str) -> TestFile {
    TestFile::new("nn-tests", name)
}

/// A tiny model with detector-style names: `backbone.stem.*`, `head.cls.p3.pw0.*`, `head.cls.p3.out.*`.
pub(crate) fn net(nc: usize, rng: &mut Rng) -> Sequential {
    let p3 = Sequential::new()
        .push_named("pw0", ConvBnAct::new(4, 4, 1, 1, 1, Activation::Silu, rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, nc, 1), rng));
    Sequential::new()
        .push_named(
            "backbone",
            Sequential::new().push_named("stem", ConvBnAct::new(3, 4, 3, 2, 1, Activation::Silu, rng)),
        )
        .push_named("head", Sequential::new().push_named("cls", Sequential::new().push_named("p3", p3)))
}

/// The same model with fused Conv blocks (`conv.weight`, `conv.bias`).
pub(crate) fn net_fused(nc: usize, rng: &mut Rng) -> Sequential {
    let p3 = Sequential::new()
        .push_named("pw0", ConvBnAct::new_fused(4, 4, 1, 1, 1, Activation::Silu, rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, nc, 1), rng));
    Sequential::new()
        .push_named(
            "backbone",
            Sequential::new().push_named("stem", ConvBnAct::new_fused(3, 4, 3, 2, 1, Activation::Silu, rng)),
        )
        .push_named("head", Sequential::new().push_named("cls", Sequential::new().push_named("p3", p3)))
}

pub(crate) fn classes(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("class{i}")).collect()
}

pub(crate) fn meta(kind: FileKind, nc: usize) -> Metadata {
    Metadata::new(kind, ModelConfig::new("n", nc).unwrap(), classes(nc), 64).unwrap()
}

pub(crate) fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

pub(crate) fn state_bits(m: &dyn Module) -> Vec<(String, Vec<u32>)> {
    m.named_tensors().into_iter().map(|t| (t.name, bits(&t.tensor))).collect()
}

/// Give BatchNorms non-trivial statistics and affine parameters.
pub(crate) fn perturb(m: &dyn Module, rng: &mut Rng) {
    m.train();
    for _ in 0..2 {
        m_forward(m, &Tensor::randn([2, 3, 8, 8], 0.3, 1.5, rng));
    }
    let _g = no_grad();
    for (n, t) in m.named_parameters() {
        if n.ends_with("bn.weight") {
            t.copy_from(&Tensor::rand_uniform(t.shape().clone(), 0.5, 1.5, rng)).unwrap();
        } else if n.ends_with("bias") {
            t.copy_from(&Tensor::rand_uniform(t.shape().clone(), -0.5, 0.5, rng)).unwrap();
        }
    }
}

pub(crate) fn m_forward(m: &dyn Module, x: &Tensor) -> Tensor {
    (m as &dyn std::any::Any)
        .downcast_ref::<Sequential>()
        .expect("test models are Sequential")
        .forward(x)
        .unwrap()
}

// ----- an independent safetensors reader (no folo-nn code) -----

pub(crate) struct Parsed {
    pub(crate) header_len: usize,
    pub(crate) raw_header: Vec<u8>,
    pub(crate) header: serde_json::Map<String, Value>,
    pub(crate) data: Vec<u8>,
}

pub(crate) fn parse_independently(bytes: &[u8]) -> Parsed {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let raw_header = bytes[8..8 + n].to_vec();
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&raw_header).unwrap();
    Parsed { header_len: n, raw_header, header, data: bytes[8 + n..].to_vec() }
}

pub(crate) fn le_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

pub(crate) fn rewrite(map: &BTreeMap<String, String>, tensors: &[(String, Tensor)]) -> Result<ModelFile> {
    let mut buf = Vec::new();
    safetensors::write(&mut buf, Some(map), tensors).unwrap();
    ModelFile::from_bytes(buf)
}
