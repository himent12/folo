//! Weights files and export: Conv+BN fusion in `F32`, then an optional `F16` cast (docs/FORMAT.md, Export).

use std::path::Path;

use folo_core::{DType, Tensor};

use super::file::{LoadOptions, ModelFile, Purpose, Source};
use super::meta::{FileKind, FloatDType, Metadata};
use super::safetensors;
use crate::module::Module;
use crate::{Error, Result, fuse_conv_bn};

/// Fold every BatchNorm of a named tensor set into its conv: each `<p>.bn.{weight, bias, running_mean,
/// running_var}` with `<p>.conv.weight` (and an optional `<p>.conv.bias`) becomes `<p>.conv.weight` and
/// `<p>.conv.bias` (docs/MODEL.md §5, fused files). Other tensors pass through. Order is preserved, with the
/// fused bias right after its weight. An incomplete BatchNorm set is an error.
pub fn fuse_bn_tensors(tensors: Vec<(String, Tensor)>, eps: f64) -> Result<Vec<(String, Tensor)>> {
    let index: std::collections::HashMap<&str, usize> =
        tensors.iter().enumerate().map(|(i, (n, _))| (n.as_str(), i)).collect();
    let get = |name: &str| index.get(name).map(|&i| &tensors[i].1);
    let mut fused = std::collections::HashMap::new();
    let mut consumed = std::collections::HashSet::new();
    for (name, _) in &tensors {
        let Some(p) = name.strip_suffix(".bn.weight") else { continue };
        let need = |s: &str| -> Result<&Tensor> {
            get(&format!("{p}.{s}"))
                .ok_or_else(|| Error::Invalid(format!("fuse: `{p}` has a BatchNorm but no `{p}.{s}`")))
        };
        let w = need("conv.weight")?;
        let cb = get(&format!("{p}.conv.bias"));
        let (fw, fb) = fuse_conv_bn(
            w,
            cb,
            need("bn.weight")?,
            need("bn.bias")?,
            need("bn.running_mean")?,
            need("bn.running_var")?,
            eps,
        )?;
        for s in ["bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "conv.bias"] {
            consumed.insert(format!("{p}.{s}"));
        }
        fused.insert(format!("{p}.conv.weight"), (fw, fb));
    }
    for (name, _) in &tensors {
        if name.contains(".bn.") && !consumed.contains(name) {
            return Err(Error::Invalid(format!("fuse: `{name}` has no matching `.bn.weight`")));
        }
    }
    let mut out = Vec::with_capacity(tensors.len());
    for (name, t) in tensors {
        if consumed.contains(&name) {
            continue;
        }
        match fused.remove(&name) {
            Some((w, b)) => {
                let bias = format!("{}.bias", name.strip_suffix(".weight").expect("conv.weight"));
                out.push((name, w));
                out.push((bias, b));
            }
            None => out.push((name, t)),
        }
    }
    Ok(out)
}

/// Write a weights file: `meta` (kind forced to `weights`, checkpoint-only keys dropped) and every tensor of
/// `model` as it is. `folo.dtype` is set from the model's floating tensors.
pub fn save_weights(path: impl AsRef<Path>, meta: &Metadata, model: &dyn Module) -> Result<()> {
    let tensors: Vec<(String, Tensor)> =
        model.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    let mut m = meta.for_weights();
    m.dtype = float_dtype(&tensors)?;
    safetensors::write_file(path.as_ref(), Some(&m.to_map()), &tensors)
}

fn float_dtype(tensors: &[(String, Tensor)]) -> Result<FloatDType> {
    let mut seen = None;
    for (n, t) in tensors {
        let d = match t.dtype() {
            DType::F32 => FloatDType::F32,
            DType::F16 => FloatDType::F16,
            _ => continue,
        };
        if seen.is_some_and(|s| s != d) {
            return Err(Error::Invalid(format!("mixed floating dtypes (at `{n}`): a .folo file has one")));
        }
        seen = Some(d);
    }
    Ok(seen.unwrap_or(FloatDType::F32))
}

/// Export a named tensor set (the EMA or live weights, in `F32`): fuse Conv+BN in `F32` unless `meta` says it
/// is already fused, cast every floating tensor to `dtype`, and write a fused `weights` file.
pub fn export_tensors(
    path: impl AsRef<Path>,
    meta: &Metadata,
    tensors: Vec<(String, Tensor)>,
    dtype: FloatDType,
) -> Result<()> {
    let tensors = if meta.fused { tensors } else { fuse_bn_tensors(tensors, meta.config.bn_eps)? };
    let tensors = tensors
        .into_iter()
        .map(|(n, t)| {
            let t = if t.dtype().is_float() { t.to_dtype(DType::F32)?.to_dtype(dtype.core())? } else { t };
            Ok((n, t))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut m = meta.for_weights();
    m.kind = FileKind::Weights;
    m.fused = true;
    m.dtype = dtype;
    safetensors::write_file(path.as_ref(), Some(&m.to_map()), &tensors)
}

/// Export a module's current weights (see [`export_tensors`]).
pub fn export_module(
    path: impl AsRef<Path>,
    meta: &Metadata,
    model: &dyn Module,
    dtype: FloatDType,
) -> Result<()> {
    let tensors = model.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    export_tensors(path, meta, tensors, dtype)
}

impl ModelFile {
    /// The tensors inference and export use: the EMA set if present (prefix stripped), otherwise the model
    /// namespace, decoded and converted to `F32`, sorted by name.
    pub fn inference_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let source = if self.has_ema() { Source::Ema } else { Source::Model };
        self.source_names(source)
            .into_iter()
            .map(|(m, f)| {
                let t = self.tensor(f)?;
                let t = if t.dtype() == DType::F16 { t.to_dtype(DType::F32)? } else { t };
                Ok((m, t))
            })
            .collect()
    }

    /// `folo export`: take the EMA state (or the model namespace), fuse Conv+BN in `F32`, optionally cast to
    /// `F16`, and write a fused `weights` file to `path`. Unknown `folo.*` keys are preserved; checkpoint-only
    /// keys are dropped.
    pub fn export(&self, path: impl AsRef<Path>, dtype: FloatDType) -> Result<()> {
        export_tensors(path, self.metadata(), self.inference_tensors()?, dtype)
    }

    /// [`ModelFile::load`] with strict inference options (the common case for prediction).
    pub fn load_for_inference(&self, model: &dyn Module) -> Result<super::file::LoadReport> {
        self.load(model, &LoadOptions::strict(Purpose::Inference))
    }
}
