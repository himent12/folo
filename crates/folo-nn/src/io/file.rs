//! [`ModelFile`]: an opened `.folo` file, its namespaces, and the loading rules of docs/FORMAT.md.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use folo_core::{DType, Tensor, no_grad};

use super::checkpoint::{self, CheckpointMeta};
use super::meta::{FileKind, Metadata};
use super::safetensors::{Container, FileDType, TensorInfo, decode};
use crate::module::Module;
use crate::{Error, Result};

/// The namespace of a tensor name, decided by its first dotted segment (docs/FORMAT.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Namespace {
    /// Model parameters and buffers.
    Model,
    /// `ema.*`.
    Ema,
    /// `optim.*`.
    Optim,
    /// `sched.*`.
    Sched,
    /// Top-level `__*`, reserved (ignored with a warning).
    Reserved,
}

impl Namespace {
    /// The namespace of `name`.
    pub fn of(name: &str) -> Namespace {
        let first = name.split('.').next().unwrap_or("");
        if first.starts_with("__") {
            return Namespace::Reserved;
        }
        match first {
            "ema" => Namespace::Ema,
            "optim" => Namespace::Optim,
            "sched" => Namespace::Sched,
            _ => Namespace::Model,
        }
    }
}

/// Why loading needs the tensors (docs/FORMAT.md loading rule 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Inference or export: the EMA weights if present, otherwise the model namespace. Fused files are allowed.
    Inference,
    /// Training (resume or fine-tune): the model namespace. Fused files are refused.
    Training,
}

/// Options for [`ModelFile::load`].
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Strict (the default): every name must match with the same shape, and the class lists must agree.
    /// Partial (`false`, transfer learning): load what matches, skip and report the rest.
    pub strict: bool,
    /// Which tensors to load.
    pub purpose: Purpose,
    /// The target model's class list, if it has one. Strict loading requires `folo.classes` to equal it;
    /// partial loading skips the class-output layers when they differ.
    pub classes: Option<Vec<String>>,
    /// Which tensors are the classification output layers (docs/MODEL.md §5); defaults to
    /// [`is_class_output`].
    pub class_output: fn(&str) -> bool,
}

impl LoadOptions {
    /// Strict loading for `purpose`.
    pub fn strict(purpose: Purpose) -> Self {
        LoadOptions { strict: true, purpose, classes: None, class_output: is_class_output }
    }

    /// Partial (transfer) loading for `purpose`.
    pub fn partial(purpose: Purpose) -> Self {
        LoadOptions { strict: false, ..Self::strict(purpose) }
    }

    /// Set the target model's class list.
    pub fn with_classes(mut self, classes: Vec<String>) -> Self {
        self.classes = Some(classes);
        self
    }
}

/// docs/MODEL.md §5: the class-dependent output layers `head.cls.{p3,p4,p5}.out.{weight,bias}` (and the
/// one-to-one branch's `head.o2o.cls.*.out.*`, §6.6).
pub fn is_class_output(name: &str) -> bool {
    let rest = name.strip_prefix("head.o2o.cls.").or_else(|| name.strip_prefix("head.cls."));
    match rest {
        Some(r) => {
            let mut p = r.split('.');
            matches!(
                (p.next(), p.next(), p.next(), p.next()),
                (Some("p3" | "p4" | "p5"), Some("out"), Some("weight" | "bias"), None)
            )
        }
        None => false,
    }
}

/// Where the loaded tensors came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The model namespace.
    Model,
    /// The EMA namespace (prefix stripped).
    Ema,
}

/// Why a tensor was not loaded (partial loading).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The model has no tensor of that name.
    NotInModel,
    /// Same name, different shape.
    ShapeMismatch {
        /// Shape in the file.
        file: Vec<usize>,
        /// Shape in the model.
        model: Vec<usize>,
    },
    /// The file dtype does not convert to the model's (only `F16 → F32` and identical dtypes do).
    DTypeMismatch {
        /// Dtype in the file.
        file: &'static str,
        /// Dtype in the model.
        model: DType,
    },
    /// A classification output layer, skipped because the class lists differ.
    ClassMismatch,
}

/// A tensor of the file that was not loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    /// Model name (EMA prefix stripped).
    pub name: String,
    /// Why.
    pub reason: SkipReason,
}

/// What [`ModelFile::load`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadReport {
    /// Which namespace was read.
    pub source: Source,
    /// Loaded tensors, in the model's canonical order.
    pub loaded: Vec<String>,
    /// File tensors not loaded, sorted by name.
    pub skipped: Vec<Skipped>,
    /// Model tensors that received no value, in canonical order.
    pub missing: Vec<String>,
}

enum Bytes {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Mapped(m) => m,
            Bytes::Owned(v) => v,
        }
    }
}

/// An opened, validated `.folo` file.
///
/// Opening checks the container (docs/FORMAT.md reader acceptance rules), the metadata, the namespaces
/// (a weights file holds only the model namespace; a checkpoint's EMA set is complete and its contracts
/// agree with its counters), and `folo.dtype`. Tensors are decoded on demand.
///
/// On Windows a memory-mapped file cannot be replaced while it is open: drop the `ModelFile` (for example
/// after resuming) before writing a new checkpoint to the same path.
pub struct ModelFile {
    bytes: Bytes,
    container: Container,
    meta: Metadata,
    index: HashMap<String, usize>,
    warnings: Vec<String>,
    checkpoint: Option<CheckpointMeta>,
}

impl ModelFile {
    /// Open `path`, memory-mapping it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let f = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: the map is read-only and private to this value. As with every memory-mapped reader
        // (safetensors included), another process truncating or rewriting the file while it is mapped is
        // outside what Rust can guard against; FOLO's own writers never modify a file in place: they write
        // a fresh temporary file and rename it over the old file, which leaves an existing mapping intact on Unix and is
        // refused while the file is mapped on Windows.
        let map = unsafe { memmap2::Mmap::map(&f) };
        let bytes = match map {
            Ok(m) => Bytes::Mapped(m),
            Err(_) => Bytes::Owned(std::fs::read(path).map_err(|e| Error::io(path, e))?),
        };
        Self::from_parts(bytes, None)
    }

    /// Parse an in-memory file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_parts(Bytes::Owned(bytes), None)
    }

    /// The import path for a plain safetensors file (no `folo.*` metadata): the caller supplies the metadata
    /// the file would have had (`folo import --config <json> --classes <json>`). Must be a `weights` file.
    pub fn import(bytes: Vec<u8>, meta: Metadata) -> Result<Self> {
        if meta.kind != FileKind::Weights {
            return Err(Error::Metadata(
                "import produces a weights file; folo.kind must be `weights`".into(),
            ));
        }
        Self::from_parts(Bytes::Owned(bytes), Some(meta))
    }

    fn from_parts(bytes: Bytes, imported: Option<Metadata>) -> Result<Self> {
        let container = Container::parse(&bytes)?;
        let meta = match imported {
            Some(m) => m,
            None => Metadata::from_map(container.metadata.as_ref().unwrap_or(&BTreeMap::new()))?,
        };
        let mut warnings = Vec::new();
        let mut index = HashMap::with_capacity(container.tensors.len());
        let mut counts: HashMap<Namespace, usize> = HashMap::new();
        for (i, t) in container.tensors.iter().enumerate() {
            let ns = Namespace::of(&t.name);
            *counts.entry(ns).or_default() += 1;
            match ns {
                Namespace::Reserved => warnings.push(format!("ignoring reserved tensor `{}`", t.name)),
                Namespace::Model if t.name.is_empty() || t.name.split('.').any(str::is_empty) => {
                    return Err(Error::Metadata(format!("invalid tensor name `{}`", t.name)));
                }
                Namespace::Model
                    if t.dtype.to_core().is_some_and(|d| d.is_float())
                        && t.dtype.to_core() != Some(meta.dtype.core()) =>
                {
                    return Err(Error::Metadata(format!(
                        "tensor `{}` is {}, but folo.dtype is {}",
                        t.name,
                        t.dtype.as_str(),
                        meta.dtype.as_str()
                    )));
                }
                Namespace::Ema | Namespace::Optim | Namespace::Sched if !t.name.contains('.') => {
                    return Err(Error::Metadata(format!("invalid tensor name `{}`", t.name)));
                }
                _ => {}
            }
            index.insert(t.name.clone(), i);
        }
        if meta.kind == FileKind::Weights {
            let extra: Vec<&str> = container
                .tensors
                .iter()
                .filter(|t| {
                    matches!(Namespace::of(&t.name), Namespace::Ema | Namespace::Optim | Namespace::Sched)
                })
                .map(|t| t.name.as_str())
                .collect();
            if !extra.is_empty() {
                return Err(Error::Metadata(format!(
                    "a weights file holds only the model namespace, but it has {}",
                    extra.join(", ")
                )));
            }
        }
        let mut file = ModelFile { bytes, container, meta, index, warnings, checkpoint: None };
        if file.meta.kind == FileKind::Checkpoint {
            file.checkpoint = Some(checkpoint::validate(&file)?);
        }
        Ok(file)
    }

    /// The metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.meta
    }

    /// The parsed checkpoint contracts (`None` for a weights file).
    pub fn checkpoint(&self) -> Option<&CheckpointMeta> {
        self.checkpoint.as_ref()
    }

    /// Warnings from opening (ignored reserved tensors).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Every tensor entry, in header order.
    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.container.tensors
    }

    /// The entry of `name`.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.container.tensors[i])
    }

    /// Full names in namespace `ns`, sorted.
    pub fn names(&self, ns: Namespace) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .container
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| Namespace::of(n) == ns)
            .collect();
        v.sort_unstable();
        v
    }

    /// True if the file has `ema.*` model tensors.
    pub fn has_ema(&self) -> bool {
        self.container.tensors.iter().any(|t| t.name.starts_with("ema.") && !t.name.starts_with("ema.__"))
    }

    /// Decode a tensor (CPU).
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name).ok_or_else(|| Error::Load(format!("the file has no tensor `{name}`")))?;
        decode(info, self.container.data(&self.bytes, info))
    }

    /// A scalar `I64` counter such as `optim.__step`.
    pub(crate) fn counter(&self, name: &str) -> Result<u64> {
        let info = self.info(name).ok_or_else(|| Error::Metadata(format!("checkpoint has no `{name}`")))?;
        if info.dtype != FileDType::I64 || !info.shape.is_empty() {
            return Err(Error::Metadata(format!("`{name}` must be a scalar I64")));
        }
        let v = self.tensor(name)?.item::<i64>()?;
        u64::try_from(v).map_err(|_| Error::Metadata(format!("`{name}` is negative ({v})")))
    }

    /// `(model name, file name)` of the chosen source, sorted by model name.
    pub(crate) fn source_names(&self, source: Source) -> Vec<(String, &str)> {
        let mut out: Vec<(String, &str)> = match source {
            Source::Model => self.names(Namespace::Model).into_iter().map(|n| (n.to_string(), n)).collect(),
            Source::Ema => self
                .names(Namespace::Ema)
                .into_iter()
                .filter(|n| !n.starts_with("ema.__"))
                .map(|n| (n["ema.".len()..].to_string(), n))
                .collect(),
        };
        out.sort();
        out
    }

    /// Load the file into `model` following docs/FORMAT.md's loading rules (source selection, strict or
    /// partial, class-list checks, `F16 → F32` conversion, fused files for inference only). The values are
    /// written into the model's tensors in place, under `no_grad`. Strict loading checks everything before
    /// writing anything, and its error lists every offending name.
    pub fn load(&self, model: &dyn Module, opts: &LoadOptions) -> Result<LoadReport> {
        if self.meta.fused && opts.purpose == Purpose::Training {
            return Err(Error::Load(
                "this file has fused Conv+BN weights (folo.fused = \"true\"), which can only be loaded for \
                 inference; load an unfused weights file or a checkpoint to train"
                    .into(),
            ));
        }
        let source =
            if opts.purpose == Purpose::Inference && self.has_ema() { Source::Ema } else { Source::Model };
        let file_names = self.source_names(source);
        let by_name: HashMap<&str, &str> = file_names.iter().map(|(m, f)| (m.as_str(), *f)).collect();
        let classes_differ = opts.classes.as_ref().is_some_and(|c| *c != self.meta.classes);
        if opts.strict && classes_differ {
            return Err(Error::Load(format!(
                "the file's classes {:?} differ from the model's {:?}; use partial loading (strict: false) to \
                 transfer the other weights",
                self.meta.classes,
                opts.classes.as_ref().expect("checked")
            )));
        }

        let targets = model.named_tensors();
        let mut model_names = BTreeSet::new();
        let mut plan: Vec<(&Tensor, &str, &str)> = Vec::new();
        let mut skipped = Vec::new();
        let mut absent = Vec::new();
        let mut problems = Vec::new();
        for t in &targets {
            model_names.insert(t.name.as_str());
            let Some(&fname) = by_name.get(t.name.as_str()) else {
                absent.push(t.name.clone());
                continue;
            };
            let info = self.info(fname).expect("indexed");
            let reason = if !opts.strict && classes_differ && (opts.class_output)(&t.name) {
                Some(SkipReason::ClassMismatch)
            } else if info.shape != t.tensor.dims() {
                Some(SkipReason::ShapeMismatch { file: info.shape.clone(), model: t.tensor.dims().to_vec() })
            } else if !converts(info.dtype, t.tensor.dtype()) {
                Some(SkipReason::DTypeMismatch { file: info.dtype.as_str(), model: t.tensor.dtype() })
            } else {
                None
            };
            match reason {
                None => plan.push((&t.tensor, t.name.as_str(), fname)),
                Some(r) => {
                    if opts.strict {
                        problems.push(describe(&t.name, &r));
                    }
                    skipped.push(Skipped { name: t.name.clone(), reason: r });
                }
            }
        }
        for (m, _) in &file_names {
            if !model_names.contains(m.as_str()) {
                if opts.strict {
                    problems.push(format!("unexpected `{m}` (not in the model)"));
                }
                skipped.push(Skipped { name: m.clone(), reason: SkipReason::NotInModel });
            }
        }
        if opts.strict {
            problems.extend(absent.iter().map(|n| format!("missing `{n}`")));
            if !problems.is_empty() {
                return Err(Error::Load(format!("strict load failed: {}", problems.join("; "))));
            }
        }
        let _g = no_grad();
        let mut loaded = Vec::with_capacity(plan.len());
        for (dst, name, fname) in plan {
            let src = self.tensor(fname)?.to_dtype(dst.dtype())?.to_device(dst.device())?;
            dst.copy_from(&src)?;
            loaded.push(name.to_string());
        }
        skipped.sort_by(|a, b| a.name.cmp(&b.name));
        let done: BTreeSet<&str> = loaded.iter().map(String::as_str).collect();
        let missing =
            targets.iter().filter(|t| !done.contains(t.name.as_str())).map(|t| t.name.clone()).collect();
        Ok(LoadReport { source, loaded, skipped, missing })
    }
}

fn converts(file: FileDType, model: DType) -> bool {
    file.to_core() == Some(model) || (file == FileDType::F16 && model == DType::F32)
}

fn describe(name: &str, r: &SkipReason) -> String {
    match r {
        SkipReason::ShapeMismatch { file, model } => {
            format!("`{name}` has shape {file:?} in the file, {model:?} in the model")
        }
        SkipReason::DTypeMismatch { file, model } => {
            format!("`{name}` is {file} in the file, {model} in the model")
        }
        SkipReason::ClassMismatch => format!("`{name}` depends on the class list"),
        SkipReason::NotInModel => format!("unexpected `{name}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(Namespace::of("backbone.stem.conv.weight"), Namespace::Model);
        assert_eq!(Namespace::of("ema.backbone.stem.conv.weight"), Namespace::Ema);
        assert_eq!(Namespace::of("optim.__step"), Namespace::Optim);
        assert_eq!(Namespace::of("sched.__step"), Namespace::Sched);
        assert_eq!(Namespace::of("__future"), Namespace::Reserved);
        assert_eq!(Namespace::of("emax.w"), Namespace::Model);
        assert!(is_class_output("head.cls.p3.out.weight"));
        assert!(is_class_output("head.o2o.cls.p5.out.bias"));
        assert!(!is_class_output("head.cls.p3.pw1.conv.weight"));
        assert!(!is_class_output("head.box.p3.out.bias"));
        assert!(!is_class_output("head.cls.p6.out.bias"));
    }
}
