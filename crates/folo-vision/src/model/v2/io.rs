//! `.folo` weight IO for [`FoloV2Model`] through folo-nn (docs/FORMAT.md, docs/MODEL.md §11).

use std::path::Path;

use folo_core::{DType, Device};
use folo_nn::io::{
    FileKind, FloatDType, LoadOptions, LoadReport, Metadata, ModelFile, Purpose, export_module, save_weights,
};
use folo_nn::{Error, Module, Result};

use super::{FoloV2Config, FoloV2Model};

impl FoloV2Model {
    /// The metadata of a weights file of this model (`folo.arch = "folo2-<scale>"`, `folo.config`,
    /// `folo.classes`, `folo.imgsz`, and `folo.fused` when fused).
    pub fn metadata(&self) -> Result<Metadata> {
        let mc = self.config.to_model_config()?;
        let mut m = Metadata::new(FileKind::Weights, mc, self.classes.clone(), self.imgsz)?;
        m.fused = self.fused;
        Ok(m)
    }

    /// Write the current tensors (parameters and BatchNorm buffers, or the fused set) as a `weights` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_weights(path, &self.metadata()?, self)
    }

    /// Export a fused `.folo` weights file at `dtype`: folds all `WS-QARepConv3x3` multi-branch kernels and
    /// BatchNorms on a detached `F32` clone (leaving `self` untouched), casts to `dtype`, and writes atomically.
    pub fn export(&self, path: impl AsRef<Path>, dtype: FloatDType) -> Result<()> {
        let mut fused = self.fused_eval_clone()?;
        fused.to_device(Device::Cpu)?;
        fused.to_dtype(DType::F32)?;
        export_module(path, &fused.metadata()?, &fused, dtype)
    }

    /// Build the [`FoloV2Model`] described by `file` (`folo.config` with `family: "folo2"`) and load it strictly.
    pub fn from_file(file: &ModelFile, purpose: Purpose) -> Result<(Self, LoadReport)> {
        let meta = file.metadata();
        let cfg = FoloV2Config::from_model_config(&meta.config)?;
        let mut model = FoloV2Model::new(cfg)?;
        model.set_classes(meta.classes.clone())?;
        model.set_imgsz(meta.imgsz);
        if meta.fused && purpose == Purpose::Inference {
            model.fuse()?;
        }
        let report = model.load(file, LoadOptions::strict(purpose))?;
        if purpose == Purpose::Inference {
            model.eval();
        }
        Ok((model, report))
    }

    /// Open a `.folo` file for inference ([`FoloV2Model::from_file`] with [`Purpose::Inference`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_file(&ModelFile::open(path)?, Purpose::Inference)?.0)
    }

    /// [`FoloV2Model::from_file`], then move the model to `device`.
    pub fn from_file_on(file: &ModelFile, purpose: Purpose, device: Device) -> Result<(Self, LoadReport)> {
        let (mut model, report) = Self::from_file(file, purpose)?;
        model.to_device(device)?;
        Ok((model, report))
    }

    /// [`FoloV2Model::open`] straight onto `device`.
    pub fn open_on(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Ok(Self::from_file_on(&ModelFile::open(path)?, Purpose::Inference, device)?.0)
    }

    /// A fused, eval-mode [`FoloV2Model`] ready for inference on `device` in `dtype` (`F32` or `F16`).
    pub fn from_file_for_inference(file: &ModelFile, device: Device, dtype: DType) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!("inference dtype must be f32 or f16, got {dtype}")));
        }
        let meta = file.metadata();
        if dtype == DType::F16 && meta.fused && meta.dtype == FloatDType::F16 {
            let cfg = FoloV2Config::from_model_config(&meta.config)?;
            let mut model = FoloV2Model::new(cfg)?;
            model.set_classes(meta.classes.clone())?;
            model.set_imgsz(meta.imgsz);
            model.fuse()?;
            model.to_dtype(DType::F16)?;
            model.to_device(device)?;
            model.load(file, LoadOptions::strict(Purpose::Inference))?;
            return Ok(model);
        }
        let (mut model, _) = Self::from_file(file, Purpose::Inference)?;
        model.fuse()?;
        model.to_dtype(dtype)?;
        model.to_device(device)?;
        Ok(model)
    }

    /// Load `file` into this model with `opts` (FORMAT.md loading rules).
    pub fn load(&self, file: &ModelFile, mut opts: LoadOptions) -> Result<LoadReport> {
        if self.fused && opts.purpose == Purpose::Training {
            return Err(Error::Load(
                "this model is fused (WS-QARep + Conv+BN folded) and can only be loaded for inference; build an \
                 unfused model to train"
                    .into(),
            ));
        }
        if opts.classes.is_none() {
            opts.classes = Some(self.classes.clone());
        }
        file.load(self, &opts)
    }

    /// Transfer learning: partial loading for training (copies loaded `head.cls.*` weights into `head.o2o.*.cls`
    /// when initialising a dual-head `o2o` model from an `o2m` checkpoint).
    pub fn transfer_from(&self, file: &ModelFile) -> Result<LoadReport> {
        let mut report = self.load(file, LoadOptions::partial(Purpose::Training))?;
        if self.has_o2o() && !report.loaded.iter().any(|n| n.starts_with("head.o2o.")) {
            let tensors: std::collections::HashMap<String, folo_core::Tensor> =
                self.state().into_iter().collect();
            let mut copied = Vec::new();
            for name in &report.loaded {
                let Some(rest) = name.strip_prefix("head.cls.") else { continue };
                let Some((level, suffix)) = rest.split_once('.') else { continue };
                let target = format!("head.o2o.{level}.cls.{suffix}");
                if let (Some(src), Some(dst)) = (tensors.get(name), tensors.get(&target))
                    && src.shape() == dst.shape()
                {
                    folo_nn::init::copy_(dst, src)?;
                    copied.push(target);
                }
            }
            report.missing.retain(|n| !copied.contains(n));
            report.loaded.extend(copied);
        }
        Ok(report)
    }

    /// Every tensor (parameters and buffers) with its name, in canonical order.
    pub fn state(&self) -> Vec<(String, folo_core::Tensor)> {
        self.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect()
    }
}
