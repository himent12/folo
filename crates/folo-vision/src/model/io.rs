//! `.folo` weight IO for [`FoloModel`] through folo-nn (docs/FORMAT.md).

use std::path::Path;

use folo_core::{DType, Device};
use folo_nn::io::{
    FileKind, FloatDType, LoadOptions, LoadReport, Metadata, ModelFile, Purpose, export_module, save_weights,
};
use folo_nn::{Error, Module, Result};

use super::{FoloModel, quant};

impl FoloModel {
    /// The metadata of a weights file of this model: `folo.arch`, `folo.config`, `folo.classes`,
    /// `folo.imgsz`, `folo.fused` when fused, and `folo.quant` when an int8 calibration plan is active.
    pub fn metadata(&self) -> Result<Metadata> {
        let mut m = Metadata::new(FileKind::Weights, self.config.clone(), self.classes.clone(), self.imgsz)?;
        m.fused = self.fused;
        if let Some(cal) = self.int8() {
            let matches_weights = match cal.weights.as_deref() {
                Some(want) => quant::fingerprint(self)? == want,
                None => true,
            };
            if matches_weights {
                cal.store(&mut m)?;
            }
        }
        Ok(m)
    }

    /// Write the current tensors (parameters and BatchNorm buffers, or the fused set) as a `weights` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_weights(path, &self.metadata()?, self)
    }

    /// `folo export` from memory: fuse Conv+BN in `F32` (unless already fused), cast to `dtype` and write a
    /// fused `weights` file. The model itself is not modified.
    pub fn export(&self, path: impl AsRef<Path>, dtype: FloatDType) -> Result<()> {
        let mut meta = self.metadata()?;
        if dtype != FloatDType::F32 {
            meta.extra.remove(quant::META_KEY);
        }
        export_module(path, &meta, self, dtype)
    }

    /// Build the model a file describes (`folo.config`, `folo.classes`, `folo.imgsz`) and load it strictly.
    ///
    /// For [`Purpose::Inference`] the EMA weights are used when present, a fused file gives a fused model, and
    /// the model is returned in eval mode. [`Purpose::Training`] reads the model namespace, refuses fused files,
    /// and returns the model in training mode.
    pub fn from_file(file: &ModelFile, purpose: Purpose) -> Result<(Self, LoadReport)> {
        let meta = file.metadata();
        let mut model = FoloModel::new(meta.config.clone())?;
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

    /// Open a `.folo` file for inference ([`FoloModel::from_file`] with [`Purpose::Inference`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_file(&ModelFile::open(path)?, Purpose::Inference)?.0)
    }

    /// [`FoloModel::from_file`], then move the model to `device` (the tensors are read on the CPU and
    /// uploaded once each).
    pub fn from_file_on(file: &ModelFile, purpose: Purpose, device: Device) -> Result<(Self, LoadReport)> {
        let (mut model, report) = Self::from_file(file, purpose)?;
        model.to_device(device)?;
        Ok((model, report))
    }

    /// [`FoloModel::open`] straight onto `device` (for example `folo_gpu::init()?`).
    pub fn open_on(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Ok(Self::from_file_on(&ModelFile::open(path)?, Purpose::Inference, device)?.0)
    }

    /// A fused, eval-mode model for inference on `device` in `dtype` (`F32`, or `F16` on a device that computes
    /// in it, `folo_core::backend::computes_f16`).
    ///
    /// A fused `F16` file (`folo export --format folo-f16`) loaded as `F16` never goes through `F32`: the
    /// fused model is converted and moved first, and each file tensor is uploaded as the `F16` it is. Any other
    /// file is loaded in `F32` (FORMAT.md's `F16 → F32` rule), fused in `F32` and converted on the CPU, then moved.
    pub fn from_file_for_inference(file: &ModelFile, device: Device, dtype: DType) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!("inference dtype must be f32 or f16, got {dtype}")));
        }
        let meta = file.metadata();
        if dtype == DType::F16 && meta.fused && meta.dtype == FloatDType::F16 {
            let mut model = FoloModel::new(meta.config.clone())?;
            model.set_classes(meta.classes.clone())?;
            model.set_imgsz(meta.imgsz);
            model.fuse()?;
            model.to_dtype(DType::F16)?;
            model.to_device(device)?;
            model.load(file, LoadOptions::strict(Purpose::Inference))?;
            return Ok(model);
        }
        // Fused and converted on the CPU, then uploaded once (fusing on the device would read every block back).
        let (mut model, _) = Self::from_file(file, Purpose::Inference)?;
        model.fuse()?;
        model.to_dtype(dtype)?;
        model.to_device(device)?;
        Ok(model)
    }

    /// Load a fused CPU `F32` model from `file` and attach its stored `folo.quant` int8 calibration plan.
    pub fn from_file_int8(file: &ModelFile) -> Result<Self> {
        let cal = quant::Calibration::from_metadata(file.metadata())?.ok_or_else(|| {
            Error::Metadata(format!("`.folo` file has no `{}` int8 calibration metadata", quant::META_KEY))
        })?;
        let mut model = Self::from_file_for_inference(file, Device::Cpu, DType::F32)?;
        model.set_int8(Some(cal))?;
        Ok(model)
    }

    /// Open a quantized `.folo` file for CPU int8 inference ([`FoloModel::from_file_int8`]).
    pub fn open_int8(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_int8(&ModelFile::open(path)?)
    }

    /// Load `file` into this model with `opts` (FORMAT.md loading rules). When `opts.classes` is unset it is
    /// filled with this model's class list, so strict loading requires identical classes and partial loading
    /// skips the class-output layers (`head.cls.*.out.*`) when they differ. A fused model can only be loaded
    /// for inference: its tensor set has no BatchNorm to train, so [`Purpose::Training`] is refused before any
    /// tensor is written.
    pub fn load(&self, file: &ModelFile, mut opts: LoadOptions) -> Result<LoadReport> {
        if self.fused && opts.purpose == Purpose::Training {
            return Err(Error::Load(
                "this model is fused (Conv+BN folded) and can only be loaded for inference; build an unfused                  model to train"
                    .into(),
            ));
        }
        if opts.classes.is_none() {
            opts.classes = Some(self.classes.clone());
        }
        let report = file.load(self, &opts)?;
        // Invalidate a fingerprinted int8 plan if replacement weights no longer match its recorded fingerprint.
        if let Some(plan) = self.active_int8()
            && let Some(want) = plan.calibration().weights.as_deref()
            && quant::fingerprint(self)? != want
        {
            plan.invalidate();
        }
        Ok(report)
    }

    /// Transfer learning: partial loading for training. Every tensor whose name, shape and dtype match is
    /// loaded; with a different class list the class-output layers are skipped (keeping this model's
    /// initialisation and priors), and so is anything whose shape changed with the class count.
    ///
    /// Into a model with the one-to-one head (`head = "o2o"`) from a file without one (an `o2m` model), the
    /// one-to-one branch starts as a copy of the one-to-many head tensors that loaded (as YOLOv10 initialises it);
    /// they are listed in the report's `loaded` as `head.o2o.*`.
    pub fn transfer_from(&self, file: &ModelFile) -> Result<LoadReport> {
        let mut report = self.load(file, LoadOptions::partial(Purpose::Training))?;
        if self.has_o2o() && !report.loaded.iter().any(|n| n.starts_with("head.o2o.")) {
            let tensors: std::collections::HashMap<String, folo_core::Tensor> =
                self.state().into_iter().collect();
            let mut copied = Vec::new();
            for name in &report.loaded {
                let Some(rest) = name.strip_prefix("head.") else { continue };
                let target = format!("head.o2o.{rest}");
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

    /// Every tensor (parameters and buffers) with its name, in canonical order: what a weights file holds.
    pub fn state(&self) -> Vec<(String, folo_core::Tensor)> {
        self.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect()
    }
}
