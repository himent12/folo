//! `.folo` weight IO for [`FoloModel`] through folo-nn (docs/FORMAT.md).

use std::path::Path;

use folo_nn::io::{
    FileKind, FloatDType, LoadOptions, LoadReport, Metadata, ModelFile, Purpose, export_module, save_weights,
};
use folo_nn::{Error, Module, Result};

use super::FoloModel;

impl FoloModel {
    /// The metadata of a weights file of this model: `folo.arch`, `folo.config`, `folo.classes`,
    /// `folo.imgsz`, and `folo.fused` when fused.
    pub fn metadata(&self) -> Result<Metadata> {
        let mut m = Metadata::new(FileKind::Weights, self.config.clone(), self.classes.clone(), self.imgsz)?;
        m.fused = self.fused;
        Ok(m)
    }

    /// Write the current tensors (parameters and BatchNorm buffers, or the fused set) as a `weights` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_weights(path, &self.metadata()?, self)
    }

    /// `folo export` from memory: fuse Conv+BN in `F32` (unless already fused), cast to `dtype` and write a
    /// fused `weights` file. The model itself is not modified.
    pub fn export(&self, path: impl AsRef<Path>, dtype: FloatDType) -> Result<()> {
        export_module(path, &self.metadata()?, self, dtype)
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
        file.load(self, &opts)
    }

    /// Transfer learning: partial loading for training. Every tensor whose name, shape and dtype match is
    /// loaded; with a different class list the class-output layers are skipped (keeping this model's
    /// initialisation and priors), and so is anything whose shape changed with the class count.
    pub fn transfer_from(&self, file: &ModelFile) -> Result<LoadReport> {
        self.load(file, LoadOptions::partial(Purpose::Training))
    }

    /// Every tensor (parameters and buffers) with its name, in canonical order: what a weights file holds.
    pub fn state(&self) -> Vec<(String, folo_core::Tensor)> {
        self.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect()
    }
}
