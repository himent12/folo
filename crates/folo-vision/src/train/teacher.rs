//! The frozen teacher of a distillation run (`TrainConfig.distill`, docs/MODEL.md §8.9): loaded once when the
//! trainer is built, fused, in eval mode on the trainer's device (`F16` on a GPU under mixed precision, else `F32`),
//! and run under `no_grad` on each training batch to build the [`DistillTargets`] of that batch.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::ModelFile;

use crate::error::{Error, Result};
use crate::loss::distill::{Consensus, DistillConfig, DistillTargets, FlipMap, TeacherView};
use crate::model::{FoloModel, HeadMode};

/// The file `distill.teacher` names: a `.folo` file, or the cached file of a published name (`folo weights pull`).
/// An existing file wins over a published name, as for `--weights`. Errors (`Error::Config`) name the key and say
/// how to get the file.
pub fn resolve_teacher(teacher: &str) -> Result<PathBuf> {
    let p = Path::new(teacher);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if let Some(w) = folo_api::weights::find(teacher) {
        let path = folo_api::weights::cache_dir().join(w.asset);
        if path.is_file() {
            return Ok(path);
        }
        return Err(Error::Config(format!(
            "the teacher `{teacher}` (distill.teacher) is not downloaded ({} does not exist); run `folo weights pull \
             {teacher}`",
            folo_api::json::path(&path)
        )));
    }
    Err(Error::Config(format!(
        "the teacher {} (distill.teacher) does not exist; pass a .folo weights file or a published name ({})",
        folo_api::json::path(p),
        folo_api::weights::names()
    )))
}

/// A loaded teacher and the flip maps of the input shapes it has seen.
pub(super) struct Teacher {
    model: FoloModel,
    dtype: DType,
    cfg: DistillConfig,
    flips: HashMap<(usize, usize), FlipMap>,
}

impl Teacher {
    /// Loads `cfg.teacher` for `student` on `device` (`F16` when `amp` on a GPU) and checks that it can teach it:
    /// the same number of classes, `reg_max` and strides, so both heads have the same anchors and channels. Returns
    /// the teacher and the log lines of its loading (an info line, and a warning when the class names differ).
    pub(super) fn load(
        cfg: &DistillConfig,
        student: &FoloModel,
        device: Device,
        amp: bool,
    ) -> Result<(Self, Vec<(folo_api::LogLevel, String)>)> {
        let path = resolve_teacher(&cfg.teacher)?;
        let shown = folo_api::json::path(&path);
        let file = ModelFile::open(&path)?;
        let dtype = if amp && !device.is_cpu() { DType::F16 } else { DType::F32 };
        let model = FoloModel::from_file_for_inference(&file, device, dtype)?;
        let incompatible = |what: &str, t: String, s: String| {
            Err(Error::Config(format!(
                "the teacher {shown} has {what} {t}, the student {s}: distillation needs a teacher with the student's \
                 number of classes, reg_max and strides (distill.teacher)"
            )))
        };
        if model.num_classes() != student.num_classes() {
            return incompatible("nc", model.num_classes().to_string(), student.num_classes().to_string());
        }
        if model.head.reg_max != student.head.reg_max {
            return incompatible("reg_max", model.head.reg_max.to_string(), student.head.reg_max.to_string());
        }
        if model.strides() != student.strides() {
            return incompatible(
                "strides",
                format!("{:?}", model.strides()),
                format!("{:?}", student.strides()),
            );
        }
        // Its weights never change, so the recorded GPU forwards replay for the whole run.
        model.set_forward_plans(true);
        let mut logs = vec![(
            folo_api::LogLevel::Info,
            format!(
                "distilling from {shown} ({}, {} parameters, {dtype}, consensus {}): cls_gain {}, dfl_gain {}, \
                 temperature {}",
                model.arch(),
                model.num_parameters(),
                cfg.consensus.as_str(),
                cfg.cls_gain,
                cfg.dfl_gain,
                cfg.temperature
            ),
        )];
        if model.classes() != student.classes() {
            logs.push((
                folo_api::LogLevel::Warn,
                format!(
                    "the teacher's class names differ from the dataset's (teacher {:?}, dataset {:?}); classes are \
                     matched by index",
                    model.classes(),
                    student.classes()
                ),
            ));
        }
        Ok((Teacher { model, dtype, cfg: cfg.clone(), flips: HashMap::new() }, logs))
    }

    /// The targets of the batch `x` (`[B, 3, H, W]`, the student's input), without autograd: one teacher forward,
    /// or two for the flip consensus (the second on `x` mirrored along `W`, mapped back and averaged).
    pub(super) fn targets(&mut self, x: &Tensor) -> Result<DistillTargets> {
        let _g = no_grad();
        let (model, dtype, temp) = (&self.model, self.dtype, self.cfg.temperature);
        let view = |x: &Tensor| -> Result<TeacherView> {
            let out = model.forward_head(&x.to_dtype(dtype)?, HeadMode::O2m)?;
            Ok(TeacherView::new(&out.box_raw, &out.cls_raw, out.reg_max, temp)?)
        };
        let plain = view(x)?;
        let v = match self.cfg.consensus {
            Consensus::None => plain,
            Consensus::Flip => {
                let &[_, _, h, w] = x.dims() else {
                    return Err(Error::Config(format!("the batch {} is not [B, 3, H, W]", x.shape())));
                };
                let map = match self.flips.entry((h, w)) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(FlipMap::new(&self.model.anchors(h, w), w, x.device())?)
                    }
                };
                let mirrored = view(&map.flip_input(x)?)?.mapped(map)?;
                plain.average(&mirrored)?
            }
        };
        Ok(DistillTargets::new(&v)?)
    }
}
