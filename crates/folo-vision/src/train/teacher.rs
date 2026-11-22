//! The frozen teacher of a distillation run (`TrainConfig.distill`, docs/MODEL.md §8.9): loaded once when the
//! trainer is built, fused, in eval mode on the trainer's device (`F16` on a GPU under mixed precision, else `F32`),
//! and run under `no_grad` on each training batch: its raw outputs feed the fused distillation kernel on a GPU, the
//! [`DistillTargets`] of that batch elsewhere.

use std::path::{Path, PathBuf};

use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::ModelFile;

use crate::error::{Error, Result};
use crate::loss::DistillAssignment;
use crate::loss::distill::{
    DistillConfig, DistillLoss, DistillTargets, TeacherView, distill_loss_fused_with_context,
    distill_loss_with_context, fused_supported, gt_verify_weights_cpu,
};
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

/// Refuses a teacher file that is `run_dir`'s own `weights/last.folo` or `weights/best.folo` (compared after
/// resolving links and `..`): training overwrites both, so the next resume would load another teacher.
fn refuse_own_checkpoint(teacher: &Path, run_dir: &Path) -> Result<()> {
    let Ok(t) = teacher.canonicalize() else { return Ok(()) };
    for name in ["last.folo", "best.folo"] {
        if run_dir.join("weights").join(name).canonicalize().is_ok_and(|c| c == t) {
            return Err(Error::Config(format!(
                "the teacher {} (distill.teacher) is this run's own checkpoint, which training overwrites; copy it \
                 first and pass the copy",
                folo_api::json::path(teacher)
            )));
        }
    }
    Ok(())
}

/// A loaded teacher.
pub(super) struct Teacher {
    model: FoloModel,
    dtype: DType,
    cfg: DistillConfig,
}

impl Teacher {
    /// Loads `cfg.teacher` for `student` on `device` (`F16` when `amp` on a GPU) and checks that it can teach it:
    /// the same number of classes, `reg_max` and strides, so both heads have the same anchors and channels. Returns
    /// the teacher and the log lines of its loading (an info line, and a warning when the class names differ).
    /// With a run folder `run_dir`, a teacher that is that run's own `weights/last.folo` or `best.folo` is refused.
    pub(super) fn load(
        cfg: &DistillConfig,
        student: &FoloModel,
        device: Device,
        amp: bool,
        run_dir: Option<&Path>,
    ) -> Result<(Self, Vec<(folo_api::LogLevel, String)>)> {
        let path = resolve_teacher(&cfg.teacher)?;
        if let Some(dir) = run_dir {
            refuse_own_checkpoint(&path, dir)?;
        }
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
                "distilling from {shown} ({}, {} parameters, {dtype}): cls_gain {}, dfl_gain {}, temperature {}",
                model.arch(),
                model.num_parameters(),
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
        Ok((Teacher { model, dtype, cfg: cfg.clone() }, logs))
    }

    /// The teacher's side of the batch `x` (`[B, 3, H, W]`, the student's input), without autograd: one teacher
    /// forward, kept raw where the fused kernel runs ([`fused_supported`]), else turned into [`DistillTargets`].
    pub(super) fn batch(&self, x: &Tensor) -> Result<TeacherBatch> {
        let _g = no_grad();
        let out = self.model.forward_head(&x.to_dtype(self.dtype)?, HeadMode::O2m)?;
        if fused_supported(&out.cls_raw, out.reg_max) {
            return Ok(TeacherBatch::Raw { box_raw: out.box_raw, cls_raw: out.cls_raw });
        }
        let view = TeacherView::new(&out.box_raw, &out.cls_raw, out.reg_max, self.cfg.temperature)?;
        let raw_box = self.cfg.gt_verify.then_some(out.box_raw);
        Ok(TeacherBatch::Targets { targets: DistillTargets::new(&view)?, raw_box, reg_max: out.reg_max })
    }
}

/// One batch's teacher outputs, as [`Teacher::batch`] gives them.
pub(super) enum TeacherBatch {
    /// The raw head outputs (`F16` under mixed precision), for [`distill_loss_fused_with_context`].
    Raw { box_raw: Tensor, cls_raw: Tensor },
    /// The constants of [`distill_loss_with_context`].
    Targets { targets: DistillTargets, raw_box: Option<Tensor>, reg_max: usize },
}

impl TeacherBatch {
    /// The distillation term of the student's f32 one-to-many outputs against this batch (docs/MODEL.md §8.9).
    pub(super) fn loss(
        &self,
        box_raw: &Tensor,
        cls_raw: &Tensor,
        anchors: &crate::decode::Anchors,
        target_score_sum: f32,
        cfg: &DistillConfig,
        asg: Option<&DistillAssignment>,
    ) -> Result<DistillLoss> {
        Ok(match self {
            TeacherBatch::Raw { box_raw: tb, cls_raw: tc } => {
                let dev_asg = match asg {
                    Some(DistillAssignment::Device(a)) => Some(a),
                    _ => None,
                };
                distill_loss_fused_with_context(box_raw, cls_raw, tb, tc, target_score_sum, cfg, dev_asg)?
            }
            TeacherBatch::Targets { targets, raw_box, reg_max } => {
                let host_asg = match asg {
                    Some(DistillAssignment::Host(a)) => Some(a),
                    _ => None,
                };
                let ver_w = match (cfg.gt_verify, raw_box, host_asg) {
                    (true, Some(tb), Some(a)) => {
                        Some(gt_verify_weights_cpu(box_raw, tb, anchors, a, *reg_max)?)
                    }
                    _ => None,
                };
                distill_loss_with_context(
                    box_raw,
                    cls_raw,
                    targets,
                    target_score_sum,
                    cfg,
                    host_asg,
                    ver_w.as_ref(),
                )?
            }
        })
    }
}
