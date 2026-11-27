//! GPU parity tests for FOLO v2-Iso ([`FoloV2Model`]):
//! - Candidate-first GPU head (`v2/lazy_gpu.rs`) against the dense GPU path, bit for bit, in `F32` and `F16`,
//!   for both `o2m` and `o2o` (`PeakSuppress`), with `folo_gpu::poison_uninit(true)`, across sparse, empty,
//!   and slot-overflow thresholds, with DFL box fusion (`Dfl` / `Sebf`) and score rules (`Elc` / `DflConsensus`),
//!   and with [`folo_gpu::ForwardPlans`] recording + replay.
//! - Dense GPU `ForwardPlans` recording + bit-identical replay for `FoloV2Model` (including `o2o` + `iou_head`)
//!   and end-to-end [`Predictor::new_v2`] GPU parity against CPU.
//! - GPU dense distillation (`--teacher`) with a `folo2-n` teacher (exercising teacher GPU `ForwardPlans`,
//!   `iou_head` + `bckd_gamma == 0` and `bckd_gamma > 0` + `gt_verify`).

use std::sync::{Arc, Mutex, MutexGuard};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Image, Split};
use folo_nn::Module;
use folo_nn::io::ModelFile;
use folo_test_support::{TestDir, TestFile};
use folo_vision::fusion::{BoxFusion, ScoreRule};
use folo_vision::model::HeadMode;
use folo_vision::model::v2::{DEFAULT_IOU_FUSION_ALPHA, FoloV2Config, FoloV2Model};
use folo_vision::nms::{Detection, NmsConfig};
use folo_vision::predict::{PredictConfig, Predictor};
use folo_vision::train::{OptimizerChoice, ResumeOverrides};
use folo_vision::{DistillConfig, TrainConfig, TrainData, TrainOptions, Trainer};

fn gpu() -> Option<(Device, folo_gpu::GpuContext, MutexGuard<'static, ()>)> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    folo_gpu::request_tensor_cores();
    folo_gpu::poison_uninit(true);
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    let dev = folo_gpu::init().ok()?;
    Some((dev, ctx, guard))
}

fn bits(d: &[Detection]) -> Vec<[u32; 6]> {
    d.iter()
        .map(|d| {
            let b = d.bbox;
            [d.class, d.score.to_bits(), b.x1.to_bits(), b.y1.to_bits(), b.x2.to_bits(), b.y2.to_bits()]
        })
        .collect()
}

/// Build a `folo2-n` model with non-trivial BN stats, active `FullPadGate` weights, and controlled `cls` biases
/// so sparse, moderate, and overflow candidate regimes are all reachable.
fn v2_model(
    nc: usize,
    head: HeadMode,
    iou_head: bool,
    dev: Device,
    dtype: DType,
    cls_bias: f32,
) -> FoloV2Model {
    let cfg = FoloV2Config::new("n", nc).unwrap().with_head(head).with_iou_head(iou_head);
    let m = FoloV2Model::with_rng(cfg, &mut Rng::seed(13)).unwrap();
    let rng = &mut Rng::seed(29);
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            let n = t.tensor.numel();
            if t.name.ends_with("running_mean") {
                t.tensor.copy_from(&Tensor::rand_uniform([n], -0.2, 0.2, rng)).unwrap();
            } else if t.name.ends_with("running_var") {
                t.tensor.copy_from(&Tensor::rand_uniform([n], 0.5, 1.5, rng)).unwrap();
            } else if t.name.contains("pad_p") && t.name.ends_with(".weight") {
                t.tensor
                    .copy_from(&Tensor::rand_uniform(t.tensor.shape().clone(), -0.15, 0.15, rng))
                    .unwrap();
            } else if (t.name.contains("head.cls.") || t.name.contains("head.o2o."))
                && t.name.ends_with(".out.bias")
            {
                let b = t.tensor.zeros_like().unwrap().add_scalar(cls_bias).unwrap();
                t.tensor.copy_from(&b).unwrap();
            }
        }
    }
    let path = TestFile::new(&format!("v2-gpu-{}-{nc}-{iou_head}", head.as_str()), "m.folo");
    m.save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    FoloV2Model::from_file_for_inference(&file, dev, dtype).unwrap()
}

fn check_lazy_vs_dense(m: &FoloV2Model, x: &Tensor, nms: &NmsConfig, what: &str) {
    assert!(m.candidates_first_applies(x.device(), nms), "{what}: candidates_first_applies must be true");
    let want = {
        let _g = no_grad();
        m.forward(x).unwrap().detections(nms, 0.0).unwrap()
    };
    let cands = m.predict_candidates(x, nms).unwrap();
    let got: Vec<Vec<Detection>> = cands.iter().map(|c| m.finish_candidates(c, nms)).collect();
    assert_eq!(got.len(), want.len(), "{what}: batch size");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(bits(g), bits(w), "{what}: image {i} mismatch (got {} vs want {})", g.len(), w.len());
    }
}

#[test]
fn v2_gpu_lazy_head_matches_dense_bit_for_bit_f32_and_f16() {
    let Some((dev, ctx, _guard)) = gpu() else { return };
    let mut dtypes = vec![DType::F32];
    if ctx.caps().shader_f16 {
        dtypes.push(DType::F16);
    }

    for dtype in dtypes {
        for head in [HeadMode::O2m, HeadMode::O2o] {
            let m = v2_model(5, head, false, dev, dtype, -0.4);
            let mut rng = Rng::seed(7);
            // 2-image batch at 160x160: P3 has 20x20=400, P4 10x10=100, P5 5x5=25 -> 525 anchors.
            let x = Tensor::rand_uniform([2, 3, 160, 160], 0.0, 1.0, &mut rng)
                .to_device(dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();

            // Empty (0.9999), sparse (0.45), moderate (0.35), and full-map (0.001).
            for conf in [0.9999, 0.45, 0.35, 0.001] {
                let nms =
                    NmsConfig { conf, iou: 1.0, max_det: 1000, multi_label: false, ..NmsConfig::default() };
                let tag = format!("{dtype:?} {} conf={conf}", head.as_str());
                check_lazy_vs_dense(&m, &x, &nms, &tag);
            }

            // DFL fusion (`Dfl`, `Sebf`) and score rules (`Elc`, `DflConsensus`) on `o2m` (and `o2o` ignores
            // them cleanly with bit-exact parity).
            for (fusion, score) in [
                (BoxFusion::Dfl, ScoreRule::None),
                (BoxFusion::Sebf, ScoreRule::Elc),
                (BoxFusion::Dfl, ScoreRule::DflConsensus),
            ] {
                let nms = NmsConfig {
                    conf: 0.38,
                    iou: 0.7,
                    max_det: 100,
                    multi_label: false,
                    fusion,
                    score,
                    ..NmsConfig::default()
                };
                let tag = format!("{dtype:?} {} {fusion:?} {score:?}", head.as_str());
                check_lazy_vs_dense(&m, &x, &nms, &tag);
            }

            // Slot-overflow fallback test: at 320x320, P3 has 40x40=1600 > GPU_SITE_SLOTS[0]=1024; at conf=0.0
            // every anchor passes, forcing overflow -> dense fallback bit-exactness.
            let x_big = Tensor::rand_uniform([1, 3, 320, 320], 0.0, 1.0, &mut rng)
                .to_device(dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let overflow_nms =
                NmsConfig { conf: 0.0, iou: 0.7, max_det: 300, multi_label: false, ..NmsConfig::default() };
            check_lazy_vs_dense(&m, &x_big, &overflow_nms, &format!("{dtype:?} {} overflow", head.as_str()));

            // Plans are opt-in on standalone `FoloV2Model`: zero replays before `set_forward_plans(true)`.
            assert_eq!(
                m.plan_replays(),
                0,
                "{dtype:?} {} standalone model must start with plans off",
                head.as_str()
            );

            // Recorded ForwardPlans on the candidate-first GPU path: 1st call warms, 2nd records, 3rd+ replays.
            m.set_forward_plans(true);
            let plan_nms = NmsConfig {
                conf: 0.38,
                iou: 0.7,
                max_det: 100,
                multi_label: false,
                fusion: BoxFusion::Sebf,
                score: ScoreRule::DflConsensus,
                ..NmsConfig::default()
            };
            let want = {
                let _g = no_grad();
                m.forward(&x).unwrap().detections(&plan_nms, 0.0).unwrap()
            };
            let replays_before_sites = m.plan_replays();
            for step in 0..4 {
                let cands = m.predict_candidates(&x, &plan_nms).unwrap();
                let got: Vec<Vec<Detection>> =
                    cands.iter().map(|c| m.finish_candidates(c, &plan_nms)).collect();
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(bits(g), bits(w), "{dtype:?} {} plan step {step} img {i}", head.as_str());
                }
            }
            let site_replays = m.plan_replays() - replays_before_sites;
            assert!(
                site_replays >= 2,
                "{dtype:?} {} expected >= 2 site plan replays, got {site_replays} (refusal: {:?})",
                head.as_str(),
                m.plan_refusal()
            );
        }
    }
}

#[test]
fn v2_gpu_forward_plans_dense_iou_head_and_predictor_parity() {
    let Some((dev, _ctx, _guard)) = gpu() else { return };

    // 1) Dense `ForwardPlans` on `folo2-n` with `o2o` + `iou_head` (3 output tensors).
    let m_gpu = v2_model(4, HeadMode::O2o, true, dev, DType::F32, -0.2);
    let mut m_cpu = m_gpu.fused_eval_clone().unwrap();
    m_cpu.to_device(Device::Cpu).unwrap();

    // `iou_head` must disable candidate-first and go through dense `forward_planned`.
    assert!(!m_gpu.candidates_first_applies(dev, &NmsConfig::default()));

    let mut rng = Rng::seed(91);
    let x_cpu = Tensor::rand_uniform([1, 3, 128, 128], 0.0, 1.0, &mut rng);
    let x_gpu = x_cpu.to_device(dev).unwrap();

    // Reference without plans.
    let (ref_b, ref_c, ref_i) = {
        let _g = no_grad();
        let o = m_gpu.forward(&x_gpu).unwrap();
        (
            o.box_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap(),
            o.cls_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap(),
            o.iou_raw.as_ref().unwrap().to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap(),
        )
    };
    let cpu_out = {
        let _g = no_grad();
        m_cpu.forward(&x_cpu).unwrap()
    };
    let cpu_b = cpu_out.box_raw.to_vec::<f32>().unwrap();
    let cpu_c = cpu_out.cls_raw.to_vec::<f32>().unwrap();
    let cpu_i = cpu_out.iou_raw.as_ref().unwrap().to_vec::<f32>().unwrap();
    let max_diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(max_diff(&ref_b, &cpu_b) < 2e-3, "box_raw GPU vs CPU: {}", max_diff(&ref_b, &cpu_b));
    assert!(max_diff(&ref_c, &cpu_c) < 2e-3, "cls_raw GPU vs CPU: {}", max_diff(&ref_c, &cpu_c));
    assert!(max_diff(&ref_i, &cpu_i) < 2e-3, "iou_raw GPU vs CPU: {}", max_diff(&ref_i, &cpu_i));

    // Plans must be off by default on a standalone `FoloV2Model`.
    assert_eq!(m_gpu.plan_replays(), 0, "standalone FoloV2Model must start with plans off");

    // Enable plans: 4 calls -> warmup + record + 2 replays, bit-exact with un-planned reference.
    m_gpu.set_forward_plans(true);
    for step in 0..4 {
        let _g = no_grad();
        let o = m_gpu.forward(&x_gpu).unwrap();
        let b = o.box_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap();
        let c = o.cls_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap();
        let i = o.iou_raw.as_ref().unwrap().to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {step} box_raw"
        );
        assert_eq!(
            c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {step} cls_raw"
        );
        assert_eq!(
            i.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_i.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {step} iou_raw"
        );
    }
    assert!(
        m_gpu.plan_replays() >= 2,
        "expected >= 2 dense iou_head plan replays, got {} (refusal: {:?})",
        m_gpu.plan_replays(),
        m_gpu.plan_refusal()
    );

    // Mutating a public layer activation while plans are enabled must never replay stale dispatches
    // (on either dense `forward` or candidate-first `predict_candidates`).
    let mut m_mut = v2_model(3, HeadMode::O2m, false, dev, DType::F32, -0.3);
    m_mut.set_forward_plans(true);
    let mut_nms = NmsConfig { conf: 0.35, iou: 0.7, max_det: 100, ..NmsConfig::default() };
    for _ in 0..3 {
        let _g = no_grad();
        let _ = m_mut.forward(&x_gpu).unwrap();
        let _ = m_mut.predict_candidates(&x_gpu, &mut_nms).unwrap();
    }
    let before_box = {
        let _g = no_grad();
        m_mut.forward(&x_gpu).unwrap().box_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap()
    };
    let before_cands: Vec<Vec<Detection>> = m_mut
        .predict_candidates(&x_gpu, &mut_nms)
        .unwrap()
        .iter()
        .map(|c| m_mut.finish_candidates(c, &mut_nms))
        .collect();
    m_mut.head.levels[0].box_branch.conv0.act = folo_nn::Activation::Identity;
    let after_planned_box = {
        let _g = no_grad();
        m_mut.forward(&x_gpu).unwrap().box_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap()
    };
    let after_planned_cands: Vec<Vec<Detection>> = m_mut
        .predict_candidates(&x_gpu, &mut_nms)
        .unwrap()
        .iter()
        .map(|c| m_mut.finish_candidates(c, &mut_nms))
        .collect();
    m_mut.set_forward_plans(false);
    let after_unplanned_box = {
        let _g = no_grad();
        m_mut.forward(&x_gpu).unwrap().box_raw.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap()
    };
    let after_unplanned_cands: Vec<Vec<Detection>> = m_mut
        .predict_candidates(&x_gpu, &mut_nms)
        .unwrap()
        .iter()
        .map(|c| m_mut.finish_candidates(c, &mut_nms))
        .collect();
    assert_ne!(before_box, after_unplanned_box, "activation change must alter box_raw");
    assert_eq!(after_planned_box, after_unplanned_box, "dense plan must not replay stale activation");
    assert_ne!(bits(&before_cands[0]), bits(&after_unplanned_cands[0]), "activation change must alter boxes");
    assert_eq!(
        bits(&after_planned_cands[0]),
        bits(&after_unplanned_cands[0]),
        "site plan must not replay stale activation"
    );

    // 2) End-to-end `Predictor::new_v2` on GPU (both candidate-first and dense `iou_head`) replays plans
    // bit-identically across calls and agrees with CPU `Predictor::new_v2` on top detections.
    let img = Image::new(128, 96, [120, 120, 120]);
    let pcfg = PredictConfig { imgsz: 128, conf: 0.35, ..PredictConfig::default() };
    for iou_head in [false, true] {
        let mg = v2_model(3, HeadMode::O2m, iou_head, dev, DType::F32, -0.3);
        let mut mc = mg.fused_eval_clone().unwrap();
        mc.to_device(Device::Cpu).unwrap();
        let pg = Predictor::new_v2(mg, pcfg.clone()).unwrap();
        let pc = Predictor::new_v2(mc, pcfg.clone()).unwrap();
        let ref_pred = pc.predict(&img).unwrap();
        let first = pg.predict(&img).unwrap();
        for step in 1..4 {
            let next = pg.predict(&img).unwrap();
            assert_eq!(
                bits(&next.detections),
                bits(&first.detections),
                "Predictor::new_v2 (iou_head={iou_head}) step {step} diverged across plan replay"
            );
        }
        assert!(
            pg.detector().plan_replays() >= 2,
            "Predictor::new_v2 (iou_head={iou_head}) expected >= 2 plan replays, got {} (refusal: {:?})",
            pg.detector().plan_replays(),
            pg.detector().plan_refusal()
        );
        // Top-scoring detections (well above `conf`) match CPU within GPU f32 tolerance.
        let top_k = first.detections.len().min(ref_pred.detections.len()).min(5);
        assert!(top_k > 0, "iou_head={iou_head}: expected non-empty detections");
        for (i, (g, c)) in first.detections[..top_k].iter().zip(&ref_pred.detections[..top_k]).enumerate() {
            assert_eq!(g.class, c.class, "iou_head={iou_head} top[{i}] class");
            assert!(
                (g.score - c.score).abs() < 2e-3,
                "iou_head={iou_head} top[{i}] score: gpu={} cpu={}",
                g.score,
                c.score
            );
            for (gv, cv) in [
                (g.bbox.x1, c.bbox.x1),
                (g.bbox.y1, c.bbox.y1),
                (g.bbox.x2, c.bbox.x2),
                (g.bbox.y2, c.bbox.y2),
            ] {
                assert!((gv - cv).abs() < 0.25, "iou_head={iou_head} top[{i}] box coord: gpu={gv} cpu={cv}");
            }
        }
        let _ = DEFAULT_IOU_FUSION_ALPHA;
    }
}

#[test]
fn v2_gpu_distillation_with_iou_head_and_teacher_plans() {
    let Some((_dev, _ctx, _guard)) = gpu() else { return };
    let dir = TestDir::new("gpu-v2-distill");

    let cfg = SyntheticConfig {
        images: 10,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 5,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 0.2,
        ..SyntheticConfig::default()
    };
    let data = || {
        let split = |s| Arc::new(SyntheticDataset::new(cfg.clone(), s).unwrap());
        TrainData { train: split(Split::Train), val: split(Split::Val), classes: cfg.classes.clone() }
    };

    // Save a `folo2-n` teacher with `o2o` + `iou_head` (exercises teacher GPU `ForwardPlans` via `O2m`).
    let t_cfg = FoloV2Config::new("n", 2).unwrap().with_head(HeadMode::O2o).with_iou_head(true);
    let mut t_model = FoloV2Model::with_rng(t_cfg, &mut Rng::seed(41)).unwrap();
    t_model.set_classes(cfg.classes.clone()).unwrap();
    let t_path = dir.join("v2-teacher.folo");
    t_model.save(&t_path).unwrap();
    let t_str = t_path.to_string_lossy().into_owned();

    let base = TrainConfig {
        arch: "folo2-n".into(),
        epochs: 1,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        device: "gpu".into(),
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 0.0,
        workers: 1,
        augment: AugmentConfig::none(),
        amp: false,
        ..TrainConfig::default()
    };

    // Build an epoch-0 `folo2-n` checkpoint with `iou_head = true` so `Trainer::resume` trains an `iou_head`
    // student on GPU under both plain (`bckd_gamma == 0`) and context-aware (`bckd_gamma > 0`, `gt_verify`)
    // distillation with 4 batches of 2 (so teacher GPU `ForwardPlans` replay).
    let ckpt_seed = dir.join("v2-iou-seed.folo");
    {
        use folo_nn::io::{BestState, CheckpointParts, FileKind, Metadata, RngState, save_checkpoint};
        let d0 = data();
        let v2_cfg = FoloV2Config::new("n", 2).unwrap().with_iou_head(true);
        let mut v2 = FoloV2Model::new(v2_cfg.clone()).unwrap();
        v2.set_classes(d0.classes.clone()).unwrap();
        v2.set_imgsz(base.imgsz);
        let mc = v2.config().to_model_config().unwrap();
        let meta = Metadata::new(FileKind::Checkpoint, mc, d0.classes.clone(), base.imgsz).unwrap();
        let mut train_json = serde_json::to_value(&base).unwrap();
        train_json.as_object_mut().unwrap().insert("train_images".into(), serde_json::json!(d0.train.len()));
        let groups = folo_nn::optim::StandardGroups::from_module(&v2).into_groups(
            base.lr0,
            base.momentum,
            base.weight_decay,
        );
        let opt = folo_nn::optim::AdamW::new(groups).unwrap();
        let sched = folo_nn::LrSchedule::new(folo_nn::DecayKind::Linear, 1, 4, 0.01, 0.0, 0.1, 0.8).unwrap();
        let mut shadow = FoloV2Model::new(v2_cfg).unwrap();
        shadow.set_classes(d0.classes.clone()).unwrap();
        shadow.set_imgsz(base.imgsz);
        let ema = folo_nn::ModelEma::with_params(shadow, &v2, 0.9999, 2000.0).unwrap();
        let rng_state = RngState { seed: base.seed, streams: Default::default() };
        let parts = CheckpointParts {
            model: &v2,
            ema: Some(&ema),
            optimizer: &opt,
            schedule: &sched,
            rng: &rng_state,
            epoch: 0,
            best: BestState::default(),
            train_config: &train_json,
        };
        save_checkpoint(&ckpt_seed, &meta, &parts).unwrap();
    }

    // Plain GPU distillation (`bckd_gamma == 0`, `iou_head == true`).
    let over_plain =
        ResumeOverrides { distill: Some(DistillConfig::new(&t_str)), ..ResumeOverrides::default() };
    let mut tr_plain = Trainer::resume(&ckpt_seed, data(), TrainOptions::default(), over_plain).unwrap();
    assert!(tr_plain.model_v2().unwrap().config().iou_head);
    let summary_plain = tr_plain.run().unwrap();
    assert_eq!(summary_plain.epochs_completed, 1);

    // Context-aware GPU distillation (`bckd_gamma > 0`, `gt_verify == true`, `iou_head == true`).
    let mut d_ctx = DistillConfig::new(&t_str);
    d_ctx.bckd_gamma = 0.5;
    d_ctx.gt_verify = true;
    let over_ctx = ResumeOverrides { distill: Some(d_ctx), ..ResumeOverrides::default() };
    let mut tr_ctx = Trainer::resume(&ckpt_seed, data(), TrainOptions::default(), over_ctx).unwrap();
    assert!(tr_ctx.model_v2().unwrap().config().iou_head);
    let summary_ctx = tr_ctx.run().unwrap();
    assert_eq!(summary_ctx.epochs_completed, 1);

    // Confirm weights from both runs are finite and differ (context-aware distillation changed the updates).
    let w_plain: Vec<f32> = tr_plain.detector().named_tensors()[0]
        .tensor
        .to_device(Device::Cpu)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let w_ctx: Vec<f32> =
        tr_ctx.detector().named_tensors()[0].tensor.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap();
    assert!(w_plain.iter().all(|v| v.is_finite()) && w_ctx.iter().all(|v| v.is_finite()));
    assert_ne!(w_plain, w_ctx, "bckd_gamma + gt_verify must change folo2 GPU updates");
}
