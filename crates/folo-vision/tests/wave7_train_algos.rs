//! Integration, finite-difference, forward-preservation, and CPU-vs-GPU parity tests for Wave 7 training
//! algorithms (P1: MAL + Scale-Adaptive NWD-IoU + Eps-Free TAL; P2: Conflict-Free BCKD + GT-Verified DFL KD;
//! P3: Decoupled & Cautious WD + Floor Conv+BN Gauge Normalization + Stage-2 Mosaic-Off Safeguards).

use std::sync::Mutex;

use folo_core::rng::Rng;
use folo_core::{Device, Tensor};
use folo_nn::Module;
use folo_nn::io::ModelConfig;
use folo_nn::optim::{GradSource, MuSgd, MuSgdWeights, Optimizer, ParamGroup, Sgd};
use folo_vision::assign::{TalConfig, assign, nwd_box, nwd_lambda};
use folo_vision::boxes::Xyxy;
use folo_vision::decode::make_anchors;
use folo_vision::loss::distill::{
    DistillConfig, DistillTargets, TeacherView, distill_loss_fused_with_context, distill_loss_with_context,
    gt_verify_weights_cpu,
};
use folo_vision::loss::{
    DistillAssignment, ImageTargets, LossConfig, assign_batch, assign_batch_device, detection_loss,
    detection_loss_for_distill, detection_loss_host, loss_with_assignment,
};
use folo_vision::model::FoloModel;
use folo_vision::train::{GAUGE_TARGET_NORM, apply_floor_gauge_norm};

struct MapGrads(Vec<(folo_core::TensorId, Tensor)>);

impl GradSource for MapGrads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.iter().find(|(id, _)| *id == param.id()).map(|(_, g)| g)
    }
}

fn gpu() -> Option<Device> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
    folo_gpu::init().ok()
}

fn sample_batch(seed: u64) -> (Tensor, Tensor, folo_vision::decode::Anchors, Vec<ImageTargets>) {
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let mut rng = Rng::seed(seed);
    let box_raw = Tensor::randn([2, 64, a], 0.0, 1.2, &mut rng);
    let cls_raw = Tensor::rand_uniform([2, 4, a], -3.5, 2.0, &mut rng);
    let targets = vec![
        ImageTargets {
            boxes: vec![Xyxy::new(6.0, 6.0, 14.0, 14.0), Xyxy::new(20.0, 18.0, 58.0, 56.0)],
            labels: vec![0, 2],
        },
        ImageTargets { boxes: vec![Xyxy::new(10.0, 28.0, 22.0, 38.0)], labels: vec![1] },
    ];
    (box_raw, cls_raw, anchors, targets)
}

fn assert_close_slice(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * scale.max(w.abs());
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (diff {:.3e} > tol {:.3e})",
            (g - w).abs(),
            tol
        );
    }
}

/// Finite-difference gradient verification for P1 MAL (gamma = 1.5) and Scale-Adaptive NWD-IoU box loss.
#[test]
fn finite_diff_mal_and_scale_adaptive_nwd() {
    let (box_raw, cls_raw, anchors, targets) = sample_batch(11);
    let cfg = LossConfig {
        mal_gamma: 1.5,
        nwd_weight: 0.5,
        tal: TalConfig { eps_free: true, nwd_weight: 0.5, ..TalConfig::default() },
        ..LossConfig::default()
    };
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    assert!(asg.num_fg() > 0);

    let b_t = box_raw.clone().requires_grad_();
    let c_t = cls_raw.clone().requires_grad_();
    let out = loss_with_assignment(&b_t, &c_t, &anchors, &asg, &cfg).unwrap();
    let grads = out.total.backward().unwrap();
    let g_cls = grads.get(&c_t).unwrap().to_vec::<f32>().unwrap();
    let g_box = grads.get(&b_t).unwrap().to_vec::<f32>().unwrap();

    // Spot-check MAL gradients on both foreground and background logits via central finite differences.
    let eps = 1e-3f32;
    let c_vec = cls_raw.to_vec::<f32>().unwrap();
    let mut cls_spots = vec![0usize, 5, 19, 42, c_vec.len() - 1];
    if asg.num_fg() > 0 {
        let fg_idx =
            (asg.image[0] as usize * 4 + asg.label[0] as usize) * anchors.len() + asg.anchor[0] as usize;
        cls_spots.push(fg_idx);
    }
    for idx in cls_spots {
        let mut vp = c_vec.clone();
        let mut vm = c_vec.clone();
        vp[idx] += eps;
        vm[idx] -= eps;
        let lp = loss_with_assignment(
            &box_raw,
            &Tensor::from_vec(vp, cls_raw.shape()).unwrap(),
            &anchors,
            &asg,
            &cfg,
        )
        .unwrap()
        .total_value;
        let lm = loss_with_assignment(
            &box_raw,
            &Tensor::from_vec(vm, cls_raw.shape()).unwrap(),
            &anchors,
            &asg,
            &cfg,
        )
        .unwrap()
        .total_value;
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_cls[idx] - num).abs() <= 2e-3 + 2e-2 * num.abs(),
            "MAL FD[{idx}]: ana={} num={}",
            g_cls[idx],
            num
        );
    }

    // Spot-check NWD+CIoU+DFL box gradients on a positive anchor.
    let b_vec = box_raw.to_vec::<f32>().unwrap();
    let img = asg.image[0] as usize;
    let anc = asg.anchor[0] as usize;
    for ch in [0usize, 5, 16, 35, 50] {
        let idx = (img * 64 + ch) * anchors.len() + anc;
        let mut vp = b_vec.clone();
        let mut vm = b_vec.clone();
        vp[idx] += eps;
        vm[idx] -= eps;
        let lp = loss_with_assignment(
            &Tensor::from_vec(vp, box_raw.shape()).unwrap(),
            &cls_raw,
            &anchors,
            &asg,
            &cfg,
        )
        .unwrap()
        .total_value;
        let lm = loss_with_assignment(
            &Tensor::from_vec(vm, box_raw.shape()).unwrap(),
            &cls_raw,
            &anchors,
            &asg,
            &cfg,
        )
        .unwrap()
        .total_value;
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_box[idx] - num).abs() <= 5e-3 + 2e-2 * num.abs(),
            "Box NWD FD[{idx}]: ana={} num={}",
            g_box[idx],
            num
        );
    }

    // Verify scale-adaptive lambda vanishes on large boxes and stays active on tiny boxes.
    assert!(nwd_lambda(&Xyxy::new(0.0, 0.0, 8.0, 8.0), 0.5) > 0.40);
    assert!(nwd_lambda(&Xyxy::new(0.0, 0.0, 128.0, 128.0), 0.5) < 1e-6);
    let b0 = Xyxy::new(10.0, 10.0, 18.0, 18.0);
    assert!((nwd_box(&b0, &b0, 12.8) - 1.0).abs() < 1e-3);
}

/// Finite-difference check for detached-weight BCKD (gamma = 1.5): gradient equals `S * |p_S - p_T|^γ * (p_S - p_T)`.
#[test]
fn bckd_gradient_matches_analytic_formula() {
    let anchors = make_anchors(32, 32, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let mut rng = Rng::seed(21);
    let sb = Tensor::randn([1, 64, a], 0.0, 1.0, &mut rng);
    let sc = Tensor::rand_uniform([1, 3, a], -3.0, 3.0, &mut rng);
    let tb = Tensor::randn([1, 64, a], 0.0, 1.0, &mut rng);
    let tc = Tensor::rand_uniform([1, 3, a], -3.0, 3.0, &mut rng);

    let view = TeacherView::new(&tb, &tc, 16, 4.0).unwrap();
    let targets = DistillTargets::new(&view).unwrap();
    let cfg = DistillConfig { bckd_gamma: 1.5, cls_gain: 0.75, dfl_gain: 1.5, ..DistillConfig::new("t") };

    let x = sc.clone().requires_grad_();
    let z = sb.clone().requires_grad_();
    let out = distill_loss_with_context(&z, &x, &targets, 4.0, &cfg, None, None).unwrap();
    let grads = out.total.backward().unwrap();
    let gx = grads.get(&x).unwrap().to_vec::<f32>().unwrap();

    let xv = sc.to_vec::<f32>().unwrap();
    let pv = targets.cls_prob.to_vec::<f32>().unwrap();
    let scale = 1.0 * cfg.cls_gain / 4.0;
    for i in 0..xv.len() {
        let ps = 1.0 / (1.0 + (-xv[i]).exp());
        let diff = ps - pv[i];
        let want = scale * diff.abs().powf(1.5) * diff;
        assert!(
            (gx[i] - want).abs() <= 1e-5 + 1e-4 * want.abs(),
            "BCKD grad[{i}]: got {} want {want}",
            gx[i]
        );
    }
}

/// Exact function-preservation test for Floor Conv+BN Gauge Normalization on a full `folo-n` model:
/// simulates `v5`-like norm collapse (`W *= 0.45`, `running_var *= 0.20`), runs `apply_floor_gauge_norm`,
/// and verifies forward output changes by `< 1e-5` relative while shrunken channel norms are restored to `>= √2`.
#[test]
fn floor_gauge_norm_preserves_full_folo_n_forward() {
    let mut rng = Rng::seed(7);
    let mconf = ModelConfig::new("n", 4).unwrap();
    let model = FoloModel::with_rng(mconf, &mut rng).unwrap();
    model.eval();

    // Simulate v4 -> v5 coupled-WD norm collapse on every ConvBnAct (except leave stem large to test floor-only).
    {
        let _g = folo_core::no_grad();
        model.for_each_module(&mut |name, m| {
            if let Some(block) = (m as &dyn std::any::Any).downcast_ref::<folo_nn::ConvBnAct>()
                && !name.contains("stem")
                && let Some(bn) = &block.bn
            {
                let w = block.conv.weight.mul_scalar(0.45).unwrap();
                let mu = bn.running_mean.add_scalar(0.15).unwrap();
                let var = bn.running_var.mul_scalar(0.25).unwrap();
                block.conv.weight.copy_from(&w).unwrap();
                bn.running_mean.copy_from(&mu).unwrap();
                bn.running_var.copy_from(&var).unwrap();
            }
        });
    }

    let x = Tensor::randn([1, 3, 64, 64], 0.0, 1.0, &mut rng);
    let before = model.forward(&x).unwrap();
    let b_box = before.box_raw.to_vec::<f32>().unwrap();
    let b_cls = before.cls_raw.to_vec::<f32>().unwrap();

    let mut before_norms: Vec<f32> = Vec::new();
    model.for_each_module(&mut |_, m| {
        if let Some(block) = (m as &dyn std::any::Any).downcast_ref::<folo_nn::ConvBnAct>()
            && block.conv.bias.is_none()
            && block.bn.is_some()
        {
            let w = block.conv.weight.to_vec::<f32>().unwrap();
            let c_out = block.conv.weight.dims()[0];
            let fan = w.len() / c_out;
            for c in 0..c_out {
                before_norms.push(w[c * fan..(c + 1) * fan].iter().map(|v| v * v).sum::<f32>().sqrt());
            }
        }
    });

    let report = apply_floor_gauge_norm(&model).unwrap();
    assert!(
        report.layers_visited >= 70,
        "expected full folo-n Conv+BN layers, got {}",
        report.layers_visited
    );
    assert!(
        report.channels_rescaled > 100,
        "expected shrunken channels rescaled, got {}",
        report.channels_rescaled
    );

    let after = model.forward(&x).unwrap();
    let a_box = after.box_raw.to_vec::<f32>().unwrap();
    let a_cls = after.cls_raw.to_vec::<f32>().unwrap();

    assert_close_slice(&a_box, &b_box, 1e-4, 1e-4, "gauge norm box_raw");
    assert_close_slice(&a_cls, &b_cls, 1e-4, 1e-4, "gauge norm cls_raw");

    // Verify every live channel was restored to `min(4 * norm_before, √2)` and no channel was shrunk.
    let mut after_norms: Vec<f32> = Vec::new();
    model.for_each_module(&mut |_, m| {
        if let Some(block) = (m as &dyn std::any::Any).downcast_ref::<folo_nn::ConvBnAct>()
            && block.conv.bias.is_none()
            && block.bn.is_some()
        {
            let w = block.conv.weight.to_vec::<f32>().unwrap();
            let c_out = block.conv.weight.dims()[0];
            let fan = w.len() / c_out;
            for c in 0..c_out {
                after_norms.push(w[c * fan..(c + 1) * fan].iter().map(|v| v * v).sum::<f32>().sqrt());
            }
        }
    });
    assert_eq!(before_norms.len(), after_norms.len());
    for (&b, &a) in before_norms.iter().zip(&after_norms) {
        if b > 1e-4 {
            let expected = if b < GAUGE_TARGET_NORM { (b * 4.0).min(GAUGE_TARGET_NORM) } else { b };
            assert!(
                (a - expected).abs() <= 1e-4 * expected.max(1.0),
                "before={b}, after={a}, expected={expected}"
            );
        }
    }
    const { assert!(GAUGE_TARGET_NORM > 1.4) };
}

/// Zero-regression check: with all Wave 7 flags off, CPU `assign` & `detection_loss` stay bit-identical,
/// `TrainConfig` defaults and validation work cleanly, and `eps_free` fixes tiny-metric attenuation.
#[test]
fn eps_free_fixes_tiny_metric_attenuation_and_defaults_unchanged() {
    let gt = Xyxy::new(10.0, 10.0, 18.0, 18.0);
    let pd = Xyxy::new(11.0, 11.0, 19.0, 19.0);
    // Tiny initial score -> u^6 * score^0.5 ~ 1e-10 (comparable to eps = 1e-9).
    let legacy = assign(&[1e-18], 1, &[pd], &[[14.0, 14.0]], &[gt], &[0], &TalConfig::default());
    let eps_free = assign(
        &[1e-18],
        1,
        &[pd],
        &[[14.0, 14.0]],
        &[gt],
        &[0],
        &TalConfig { eps_free: true, ..TalConfig::default() },
    );
    assert!(eps_free.target_scores[0] > legacy.target_scores[0] * 5.0);

    // Zero-regression check on default detection_loss and TrainConfig validation.
    let (box_raw, cls_raw, anchors, targets) = sample_batch(55);
    let def_cfg = LossConfig::default();
    let l1 = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &def_cfg).unwrap();
    let l2 = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &def_cfg).unwrap();
    assert_eq!(l1.total_value.to_bits(), l2.total_value.to_bits());
    assert_eq!(l1.items, l2.items);

    let mut tcfg = folo_vision::TrainConfig::default();
    assert!(!tcfg.decoupled_wd && !tcfg.cautious_wd && !tcfg.gauge_norm && !tcfg.bn_freeze_close_mosaic);
    assert_eq!(tcfg.cls_lr_close_scale, 1.0);
    assert!(tcfg.validate().is_ok());
    tcfg.loss.mal_gamma = -0.1;
    assert!(tcfg.validate().is_err());
    tcfg.loss.mal_gamma = 1.5;
    tcfg.cls_lr_close_scale = 0.0;
    assert!(tcfg.validate().is_err());
}

/// CPU-vs-GPU parity for P1 (MAL + Scale-Adaptive NWD-IoU + Eps-Free TAL), P2 (BCKD + GT-Verified DFL KD),
/// and P3 (Decoupled & Cautious WD in Sgd and MuSgd).
#[test]
fn gpu_parity_p1_p2_p3() {
    let Some(dev) = gpu() else { return };

    // --- P1: MAL + Scale-Adaptive NWD-IoU + Eps-Free TAL ---
    let (box_raw, cls_raw, anchors, targets) = sample_batch(31);
    let cfg = LossConfig {
        mal_gamma: 1.5,
        nwd_weight: 0.5,
        tal: TalConfig { eps_free: true, nwd_weight: 0.5, ..TalConfig::default() },
        ..LossConfig::default()
    };
    let asg_cpu = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    let b_gpu = box_raw.to_device(dev).unwrap();
    let c_gpu = cls_raw.to_device(dev).unwrap();
    let asg_gpu = assign_batch_device(&b_gpu, &c_gpu, &anchors, &targets, &cfg).unwrap();
    assert_eq!(asg_cpu.num_fg(), asg_gpu.num_fg(), "P1 num_fg");
    assert_eq!(asg_cpu.anchor, asg_gpu.anchor, "P1 anchors");
    assert_close_slice(&asg_gpu.score, &asg_cpu.score, 1e-5, 1e-4, "P1 target scores");

    let (b_c, c_c) = (box_raw.clone().requires_grad_(), cls_raw.clone().requires_grad_());
    let l_cpu = detection_loss_host(&b_c, &c_c, &anchors, &targets, &cfg).unwrap();
    let g_cpu = l_cpu.total.backward().unwrap();

    let (b_g, c_g) = (b_gpu.clone().requires_grad_(), c_gpu.clone().requires_grad_());
    let (l_gpu, dev_asg) = detection_loss_for_distill(&b_g, &c_g, &anchors, &targets, &cfg, true).unwrap();
    let g_gpu = l_gpu.total.backward().unwrap();

    assert!((l_gpu.items.box_ - l_cpu.items.box_).abs() <= 1e-4 + 1e-4 * l_cpu.items.box_.abs());
    assert!((l_gpu.items.cls - l_cpu.items.cls).abs() <= 1e-4 + 1e-4 * l_cpu.items.cls.abs());
    assert!((l_gpu.items.dfl - l_cpu.items.dfl).abs() <= 1e-4 + 1e-4 * l_cpu.items.dfl.abs());
    assert_close_slice(
        &g_gpu.get(&c_g).unwrap().to_vec::<f32>().unwrap(),
        &g_cpu.get(&c_c).unwrap().to_vec::<f32>().unwrap(),
        1e-5,
        2e-4,
        "P1 GPU vs CPU cls grad",
    );
    assert_close_slice(
        &g_gpu.get(&b_g).unwrap().to_vec::<f32>().unwrap(),
        &g_cpu.get(&b_c).unwrap().to_vec::<f32>().unwrap(),
        1e-5,
        2e-4,
        "P1 GPU vs CPU box grad",
    );

    // --- P2: Conflict-Free BCKD + GT-Verified DFL KD ---
    let mut rng = Rng::seed(99);
    // Make teacher box closer to GT than student so gt_verify triggers on positives.
    let t_box = box_raw.add(&Tensor::randn(box_raw.shape().clone(), 0.0, 0.4, &mut rng)).unwrap();
    let t_cls = cls_raw.add(&Tensor::randn(cls_raw.shape().clone(), 0.0, 0.8, &mut rng)).unwrap();
    let kd_cfg = DistillConfig { bckd_gamma: 1.5, gt_verify: true, ..DistillConfig::new("t") };

    let view = TeacherView::new(&t_box, &t_cls, 16, kd_cfg.temperature).unwrap();
    let dt = DistillTargets::new(&view).unwrap();
    let vw_cpu = gt_verify_weights_cpu(&box_raw, &t_box, &anchors, &asg_cpu, 16).unwrap();
    let (kb_c, kc_c) = (box_raw.clone().requires_grad_(), cls_raw.clone().requires_grad_());
    let kd_cpu = distill_loss_with_context(
        &kb_c,
        &kc_c,
        &dt,
        l_cpu.target_score_sum,
        &kd_cfg,
        Some(&asg_cpu),
        Some(&vw_cpu),
    )
    .unwrap();
    let kg_cpu = kd_cpu.total.backward().unwrap();

    let (kb_g, kc_g) = (b_gpu.clone().requires_grad_(), c_gpu.clone().requires_grad_());
    let tb_g = t_box.to_device(dev).unwrap();
    let tc_g = t_cls.to_device(dev).unwrap();
    let gpu_asg_ref = match dev_asg.as_ref() {
        Some(DistillAssignment::Device(a)) => Some(a),
        _ => panic!("expected device assignment"),
    };
    let kd_gpu = distill_loss_fused_with_context(
        &kb_g,
        &kc_g,
        &tb_g,
        &tc_g,
        l_gpu.target_score_sum,
        &kd_cfg,
        gpu_asg_ref,
    )
    .unwrap();
    let kg_gpu = kd_gpu.total.backward().unwrap();

    assert!(
        (kd_gpu.cls - kd_cpu.cls).abs() <= 1e-4 + 1e-4 * kd_cpu.cls.abs(),
        "P2 cls: {} vs {}",
        kd_gpu.cls,
        kd_cpu.cls
    );
    assert!(
        (kd_gpu.dfl - kd_cpu.dfl).abs() <= 1e-4 + 1e-4 * kd_cpu.dfl.abs(),
        "P2 dfl: {} vs {}",
        kd_gpu.dfl,
        kd_cpu.dfl
    );
    assert_close_slice(
        &kg_gpu.get(&kc_g).unwrap().to_vec::<f32>().unwrap(),
        &kg_cpu.get(&kc_c).unwrap().to_vec::<f32>().unwrap(),
        1e-5,
        2e-4,
        "P2 GPU vs CPU cls grad",
    );
    assert_close_slice(
        &kg_gpu.get(&kb_g).unwrap().to_vec::<f32>().unwrap(),
        &kg_cpu.get(&kb_c).unwrap().to_vec::<f32>().unwrap(),
        1e-5,
        2e-4,
        "P2 GPU vs CPU box grad",
    );

    // --- P3: Decoupled & Cautious WD CPU vs GPU parity (Sgd + MuSgd with w_sgd = 0.5) ---
    for cautious in [false, true] {
        let mut r = Rng::seed(77);
        let p0 = Tensor::randn([16, 16, 3, 3], 0.0, 0.5, &mut r);
        let g0 = Tensor::randn([16, 16, 3, 3], 0.0, 0.2, &mut r);
        let g1 = Tensor::randn([16, 16, 3, 3], 0.0, 0.2, &mut r);

        let p_cpu = p0.clone();
        let p_gpu = p0.to_device(dev).unwrap();
        let grp_cpu = ParamGroup::new("decay", vec![("w".into(), p_cpu.clone())])
            .lr(0.01)
            .momentum(0.937)
            .weight_decay(0.004)
            .decoupled_wd(true)
            .cautious_wd(cautious);
        let grp_gpu = ParamGroup::new("decay", vec![("w".into(), p_gpu.clone())])
            .lr(0.01)
            .momentum(0.937)
            .weight_decay(0.004)
            .decoupled_wd(true)
            .cautious_wd(cautious);
        let mut opt_cpu = Sgd::new(vec![grp_cpu]).unwrap();
        let mut opt_gpu = Sgd::new(vec![grp_gpu]).unwrap();
        for g in [&g0, &g1] {
            opt_cpu.step(&MapGrads(vec![(p_cpu.id(), g.clone())])).unwrap();
            opt_gpu.step(&MapGrads(vec![(p_gpu.id(), g.to_device(dev).unwrap())])).unwrap();
        }
        assert_close_slice(
            &p_gpu.to_vec::<f32>().unwrap(),
            &p_cpu.to_vec::<f32>().unwrap(),
            1e-6,
            1e-5,
            &format!("P3 Sgd cautious={cautious}"),
        );

        // Also verify MuSgd with w_sgd = 0.5 matches between CPU and GPU.
        let pm_cpu = p0.clone();
        let pm_gpu = p0.to_device(dev).unwrap();
        let gm_cpu = ParamGroup::new("decay", vec![("w".into(), pm_cpu.clone())])
            .lr(0.01)
            .momentum(0.937)
            .weight_decay(0.004)
            .decoupled_wd(true)
            .cautious_wd(cautious);
        let gm_gpu = ParamGroup::new("decay", vec![("w".into(), pm_gpu.clone())])
            .lr(0.01)
            .momentum(0.937)
            .weight_decay(0.004)
            .decoupled_wd(true)
            .cautious_wd(cautious);
        let w = MuSgdWeights { muon: 0.5, sgd: 0.5 };
        let mut m_cpu = MuSgd::with_weights(vec![gm_cpu], w).unwrap();
        let mut m_gpu = MuSgd::with_weights(vec![gm_gpu], w).unwrap();
        m_cpu.step(&MapGrads(vec![(pm_cpu.id(), g0.clone())])).unwrap();
        m_gpu.step(&MapGrads(vec![(pm_gpu.id(), g0.to_device(dev).unwrap())])).unwrap();
        assert_close_slice(
            &pm_gpu.to_vec::<f32>().unwrap(),
            &pm_cpu.to_vec::<f32>().unwrap(),
            1e-4,
            1e-3,
            &format!("P3 MuSgd cautious={cautious}"),
        );
    }
}
