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

/// Regression tests for Round-1 review findings:
/// - fractional `0 < mal_gamma < 1` at `x = -120` on the composite (`!fused`) path stays 100% finite,
/// - `wd_eff` is reported consistently in `run_config()` and `resolved_train()`,
/// - unsupported P3 optimizer combinations (`AdamW + cautious_wd`, non-`MuSGD + cls_lr_close_scale`) error,
/// - cross-class `--weights` transfer with `mal_gamma > 0` initializes fresh class biases to `-4.59512`,
/// - Wave-7 trainer config round-trips cleanly through checkpoint save + `Trainer::resume`.
#[test]
fn composite_fractional_mal_and_p3_validation_and_resume_regressions() {
    use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
    use folo_data::{AugmentConfig, Split};
    use folo_test_support::{TestDir, TestFile};
    use folo_vision::train::{OptimizerChoice, ResumeOverrides};
    use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};
    use std::sync::Arc;

    // 1. Composite `!fused` MAL with fractional gamma = 0.5 at x = -120 on an empty-target batch (CPU + GPU).
    let anchors = make_anchors(32, 32, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let empty = vec![ImageTargets { boxes: vec![], labels: vec![] }];
    let cfg_frac = LossConfig { mal_gamma: 0.5, ..LossConfig::default() };
    let mut devices = vec![Device::Cpu];
    if let Some(dev) = gpu() {
        devices.push(dev);
    }
    for dev in devices {
        let bx = Tensor::zeros([1, 64, a]).to_device(dev).unwrap().requires_grad_();
        let cl = Tensor::full([1, 3, a], -120.0).to_device(dev).unwrap().requires_grad_();
        let out = detection_loss_host(&bx, &cl, &anchors, &empty, &cfg_frac).unwrap();
        assert!(out.total_value.is_finite(), "{dev}: fractional MAL loss non-finite");
        let grads = out.total.backward().unwrap();
        let g = grads.get(&cl).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "{dev}: fractional MAL produced NaN/Inf grad at x=-120");
    }

    let make_data = |classes: Vec<String>| {
        let syn = SyntheticConfig {
            images: 6,
            classes: classes.clone(),
            imgsz: 64,
            seed: 9,
            min_objects: 1,
            max_objects: 2,
            empty_fraction: 0.0,
            small_fraction: 0.0,
            vary_aspect: false,
            val_fraction: 2.0 / 6.0,
            ..SyntheticConfig::default()
        };
        TrainData {
            train: Arc::new(SyntheticDataset::new(syn.clone(), Split::Train).unwrap()),
            val: Arc::new(SyntheticDataset::new(syn, Split::Val).unwrap()),
            classes,
        }
    };

    // 2. P3 optimizer compatibility validation & consistent `wd_eff` reporting.
    let bad_adamw = TrainConfig {
        epochs: 1,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        optimizer: OptimizerChoice::Adamw,
        cautious_wd: true,
        device: "cpu".into(),
        workers: 0,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    };
    assert!(
        Trainer::new(bad_adamw, make_data(vec!["circle".into(), "square".into()]), TrainOptions::default())
            .is_err()
    );

    let bad_sgd_cls = TrainConfig {
        epochs: 1,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        optimizer: OptimizerChoice::Sgd,
        cls_lr_close_scale: 0.5,
        device: "cpu".into(),
        workers: 0,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    };
    assert!(
        Trainer::new(bad_sgd_cls, make_data(vec!["circle".into(), "square".into()]), TrainOptions::default())
            .is_err()
    );

    // 3. Cross-class transfer (`nc = 2 -> 3`) with `mal_gamma > 0` keeps fresh class biases at `-4.59512`,
    //    `wd_eff` matches both `optimizer.weight_decay` and `resolved.weight_decay`, and checkpoint resumes cleanly.
    let src_weights = TestFile::new("wave7-xfer", "src.folo");
    let mut m_src = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
    m_src.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m_src.save(&src_weights).unwrap();

    let run_dir = TestDir::new("wave7-resume");
    let w7_cfg = TrainConfig {
        epochs: 2,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        optimizer: OptimizerChoice::Musgd,
        lr0: 0.005,
        warmup_epochs: 0.0,
        workers: 0,
        device: "cpu".into(),
        augment: AugmentConfig::none(),
        loss: LossConfig {
            mal_gamma: 1.5,
            nwd_weight: 0.5,
            tal: TalConfig { eps_free: true, nwd_weight: 0.5, ..TalConfig::default() },
            ..LossConfig::default()
        },
        decoupled_wd: true,
        cautious_wd: true,
        wd_eff: Some(0.004),
        gauge_norm: true,
        ..TrainConfig::default()
    };
    let trainer = Trainer::new(
        w7_cfg.clone(),
        make_data(vec!["circle".into(), "square".into(), "triangle".into()]),
        TrainOptions { weights: Some(src_weights.path().to_path_buf()), ..TrainOptions::default() },
    )
    .unwrap();
    let rc = trainer.run_config();
    assert_eq!(rc["optimizer"]["weight_decay"].as_f64(), Some(0.004));
    assert_eq!(rc["resolved"]["weight_decay"].as_f64(), Some(0.004));
    let cls_bias = trainer
        .model()
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "head.cls.p3.out.bias")
        .unwrap()
        .1
        .to_vec::<f32>()
        .unwrap();
    assert!(
        (cls_bias[0] - (-4.595_12)).abs() < 1e-4,
        "cross-class transfer with mal_gamma > 0 must initialize fresh cls bias to -4.59512, got {}",
        cls_bias[0]
    );
    let ckpt = run_dir.join("last.folo");
    trainer.save_checkpoint(&ckpt).unwrap();
    let resumed = Trainer::resume(
        &ckpt,
        make_data(vec!["circle".into(), "square".into(), "triangle".into()]),
        TrainOptions::default(),
        ResumeOverrides::default(),
    )
    .unwrap();
    let rrc = resumed.run_config();
    assert_eq!(rrc["resolved"]["weight_decay"].as_f64(), Some(0.004));
    assert_eq!(rrc["train"]["decoupled_wd"].as_bool(), Some(true));
    assert_eq!(rrc["train"]["cautious_wd"].as_bool(), Some(true));
}

/// GPU distillation safety & fallback regressions:
/// - `gt_verify = true` still flags non-finite teacher `box_raw` at zero-weight anchors (CPU == GPU),
/// - `folo_gpu::distill::distill_loss_with_context` validates `gt_asg` dimensions and rejects `gt_verify` w/o `asg`,
/// - in-place mutation of returned `Assignment.dense` invalidates backward via autograd version tracking,
/// - GPU trainer with `loss.tal.topk = 65` (host TAL fallback) + `gt_verify` + `bckd_gamma` runs cleanly on both
///   empty-target and matched-positive batches.
#[test]
fn gpu_distill_safety_and_topk65_fallback_regressions() {
    use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
    use folo_data::{AugmentConfig, Split};
    use folo_test_support::TestFile;
    use folo_vision::train::OptimizerChoice;
    use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};
    use std::sync::Arc;

    let (box_raw, cls_raw, anchors, targets) = sample_batch(99);
    let a = anchors.len();
    let cfg = LossConfig::default();

    // --- CPU regressions (always execute, even on CPU-only CI runners without a GPU) ---
    // 1a. Corrupt teacher box_raw with +Inf on a zero-object batch (all gt_verify weights == 0): CPU reports NaN.
    let empty_targets = vec![ImageTargets { boxes: vec![], labels: vec![] }; 2];
    let mut tb_vec = box_raw.to_vec::<f32>().unwrap();
    tb_vec[0] = f32::INFINITY;
    let tb_bad = Tensor::from_vec(tb_vec, box_raw.shape()).unwrap();
    let tc = cls_raw.clone();
    let dcfg = DistillConfig { bckd_gamma: 1.5, gt_verify: true, ..DistillConfig::new("t") };

    let (sup_c, asg_c) =
        detection_loss_for_distill(&box_raw, &cls_raw, &anchors, &empty_targets, &cfg, true).unwrap();
    let Some(DistillAssignment::Host(ha)) = asg_c else { panic!("expected host assignment") };
    let view_c = TeacherView::new(&tb_bad, &tc, 16, dcfg.temperature).unwrap();
    let tgt_c = DistillTargets::new(&view_c).unwrap();
    let vw_c = gt_verify_weights_cpu(&box_raw, &tb_bad, &anchors, &ha, 16).unwrap();
    let kd_c = distill_loss_with_context(
        &box_raw,
        &cls_raw,
        &tgt_c,
        sup_c.target_score_sum,
        &dcfg,
        Some(&ha),
        Some(&vw_c),
    )
    .unwrap();
    assert!(kd_c.dfl.is_nan(), "CPU gt_verify must propagate NaN from +Inf teacher box logit");

    // 1b. Public host `distill_loss_with_context` rejects `gt_verify=true` with `anchor_ver_w=None`,
    //     and accepts an explicit all-zero weight tensor for an empty assignment (`dfl == 0.0`).
    let view_ok = TeacherView::new(&box_raw, &tc, 16, dcfg.temperature).unwrap();
    let tgt_ok = DistillTargets::new(&view_ok).unwrap();
    assert!(
        distill_loss_with_context(
            &box_raw,
            &cls_raw,
            &tgt_ok,
            sup_c.target_score_sum,
            &dcfg,
            Some(&ha),
            None
        )
        .is_err(),
        "host distill_loss_with_context must reject gt_verify=true with anchor_ver_w=None"
    );
    let zeros_vw = Tensor::zeros([2, 1, 1, a]);
    let kd_zero = distill_loss_with_context(
        &box_raw,
        &cls_raw,
        &tgt_ok,
        sup_c.target_score_sum,
        &dcfg,
        Some(&ha),
        Some(&zeros_vw),
    )
    .unwrap();
    assert_eq!(kd_zero.dfl, 0.0, "all-zero anchor_ver_w must produce zero DFL KD");

    // 1c. Epsilon-free TAL `(m / m_max) * u_max` preserves `u_max` even when `m * u_max` would flush to zero.
    {
        let u_box = Xyxy::new(10.0, 10.0, 20.0, 20.0);
        // Concentric 1×1 square inside 10×10 square: identical center & aspect ratio so `ciou == iou == 0.01`,
        // `m = 1e-32 * 0.01^6 = 1e-44 > 0`, while `m * 0.01 = 1e-46` flushes to 0.0 in f32.
        let pred_b = [Xyxy::new(14.5, 14.5, 15.5, 15.5)];
        let tiny_scores = [1e-32f32];
        let tal_cfg = TalConfig { eps_free: true, alpha: 1.0, beta: 6.0, ..TalConfig::default() };
        let asg_ef =
            folo_vision::assign::assign(&tiny_scores, 1, &pred_b, &[[15.0, 15.0]], &[u_box], &[0], &tal_cfg);
        let iou = folo_vision::boxes::box_iou(&pred_b[0], &u_box);
        let m_raw = tiny_scores[0].powf(1.0) * iou.powf(6.0);
        assert!(m_raw > 0.0 && (m_raw * iou) == 0.0, "test setup must underflow m * u_max");
        assert!(asg_ef.fg_mask[0]);
        assert!(
            (asg_ef.target_scores[0] - iou).abs() <= 1e-5 * iou,
            "eps_free TAL top anchor should preserve u_max ({iou:e}), got {:e}",
            asg_ef.target_scores[0]
        );
    }

    // 1d. Recoverable non-finite student batch with `distill.gt_verify = true` must be skipped (BN restored),
    //     NOT hard-abort training with `Error::Config` (runs on CPU always, and on GPU below when available).
    let teacher_file = TestFile::new("wave7-topk65", "teacher.folo");
    let mut tm = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
    tm.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    tm.save(&teacher_file).unwrap();
    let two_img_ds = |split: Split| {
        Arc::new(folo_data::InMemoryDataset {
            samples: vec![
                folo_data::Sample {
                    image: folo_data::Image::new(64, 64, [255, 255, 255]),
                    boxes: vec![folo_data::BBox::new(0, 0.5, 0.5, 0.4, 0.4)],
                },
                folo_data::Sample {
                    image: folo_data::Image::new(64, 64, [0, 0, 0]),
                    boxes: vec![folo_data::BBox::new(0, 0.5, 0.5, 0.4, 0.4)],
                },
            ],
            classes: vec!["circle".into(), "square".into()],
            split,
        })
    };
    let bad_init = TestFile::new("wave7-nonfinite-gtverify", "init.folo");
    {
        let mut init_m = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
        init_m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
        let _g = folo_core::no_grad();
        let w = &init_m.backbone.stem.conv.weight;
        let big = Tensor::full(w.dims(), 1e38f32);
        w.copy_from(&big).unwrap();
        init_m.save(&bad_init).unwrap();
    }
    let run_nonfinite_recovery = |dev_str: &str| {
        let nf_data = TrainData {
            train: two_img_ds(Split::Train),
            val: two_img_ds(Split::Val),
            classes: vec!["circle".into(), "square".into()],
        };
        let nf_cfg = TrainConfig {
            epochs: 1,
            batch: 1,
            nbs: 1,
            imgsz: 64,
            optimizer: OptimizerChoice::Sgd,
            lr0: 0.001,
            warmup_epochs: 0.0,
            workers: 0,
            device: dev_str.into(),
            augment: AugmentConfig::none(),
            distill: Some(DistillConfig {
                bckd_gamma: 1.5,
                gt_verify: true,
                ..DistillConfig::new(teacher_file.path().to_string_lossy())
            }),
            ..TrainConfig::default()
        };
        let opts = TrainOptions { weights: Some(bad_init.path().to_path_buf()), ..TrainOptions::default() };
        let mut nf_tr = Trainer::new(nf_cfg, nf_data, opts).unwrap();
        let nf_summary = nf_tr.run().unwrap();
        assert_eq!(nf_summary.epochs_completed, 1, "dev={dev_str}");
        assert!(
            nf_summary.epochs[0].train_loss.total.is_finite(),
            "trainer ({dev_str}) with gt_verify=true must skip the non-finite batch and complete the valid batch"
        );
    };
    run_nonfinite_recovery("cpu");

    // --- GPU regressions ---
    let Some(dev) = gpu() else { return };
    let sb_g = box_raw.to_device(dev).unwrap().requires_grad_();
    let sc_g = cls_raw.to_device(dev).unwrap().requires_grad_();
    let tb_g = tb_bad.to_device(dev).unwrap();
    let tc_g = tc.to_device(dev).unwrap();
    let (sup_g, asg_g) =
        detection_loss_for_distill(&sb_g, &sc_g, &anchors, &empty_targets, &cfg, true).unwrap();
    let Some(DistillAssignment::Device(da)) = asg_g else { panic!("expected device assignment") };
    let kd_g =
        distill_loss_fused_with_context(&sb_g, &sc_g, &tb_g, &tc_g, sup_g.target_score_sum, &dcfg, Some(&da))
            .unwrap();
    assert!(kd_g.dfl.is_nan(), "GPU gt_verify must not swallow +Inf teacher box logit on zero-weight anchor");

    // 2. Reject gt_verify=true without Assignment, mismatched Assignment dimensions, and oversized desc.g.
    assert!(
        distill_loss_fused_with_context(
            &sb_g,
            &sc_g,
            &box_raw.to_device(dev).unwrap(),
            &tc_g,
            1.0,
            &dcfg,
            None
        )
        .is_err(),
        "gt_verify without assignment must error"
    );
    let small_anchors = make_anchors(32, 32, &[8, 16, 32], 0.5);
    let mut rng = Rng::seed(101);
    let sb_s = Tensor::randn([2, 64, small_anchors.len()], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
    let sc_s = Tensor::randn([2, 4, small_anchors.len()], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
    assert!(
        distill_loss_fused_with_context(&sb_s, &sc_s, &sb_s, &sc_s, 1.0, &dcfg, Some(&da)).is_err(),
        "mismatched assignment anchor count must error"
    );
    let mut bad_da = da;
    bad_da.desc.g = usize::MAX;
    assert!(
        distill_loss_fused_with_context(
            &sb_g,
            &sc_g,
            &box_raw.to_device(dev).unwrap(),
            &tc_g,
            1.0,
            &dcfg,
            Some(&bad_da)
        )
        .is_err(),
        "oversized Assignment::desc.g must return Err, not panic"
    );

    // 3. In-place mutation of returned `Assignment.dense` invalidates both detection and BCKD backward.
    let sb_m = box_raw.to_device(dev).unwrap().requires_grad_();
    let sc_m = cls_raw.to_device(dev).unwrap().requires_grad_();
    let (sup_m, asg_m) = detection_loss_for_distill(&sb_m, &sc_m, &anchors, &targets, &cfg, true).unwrap();
    let Some(DistillAssignment::Device(da_m)) = asg_m else { panic!("expected device assignment") };
    let kd_m = distill_loss_fused_with_context(
        &sb_m,
        &sc_m,
        &box_raw.to_device(dev).unwrap(),
        &tc_g,
        sup_m.target_score_sum,
        &dcfg,
        Some(&da_m),
    )
    .unwrap();
    let zeros_dense = da_m.dense.zeros_like().unwrap();
    da_m.dense.copy_from(&zeros_dense).unwrap();
    assert!(sup_m.total.backward().is_err(), "mutating Assignment.dense must invalidate detection backward");
    assert!(kd_m.total.backward().is_err(), "mutating Assignment.dense must invalidate BCKD backward");

    // 4. GPU Trainer run with `loss.tal.topk = 65` (host TAL fallback) + `gt_verify` + `bckd_gamma`,
    //    exercising both an empty-object image and a matched-positive image, plus GPU bad-batch recovery.
    let syn = SyntheticConfig {
        images: 4,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 17,
        min_objects: 1,
        max_objects: 2,
        empty_fraction: 0.5,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 0.25,
        ..SyntheticConfig::default()
    };
    let data = TrainData {
        train: Arc::new(SyntheticDataset::new(syn.clone(), Split::Train).unwrap()),
        val: Arc::new(SyntheticDataset::new(syn, Split::Val).unwrap()),
        classes: vec!["circle".into(), "square".into()],
    };
    let tcfg = TrainConfig {
        epochs: 1,
        batch: 1,
        nbs: 1,
        imgsz: 64,
        optimizer: OptimizerChoice::Sgd,
        lr0: 0.001,
        warmup_epochs: 0.0,
        workers: 0,
        device: "gpu:0".into(),
        augment: AugmentConfig::none(),
        loss: LossConfig {
            mal_gamma: 1.5,
            tal: TalConfig { topk: 65, ..TalConfig::default() },
            ..LossConfig::default()
        },
        distill: Some(DistillConfig {
            bckd_gamma: 1.5,
            gt_verify: true,
            ..DistillConfig::new(teacher_file.path().to_string_lossy())
        }),
        ..TrainConfig::default()
    };
    let mut tr = Trainer::new(tcfg, data, TrainOptions::default()).unwrap();
    let summary = tr.run().unwrap();
    assert_eq!(summary.epochs_completed, 1);
    assert!(summary.epochs[0].train_loss.total.is_finite());
    run_nonfinite_recovery("gpu:0");
}
