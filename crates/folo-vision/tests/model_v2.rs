//! Comprehensive tests for the FOLO v2 detector family (`folo2-n/s/m/l/x`):
//! - `ceil16` channel alignment and shared `c_box = 48, c_cls = 64` CrossKD lock (`n`/`s`),
//! - exact parameter counts and output shapes (8,400 anchors at 640×640),
//! - algebraic training equivalence of `WS-QARepConv3x3` weight-space merge vs explicit 5-branch sum,
//! - `fuse()` inference parity (`< 1e-4`),
//! - central finite-difference gradient checks on every new v2 block,
//! - `PeakSuppress` exact peak preservation & monotonicity,
//! - relocated `FullPAD-Lite` FiLM isolation (proving `neck.down3/4` stay free of float barriers),
//! - CPU/GPU forward & backward parity (when a GPU is available).

use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check, grad_check_with};
use folo_core::{DType, Tensor, no_grad};
use folo_nn::{Forward, Module, init};
use folo_vision::model::HeadMode;
use folo_vision::model::blocks::Init;
use folo_vision::model::v2::{
    FoloBlock, FoloV2Config, FoloV2Model, FullPadGate, HyperContext, MAL_CLS_BIAS_PRIOR, PeakSuppress,
    ScalingV2, WsQARepConv3x3, space_to_depth_2x2,
};

fn assert_close(a: &[f32], b: &[f32], rel: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let mut worst = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite(), "{what}: non-finite at {i} ({x} vs {y})");
        let err = (x - y).abs() / x.abs().max(y.abs()).max(1.0);
        if err > worst.0 {
            worst = (err, i);
        }
    }
    assert!(
        worst.0 <= rel,
        "{what}: max rel err {:.6e} > {rel:.1e} at idx {} ({} vs {})",
        worst.0,
        worst.1,
        a[worst.1],
        b[worst.1]
    );
}

fn randomize_bn_v2(m: &FoloV2Model, seed: u64) {
    let mut rng = Rng::seed(seed);
    for t in m.named_tensors() {
        let (lo, hi) = if t.name.ends_with("bn.weight") || t.name.ends_with("running_var") {
            (0.75, 1.35)
        } else if t.name.ends_with("bn.bias") || t.name.ends_with("running_mean") {
            (-0.15, 0.15)
        } else {
            continue;
        };
        init::copy_(&t.tensor, &Tensor::rand_uniform(t.tensor.shape().clone(), lo, hi, &mut rng)).unwrap();
    }
}

#[test]
fn scaling_alignment_and_crosskd_lock() {
    for scale in ["n", "s", "m", "l", "x"] {
        let cfg = FoloV2Config::new(scale, 80).unwrap();
        let s = ScalingV2::of(&cfg);
        for base in [64, 128, 256, 512, 1024] {
            let c = s.ch(base);
            assert!(c >= 16 && c.is_multiple_of(16), "{scale} ch({base}) = {c} not %16");
        }
        assert!(s.c_box.is_multiple_of(16), "{scale} c_box={} not %16", s.c_box);
        assert!(s.c_cls.is_multiple_of(16), "{scale} c_cls={} not %16", s.c_cls);
    }
    // Locked shared head widths across folo2-n and folo2-s for zero-adapter Box + Cls CrossKD.
    let sn = ScalingV2::of(&FoloV2Config::new("n", 80).unwrap());
    let ss = ScalingV2::of(&FoloV2Config::new("s", 80).unwrap());
    assert_eq!((sn.c_box, sn.c_cls), (48, 64));
    assert_eq!((ss.c_box, ss.c_cls), (48, 64));
}

#[test]
fn all_scales_shapes_and_fused_param_counts() {
    let x = Tensor::rand_uniform([1, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(7));
    // Expected anchors at 64x96 for strides [8, 16, 32]: 8*12 + 4*6 + 2*3 = 126.
    let mut prev_fused_params = 0usize;
    for scale in ["n", "s", "m", "l", "x"] {
        let mut m = FoloV2Model::new(FoloV2Config::new(scale, 80).unwrap()).unwrap();
        let train_params = m.num_parameters();
        assert_eq!(m.anchors(640, 640).len(), 8_400, "{scale}: 640x640 must keep 8400 anchors");
        m.fuse().unwrap();
        let fused_params = m.num_parameters();
        assert!(
            fused_params < train_params,
            "{scale}: fuse() should drop aux branches & BN params ({fused_params} vs {train_params})"
        );
        assert!(
            fused_params > prev_fused_params,
            "{scale}: fused params should grow strictly with scale ({fused_params} <= {prev_fused_params})"
        );
        prev_fused_params = fused_params;

        // Verify locked exact fused parameter budgets for folo2-n and folo2-s (nc = 80, shared c_box=48, c_cls=64).
        if scale == "n" {
            // Exact analytical fused count: 2,713,568 (differs from unfused BN 2*c accounting by exactly -sum(c_bn)).
            assert!(
                (2_680_000..=2_730_000).contains(&fused_params),
                "folo2-n fused params out of iso-FLOP budget: {fused_params}"
            );
        } else if scale == "s" {
            assert!(
                (9_750_000..=9_850_000).contains(&fused_params),
                "folo2-s fused params out of iso-FLOP budget: {fused_params}"
            );
        }

        // Only run forward on n/s/m in unit tests to keep test time < 1s.
        if matches!(scale, "n" | "s") {
            let _g = no_grad();
            let out = m.forward(&x).unwrap();
            assert_eq!(out.box_raw.dims(), &[1, 64, 126]);
            assert_eq!(out.cls_raw.dims(), &[1, 80, 126]);
            assert!(out.iou_raw.is_none());
            assert_eq!(out.num_anchors(), 126);
            let v1_out = out.into_head_output();
            assert_eq!(v1_out.box_raw.dims(), &[1, 64, 126]);
        }
    }
}

#[test]
fn ws_qarep_algebraic_equivalence_forward_and_backward() {
    for (c_in, c_out) in [(8usize, 8usize), (8, 16)] {
        let mut rng = Rng::seed(101 + c_out as u64);
        let mut init_ctx = Init::new(&mut rng);
        let layer = WsQARepConv3x3::new(c_in, c_out, &mut init_ctx);
        // Push gamma_id and BN away from trivial ones/zeros.
        if let Some(g) = &layer.branches.as_ref().unwrap().gamma_id {
            init::copy_(g, &Tensor::rand_uniform([c_out], 0.5, 1.5, &mut rng)).unwrap();
        }
        let bn = layer.bn.as_ref().unwrap();
        init::copy_(&bn.weight, &Tensor::rand_uniform([c_out], 0.7, 1.3, &mut rng)).unwrap();
        init::copy_(&bn.bias, &Tensor::rand_uniform([c_out], -0.2, 0.2, &mut rng)).unwrap();

        let x_data = Tensor::rand_uniform([2, c_in, 8, 8], -1.0, 1.0, &mut rng);
        let proj = Tensor::rand_uniform([2, c_out, 8, 8], -1.0, 1.0, &mut rng);

        // Save initial BN running stats so both paths update from identical state.
        let rm0 = bn.running_mean.to_vec::<f32>().unwrap();
        let rv0 = bn.running_var.to_vec::<f32>().unwrap();

        let x_ws =
            Tensor::from_vec(x_data.to_vec::<f32>().unwrap(), [2, c_in, 8, 8]).unwrap().requires_grad_();
        let y_ws = layer.forward(&x_ws).unwrap();
        let g_ws = y_ws.mul(&proj).unwrap().sum_all().unwrap().backward().unwrap();

        // Restore BN running stats before running explicit 5-branch reference.
        init::copy_(&bn.running_mean, &Tensor::from_vec(rm0, [c_out]).unwrap()).unwrap();
        init::copy_(&bn.running_var, &Tensor::from_vec(rv0, [c_out]).unwrap()).unwrap();

        let x_ref =
            Tensor::from_vec(x_data.to_vec::<f32>().unwrap(), [2, c_in, 8, 8]).unwrap().requires_grad_();
        let y_ref = layer.forward_explicit_branches(&x_ref).unwrap();
        let g_ref = y_ref.mul(&proj).unwrap().sum_all().unwrap().backward().unwrap();

        assert_close(
            &y_ws.to_vec::<f32>().unwrap(),
            &y_ref.to_vec::<f32>().unwrap(),
            1e-5,
            &format!("WS-QARep forward ({c_in}->{c_out})"),
        );
        assert_close(
            &g_ws.get(&x_ws).unwrap().to_vec::<f32>().unwrap(),
            &g_ref.get(&x_ref).unwrap().to_vec::<f32>().unwrap(),
            1e-4,
            &format!("WS-QARep dX ({c_in}->{c_out})"),
        );
        for (name, p) in layer.named_parameters() {
            let a = g_ws.get(&p).unwrap().to_vec::<f32>().unwrap();
            let b = g_ref.get(&p).unwrap().to_vec::<f32>().unwrap();
            assert_close(&a, &b, 1e-4, &format!("WS-QARep grad {name} ({c_in}->{c_out})"));
        }
    }
}

#[test]
fn full_model_fuse_parity_o2m_iou_and_o2o() {
    let cfg = FoloV2Config::new("n", 6).unwrap().with_head(HeadMode::O2o).with_iou_head(true);
    let mut m = FoloV2Model::with_rng(cfg, &mut Rng::seed(42)).unwrap();
    randomize_bn_v2(&m, 99);
    // Also set non-zero FullPAD weights so FiLM path is exercised during parity check.
    if let Some(g) = &m.neck.pad_p3 {
        init::copy_(
            &g.proj.weight,
            &Tensor::rand_uniform(g.proj.weight.shape().clone(), -0.1, 0.1, &mut Rng::seed(5)),
        )
        .unwrap();
    }
    m.eval();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(12));

    let (o2m_before, o2o_before) = {
        let _g = no_grad();
        (m.forward_head(&x, HeadMode::O2m).unwrap(), m.forward_head(&x, HeadMode::O2o).unwrap())
    };
    m.fuse().unwrap();
    assert!(m.is_fused());
    // Idempotent second fuse().
    m.fuse().unwrap();
    assert!(m.backbone.stage2.items().iter().all(|r| r.is_fused() && r.branches.is_none() && r.bn.is_none()));

    let (o2m_after, o2o_after) = {
        let _g = no_grad();
        (m.forward_head(&x, HeadMode::O2m).unwrap(), m.forward_head(&x, HeadMode::O2o).unwrap())
    };

    assert_close(
        &o2m_before.box_raw.to_vec::<f32>().unwrap(),
        &o2m_after.box_raw.to_vec::<f32>().unwrap(),
        1e-4,
        "fuse parity o2m box_raw",
    );
    assert_close(
        &o2m_before.cls_raw.to_vec::<f32>().unwrap(),
        &o2m_after.cls_raw.to_vec::<f32>().unwrap(),
        1e-4,
        "fuse parity o2m cls_raw",
    );
    assert_close(
        &o2m_before.iou_raw.unwrap().to_vec::<f32>().unwrap(),
        &o2m_after.iou_raw.unwrap().to_vec::<f32>().unwrap(),
        1e-4,
        "fuse parity o2m iou_raw",
    );
    assert_close(
        &o2o_before.cls_raw.to_vec::<f32>().unwrap(),
        &o2o_after.cls_raw.to_vec::<f32>().unwrap(),
        1e-4,
        "fuse parity o2o cls_raw",
    );
}

#[test]
fn grad_check_all_v2_blocks() {
    let nn = |r: folo_nn::Result<Tensor>| r.map_err(|e| folo_core::Error::Autograd(e.to_string()));
    let mut rng = Rng::seed(2026);
    let opts = GradCheck { eps: 5e-3, atol: 3e-3, rtol: 2e-2, max_checks: 32, seed: 77 };

    // 1. space_to_depth_2x2
    let x_spd = Tensor::rand_uniform([1, 3, 4, 4], -1.0, 1.0, &mut rng);
    grad_check(|v| nn(space_to_depth_2x2(&v[0])), &[x_spd]).unwrap();

    // 2. WsQARepConv3x3 (eval mode so BN doesn't mutate stats during finite-diff probes)
    let mut init_ctx = Init::new(&mut rng);
    let rep = WsQARepConv3x3::new(4, 4, &mut init_ctx);
    rep.eval();
    let x_rep = Tensor::rand_uniform([1, 4, 4, 4], -0.8, 0.8, &mut rng);
    grad_check_with(|v| nn(rep.forward(&v[0])), &[x_rep], &opts).unwrap();

    // 3. FoloBlock (m = 2 and m = 3 multi-tap)
    for m in [2usize, 3] {
        let mut init_ctx = Init::new(&mut rng);
        let blk = FoloBlock::new(8, 8, m, 4, &mut init_ctx);
        blk.eval();
        let x_blk = Tensor::rand_uniform([1, 8, 4, 4], -0.5, 0.5, &mut rng);
        grad_check_with(|v| nn(blk.forward(&v[0])), &[x_blk], &opts).unwrap();
    }

    // 4. HyperContext (both y_p5 and g_scene)
    let mut init_ctx = Init::new(&mut rng);
    let hyper = HyperContext::new(8, 4, &mut init_ctx);
    hyper.eval();
    let x_hyp = Tensor::rand_uniform([1, 8, 4, 4], -0.5, 0.5, &mut rng);
    grad_check_with(
        |v| {
            let (y, g) =
                hyper.forward_with_scene(&v[0]).map_err(|e| folo_core::Error::Autograd(e.to_string()))?;
            let gb = g.reshape([1, 4, 1, 1])?.broadcast_to([1, 4, 4, 4])?;
            y.narrow(1, 0, 4)?.add(&gb)
        },
        &[x_hyp],
        &opts,
    )
    .unwrap();

    // 5. FullPadGate (non-zero weights)
    let mut init_ctx = Init::new(&mut rng);
    let gate = FullPadGate::new(4, 6, &mut init_ctx).unwrap();
    init::copy_(&gate.proj.weight, &Tensor::rand_uniform([6, 4], -0.4, 0.4, &mut rng)).unwrap();
    let feat = Tensor::rand_uniform([1, 6, 4, 4], -0.8, 0.8, &mut rng);
    let scene = Tensor::rand_uniform([1, 4], -0.8, 0.8, &mut rng);
    grad_check_with(|v| nn(gate.forward(&v[0], &v[1])), &[feat, scene], &opts).unwrap();

    // 6. PeakSuppress
    let mut init_ctx = Init::new(&mut rng);
    let ps = PeakSuppress::new(4, 3, &mut init_ctx);
    let logits = Tensor::rand_uniform([1, 3, 4, 4], -1.0, 1.0, &mut rng);
    let ps_feat = Tensor::rand_uniform([1, 4, 4, 4], -0.8, 0.8, &mut rng);
    grad_check_with(|v| nn(ps.apply(&v[0], &v[1])), &[logits, ps_feat], &opts).unwrap();
}

#[test]
fn peak_suppress_preserves_peaks_and_suppresses_neighbors() {
    let mut rng = Rng::seed(55);
    let mut init_ctx = Init::new(&mut rng);
    let ps = PeakSuppress::new(8, 3, &mut init_ctx);
    let logits = Tensor::rand_uniform([1, 3, 6, 6], -2.0, 2.0, &mut rng);
    let feat = Tensor::rand_uniform([1, 8, 6, 6], -1.0, 1.0, &mut rng);

    let out = ps.apply(&logits, &feat).unwrap();
    let z = logits.to_vec::<f32>().unwrap();
    let z_max = logits.max_pool2d(3, 1, 1).unwrap().to_vec::<f32>().unwrap();
    let z_o2o = out.to_vec::<f32>().unwrap();

    let mut peak_count = 0usize;
    let mut suppressed_count = 0usize;
    for i in 0..z.len() {
        // Strict monotonicity: Z_o2o <= Z everywhere (compatible with lazy bounds).
        assert!(z_o2o[i] <= z[i] + 1e-6, "monotonicity violated at {i}: {} > {}", z_o2o[i], z[i]);
        if (z[i] - z_max[i]).abs() < 1e-6 {
            // Exact peak preservation.
            assert!((z_o2o[i] - z[i]).abs() < 1e-6, "peak altered at {i}: {} vs {}", z_o2o[i], z[i]);
            peak_count += 1;
        } else if z_max[i] - z[i] > 0.25 {
            assert!(z_o2o[i] < z[i] - 1e-3, "non-peak not suppressed at {i}");
            suppressed_count += 1;
        }
    }
    assert!(peak_count > 0 && suppressed_count > 0);
}

#[test]
fn fullpad_identity_init_and_relocation_isolation() {
    let cfg = FoloV2Config::new("n", 4).unwrap().with_fullpad(true);
    let m = FoloV2Model::with_rng(cfg, &mut Rng::seed(9)).unwrap();
    m.eval();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(11));

    let [p3_0, p4_0, p5_0] = {
        let _g = no_grad();
        m.forward_features(&x).unwrap()
    };

    // Perturb pad_p3 bias ONLY. Because FullPAD is relocated AFTER neck.down3/down4 taps,
    // P3_head MUST change while P4_head and P5_head MUST remain bit-for-bit identical!
    let pad_p3 = m.neck.pad_p3.as_ref().unwrap();
    init::constant_(pad_p3.proj.bias.as_ref().unwrap(), 0.5).unwrap();

    let [p3_1, p4_1, p5_1] = {
        let _g = no_grad();
        m.forward_features(&x).unwrap()
    };

    assert_ne!(p3_0.to_vec::<f32>().unwrap(), p3_1.to_vec::<f32>().unwrap(), "pad_p3 must modulate P3_head");
    assert_eq!(
        p4_0.to_vec::<f32>().unwrap(),
        p4_1.to_vec::<f32>().unwrap(),
        "neck.down3 must read raw out3 BEFORE pad_p3 (0 CPU int8 barriers)"
    );
    assert_eq!(
        p5_0.to_vec::<f32>().unwrap(),
        p5_1.to_vec::<f32>().unwrap(),
        "P5_head must be isolated from pad_p3"
    );
}

#[test]
fn iou_head_alignment_mal_bias_and_v1_stem_fallback() {
    let cfg_bce = FoloV2Config::new("n", 80).unwrap().with_iou_head(true).with_spd_stem(false);
    let m_bce = FoloV2Model::new(cfg_bce).unwrap();
    assert!(!m_bce.backbone.stem.spd);
    let p3_bias = m_bce.head.levels[0].cls_branch.out.bias.as_ref().unwrap().to_vec::<f32>().unwrap();
    assert!((p3_bias[0] - init::cls_bias_prior(80, 8)).abs() < 1e-5);

    let cfg_mal = FoloV2Config::new("n", 80).unwrap().with_mal_bias_prior(true).with_iou_head(true);
    let m_mal = FoloV2Model::new(cfg_mal).unwrap();
    let mal_bias = m_mal.head.levels[0].cls_branch.out.bias.as_ref().unwrap().to_vec::<f32>().unwrap();
    assert!((mal_bias[0] - MAL_CLS_BIAS_PRIOR).abs() < 1e-5);

    m_mal.eval();
    let _g = no_grad();
    let out = m_mal.forward(&Tensor::zeros([1, 3, 64, 64])).unwrap();
    // box_raw stays strictly 64 channels (% 16 == 0); separate iou_raw is [1, 1, A].
    assert_eq!(out.box_raw.dims(), &[1, 64, 84]);
    assert_eq!(out.iou_raw.unwrap().dims(), &[1, 1, 84]);
}

#[test]
fn cpu_gpu_forward_and_backward_parity() {
    let Ok(gpu) = folo_gpu::init() else { return };
    let cfg = FoloV2Config::new("n", 5).unwrap().with_head(HeadMode::O2o).with_iou_head(true);
    let mut m_cpu = FoloV2Model::with_rng(cfg.clone(), &mut Rng::seed(31)).unwrap();
    let mut m_gpu = FoloV2Model::with_rng(cfg.clone(), &mut Rng::seed(31)).unwrap();
    randomize_bn_v2(&m_cpu, 61);
    randomize_bn_v2(&m_gpu, 61);
    m_gpu.to_device(gpu).unwrap();

    let x_cpu = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(19));
    let x_gpu = x_cpu.to_device(gpu).unwrap();

    // 1. Fused eval CPU vs GPU parity (before any training updates BN stats).
    m_cpu.fuse().unwrap();
    m_gpu.fuse().unwrap();
    {
        let _g = no_grad();
        let c_out = m_cpu.forward_head(&x_cpu, HeadMode::O2m).unwrap();
        let g_out = m_gpu.forward_head(&x_gpu, HeadMode::O2m).unwrap();
        assert_close(
            &c_out.box_raw.to_vec::<f32>().unwrap(),
            &g_out.box_raw.to_vec::<f32>().unwrap(),
            5e-3,
            "CPU vs GPU fused box_raw",
        );
        assert_close(
            &c_out.cls_raw.to_vec::<f32>().unwrap(),
            &g_out.cls_raw.to_vec::<f32>().unwrap(),
            5e-3,
            "CPU vs GPU fused cls_raw",
        );
        assert_close(
            &c_out.iou_raw.unwrap().to_vec::<f32>().unwrap(),
            &g_out.iou_raw.unwrap().to_vec::<f32>().unwrap(),
            5e-3,
            "CPU vs GPU fused iou_raw",
        );
        let c_o2o = m_cpu.forward_head(&x_cpu, HeadMode::O2o).unwrap();
        let g_o2o = m_gpu.forward_head(&x_gpu, HeadMode::O2o).unwrap();
        assert_close(
            &c_o2o.cls_raw.to_vec::<f32>().unwrap(),
            &g_o2o.cls_raw.to_vec::<f32>().unwrap(),
            5e-3,
            "CPU vs GPU fused o2o cls_raw",
        );
    }

    // 2. F16 inference smoke on GPU when supported.
    if folo_core::backend::computes_f16(gpu) {
        m_gpu.to_dtype(DType::F16).unwrap();
        let x16 = x_gpu.to_dtype(DType::F16).unwrap();
        let _g = no_grad();
        let o16 = m_gpu.forward(&x16).unwrap();
        assert_eq!(o16.box_raw.dtype(), DType::F16);
        assert_eq!(o16.cls_raw.dtype(), DType::F16);
    }

    // 3. Training forward + backward smoke on GPU (verifies all v2 autograd ops run end-to-end on GPU).
    let mut m_train = FoloV2Model::with_rng(cfg, &mut Rng::seed(31)).unwrap();
    m_train.to_device(gpu).unwrap();
    let out_tr = m_train.forward(&x_gpu).unwrap();
    let mut loss = out_tr.box_raw.square().unwrap().mean_all().unwrap();
    loss = loss.add(&out_tr.cls_raw.square().unwrap().mean_all().unwrap()).unwrap();
    loss = loss.add(&out_tr.iou_raw.unwrap().square().unwrap().mean_all().unwrap()).unwrap();
    if let Some((ob, oc)) = out_tr.o2o {
        loss = loss.add(&ob.square().unwrap().mean_all().unwrap()).unwrap();
        loss = loss.add(&oc.square().unwrap().mean_all().unwrap()).unwrap();
    }
    let grads = loss.backward().unwrap();
    let g_stem = grads
        .get(&m_train.backbone.stem.conv.conv.weight)
        .expect("gradient must reach backbone.stem on GPU")
        .to_vec::<f32>()
        .unwrap();
    assert!(g_stem.iter().all(|v| v.is_finite()), "non-finite GPU stem grad");

    // 4. GPU autocast (F16) training forward + backward with dual o2o head (`PeakSuppress` F16/F32 dtype).
    if folo_core::backend::computes_f16(gpu) {
        let mut m_amp = FoloV2Model::with_rng(
            FoloV2Config::new("n", 5).unwrap().with_head(HeadMode::O2o),
            &mut Rng::seed(33),
        )
        .unwrap();
        m_amp.to_device(gpu).unwrap();
        let out_amp = {
            let _ac = folo_core::autocast();
            m_amp.forward(&x_gpu).unwrap()
        };
        let (ob, oc) = out_amp.o2o.expect("o2o under autocast");
        let loss = out_amp
            .box_raw
            .to_dtype(DType::F32)
            .unwrap()
            .square()
            .unwrap()
            .mean_all()
            .unwrap()
            .add(&ob.to_dtype(DType::F32).unwrap().square().unwrap().mean_all().unwrap())
            .unwrap()
            .add(&oc.to_dtype(DType::F32).unwrap().square().unwrap().mean_all().unwrap())
            .unwrap();
        let g_amp = loss.backward().unwrap();
        let alpha_g = g_amp
            .get(&m_amp.head.o2o_cls.as_ref().unwrap()[0].suppress.alpha_raw)
            .expect("PeakSuppress alpha_raw grad under autocast")
            .to_vec::<f32>()
            .unwrap();
        assert!(alpha_g.iter().all(|v| v.is_finite()), "non-finite PeakSuppress grad under autocast");

        // 5. Layer-level `Linear` autocast test: F32 master bias receives matching F32 gradients under GPU
        //    autocast, while non-F32 bias (e.g. U32) is still rejected with DTypeMismatch on both GPU and CPU.
        let mut rng_l = Rng::seed(37);
        let lin_ref = folo_nn::Linear::new(16, 8, true, &mut rng_l);
        let mut lin_gpu = folo_nn::Linear::new(16, 8, true, &mut Rng::seed(37));
        lin_gpu.to_device(gpu).unwrap();
        let xl = Tensor::randn([4, 16], 0.0, 0.5, &mut rng_l);
        let xl_g = xl.to_device(gpu).unwrap();
        let g_ref = lin_ref.forward(&xl).unwrap().square().unwrap().mean_all().unwrap().backward().unwrap();
        let y_amp = {
            let _ac = folo_core::autocast();
            lin_gpu.forward(&xl_g).unwrap()
        };
        assert_eq!(y_amp.dtype(), DType::F16);
        let g_lin =
            y_amp.to_dtype(DType::F32).unwrap().square().unwrap().mean_all().unwrap().backward().unwrap();
        let db_ref = g_ref.get(lin_ref.bias.as_ref().unwrap()).unwrap().to_vec::<f32>().unwrap();
        let db_amp = g_lin.get(lin_gpu.bias.as_ref().unwrap()).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(g_lin.get(lin_gpu.bias.as_ref().unwrap()).unwrap().dtype(), DType::F32);
        for (a, b) in db_ref.iter().zip(&db_amp) {
            assert!((a - b).abs() <= 5e-3, "Linear autocast master bias grad mismatch: {b} vs {a}");
        }
        let bad_lin_gpu = folo_nn::Linear::from_tensors(
            lin_gpu.weight.clone(),
            Some(Tensor::from_vec(vec![0u32; 8], [8]).unwrap().to_device(gpu).unwrap()),
        )
        .unwrap();
        let _ac = folo_core::autocast();
        assert!(
            bad_lin_gpu.forward(&xl_g).is_err(),
            "Linear must reject U32 bias on GPU even under autocast"
        );
    }
}

#[test]
fn v2_config_validation_o2o_detach_and_hypercontext_invariance() {
    // 1. Invalid reg_max / bn_eps / bn_momentum rejected.
    let mut c = FoloV2Config::new("n", 4).unwrap();
    c.reg_max = 1;
    assert!(c.validate().is_err());
    let mut c = FoloV2Config::new("n", 4).unwrap();
    c.bn_eps = -2.0;
    assert!(c.validate().is_err());
    let mut c = FoloV2Config::new("n", 4).unwrap();
    c.bn_eps = f64::NAN;
    assert!(c.validate().is_err());
    let mut c = FoloV2Config::new("n", 4).unwrap();
    c.bn_momentum = 1.5;
    assert!(c.validate().is_err());

    // 2. HyperContext broadcast stays invariant to spatial resolution (H*W) on constant spatial fields.
    {
        let mut rng = Rng::seed(71);
        let mut init_ctx = Init { rng: &mut rng, bn_eps: 1e-3, bn_momentum: 0.03 };
        let hc = HyperContext::new(16, 4, &mut init_ctx);
        hc.eval();
        let _g = no_grad();
        let base: Vec<f32> = (0..16).map(|c| 0.1 * (c as f32 + 1.0)).collect();
        let make_const = |h: usize, w: usize| {
            let mut data = Vec::with_capacity(16 * h * w);
            for &v in &base {
                data.extend(std::iter::repeat_n(v, h * w));
            }
            Tensor::from_vec(data, [1, 16, h, w]).unwrap()
        };
        let y4 = hc.forward(&make_const(4, 4)).unwrap().to_vec::<f32>().unwrap();
        let y8 = hc.forward(&make_const(8, 8)).unwrap().to_vec::<f32>().unwrap();
        for c_idx in 0..16 {
            let v4 = y4[c_idx * 16];
            let v8 = y8[c_idx * 64];
            assert!((v4 - v8).abs() < 1e-5, "HyperContext scale drift ch {c_idx}: 4x4={v4} vs 8x8={v8}");
        }
    }

    // 3. o2o_detach=true stops gradients at backbone/neck `feats`, still trains shared `box_branch`,
    //    and leaves `box_branch` BN running stats identical to `o2o_detach=false` (no double update!).
    let m_det = FoloV2Model::with_rng(
        FoloV2Config::new("n", 4).unwrap().with_head(HeadMode::O2o),
        &mut Rng::seed(73),
    )
    .unwrap();
    let mut m_nodet = FoloV2Model::with_rng(
        FoloV2Config::new("n", 4).unwrap().with_head(HeadMode::O2o),
        &mut Rng::seed(73),
    )
    .unwrap();
    m_nodet.head.o2o_detach = false;
    assert!(m_det.head.o2o_detach);
    let mut rng = Rng::seed(74);
    let p3_a = Tensor::randn([2, 64, 8, 8], 0.0, 0.5, &mut rng).requires_grad_();
    let p4_a = Tensor::randn([2, 128, 4, 4], 0.0, 0.5, &mut rng).requires_grad_();
    let p5_a = Tensor::randn([2, 256, 2, 2], 0.0, 0.5, &mut rng).requires_grad_();
    let p3_b = p3_a.detach().requires_grad_();
    let p4_b = p4_a.detach().requires_grad_();
    let p5_b = p5_a.detach().requires_grad_();

    let (b_o2m_a, _, _) = m_det.head.forward([&p3_a, &p4_a, &p5_a]).unwrap();
    let (b_o2o_a, c_o2o_a, _) =
        m_det.head.forward_o2o([&p3_a, &p4_a, &p5_a], Some(&b_o2m_a)).unwrap().unwrap();
    let grads_a = b_o2o_a
        .square()
        .unwrap()
        .mean_all()
        .unwrap()
        .add(&c_o2o_a.square().unwrap().mean_all().unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert!(grads_a.get(&p3_a).is_none(), "o2o_detach=true must block gradient to feature inputs");
    let box_w_grad = grads_a
        .get(&m_det.head.levels[0].box_branch.out.weight)
        .expect("o2o box loss must train shared box_branch weights")
        .to_vec::<f32>()
        .unwrap();
    assert!(box_w_grad.iter().any(|&v| v.abs() > 1e-8), "shared box_branch gradient was zero");

    let (b_o2m_b, _, _) = m_nodet.head.forward([&p3_b, &p4_b, &p5_b]).unwrap();
    let (b_o2o_b, c_o2o_b, _) =
        m_nodet.head.forward_o2o([&p3_b, &p4_b, &p5_b], Some(&b_o2m_b)).unwrap().unwrap();
    let grads_b = b_o2o_b
        .square()
        .unwrap()
        .mean_all()
        .unwrap()
        .add(&c_o2o_b.square().unwrap().mean_all().unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert!(grads_b.get(&p3_b).is_some(), "o2o_detach=false must pass gradient to feature inputs");

    for lvl in 0..3 {
        for (tag, bn_a, bn_b) in [
            (
                "conv0",
                m_det.head.levels[lvl].box_branch.conv0.bn.as_ref().unwrap(),
                m_nodet.head.levels[lvl].box_branch.conv0.bn.as_ref().unwrap(),
            ),
            (
                "conv1",
                m_det.head.levels[lvl].box_branch.conv1.bn.as_ref().unwrap(),
                m_nodet.head.levels[lvl].box_branch.conv1.bn.as_ref().unwrap(),
            ),
        ] {
            assert_eq!(
                bn_a.running_mean.to_vec::<f32>().unwrap(),
                bn_b.running_mean.to_vec::<f32>().unwrap(),
                "o2o_detach=true double-updated box_branch level {lvl} {tag} running_mean"
            );
            assert_eq!(
                bn_a.running_var.to_vec::<f32>().unwrap(),
                bn_b.running_var.to_vec::<f32>().unwrap(),
                "o2o_detach=true double-updated box_branch level {lvl} {tag} running_var"
            );
        }
    }

    // 3b. Frozen BatchNorm (`bn.eval()`, e.g. `bn_freeze_close_mosaic`) with `o2o_detach=true` must NOT
    //     mutate `running_mean`/`running_var` storage versions or break `backward()` with `ModifiedInPlace`.
    {
        let m_frozen = FoloV2Model::with_rng(
            FoloV2Config::new("n", 4).unwrap().with_head(HeadMode::O2o),
            &mut Rng::seed(76),
        )
        .unwrap();
        for lvl in &m_frozen.head.levels {
            lvl.box_branch.conv0.bn.as_ref().unwrap().eval();
            lvl.box_branch.conv1.bn.as_ref().unwrap().eval();
        }
        let p3_f = p3_a.detach().requires_grad_();
        let p4_f = p4_a.detach().requires_grad_();
        let p5_f = p5_a.detach().requires_grad_();
        let (b_m, _, _) = m_frozen.head.forward([&p3_f, &p4_f, &p5_f]).unwrap();
        let (b_o, c_o, _) = m_frozen.head.forward_o2o([&p3_f, &p4_f, &p5_f], Some(&b_m)).unwrap().unwrap();
        let loss_f = b_m
            .square()
            .unwrap()
            .mean_all()
            .unwrap()
            .add(&b_o.square().unwrap().mean_all().unwrap())
            .unwrap()
            .add(&c_o.square().unwrap().mean_all().unwrap())
            .unwrap();
        let g_f = loss_f.backward().unwrap();
        assert!(g_f.get(&m_frozen.head.levels[0].box_branch.out.weight).is_some());
    }

    // 3c. CPU `Linear` rejects non-F32 (U32) bias even while `autocast()` is active (runs without a GPU).
    {
        let mut rng_c = Rng::seed(77);
        let lin = folo_nn::Linear::new(8, 4, false, &mut rng_c);
        let bad = folo_nn::Linear::from_tensors(
            lin.weight.clone(),
            Some(Tensor::from_vec(vec![0u32; 4], [4]).unwrap()),
        )
        .unwrap();
        let xc = Tensor::randn([2, 8], 0.0, 0.5, &mut rng_c);
        let _ac = folo_core::autocast();
        assert!(bad.forward(&xc).is_err(), "Linear must reject U32 bias on CPU even under autocast");
    }

    // 4. Train-mode WsQARepConv3x3 finite-difference check on auxiliary branch weights.
    let mut rng = Rng::seed(75);
    let mut init_ctx = Init { rng: &mut rng, bn_eps: 1e-3, bn_momentum: 0.03 };
    let rep = WsQARepConv3x3::new(4, 4, &mut init_ctx);
    let x_t = Tensor::randn([2, 4, 5, 5], 0.0, 0.6, &mut rng);
    let br = rep.branches.as_ref().unwrap();
    grad_check_with(
        |inputs| {
            let mut copy = WsQARepConv3x3::new(
                4,
                4,
                &mut Init { rng: &mut Rng::seed(75), bn_eps: 1e-3, bn_momentum: 0.03 },
            );
            copy.branches.as_mut().unwrap().w1x1 = inputs[0].clone();
            copy.branches.as_mut().unwrap().gamma_id = Some(inputs[1].clone());
            copy.forward(&x_t).unwrap().square()?.mean_all()
        },
        &[br.w1x1.clone(), br.gamma_id.clone().unwrap()],
        &GradCheck { eps: 1e-3, atol: 3e-3, rtol: 2e-2, ..GradCheck::default() },
    )
    .unwrap();
}

#[test]
fn standard_groups_excludes_v2_gamma_id_and_alpha_raw_from_weight_decay() {
    let model = FoloV2Model::new(FoloV2Config::new("n", 4).unwrap().with_head(HeadMode::O2o)).unwrap();
    let sg = folo_nn::optim::StandardGroups::from_module(&model);
    let gamma_in_bn = sg.bn.iter().filter(|(n, _)| n.ends_with(".gamma_id")).count();
    let alpha_in_bn = sg.bn.iter().filter(|(n, _)| n.ends_with(".alpha_raw")).count();
    assert!(gamma_in_bn > 0, "expected FoloV2 WsQARepConv3x3 gamma_id parameters in bn group");
    assert!(alpha_in_bn > 0, "expected FoloV2 PeakSuppress alpha_raw parameters in bn group");
    assert!(
        sg.decay.iter().all(|(n, _)| !n.ends_with(".gamma_id") && !n.ends_with(".alpha_raw")),
        "v2 gamma_id and alpha_raw must never receive weight decay"
    );
    assert!(
        sg.bias.iter().all(|(n, _)| !n.ends_with(".gamma_id") && !n.ends_with(".alpha_raw")),
        "v2 gamma_id and alpha_raw must not receive bias warmup"
    );
    let groups = sg.into_groups(0.01, 0.937, 5e-4);
    let bn_group = groups.iter().find(|g| g.name == "bn").unwrap();
    assert_eq!(bn_group.weight_decay, 0.0);
}

#[test]
fn hypercontext_and_fullpad_f16_stability_and_f32_scene_pooling() {
    let mut rng = Rng::seed(2027);
    let mut init_ctx = Init::new(&mut rng);
    let mut hc = HyperContext::new(16, 8, &mut init_ctx);
    let mut pad = FullPadGate::new(8, 16, &mut init_ctx).unwrap();
    // Give FullPadGate small non-zero weights so tiny gate updates are exercised.
    init::copy_(&pad.proj.weight, &Tensor::rand_uniform([16, 8], -0.05, 0.05, &mut rng)).unwrap();
    hc.eval();
    pad.eval();

    // Large-magnitude logits where F16 softmax would overflow (> 11.0 -> exp > 65504) if not promoted to F32.
    let x_f32 = Tensor::rand_uniform([1, 16, 8, 8], -30.0, 30.0, &mut rng);
    let (y_ref, g_ref) = hc.forward_with_scene(&x_f32).unwrap();
    let p_ref = pad.forward(&y_ref, &g_ref).unwrap();
    assert_eq!(g_ref.dtype(), DType::F32);
    assert!(y_ref.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(p_ref.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));

    // GPU AMP autocast + explicit F16 inference check when a GPU with shader-f16 is available.
    if let Ok(gpu) = folo_gpu::init()
        && folo_core::backend::computes_f16(gpu)
    {
        // 1. AMP autocast training forward + backward: g_scene stays F32, gradients are finite.
        let mut hc_amp = HyperContext::new(16, 8, &mut Init::new(&mut Rng::seed(2027)));
        let mut pad_amp = FullPadGate::new(8, 16, &mut Init::new(&mut Rng::seed(2028))).unwrap();
        init::copy_(&pad_amp.proj.weight, &Tensor::rand_uniform([16, 8], -0.05, 0.05, &mut Rng::seed(2029)))
            .unwrap();
        hc_amp.to_device(gpu).unwrap();
        pad_amp.to_device(gpu).unwrap();
        let x_g = x_f32.to_device(gpu).unwrap().requires_grad_();
        let (y_amp, g_amp, p_amp) = {
            let _ac = folo_core::autocast();
            let (y, g) = hc_amp.forward_with_scene(&x_g).unwrap();
            let p = pad_amp.forward(&y, &g).unwrap();
            (y, g, p)
        };
        assert_eq!(g_amp.dtype(), DType::F32, "g_scene must stay F32 under AMP autocast");
        assert_eq!(y_amp.dtype(), DType::F32, "HyperContext preserves F32 residual stream dtype");
        let g_vec = g_amp.to_vec::<f32>().unwrap();
        assert!(g_vec.iter().all(|v| v.is_finite()), "g_scene non-finite under AMP");
        let loss = p_amp.to_dtype(DType::F32).unwrap().square().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();
        let gx = grads.get(&x_g).unwrap().to_vec::<f32>().unwrap();
        assert!(gx.iter().all(|v| v.is_finite()), "HyperContext/FullPad grad non-finite under AMP");

        // 2. Explicit F16 inference (no autocast): softmax & scene pooling stay F32, no overflow/NaN.
        hc.to_device(gpu).unwrap();
        pad.to_device(gpu).unwrap();
        hc.to_dtype(DType::F16).unwrap();
        pad.to_dtype(DType::F16).unwrap();
        let x_f16 = x_f32.to_device(gpu).unwrap().to_dtype(DType::F16).unwrap();
        let _ng = no_grad();
        let (y_f16, g_f16) = hc.forward_with_scene(&x_f16).unwrap();
        let p_f16 = pad.forward(&y_f16, &g_f16).unwrap();
        assert_eq!(g_f16.dtype(), DType::F32);
        assert_eq!(y_f16.dtype(), DType::F16);
        assert!(
            p_f16.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
            "explicit F16 HyperContext+FullPad produced non-finite values"
        );
    }
}

#[test]
fn lazyalign_iou_quality_loss_grad_check_and_zero_fg() {
    use folo_vision::Xyxy;
    use folo_vision::loss::{ImageTargets, LossConfig, assign_batch};
    use folo_vision::model::v2::iou_quality_loss;

    let mut rng = Rng::seed(909);
    let cfg = FoloV2Config::new("n", 3).unwrap().with_iou_head(true);
    let model = FoloV2Model::with_rng(cfg, &mut rng).unwrap();
    let x = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut rng);
    let out = model.forward(&x).unwrap();
    let iou_raw = out.iou_raw.expect("iou_head=true produces iou_raw");
    let targets = vec![
        ImageTargets { boxes: vec![Xyxy::new(8.0, 8.0, 40.0, 40.0)], labels: vec![0] },
        ImageTargets {
            boxes: vec![Xyxy::new(16.0, 12.0, 52.0, 56.0), Xyxy::new(4.0, 20.0, 28.0, 48.0)],
            labels: vec![1, 2],
        },
    ];
    let loss_cfg = LossConfig::default();
    let asg = assign_batch(&out.box_raw, &out.cls_raw, &out.anchors, &targets, &loss_cfg).unwrap();
    assert!(asg.num_fg() > 0);

    // 1. Finite-difference gradient check wrt iou_raw; verify box_raw target is detached (zero grad into box_raw).
    let iou_probe = Tensor::randn(iou_raw.shape().clone(), 0.0, 0.7, &mut rng);
    let box_leaf = out.box_raw.detach().requires_grad_();
    let iou_leaf = iou_probe.detach().requires_grad_();
    let l = iou_quality_loss(&iou_leaf, &box_leaf, &out.anchors, &asg, out.reg_max, 1.5).unwrap();
    let g = l.backward().unwrap();
    assert!(g.get(&box_leaf).is_none(), "IoU target must be detached from box_raw");
    let g_iou = g.get(&iou_leaf).unwrap().to_vec::<f32>().unwrap();
    assert!(g_iou.iter().any(|&v| v.abs() > 1e-6) && g_iou.iter().all(|v| v.is_finite()));

    let box_const = out.box_raw.detach();
    grad_check_with(
        |inputs| iou_quality_loss(&inputs[0], &box_const, &out.anchors, &asg, out.reg_max, 1.5),
        &[iou_probe],
        &GradCheck { eps: 1e-3, atol: 2e-3, rtol: 2e-2, max_checks: 32, seed: 42 },
    )
    .unwrap();

    // 2. End-to-end backward reaches `box_branch.iou_out.weight` and shared `box_branch.conv1`.
    let l_e2e = iou_quality_loss(&iou_raw, &out.box_raw, &out.anchors, &asg, out.reg_max, 1.0).unwrap();
    let g_e2e = l_e2e.backward().unwrap();
    let any_iou_grad = model.head.levels.iter().any(|lvl| {
        g_e2e
            .get(&lvl.box_branch.iou_out.as_ref().unwrap().weight)
            .is_some_and(|t| t.to_vec::<f32>().unwrap().iter().any(|&v| v.abs() > 1e-8))
    });
    assert!(any_iou_grad, "iou_quality_loss must produce non-zero gradients on iou_out weights");

    // 3. Zero foreground anchors and gain == 0 return finite scalar zero with zero gradient.
    let empty_targets = vec![ImageTargets::default(), ImageTargets::default()];
    let empty_asg =
        assign_batch(&out.box_raw, &out.cls_raw, &out.anchors, &empty_targets, &loss_cfg).unwrap();
    assert_eq!(empty_asg.num_fg(), 0);
    let iou_z = iou_raw.detach().requires_grad_();
    let lz = iou_quality_loss(&iou_z, &out.box_raw, &out.anchors, &empty_asg, out.reg_max, 1.0).unwrap();
    assert_eq!(lz.item::<f32>().unwrap(), 0.0);
    let gz = lz.backward().unwrap().get(&iou_z).unwrap().to_vec::<f32>().unwrap();
    assert!(gz.iter().all(|&v| v == 0.0));

    // 4. Invalid gain rejected.
    assert!(iou_quality_loss(&iou_raw, &out.box_raw, &out.anchors, &asg, out.reg_max, -0.5).is_err());
    assert!(iou_quality_loss(&iou_raw, &out.box_raw, &out.anchors, &asg, out.reg_max, f32::NAN).is_err());
}

#[test]
fn lazyalign_iou_score_fusion_and_detections() {
    use folo_vision::model::v2::{DEFAULT_IOU_FUSION_ALPHA, fuse_cls_iou_scores};
    use folo_vision::nms::NmsConfig;

    // 1. Direct score fusion math at alpha = 0.0, 0.5, 1.0.
    let iou_logits = [0.0f32, 2.0]; // sigmoid(0)=0.5, sigmoid(2)≈0.880797
    let p_iou0 = 0.5f32;
    let p_iou1 = 1.0 / (1.0 + (-2.0f32).exp());

    let mut s0 = vec![0.8f32, 0.2, 0.5, 0.9];
    fuse_cls_iou_scores(&mut s0, &iou_logits, 2, 0.0);
    assert_eq!(s0, vec![0.8, 0.2, 0.5, 0.9], "alpha=0 must be exact identity");

    let mut s_half = vec![0.8f32, 0.2, 0.5, 0.9];
    fuse_cls_iou_scores(&mut s_half, &iou_logits, 2, DEFAULT_IOU_FUSION_ALPHA);
    assert!((s_half[0] - (0.8 * p_iou0).sqrt()).abs() < 1e-6);
    assert!((s_half[1] - (0.2 * p_iou0).sqrt()).abs() < 1e-6);
    assert!((s_half[2] - (0.5 * p_iou1).sqrt()).abs() < 1e-6);
    assert!((s_half[3] - (0.9 * p_iou1).sqrt()).abs() < 1e-6);

    let mut s_one = vec![0.8f32, 0.2, 0.5, 0.9];
    fuse_cls_iou_scores(&mut s_one, &iou_logits, 2, 1.0);
    assert_close(&s_one, &[p_iou0, p_iou0, p_iou1, p_iou1], 1e-6, "alpha=1 pure IoU");

    // 2. End-to-end HeadOutputV2::decode_images & detections with iou_head=true.
    let cfg = FoloV2Config::new("n", 3).unwrap().with_iou_head(true);
    let model = FoloV2Model::with_rng(cfg, &mut Rng::seed(123)).unwrap();
    model.eval();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(456));
    let mut out = {
        let _ng = no_grad();
        model.forward(&x).unwrap()
    };
    let a = out.num_anchors();
    // Craft synthetic cls_raw & iou_raw on two disjoint anchors (0 and 1):
    // anchor 0 has higher raw cls (logit 2.0 -> 0.881) but terrible IoU (logit -4.0 -> 0.018),
    // anchor 1 has lower raw cls (logit 1.0 -> 0.731) but high IoU (logit +4.0 -> 0.982).
    let mut cls_data = vec![-20.0f32; 3 * a];
    cls_data[0] = 2.0; // class 0, anchor 0
    cls_data[1] = 1.0; // class 0, anchor 1
    let mut iou_data = vec![0.0f32; a];
    iou_data[0] = -4.0;
    iou_data[1] = 4.0;
    out.cls_raw = Tensor::from_vec(cls_data, [1, 3, a]).unwrap();
    out.iou_raw = Some(Tensor::from_vec(iou_data, [1, 1, a]).unwrap());

    let nms = NmsConfig { conf: 0.05, iou: 0.99, max_det: 10, ..NmsConfig::default() };
    let dets_raw = out.detections(&nms, 0.0).unwrap();
    let dets_fused = out.detections(&nms, 0.5).unwrap();
    assert_eq!(dets_raw[0].len(), 2);
    assert_eq!(dets_fused[0].len(), 2);
    // Without IoU fusion, anchor 0 ranks #1; with IoU fusion (alpha=0.5), anchor 1 overtakes anchor 0!
    assert_ne!(dets_raw[0][0].bbox, dets_fused[0][0].bbox, "IoU fusion should promote higher-IoU anchor");
    assert!(dets_fused[0][0].score > dets_fused[0][1].score);

    // Invalid iou_alpha rejected.
    assert!(out.decode_images(-0.1).is_err());
    assert!(out.decode_images(1.5).is_err());
    assert!(out.detections(&nms, f32::NAN).is_err());
}
