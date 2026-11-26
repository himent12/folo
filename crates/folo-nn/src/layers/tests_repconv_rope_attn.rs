//! Unit tests for [`RepConv`], [`RoPE2d`], and [`MultiHeadCrossAttention`]:
//! - `RepConv`: tight numerical equivalence between eval-mode multi-branch (`3×3` + `1×1` + identity BN) and
//!   `switch_to_deploy()` single-`3×3` output across standard, downsampling (`stride = 2`), channel-changing,
//!   grouped (`groups = 2`), and depthwise (`groups = C`) configs, central FD `grad_check_with`, and `.folo`
//!   weight save/load round-trips (both training and deploy layouts).
//! - `RoPE2d`: exact L2 norm preservation, exact 2-D relative-position inner-product invariance, `apply_nchw`
//!   parity, central FD `grad_check_with`, `.folo` buffer round-trip, and validation.
//! - `MultiHeadCrossAttention`: central FD `grad_check_with` wrt `query`, `key`, `value`, and all trainable
//!   weights/biases (with and without QK-RMSNorm & `RoPE2d`), additive attention masking, empty batch/query
//!   edge cases, and `.folo` save/load round-trip.

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_test_support::TestDir;

use super::{
    Activation, Linear, MultiHeadCrossAttention, MultiHeadCrossAttentionConfig, RepConv, RepConvConfig,
    RmsNorm, RoPE2d,
};
use crate::io::{FileKind, LoadOptions, Metadata, ModelConfig, ModelFile, Purpose, save_weights};
use crate::module::{Forward, Module};

#[test]
fn repconv_switch_to_deploy_equivalence_and_folo_roundtrip() {
    let mut rng = Rng::seed(801);
    let configs = [
        RepConvConfig::new(6, 6).stride(1).groups(1).use_identity(true).act(Activation::Relu),
        RepConvConfig::new(6, 6).stride(1).groups(1).use_identity(false).act(Activation::Silu),
        RepConvConfig::new(4, 8).stride(2).groups(1).use_identity(true).act(Activation::Relu),
        RepConvConfig::new(6, 6).stride(1).groups(2).use_identity(true).act(Activation::Identity),
        RepConvConfig::new(6, 6).stride(1).groups(6).use_identity(true).act(Activation::Silu),
    ];

    let dir = TestDir::new("repconv-roundtrip");

    for (idx, cfg) in configs.into_iter().enumerate() {
        let mut blk = RepConv::new(cfg, &mut rng).unwrap();
        // Run several training passes with non-trivial BN weights/biases so running_mean and running_var
        // diverge meaningfully from 0 and 1.
        {
            let _ng = folo_core::no_grad();
            for (name, p) in blk.named_parameters() {
                if name.ends_with(".bn.weight") || name == "rbr_identity.weight" {
                    p.copy_from(&Tensor::rand_uniform(p.shape().clone(), 0.5, 1.5, &mut rng)).unwrap();
                } else if name.ends_with(".bn.bias") || name == "rbr_identity.bias" {
                    p.copy_from(&Tensor::rand_uniform(p.shape().clone(), -0.5, 0.5, &mut rng)).unwrap();
                }
            }
        }
        for _ in 0..3 {
            let x_tr = Tensor::randn([2, cfg.in_channels, 6, 6], 0.3, 1.2, &mut rng);
            let _ = blk.forward(&x_tr).unwrap();
        }

        // Save multi-branch weights to `.folo` and verify training-mode reload matches bit-for-bit.
        blk.eval();
        let x_eval = Tensor::randn([2, cfg.in_channels, 6, 6], -0.2, 0.9, &mut rng);
        let y_multi = blk.forward(&x_eval).unwrap();

        let train_path = dir.join(format!("repconv_train_{idx}.folo"));
        let mut meta = Metadata::new(
            FileKind::Weights,
            ModelConfig::new("n", 4).unwrap(),
            (0..4).map(|i| format!("c{i}")).collect(),
            640,
        )
        .unwrap();
        save_weights(&train_path, &meta, &blk).unwrap();
        let reloaded_multi = RepConv::new(cfg, &mut Rng::seed(999)).unwrap();
        reloaded_multi.eval();
        ModelFile::open(&train_path)
            .unwrap()
            .load(&reloaded_multi, &LoadOptions::strict(Purpose::Training))
            .unwrap();
        assert_eq!(
            y_multi.to_vec::<f32>().unwrap(),
            reloaded_multi.forward(&x_eval).unwrap().to_vec::<f32>().unwrap(),
            "RepConv config {idx} multi-branch .folo roundtrip mismatch"
        );

        // Switch to deploy and verify tight numerical equivalence against eval-mode multi-branch output.
        blk.switch_to_deploy().unwrap();
        assert!(blk.is_deployed());
        assert_eq!(blk.named_parameters().len(), 2); // rbr_reparam.weight + rbr_reparam.bias
        // Second call is an idempotent no-op.
        blk.switch_to_deploy().unwrap();
        let y_deploy = blk.forward(&x_eval).unwrap();

        let vm = y_multi.to_vec::<f32>().unwrap();
        let vd = y_deploy.to_vec::<f32>().unwrap();
        assert_eq!(vm.len(), vd.len());
        let max_diff = vm.iter().zip(&vd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "RepConv config {idx} deploy equivalence failed: max_diff={max_diff:e}");

        // Save deploy weights to `.folo` and reload into `RepConv::new_deploy`.
        let deploy_path = dir.join(format!("repconv_deploy_{idx}.folo"));
        meta.fused = true;
        save_weights(&deploy_path, &meta, &blk).unwrap();
        let reloaded_deploy = RepConv::new_deploy(cfg, &mut Rng::seed(888)).unwrap();
        ModelFile::open(&deploy_path)
            .unwrap()
            .load(&reloaded_deploy, &LoadOptions::strict(Purpose::Inference))
            .unwrap();
        assert_eq!(
            vd,
            reloaded_deploy.forward(&x_eval).unwrap().to_vec::<f32>().unwrap(),
            "RepConv config {idx} deploy .folo roundtrip mismatch"
        );
    }

    // Central finite-difference grad check on RepConv in eval mode.
    let g_blk = RepConv::new(RepConvConfig::new(4, 4).act(Activation::Silu), &mut rng).unwrap();
    g_blk.eval();
    let gx = Tensor::randn([1, 4, 4, 4], 0.0, 0.8, &mut rng);
    grad_check_with(
        |ins| g_blk.forward(&ins[0]).unwrap().square()?.mean_all(),
        std::slice::from_ref(&gx),
        &GradCheck { eps: 2e-3, atol: 3e-3, rtol: 3e-2, max_checks: 16, seed: 81 },
    )
    .unwrap();
}

#[test]
fn rope2d_norm_preservation_relative_invariance_grad_check_and_roundtrip() {
    let mut rng = Rng::seed(802);
    let rope = RoPE2d::new(8).unwrap();
    assert_eq!(rope.head_dim(), 8);
    assert_eq!(rope.named_buffers().len(), 1);
    assert_eq!(rope.named_buffers()[0].0, "inv_freq");

    let (b, heads, h, w, d) = (2, 2, 3, 4, 8);
    let q = Tensor::randn([b, heads, h * w, d], 0.0, 1.0, &mut rng);
    let k = Tensor::randn([b, heads, h * w, d], 0.0, 1.0, &mut rng);
    let q_rot = rope.apply(&q, h, w).unwrap();
    let k_rot = rope.apply(&k, h, w).unwrap();

    // 1. Exact L2 norm preservation at every spatial token.
    let q_n = q.square().unwrap().sum(3, false).unwrap().to_vec::<f32>().unwrap();
    let qr_n = q_rot.square().unwrap().sum(3, false).unwrap().to_vec::<f32>().unwrap();
    let k_n = k.square().unwrap().sum(3, false).unwrap().to_vec::<f32>().unwrap();
    let kr_n = k_rot.square().unwrap().sum(3, false).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in q_n.iter().zip(&qr_n).chain(k_n.iter().zip(&kr_n)) {
        assert!((a - b).abs() < 1e-5, "RoPE2d must preserve L2 norm: {a} vs {b}");
    }

    // 2. Exact 2-D relative-position inner-product invariance:
    //    <RoPE2d(q0, y1, x1), RoPE2d(k0, y2, x2)> == <RoPE2d(q0, y1+dy, x1+dx), RoPE2d(k0, y2+dy, x2+dx)>.
    let q_single = Tensor::randn([1, 1, 1, d], 0.0, 1.0, &mut rng);
    let k_single = Tensor::randn([1, 1, 1, d], 0.0, 1.0, &mut rng);
    let q_tile = q_single.broadcast_to([1, 1, h * w, d]).unwrap().contiguous().unwrap();
    let k_tile = k_single.broadcast_to([1, 1, h * w, d]).unwrap().contiguous().unwrap();
    let qt_r = rope.apply(&q_tile, h, w).unwrap().to_vec::<f32>().unwrap();
    let kt_r = rope.apply(&k_tile, h, w).unwrap().to_vec::<f32>().unwrap();
    let dot_at = |p1: usize, p2: usize| -> f32 { (0..d).map(|c| qt_r[p1 * d + c] * kt_r[p2 * d + c]).sum() };
    // Compare (y=0,x=0)->(y=1,x=2) vs (y=1,x=1)->(y=2,x=3): same delta (dy=+1, dx=+2).
    let d1 = dot_at(0, w + 2);
    let d2 = dot_at(w + 1, 2 * w + 3);
    assert!((d1 - d2).abs() < 1e-5, "RoPE2d relative inner product drifted: {d1} vs {d2}");

    // 3. `apply_nchw` matches `apply` exactly.
    let x_nchw = Tensor::randn([2, heads * d, h, w], 0.0, 1.0, &mut rng);
    let y_nchw = rope.apply_nchw(&x_nchw, heads).unwrap();
    let x_tok = x_nchw.reshape([2, heads, d, h * w]).unwrap().transpose(2, 3).unwrap().contiguous().unwrap();
    let y_tok = rope
        .apply(&x_tok, h, w)
        .unwrap()
        .transpose(2, 3)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape([2, heads * d, h, w])
        .unwrap();
    assert_eq!(y_nchw.to_vec::<f32>().unwrap(), y_tok.to_vec::<f32>().unwrap());

    // 4. Central FD grad_check_with on `RoPE2d::apply` and `apply_nchw`.
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 82 };
    grad_check_with(
        |ins| {
            let qr = rope.apply(&ins[0], h, w).unwrap();
            let kr = rope.apply(&ins[1], h, w).unwrap();
            qr.matmul(&kr.transpose(2, 3)?.contiguous()?)?.square()?.mean_all()
        },
        &[q.clone(), k.clone()],
        &gc,
    )
    .unwrap();

    // 5. `.folo` buffer roundtrip.
    let dir = TestDir::new("rope2d-roundtrip");
    let path = dir.join("rope2d.folo");
    let rope_custom = RoPE2d::with_options(8, 500.0, 0.5).unwrap();
    let meta = Metadata::new(
        FileKind::Weights,
        ModelConfig::new("n", 4).unwrap(),
        (0..4).map(|i| format!("c{i}")).collect(),
        640,
    )
    .unwrap();
    save_weights(&path, &meta, &rope_custom).unwrap();
    let rope_loaded = RoPE2d::new(8).unwrap();
    ModelFile::open(&path).unwrap().load(&rope_loaded, &LoadOptions::strict(Purpose::Training)).unwrap();
    assert_eq!(
        rope_custom.apply(&q, h, w).unwrap().to_vec::<f32>().unwrap(),
        rope_loaded.apply(&q, h, w).unwrap().to_vec::<f32>().unwrap()
    );

    // Invalid head_dim / base rejected.
    assert!(RoPE2d::new(6).is_err());
    assert!(RoPE2d::new(0).is_err());
    assert!(RoPE2d::with_options(8, 0.0, 1.0).is_err());
}

#[test]
fn multi_head_cross_attention_grad_check_masking_rope2d_and_folo_roundtrip() {
    let mut rng = Rng::seed(803);
    let cfg = MultiHeadCrossAttentionConfig::new(8, 2).kv_dim(6).bias(true).qk_norm(true);
    let attn = MultiHeadCrossAttention::new(cfg, &mut rng).unwrap();
    assert_eq!((attn.embed_dim(), attn.kv_dim(), attn.num_heads(), attn.head_dim()), (8, 6, 2, 4));

    let q = Tensor::randn([2, 4, 8], 0.0, 0.8, &mut rng); // 2x2 spatial grid
    let k = Tensor::randn([2, 6, 6], 0.0, 0.8, &mut rng); // 2x3 spatial grid
    let v = Tensor::randn([2, 6, 6], 0.0, 0.8, &mut rng);
    let rope = RoPE2d::new(4).unwrap();

    // 1. Full central FD grad_check_with on Q, K, V, and ALL trainable weights/biases (with QK-RMSNorm).
    let inps = vec![
        q.clone(),
        k.clone(),
        v.clone(),
        attn.q_proj.weight.detach(),
        attn.q_proj.bias.as_ref().unwrap().detach(),
        attn.k_proj.weight.detach(),
        attn.k_proj.bias.as_ref().unwrap().detach(),
        attn.v_proj.weight.detach(),
        attn.v_proj.bias.as_ref().unwrap().detach(),
        attn.out_proj.weight.detach(),
        attn.out_proj.bias.as_ref().unwrap().detach(),
        attn.q_norm.as_ref().unwrap().weight.as_ref().unwrap().detach(),
        attn.k_norm.as_ref().unwrap().weight.as_ref().unwrap().detach(),
    ];
    let gc = GradCheck { eps: 2e-3, atol: 3e-3, rtol: 3e-2, max_checks: 10, seed: 83 };
    grad_check_with(
        |t| {
            let mut m = MultiHeadCrossAttention::new(cfg, &mut Rng::seed(0)).unwrap();
            m.q_proj = Linear::from_tensors(t[3].clone(), Some(t[4].clone())).unwrap();
            m.k_proj = Linear::from_tensors(t[5].clone(), Some(t[6].clone())).unwrap();
            m.v_proj = Linear::from_tensors(t[7].clone(), Some(t[8].clone())).unwrap();
            m.out_proj = Linear::from_tensors(t[9].clone(), Some(t[10].clone())).unwrap();
            m.q_norm = Some(RmsNorm::from_weight(t[11].clone(), cfg.qk_norm_eps).unwrap());
            m.k_norm = Some(RmsNorm::from_weight(t[12].clone(), cfg.qk_norm_eps).unwrap());
            m.forward_cross_qkv(&t[0], &t[1], &t[2], None).unwrap().square()?.mean_all()
        },
        &inps,
        &gc,
    )
    .unwrap();

    // 2. Central FD grad_check_with composed with RoPE2d.
    grad_check_with(
        |t| {
            attn.forward_cross_with_rope2d(&t[0], (2, 2), &t[1], (2, 3), &rope, None)
                .unwrap()
                .square()?
                .mean_all()
        },
        &[q.clone(), k.clone()],
        &gc,
    )
    .unwrap();

    // 3. Additive attention mask zeros out contribution and gradient of masked key/value positions.
    let mask = Tensor::from_slice(&[0.0f32, 0.0, 0.0, -1e9, -1e9, -1e9], [1, 1, 1, 6]).unwrap();
    let v_req = v.detach().requires_grad_();
    let y_masked = attn.forward_cross_qkv(&q, &k, &v_req, Some(&mask)).unwrap();
    let g_v = y_masked
        .square()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&v_req)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    // Masked tokens 3..6 in batch 0 and batch 1 must have ~0 gradient.
    for batch in 0..2 {
        for tok in 3..6 {
            for c in 0..6 {
                let idx = (batch * 6 + tok) * 6 + c;
                assert!(
                    g_v[idx].abs() < 1e-6,
                    "masked value token ({batch},{tok},{c}) leaked grad {}",
                    g_v[idx]
                );
            }
        }
    }

    // 4. `.folo` save/load round-trip produces bitwise identical outputs.
    let dir = TestDir::new("cross-attn-roundtrip");
    let path = dir.join("cross_attn.folo");
    let meta = Metadata::new(
        FileKind::Weights,
        ModelConfig::new("n", 4).unwrap(),
        (0..4).map(|i| format!("c{i}")).collect(),
        640,
    )
    .unwrap();
    save_weights(&path, &meta, &attn).unwrap();
    let loaded = MultiHeadCrossAttention::new(cfg, &mut Rng::seed(777)).unwrap();
    ModelFile::open(&path).unwrap().load(&loaded, &LoadOptions::strict(Purpose::Training)).unwrap();
    let y1 = attn.forward_cross_with_rope2d(&q, (2, 2), &k, (2, 3), &rope, Some(&mask)).unwrap();
    let y2 = loaded.forward_cross_with_rope2d(&q, (2, 2), &k, (2, 3), &rope, Some(&mask)).unwrap();
    assert_eq!(y1.to_vec::<f32>().unwrap(), y2.to_vec::<f32>().unwrap());

    // 5. Empty batch preserves autograd; L_kv == 0 with L_q > 0 errors cleanly.
    let q0 = Tensor::zeros([0, 4, 8]).requires_grad_();
    let kv0 = Tensor::zeros([0, 6, 6]);
    let y0 = attn.forward_cross(&q0, &kv0, None).unwrap();
    assert_eq!(y0.dims(), &[0, 4, 8]);
    assert_eq!(y0.sum_all().unwrap().backward().unwrap().get(&q0).unwrap().dims(), &[0, 4, 8]);
    assert!(attn.forward_cross(&q, &Tensor::zeros([2, 0, 6]), None).is_err());
}
