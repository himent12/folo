//! Candidate-first prediction on the GPU (docs/MODEL.md §9.5) against the dense GPU path, bit for bit: every
//! detection of `predict_candidates` + `finish_candidates` equals the dense forward's `detections` (with NMS at
//! IoU 1 and no cap, so every candidate is compared), in f32, in f16 on the regular kernels and on tensor cores,
//! for `o2m` and `o2o` heads, at thresholds from none passing to every anchor passing (more candidates than the
//! site lists hold, so the dense head serves those images). With DFL box fusion (MODEL.md §9.4) also every
//! candidate's side precisions against the dense `HeadOutput::side_precisions`, bit for bit, and the fused
//! detections.
//!
//! This binary asks for tensor cores before the shared GPU context exists and runs the f16 cases with them on
//! and off (on only where the adapter has them). It fills every buffer the GPU backend allocates with NaN
//! (`folo_gpu::poison_uninit`), so a site value can only match by being computed: the dense forward that runs first
//! would otherwise leave its maps in the pool for the sites path to reuse.

use std::sync::{Mutex, MutexGuard};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::ModelFile;
use folo_test_support::TestFile;
use folo_vision::fusion::{BoxFusion, ScoreRule};
use folo_vision::model::{FoloModel, HeadMode, ImageCandidates, ModelConfig};
use folo_vision::nms::{Detection, NmsConfig};

/// The GPU device and its context (tensor cores requested; a lock, since the tests flip the switch), or `None`.
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

/// A folo-n detector with `nc` classes and `head` (`o2m` or `o2o`), with non-trivial BatchNorm statistics and
/// constant `cls_bias` class biases, loaded fused for inference on `dev` in `dtype`.
fn model_with_bias(nc: usize, head: &str, dev: Device, dtype: DType, cls_bias: f32) -> FoloModel {
    let mut c = ModelConfig::new("n", nc).unwrap();
    c.head = head.into();
    let m = FoloModel::new(c).unwrap();
    let rng = &mut Rng::seed(3);
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            let n = t.tensor.numel();
            if t.name.ends_with("running_mean") {
                t.tensor.copy_from(&Tensor::rand_uniform([n], -0.2, 0.2, rng)).unwrap();
            } else if t.name.ends_with("running_var") {
                t.tensor.copy_from(&Tensor::rand_uniform([n], 0.5, 1.5, rng)).unwrap();
            } else if t.name.contains("cls.") && t.name.ends_with(".out.bias") {
                let b = t.tensor.zeros_like().unwrap().add_scalar(cls_bias).unwrap();
                t.tensor.copy_from(&b).unwrap();
            }
        }
    }
    let path = TestFile::new(&format!("lazy-gpu-{head}-{nc}"), "m.folo");
    m.save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    FoloModel::from_file_for_inference(&file, dev, dtype).unwrap()
}

fn model(nc: usize, head: &str, dev: Device, dtype: DType) -> FoloModel {
    model_with_bias(nc, head, dev, dtype, 0.0)
}

fn bits(d: &[Detection]) -> Vec<[u32; 6]> {
    d.iter()
        .map(|d| {
            let b = d.bbox;
            [d.class, d.score.to_bits(), b.x1.to_bits(), b.y1.to_bits(), b.x2.to_bits(), b.y2.to_bits()]
        })
        .collect()
}

/// With DFL fusion on an `o2m` head, each candidate's side precisions in `got` against the dense ones (`dense`,
/// per image per anchor), bit for bit; otherwise that `got` carries none.
fn check_precisions(
    m: &FoloModel,
    got: &[ImageCandidates],
    dense: &[Vec<[f32; 4]>],
    nms: &NmsConfig,
    what: &str,
) {
    let dfl = nms.fusion == BoxFusion::Dfl && m.head_mode() == HeadMode::O2m;
    for (i, c) in got.iter().enumerate() {
        if !dfl {
            assert_eq!(c.precisions, None, "{what}: image {i}");
            continue;
        }
        let p = c.precisions.as_ref().unwrap_or_else(|| panic!("{what}: image {i} has no precisions"));
        assert_eq!(p.len(), c.cands.len(), "{what}: image {i}");
        for (c, p) in c.cands.iter().zip(p) {
            let want = dense[i][c.anchor as usize];
            assert_eq!(p.map(f32::to_bits), want.map(f32::to_bits), "{what}: image {i}, anchor {}", c.anchor);
        }
    }
}

/// `on` is `off` with the consensus score: the same boxes and classes, each score at most `off`'s and at least
/// `off`'s over `√10`. Returns how many scores it lowered.
fn lowered_by_consensus(on: &[Detection], off: &[Detection]) -> usize {
    let key = |d: &Detection| {
        let b = d.bbox;
        ([d.class, b.x1.to_bits(), b.y1.to_bits(), b.x2.to_bits(), b.y2.to_bits()], d.score.to_bits())
    };
    let (mut a, mut b): (Vec<_>, Vec<_>) = (on.iter().map(key).collect(), off.iter().map(key).collect());
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a.len(), b.len(), "the same number of detections");
    let mut lowered = 0;
    for ((ka, sa), (kb, sb)) in a.iter().zip(&b) {
        assert_eq!(ka, kb, "the same box and class");
        let (s, was) = (f32::from_bits(*sa), f32::from_bits(*sb));
        assert!(s <= was && f64::from(s) >= f64::from(was) / 10f64.sqrt() * (1.0 - 1e-6), "{s} from {was}");
        lowered += usize::from(s < was);
    }
    lowered
}

/// Candidate-first against dense on `x` with the box fusion `fusion` and the score rule `score`, at thresholds picked
/// from the dense scores: none passing, the top 1%, 5% and 30% of all anchors, and every anchor. With the consensus
/// score, also that it changed scores and nothing else. Returns how many detections were compared.
fn check(m: &FoloModel, x: &Tensor, what: &str, fusion: BoxFusion, score: ScoreRule) -> usize {
    let _g = no_grad();
    let dense = m.forward(x).unwrap();
    let (b, nc, a) = (dense.batch(), dense.num_classes, dense.num_anchors());
    let cls = dense.cls_raw.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let mut best: Vec<f32> = (0..b * a)
        .map(|i| {
            let (img, j) = (i / a, i % a);
            (0..nc).map(|c| cls[(img * nc + c) * a + j]).fold(f32::NEG_INFINITY, f32::max)
        })
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    best.sort_by(f32::total_cmp);
    let at = |q: f64| best[((best.len() as f64 * q) as usize).min(best.len() - 1)];
    let (mut compared, mut changed) = (0, 0);
    let mut confs = vec![best[best.len() - 1] + 1e-3, at(0.99), at(0.95), at(0.7), best[0] - 1e-3];
    if score.is_calibrated() {
        // Singleton anchors around 0.50 shift to ~0.29 under ELC and ~0.097 under DflConsensus (untrained wide
        // DFL bins), so include both 0.15 (exercises `post_filter_calibrated`) and 0.05 (non-empty for both).
        confs.push(0.15);
        confs.push(0.05);
    }
    for conf in confs {
        let nms = NmsConfig {
            conf,
            iou: 1.0,
            max_det: usize::MAX,
            max_candidates: usize::MAX,
            fusion,
            score,
            ..NmsConfig::default()
        };
        assert!(m.candidates_first_applies(x.device(), &nms), "{what}: candidate-first applies");
        let want = dense.detections(&nms).unwrap();
        let cands = m.predict_candidates(x, &nms).unwrap();
        if fusion == BoxFusion::Dfl {
            check_precisions(
                m,
                &cands,
                &dense.side_precisions().unwrap(),
                &nms,
                &format!("{what} at conf {conf}"),
            );
        }
        let got: Vec<Vec<Detection>> = cands.iter().map(|c| m.finish_candidates(c, &nms)).collect();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(bits(g), bits(w), "{what}: image {i} at conf {conf}");
            compared += w.len();
        }
        match score {
            ScoreRule::None => {}
            ScoreRule::Consensus => {
                let off = dense.detections(&NmsConfig { score: ScoreRule::None, ..nms.clone() }).unwrap();
                changed += got.iter().zip(&off).map(|(on, off)| lowered_by_consensus(on, off)).sum::<usize>();
            }
            ScoreRule::Elc | ScoreRule::DflConsensus | ScoreRule::DflRepulse => {
                let off = dense.detections(&NmsConfig { score: ScoreRule::None, ..nms.clone() }).unwrap();
                changed += got.iter().zip(&off).filter(|(on, off)| bits(on) != bits(off)).count();
            }
        }
    }
    assert!(score == ScoreRule::None || changed > 0, "{what}: {score:?} changed nothing");
    compared
}

fn input(dims: [usize; 4], seed: u64, dev: Device, dtype: DType) -> Tensor {
    let x = Tensor::rand_uniform(dims, 0.0, 1.0, &mut Rng::seed(seed)).to_device(dev).unwrap();
    x.to_dtype(dtype).unwrap()
}

#[test]
fn gpu_candidates_first_matches_dense_bit_for_bit() {
    let Some((dev, ctx, _g)) = gpu() else { return };
    let tc = ctx.caps().coop_f16;
    let mut dtypes = vec![(DType::F32, false)];
    if ctx.caps().shader_f16 {
        dtypes.push((DType::F16, false));
        if tc {
            dtypes.push((DType::F16, true));
        }
    }
    for head in ["o2m", "o2o"] {
        for &(dtype, on) in &dtypes {
            let m = model(3, head, dev, dtype);
            ctx.set_tensor_cores(on);
            // 320²: P3 has more pixels than its site list, so "every anchor" overflows; 384×640 is rectangular.
            for (dims, seed) in [([2, 3, 320, 320], 1), ([1, 3, 384, 640], 2), ([3, 3, 128, 96], 3)] {
                let what = format!("{head} {dtype:?} tensor cores {on} {dims:?}");
                let n = check(&m, &input(dims, seed, dev, dtype), &what, BoxFusion::None, ScoreRule::None);
                assert!(n > 0, "{what}: nothing compared");
            }
            assert_eq!(ctx.tensor_cores(), on, "{}", ctx.tensor_cores_status());
        }
    }
    ctx.set_tensor_cores(true);
}

/// Box fusion: score voting and DFL evidence fusion give the dense GPU path's detections bit for bit, DFL fusion
/// from side precisions that are the dense ones bit for bit (the site columns' moments, and the dense head's for
/// an image that overflows), in f32 and f16, on the regular kernels and on tensor cores.
#[test]
fn gpu_box_fusion_matches_dense_bit_for_bit() {
    let Some((dev, ctx, _g)) = gpu() else { return };
    let mut dtypes = vec![(DType::F32, false)];
    if ctx.caps().shader_f16 {
        dtypes.push((DType::F16, false));
        if ctx.caps().coop_f16 {
            dtypes.push((DType::F16, true));
        }
    }
    for &(dtype, on) in &dtypes {
        let m = model(3, "o2m", dev, dtype);
        ctx.set_tensor_cores(on);
        for fusion in [BoxFusion::Vote, BoxFusion::Dfl] {
            // 320²: every anchor passing overflows P3's site list; 384×640 is rectangular.
            for (dims, seed) in [([2, 3, 320, 320], 4), ([1, 3, 384, 640], 5)] {
                let what = format!("{fusion:?} {dtype:?} tensor cores {on} {dims:?}");
                let n = check(&m, &input(dims, seed, dev, dtype), &what, fusion, ScoreRule::None);
                assert!(n > 0, "{what}: nothing compared");
                // The consensus / ELC / DFL-consensus / DFL-repulse rules run on the host after NMS.
                for rule in
                    [ScoreRule::Consensus, ScoreRule::Elc, ScoreRule::DflConsensus, ScoreRule::DflRepulse]
                {
                    let n = check(&m, &input(dims, seed, dev, dtype), &what, fusion, rule);
                    assert!(n > 0, "{what}, {rule:?}: nothing compared");
                }
            }
        }
        // Also test `ScoreRule::DflRepulse` without box fusion (entropy-only `[B, 1, Σ cap]` GPU stats path).
        let n = check(
            &m,
            &input([1, 3, 384, 640], 5, dev, dtype),
            "None+DflRepulse",
            BoxFusion::None,
            ScoreRule::DflRepulse,
        );
        assert!(n > 0, "None+DflRepulse: nothing compared");
        // The fusion moves boxes, so the comparison above sees the precisions.
        let x = input([1, 3, 384, 640], 5, dev, dtype);
        let _g = no_grad();
        let out = m.forward(&x).unwrap();
        let nms = |fusion| NmsConfig { conf: 0.3, fusion, ..NmsConfig::default() };
        let (dfl, vote) =
            (out.detections(&nms(BoxFusion::Dfl)).unwrap(), out.detections(&nms(BoxFusion::Vote)).unwrap());
        assert_ne!(bits(&dfl[0]), bits(&vote[0]), "{dtype:?}: DFL fusion moved no box past voting");
        // Exercise calibrated rules with real NMS suppression on GPU candidates-first.
        for (rule, iou) in
            [(ScoreRule::Elc, 0.70), (ScoreRule::DflConsensus, 0.70), (ScoreRule::DflRepulse, 0.75)]
        {
            let cfg =
                NmsConfig { conf: 0.3, iou, fusion: BoxFusion::Dfl, score: rule, ..NmsConfig::default() };
            let want = out.detections(&cfg).unwrap();
            let cands = m.predict_candidates(&x, &cfg).unwrap();
            let got: Vec<Vec<Detection>> = cands.iter().map(|c| m.finish_candidates(c, &cfg)).collect();
            assert_eq!(bits(&got[0]), bits(&want[0]), "{dtype:?} tensor cores {on}: {rule:?} at iou={iou}");
        }
        assert_eq!(ctx.tensor_cores(), on, "{}", ctx.tensor_cores_status());
    }
    ctx.set_tensor_cores(true);
}

/// One batch in which one image overflows P3's site list (more than 1024 candidates at 320²) and the other fits:
/// the overflowing image takes the dense head's candidates, the other its own, each bit for bit the dense path.
#[test]
fn a_batch_mixing_overflow_and_fit_matches_dense_per_image() {
    let Some((dev, ctx, _g)) = gpu() else { return };
    ctx.set_tensor_cores(false);
    let _g = no_grad();
    let m = model(3, "o2m", dev, DType::F32);
    // A high-contrast random image and a flat grey one: this random-init model's scores hardly depend on an image
    // in [0, 1] (about 66 distinct P3 scores), so the contrast makes the two maps differ.
    let a = Tensor::rand_uniform([1, 3, 320, 320], -40.0, 40.0, &mut Rng::seed(11));
    let b = a.zeros_like().unwrap().add_scalar(0.5).unwrap();
    let x = Tensor::cat(&[&a, &b], 0).unwrap().to_device(dev).unwrap();
    let dense = m.forward(&x).unwrap();
    let (nc, an) = (dense.num_classes, dense.num_anchors());
    let p3 = 40 * 40;
    let cls = dense.cls_raw.to_vec::<f32>().unwrap();
    // Each image's 1025th best P3 score: a threshold below it overflows that image's P3 list.
    let kth: Vec<f32> = (0..2)
        .map(|img| {
            let mut best: Vec<f32> = (0..p3)
                .map(|j| (0..nc).map(|c| cls[(img * nc + c) * an + j]).fold(f32::NEG_INFINITY, f32::max))
                .map(|l| 1.0 / (1.0 + (-l).exp()))
                .collect();
            best.sort_by(|u, v| v.total_cmp(u));
            best[1024]
        })
        .collect();
    let conf = (kth[0] + kth[1]) / 2.0;
    let nms =
        NmsConfig { conf, iou: 1.0, max_det: usize::MAX, max_candidates: usize::MAX, ..NmsConfig::default() };
    let got = m.predict_candidates(&x, &nms).unwrap();
    let in_p3 = |img: usize| got[img].cands.iter().filter(|c| (c.anchor as usize) < p3).count();
    let (over, fit) = if kth[0] > kth[1] { (0, 1) } else { (1, 0) };
    assert!(
        in_p3(over) > 1024 && in_p3(fit) <= 1024,
        "P3 candidates {} and {} ({kth:?})",
        in_p3(0),
        in_p3(1)
    );
    let want = dense.detections(&nms).unwrap();
    for img in 0..2 {
        assert_eq!(bits(&m.finish_candidates(&got[img], &nms)), bits(&want[img]), "image {img}");
    }
    // DFL fusion + DFL-consensus scoring on a genuine mixed fit/overflow batch (including bin_precision mapping).
    // Bias class logits to -2.6 so the 1025th P3 scores straddle below `CALIBRATED_PRE_CONF = 0.10` (ensuring
    // `candidate_conf == conf`), so one image overflows P3 (>1024) and the other fits (<=1024) on the GPU sites path.
    let m_low = model_with_bias(3, "o2m", dev, DType::F32, -2.6);
    let dense_low = m_low.forward(&x).unwrap();
    let cls_low = dense_low.cls_raw.to_vec::<f32>().unwrap();
    let kth_low: Vec<f32> = (0..2)
        .map(|img| {
            let mut best: Vec<f32> = (0..p3)
                .map(|j| (0..nc).map(|c| cls_low[(img * nc + c) * an + j]).fold(f32::NEG_INFINITY, f32::max))
                .map(|l| 1.0 / (1.0 + (-l).exp()))
                .collect();
            best.sort_by(|u, v| v.total_cmp(u));
            best[1024]
        })
        .collect();
    let conf_low = (kth_low[0] + kth_low[1]) / 2.0;
    assert!(conf_low <= 0.10, "expected low-bias threshold <= 0.10, got {conf_low} ({kth_low:?})");
    let nms_low =
        NmsConfig { conf: conf_low, fusion: BoxFusion::Dfl, score: ScoreRule::DflConsensus, ..nms.clone() };
    let got_low = m_low.predict_candidates(&x, &nms_low).unwrap();
    let in_p3_low = |img: usize| got_low[img].cands.iter().filter(|c| (c.anchor as usize) < p3).count();
    let (over_low, fit_low) = if kth_low[0] > kth_low[1] { (0, 1) } else { (1, 0) };
    assert!(
        in_p3_low(over_low) > 1024 && in_p3_low(fit_low) <= 1024,
        "DflConsensus P3 candidates {} and {} ({kth_low:?})",
        in_p3_low(0),
        in_p3_low(1)
    );
    let dense_low_side = dense_low.side_precisions().unwrap();
    check_precisions(&m_low, &got_low, &dense_low_side, &nms_low, "mixed batch, DFL-consensus");
    let raw_box = dense_low.box_raw.to_vec::<f32>().unwrap();
    let rm = dense_low.reg_max;
    for (img, ic) in got_low.iter().enumerate() {
        let bp = ic.bin_precisions.as_ref().expect("mixed batch bin_precisions");
        assert_eq!(bp.len(), ic.cands.len());
        let rg = &raw_box[img * 4 * rm * an..(img + 1) * 4 * rm * an];
        for (c, &p) in ic.cands.iter().zip(bp) {
            let a_idx = c.anchor as usize;
            let row: Vec<f32> = (0..4 * rm).map(|k| rg[k * an + a_idx]).collect();
            let want_bp = folo_vision::decode::bin_precisions(&row, rm);
            for side in 0..4 {
                assert!(
                    (p[side] - want_bp[side]).abs() <= 1e-3 * want_bp[side].max(1.0),
                    "mixed batch bin_precision mismatch img={img} anchor={a_idx} side={side}: {} vs {}",
                    p[side],
                    want_bp[side]
                );
            }
        }
    }
    let want_low = dense_low.detections(&nms_low).unwrap();
    for img in 0..2 {
        assert_eq!(
            bits(&m_low.finish_candidates(&got_low[img], &nms_low)),
            bits(&want_low[img]),
            "DFL-consensus mixed fit/overflow: image {img}"
        );
    }
    // Also check single-label post-filtered (`conf > 0.10`) DFL-consensus parity on the zero-bias model.
    let nms = NmsConfig { fusion: BoxFusion::Dfl, score: ScoreRule::DflConsensus, ..nms };
    let got = m.predict_candidates(&x, &nms).unwrap();
    let dense_side = dense.side_precisions().unwrap();
    check_precisions(&m, &got, &dense_side, &nms, "mixed batch, DFL");
    let want = dense.detections(&nms).unwrap();
    for img in 0..2 {
        assert_eq!(
            bits(&m.finish_candidates(&got[img], &nms)),
            bits(&want[img]),
            "DFL-consensus: image {img}"
        );
    }
    // Also check mixed fit/overflow batch with `DflRepulse` (entropies + strides + precisions bit-exact vs dense).
    let nms_rep =
        NmsConfig { conf: conf_low, fusion: BoxFusion::Dfl, score: ScoreRule::DflRepulse, ..nms.clone() };
    let got_rep = m_low.predict_candidates(&x, &nms_rep).unwrap();
    let want_rep = dense_low.detections(&nms_rep).unwrap();
    for img in 0..2 {
        assert_eq!(got_rep[img].entropies.as_ref().unwrap().len(), got_rep[img].cands.len());
        assert_eq!(got_rep[img].strides.as_ref().unwrap().len(), got_rep[img].cands.len());
        assert_eq!(
            bits(&m_low.finish_candidates(&got_rep[img], &nms_rep)),
            bits(&want_rep[img]),
            "DFL-repulse mixed fit/overflow: image {img}"
        );
    }
    // Verify sparse-gather `candidates_on_device` with exactly 1 surviving candidate matches dense bit-for-bit.
    // Run on the high-contrast random image `a` alone (image `b` is flat 0.5, so its interior anchors tie).
    let x1 = a.to_device(dev).unwrap();
    let dense1 = m_low.forward(&x1).unwrap();
    let dense1_side = dense1.side_precisions().unwrap();
    let cls1 = dense1.cls_raw.to_vec::<f32>().unwrap();
    let mut all_scores: Vec<f32> = (0..an)
        .map(|j| (0..nc).map(|c| cls1[c * an + j]).fold(f32::NEG_INFINITY, f32::max))
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    all_scores.sort_by(|u, v| v.total_cmp(u));
    let conf_one = (all_scores[0] + all_scores[1]) * 0.5;
    assert!(
        all_scores[0] > conf_one && conf_one > all_scores[1],
        "expected distinct top-2 scores for 1-candidate test, got {} and {}",
        all_scores[0],
        all_scores[1]
    );
    // `ScoreRule::None` keeps `candidate_conf == conf_one` regardless of `0.10`, deterministically exercising the
    // `total_cands == 1` 2-column padding path in `candidates_on_device`; also check `DflRepulse` at `conf_one`.
    for rule in [ScoreRule::None, ScoreRule::DflRepulse] {
        let nms_one = NmsConfig { conf: conf_one, fusion: BoxFusion::Dfl, score: rule, ..nms.clone() };
        let got_one = m_low.predict_candidates(&x1, &nms_one).unwrap();
        let want_one = dense1.detections(&nms_one).unwrap();
        if rule == ScoreRule::None || conf_one <= 0.10 {
            assert_eq!(
                got_one[0].cands.len(),
                1,
                "{rule:?} single-candidate sparse gather must have exactly 1 surviving candidate"
            );
            assert_eq!(want_one[0].len(), 1);
            check_precisions(&m_low, &got_one, &dense1_side, &nms_one, "single-candidate sparse gather");
        }
        assert_eq!(
            bits(&m_low.finish_candidates(&got_one[0], &nms_one)),
            bits(&want_one[0]),
            "{rule:?} single-candidate sparse gather"
        );
    }
    // `o2o` ignores `--score` (DflConsensus and DflRepulse match None bit-for-bit).
    let m_o2o = model(3, "o2o", dev, DType::F32);
    let o2o_none = NmsConfig { conf: 0.25, score: ScoreRule::None, ..NmsConfig::default() };
    let d_none = m_o2o.predict_raw(&x, &o2o_none).unwrap();
    for rule in [ScoreRule::DflConsensus, ScoreRule::DflRepulse] {
        let o2o_sc = NmsConfig { conf: 0.25, score: rule, ..NmsConfig::default() };
        let d_sc = m_o2o.predict_raw(&x, &o2o_sc).unwrap();
        for img in 0..2 {
            assert_eq!(
                bits(&d_none.images[img].detections),
                bits(&d_sc.images[img].detections),
                "o2o {rule:?} img={img}"
            );
        }
    }
    ctx.set_tensor_cores(true);
}
