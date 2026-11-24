//! Candidate-first prediction on the CPU (docs/MODEL.md §9.5): the records and detections of
//! `FoloModel::predict_candidates` are the dense path's bit for bit, at square, rectangular and odd-sided inputs,
//! batches, sparse and dense candidate sets, both heads (with ties in score), class filters, every box fusion
//! (MODEL.md §9.4, with DFL fusion's side precisions), and through the predictor; the dense forward
//! (`HeadOutput`) is unchanged.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::Module;
use folo_vision::fusion::{BoxFusion, ScoreRule};
use folo_vision::model::{FoloModel, HeadMode, HeadOutput, ModelConfig};
use folo_vision::nms::{Candidate, Detection, NmsConfig};
use folo_vision::{PredictConfig, Predictor};

fn model(scale: &str, nc: usize, head: &str, seed: u64) -> FoloModel {
    let mut c = ModelConfig::new(scale, nc).unwrap();
    c.head = head.into();
    let mut m = FoloModel::with_rng(c, &mut Rng::seed(seed)).unwrap();
    m.fuse().unwrap();
    m
}

fn input(b: usize, h: usize, w: usize, seed: u64) -> Tensor {
    Tensor::rand_uniform([b, 3, h, w], 0.0, 1.0, &mut Rng::seed(seed))
}

/// Per image, every anchor's `(best score, first best class)` and its decoded box, from the dense outputs.
fn dense_best(out: &HeadOutput) -> Vec<Vec<(f32, u32, folo_vision::Xyxy)>> {
    let nc = out.num_classes;
    out.decode_images()
        .unwrap()
        .into_iter()
        .map(|(boxes, scores)| {
            boxes
                .iter()
                .zip(scores.chunks_exact(nc))
                .map(|(b, row)| {
                    let (mut bc, mut bs) = (0, row[0]);
                    for (c, &s) in row.iter().enumerate().skip(1) {
                        if s > bs {
                            (bc, bs) = (c, s);
                        }
                    }
                    (bs, bc as u32, *b)
                })
                .collect()
        })
        .collect()
}

/// A `conf` that about `frac` of the anchors of every level of every image pass (at least one per level).
fn conf_for(out: &HeadOutput, best: &[Vec<(f32, u32, folo_vision::Xyxy)>], frac: f64) -> f32 {
    let mut conf = f32::INFINITY;
    for img in best {
        for lv in &out.anchors.levels {
            let n = lv.height * lv.width;
            let mut s: Vec<f32> = img[lv.start..lv.start + n].iter().map(|x| x.0).collect();
            s.sort_by(f32::total_cmp);
            let k = ((n as f64 * (1.0 - frac)) as usize).min(n - 1);
            conf = conf.min(if k == 0 { 0.0 } else { s[k - 1] });
        }
    }
    conf
}

/// The dense path's records of one image under `nms` for the branch `mode` (no class filter): every anchor that
/// passes, or for `o2o` the `max_det` best by score then anchor, in anchor order.
fn expected(best: &[(f32, u32, folo_vision::Xyxy)], nms: &NmsConfig, mode: HeadMode) -> Vec<Candidate> {
    let mut v: Vec<Candidate> = best
        .iter()
        .enumerate()
        .filter(|(_, b)| b.0 > nms.conf)
        .map(|(a, b)| Candidate { anchor: a as u32, class: b.1, score: b.0, bbox: b.2 })
        .collect();
    if mode == HeadMode::O2o && v.len() > nms.max_det {
        v.sort_by(|x, y| y.score.total_cmp(&x.score).then(x.anchor.cmp(&y.anchor)));
        v.truncate(nms.max_det);
        v.sort_by_key(|c| c.anchor);
    }
    v
}

fn cand_bits(c: &Candidate) -> [u32; 7] {
    let b = c.bbox;
    [c.anchor, c.class, c.score.to_bits(), b.x1.to_bits(), b.y1.to_bits(), b.x2.to_bits(), b.y2.to_bits()]
}

fn det_bits(d: &[Vec<Detection>]) -> Vec<Vec<[u32; 6]>> {
    d.iter()
        .map(|v| {
            v.iter()
                .map(|d| {
                    let b = d.bbox;
                    [
                        d.class,
                        d.score.to_bits(),
                        b.x1.to_bits(),
                        b.y1.to_bits(),
                        b.x2.to_bits(),
                        b.y2.to_bits(),
                    ]
                })
                .collect()
        })
        .collect()
}

/// Records and detections of the candidate-first path against the dense path's, bit for bit (with DFL fusion
/// also each record's side precisions, against [`HeadOutput::side_precisions`]); returns the candidates per
/// image.
fn check(m: &FoloModel, x: &Tensor, nms: &NmsConfig) -> Vec<usize> {
    // Whatever the box branch reads that it did not compute is NaN (never the dense forward's values left in a
    // recycled buffer), so a record can only match by computing what it reads.
    folo_core::cpu::poison_uninit(true);
    let _g = no_grad();
    let out = m.forward(x).unwrap();
    let best = dense_best(&out);
    let got = m.predict_candidates(x, nms).unwrap();
    assert_eq!(got.len(), x.dims()[0]);
    let dfl = nms.fusion.wants_side_precision() && m.head_mode() == HeadMode::O2m;
    let prec = if dfl { Some(out.side_precisions().unwrap()) } else { None };
    for (i, (g, b)) in got.iter().zip(&best).enumerate() {
        let want = expected(b, nms, m.head_mode());
        let (gb, wb): (Vec<_>, Vec<_>) =
            (g.cands.iter().map(cand_bits).collect(), want.iter().map(cand_bits).collect());
        assert_eq!(gb.len(), wb.len(), "image {i}: candidate count");
        for (k, (g, w)) in gb.iter().zip(&wb).enumerate() {
            assert_eq!(g, w, "image {i}, record {k}");
        }
        match &prec {
            Some(p) => {
                let gp = g.precisions.as_ref().expect("DFL fusion needs the precisions");
                assert_eq!(gp.len(), want.len(), "image {i}: precision count");
                for (k, (gp, c)) in gp.iter().zip(&want).enumerate() {
                    let wp = p[i][c.anchor as usize];
                    assert_eq!(
                        gp.map(f32::to_bits),
                        wp.map(f32::to_bits),
                        "image {i}, record {k}: precisions"
                    );
                }
            }
            None => assert_eq!(g.precisions, None, "image {i}"),
        }
    }
    assert_eq!(det_bits(&m.predict_detections(x, nms).unwrap()), det_bits(&out.detections(nms).unwrap()));
    got.iter().map(|c| c.cands.len()).collect()
}

/// folo-n at 640², 320² (two images), 384×640, 352² and 352×640 (P5 has 11 rows), with a few percent of the
/// anchors of every level passing, and with every anchor passing.
#[test]
fn folo_n_matches_dense_bit_for_bit() {
    let m = model("n", 20, "o2m", 1);
    for (b, h, w) in [(1, 640, 640), (2, 320, 320), (1, 384, 640), (1, 352, 352), (2, 352, 640)] {
        let x = input(b, h, w, (h * w + b) as u64);
        let out = {
            let _g = no_grad();
            m.forward(&x).unwrap()
        };
        let best = dense_best(&out);
        for frac in [0.03, 1.0] {
            let nms = NmsConfig { conf: conf_for(&out, &best, frac), ..NmsConfig::default() };
            let n = check(&m, &x, &nms);
            assert!(n.iter().all(|&n| n > 0), "{b}x{h}x{w} at {frac}: no candidates");
        }
    }
}

/// folo-s shapes (wider box branch, 128-channel P3), and a batch of five whose images range from no candidate
/// at all (a constant image) to many.
#[test]
fn folo_s_and_mixed_batches_match_dense() {
    let m = model("s", 7, "o2m", 2);
    let x = input(1, 320, 320, 5);
    let out = {
        let _g = no_grad();
        m.forward(&x).unwrap()
    };
    let nms = NmsConfig { conf: conf_for(&out, &dense_best(&out), 0.05), ..NmsConfig::default() };
    check(&m, &x, &nms);

    let m = model("n", 20, "o2m", 3);
    let mut parts: Vec<Tensor> = (0..4).map(|i| input(1, 256, 320, 10 + i)).collect();
    parts.insert(2, Tensor::full([1, 3, 256, 320], 0.5));
    let x = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0).unwrap();
    let out = {
        let _g = no_grad();
        m.forward(&x).unwrap()
    };
    let best = dense_best(&out);
    // The constant image's interior anchors share one score; a conf just above it leaves that image few or none.
    let flat = best[2][out.anchors.levels[0].width * 10 + 10].0;
    let n = check(&m, &x, &NmsConfig { conf: flat, ..NmsConfig::default() });
    assert!(n.iter().any(|&n| n > 100), "{n:?}");
    let none = NmsConfig { conf: 1.0, ..NmsConfig::default() };
    assert_eq!(check(&m, &x, &none), [0; 5]);
}

/// The one-to-one head: its top-k at any conf (`max_det` below the passing count), and with every class
/// logit of a level equal (zeroed class weights), so the selection rests on the anchor order of the ties.
#[test]
fn o2o_top_k_and_ties_match_dense() {
    let m = model("n", 5, "o2o", 4);
    assert_eq!(m.head_mode(), HeadMode::O2o);
    let x = input(2, 320, 384, 6);
    let out = {
        let _g = no_grad();
        m.forward(&x).unwrap()
    };
    let conf = conf_for(&out, &dense_best(&out), 0.2);
    for max_det in [0, 1, 37, 300, 100_000] {
        check(&m, &x, &NmsConfig { conf, max_det, ..NmsConfig::default() });
    }
    {
        let _g = no_grad();
        for l in m.head.o2o.as_ref().unwrap() {
            let wt = &l.cls_branch.out.weight;
            wt.copy_from(&Tensor::zeros(wt.shape().clone())).unwrap();
        }
    }
    let n = check(&m, &x, &NmsConfig { conf: 0.0, max_det: 300, ..NmsConfig::default() });
    assert_eq!(n, [300, 300]);
}

/// Class filters: NMS drops other classes before it runs, the top-k after its selection; both as the dense path.
#[test]
fn class_filters_match_dense() {
    for head in ["o2m", "o2o"] {
        let m = model("n", 6, head, 7);
        let x = input(1, 320, 320, 8);
        let out = {
            let _g = no_grad();
            m.forward(&x).unwrap()
        };
        let conf = conf_for(&out, &dense_best(&out), 0.1);
        let nms = NmsConfig { conf, max_det: 40, classes: Some(vec![1, 4]), ..NmsConfig::default() };
        let _g = no_grad();
        let got = m.predict_detections(&x, &nms).unwrap();
        assert_eq!(det_bits(&got), det_bits(&out.detections(&nms).unwrap()), "{head}");
        assert!(got[0].iter().all(|d| d.class == 1 || d.class == 4));
    }
}

/// Box fusion (MODEL.md §9.4): score voting and DFL evidence fusion, after NMS and with the top-k head (which
/// ignores them), give the dense path's detections bit for bit, DFL fusion from the dense side precisions.
#[test]
fn box_fusion_matches_dense() {
    for (head, seed) in [("o2m", 14), ("o2o", 15)] {
        let m = model("n", 4, head, seed);
        let x = input(2, 288, 352, seed + 100);
        let out = {
            let _g = no_grad();
            m.forward(&x).unwrap()
        };
        let best = dense_best(&out);
        for frac in [0.05, 1.0] {
            let conf = conf_for(&out, &best, frac);
            for fusion in [BoxFusion::Vote, BoxFusion::Dfl] {
                // A low IoU threshold so NMS suppresses (and fusion merges) many boxes per cluster.
                let nms = NmsConfig { conf, iou: 0.3, fusion, fusion_iou: 0.4, ..NmsConfig::default() };
                let n = check(&m, &x, &nms);
                assert!(n.iter().all(|&n| n > 0), "{head} {fusion:?} at {frac}: no candidates");
                if head == "o2m" {
                    let fused = det_bits(&m.predict_detections(&x, &nms).unwrap());
                    let plain = det_bits(
                        &out.detections(&NmsConfig { fusion: BoxFusion::None, ..nms.clone() }).unwrap(),
                    );
                    assert_ne!(fused, plain, "{fusion:?} at {frac} moved no box: the test cannot see it");
                }
            }
        }
    }
}

/// The consensus score (MODEL.md §9.4), alone and on top of DFL fusion, gives the dense path's detections bit for
/// bit; the top-k head has no clusters and ignores it.
#[test]
fn consensus_score_matches_dense() {
    for (head, seed) in [("o2m", 16), ("o2o", 17)] {
        let m = model("n", 4, head, seed);
        let x = input(2, 288, 352, seed + 100);
        let out = {
            let _g = no_grad();
            m.forward(&x).unwrap()
        };
        let best = dense_best(&out);
        let conf = conf_for(&out, &best, 0.3);
        for fusion in [BoxFusion::None, BoxFusion::Dfl] {
            let plain = NmsConfig { conf, iou: 0.3, fusion, ..NmsConfig::default() };
            let nms = NmsConfig { score: ScoreRule::Consensus, ..plain.clone() };
            let n = check(&m, &x, &nms);
            assert!(n.iter().all(|&n| n > 0), "{head} {fusion:?}: no candidates");
            let (on, off) =
                (m.predict_detections(&x, &nms).unwrap(), m.predict_detections(&x, &plain).unwrap());
            if head == "o2m" {
                assert_ne!(
                    det_bits(&on),
                    det_bits(&off),
                    "{fusion:?}: no score moved, the test cannot see it"
                );
            } else {
                assert_eq!(det_bits(&on), det_bits(&off), "the top-k head ignores the score rule");
            }
        }
    }
}

/// The predictor takes the candidate-first path and returns what prepare → forward → detections gives, with
/// and without DFL fusion.
#[test]
fn predictor_matches_its_dense_steps() {
    for fusion in [BoxFusion::None, BoxFusion::Dfl] {
        let cfg = PredictConfig { conf: 0.0, fusion, ..PredictConfig::default() };
        let p = Predictor::new(model("n", 4, "o2m", 9), cfg).unwrap();
        let mut rng = Rng::seed(11);
        let data = (0..300 * 200 * 3).map(|_| rng.uniform(0.0, 255.0) as u8).collect();
        let img = folo_data::Image::from_raw(300, 200, data).unwrap();
        let got = p.predict(&img).unwrap();
        let batch = p.prepare(std::slice::from_ref(&img), true).unwrap();
        let c = p.config();
        let (conf, iou, max_det, fusion_iou) = (c.conf, c.iou, c.max_det, c.fusion_iou);
        let nms = NmsConfig { conf, iou, max_det, fusion, fusion_iou, ..NmsConfig::default() };
        let dense = p.forward(&batch).unwrap().detections(&nms).unwrap();
        let want = p.finish_detections(&batch, dense);
        assert!(!want[0].is_empty());
        assert_eq!(det_bits(&[got.detections]), det_bits(&want), "{fusion:?}");
    }
}

/// The dense forward is untouched: a fused model's `HeadOutput` still holds every anchor's raw box outputs, the
/// box branches' own outputs concatenated; and only fused, eval-mode, single-label CPU prediction is
/// candidate-first.
#[test]
fn dense_forward_is_unchanged_and_lazy_only_where_it_applies() {
    let m = model("n", 3, "o2m", 12);
    let x = input(1, 128, 192, 13);
    let _g = no_grad();
    let out = m.forward(&x).unwrap();
    assert_eq!(out.box_raw.dims(), [1, 64, out.anchors.len()]);
    let feats = {
        let mut f = Vec::new();
        m.forward_traced(&x, &mut |i, t| {
            if [16, 19, 22].contains(&i) {
                f.push(t.clone());
            }
        })
        .unwrap();
        f
    };
    let per_level: Vec<Tensor> = m
        .head
        .levels
        .iter()
        .zip(&feats)
        .map(|(l, f)| {
            use folo_nn::Forward;
            let y = l.box_branch.forward(f).unwrap();
            let (b, c, h, w) = y.dims4().unwrap();
            y.reshape([b, c, h * w]).unwrap()
        })
        .collect();
    let cat = Tensor::cat(&per_level.iter().collect::<Vec<_>>(), 2).unwrap();
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&out.box_raw), bits(&cat));

    let nms = NmsConfig::default();
    assert!(m.candidates_first_applies(x.device(), &nms));
    assert!(
        !m.candidates_first_applies(x.device(), &NmsConfig { multi_label: true, ..NmsConfig::default() })
    );
    let unfused = FoloModel::new(ModelConfig::new("n", 3).unwrap()).unwrap();
    unfused.eval();
    assert!(!unfused.candidates_first_applies(x.device(), &nms));
    assert!(unfused.predict_candidates(&x, &nms).is_err());
    assert_eq!(unfused.predict_detections(&x, &nms).unwrap().len(), 1, "unfused models predict densely");
}
