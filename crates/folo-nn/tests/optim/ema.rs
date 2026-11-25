//! ModelEma hand-computed values, bit-exact 100-step parity, aliasing, and thread-count determinism.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::*;

use crate::close;

#[test]
fn ema_values() {
    let live = Conv2d::from_tensors(
        Conv2dConfig::new(1, 1, 1),
        Tensor::full([1, 1, 1, 1], 2.0).requires_grad_(),
        Some(Tensor::zeros([1]).requires_grad_()),
    )
    .unwrap();
    let shadow = Conv2d::from_tensors(
        Conv2dConfig::new(1, 1, 1),
        Tensor::zeros([1, 1, 1, 1]),
        Some(Tensor::zeros([1])),
    )
    .unwrap();
    let mut ema = ModelEma::new(shadow, &live).unwrap();
    assert_eq!(ema.module().weight.to_vec::<f32>().unwrap(), [2.0]);
    assert!(!ema.module().weight.requires_grad());
    let d = |u: f64| 0.9999 * (1.0 - (-u / 2000.0).exp());
    assert!((d(1.0) - 4.99825e-4).abs() < 1e-9);
    let mut want = 2.0f64;
    for (u, v) in [(1.0, 4.0f32), (2.0, 6.0)] {
        {
            let _g = no_grad();
            live.weight.fill_assign(v).unwrap();
        }
        ema.update(&live).unwrap();
        want = d(u) * want + (1.0 - d(u)) * v as f64;
        close(&ema.module().weight.to_vec::<f32>().unwrap(), &[want], 1e-6);
    }
    assert_eq!(ema.updates(), 2);
    // BatchNorm running statistics are tracked too.
    let bn = BatchNorm2d::new(2);
    let mut e2 = ModelEma::new(BatchNorm2d::new(2), &bn).unwrap();
    bn.forward(&Tensor::full([2, 2, 2, 2], 3.0)).unwrap();
    e2.update(&bn).unwrap();
    let rm = 0.03 * 3.0;
    close(&e2.module().running_mean.to_vec::<f32>().unwrap(), &[(1.0 - d(1.0)) * rm; 2], 1e-6);
}

struct EmaMixedNet {
    seq: Sequential,
    list: ModuleList<Conv2d>,
    big: Tensor,
    odd: Tensor,
    int_buf: Tensor,
}

impl EmaMixedNet {
    fn new(rng: &mut Rng) -> Self {
        let seq = Sequential::new()
            .push_named("stem", ConvBnAct::new(3, 17, 3, 1, 1, Activation::Silu, rng))
            .push_named("bn", BatchNorm2d::new(17));
        let list = ModuleList::new(vec![
            Conv2d::new(Conv2dConfig::new(17, 31, 3), rng),
            Conv2d::new(Conv2dConfig::new(31, 33, 1), rng),
        ]);
        let big = Tensor::randn([65_573], 0.0, 1.0, rng).requires_grad_();
        let odd = Tensor::randn([39], 0.0, 1.0, rng).requires_grad_();
        let int_buf = Tensor::zeros_with([4], folo_core::DType::I64, folo_core::Device::Cpu).unwrap();
        EmaMixedNet { seq, list, big, odd, int_buf }
    }
}

impl Module for EmaMixedNet {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("big", &self.big, TensorKind::Parameter);
        f("odd", &self.odd, TensorKind::Parameter);
        f("int_buf", &self.int_buf, TensorKind::Buffer);
    }

    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("seq", &self.seq);
        f("", &self.list);
    }
}

fn tracked_tensors(m: &dyn Module) -> Vec<(String, Tensor)> {
    m.named_tensors()
        .into_iter()
        .filter(|t| t.kind == TensorKind::Parameter || t.tensor.dtype() == folo_core::DType::F32)
        .map(|t| (t.name, t.tensor))
        .collect()
}

fn perturb_tracked(m: &dyn Module, rng: &mut Rng) {
    let _g = no_grad();
    for (_, t) in tracked_tensors(m) {
        let delta = Tensor::randn(t.shape().clone(), 0.0, 0.05, rng);
        t.add_assign(&delta).unwrap();
    }
}

fn tracked_bits(m: &dyn Module) -> Vec<(String, Vec<u32>)> {
    tracked_tensors(m)
        .into_iter()
        .map(|(n, t)| (n, t.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect()))
        .collect()
}

#[test]
fn ema_bit_exact_100_steps_and_resume_reset() {
    let mut rng = Rng::seed(42);
    let live = EmaMixedNet::new(&mut rng);
    let mut ema = ModelEma::with_params(EmaMixedNet::new(&mut rng), &live, 0.999, 50.0).unwrap();
    let mut ref_state: Vec<Vec<f32>> =
        tracked_tensors(&live).into_iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();

    let x = Tensor::randn([2, 3, 4, 4], 0.0, 1.0, &mut rng);
    for step in 1..=100u64 {
        perturb_tracked(&live, &mut rng);
        if step % 25 == 0 {
            let _ = live.seq.forward(&x).unwrap();
        }
        ema.update(&live).unwrap();
        let d = 0.999 * (1.0 - (-(step as f64) / 50.0).exp());
        let (d32, w) = (d as f32, (1.0 - d) as f32);
        for (e, (_, m)) in ref_state.iter_mut().zip(tracked_tensors(&live)) {
            for (ev, mv) in e.iter_mut().zip(m.to_vec::<f32>().unwrap()) {
                *ev = *ev * d32 + mv * w;
            }
        }
    }
    assert_eq!(ema.updates(), 100);
    for ((name, got), want) in tracked_bits(ema.module()).iter().zip(&ref_state) {
        let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, &want_bits, "tensor `{name}` diverged after 100 steps");
    }

    // Resume/reload consistency via EmaState::restore.
    let saved: std::collections::HashMap<String, Tensor> = ema
        .ema_tensors()
        .into_iter()
        .map(|(n, t)| (n, Tensor::from_vec(t.to_vec::<f32>().unwrap(), t.shape().clone()).unwrap()))
        .collect();
    let mut ema_b = ModelEma::with_params(EmaMixedNet::new(&mut rng), &live, 0.5, 5.0).unwrap();
    ema_b.restore(&saved, ema.updates(), ema.contract()).unwrap();
    assert_eq!(tracked_bits(ema.module()), tracked_bits(ema_b.module()));
    for _ in 0..15 {
        perturb_tracked(&live, &mut rng);
        ema.update(&live).unwrap();
        ema_b.update(&live).unwrap();
    }
    assert_eq!(tracked_bits(ema.module()), tracked_bits(ema_b.module()));

    // Reset matches a fresh EMA.
    ema.reset(&live).unwrap();
    assert_eq!(ema.updates(), 0);
    let mut fresh = ModelEma::with_params(EmaMixedNet::new(&mut rng), &live, 0.999, 50.0).unwrap();
    for _ in 0..5 {
        perturb_tracked(&live, &mut rng);
        ema.update(&live).unwrap();
        fresh.update(&live).unwrap();
    }
    assert_eq!(tracked_bits(ema.module()), tracked_bits(fresh.module()));
}

#[test]
fn ema_mismatches_non_contiguous_aliasing_and_thread_determinism() {
    struct Single(&'static str, Tensor);
    impl Module for Single {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f(self.0, &self.1, TensorKind::Parameter);
        }
    }
    let live = Single("weight", Tensor::full([4], 2.0).requires_grad_());
    let mut ema = ModelEma::new(Single("weight", Tensor::zeros([4])), &live).unwrap();
    ema.update(&live).unwrap();
    let before = tracked_bits(ema.module());
    assert!(ema.update(&Single("other", Tensor::full([4], 9.0))).is_err());
    assert!(ema.update(&Single("weight", Tensor::full([5], 9.0))).is_err());
    assert!(
        ema.update(&Single(
            "weight",
            Tensor::full_with([4], 9.0, folo_core::DType::F16, folo_core::Device::Cpu).unwrap()
        ))
        .is_err()
    );
    assert!(ema.update(&Sequential::new()).is_err());
    assert_eq!((ema.updates(), tracked_bits(ema.module())), (1, before));

    // Non-contiguous EMA & non-contiguous source parity.
    let base_live = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let base_ema = Tensor::zeros([3, 2]).t().unwrap();
    let mut ema_nc =
        ModelEma::with_params(Single("w", base_ema), &Single("w", base_live.clone()), 0.9, 10.0).unwrap();
    let mut ema_c =
        ModelEma::with_params(Single("w", Tensor::zeros([2, 3])), &Single("w", base_live.clone()), 0.9, 10.0)
            .unwrap();
    {
        let _g = no_grad();
        base_live.add_assign(&Tensor::full([2, 3], 1.5)).unwrap();
    }
    ema_nc.update(&Single("w", base_live.clone())).unwrap();
    ema_c.update(&Single("w", base_live)).unwrap();
    assert_eq!(tracked_bits(ema_nc.module()), tracked_bits(ema_c.module()));

    let src_t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap().t().unwrap();
    let src_c = Tensor::from_vec(src_t.to_vec::<f32>().unwrap(), [2, 3]).unwrap();
    let mut s_nc = ModelEma::with_params(
        Single("w", Tensor::zeros([2, 3])),
        &Single("w", Tensor::full([2, 3], 2.0)),
        0.9,
        10.0,
    )
    .unwrap();
    let mut s_c = ModelEma::with_params(
        Single("w", Tensor::zeros([2, 3])),
        &Single("w", Tensor::full([2, 3], 2.0)),
        0.9,
        10.0,
    )
    .unwrap();
    s_nc.update(&Single("w", src_t)).unwrap();
    s_c.update(&Single("w", src_c)).unwrap();
    assert_eq!(tracked_bits(s_nc.module()), tracked_bits(s_c.module()));

    // Thread count determinism (1 vs 4).
    let mut rng = Rng::seed(99);
    let m_live = EmaMixedNet::new(&mut rng);
    let mut e1 = ModelEma::with_params(EmaMixedNet::new(&mut rng), &m_live, 0.99, 10.0).unwrap();
    let mut e4 = ModelEma::with_params(EmaMixedNet::new(&mut rng), &m_live, 0.99, 10.0).unwrap();
    perturb_tracked(&m_live, &mut rng);
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| e1.update(&m_live).unwrap());
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| e4.update(&m_live).unwrap());
    assert_eq!(tracked_bits(e1.module()), tracked_bits(e4.module()));

    // Self-alias (detach + transpose), crossed alias, tied weights, and in-place buffer dtype change.
    struct Pair(Tensor, Tensor);
    impl Module for Pair {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("a", &self.0, TensorKind::Parameter);
            f("b", &self.1, TensorKind::Parameter);
        }
    }
    let w0 = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], [2, 2]).unwrap();
    let mut ema_al = ModelEma::with_params(
        Pair(Tensor::zeros([2, 2]), Tensor::zeros([2, 2])),
        &Pair(w0.clone(), w0),
        0.5,
        1.0,
    )
    .unwrap();
    let d = ema_al.decay_at(1);
    let (d32, w) = (d as f32, (1.0 - d) as f32);
    let alias_src = Pair(ema_al.module().0.detach(), ema_al.module().1.t().unwrap());
    ema_al.update(&alias_src).unwrap();
    assert_eq!(
        ema_al.module().0.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        [2.0f32, 4.0, 6.0, 8.0].iter().map(|&v| (v * d32 + v * w).to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        ema_al.module().1.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        [2.0f32, 4.0, 6.0, 8.0]
            .iter()
            .zip([2.0f32, 6.0, 4.0, 8.0])
            .map(|(&e, m)| (e * d32 + m * w).to_bits())
            .collect::<Vec<_>>()
    );

    let n = 65_552;
    let mut ema_cross = ModelEma::with_params(
        Pair(Tensor::zeros([n]), Tensor::zeros([n])),
        &Pair(Tensor::full([n], 1.0), Tensor::full([n], 3.0)),
        0.8,
        2.0,
    )
    .unwrap();
    let crossed = Pair(ema_cross.module().1.clone(), ema_cross.module().0.clone());
    let d1 = ema_cross.decay_at(1);
    let (d1_32, w1) = (d1 as f32, (1.0 - d1) as f32);
    ema_cross.update(&crossed).unwrap();
    let a_want = 1.0f32 * d1_32 + 3.0f32 * w1;
    let b_want = 3.0f32 * d1_32 + a_want * w1;
    assert_eq!(ema_cross.module().0.to_vec::<f32>().unwrap()[0].to_bits(), a_want.to_bits());
    assert_eq!(ema_cross.module().1.to_vec::<f32>().unwrap()[0].to_bits(), b_want.to_bits());

    let tied_w = Tensor::full([n], 2.0);
    let mut ema_tied = ModelEma::with_params(
        Pair(tied_w.clone(), tied_w),
        &Pair(Tensor::full([n], 2.0), Tensor::full([n], 2.0)),
        0.8,
        2.0,
    )
    .unwrap();
    ema_tied.update(&Pair(Tensor::full([n], 5.0), Tensor::full([n], 9.0))).unwrap();
    let step2 = (2.0f32 * d1_32 + 5.0f32 * w1) * d1_32 + 9.0f32 * w1;
    assert_eq!(ema_tied.module().0.to_vec::<f32>().unwrap()[0].to_bits(), step2.to_bits());

    struct WithBuf(Tensor, Tensor);
    impl Module for WithBuf {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("w", &self.0, TensorKind::Parameter);
            f("buf", &self.1, TensorKind::Buffer);
        }
    }
    let live_b = WithBuf(Tensor::full([4], 1.0), Tensor::full([4], 2.0));
    let mut ema_b = ModelEma::new(WithBuf(Tensor::zeros([4]), Tensor::zeros([4])), &live_b).unwrap();
    assert_eq!(ema_b.ema_tensors().len(), 2);
    live_b.1.to_dtype_(folo_core::DType::F16).unwrap();
    ema_b.module().1.to_dtype_(folo_core::DType::F16).unwrap();
    assert_eq!(ema_b.ema_tensors().len(), 1);
    ema_b.update(&live_b).unwrap();
}
