//! Multi-group GPU optimizer coalescing (`step_many`) and cross-group alias fallback tests.

use super::*;

#[test]
fn multi_group_coalesced_gpu_sgd_and_adamw_match_cpu() {
    let dev = gpu!();
    let mut rng = Rng::seed(2026);
    let shapes: [&[usize]; 6] = [&[8, 4, 3, 3], &[4, 4, 1, 1], &[8], &[4], &[8], &[4]];
    let init: Vec<Vec<f32>> = shapes
        .iter()
        .map(|s| Tensor::randn(s.to_vec(), 0.0, 0.5, &mut rng).to_vec::<f32>().unwrap())
        .collect();

    for kind in [OptimKind::Sgd, OptimKind::AdamW] {
        let mk_ps = |d: Device| -> Vec<Tensor> {
            shapes
                .iter()
                .zip(&init)
                .map(|(s, v)| {
                    Tensor::from_vec(v.clone(), s.to_vec())
                        .unwrap()
                        .to_device(d)
                        .unwrap()
                        .detach()
                        .requires_grad_()
                })
                .collect()
        };
        let cpu_ps = mk_ps(Device::Cpu);
        let gpu_ps = mk_ps(dev);
        // Groups 1 ("bn") and 2 ("bias") share identical OptimStep (lr, momentum, wd = 0) so step_many
        // coalesces them into one GPU optimizer_step_many_ batch, while group 0 ("decay") has wd > 0.
        let mk_groups = |ps: &[Tensor]| {
            vec![
                ParamGroup::new("decay", vec![("w0".into(), ps[0].clone()), ("w1".into(), ps[1].clone())])
                    .lr(0.02)
                    .momentum(0.9)
                    .weight_decay(5e-4),
                ParamGroup::new("bn", vec![("bn0".into(), ps[2].clone()), ("bn1".into(), ps[3].clone())])
                    .lr(0.02)
                    .momentum(0.9),
                ParamGroup::new("bias", vec![("b0".into(), ps[4].clone()), ("b1".into(), ps[5].clone())])
                    .lr(0.02)
                    .momentum(0.9),
            ]
        };
        let (mut opt_c, mut opt_g): (Box<dyn Optimizer>, Box<dyn Optimizer>) = match kind {
            OptimKind::Sgd => (
                Box::new(Sgd::new(mk_groups(&cpu_ps)).unwrap()),
                Box::new(Sgd::new(mk_groups(&gpu_ps)).unwrap()),
            ),
            OptimKind::AdamW => (
                Box::new(AdamW::new(mk_groups(&cpu_ps)).unwrap()),
                Box::new(AdamW::new(mk_groups(&gpu_ps)).unwrap()),
            ),
            OptimKind::MuSgd => unreachable!(),
        };
        for _ in 0..3 {
            let mut gc = folo_core::Grads::default();
            let mut gg = folo_core::Grads::default();
            for (cp, gp) in cpu_ps.iter().zip(&gpu_ps) {
                let g = Tensor::randn(cp.shape().clone(), 0.0, 0.2, &mut rng);
                gc.insert(cp, g.clone());
                gg.insert(gp, g.to_device(dev).unwrap());
            }
            opt_c.step(&gc).unwrap();
            opt_g.step(&gg).unwrap();
        }
        for (i, (cp, gp)) in cpu_ps.iter().zip(&gpu_ps).enumerate() {
            close(gp, cp, 1e-5, 1e-5, &format!("{kind:?} param {i}"));
        }
    }
}

#[test]
fn multi_group_gpu_cross_group_alias_falls_back_cleanly() {
    let dev = gpu!();
    // Two GPU groups with identical OptimStep where group 1's gradient aliases group 0's parameter storage.
    // Coalescing must decline merging and update group 0 then group 1 sequentially (matching CPU).
    let p0_c = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [4]).unwrap().requires_grad_();
    let p1_c = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], [4]).unwrap().requires_grad_();
    let p0_g = p0_c.to_device(dev).unwrap().detach().requires_grad_();
    let p1_g = p1_c.to_device(dev).unwrap().detach().requires_grad_();

    let mk = |p0: &Tensor, p1: &Tensor| {
        Sgd::new(vec![
            ParamGroup::new("bn", vec![("p0".into(), p0.clone())]).lr(0.1).momentum(0.0),
            ParamGroup::new("bias", vec![("p1".into(), p1.clone())]).lr(0.1).momentum(0.0),
        ])
        .unwrap()
    };
    let mut opt_c = mk(&p0_c, &p1_c);
    let mut opt_g = mk(&p0_g, &p1_g);

    let g0 = Tensor::from_slice(&[0.5f32, -0.5, 1.0, -1.0], [4]).unwrap();
    let mut gc = folo_core::Grads::default();
    gc.insert(&p0_c, g0.clone());
    gc.insert(&p1_c, p0_c.detach());
    let mut gg = folo_core::Grads::default();
    gg.insert(&p0_g, g0.to_device(dev).unwrap());
    gg.insert(&p1_g, p0_g.detach());

    opt_c.step(&gc).unwrap();
    opt_g.step(&gg).unwrap();
    close(&p0_g, &p0_c, 1e-6, 1e-6, "cross-alias p0");
    close(&p1_g, &p1_c, 1e-6, 1e-6, "cross-alias p1");
}
