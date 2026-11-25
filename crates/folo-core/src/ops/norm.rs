//! Layer normalisation ([`Tensor::layer_norm`]) and RMS normalisation ([`Tensor::rms_norm`]) over the last
//! dimension, backed by optional [`crate::backend::Backend`] kernels with composite fallbacks.

use smallvec::SmallVec;

use super::with_views_dyn;
use crate::{DType, Error, Result, Shape, Tensor};

fn check_norm_args(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
    op: &'static str,
) -> Result<(usize, usize)> {
    x.check_float(op)?;
    if x.rank() == 0 {
        return Err(Error::invalid_shape(op, x.shape(), "needs rank >= 1"));
    }
    let d = x.dims()[x.rank() - 1];
    if d == 0 {
        return Err(Error::invalid_shape(op, x.shape(), "last dimension must be non-zero"));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(Error::invalid_arg(op, format!("eps must be finite and >= 0 (got {eps})")));
    }
    for param in [weight, bias].into_iter().flatten() {
        x.check_same_device(param, op)?;
        param.check_float_like(x, op)?;
        if param.dims() != [d] {
            return Err(Error::ShapeMismatch { op, lhs: Shape::new(&[d]), rhs: param.shape().clone() });
        }
    }
    Ok((x.numel() / d, d))
}

impl Tensor {
    /// Layer normalisation over the last dimension `d` of `self` (`[.., d]`):
    /// `y = ((x - mean) / sqrt(var + eps)) * weight + bias` with biased variance.
    ///
    /// `weight` and `bias`, when given, have shape `[d]` and match `self`'s device and dtype. `F16` inputs are
    /// normalised in `F32` and rounded back to `F16`.
    pub fn layer_norm(&self, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
        let (outer, d) = check_norm_args(self, weight, bias, eps, "layer_norm")?;
        if self.dtype() == DType::F16 {
            let x32 = self.to_dtype(DType::F32)?;
            let w32 = weight.map(|w| w.to_dtype(DType::F32)).transpose()?;
            let b32 = bias.map(|b| b.to_dtype(DType::F32)).transpose()?;
            return x32.layer_norm(w32.as_ref(), b32.as_ref(), eps)?.to_dtype(DType::F16);
        }

        let b = self.backend()?;
        let mut ts: SmallVec<[&Tensor; 3]> = SmallVec::new();
        ts.push(self);
        let w_idx = weight.map(|w| {
            let i = ts.len();
            ts.push(w);
            i
        });
        let b_idx = bias.map(|bt| {
            let i = ts.len();
            ts.push(bt);
            i
        });
        let fused =
            with_views_dyn(&ts, |v| b.layer_norm(v[0], w_idx.map(|i| v[i]), b_idx.map(|i| v[i]), eps));
        match fused {
            Ok((ys, ms, rs)) => {
                let out = Tensor::wrap(ys, self.shape().clone(), "layer_norm")?;
                let mean = Tensor::wrap(ms, Shape::new(&[outer]), "layer_norm")?;
                let rstd = Tensor::wrap(rs, Shape::new(&[outer]), "layer_norm")?;
                let x = self.detach();
                let w = weight.map(Tensor::detach);
                let need_w = weight.is_some_and(Tensor::requires_grad);
                let need_b = bias.is_some_and(Tensor::requires_grad);
                let has_w = weight.is_some();
                let has_b = bias.is_some();
                Ok(out.track_saved("layer_norm", &ts, &[&mean.clone(), &rstd.clone()], move |g| {
                    x.check_same_device(g, "layer_norm")?;
                    g.check_float_like(&x, "layer_norm")?;
                    let b = x.backend()?;
                    let mut bts: SmallVec<[&Tensor; 5]> = SmallVec::new();
                    bts.push(&x);
                    bts.push(g);
                    bts.push(&mean);
                    bts.push(&rstd);
                    let wi = w.as_ref().map(|wt| {
                        let i = bts.len();
                        bts.push(wt);
                        i
                    });
                    let res = with_views_dyn(&bts, |v| {
                        b.layer_norm_backward(v[0], v[1], wi.map(|i| v[i]), v[2], v[3], need_w, need_b)
                    });
                    match res {
                        Ok((dxs, dws, dbs)) => {
                            let mut out_grads: SmallVec<[Option<Tensor>; 2]> = SmallVec::new();
                            out_grads.push(Some(Tensor::wrap(dxs, x.shape().clone(), "layer_norm")?));
                            if has_w {
                                out_grads.push(
                                    dws.map(|s| Tensor::wrap(s, Shape::new(&[d]), "layer_norm"))
                                        .transpose()?,
                                );
                            }
                            if has_b {
                                out_grads.push(
                                    dbs.map(|s| Tensor::wrap(s, Shape::new(&[d]), "layer_norm"))
                                        .transpose()?,
                                );
                            }
                            Ok(out_grads)
                        }
                        Err(Error::NotSupported { .. }) => layer_norm_backward_composite(
                            &x,
                            g,
                            w.as_ref(),
                            &mean,
                            &rstd,
                            has_w,
                            has_b,
                            need_w,
                            need_b,
                        ),
                        Err(e) => Err(e),
                    }
                }))
            }
            Err(Error::NotSupported { .. }) => {
                let mean = self.mean(-1, true)?;
                let diff = self.sub(&mean)?;
                let var = diff.square()?.mean(-1, true)?;
                let mut y = diff.mul(&var.add_scalar(eps)?.rsqrt()?)?;
                if let Some(w) = weight {
                    y = y.mul(w)?;
                }
                if let Some(bt) = bias {
                    y = y.add(bt)?;
                }
                Ok(y)
            }
            Err(e) => Err(e),
        }
    }

    /// Root-mean-square normalisation over the last dimension `d` of `self` (`[.., d]`):
    /// `y = (x / sqrt(mean(x^2) + eps)) * weight`.
    ///
    /// `weight`, when given, has shape `[d]` and matches `self`'s device and dtype. `F16` inputs are normalised
    /// in `F32` and rounded back to `F16`.
    pub fn rms_norm(&self, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
        let (outer, d) = check_norm_args(self, weight, None, eps, "rms_norm")?;
        if self.dtype() == DType::F16 {
            let x32 = self.to_dtype(DType::F32)?;
            let w32 = weight.map(|w| w.to_dtype(DType::F32)).transpose()?;
            return x32.rms_norm(w32.as_ref(), eps)?.to_dtype(DType::F16);
        }

        let b = self.backend()?;
        let mut ts: SmallVec<[&Tensor; 2]> = SmallVec::new();
        ts.push(self);
        let w_idx = weight.map(|w| {
            let i = ts.len();
            ts.push(w);
            i
        });
        let fused = with_views_dyn(&ts, |v| b.rms_norm(v[0], w_idx.map(|i| v[i]), eps));
        match fused {
            Ok((ys, rs)) => {
                let out = Tensor::wrap(ys, self.shape().clone(), "rms_norm")?;
                let rstd = Tensor::wrap(rs, Shape::new(&[outer]), "rms_norm")?;
                let x = self.detach();
                let w = weight.map(Tensor::detach);
                let need_w = weight.is_some_and(Tensor::requires_grad);
                let has_w = weight.is_some();
                Ok(out.track_saved("rms_norm", &ts, &[&rstd.clone()], move |g| {
                    x.check_same_device(g, "rms_norm")?;
                    g.check_float_like(&x, "rms_norm")?;
                    let b = x.backend()?;
                    let mut bts: SmallVec<[&Tensor; 4]> = SmallVec::new();
                    bts.push(&x);
                    bts.push(g);
                    bts.push(&rstd);
                    let wi = w.as_ref().map(|wt| {
                        let i = bts.len();
                        bts.push(wt);
                        i
                    });
                    let res = with_views_dyn(&bts, |v| {
                        b.rms_norm_backward(v[0], v[1], wi.map(|i| v[i]), v[2], need_w)
                    });
                    match res {
                        Ok((dxs, dws)) => {
                            let mut out_grads: SmallVec<[Option<Tensor>; 2]> = SmallVec::new();
                            out_grads.push(Some(Tensor::wrap(dxs, x.shape().clone(), "rms_norm")?));
                            if has_w {
                                out_grads.push(
                                    dws.map(|s| Tensor::wrap(s, Shape::new(&[d]), "rms_norm")).transpose()?,
                                );
                            }
                            Ok(out_grads)
                        }
                        Err(Error::NotSupported { .. }) => {
                            rms_norm_backward_composite(&x, g, w.as_ref(), &rstd, has_w, need_w)
                        }
                        Err(e) => Err(e),
                    }
                }))
            }
            Err(Error::NotSupported { .. }) => {
                let ms = self.square()?.mean(-1, true)?;
                let mut y = self.mul(&ms.add_scalar(eps)?.rsqrt()?)?;
                if let Some(w) = weight {
                    y = y.mul(w)?;
                }
                Ok(y)
            }
            Err(e) => Err(e),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_backward_composite(
    x: &Tensor,
    g: &Tensor,
    w: Option<&Tensor>,
    mean: &Tensor,
    rstd: &Tensor,
    has_w: bool,
    has_b: bool,
    need_w: bool,
    need_b: bool,
) -> Result<SmallVec<[Option<Tensor>; 2]>> {
    let stat_shape = x.shape().with_dim(x.rank() - 1, 1);
    let m = mean.reshape(&stat_shape)?;
    let r = rstd.reshape(&stat_shape)?;
    let xh = x.sub(&m)?.mul(&r)?;
    let gw = match w {
        Some(wt) => g.mul(wt)?,
        None => g.clone(),
    };
    let c1 = gw.mean(-1, true)?;
    let c2 = gw.mul(&xh)?.mean(-1, true)?;
    let dx = r.mul(&gw.sub(&c1.add(&xh.mul(&c2)?)?)?)?;
    let mut out: SmallVec<[Option<Tensor>; 2]> = SmallVec::new();
    out.push(Some(dx));
    let reduce_dims: Vec<usize> = (0..x.rank() - 1).collect();
    if has_w {
        out.push(if need_w {
            let prod = g.mul(&xh)?;
            Some(if reduce_dims.is_empty() { prod } else { prod.sum(reduce_dims.clone(), false)? })
        } else {
            None
        });
    }
    if has_b {
        out.push(if need_b {
            Some(if reduce_dims.is_empty() { g.clone() } else { g.sum(reduce_dims, false)? })
        } else {
            None
        });
    }
    Ok(out)
}

fn rms_norm_backward_composite(
    x: &Tensor,
    g: &Tensor,
    w: Option<&Tensor>,
    rstd: &Tensor,
    has_w: bool,
    need_w: bool,
) -> Result<SmallVec<[Option<Tensor>; 2]>> {
    let stat_shape = x.shape().with_dim(x.rank() - 1, 1);
    let r = rstd.reshape(&stat_shape)?;
    let xh = x.mul(&r)?;
    let gw = match w {
        Some(wt) => g.mul(wt)?,
        None => g.clone(),
    };
    let c = gw.mul(&xh)?.mean(-1, true)?;
    let dx = r.mul(&gw.sub(&xh.mul(&c)?)?)?;
    let mut out: SmallVec<[Option<Tensor>; 2]> = SmallVec::new();
    out.push(Some(dx));
    if has_w {
        let reduce_dims: Vec<usize> = (0..x.rank() - 1).collect();
        out.push(if need_w {
            let prod = g.mul(&xh)?;
            Some(if reduce_dims.is_empty() { prod } else { prod.sum(reduce_dims, false)? })
        } else {
            None
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use crate::rng::Rng;

    #[test]
    fn norm_validation_and_aliasing() {
        let s = Tensor::scalar(1.0f32);
        assert!(s.layer_norm(None, None, 1e-5).is_err());
        assert!(s.rms_norm(None, 1e-5).is_err());

        let x = Tensor::randn([2, 6], 0.0, 1.0, &mut Rng::seed(101));
        assert!(x.layer_norm(None, None, -1e-5).is_err());
        assert!(x.layer_norm(None, None, f32::NAN).is_err());
        assert!(x.rms_norm(None, f32::INFINITY).is_err());

        let bad_w = Tensor::ones([5]);
        assert!(x.layer_norm(Some(&bad_w), None, 1e-5).is_err());
        assert!(x.layer_norm(None, Some(&bad_w), 1e-5).is_err());
        assert!(x.rms_norm(Some(&bad_w), 1e-5).is_err());

        let zero_last = Tensor::zeros([2, 0]);
        assert!(zero_last.layer_norm(None, None, 1e-5).is_err());
        assert!(zero_last.rms_norm(None, 1e-5).is_err());

        // Empty outer batch with non-zero last dim succeeds and produces zero param grads.
        let empty = Tensor::zeros([0, 6]).requires_grad_();
        let w = Tensor::ones([6]).requires_grad_();
        let b = Tensor::zeros([6]).requires_grad_();
        let y = empty.layer_norm(Some(&w), Some(&b), 1e-5).unwrap();
        assert_eq!(y.dims(), &[0, 6]);
        let g = y.backward_with(&Tensor::zeros([0, 6])).unwrap();
        assert_eq!(g.get(&w).unwrap().to_vec::<f32>().unwrap(), vec![0.0; 6]);
        assert_eq!(g.get(&b).unwrap().to_vec::<f32>().unwrap(), vec![0.0; 6]);

        // Aliasing same storage cell (`x` used as weight and bias on 1D tensor) must not deadlock.
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [4]).unwrap().requires_grad_();
        let yn = v.layer_norm(Some(&v), Some(&v), 1e-5).unwrap();
        let gn = yn.sum_all().unwrap().backward().unwrap();
        assert_eq!(gn.get(&v).unwrap().dims(), &[4]);
        let yr = v.rms_norm(Some(&v), 1e-5).unwrap();
        let gr = yr.sum_all().unwrap().backward().unwrap();
        assert_eq!(gr.get(&v).unwrap().dims(), &[4]);
    }

    #[test]
    fn norm_matches_composite_formulas() {
        let mut rng = Rng::seed(102);
        let x = Tensor::randn([4, 3, 19], 0.2, 1.5, &mut rng);
        let w = Tensor::randn([19], 1.0, 0.3, &mut rng);
        let b = Tensor::randn([19], 0.0, 0.2, &mut rng);
        let eps = 1e-5;

        let got_ln = x.layer_norm(Some(&w), Some(&b), eps).unwrap().to_vec::<f32>().unwrap();
        let mean = x.mean(-1, true).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.square().unwrap().mean(-1, true).unwrap();
        let ref_ln = diff
            .mul(&var.add_scalar(eps).unwrap().rsqrt().unwrap())
            .unwrap()
            .mul(&w)
            .unwrap()
            .add(&b)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in got_ln.iter().zip(&ref_ln) {
            assert!((a - b).abs() < 1e-5, "ln {a} vs {b}");
        }

        let got_rms = x.rms_norm(Some(&w), eps).unwrap().to_vec::<f32>().unwrap();
        let ms = x.square().unwrap().mean(-1, true).unwrap();
        let ref_rms = x
            .mul(&ms.add_scalar(eps).unwrap().rsqrt().unwrap())
            .unwrap()
            .mul(&w)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in got_rms.iter().zip(&ref_rms) {
            assert!((a - b).abs() < 1e-5, "rms {a} vs {b}");
        }
    }
}
