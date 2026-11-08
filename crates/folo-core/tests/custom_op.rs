//! The public custom-op API, used the way a downstream crate would (a fused kernel with its own backward).

use folo_core::rng::Rng;
use folo_core::testing::grad_check;
use folo_core::{Error, Result, Tensor, no_grad};

/// Fused `silu(x * scale + shift)` with per-channel `scale`/`shift` of shape `[C, 1]` (a BN-style affine plus
/// activation), forward computed on raw slices, backward written by hand.
fn fused_affine_silu(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let (c, n) = x.dims2()?;
    let xs = x.to_vec::<f32>()?;
    let (sc, sh) = (scale.to_vec::<f32>()?, shift.to_vec::<f32>()?);
    let mut out = vec![0f32; c * n];
    for i in 0..c {
        for j in 0..n {
            let z = xs[i * n + j] * sc[i] + sh[i];
            out[i * n + j] = z / (1.0 + (-z).exp());
        }
    }
    let out = Tensor::from_vec(out, [c, n])?;
    // The closure keeps detached handles (no graph) of what it needs.
    let (xd, scd, shd) = (x.detach(), scale.detach(), shift.detach());
    Tensor::custom_op("fused_affine_silu", &[x, scale, shift], out, move |g| {
        let z = xd.mul(&scd)?.add(&shd)?;
        let s = z.sigmoid()?;
        let dz = g.mul(&s.mul(&z.mul(&s.rsub_scalar(1.0)?)?.add_scalar(1.0)?)?)?;
        Ok(vec![Some(dz.mul(&scd)?), Some(dz.mul(&xd)?.sum(1, true)?), Some(dz.sum(1, true)?)])
    })
}

#[test]
fn custom_op_grad_checks() {
    let mut rng = Rng::seed(5);
    let x = Tensor::randn([3, 4], 0.0, 1.0, &mut rng);
    let scale = Tensor::randn([3, 1], 1.0, 0.3, &mut rng);
    let shift = Tensor::randn([3, 1], 0.0, 0.5, &mut rng);
    grad_check(|v| fused_affine_silu(&v[0], &v[1], &v[2]), &[x, scale, shift]).unwrap();
}

#[test]
fn custom_op_semantics() -> Result<()> {
    let x = Tensor::ones([2]).requires_grad_();
    let out = {
        let _g = no_grad();
        x.mul_scalar(2.0)?
    };
    // Wrong arity is reported at backward time.
    let bad = Tensor::custom_op("bad", &[&x], out.clone(), |g| Ok(vec![Some(g.clone()), None]))?;
    assert!(matches!(bad.sum_all()?.backward(), Err(Error::Autograd(_))));
    // Wrong gradient shape is reported too.
    let bad = Tensor::custom_op("bad_shape", &[&x], out.clone(), |_| Ok(vec![Some(Tensor::ones([3]))]))?;
    assert!(bad.sum_all()?.backward().is_err());
    // No input requires grad: nothing is recorded.
    let c = Tensor::custom_op("const", &[&Tensor::ones([2])], out.clone(), |_| Ok(vec![None]))?;
    assert!(!c.requires_grad());
    // The output is version-checked.
    let y = Tensor::custom_op("scale2", &[&x], out, |g| Ok(vec![Some(g.mul_scalar(2.0)?)]))?;
    let loss = y.sum_all()?;
    assert_eq!(loss.backward()?.get(&x).unwrap().to_vec::<f32>()?, vec![2.0, 2.0]);
    {
        let _g = no_grad();
        y.fill_assign(0.0)?;
    }
    assert!(matches!(loss.backward(), Err(Error::ModifiedInPlace { .. })));
    Ok(())
}

#[test]
fn custom_op_gradient_dtype_must_match() -> Result<()> {
    let x = Tensor::ones([1]).requires_grad_();
    let out = x.detach();
    let bad = Tensor::custom_op("u8_grad", &[&x], out, |_| Ok(vec![Some(Tensor::from_slice(&[1u8], [1])?)]))?;
    assert!(matches!(bad.sum_all()?.backward(), Err(Error::Autograd(_))));
    Ok(())
}
