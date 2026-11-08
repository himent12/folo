//! Autograd semantics and the backend registry contract.

use std::sync::Arc;

use folo_core::backend::{Backend, View, available_devices, register_backend};
use folo_core::{CpuStorage, DType, Device, DeviceStorage, Error, Layout, Result, Storage, Tensor, no_grad};

#[test]
fn grads_for_leaves_only() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3])?.requires_grad_();
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], [3])?;
    let h = a.mul(&b)?; // intermediate
    let loss = h.sum_all()?;
    let g = loss.backward()?;
    assert_eq!(g.get(&a).unwrap().to_vec::<f32>()?, vec![4.0, 5.0, 6.0]);
    assert!(g.get(&b).is_none(), "b does not require grad");
    assert!(g.get(&h).is_none(), "intermediates are not kept");
    assert_eq!(loss.grad_fn_name(), Some("sum"));
    assert!(a.is_leaf() && !h.is_leaf());
    Ok(())
}

#[test]
fn diamond_graph_and_repeated_backward() -> Result<()> {
    let x = Tensor::from_slice(&[2.0f32], [1])?.requires_grad_();
    let y = x.square()?; // x^2
    let z = y.mul(&x)?.add(&y)?; // x^3 + x^2 -> 3x^2 + 2x = 16 at x = 2
    let g1 = z.sum_all()?.backward()?;
    assert_eq!(g1.get(&x).unwrap().to_vec::<f32>()?, vec![16.0]);
    // Graphs are retained: backward again gives the same answer.
    let g2 = z.sum_all()?.backward()?;
    assert_eq!(g2.get(&x).unwrap().to_vec::<f32>()?, vec![16.0]);
    Ok(())
}

#[test]
fn no_grad_and_detach() -> Result<()> {
    let w = Tensor::ones([2]).requires_grad_();
    {
        let _g = no_grad();
        assert!(!folo_core::is_grad_enabled());
        let y = w.mul_scalar(3.0)?;
        assert!(!y.requires_grad());
        {
            let _g2 = no_grad();
        }
        assert!(!folo_core::is_grad_enabled(), "nested guard restores the outer state");
    }
    assert!(folo_core::is_grad_enabled());
    let d = w.detach();
    assert!(!d.requires_grad() && d.same_storage(&w));
    let loss = w.mul(&d)?.sum_all()?;
    assert_eq!(loss.backward()?.get(&w).unwrap().to_vec::<f32>()?, vec![1.0, 1.0]);
    assert!(Tensor::ones([2]).sum_all()?.backward().is_err(), "no grad to compute");
    assert!(w.mul_scalar(2.0)?.backward().is_err(), "backward needs a scalar");
    let seeded = w.mul_scalar(2.0)?.backward_with(&Tensor::from_slice(&[1.0f32, 10.0], [2])?)?;
    assert_eq!(seeded.get(&w).unwrap().to_vec::<f32>()?, vec![2.0, 20.0]);
    w.set_requires_grad(false);
    assert!(!w.requires_grad());
    Ok(())
}

#[test]
fn deep_graph_drops_without_stack_overflow() -> Result<()> {
    // Runs on a small-stack thread to make sure graph teardown is not recursive.
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| -> Result<()> {
            let x = Tensor::ones([1]).requires_grad_();
            let mut y = x.clone();
            for _ in 0..20_000 {
                y = y.add_scalar(1.0)?;
            }
            assert_eq!(y.item::<f32>()?, 20_001.0);
            drop(y);
            Ok(())
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn errors_name_the_op_and_shapes() {
    let a = Tensor::zeros([2, 3]);
    let e = a.matmul(&Tensor::zeros([4, 5])).unwrap_err().to_string();
    assert!(e.contains("matmul") && e.contains("[2, 3]") && e.contains("[4, 5]"), "{e}");
    let e = a.reshape([7, -1]).unwrap_err().to_string();
    assert!(e.contains("reshape"), "{e}");
    let e = a.sum(5, false).unwrap_err().to_string();
    assert!(e.contains("sum") && e.contains("rank-2"), "{e}");
    let e = Tensor::zeros([1, 3, 4, 4]).conv2d(&Tensor::zeros([2, 2, 3, 3]), None, 1, 1, 1, 1).unwrap_err();
    assert!(matches!(e, Error::ShapeMismatch { op: "conv2d", .. }), "{e}");
}

/// A fake device backend: implements only memory ops by keeping host vectors in its own buffer type.
struct FakeGpu;
struct FakeBuf(CpuStorage);

impl FakeGpu {
    fn host<'a>(&self, v: &View<'a>) -> &'a CpuStorage {
        match v.storage {
            Storage::Device(d) => &d.downcast_ref::<FakeBuf>().expect("our buffer").0,
            Storage::Cpu(_) => panic!("fake gpu got host storage"),
        }
    }
}

impl Backend for FakeGpu {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn full(&self, device: Device, dtype: DType, len: usize, value: f64) -> Result<Storage> {
        Ok(Storage::Device(DeviceStorage::new(
            device,
            dtype,
            len,
            FakeBuf(CpuStorage::full(dtype, len, value)),
        )))
    }
    fn upload(&self, device: Device, data: &CpuStorage) -> Result<Storage> {
        Ok(Storage::Device(DeviceStorage::new(device, data.dtype(), data.len(), FakeBuf(data.clone()))))
    }
    fn download(&self, src: View<'_>) -> Result<CpuStorage> {
        let host = Storage::Cpu(self.host(&src).clone());
        folo_core::backend::backend(Device::Cpu)?.download(View::new(&host, src.layout))
    }
    fn copy_strided(&self, src: View<'_>) -> Result<Storage> {
        let dev = src.storage.device();
        self.upload(dev, &self.download(src)?)
    }
    fn copy_into(&self, _dst: &mut Storage, _l: &Layout, _src: View<'_>) -> Result<()> {
        Err(Error::NotSupported { op: "copy_into", backend: "fake" })
    }
    fn cast(&self, src: View<'_>, dtype: DType) -> Result<Storage> {
        let host = Storage::Cpu(self.download(src)?);
        let l = Layout::contiguous(src.shape().clone());
        let c = folo_core::backend::backend(Device::Cpu)?.cast(View::new(&host, &l), dtype)?;
        let dev = src.storage.device();
        match c {
            Storage::Cpu(c) => self.upload(dev, &c),
            Storage::Device(_) => unreachable!(),
        }
    }
}

#[test]
fn backend_registry_contract() -> Result<()> {
    let dev = Device::Gpu(7);
    assert!(matches!(Tensor::ones([2]).to_device(dev), Err(Error::BackendNotRegistered(Device::Gpu(7)))));
    register_backend(dev, Arc::new(FakeGpu))?;
    assert!(register_backend(Device::Cpu, Arc::new(FakeGpu)).is_err());
    assert!(available_devices().contains(&dev));

    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2])?.requires_grad_();
    let g = x.to_device(dev)?;
    assert_eq!(g.device(), dev);
    // Views and downloads work through the memory methods.
    assert_eq!(g.t()?.to_vec::<f32>()?, vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(g.to_dtype(DType::U8)?.to_vec::<u8>()?, vec![1, 2, 3, 4]);
    // Unimplemented compute ops report NotSupported with the backend name.
    match g.exp() {
        Err(Error::NotSupported { op, backend }) => assert_eq!((op, backend), ("unary", "fake")),
        other => panic!("expected NotSupported, got {other:?}"),
    }
    assert!(matches!(g.add(&x), Err(Error::DeviceMismatch { .. })));
    // Round trip back to the CPU keeps the graph: gradients reach the CPU leaf.
    let back = g.to_device(Device::Cpu)?;
    let loss = back.mul(&back)?.sum_all()?;
    let grads = loss.backward()?;
    assert_eq!(grads.get(&x).unwrap().to_vec::<f32>()?, vec![2.0, 4.0, 6.0, 8.0]);
    Ok(())
}

#[test]
fn saved_indices_masks_and_outputs_are_version_checked() -> Result<()> {
    let expect_modified = |r: Result<folo_core::Grads>| match r {
        Err(Error::ModifiedInPlace { .. }) => {}
        other => panic!("expected ModifiedInPlace, got {other:?}"),
    };
    // Index tensor overwritten after index_select.
    let x = Tensor::from_slice(&[1.0f32, 2.0], [2])?.requires_grad_();
    let idx = Tensor::from_slice(&[0i64], [1])?;
    let y = x.index_select(0, &idx)?.sum_all()?;
    idx.copy_from(&Tensor::from_slice(&[1i64], [1])?)?;
    expect_modified(y.backward());
    // Mask overwritten after where_cond.
    let mask = Tensor::from_slice(&[1.0f32, 0.0], [2])?;
    let y = mask.where_cond(&x, &Tensor::scalar(0.0))?.sum_all()?;
    mask.fill_assign(1.0)?;
    expect_modified(y.backward());
    // The op's own output overwritten (softmax saves it for backward).
    let s = x.softmax(0)?;
    let loss = s.mul(&Tensor::from_slice(&[1.0f32, 0.0], [2])?)?.sum_all()?;
    {
        let _g = no_grad();
        s.detach().fill_assign(0.0)?;
    }
    expect_modified(loss.backward());
    // topk indices returned to the caller.
    let (v, i) = x.topk(1, true)?;
    let loss = v.sum_all()?;
    i.copy_from(&Tensor::from_slice(&[0i64], [1])?)?;
    expect_modified(loss.backward());
    Ok(())
}

#[test]
fn copy_into_rejects_bad_layouts() -> Result<()> {
    let b = folo_core::backend::backend(Device::Cpu)?;
    let mut dst = Storage::Cpu(CpuStorage::F32(vec![0.0; 511]));
    let src = Storage::Cpu(CpuStorage::F32(vec![1.0; 256 * 256]));
    let src_l = Layout::contiguous([256usize, 256]);
    let dims: folo_core::DimVec = [1usize, 1].into_iter().collect();
    let overlapping = Layout::new(folo_core::Shape::new(&[256, 256]), dims, 0);
    assert!(b.copy_into(&mut dst, &overlapping, View::new(&src, &src_l)).is_err());
    let out_of_bounds = Layout::contiguous([256usize, 256]);
    assert!(b.copy_into(&mut dst, &out_of_bounds, View::new(&src, &src_l)).is_err());
    Ok(())
}

#[test]
fn edge_case_gradients() -> Result<()> {
    // d/db (a / b) at tiny values stays finite.
    let a = Tensor::full([1], 1e-30);
    let b = Tensor::full([1], 1e-30).requires_grad_();
    let g = a.div(&b)?.sum_all()?.backward()?;
    let gb = g.get(&b).unwrap().item::<f32>()?;
    assert!(gb.is_finite() && (gb / -1e30 - 1.0).abs() < 1e-3, "{gb}");
    // 0^0 has a zero base gradient, not NaN.
    let base = Tensor::zeros([1]).requires_grad_();
    let g = base.pow(&Tensor::zeros([1]))?.sum_all()?.backward()?;
    assert_eq!(g.get(&base).unwrap().item::<f32>()?, 0.0);
    // Shared-weight matmul backward with an empty contraction dim.
    let x = Tensor::zeros([2, 3, 0]);
    let w = Tensor::zeros([0, 4]).requires_grad_();
    let g = x.matmul(&w)?.sum_all()?.backward()?;
    assert_eq!(g.get(&w).unwrap().dims(), &[0, 4]);
    // Empty spatial dims: upsampling returns an empty tensor, pooling reports an error.
    let e = Tensor::zeros([1, 1, 0, 3]).requires_grad_();
    let u = e.upsample_nearest2d(2)?;
    assert_eq!(u.dims(), &[1, 1, 0, 6]);
    assert_eq!(u.sum_all()?.backward()?.get(&e).unwrap().dims(), &[1, 1, 0, 3]);
    assert!(Tensor::zeros([1, 1, 0, 3]).max_pool2d(2, 2, 1).is_err());
    Ok(())
}

#[test]
fn reciprocal_gradients_do_not_square() -> Result<()> {
    let x = Tensor::full([1], 1e-30).requires_grad_();
    let g = x.rdiv_scalar(1e-30)?.sum_all()?.backward()?;
    let v = g.get(&x).unwrap().item::<f32>()?;
    assert!(v.is_finite() && (v / -1e30 - 1.0).abs() < 1e-3, "{v}");
    let g = x.recip()?.backward_with(&Tensor::full([1], 1e-30))?;
    let v = g.get(&x).unwrap().item::<f32>()?;
    assert!(v.is_finite() && (v / -1e30 - 1.0).abs() < 1e-3, "{v}");
    assert_eq!(Tensor::scalar(1e-39).div_scalar(1e-39)?.item::<f32>()?, 1.0);
    assert_eq!((Tensor::scalar(1e-39) / 1e-39)?.item::<f32>()?, 1.0);
    Ok(())
}

#[test]
fn sigmoid_tail_and_pow_subnormal_base() -> Result<()> {
    let x = Tensor::full([1], -90.0).requires_grad_();
    let y = x.silu()?;
    let v = y.item::<f32>()?;
    assert!(v < 0.0 && ((v as f64) / (-90.0 * (-90f64).exp()) - 1.0).abs() < 1e-3, "{v}");
    let g = y.backward_with(&Tensor::full([1], 1e38))?.get(&x).unwrap().item::<f32>()?;
    let want = 1e38 * (-89.0f64) * (-90f64).exp();
    assert!(((g as f64) / want - 1.0).abs() < 1e-3, "{g} vs {want}");
    // d/db a^b = a^b ln a must keep subnormal positive bases exact.
    let a = Tensor::full([1], 1e-40);
    let b = Tensor::zeros([1]).requires_grad_();
    let g = a.pow(&b)?.sum_all()?.backward()?.get(&b).unwrap().item::<f32>()?;
    assert!((g - (1e-40f64).ln() as f32).abs() < 1e-2, "{g}");
    Ok(())
}
