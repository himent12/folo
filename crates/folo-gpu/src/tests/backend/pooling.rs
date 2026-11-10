//! Max, average and adaptive pooling and nearest upsampling.

use folo_core::Tensor;

use super::fixtures::{assert_close_t, check, dev, randn};

#[test]
fn pooling_and_upsampling() {
    let dev = dev!();
    let x = randn(&[2, 3, 11, 13], 70);
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool k3 s2 p1", |v| v[0].max_pool2d(3, 2, 1));
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool k5 s1 p2 (SPPF)", |v| v[0].max_pool2d(5, 1, 2));
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool k2 s2", |v| v[0].max_pool2d(2, 2, 0));
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "avg_pool k2 s2", |v| v[0].avg_pool2d(2, 2, 0));
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "avg_pool k3 s1 p1", |v| v[0].avg_pool2d(3, 1, 1));
    for hw in [[1usize, 1], [3, 5], [11, 13], [20, 7]] {
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("adaptive {hw:?}"), move |v| {
            v[0].adaptive_avg_pool2d(hw)
        });
    }
    // Window arithmetic that would wrap u32 is rejected rather than computed wrongly (65536 · 65537 > 2^32).
    let tall = Tensor::ones([1, 1, 65536, 1]).to_device(dev).unwrap();
    let e = tall.adaptive_avg_pool2d([65536, 1]).unwrap_err().to_string();
    assert!(e.contains("overflows"), "{e}");
    assert!(tall.adaptive_avg_pool2d([1, 1]).is_ok());
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "upsample 2x", |v| v[0].upsample_nearest2d(2));
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "upsample 2x3", |v| v[0].upsample_nearest2d_hw([2, 3]));
    // Huge gradients: each window's share is scaled before it is summed, so nothing overflows.
    let small = randn(&[1, 1, 3, 3], 71);
    let (cs, gs) = (small.detach().requires_grad_(), small.to_device(dev).unwrap().requires_grad_());
    let big = Tensor::full([1, 1, 3, 3], 1e38);
    let gc = cs.avg_pool2d(3, 1, 1).unwrap().backward_with(&big).unwrap();
    let gg = gs.avg_pool2d(3, 1, 1).unwrap().backward_with(&big.to_device(dev).unwrap()).unwrap();
    assert_close_t(gg.get(&gs).unwrap(), gc.get(&cs).unwrap(), 0.0, 1e-6, "avg_pool backward of 1e38");
    // Ties: constant planes pick the first element of each window.
    let flat = Tensor::ones([1, 1, 4, 4]);
    check(dev, &[flat], 0.0, 0.0, "max_pool ties", |v| v[0].max_pool2d(2, 1, 0));
    // A transposed (non-dense) input.
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool of transposed", |v| {
        v[0].transpose(2, 3)?.max_pool2d(3, 2, 1)
    });
}
