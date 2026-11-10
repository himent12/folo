//! Matmul on strided and batched operands.

use folo_core::Tensor;

use super::fixtures::{check, dev, randn};

#[test]
fn matmul_layouts() {
    let dev = dev!();
    let a = randn(&[5, 37, 19], 20);
    let b = randn(&[19, 23], 21);
    let bb = randn(&[5, 19, 23], 22);
    check(dev, &[a.clone(), b.clone()], 1e-4, 1e-4, "shared rhs (folded)", |v| v[0].matmul(&v[1]));
    check(dev, &[a.clone(), bb.clone()], 1e-4, 1e-4, "batched", |v| v[0].matmul(&v[1]));
    check(dev, &[a.clone(), bb.clone()], 1e-4, 1e-4, "transposed views", |v| {
        v[0].transpose(1, 2)?
            .contiguous()?
            .transpose(1, 2)?
            .matmul(&v[1].transpose(1, 2)?.contiguous()?.transpose(1, 2)?)
    });
    check(dev, &[bb.clone(), a.clone()], 1e-4, 1e-4, "both transposed", |v| {
        v[0].transpose(1, 2)?.matmul(&v[1].transpose(1, 2)?)
    });
    let m4 = randn(&[2, 1, 6, 8], 23);
    let n4 = randn(&[3, 8, 4], 24);
    check(dev, &[m4, n4], 1e-4, 1e-4, "4-d broadcast batch", |v| v[0].matmul(&v[1]));
    let big = randn(&[64, 128], 25);
    let w = randn(&[128, 96], 26);
    check(dev, &[big.clone(), w.clone()], 1e-4, 1e-4, "aligned 2-d", |v| v[0].matmul(&v[1]));
    check(dev, &[big, w], 1e-4, 1e-4, "offset slices", |v| {
        v[0].narrow(1, 3, 101)?.matmul(&v[1].narrow(0, 1, 101)?.narrow(1, 2, 90)?)
    });
    let v1 = randn(&[1, 7], 27);
    let v2 = randn(&[7, 1], 28);
    check(dev, &[v1.clone(), v2.clone()], 1e-5, 1e-5, "dot", |v| v[0].matmul(&v[1]));
    check(dev, &[v2, v1], 1e-5, 1e-5, "outer", |v| v[0].matmul(&v[1]));
    let e1 = Tensor::zeros([3, 0]);
    let e2 = Tensor::zeros([0, 4]);
    check(dev, &[e1, e2], 0.0, 0.0, "k = 0", |v| v[0].matmul(&v[1]));
}

#[test]
fn matmul_batches_beyond_one_dispatch() {
    // More batch matrices than one dispatch's 65535 workgroup layers: one level (per-anchor dot products),
    // two levels (a transposed batch that does not coalesce) with the outer and with the inner count too large.
    let dev = dev!();
    let a = randn(&[70000, 1, 3], 90);
    let b = randn(&[70000, 3, 1], 91);
    check(dev, &[a, b], 1e-5, 1e-5, "batch 70000", |v| v[0].matmul(&v[1]));
    let a = randn(&[300, 300, 2, 3], 92);
    let b = randn(&[300, 300, 3, 2], 93);
    check(dev, &[a, b], 1e-5, 1e-5, "batch 300 x 300", |v| {
        v[0].transpose(0, 1)?.matmul(&v[1].transpose(0, 1)?)
    });
    let a = randn(&[70000, 2, 1, 2], 94);
    let b = randn(&[2, 2, 3], 95);
    // The broadcast operand's gradient sums 70000 products: reduction tolerance.
    check(dev, &[a, b], 1e-3, 1e-4, "batch 2 x 70000", |v| {
        v[0].transpose(0, 1)?.matmul(&v[1].reshape([2, 1, 2, 3])?.expand([2, 70000, 2, 3])?)
    });
}
