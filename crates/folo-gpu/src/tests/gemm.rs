//! GEMM tests: every transpose combination, odd sizes, batches, bias + activation, f16.

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::gemm::{self, BatchStrides, GemmBias, GemmDesc, TileConfig};
use crate::{DType, GpuTensor, UnaryOp};

/// Element count an operand needs: offset of the last batch plus one matrix.
fn operand_len(s: &BatchStrides, batch: (usize, usize), rows: usize, cols: usize, ld: usize) -> usize {
    s.offset + s.outer * (batch.0 - 1) + s.inner * (batch.1 - 1) + (rows - 1) * ld + cols
}

/// Reference for `d` (row-major, leading dims and batch strides from `d`), accumulating in f64.
pub(crate) fn cpu_gemm(d: &GemmDesc, a: &[f32], b: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut c = vec![0f32; operand_len(&d.c, d.batch, d.m, d.n, d.ldc)];
    for b1 in 0..d.batch.0 {
        for b2 in 0..d.batch.1 {
            let oa = d.a.offset + b1 * d.a.outer + b2 * d.a.inner;
            let ob = d.b.offset + b1 * d.b.outer + b2 * d.b.inner;
            let oc = d.c.offset + b1 * d.c.outer + b2 * d.c.inner;
            let obias = d.bias_strides.offset + b1 * d.bias_strides.outer + b2 * d.bias_strides.inner;
            for i in 0..d.m {
                for j in 0..d.n {
                    let mut s = 0f64;
                    for k in 0..d.k {
                        let av = if d.trans_a { a[oa + k * d.lda + i] } else { a[oa + i * d.lda + k] };
                        let bv = if d.trans_b { b[ob + j * d.ldb + k] } else { b[ob + k * d.ldb + j] };
                        s += av as f64 * bv as f64;
                    }
                    let mut v = s as f32 * d.alpha;
                    match d.bias {
                        GemmBias::Row => v += bias.unwrap()[obias + i],
                        GemmBias::Col => v += bias.unwrap()[obias + j],
                        GemmBias::None => {}
                    }
                    c[oc + i * d.ldc + j] = d.act.apply(v);
                }
            }
        }
    }
    c
}

fn run(d: &GemmDesc, cfg: Option<TileConfig>, seed: u64) {
    let ctx = gpu!();
    if d.dtype == DType::F16 && !ctx.caps().shader_f16 {
        eprintln!("note: skipping f16 GEMM test, adapter lacks shader-f16");
        return;
    }
    let mut rng = Rng::new(seed);
    let (ar, ac) = if d.trans_a { (d.k, d.m) } else { (d.m, d.k) };
    let (br, bc) = if d.trans_b { (d.n, d.k) } else { (d.k, d.n) };
    let batches = d.batch.0 * d.batch.1;
    let mut a = rng.vec(operand_len(&d.a, d.batch, ar, ac, d.lda), -1.0, 1.0);
    let mut b = rng.vec(operand_len(&d.b, d.batch, br, bc, d.ldb), -1.0, 1.0);
    let bias_len = match d.bias {
        GemmBias::None => 1,
        GemmBias::Row => d.bias_strides.offset + d.m * batches,
        GemmBias::Col => d.bias_strides.offset + d.n * batches,
    };
    let mut bias = rng.vec(bias_len, -1.0, 1.0);
    if d.dtype == DType::F16 {
        a = f16_round(&a);
        b = f16_round(&b);
        bias = f16_round(&bias);
    }
    let up = |v: &[f32]| match d.dtype {
        DType::F32 => GpuTensor::from_slice(&ctx, v, &[v.len()]).unwrap(),
        DType::F16 => GpuTensor::from_f16(&ctx, &to_f16(v), &[v.len()]).unwrap(),
    };
    let want = cpu_gemm(d, &a, &b, Some(&bias));
    let (ga, gb, gbias) = (up(&a), up(&b), up(&bias));
    let gc = GpuTensor::zeros(&ctx, &[want.len()], d.dtype).unwrap();
    let bias_buf = (d.bias != GemmBias::None).then(|| gbias.raw());
    match cfg {
        Some(cfg) => gemm::gemm_with_config(&ctx, d, ga.raw(), gb.raw(), bias_buf, gc.raw(), cfg).unwrap(),
        None => gemm::gemm(&ctx, d, ga.raw(), gb.raw(), bias_buf, gc.raw()).unwrap(),
    }
    let got = gc.to_vec().unwrap();
    // f32: inputs in [-1, 1], f32 FMA accumulation vs an f64 reference. f16: the output is rounded to 11 bits.
    let (atol, rtol) = if d.dtype == DType::F16 { (2e-3, 2e-3) } else { (2e-5 * (d.k as f32).sqrt(), 1e-5) };
    assert_close(&got, &want, atol, rtol, &format!("gemm {d:?} {cfg:?}"));
}

#[test]
fn gemm_all_transposes_odd_sizes() {
    let shapes = [(1, 1, 1), (3, 5, 7), (67, 45, 33), (130, 129, 17), (64, 64, 64), (128, 256, 48)];
    for (seed, (m, n, k)) in shapes.into_iter().enumerate() {
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            for cfg in [TileConfig::SMALL, TileConfig::MEDIUM, TileConfig::LARGE] {
                run(&GemmDesc::new(m, n, k, ta, tb, 1), Some(cfg), seed as u64);
            }
        }
    }
}

#[test]
fn gemm_vec4_aligned_with_ragged_tiles() {
    // Multiples of 4 (vec4 loads) but not of the tile sizes, so bounds checks and vec4 paths combine.
    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
        run(&GemmDesc::new(132, 68, 52, ta, tb, 1), Some(TileConfig::LARGE), 7);
        run(&GemmDesc::new(132, 68, 52, ta, tb, 1), Some(TileConfig::SMALL), 8);
        run(&GemmDesc::new(132, 260, 52, ta, tb, 1), Some(TileConfig::MEDIUM), 9);
    }
}

#[test]
fn gemm_batched_and_strided() {
    run(&GemmDesc::new(33, 20, 12, false, false, 3), None, 11);
    run(&GemmDesc::new(32, 64, 16, true, true, 5), None, 12);
    // Shared A across the outer batch (stride 0), two-level batch, padded leading dims and offsets.
    let mut d = GemmDesc::new(24, 40, 20, false, false, 2);
    d.batch = (2, 3);
    d.a = BatchStrides { offset: 4, outer: 0, inner: 24 * 20 };
    d.ldb = 44;
    d.b = BatchStrides { offset: 0, outer: 3 * 20 * 44, inner: 20 * 44 };
    d.c = BatchStrides { offset: 8, outer: 3 * 24 * 40, inner: 24 * 40 };
    run(&d, None, 13);
    // Unaligned offset forces scalar loads.
    d.a.offset = 3;
    run(&d, Some(TileConfig::LARGE), 14);
}

#[test]
fn gemm_bias_activation_alpha() {
    let mut d = GemmDesc::new(70, 50, 30, false, false, 1);
    d.bias = GemmBias::Row;
    d.act = UnaryOp::Silu;
    d.alpha = 0.5;
    run(&d, None, 21);
    d.bias = GemmBias::Col;
    d.act = UnaryOp::Relu;
    run(&d, Some(TileConfig::LARGE), 22);
    let mut d = GemmDesc::new(64, 64, 32, false, true, 2);
    d.bias = GemmBias::Col;
    d.bias_strides = BatchStrides { offset: 0, outer: 64, inner: 0 };
    d.act = UnaryOp::LeakyRelu(0.1);
    run(&d, None, 23);
}

#[test]
fn gemm_f16() {
    for (ta, tb) in [(false, false), (true, true)] {
        let mut d = GemmDesc::new(96, 72, 40, ta, tb, 2);
        d.dtype = DType::F16;
        run(&d, None, 31);
        let mut d = GemmDesc::new(37, 29, 19, ta, tb, 1);
        d.dtype = DType::F16;
        d.bias = GemmBias::Row;
        d.act = UnaryOp::Sigmoid;
        run(&d, Some(TileConfig::LARGE), 32);
    }
}

#[test]
fn matmul_tensor_api() {
    let ctx = gpu!();
    let mut rng = Rng::new(41);
    let (a, b) = (rng.vec(2 * 3 * 5, -1.0, 1.0), rng.vec(2 * 5 * 4, -1.0, 1.0));
    let ga = GpuTensor::from_slice(&ctx, &a, &[2, 3, 5]).unwrap();
    let gb = GpuTensor::from_slice(&ctx, &b, &[2, 5, 4]).unwrap();
    let got = ga.matmul(&gb).unwrap();
    assert_eq!(got.shape(), &[2, 3, 4]);
    let want = cpu_gemm(&GemmDesc::new(3, 4, 5, false, false, 2), &a, &b, None);
    assert_close(&got.to_vec().unwrap(), &want, 1e-5, 1e-5, "matmul");
    // A 2-D rhs broadcasts over the batch.
    let gw = GpuTensor::from_slice(&ctx, &b[..20], &[5, 4]).unwrap();
    let got = ga.matmul(&gw).unwrap().to_vec().unwrap();
    let mut d = GemmDesc::new(3, 4, 5, false, false, 2);
    d.b = BatchStrides::default();
    assert_close(&got, &cpu_gemm(&d, &a, &b[..20], None), 1e-5, 1e-5, "matmul broadcast rhs");
    // Transposed rhs.
    let gbt = GpuTensor::from_slice(&ctx, &b[..20], &[4, 5]).unwrap();
    let got = ga.matmul_t(&gbt, false, true).unwrap().to_vec().unwrap();
    let mut d = GemmDesc::new(3, 4, 5, false, true, 2);
    d.b = BatchStrides::default();
    assert_close(&got, &cpu_gemm(&d, &a, &b[..20], None), 1e-5, 1e-5, "matmul_t");
    // Shape errors are reported, not panics.
    assert!(ga.matmul(&ga).is_err());
}
