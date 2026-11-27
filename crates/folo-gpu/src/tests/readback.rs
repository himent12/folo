//! Tests for submission-targeted, deferred, and single-submission batched GPU readbacks.

use folo_core::{DType, Device, Tensor, no_grad};

use super::gpu;
use crate::{GpuContext, GpuError, GpuTensor, UnaryOp};

/// `read_deferred` lets subsequent GPU work be encoded and submitted before the earlier readback is collected,
/// and `read_batch` / `read_batch_async` download multiple ranges in a single queue submission.
#[test]
fn deferred_and_batched_buffer_readbacks() {
    let _shared = gpu!();
    let ctx = GpuContext::new().unwrap();

    let a_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let b_data: Vec<f32> = (0..512).map(|i| -(i as f32) * 0.25).collect();
    let c_bytes: Vec<u8> = (0..37).map(|i| (i * 19 % 251) as u8).collect();

    let a = GpuTensor::from_slice(&ctx, &a_data, &[1024]).unwrap();
    let b = GpuTensor::from_slice(&ctx, &b_data, &[512]).unwrap();
    let c = ctx.upload(&c_bytes).unwrap();

    // Deferred readback overlapped with subsequent GPU compute: compute step 1, start deferred readback,
    // immediately enqueue step 2 and flush it, then collect step 1's readback without losing accuracy.
    let step1 = a.unary(UnaryOp::Square).unwrap();
    let rb = ctx.read_deferred(step1.raw(), 0, (1024 * 4) as u64).unwrap();
    assert_eq!(rb.len(), 1024 * 4);
    assert!(!rb.is_empty());

    let mut step2 = b.clone();
    for _ in 0..8 {
        step2 = step2.unary(UnaryOp::Relu).unwrap();
    }
    ctx.flush();

    let step1_bytes = rb.into_vec().unwrap();
    let step1_got: &[f32] = bytemuck::cast_slice(&step1_bytes);
    for (g, &x) in step1_got.iter().zip(&a_data) {
        assert_eq!(*g, x * x);
    }

    // Batched readback of 4 ranges (including empty, non-4-byte length, and duplicate source buffer):
    // must take exactly 1 queue submission.
    let before = ctx.work_stats();
    let batch = ctx
        .read_batch(&[
            (step2.raw(), 0, (512 * 4) as u64),
            (c.raw(), 0, 0),
            (c.raw(), 0, c_bytes.len() as u64),
            (a.raw(), 16, 32),
        ])
        .unwrap();
    let after = ctx.work_stats();
    assert_eq!(after.submissions - before.submissions, 1, "read_batch must submit once");
    assert_eq!(batch.len(), 4);
    let step2_got: &[f32] = bytemuck::cast_slice(&batch[0]);
    for (g, &x) in step2_got.iter().zip(&b_data) {
        assert_eq!(*g, x.max(0.0));
    }
    assert!(batch[1].is_empty());
    assert_eq!(batch[2], c_bytes);
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&batch[3]), &a_data[4..12]);

    // Async batched readback matches byte for byte and also submits once.
    let before = ctx.work_stats();
    let async_batch = pollster::block_on(ctx.read_batch_async(&[
        (a.raw(), 0, 64),
        (c.raw(), 0, c_bytes.len() as u64),
        (a.raw(), 0, 0),
    ]))
    .unwrap();
    let after = ctx.work_stats();
    assert_eq!(after.submissions - before.submissions, 1, "read_batch_async must submit once");
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&async_batch[0]), &a_data[..16]);
    assert_eq!(async_batch[1], c_bytes);
    assert!(async_batch[2].is_empty());
}

/// Validation errors in `read_batch_deferred` (including missing `COPY_SRC` usage and overflowing lengths)
/// reject atomically before encoding or submitting anything, and never corrupt an earlier in-flight readback.
#[test]
fn read_batch_validation_is_atomic() {
    let _shared = gpu!();
    let ctx = GpuContext::new().unwrap();
    ctx.set_wait_spin(std::time::Duration::ZERO);
    let buf = ctx.upload(&[1u8; 64]).unwrap();
    let no_copy_src = ctx.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("no copy src"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.sync().unwrap();

    // Start an earlier valid deferred readback, then reject invalid batches (including missing `COPY_SRC` and
    // `u64::MAX` length overflow) before collecting the earlier handle.
    let valid_in_flight = ctx.read_deferred(buf.raw(), 0, 64).unwrap();
    let before = ctx.work_stats();
    // Missing COPY_SRC usage on source buffer.
    let err = ctx.read_batch_deferred(&[(buf.raw(), 0, 16), (&no_copy_src, 0, 16)]).err();
    assert!(matches!(err, Some(GpuError::Shape(_))), "missing COPY_SRC must fail preflight: {err:?}");
    // Unaligned offset on second entry.
    let err = ctx.read_batch_deferred(&[(buf.raw(), 0, 16), (buf.raw(), 2, 8)]).err();
    assert!(matches!(err, Some(GpuError::Shape(_))));
    // Out-of-bounds length on second entry.
    let err = ctx.read_batch_deferred(&[(buf.raw(), 0, 16), (buf.raw(), 0, 1024)]).err();
    assert!(matches!(err, Some(GpuError::Shape(_))));
    // Overflowing `next_multiple_of` length.
    let err = ctx.read_batch_deferred(&[(buf.raw(), 0, u64::MAX - 1)]).err();
    assert!(matches!(err, Some(GpuError::Shape(_))));
    let after = ctx.work_stats();
    assert_eq!(after.submissions, before.submissions, "rejected batch must not submit");
    assert_eq!(valid_in_flight.into_vec().unwrap(), vec![1u8; 64]);

    // Empty / all-zero batch submits nothing and returns immediately-ready handles.
    let before_empty = ctx.work_stats();
    let empty = ctx.read_batch_deferred(&[(buf.raw(), 0, 0), (buf.raw(), 4, 0)]).unwrap();
    assert_eq!(ctx.work_stats().submissions, before_empty.submissions);
    for rb in empty {
        assert!(rb.is_empty());
        assert!(rb.poll_ready().unwrap());
        assert!(rb.into_vec().unwrap().is_empty());
    }
}

/// Dropping an unfinished `Readback` retires its staging buffer instead of returning it to the reuse pool while
/// the original submission may still be in flight, so an immediately following readback of the same size class
/// can never observe the abandoned submission's bytes.
#[test]
fn dropped_deferred_readback_unmaps_cleanly() {
    let _shared = gpu!();
    let ctx = GpuContext::new().unwrap();
    let n = 4096usize;
    let bytes = (n * 4) as u64;
    let a = GpuTensor::from_slice(&ctx, &vec![1.0f32; n], &[n]).unwrap();
    let b = GpuTensor::from_slice(&ctx, &vec![2.0f32; n], &[n]).unwrap();
    let one = GpuTensor::from_slice(&ctx, &[0.0f32], &[1]).unwrap();
    ctx.sync().unwrap();

    for _ in 0..8 {
        // Queue nontrivial work before the abandoned readback so its submission is still executing when dropped.
        let mut slow = a.clone();
        for _ in 0..32 {
            slow = slow.add(&one).unwrap();
        }
        let abandoned = ctx.read_deferred(slow.raw(), 0, bytes).unwrap();
        drop(abandoned);

        // Immediately read a different buffer of the exact same size class: must return `2.0`, never `1.0`.
        let mut second = b.clone();
        for _ in 0..4 {
            second = second.add(&one).unwrap();
        }
        let got_bytes = ctx.read(second.raw(), 0, bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&got_bytes);
        assert!(got.iter().all(|&v| v == 2.0), "stale bytes from abandoned readback leaked into next read");
    }
    ctx.sync().unwrap();
}

/// `poll_ready` and `finish` / `finish_async` on a `Readback` report `DeviceLost` when the device is destroyed.
#[test]
fn deferred_readback_reports_device_loss() {
    let _shared = gpu!();
    let ctx = GpuContext::new().unwrap();
    let buf = ctx.upload(&[7u8; 64]).unwrap();
    ctx.sync().unwrap();

    let rb = ctx.read_deferred(buf.raw(), 0, 64).unwrap();
    let rb_after = ctx.read_deferred(buf.raw(), 0, 64).unwrap();
    ctx.destroy_device();
    // Non-blocking `poll_ready` must never panic on a destroyed device; `into_vec` must report `DeviceLost`.
    let _ = rb.poll_ready();
    let finished = rb.into_vec();
    assert!(
        matches!(finished, Err(GpuError::DeviceLost(_))),
        "expected DeviceLost on finish, got {finished:?}"
    );
    // Once device loss is recorded, `poll_ready` reports `DeviceLost` deterministically.
    assert!(matches!(rb_after.poll_ready(), Err(GpuError::DeviceLost(_))));
    assert!(matches!(rb_after.into_vec(), Err(GpuError::DeviceLost(_))));

    let ctx2 = GpuContext::new().unwrap();
    let buf2 = ctx2.upload(&[9u8; 64]).unwrap();
    ctx2.sync().unwrap();
    ctx2.destroy_device();
    let res2 = ctx2.read_deferred(buf2.raw(), 0, 64).and_then(|r| pollster::block_on(r.into_vec_async()));
    assert!(matches!(res2, Err(GpuError::DeviceLost(_))), "expected DeviceLost, got {res2:?}");
}

/// `download_deferred`, `download_many`, and `download_many_async` handle contiguous, strided, unaligned-offset,
/// empty, CPU, and duplicate GPU tensors in a single queue submission.
#[test]
fn tensor_deferred_and_batched_downloads() {
    let _g = no_grad();
    let dev = match crate::init() {
        Ok(d) => d,
        Err(_) => return,
    };
    let Some(ctx) = crate::context_for(dev) else { return };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        return;
    }

    let cpu_a = Tensor::from_slice(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), [4, 6]).unwrap();
    let gpu_a = cpu_a.to_device(dev).unwrap();
    let gpu_strided = gpu_a.transpose(0, 1).unwrap();
    let cpu_u8 = Tensor::from_slice(&(0u8..10).collect::<Vec<_>>(), [10]).unwrap();
    let gpu_u8_unaligned = cpu_u8.to_device(dev).unwrap().narrow(0, 1, 7).unwrap();
    let gpu_empty = Tensor::zeros([0, 4]).to_device(dev).unwrap();
    let cpu_only = Tensor::from_slice(&[1.5f32, -2.5, 3.25], [3]).unwrap();

    // `download_many` downloads all 6 tensors (including strided gather + unaligned u8 gather) in 1 submission.
    let got =
        crate::download_many(&ctx, &[&gpu_a, &gpu_strided, &gpu_u8_unaligned, &gpu_empty, &cpu_only, &gpu_a])
            .unwrap();
    assert_eq!(got.len(), 6);
    assert_eq!(got[0].device(), Device::Cpu);
    assert_eq!(got[0].to_vec::<f32>().unwrap(), cpu_a.to_vec::<f32>().unwrap());
    assert_eq!(got[1].to_vec::<f32>().unwrap(), cpu_a.transpose(0, 1).unwrap().to_vec::<f32>().unwrap());
    assert_eq!(got[2].dtype(), DType::U8);
    assert_eq!(got[2].to_vec::<u8>().unwrap(), vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(got[3].dims(), &[0, 4]);
    assert_eq!(got[3].to_vec::<f32>().unwrap(), Vec::<f32>::new());
    assert_eq!(got[4].to_vec::<f32>().unwrap(), vec![1.5, -2.5, 3.25]);
    assert_eq!(got[5].to_vec::<f32>().unwrap(), cpu_a.to_vec::<f32>().unwrap());

    // `download_many_async` matches too.
    let got_async =
        pollster::block_on(crate::download_many_async(&ctx, &[&gpu_strided, &gpu_u8_unaligned])).unwrap();
    assert_eq!(
        got_async[0].to_vec::<f32>().unwrap(),
        cpu_a.transpose(0, 1).unwrap().to_vec::<f32>().unwrap()
    );
    assert_eq!(got_async[1].to_vec::<u8>().unwrap(), vec![1, 2, 3, 4, 5, 6, 7]);

    // `download_deferred` overlapped with next GPU op.
    let dl = crate::download_deferred(&ctx, &gpu_a).unwrap();
    let next = gpu_a.mul_scalar(2.0).unwrap();
    ctx.flush();
    let res0 = dl.finish().unwrap();
    assert_eq!(res0.to_vec::<f32>().unwrap(), cpu_a.to_vec::<f32>().unwrap());
    let res1 = crate::download_deferred(&ctx, &next).unwrap().finish().unwrap();
    assert_eq!(
        res1.to_vec::<f32>().unwrap(),
        cpu_a.to_vec::<f32>().unwrap().into_iter().map(|v| v * 2.0).collect::<Vec<_>>()
    );
}
