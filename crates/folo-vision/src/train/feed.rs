//! Feeding batches to a GPU: a thread takes the loader's `u8` batches and stages each one's pixels in a mapped
//! staging buffer ([`folo_gpu::GpuContext::stage`]) while the current step runs, so the training thread only
//! encodes a buffer copy and a `u8 → f32 / 255` conversion per batch.

use std::sync::mpsc::{Receiver, sync_channel};
use std::thread::JoinHandle;

use folo_core::{DType, Device, Tensor};
use folo_data::{Batch, EpochIter};
use folo_gpu::{GpuContext, Staged};

use crate::error::{Error, Result};

/// One epoch's batches, staged ahead on a thread. Dropping it stops the thread (and the loader).
pub(crate) struct Feeder {
    rx: Option<Receiver<Result<(Batch, Staged)>>>,
    thread: Option<JoinHandle<()>>,
}

impl Feeder {
    /// Stage `iter`'s batches for `ctx`, keeping one batch ahead of the consumer (plus the one being staged).
    pub(crate) fn new(iter: EpochIter, ctx: GpuContext) -> Result<Self> {
        let (tx, rx) = sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("folo-feed".into())
            .spawn(move || {
                for b in iter {
                    let item = b.map_err(Error::from).and_then(|mut b| {
                        let staged = ctx.stage(&b.images_u8).map_err(|e| Error::Failed(e.to_string()))?;
                        b.images_u8 = Vec::new();
                        Ok((b, staged))
                    });
                    let failed = item.is_err();
                    if tx.send(item).is_err() || failed {
                        return;
                    }
                }
            })
            .map_err(|e| Error::Failed(format!("failed to spawn the batch feeder: {e}")))?;
        Ok(Feeder { rx: Some(rx), thread: Some(thread) })
    }
}

impl Iterator for Feeder {
    type Item = Result<(Batch, Staged)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.as_ref()?.recv().ok()
    }
}

impl Drop for Feeder {
    fn drop(&mut self) {
        // Closing the channel makes the thread's next send fail; it then drops the loader and exits.
        self.rx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The `[n, 3, h, w]` f32 input in `0..1` of a staged `u8` batch: one copy to the device, then a cast and the
/// division by 255 there (the loader's `i / 255.0`).
pub(crate) fn staged_tensor(
    ctx: &GpuContext,
    device: Device,
    staged: &Staged,
    shape: [usize; 4],
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    if staged.len() != n as u64 {
        return Err(Error::Failed(format!("staged batch holds {} bytes, expected {n}", staged.len())));
    }
    let buf = ctx.upload_staged(staged).map_err(|e| Error::Failed(e.to_string()))?;
    let x = folo_gpu::tensor_from_buffer(device, DType::U8, shape, buf)?;
    Ok(x.to_dtype(DType::F32)?.div_scalar(255.0)?)
}
