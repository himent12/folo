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
                        let staged = ctx.stage(&b.images_u8).map_err(|e| Error::Core(e.into()))?;
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
    let buf = ctx.upload_staged(staged).map_err(|e| Error::Core(e.into()))?;
    let x = folo_gpu::tensor_from_buffer(device, DType::U8, shape, buf)?;
    Ok(x.to_dtype(DType::F32)?.div_scalar(255.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
    use folo_data::{DataLoader, LoaderConfig, Split};

    /// A batch staged on a lost device comes out of the feeder as a folo-core error that folo-gpu recognises as
    /// device loss (so the trainer names the checkpoint to resume from), not as a generic failure.
    #[test]
    fn feeder_reports_a_lost_device() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("note: skipping GPU test, no adapter");
            return;
        };
        if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
            eprintln!("note: skipping GPU test on a software adapter");
            return;
        }
        let cfg = SyntheticConfig { images: 4, imgsz: 64, ..SyntheticConfig::default() };
        let ds = std::sync::Arc::new(SyntheticDataset::new(cfg, Split::Train).unwrap());
        let lc =
            LoaderConfig { batch_size: 2, imgsz: 64, workers: 1, u8_images: true, ..LoaderConfig::default() };
        let loader = DataLoader::new(ds, lc).unwrap();
        ctx.destroy_device();
        let mut feed = Feeder::new(loader.epoch(0), ctx).unwrap();
        match feed.next() {
            Some(Err(Error::Core(e))) => assert!(folo_gpu::is_device_lost(&e), "{e}"),
            other => panic!("expected a device-lost error, got {:?}", other.map(|r| r.map(|_| ()))),
        }
        assert!(feed.next().is_none(), "the feeder stops after the error");
    }
}
