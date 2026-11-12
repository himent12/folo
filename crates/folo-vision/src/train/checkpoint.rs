//! Checkpoints: [`Trainer::save_checkpoint`] (the complete FORMAT.md training state), the host snapshot of it the
//! epoch's files are written from (`train/writer/`), and the batched device-to-host download behind both.

use std::collections::BTreeMap;
use std::path::Path;

use folo_core::Device;
use folo_core::rng::Rng;
use folo_nn::EmaState;
use folo_nn::io::{CheckpointParts, FileContents, Metrics, RngState};

use super::{
    AMP_KEY, CLOSE_MOSAIC_KEY, NONFINITE_STREAK_KEY, TRAIN_IMAGES_KEY, Trainer, f32_decimal, rfc3339_now,
};
use crate::error::{Error, Result};

impl Trainer {
    /// Write the complete training state (FORMAT.md checkpoint) to `path`, atomically.
    ///
    /// Only possible at an epoch boundary (FORMAT.md v1 stores no loader position or pending gradients): before
    /// the first epoch, between epochs, or after `run` returned at one. A trainer stopped or failed inside an
    /// epoch refuses.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let (map, tensors) = self.checkpoint_snapshot()?;
        Ok(folo_nn::io::safetensors::write_file(path.as_ref(), Some(&map), &tensors)?)
    }

    /// The complete training state as [`Trainer::save_checkpoint`] writes it, with every tensor on the host: what
    /// the checkpoint writer thread is handed (`train/writer/`). Only at an epoch boundary.
    pub(super) fn checkpoint_snapshot(&self) -> Result<FileContents> {
        if !self.at_boundary {
            return Err(Error::Config(format!(
                "a checkpoint can only be written at an epoch boundary; this run stopped inside epoch {}",
                self.epoch + 1
            )));
        }
        let mut meta = self.model.metadata()?;
        meta.metrics = self.best_metrics.as_ref().map(|m| Metrics {
            map50: m.map50 as f64,
            map50_95: m.map50_95 as f64,
            precision: m.precision as f64,
            recall: m.recall as f64,
            best_f1_conf: m.best_f1_conf.map(f32_decimal),
        });
        meta.created = Some(rfc3339_now());
        meta.extra = self.extra_meta.clone();
        meta.extra.insert(NONFINITE_STREAK_KEY.into(), self.consecutive_bad.to_string());
        if let Some(s) = &self.scaler {
            meta.extra.insert(AMP_KEY.into(), s.to_json());
        }
        let mut streams = BTreeMap::new();
        // Every random draw after initialisation is derived from (seed, epoch, ...), so the seed is the whole
        // RNG state; the init stream is recorded for completeness.
        streams.insert("init".to_string(), Rng::seed(self.cfg.seed));
        let rng = RngState { seed: self.cfg.seed, streams };
        let mut train = serde_json::to_value(&self.cfg).expect("config serialises");
        train[TRAIN_IMAGES_KEY] = serde_json::json!(self.data.train.len());
        train[CLOSE_MOSAIC_KEY] = serde_json::json!(self.cfg.augment.close_mosaic_epochs(self.cfg.epochs));
        let parts = CheckpointParts {
            model: &self.model,
            ema: self.ema.as_ref().map(|e| e as &dyn EmaState),
            optimizer: self.opt.as_ref(),
            schedule: &self.sched,
            rng: &rng,
            epoch: self.epoch as u64,
            best: self.best,
            train_config: &train,
        };
        let (map, mut tensors) = folo_nn::io::checkpoint_contents(&meta, &parts)?;
        // Tensors already on the host are the live ones (a CPU run's weights, BatchNorm statistics, optimizer
        // state): the writer thread writes the snapshot while the next epoch updates them in place, so they are
        // copied. Device tensors become new host copies in the download.
        for (_, t) in tensors.iter_mut().filter(|(_, t)| t.device().is_cpu()) {
            *t = folo_core::Tensor::from_storage(
                folo_core::Storage::Cpu(t.to_cpu_storage()?),
                t.shape().clone(),
            )?;
        }
        download_all(&mut tensors)?;
        Ok((map, tensors))
    }
}

/// Bytes [`download_all`] concatenates into one device buffer at most (also never above the device's buffer limits).
pub(super) const DOWNLOAD_CHUNK: u64 = 128 << 20;

/// Replaces the device tensors of a checkpoint by host copies, downloaded with one readback per dtype and chunk of
/// at most [`DOWNLOAD_CHUNK`] bytes: the tensors of a chunk are concatenated on the device first. Read one by one,
/// every tensor would wait for the GPU (about a thousand waits for the weights, EMA and optimizer state of
/// folo-n); in one piece, folo-x's state (up to ~900 MB) would exceed many adapters' buffer limits and add a full
/// copy of the training state to VRAM. The bytes are the same.
pub(crate) fn download_all(tensors: &mut [(String, folo_core::Tensor)]) -> Result<()> {
    download_all_chunked(tensors, DOWNLOAD_CHUNK)
}

/// [`download_all`] with chunks of at most `chunk` bytes (a tensor larger than that is a chunk of its own).
pub(super) fn download_all_chunked(tensors: &mut [(String, folo_core::Tensor)], chunk: u64) -> Result<()> {
    let _g = folo_core::no_grad();
    let mut groups: Vec<((Device, folo_core::DType), Vec<usize>)> = Vec::new();
    for (i, (_, t)) in tensors.iter_mut().enumerate() {
        if t.device().is_cpu() {
            continue;
        }
        if t.numel() == 0 {
            *t = t.to_device(Device::Cpu)?;
            continue;
        }
        let key = (t.device(), t.dtype());
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(i),
            None => groups.push((key, vec![i])),
        }
    }
    for ((dev, dtype), idx) in groups {
        let limit = folo_gpu::context_for(dev)
            .map_or(u64::MAX, |c| c.caps().max_buffer_size.min(c.caps().max_storage_buffer_binding_size));
        let max = chunk.min(limit).max(1);
        let elem = dtype.size_in_bytes() as u64;
        let mut rest = &idx[..];
        while !rest.is_empty() {
            // As many tensors as fit in `max` bytes, at least one.
            let mut bytes = 0;
            let take = rest
                .iter()
                .take_while(|&&i| {
                    bytes += tensors[i].1.numel() as u64 * elem;
                    bytes <= max
                })
                .count()
                .max(1);
            let (part, tail) = rest.split_at(take);
            rest = tail;
            let flat = part
                .iter()
                .map(|&i| tensors[i].1.reshape([tensors[i].1.numel()]))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let host = folo_core::Tensor::cat(&flat, 0)?.to_device(Device::Cpu)?;
            drop(flat);
            let mut start = 0;
            for &i in part {
                let t = &tensors[i].1;
                let n = t.numel();
                let piece = host.narrow(0, start, n)?.reshape(t.shape().clone())?;
                start += n;
                tensors[i].1 = piece;
            }
        }
    }
    Ok(())
}
