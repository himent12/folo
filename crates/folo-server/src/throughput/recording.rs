//! [`Throughput`]: the table in memory and in its file, and recording samples into it.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use folo_api::{PhaseKind, RunEstimate, ThroughputTable};

use crate::index::write_atomic;
use crate::util::{rfc3339, unix_now};

use super::api_table::table;
use super::run_estimate::{EstimateInput, estimate};
use super::{DeviceId, InferRecord, InferSample, Series, TableFile, TrainRecord, TrainSample, WRITE_EVERY};

struct Inner {
    file: TableFile,
    dirty: bool,
    last_write: Option<Instant>,
}

/// The table, persisted (real store) or in memory (mock).
pub struct Throughput {
    path: Option<PathBuf>,
    inner: Mutex<Inner>,
}

impl Throughput {
    /// Opens `path` (a missing file is an empty table; a malformed one is logged and replaced on the next write:
    /// it only holds measurements, which come back).
    pub fn open(path: PathBuf) -> Throughput {
        let file = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!("{}: {e}; starting a new throughput table", path.display());
                TableFile::default()
            }),
            Err(_) => TableFile::default(),
        };
        Throughput { path: Some(path), inner: Mutex::new(Inner { file, dirty: false, last_write: None }) }
    }

    /// A table that is never written.
    pub fn in_memory(file: TableFile) -> Throughput {
        Throughput { path: None, inner: Mutex::new(Inner { file, dirty: false, last_write: None }) }
    }

    /// A copy of the rows.
    pub fn file(&self) -> TableFile {
        self.inner.lock().unwrap().file.clone()
    }

    /// Adds a training sample and writes the file.
    pub fn record_train(&self, s: TrainSample) -> TrainRecord {
        let mut inner = self.inner.lock().unwrap();
        let rows = &mut inner.file.training;
        let i = match rows.iter().position(|r| r.key == s.key) {
            Some(i) => i,
            None => {
                rows.push(TrainRecord {
                    key: s.key.clone(),
                    mosaic: Series::default(),
                    close_mosaic: Series::default(),
                    val: Series::default(),
                    startup: Series::default(),
                    peak_vram_bytes: None,
                    probes: 0,
                    updated: String::new(),
                });
                rows.len() - 1
            }
        };
        let r = &mut rows[i];
        if let Some(v) = s.train_imgs_per_s {
            match s.phase {
                PhaseKind::Mosaic => r.mosaic.push(v),
                PhaseKind::CloseMosaic => r.close_mosaic.push(v),
            }
        }
        if let Some(v) = s.val_imgs_per_s {
            r.val.push(v);
        }
        if let Some(v) = s.startup_s {
            r.startup.push(v);
        }
        if let Some(b) = s.peak_vram_bytes.filter(|&b| b > 0) {
            r.peak_vram_bytes = Some(r.peak_vram_bytes.map_or(b, |m| m.max(b)));
        }
        r.probes += u64::from(s.probe);
        r.updated = rfc3339(unix_now());
        let out = r.clone();
        inner.dirty = true;
        self.write(&mut inner);
        out
    }

    /// Adds a prediction; the file is written at most every [`WRITE_EVERY`] (and by [`Throughput::flush`]).
    pub fn record_infer(&self, s: InferSample) -> InferRecord {
        let mut inner = self.inner.lock().unwrap();
        let rows = &mut inner.file.inference;
        let i = match rows.iter().position(|r| r.key == s.key) {
            Some(i) => i,
            None => {
                rows.push(InferRecord {
                    key: s.key.clone(),
                    total: Series::default(),
                    inference: Series::default(),
                    updated: String::new(),
                });
                rows.len() - 1
            }
        };
        let r = &mut rows[i];
        r.total.push(s.total_ms);
        r.inference.push(s.inference_ms);
        r.updated = rfc3339(unix_now());
        let out = r.clone();
        inner.dirty = true;
        if inner.last_write.is_none_or(|t| t.elapsed() >= WRITE_EVERY) {
            self.write(&mut inner);
        }
        out
    }

    /// Writes pending samples.
    pub fn flush(&self) {
        let mut inner = self.inner.lock().unwrap();
        self.write(&mut inner);
    }

    fn write(&self, inner: &mut Inner) {
        let Some(path) = &self.path else { return };
        if !inner.dirty {
            return;
        }
        let bytes = serde_json::to_vec_pretty(&inner.file).expect("the table serializes");
        match write_atomic(path, &bytes) {
            Ok(()) => {
                inner.dirty = false;
                inner.last_write = Some(Instant::now());
            }
            Err(e) => tracing::warn!("writing {}: {e}", path.display()),
        }
    }

    /// `GET /api/system/throughput`: measured rows, then reference rows for `devices` that no measured row covers.
    pub fn table(&self, auto: &DeviceId, devices: &[DeviceId]) -> ThroughputTable {
        table(&self.inner.lock().unwrap().file, auto, devices)
    }

    /// `POST /api/runs/estimate` on this table.
    pub fn estimate(&self, input: &EstimateInput) -> RunEstimate {
        estimate(&self.inner.lock().unwrap().file.training, input)
    }
}
