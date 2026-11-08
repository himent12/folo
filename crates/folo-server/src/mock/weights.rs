//! Mock weights (`GET /api/models/{id}/weights`): real `.folo` files of the folo-n architecture with seeded random
//! weights and the model's classes and image size, generated on first request and kept in memory (one file per
//! class list, image size and format), so Studio's in-browser detector can load them. Random weights detect nothing
//! useful; the point is a file that loads and runs exactly like a trained one.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::SystemTime;

use axum::body::Bytes;
use folo_api::Model;
use folo_core::rng::Rng;
use folo_nn::io::{FloatDType, ModelConfig, ModelFile, write_weights};
use folo_vision::FoloModel;

use crate::error::{Error, Result};
use crate::store::{Weights, WeightsBody, WeightsFormat, weights_file_name};
use crate::util::fnv1a;

type Key = (Vec<String>, u32, WeightsFormat);

/// The generated files (bytes and their ETag).
pub(super) struct MockWeights {
    /// `Last-Modified` of every file: when the store was created (the bytes are deterministic, but a later build
    /// may generate different ones, so a restart must not answer an older `If-Modified-Since` with 304).
    born: SystemTime,
    files: Mutex<HashMap<Key, (Bytes, String)>>,
}

impl MockWeights {
    pub(super) fn new() -> MockWeights {
        MockWeights { born: SystemTime::now(), files: Mutex::default() }
    }

    /// The weights of `model` in `format`.
    pub(super) fn get(&self, model: &Model, format: WeightsFormat) -> Result<Weights> {
        let (bytes, etag) = self.file(&model.classes, model.imgsz, format)?;
        Ok(Weights {
            len: bytes.len() as u64,
            body: WeightsBody::Bytes(bytes),
            modified: self.born,
            etag,
            file_name: weights_file_name(&model.name, format),
        })
    }

    fn file(&self, classes: &[String], imgsz: u32, format: WeightsFormat) -> Result<(Bytes, String)> {
        let key = (classes.to_vec(), imgsz, format);
        if let Some(f) = self.files.lock().unwrap().get(&key) {
            return Ok(f.clone());
        }
        let bytes = match format {
            WeightsFormat::Stored => generate(classes, imgsz)?,
            WeightsFormat::F16 => {
                let (f32_bytes, _) = self.file(classes, imgsz, WeightsFormat::Stored)?;
                let file = ModelFile::from_bytes(f32_bytes.to_vec()).map_err(internal)?;
                let mut out = Vec::new();
                file.export_to(&mut out, FloatDType::F16).map_err(internal)?;
                out
            }
        };
        let etag = format!("\"{:x}-{:016x}\"", bytes.len(), fnv1a(&bytes));
        let f = (Bytes::from(bytes), etag);
        // Two first requests may both generate; the bytes are identical, so either insert is fine.
        Ok(self.files.lock().unwrap().entry(key).or_insert(f).clone())
    }
}

fn internal(e: folo_nn::Error) -> Error {
    Error::internal(format!("generating mock weights: {e}"))
}

/// A folo-n weights file with `classes` and `imgsz`, its weights drawn from a seed derived from the class names.
fn generate(classes: &[String], imgsz: u32) -> Result<Vec<u8>> {
    let config = ModelConfig::new("n", classes.len()).map_err(internal)?;
    let seed = fnv1a(classes.join("\n").as_bytes());
    let mut model = FoloModel::with_rng(config, &mut Rng::seed(seed)).map_err(internal)?;
    model.set_classes(classes.to_vec()).map_err(internal)?;
    model.set_imgsz(imgsz);
    let mut out = Vec::new();
    write_weights(&mut out, &model.metadata().map_err(internal)?, &model).map_err(internal)?;
    Ok(out)
}
