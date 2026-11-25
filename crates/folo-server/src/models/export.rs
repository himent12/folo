//! Model exports (`POST /api/models/{id}/export`) to `<data dir>/exports/` and shared [`onnx_export`] options.

use folo_api::{ExportFormat, ExportResult, ExportSpec};
use folo_nn::io::FloatDType;
use folo_vision::export::onnx::{self, OnnxOptions};

use crate::engine::{self, engine_error, nn_error};
use crate::error::{Error, Result};

use super::{ModelRef, Models};

/// Longest [`export_stem`], in bytes: file names hold 255, and the export's suffix and `-2`, ... follow.
const MAX_STEM_BYTES: usize = 200;

impl Models {
    /// Exports `m` to `<data dir>/exports/`: `folo-f16` as `<name>-f16.folo`, `onnx` as `<name>.onnx` (the
    /// defaults of `folo export --format onnx`: f32, batch 1, the model's imgsz, raw head outputs; `-decoded` with
    /// `decode`, `-f16` with `half`, as the CLI names them). `-2`, `-3`, ... when the name is taken; never
    /// overwriting.
    pub fn export(&self, m: &ModelRef, spec: ExportSpec) -> Result<ExportResult> {
        let format = spec.format;
        let file = engine::read_model_file(&m.path)?;
        let dir = self.data_dir.join("exports");
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        let base = export_stem(&m.model.name);
        let (suffix, ext) = match format {
            ExportFormat::FoloF16 => ("-f16", "folo"),
            ExportFormat::Onnx => (spec.onnx.file_suffix(), "onnx"),
        };
        // Written under a temporary name and moved to the first free name only once complete, in one step that
        // fails when the name exists (folo_nn::io::publish_free): two exports never share or replace a file, and
        // one that fails or is killed leaves no file under the name.
        let name = |n: u32| {
            let tag = if n == 1 { String::new() } else { format!("-{n}") };
            dir.join(format!("{base}{suffix}{tag}.{ext}"))
        };
        let creating = |e: std::io::Error| Error::internal(format!("creating {}: {e}", name(1).display()));
        let tmp = folo_nn::io::temp_path(&dir).map_err(creating)?;
        let written = match format {
            ExportFormat::FoloF16 => file.export(&tmp, FloatDType::F16).map_err(nn_error),
            ExportFormat::Onnx => {
                onnx_export(&file, spec.onnx).and_then(|x| x.write_to(&tmp).map_err(engine_error))
            }
        }
        .and_then(|()| folo_nn::io::publish_free(&tmp, name).map_err(creating));
        let out = match written {
            Ok(out) => out,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        let bytes = std::fs::metadata(&out).map(|x| x.len()).unwrap_or(0);
        Ok(ExportResult { path: folo_api::json::path(&out), bytes })
    }
}

/// The start of an export's file name for a model called `name` (a run's name is any text): the characters no file
/// name may hold, on Windows either, become `-`, and it is cut to [`MAX_STEM_BYTES`], so every model can be exported.
fn export_stem(name: &str) -> String {
    let mut stem = String::new();
    for c in name.chars() {
        let c = if c.is_control() || matches!(c, '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*') {
            '-'
        } else {
            c
        };
        if stem.len() + c.len_utf8() > MAX_STEM_BYTES {
            break;
        }
        stem.push(c);
    }
    stem
}

/// The `onnx` export of `POST /api/models/{id}/export` (and `/weights?format=onnx`): `folo export --format onnx`'s
/// defaults (f32, batch 1, the model's training imgsz, raw head outputs) unless a switch says otherwise, with the
/// file's metadata. A model the exporter cannot map is `501 not_implemented`.
pub fn onnx_export(
    file: &folo_nn::io::ModelFile,
    switches: folo_api::OnnxSwitches,
) -> Result<onnx::OnnxExport> {
    let mut opts = OnnxOptions::new(file.metadata().imgsz);
    (opts.decode, opts.dynamic, opts.half) = (switches.decode, switches.dynamic, switches.half);
    opts.metadata = file.metadata().for_weights().to_map();
    onnx::export_file(file, &opts).map_err(engine_error)
}
