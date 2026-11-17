//! Post-training int8 quantisation of the backbone and neck (docs/MODEL.md §9.6).
//!
//! - [`walk`]: the quantised graph of layers 0–22 over an executor ([`walk::Exec`]).
//! - [`calib`]: range calibration over sample images ([`calibrate`]), and the stored ranges ([`Calibration`]).
//! - [`exec`]: the int8 executor on folo-core's kernels ([`Int8Plan`]).
//! - [`fake`]: the fake-quant reference executor ([`FakeQuant`]): `f32` arithmetic on the int8 plan's grids.

pub mod calib;
pub mod exec;
pub mod fake;
pub mod walk;

pub use calib::{Calibration, Method, calibrate, calibrate_sample};
pub use exec::Int8Plan;
pub use fake::FakeQuant;

/// An affine u8 grid (folo-core's [`folo_core::cpu::q8::Grid`]): `x ≈ scale · (q − zero)`.
pub use folo_core::cpu::q8::Grid as QParam;
pub use folo_core::cpu::q8::quantize_weights;

use folo_nn::io::Metadata;
use folo_nn::{Error, Module, Result};

use crate::model::FoloModel;

/// The `.folo` metadata key that holds a model's [`Calibration`] (JSON), written by `folo quantize`.
pub const META_KEY: &str = "folo.quant";

impl Calibration {
    /// The calibration a file's metadata holds ([`META_KEY`]), if any.
    pub fn from_metadata(meta: &Metadata) -> Result<Option<Calibration>> {
        let Some(s) = meta.extra.get(META_KEY) else { return Ok(None) };
        serde_json::from_str(s)
            .map(Some)
            .map_err(|e| Error::Metadata(format!("`{META_KEY}` is not a calibration: {e}")))
    }

    /// Store this calibration in `meta` ([`META_KEY`]).
    pub fn store(&self, meta: &mut Metadata) -> Result<()> {
        let s = serde_json::to_string(self).map_err(|e| Error::Metadata(e.to_string()))?;
        meta.extra.insert(META_KEY.to_string(), s);
        Ok(())
    }
}

/// A fingerprint of a model's tensors (names, shapes, dtypes and bits; FNV-1a, 64 bits, hex): a calibration
/// records the one of the weights it was measured on, and [`FoloModel::set_int8`] refuses other weights.
pub fn fingerprint(model: &FoloModel) -> Result<String> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    };
    let mut ts = model.named_tensors();
    ts.sort_by(|a, b| a.name.cmp(&b.name));
    for t in ts {
        eat(t.name.as_bytes());
        eat(format!("{:?}{:?}", t.tensor.dims(), t.tensor.dtype()).as_bytes());
        let v = t.tensor.to_dtype(folo_core::DType::F32)?.to_vec::<f32>()?;
        for x in v {
            eat(&x.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{h:016x}"))
}
