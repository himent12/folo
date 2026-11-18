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
/// Whether the int8 convs run on AVX-VNNI here (else a portable kernel, several times slower than `f32`).
pub use folo_core::cpu::q8::uses_vnni;

use folo_nn::io::Metadata;
use folo_nn::{Activation, ConvBnAct, Error, Module, Result};

use crate::model::FoloModel;

/// The output size of `conv` on an `[n, c, h, w]` input when it runs on the int8 kernel; `None` when it runs in
/// `f32` (the int8 executor dequantises its inputs, and the fake-quant reference keeps its `f32` weights). A conv
/// runs in int8 when it is fused, dense and undilated, with SiLU or no activation, its weight is
/// `[out, in, k, k]` as configured with `in = c`, [`folo_core::cpu::q8::QConv::supports`] its geometry, and
/// [`folo_core::cpu::q8::conv_size`] its sizes (the kernel fits the padded input, nothing overflows), and its
/// residual `res`, if any, is on the output's shape (the `f32` path broadcasts one that is not).
pub(crate) fn int8_able(
    conv: &ConvBnAct,
    [n, c, h, w]: [usize; 4],
    res: Option<[usize; 4]>,
) -> Option<(usize, usize)> {
    let cfg = conv.conv.config();
    let (o, k) = (cfg.out_channels, cfg.kernel);
    let fits = conv.bn.is_none()
        && cfg.groups == 1
        && cfg.dilation == 1
        && matches!(conv.act, Activation::Silu | Activation::Identity)
        && cfg.in_channels == c
        && conv.conv.weight.dims() == [o, c, k, k]
        && conv.conv.bias.as_ref().is_none_or(|b| b.dims() == [o])
        && folo_core::cpu::q8::QConv::supports((o, c, k), cfg.stride, cfg.padding);
    if !fits {
        return None;
    }
    folo_core::cpu::q8::conv_size(n, c, o, h, w, k, cfg.stride, cfg.padding)
        .filter(|&(oh, ow)| res.is_none_or(|r| r == [n, o, oh, ow]))
}

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
