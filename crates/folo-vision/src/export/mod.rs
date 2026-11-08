//! Export of FOLO models to other runtimes. `.folo` export itself lives in folo-nn (docs/FORMAT.md, Export).
//!
//! - [`onnx`]: an ONNX graph (opset 17) of a fused [`crate::FoloModel`], for onnxruntime, TensorRT, OpenVINO and
//!   the CoreML converters (`folo export --format onnx`, docs/CLI.md).

pub mod onnx;
