//! FOLO tensor core: n-d tensors, CPU kernels, reverse-mode autograd.
//!
//! Everything else in FOLO builds on this crate. The API follows PyTorch semantics (broadcasting, negative
//! dims, `keepdim`, views that share storage) with Rust ergonomics: ops return [`Result`], tensors are cheap
//! `Arc` handles, and gradients come back as a [`Grads`] map instead of being stored on tensors.
//!
//! ```
//! use folo_core::{no_grad, rng::Rng, Result, Tensor};
//!
//! fn main() -> Result<()> {
//!     let mut rng = Rng::seed(0);
//!     // A tiny conv layer + loss.
//!     let x = Tensor::randn([2, 3, 8, 8], 0.0, 1.0, &mut rng);
//!     let w = Tensor::randn([4, 3, 3, 3], 0.0, 0.1, &mut rng).requires_grad_();
//!     let b = Tensor::zeros([4]).requires_grad_();
//!     let y = x.conv2d(&w, Some(&b), 1, 1, 1, 1)?.silu()?;
//!     assert_eq!(y.dims(), &[2, 4, 8, 8]);
//!     let loss = y.square()?.mean_all()?;
//!
//!     // Reverse mode: gradients for every leaf that requires grad.
//!     let grads = loss.backward()?;
//!     let gw = grads.get(&w).expect("w contributed to the loss");
//!     assert_eq!(gw.shape(), w.shape());
//!
//!     // An SGD step, in place, outside the graph.
//!     let _guard = no_grad();
//!     w.add_scaled_assign(gw, -0.1)?;
//!     Ok(())
//! }
//! ```
//!
//! # Map
//! - [`Tensor`] (`tensor.rs`): construction, inspection, data access. Its ops are methods defined in the private
//!   `ops/` folder (elementwise, reductions, matmul and conv, views, copies, in-place updates, fused ops).
//! - [`Shape`], [`Layout`], [`DType`], [`Device`], [`Storage`]: what a tensor is (`shape.rs`, `layout.rs`,
//!   `dtype.rs`, `device.rs`, `storage.rs`); [`Error`] in `error.rs`.
//! - [`autograd`]: [`no_grad`], [`Grads`]. [`testing::grad_check`] verifies gradients numerically.
//! - [`autocast`](mod@autocast): mixed precision for the forward pass.
//! - [`backend`]: the [`backend::Backend`] trait device kernels implement and the runtime registry
//!   (folo-gpu registers itself there). [`cpu`] is the built-in implementation; [`cpu::gemm`] exposes the
//!   GEMM for raw slices.
//! - [`rng`]: the seeded generator every random op takes.
//!
//! See `crates/folo-core/README.md` for where things are, and `docs/ARCHITECTURE.md` for how this crate fits into
//! FOLO.

pub mod autocast;
pub mod autograd;
pub mod backend;
pub mod cpu;
mod device;
mod dtype;
mod error;
mod guard;
mod layout;
mod ops;
pub mod rng;
mod shape;
mod storage;
mod tensor;
pub mod testing;

pub use autocast::{AutocastGuard, autocast, autocast_dtype};
pub use autograd::{Grads, NoGradGuard, is_grad_enabled, no_grad, with_no_grad};
pub use backend::{
    BinaryOp, ClassSites, Conv2dParams, DetectDecode, LetterboxImage, Pool2dParams, ReduceOp, UnaryOp,
};
pub use device::Device;
pub use dtype::{DType, Element};
pub use error::{Error, Result};
pub use guard::{StepGuard, step_guard_allows};
pub use half::f16;
pub use layout::Layout;
pub use ops::BatchNormTrain;
pub use shape::{AllDims, Dim, DimVec, Dims, Shape, ShapeSpec};
pub use storage::{CpuStorage, DeviceStorage, Storage};
pub use tensor::{Tensor, TensorId};
