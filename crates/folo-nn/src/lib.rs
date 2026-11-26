//! FOLO neural-network layers, optimizers, schedules, EMA and the `.folo` weight format.
//!
//! Everything the detector and the trainer need below the model definition (docs/NN.md has the overview):
//! - [`Module`] / [`Forward`]: stable dotted names, parameters vs buffers, train/eval propagation, visitors.
//! - Layers: [`Conv2d`], [`BatchNorm2d`], [`Linear`], [`Activation`], [`MaxPool2d`], [`Upsample`],
//!   [`Concat`], [`Identity`], [`Sequential`], [`ModuleList`], and [`ConvBnAct`] (docs/MODEL.md's `Conv`).
//! - [`init`]: seeded Kaiming / Xavier and the head priors.
//! - [`optim`]: [`optim::Sgd`], [`optim::AdamW`], [`optim::MuSgd`], parameter groups, gradient accumulation and
//!   clipping.
//! - [`LrSchedule`] (warmup + linear / cosine / WSD / cosine-restarts) and [`ModelEma`].
//! - [`amp`]: [`GradScaler`], the dynamic loss scaling of mixed-precision training.
//! - [`io`]: `.folo` read / write, strict and partial loading, checkpoints, export with Conv+BN fusion.
//!
//! Files: `module.rs` ([`Module`], [`Forward`]), `layers/`, `init.rs`, `optim/`, `sched.rs` ([`LrSchedule`]),
//! `ema.rs` ([`ModelEma`]), `amp.rs`, `io/` and `error.rs` ([`Error`]). `crates/folo-nn/README.md` maps them.
//!
//! ```
//! use folo_core::{rng::Rng, Tensor};
//! use folo_nn::optim::{GradAccumulator, Optimizer, Sgd, StandardGroups};
//! use folo_nn::{Activation, ConvBnAct, DecayKind, Forward, LrSchedule, ModelEma, Module, Sequential};
//!
//! # fn main() -> folo_nn::Result<()> {
//! let build = |rng: &mut Rng| {
//!     Sequential::new()
//!         .push_named("stem", ConvBnAct::new(3, 8, 3, 2, 1, Activation::Silu, rng))
//!         .push_named("out", ConvBnAct::new(8, 4, 1, 1, 1, Activation::Identity, rng))
//! };
//! let mut rng = Rng::seed(0);
//! let model = build(&mut rng);
//! assert_eq!(model.named_parameters()[0].0, "stem.conv.weight");
//!
//! let mut ema = ModelEma::new(build(&mut Rng::seed(1)), &model)?;
//! let mut opt = Sgd::new(StandardGroups::from_module(&model).into_groups(0.01, 0.937, 5e-4))?;
//! let mut sched = LrSchedule::new(DecayKind::Linear, 2, 3, 0.01, 0.0, 0.0, 0.8)?;
//! let mut acc = GradAccumulator::new(&model.parameters());
//!
//! for _ in 0..3 {
//!     let x = Tensor::rand_uniform([2, 3, 16, 16], 0.0, 1.0, &mut rng);
//!     let loss = model.forward(&x)?.square()?.mean_all()?;
//!     acc.add(&loss.backward()?)?;
//!     sched.apply(&mut opt);
//!     acc.clip_grad_norm(10.0)?;
//!     opt.step(&acc)?;
//!     ema.update(&model)?;
//!     sched.advance();
//!     acc.zero();
//! }
//! assert_eq!((opt.step_count(), sched.step(), ema.updates()), (3, 3, 3));
//! # Ok(())
//! # }
//! ```

pub mod amp;
mod ema;
mod error;
pub mod init;
pub mod io;
mod layers;
mod module;
pub mod optim;
mod sched;
mod swa;

pub use amp::{AmpContract, GradScaler, ScaleUpdate, unscale_, usable_scale};
pub use ema::{
    EMA_DECAY, EMA_MAX_WINDOW, EMA_TAU, EmaContract, EmaMode, EmaPowerConfig, EmaState, ModelEma,
    PEMA_GAMMA_0, PEMA_LF_MIN, PEMA_RHO, ema_for_run,
};
pub use error::{Error, Result};
pub use layers::{
    Activation, BN_EPS, BN_MOMENTUM, BatchNorm2d, Concat, Conv2d, Conv2dConfig, ConvBnAct, Identity, Linear,
    MaxPool2d, ModuleList, RunningStats, Sequential, Upsample, fuse_conv_bn,
};
pub use module::{Forward, Layer, Mode, Module, NamedTensor, TensorKind};
pub use sched::{
    DEFAULT_ONE_CYCLE_DIV_FACTOR, DEFAULT_ONE_CYCLE_PCT_START, DEFAULT_POLY_POWER, DEFAULT_RESTART_CYCLES,
    DEFAULT_RESTART_MULT, DecayKind, LrSchedule, SchedContract, SchedUnit, WARMUP_FLOOR,
    WSD_DEFAULT_DECAY_FRACTION, WsdTail, warmup_iters, warmup_iters_floor,
};
pub use swa::SwaModel;
