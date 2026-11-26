//! Layers: convolution, BatchNorm, linear, activations, pooling, upsampling, containers and the `Conv` block.
//!
//! - `conv.rs`: [`Conv2d`] and [`Conv2dConfig`].
//! - `batchnorm/`: [`BatchNorm2d`], its fused CPU training kernel, and [`RunningStats`], the running-statistics
//!   guard for non-finite batches.
//! - `conv_bn_act.rs`: [`ConvBnAct`] (docs/MODEL.md's `Conv`) and [`fuse_conv_bn`].
//! - `simple.rs`: [`Linear`], [`Activation`], [`MaxPool2d`], [`Upsample`], [`Concat`] and [`Identity`].
//! - `regularize.rs`: [`DropPath`] (stochastic depth) and [`LayerScale`] (per-channel residual scaling).
//! - `spectral_norm.rs`: [`SpectralNorm`], [`power_iteration`], [`spectral_norm`], and [`spectral_normalize`].
//! - `container.rs`: [`Sequential`] and [`ModuleList`].

mod batchnorm;
mod container;
mod conv;
mod conv_bn_act;
mod regularize;
mod simple;
mod spectral_norm;

pub use batchnorm::{BN_EPS, BN_MOMENTUM, BatchNorm2d, RunningStats};
pub use container::{ModuleList, Sequential};
pub use conv::{Conv2d, Conv2dConfig};
pub use conv_bn_act::{ConvBnAct, fuse_conv_bn};
pub use regularize::{ChannelLayout, DropPath, LAYERSCALE_DEFAULT_INIT, LayerScale};
pub use simple::{Activation, Concat, Identity, Linear, MaxPool2d, Upsample};
pub use spectral_norm::{
    PowerIterationResult, SPECTRAL_NORM_EPS, SpectralNorm, SpectralNormTarget, power_iteration,
    spectral_norm, spectral_normalize,
};
