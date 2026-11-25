//! Layers: convolution, BatchNorm, linear, activations, pooling, upsampling, containers and the `Conv` block.
//!
//! - `conv.rs`: [`Conv2d`] and [`Conv2dConfig`].
//! - `batchnorm/`: [`BatchNorm2d`], its fused CPU training kernel, and [`RunningStats`], the running-statistics
//!   guard for non-finite batches.
//! - `conv_bn_act.rs`: [`ConvBnAct`] (docs/MODEL.md's `Conv`) and [`fuse_conv_bn`].
//! - `simple.rs`: [`Linear`], [`Activation`], [`MaxPool2d`], [`Upsample`], [`Concat`] and [`Identity`].
//! - `container.rs`: [`Sequential`] and [`ModuleList`].

mod batchnorm;
mod container;
mod conv;
mod conv_bn_act;
mod simple;

pub use batchnorm::{BN_EPS, BN_MOMENTUM, BatchNorm2d, RunningStats};
pub use container::{ModuleList, Sequential};
pub use conv::{Conv2d, Conv2dConfig};
pub use conv_bn_act::{ConvBnAct, fuse_conv_bn};
pub use simple::{Activation, Concat, Identity, Linear, MaxPool2d, Upsample};
