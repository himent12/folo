//! Layers: convolution, BatchNorm, linear, activations, pooling, upsampling, containers and the `Conv` block.
//!
//! - `conv.rs`: [`Conv2d`] and [`Conv2dConfig`].
//! - `batchnorm/`: [`BatchNorm2d`], its fused CPU training kernel, and [`RunningStats`], the running-statistics
//!   guard for non-finite batches.
//! - `conv_bn_act.rs`: [`ConvBnAct`] (docs/MODEL.md's `Conv`) and [`fuse_conv_bn`].
//! - `simple.rs`: [`Linear`], [`Activation`], [`MaxPool2d`], [`Upsample`], [`Concat`] and [`Identity`].
//! - `glu.rs`: [`GatedFfn`] (`SwiGLU` / `GeGLU`), [`GluKind`], [`swiglu`], [`geglu`], and [`gelu_tanh`].
//! - `rmsnorm.rs`: [`RmsNorm`] (Root Mean Square Layer Normalization).
//! - `regularize.rs`: [`DropPath`] (stochastic depth) and [`LayerScale`] (per-channel residual scaling).
//! - `spectral_norm.rs`: [`SpectralNorm`], [`power_iteration`], [`spectral_norm`], and [`spectral_normalize`].
//! - `attention.rs`: [`SqueezeExcitation`], [`Eca`] ([`eca_kernel_size`]), [`ChannelAttention`],
//!   [`SpatialAttention`], and [`Cbam`].
//! - `ws_conv.rs`: [`WeightStandardizedConv2d`] and [`standardize_weight`].
//! - `convnext.rs`: [`LayerNorm`] and [`ConvNeXtBlock`] ([`ConvNeXtConfig`], [`ConvNeXtNorm`]).
//! - `mbconv.rs`: [`InvertedResidual`] (MBConv) and [`InvertedResidualConfig`].
//! - `dropblock.rs`: [`DropBlock2d`] and [`dropblock_gamma`].
//! - `container.rs`: [`Sequential`] and [`ModuleList`].

mod attention;
mod batchnorm;
mod container;
mod conv;
mod conv_bn_act;
mod convnext;
mod dropblock;
mod glu;
mod mbconv;
mod regularize;
mod rmsnorm;
mod simple;
mod spectral_norm;
#[cfg(test)]
mod tests_attention;
#[cfg(test)]
mod tests_blocks;
mod ws_conv;

pub use attention::{Cbam, ChannelAttention, Eca, SpatialAttention, SqueezeExcitation, eca_kernel_size};
pub use batchnorm::{BN_EPS, BN_MOMENTUM, BatchNorm2d, RunningStats};
pub use container::{ModuleList, Sequential};
pub use conv::{Conv2d, Conv2dConfig};
pub use conv_bn_act::{ConvBnAct, fuse_conv_bn};
pub use convnext::{
    CONVNEXT_DEFAULT_EXPAND, CONVNEXT_DEFAULT_KERNEL, CONVNEXT_DEFAULT_LAYER_SCALE, ConvNeXtBlock,
    ConvNeXtConfig, ConvNeXtNorm, ConvNeXtNormLayer, LAYERNORM_DEFAULT_EPS, LayerNorm,
};
pub use dropblock::{DropBlock2d, dropblock_gamma};
pub use glu::{GatedFfn, GluKind, geglu, gelu_tanh, swiglu};
pub use mbconv::{InvertedResidual, InvertedResidualConfig};
pub use regularize::{ChannelLayout, DropPath, LAYERSCALE_DEFAULT_INIT, LayerScale};
pub use rmsnorm::{RMSNORM_DEFAULT_EPS, RmsNorm};
pub use simple::{Activation, Concat, Identity, Linear, MaxPool2d, Upsample};
pub use spectral_norm::{
    PowerIterationResult, SPECTRAL_NORM_EPS, SpectralNorm, SpectralNormTarget, power_iteration,
    spectral_norm, spectral_normalize,
};
pub use ws_conv::{WS_CONV_DEFAULT_EPS, WeightStandardizedConv2d, standardize_weight};
