//! WebAssembly SIMD128-accelerated tensor and post-processing kernels with portable scalar fallbacks.
//!
//! When compiled for `wasm32` with `-C target-feature=+simd128`, the inner loops run on 128-bit WASM SIMD
//! instructions (`core::arch::wasm32`); natively and on non-SIMD builds, portable scalar loops produce matching
//! results so the workspace's native test suite exercises every contract directly.
//!
//! - [`pool`]: 2D NCHW max pooling ([`max_pool2d_nchw`]) and average pooling ([`avg_pool2d_nchw`]).
//! - [`upsample`]: 2D NCHW nearest-neighbour ([`upsample_nearest2d_nchw`]) and bilinear
//!   ([`upsample_bilinear2d_nchw`]) upsampling.
//! - [`post`]: row-wise vectorized softmax ([`softmax_rows`]) and batched axis-aligned box IoU
//!   ([`box_iou_matrix`], [`box_iou_one_to_many`]).

pub mod pool;
pub mod post;
pub mod upsample;

pub use pool::{Pool2dSpec, avg_pool2d_nchw, max_pool2d_nchw};
pub use post::{box_iou_matrix, box_iou_one_to_many, softmax_rows};
pub use upsample::{upsample_bilinear2d_nchw, upsample_nearest2d_nchw};

/// Returns `true` when compiled with WebAssembly SIMD128 (`target_arch = "wasm32"` and `target_feature = "simd128"`).
#[inline]
pub const fn simd128_enabled() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}
