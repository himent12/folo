// Adapted from GPUI (Apache-2.0): crates/gpui/src/assets.rs (RenderImage)
//! Decoded images: `RenderImage` (frames of BGRA pixels, as GPUI's).

use crate::{DevicePixels, Pixels, Size, size};
use ::image::{Delay, Frame};
use smallvec::SmallVec;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// A process-unique image id (tile hashes and caches use it, never the pixels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(pub u64);

/// A decoded image ready to draw: one frame, or several for an animation. Each frame's bytes are BGRA
/// (as GPUI's), straight alpha, held in the frame's `RgbaImage` buffer.
pub struct RenderImage {
    /// The image's process-unique id.
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Wraps the given frames (BGRA bytes) with a new id.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self { id: ImageId(NEXT_ID.fetch_add(1, Ordering::Relaxed)), scale_factor: 1.0, data: data.into() }
    }

    /// The bytes of frame `frame_index` (BGRA), if it exists.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data.get(frame_index).map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// The size of frame `frame_index`, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        let (width, height) = self.data[frame_index].buffer().dimensions();
        size(width.into(), height.into())
    }

    /// The size of frame `frame_index` for display, adjusted for the scale factor.
    pub fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index).map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// The delay of frame `frame_index` from the previous one.
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data[frame_index].delay()
    }

    /// How many frames the image has.
    pub fn frame_count(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData").field("id", &self.id).field("size", &self.size(0)).finish()
    }
}
