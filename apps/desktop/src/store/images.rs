//! The decoded-image cache: `RenderImage`s keyed by dataset, image and size, and the conversion from
//! backend pixels.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use folo_ui::RenderImage;

use crate::api::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ImgKind {
    Thumb,
    Full,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ImgKey {
    pub dataset: String,
    pub image: String,
    pub kind: ImgKind,
}

pub enum ImgSlot {
    Loading,
    Ready(Arc<RenderImage>),
    Failed,
}

/// Decoded images keyed by dataset/image/size, evicted oldest-first past a budget.
#[derive(Default)]
pub struct ImageCache {
    pub slots: HashMap<ImgKey, ImgSlot>,
    order: VecDeque<ImgKey>,
}

impl ImageCache {
    pub const BUDGET: usize = 600;

    pub fn get(&self, k: &ImgKey) -> Option<&ImgSlot> {
        self.slots.get(k)
    }

    /// Mark `k` as loading; returns false if it is already known.
    pub fn begin(&mut self, k: &ImgKey) -> bool {
        if self.slots.contains_key(k) {
            return false;
        }
        self.slots.insert(k.clone(), ImgSlot::Loading);
        self.order.push_back(k.clone());
        true
    }

    /// Forget everything (a new data source); returns the decoded images to free on the GPU.
    pub fn clear(&mut self) -> Vec<Arc<RenderImage>> {
        self.order.clear();
        self.slots
            .drain()
            .filter_map(|(_, s)| match s {
                ImgSlot::Ready(img) => Some(img),
                _ => None,
            })
            .collect()
    }

    /// Store a result; returns images evicted to stay within budget (free them on the GPU).
    pub fn finish(&mut self, k: ImgKey, img: Option<Arc<RenderImage>>) -> Vec<Arc<RenderImage>> {
        // A request evicted while it was still loading has no slot (and no place in `order`) any
        // more: drop its result instead of resurrecting an untracked entry.
        if !matches!(self.slots.get(&k), Some(ImgSlot::Loading)) {
            return img.into_iter().collect();
        }
        self.slots.insert(k, img.map_or(ImgSlot::Failed, ImgSlot::Ready));
        let mut evicted = Vec::new();
        while self.order.len() > Self::BUDGET {
            if let Some(old) = self.order.pop_front()
                && let Some(ImgSlot::Ready(img)) = self.slots.remove(&old)
            {
                evicted.push(img);
            }
        }
        evicted
    }
}

/// Decodes an encoded image upright, as the engine does (`folo_data::decode`: turned by its JPEG EXIF
/// orientation), so the boxes the server or the local predictor return land on what is shown. Keeps alpha.
pub fn decode_upright(bytes: &[u8]) -> Option<image::RgbaImage> {
    let mut img = image::load_from_memory(bytes).ok()?;
    if let Some(o) = folo_data::jpeg_exif_orientation(bytes).and_then(image::metadata::Orientation::from_exif)
    {
        img.apply_orientation(o);
    }
    Some(img.to_rgba8())
}

/// Turn backend pixels into a `RenderImage` (BGRA frames); encoded ones upright ([`decode_upright`]).
pub fn to_render_image(data: ImageData) -> Option<Arc<RenderImage>> {
    let (w, h, mut px) = match data {
        ImageData::Rgba { width, height, pixels } => (width, height, pixels),
        ImageData::Encoded(bytes) => {
            let img = decode_upright(&bytes)?;
            let (w, h) = img.dimensions();
            (w, h, img.into_raw())
        }
    };
    for p in px.as_chunks_mut::<4>().0 {
        p.swap(0, 2);
    }
    let buf = image::RgbaImage::from_raw(w, h, px)?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(buf)])))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_cache_clear_forgets_slots_and_returns_images() {
        let mut c = ImageCache::default();
        let k = |i: &str| ImgKey { dataset: "ds".into(), image: i.into(), kind: ImgKind::Thumb };
        let img = Arc::new(RenderImage::new(vec![image::Frame::new(image::RgbaImage::new(1, 1))]));
        assert!(c.begin(&k("a")));
        c.finish(k("a"), Some(img));
        assert!(c.begin(&k("b")));
        c.finish(k("b"), None); // a failure
        assert!(c.begin(&k("c"))); // still loading
        let freed = c.clear();
        assert_eq!(freed.len(), 1);
        // Nothing is remembered: every key can be fetched again, failures included.
        assert!(c.get(&k("a")).is_none() && c.get(&k("b")).is_none());
        assert!(c.begin(&k("b")));
        // A request that was in flight across the clear lands nowhere.
        let late = Arc::new(RenderImage::new(vec![image::Frame::new(image::RgbaImage::new(1, 1))]));
        assert_eq!(c.finish(k("c"), Some(late)).len(), 1);
    }

    #[test]
    fn cache_evicts_oldest_and_drops_late_completions() {
        let mut c = ImageCache::default();
        let key = |i: usize| ImgKey { dataset: "d".into(), image: i.to_string(), kind: ImgKind::Thumb };
        for i in 0..=ImageCache::BUDGET {
            assert!(c.begin(&key(i)));
        }
        // A newer request finishes first and evicts the oldest, still-loading one.
        c.finish(key(ImageCache::BUDGET), None);
        assert!(c.get(&key(0)).is_none());
        // Its late completion must not reappear outside the eviction order.
        c.finish(key(0), None);
        assert!(c.get(&key(0)).is_none());
        assert_eq!(c.slots.len(), ImageCache::BUDGET);
        // The next one in goes the same way: the oldest left out.
        assert!(c.begin(&key(ImageCache::BUDGET + 1)));
        c.finish(key(ImageCache::BUDGET + 1), None);
        assert!(c.get(&key(1)).is_none() && c.get(&key(2)).is_some());
        assert_eq!(c.slots.len(), ImageCache::BUDGET);
    }

    /// `jpeg` with an EXIF APP1 segment holding `orientation` right after its SOI marker.
    fn with_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        let mut seg = b"Exif\0\0II\x2A\x00\x08\x00\x00\x00".to_vec();
        for v in [1u16, 0x0112, 3] {
            seg.extend_from_slice(&v.to_le_bytes());
        }
        seg.extend_from_slice(&1u32.to_le_bytes());
        seg.extend_from_slice(&orientation.to_le_bytes());
        seg.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&seg);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn an_exif_rotated_photo_is_shown_as_the_engine_sees_it() {
        // 48x32 in four colour blocks, so every turn and mirror reads differently.
        let mut img = folo_data::Image::new(48, 32, [240, 20, 20]);
        img.fill_rect(24, 0, 24, 16, [20, 240, 20]);
        img.fill_rect(0, 16, 24, 16, [20, 20, 240]);
        img.fill_rect(24, 16, 24, 16, [240, 240, 20]);
        let jpeg = folo_data::encode(&img, folo_data::ImageFormat::Jpeg(95)).unwrap();
        for o in 1..=8u16 {
            let bytes = with_exif_orientation(&jpeg, o);
            let engine = folo_data::decode(&bytes).unwrap();
            let shown = decode_upright(&bytes).unwrap();
            assert_eq!(shown.dimensions(), (engine.width, engine.height), "orientation {o}");
            let rgb: Vec<u8> = shown.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
            assert!(rgb == engine.data, "orientation {o}: the pixels differ from the engine's");
            // The render image the Playground draws has the same upright size.
            let render = to_render_image(ImageData::Encoded(bytes)).unwrap();
            let s = render.size(0);
            assert_eq!((s.width.0 as u32, s.height.0 as u32), (engine.width, engine.height));
        }
    }

    #[test]
    fn rgba_converts_to_bgra() {
        let img =
            to_render_image(ImageData::Rgba { width: 1, height: 1, pixels: vec![10, 20, 30, 255] }).unwrap();
        assert_eq!(img.as_bytes(0).unwrap(), &[30, 20, 10, 255]);
    }
}
