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

/// Decoded images keyed by dataset/image/size, evicted oldest-first past a count and a byte budget.
pub struct ImageCache {
    pub slots: HashMap<ImgKey, ImgSlot>,
    order: VecDeque<ImgKey>,
    /// Decoded bytes held by the `Ready` slots.
    bytes: usize,
    /// [`Self::BYTE_BUDGET`] (smaller in tests).
    byte_budget: usize,
    /// How often each image failed in a row (its automatic retries: [`Self::retry_delay`]).
    failures: HashMap<ImgKey, Failure>,
    /// The token the next failure gets (every failure a new one).
    next_token: u64,
    /// The attempt each loading slot is on, a new one at every [`Self::begin`]: a fetch started for a
    /// slot evicted and asked for again meanwhile (a grid flung far past and back) is not the new
    /// slot's, and its answer is dropped ([`Self::attempt`]).
    loading: HashMap<ImgKey, u64>,
    next_attempt: u64,
}

/// An image's failures in a row, and its latest one's token: a retry timer set for an older failure
/// (one a reconnect or a Retry already answered) leaves a newer one alone.
#[derive(Clone, Copy, Default)]
struct Failure {
    count: u32,
    token: u64,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            byte_budget: Self::BYTE_BUDGET,
            failures: HashMap::new(),
            next_token: 0,
            loading: HashMap::new(),
            next_attempt: 0,
        }
    }
}

/// The decoded bytes of an image (its BGRA frame).
fn image_bytes(img: &RenderImage) -> usize {
    img.as_bytes(0).map_or(0, <[u8]>::len)
}

impl ImageCache {
    /// At most this many entries (thumbnails are small: 600 of 320 px are under 250 MB).
    pub const BUDGET: usize = 600;
    /// At most this many decoded bytes: a few full-size photos would otherwise hold gigabytes.
    pub const BYTE_BUDGET: usize = 768 << 20;
    /// Full-size images are decoded at most this long a side (64 MB each), so the byte budget always
    /// holds the one on screen with the grid behind it, and the texture fits every GPU.
    pub const FULL_MAX_SIDE: u32 = 4096;

    /// A cache holding at most `byte_budget` decoded bytes (tests of the app around it).
    #[cfg(test)]
    pub fn with_byte_budget(byte_budget: usize) -> Self {
        Self { byte_budget, ..Default::default() }
    }

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
        self.loading.insert(k.clone(), self.next_attempt);
        self.next_attempt += 1;
        true
    }

    /// The attempt `k` is loading on (`None` unless it is loading): a fetch carries the one its slot
    /// had when it started, and its answer is that slot's only while this still says the same.
    pub fn attempt(&self, k: &ImgKey) -> Option<u64> {
        self.loading.get(k).copied()
    }

    /// Forget everything (a new data source); returns the decoded images to free on the GPU.
    pub fn clear(&mut self) -> Vec<Arc<RenderImage>> {
        self.order.clear();
        self.failures.clear();
        self.loading.clear();
        self.bytes = 0;
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
        self.loading.remove(&k);
        self.bytes += img.as_deref().map_or(0, image_bytes);
        if img.is_some() {
            self.failures.remove(&k);
            // The image loads now: its other size, if it failed (its retries perhaps spent), is asked
            // for again at its next draw (a thumbnail that failed recovers once the inspector loads).
            let kind = match k.kind {
                ImgKind::Thumb => ImgKind::Full,
                ImgKind::Full => ImgKind::Thumb,
            };
            self.retry(&ImgKey { kind, ..k.clone() });
        } else {
            let f = self.failures.entry(k.clone()).or_default();
            f.count += 1;
            f.token = self.next_token;
            self.next_token += 1;
        }
        self.slots.insert(k.clone(), img.map_or(ImgSlot::Failed, ImgSlot::Ready));
        let mut evicted = Vec::new();
        // The image just finished stays (it is the one asked for): past the byte budget everything
        // older goes first.
        while self.order.len() > Self::BUDGET || (self.bytes > self.byte_budget && self.order.len() > 1) {
            let Some(old) = self.order.pop_front() else { break };
            if old == k {
                self.order.push_back(old);
                continue;
            }
            self.failures.remove(&old);
            self.loading.remove(&old);
            if let Some(ImgSlot::Ready(img)) = self.slots.remove(&old) {
                self.bytes -= image_bytes(&img);
                evicted.push(img);
            }
        }
        evicted
    }

    /// Forget the failures, so those images are asked for again (the server is back: what failed
    /// while it was away may well load now), each with its automatic retries afresh.
    pub fn forget_failed(&mut self) {
        self.slots.retain(|_, s| !matches!(s, ImgSlot::Failed));
        self.order.retain(|k| self.slots.contains_key(k));
        self.failures.clear();
    }

    /// Automatic retries of an image that failed (a dropped body, a file being rewritten) before it
    /// waits for the server to come back or a Retry.
    pub const RETRIES: u32 = 3;

    /// When to ask again for `k`, which just failed: 2, 8 then 32 s after its 1st, 2nd and 3rd
    /// failure in a row, with that failure's token for [`Self::forget`]; `None` past
    /// [`Self::RETRIES`] (a corrupt file is not fetched for ever).
    pub fn retry_delay(&self, k: &ImgKey) -> Option<(std::time::Duration, u64)> {
        let f = self.failures.get(k).copied().unwrap_or_default();
        (1..=Self::RETRIES)
            .contains(&f.count)
            .then(|| (std::time::Duration::from_secs(2 << (2 * (f.count - 1))), f.token))
    }

    /// Whether `k` failed (and waits for a retry).
    pub fn failed(&self, k: &ImgKey) -> bool {
        matches!(self.slots.get(k), Some(ImgSlot::Failed))
    }

    /// A Retry by hand: `k` is asked for again at the next draw, its automatic retries afresh.
    pub fn retry(&mut self, k: &ImgKey) {
        if self.failed(k) {
            self.slots.remove(k);
            self.order.retain(|o| o != k);
        }
        self.failures.remove(k);
    }

    /// Forget `k` if its latest failure is still the one `token` names (a timer set for an older
    /// one does nothing), so the next draw asks for it again; its count of failures stays.
    pub fn forget(&mut self, k: &ImgKey, token: u64) {
        if self.failed(k) && self.failures.get(k).is_some_and(|f| f.token == token) {
            self.slots.remove(k);
            self.order.retain(|o| o != k);
            // Counts of images forgotten and never drawn again are dropped once they pile up.
            if self.failures.len() > Self::BUDGET {
                let slots = &self.slots;
                self.failures.retain(|o, _| o == k || slots.contains_key(o));
            }
        }
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
    to_render_image_within(data, None)
}

/// [`to_render_image`], scaled down (aspect kept) so its long side is at most `max_side`.
pub fn to_render_image_within(data: ImageData, max_side: Option<u32>) -> Option<Arc<RenderImage>> {
    let mut img = match data {
        ImageData::Rgba { width, height, pixels } => image::RgbaImage::from_raw(width, height, pixels)?,
        ImageData::Encoded(bytes) => decode_upright(&bytes)?,
    };
    if let Some(m) = max_side {
        let (w, h) = img.dimensions();
        let long = w.max(h);
        if long > m {
            let side = |v: u32| (u64::from(v) * u64::from(m) / u64::from(long)).max(1) as u32;
            img = image::imageops::thumbnail(&img, side(w), side(h));
        }
    }
    let (w, h) = img.dimensions();
    let mut px = img.into_raw();
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

    #[test]
    fn a_slot_evicted_and_asked_for_again_is_on_a_new_attempt() {
        let mut c = ImageCache::default();
        let key = |i: usize| ImgKey { dataset: "d".into(), image: i.to_string(), kind: ImgKind::Thumb };
        assert!(c.begin(&key(0)));
        let old = c.attempt(&key(0)).expect("loading");
        // Evicted while its fetch is on its way, then drawn again: a second fetch, a new attempt.
        for i in 1..=ImageCache::BUDGET {
            assert!(c.begin(&key(i)));
            c.finish(key(i), None);
        }
        assert!(c.get(&key(0)).is_none() && c.attempt(&key(0)).is_none());
        assert!(c.begin(&key(0)));
        let new = c.attempt(&key(0)).expect("loading again");
        assert_ne!(old, new, "the first fetch's answer is told apart from the second's");
        // Only loading slots have one: nothing piles up for the finished and evicted ones.
        c.finish(key(0), Some(render(1, 1)));
        assert_eq!((c.attempt(&key(0)), c.loading.len()), (None, 0));
        assert!(c.begin(&key(ImageCache::BUDGET + 1)));
        assert_eq!(c.loading.len(), 1);
        assert!(c.clear().len() == 1 && c.loading.is_empty());
    }

    fn render(w: u32, h: u32) -> Arc<RenderImage> {
        Arc::new(RenderImage::new(vec![image::Frame::new(image::RgbaImage::new(w, h))]))
    }

    #[test]
    fn big_images_are_held_within_a_byte_budget_keeping_the_newest() {
        // 10x10 = 400 bytes each; the budget holds two.
        let mut c = ImageCache { byte_budget: 900, ..Default::default() };
        let key = |i: usize| ImgKey { dataset: "d".into(), image: i.to_string(), kind: ImgKind::Full };
        for i in 0..5 {
            assert!(c.begin(&key(i)));
            let evicted = c.finish(key(i), Some(render(10, 10)));
            assert_eq!(evicted.len(), usize::from(i >= 2), "image {i}");
            assert!(c.bytes <= 900);
        }
        assert!(c.get(&key(2)).is_none() && matches!(c.get(&key(4)), Some(ImgSlot::Ready(_))));
        // One image over the whole budget still shows: everything older goes, it stays.
        assert!(c.begin(&key(9)));
        assert_eq!(c.finish(key(9), Some(render(20, 20))).len(), 2);
        assert!(matches!(c.get(&key(9)), Some(ImgSlot::Ready(_))));
        assert_eq!(c.bytes, 1600);
        // The next result in evicts the big one (the inspector moved on).
        assert!(c.begin(&key(10)));
        assert_eq!(c.finish(key(10), Some(render(10, 10))).len(), 1);
        assert_eq!(c.bytes, 400);
        assert_eq!(c.clear().len(), 1);
        assert_eq!(c.bytes, 0);
    }

    #[test]
    fn failures_are_forgotten_on_request_and_ready_images_kept() {
        let mut c = ImageCache::default();
        let key = |i: &str| ImgKey { dataset: "d".into(), image: i.into(), kind: ImgKind::Thumb };
        for (k, img) in [("ok", Some(render(1, 1))), ("down", None)] {
            assert!(c.begin(&key(k)));
            c.finish(key(k), img);
        }
        assert!(c.begin(&key("loading")));
        c.forget_failed();
        assert!(c.get(&key("down")).is_none(), "asked for again");
        assert!(matches!(c.get(&key("ok")), Some(ImgSlot::Ready(_))));
        assert!(matches!(c.get(&key("loading")), Some(ImgSlot::Loading)));
        assert!(c.begin(&key("down")));
        assert_eq!(c.order.len(), 3);
    }

    #[test]
    fn a_retry_timer_set_for_an_older_failure_leaves_a_newer_one_alone() {
        let mut c = ImageCache::default();
        let k = ImgKey { dataset: "d".into(), image: "i".into(), kind: ImgKind::Full };
        let fail = |c: &mut ImageCache| {
            assert!(c.begin(&k));
            c.finish(k.clone(), None);
            c.retry_delay(&k)
        };
        let (wait, old) = fail(&mut c).unwrap();
        assert_eq!(wait.as_secs(), 2);
        // A reconnect answers it; the image fails again: a new first failure.
        c.forget_failed();
        let (wait, new) = fail(&mut c).unwrap();
        assert_eq!(wait.as_secs(), 2);
        c.forget(&k, old);
        assert!(c.failed(&k), "the older timer does nothing");
        c.forget(&k, new);
        assert!(c.get(&k).is_none(), "its own timer asks again");
        let (_, second) = fail(&mut c).unwrap();
        c.forget(&k, second);
        let (_, third) = fail(&mut c).unwrap();
        c.forget(&k, third);
        // The fourth failure in a row: no timer, and no older one asks again either.
        assert!(fail(&mut c).is_none());
        for t in [old, new, second, third] {
            c.forget(&k, t);
        }
        assert!(c.failed(&k));
        // A Retry by hand starts it over, its budget too.
        c.retry(&k);
        assert!(c.get(&k).is_none());
        assert_eq!(fail(&mut c).unwrap().0.as_secs(), 2);
    }

    #[test]
    fn an_image_that_loads_at_one_size_has_its_other_size_asked_for_again() {
        let mut c = ImageCache::default();
        let key = |kind| ImgKey { dataset: "d".into(), image: "i".into(), kind };
        let (thumb, full) = (key(ImgKind::Thumb), key(ImgKind::Full));
        // The thumbnail fails past its retries.
        for _ in 0..=ImageCache::RETRIES {
            c.slots.remove(&thumb);
            c.order.retain(|o| o != &thumb);
            assert!(c.begin(&thumb));
            c.finish(thumb.clone(), None);
        }
        assert!(c.retry_delay(&thumb).is_none());
        // Another image loading leaves it failed.
        assert!(c.begin(&full));
        let other = ImgKey { image: "j".into(), ..thumb.clone() };
        assert!(c.begin(&other));
        c.finish(other, Some(render(1, 1)));
        assert!(c.failed(&thumb));
        // Its full size loads: the thumbnail is asked for again, its retries afresh.
        c.finish(full.clone(), Some(render(1, 1)));
        assert!(c.get(&thumb).is_none());
        assert!(c.begin(&thumb));
        c.finish(thumb.clone(), None);
        assert_eq!(c.retry_delay(&thumb).unwrap().0.as_secs(), 2);
        assert!(matches!(c.get(&full), Some(ImgSlot::Ready(_))), "a thumbnail's failure leaves it alone");
    }

    #[test]
    fn full_images_are_scaled_down_to_the_cap_keeping_their_aspect() {
        let data = || ImageData::Rgba { width: 9000, height: 30, pixels: vec![7; 9000 * 30 * 4] };
        let s = to_render_image_within(data(), Some(ImageCache::FULL_MAX_SIDE)).unwrap().size(0);
        assert_eq!((s.width.0, s.height.0), (4096, 13));
        // Thin ones keep at least a pixel; small ones are left alone.
        let thin = ImageData::Rgba { width: 1, height: 20000, pixels: vec![7; 20000 * 4] };
        let s = to_render_image_within(thin, Some(4096)).unwrap().size(0);
        assert_eq!((s.width.0, s.height.0), (1, 4096));
        let s = to_render_image_within(data(), None).unwrap().size(0);
        assert_eq!((s.width.0, s.height.0), (9000, 30));
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
