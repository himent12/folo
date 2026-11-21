//! The per-folder half of the scan cache: image header probes and YOLO label parses, keyed by file name and
//! checked against the file's [`Stamp`].

use super::listing::{Entry, Listing};
use super::{Dec, Enc, Settings, Stamp};
use crate::bbox::BBox;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// What reading an image's header gave.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ImageProbe {
    /// `(width, height)` after EXIF orientation.
    Size(u32, u32),
    /// The codec's message (a [`crate::DataError::Image`] without its path): the image cannot be decoded.
    Corrupt(String),
}

/// What parsing a YOLO label file gave.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LabelParse {
    /// The boxes and the warnings about them.
    Parsed { boxes: Vec<BBox>, warnings: Vec<String> },
    /// Why the file is broken (the message of its [`super::super::IssueKind::BrokenLabel`] issue).
    Broken(String),
}

/// A result learnt during this open, for the cache.
pub(crate) enum Fresh {
    Image(PathBuf, Stamp, ImageProbe),
    Label(PathBuf, Stamp, Option<usize>, LabelParse),
}

#[derive(Default)]
struct DirCache {
    images: HashMap<String, (Stamp, ImageProbe)>,
    /// `(stamp, number of classes the labels were checked against, result)`
    labels: HashMap<String, (Stamp, Option<u64>, LabelParse)>,
}

pub(super) const KIND: &str = "dir";

/// The listings and cached results of one [`super::super::build_entries`] call.
pub(crate) struct Probes {
    settings: Settings,
    listing: Listing,
    /// Cached results of the listed folders (when caching is on and the folder has a [`key`]), by folder.
    dirs: HashMap<PathBuf, DirCache>,
    /// The [`key`] of every folder in `dirs`.
    keys: HashMap<PathBuf, String>,
    now: SystemTime,
}

impl Probes {
    /// Lists the folders of `files` (images and label files) and loads their caches.
    pub fn new<'a>(files: impl Iterator<Item = &'a Path>, settings: Settings) -> Self {
        let listing = Listing::new(files, &settings);
        let (mut dirs, mut keys) = (HashMap::new(), HashMap::new());
        if let Some(cache) = &settings.dir {
            for dir in listing.dirs() {
                let Some(k) = key(dir) else { continue };
                let loaded = super::read(cache, KIND, &k).and_then(|b| decode(&b)).unwrap_or_default();
                dirs.insert(dir.to_path_buf(), loaded);
                keys.insert(dir.to_path_buf(), k);
            }
        }
        let now = settings.now.unwrap_or_else(SystemTime::now);
        Probes { settings, listing, dirs, keys, now }
    }

    /// What is at `path` (see [`Listing::stat`]).
    pub fn stat(&self, path: &Path) -> Option<Entry> {
        self.listing.stat(path)
    }

    /// Whether results for `path`, last seen with `stamp`, may be cached.
    pub fn cacheable(&self, path: &Path, stamp: Option<Stamp>) -> Option<Stamp> {
        let dir = path.parent()?;
        // Names that are not UTF-8 are not cached.
        let _name = path.file_name()?.to_str()?;
        stamp.filter(|s| self.dirs.contains_key(dir) && s.settled(self.now))
    }

    /// The cached header probe of `path`, when its stamp still matches.
    pub fn image(&self, path: &Path, stamp: Stamp) -> Option<&ImageProbe> {
        let (s, p) = self.dirs.get(path.parent()?)?.images.get(path.file_name()?.to_str()?)?;
        (*s == stamp).then_some(p)
    }

    /// The cached parse of the label file `path`, when its stamp and the number of classes still match.
    pub fn label(&self, path: &Path, stamp: Stamp, num_classes: Option<usize>) -> Option<&LabelParse> {
        let (s, nc, p) = self.dirs.get(path.parent()?)?.labels.get(path.file_name()?.to_str()?)?;
        (*s == stamp && *nc == num_classes.map(|n| n as u64)).then_some(p)
    }

    /// Writes the caches of the folders that learnt something: their old entries for files still there unchanged,
    /// plus `fresh`.
    pub fn save(mut self, fresh: Vec<Fresh>) {
        let Some(cache) = self.settings.dir.take() else { return };
        if fresh.is_empty() {
            return;
        }
        let mut touched: HashMap<PathBuf, DirCache> = HashMap::new();
        for f in fresh {
            let (path, name) = match &f {
                Fresh::Image(p, ..) | Fresh::Label(p, ..) => (p, p.file_name().and_then(|n| n.to_str())),
            };
            let (Some(dir), Some(name)) = (path.parent(), name) else { continue };
            let name = name.to_string();
            let d = match touched.get_mut(dir) {
                Some(d) => d,
                None => {
                    let old = self.dirs.remove(dir).unwrap_or_default();
                    touched.entry(dir.to_path_buf()).or_insert(self.still_valid(dir, old))
                }
            };
            match f {
                Fresh::Image(_, s, p) => {
                    d.images.insert(name, (s, p));
                }
                Fresh::Label(_, s, nc, p) => {
                    d.labels.insert(name, (s, nc.map(|n| n as u64), p));
                }
            }
        }
        for (dir, d) in touched {
            if let Some(k) = self.keys.get(&dir) {
                super::write(&cache, KIND, k, &encode(&d));
            }
        }
    }

    /// `old` without the entries of files that are gone or changed.
    fn still_valid(&self, dir: &Path, mut old: DirCache) -> DirCache {
        let same = |name: &str, s: &Stamp| self.stat(&dir.join(name)).and_then(|e| e.stamp) == Some(*s);
        old.images.retain(|n, (s, _)| same(n, s));
        old.labels.retain(|n, (s, _, _)| same(n, s));
        old
    }
}

/// The cache key of a folder: its canonical path (links resolved, so two spellings of one folder share a key and
/// two folders never do). `None` (the folder is not cached) when it cannot be canonicalised or is not UTF-8.
fn key(dir: &Path) -> Option<String> {
    std::fs::canonicalize(dir).ok()?.into_os_string().into_string().ok()
}

fn encode(d: &DirCache) -> Vec<u8> {
    let mut e = Enc::default();
    e.len(d.images.len());
    for (name, (s, p)) in &d.images {
        e.str(name);
        e.stamp(*s);
        match p {
            ImageProbe::Size(w, h) => {
                e.u8(0);
                e.u32(*w);
                e.u32(*h);
            }
            ImageProbe::Corrupt(m) => {
                e.u8(1);
                e.str(m);
            }
        }
    }
    e.len(d.labels.len());
    for (name, (s, nc, p)) in &d.labels {
        e.str(name);
        e.stamp(*s);
        e.u64(nc.unwrap_or(u64::MAX));
        match p {
            LabelParse::Parsed { boxes, warnings } => {
                e.u8(0);
                e.len(boxes.len());
                for b in boxes {
                    e.u32(b.class);
                    for v in [b.cx, b.cy, b.w, b.h] {
                        e.f32(v);
                    }
                }
                e.len(warnings.len());
                for w in warnings {
                    e.str(w);
                }
            }
            LabelParse::Broken(m) => {
                e.u8(1);
                e.str(m);
            }
        }
    }
    e.0
}

fn decode(bytes: &[u8]) -> Option<DirCache> {
    let mut d = Dec(bytes);
    let mut out = DirCache::default();
    for _ in 0..d.len(25)? {
        let name = d.str()?.to_string();
        let s = d.stamp()?;
        let p = match d.u8()? {
            0 => ImageProbe::Size(d.u32()?, d.u32()?),
            1 => ImageProbe::Corrupt(d.str()?.to_string()),
            _ => return None,
        };
        out.images.insert(name, (s, p));
    }
    for _ in 0..d.len(33)? {
        let name = d.str()?.to_string();
        let s = d.stamp()?;
        let nc = Some(d.u64()?).filter(|&n| n != u64::MAX);
        let p = match d.u8()? {
            0 => {
                let mut boxes = Vec::new();
                for _ in 0..d.len(20)? {
                    let class = d.u32()?;
                    boxes.push(BBox::new(class, d.f32()?, d.f32()?, d.f32()?, d.f32()?));
                }
                let mut warnings = Vec::new();
                for _ in 0..d.len(8)? {
                    warnings.push(d.str()?.to_string());
                }
                LabelParse::Parsed { boxes, warnings }
            }
            1 => LabelParse::Broken(d.str()?.to_string()),
            _ => return None,
        };
        out.labels.insert(name, (s, nc, p));
    }
    d.is_empty().then_some(out)
}
