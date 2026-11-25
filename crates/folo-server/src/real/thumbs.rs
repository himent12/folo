//! Bounded in-memory LRU cache for dataset image thumbnails, validated by file metadata and evicted on dataset
//! removal.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::SystemTime;

/// Maximum thumbnail entries held in memory per [`RealStore`](super::RealStore).
pub(super) const MAX_THUMB_ENTRIES: usize = 256;
/// Maximum total JPEG thumbnail bytes held in memory per [`RealStore`](super::RealStore) (16 MiB).
pub(super) const MAX_THUMB_BYTES: usize = 16 * 1024 * 1024;

/// File metadata stamp for a cached thumbnail source image.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ImageStamp {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    ctime: (i64, i64),
    #[cfg(windows)]
    created: u64,
}

impl ImageStamp {
    pub(super) fn from_metadata(md: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                len: md.len(),
                modified: md.modified().ok(),
                dev: md.dev(),
                ino: md.ino(),
                ctime: (md.ctime(), md.ctime_nsec()),
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            Self { len: md.len(), modified: md.modified().ok(), created: md.creation_time() }
        }
        #[cfg(not(any(unix, windows)))]
        {
            Self { len: md.len(), modified: md.modified().ok() }
        }
    }
}

/// Exact lookup key for one rendered dataset thumbnail (no lossy hash key).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ThumbKey {
    pub dataset_id: String,
    pub path: PathBuf,
    pub side: u32,
    pub stamp: ImageStamp,
}

/// Bounded LRU cache of encoded JPEG thumbnails capped by both entry count and total byte size.
pub(super) struct ThumbCache {
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    map: HashMap<ThumbKey, Vec<u8>>,
    order: VecDeque<ThumbKey>,
}

impl Default for ThumbCache {
    fn default() -> Self {
        Self::with_limits(MAX_THUMB_ENTRIES, MAX_THUMB_BYTES)
    }
}

impl ThumbCache {
    pub(super) fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self { max_entries, max_bytes, bytes: 0, map: HashMap::new(), order: VecDeque::new() }
    }

    pub(super) fn get(&mut self, key: &ThumbKey) -> Option<Vec<u8>> {
        let val = self.map.get(key)?.clone();
        self.touch(key);
        Some(val)
    }

    pub(super) fn insert(&mut self, key: ThumbKey, val: Vec<u8>) {
        if self.max_entries == 0 || val.len() > self.max_bytes {
            return;
        }
        if let Some(old) = self.map.insert(key.clone(), val) {
            self.bytes = self.bytes.saturating_sub(old.len());
            self.bytes += self.map[&key].len();
            self.touch(&key);
        } else {
            self.bytes += self.map[&key].len();
            self.order.push_back(key);
        }
        self.evict_excess();
    }

    /// Drops every cached thumbnail belonging to `dataset_id`.
    pub(super) fn evict_dataset(&mut self, dataset_id: &str) {
        self.order.retain(|k| {
            if k.dataset_id == dataset_id {
                if let Some(v) = self.map.remove(k) {
                    self.bytes = self.bytes.saturating_sub(v.len());
                }
                false
            } else {
                true
            }
        });
    }

    fn touch(&mut self, key: &ThumbKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key)
            && let Some(k) = self.order.remove(pos)
        {
            self.order.push_back(k);
        }
    }

    fn evict_excess(&mut self) {
        while self.map.len() > self.max_entries || self.bytes > self.max_bytes {
            let Some(oldest) = self.order.pop_front() else { break };
            if let Some(v) = self.map.remove(&oldest) {
                self.bytes = self.bytes.saturating_sub(v.len());
            }
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    pub(super) fn total_bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ds: &str, name: &str, side: u32) -> ThumbKey {
        ThumbKey {
            dataset_id: ds.into(),
            path: PathBuf::from(name),
            side,
            stamp: ImageStamp {
                len: 100,
                modified: None,
                #[cfg(unix)]
                dev: 1,
                #[cfg(unix)]
                ino: 1,
                #[cfg(unix)]
                ctime: (0, 0),
                #[cfg(windows)]
                created: 1,
            },
        }
    }

    #[test]
    fn thumb_cache_enforces_entry_and_byte_limits_with_lru_order() {
        let mut cache = ThumbCache::with_limits(3, 10);
        let (k1, k2, k3, k4) =
            (key("d1", "a", 64), key("d1", "b", 64), key("d1", "c", 64), key("d2", "d", 64));
        cache.insert(k1.clone(), vec![1; 4]);
        cache.insert(k2.clone(), vec![2; 4]);
        // Touch k1 so k2 becomes least recently used.
        assert_eq!(cache.get(&k1), Some(vec![1; 4]));
        // Adding 4 bytes exceeds max_bytes (10), evicting k2 (not k1).
        cache.insert(k3.clone(), vec![3; 4]);
        assert!(cache.get(&k2).is_none());
        assert!(cache.get(&k1).is_some());
        assert_eq!((cache.len(), cache.total_bytes()), (2, 8));

        // Oversized values are never cached.
        cache.insert(k4.clone(), vec![9; 11]);
        assert!(cache.get(&k4).is_none());
        assert_eq!((cache.len(), cache.total_bytes()), (2, 8));

        // Dataset eviction drops only that dataset's entries and updates byte accounting.
        cache.insert(key("d2", "e", 64), vec![4; 2]);
        assert_eq!((cache.len(), cache.total_bytes()), (3, 10));
        cache.evict_dataset("d1");
        assert_eq!((cache.len(), cache.total_bytes()), (1, 2));
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&key("d2", "e", 64)).is_some());
    }
}
