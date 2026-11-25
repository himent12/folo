//! Bounded LRU cache of recent `predict_batch` payloads (`folo://predictions/{batch_id}`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde_json::Value;

/// Maximum number of `predict_batch` results kept in memory for a session.
pub(crate) const MAX_BATCHES: usize = 64;

/// Approximate serialized-JSON byte budget across all cached `predict_batch` results.
/// Any single batch larger than this budget is rejected without evicting existing entries.
pub(crate) const MAX_BATCH_BYTES: usize = 32 * 1024 * 1024;

struct Entry {
    value: Arc<Value>,
    bytes: usize,
}

/// Outcome of looking up `folo://predictions/{batch_id}` in [`BatchCache`].
pub(crate) enum BatchLookup {
    /// Cached payload (promoted to most recently used).
    Found(Arc<Value>),
    /// The batch ID was previously issued by this server and later evicted by the cache bounds.
    Evicted,
    /// No such batch ID was issued in this session.
    NotFound,
}

/// Bounded LRU cache of `predict_batch` results, capped by entry count and approximate JSON size.
pub(crate) struct BatchCache {
    max_entries: usize,
    max_bytes: usize,
    total_bytes: usize,
    seq: u64,
    /// Keys ordered from least recently used (front) to most recently used (back).
    order: VecDeque<String>,
    entries: HashMap<String, Entry>,
}

impl BatchCache {
    /// Creates a cache with the default [`MAX_BATCHES`] and [`MAX_BATCH_BYTES`] bounds.
    pub(crate) fn new() -> Self {
        Self::with_limits(MAX_BATCHES, MAX_BATCH_BYTES)
    }

    /// Creates a cache with explicit entry-count and byte-budget limits.
    pub(crate) fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            total_bytes: 0,
            seq: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Allocates the next `pb_<n>` ID, builds its payload, and inserts it atomically under the
    /// cache lock. Returns `None` without advancing the ID sequence or evicting existing entries if
    /// the built payload exceeds the cache's byte budget (or capacity is 0).
    pub(crate) fn try_insert_next(&mut self, build: impl FnOnce(&str) -> Value) -> Option<String> {
        if self.max_entries == 0 {
            return None;
        }
        let next_seq = self.seq.saturating_add(1);
        let id = format!("pb_{next_seq}");
        let value = Arc::new(build(&id));
        if self.try_insert(id.clone(), value) {
            self.seq = next_seq;
            Some(id)
        } else {
            None
        }
    }

    /// Inserts or replaces `id` as the most recently used entry, evicting least recently used
    /// entries until the cache fits within both `max_entries` and `max_bytes`.
    /// Returns `false` (leaving existing entries untouched) if `max_entries == 0` or `value` alone
    /// exceeds `max_bytes`.
    pub(crate) fn try_insert(&mut self, id: String, value: Arc<Value>) -> bool {
        if self.max_entries == 0 {
            return false;
        }
        let bytes = estimate_json_bytes(&value);
        if bytes > self.max_bytes {
            return false;
        }
        if let Some(old) = self.entries.get_mut(&id) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes).saturating_add(bytes);
            old.value = value;
            old.bytes = bytes;
            self.promote(&id);
        } else {
            self.order.push_back(id.clone());
            self.entries.insert(id, Entry { value, bytes });
            self.total_bytes = self.total_bytes.saturating_add(bytes);
        }
        while self.entries.len() > self.max_entries || self.total_bytes > self.max_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
            }
        }
        true
    }

    /// Returns the cached payload for `id` and marks it most recently used.
    pub(crate) fn get(&mut self, id: &str) -> Option<Arc<Value>> {
        let value = Arc::clone(&self.entries.get(id)?.value);
        self.promote(id);
        Some(value)
    }

    /// Looks up `id`, distinguishing a previously issued (and evicted) batch from an unknown ID.
    pub(crate) fn lookup(&mut self, id: &str) -> BatchLookup {
        if let Some(v) = self.get(id) {
            return BatchLookup::Found(v);
        }
        if is_issued_batch_id(id, self.seq) { BatchLookup::Evicted } else { BatchLookup::NotFound }
    }

    fn promote(&mut self, id: &str) {
        if self.order.back().is_some_and(|k| k == id) {
            return;
        }
        if let Some(pos) = self.order.iter().position(|k| k == id)
            && let Some(key) = self.order.remove(pos)
        {
            self.order.push_back(key);
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }
}

/// Returns true iff `id` is a canonical `pb_<n>` ID with `1 <= n <= issued`.
fn is_issued_batch_id(id: &str, issued: u64) -> bool {
    let Some(rest) = id.strip_prefix("pb_") else {
        return false;
    };
    if rest.is_empty() || rest.starts_with('0') || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    rest.parse::<u64>().is_ok_and(|n| n <= issued)
}

/// Estimates the serialized JSON size of `v` iteratively without allocating a JSON string.
fn estimate_json_bytes(v: &Value) -> usize {
    let mut total = 0usize;
    let mut stack = vec![v];
    while let Some(cur) = stack.pop() {
        match cur {
            Value::Null => total = total.saturating_add(4),
            Value::Bool(true) => total = total.saturating_add(4),
            Value::Bool(false) => total = total.saturating_add(5),
            Value::Number(_) => total = total.saturating_add(8),
            Value::String(s) => total = total.saturating_add(s.len()).saturating_add(2),
            Value::Array(items) => {
                total = total.saturating_add(2).saturating_add(items.len());
                stack.extend(items.iter());
            }
            Value::Object(map) => {
                total = total.saturating_add(2);
                for (k, val) in map {
                    total = total.saturating_add(k.len()).saturating_add(4);
                    stack.push(val);
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn evicts_least_recently_used_and_promotes_on_get() {
        let mut cache = BatchCache::with_limits(3, usize::MAX);
        assert!(cache.try_insert("pb_1".into(), Arc::new(json!({ "id": 1 }))));
        assert!(cache.try_insert("pb_2".into(), Arc::new(json!({ "id": 2 }))));
        assert!(cache.try_insert("pb_3".into(), Arc::new(json!({ "id": 3 }))));
        assert_eq!(cache.len(), 3);

        // Accessing pb_1 promotes it to MRU, so pb_2 becomes LRU.
        assert_eq!(cache.get("pb_1").unwrap()["id"], 1);

        assert!(cache.try_insert("pb_4".into(), Arc::new(json!({ "id": 4 }))));
        assert_eq!(cache.len(), 3);
        assert!(cache.contains("pb_1"));
        assert!(!cache.contains("pb_2"), "pb_2 should have been evicted as LRU");
        assert!(cache.contains("pb_3"));
        assert!(cache.contains("pb_4"));

        // Re-inserting an existing key updates its value and promotes it without growing the cache.
        assert!(cache.try_insert("pb_3".into(), Arc::new(json!({ "id": 33 }))));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get("pb_3").unwrap()["id"], 33);

        // Now order is pb_1 (LRU), pb_4, pb_3 (MRU).
        assert!(cache.try_insert("pb_5".into(), Arc::new(json!({ "id": 5 }))));
        assert!(!cache.contains("pb_1"));
        assert!(cache.contains("pb_3"));
        assert!(cache.contains("pb_4"));
        assert!(cache.contains("pb_5"));
    }

    #[test]
    fn enforces_byte_budget_and_rejects_oversized_single_entry() {
        let payload = |n: usize| Arc::new(json!({ "blob": "x".repeat(n) }));
        let one_size = estimate_json_bytes(&payload(100));
        let budget = one_size * 2 + 10;
        let mut cache = BatchCache::with_limits(10, budget);

        assert!(cache.try_insert("pb_1".into(), payload(100)));
        assert!(cache.try_insert("pb_2".into(), payload(100)));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), one_size * 2);

        // Third insert exceeds byte budget, evicting pb_1.
        assert!(cache.try_insert("pb_3".into(), payload(100)));
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains("pb_1"));
        assert!(cache.contains("pb_2"));
        assert!(cache.contains("pb_3"));
        assert_eq!(cache.total_bytes(), one_size * 2);

        // A single entry larger than the budget is rejected without evicting valid entries or
        // leaving the cache over budget.
        assert!(!cache.try_insert("pb_big".into(), payload(1000)));
        assert_eq!(cache.len(), 2);
        assert!(cache.contains("pb_2"));
        assert!(cache.contains("pb_3"));
        assert!(!cache.contains("pb_big"));
        assert!(cache.total_bytes() <= budget);

        // Same through `try_insert_next`: rejected entry does not advance `seq` or evict entries.
        assert!(cache.try_insert_next(|id| json!({ "batch_id": id, "blob": "x".repeat(1000) })).is_none());
        assert_eq!(cache.len(), 2);
        assert!(cache.contains("pb_2"));
        assert!(cache.contains("pb_3"));
        assert_eq!(cache.try_insert_next(|id| json!({ "batch_id": id })).as_deref(), Some("pb_1"));
    }

    #[test]
    fn zero_capacity_never_stores_entries() {
        let mut cache = BatchCache::with_limits(0, 1024);
        assert!(!cache.try_insert("pb_1".into(), Arc::new(json!({ "id": 1 }))));
        assert!(cache.try_insert_next(|id| json!({ "batch_id": id })).is_none());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.get("pb_1").is_none());
    }

    #[test]
    fn insert_next_and_lookup_distinguish_evicted_from_unknown() {
        let mut cache = BatchCache::with_limits(2, usize::MAX);
        let id1 = cache.try_insert_next(|id| json!({ "batch_id": id })).unwrap();
        let id2 = cache.try_insert_next(|id| json!({ "batch_id": id })).unwrap();
        let id3 = cache.try_insert_next(|id| json!({ "batch_id": id })).unwrap();
        assert_eq!((id1.as_str(), id2.as_str(), id3.as_str()), ("pb_1", "pb_2", "pb_3"));
        assert!(matches!(cache.lookup("pb_1"), BatchLookup::Evicted));
        assert!(matches!(cache.lookup("pb_2"), BatchLookup::Found(v) if v["batch_id"] == "pb_2"));
        assert!(matches!(cache.lookup("pb_3"), BatchLookup::Found(v) if v["batch_id"] == "pb_3"));
        assert!(matches!(cache.lookup("pb_4"), BatchLookup::NotFound));
        assert!(matches!(cache.lookup("pb_01"), BatchLookup::NotFound));
        assert!(matches!(cache.lookup("pb_0"), BatchLookup::NotFound));
        assert!(matches!(cache.lookup("bogus"), BatchLookup::NotFound));
    }
}
