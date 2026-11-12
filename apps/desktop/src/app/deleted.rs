//! What was deleted lately (here or by another client): runs, models and datasets by id, with their
//! names. A link to one left behind (an earlier toast's Open, Back, the palette) opens its list instead,
//! saying so. Only the latest few hundred are kept; a link to an older one opens its page, which says
//! it was not found, as for any id the server doesn't know.

use std::collections::{HashMap, VecDeque};

/// How many deletions are remembered.
pub const KEEP: usize = 256;

/// The latest [`KEEP`] deletions, oldest first.
#[derive(Default)]
pub struct Deleted {
    names: HashMap<String, String>,
    order: VecDeque<String>,
}

impl Deleted {
    /// Notes `id` (called `name`) as deleted now; the oldest is forgotten past [`KEEP`].
    pub fn insert(&mut self, id: String, name: String) {
        if self.names.insert(id.clone(), name).is_some() {
            self.order.retain(|i| *i != id);
        }
        self.order.push_back(id);
        while self.order.len() > KEEP {
            if let Some(old) = self.order.pop_front() {
                self.names.remove(&old);
            }
        }
    }

    /// `id` is listed again (its folder registered again): no longer deleted.
    pub fn remove(&mut self, id: &str) {
        if self.names.remove(id).is_some() {
            self.order.retain(|i| i != id);
        }
    }

    /// The name of `id`, if it was deleted lately.
    pub fn get(&self, id: &str) -> Option<&String> {
        self.names.get(id)
    }

    pub fn clear(&mut self) {
        self.names.clear();
        self.order.clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.order.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{Deleted, KEEP};

    #[test]
    fn only_the_latest_deletions_are_kept() {
        let mut d = Deleted::default();
        for i in 0..KEEP + 44 {
            d.insert(format!("run_{i}"), format!("run {i}"));
        }
        assert_eq!(d.len(), KEEP);
        // The oldest are forgotten (their links get the ordinary not-found page); the latest stay.
        assert!(d.get("run_0").is_none() && d.get("run_43").is_none());
        assert_eq!(d.get("run_44").map(String::as_str), Some("run 44"));
        assert_eq!(
            d.get(&format!("run_{}", KEEP + 43)).map(String::as_str),
            Some(&*format!("run {}", KEEP + 43))
        );
        // Deleted again: counted once, as the latest.
        d.insert("run_44".into(), "run 44".into());
        assert_eq!(d.len(), KEEP);
        d.insert("one more".into(), "x".into());
        assert!(d.get("run_44").is_some() && d.get("run_45").is_none());
        // Listed again: gone from the memory, and its slot with it.
        d.remove("run_44");
        assert!(d.get("run_44").is_none());
        assert_eq!(d.len(), KEEP - 1);
    }
}
