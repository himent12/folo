// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform.rs (ClipboardItem)
//! `ClipboardItem`: what goes on and comes off the clipboard (text, optionally with metadata).

/// A clipboard item that should be copied to the clipboard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipboardItem {
    entries: Vec<ClipboardEntry>,
}

/// One entry of a clipboard item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClipboardEntry {
    /// A string entry.
    String(ClipboardString),
}

/// A clipboard string: its text and optional metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipboardString {
    /// The text.
    pub text: String,
    /// Metadata the app attached (round-trips only through this app).
    pub metadata: Option<String>,
}

impl ClipboardString {
    /// A string entry with no metadata.
    pub fn new(text: String) -> Self {
        Self { text, metadata: None }
    }
}

impl ClipboardItem {
    /// Create a new ClipboardItem::String with no associated metadata.
    pub fn new_string(text: String) -> Self {
        Self { entries: vec![ClipboardEntry::String(ClipboardString::new(text))] }
    }

    /// Create a new ClipboardItem::String with the given text and associated metadata.
    pub fn new_string_with_metadata(text: String, metadata: String) -> Self {
        Self { entries: vec![ClipboardEntry::String(ClipboardString { text, metadata: Some(metadata) })] }
    }

    /// Concatenates together all the ClipboardString entries in the item. `None` if there are none.
    pub fn text(&self) -> Option<String> {
        let mut answer = String::new();
        let mut any_entries = false;
        for entry in self.entries.iter() {
            let ClipboardEntry::String(ClipboardString { text, metadata: _ }) = entry;
            answer.push_str(text);
            any_entries = true;
        }
        if any_entries { Some(answer) } else { None }
    }

    /// If this item is one string entry, its metadata.
    pub fn metadata(&self) -> Option<&String> {
        match self.entries.as_slice() {
            [ClipboardEntry::String(s)] => s.metadata.as_ref(),
            _ => None,
        }
    }

    /// The item's entries.
    pub fn entries(&self) -> &[ClipboardEntry] {
        &self.entries
    }
}
