//! What a tool returns: the size limits of docs/MCP.md ("Paging and size limits"), one-line summaries and
//! pages.

use rmcp::model::ContentBlock;
use serde::Serialize;
use serde_json::Value;

use crate::errors::{Error, invalid_arguments};
use crate::types::*;

/// No tool result is larger than this many bytes of JSON (docs/MCP.md "Paging and size limits").
pub const MAX_RESULT_BYTES: usize = 32 * 1024;
/// List pages stop adding items once they reach this many bytes, leaving room for the envelope.
const PAGE_BUDGET: usize = 30 * 1024;

/// What a successful tool call returns.
pub struct Output {
    /// `structuredContent`.
    pub value: Value,
    /// The one-line text item.
    pub summary: String,
    /// Extra content (resource links).
    pub links: Vec<ContentBlock>,
}

/// Longest one-line summary, in characters.
pub(super) const MAX_SUMMARY_CHARS: usize = 300;
/// Longest name (dataset, run) accepted from tool arguments, in characters.
pub(super) const MAX_NAME_CHARS: usize = 128;
/// List summaries carry at most this many class names (`num_classes` counts all).
const SUMMARY_CLASSES: usize = 100;

/// `s` cut to at most `max` characters (with an ellipsis when cut).
pub(crate) fn short(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        None => s.to_string(),
        Some((i, _)) => format!("{}…", &s[..i]),
    }
}

/// A successful result. Every tool result goes through here, so none exceeds [`MAX_RESULT_BYTES`]: callers shrink
/// what has a bounded form, and anything still too large is the `result_too_large` tool error.
pub(super) fn out<T: Serialize>(value: &T, summary: impl Into<String>) -> Result<Output, Error> {
    let value =
        folo_api::json::to_value(value).map_err(|e| Error::internal(format!("serializing a result: {e}")))?;
    let n = json_len(&value);
    if n > MAX_RESULT_BYTES {
        return Err(Error::new(
            http::StatusCode::PAYLOAD_TOO_LARGE,
            "result_too_large",
            format!("the result would be {n} bytes of JSON, over the {MAX_RESULT_BYTES}-byte limit"),
        )
        .hint(
            "read the matching folo:// resource instead (e.g. folo://datasets/{id}), or narrow the request",
        ));
    }
    Ok(Output { value, summary: short(&summary.into(), MAX_SUMMARY_CHARS), links: Vec::new() })
}

/// The first [`SUMMARY_CLASSES`] class names, each cut to 64 characters.
pub(super) fn summary_classes(classes: &[String]) -> Vec<String> {
    classes.iter().take(SUMMARY_CLASSES).map(|c| short(c, 64)).collect()
}

pub(super) fn json_len<T: Serialize>(v: &T) -> usize {
    serde_json::to_vec(v).map_or(usize::MAX, |b| b.len())
}

/// `(offset, limit)` of a page request.
pub(super) fn page_window(cursor: Option<&str>, limit: Option<u32>) -> Result<(u64, u64), Error> {
    let offset = match cursor {
        None => 0,
        Some(c) => c.strip_prefix("o").and_then(|n| n.parse::<u64>().ok()).ok_or_else(|| {
            invalid_arguments(format!(
                "invalid cursor `{}`; pass a next_cursor from a previous page",
                short(c, 40)
            ))
        })?,
    };
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if !(1..=MAX_LIMIT).contains(&limit) {
        return Err(invalid_arguments(format!("limit must be in 1..={MAX_LIMIT}, got {limit}")));
    }
    Ok((offset, limit as u64))
}

pub(super) fn cursor_at(offset: u64) -> String {
    format!("o{offset}")
}

/// A page of `items` (already starting at `offset`), shortened to fit [`PAGE_BUDGET`].
pub(super) fn fit_page<T: Serialize>(
    items: impl IntoIterator<Item = T>,
    offset: u64,
    limit: u64,
    total: u64,
) -> Page<T> {
    let mut page = Vec::new();
    let mut bytes = 0;
    for item in items.into_iter().take(limit as usize) {
        let n = json_len(&item) + 1;
        if !page.is_empty() && bytes + n > PAGE_BUDGET {
            break;
        }
        bytes += n;
        page.push(item);
    }
    let end = offset.saturating_add(page.len() as u64);
    Page { next_cursor: (end < total).then(|| cursor_at(end)), total, items: page }
}

/// A page of an in-memory list.
pub(super) fn page_of<T: Serialize>(
    all: Vec<T>,
    cursor: Option<&str>,
    limit: Option<u32>,
) -> Result<Page<T>, Error> {
    let (offset, limit) = page_window(cursor, limit)?;
    let total = all.len() as u64;
    let skip = usize::try_from(offset).unwrap_or(usize::MAX);
    Ok(fit_page(all.into_iter().skip(skip), offset, limit, total))
}
