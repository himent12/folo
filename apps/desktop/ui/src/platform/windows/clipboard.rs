//! The clipboard: text as `CF_UNICODETEXT`, and a string's metadata in GPUI's private format next to a
//! hash of the text (so metadata left over from an older copy is ignored).

use crate::{ClipboardEntry, ClipboardItem};
use anyhow::{Result, anyhow};
use std::sync::OnceLock;
use ::windows::Win32::Foundation::{HANDLE, HGLOBAL};
use ::windows::Win32::System::DataExchange::*;
use ::windows::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock};
use ::windows::Win32::System::Ole::CF_UNICODETEXT;
use ::windows::core::w;

fn hash_format() -> u32 {
    static F: OnceLock<u32> = OnceLock::new();
    // SAFETY: registering a constant format name.
    *F.get_or_init(|| unsafe { RegisterClipboardFormatW(w!("GPUI internal text hash")) })
}

fn metadata_format() -> u32 {
    static F: OnceLock<u32> = OnceLock::new();
    // SAFETY: registering a constant format name.
    *F.get_or_init(|| unsafe { RegisterClipboardFormatW(w!("GPUI internal metadata")) })
}

/// FNV-1a of the text: what the metadata entry is tied to.
fn text_hash(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Opens the clipboard, retrying briefly: another process may hold it for a moment.
fn with_clipboard<T>(f: impl FnOnce() -> T) -> Option<T> {
    for attempt in 0..5 {
        // SAFETY: no owner window; the clipboard is closed again below.
        if unsafe { OpenClipboard(None) }.is_ok() {
            let result = f();
            // SAFETY: we opened it.
            let _ = unsafe { CloseClipboard() };
            return Some(result);
        }
        std::thread::sleep(std::time::Duration::from_millis(2 << attempt));
    }
    None
}

/// Puts the item's first string on the clipboard (an item with no entries clears it).
pub(crate) fn write_to_clipboard(item: ClipboardItem) {
    let _ = with_clipboard(|| -> Result<()> {
        // SAFETY: the clipboard is open.
        unsafe { EmptyClipboard() }?;
        let Some(ClipboardEntry::String(string)) = item.entries().first() else { return Ok(()) };
        let wide: Vec<u16> = string.text.encode_utf16().chain(Some(0)).collect();
        set_data(&wide, CF_UNICODETEXT.0 as u32)?;
        if let Some(metadata) = &string.metadata {
            let hash = text_hash(&string.text).to_ne_bytes();
            set_data(&hash, hash_format())?;
            let wide: Vec<u16> = metadata.encode_utf16().chain(Some(0)).collect();
            set_data(&wide, metadata_format())?;
        }
        Ok(())
    });
}

/// The clipboard's text, with its metadata when it was written with some for this same text.
pub(crate) fn read_from_clipboard() -> Option<ClipboardItem> {
    with_clipboard(|| {
        let text = read_wide(CF_UNICODETEXT.0 as u32)?;
        let metadata = read_bytes(hash_format())
            .filter(|h| h.len() >= 8 && h[..8] == text_hash(&text).to_ne_bytes())
            .and_then(|_| read_wide(metadata_format()));
        Some(match metadata {
            Some(metadata) => ClipboardItem::new_string_with_metadata(text, metadata),
            None => ClipboardItem::new_string(text),
        })
    })
    .flatten()
}

fn set_data<T: Copy>(data: &[T], format: u32) -> Result<()> {
    let bytes = std::mem::size_of_val(data);
    // SAFETY: a fresh moveable allocation of `bytes`, locked, filled, unlocked, then owned by the
    // clipboard once SetClipboardData succeeds.
    unsafe {
        let global = GlobalAlloc(GMEM_MOVEABLE, bytes)?;
        let ptr = GlobalLock(global);
        if ptr.is_null() {
            return Err(anyhow!("GlobalLock failed"));
        }
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, bytes);
        let _ = GlobalUnlock(global);
        SetClipboardData(format, Some(HANDLE(global.0)))?;
    }
    Ok(())
}

fn read_bytes(format: u32) -> Option<Vec<u8>> {
    // SAFETY: the clipboard is open; the handle is locked while it is read and owned by the clipboard.
    unsafe {
        IsClipboardFormatAvailable(format).ok()?;
        let handle = GetClipboardData(format).ok()?;
        let global = HGLOBAL(handle.0);
        let size = GlobalSize(global);
        let ptr = GlobalLock(global);
        if ptr.is_null() {
            return None;
        }
        let bytes = std::slice::from_raw_parts(ptr as *const u8, size).to_vec();
        let _ = GlobalUnlock(global);
        Some(bytes)
    }
}

fn read_wide(format: u32) -> Option<String> {
    let bytes = read_bytes(format)?;
    let units: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_ne_bytes([c[0], c[1]])).collect();
    let end = units.iter().position(|&u| u == 0).unwrap_or(units.len());
    Some(String::from_utf16_lossy(&units[..end]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_hash_is_stable() {
        assert_eq!(text_hash(""), 0xcbf2_9ce4_8422_2325);
        assert_ne!(text_hash("a"), text_hash("b"));
    }
}
