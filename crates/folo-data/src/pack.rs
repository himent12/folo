//! Dataset packing: an opened dataset as one tar stream in the canonical YOLO layout, written on the fly (no
//! temporary files), so a dataset of any format and layout can be sent to another FOLO server as FOLO reads it
//! (docs/DATA.md, Packing; docs/CLOUD.md, Dataset).
//!
//! The stream holds `data.yaml` (the class names in order and the split folders), then for every split, every image
//! in dataset order as `images/<split>/<index>.<ext>` (the source file's bytes, its own extension) followed by
//! `labels/<split>/<index>.txt` (written from the parsed boxes; empty for an image without boxes). `<index>` is the
//! image's 0-based position in its split, zero-padded to 7 digits (more for larger splits), so names sort in dataset
//! order. Headers are ustar with a fixed mode, owner and modification time: the bytes depend only on the dataset.
//! Reading the stream back with [`crate::open`] gives the same class names, split sizes, image order and boxes.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::Path;

use crate::convert::image_extension;
use crate::dataset::{Dataset, LoadedDataset};
use crate::error::{DataError, Result};
use crate::{BBox, Split};

/// Size of a tar block: headers are one block, file contents are padded to whole blocks.
const BLOCK: u64 = 512;
/// The end-of-archive marker: two zero blocks.
const END: [u8; 1024] = [0; 1024];
/// Modification time of every entry (the Unix epoch), so the stream depends only on the dataset.
const MTIME: u64 = 0;
/// Fewest digits of an image's index in its file name.
const MIN_DIGITS: usize = 7;

/// One file of the stream.
enum Content<'a> {
    /// A file copied byte for byte, of this length.
    File(&'a Path, u64),
    /// Text written by the packer.
    Text(&'a str),
}

impl Content<'_> {
    fn len(&self) -> u64 {
        match self {
            Content::File(_, len) => *len,
            Content::Text(t) => t.len() as u64,
        }
    }
}

/// Writes `ds` to `out` as a tar stream (see the module docs) and returns the bytes written, which equal
/// [`packed_len`]. Images are streamed from their files, one at a time. Fails with [`DataError::Config`] before
/// writing anything when the dataset has no training or validation images (the YOLO reader would not open it), and
/// with [`DataError::Io`] when an image file cannot be read (or changed size since it was measured) or `out` fails.
pub fn pack<W: Write + ?Sized>(ds: &LoadedDataset, out: &mut W) -> Result<u64> {
    let mut written = 0u64;
    visit(ds, |name, content| {
        let mut header = tar::Header::new_ustar();
        header.set_path(name).map_err(|e| DataError::io(name, e))?;
        header.set_size(content.len());
        header.set_mode(0o644);
        header.set_mtime(MTIME);
        header.set_uid(0);
        header.set_gid(0);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();
        // A failing `out` is reported with the entry it was writing.
        out.write_all(header.as_bytes()).map_err(|e| DataError::io(name, e))?;
        match content {
            Content::File(file, len) => {
                let f = std::fs::File::open(file).map_err(|e| DataError::io(file, e))?;
                let copied = std::io::copy(&mut f.take(len), out).map_err(|e| DataError::io(file, e))?;
                if copied != len {
                    let e = std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!("the file shrank from {len} to {copied} bytes while it was packed"),
                    );
                    return Err(DataError::io(file, e));
                }
            }
            Content::Text(text) => out.write_all(text.as_bytes()).map_err(|e| DataError::io(name, e))?,
        }
        let padding = padded(content.len()) - content.len();
        out.write_all(&END[..padding as usize]).map_err(|e| DataError::io(name, e))?;
        written += BLOCK + padded(content.len());
        Ok(())
    })?;
    out.write_all(&END).map_err(|e| DataError::io("the end of the archive", e))?;
    Ok(written + END.len() as u64)
}

/// The exact length of [`pack`]'s stream for `ds`, from the image files' sizes (nothing is read), for progress.
pub fn packed_len(ds: &LoadedDataset) -> Result<u64> {
    let mut len = 0;
    visit(ds, |_, content| {
        len += BLOCK + padded(content.len());
        Ok(())
    })?;
    Ok(len + END.len() as u64)
}

/// `len` rounded up to whole blocks.
fn padded(len: u64) -> u64 {
    len.div_ceil(BLOCK) * BLOCK
}

/// Calls `f` with every file of the stream in order: its name in the archive and its content.
fn visit(ds: &LoadedDataset, mut f: impl FnMut(&str, Content) -> Result<()>) -> Result<()> {
    if ds.train.is_empty() && ds.val.is_empty() {
        return Err(DataError::Config(format!(
            "nothing to pack: {} has no training or validation images",
            crate::paths::shown(&ds.root)
        )));
    }
    f("data.yaml", Content::Text(&yaml(ds)))?;
    let mut label = String::new();
    for split in Split::ALL {
        let d = ds.split(split);
        let digits = MIN_DIGITS.max(d.len().saturating_sub(1).to_string().len());
        for (i, e) in d.entries().iter().enumerate() {
            let len = std::fs::metadata(&e.image).map_err(|err| DataError::io(&e.image, err))?.len();
            let name = format!("images/{split}/{i:0digits$}.{}", image_extension(&e.image));
            f(&name, Content::File(&e.image, len))?;
            label.clear();
            write_label(&mut label, &e.boxes);
            f(&format!("labels/{split}/{i:0digits$}.txt"), Content::Text(&label))?;
        }
    }
    Ok(())
}

/// `data.yaml`: the split folders that have images and the class names in order.
fn yaml(ds: &LoadedDataset) -> String {
    let mut yaml = String::from("# packed by FOLO\n");
    for split in Split::ALL.into_iter().filter(|&s| !ds.split(s).is_empty()) {
        let _ = writeln!(yaml, "{split}: images/{split}");
    }
    yaml.push_str("names:\n");
    for (i, c) in ds.classes.iter().enumerate() {
        // JSON strings are valid YAML double-quoted scalars.
        let _ = writeln!(yaml, "  {i}: {}", serde_json::to_string(c).expect("strings serialize"));
    }
    yaml
}

/// YOLO label lines of `boxes`. `f32`'s `Display` is the shortest text that parses back to the same value, so the
/// reader gets the boxes exactly.
fn write_label(out: &mut String, boxes: &[BBox]) {
    for b in boxes {
        let _ = writeln!(out, "{} {} {} {} {}", b.class, b.cx, b.cy, b.w, b.h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip_exactly() {
        let boxes = [BBox::new(0, 0.1, 1.0 / 3.0, 0.2, 2.0 / 7.0), BBox::new(4, 0.5, 0.5, 1.0, 1.0)];
        let mut text = String::new();
        write_label(&mut text, &boxes);
        let parsed = crate::dataset::labels::parse_yolo(&text, Some(5)).unwrap();
        assert_eq!(parsed.boxes, boxes);
        assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
    }

    #[test]
    fn lengths_are_whole_blocks() {
        assert_eq!((padded(0), padded(1), padded(512), padded(513)), (0, 512, 512, 1024));
    }
}
