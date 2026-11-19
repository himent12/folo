//! The COCO half of the scan cache: [`Annotations`] (everything derived from a dataset's JSON files) keyed by the
//! files' paths and checked against their [`Stamp`]s.

use super::{Dec, Enc, Settings, Stamp};
use crate::dataset::coco::{Annotations, ImageAnnotations, SplitAnnotations};
use crate::dataset::{Issue, IssueKind, Split};
use crate::error::Result;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub(super) const KIND: &str = "coco";

/// [`Annotations`] of `jsons` (and `wanted`), from the cache while every file is unchanged, else from `read` (and
/// then cached, for a dataset with at least [`Settings::min_files`] images).
pub(crate) fn load_or_read(
    settings: &Settings,
    jsons: &[(Split, PathBuf)],
    wanted: &[Split],
    read: impl FnOnce() -> Result<Annotations>,
) -> Result<Annotations> {
    let Some(dir) = &settings.dir else { return read() };
    let Some(key) = key(jsons, wanted) else { return read() };
    // Stamps before the files are read: a file changed while it is read then misses next time.
    let stamps: Option<Vec<Stamp>> =
        jsons.iter().map(|(_, p)| std::fs::metadata(p).ok().and_then(|m| Stamp::of(&m))).collect();
    let Some(stamps) = stamps else { return read() };
    if let Some(a) = super::read(dir, KIND, &key).and_then(|b| decode(&b, &stamps)) {
        return Ok(a);
    }
    let a = read()?;
    let images: usize = a.splits.iter().map(|s| s.images.len()).sum();
    let now = settings.now.unwrap_or_else(SystemTime::now);
    if images >= settings.min_files
        && stamps.iter().all(|s| s.settled(now))
        && let Some(bytes) = encode(&a, &stamps)
    {
        super::write(dir, KIND, &key, &bytes);
    }
    Ok(a)
}

/// The files as given (issue paths are built from them) and as canonical paths (links resolved: the identity of
/// each file), and the wanted splits. `None` (not cached) for a path that cannot be canonicalised or is not UTF-8.
fn key(jsons: &[(Split, PathBuf)], wanted: &[Split]) -> Option<String> {
    let mut k = String::from("wanted");
    for s in Split::ALL {
        k.push_str(if wanted.contains(&s) { " 1" } else { " 0" });
    }
    for (split, p) in jsons {
        k.push('\n');
        k.push_str(split.as_str());
        k.push('\n');
        k.push_str(p.to_str()?);
        k.push('\n');
        k.push_str(std::fs::canonicalize(p).ok()?.to_str()?);
    }
    Some(k)
}

fn kind_code(k: IssueKind) -> u8 {
    match k {
        IssueKind::CorruptImage => 0,
        IssueKind::MissingImage => 1,
        IssueKind::BrokenLabel => 2,
        IssueKind::LabelWarning => 3,
        IssueKind::Other => 4,
    }
}

fn code_kind(c: u8) -> Option<IssueKind> {
    Some(match c {
        0 => IssueKind::CorruptImage,
        1 => IssueKind::MissingImage,
        2 => IssueKind::BrokenLabel,
        3 => IssueKind::LabelWarning,
        4 => IssueKind::Other,
        _ => return None,
    })
}

fn split_code(s: Split) -> u8 {
    match s {
        Split::Train => 0,
        Split::Val => 1,
        Split::Test => 2,
    }
}

fn path_str(p: &Path) -> Option<&str> {
    p.to_str()
}

/// `None` when a path is not UTF-8 (such a dataset is not cached).
pub(crate) fn encode(a: &Annotations, stamps: &[Stamp]) -> Option<Vec<u8>> {
    let mut e = Enc::default();
    e.len(stamps.len());
    for s in stamps {
        e.stamp(*s);
    }
    e.len(a.issues.len());
    for i in &a.issues {
        e.u8(kind_code(i.kind));
        e.str(path_str(&i.path)?);
        e.str(&i.message);
    }
    e.len(a.classes.len());
    for c in &a.classes {
        e.str(c);
    }
    e.len(a.other_files.len());
    for f in &a.other_files {
        e.str(path_str(f)?);
    }
    e.len(a.splits.len());
    for s in &a.splits {
        e.u8(split_code(s.split));
        e.str(path_str(&s.path)?);
        e.len(s.files.len());
        for f in &s.files {
            e.str(f);
        }
        e.len(s.images.len());
        for im in &s.images {
            e.u64(im.row as u64);
            match im.declared {
                Some((w, h)) => {
                    e.u8(1);
                    e.u32(w);
                    e.u32(h);
                }
                None => e.u8(0),
            }
            e.len(im.boxes.len());
            for (class, b) in &im.boxes {
                e.u32(*class);
                for v in b {
                    e.f32(*v);
                }
            }
            e.opt_str(im.broken.as_deref());
        }
    }
    Some(e.0)
}

/// The cached [`Annotations`], when they were stored with exactly `stamps`.
pub(crate) fn decode(bytes: &[u8], stamps: &[Stamp]) -> Option<Annotations> {
    let mut d = Dec(bytes);
    let n = d.len(16)?;
    if n != stamps.len() {
        return None;
    }
    for s in stamps {
        if d.stamp()? != *s {
            return None;
        }
    }
    let mut issues = Vec::new();
    for _ in 0..d.len(17)? {
        let kind = code_kind(d.u8()?)?;
        let path = PathBuf::from(d.str()?);
        issues.push(Issue { kind, path, message: d.str()?.to_string() });
    }
    let mut classes = Vec::new();
    for _ in 0..d.len(8)? {
        classes.push(d.str()?.to_string());
    }
    let mut other_files = Vec::new();
    for _ in 0..d.len(8)? {
        other_files.push(PathBuf::from(d.str()?));
    }
    let mut splits = Vec::new();
    for _ in 0..d.len(25)? {
        let split = *Split::ALL.get(usize::from(d.u8()?))?;
        let path = PathBuf::from(d.str()?);
        let mut files = Vec::new();
        for _ in 0..d.len(8)? {
            files.push(d.str()?.to_string());
        }
        let mut images = Vec::new();
        for _ in 0..d.len(18)? {
            let row = usize::try_from(d.u64()?).ok()?;
            if row >= files.len() {
                return None;
            }
            let declared = match d.u8()? {
                0 => None,
                1 => Some((d.u32()?, d.u32()?)),
                _ => return None,
            };
            let mut boxes = Vec::new();
            for _ in 0..d.len(20)? {
                let class = d.u32()?;
                boxes.push((class, [d.f32()?, d.f32()?, d.f32()?, d.f32()?]));
            }
            let broken = d.opt_str()?.map(str::to_string);
            images.push(ImageAnnotations { row, declared, boxes, broken });
        }
        splits.push(SplitAnnotations { split, path, files, images });
    }
    d.is_empty().then_some(Annotations { issues, classes, other_files, splits })
}
