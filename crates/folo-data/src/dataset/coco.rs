//! COCO instances JSON.

use super::{
    Candidate, FileDataset, Format, IssueKind, LabelSource, LoadReport, LoadedDataset, OpenOptions, Split,
};
use crate::error::{DataError, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Guesses the split from a JSON file name or its folder (`instances_val2017.json`, `valid/_annotations.coco.json`).
pub(crate) fn guess_split(path: &Path) -> Split {
    let check = |s: &str| {
        let s = s.to_ascii_lowercase();
        if s.contains("train") {
            Some(Split::Train)
        } else if s.contains("val") {
            Some(Split::Val)
        } else if s.contains("test") {
            Some(Split::Test)
        } else {
            None
        }
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let parent = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    check(stem).or_else(|| check(parent)).unwrap_or(Split::Train)
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")))
        .collect();
    v.sort();
    v
}

/// COCO annotation files under `dir`: `annotations/instances_*.json` (or other non-caption JSON there), then JSON at
/// the root and in `train|val|valid|test/` (the Roboflow layout).
pub(crate) fn find_jsons(dir: &Path) -> Vec<(Split, PathBuf)> {
    let ann = json_files(&dir.join("annotations"));
    let instances: Vec<PathBuf> = ann
        .iter()
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("instances")))
        .cloned()
        .collect();
    let mut files = if instances.is_empty() {
        ann.into_iter()
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                !n.starts_with("captions") && !n.starts_with("person_keypoints")
            })
            .collect()
    } else {
        instances
    };
    files.extend(json_files(dir).into_iter().filter(|p| looks_like_coco(p)));
    for sub in ["train", "val", "valid", "test"] {
        files.extend(json_files(&dir.join(sub)).into_iter().filter(|p| looks_like_coco(p)));
    }
    files.into_iter().map(|p| (guess_split(&p), p)).collect()
}

/// Cheap check that a JSON at the dataset root is COCO and not some unrelated file. Files named like COCO
/// annotations are always taken (so a corrupt one is reported rather than silently ignored).
fn looks_like_coco(p: &Path) -> bool {
    use std::io::Read;
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    if name.contains("coco") || name.starts_with("instances") || name.contains("annotations") {
        return true;
    }
    let Ok(f) = std::fs::File::open(p) else { return false };
    let mut head = Vec::new();
    if f.take(1 << 20).read_to_end(&mut head).is_err() {
        return false;
    }
    let head = String::from_utf8_lossy(&head);
    head.contains("\"images\"") && (head.contains("\"annotations\"") || head.contains("\"categories\""))
}

fn id_key(v: &Value) -> Option<String> {
    match v {
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// `(id key, file_name, declared size)` of a COCO image.
type ImageRow = (String, String, Option<(u32, u32)>);
/// Pixel boxes of one image and the first reason its annotations are broken, if any.
type ImageAnns = (Vec<(u32, [f32; 4])>, Option<String>);

struct Parsed {
    split: Split,
    path: PathBuf,
    images: Vec<ImageRow>,
    /// `(image id key, category id, bbox)`
    anns: Vec<(String, u64, Value)>,
}

pub(crate) fn open_jsons(
    root: &Path,
    jsons: Vec<(Split, PathBuf)>,
    opts: &OpenOptions,
) -> Result<LoadedDataset> {
    let mut report = LoadReport::default();
    let mut categories: BTreeMap<u64, String> = BTreeMap::new();
    let mut parsed = Vec::new();
    for (split, path) in jsons {
        let bytes = std::fs::read(&path).map_err(|e| DataError::io(&path, e))?;
        let v: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                report.push(IssueKind::BrokenLabel, &path, format!("invalid JSON: {e}"));
                continue;
            }
        };
        let Some(images) = v.get("images").and_then(Value::as_array) else {
            report.push(IssueKind::BrokenLabel, &path, "no `images` array: not a COCO file");
            continue;
        };
        for c in v.get("categories").and_then(Value::as_array).into_iter().flatten() {
            let (Some(id), Some(name)) =
                (c.get("id").and_then(Value::as_u64), c.get("name").and_then(Value::as_str))
            else {
                report.push(IssueKind::BrokenLabel, &path, format!("bad category {c}"));
                continue;
            };
            if let Some(prev) = categories.insert(id, name.to_string())
                && prev != name
            {
                report.push(
                    IssueKind::Other,
                    &path,
                    format!("category {id} is named both {prev:?} and {name:?}"),
                );
            }
        }
        let mut imgs = Vec::with_capacity(images.len());
        for im in images {
            let (Some(id), Some(file)) =
                (im.get("id").and_then(id_key), im.get("file_name").and_then(Value::as_str))
            else {
                report.push(
                    IssueKind::BrokenLabel,
                    &path,
                    format!("image entry without id or file_name: {im}"),
                );
                continue;
            };
            let dim = |k| im.get(k).and_then(Value::as_u64).filter(|&d| d > 0 && d <= u32::MAX as u64);
            let declared = dim("width").zip(dim("height")).map(|(w, h)| (w as u32, h as u32));
            imgs.push((id, file.to_string(), declared));
        }
        let mut anns = Vec::new();
        for a in v.get("annotations").and_then(Value::as_array).into_iter().flatten() {
            if a.get("iscrowd").and_then(Value::as_u64) == Some(1) {
                continue;
            }
            let Some(img) = a.get("image_id").and_then(id_key) else {
                report.push(IssueKind::BrokenLabel, &path, format!("annotation without image_id: {a}"));
                continue;
            };
            let cat = a.get("category_id").and_then(Value::as_u64).unwrap_or(u64::MAX);
            anns.push((img, cat, a.get("bbox").cloned().unwrap_or(Value::Null)));
        }
        parsed.push(Parsed { split, path, images: imgs, anns });
    }
    if parsed.is_empty() {
        return Err(DataError::Format(format!("no readable COCO annotation file in {}", root.display())));
    }
    let remap: HashMap<u64, u32> = categories.keys().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    let classes: Vec<String> = categories.into_values().collect();

    let mut by_split: BTreeMap<Split, Vec<Candidate>> = BTreeMap::new();
    for p in parsed {
        let base = image_base(root, &p);
        let mut per_image: HashMap<&str, ImageAnns> = HashMap::new();
        let known: HashMap<&str, ()> = p.images.iter().map(|(id, _, _)| (id.as_str(), ())).collect();
        for (img, cat, bbox) in &p.anns {
            if !known.contains_key(img.as_str()) {
                report.push(IssueKind::Other, &p.path, format!("annotation for unknown image id {img}"));
                continue;
            }
            let entry = per_image.entry(img.as_str()).or_default();
            let Some(&class) = remap.get(cat) else {
                entry.1.get_or_insert_with(|| format!("unknown category id {cat}"));
                continue;
            };
            // Exactly four numbers; anything else (null, strings, extra values) makes the annotation broken.
            let nums: Option<Vec<f64>> =
                bbox.as_array().filter(|a| a.len() == 4).and_then(|a| a.iter().map(Value::as_f64).collect());
            match nums.as_deref() {
                Some(&[x, y, w, h])
                    if [x, y, x + w, y + h].iter().all(|v| v.is_finite() && v.abs() < 1e9)
                        && w >= 0.0
                        && h >= 0.0 =>
                {
                    entry.0.push((class, [x as f32, y as f32, (x + w) as f32, (y + h) as f32]));
                }
                _ => {
                    entry.1.get_or_insert_with(|| format!("bad bbox {bbox}"));
                }
            }
        }
        let list = by_split.entry(p.split).or_default();
        for (id, file, declared) in &p.images {
            let (boxes, broken) = per_image.remove(id.as_str()).unwrap_or_default();
            list.push(Candidate {
                image: base.join(file),
                labels: LabelSource::Coco { boxes, declared: *declared, broken, label: p.path.clone() },
            });
        }
    }
    let mut built: BTreeMap<Split, Vec<super::ImageEntry>> = BTreeMap::new();
    for (split, cands) in by_split {
        let entries = super::build_entries(cands, Some(classes.len()), opts, &mut report);
        built.entry(split).or_default().extend(entries);
    }
    let mut take = |s: Split| FileDataset::new(s, classes.clone(), built.remove(&s).unwrap_or_default());
    let (train, val, test) = (take(Split::Train), take(Split::Val), take(Split::Test));
    Ok(LoadedDataset { format: Format::Coco, root: root.to_path_buf(), classes, train, val, test, report })
}

/// Picks the folder the JSON's `file_name`s are relative to by probing a few images.
fn image_base(root: &Path, p: &Parsed) -> PathBuf {
    let json_dir = p.path.parent().unwrap_or(root).to_path_buf();
    let stem = p.path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let tag = stem.strip_prefix("instances_").unwrap_or(stem);
    let mut cands = vec![
        json_dir.clone(),
        root.join("images").join(tag),
        root.join(tag),
        root.join("images").join(p.split.as_str()),
        root.join(p.split.as_str()),
    ];
    if p.split == Split::Val {
        cands.push(root.join("images").join("valid"));
        cands.push(root.join("valid"));
    }
    cands.push(root.join("images"));
    cands.push(root.to_path_buf());
    // Probe the first few images and an even spread over the rest, so a run of missing files at the start
    // does not hide the right folder.
    let step = (p.images.len() / 32).max(1);
    let mut probe: Vec<&str> = p.images.iter().take(8).map(|(_, f, _)| f.as_str()).collect();
    probe.extend(p.images.iter().step_by(step).take(32).map(|(_, f, _)| f.as_str()));
    cands
        .iter()
        .map(|c| (c, probe.iter().filter(|f| c.join(f).is_file()).count()))
        .filter(|&(_, n)| n > 0)
        .max_by_key(|&(c, n)| (n, std::cmp::Reverse(cands.iter().position(|x| x == c))))
        .map(|(c, _)| c.clone())
        .unwrap_or(json_dir)
}
