//! YOLO format: `data.yaml`, `images/` + `labels/`.

use super::{
    Candidate, FileDataset, Format, IssueKind, LabelSource, LoadReport, LoadedDataset, OpenOptions, Split,
    build_entries, default_names,
};
use crate::error::{DataError, Result};
use crate::image::is_image_path;
use serde_yaml::Value;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

/// The YOLO label path for an image: the last `images` directory component becomes `labels` and the extension
/// becomes `.txt` (`ds/images/train/a.jpg` → `ds/labels/train/a.txt`). Without an `images` component the label sits
/// next to the image.
pub fn label_path_for(image: &Path) -> PathBuf {
    let comps: Vec<Component> = image.components().collect();
    let pos = comps.iter().rposition(|c| c.as_os_str() == OsStr::new("images"));
    let mut out = PathBuf::new();
    for (i, c) in comps.iter().enumerate() {
        if Some(i) == pos && i + 1 < comps.len() {
            out.push("labels");
        } else {
            out.push(c.as_os_str());
        }
    }
    out.set_extension("txt");
    out
}

/// A `data.yaml` (or `dataset.yaml`, or any YAML with a `train` key) directly inside `dir`.
pub(crate) fn find_yaml(dir: &Path) -> Option<PathBuf> {
    for name in ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut yamls: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "yaml" || e == "yml"))
        .collect();
    yamls.sort();
    yamls.into_iter().find(|p| {
        std::fs::read_to_string(p)
            .ok()
            .and_then(|t| serde_yaml::from_str::<Value>(&t).ok())
            .is_some_and(|v| v.get("train").is_some())
    })
}

/// Whether `dir` has YOLO `images/` and `labels/` folders (at the root or under `train|valid|val|test/`).
pub(crate) fn looks_like_yolo_dir(dir: &Path) -> bool {
    (dir.join("images").is_dir() && dir.join("labels").is_dir())
        || ["train", "valid", "val"].iter().any(|s| dir.join(s).join("images").is_dir())
}

pub(crate) fn open_yaml(yaml: &Path, opts: &OpenOptions) -> Result<LoadedDataset> {
    let text = std::fs::read_to_string(yaml).map_err(|e| DataError::io(yaml, e))?;
    let v: Value = serde_yaml::from_str(&text)
        .map_err(|e| DataError::Format(format!("{}: invalid YAML: {e}", yaml.display())))?;
    if !v.is_mapping() {
        return Err(DataError::Format(format!("{}: expected a mapping at the top level", yaml.display())));
    }
    let yaml_dir = yaml.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut report = LoadReport::default();
    let root = match v.get("path").and_then(scalar_string) {
        Some(p) if !p.is_empty() => {
            let p = PathBuf::from(p);
            let candidate = if p.is_absolute() { p } else { yaml_dir.join(p) };
            if candidate.is_dir() {
                candidate
            } else {
                report.push(
                    IssueKind::Other,
                    yaml,
                    format!("`path` {} does not exist; using the YAML's directory", candidate.display()),
                );
                yaml_dir.clone()
            }
        }
        _ => yaml_dir.clone(),
    };
    let names = parse_names(&v, yaml)?;
    let mut splits: BTreeMap<Split, Vec<PathBuf>> = BTreeMap::new();
    for (split, keys) in
        [(Split::Train, &["train"][..]), (Split::Val, &["val", "valid"][..]), (Split::Test, &["test"][..])]
    {
        let Some(val) = keys.iter().find_map(|k| v.get(*k)) else { continue };
        let sources: Vec<String> = match val {
            Value::Null => Vec::new(),
            Value::Sequence(seq) => seq.iter().filter_map(scalar_string).collect(),
            other => scalar_string(other).into_iter().collect(),
        };
        let mut images = Vec::new();
        for src in sources {
            match resolve(&root, &yaml_dir, &src) {
                Some(p) => collect_images(&p, &root, &mut images, &mut report),
                None => report.push(IssueKind::Other, yaml, format!("{split} path {src:?} does not exist")),
            }
        }
        splits.insert(split, images);
    }
    if splits.get(&Split::Train).is_none_or(|v| v.is_empty())
        && splits.get(&Split::Val).is_none_or(|v| v.is_empty())
    {
        return Err(DataError::Format(format!("{}: no train or val images found", yaml.display())));
    }
    finish(root, names, splits, opts, report)
}

pub(crate) fn open_dir(root: &Path, opts: &OpenOptions) -> Result<LoadedDataset> {
    let mut report = LoadReport::default();
    let mut splits: BTreeMap<Split, Vec<PathBuf>> = BTreeMap::new();
    let images = root.join("images");
    let split_dirs =
        [(Split::Train, "train"), (Split::Val, "val"), (Split::Val, "valid"), (Split::Test, "test")];
    let mut found = false;
    for (split, name) in split_dirs {
        for dir in [images.join(name), root.join(name).join("images")] {
            if dir.is_dir() {
                collect_images(&dir, root, splits.entry(split).or_default(), &mut report);
                found = true;
            }
        }
    }
    if !found && images.is_dir() {
        collect_images(&images, root, splits.entry(Split::Train).or_default(), &mut report);
    }
    let names = ["classes.txt", "labels/classes.txt"]
        .iter()
        .map(|n| root.join(n))
        .find(|p| p.is_file())
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect::<Vec<_>>());
    finish(root.to_path_buf(), names, splits, opts, report)
}

fn finish(
    root: PathBuf,
    names: Option<Vec<String>>,
    splits: BTreeMap<Split, Vec<PathBuf>>,
    opts: &OpenOptions,
    mut report: LoadReport,
) -> Result<LoadedDataset> {
    let nc = names.as_ref().map(Vec::len);
    let mut built: BTreeMap<Split, Vec<super::ImageEntry>> = BTreeMap::new();
    for (split, images) in splits {
        let candidates = images
            .into_iter()
            .map(|image| {
                let label = label_path_for(&image);
                Candidate { image, labels: LabelSource::Yolo(label) }
            })
            .collect();
        built.insert(split, build_entries(candidates, nc, opts, &mut report));
    }
    let classes = names.unwrap_or_else(|| {
        let max = built.values().flatten().flat_map(|e| e.boxes.iter().map(|b| b.class as usize + 1)).max();
        default_names(max.unwrap_or(0))
    });
    let mut take = |s: Split| FileDataset::new(s, classes.clone(), built.remove(&s).unwrap_or_default());
    let (train, val, test) = (take(Split::Train), take(Split::Val), take(Split::Test));
    Ok(LoadedDataset { format: Format::Yolo, root, classes, train, val, test, report })
}

fn scalar_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// `names` as a list or an `{id: name}` map, falling back to `nc`.
fn parse_names(v: &Value, yaml: &Path) -> Result<Option<Vec<String>>> {
    let nc = v.get("nc").and_then(Value::as_u64).map(|n| n as usize);
    let names = match v.get("names") {
        Some(Value::Sequence(seq)) => {
            Some(seq.iter().map(|x| scalar_string(x).unwrap_or_default()).collect::<Vec<_>>())
        }
        Some(Value::Mapping(map)) => {
            let mut by_id = BTreeMap::new();
            for (k, name) in map {
                let id = k.as_u64().or_else(|| k.as_str().and_then(|s| s.trim().parse().ok())).ok_or_else(
                    || DataError::Format(format!("{}: names key {k:?} is not a class id", yaml.display())),
                )?;
                if id > 100_000 {
                    return Err(DataError::Format(format!(
                        "{}: class id {id} is implausibly large",
                        yaml.display()
                    )));
                }
                by_id.insert(id as usize, scalar_string(name).unwrap_or_default());
            }
            let n = by_id.keys().next_back().map_or(0, |m| m + 1);
            Some((0..n).map(|i| by_id.remove(&i).unwrap_or_else(|| format!("class{i}"))).collect())
        }
        Some(Value::Null) | None => None,
        Some(other) => {
            return Err(DataError::Format(format!(
                "{}: `names` must be a list or a map, got {other:?}",
                yaml.display()
            )));
        }
    };
    match (names, nc) {
        (Some(n), Some(c)) if n.len() != c => Err(DataError::Format(format!(
            "{}: `nc` is {c} but `names` has {} entries",
            yaml.display(),
            n.len()
        ))),
        (Some(n), _) => Ok(Some(n)),
        (None, Some(c)) => Ok(Some(default_names(c))),
        (None, None) => Ok(None),
    }
}

/// Resolves a split source relative to the root; tolerates the `../train/images` paths Roboflow exports.
fn resolve(root: &Path, yaml_dir: &Path, src: &str) -> Option<PathBuf> {
    let p = Path::new(src);
    if p.is_absolute() {
        return p.exists().then(|| p.to_path_buf());
    }
    let stripped: PathBuf =
        p.components().skip_while(|c| matches!(c, Component::ParentDir | Component::CurDir)).collect();
    [root.join(p), yaml_dir.join(p), root.join(&stripped), yaml_dir.join(&stripped)]
        .into_iter()
        .find(|c| c.exists())
}

/// Adds images from a directory (recursively), a `.txt` list, or a single image file.
fn collect_images(src: &Path, root: &Path, out: &mut Vec<PathBuf>, report: &mut LoadReport) {
    if src.is_dir() {
        let mut found: Vec<PathBuf> = walkdir::WalkDir::new(src)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && is_image_path(e.path()))
            .map(|e| e.into_path())
            .collect();
        found.sort();
        out.extend(found);
    } else if src.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt")) {
        let text = match std::fs::read_to_string(src) {
            Ok(t) => t,
            Err(e) => {
                report.push(IssueKind::Other, src, format!("cannot read image list: {e}"));
                return;
            }
        };
        let parent = src.parent().unwrap_or(Path::new("."));
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let p = Path::new(line);
            let path = if p.is_absolute() {
                p.to_path_buf()
            } else if let Some(rest) = line.strip_prefix("./") {
                parent.join(rest)
            } else if root.join(p).exists() {
                root.join(p)
            } else {
                parent.join(p)
            };
            out.push(path);
        }
    } else if is_image_path(src) {
        out.push(src.to_path_buf());
    } else {
        report.push(IssueKind::Other, src, "not a directory, .txt image list or image");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_paths() {
        assert_eq!(
            label_path_for(Path::new("ds/images/train/a.jpg")),
            PathBuf::from("ds/labels/train/a.txt")
        );
        assert_eq!(
            label_path_for(Path::new("x/images/ds/images/v/b.PNG")),
            PathBuf::from("x/images/ds/labels/v/b.txt")
        );
        assert_eq!(label_path_for(Path::new("flat/c.jpeg")), PathBuf::from("flat/c.txt"));
    }
}
