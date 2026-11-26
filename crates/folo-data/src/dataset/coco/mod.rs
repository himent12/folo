//! COCO instances JSON.

mod fields;
mod parse;

use super::{
    Candidate, FileDataset, Format, IssueKind, LabelSource, LoadReport, LoadedDataset, OpenOptions, Split,
};
use crate::error::{DataError, Result};
use fields::{IdKey, ParsedBbox, whole_number_value};
use parse::{ParsedImage, Sections};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Ordinary words that contain `val` or `test` and name no split, so a dataset in a folder called `valves` or `latest`
/// is not a validation or test set.
const NOT_A_SPLIT: &[&str] = &[
    "approval",
    "arrival",
    "arrivals",
    "carnival",
    "carnivals",
    "contest",
    "contests",
    "fastest",
    "festival",
    "festivals",
    "greatest",
    "interval",
    "intervals",
    "latest",
    "medieval",
    "protest",
    "protests",
    "removal",
    "retrieval",
    "rival",
    "rivals",
    "survival",
    "valley",
    "valleys",
    "value",
    "values",
    "valve",
    "valves",
];

/// Guesses the split from a JSON file name, then its folder's (`instances_val2017.json`,
/// `valid/_annotations.coco.json`).
pub(crate) fn guess_split(path: &Path) -> Split {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let parent = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    split_in_name(stem).or_else(|| split_in_name(parent)).unwrap_or(Split::Train)
}

/// The first of `train`, `val`, `test` found anywhere in a name (`valsubset`, `instances_myval`), ignoring the runs of
/// letters that are exactly a [`NOT_A_SPLIT`] word.
fn split_in_name(name: &str) -> Option<Split> {
    let name = name.to_ascii_lowercase();
    let words = name.split(|c: char| !c.is_alphabetic()).filter(|w| !NOT_A_SPLIT.contains(w));
    let has = |part: &str| words.clone().any(|w| w.contains(part));
    [(Split::Train, "train"), (Split::Val, "val"), (Split::Test, "test")]
        .into_iter()
        .find_map(|(split, part)| has(part).then_some(split))
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = crate::guard::check(dir).and_then(|_| std::fs::read_dir(dir)) else { return Vec::new() };
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
    let Ok(f) = crate::guard::check(p).and_then(|_| std::fs::File::open(p)) else { return false };
    let mut head = Vec::new();
    if f.take(1 << 20).read_to_end(&mut head).is_err() {
        return false;
    }
    let head = String::from_utf8_lossy(&head);
    head.contains("\"images\"") && (head.contains("\"annotations\"") || head.contains("\"categories\""))
}

/// The category id of an annotation that has none, or one that is not a whole number.
const NO_CATEGORY: u64 = u64::MAX;

/// `(id key, file_name, declared size)` of a COCO image.
type ImageRow = (IdKey, String, Option<(u32, u32)>);
/// Pixel boxes of one image and the first reason its annotations are broken, if any.
type ImageAnns = (Vec<(u32, [f32; 4])>, Option<String>);

struct Parsed {
    split: Split,
    path: PathBuf,
    images: Vec<ImageRow>,
    /// `(image id key, category id, bbox)`
    anns: Vec<(IdKey, u64, ParsedBbox)>,
}

pub(crate) fn open_jsons(
    root: &Path,
    jsons: Vec<(Split, PathBuf)>,
    opts: &OpenOptions,
    wanted: &[Split],
) -> Result<LoadedDataset> {
    let settings = super::cache::Settings::current();
    let ann = super::cache::coco::load_or_read(&settings, &jsons, wanted, || {
        read_annotations(root, &jsons, wanted)
    })?;
    let Annotations { issues, skipped_image_rows, classes, other_files, splits } = ann;
    let mut report = LoadReport::from_issues(issues, skipped_image_rows);
    let mut deps: std::collections::BTreeSet<PathBuf> = std::iter::once(root.to_path_buf()).collect();
    deps.extend(other_files);
    let mut by_split: BTreeMap<Split, Vec<Candidate>> = BTreeMap::new();
    for p in splits {
        deps.insert(p.path.clone());
        let files: Vec<&str> = p.files.iter().map(String::as_str).collect();
        let base = image_base(root, &p.path, p.split, &files);
        let list = by_split.entry(p.split).or_default();
        list.reserve(p.images.len());
        for im in p.images {
            let image = base.join(&p.files[im.row]);
            let parent = crate::paths::parent_dir(&image);
            if !deps.contains(parent) {
                deps.insert(parent.to_path_buf());
            }
            list.push(Candidate {
                image,
                labels: LabelSource::Coco {
                    boxes: im.boxes,
                    declared: im.declared,
                    broken: im.broken,
                    label: p.path.clone(),
                },
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
    let sources = deps.into_iter().collect();
    Ok(LoadedDataset {
        format: Format::Coco,
        root: root.to_path_buf(),
        classes,
        train,
        val,
        test,
        report,
        sources,
    })
}

/// What [`open_jsons`] derives from the JSON files alone, before it looks for the images: the part the scan cache
/// keeps (`cache::coco`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Annotations {
    /// The issues found in the files, in order.
    pub issues: Vec<super::Issue>,
    /// Malformed entries in COCO `images` arrays dropped before candidate building.
    pub skipped_image_rows: usize,
    /// Class names, by category id.
    pub classes: Vec<String>,
    /// Files of the splits not wanted (read for their categories only).
    pub other_files: Vec<PathBuf>,
    /// The wanted splits' files, in order.
    pub splits: Vec<SplitAnnotations>,
}

/// One wanted COCO file.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SplitAnnotations {
    pub split: Split,
    pub path: PathBuf,
    /// Every image entry's `file_name`, in file order (duplicates included: they count when the image folder
    /// is chosen).
    pub files: Vec<String>,
    /// The images kept (the first entry of every id), in file order.
    pub images: Vec<ImageAnnotations>,
}

/// One image of a COCO file and its boxes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageAnnotations {
    /// Index of its `file_name` in [`SplitAnnotations::files`].
    pub row: usize,
    /// The size the JSON declares.
    pub declared: Option<(u32, u32)>,
    /// `(class, pixel xyxy)`.
    pub boxes: Vec<(u32, [f32; 4])>,
    /// The first reason its annotations are broken.
    pub broken: Option<String>,
}

/// Reads and parses the COCO files (the uncached half of [`open_jsons`]).
pub(crate) fn read_annotations(
    root: &Path,
    jsons: &[(Split, PathBuf)],
    wanted: &[Split],
) -> Result<Annotations> {
    let mut report = LoadReport::default();
    let mut skipped_image_rows = 0usize;
    let mut categories: BTreeMap<u64, String> = BTreeMap::new();
    let mut parsed = Vec::with_capacity(jsons.len());
    // Files of the splits not wanted, read for their categories only (every file's count for the class ids).
    let mut other_files = Vec::new();
    for (split, path) in jsons {
        let (split, path) = (*split, path.clone());
        let bytes = crate::guard::check(&path)
            .and_then(|_| std::fs::read(&path))
            .map_err(|e| DataError::io(&path, e))?;
        let sections = if wanted.contains(&split) {
            Sections::parse(&bytes)
        } else {
            Sections::parse_categories(&bytes)
        };
        let sections = match sections {
            Ok(s) => s,
            Err(e) => {
                report.push(IssueKind::BrokenLabel, &path, format!("invalid JSON: {e}"));
                continue;
            }
        };
        drop(bytes);
        let Some(raw_images) = sections.images else {
            report.push(IssueKind::BrokenLabel, &path, "no `images` array: not a COCO file");
            continue;
        };
        for c in sections.categories.as_ref().and_then(Value::as_array).into_iter().flatten() {
            let (Some(id), Some(name)) =
                (c.get("id").and_then(whole_number_value), c.get("name").and_then(Value::as_str))
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
        if !wanted.contains(&split) {
            other_files.push(path);
            continue;
        }
        let mut imgs = Vec::with_capacity(raw_images.len());
        for im in raw_images {
            match im {
                ParsedImage::Valid { id, file_name, declared } => imgs.push((id, file_name, declared)),
                ParsedImage::Bad(msg) => {
                    skipped_image_rows += 1;
                    report.push(
                        IssueKind::BrokenLabel,
                        &path,
                        format!("image entry without id or file_name: {msg}"),
                    );
                }
            }
        }
        let mut anns = Vec::with_capacity(sections.annotations.len());
        for a in sections.annotations {
            if a.iscrowd {
                continue;
            }
            let Some(img) = a.image_id else {
                report.push(
                    IssueKind::BrokenLabel,
                    &path,
                    format!("annotation without image_id: {}", a.text),
                );
                continue;
            };
            let cat = a.category_id.unwrap_or(NO_CATEGORY);
            anns.push((img, cat, a.bbox));
        }
        parsed.push(Parsed { split, path, images: imgs, anns });
    }
    if parsed.is_empty() && other_files.is_empty() {
        return Err(DataError::Format(format!(
            "no readable COCO annotation file in {}",
            crate::paths::shown(root)
        )));
    }
    let remap: HashMap<u64, u32> = categories.keys().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    let classes: Vec<String> = categories.into_values().collect();

    let mut splits = Vec::with_capacity(parsed.len());
    for p in parsed {
        // Single pre-sized lookup table from image id to its first row index, avoiding three separate maps and
        // hashing every annotation twice.
        let mut first_row: HashMap<&IdKey, usize> = HashMap::with_capacity(p.images.len());
        let mut dup_of: Vec<Option<usize>> = Vec::with_capacity(p.images.len());
        for (row, (id, _, _)) in p.images.iter().enumerate() {
            match first_row.entry(id) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(row);
                    dup_of.push(None);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    dup_of.push(Some(*e.get()));
                }
            }
        }
        let mut per_image: Vec<ImageAnns> = (0..p.images.len()).map(|_| (Vec::new(), None)).collect();
        for (img, cat, bbox) in &p.anns {
            let Some(&slot) = first_row.get(img) else {
                report.push(IssueKind::Other, &p.path, format!("annotation for unknown image id {img}"));
                continue;
            };
            let entry = &mut per_image[slot];
            let Some(&class) = remap.get(cat) else {
                entry.1.get_or_insert_with(|| match *cat {
                    NO_CATEGORY => "category_id is missing or not a whole number".to_string(),
                    cat => format!("unknown category id {cat}"),
                });
                continue;
            };
            match bbox {
                ParsedBbox::Valid([x, y, w, h])
                    if [*x, *y, x + w, y + h].iter().all(|v| v.is_finite() && v.abs() < 1e9)
                        && *w >= 0.0
                        && *h >= 0.0 =>
                {
                    entry.0.push((class, [*x as f32, *y as f32, (x + w) as f32, (y + h) as f32]));
                }
                ParsedBbox::Valid([x, y, w, h]) => {
                    entry.1.get_or_insert_with(|| {
                        if *w < 0.0 || *h < 0.0 {
                            format!("bad bbox [{x},{y},{w},{h}]: width and height must be non-negative")
                        } else {
                            format!("bad bbox [{x},{y},{w},{h}]: coordinates out of range")
                        }
                    });
                }
                ParsedBbox::Missing => {
                    entry.1.get_or_insert_with(|| "bad bbox null".to_string());
                }
                ParsedBbox::Invalid(v) => {
                    entry.1.get_or_insert_with(|| format!("bad bbox {v}"));
                }
            }
        }
        // Second entries reusing an image id are left out (their annotations cannot be told apart from the first's).
        let mut images = Vec::with_capacity(first_row.len());
        for (row, (id, file, declared)) in p.images.iter().enumerate() {
            if let Some(kept_row) = dup_of[row] {
                let kept = &p.images[kept_row].1;
                report.push(
                    IssueKind::Other,
                    &p.path,
                    format!("image id {id} is used by both {kept} and {file}; {file} is left out"),
                );
                continue;
            }
            let (boxes, broken) = std::mem::take(&mut per_image[row]);
            images.push(ImageAnnotations { row, declared: *declared, boxes, broken });
        }
        drop(first_row);
        let files = p.images.into_iter().map(|(_, file, _)| file).collect();
        splits.push(SplitAnnotations { split: p.split, path: p.path, files, images });
    }
    Ok(Annotations { issues: report.issues, skipped_image_rows, classes, other_files, splits })
}

/// Picks the folder the JSON's `file_name`s are relative to by probing a few images.
fn image_base(root: &Path, json: &Path, split: Split, files: &[&str]) -> PathBuf {
    let json_dir = match json.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => root,
    }
    .to_path_buf();
    let stem = json.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let tag = stem.strip_prefix("instances_").unwrap_or(stem);
    let mut cands = vec![
        json_dir.clone(),
        root.join("images").join(tag),
        root.join(tag),
        root.join("images").join(split.as_str()),
        root.join(split.as_str()),
    ];
    if split == Split::Val {
        cands.push(root.join("images").join("valid"));
        cands.push(root.join("valid"));
    }
    cands.push(root.join("images"));
    cands.push(root.to_path_buf());
    // Probe the first few images and an even spread over the rest, so a run of missing files at the start
    // does not hide the right folder.
    let n = files.len();
    let mut idx: Vec<usize> = (0..n.min(8)).collect();
    if n > 1 {
        // 32 points from the first to the last entry inclusive.
        idx.extend((0..32).map(|k| k * (n - 1) / 31));
    }
    idx.sort_unstable();
    idx.dedup();
    let probe: Vec<&str> = idx.iter().map(|&i| files[i]).collect();
    cands
        .iter()
        .map(|c| (c, probe.iter().filter(|f| c.join(f).is_file()).count()))
        .filter(|&(_, n)| n > 0)
        .max_by_key(|&(c, n)| (n, std::cmp::Reverse(cands.iter().position(|x| x == c))))
        .map(|(c, _)| c.clone())
        .unwrap_or(json_dir)
}

/// One COCO annotation as pycocotools' `COCOeval` uses it.
#[derive(Clone, Debug, PartialEq)]
pub struct CocoAnnotation {
    /// `category_id` as in the file.
    pub category_id: u64,
    /// Pixel `[x, y, w, h]`.
    pub bbox: [f64; 4],
    /// `area` (COCO instances: the segmentation area); the box area when the file has none.
    pub area: f64,
    /// `iscrowd: 1`.
    pub iscrowd: bool,
}

/// One COCO image with its annotations.
#[derive(Clone, Debug, PartialEq)]
pub struct CocoImage {
    /// `id`.
    pub id: u64,
    /// `file_name` as in the file.
    pub file_name: String,
    /// The image file as the dataset loader resolves it (set by [`open_coco_ground_truth`]; `None` from
    /// [`read_coco_ground_truth`]), so a [`FileDataset`] entry can be matched by its exact path.
    pub path: Option<PathBuf>,
    /// Every annotation of the image, crowd regions included, in file order.
    pub annotations: Vec<CocoAnnotation>,
}

/// The ground truth of one COCO instances file, for exact COCO evaluation (crowd regions and annotation areas,
/// which [`open`](crate::open) drops) and for writing COCO results files.
#[derive(Clone, Debug, PartialEq)]
pub struct CocoGroundTruth {
    /// The file.
    pub path: PathBuf,
    /// Images in file order.
    pub images: Vec<CocoImage>,
    /// `(id, name)` sorted by id: class `i` of a dataset [`open`](crate::open)ed from this file is entry `i`.
    pub categories: Vec<(u64, String)>,
}

mod gt {
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct File {
        pub images: Vec<Image>,
        #[serde(default)]
        pub annotations: Vec<Ann>,
        #[serde(default)]
        pub categories: Vec<Cat>,
    }
    #[derive(Deserialize)]
    pub struct Image {
        pub id: u64,
        pub file_name: String,
    }
    #[derive(Deserialize)]
    pub struct Ann {
        pub image_id: u64,
        pub category_id: u64,
        pub bbox: [f64; 4],
        pub area: Option<f64>,
        #[serde(default)]
        pub iscrowd: u8,
        // Unknown fields, `segmentation` (the bulk of an instances file) included, are skipped without being built.
    }
    #[derive(Deserialize)]
    pub struct Cat {
        pub id: u64,
        pub name: String,
    }
}

/// Reads a COCO instances file strictly (numeric image and category ids, four-number boxes), keeping crowd regions
/// and areas. Unlike [`open`](crate::open) it fails on the first malformed entry.
pub fn read_coco_ground_truth(path: impl AsRef<Path>) -> Result<CocoGroundTruth> {
    let path = path.as_ref();
    let file = crate::guard::check(path)
        .and_then(|_| std::fs::File::open(path))
        .map_err(|e| DataError::io(path, e))?;
    let f: gt::File = serde_json::from_reader(std::io::BufReader::with_capacity(1 << 20, file))
        .map_err(|e| DataError::Format(format!("{}: not a COCO instances file: {e}", path.display())))?;
    let mut index: HashMap<u64, usize> = HashMap::with_capacity(f.images.len());
    let mut images: Vec<CocoImage> = Vec::with_capacity(f.images.len());
    for im in f.images {
        if index.insert(im.id, images.len()).is_some() {
            return Err(DataError::Format(format!("{}: duplicate image id {}", path.display(), im.id)));
        }
        images.push(CocoImage { id: im.id, file_name: im.file_name, path: None, annotations: Vec::new() });
    }
    for a in f.annotations {
        let Some(&i) = index.get(&a.image_id) else {
            return Err(DataError::Format(format!(
                "{}: annotation for unknown image id {}",
                path.display(),
                a.image_id
            )));
        };
        images[i].annotations.push(CocoAnnotation {
            category_id: a.category_id,
            bbox: a.bbox,
            area: a.area.unwrap_or(a.bbox[2] * a.bbox[3]),
            iscrowd: a.iscrowd == 1,
        });
    }
    let mut categories: Vec<(u64, String)> = f.categories.into_iter().map(|c| (c.id, c.name)).collect();
    categories.sort_by_key(|c| c.0);
    categories.dedup_by_key(|c| c.0);
    Ok(CocoGroundTruth { path: path.to_path_buf(), images, categories })
}

/// The ground truth of the one annotation file of `split` in the COCO dataset at `path` (a COCO `.json` or folder),
/// with every image's [`CocoImage::path`] resolved exactly as [`open`](crate::open) resolves it. Fails when the
/// split has no annotation file or several.
pub fn open_coco_ground_truth(path: impl AsRef<Path>, split: Split) -> Result<CocoGroundTruth> {
    let path = path.as_ref();
    let super::Source::Coco { root, jsons } = detect_coco(path)? else {
        return Err(DataError::Format(format!("{} is not a COCO dataset", crate::paths::shown(path))));
    };
    let mut of_split = jsons.into_iter().filter(|(s, _)| *s == split).map(|(_, p)| p);
    let (Some(json), None) = (of_split.next(), of_split.next()) else {
        return Err(DataError::Format(format!(
            "{}: expected exactly one COCO annotation file for the {} split",
            crate::paths::shown(path),
            split.as_str()
        )));
    };
    let mut gt = read_coco_ground_truth(&json)?;
    let files: Vec<&str> = gt.images.iter().map(|im| im.file_name.as_str()).collect();
    let base = image_base(&root, &json, split, &files);
    for im in &mut gt.images {
        im.path = Some(base.join(&im.file_name));
    }
    Ok(gt)
}

/// The COCO annotation files of the dataset at `path` (a COCO `.json` or a COCO dataset folder, as
/// [`open`](crate::open) finds them) with the split each one feeds.
pub fn coco_annotation_files(path: impl AsRef<Path>) -> Result<Vec<(Split, PathBuf)>> {
    match detect_coco(path.as_ref())? {
        super::Source::Coco { jsons, .. } => Ok(jsons),
        _ => Err(DataError::Format(format!("{} is not a COCO dataset", crate::paths::shown(path.as_ref())))),
    }
}

/// [`super::detect`] restricted to COCO. When there is no COCO dataset at `path` but a YOLO one, the error says
/// so (rather than that no dataset was found), so a caller can point at `folo dataset convert`.
fn detect_coco(path: &Path) -> Result<super::Source> {
    super::detect(path, Format::Coco).map_err(|e| match (&e, super::detect(path, Format::Yolo)) {
        (DataError::Format(_), Ok(_)) => {
            DataError::Format(format!("{} is a YOLO dataset, not a COCO one", crate::paths::shown(path)))
        }
        _ => e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type Fields = (Option<IdKey>, Option<u64>, ParsedBbox, bool);

    fn fields(s: &Sections) -> Vec<Fields> {
        s.annotations
            .iter()
            .map(|a| {
                let bbox = match &a.bbox {
                    ParsedBbox::Missing => ParsedBbox::Missing,
                    ParsedBbox::Valid(n) => ParsedBbox::Valid(*n),
                    ParsedBbox::Invalid(v) => ParsedBbox::Invalid(v.clone()),
                };
                (a.image_id.clone(), a.category_id, bbox, a.iscrowd)
            })
            .collect()
    }

    type ImageSummary<'a> = (&'a IdKey, &'a str, Option<(u32, u32)>);

    fn image_summary(s: &Sections) -> Option<Vec<Option<ImageSummary<'_>>>> {
        s.images.as_ref().map(|imgs| {
            imgs.iter()
                .map(|im| match im {
                    ParsedImage::Valid { id, file_name, declared } => {
                        Some((id, file_name.as_str(), *declared))
                    }
                    ParsedImage::Bad(_) => None,
                })
                .collect()
        })
    }

    /// The single-pass streaming parse agrees with reading the whole file as a `Value`, including on malformed input.
    #[test]
    fn streaming_parse_matches_value_parse() {
        let cases = [
            r#"{"images":[{"id":1,"file_name":"a.jpg","width":640,"height":480,"coco_url":"http://x"}],"annotations":[{"id":1,"image_id":1,"category_id":2,"bbox":[0,0,10,10],"segmentation":[[0,0,1,1]],"iscrowd":0}],"categories":[{"id":2,"name":"x"}]}"#,
            // Positional arrays and scalars are not annotations or images.
            r#"{"images":[[1,"a.jpg"], 9, null, {"id":"1","file_name":"b.jpg"}],"annotations":[[1,1,[0,0,10,10],0], 7, "s", null, {"image_id":3,"bbox":[1,2,3]}, {"image_id":4,"bbox":[1,2,3,4,5]}, {"image_id":5,"bbox":[1,"x",3,4]}]}"#,
            // Duplicate fields: the last one wins (including resetting to null/invalid).
            r#"{"images":[{"id":1,"id":2.0,"file_name":"a.jpg","width":10,"height":20}],"annotations":[{"image_id":1,"image_id":2,"bbox":null,"iscrowd":1,"iscrowd":0}]}"#,
            // `annotations` not an array; missing sections.
            r#"{"images":[],"annotations":{"image_id":1}}"#,
            r#"{"categories":[]}"#,
            // Duplicate top-level keys.
            r#"{"images":[1],"images":[],"annotations":[{"image_id":5}],"annotations":[{"image_id":6}]}"#,
            // The top level is not an object (an array would fill a derived struct positionally).
            r#"[[{"id":1,"file_name":"a.jpg"}], [], [{"image_id":1}]]"#,
            r#"  [1, 2, 3]"#,
            r#"{"images":[], "annotations":[ {"image_id": "s1", "iscrowd": 1}, {"image_id": "01", "iscrowd": 1.0}, {"image_id": "-0", "iscrowd": true}, {"category_id": 4.5, "bbox": {"x": 1}} ]}"#,
            // Escaped keys and huge u64 ids (> i64::MAX).
            r#"{"images":[{"\u0069d":18446744073709551615,"file_name":"big.jpg"}],"annotations":[{"\u0069mage_id":18446744073709551615,"category_id":0,"bbox":[1,2,3,4]}]}"#,
        ];
        for c in cases {
            let a = Sections::parse(c.as_bytes()).unwrap();
            let b = Sections::parse_value(serde_json::from_str(c).unwrap()).unwrap();
            assert_eq!(image_summary(&a), image_summary(&b), "{c}");
            assert_eq!(a.categories, b.categories, "{c}");
            assert_eq!(fields(&a), fields(&b), "{c}");
            let ca = Sections::parse_categories(c.as_bytes()).unwrap();
            assert_eq!(ca.images.is_some(), b.images.is_some(), "categories images {c}");
            assert_eq!(ca.categories, b.categories, "categories {c}");
            assert!(ca.annotations.is_empty());
        }
        assert!(Sections::parse(br#"{"images": ["#).is_err());
        assert!(Sections::parse(br#"{"images": []} trailing"#).is_err());
    }

    #[test]
    fn id_key_matches_value_equivalence_classes() {
        assert_eq!(IdKey::from_str_id("1"), IdKey::Int(1));
        assert_eq!(IdKey::from_f64(1.0), IdKey::Int(1));
        assert_eq!(IdKey::from_f64(-0.0), IdKey::Int(0));
        assert_eq!(IdKey::from_str_id("0"), IdKey::Int(0));
        assert_ne!(IdKey::from_str_id("01"), IdKey::Int(1));
        assert_ne!(IdKey::from_str_id("-0"), IdKey::Int(0));
        assert_eq!(IdKey::from_u64(u64::MAX).to_string(), u64::MAX.to_string());
        assert_eq!(IdKey::from_str_id(&u64::MAX.to_string()), IdKey::from_u64(u64::MAX));
    }

    #[test]
    fn overflowing_number_in_one_row_retains_valid_rows_and_rejects_invalid_utf8_in_skipped_fields() {
        // An overflowing float in one annotation's bbox isolates only that row instead of discarding the split.
        let json = br#"{"images":[{"id":1,"file_name":"a.jpg","width":100,"height":100},{"id":2,"file_name":"b.jpg","width":100,"height":100}],"annotations":[{"id":1,"image_id":1,"category_id":0,"bbox":[0,0,1e400,2]},{"id":2,"image_id":2,"category_id":0,"bbox":[1,2,10,10]}],"categories":[{"id":0,"name":"c"}]}"#;
        let parsed = Sections::parse(json).expect("split should survive one overflowing annotation");
        let imgs = parsed.images.expect("images array retained");
        assert_eq!(imgs.len(), 2);
        assert_eq!(parsed.annotations.len(), 2);
        assert!(parsed.annotations[0].image_id.is_none());
        assert!(!parsed.annotations[0].text.is_empty());
        assert_eq!(parsed.annotations[1].image_id, Some(IdKey::Int(2)));
        assert_eq!(parsed.annotations[1].bbox, ParsedBbox::Valid([1.0, 2.0, 10.0, 10.0]));

        // Invalid UTF-8 inside a skipped field (such as `segmentation`) must still fail parsing.
        let mut bad_utf8 = br#"{"images":[{"id":1,"file_name":"a.jpg"}],"annotations":[{"id":1,"image_id":1,"category_id":0,"bbox":[1,2,3,4],"segmentation":""#.to_vec();
        bad_utf8.push(0xFF);
        bad_utf8.extend_from_slice(br#""}]}"#);
        assert!(Sections::parse(&bad_utf8).is_err());
        assert!(Sections::parse_categories(&bad_utf8).is_err());
    }
}
