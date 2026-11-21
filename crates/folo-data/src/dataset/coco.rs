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

/// An image id as a map key: a string as it is, a number as JSON writes it, except that an integral float is its
/// integer (`"image_id": 1.0` names `"id": 1`).
fn id_key(v: &Value) -> Option<String> {
    match v {
        Value::Number(n) => Some(match n.as_f64() {
            Some(f) if n.is_f64() && f.fract() == 0.0 && f.abs() < (1u64 << 53) as f64 => {
                (f as i64).to_string()
            }
            _ => n.to_string(),
        }),
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

/// The fields of one `annotations` entry the loader reads. Everything else, `segmentation` included (the bulk of an
/// instances file), is skipped without being built.
#[derive(Debug, Default, PartialEq, serde::Deserialize)]
#[serde(default)]
struct AnnFields {
    image_id: Option<Value>,
    category_id: Option<Value>,
    bbox: Option<Value>,
    iscrowd: Option<Value>,
    /// The entry's JSON text, for messages (only kept for entries without an image id).
    #[serde(skip)]
    text: String,
}

impl AnnFields {
    fn from_value(a: &Value) -> Self {
        let field = |k: &str| a.get(k).filter(|v| !v.is_null()).cloned();
        AnnFields {
            image_id: field("image_id"),
            category_id: field("category_id"),
            bbox: field("bbox"),
            iscrowd: field("iscrowd"),
            text: String::new(),
        }
    }
}

/// The top level of a COCO file, annotations borrowed as raw JSON so they are never built as a `Value` tree
/// (that tree is about 15x the file: 7 GB for COCO's 448 MB `instances_train2017.json`).
#[derive(serde::Deserialize)]
struct Top<'a> {
    images: Option<Value>,
    categories: Option<Value>,
    #[serde(borrow)]
    annotations: Option<&'a serde_json::value::RawValue>,
}

/// Whether JSON text is an object. A derived struct deserializer also accepts a positional array
/// (`[1, 1, [0, 0, 10, 10]]` would fill `image_id`, `category_id`, `bbox`), which the `Value` reading never does.
fn is_object(json: &[u8]) -> bool {
    json.trim_ascii_start().first() == Some(&b'{')
}

/// What the loader reads from one COCO file.
#[derive(Debug, PartialEq)]
struct Sections {
    images: Option<Value>,
    categories: Option<Value>,
    annotations: Vec<AnnFields>,
}

impl Sections {
    /// Same result as reading the file as one `Value` (a non-array `annotations` means none, a non-object entry
    /// has no fields), with a fraction of the memory.
    fn parse(bytes: &[u8]) -> serde_json::Result<Self> {
        if !is_object(bytes) {
            return Self::parse_value(serde_json::from_slice(bytes)?);
        }
        let top: Top = match serde_json::from_slice(bytes) {
            Ok(t) => t,
            // Duplicate top-level keys and the like: fall back to the `Value` reading (last key wins).
            Err(e) if !e.is_syntax() && !e.is_eof() => {
                return Self::parse_value(serde_json::from_slice(bytes)?);
            }
            Err(e) => return Err(e),
        };
        let raws: Vec<&serde_json::value::RawValue> =
            top.annotations.and_then(|a| serde_json::from_str(a.get()).ok()).unwrap_or_default();
        let annotations = raws
            .into_iter()
            .map(|raw| {
                let typed = if is_object(raw.get().as_bytes()) {
                    serde_json::from_str::<AnnFields>(raw.get()).ok()
                } else {
                    None
                };
                let mut a = typed.unwrap_or_else(|| {
                    // Not an object (no fields, as with `Value::get`), or a duplicate field (last key wins).
                    serde_json::from_str::<Value>(raw.get())
                        .map(|v| AnnFields::from_value(&v))
                        .unwrap_or_default()
                });
                if a.image_id.is_none() {
                    a.text = raw.get().to_string();
                }
                a
            })
            .collect();
        Ok(Sections { images: top.images, categories: top.categories, annotations })
    }

    /// [`Sections::parse`] for a file of a split that is not opened: the categories as `parse` reads them, and
    /// `images` only as far as whether it is an array (an empty one stands for it), no annotations.
    fn parse_categories(bytes: &[u8]) -> serde_json::Result<Self> {
        /// Whether a JSON value is an array, read without building it.
        struct IsArray(bool);
        impl<'de> serde::Deserialize<'de> for IsArray {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                struct V;
                impl<'de> serde::de::Visitor<'de> for V {
                    type Value = IsArray;
                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str("any JSON value")
                    }
                    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut a: A) -> Result<IsArray, A::Error> {
                        while a.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                        Ok(IsArray(true))
                    }
                    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<IsArray, A::Error> {
                        while a.next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?.is_some() {}
                        Ok(IsArray(false))
                    }
                    fn visit_bool<E>(self, _: bool) -> Result<IsArray, E> {
                        Ok(IsArray(false))
                    }
                    fn visit_i64<E>(self, _: i64) -> Result<IsArray, E> {
                        Ok(IsArray(false))
                    }
                    fn visit_u64<E>(self, _: u64) -> Result<IsArray, E> {
                        Ok(IsArray(false))
                    }
                    fn visit_f64<E>(self, _: f64) -> Result<IsArray, E> {
                        Ok(IsArray(false))
                    }
                    fn visit_str<E>(self, _: &str) -> Result<IsArray, E> {
                        Ok(IsArray(false))
                    }
                    fn visit_unit<E>(self) -> Result<IsArray, E> {
                        Ok(IsArray(false))
                    }
                }
                d.deserialize_any(V)
            }
        }
        #[derive(serde::Deserialize)]
        struct CategoriesTop {
            images: Option<IsArray>,
            categories: Option<Value>,
        }
        let marker = |is_array: bool| is_array.then(|| Value::Array(Vec::new()));
        let from_value = |v: Value| Sections {
            images: marker(v.get("images").is_some_and(Value::is_array)),
            categories: v.get("categories").cloned(),
            annotations: Vec::new(),
        };
        if !is_object(bytes) {
            return Ok(from_value(serde_json::from_slice(bytes)?));
        }
        match serde_json::from_slice::<CategoriesTop>(bytes) {
            Ok(t) => Ok(Sections {
                images: marker(t.images.is_some_and(|a| a.0)),
                categories: t.categories,
                annotations: Vec::new(),
            }),
            // Duplicate top-level keys and the like: the `Value` reading (last key wins), as `parse` does.
            Err(e) if !e.is_syntax() && !e.is_eof() => Ok(from_value(serde_json::from_slice(bytes)?)),
            Err(e) => Err(e),
        }
    }

    fn parse_value(v: Value) -> serde_json::Result<Self> {
        let annotations = v
            .get("annotations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|a| AnnFields { text: a.to_string(), ..AnnFields::from_value(a) })
            .collect();
        Ok(Sections {
            images: v.get("images").cloned(),
            categories: v.get("categories").cloned(),
            annotations,
        })
    }
}

pub(crate) fn open_jsons(
    root: &Path,
    jsons: Vec<(Split, PathBuf)>,
    opts: &OpenOptions,
    wanted: &[Split],
) -> Result<LoadedDataset> {
    Ok(open_jsons_impl(root, jsons, opts, wanted, None::<super::ImageReader<'_, ()>>)?.0)
}

/// [`open_jsons`] using `reader` to read each COCO image once (for [`crate::check`]), returning the dataset and
/// a [`super::CheckedImage`] for every kept entry and every unreadable entry without a declared size.
pub(crate) fn open_jsons_with<T: Send>(
    root: &Path,
    jsons: Vec<(Split, PathBuf)>,
    opts: &OpenOptions,
    wanted: &[Split],
    reader: super::ImageReader<'_, T>,
) -> Result<(LoadedDataset, Vec<super::CheckedImage<T>>)> {
    open_jsons_impl(root, jsons, opts, wanted, Some(reader))
}

fn open_jsons_impl<T: Send>(
    root: &Path,
    jsons: Vec<(Split, PathBuf)>,
    opts: &OpenOptions,
    wanted: &[Split],
    reader: Option<super::ImageReader<'_, T>>,
) -> Result<(LoadedDataset, Vec<super::CheckedImage<T>>)> {
    let settings = super::cache::Settings::current();
    let ann = super::cache::coco::load_or_read(&settings, &jsons, wanted, || {
        read_annotations(root, &jsons, wanted)
    })?;
    let Annotations { issues, classes, other_files, splits } = ann;
    let mut report = LoadReport { issues };
    let mut deps: std::collections::BTreeSet<PathBuf> = std::iter::once(root.to_path_buf()).collect();
    deps.extend(other_files);
    let mut by_split: BTreeMap<Split, Vec<Candidate>> = BTreeMap::new();
    for p in splits {
        deps.insert(p.path.clone());
        let files: Vec<&str> = p.files.iter().map(String::as_str).collect();
        let base = image_base(root, &p.path, p.split, &files);
        let list = by_split.entry(p.split).or_default();
        for im in p.images {
            let image = base.join(&p.files[im.row]);
            deps.insert(crate::paths::parent_dir(&image).to_path_buf());
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
    let mut checks = Vec::new();
    for (split, cands) in by_split {
        let (entries, split_checks) =
            super::build_entries_with(cands, Some(classes.len()), opts, &mut report, reader);
        built.entry(split).or_default().extend(entries);
        checks.extend(split_checks);
    }
    let mut take = |s: Split| FileDataset::new(s, classes.clone(), built.remove(&s).unwrap_or_default());
    let (train, val, test) = (take(Split::Train), take(Split::Val), take(Split::Test));
    let sources = deps.into_iter().collect();
    Ok((
        LoadedDataset {
            format: Format::Coco,
            root: root.to_path_buf(),
            classes,
            train,
            val,
            test,
            report,
            sources,
        },
        checks,
    ))
}

/// What [`open_jsons`] derives from the JSON files alone, before it looks for the images: the part the scan cache
/// keeps (`cache::coco`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Annotations {
    /// The issues found in the files, in order.
    pub issues: Vec<super::Issue>,
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
    let mut categories: BTreeMap<u64, String> = BTreeMap::new();
    let mut parsed = Vec::new();
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
        let Some(images) = sections.images.as_ref().and_then(Value::as_array) else {
            report.push(IssueKind::BrokenLabel, &path, "no `images` array: not a COCO file");
            continue;
        };
        for c in sections.categories.as_ref().and_then(Value::as_array).into_iter().flatten() {
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
        if !wanted.contains(&split) {
            other_files.push(path);
            continue;
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
        let mut anns = Vec::with_capacity(sections.annotations.len());
        for a in sections.annotations {
            if a.iscrowd.as_ref().and_then(Value::as_u64) == Some(1) {
                continue;
            }
            let Some(img) = a.image_id.as_ref().and_then(id_key) else {
                report.push(
                    IssueKind::BrokenLabel,
                    &path,
                    format!("annotation without image_id: {}", a.text),
                );
                continue;
            };
            let cat = a.category_id.as_ref().and_then(Value::as_u64).unwrap_or(u64::MAX);
            anns.push((img, cat, a.bbox.unwrap_or(Value::Null)));
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
        // Image id -> the file of its first entry: a second entry with the id is left out (its annotations cannot
        // be told apart from the first's).
        let mut first: HashMap<&str, &str> = HashMap::new();
        let mut images = Vec::with_capacity(p.images.len());
        for (row, (id, file, declared)) in p.images.iter().enumerate() {
            if let Some(&kept) = first.get(id.as_str()) {
                report.push(
                    IssueKind::Other,
                    &p.path,
                    format!("image id {id} is used by both {kept} and {file}; {file} is left out"),
                );
                continue;
            }
            first.insert(id, file);
            let (boxes, broken) = per_image.remove(id.as_str()).unwrap_or_default();
            images.push(ImageAnnotations { row, declared: *declared, boxes, broken });
        }
        let files = p.images.into_iter().map(|(_, file, _)| file).collect();
        splits.push(SplitAnnotations { split: p.split, path: p.path, files, images });
    }
    Ok(Annotations { issues: report.issues, classes, other_files, splits })
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

    type Fields = (Option<Value>, Option<Value>, Option<Value>, Option<Value>);

    fn fields(s: &Sections) -> Vec<Fields> {
        s.annotations
            .iter()
            .map(|a| (a.image_id.clone(), a.category_id.clone(), a.bbox.clone(), a.iscrowd.clone()))
            .collect()
    }

    /// The borrowed parse agrees with reading the whole file as a `Value`, including on malformed input.
    #[test]
    fn borrowed_parse_matches_value_parse() {
        let cases = [
            r#"{"images":[{"id":1,"file_name":"a.jpg"}],"annotations":[{"id":1,"image_id":1,"category_id":2,"bbox":[0,0,10,10],"segmentation":[[0,0,1,1]],"iscrowd":0}],"categories":[{"id":2,"name":"x"}]}"#,
            // Positional arrays and scalars are not annotations.
            r#"{"images":[],"annotations":[[1,1,[0,0,10,10],0], 7, "s", null, {"image_id":3}]}"#,
            // Duplicate fields: the last one wins.
            r#"{"images":[],"annotations":[{"image_id":1,"image_id":2,"bbox":null}]}"#,
            // `annotations` not an array; missing sections.
            r#"{"images":[],"annotations":{"image_id":1}}"#,
            r#"{"categories":[]}"#,
            // Duplicate top-level keys.
            r#"{"images":[1],"images":[],"annotations":[{"image_id":5}],"annotations":[{"image_id":6}]}"#,
            // The top level is not an object (an array would fill a derived struct positionally).
            r#"[[{"id":1,"file_name":"a.jpg"}], [], [{"image_id":1}]]"#,
            r#"  [1, 2, 3]"#,
            r#"{"images":[], "annotations":[ {"image_id": "s1", "iscrowd": 1}, {"category_id": 4.5} ]}"#,
        ];
        for c in cases {
            let a = Sections::parse(c.as_bytes()).unwrap();
            let b = Sections::parse_value(serde_json::from_str(c).unwrap()).unwrap();
            assert_eq!((&a.images, &a.categories), (&b.images, &b.categories), "{c}");
            assert_eq!(fields(&a), fields(&b), "{c}");
        }
        assert!(Sections::parse(br#"{"images": ["#).is_err());
    }
}
