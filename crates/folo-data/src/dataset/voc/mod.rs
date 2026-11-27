//! Pascal VOC XML detection datasets (`Annotations/*.xml`, `JPEGImages/` or colocated images, optional
//! `ImageSets/Main/{train,val,trainval,test}.txt`).

pub(crate) mod xml;

use super::labels::strip_bom;
use super::{
    Candidate, FileDataset, Format, IssueKind, LabelSource, LoadReport, LoadedDataset, OpenOptions, Split,
    build_entries,
};
use crate::error::{DataError, Result};
use crate::image::IMAGE_EXTENSIONS;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

/// Options for [`open_voc_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocOptions {
    /// Whether `<difficult>1</difficult>` objects are kept (`true` by default).
    pub include_difficult: bool,
    /// Read every image header while opening so corrupt files are skipped up front and sizes are verified.
    pub verify_images: bool,
}

impl Default for VocOptions {
    fn default() -> Self {
        Self { include_difficult: true, verify_images: true }
    }
}

/// One `<object>` parsed from a Pascal VOC XML annotation file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VocObject {
    /// Decoded class name from `<name>`.
    pub name: String,
    /// Pixel corners `[xmin, ymin, xmax, ymax]`.
    pub bbox: [f32; 4],
    /// Whether `<difficult>` was truthy (`1`, `true`, `yes`).
    pub difficult: bool,
}

/// A parsed Pascal VOC `<annotation>` XML document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VocAnnotation {
    /// Optional `<filename>` text (trimmed, XML entities decoded).
    pub filename: Option<String>,
    /// Optional `<folder>` text (trimmed, XML entities decoded).
    pub folder: Option<String>,
    /// Optional declared `(width, height)` from `<size>`, when both are > 0.
    pub size: Option<(u32, u32)>,
    /// Parsed `<object>` entries, in document order.
    pub objects: Vec<VocObject>,
}

/// Parses one Pascal VOC XML document into a [`VocAnnotation`].
pub fn parse_voc_xml(text: &str) -> std::result::Result<VocAnnotation, String> {
    let root = xml::parse_document(text)?;
    if root.name != "annotation" {
        return Err(format!("expected root element <annotation>, got <{}>", root.name));
    }
    let filename = root.child_text("filename").filter(|s| !s.is_empty()).map(String::from);
    let folder = root.child_text("folder").filter(|s| !s.is_empty()).map(String::from);
    let size = parse_size(&root)?;
    let mut objects = Vec::new();
    for (idx, obj) in root.children_named("object").enumerate() {
        let n = idx + 1;
        let name = obj
            .child_text("name")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("object #{n}: missing or empty <name>"))?
            .to_string();
        let difficult = match obj.child_text("difficult") {
            None | Some("") => false,
            Some(v) => parse_bool_flag(v).ok_or_else(|| {
                format!("object #{n} ({name}): invalid <difficult> value {v:?} (expected 0/1 or true/false)")
            })?,
        };
        let bnd = obj.child("bndbox").ok_or_else(|| format!("object #{n} ({name}): missing <bndbox>"))?;
        let xmin = parse_coord(bnd, "xmin", n, &name)?;
        let ymin = parse_coord(bnd, "ymin", n, &name)?;
        let xmax = parse_coord(bnd, "xmax", n, &name)?;
        let ymax = parse_coord(bnd, "ymax", n, &name)?;
        if xmax < xmin || ymax < ymin {
            return Err(format!(
                "object #{n} ({name}): invalid <bndbox> [{xmin}, {ymin}, {xmax}, {ymax}] (max < min)"
            ));
        }
        objects.push(VocObject {
            name,
            bbox: [xmin as f32, ymin as f32, xmax as f32, ymax as f32],
            difficult,
        });
    }
    Ok(VocAnnotation { filename, folder, size, objects })
}

fn parse_size(root: &xml::Element) -> std::result::Result<Option<(u32, u32)>, String> {
    let Some(size) = root.child("size") else {
        return Ok(None);
    };
    let w_raw = size.child_text("width").filter(|s| !s.is_empty());
    let h_raw = size.child_text("height").filter(|s| !s.is_empty());
    match (w_raw, h_raw) {
        (None, None) => Ok(None),
        (Some(w), Some(h)) => {
            let w = parse_dim(w, "width")?;
            let h = parse_dim(h, "height")?;
            if w == 0 || h == 0 { Ok(None) } else { Ok(Some((w, h))) }
        }
        _ => Err("<size> must provide both <width> and <height>".into()),
    }
}

fn parse_dim(s: &str, field: &str) -> std::result::Result<u32, String> {
    if let Ok(v) = s.parse::<u32>() {
        return Ok(v);
    }
    if let Ok(f) = s.parse::<f64>()
        && f.is_finite()
        && f >= 0.0
        && f.fract() == 0.0
        && f <= u32::MAX as f64
    {
        return Ok(f as u32);
    }
    Err(format!("<size><{field}> {s:?} is not a non-negative integer"))
}

fn parse_coord(
    bnd: &xml::Element,
    field: &str,
    obj_idx: usize,
    name: &str,
) -> std::result::Result<f64, String> {
    let raw = bnd
        .child_text(field)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("object #{obj_idx} ({name}): missing <bndbox><{field}>"))?;
    let v = raw
        .parse::<f64>()
        .map_err(|_| format!("object #{obj_idx} ({name}): invalid <{field}> value {raw:?}"))?;
    if !v.is_finite() || v.abs() >= 1e9 {
        return Err(format!("object #{obj_idx} ({name}): <{field}> value {raw:?} is out of range"));
    }
    Ok(v)
}

fn parse_bool_flag(s: &str) -> Option<bool> {
    match s.to_ascii_lowercase().as_str() {
        "0" | "false" | "no" => Some(false),
        "1" | "true" | "yes" => Some(true),
        _ => None,
    }
}

/// Opens a Pascal VOC dataset with VOC-specific options such as [`VocOptions::include_difficult`].
pub fn open_voc_with(path: impl AsRef<Path>, opts: &VocOptions) -> Result<LoadedDataset> {
    let path = path.as_ref();
    let super::Source::Voc { root, single_xml } = super::detect(path, Format::Voc)? else {
        return Err(DataError::Format(format!("{} is not a Pascal VOC dataset", crate::paths::shown(path))));
    };
    let open_opts = OpenOptions { format: Format::Voc, verify_images: opts.verify_images };
    open_voc_inner(&root, single_xml.as_deref(), &open_opts, &Split::ALL, opts.include_difficult)
}

/// Whether `dir` looks like a Pascal VOC dataset directory without reading every annotation.
pub(crate) fn looks_like_voc_dir(dir: &Path) -> bool {
    if let Some(ann_dir) = find_annotations_dir(dir)
        && has_xml_files(&ann_dir)
    {
        return true;
    }
    for sub in ["train", "val", "valid", "test"] {
        let sub_dir = dir.join(sub);
        if sub_dir.is_dir() {
            if let Some(ann) = find_annotations_dir(&sub_dir)
                && has_xml_files(&ann)
            {
                return true;
            }
            if has_voc_xml_files(&sub_dir) {
                return true;
            }
        }
    }
    has_voc_xml_files(dir)
}

fn find_annotations_dir(dir: &Path) -> Option<PathBuf> {
    for name in ["Annotations", "annotations"] {
        let p = dir.join(name);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

fn find_imagesets_main_dir(dir: &Path) -> Option<PathBuf> {
    for rel in ["ImageSets/Main", "imagesets/main", "ImageSets/main", "imagesets/Main"] {
        let p = dir.join(rel);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

fn xml_files_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = crate::guard::check(dir).and_then(|_| std::fs::read_dir(dir)) else { return Vec::new() };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("xml")))
        .collect();
    out.sort();
    out
}

fn has_xml_files(dir: &Path) -> bool {
    let Ok(rd) = crate::guard::check(dir).and_then(|_| std::fs::read_dir(dir)) else { return false };
    rd.filter_map(|e| e.ok().map(|e| e.path()))
        .any(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("xml")))
}

/// Checks if `dir` has at least one `.xml` file that starts like a Pascal VOC `<annotation>` document.
fn has_voc_xml_files(dir: &Path) -> bool {
    xml_files_in(dir).into_iter().take(8).any(|p| looks_like_voc_xml(&p))
}

fn looks_like_voc_xml(p: &Path) -> bool {
    use std::io::Read;
    let Ok(f) = crate::guard::check(p).and_then(|_| std::fs::File::open(p)) else { return false };
    let mut head = Vec::with_capacity(4096);
    if f.take(4096).read_to_end(&mut head).is_err() {
        return false;
    }
    let head = String::from_utf8_lossy(&head);
    head.contains("<annotation")
}

/// One discovered VOC item before XML parsing.
#[derive(Clone, Debug)]
pub(crate) enum VocItem {
    /// An XML annotation file on disk.
    Xml(PathBuf),
    /// An image ID from `ImageSets/Main/*.txt` whose XML file does not exist (background image or missing image).
    StemOnly { stem: String, split_file: PathBuf },
}

/// Result of scanning a VOC dataset's layout and split assignments.
pub(crate) struct VocScan {
    pub root: PathBuf,
    pub splits: BTreeMap<Split, Vec<VocItem>>,
    pub sources: BTreeSet<PathBuf>,
    pub report: LoadReport,
}

/// Discovers splits and annotation files in a VOC dataset.
pub(crate) fn scan_voc(root: &Path, single_xml: Option<&Path>) -> VocScan {
    let mut report = LoadReport::default();
    let mut splits: BTreeMap<Split, Vec<VocItem>> = BTreeMap::new();
    let mut sources: BTreeSet<PathBuf> = BTreeSet::new();
    sources.insert(root.to_path_buf());

    if let Some(xml_path) = single_xml {
        let split = guess_split_from_parent(xml_path);
        sources.insert(xml_path.to_path_buf());
        sources.insert(crate::paths::parent_dir(xml_path).to_path_buf());
        splits.insert(split, vec![VocItem::Xml(xml_path.to_path_buf())]);
        return VocScan { root: root.to_path_buf(), splits, sources, report };
    }

    let ann_dir = find_annotations_dir(root);
    if let Some(ref d) = ann_dir {
        sources.insert(d.clone());
    }

    // 1. Classic Pascal VOC `ImageSets/Main/{train,val,trainval,test}.txt`
    if let Some(main_dir) = find_imagesets_main_dir(root) {
        let train_txt = find_split_txt(&main_dir, "train.txt");
        let val_txt = find_split_txt(&main_dir, "val.txt");
        let trainval_txt = find_split_txt(&main_dir, "trainval.txt");
        let test_txt = find_split_txt(&main_dir, "test.txt");

        let mut split_files: Vec<(Split, PathBuf)> = Vec::new();
        if let Some(p) = train_txt.clone() {
            split_files.push((Split::Train, p));
        }
        if let Some(p) = val_txt.clone() {
            split_files.push((Split::Val, p));
        }
        // Only fall back to `trainval.txt` when neither `train.txt` nor `val.txt` is present
        // (standard VOC ships all three where `trainval.txt` is the union of `train.txt` and `val.txt`).
        if train_txt.is_none()
            && val_txt.is_none()
            && let Some(p) = trainval_txt
        {
            split_files.push((Split::Train, p));
        }
        if let Some(p) = test_txt {
            split_files.push((Split::Test, p));
        }

        if !split_files.is_empty() {
            sources.insert(main_dir);
            let xml_index = build_xml_stem_index(ann_dir.as_deref().unwrap_or(root));
            for (split, txt) in split_files {
                sources.insert(txt.clone());
                let items = read_imageset_file(&txt, &xml_index, &mut report);
                splits.entry(split).or_default().extend(items);
            }
            return VocScan { root: root.to_path_buf(), splits, sources, report };
        }
    }

    // 2. Split subdirectories (`train/`, `val/`, `valid/`, `test/`).
    let mut found_split_dirs = false;
    for (split, name) in
        [(Split::Train, "train"), (Split::Val, "val"), (Split::Val, "valid"), (Split::Test, "test")]
    {
        let sub = root.join(name);
        if !sub.is_dir() {
            continue;
        }
        let xmls = if let Some(sub_ann) = find_annotations_dir(&sub) {
            sources.insert(sub_ann.clone());
            xml_files_in(&sub_ann)
        } else {
            xml_files_in(&sub)
        };
        if !xmls.is_empty() {
            found_split_dirs = true;
            sources.insert(sub);
            splits.entry(split).or_default().extend(xmls.into_iter().map(VocItem::Xml));
        }
    }
    if found_split_dirs {
        return VocScan { root: root.to_path_buf(), splits, sources, report };
    }

    // 3. Flat `Annotations/*.xml` or root `*.xml` -> `Split::Train`.
    let xmls = if let Some(ref d) = ann_dir { xml_files_in(d) } else { xml_files_in(root) };
    if !xmls.is_empty() {
        splits.insert(Split::Train, xmls.into_iter().map(VocItem::Xml).collect());
    }

    VocScan { root: root.to_path_buf(), splits, sources, report }
}

fn find_split_txt(dir: &Path, name: &str) -> Option<PathBuf> {
    let p = dir.join(name);
    if p.is_file() {
        return Some(p);
    }
    None
}

fn guess_split_from_parent(xml_path: &Path) -> Split {
    let parent = crate::paths::parent_dir(xml_path);
    let check_dir = if parent.file_name().is_some_and(|n| n.eq_ignore_ascii_case("annotations")) {
        crate::paths::parent_dir(parent)
    } else {
        parent
    };
    let name = check_dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    match name.as_str() {
        "val" | "valid" | "validation" => Split::Val,
        "test" => Split::Test,
        _ => Split::Train,
    }
}

fn build_xml_stem_index(dir: &Path) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    for p in xml_files_in(dir) {
        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
            map.entry(stem.to_string()).or_insert(p);
        }
    }
    map
}

fn read_imageset_file(
    txt: &Path,
    xml_index: &HashMap<String, PathBuf>,
    report: &mut LoadReport,
) -> Vec<VocItem> {
    let text = match crate::guard::check(txt).and_then(|_| std::fs::read_to_string(txt)) {
        Ok(t) => t,
        Err(e) => {
            report.push(IssueKind::Other, txt, format!("cannot read ImageSets split file: {e}"));
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw_line in strip_bom(&text).lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(token) = line.split_whitespace().next() else { continue };
        let stem = strip_known_ext(token);
        if stem.is_empty() || !seen.insert(stem.to_string()) {
            continue;
        }
        if let Some(xml_path) = xml_index.get(stem) {
            out.push(VocItem::Xml(xml_path.clone()));
        } else {
            out.push(VocItem::StemOnly { stem: stem.to_string(), split_file: txt.to_path_buf() });
        }
    }
    out
}

fn strip_known_ext(token: &str) -> &str {
    let p = Path::new(token);
    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
        let lower = ext.to_ascii_lowercase();
        if lower == "xml" || IMAGE_EXTENSIONS.contains(&lower.as_str()) {
            return p.file_stem().and_then(|s| s.to_str()).unwrap_or(token);
        }
    }
    token
}

/// Opens a Pascal VOC dataset from `root` (or `single_xml`), reading only `wanted` splits' images.
pub(crate) fn open_voc(
    root: &Path,
    single_xml: Option<&Path>,
    opts: &OpenOptions,
    wanted: &[Split],
) -> Result<LoadedDataset> {
    open_voc_inner(root, single_xml, opts, wanted, true)
}

enum ParsedItem {
    Valid { xml_path: PathBuf, ann: VocAnnotation },
    Broken { xml_path: PathBuf, message: String },
    StemOnly { stem: String, split_file: PathBuf },
}

fn open_voc_inner(
    root: &Path,
    single_xml: Option<&Path>,
    opts: &OpenOptions,
    wanted: &[Split],
    include_difficult: bool,
) -> Result<LoadedDataset> {
    let VocScan { root, splits, mut sources, mut report } = scan_voc(root, single_xml);

    // Optional `classes.txt` gives a pinned class order; any extra names found in XMLs are appended sorted.
    let mut classes = read_classes_txt(&root, &mut report, &mut sources);
    let mut class_set: HashSet<String> = classes.iter().cloned().collect();
    let mut discovered_names: BTreeSet<String> = BTreeSet::new();

    // Parse XMLs across all splits so `classes` is identical regardless of `wanted`.
    let mut parsed_splits: BTreeMap<Split, Vec<ParsedItem>> = BTreeMap::new();
    for (split, items) in splits {
        let is_wanted = wanted.contains(&split);
        let mut parsed = Vec::with_capacity(if is_wanted { items.len() } else { 0 });
        for item in items {
            match item {
                VocItem::Xml(xml_path) => match read_and_parse_xml(&xml_path) {
                    Ok(ann) => {
                        for obj in &ann.objects {
                            if !class_set.contains(&obj.name) {
                                discovered_names.insert(obj.name.clone());
                            }
                        }
                        if is_wanted {
                            parsed.push(ParsedItem::Valid { xml_path, ann });
                        }
                    }
                    Err(msg) => {
                        if is_wanted {
                            parsed.push(ParsedItem::Broken { xml_path, message: msg });
                        }
                    }
                },
                VocItem::StemOnly { stem, split_file } => {
                    if is_wanted {
                        parsed.push(ParsedItem::StemOnly { stem, split_file });
                    }
                }
            }
        }
        if is_wanted {
            parsed_splits.insert(split, parsed);
        }
    }

    for name in discovered_names {
        if class_set.insert(name.clone()) {
            classes.push(name);
        }
    }
    let class_map: HashMap<&str, u32> =
        classes.iter().enumerate().map(|(i, n)| (n.as_str(), i as u32)).collect();

    let mut built: BTreeMap<Split, Vec<super::ImageEntry>> = BTreeMap::new();
    for (split, items) in parsed_splits {
        let mut candidates = Vec::with_capacity(items.len());
        for item in items {
            match item {
                ParsedItem::Valid { xml_path, ann } => {
                    let image = resolve_voc_image(
                        &root,
                        split,
                        &xml_path,
                        ann.filename.as_deref(),
                        ann.folder.as_deref(),
                    );
                    sources.insert(crate::paths::parent_dir(&image).to_path_buf());
                    sources.insert(crate::paths::parent_dir(&xml_path).to_path_buf());
                    let boxes: Vec<(u32, [f32; 4])> = ann
                        .objects
                        .into_iter()
                        .filter(|o| include_difficult || !o.difficult)
                        .filter_map(|o| class_map.get(o.name.as_str()).map(|&c| (c, o.bbox)))
                        .collect();
                    candidates.push(Candidate {
                        image,
                        labels: LabelSource::Voc {
                            boxes,
                            declared: ann.size,
                            broken: None,
                            label: Some(xml_path),
                        },
                    });
                }
                ParsedItem::Broken { xml_path, message } => {
                    let image = resolve_voc_image(&root, split, &xml_path, None, None);
                    sources.insert(crate::paths::parent_dir(&xml_path).to_path_buf());
                    if image.is_file() {
                        candidates.push(Candidate {
                            image,
                            labels: LabelSource::Voc {
                                boxes: Vec::new(),
                                declared: None,
                                broken: Some(message),
                                label: Some(xml_path),
                            },
                        });
                    } else {
                        // Even when the image is missing or unresolvable from a corrupt XML, report the broken XML.
                        report.push(IssueKind::BrokenLabel, &xml_path, message);
                        report.record_skipped_label_image();
                    }
                }
                ParsedItem::StemOnly { stem, split_file } => {
                    if let Some(image) = resolve_stem_image(&root, split, &stem) {
                        sources.insert(crate::paths::parent_dir(&image).to_path_buf());
                        candidates.push(Candidate {
                            image,
                            labels: LabelSource::Voc {
                                boxes: Vec::new(),
                                declared: None,
                                broken: None,
                                label: None,
                            },
                        });
                    } else {
                        report.push(
                            IssueKind::MissingImage,
                            &split_file,
                            format!("no annotation XML or image found for {stem:?}"),
                        );
                    }
                }
            }
        }
        built.insert(split, build_entries(candidates, Some(classes.len()), opts, &mut report));
    }

    let mut take = |s: Split| FileDataset::new(s, classes.clone(), built.remove(&s).unwrap_or_default());
    let (train, val, test) = (take(Split::Train), take(Split::Val), take(Split::Test));
    Ok(LoadedDataset {
        format: Format::Voc,
        root,
        classes,
        train,
        val,
        test,
        report,
        sources: sources.into_iter().collect(),
    })
}

fn read_classes_txt(root: &Path, report: &mut LoadReport, sources: &mut BTreeSet<PathBuf>) -> Vec<String> {
    let p = root.join("classes.txt");
    if !p.is_file() {
        return Vec::new();
    }
    sources.insert(p.clone());
    match crate::guard::check(&p).and_then(|_| std::fs::read_to_string(&p)) {
        Ok(text) => {
            strip_bom(&text).lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()
        }
        Err(e) => {
            report.push(IssueKind::Other, &p, format!("cannot read class names: {e}"));
            Vec::new()
        }
    }
}

pub(crate) fn read_and_parse_xml(path: &Path) -> std::result::Result<VocAnnotation, String> {
    let bytes = crate::guard::check(path)
        .and_then(|_| std::fs::read(path))
        .map_err(|e| format!("cannot read XML: {e}"))?;
    let text = std::str::from_utf8(&bytes).map_err(|_| "XML file is not valid UTF-8 text".to_string())?;
    parse_voc_xml(text)
}

/// Candidate directories where images for a VOC annotation may live, in priority order.
fn image_search_dirs(
    root: &Path,
    split: Split,
    xml_path: Option<&Path>,
    folder: Option<&str>,
) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut push_unique = |p: PathBuf| {
        if !dirs.contains(&p) {
            dirs.push(p);
        }
    };
    if let Some(xml) = xml_path {
        let xml_dir = crate::paths::parent_dir(xml);
        push_unique(xml_dir.to_path_buf());
        if xml_dir.file_name().is_some_and(|n| n.eq_ignore_ascii_case("annotations")) {
            let up = crate::paths::parent_dir(xml_dir);
            push_unique(up.join("JPEGImages"));
            push_unique(up.join("jpegimages"));
            push_unique(up.join("images"));
            push_unique(up.to_path_buf());
        }
    }
    push_unique(root.join("JPEGImages"));
    push_unique(root.join("jpegimages"));
    push_unique(root.join("images").join(split.as_str()));
    push_unique(root.join("images"));
    push_unique(root.join(split.as_str()));
    if split == Split::Val {
        push_unique(root.join("images").join("valid"));
        push_unique(root.join("valid"));
    }
    if let Some(f) = folder
        && let Some(safe) = safe_relative_subpath(f)
    {
        push_unique(root.join(safe));
    }
    push_unique(root.to_path_buf());
    dirs
}

/// Strips any path traversal (`..`, root, prefixes) from an XML `<folder>` or `<filename>` relative hint.
fn safe_relative_subpath(raw: &str) -> Option<PathBuf> {
    let normalized = raw.replace('\\', "/");
    let p = Path::new(&normalized);
    if p.is_absolute() {
        return None;
    }
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

/// Extracts just the basename from a `<filename>` (handles exports that embed `C:\path\img.jpg` or `sub/img.jpg`).
fn filename_basename(raw: &str) -> Option<&str> {
    let s = raw.rsplit(['/', '\\']).next().unwrap_or(raw).trim();
    (!s.is_empty() && s != "." && s != "..").then_some(s)
}

/// Resolves the image file on disk corresponding to a VOC XML annotation.
pub(crate) fn resolve_voc_image(
    root: &Path,
    split: Split,
    xml_path: &Path,
    filename: Option<&str>,
    folder: Option<&str>,
) -> PathBuf {
    let dirs = image_search_dirs(root, split, Some(xml_path), folder);
    let mut names: Vec<PathBuf> = Vec::new();
    if let Some(raw) = filename {
        if let Some(rel) = safe_relative_subpath(raw) {
            names.push(rel);
        }
        if let Some(base) = filename_basename(raw) {
            let bp = PathBuf::from(base);
            if !names.contains(&bp) {
                names.push(bp);
            }
        }
    }
    // Try exact filename candidates first.
    for dir in &dirs {
        for name in &names {
            let cand = dir.join(name);
            if cand.is_file() {
                return cand;
            }
        }
    }
    // Try matching stem with known image extensions (handles extensionless `<filename>` or mismatched ext).
    let mut stems: Vec<String> = Vec::new();
    if let Some(raw) = filename
        && let Some(base) = filename_basename(raw)
        && let Some(st) = Path::new(base).file_stem().and_then(|s| s.to_str())
    {
        stems.push(st.to_string());
    }
    if let Some(st) = xml_path.file_stem().and_then(|s| s.to_str())
        && !stems.iter().any(|s| s == st)
    {
        stems.push(st.to_string());
    }
    for dir in &dirs {
        for stem in &stems {
            if let Some(found) = find_image_with_stem(dir, stem) {
                return found;
            }
        }
    }
    // Fallback path reported in `MissingImage` issue.
    let fallback_name = names
        .first()
        .cloned()
        .or_else(|| stems.first().map(|s| PathBuf::from(format!("{s}.jpg"))))
        .unwrap_or_else(|| PathBuf::from("image.jpg"));
    let fallback_dir = if root.join("JPEGImages").is_dir() {
        root.join("JPEGImages")
    } else {
        crate::paths::parent_dir(xml_path).to_path_buf()
    };
    fallback_dir.join(fallback_name)
}

fn resolve_stem_image(root: &Path, split: Split, stem: &str) -> Option<PathBuf> {
    for dir in image_search_dirs(root, split, None, None) {
        if let Some(found) = find_image_with_stem(&dir, stem) {
            return Some(found);
        }
    }
    None
}

fn find_image_with_stem(dir: &Path, stem: &str) -> Option<PathBuf> {
    for ext in IMAGE_EXTENSIONS {
        let p = dir.join(format!("{stem}.{ext}"));
        if p.is_file() {
            return Some(p);
        }
        let upper = dir.join(format!("{stem}.{}", ext.to_ascii_uppercase()));
        if upper.is_file() {
            return Some(upper);
        }
    }
    None
}

/// Formats a [`LoadedDataset`] image's boxes as a Pascal VOC `<annotation>` XML document.
pub(crate) fn format_voc_xml(
    filename: &str,
    folder: &str,
    width: u32,
    height: u32,
    classes: &[String],
    boxes: &[crate::BBox],
) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(256 + boxes.len() * 180);
    out.push_str("<annotation>\n");
    let _ = writeln!(out, "  <folder>{}</folder>", xml::escape_text(folder));
    let _ = writeln!(out, "  <filename>{}</filename>", xml::escape_text(filename));
    out.push_str("  <size>\n");
    let _ = writeln!(out, "    <width>{width}</width>");
    let _ = writeln!(out, "    <height>{height}</height>");
    out.push_str("    <depth>3</depth>\n");
    out.push_str("  </size>\n");
    out.push_str("  <segmented>0</segmented>\n");
    for b in boxes {
        let class_name = classes.get(b.class as usize).map(String::as_str).unwrap_or("unknown");
        let [x1, y1, x2, y2] = b.pixel_xyxy(width, height);
        out.push_str("  <object>\n");
        let _ = writeln!(out, "    <name>{}</name>", xml::escape_text(class_name));
        out.push_str("    <pose>Unspecified</pose>\n");
        out.push_str("    <truncated>0</truncated>\n");
        out.push_str("    <difficult>0</difficult>\n");
        out.push_str("    <bndbox>\n");
        let _ = writeln!(out, "      <xmin>{x1:.2}</xmin>");
        let _ = writeln!(out, "      <ymin>{y1:.2}</ymin>");
        let _ = writeln!(out, "      <xmax>{x2:.2}</xmax>");
        let _ = writeln!(out, "      <ymax>{y2:.2}</ymax>");
        out.push_str("    </bndbox>\n");
        out.push_str("  </object>\n");
    }
    out.push_str("</annotation>\n");
    out
}
