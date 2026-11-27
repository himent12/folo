//! Pascal VOC layout detection, `ImageSets/Main` split scanning, and guarded image path resolution.

use super::super::labels::strip_bom;
use super::super::{IssueKind, LoadReport, Split};
use crate::image::IMAGE_EXTENSIONS;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

/// One discovered VOC item before XML parsing.
#[derive(Clone, Debug)]
pub(crate) enum VocItem {
    /// An XML annotation file on disk.
    Xml(PathBuf),
    /// A validated relative image ID from `ImageSets/Main/*.txt` whose XML file does not exist.
    StemOnly { stem: PathBuf, raw: String, split_file: PathBuf },
}

/// Result of scanning a VOC dataset's layout and split assignments.
pub(crate) struct VocScan {
    pub root: PathBuf,
    pub splits: BTreeMap<Split, Vec<VocItem>>,
    pub sources: BTreeSet<PathBuf>,
    pub report: LoadReport,
}

pub(crate) fn guarded_is_file(p: &Path) -> bool {
    crate::guard::check(p).is_ok() && p.is_file()
}

pub(crate) fn guarded_is_dir(p: &Path) -> bool {
    crate::guard::check(p).is_ok() && p.is_dir()
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
        if guarded_is_dir(&sub_dir) {
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
        if guarded_is_dir(&p) {
            return Some(p);
        }
    }
    None
}

fn find_imagesets_main_dir(dir: &Path) -> Option<PathBuf> {
    for rel in ["ImageSets/Main", "imagesets/main", "ImageSets/main", "imagesets/Main"] {
        let p = dir.join(rel);
        if guarded_is_dir(&p) {
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
        if !guarded_is_dir(&sub) {
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
    guarded_is_file(&p).then_some(p)
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
        if line.is_empty() {
            continue;
        }
        // Check exact XML stem first so image IDs containing spaces, leading `#`, or multiple dots (e.g.
        // `frame.jpg` from `frame.jpg.png`) match verbatim before comment/flag/extension stripping.
        if let Some(xml_path) = xml_index.get(line) {
            if seen.insert(line.to_string()) {
                out.push(VocItem::Xml(xml_path.clone()));
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        // Strip only a trailing VOC presence column (`-1`, `0`, `1`) if present, preserving spaces inside the ID.
        let candidate = strip_trailing_voc_flag(line);
        if let Some(xml_path) = xml_index.get(candidate) {
            if seen.insert(candidate.to_string()) {
                out.push(VocItem::Xml(xml_path.clone()));
            }
            continue;
        }
        let stripped = strip_known_ext(candidate);
        if let Some(xml_path) = xml_index.get(stripped) {
            if seen.insert(stripped.to_string()) {
                out.push(VocItem::Xml(xml_path.clone()));
            }
            continue;
        }
        // Validate stem before allowing any filesystem lookup (blocks `..`, `/abs`, and Windows drive prefixes).
        let Some(safe_stem) = safe_relative_subpath(stripped) else {
            report.push(
                IssueKind::Other,
                txt,
                format!("ignoring unsafe image ID {candidate:?} in ImageSets split file"),
            );
            continue;
        };
        if seen.insert(stripped.to_string()) {
            out.push(VocItem::StemOnly {
                stem: safe_stem,
                raw: stripped.to_string(),
                split_file: txt.to_path_buf(),
            });
        }
    }
    out
}

/// Strips a trailing whitespace-separated Pascal VOC presence label (`-1`, `0`, `1`), if present.
fn strip_trailing_voc_flag(line: &str) -> &str {
    if let Some((head, tail)) = line.rsplit_once(char::is_whitespace) {
        let tail = tail.trim();
        let head = head.trim_end();
        if !head.is_empty() && matches!(tail, "-1" | "0" | "1") {
            return head;
        }
    }
    line
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

/// Split-level image directories that actually exist on disk and pass the read guard, in priority order.
pub(crate) fn existing_split_image_dirs(root: &Path, split: Split) -> Vec<PathBuf> {
    let mut raw = vec![
        root.join("JPEGImages"),
        root.join("jpegimages"),
        root.join("images").join(split.as_str()),
        root.join("images"),
        root.join(split.as_str()),
    ];
    if split == Split::Val {
        raw.push(root.join("images").join("valid"));
        raw.push(root.join("valid"));
    }
    raw.push(root.to_path_buf());
    let mut dirs = Vec::new();
    for d in raw {
        if guarded_is_dir(&d) && !dirs.contains(&d) {
            dirs.push(d);
        }
    }
    dirs
}

/// Candidate directories for one XML annotation: XML-local folders, then pre-filtered split dirs, then `<folder>`.
fn dirs_for_xml(root: &Path, split_dirs: &[PathBuf], xml_path: &Path, folder: Option<&str>) -> Vec<PathBuf> {
    let mut dirs = Vec::with_capacity(split_dirs.len() + 4);
    let push_existing = |dirs: &mut Vec<PathBuf>, p: PathBuf| {
        if guarded_is_dir(&p) && !dirs.contains(&p) {
            dirs.push(p);
        }
    };
    let xml_dir = crate::paths::parent_dir(xml_path);
    push_existing(&mut dirs, xml_dir.to_path_buf());
    if xml_dir.file_name().is_some_and(|n| n.eq_ignore_ascii_case("annotations")) {
        let up = crate::paths::parent_dir(xml_dir);
        push_existing(&mut dirs, up.join("JPEGImages"));
        push_existing(&mut dirs, up.join("jpegimages"));
        push_existing(&mut dirs, up.join("images"));
        push_existing(&mut dirs, up.to_path_buf());
    }
    for d in split_dirs {
        if !dirs.contains(d) {
            dirs.push(d.clone());
        }
    }
    if let Some(f) = folder
        && let Some(safe) = safe_relative_subpath(f)
    {
        push_existing(&mut dirs, root.join(safe));
    }
    dirs
}

/// Validates that `raw` is a safe relative subpath made exclusively of normal path components (rejects `..`,
/// absolute roots, and Windows drive prefixes such as `C:foo.jpg`).
fn safe_relative_subpath(raw: &str) -> Option<PathBuf> {
    let normalized = raw.replace('\\', "/");
    let p = Path::new(&normalized);
    if p.is_absolute() || has_windows_drive_prefix(&normalized) {
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

/// Rejects Windows drive-letter prefixes (`C:...`) on every OS so untrusted paths are rejected portably.
fn has_windows_drive_prefix(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':'
}

/// Extracts a safe single normal filename component from `<filename>` (handles exports that embed paths like
/// `C:\path\img.jpg`, while rejecting bare Windows drive-relative strings like `C:secret.jpg`).
fn filename_basename(raw: &str) -> Option<&str> {
    let s = raw.rsplit(['/', '\\']).next().unwrap_or(raw).trim();
    if s.is_empty() || has_windows_drive_prefix(s) {
        return None;
    }
    let mut comps = Path::new(s).components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(_)), None) => Some(s),
        _ => None,
    }
}

/// Resolves the image file on disk corresponding to a VOC XML annotation.
pub(crate) fn resolve_voc_image(
    root: &Path,
    split_dirs: &[PathBuf],
    xml_path: &Path,
    filename: Option<&str>,
    folder: Option<&str>,
) -> PathBuf {
    let dirs = dirs_for_xml(root, split_dirs, xml_path, folder);
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
    // Try exact filename candidates first (every probe guarded).
    for dir in &dirs {
        for name in &names {
            let cand = dir.join(name);
            if guarded_is_file(&cand) {
                return cand;
            }
        }
    }
    // Try matching stem with known image extensions (handles extensionless `<filename>` or mismatched ext).
    let mut stems: Vec<PathBuf> = Vec::new();
    if let Some(raw) = filename
        && let Some(base) = filename_basename(raw)
        && let Some(st) = Path::new(base).file_stem().and_then(|s| s.to_str())
        && let Some(safe) = safe_relative_subpath(st)
    {
        stems.push(safe);
    }
    if let Some(st) = xml_path.file_stem().and_then(|s| s.to_str())
        && let Some(safe) = safe_relative_subpath(st)
        && !stems.contains(&safe)
    {
        stems.push(safe);
    }
    for dir in &dirs {
        for stem in &stems {
            if let Some(found) = find_image_with_stem(dir, stem) {
                return found;
            }
        }
    }
    // Fallback path reported in `MissingImage` issue (always safely inside the dataset).
    let fallback_name = names
        .first()
        .cloned()
        .or_else(|| {
            stems.first().map(|s| {
                let mut p = s.as_os_str().to_os_string();
                p.push(".jpg");
                PathBuf::from(p)
            })
        })
        .unwrap_or_else(|| PathBuf::from("image.jpg"));
    let fallback_dir = if guarded_is_dir(&root.join("JPEGImages")) {
        root.join("JPEGImages")
    } else {
        crate::paths::parent_dir(xml_path).to_path_buf()
    };
    fallback_dir.join(fallback_name)
}

pub(crate) fn resolve_stem_image(split_dirs: &[PathBuf], stem: &Path) -> Option<PathBuf> {
    for dir in split_dirs {
        if let Some(found) = find_image_with_stem(dir, stem) {
            return Some(found);
        }
    }
    None
}

fn find_image_with_stem(dir: &Path, stem: &Path) -> Option<PathBuf> {
    let base = dir.join(stem);
    for ext in IMAGE_EXTENSIONS {
        let mut p = base.as_os_str().to_os_string();
        p.push(".");
        p.push(ext);
        let cand = PathBuf::from(p);
        if guarded_is_file(&cand) {
            return Some(cand);
        }
        let mut up = base.as_os_str().to_os_string();
        up.push(".");
        up.push(ext.to_ascii_uppercase());
        let upper = PathBuf::from(up);
        if guarded_is_file(&upper) {
            return Some(upper);
        }
    }
    None
}
