//! Pascal VOC XML detection datasets (`Annotations/*.xml`, `JPEGImages/` or colocated images, optional
//! `ImageSets/Main/{train,val,trainval,test}.txt`).

mod resolve;
pub(crate) mod xml;

pub(crate) use resolve::looks_like_voc_dir;

use super::labels::strip_bom;
use super::{
    Candidate, FileDataset, Format, IssueKind, LabelSource, LoadReport, LoadedDataset, OpenOptions, Split,
    build_entries,
};
use crate::error::{DataError, Result};
use rayon::prelude::*;
use resolve::{
    VocItem, VocScan, existing_split_image_dirs, guarded_is_file, resolve_stem_image, resolve_voc_image,
    scan_voc,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Pixel coordinate convention for Pascal VOC `<bndbox>` elements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VocCoordMode {
    /// Classic Pascal VOC 1-based inclusive pixel coordinates (`[1..=width]`, width `= xmax - xmin + 1`),
    /// mapped to 0-based pixel edges `[(xmin - 1).max(0), (ymin - 1).max(0), xmax, ymax]` (VOC2007/VOC2012 devkit
    /// standard). A full-image box `(1, 1, width, height)` covers `[0, width] x [0, height]` and a 1-pixel box
    /// `(x, y, x, y)` covers `[x - 1, x] x [y - 1, y]`.
    #[default]
    OneBased,
    /// Zero-based continuous pixel edges (`[0..width]`, width `= xmax - xmin`), used by some modern exporters.
    ZeroBased,
}

/// Converts raw VOC `<bndbox>` corners `[xmin, ymin, xmax, ymax]` to 0-based continuous pixel edges
/// `[x1, y1, x2, y2]` according to `mode`.
pub(crate) fn voc_box_to_xyxy(bbox: [f32; 4], mode: VocCoordMode) -> [f32; 4] {
    let [xmin, ymin, xmax, ymax] = bbox;
    match mode {
        VocCoordMode::OneBased => [(xmin - 1.0).max(0.0), (ymin - 1.0).max(0.0), xmax, ymax],
        VocCoordMode::ZeroBased => [xmin, ymin, xmax, ymax],
    }
}

/// Options for [`open_voc_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocOptions {
    /// Whether `<difficult>1</difficult>` objects are kept (`true` by default).
    pub include_difficult: bool,
    /// Read every image header while opening so corrupt files are skipped up front and sizes are verified.
    pub verify_images: bool,
    /// Coordinate convention for `<bndbox>` elements (default [`VocCoordMode::OneBased`]).
    pub coord_mode: VocCoordMode,
}

impl Default for VocOptions {
    fn default() -> Self {
        Self { include_difficult: true, verify_images: true, coord_mode: VocCoordMode::OneBased }
    }
}

/// One `<object>` parsed from a Pascal VOC XML annotation file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VocObject {
    /// Decoded class name from `<name>`.
    pub name: String,
    /// Raw pixel corners `[xmin, ymin, xmax, ymax]` as written in `<bndbox>`.
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

/// Opens a Pascal VOC dataset with VOC-specific options such as [`VocOptions::include_difficult`] and
/// [`VocOptions::coord_mode`].
pub fn open_voc_with(path: impl AsRef<Path>, opts: &VocOptions) -> Result<LoadedDataset> {
    let path = path.as_ref();
    let super::Source::Voc { root, single_xml } = super::detect(path, Format::Voc)? else {
        return Err(DataError::Format(format!("{} is not a Pascal VOC dataset", crate::paths::shown(path))));
    };
    let open_opts = OpenOptions { format: Format::Voc, verify_images: opts.verify_images };
    open_voc_inner(
        &root,
        single_xml.as_deref(),
        &open_opts,
        &Split::ALL,
        opts.include_difficult,
        opts.coord_mode,
    )
}

/// Opens a Pascal VOC dataset from `root` (or `single_xml`), reading only `wanted` splits' images.
pub(crate) fn open_voc(
    root: &Path,
    single_xml: Option<&Path>,
    opts: &OpenOptions,
    wanted: &[Split],
) -> Result<LoadedDataset> {
    open_voc_inner(root, single_xml, opts, wanted, true, VocCoordMode::default())
}

struct ResolvedItem {
    candidate: Option<Candidate>,
    issue: Option<(IssueKind, PathBuf, String, bool)>,
    sources: Vec<PathBuf>,
}

enum ParsedItem {
    Valid { xml_path: PathBuf, ann: VocAnnotation },
    Broken { xml_path: PathBuf, message: String },
    StemOnly { exact_stem: PathBuf, stripped_stem: Option<PathBuf>, raw: String, split_file: PathBuf },
}

fn open_voc_inner(
    root: &Path,
    single_xml: Option<&Path>,
    opts: &OpenOptions,
    wanted: &[Split],
    include_difficult: bool,
    coord_mode: VocCoordMode,
) -> Result<LoadedDataset> {
    let VocScan { root, splits, mut sources, mut report } = scan_voc(root, single_xml);

    // Optional `classes.txt` gives a pinned class order; any extra names found in XMLs are appended sorted.
    let mut classes = read_classes_txt(&root, &mut report, &mut sources);
    let mut class_set: HashSet<String> = classes.iter().cloned().collect();
    let mut discovered_names: BTreeSet<String> = BTreeSet::new();

    // Parse XMLs across all splits in parallel (order preserved) so `classes` is identical regardless of `wanted`.
    let mut parsed_splits: BTreeMap<Split, Vec<ParsedItem>> = BTreeMap::new();
    for (split, items) in splits {
        let is_wanted = wanted.contains(&split);
        let parsed_all: Vec<Option<ParsedItem>> = items
            .into_par_iter()
            .map(|item| match item {
                VocItem::Xml(xml_path) => match read_and_parse_xml(&xml_path) {
                    Ok(ann) => Some(ParsedItem::Valid { xml_path, ann }),
                    Err(message) => {
                        if is_wanted {
                            Some(ParsedItem::Broken { xml_path, message })
                        } else {
                            None
                        }
                    }
                },
                VocItem::StemOnly { exact_stem, stripped_stem, raw, split_file } => {
                    if is_wanted {
                        Some(ParsedItem::StemOnly { exact_stem, stripped_stem, raw, split_file })
                    } else {
                        None
                    }
                }
            })
            .collect();

        let mut kept = Vec::with_capacity(if is_wanted { parsed_all.len() } else { 0 });
        for item in parsed_all.into_iter().flatten() {
            if let ParsedItem::Valid { ref ann, .. } = item {
                for obj in &ann.objects {
                    if !class_set.contains(&obj.name) {
                        discovered_names.insert(obj.name.clone());
                    }
                }
            }
            if is_wanted {
                kept.push(item);
            }
        }
        if is_wanted {
            parsed_splits.insert(split, kept);
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
        // Pre-filter existing split-level directories once so per-image resolution avoids repeated misses.
        let split_dirs = existing_split_image_dirs(&root, split);
        let resolved: Vec<ResolvedItem> = items
            .into_par_iter()
            .map(|item| match item {
                ParsedItem::Valid { xml_path, ann } => {
                    let image = resolve_voc_image(
                        &root,
                        &split_dirs,
                        &xml_path,
                        ann.filename.as_deref(),
                        ann.folder.as_deref(),
                    );
                    let item_sources = vec![
                        crate::paths::parent_dir(&image).to_path_buf(),
                        crate::paths::parent_dir(&xml_path).to_path_buf(),
                    ];
                    let boxes: Vec<(u32, [f32; 4])> = ann
                        .objects
                        .into_iter()
                        .filter(|o| include_difficult || !o.difficult)
                        .filter_map(|o| {
                            class_map.get(o.name.as_str()).map(|&c| (c, voc_box_to_xyxy(o.bbox, coord_mode)))
                        })
                        .collect();
                    ResolvedItem {
                        candidate: Some(Candidate {
                            image,
                            labels: LabelSource::Voc {
                                boxes,
                                declared: ann.size,
                                broken: None,
                                label: Some(xml_path),
                            },
                        }),
                        issue: None,
                        sources: item_sources,
                    }
                }
                ParsedItem::Broken { xml_path, message } => {
                    let image = resolve_voc_image(&root, &split_dirs, &xml_path, None, None);
                    let item_sources = vec![crate::paths::parent_dir(&xml_path).to_path_buf()];
                    if guarded_is_file(&image) {
                        ResolvedItem {
                            candidate: Some(Candidate {
                                image,
                                labels: LabelSource::Voc {
                                    boxes: Vec::new(),
                                    declared: None,
                                    broken: Some(message),
                                    label: Some(xml_path),
                                },
                            }),
                            issue: None,
                            sources: item_sources,
                        }
                    } else {
                        ResolvedItem {
                            candidate: None,
                            issue: Some((IssueKind::BrokenLabel, xml_path, message, true)),
                            sources: item_sources,
                        }
                    }
                }
                ParsedItem::StemOnly { exact_stem, stripped_stem, raw, split_file } => {
                    if let Some(image) =
                        resolve_stem_image(&split_dirs, &exact_stem, stripped_stem.as_deref())
                    {
                        let item_sources = vec![crate::paths::parent_dir(&image).to_path_buf()];
                        ResolvedItem {
                            candidate: Some(Candidate {
                                image,
                                labels: LabelSource::Voc {
                                    boxes: Vec::new(),
                                    declared: None,
                                    broken: None,
                                    label: None,
                                },
                            }),
                            issue: None,
                            sources: item_sources,
                        }
                    } else {
                        ResolvedItem {
                            candidate: None,
                            issue: Some((
                                IssueKind::MissingImage,
                                split_file,
                                format!("no annotation XML or image found for {raw:?}"),
                                false,
                            )),
                            sources: Vec::new(),
                        }
                    }
                }
            })
            .collect();

        let mut candidates = Vec::with_capacity(resolved.len());
        for item in resolved {
            sources.extend(item.sources);
            if let Some(c) = item.candidate {
                candidates.push(c);
            }
            if let Some((kind, path, msg, skipped_label)) = item.issue {
                report.push(kind, path, msg);
                if skipped_label {
                    report.record_skipped_label_image();
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

/// Formats a [`LoadedDataset`] image's boxes as a Pascal VOC `<annotation>` XML document using the requested
/// [`VocCoordMode`].
///
/// Under [`VocCoordMode::OneBased`] (the VOC2007/VOC2012 devkit and Detectron2 `int()` standard), continuous
/// 0-based edges `[x1, x2]` with width `>= 1` px round to `xmin = round(x1) + 1`, `xmax = round(x2)` clamped
/// to `[1, width]` with `xmax >= xmin`, while sub-pixel boxes (`x2 - x1 < 1` px) map to the 1-pixel inclusive
/// box `xmin = xmax = floor(midpoint) + 1` clamped to `[1, width]`. Under [`VocCoordMode::ZeroBased`], continuous
/// 0-based edges `[x1, y1, x2, y2]` are clamped to `[0, width]` / `[0, height]` with `xmax >= xmin` and
/// `ymax >= ymin`.
pub(crate) fn format_voc_xml_with(
    filename: &str,
    folder: &str,
    width: u32,
    height: u32,
    classes: &[String],
    boxes: &[crate::BBox],
    coord_mode: VocCoordMode,
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
    let (max_w, max_h) = (i64::from(width.max(1)), i64::from(height.max(1)));
    let (wf, hf) = (f64::from(width), f64::from(height));
    for b in boxes {
        let class_name = classes.get(b.class as usize).map(String::as_str).unwrap_or("unknown");
        let [x1, y1, x2, y2] = b.pixel_xyxy(width, height);
        out.push_str("  <object>\n");
        let _ = writeln!(out, "    <name>{}</name>", xml::escape_text(class_name));
        out.push_str("    <pose>Unspecified</pose>\n");
        out.push_str("    <truncated>0</truncated>\n");
        out.push_str("    <difficult>0</difficult>\n");
        out.push_str("    <bndbox>\n");
        match coord_mode {
            VocCoordMode::OneBased => {
                let (xmin, xmax) = to_one_based_int_span(x1, x2, max_w);
                let (ymin, ymax) = to_one_based_int_span(y1, y2, max_h);
                let _ = writeln!(out, "      <xmin>{xmin}</xmin>");
                let _ = writeln!(out, "      <ymin>{ymin}</ymin>");
                let _ = writeln!(out, "      <xmax>{xmax}</xmax>");
                let _ = writeln!(out, "      <ymax>{ymax}</ymax>");
            }
            VocCoordMode::ZeroBased => {
                let xmin = normalize_zero(f64::from(x1).clamp(0.0, wf));
                let ymin = normalize_zero(f64::from(y1).clamp(0.0, hf));
                let xmax = normalize_zero(f64::from(x2).clamp(xmin, wf));
                let ymax = normalize_zero(f64::from(y2).clamp(ymin, hf));
                let _ = writeln!(out, "      <xmin>{xmin}</xmin>");
                let _ = writeln!(out, "      <ymin>{ymin}</ymin>");
                let _ = writeln!(out, "      <xmax>{xmax}</xmax>");
                let _ = writeln!(out, "      <ymax>{ymax}</ymax>");
            }
        }
        out.push_str("    </bndbox>\n");
        out.push_str("  </object>\n");
    }
    out.push_str("</annotation>\n");
    out
}

#[inline]
fn normalize_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

fn to_one_based_int_span(lo: f32, hi: f32, max_dim: i64) -> (i64, i64) {
    let (lo64, hi64) = (f64::from(lo), f64::from(hi));
    if hi64 - lo64 >= 1.0 {
        let min_i = (lo64.round() as i64).saturating_add(1).clamp(1, max_dim);
        let max_i = (hi64.round() as i64).clamp(min_i, max_dim);
        (min_i, max_i)
    } else {
        let mid = lo64 + 0.5 * (hi64 - lo64);
        let p = (mid.floor() as i64).saturating_add(1).clamp(1, max_dim);
        (p, p)
    }
}
