//! Datasets registered from a folder on disk (YOLO layout). COCO is detected but not read yet.
//!
//! This is a stopgap until `folo-data` lands: it lists images, reads class names from `data.yaml` and parses YOLO
//! txt labels for stats and box overlays. Image sizes are read lazily (header only) when a page of images is listed.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use axum::http::StatusCode;
use folo_api::{BBox, Dataset, DatasetFormat, DatasetStats, ImageInfo, ImagePage, Split, Splits};

use crate::error::{Error, Result};
use crate::stats::StatsBuilder;
use crate::store::{ImageData, ImageQuery};

const IMAGE_EXTS: [&str; 5] = ["jpg", "jpeg", "png", "bmp", "webp"];

/// Most classes a dataset may have. Label lines with a larger class index count as broken.
pub const MAX_CLASSES: usize = 4096;

/// A YOLO label in normalised `cx, cy, w, h`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YoloBox {
    /// Class index.
    pub class: u32,
    /// Centre x in 0..1.
    pub cx: f32,
    /// Centre y in 0..1.
    pub cy: f32,
    /// Width in 0..1.
    pub w: f32,
    /// Height in 0..1.
    pub h: f32,
}

/// One image of a disk dataset.
#[derive(Debug)]
pub struct DiskImage {
    /// Path relative to the dataset root, with `/` separators.
    pub rel: String,
    /// Split.
    pub split: Split,
    /// Parsed labels.
    pub labels: Vec<YoloBox>,
    dims: OnceLock<Option<(u32, u32)>>,
}

/// A scanned dataset.
#[derive(Debug)]
pub struct DiskDataset {
    /// Metadata (without stats).
    pub meta: Dataset,
    /// Root folder.
    pub root: PathBuf,
    /// Images in split order, then path order. The image id is the index.
    pub images: Vec<DiskImage>,
    /// Statistics.
    pub stats: DatasetStats,
}

/// The fields of `data.yaml` we use.
#[derive(Debug, Default, PartialEq)]
pub struct DataYaml {
    /// `path:` (dataset root override).
    pub path: Option<String>,
    /// `train:` / `val:` / `test:` entries.
    pub splits: Vec<(Split, String)>,
    /// `names:` in index order.
    pub names: Vec<String>,
    /// `nc:`.
    pub nc: Option<usize>,
}

/// A YAML scalar: trimmed, with one pair of surrounding quotes removed and their escapes resolved (`\"` and `\\`
/// in double quotes, `''` in single quotes).
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].replace("\\\\", "\u{0}").replace("\\\"", "\"").replace('\u{0}', "\\")
    } else if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        s[1..s.len() - 1].replace("''", "'")
    } else {
        s.to_string()
    }
}

/// Removes a YAML comment: a `#` at the start of the line or after whitespace, outside quoted scalars. A quote only
/// opens a scalar at a token boundary (line start, or after `:`, `-`, `[`, `{`, `,` and whitespace), so the
/// apostrophe in `person's` is plain text; `\"` inside double quotes does not close them.
fn strip_comment(line: &str) -> &str {
    for (i, c) in unquoted_chars(line) {
        if c == '#' && line[..i].chars().next_back().is_none_or(char::is_whitespace) {
            return &line[..i];
        }
    }
    line
}

/// The characters of `s` (with byte offsets) that are outside quoted scalars. A quote opens a scalar only at a
/// token boundary (start, or after `:` `-` `[` `{` `,` and whitespace); inside double quotes `\x` is an escape,
/// inside single quotes `''` is a literal quote.
fn unquoted_chars(s: &str) -> Vec<(usize, char)> {
    let mut out = Vec::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut boundary = true;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if q == '"' && c == '\\' {
                escaped = true;
            } else if c == q {
                if q == '\'' && it.peek().is_some_and(|&(_, n)| n == '\'') {
                    it.next();
                } else {
                    quote = None;
                    boundary = false;
                }
            }
            continue;
        }
        if matches!(c, '"' | '\'') && boundary {
            quote = Some(c);
            continue;
        }
        out.push((i, c));
        boundary = matches!(c, ':' | '-' | '[' | '{' | ',') || (c.is_whitespace() && boundary);
    }
    out
}

/// Splits the inside of a flow collection (`body` starts right after `[` or `{`) at top-level commas, up to the
/// closing `close` bracket; quoted commas and brackets are part of their scalar.
fn split_flow(body: &str, close: char) -> Vec<&str> {
    let mut items = Vec::new();
    let mut start = 0;
    for (i, c) in unquoted_chars(body) {
        if c == ',' || c == close {
            items.push(&body[start..i]);
            start = i + 1;
            if c == close {
                return items.into_iter().filter(|s| !s.trim().is_empty()).collect();
            }
        }
    }
    items.push(&body[start..]);
    items.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// `{0: a, 1: b}` (the part after `{`) → index / name pairs.
fn flow_map(body: &str) -> Vec<(Option<usize>, String)> {
    split_flow(body, '}')
        .into_iter()
        .map(|kv| match kv.split_once(':') {
            Some((k, v)) => (k.trim().parse().ok(), unquote(v)),
            None => (None, unquote(kv)),
        })
        .collect()
}

/// Parses the subset of YAML that YOLO `data.yaml` files use: top-level `key: value` pairs, `names` as a flow list
/// (`[a, b]`), a block list (`- a`), or an index map (`0: a` lines or `{0: a, 1: b}`). Index maps must use each
/// index `0..n` exactly once; anything else is an error (it would silently shift class names).
pub fn parse_data_yaml(text: &str) -> Result<DataYaml, String> {
    let mut y = DataYaml::default();
    let mut in_names = false;
    let mut indexed: Vec<(Option<usize>, String)> = Vec::new();
    for raw in text.lines() {
        let line = strip_comment(raw).trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indented = line.len() != trimmed.len();
        if in_names && (indented || trimmed.starts_with("- ")) {
            if let Some(item) = trimmed.strip_prefix("- ") {
                y.names.push(unquote(item));
            } else if let Some((k, v)) = trimmed.split_once(':') {
                indexed.push((k.trim().parse().ok(), unquote(v)));
            }
            continue;
        }
        in_names = false;
        let Some((key, value)) = trimmed.split_once(':') else { continue };
        let value = value.trim();
        match key.trim() {
            "path" => y.path = Some(unquote(value)).filter(|s| !s.is_empty()),
            "train" => y.splits.push((Split::Train, unquote(value))),
            "val" => y.splits.push((Split::Val, unquote(value))),
            "test" if !unquote(value).is_empty() => y.splits.push((Split::Test, unquote(value))),
            "nc" => y.nc = value.parse().ok(),
            "names" => {
                if let Some(list) = value.strip_prefix('[') {
                    y.names = split_flow(list, ']').into_iter().map(unquote).collect();
                } else if let Some(map) = value.strip_prefix('{') {
                    indexed = flow_map(map);
                } else if value.is_empty() {
                    in_names = true;
                }
            }
            _ => {}
        }
    }
    if y.names.is_empty() && !indexed.is_empty() {
        let mut slots: Vec<Option<String>> = vec![None; indexed.len()];
        for (k, name) in indexed {
            let i = k.ok_or_else(|| "names: map keys must be class indices".to_string())?;
            let slot = slots
                .get_mut(i)
                .ok_or_else(|| format!("names: index {i} leaves a gap (indices must be 0..n, each once)"))?;
            if slot.replace(name).is_some() {
                return Err(format!("names: index {i} appears twice"));
            }
        }
        y.names = slots.into_iter().map(|s| s.expect("n keys in 0..n, no duplicates: all filled")).collect();
    }
    Ok(y)
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

fn walk_images(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk_images(&p, out);
        } else if is_image(&p) {
            out.push(p);
        }
    }
}

/// YOLO label path for an image: the last `images` path component becomes `labels`, the extension `.txt`.
pub fn label_path(img: &Path) -> PathBuf {
    let comps: Vec<_> = img.components().collect();
    let idx = comps.iter().rposition(|c| c.as_os_str() == "images");
    let mut p = PathBuf::new();
    for (i, c) in comps.iter().enumerate() {
        if Some(i) == idx {
            p.push("labels");
        } else {
            p.push(c.as_os_str());
        }
    }
    p.set_extension("txt");
    p
}

/// Parses YOLO label text. Returns the boxes and the number of broken lines. Segment polygons (`class x1 y1 x2 y2
/// ...`) are reduced to their bounding box.
pub fn parse_labels(text: &str, nc: usize) -> (Vec<YoloBox>, u64) {
    let mut boxes = Vec::new();
    let mut broken = 0;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let class = it.next().and_then(|c| c.parse::<u32>().ok());
        let vals: Option<Vec<f32>> = it.map(|v| v.parse::<f32>().ok()).collect();
        // Non-finite values are broken before any min / max (which would silently drop a NaN).
        let vals = vals.filter(|v| v.iter().all(|x| x.is_finite()));
        let (Some(class), Some(vals)) = (class, vals) else {
            broken += 1;
            continue;
        };
        let b = if vals.len() == 4 {
            Some(YoloBox { class, cx: vals[0], cy: vals[1], w: vals[2], h: vals[3] })
        } else if vals.len() >= 6 && vals.len() % 2 == 0 {
            let xs = vals.iter().step_by(2);
            let ys = vals.iter().skip(1).step_by(2);
            let (x1, x2) = xs.fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let (y1, y2) = ys.fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            Some(YoloBox { class, cx: (x1 + x2) / 2.0, cy: (y1 + y2) / 2.0, w: x2 - x1, h: y2 - y1 })
        } else {
            None
        };
        let in_range = |v: f32| v.is_finite() && (-0.01..=1.01).contains(&v);
        // The part of the box inside the image must have an area.
        let visible = |b: &YoloBox| {
            let (x1, x2) = ((b.cx - b.w / 2.0).max(0.0), (b.cx + b.w / 2.0).min(1.0));
            let (y1, y2) = ((b.cy - b.h / 2.0).max(0.0), (b.cy + b.h / 2.0).min(1.0));
            x2 > x1 && y2 > y1
        };
        match b {
            Some(b)
                if (b.class as usize) < nc
                    && in_range(b.cx)
                    && in_range(b.cy)
                    && in_range(b.w)
                    && in_range(b.h)
                    && b.w > 0.0
                    && b.h > 0.0
                    && visible(&b) =>
            {
                boxes.push(b)
            }
            _ => broken += 1,
        }
    }
    (boxes, broken)
}

/// Resolves a `data.yaml` split entry (a folder or a `.txt` list of images) to image paths.
fn split_images(base: &Path, entry: &str) -> Vec<PathBuf> {
    let p = base.join(entry);
    let mut out = Vec::new();
    if p.is_dir() {
        walk_images(&p, &mut out);
    } else if p.extension().is_some_and(|e| e == "txt")
        && let Ok(list) = std::fs::read_to_string(&p)
    {
        out.extend(
            list.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(|l| base.join(l))
                .filter(|q| is_image(q)),
        );
    }
    out
}

impl DiskDataset {
    /// Scans `path`. `id`, `name` and `created` become the metadata.
    pub fn scan(
        id: String,
        name: Option<String>,
        path: &str,
        format: DatasetFormat,
        created: String,
    ) -> Result<Self> {
        let root = PathBuf::from(path);
        if !root.is_dir() {
            return Err(Error::new(
                StatusCode::NOT_FOUND,
                "path_not_found",
                format!("'{path}' is not a folder"),
            )
            .hint("pass the dataset root: the folder holding data.yaml or images/ and labels/"));
        }
        let yaml_path =
            ["data.yaml", "dataset.yaml", "data.yml"].iter().map(|f| root.join(f)).find(|p| p.is_file());
        let looks_yolo = yaml_path.is_some() || root.join("images").is_dir() || root.join("labels").is_dir();
        let looks_coco = root.join("annotations").is_dir();
        let format = match format {
            DatasetFormat::Auto if looks_yolo => DatasetFormat::Yolo,
            DatasetFormat::Auto if looks_coco => DatasetFormat::Coco,
            DatasetFormat::Auto => {
                return Err(Error::new(
                    StatusCode::BAD_REQUEST,
                    "unknown_format",
                    "could not detect the format",
                )
                .hint("expected data.yaml or images/ + labels/ (YOLO), or annotations/*.json (COCO)"));
            }
            f => f,
        };
        if format == DatasetFormat::Coco {
            return Err(Error::not_implemented("reading COCO datasets"));
        }
        let yaml = match &yaml_path {
            Some(p) => {
                let text = std::fs::read_to_string(p).map_err(|e| Error::internal(e.to_string()))?;
                parse_data_yaml(&text).map_err(|e| {
                    Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", format!("{}: {e}", p.display()))
                })?
            }
            None => DataYaml::default(),
        };
        let base = match &yaml.path {
            Some(p) if Path::new(p).is_absolute() => PathBuf::from(p),
            Some(p) => root.join(p),
            None => root.clone(),
        };
        let mut by_split: Vec<(Split, Vec<PathBuf>)> = Vec::new();
        if !yaml.splits.is_empty() {
            for (split, entry) in &yaml.splits {
                by_split.push((*split, split_images(&base, entry)));
            }
        } else {
            for split in Split::ALL {
                let dir = root.join("images").join(split.as_str());
                if dir.is_dir() {
                    by_split.push((split, split_images(&dir, ".")));
                }
            }
            if by_split.is_empty() {
                by_split.push((Split::Train, split_images(&root.join("images"), ".")));
            }
        }

        if yaml.nc.is_some_and(|n| n > MAX_CLASSES) || yaml.names.len() > MAX_CLASSES {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_dataset",
                format!("data.yaml declares more than {MAX_CLASSES} classes"),
            ));
        }
        let mut images = Vec::new();
        let mut max_class = 0usize;
        let mut nc = if yaml.names.is_empty() { yaml.nc.unwrap_or(MAX_CLASSES) } else { yaml.names.len() };
        let mut builder_labels = Vec::new();
        let mut broken_total = 0;
        for (split, paths) in by_split {
            for p in paths {
                let (labels, broken) = match std::fs::read_to_string(label_path(&p)) {
                    Ok(t) => parse_labels(&t, nc),
                    Err(_) => (Vec::new(), 0),
                };
                broken_total += broken;
                max_class = labels.iter().map(|b| b.class as usize + 1).fold(max_class, usize::max);
                let rel = p.strip_prefix(&root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
                builder_labels.push(labels.iter().map(|b| (b.class, b.w, b.h)).collect::<Vec<_>>());
                images.push(DiskImage { rel, split, labels, dims: OnceLock::new() });
            }
        }
        let classes: Vec<String> = if yaml.names.is_empty() {
            nc = yaml.nc.unwrap_or(max_class);
            (0..nc).map(|i| format!("class{i}")).collect()
        } else {
            yaml.names.clone()
        };
        let mut sb = StatsBuilder::new(classes.len());
        for l in builder_labels {
            sb.add_image(l);
        }
        sb.add_broken(broken_total);
        let mut splits = Splits::default();
        for im in &images {
            match im.split {
                Split::Train => splits.train += 1,
                Split::Val => splits.val += 1,
                Split::Test => splits.test += 1,
            }
        }
        let name = name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| {
            root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "dataset".into())
        });
        let meta =
            Dataset { id, name, path: path.to_string(), format, classes, created, splits, stats: None };
        Ok(DiskDataset { meta, root, images, stats: sb.finish() })
    }

    fn image_at(&self, image_id: &str) -> Result<&DiskImage> {
        image_id
            .parse::<usize>()
            .ok()
            .and_then(|i| self.images.get(i))
            .ok_or_else(|| Error::not_found("image", image_id))
    }

    fn dims(&self, im: &DiskImage) -> Option<(u32, u32)> {
        *im.dims.get_or_init(|| image::image_dimensions(self.root.join(&im.rel)).ok())
    }

    /// A page of images.
    pub fn list_images(&self, q: &ImageQuery) -> ImagePage {
        let matches = |im: &DiskImage| {
            q.split.is_none_or(|s| s == im.split)
                && q.class.is_none_or(|c| im.labels.iter().any(|b| b.class == c))
        };
        let total = self.images.iter().filter(|im| matches(im)).count() as u64;
        let images = self
            .images
            .iter()
            .enumerate()
            .filter(|(_, im)| matches(im))
            .skip(q.offset as usize)
            .take(q.limit as usize)
            .map(|(i, im)| {
                let (w, h) = self.dims(im).unwrap_or((0, 0));
                let (wf, hf) = (w as f32, h as f32);
                let boxes = im
                    .labels
                    .iter()
                    .map(|b| BBox {
                        class: b.class,
                        x1: ((b.cx - b.w / 2.0) * wf).clamp(0.0, wf),
                        y1: ((b.cy - b.h / 2.0) * hf).clamp(0.0, hf),
                        x2: ((b.cx + b.w / 2.0) * wf).clamp(0.0, wf),
                        y2: ((b.cy + b.h / 2.0) * hf).clamp(0.0, hf),
                    })
                    .collect();
                ImageInfo {
                    id: i.to_string(),
                    file: im.rel.clone(),
                    split: im.split,
                    width: w,
                    height: h,
                    boxes,
                }
            })
            .collect();
        ImagePage { total, images }
    }

    /// Original bytes, or a JPEG thumbnail.
    pub fn image(&self, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        let im = self.image_at(image_id)?;
        let path = self.root.join(&im.rel);
        let bytes =
            std::fs::read(&path).map_err(|e| Error::internal(format!("reading {}: {e}", path.display())))?;
        match thumb {
            None => {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                let content_type = match ext.as_str() {
                    "png" => "image/png",
                    "bmp" => "image/bmp",
                    "webp" => "image/webp",
                    _ => "image/jpeg",
                };
                Ok(ImageData { content_type, bytes, immutable: false })
            }
            Some(side) => {
                let img = image::load_from_memory(&bytes).map_err(|e| {
                    Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
                })?;
                let rgb = if side >= img.width().max(img.height()) {
                    img.to_rgb8()
                } else {
                    img.thumbnail(side, side).to_rgb8()
                };
                let mut out = Vec::new();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85)
                    .encode(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
                    .map_err(|e| Error::internal(e.to_string()))?;
                Ok(ImageData { content_type: "image/jpeg", bytes: out, immutable: false })
            }
        }
    }

    /// Ground truth of an image in pixels plus its size (for mock predictions on disk images).
    pub fn truth(&self, image_id: &str) -> Result<(u32, u32, Vec<BBox>)> {
        self.image_at(image_id)?;
        let offset = image_id.parse().unwrap_or(0);
        let mut page = self.list_images(&ImageQuery { offset, limit: 1, ..Default::default() });
        let info = page.images.pop().ok_or_else(|| Error::not_found("image", image_id))?;
        Ok((info.width, info.height, info.boxes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_forms() {
        let p = |t: &str| parse_data_yaml(t).unwrap();
        let y = p("path: ../d\ntrain: images/train\nval: images/val\ntest:\nnames: ['a', \"b\"]\n");
        assert_eq!(y.path.as_deref(), Some("../d"));
        assert_eq!(y.splits.len(), 2);
        assert_eq!(y.names, vec!["a", "b"]);
        let y = p("names:\n  0: person\n  1: car # comment\nnc: 2\n");
        assert_eq!(y.names, vec!["person", "car"]);
        assert_eq!(y.nc, Some(2));
        let y = p("nc: 5000 # classes\nnames: # the classes\n  0: a#1 # first\n  1: 'b # c'\n");
        assert_eq!(y.nc, Some(5000));
        assert_eq!(y.names, vec!["a#1", "b # c"]);
        let y = p("# header\nnames: ['x', \"y#z\"] # trailing\n");
        assert_eq!(y.names, vec!["x", "y#z"]);
        let y = p("names:\n- body\n- Head\ntrain: x\n");
        assert_eq!(y.names, vec!["body", "Head"]);
        assert_eq!(y.splits, vec![(Split::Train, "x".to_string())]);
        // Apostrophes inside plain scalars, escapes inside quoted ones.
        let y = p("names:\n  0: person's # comment\n  1: \"a \\\" # b\"\n  2: 'it''s'\n");
        assert_eq!(y.names, vec!["person's", "a \" # b", "it's"]);
        // Flow index maps; gaps and duplicates are errors, never shifted names.
        assert_eq!(p("names: {0: cat, 1: dog}\n").names, vec!["cat", "dog"]);
        assert!(parse_data_yaml("names: {0: cat, 2: dog}\n").unwrap_err().contains("gap"));
        assert!(parse_data_yaml("names:\n  0: cat\n  2: dog\n").unwrap_err().contains("gap"));
        assert!(parse_data_yaml("names:\n  0: cat\n  0: dog\n").unwrap_err().contains("twice"));
        // Doubled single quotes stay inside the scalar, so a `#` after them is not a comment.
        assert_eq!(p("names:\n  0: 'it''s # fine' # comment\n").names, vec!["it's # fine"]);
        // Quoted commas and brackets in flow collections belong to their scalar.
        assert_eq!(p("names: {0: \"cat,dog\", 1: bird}\n").names, vec!["cat,dog", "bird"]);
        assert_eq!(p("names: {0: \"cat}dog\", 1: bird}\n").names, vec!["cat}dog", "bird"]);
        assert_eq!(p("names: ['a,b', \"c]d\", e] # x\n").names, vec!["a,b", "c]d", "e"]);
    }

    #[test]
    fn labels_and_paths() {
        let (b, broken) =
            parse_labels("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.3 0.1 0.3 0.3\n5 0.5 0.5 0.1 0.1\nx\n", 2);
        assert_eq!(b.len(), 2);
        assert_eq!(broken, 2);
        assert!((b[1].w - 0.2).abs() < 1e-6);
        assert_eq!(label_path(Path::new("d/images/train/a.jpg")), Path::new("d/labels/train/a.txt"));
        // Inside the parse tolerance but entirely outside the image on one edge: broken, never an inverted box.
        for line in [
            "0 -0.005 0.5 0.002 0.2",
            "0 1.005 0.5 0.002 0.2",
            "0 0.5 -0.005 0.2 0.002",
            "0 0.5 1.005 0.2 0.002",
        ] {
            assert_eq!(parse_labels(line, 1), (vec![], 1), "{line}");
        }
        // NaN / inf anywhere (boxes or polygons) is broken.
        for line in ["0 NaN 0.2 0.4 0.4 0.6 0.6", "0 0.5 0.5 inf 0.1", "0 0.2 0.2 0.4 0.4 0.6 NaN"] {
            assert_eq!(parse_labels(line, 1), (vec![], 1), "{line}");
        }
        assert_eq!(parse_labels("0 0.995 0.5 0.02 0.2", 1).0.len(), 1);
    }

    #[test]
    fn class_counts_are_bounded() {
        let dir = std::env::temp_dir().join(format!("folo-disk-bounds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("images/train")).unwrap();
        std::fs::create_dir_all(dir.join("labels/train")).unwrap();
        image::RgbImage::new(8, 8).save(dir.join("images/train/a.png")).unwrap();
        // No names / nc: a huge class index is a broken label, not a 4-billion-entry class list.
        std::fs::write(
            dir.join("labels/train/a.txt"),
            "4294967295 0.5 0.5 0.1 0.1
2 0.5 0.5 0.1 0.1
",
        )
        .unwrap();
        let path = dir.to_str().unwrap();
        let ds = DiskDataset::scan("d".into(), None, path, DatasetFormat::Auto, "t".into()).unwrap();
        assert_eq!(ds.meta.classes.len(), 3);
        assert_eq!(ds.stats.broken_labels, 1);
        std::fs::write(
            dir.join("data.yaml"),
            "nc: 1000000000
",
        )
        .unwrap();
        let e = DiskDataset::scan("d".into(), None, path, DatasetFormat::Auto, "t".into()).unwrap_err();
        assert_eq!(e.body.code, "invalid_dataset");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn scan_a_tiny_yolo_folder() {
        let dir = std::env::temp_dir().join(format!("folo-disk-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for s in ["train", "val"] {
            std::fs::create_dir_all(dir.join("images").join(s)).unwrap();
            std::fs::create_dir_all(dir.join("labels").join(s)).unwrap();
        }
        std::fs::write(dir.join("data.yaml"), "train: images/train\nval: images/val\nnames: [cat, dog]\n")
            .unwrap();
        let img = image::RgbImage::from_pixel(40, 20, image::Rgb([200, 10, 10]));
        img.save(dir.join("images/train/a.png")).unwrap();
        img.save(dir.join("images/val/b.png")).unwrap();
        std::fs::write(dir.join("labels/train/a.txt"), "1 0.5 0.5 0.5 0.5\n0 0.995 0.5 0.02 0.2\n").unwrap();
        let ds =
            DiskDataset::scan("ds_1".into(), None, dir.to_str().unwrap(), DatasetFormat::Auto, "t".into())
                .unwrap();
        assert_eq!(ds.meta.classes, vec!["cat", "dog"]);
        assert_eq!(ds.meta.splits, Splits { train: 1, val: 1, test: 0 });
        assert_eq!(ds.stats.class_counts, vec![1, 1]);
        assert_eq!(ds.stats.empty_images, 1);
        let page = ds.list_images(&ImageQuery { limit: 10, ..Default::default() });
        assert_eq!(page.total, 2);
        assert_eq!(page.images[0].width, 40);
        assert_eq!(page.images[0].boxes[0].x1, 10.0);
        // A box poking out of the right edge is clipped, never inverted.
        let edge = page.images[0].boxes[1];
        assert!(edge.x1 < edge.x2 && edge.x2 == 40.0, "{edge:?}");
        let th = ds.image("0", Some(10)).unwrap();
        assert_eq!(image::load_from_memory(&th.bytes).unwrap().width(), 10);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
