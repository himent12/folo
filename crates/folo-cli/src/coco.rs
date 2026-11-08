//! `folo val --coco` / `--save-json`: pycocotools-exact evaluation of a COCO split and COCO results files.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use folo_data::{CocoGroundTruth, CocoImage, Dataset, FileDataset, Split};
use folo_vision::cocoeval::{CocoEval, CocoStats, CocoTruth};
use folo_vision::{Detection, Xyxy};

use crate::output::{CliError, ExitCode};

/// The ground truth of `split` of the COCO dataset at `data`.
pub fn ground_truth(data: &str, split: Split) -> Result<CocoGroundTruth, CliError> {
    let files = folo_data::coco_annotation_files(data).map_err(|e| {
        CliError::new(ExitCode::Usage, "usage", format!("--coco needs a COCO dataset: {e}"))
            .hint("pass the COCO folder (annotations/instances_<split>.json) or the instances JSON itself")
    })?;
    let mut of_split = files.into_iter().filter(|(s, _)| *s == split);
    let (Some((_, path)), None) = (of_split.next(), of_split.next()) else {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--coco needs exactly one COCO annotation file for the {} split", split.as_str()),
        )
        .hint("pass the instances JSON of that split as --data"));
    };
    folo_data::read_coco_ground_truth(&path)
        .map_err(|e| CliError::new(ExitCode::Runtime, "invalid_dataset", e.to_string()))
}

/// Matches dataset images to COCO images by `file_name` (the image path ends with it).
fn by_file_name<'a>(gt: &'a CocoGroundTruth, part: &FileDataset) -> HashMap<usize, &'a CocoImage> {
    let names: HashMap<&str, &CocoImage> = gt.images.iter().map(|im| (im.file_name.as_str(), im)).collect();
    let mut out = HashMap::new();
    for (i, e) in part.entries().iter().enumerate() {
        // Try the last 1, 2, 3 path components (file names may include folders).
        let comps: Vec<String> = e
            .image
            .components()
            .rev()
            .take(3)
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        for k in 1..=comps.len() {
            let key = comps[..k].iter().rev().cloned().collect::<Vec<_>>().join("/");
            if let Some(&im) = names.get(key.as_str()) {
                out.insert(i, im);
                break;
            }
        }
    }
    out
}

/// What [`evaluate`] returns.
pub struct CocoResult {
    pub stats: CocoStats,
    /// GT images the loader skipped (unreadable): evaluated with no detections, as pycocotools would.
    pub missing_images: usize,
}

/// Category id of each class (`classes[i]` must be the name of `categories[i]`, sorted by id).
pub fn category_ids(gt: &CocoGroundTruth, classes: &[String]) -> Result<Vec<u64>, CliError> {
    let names: Vec<&str> = gt.categories.iter().map(|c| c.1.as_str()).collect();
    if names != classes.iter().map(String::as_str).collect::<Vec<_>>() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{}: its categories do not match the dataset's classes", gt.path.display()),
        ));
    }
    Ok(gt.categories.iter().map(|c| c.0).collect())
}

/// pycocotools-exact evaluation: every GT image of the split (in id order), crowd regions, annotation areas.
pub fn evaluate(
    gt: &CocoGroundTruth,
    part: &FileDataset,
    detections: &[(usize, Vec<Detection>)],
) -> Result<CocoResult, CliError> {
    let cats = category_ids(gt, part.class_names())?;
    let class_of: HashMap<u64, u32> = cats.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
    let images = by_file_name(gt, part);
    let mut dets_of: HashMap<u64, &[Detection]> = HashMap::new();
    for (index, d) in detections {
        if let Some(im) = images.get(index) {
            dets_of.entry(im.id).or_insert(d.as_slice());
        }
    }
    let mut ev = CocoEval::new(cats.len());
    let mut missing = 0;
    let mut truths = Vec::new();
    for im in &gt.images {
        truths.clear();
        for a in &im.annotations {
            let Some(&class) = class_of.get(&a.category_id) else { continue };
            let [x, y, w, h] = a.bbox;
            truths.push(CocoTruth {
                bbox: Xyxy::new(x as f32, y as f32, (x + w) as f32, (y + h) as f32),
                class,
                area: a.area,
                iscrowd: a.iscrowd,
            });
        }
        let dets = dets_of.get(&im.id).copied().unwrap_or_else(|| {
            missing += 1;
            &[]
        });
        let dets: Vec<Detection> = dets.iter().filter(|d| (d.class as usize) < cats.len()).copied().collect();
        ev.add_image(im.id, &dets, &truths);
    }
    Ok(CocoResult { stats: ev.compute(), missing_images: missing })
}

/// Write `detections` as a COCO results file (`[{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]`),
/// the input of pycocotools' `COCO.loadRes`. Coordinates keep full f32 precision (`w = x2 - x1` in f64, so
/// `x + w` is `x2` again). Returns the number of detections written.
pub fn write_results(
    path: &Path,
    gt: &CocoGroundTruth,
    part: &FileDataset,
    detections: &[(usize, Vec<Detection>)],
) -> Result<usize, CliError> {
    let cats = category_ids(gt, part.class_names())?;
    let images = by_file_name(gt, part);
    let io = |e: std::io::Error| {
        CliError::new(ExitCode::Runtime, "io_error", format!("cannot write {}: {e}", path.display()))
    };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
    let mut n = 0usize;
    let mut seen = std::collections::HashSet::new();
    f.write_all(b"[").map_err(io)?;
    for (index, dets) in detections {
        let Some(im) = images.get(index) else { continue };
        if !seen.insert(im.id) {
            continue;
        }
        for d in dets {
            let Some(&cat) = cats.get(d.class as usize) else { continue };
            let (x1, y1) = (d.bbox.x1 as f64, d.bbox.y1 as f64);
            let (w, h) = (d.bbox.x2 as f64 - x1, d.bbox.y2 as f64 - y1);
            let sep = if n == 0 { "" } else { "," };
            write!(
                f,
                "{sep}\n{{\"image_id\":{},\"category_id\":{cat},\"bbox\":[{x1},{y1},{w},{h}],\"score\":{}}}",
                im.id, d.score as f64
            )
            .map_err(io)?;
            n += 1;
        }
    }
    f.write_all(b"\n]\n").map_err(io)?;
    f.flush().map_err(io)?;
    Ok(n)
}
