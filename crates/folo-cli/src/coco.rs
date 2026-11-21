//! `folo val --coco` / `--save-json`: pycocotools-exact evaluation of a COCO split and COCO results files.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use folo_data::{CocoGroundTruth, CocoImage, Dataset, FileDataset, Split};
use folo_vision::Detection;
use folo_vision::cocoeval::{CocoEval, CocoStats, CocoTruth};

use crate::output::{CliError, ExitCode};

/// The ground truth of `split` of the COCO dataset at `data`, image paths resolved like the loader's.
pub fn ground_truth(data: &str, split: Split) -> Result<CocoGroundTruth, CliError> {
    folo_data::open_coco_ground_truth(data, split).map_err(|e| match e {
        folo_data::DataError::Format(m) => CliError::new(ExitCode::Usage, "usage", format!("--coco: {m}")).hint(
            folo_api::json::cli_command(
                "--coco and --save-json need COCO annotations: pass a COCO folder (annotations/instances_<split>.json) \
                 or the split's instances JSON; `folo dataset convert {data} <out> --to coco` writes one from a YOLO \
                 dataset",
                &[("data", Path::new(data))],
            ),
        ),
        e => CliError::new(ExitCode::Runtime, "invalid_dataset", e.to_string()),
    })
}

/// Dataset image index -> COCO image, by the exact path both resolve the image to.
fn by_path<'a>(gt: &'a CocoGroundTruth, part: &FileDataset) -> HashMap<usize, &'a CocoImage> {
    let paths: HashMap<&Path, &CocoImage> =
        gt.images.iter().filter_map(|im| im.path.as_deref().map(|p| (p, im))).collect();
    part.entries()
        .iter()
        .enumerate()
        .filter_map(|(i, e)| paths.get(e.image.as_path()).map(|&im| (i, im)))
        .collect()
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
            format!("{}: its categories do not match the dataset's classes", folo_api::json::path(&gt.path)),
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
    let images = by_path(gt, part);
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
            truths.push(CocoTruth {
                // f64 as in the file: rounding to f32 could move an IoU across a threshold.
                xywh: a.bbox,
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

/// Refuses, before the evaluation, a `--save-json` file that plainly cannot be written, and creates its folder (as
/// `predict --save` and `train --project` do). Best effort: the final name is never created or removed here, so an
/// existing file keeps its content and a link keeps its target, and a new file is only tried under a temporary name
/// in the same folder. [`write_results`] can still fail (a full disk, a folder that changed meanwhile).
pub fn prepare_results(path: &Path) -> Result<(), CliError> {
    let io = |e: std::io::Error| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot write {}: {e}", folo_api::json::path(path)),
        )
    };
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(io)?;
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => Err(io(std::io::ErrorKind::IsADirectory.into())),
        // Not truncated: an earlier results file stays until the new one is written.
        Ok(_) => std::fs::OpenOptions::new().write(true).open(path).map(drop).map_err(io),
        // A link to a file not there yet: the writer creates its target.
        Err(_) if std::fs::symlink_metadata(path).is_ok() => Ok(()),
        Err(_) => {
            let tmp = folo_nn::io::temp_path(dir).map_err(io)?;
            let _ = std::fs::remove_file(&tmp);
            Ok(())
        }
    }
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
    let images = by_path(gt, part);
    let io = |e: std::io::Error| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot write {}: {e}", folo_api::json::path(path)),
        )
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

#[cfg(test)]
mod tests {
    use folo_data::ImageFormat;
    use folo_data::synthetic::{SyntheticConfig, write_yolo};
    use folo_test_support::TestDir;

    /// `--coco` / `--save-json` on a YOLO dataset: one readable hint (it had two runs of 18 spaces in it).
    #[test]
    fn yolo_dataset_hint_reads_as_one_sentence() {
        let dir = TestDir::new("coco-hint");
        let cfg = SyntheticConfig { images: 2, imgsz: 64, ..SyntheticConfig::default() };
        write_yolo(&cfg, &dir.join("shapes"), ImageFormat::Png).unwrap();
        let data = dir.join("shapes");
        let e = super::ground_truth(data.to_str().unwrap(), folo_data::Split::Val).unwrap_err();
        assert_eq!(e.code, "usage");
        let hint = e.hint.unwrap();
        assert!(!hint.contains("  "), "{hint}");
    }

    /// `val --save-json` in a new folder works, and one that cannot be written fails before the evaluation.
    #[test]
    fn a_results_file_is_prepared_before_the_evaluation() {
        let dir = TestDir::new("coco-prepare");
        // A folder that does not exist yet is created, and no empty results file stays in it.
        let new = dir.join("eval").join("deep").join("dets.json");
        super::prepare_results(&new).unwrap();
        assert!(new.parent().unwrap().is_dir() && !new.exists());
        assert_eq!(std::fs::read_dir(new.parent().unwrap()).unwrap().count(), 0, "no temporary file stays");
        // A link to a file not written yet is left for the writer, which creates its target.
        #[cfg(unix)]
        {
            let link = dir.join("eval").join("latest.json");
            std::os::unix::fs::symlink("run-2.json", &link).unwrap();
            super::prepare_results(&link).unwrap();
            assert!(
                std::fs::symlink_metadata(&link).is_ok() && !dir.join("eval").join("run-2.json").exists()
            );
        }
        // An earlier results file keeps its content until the new one is written.
        std::fs::write(&new, b"[]").unwrap();
        super::prepare_results(&new).unwrap();
        assert_eq!(std::fs::read(&new).unwrap(), b"[]");
        // A folder in the file's place, and a file in the folder's place, are refused now.
        let e = super::prepare_results(new.parent().unwrap()).unwrap_err();
        assert_eq!(e.code, "io_error");
        let e = super::prepare_results(&new.join("dets.json")).unwrap_err();
        assert!(e.code == "io_error" && e.message.starts_with("cannot write "), "{}", e.message);
    }
}
