//! `folo val --coco` / `--save-json`: pycocotools-exact evaluation of a COCO split and COCO results files.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use folo_data::{CocoGroundTruth, CocoImage, FileDataset, Split};
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

/// Category id of each class in `classes`, or `None` for classes declared only in another split of the dataset.
///
/// Exact 1:1 matches (`gt.categories` names equal `classes` in order) map every class directly. Otherwise, a split
/// file is accepted only when `gt.categories` is a non-empty `(id, name)` subset of the COCO dataset's full
/// categories (`dataset_cats`, matching `classes` 1:1) **and** every annotation in `gt` names a category declared
/// by `gt.categories`; anything else is refused as `invalid_dataset`.
pub fn category_ids(
    gt: &CocoGroundTruth,
    classes: &[String],
    dataset_cats: Option<&[(u64, String)]>,
) -> Result<Vec<Option<u64>>, CliError> {
    let mismatch = || {
        CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{}: its categories do not match the dataset's classes", folo_api::json::path(&gt.path)),
        )
    };
    if gt.categories.len() == classes.len()
        && gt.categories.iter().zip(classes).all(|((_, name), c)| name == c)
    {
        return Ok(gt.categories.iter().map(|&(id, _)| Some(id)).collect());
    }
    if gt.categories.is_empty() || gt.categories.len() >= classes.len() {
        return Err(mismatch());
    }
    let Some(ds_cats) = dataset_cats else {
        return Err(mismatch());
    };
    if ds_cats.len() != classes.len() || !ds_cats.iter().zip(classes).all(|((_, name), c)| name == c) {
        return Err(mismatch());
    }
    let mut out = vec![None; classes.len()];
    let mut declared = HashSet::with_capacity(gt.categories.len());
    for (id, name) in &gt.categories {
        let Some(idx) = ds_cats.iter().position(|(did, dname)| did == id && dname == name) else {
            return Err(mismatch());
        };
        out[idx] = Some(*id);
        declared.insert(*id);
    }
    if gt.images.iter().flat_map(|im| &im.annotations).any(|a| !declared.contains(&a.category_id)) {
        return Err(mismatch());
    }
    Ok(out)
}

/// Union of `(id, name)` categories across every COCO annotation file of `data`, sorted by id (matching
/// `folo_data::open_splits`), streaming past `images` and `annotations`. Returns `None` if any file cannot be read
/// or two files assign different names to the same category id.
fn dataset_categories(data: &str) -> Option<Vec<(u64, String)>> {
    #[derive(serde::Deserialize)]
    struct Cat {
        id: u64,
        name: String,
    }
    #[derive(serde::Deserialize)]
    struct FileCats {
        #[serde(default)]
        categories: Vec<Cat>,
    }
    let jsons = folo_data::coco_annotation_files(data).ok()?;
    let mut all: BTreeMap<u64, String> = BTreeMap::new();
    for (_, path) in jsons {
        let file = std::fs::File::open(&path).ok()?;
        let parsed: FileCats =
            serde_json::from_reader(std::io::BufReader::with_capacity(1 << 20, file)).ok()?;
        for c in parsed.categories {
            if let Some(prev) = all.insert(c.id, c.name.clone())
                && prev != c.name
            {
                return None;
            }
        }
    }
    Some(all.into_iter().collect())
}

/// Resolve and validate `gt`'s category mapping against `classes` (reading the dataset's full COCO category table
/// only when `gt.categories` is a proper non-empty subset of `classes`).
pub fn resolve_categories(
    data: &str,
    gt: &CocoGroundTruth,
    classes: &[String],
) -> Result<Vec<Option<u64>>, CliError> {
    if gt.categories.len() == classes.len()
        && gt.categories.iter().zip(classes).all(|((_, name), c)| name == c)
    {
        return category_ids(gt, classes, None);
    }
    let ds_cats = if !gt.categories.is_empty() && gt.categories.len() < classes.len() {
        dataset_categories(data)
    } else {
        None
    };
    category_ids(gt, classes, ds_cats.as_deref())
}

/// Category id of a detection's class in this split, or `None` when the class is absent from this split.
/// Shared by [`evaluate`] and [`write_results`] so `coco.detections` and `--save-json` always apply the same filter.
fn category_of(cats: &[Option<u64>], class: u32) -> Option<u64> {
    cats.get(class as usize).copied().flatten()
}

/// pycocotools-exact evaluation: every GT image of the split (in id order), crowd regions, annotation areas.
pub fn evaluate(
    gt: &CocoGroundTruth,
    cats: &[Option<u64>],
    part: &FileDataset,
    detections: &[(usize, Vec<Detection>)],
) -> CocoResult {
    let class_of: HashMap<u64, u32> =
        cats.iter().enumerate().filter_map(|(i, c)| c.map(|id| (id, i as u32))).collect();
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
        let dets: Vec<Detection> =
            dets.iter().filter(|d| category_of(cats, d.class).is_some()).copied().collect();
        ev.add_image(im.id, &dets, &truths);
    }
    CocoResult { stats: ev.compute(), missing_images: missing }
}

/// Creates the folder of a `--save-json` file before the evaluation (as `predict --save` and `train --project` do)
/// and refuses three things: a folder that cannot be created, a folder in the file's place, and a path whose lookup
/// fails with an error other than "not found". The path itself is only looked up (`metadata`), never opened, so
/// nothing is tried about whether the file can be written: [`write_results`] reports that after the evaluation.
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
        // A lookup that fails for another reason than "not there" (a name too long, a loop of links).
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io(e)),
        // Something that is not a folder, or nothing yet (also a link to a file not there yet): the writer's to try.
        _ => Ok(()),
    }
}

/// Write `detections` as a COCO results file (`[{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]`),
/// the input of pycocotools' `COCO.loadRes`. Coordinates keep full f32 precision (`w = x2 - x1` in f64, so
/// `x + w` is `x2` again). Returns the number of detections written.
pub fn write_results(
    path: &Path,
    gt: &CocoGroundTruth,
    cats: &[Option<u64>],
    part: &FileDataset,
    detections: &[(usize, Vec<Detection>)],
) -> Result<usize, CliError> {
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
            let Some(cat) = category_of(cats, d.class) else { continue };
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

    /// `val --save-json` in a new folder works, and a path that can be seen to fail is refused before the evaluation.
    #[test]
    fn a_results_file_is_prepared_before_the_evaluation() {
        let dir = TestDir::new("coco-prepare");
        // A folder that does not exist yet is created, and nothing is written into it.
        let new = dir.join("eval").join("deep").join("dets.json");
        super::prepare_results(&new).unwrap();
        assert!(new.parent().unwrap().is_dir() && !new.exists());
        assert_eq!(std::fs::read_dir(new.parent().unwrap()).unwrap().count(), 0, "no file is created");
        // Nor into a folder that holds other files: they stay as they are, and no file is added.
        let folder = dir.join("held");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("other.tmp"), b"old").unwrap();
        super::prepare_results(&folder.join("dets.json")).unwrap();
        assert_eq!(std::fs::read(folder.join("other.tmp")).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 1, "no file is created");
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
        // A name the system refuses is refused now, not after the evaluation.
        #[cfg(unix)]
        {
            let long = dir.join("x".repeat(256));
            assert_eq!(super::prepare_results(&long).unwrap_err().code, "io_error");
            let looped = dir.join("eval").join("loop.json");
            std::os::unix::fs::symlink("loop.json", &looped).unwrap();
            assert_eq!(super::prepare_results(&looped).unwrap_err().code, "io_error");
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

    #[test]
    fn subset_categories_map_to_dataset_classes_and_reject_mismatches() {
        let classes: Vec<String> = ["circle", "square", "triangle"].into_iter().map(String::from).collect();
        let ds_cats: Vec<(u64, String)> =
            vec![(7, "circle".into()), (19, "square".into()), (42, "triangle".into())];
        let make = |cats: &[(u64, &str)], ann_cats: &[u64]| folo_data::CocoGroundTruth {
            path: std::path::PathBuf::from("instances_val.json"),
            images: vec![folo_data::CocoImage {
                id: 1,
                file_name: "a.png".into(),
                path: Some(std::path::PathBuf::from("a.png")),
                annotations: ann_cats
                    .iter()
                    .map(|&category_id| folo_data::CocoAnnotation {
                        category_id,
                        bbox: [1.0, 2.0, 10.0, 10.0],
                        area: 100.0,
                        iscrowd: false,
                    })
                    .collect(),
            }],
            categories: cats.iter().map(|&(id, n)| (id, n.to_string())).collect(),
        };
        // Exact 1:1 match.
        assert_eq!(
            super::category_ids(
                &make(&[(7, "circle"), (19, "square"), (42, "triangle")], &[7, 19]),
                &classes,
                None
            )
            .unwrap(),
            vec![Some(7), Some(19), Some(42)]
        );
        // Valid non-empty (id, name) subset with all annotations declared.
        let cats =
            super::category_ids(&make(&[(7, "circle"), (42, "triangle")], &[42]), &classes, Some(&ds_cats))
                .unwrap();
        assert_eq!(cats, vec![Some(7), None, Some(42)]);

        // Refused: empty categories, missing dataset table, wrong id for name, unknown category, undeclared annotation.
        for (bad, ds) in [
            (make(&[], &[]), Some(ds_cats.as_slice())),
            (make(&[(19, "square")], &[19]), None),
            (make(&[(99, "square")], &[99]), Some(ds_cats.as_slice())),
            (make(&[(7, "circle"), (20, "unknown")], &[7]), Some(ds_cats.as_slice())),
            (make(&[(19, "square")], &[7, 19]), Some(ds_cats.as_slice())),
            (make(&[(42, "triangle"), (19, "square"), (7, "circle")], &[7]), Some(ds_cats.as_slice())),
        ] {
            assert_eq!(super::category_ids(&bad, &classes, ds).unwrap_err().code, "invalid_dataset");
        }

        // Shared filter: detections of classes omitted from a subset split are dropped from both evaluate() and
        // write_results(), so coco.detections equals the number of rows written to --save-json.
        let dir = TestDir::new("coco-subset-filter");
        let img = dir.join("a.png");
        let gt = folo_data::CocoGroundTruth {
            path: dir.join("instances_val.json"),
            images: vec![folo_data::CocoImage {
                id: 1,
                file_name: "a.png".into(),
                path: Some(img.clone()),
                annotations: vec![folo_data::CocoAnnotation {
                    category_id: 19,
                    bbox: [10.0, 10.0, 20.0, 20.0],
                    area: 400.0,
                    iscrowd: false,
                }],
            }],
            categories: vec![(19, "square".into())],
        };
        let ds_two = vec![(7, "circle".into()), (19, "square".into())];
        let two_classes = vec!["circle".into(), "square".into()];
        let cats = super::category_ids(&gt, &two_classes, Some(&ds_two)).unwrap();
        assert_eq!(cats, vec![None, Some(19)]);
        let part = folo_data::FileDataset::new(
            folo_data::Split::Val,
            two_classes,
            vec![folo_data::ImageEntry { image: img, label: None, size: Some((64, 64)), boxes: Vec::new() }],
        );
        let dets = vec![(
            0usize,
            vec![
                folo_vision::Detection {
                    bbox: folo_vision::Xyxy { x1: 2.0, y1: 2.0, x2: 8.0, y2: 8.0 },
                    score: 0.9,
                    class: 0, // circle: absent from val split
                },
                folo_vision::Detection {
                    bbox: folo_vision::Xyxy { x1: 10.0, y1: 10.0, x2: 30.0, y2: 30.0 },
                    score: 0.8,
                    class: 1, // square: present in val split (category_id 19)
                },
            ],
        )];
        let res = super::evaluate(&gt, &cats, &part, &dets);
        let out_json = dir.join("dets.json");
        let written = super::write_results(&out_json, &gt, &cats, &part, &dets).unwrap();
        assert_eq!(res.stats.detections, 1);
        assert_eq!(written, 1);
        let rows: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(&out_json).unwrap()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["category_id"].as_u64(), Some(19));
    }
}
