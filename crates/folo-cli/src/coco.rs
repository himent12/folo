//! `folo val --coco` / `--save-json`: pycocotools-exact evaluation of a COCO split and COCO results files.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use folo_data::{CocoGroundTruth, CocoImage, FileDataset, Issue, IssueKind, Split};
use folo_vision::Detection;
use folo_vision::cocoeval::{CocoEval, CocoStats, CocoTruth};
use serde::de::{self, Deserialize, Deserializer, IgnoredAny, MapAccess, Visitor};

use crate::output::{CliError, ExitCode};

/// The ground truth of `split` of the COCO dataset at `data`, image paths resolved like the loader's.
pub fn ground_truth(data: &str, split: Split) -> Result<CocoGroundTruth, CliError> {
    folo_data::open_coco_ground_truth(data, split).map_err(|e| match e {
        folo_data::DataError::Format(m)
            if m.contains("not a COCO instances file")
                || m.contains("duplicate image id")
                || m.contains("annotation for unknown image id") =>
        {
            CliError::new(ExitCode::Runtime, "invalid_dataset", format!("--coco: {m}")).hint(
                folo_api::json::cli_command(
                    "run `folo dataset check {data}` to inspect the COCO annotations",
                    &[("data", Path::new(data))],
                ),
            )
        }
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

/// Validated bidirectional mapping between model / dataset class indices (`0..num_classes`) and the COCO
/// `category_id`s declared in the evaluated split's ground-truth JSON.
///
/// When a split JSON declares a strict subset of the dataset's categories (e.g. only classes present in `val`),
/// classes absent from the split map to `None` in `class_to_cat`, while present classes keep their dataset-wide
/// index so detections are never written or scored under another class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CategoryMap {
    /// `class_to_cat[class_idx]` is `Some(category_id)` when that class is declared in `gt.categories`.
    class_to_cat: Vec<Option<u64>>,
    /// `cat_to_class[category_id]` is the dataset/model class index.
    cat_to_class: HashMap<u64, u32>,
}

impl CategoryMap {
    /// Total number of classes in the dataset (and model).
    pub fn num_classes(&self) -> usize {
        self.class_to_cat.len()
    }

    /// COCO `category_id` for a model/dataset class index, or `None` if that class is not in this split's JSON.
    pub fn category_of(&self, class: u32) -> Option<u64> {
        self.class_to_cat.get(class as usize).copied().flatten()
    }

    /// Model/dataset class index for a COCO `category_id` declared in this split's JSON.
    pub fn class_of(&self, category_id: u64) -> Option<u32> {
        self.cat_to_class.get(&category_id).copied()
    }
}

/// Validates COCO ground-truth categories, annotations, and model/dataset class alignment *before* inference (#422).
///
/// Refuses:
/// - empty `categories` or empty `images` in the split's COCO JSON,
/// - any annotation referencing a `category_id` not declared in `gt.categories`,
/// - a model whose class list differs from the dataset's full class order (`model_classes != dataset_classes`),
/// - a split whose `categories` conflict with or fall outside the dataset's merged category table.
///
/// Accepts a split whose `categories` are either the full dataset class list or a non-empty strict subset of the
/// dataset's `(id, name)` category table.
pub fn validate_categories(
    data: &str,
    gt: &CocoGroundTruth,
    dataset_classes: &[String],
    dataset_issues: &[Issue],
    model_classes: &[String],
) -> Result<CategoryMap, CliError> {
    let gt_path = folo_api::json::path(&gt.path);
    // Refuse conflicting category names across splits even when `gt.categories` happens to match the winning
    // `dataset_classes` (fast path below would otherwise skip inspecting the other split files).
    if let Some(conflict) = dataset_issues.iter().find(|i| {
        i.kind == IssueKind::Other
            && i.message.starts_with("category ")
            && i.message.contains(" is named both ")
    }) {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{}: {}", folo_api::json::path(&conflict.path), conflict.message),
        )
        .hint("make category id -> name mappings consistent across every COCO split JSON"));
    }
    if gt.images.is_empty() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{gt_path}: COCO ground truth has no images"),
        ));
    }
    if gt.categories.is_empty() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{gt_path}: COCO ground truth declares no categories"),
        )
        .hint("add a non-empty `categories` array to the COCO annotation file"));
    }
    if dataset_classes.is_empty() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{gt_path}: dataset declares no classes"),
        ));
    }
    if model_classes != dataset_classes {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!(
                "model classes {model_classes:?} do not match the dataset's classes {dataset_classes:?}; \
                 --coco and --save-json require the model's classes to match the dataset's full class order"
            ),
        )
        .hint(
            "evaluate with a model trained on the same class list and order, or omit --coco / --save-json",
        ));
    }

    let gt_cat_ids: HashSet<u64> = gt.categories.iter().map(|(id, _)| *id).collect();
    for im in &gt.images {
        for ann in &im.annotations {
            if !gt_cat_ids.contains(&ann.category_id) {
                return Err(CliError::new(
                    ExitCode::Runtime,
                    "invalid_dataset",
                    format!(
                        "{gt_path}: annotation on image {} references undeclared category_id {}",
                        im.id, ann.category_id
                    ),
                )
                .hint("declare every referenced category_id in the split's `categories` array"));
            }
        }
    }

    // Fast path (the common case, including COCO val2017): `gt.categories` already lists every dataset class in
    // exact id-sorted order, so no other split JSON needs to be opened.
    let gt_names: Vec<&str> = gt.categories.iter().map(|c| c.1.as_str()).collect();
    let ds_names: Vec<&str> = dataset_classes.iter().map(String::as_str).collect();
    if gt_names == ds_names {
        let class_to_cat: Vec<Option<u64>> = gt.categories.iter().map(|(id, _)| Some(*id)).collect();
        let cat_to_class: HashMap<u64, u32> =
            gt.categories.iter().enumerate().map(|(i, (id, _))| (*id, i as u32)).collect();
        return Ok(CategoryMap { class_to_cat, cat_to_class });
    }

    // Subset path: read only the `categories` arrays across all COCO annotation files in `data` (streaming past
    // `images` and `annotations` without allocating) to verify that `gt.categories` is a consistent subset of the
    // dataset's merged `(id, name)` table and map each category id to its dataset-wide class index.
    let files = folo_data::coco_annotation_files(data)
        .map_err(|e| CliError::new(ExitCode::Runtime, "invalid_dataset", e.to_string()))?;
    let mut merged: BTreeMap<u64, String> = BTreeMap::new();
    for (_, path) in files {
        let cats = read_json_categories(&path)?;
        for (id, name) in cats {
            if let Some(prev) = merged.insert(id, name.clone())
                && prev != name
            {
                return Err(CliError::new(
                    ExitCode::Runtime,
                    "invalid_dataset",
                    format!(
                        "{}: category {id} is named both {prev:?} and {name:?} across splits",
                        folo_api::json::path(&path)
                    ),
                ));
            }
        }
    }
    let merged_names: Vec<&str> = merged.values().map(String::as_str).collect();
    if merged_names != ds_names {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("{gt_path}: its categories do not match the dataset's classes"),
        ));
    }

    let id_to_class: HashMap<u64, u32> = merged.keys().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    let mut class_to_cat = vec![None; dataset_classes.len()];
    let mut cat_to_class = HashMap::with_capacity(gt.categories.len());
    for (id, name) in &gt.categories {
        match merged.get(id) {
            Some(expected) if expected == name => {
                let idx = id_to_class[id];
                class_to_cat[idx as usize] = Some(*id);
                cat_to_class.insert(*id, idx);
            }
            _ => {
                return Err(CliError::new(
                    ExitCode::Runtime,
                    "invalid_dataset",
                    format!("{gt_path}: category ({id}, {name:?}) does not match the dataset's categories"),
                ));
            }
        }
    }
    Ok(CategoryMap { class_to_cat, cat_to_class })
}

/// Streams only the top-level `"categories"` array from a COCO JSON file, skipping `"images"` and `"annotations"`
/// without allocating so large training splits remain fast.
fn read_json_categories(path: &Path) -> Result<Vec<(u64, String)>, CliError> {
    struct CatEntry {
        id: u64,
        name: String,
    }
    impl<'de> Deserialize<'de> for CatEntry {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = CatEntry;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a COCO category object with `id` and `name`")
                }
                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<CatEntry, A::Error> {
                    let mut id: Option<u64> = None;
                    let mut name: Option<String> = None;
                    while let Some(key) = map.next_key::<std::borrow::Cow<'de, str>>()? {
                        match key.as_ref() {
                            "id" => {
                                let v: serde_json::Value = map.next_value()?;
                                id = v.as_u64().or_else(|| {
                                    v.as_f64()
                                        .filter(|f| {
                                            f.is_finite()
                                                && *f >= 0.0
                                                && f.fract() == 0.0
                                                && *f <= u64::MAX as f64
                                        })
                                        .map(|f| f as u64)
                                });
                                if id.is_none() {
                                    return Err(de::Error::custom(
                                        "category `id` must be a non-negative whole number",
                                    ));
                                }
                            }
                            "name" => name = Some(map.next_value::<String>()?),
                            _ => {
                                let _ = map.next_value::<IgnoredAny>()?;
                            }
                        }
                    }
                    Ok(CatEntry {
                        id: id.ok_or_else(|| de::Error::missing_field("id"))?,
                        name: name.ok_or_else(|| de::Error::missing_field("name"))?,
                    })
                }
            }
            d.deserialize_map(V)
        }
    }
    struct OnlyCategories(Vec<CatEntry>);
    impl<'de> Deserialize<'de> for OnlyCategories {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = OnlyCategories;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a COCO JSON object")
                }
                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<OnlyCategories, A::Error> {
                    let mut cats = Vec::new();
                    while let Some(key) = map.next_key::<std::borrow::Cow<'de, str>>()? {
                        if key.as_ref() == "categories" {
                            cats = map.next_value::<Vec<CatEntry>>()?;
                        } else {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                    Ok(OnlyCategories(cats))
                }
            }
            d.deserialize_map(V)
        }
    }
    let file = std::fs::File::open(path).map_err(|e| {
        CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("cannot read {}: {e}", folo_api::json::path(path)),
        )
    })?;
    let parsed: OnlyCategories = serde_json::from_reader(std::io::BufReader::with_capacity(1 << 20, file))
        .map_err(|e| {
            CliError::new(
                ExitCode::Runtime,
                "invalid_dataset",
                format!("{}: invalid COCO categories: {e}", folo_api::json::path(path)),
            )
        })?;
    Ok(parsed.0.into_iter().map(|c| (c.id, c.name)).collect())
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

/// pycocotools-exact evaluation: every GT image of the split (in id order), crowd regions, annotation areas.
pub fn evaluate(
    gt: &CocoGroundTruth,
    cats: &CategoryMap,
    part: &FileDataset,
    detections: &[(usize, Vec<Detection>)],
) -> CocoResult {
    let images = by_path(gt, part);
    let mut dets_of: HashMap<u64, &[Detection]> = HashMap::new();
    for (index, d) in detections {
        if let Some(im) = images.get(index) {
            dets_of.entry(im.id).or_insert(d.as_slice());
        }
    }
    let mut ev = CocoEval::new(cats.num_classes());
    let mut missing = 0;
    let mut truths = Vec::new();
    for im in &gt.images {
        truths.clear();
        for a in &im.annotations {
            let Some(class) = cats.class_of(a.category_id) else { continue };
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
            dets.iter().filter(|d| cats.category_of(d.class).is_some()).copied().collect();
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
    cats: &CategoryMap,
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
    let mut seen = HashSet::new();
    f.write_all(b"[").map_err(io)?;
    for (index, dets) in detections {
        let Some(im) = images.get(index) else { continue };
        if !seen.insert(im.id) {
            continue;
        }
        for d in dets {
            let Some(cat) = cats.category_of(d.class) else { continue };
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
    fn validate_categories_accepts_subset_and_rejects_empty_undeclared_or_reordered_model() {
        let dir = TestDir::new("coco-validate-cats");
        let ann = dir.join("annotations");
        std::fs::create_dir_all(&ann).unwrap();
        let train_json = ann.join("instances_train.json");
        let val_json = ann.join("instances_val.json");
        std::fs::write(
            &train_json,
            r#"{"images":[{"id":1,"file_name":"a.png"}],"annotations":[],"categories":[{"id":1,"name":"cat"},{"id":2,"name":"dog"},{"id":3,"name":"bird"}]}"#,
        )
        .unwrap();
        std::fs::write(
            &val_json,
            r#"{"images":[{"id":2,"file_name":"b.png"}],"annotations":[{"id":1,"image_id":2,"category_id":3,"bbox":[1,1,10,10]}],"categories":[{"id":1,"name":"cat"},{"id":3,"name":"bird"}]}"#,
        )
        .unwrap();
        let gt = folo_data::open_coco_ground_truth(dir.path(), folo_data::Split::Val).unwrap();
        let full = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()];

        // Valid strict subset with matching model classes maps class 0 -> 1, class 1 -> None, class 2 -> 3.
        let map = super::validate_categories(dir.path().to_str().unwrap(), &gt, &full, &[], &full).unwrap();
        assert_eq!(map.category_of(0), Some(1));
        assert_eq!(map.category_of(1), None);
        assert_eq!(map.category_of(2), Some(3));
        assert_eq!(map.class_of(1), Some(0));
        assert_eq!(map.class_of(2), None);
        assert_eq!(map.class_of(3), Some(2));

        // Conflicting category names reported by the loader are refused even on the fast path.
        let conflict = folo_data::Issue {
            kind: folo_data::IssueKind::Other,
            path: val_json.clone(),
            message: "category 1 is named both \"dog\" and \"cat\"".into(),
        };
        let err = super::validate_categories(dir.path().to_str().unwrap(), &gt, &full, &[conflict], &full)
            .unwrap_err();
        assert_eq!(err.code, "invalid_dataset");
        assert!(err.message.contains("is named both"), "{}", err.message);

        // Reordered or subset model classes are refused.
        let reordered = vec!["dog".to_string(), "cat".to_string(), "bird".to_string()];
        let err = super::validate_categories(dir.path().to_str().unwrap(), &gt, &full, &[], &reordered)
            .unwrap_err();
        assert_eq!(err.code, "invalid_dataset");
        let subset_model = vec!["cat".to_string(), "bird".to_string()];
        let err = super::validate_categories(dir.path().to_str().unwrap(), &gt, &full, &[], &subset_model)
            .unwrap_err();
        assert_eq!(err.code, "invalid_dataset");

        // Undeclared category_id in an annotation is refused.
        let mut bad_ann = gt.clone();
        bad_ann.images[0].annotations[0].category_id = 2;
        let err = super::validate_categories(dir.path().to_str().unwrap(), &bad_ann, &full, &[], &full)
            .unwrap_err();
        assert_eq!(err.code, "invalid_dataset");
        assert!(err.message.contains("undeclared category_id 2"), "{}", err.message);

        // Empty categories are refused.
        let mut empty_cats = gt.clone();
        empty_cats.categories.clear();
        let err = super::validate_categories(dir.path().to_str().unwrap(), &empty_cats, &full, &[], &full)
            .unwrap_err();
        assert_eq!(err.code, "invalid_dataset");
        assert!(err.message.contains("no categories"), "{}", err.message);
    }
}
