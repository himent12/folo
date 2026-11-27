//! Integration tests for dataset merging, class remapping, subset filtering, and perceptual (`dHash`)
//! near-duplicate / cross-split leakage auditing.

mod common;

use common::{TempDir, write, write_png};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{
    AuditOptions, ClassNormalize, CollisionPolicy, Dataset, FilterConfig, Format, Image, ImageFormat,
    LetterboxOptions, MergeConfig, MergeSplitPolicy, OpenOptions, RemapConfig, Split, SplitConfig,
    dhash_file, dhash_image, filter_dataset, hamming_distance, letterbox, merge_datasets, merge_paths, open,
    open_with, remap_classes, write_filtered, write_image, write_merged,
};
use std::collections::HashMap;

fn save_img(img: &Image, path: &std::path::Path, fmt: ImageFormat) {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    write_image(img, path, fmt).unwrap();
}

fn crop_border(img: &Image, margin_x: u32, margin_y: u32) -> Image {
    let nw = img.width - 2 * margin_x;
    let nh = img.height - 2 * margin_y;
    let mut out = Image::new(nw, nh, [0, 0, 0]);
    for y in 0..nh {
        for x in 0..nw {
            out.set_pixel(x, y, img.pixel(x + margin_x, y + margin_y));
        }
    }
    out
}

#[test]
fn merge_yolo_coco_and_voc_with_class_unification_remap_and_collision_resolution() {
    let t = TempDir::new("merge-3way");

    // 1. YOLO dataset: classes ["Person", "Car", "Noise"], colliding filename `0001.png` in train.
    let yolo_dir = t.join("yolo_src");
    write(&yolo_dir.join("data.yaml"), "train: images/train\nval: images/val\nnames: [Person, Car, Noise]\n");
    write_png(&yolo_dir.join("images/train/0001.png"), 64, 64, [200, 10, 10]);
    write(&yolo_dir.join("labels/train/0001.txt"), "0 0.5 0.5 0.4 0.4\n2 0.2 0.2 0.1 0.1\n");
    write_png(&yolo_dir.join("images/train/yolo_only.png"), 64, 64, [180, 30, 10]);
    write(&yolo_dir.join("labels/train/yolo_only.txt"), "2 0.5 0.5 0.2 0.2\n"); // only dropped class -> empty
    write_png(&yolo_dir.join("images/val/v1.png"), 64, 64, [150, 50, 20]);
    write(&yolo_dir.join("labels/val/v1.txt"), "1 0.5 0.5 0.3 0.3\n");

    // 2. COCO dataset: classes ["person", "Truck"], colliding filename `0001.png` in train.
    let coco_dir = t.join("coco_src");
    write_png(&coco_dir.join("images/train/0001.png"), 100, 100, [10, 200, 10]);
    write_png(&coco_dir.join("images/val/v2.png"), 100, 100, [20, 180, 30]);
    write(
        &coco_dir.join("annotations/instances_train.json"),
        r#"{
            "images": [{"id": 1, "file_name": "0001.png", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "Truck"}]
        }"#,
    );
    write(
        &coco_dir.join("annotations/instances_val.json"),
        r#"{
            "images": [{"id": 2, "file_name": "v2.png", "width": 100, "height": 100}],
            "annotations": [
                {"id": 3, "image_id": 2, "category_id": 2, "bbox": [20, 20, 40, 40], "area": 1600, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "Truck"}]
        }"#,
    );

    // 3. Pascal VOC dataset: classes ["  PERSON ", "Bicycle"], colliding filename `0001.png` in train.
    let voc_dir = t.join("voc_src");
    write_png(&voc_dir.join("JPEGImages/0001.png"), 80, 80, [10, 10, 200]);
    write(
        &voc_dir.join("Annotations/0001.xml"),
        "<annotation><filename>0001.png</filename><size><width>80</width><height>80</height></size>\
         <object><name>  PERSON </name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>40</xmax><ymax>40</ymax></bndbox></object>\
         <object><name>Bicycle</name><bndbox><xmin>41</xmin><ymin>41</ymin><xmax>80</xmax><ymax>80</ymax></bndbox></object>\
         </annotation>",
    );
    write(&voc_dir.join("ImageSets/Main/train.txt"), "0001\n");

    let mut class_map = HashMap::new();
    class_map.insert("truck".into(), Some("Car".into())); // unify Truck -> Car
    class_map.insert("noise".into(), None); // drop Noise

    let cfg = MergeConfig {
        normalize: ClassNormalize::CaseAndWhitespace,
        class_map,
        drop_unmapped: false,
        drop_empty_images: true, // drops yolo_only.png whose only box was Noise
        splits: MergeSplitPolicy::Preserve,
        collisions: CollisionPolicy::PrefixOnCollision,
        source_prefixes: vec!["yolo".into(), "coco".into(), "voc".into()],
    };

    let out = t.join("merged_yolo");
    let report = merge_paths(&[&yolo_dir, &coco_dir, &voc_dir], &out, &cfg).unwrap();
    assert_eq!(report.classes, vec!["Person", "Car", "Bicycle"]);
    assert_eq!(report.splits, [3, 2, 0]); // 3 train (one from each), 2 val (yolo + coco)
    assert_eq!(report.boxes, 7); // yolo:1+1, coco:2+1, voc:2 = 7
    assert_eq!(report.renamed_collisions, 3); // all three colliding `0001.png` prefixed

    // Verify clean prefixed filenames on disk with zero overwrites.
    assert!(out.join("images/train/yolo_0001.png").is_file());
    assert!(out.join("images/train/coco_0001.png").is_file());
    assert!(out.join("images/train/voc_0001.png").is_file());
    assert!(out.join("images/val/v1.png").is_file());
    assert!(out.join("images/val/v2.png").is_file());

    // Reload merged YOLO dataset and verify every box & class round-trips cleanly.
    let reloaded = open(&out).unwrap();
    assert_eq!(reloaded.format, Format::Yolo);
    assert!(reloaded.report.issues.is_empty(), "{:#?}", reloaded.report);
    assert_eq!(reloaded.classes, vec!["Person", "Car", "Bicycle"]);
    assert_eq!(reloaded.split_sizes(), [3, 2, 0]);
    assert_eq!(reloaded.stats().boxes, 7);
    assert_eq!(reloaded.stats().class_counts, vec![3, 3, 1]); // Person:3, Car(+Truck):3, Bicycle:1

    // Also verify PoolToTrain and Stratified split policies via in-memory `merge_datasets` + `write_merged`.
    let d_yolo = open(&yolo_dir).unwrap();
    let d_coco = open(&coco_dir).unwrap();
    let d_voc = open(&voc_dir).unwrap();

    let pooled = merge_datasets(
        &[&d_yolo, &d_coco, &d_voc],
        &MergeConfig { splits: MergeSplitPolicy::PoolToTrain, ..cfg.clone() },
    )
    .unwrap();
    assert_eq!(pooled.split_sizes(), [5, 0, 0]);

    let strat_out = t.join("merged_stratified");
    let strat_rep = write_merged(
        &[&d_yolo, &d_coco, &d_voc],
        &strat_out,
        &MergeConfig {
            splits: MergeSplitPolicy::Stratified(SplitConfig::train_val(0.6, 0.4, 42)),
            collisions: CollisionPolicy::AlwaysPrefix,
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(strat_rep.splits, [3, 2, 0]);
    assert_eq!(strat_rep.renamed_collisions, 5);
    let strat_reloaded = open(&strat_out).unwrap();
    assert_eq!(strat_reloaded.split_sizes(), [3, 2, 0]);
    assert!(strat_reloaded.audit().is_clean());
}

#[test]
fn remap_classes_and_filter_dataset_carve_clean_subsets() {
    let t = TempDir::new("remap-filter");
    let src = t.join("src");
    write(
        &src.join("data.yaml"),
        "train: images/train\nval: images/val\ntest: images/test\nnames: [person, car, truck, dog]\n",
    );

    // train/small_img.png: 50x50, person (tiny 0.04x0.04 => 2x2px, area=0.0016), car (0.4x0.4 => 20x20px, area=0.16)
    write_png(&src.join("images/train/small_img.png"), 50, 50, [10, 20, 30]);
    write(
        &src.join("labels/train/small_img.txt"),
        "0 0.5 0.5 0.04 0.04\n1 0.5 0.5 0.4 0.4\n3 0.2 0.2 0.2 0.2\n",
    );
    // train/large_img.png: 200x100, person (0.2x0.4 => 40x40px, area=0.08), truck (0.9x0.9 => 180x90px, area=0.81)
    write_png(&src.join("images/train/large_img.png"), 200, 100, [40, 50, 60]);
    write(&src.join("labels/train/large_img.txt"), "0 0.3 0.5 0.2 0.4\n2 0.5 0.5 0.9 0.9\n");
    // train/dog_only.png: 200x100, only dog box
    write_png(&src.join("images/train/dog_only.png"), 200, 100, [70, 80, 90]);
    write(&src.join("labels/train/dog_only.txt"), "3 0.5 0.5 0.5 0.5\n");
    // val/v.png: 200x100, car (0.2x0.3 => 40x30px, area=0.06)
    write_png(&src.join("images/val/v.png"), 200, 100, [100, 110, 120]);
    write(&src.join("labels/val/v.txt"), "1 0.5 0.5 0.2 0.3\n");
    // test/te.png: 200x100, person
    write_png(&src.join("images/test/te.png"), 200, 100, [130, 140, 150]);
    write(&src.join("labels/test/te.txt"), "0 0.5 0.5 0.3 0.3\n");

    let ds = open(&src).unwrap();
    assert_eq!(ds.split_sizes(), [3, 1, 1]);

    // 1. Remap: car + truck -> vehicle, drop dog, keep person, drop empty images.
    let mut map = HashMap::new();
    map.insert("car".into(), Some("vehicle".into()));
    map.insert("truck".into(), Some("vehicle".into()));
    map.insert("dog".into(), None);
    let remapped = ds
        .remap_classes(&RemapConfig {
            map,
            normalize: ClassNormalize::Exact,
            drop_unmapped: false,
            drop_empty_images: true,
        })
        .unwrap();
    assert_eq!(remapped.classes, vec!["person", "vehicle"]);
    assert_eq!(remapped.split_sizes(), [2, 1, 1]); // dog_only dropped!
    assert_eq!(remapped.train.boxes(1)[1].class, 1); // truck -> vehicle (id 1)

    // Also test remap_classes free function with drop_unmapped=true.
    let mut only_person = HashMap::new();
    only_person.insert("PERSON".into(), Some("human".into()));
    let human_only = remap_classes(
        &ds,
        &RemapConfig {
            map: only_person,
            normalize: ClassNormalize::CaseAndWhitespace,
            drop_unmapped: true,
            drop_empty_images: true,
        },
    )
    .unwrap();
    assert_eq!(human_only.classes, vec!["human"]);
    assert_eq!(human_only.split_sizes(), [2, 0, 1]);

    // 2. Filter subset: whitelist person + vehicle, drop test split, area in [0.01, 0.50],
    //    box width >= 25px, min image width >= 100px, drop empty images.
    let subset_cfg = FilterConfig {
        include_classes: Some(vec!["Person".into(), "VEHICLE".into()]),
        exclude_classes: vec![],
        normalize: ClassNormalize::CaseAndWhitespace,
        compact_classes: true,
        splits: Some(vec![Split::Train, Split::Val]),
        min_box_area: Some(0.01),
        max_box_area: Some(0.50),
        min_box_width_px: Some(25.0),
        min_image_width: Some(100),
        drop_empty_images: true,
        ..Default::default()
    };
    let filtered = filter_dataset(&remapped, &subset_cfg).unwrap();
    assert_eq!(filtered.classes, vec!["person", "vehicle"]);
    // train: small_img dropped (width 50 < 100); large_img keeps person (area 0.08, 40px) & drops vehicle (area 0.81 > 0.50)
    // val: v.png keeps vehicle (area 0.06, 40px)
    // test: dropped by splits selection
    assert_eq!(filtered.split_sizes(), [1, 1, 0]);
    assert_eq!(filtered.train.boxes(0).len(), 1);
    assert_eq!(filtered.train.boxes(0)[0].class, 0);
    assert_eq!(filtered.val.boxes(0).len(), 1);
    assert_eq!(filtered.val.boxes(0)[0].class, 1);

    // 3. Test pixel-dimension probing when dataset was opened with verify_images: false (entry.size == None).
    let unverified = open_with(&src, &OpenOptions { format: Format::Yolo, verify_images: false }).unwrap();
    assert_eq!(unverified.train.entries()[0].size, None);
    let probed_filter = unverified
        .filter(&FilterConfig {
            exclude_classes: vec!["dog".into()],
            min_box_width_px: Some(30.0),
            drop_empty_images: true,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(probed_filter.classes, vec!["person", "car", "truck"]);
    // small_img boxes are 2px and 20px -> both < 30px -> dropped; large_img (40px, 180px), val (40px), test (60px) survive
    assert_eq!(probed_filter.split_sizes(), [1, 1, 1]);
    assert_eq!(probed_filter.train.entries()[0].size, Some((200, 100)));

    // 4. Materialize filtered subset on disk via `write_filtered` and reopen cleanly.
    let sub_out = t.join("filtered_yolo");
    let conv = write_filtered(&remapped, &sub_out, Format::Yolo, &subset_cfg).unwrap();
    assert_eq!(conv.splits, [1, 1, 0]);
    assert_eq!(conv.boxes, 2);
    let sub_reloaded = open(&sub_out).unwrap();
    assert_eq!(sub_reloaded.classes, vec!["person", "vehicle"]);
    assert_eq!(sub_reloaded.split_sizes(), [1, 1, 0]);
}

#[test]
fn perceptual_dhash_audit_catches_recompressed_resized_and_cropped_split_leakage() {
    let t = TempDir::new("audit-dhash");
    let r = t.path();
    write(&r.join("data.yaml"), "train: images/train\nval: images/val\ntest: images/test\nnames: [obj]\n");

    // Use realistic procedural scenes from SyntheticDataset so each scene has distinct geometry & shading.
    let synth = SyntheticDataset::new(
        SyntheticConfig { images: 8, imgsz: 160, seed: 2026, ..Default::default() },
        Split::Train,
    )
    .unwrap();
    let scene1 = synth.get(0).unwrap().image;
    let scene2 = synth.get(1).unwrap().image;
    let scene3 = synth.get(2).unwrap().image;
    let scene4 = synth.get(3).unwrap().image;

    // Scene 1 in train (160x160 PNG) -> leaked to val as JPEG q=75 (recompressed) AND to test as 80x80 PNG (resized).
    save_img(&scene1, &r.join("images/train/s1.png"), ImageFormat::Png);
    write(&r.join("labels/train/s1.txt"), "0 0.5 0.5 0.3 0.3\n");
    save_img(&scene1, &r.join("images/val/s1_recomp.jpg"), ImageFormat::Jpeg(75));
    write(&r.join("labels/val/s1_recomp.txt"), "0 0.5 0.5 0.3 0.3\n");
    let (scene1_resized, _) =
        letterbox(&scene1, scene1.width / 2, scene1.height / 2, &LetterboxOptions::default());
    save_img(&scene1_resized, &r.join("images/test/s1_small.png"), ImageFormat::Png);
    write(&r.join("labels/test/s1_small.txt"), "0 0.5 0.5 0.3 0.3\n");

    // Scene 2 in train (160x160 PNG) -> leaked to val as lightly cropped (4px border crop = 2.5% per side).
    save_img(&scene2, &r.join("images/train/s2.png"), ImageFormat::Png);
    write(&r.join("labels/train/s2.txt"), "0 0.4 0.4 0.2 0.2\n");
    let scene2_cropped = crop_border(&scene2, 4, 4);
    save_img(&scene2_cropped, &r.join("images/val/s2_crop.png"), ImageFormat::Png);
    write(&r.join("labels/val/s2_crop.txt"), "0 0.4 0.4 0.2 0.2\n");

    // Scene 3 in train + exact byte duplicate in train (should be exact duplicate ONLY, NOT near-duplicate).
    save_img(&scene3, &r.join("images/train/s3.png"), ImageFormat::Png);
    write(&r.join("labels/train/s3.txt"), "0 0.6 0.6 0.2 0.2\n");
    std::fs::copy(r.join("images/train/s3.png"), r.join("images/train/s3_exact_copy.png")).unwrap();
    write(&r.join("labels/train/s3_exact_copy.txt"), "0 0.6 0.6 0.2 0.2\n");

    // Distinct unique scene 4 in val (must NOT match any train image).
    save_img(&scene4, &r.join("images/val/s4_unique.png"), ImageFormat::Png);
    write(&r.join("labels/val/s4_unique.txt"), "0 0.5 0.5 0.4 0.4\n");

    // Direct helper checks.
    let h1 = dhash_image(&scene1);
    let h1_small = dhash_image(&scene1_resized);
    let h2 = dhash_image(&scene2);
    let h2_crop = dhash_image(&scene2_cropped);
    assert!(
        hamming_distance(h1, h1_small) <= 6,
        "resized dist={} ({h1:016x} vs {h1_small:016x}), crop dist={} ({h2:016x} vs {h2_crop:016x})",
        hamming_distance(h1, h1_small),
        hamming_distance(h2, h2_crop),
    );
    assert_eq!(h1, dhash_file(&r.join("images/train/s1.png")).unwrap());
    let h4 = dhash_image(&scene4);
    assert!(hamming_distance(h1, h4) > 10, "distinct scenes should have large dHash distance");
    assert!(hamming_distance(dhash_image(&scene3), h4) > 10);

    let ds = open(r).unwrap();

    // Default audit (streaming exact byte hash only) finds exact duplicate s3 but NOT perceptual leaks.
    let exact_only = ds.audit();
    assert_eq!(exact_only.duplicate_groups.len(), 1);
    assert!(exact_only.split_leakage.is_empty());
    assert!(exact_only.near_duplicate_groups.is_empty());
    assert!(exact_only.near_duplicate_leakage.is_empty());
    assert!(!exact_only.has_split_leakage());

    // Perceptual dHash audit catches BOTH scene 1 (recompressed + resized across train/val/test)
    // AND scene 2 (lightly cropped across train/val), while leaving exact-only s3 out of near_duplicate_groups
    // and keeping distinct scene 4 unflagged.
    let report = ds.audit_with(&AuditOptions::perceptual(8));
    assert_eq!(report.duplicate_groups.len(), 1);
    assert!(report.has_near_duplicates());
    assert!(report.has_near_duplicate_leakage());
    assert!(report.has_split_leakage());
    assert!(report.has_train_val_leakage());
    assert!(!report.is_clean());
    assert_eq!(report.near_duplicate_groups.len(), 2, "{:#?}", report.near_duplicate_groups);
    assert_eq!(report.near_duplicate_leakage.len(), 2, "{:#?}", report.near_duplicate_leakage);

    let g1 = report
        .near_duplicate_leakage
        .iter()
        .find(|g| g.images.iter().any(|i| i.path.ends_with("s1.png")))
        .expect("scene1 leakage group");
    assert_eq!(g1.splits, vec![Split::Train, Split::Val, Split::Test]);
    assert_eq!(g1.images.len(), 3);
    assert!(g1.max_distance <= 8);

    let g2 = report
        .near_duplicate_leakage
        .iter()
        .find(|g| g.images.iter().any(|i| i.path.ends_with("s2.png")))
        .expect("scene2 leakage group");
    assert_eq!(g2.splits, vec![Split::Train, Split::Val]);
    assert_eq!(g2.images.len(), 2);
    assert!(g2.max_distance <= 8);

    // Tight threshold (0) excludes cropped pair if distance > 0 while keeping exact dHash matches.
    let strict = ds.audit_with(&AuditOptions::perceptual(0));
    assert!(strict.near_duplicate_groups.iter().all(|g| g.max_distance == 0));

    // Corrupt image file added after `open` is reported in `unreadable` without panicking.
    write(&r.join("images/val/s4_unique.png"), "not a valid png");
    let with_corrupt = ds.audit_with(&AuditOptions::perceptual(6));
    assert_eq!(with_corrupt.unreadable.len(), 1);
    assert!(with_corrupt.unreadable[0].path.ends_with("s4_unique.png"));
}
