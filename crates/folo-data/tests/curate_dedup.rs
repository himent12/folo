//! End-to-end integration tests for exact and perceptual (`dHash`) dataset deduplication (`curate::dedup`).

use folo_data::{
    AuditOptions, CrossSplitPolicy, DedupKeepPolicy, DedupMatchKind, DedupOptions, Image, ImageFormat,
    OpenOptions, Split, deduplicate_dataset, deduplicate_path, encode, open, open_with, write_deduplicated,
};
use folo_test_support::TestDir;
use std::path::Path;

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, bytes).unwrap();
}

/// Deterministic structured image with distinct horizontal gradients so different seeds yield large dHash distances
/// while resizes and JPEG recompressions of the same seed stay within small Hamming distance.
fn patterned_image(w: u32, h: u32, variant: u32) -> Image {
    let mut img = Image::new(w, h, [0, 0, 0]);
    for y in 0..h {
        for x in 0..w {
            let fx = x * 255 / w.max(1);
            let fy = y * 255 / h.max(1);
            let (r, g, b) = match variant {
                0 => {
                    // Left-to-right ramp + dark block in top-right quadrant.
                    let block = if x > w / 2 && y < h / 2 { 110u32 } else { 0 };
                    (fx.saturating_sub(block), fy, 255 - fx)
                }
                1 => {
                    // Right-to-left ramp + bright central stripe.
                    let stripe = if x > w / 3 && x < (2 * w) / 3 { 90u32 } else { 0 };
                    ((255 - fx).saturating_add(stripe).min(255), fx, fy)
                }
                _ => {
                    // Checkerboard blocks (distinct gradient sign pattern from variants 0 and 1).
                    let cx = (x * 4 / w.max(1)) & 1;
                    let cy = (y * 4 / h.max(1)) & 1;
                    let hi = if (cx ^ cy) == 1 { 235 } else { 25 };
                    (hi, 255 - hi, (fx + fy) / 2)
                }
            };
            img.set_pixel(x, y, [r as u8, g as u8, b as u8]);
        }
    }
    img
}

/// Builds a YOLO fixture containing:
/// - `train/a_hi.png` (64×48, variant 0, 1 box) and `train/a_lo.jpg` (32×24 JPEG near-dup of variant 0, 3 boxes)
/// - `val/a_leak.jpg` (48×36 JPEG near-dup of variant 0, 2 boxes) — cross-split perceptual leak!
/// - `train/exact_1.png` (40×32, variant 1, 1 box) and `val/exact_2.png` (exact byte copy, 4 boxes)
/// - `test/exact_3.png` (exact byte copy of variant 1, 2 boxes) — 3-split exact leak!
/// - `train/unique_t.png` (40×32, variant 2, 1 box) and `val/unique_v.png` (40×32, variant 2 flipped, 1 box)
fn build_fixture(root: &Path) {
    let a_hi = encode(&patterned_image(64, 48, 0), ImageFormat::Png).unwrap();
    let a_lo = encode(&patterned_image(32, 24, 0), ImageFormat::Jpeg(82)).unwrap();
    let a_leak = encode(&patterned_image(48, 36, 0), ImageFormat::Jpeg(75)).unwrap();

    let exact_bytes = encode(&patterned_image(40, 32, 1), ImageFormat::Png).unwrap();
    let unique_t = encode(&patterned_image(40, 32, 2), ImageFormat::Png).unwrap();
    let mut unique_v_img = patterned_image(40, 32, 2);
    // Flip horizontally so it has a completely different dHash from `unique_t`.
    unique_v_img = unique_v_img.oriented(2);
    let unique_v = encode(&unique_v_img, ImageFormat::Png).unwrap();

    write_file(
        &root.join("data.yaml"),
        b"train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: circle\n  1: square\n",
    );

    write_file(&root.join("images/train/a_hi.png"), &a_hi);
    write_file(&root.join("labels/train/a_hi.txt"), b"0 0.5 0.5 0.2 0.2\n");

    write_file(&root.join("images/train/a_lo.jpg"), &a_lo);
    write_file(
        &root.join("labels/train/a_lo.txt"),
        b"0 0.2 0.2 0.1 0.1\n0 0.5 0.5 0.2 0.2\n1 0.8 0.8 0.1 0.1\n",
    );

    write_file(&root.join("images/train/exact_1.png"), &exact_bytes);
    write_file(&root.join("labels/train/exact_1.txt"), b"1 0.5 0.5 0.3 0.3\n");

    write_file(&root.join("images/train/unique_t.png"), &unique_t);
    write_file(&root.join("labels/train/unique_t.txt"), b"0 0.4 0.4 0.2 0.2\n");

    write_file(&root.join("images/val/a_leak.jpg"), &a_leak);
    write_file(&root.join("labels/val/a_leak.txt"), b"0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n");

    write_file(&root.join("images/val/exact_2.png"), &exact_bytes);
    write_file(
        &root.join("labels/val/exact_2.txt"),
        b"0 0.2 0.2 0.1 0.1\n0 0.4 0.4 0.1 0.1\n1 0.6 0.6 0.1 0.1\n1 0.8 0.8 0.1 0.1\n",
    );

    write_file(&root.join("images/val/unique_v.png"), &unique_v);
    write_file(&root.join("labels/val/unique_v.txt"), b"1 0.5 0.5 0.2 0.2\n");

    write_file(&root.join("images/test/exact_3.png"), &exact_bytes);
    write_file(&root.join("labels/test/exact_3.txt"), b"1 0.5 0.5 0.3 0.3\n0 0.2 0.2 0.1 0.1\n");
}

#[test]
fn dedup_dry_run_and_materialize_preserve_source_and_clean_audit() {
    let dir = TestDir::new("curate-dedup-e2e");
    let src = dir.join("src");
    let out = dir.join("out");
    build_fixture(&src);

    let ds = open(&src).unwrap();
    assert_eq!(ds.split_sizes(), [4, 3, 1]);

    // 1. Dry-run preview must not create `out` on disk while reporting exact groups/kept/dropped.
    let dry_opts = DedupOptions {
        perceptual: true,
        max_hamming_distance: 6,
        keep: DedupKeepPolicy::KeepLargestResolution,
        cross_split: CrossSplitPolicy::PreferTrain,
        dry_run: true,
        ..Default::default()
    };
    let preview = write_deduplicated(&ds, &out, &dry_opts).unwrap();
    assert!(!out.exists(), "dry_run must not create output directory");
    assert!(preview.dry_run);
    assert!(preview.out_root.is_none());
    assert_eq!(preview.input_splits, [4, 3, 1]);
    assert_eq!(preview.groups.len(), 2, "expected 1 perceptual + 1 exact group: {preview:?}");
    assert_eq!(preview.cross_split_groups(), 2);
    assert_eq!(preview.total_dropped(), 4);
    assert_eq!(preview.output_splits, [3, 1, 0]);
    assert_eq!(preview.dropped_splits, [1, 2, 1]);
    assert!(preview.unreadable.is_empty());

    // Group 0: variant 0 (`a_hi` kept in train because PreferTrain + KeepLargestResolution 64x48 > 32x24).
    let g_perc = preview.groups.iter().find(|g| g.kind == DedupMatchKind::Perceptual).unwrap();
    assert!(g_perc.cross_split);
    assert_eq!(g_perc.kept.split, Split::Train);
    assert!(g_perc.kept.path.ends_with("a_hi.png"));
    assert_eq!(g_perc.kept.resolution, Some((64, 48)));
    assert_eq!(g_perc.dropped.len(), 2);

    // Group 1: exact 3-split leak (`exact_1.png` kept in train, `val/exact_2` & `test/exact_3` dropped).
    let g_exact = preview.groups.iter().find(|g| g.kind == DedupMatchKind::Exact).unwrap();
    assert!(g_exact.cross_split);
    assert_eq!(g_exact.max_distance, 0);
    assert_eq!(g_exact.kept.split, Split::Train);
    assert!(g_exact.kept.path.ends_with("exact_1.png"));
    assert_eq!(g_exact.dropped.len(), 2);

    // 2. Materialise to `out` via `deduplicate_path`, verify clean reopen & clean perceptual audit.
    let mut write_opts = dry_opts;
    write_opts.dry_run = false;
    let written = deduplicate_path(&src, &out, &write_opts).unwrap();
    assert!(!written.dry_run);
    assert_eq!(written.out_root.as_deref(), Some(out.as_path()));
    assert_eq!(written.output_splits, preview.output_splits);
    assert_eq!(written.dropped_splits, preview.dropped_splits);
    assert_eq!(written.groups, preview.groups);

    // Reopen output YOLO dataset: zero exact or perceptual duplicates remain.
    let clean = open(&out).unwrap();
    assert_eq!(clean.split_sizes(), [3, 1, 0]);
    let audit = clean.audit_with(&AuditOptions::perceptual(6));
    assert!(audit.is_clean(), "deduplicated output must pass perceptual audit cleanly: {audit:?}");

    // Source dataset must remain completely untouched on disk.
    let src_after = open(&src).unwrap();
    assert_eq!(src_after.split_sizes(), [4, 3, 1]);

    // Refuses writing into an existing non-empty directory (including source or already-written `out`).
    assert!(write_deduplicated(&ds, &src, &write_opts).is_err());
    assert!(write_deduplicated(&ds, &out, &write_opts).is_err());
}

#[test]
fn dedup_keep_and_cross_split_policies() {
    let dir = TestDir::new("curate-dedup-policies");
    let src = dir.join("src");
    build_fixture(&src);
    let ds = open(&src).unwrap();

    // 1. `KeepMostBoxes` + `PreferTrain`: in variant 0, `train/a_lo.jpg` has 3 boxes > `train/a_hi.png`'s 1 box.
    let (pruned, rep) = ds
        .deduplicate(&DedupOptions {
            perceptual: true,
            keep: DedupKeepPolicy::KeepMostBoxes,
            cross_split: CrossSplitPolicy::PreferTrain,
            ..Default::default()
        })
        .unwrap();
    let g_perc = rep.groups.iter().find(|g| g.kind == DedupMatchKind::Perceptual).unwrap();
    assert!(
        g_perc.kept.path.ends_with("a_lo.jpg"),
        "expected a_lo.jpg (3 boxes), got {:?}",
        g_perc.kept.path
    );
    assert_eq!(g_perc.kept.boxes, 3);
    assert_eq!(pruned.split_sizes(), [3, 1, 0]);

    // 2. `PreferVal`: protects validation set by keeping `val/a_leak.jpg` and `val/exact_2.png`.
    let (_, rep_val) = deduplicate_dataset(
        &ds,
        &DedupOptions {
            perceptual: true,
            keep: DedupKeepPolicy::KeepFirst,
            cross_split: CrossSplitPolicy::PreferVal,
            ..Default::default()
        },
    )
    .unwrap();
    for g in &rep_val.groups {
        assert_eq!(g.kept.split, Split::Val, "PreferVal must keep val copy: {g:?}");
    }
    assert_eq!(rep_val.output_splits, [1, 3, 0]);

    // 3. `PreferTest`: keeps `test/exact_3.png` for the 3-split group, and `val/a_leak.jpg` for the train+val group.
    let (_, rep_test) = deduplicate_dataset(
        &ds,
        &DedupOptions {
            perceptual: true,
            keep: DedupKeepPolicy::KeepFirst,
            cross_split: CrossSplitPolicy::PreferTest,
            ..Default::default()
        },
    )
    .unwrap();
    let g_ex = rep_test.groups.iter().find(|g| g.kind == DedupMatchKind::Exact).unwrap();
    assert_eq!(g_ex.kept.split, Split::Test);
    let g_p = rep_test.groups.iter().find(|g| g.kind == DedupMatchKind::Perceptual).unwrap();
    assert_eq!(g_p.kept.split, Split::Val);
    assert_eq!(rep_test.output_splits, [1, 2, 1]);

    // 4. `IgnoreSplit` + `KeepMostBoxes`: exact group keeps `val/exact_2.png` (4 boxes > train 1, test 2).
    let (_, rep_ign) = deduplicate_dataset(
        &ds,
        &DedupOptions {
            perceptual: true,
            keep: DedupKeepPolicy::KeepMostBoxes,
            cross_split: CrossSplitPolicy::IgnoreSplit,
            ..Default::default()
        },
    )
    .unwrap();
    let g_ex = rep_ign.groups.iter().find(|g| g.kind == DedupMatchKind::Exact).unwrap();
    assert_eq!(g_ex.kept.split, Split::Val);
    assert_eq!(g_ex.kept.boxes, 4);

    // 5. `WithinSplitOnly`: only drops `train/a_lo.jpg` inside train; keeps `val/a_leak.jpg`, `val/exact_2.png`,
    //    and `test/exact_3.png` untouched across splits.
    let (_, rep_within) = deduplicate_dataset(
        &ds,
        &DedupOptions {
            perceptual: true,
            keep: DedupKeepPolicy::KeepLargestResolution,
            cross_split: CrossSplitPolicy::WithinSplitOnly,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rep_within.groups.len(), 1);
    assert!(!rep_within.groups[0].cross_split);
    assert_eq!(rep_within.groups[0].kept.split, Split::Train);
    assert!(rep_within.groups[0].kept.path.ends_with("a_hi.png"));
    assert_eq!(rep_within.output_splits, [3, 3, 1]);

    // 6. `perceptual: false` (exact-byte only) + `KeepLargestResolution` (exercises lazy `probe_size` path).
    let (_, rep_exact_only) = deduplicate_dataset(
        &ds,
        &DedupOptions {
            perceptual: false,
            keep: DedupKeepPolicy::KeepLargestResolution,
            cross_split: CrossSplitPolicy::PreferTrain,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rep_exact_only.groups.len(), 1);
    assert_eq!(rep_exact_only.groups[0].kind, DedupMatchKind::Exact);
    assert_eq!(rep_exact_only.groups[0].kept.resolution, Some((40, 32)));
    assert_eq!(rep_exact_only.output_splits, [4, 2, 0]);
}

#[test]
fn dedup_reports_unreadable_without_false_grouping() {
    let dir = TestDir::new("curate-dedup-unreadable");
    let src = dir.join("src");
    build_fixture(&src);
    let ds = open(&src).unwrap();

    // Remove one unique image after opening to simulate a missing file at dedup time.
    std::fs::remove_file(src.join("images/train/unique_t.png")).unwrap();

    let (_, rep) = deduplicate_dataset(&ds, &DedupOptions { dry_run: true, ..Default::default() }).unwrap();
    assert_eq!(rep.unreadable.len(), 1);
    assert!(rep.unreadable[0].path.ends_with("unique_t.png"));
    assert!(!rep.is_clean());
    // Legitimate duplicate groups are still reported cleanly.
    assert_eq!(rep.groups.len(), 2);

    // A dataset opened with `verify_images: false` containing a corrupt `.png` reports it on `dry_run`
    // and refuses `write_deduplicated` BEFORE creating `out` (preventing entries disappearing on reopen).
    let corrupt_src = dir.join("corrupt_src");
    let corrupt_out = dir.join("corrupt_out");
    let valid_png = encode(&patterned_image(40, 32, 0), ImageFormat::Png).unwrap();
    write_file(&corrupt_src.join("data.yaml"), b"train: images/train\nnames:\n  0: circle\n");
    write_file(&corrupt_src.join("images/train/good.png"), &valid_png);
    write_file(&corrupt_src.join("labels/train/good.txt"), b"0 0.5 0.5 0.2 0.2\n");
    write_file(&corrupt_src.join("images/train/broken.png"), b"not-a-valid-png-header");
    write_file(&corrupt_src.join("labels/train/broken.txt"), b"0 0.5 0.5 0.2 0.2\n");

    let unverified =
        open_with(&corrupt_src, &OpenOptions { verify_images: false, ..Default::default() }).unwrap();
    assert_eq!(unverified.split_sizes(), [2, 0, 0]);
    let dry_rep =
        write_deduplicated(&unverified, &corrupt_out, &DedupOptions { dry_run: true, ..Default::default() })
            .unwrap();
    assert_eq!(dry_rep.unreadable.len(), 1);
    assert!(dry_rep.unreadable[0].path.ends_with("broken.png"));
    assert!(!corrupt_out.exists());

    let err = write_deduplicated(&unverified, &corrupt_out, &DedupOptions::default()).unwrap_err();
    assert!(err.to_string().contains("unreadable"), "unexpected error: {err}");
    assert!(!corrupt_out.exists(), "failed write_deduplicated must not create output dir");

    // Exact-byte mode (`perceptual: false`) must also refuse unreadable/corrupt surviving images before claiming `out`.
    let err_exact = write_deduplicated(
        &unverified,
        &corrupt_out,
        &DedupOptions { perceptual: false, ..Default::default() },
    )
    .unwrap_err();
    assert!(err_exact.to_string().contains("unreadable"), "unexpected error: {err_exact}");
    assert!(!corrupt_out.exists(), "failed exact-mode write_deduplicated must not create output dir");

    // Stem collision disambiguation (`same.png` + `same.jpg` in one split) and test-only refusal.
    let coll_src = dir.join("coll_src");
    let coll_out = dir.join("coll_out");
    let img_a = encode(&patterned_image(40, 32, 0), ImageFormat::Png).unwrap();
    let img_b = encode(&patterned_image(40, 32, 1), ImageFormat::Jpeg(90)).unwrap();
    write_file(&coll_src.join("data.yaml"), b"train: images/train\nnames:\n  0: circle\n");
    write_file(&coll_src.join("images/train/same.png"), &img_a);
    write_file(&coll_src.join("labels/train/same.txt"), b"0 0.5 0.5 0.2 0.2\n");
    write_file(&coll_src.join("images/train/sub/same.jpg"), &img_b);
    write_file(&coll_src.join("labels/train/sub/same.txt"), b"0 0.4 0.4 0.2 0.2\n");
    let coll_ds = open(&coll_src).unwrap();
    let coll_rep = write_deduplicated(&coll_ds, &coll_out, &DedupOptions::default()).unwrap();
    assert_eq!(coll_rep.output_splits, [2, 0, 0]);
    assert_eq!(coll_rep.renamed_collisions, 2);
    assert_eq!(open(&coll_out).unwrap().split_sizes(), [2, 0, 0]);

    // Refuses to write when only `test` images survive (e.g. `PreferTest` drops the only `train` duplicate).
    let test_only_src = dir.join("test_only_src");
    let test_only_out = dir.join("test_only_out");
    write_file(
        &test_only_src.join("data.yaml"),
        b"train: images/train\ntest: images/test\nnames:\n  0: circle\n",
    );
    write_file(&test_only_src.join("images/train/dup.png"), &img_a);
    write_file(&test_only_src.join("labels/train/dup.txt"), b"0 0.5 0.5 0.2 0.2\n");
    write_file(&test_only_src.join("images/test/dup.png"), &img_a);
    write_file(&test_only_src.join("labels/test/dup.txt"), b"0 0.5 0.5 0.2 0.2\n");
    let test_only_ds = open(&test_only_src).unwrap();
    let err_test_only = write_deduplicated(
        &test_only_ds,
        &test_only_out,
        &DedupOptions { cross_split: CrossSplitPolicy::PreferTest, ..Default::default() },
    )
    .unwrap_err();
    assert!(err_test_only.to_string().contains("only has test images"), "unexpected error: {err_test_only}");
    assert!(!test_only_out.exists(), "refused test-only write must not create output dir");
}
