//! Deterministic unit and headless UI tests for the Dataset Curate & Export CLI Builder
//! (including `folo dataset export --format coco` matching `crates/folo-cli/src/cli/dataset.rs`).

use super::*;
use crate::api::{Backend, DatasetFormat, Split, mock::MockBackend};
use crate::app::tests::{drive, open};
use crate::app::{DsTab, Route};

#[test]
fn builds_verified_convert_split_filter_dedup_and_export_coco_cli_commands() {
    let api = MockBackend::new(4200, true);
    let mut ds = api.dataset("ds_3b91").unwrap();
    ds.path = "/data/my shapes/data.yaml".into();
    ds.format = DatasetFormat::Yolo;
    ds.classes = vec!["circle".into(), "stop sign".into()];

    let mut st = DatasetCliState::default();
    st.sync_dataset(&ds);
    assert_eq!(st.convert_to, DatasetTargetFormat::Coco);

    // 1. Convert: strips data.yaml to derive sibling output dir, emits --from yolo, and quotes spaces safely.
    let conv = build_dataset_cli(&ds, &st);
    assert_eq!(
        conv.command,
        "folo dataset convert \"/data/my shapes/data.yaml\" \"/data/my shapes-coco\" --from yolo --to coco"
    );
    assert!(conv.quote_note.is_none());

    // 2. Split: includes --out, --from, --val, optional --test and --to.
    st.mode = DatasetCliMode::Split;
    st.split_val_idx = 1; // 0.2
    st.split_test_idx = 1; // 0.1
    st.split_to = Some(DatasetTargetFormat::Voc);
    let sp = build_dataset_cli(&ds, &st);
    assert_eq!(
        sp.command,
        "folo dataset split \"/data/my shapes/data.yaml\" --out \"/data/my shapes-split\" --from yolo --val 0.2 --test 0.1 --to voc"
    );

    // 3. Filter: uses exact verified flags --from, --split, --include-class, --min-box-area, --drop-empty-images.
    st.mode = DatasetCliMode::Filter;
    st.filter_split = Some(Split::Train);
    st.filter_class_mode = FilterClassMode::Include;
    st.filter_class = Some(1);
    st.filter_drop_tiny = true;
    st.filter_drop_empty = true;
    let flt = build_dataset_cli(&ds, &st);
    assert_eq!(
        flt.command,
        "folo dataset filter \"/data/my shapes/data.yaml\" --out \"/data/my shapes-filtered\" --from yolo --split train --include-class \"stop sign\" --min-box-area 0.0025 --drop-empty-images"
    );

    // 4. Dedup check: uses real `folo dataset check --format <fmt> --perceptual [--dhash-threshold] [--fix-dry-run]`.
    st.mode = DatasetCliMode::DedupCheck;
    st.check_perceptual = true;
    st.check_dhash_idx = 0; // 4 bits (non-default -> emitted)
    st.check_fix_dry_run = true;
    let chk = build_dataset_cli(&ds, &st);
    assert_eq!(
        chk.command,
        "folo dataset check \"/data/my shapes/data.yaml\" --format yolo --perceptual --dhash-threshold 4 --fix-dry-run"
    );

    // 4b. Auto format omits explicit --from / --format flags.
    ds.format = DatasetFormat::Auto;
    assert_eq!(
        build_dataset_cli(&ds, &st).command,
        "folo dataset check \"/data/my shapes/data.yaml\" --perceptual --dhash-threshold 4 --fix-dry-run"
    );
    ds.format = DatasetFormat::Yolo;

    // 5. Export COCO: verified exact flags `--out`, `--format coco`, `--from`, optional `--split`, `--images`, `--pretty`.
    st.mode = DatasetCliMode::ExportCoco;
    st.export_coco_split = None;
    st.export_coco_images = CocoImageMode::Copy;
    st.export_coco_pretty = false;
    let coco_default = build_dataset_cli(&ds, &st);
    assert_eq!(
        coco_default.command,
        "folo dataset export \"/data/my shapes/data.yaml\" --out \"/data/my shapes-coco-export\" --format coco --from yolo --images copy"
    );
    assert!(coco_default.copy_blocked.is_none());

    st.export_coco_split = Some(Split::Val);
    st.export_coco_images = CocoImageMode::Reference;
    st.export_coco_pretty = true;
    let coco_opts = build_dataset_cli(&ds, &st);
    assert_eq!(
        coco_opts.command,
        "folo dataset export \"/data/my shapes/data.yaml\" --out \"/data/my shapes-coco-export\" --format coco --from yolo --split val --images reference --pretty"
    );

    st.export_coco_images = CocoImageMode::Link;
    st.export_coco_split = Some(Split::Test);
    ds.format = DatasetFormat::Auto;
    let coco_link = build_dataset_cli(&ds, &st);
    assert_eq!(
        coco_link.command,
        "folo dataset export \"/data/my shapes/data.yaml\" --out \"/data/my shapes-coco-export\" --format coco --split test --images link --pretty"
    );
    ds.format = DatasetFormat::Yolo;

    // 6a. Leading '<' path never bypasses quote_path (safely quoted instead of raw shell redirection).
    ds.path = "<bad;id>/data.yaml".into();
    st.mode = DatasetCliMode::Convert;
    let angle_cmd = build_dataset_cli(&ds, &st);
    assert_eq!(
        angle_cmd.command,
        "folo dataset convert \"<bad;id>/data.yaml\" \"<bad;id>-coco\" --from yolo --to coco"
    );
    assert!(angle_cmd.copy_blocked.is_none());

    // 6b. Unquotable shell-unsafe path falls back to placeholders and blocks clipboard copying (including ExportCoco).
    ds.path = "/data/bad`$(rm -rf)/data.yaml".into();
    let unsafe_cmd = build_dataset_cli(&ds, &st);
    assert_eq!(unsafe_cmd.command, "folo dataset convert <dataset> <output> --from yolo --to coco");
    assert!(unsafe_cmd.quote_note.is_some());
    assert!(unsafe_cmd.copy_blocked.is_some());

    st.mode = DatasetCliMode::ExportCoco;
    st.export_coco_split = None;
    st.export_coco_images = CocoImageMode::Copy;
    st.export_coco_pretty = false;
    let unsafe_coco = build_dataset_cli(&ds, &st);
    assert_eq!(
        unsafe_coco.command,
        "folo dataset export <dataset> --out <output> --format coco --from yolo --images copy"
    );
    assert!(unsafe_coco.copy_blocked.is_some());

    // 7. Standard COCO manifest under `annotations/` derives sibling of the dataset root and emits --from coco.
    ds.path = "/data/coco/annotations/instances_train.json".into();
    ds.format = DatasetFormat::Coco;
    st.mode = DatasetCliMode::Convert;
    st.convert_to = DatasetTargetFormat::Yolo;
    let coco_conv = build_dataset_cli(&ds, &st);
    assert_eq!(
        coco_conv.command,
        "folo dataset convert /data/coco/annotations/instances_train.json /data/coco-yolo --from coco --to yolo"
    );
    assert!(coco_conv.copy_blocked.is_none());

    // 8. Comma in class name blocks `folo dataset filter --include-class` (clap splits on ',').
    ds.classes = vec!["circle".into(), "person, ped".into()];
    st.mode = DatasetCliMode::Filter;
    st.filter_class = Some(1);
    let comma_flt = build_dataset_cli(&ds, &st);
    assert!(comma_flt.copy_blocked.is_some());
    assert!(comma_flt.command.contains("--include-class <class>"));
}

#[test]
fn export_coco_panel_controls_wire_into_dataset_statistics_tab() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Dataset("ds_8f2c".into()), w, cx);
        this.ds_tab = DsTab::Statistics;
        this.ds_cli.mode = DatasetCliMode::ExportCoco;
        cx.notify();
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "ds-cli-mode"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "ds-cli-coco-split-0"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "ds-cli-coco-img-copy"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "ds-cli-coco-img-link"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "ds-cli-coco-img-reference"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "ds-cli-coco-pretty"), "{keys:?}");
        assert!(keys.iter().any(|k| k == "ds-cli-copy"), "{keys:?}");
    });
}

#[test]
fn export_coco_disables_and_strips_empty_splits_across_datasets() {
    let api = MockBackend::new(4200, true);
    let mut ds_with_test = api.dataset("ds_3b91").unwrap();
    ds_with_test.id = "ds_a".into();
    ds_with_test.path = "/data/a/data.yaml".into();
    ds_with_test.format = DatasetFormat::Yolo;
    ds_with_test.splits.test = 20;

    let mut ds_no_test = ds_with_test.clone();
    ds_no_test.id = "ds_b".into();
    ds_no_test.path = "/data/b/data.yaml".into();
    ds_no_test.splits.test = 0;

    let mut st = DatasetCliState {
        mode: DatasetCliMode::ExportCoco,
        export_coco_split: Some(Split::Test),
        ..Default::default()
    };
    st.sync_dataset(&ds_with_test);
    assert_eq!(st.export_coco_split, Some(Split::Test));
    let cmd_a = build_dataset_cli(&ds_with_test, &st);
    assert!(cmd_a.command.contains("--split test"), "{}", cmd_a.command);

    // Even before sync_dataset runs, build_dataset_cli refuses to emit an empty split.
    let cmd_b_direct = build_dataset_cli(&ds_no_test, &st);
    assert!(!cmd_b_direct.command.contains("--split test"), "{}", cmd_b_direct.command);

    // sync_dataset clears the stale empty split when switching datasets.
    st.sync_dataset(&ds_no_test);
    assert_eq!(st.export_coco_split, None);
    let cmd_b = build_dataset_cli(&ds_no_test, &st);
    assert!(!cmd_b.command.contains("--split"), "{}", cmd_b.command);
}
