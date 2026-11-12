# Shapes: the convergence smoke test

docs/MODEL.md §10.6 (the M2 gate): train folo-n from scratch on folo-data's synthetic shapes, 4 classes, 2000 train
and 200 val images at 320×320, and reach **mAP@0.5 ≥ 0.90 and mAP@0.5:0.95 ≥ 0.75 by epoch 30** (EMA model) with no
non-finite loss.

## Commands

```sh
# 1. The dataset: circle / square / triangle / star, 1 to 8 objects per image, seed 0 (about 3 s, ~60 MB of JPEGs).
cargo run --release -p folo-data --example shapes -- D:/folo-data/shapes-smoke
#    `cargo folo dataset synth D:/folo-data/shapes-alt --images 2200 --val-frac 0.090909 --imgsz 320 \
#      --classes circle,square,triangle,star` is close, but its generator defaults keep 5 % empty images.

# 2. Train (MODEL.md's run; `auto` picks MuSGD at lr0 0.01, 125 batches = 32 optimizer steps per epoch,
#    96 warmup steps (3 epochs), mosaic off for the last 5 epochs).
cargo folo train --data D:/folo-data/shapes-smoke --arch folo-n --imgsz 320 --epochs 30 --batch 16 \
  --close-mosaic 5 --seed 0 --project runs --name smoke-cpu
#    add --json for one JSON record per event on stdout (metrics.jsonl in the run folder has every step).
#    Interrupted? continue bit-exactly with:  cargo folo train --resume runs/smoke-cpu/weights/last.folo

# 3. Evaluate, predict, export.
cargo folo val --model runs/smoke-cpu/weights/best.folo --data D:/folo-data/shapes-smoke
cargo folo predict --model runs/smoke-cpu/weights/best.folo D:/folo-data/shapes-smoke/images/val --save runs/smoke-cpu/pred
cargo folo export --model runs/smoke-cpu/weights/best.folo --format folo-f16
```

`cargo folo` is the alias for `cargo run --release -p folo-cli --` (`.cargo/config.toml`).

## Result (CPU)

Measured 2026-11-08 on the dev machine (28 threads, Windows), commit of PR #33, `--device cpu`, when `auto` still
picked AdamW at lr0 0.00125 with a 100-step warmup here (docs/MODEL.md §10.6 has the current rule's run): 30 epochs in
72 min (4326 s; 16–18 img/s training when the machine was otherwise idle, slower while other builds shared it). No
non-finite batch. **Pass**: the gate (mAP50 ≥ 0.90, mAP50-95 ≥ 0.75) was first met at epoch 9 (0.936 / 0.772).
The reference (Ultralytics YOLO11n, MODEL.md §10.6) reached 0.854 / 0.904 / 0.933 / 0.948 mAP50 at epochs
6 / 11 / 16 / 30; FOLO: 0.824 / 0.947 / 0.970 / 0.980. The loss drops visibly when mosaic closes (epoch 26).

EMA validation after each epoch (train losses are the epoch means of the reported items):

| epoch | box | cls | dfl | P | R | mAP50 | mAP50-95 | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.284 | 4.335 | 4.273 | 0.042 | 0.002 | 0.002 | 0.000 | 120 |
| 2 | 2.971 | 3.012 | 3.026 | 0.121 | 0.285 | 0.071 | 0.029 | 114 |
| 3 | 2.153 | 2.176 | 1.977 | 0.276 | 0.611 | 0.305 | 0.174 | 86 |
| 4 | 1.684 | 1.810 | 1.510 | 0.445 | 0.669 | 0.544 | 0.362 | 116 |
| 5 | 1.375 | 1.509 | 1.294 | 0.542 | 0.760 | 0.665 | 0.504 | 96 |
| 6 | 1.209 | 1.284 | 1.180 | 0.772 | 0.718 | 0.824 | 0.651 | 92 |
| 7 | 1.105 | 1.117 | 1.125 | 0.773 | 0.800 | 0.878 | 0.698 | 151 |
| 8 | 1.008 | 0.981 | 1.083 | 0.778 | 0.835 | 0.887 | 0.721 | 169 |
| 9 | 0.972 | 0.914 | 1.063 | 0.917 | 0.839 | 0.936 | 0.772 | 146 |
| 10 | 0.922 | 0.840 | 1.041 | 0.926 | 0.857 | 0.942 | 0.788 | 245 |
| 11 | 0.884 | 0.796 | 1.029 | 0.931 | 0.856 | 0.947 | 0.802 | 155 |
| 12 | 0.846 | 0.745 | 1.019 | 0.916 | 0.881 | 0.948 | 0.801 | 150 |
| 13 | 0.816 | 0.713 | 1.000 | 0.931 | 0.895 | 0.960 | 0.830 | 113 |
| 14 | 0.804 | 0.692 | 0.993 | 0.922 | 0.906 | 0.964 | 0.830 | 117 |
| 15 | 0.755 | 0.642 | 0.976 | 0.941 | 0.901 | 0.966 | 0.829 | 157 |
| 16 | 0.748 | 0.626 | 0.971 | 0.929 | 0.933 | 0.970 | 0.839 | 106 |
| 17 | 0.735 | 0.610 | 0.970 | 0.949 | 0.914 | 0.970 | 0.858 | 100 |
| 18 | 0.713 | 0.598 | 0.963 | 0.959 | 0.879 | 0.966 | 0.856 | 205 |
| 19 | 0.701 | 0.581 | 0.961 | 0.959 | 0.928 | 0.975 | 0.866 | 147 |
| 20 | 0.695 | 0.568 | 0.957 | 0.981 | 0.905 | 0.975 | 0.874 | 205 |
| 21 | 0.678 | 0.559 | 0.957 | 0.966 | 0.924 | 0.975 | 0.868 | 179 |
| 22 | 0.666 | 0.548 | 0.946 | 0.969 | 0.910 | 0.973 | 0.860 | 202 |
| 23 | 0.659 | 0.535 | 0.947 | 0.955 | 0.923 | 0.975 | 0.878 | 240 |
| 24 | 0.637 | 0.520 | 0.940 | 0.959 | 0.930 | 0.975 | 0.883 | 251 |
| 25 | 0.640 | 0.516 | 0.940 | 0.981 | 0.932 | 0.979 | 0.887 | 110 |
| 26 | 0.522 | 0.438 | 0.876 | 0.981 | 0.929 | 0.980 | 0.888 | 99 |
| 27 | 0.503 | 0.420 | 0.873 | 0.969 | 0.948 | 0.981 | 0.895 | 102 |
| 28 | 0.496 | 0.411 | 0.870 | 0.953 | 0.957 | 0.980 | 0.896 | 103 |
| 29 | 0.488 | 0.404 | 0.867 | 0.958 | 0.957 | 0.981 | 0.901 | 128 |
| 30 | 0.478 | 0.396 | 0.872 | 0.952 | 0.957 | 0.980 | 0.898 | 123 |

Best epoch 29 (`weights/best.folo`), per class; `folo val` on best.folo reproduces it exactly (fused EMA, 3.7 s for
the 200 val images):

| class | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|
| circle | 0.987 | 0.944 | 0.991 | 0.966 |
| square | 0.978 | 0.908 | 0.956 | 0.943 |
| triangle | 0.974 | 0.869 | 0.956 | 0.951 |
| star | 0.984 | 0.882 | 0.930 | 0.968 |

`folo predict` on the 200 val images at 320 px (single images, rect letterbox, fused f32, CPU): mean 33.0 ms per
image (preprocess 1.5, inference 30.6, postprocess 0.9).

## Result (GPU)

Measured 2026-11-08 on the same machine (RTX 4080 SUPER, Vulkan), branch `train/gpu`, the same command with
`--device gpu:0` (now also what `auto` picks):

```sh
cargo folo train --data D:/folo-data/shapes-smoke --arch folo-n --imgsz 320 --epochs 30 --batch 16 \
  --close-mosaic 5 --seed 0 --device gpu:0 --project runs --name smoke-gpu
```

30 epochs in **169 s** (2.8 min, 4–6 s per epoch including validation, about 400 img/s training) against 4326 s on
the CPU: **25.6× faster** wall clock. No non-finite batch. **Pass**: the gate was first met at epoch 9 (0.925 /
0.764), the same epoch as on the CPU, and the run ends at mAP50 0.980 / mAP50-95 0.896 (CPU 0.980 / 0.898). The two
runs are not bit-identical (sums run in other orders on the GPU) but track each other to the third decimal in the
first epochs and stay within run-to-run noise after.

| epoch | box | cls | dfl | P | R | mAP50 | mAP50-95 | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.280 | 4.335 | 4.275 | 0.000 | 0.002 | 0.000 | 0.000 | 14.0 |
| 2 | 2.966 | 3.009 | 3.053 | 0.132 | 0.247 | 0.085 | 0.034 | 5.8 |
| 3 | 2.147 | 2.164 | 1.969 | 0.245 | 0.616 | 0.290 | 0.169 | 5.5 |
| 4 | 1.674 | 1.813 | 1.509 | 0.407 | 0.630 | 0.487 | 0.311 | 5.3 |
| 5 | 1.372 | 1.537 | 1.293 | 0.548 | 0.705 | 0.653 | 0.497 | 5.1 |
| 6 | 1.218 | 1.313 | 1.186 | 0.743 | 0.692 | 0.806 | 0.632 | 4.8 |
| 7 | 1.107 | 1.141 | 1.127 | 0.794 | 0.758 | 0.862 | 0.678 | 5.1 |
| 8 | 1.019 | 1.011 | 1.090 | 0.871 | 0.789 | 0.894 | 0.732 | 6.3 |
| 9 | 0.977 | 0.923 | 1.061 | 0.865 | 0.856 | 0.925 | 0.764 | 5.5 |
| 10 | 0.936 | 0.849 | 1.041 | 0.946 | 0.788 | 0.924 | 0.777 | 5.2 |
| 11 | 0.884 | 0.807 | 1.027 | 0.933 | 0.854 | 0.945 | 0.796 | 5.0 |
| 12 | 0.849 | 0.757 | 1.015 | 0.968 | 0.848 | 0.949 | 0.789 | 4.9 |
| 13 | 0.823 | 0.718 | 0.995 | 0.948 | 0.883 | 0.957 | 0.818 | 5.3 |
| 14 | 0.799 | 0.690 | 0.988 | 0.963 | 0.875 | 0.963 | 0.815 | 5.3 |
| 15 | 0.763 | 0.643 | 0.973 | 0.930 | 0.918 | 0.966 | 0.836 | 5.7 |
| 16 | 0.737 | 0.627 | 0.964 | 0.956 | 0.924 | 0.970 | 0.852 | 9.3 |
| 17 | 0.738 | 0.616 | 0.965 | 0.960 | 0.874 | 0.964 | 0.855 | 6.9 |
| 18 | 0.714 | 0.602 | 0.958 | 0.976 | 0.897 | 0.972 | 0.856 | 5.0 |
| 19 | 0.697 | 0.579 | 0.954 | 0.969 | 0.929 | 0.976 | 0.863 | 5.6 |
| 20 | 0.689 | 0.565 | 0.951 | 0.961 | 0.926 | 0.975 | 0.866 | 5.4 |
| 21 | 0.675 | 0.552 | 0.951 | 0.976 | 0.922 | 0.976 | 0.876 | 5.0 |
| 22 | 0.664 | 0.544 | 0.940 | 0.966 | 0.912 | 0.974 | 0.864 | 4.5 |
| 23 | 0.650 | 0.530 | 0.941 | 0.975 | 0.940 | 0.979 | 0.877 | 4.6 |
| 24 | 0.639 | 0.519 | 0.936 | 0.968 | 0.944 | 0.976 | 0.886 | 4.8 |
| 25 | 0.633 | 0.514 | 0.933 | 0.981 | 0.935 | 0.980 | 0.884 | 4.7 |
| 26 | 0.522 | 0.434 | 0.873 | 0.961 | 0.953 | 0.979 | 0.882 | 5.3 |
| 27 | 0.502 | 0.415 | 0.869 | 0.973 | 0.941 | 0.979 | 0.889 | 4.8 |
| 28 | 0.489 | 0.406 | 0.865 | 0.982 | 0.944 | 0.980 | 0.893 | 4.8 |
| 29 | 0.487 | 0.402 | 0.863 | 0.976 | 0.946 | 0.980 | 0.896 | 4.9 |
| 30 | 0.475 | 0.397 | 0.867 | 0.970 | 0.954 | 0.980 | 0.896 | 4.6 |

Best epoch 30, per class: circle 0.986 / 0.943, square 0.975 / 0.900, triangle 0.975 / 0.869, star 0.984 / 0.871
(mAP50 / mAP50-95). Epoch 1 includes compiling the GPU pipelines (14 s).
