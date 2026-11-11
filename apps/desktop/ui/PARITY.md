# folo-ui: pixel parity with the GPUI app

How close `folo-desktop --shot` (headless, CPU renderer) is to the GPUI 0.2.2 app, what rule each part follows,
and every pixel that still differs, with its cause. The reference is the GPUI build's off-screen captures
(`D:/folo-build/dfw-baseline/shots/a`: 98 pairs, scale 1.5; its own run-to-run noise is at most 6 levels, in
the breathing status dot and ±1 inside some thumbnails).

## Measuring

```
tools\shot-all.ps1 -Dir <dir>                      # the 98 pairs, headless, ~5 s (release build)
python tools\parity.py --a <gpui shots> --b <dir> --out <report> --tol 8   # and --tol 2
FOLO_UI_SCENE_DUMP=<file> folo-desktop --scenario <s> ... --shot <png>     # the glyph blend + every primitive
```

## Result (98 pairs, % of pixels whose largest channel difference exceeds the tolerance)

| | worst % > 8 | median % > 8 | mean % > 8 | pairs clean at 8 | worst % > 2 | median % > 2 | pairs clean at 2 | largest difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before (`desktop/framework` 891f8ba9) | 0.8135 | 0.1349 | 0.1834 | 0 | 1.2698 | 0.2835 | 0 | 231 |
| now | 0.0040 | 0.0000 | 0.0001 | 71 | 0.0132 | 0.0000 | 62 | 102 |

## The rules, as GPUI's DX11 renderer applies them

Each is read from GPUI's source (`vendor/gpui/src/platform/windows/shaders.hlsl`, `directx_renderer.rs`,
`directx_atlas.rs`, `window.rs`, `path_builder.rs`) and pinned by a test.

- **Clip and edge pixels** (`raster/geom.rs`). `SV_ClipDistance` clips the triangles themselves, so a clip
  edge is rasterized like any edge: a pixel centre exactly on a left / top edge is in, on a right / bottom
  edge out, after D3D11 snaps positions to 1/256 px (a clip at 1553.5001 is 1553.5). Before, clips kept
  centres on both edges: every 1.5 px card border side (four clipped quads in GPUI's `Style::paint`) drew
  an extra half-covered column. Tests: `geom::tests::spans_follow_pixel_centres`,
  `tests/raster/exact.rs::clips_follow_the_top_left_rule`.
- **SVG icons** (`raster/sprite.rs`, `draw_svg`). `monochrome_sprite_fragment` applies the glyph contrast /
  gamma correction to every monochrome sprite: icons get the colour's glyph table, interpolated at the
  filtered sample. Test: `sprite::tests::svg_samples_get_the_glyph_correction`.
- **Images** (`raster/image.rs`). An image is a padding-free tile of a zero-initialised atlas sampled
  linearly with WRAP addressing: a magnified image's rim blends with transparent black (a side of 1024 px
  or more spans its own texture and wraps to its opposite edge). Bounds: origin floored, size ceiled (as
  before). Tests: `image::tests::magnified_edges_blend_with_the_empty_atlas`, `wide_images_wrap`.
- **Paths** (`path_builder.rs`, `raster/triangles.rs`, `raster/path.rs`). `PathBuilder::build` is GPUI's:
  the lyon path (lyon 1.0.19, tessellation 1.0.23, algorithms 1.0.22, the versions GPUI's lockfile has),
  dashed with lyon's path measurements, tessellated in logical px with u16 indices; the bounds are the
  vertices' box. The renderer samples each triangle at D3D11's standard 4x MSAA positions with the
  top-left rule and 1/256 px snapping, clips per sample, counts the triangles over each sample (a
  translucent stroke blends again where lyon's triangles overlap) and resolves with 8-bit rounding.
  Before: tiny-skia's analytic coverage, SVG's miter limit (lyon bevels only past twice its limit), and
  bounds that cut miters. Tests: `triangles::tests::*`, `path::tests::*`, `tests/style/path.rs`.
- **Dashed borders** (`raster/quad_eval.rs`). `max_t * dash_velocity - dash_length` is one `mad` on the GPU,
  executed fused: a 21 px side of a 1.5 px dashed box has the dash count the capture shows (two roundings
  give 3.9999998, one fewer dash). Test: `quad_tests::dashed_side_counts_dashes_like_the_gpu`.
- **Scroll thumbs** (`apps/desktop/src/ui/scroll.rs`). The 800 ms after a scroll are timed on the
  executor's clock, so a shot settles like the real app (the thumb has hidden by the 2.2 s capture).
- **Unchanged and verified:** quad / border / shadow formulas, `From<Hsla> for Rgba` (equal to the shader's
  `hsla_to_rgba`; flat areas match exactly, translucent ones included), glyph masks (DirectWrite, the same
  calls) and their correction: a float32 emulation of `monochrome_sprite_fragment` over the dumped masks
  reproduces our pixels exactly (607 of 607 in the settings label row).

## What still differs, and why it stays

Causes in the table: **D** status dot, **P** paths, **I** image atlas neighbours, **U** unexplained,
**-** nothing above 2 levels. Every pair also has 1-level differences (cause T).

- **D, the breathing status dot** (9x9 or 12x12 px, up to 10 levels). It pulses on a 2400 ms repeating
  animation; the GPUI captures catch it at an arbitrary phase (it differs between GPUI's own runs by up to
  6), the shot at a fixed 2200 ms. Noise of the reference, accepted by the brief.
- **P, paths** (chart lines and areas; 0.0003 % or less of a chart page above 8):
  - single pixels one sample (a quarter of the line's contrast, up to 59 levels) different, at path ends
    and where a path's samples fall just outside its own bounds. GPUI draws all paths of a batch into one
    MSAA target and copies it through the union of their clipped bounds when their draw orders differ
    (`draw_paths_from_intermediate`), so such samples show there; modelling it needs GPUI's per-primitive
    draw orders (its `BoundsTree`) and batching. The largest (102 in run-failed) is a dashed marker over a
    line end.
  - up to 7 levels where translucent series overlap (the raw lines under the smoothed ones): GPUI blends the
    paths of a batch per sample before the resolve; here each path is resolved, then blended. Exact would
    need a per-batch 4-sample buffer in the tile renderer.
- **I, thumbnail edges** (1680x1050 overview, one column at an image's edge, up to 21 levels): small images
  share an atlas, and the sampler reads the neighbouring tile's texels (etagere's packing and the load order
  decide which) where we read transparent black. Large images (inspector, playground) match within 2.
- **U, under detection-box outlines** (dataset, datasets, inspector: 1 to 6 pixels each, 13 levels): GPUI's
  pixel is pure black where the 0.8-alpha outline lies over an image pixel we both draw alike elsewhere;
  stable across GPUI's runs. Not explained.
- **T, 1 level on text and icon edges** (every pair, never above 2): GPUI samples glyph and icon masks
  through a LINEAR sampler at interpolated atlas coordinates; the GPU's 8-bit filter weights and the
  interpolation error pull 1/256 (sometimes 2/256) of a neighbouring texel into about 29 % of glyph pixels
  (settings label row: 434 of 607 ink pixels exact, and every other one is reproduced exactly by a 1/256 or
  2/256 bleed from its left, right, upper or lower texel). The direction depends on where each glyph's tile
  sits in GPUI's atlas, so it is not reproducible; it never exceeds 1 level on text.

## Per pair (final run; "before" is the `desktop/framework` build this work started from)

| pair | before % > 8 | % > 8 | % > 2 | max | cause |
|---|---:|---:|---:|---:|---|
| 960x620/run-live | 0.4105 | 0.0040 | 0.0072 | 57 | DP |
| 1680x1050/overview | 0.1317 | 0.0023 | 0.0122 | 59 | DIP |
| 1280x800/run-failed | 0.4068 | 0.0009 | 0.0012 | 102 | P |
| 960x620/dataset | 0.1641 | 0.0007 | 0.0007 | 13 | U |
| 960x620/run-cloud+training | 0.3802 | 0.0004 | 0.0116 | 57 | DP |
| 960x620/run-live+charts-focus | 0.4163 | 0.0004 | 0.0072 | 57 | DP |
| 1280x800/run-done | 0.6105 | 0.0003 | 0.0132 | 57 | P |
| 1280x800/run-cloud+training+light | 0.2990 | 0.0003 | 0.0131 | 55 | DP |
| 1280x800/run-cloud+released | 0.4615 | 0.0003 | 0.0101 | 59 | P |
| 1280x800/dataset+light | 0.0828 | 0.0003 | 0.0003 | 13 | U |
| 1280x800/dataset | 0.0934 | 0.0003 | 0.0003 | 13 | U |
| 1280x800/run-live | 0.3351 | 0.0001 | 0.0057 | 58 | DP |
| 1280x800/run-live+hover | 0.3276 | 0.0001 | 0.0057 | 58 | DP |
| 1280x800/run-live+light | 0.3459 | 0.0001 | 0.0055 | 54 | DP |
| 1280x800/stop-confirm+light | 0.1726 | 0.0001 | 0.0042 | 33 | DP |
| 1280x800/datasets | 0.1115 | 0.0001 | 0.0001 | 13 | U |
| 1280x800/stop-confirm | 0.0762 | 0.0001 | 0.0001 | 20 | P |
| 1280x800/run-cloud+training | 0.2910 | 0.0001 | 0.0109 | 57 | DP |
| 1280x800/model+light | 0.2260 | 0.0001 | 0.0001 | 52 | P |
| 1280x800/model | 0.2404 | 0.0001 | 0.0001 | 59 | P |
| 1680x1050/run-cloud+training | 0.2264 | 0.0001 | 0.0054 | 59 | DP |
| 1280x800/run-done+sample-focus | 0.8135 | 0.0000 | 0.0077 | 53 | P |
| 1280x800/run-live+charts-focus | 0.3400 | 0.0000 | 0.0053 | 36 | DP |
| 1280x800/inspector+light | 0.3513 | 0.0000 | 0.0000 | 13 | U |
| 1280x800/inspector | 0.3538 | 0.0000 | 0.0000 | 13 | U |
| 1680x1050/run-live | 0.2204 | 0.0000 | 0.0073 | 53 | DP |
| 1680x1050/model | 0.2571 | 0.0000 | 0.0000 | 55 | P |
| 1280x800/overview+cloud | 0.1424 | 0.0000 | 0.0083 | 5 | D |
| 1280x800/kb-focus | 0.1423 | 0.0000 | 0.0060 | 7 | D |
| 1280x800/overview+light | 0.1263 | 0.0000 | 0.0060 | 5 | D |
| 1280x800/toast | 0.1560 | 0.0000 | 0.0060 | 5 | D |
| 1280x800/run-cloud+syncing | 0.1189 | 0.0000 | 0.0052 | 4 | D |
| 960x620/overview | 0.2112 | 0.0000 | 0.0052 | 8 | D |
| 1280x800/overview | 0.1423 | 0.0000 | 0.0048 | 3 | D |
| 1280x800/runs | 0.1285 | 0.0000 | 0.0030 | 7 | D |
| 1280x800/palette+light | 0.0688 | 0.0000 | 0.0025 | 4 | D |
| 1280x800/bench+light | 0.0691 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/bench-design+proposal | 0.0843 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/bench-design | 0.0894 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/bench-entry+bottom | 0.2071 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/bench-entry+light | 0.0762 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/bench-entry | 0.0901 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/bench-loop-new | 0.1510 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/bench-loops | 0.1322 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/bench-new | 0.1354 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/bench-plan | 0.0873 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/bench | 0.0800 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/benchmarks+empty | 0.0609 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/benchmarks+light | 0.1046 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/benchmarks | 0.1155 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/cloud-connect+light | 0.1124 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-connect+rejected | 0.0712 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-connect+remote+light | 0.1123 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-connect+remote | 0.0662 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-connect | 0.0698 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-disconnect+light | 0.1306 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-disconnect | 0.0479 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-terminate+light | 0.1256 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/cloud-terminate | 0.0430 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/dataset+filter | 0.0963 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/inspector+sample | 0.3223 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/model+table-focus | 0.1132 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/models | 0.1292 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/new-run+amp | 0.0783 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/new-run+cloud+light | 0.1075 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/new-run+cloud | 0.0818 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/new-run+light | 0.1049 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/new-run | 0.0793 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/overview+empty | 0.0831 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/overview+error | 0.0623 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/overview+loading | 0.0809 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/palette | 0.0599 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/playground+canvas-focus | 0.3565 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/playground+empty | 0.0683 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/playground+light | 0.2967 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/playground+nms | 0.3088 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/playground | 0.3120 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/run-done+bottom | 0.1785 | 0.0000 | 0.0000 | 1 | - |
| 1280x800/runs+cloud | 0.1344 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings+cloud+empty | 0.1626 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings+cloud+light | 0.1972 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings+cloud+trouble+light | 0.1993 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings+cloud+trouble | 0.2225 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings+cloud | 0.2228 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings+reference | 0.2207 | 0.0000 | 0.0000 | 2 | - |
| 1280x800/settings | 0.2207 | 0.0000 | 0.0000 | 2 | - |
| 1680x1050/overview+empty | 0.0461 | 0.0000 | 0.0000 | 1 | - |
| 1680x1050/settings+cloud | 0.0726 | 0.0000 | 0.0000 | 2 | - |
| 960x620/cloud-connect+remote | 0.0984 | 0.0000 | 0.0000 | 2 | - |
| 960x620/cloud-connect | 0.1061 | 0.0000 | 0.0000 | 2 | - |
| 960x620/models | 0.2183 | 0.0000 | 0.0000 | 1 | - |
| 960x620/new-run+cloud | 0.1231 | 0.0000 | 0.0000 | 2 | - |
| 960x620/new-run | 0.1290 | 0.0000 | 0.0000 | 2 | - |
| 960x620/overview+empty | 0.1453 | 0.0000 | 0.0000 | 1 | - |
| 960x620/playground | 0.2445 | 0.0000 | 0.0000 | 2 | - |
| 960x620/run-cloud+syncing | 0.1368 | 0.0000 | 0.0000 | 1 | - |
| 960x620/settings+cloud+trouble | 0.2604 | 0.0000 | 0.0000 | 2 | - |
| 960x620/settings+cloud | 0.2499 | 0.0000 | 0.0000 | 2 | - |
