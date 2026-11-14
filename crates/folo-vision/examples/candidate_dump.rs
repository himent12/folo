//! The pre-NMS candidate dump of hill-climb 3's `fusion` lane (docs/notes/hill/r3-fusion.md): runs a model over
//! the images of a COCO split with validation's preprocessing and decode (the GPU path `folo val` takes) and writes,
//! for every anchor with any class score `> conf`, its decoded box (network-input and original-image pixels) and its
//! `4 · reg_max` DFL logits, plus every `(anchor, class, score)` pair with score `> conf`. `bench/fusion/` rebuilds
//! FOLO's post-process from it.
//!
//! ```text
//! cargo run --release -p folo-vision --example candidate_dump -- \
//!     --model <weights.folo> --data D:/folo-data/coco --split val --out <dir> \
//!     [--list <file of image file names>] [--half] [--batch 32] [--conf 0.001] [--device gpu:0] \
//!     [--features] [--aug <seed>] [--input] [--stem]
//! ```
//!
//! Output folder:
//! - `dump.json`: model, imgsz, reg_max, classes, conf, the anchor levels (stride, height, width, start), dtype.
//! - `images.jsonl`: one line per image in loader order: dataset index, file, original size, letterbox ratio and
//!   pad, and the offsets and counts of its anchors and pairs.
//! - `anchors.bin`: per candidate anchor (in anchor order within an image) 4 + 4 + 4 + 4·reg_max little-endian
//!   words: anchor index (u32), class-max score (f32), 2 unused (0), the network box `x1 y1 x2 y2` (f32), the
//!   original-image box (f32, `ImageMeta::unmap_xyxy`, clipped), then the logits (f32).
//! - `pairs.bin`: per pair 3 words: the anchor's position in the image's anchor list (u32), class (u32), score
//!   (f32); in `(anchor, class)` order.
//! - `feats.bin` (with `--features`, lane `refine`, docs/notes/hill/r3-refine.md): per image the head's three
//!   inputs (MODEL.md §4 rows 16, 19, 22: P3, P4, P5) as f16, `[C, H, W]` each, one level after the other.
//! - `input.bin` (with `--input`): per image the network input as u8 `[3, H, W]` (`R G B`, after `--aug`'s
//!   flip and zoom): what the refiner's high-resolution probes read.
//! - `stem.bin` (with `--stem`, to check a re-implementation): per image layers 0 and 1 (the stem convs) as f32
//!   `[C, H, W]`, one after the other.
//!
//! `--aug <seed>` (lane `refine`'s training views): before the forward, each letterboxed image is flipped
//! horizontally with probability 1/2 and zoomed about the input's centre by `z` drawn uniformly from 0.75..1.25
//! (bilinear, gray 114 outside), both drawn from `(seed, dataset index)`. `images.jsonl` then also has `flip`,
//! `zoom` and `targets` (the loader's boxes after the same transform, `[class, x1, y1, x2, y2]` in network
//! pixels, unclipped); the original-image boxes in `anchors.bin` are meaningless for such a view.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

use folo_core::{DType, Tensor};
use folo_data::{DataLoader, Dataset, FileDataset, LoaderConfig, OpenOptions, Split};
use folo_nn::io::ModelFile;
use folo_vision::model::{FoloModel, HeadMode};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(arg(&args, "--model").ok_or("--model is required")?);
    let data = arg(&args, "--data").ok_or("--data is required")?;
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out is required")?);
    let split = match arg(&args, "--split").as_deref().unwrap_or("val") {
        "train" => Split::Train,
        "val" => Split::Val,
        s => return Err(format!("--split {s}: train or val").into()),
    };
    let batch: usize = arg(&args, "--batch").map_or(Ok(32), |s| s.parse())?;
    let conf: f32 = arg(&args, "--conf").map_or(Ok(0.001), |s| s.parse())?;
    let device = arg(&args, "--device").unwrap_or_else(|| "gpu:0".into());
    let half = args.iter().any(|a| a == "--half");
    let features = args.iter().any(|a| a == "--features");
    let aug: Option<u64> = arg(&args, "--aug").map(|s| s.parse()).transpose()?;
    let input = args.iter().any(|a| a == "--input");
    let stem = args.iter().any(|a| a == "--stem");

    if device.starts_with("gpu") {
        folo_gpu::init()?;
    }
    let device = folo_vision::train::resolve_device(&device)?;
    let file = ModelFile::open(&model_path)?;
    let dtype = if half { DType::F16 } else { DType::F32 };
    let model = FoloModel::from_file_for_inference(&file, device, dtype)?;
    let imgsz = model.imgsz();

    let ds = folo_data::open_splits(&data, &OpenOptions::default(), &[split])?;
    let part = ds.split(split);
    let part: FileDataset = match arg(&args, "--list") {
        None => part.clone(),
        Some(list) => {
            // Image file names (no folders), one per line; dataset order is kept.
            let want: std::collections::HashSet<String> = std::fs::read_to_string(&list)?
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            let entries: Vec<_> = part
                .entries()
                .iter()
                .filter(|e| e.image.file_name().is_some_and(|n| want.contains(n.to_string_lossy().as_ref())))
                .cloned()
                .collect();
            if entries.len() != want.len() {
                eprintln!("note: {} of the {} listed images are in the split", entries.len(), want.len());
            }
            FileDataset::new(split, part.class_names().to_vec(), entries)
        }
    };
    let ds: Arc<dyn Dataset> = Arc::new(part.clone());
    std::fs::create_dir_all(&out)?;
    let loader = DataLoader::new(
        ds.clone(),
        LoaderConfig {
            workers: folo_vision::train::default_workers(),
            u8_images: !device.is_cpu(),
            ..LoaderConfig::val(batch, imgsz)
        },
    )?;

    let mut anchors_f = BufWriter::new(std::fs::File::create(out.join("anchors.bin"))?);
    let mut pairs_f = BufWriter::new(std::fs::File::create(out.join("pairs.bin"))?);
    let mut images_f = BufWriter::new(std::fs::File::create(out.join("images.jsonl"))?);
    let mut input_f =
        if input { Some(BufWriter::new(std::fs::File::create(out.join("input.bin"))?)) } else { None };
    let mut stem_f =
        if stem { Some(BufWriter::new(std::fs::File::create(out.join("stem.bin"))?)) } else { None };
    let mut feats_f =
        if features { Some(BufWriter::new(std::fs::File::create(out.join("feats.bin"))?)) } else { None };
    let (mut n_anchors, mut n_pairs, mut n_images) = (0u64, 0u64, 0usize);
    let mut header = None;
    let t0 = std::time::Instant::now();
    for b in loader.epoch(0) {
        let mut b = b?;
        // As `folo_vision::val`: bytes are converted on the device, `value / 255` after the cast.
        let bytes = !b.images_u8.is_empty();
        let views: Vec<(bool, f32)> = match aug {
            None => vec![(false, 1.0); b.meta.len()],
            Some(seed) => {
                if !bytes {
                    return Err("--aug needs a GPU device (u8 images)".into());
                }
                let [n, c, h, w] = b.shape;
                let plane = c * h * w;
                (0..n)
                    .map(|i| {
                        let r = splitmix(seed ^ (b.meta[i].index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let flip = r & 1 == 1;
                        let zoom = 0.75 + 0.5 * ((r >> 11) as f64 / (1u64 << 53) as f64) as f32;
                        let src = b.images_u8[i * plane..(i + 1) * plane].to_vec();
                        warp(&src, &mut b.images_u8[i * plane..(i + 1) * plane], c, h, w, flip, zoom);
                        (flip, zoom)
                    })
                    .collect()
            }
        };
        if input && !bytes {
            return Err("--input needs a GPU device (u8 images)".into());
        }
        let input_bytes = if input { b.images_u8.clone() } else { Vec::new() };
        let x = if bytes {
            Tensor::from_vec(std::mem::take(&mut b.images_u8), b.shape)?
        } else {
            Tensor::from_vec(std::mem::take(&mut b.images), b.shape)?
        };
        let x = if device.is_cpu() { x } else { x.to_device(device)? };
        let x = x.to_dtype(model.dtype())?;
        let x = if bytes { x.div_scalar(255.0)? } else { x };
        let mut pyramid: Vec<Tensor> = Vec::new();
        let mut stem_out: Vec<Tensor> = Vec::new();
        let output = {
            let _g = folo_core::no_grad();
            if features || stem {
                assert_eq!(model.head_mode(), HeadMode::O2m, "--features needs the o2m branch");
                model.forward_traced(&x, &mut |l, t| {
                    if features && matches!(l, 16 | 19 | 22) {
                        pyramid.push(t.clone());
                    }
                    if stem && l <= 1 {
                        stem_out.push(t.clone());
                    }
                })?
            } else {
                model.forward_head(&x, HeadMode::O2m)?
            }
        };
        // The pyramid as f16 on the host, per level [B, C, H, W].
        let pyramid: Vec<(Vec<folo_core::f16>, usize)> = pyramid
            .iter()
            .map(|t| {
                let per = t.shape().dims()[1..].iter().product::<usize>();
                let v = t.to_dtype(DType::F16)?.to_device(folo_core::Device::Cpu)?.to_vec()?;
                Ok::<_, folo_core::Error>((v, per))
            })
            .collect::<Result<_, _>>()?;
        let stem_out: Vec<(Vec<f32>, usize)> = stem_out
            .iter()
            .map(|t| {
                let per = t.shape().dims()[1..].iter().product::<usize>();
                let v = t.to_dtype(DType::F32)?.to_device(folo_core::Device::Cpu)?.to_vec()?;
                Ok::<_, folo_core::Error>((v, per))
            })
            .collect::<Result<_, _>>()?;
        let (a, nc, rm) = (output.num_anchors(), output.num_classes, output.reg_max);
        if header.is_none() {
            header = Some(serde_json::json!({
                "model": model_path.display().to_string(),
                "imgsz": imgsz,
                "reg_max": rm,
                "num_classes": nc,
                "classes": model.classes(),
                "conf": conf,
                "batch": batch,
                "dtype": if half { "f16" } else { "f32" },
                "device": device.to_string(),
                "split": split.as_str(),
                "levels": output.anchors.levels.iter().map(|l| serde_json::json!({
                    "stride": l.stride, "height": l.height, "width": l.width, "start": l.start
                })).collect::<Vec<_>>(),
            }));
        }
        // The decode `folo val` post-processes on a GPU (HeadOutput::decoded_on_device), and the raw DFL logits.
        let decoded = if output.box_raw.device().is_cpu() {
            output.decode()?.to_vec::<f32>()?
        } else {
            output.decoded_on_device()?.to_vec::<f32>()?
        };
        let raw = output.box_raw.to_device(folo_core::Device::Cpu)?.to_dtype(DType::F32)?.to_vec::<f32>()?;
        let row = 4 + nc;
        for (i, meta) in b.meta.iter().enumerate() {
            let img = &decoded[i * a * row..(i + 1) * a * row];
            let rg = &raw[i * 4 * rm * a..(i + 1) * 4 * rm * a];
            let (a0, p0) = (n_anchors, n_pairs);
            let mut local = 0u32;
            for j in 0..a {
                let r = &img[j * row..(j + 1) * row];
                let s = &r[4..];
                if !s.iter().any(|&v| v > conf) {
                    continue;
                }
                let best = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let orig = meta.unmap_xyxy([r[0], r[1], r[2], r[3]]);
                let mut w: Vec<u32> = Vec::with_capacity(12 + 4 * rm);
                w.extend([j as u32, best.to_bits(), 0, 0]);
                w.extend(r[..4].iter().map(|v| v.to_bits()));
                w.extend(orig.iter().map(|v| v.to_bits()));
                w.extend((0..4 * rm).map(|k| rg[k * a + j].to_bits()));
                for v in w {
                    anchors_f.write_all(&v.to_le_bytes())?;
                }
                for (c, &v) in s.iter().enumerate() {
                    if v > conf {
                        for v in [local, c as u32, v.to_bits()] {
                            pairs_f.write_all(&v.to_le_bytes())?;
                        }
                        n_pairs += 1;
                    }
                }
                local += 1;
                n_anchors += 1;
            }
            if let Some(f) = input_f.as_mut() {
                let plane = b.shape[1] * b.shape[2] * b.shape[3];
                f.write_all(&input_bytes[i * plane..(i + 1) * plane])?;
            }
            if let Some(f) = stem_f.as_mut() {
                for (v, per) in &stem_out {
                    for x in &v[i * per..(i + 1) * per] {
                        f.write_all(&x.to_le_bytes())?;
                    }
                }
            }
            if let Some(f) = feats_f.as_mut() {
                for (v, per) in &pyramid {
                    for x in &v[i * per..(i + 1) * per] {
                        f.write_all(&x.to_bits().to_le_bytes())?;
                    }
                }
            }
            let mut line = serde_json::json!({
                "index": meta.index,
                "file": part.entries()[meta.index].image.file_name().map(|n| n.to_string_lossy().to_string()),
                "orig": meta.orig_size,
                "ratio": meta.ratio,
                "pad": meta.pad,
                "anchor_start": a0, "anchors": n_anchors - a0,
                "pair_start": p0, "pairs": n_pairs - p0,
            });
            if aug.is_some() {
                let (flip, zoom) = views[i];
                let (w, h) = (b.shape[3] as f32, b.shape[2] as f32);
                let tx = |x: f32| {
                    let x = if flip { w - x } else { x };
                    zoom * (x - w / 2.0) + w / 2.0
                };
                let ty = |y: f32| zoom * (y - h / 2.0) + h / 2.0;
                let targets: Vec<[f32; 5]> = b
                    .targets
                    .iter()
                    .filter(|t| t[0] as usize == i)
                    .map(|t| {
                        let (cx, cy, bw, bh) = (t[2] * w, t[3] * h, t[4] * w, t[5] * h);
                        let (xa, xb) = (tx(cx - bw / 2.0), tx(cx + bw / 2.0));
                        [t[1], xa.min(xb), ty(cy - bh / 2.0), xa.max(xb), ty(cy + bh / 2.0)]
                    })
                    .collect();
                line["flip"] = flip.into();
                line["zoom"] = zoom.into();
                line["targets"] = serde_json::to_value(targets)?;
            }
            writeln!(images_f, "{line}")?;
            n_images += 1;
        }
        if n_images % 512 < batch {
            eprintln!(
                "{n_images} images, {n_anchors} anchors, {n_pairs} pairs, {:.0} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    for f in [input_f.as_mut(), stem_f.as_mut(), feats_f.as_mut()].into_iter().flatten() {
        f.flush()?;
    }
    anchors_f.flush()?;
    pairs_f.flush()?;
    images_f.flush()?;
    let mut header = header.unwrap_or_default();
    header["images"] = n_images.into();
    header["anchor_records"] = n_anchors.into();
    header["pair_records"] = n_pairs.into();
    header["anchor_words"] = (12 + 4 * model.config().reg_max).into();
    header["features"] = features.into();
    header["input"] = input.into();
    header["stem"] = stem.into();
    header["aug"] = aug.into();
    std::fs::write(out.join("dump.json"), serde_json::to_string_pretty(&header)?)?;
    eprintln!(
        "done: {n_images} images, {n_anchors} anchors, {n_pairs} pairs in {:.0} s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// SplitMix64: one well-mixed 64-bit value from a seed.
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `dst = src` flipped horizontally (if `flip`), then zoomed by `zoom` about the centre: the network-pixel point
/// `x` of `src` lands at `zoom · (x' − w/2) + w/2` (`x' = w − x` when flipped), bilinear, gray 114 outside.
fn warp(src: &[u8], dst: &mut [u8], c: usize, h: usize, w: usize, flip: bool, zoom: f32) {
    let (wf, hf) = (w as f32, h as f32);
    for y in 0..h {
        let sy = ((y as f32 + 0.5) - hf / 2.0) / zoom + hf / 2.0 - 0.5;
        for x in 0..w {
            let mut sx = ((x as f32 + 0.5) - wf / 2.0) / zoom + wf / 2.0;
            if flip {
                sx = wf - sx;
            }
            let sx = sx - 0.5;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            for ch in 0..c {
                let px = |xx: f32, yy: f32| -> f32 {
                    if xx < 0.0 || yy < 0.0 || xx >= wf || yy >= hf {
                        114.0
                    } else {
                        src[ch * h * w + yy as usize * w + xx as usize] as f32
                    }
                };
                let v = (1.0 - fy) * ((1.0 - fx) * px(x0, y0) + fx * px(x0 + 1.0, y0))
                    + fy * ((1.0 - fx) * px(x0, y0 + 1.0) + fx * px(x0 + 1.0, y0 + 1.0));
                dst[ch * h * w + y * w + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}
