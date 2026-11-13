//! The pre-NMS candidate dump of hill-climb 3's `fusion` lane (docs/notes/hill/r3-fusion.md): runs a model over
//! the images of a COCO split with validation's preprocessing and decode (the GPU path `folo val` takes) and writes,
//! for every anchor with any class score `> conf`, its decoded box (network-input and original-image pixels) and its
//! `4 · reg_max` DFL logits, plus every `(anchor, class, score)` pair with score `> conf`. `bench/fusion/` rebuilds
//! FOLO's post-process from it.
//!
//! ```text
//! cargo run --release -p folo-vision --example candidate_dump -- \
//!     --model <weights.folo> --data D:/folo-data/coco --split val --out <dir> \
//!     [--list <file of image file names>] [--half] [--batch 32] [--conf 0.001] [--device gpu:0]
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
    let (mut n_anchors, mut n_pairs, mut n_images) = (0u64, 0u64, 0usize);
    let mut header = None;
    let t0 = std::time::Instant::now();
    for b in loader.epoch(0) {
        let mut b = b?;
        // As `folo_vision::val`: bytes are converted on the device, `value / 255` after the cast.
        let bytes = !b.images_u8.is_empty();
        let x = if bytes {
            Tensor::from_vec(std::mem::take(&mut b.images_u8), b.shape)?
        } else {
            Tensor::from_vec(std::mem::take(&mut b.images), b.shape)?
        };
        let x = if device.is_cpu() { x } else { x.to_device(device)? };
        let x = x.to_dtype(model.dtype())?;
        let x = if bytes { x.div_scalar(255.0)? } else { x };
        let output = {
            let _g = folo_core::no_grad();
            model.forward_head(&x, HeadMode::O2m)?
        };
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
            let line = serde_json::json!({
                "index": meta.index,
                "file": part.entries()[meta.index].image.file_name().map(|n| n.to_string_lossy().to_string()),
                "orig": meta.orig_size,
                "ratio": meta.ratio,
                "pad": meta.pad,
                "anchor_start": a0, "anchors": n_anchors - a0,
                "pair_start": p0, "pairs": n_pairs - p0,
            });
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
    anchors_f.flush()?;
    pairs_f.flush()?;
    images_f.flush()?;
    let mut header = header.unwrap_or_default();
    header["images"] = n_images.into();
    header["anchor_records"] = n_anchors.into();
    header["pair_records"] = n_pairs.into();
    header["anchor_words"] = (12 + 4 * model.config().reg_max).into();
    std::fs::write(out.join("dump.json"), serde_json::to_string_pretty(&header)?)?;
    eprintln!(
        "done: {n_images} images, {n_anchors} anchors, {n_pairs} pairs in {:.0} s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
