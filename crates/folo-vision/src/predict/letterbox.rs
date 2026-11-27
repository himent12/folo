//! Host and device letterbox batching and shared GPU image uploads for [`super::Predictor`].

use folo_core::{DType, Device, LetterboxImage, Tensor};
use folo_data::{Image, Letterbox};
use rayon::prelude::*;

use crate::error::{Error, Result};

/// `u8` pixel → network input. A product (not `/ 255`), so the device letterbox can match it bit for bit.
const INV_255: f32 = 1.0 / 255.0;

/// The `[B, 3, H, W]` network input (`0..1`) of `imgs` letterboxed with `boxes`, built on the host. Every
/// geometry must have the same output size (the predictor uses `rect` only for images that share their canvas).
///
/// One fused pass per image ([`Letterbox::apply_planar_f32`]: resize, padding and the `u8` → `f32` scaling,
/// rows in parallel) writes straight into the tensor's buffer, which comes from folo-core's buffer cache
/// ([`folo_core::cpu::take_buffer`]), so a steady stream of predictions reuses the same input memory instead
/// of faulting in fresh pages every call. Bit for bit the planes of [`Letterbox::apply`] times `1 / 255`.
pub fn letterbox_on_host(imgs: &[Image], boxes: &[Letterbox]) -> Result<Tensor> {
    let refs: Vec<&Image> = imgs.iter().collect();
    letterbox_on_host_refs(&refs, boxes)
}

pub(super) fn letterbox_on_host_refs(imgs: &[&Image], boxes: &[Letterbox]) -> Result<Tensor> {
    let [w, h] = check_letterbox(imgs, boxes)?;
    let plane = (w * h) as usize;
    let mut input = folo_core::cpu::take_buffer(imgs.len() * 3 * plane);
    input.par_chunks_mut(3 * plane).zip(imgs.par_iter().zip(boxes)).for_each(|(dst, (&img, lb))| {
        lb.apply_planar_f32(img, folo_data::PAD_VALUE, INV_255, dst);
    });
    Ok(Tensor::from_vec(input, [imgs.len(), 3, h as usize, w as usize])?)
}

/// Packed `[B, H, W]` `u32` `[R, G, B, 0]` letterbox canvases for the direct `u8` CPU int8 stem.
pub(super) fn letterbox_on_host_rgb0(
    imgs: &[&Image],
    boxes: &[Letterbox],
) -> Result<(Vec<u32>, usize, usize)> {
    let [w, h] = check_letterbox(imgs, boxes)?;
    let plane = (w * h) as usize;
    let mut out = folo_core::cpu::q8::alloc(imgs.len(), 4, plane);
    out[..imgs.len() * plane].par_chunks_mut(plane).zip(imgs.par_iter().zip(boxes)).for_each(
        |(dst, (&img, lb))| {
            lb.apply_rgb0_u8(img, folo_data::PAD_VALUE, dst);
        },
    );
    Ok((out, h as usize, w as usize))
}

/// [`letterbox_on_host`] on `device` from the raw `u8` images, bit-exact with it (`None` when the device has
/// no letterbox kernel): the images go up as they are, with folo-data's resize taps, and the canvas is
/// built there.
pub fn letterbox_on_device(imgs: &[Image], boxes: &[Letterbox], device: Device) -> Result<Option<Tensor>> {
    letterbox_on_device_as(imgs, boxes, device, DType::F32)
}

/// [`letterbox_on_device`] writing a canvas of `dtype` (`F16` for half-precision inference: the f32 value
/// rounded once).
pub fn letterbox_on_device_as(
    imgs: &[Image],
    boxes: &[Letterbox],
    device: Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    let refs: Vec<&Image> = imgs.iter().collect();
    letterbox_on_device_as_refs(&refs, boxes, device, dtype)
}

pub(super) fn letterbox_on_device_as_refs(
    imgs: &[&Image],
    boxes: &[Letterbox],
    device: Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    let [w, h] = check_letterbox(imgs, boxes)?;
    let mut taps: Vec<u32> = Vec::new();
    let mut geoms = Vec::with_capacity(imgs.len());
    for (&img, lb) in imgs.iter().zip(boxes) {
        let [rw, rh] = lb.resized;
        let plan = folo_data::resize::resize_plan(img.width, img.height, rw, rh);
        geoms.push(LetterboxImage {
            src: [img.width as usize, img.height as usize],
            resized: [rw as usize, rh as usize],
            pad: [lb.pad[0] as usize, lb.pad[1] as usize],
            prefilter: [plan.prefilter[0] as usize, plan.prefilter[1] as usize],
            taps: taps.len(),
        });
        for t in [&plan.x0, &plan.x1, &plan.wx, &plan.y0, &plan.y1, &plan.wy] {
            taps.extend_from_slice(t);
        }
    }
    let n = taps.len();
    let taps = Tensor::from_vec(taps, [n])?.to_device(device)?;
    let dev_imgs = upload_images(imgs, device)?;
    let refs: Vec<&Tensor> = dev_imgs.iter().collect();
    let out = [h as usize, w as usize];
    Ok(Tensor::letterbox_rgb8(&refs, &taps, &geoms, out, folo_data::PAD_VALUE, INV_255, dtype)?)
}

/// The `u8` images as `[H, W, 3]` tensors on `device`. On a folo-gpu device all of them share one upload: the
/// pixels are copied in parallel straight into reused, mapped staging memory ([`folo_gpu::GpuContext::upload_with`])
/// and each image is a view of the one device buffer (at a 4-byte aligned offset, which the letterbox kernel
/// reads in place). A batch larger than the device's largest storage buffer is uploaded in several such runs,
/// each as many images as fit. Elsewhere each image is uploaded on its own.
fn upload_images(imgs: &[&Image], device: Device) -> Result<Vec<Tensor>> {
    let Some(ctx) = folo_gpu::context_for(device) else {
        let host: Vec<Tensor> = imgs
            .par_iter()
            .map(|&im| Tensor::from_vec(im.data.clone(), [im.height as usize, im.width as usize, 3]))
            .collect::<folo_core::Result<_>>()?;
        return Ok(host.iter().map(|t| t.to_device(device)).collect::<folo_core::Result<_>>()?);
    };
    let caps = ctx.caps();
    let limit = caps.max_buffer_size.min(caps.max_storage_buffer_binding_size) as usize;
    let mut out = Vec::with_capacity(imgs.len());
    let mut rest = imgs;
    while !rest.is_empty() {
        // At least one image per run (one over the limit fails in the upload, as on its own).
        let mut total = rest[0].data.len().next_multiple_of(4);
        let mut n = 1;
        while n < rest.len() && total + rest[n].data.len().next_multiple_of(4) <= limit {
            total += rest[n].data.len().next_multiple_of(4);
            n += 1;
        }
        out.extend(upload_run(&ctx, &rest[..n], device)?);
        rest = &rest[n..];
    }
    Ok(out)
}

/// [`upload_images`]' one shared upload of `imgs` on a folo-gpu device.
fn upload_run(ctx: &folo_gpu::GpuContext, imgs: &[&Image], device: Device) -> Result<Vec<Tensor>> {
    let mut offsets = Vec::with_capacity(imgs.len());
    let mut total = 0usize;
    for &im in imgs {
        offsets.push(total);
        total += im.data.len().next_multiple_of(4);
    }
    // `WriteOnly<[u8]>` is not `Send`, its fixed-size chunks are: the mapping is filled in 64 KiB chunks in
    // parallel, each from the image bytes it covers (the padding between images stays unwritten).
    const CHUNK: usize = 64 * 1024;
    let fill = |start: usize, mut dst: folo_gpu::WriteOnly<'_, [u8]>| {
        let end = start + dst.len();
        for (&im, &off) in imgs.iter().zip(&offsets) {
            let (a, b) = (start.max(off), end.min(off + im.data.len()));
            if a < b {
                dst.slice(a - start..b - start).copy_from_slice(&im.data[a - off..b - off]);
            }
        }
    };
    let (buf, ()) = ctx
        .upload_with(total as u64, |dst| {
            let body = dst.len() / CHUNK * CHUNK;
            let (chunks, tail) = dst.split_at(body);
            let (chunks, _) = chunks.into_chunks::<CHUNK>();
            let chunks: Vec<_> = chunks.into_iter().enumerate().collect();
            chunks.into_par_iter().for_each(|(i, c)| fill(i * CHUNK, c.into()));
            fill(body, tail);
        })
        .map_err(folo_core::Error::from)?;
    let all = folo_gpu::tensor_from_buffer(device, DType::U8, [total], buf)?;
    Ok(imgs
        .iter()
        .zip(offsets)
        .map(|(&im, off)| {
            all.narrow(0, off, im.data.len())?.reshape([im.height as usize, im.width as usize, 3])
        })
        .collect::<folo_core::Result<_>>()?)
}

/// The common canvas `[width, height]` of a letterbox batch: one geometry per image, computed for its size,
/// all with the same output size.
fn check_letterbox(imgs: &[&Image], boxes: &[Letterbox]) -> Result<[u32; 2]> {
    let first = boxes.first().ok_or_else(|| Error::Config("letterbox: no images".into()))?;
    if imgs.len() != boxes.len() {
        return Err(Error::Config(format!(
            "letterbox: {} images but {} geometries",
            imgs.len(),
            boxes.len()
        )));
    }
    for (&img, lb) in imgs.iter().zip(boxes) {
        if lb.orig != [img.width, img.height] || lb.out != first.out {
            return Err(Error::Config(format!(
                "letterbox: geometry {lb:?} does not fit a {}x{} image on a {:?} canvas",
                img.width, img.height, first.out
            )));
        }
    }
    Ok(first.out)
}
