// The mock's conditions as pixels: a rendered benchmark image (./render) changed on a canvas so each condition is
// recognisable and grows with its severity, then encoded as a JPEG at quality 95 (JPEG artifacts at a quality of their
// own), as the server serves an image under a condition. Approximations of crates/folo-bench's operations, close
// enough to look at; the geometry (`distance`'s boxes, `occlusion`'s patches) is ./benchConditions', so the truth the
// API returns lines up with what is drawn.

import type { Box } from '../types';
import { distanceScale, occlusionPatches } from './benchConditions';
import { hash, rng } from './rng';

type Ctx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
type Canvas = HTMLCanvasElement | OffscreenCanvas;

function makeCanvas(w: number, h: number): { canvas: Canvas; ctx: Ctx } {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(w, h);
    return { canvas, ctx: canvas.getContext('2d') as OffscreenCanvasRenderingContext2D };
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  return { canvas, ctx: canvas.getContext('2d') as CanvasRenderingContext2D };
}

function encode(canvas: Canvas, quality: number): Promise<Blob> {
  if ('convertToBlob' in canvas) return canvas.convertToBlob({ type: 'image/jpeg', quality });
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', quality));
}

/** Gaussian-ish noise of standard deviation `sigma` (levels) over every pixel, some of it per channel. */
function addNoise(ctx: Ctx, w: number, h: number, sigma: number, seed: number, chroma = 0.35) {
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const r = rng(seed);
  const g = () => (r() + r() + r() - 1.5) * 2;
  for (let i = 0; i < d.length; i += 4) {
    const l = g() * sigma;
    d[i] = d[i]! + l + g() * sigma * chroma;
    d[i + 1] = d[i + 1]! + l + g() * sigma * chroma;
    d[i + 2] = d[i + 2]! + l + g() * sigma * chroma;
  }
  ctx.putImageData(img, 0, 0);
}

export interface ConditionRender {
  condition: string;
  severity: number;
  /** The image's true boxes at full size (`occlusion` draws over them). */
  boxes: readonly Box[];
  /** The full-size width, to scale boxes and effects to a thumbnail. */
  fullWidth: number;
  /** Seeds where noise and patches fall (the benchmark's seed and the image). */
  seed: string;
}

/** The image `base` under a condition, as a JPEG. */
export async function renderCondition(base: Blob, o: ConditionRender): Promise<Blob> {
  const bmp = await createImageBitmap(base);
  const W = bmp.width;
  const H = bmp.height;
  const { canvas, ctx } = makeCanvas(W, H);
  const s = Math.max(1, Math.min(5, o.severity));
  const k = W / o.fullWidth;
  const short = Math.min(W, H);
  const seed = hash(`${o.seed}:${o.condition}:${s}`);
  let quality = 0.95;
  switch (o.condition) {
    case 'low_light':
      ctx.filter = `brightness(${1 - 0.16 * s}) contrast(${1 - 0.05 * s}) saturate(${1 - 0.1 * s})`;
      ctx.drawImage(bmp, 0, 0);
      ctx.filter = 'none';
      addNoise(ctx, W, H, 2 + 1.6 * s, seed, 0.6);
      break;
    case 'overexposure':
      ctx.filter = `brightness(${1 + 0.24 * s}) contrast(${1 - 0.07 * s}) saturate(${1 - 0.1 * s})`;
      ctx.drawImage(bmp, 0, 0);
      break;
    case 'low_contrast':
      ctx.filter = `contrast(${1 - 0.16 * s}) saturate(${1 - 0.08 * s})`;
      ctx.drawImage(bmp, 0, 0);
      break;
    case 'defocus':
      ctx.filter = `blur(${(0.0045 * s * short).toFixed(2)}px)`;
      ctx.drawImage(bmp, 0, 0);
      break;
    case 'motion_blur': {
      // The average of copies along a horizontal streak as long as 1.2% of the width per severity.
      const len = 0.012 * s * W;
      const n = 14;
      for (let i = 0; i < n; i++) {
        ctx.globalAlpha = 1 / (i + 1);
        ctx.drawImage(bmp, -len / 2 + (len * i) / (n - 1), 0);
      }
      ctx.globalAlpha = 1;
      break;
    }
    case 'sensor_noise':
      ctx.drawImage(bmp, 0, 0);
      addNoise(ctx, W, H, 5 * s, seed);
      break;
    case 'haze':
      ctx.filter = `blur(${(0.0008 * s * short).toFixed(2)}px) contrast(${1 - 0.06 * s})`;
      ctx.drawImage(bmp, 0, 0);
      ctx.filter = 'none';
      ctx.fillStyle = `rgba(206, 212, 214, ${0.135 * s})`;
      ctx.fillRect(0, 0, W, H);
      break;
    case 'color_cast':
      ctx.drawImage(bmp, 0, 0);
      ctx.globalCompositeOperation = 'multiply';
      ctx.fillStyle = `rgb(${Math.round(255 - 30 * s)}, ${Math.round(255 - 4 * s)}, ${Math.round(255 - 34 * s)})`;
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = `rgba(52, 120, 64, ${0.045 * s})`;
      ctx.fillRect(0, 0, W, H);
      break;
    case 'jpeg':
      ctx.drawImage(bmp, 0, 0);
      quality = [0.32, 0.18, 0.1, 0.05, 0.02][s - 1]!;
      break;
    case 'low_resolution': {
      // Down to 1 / (1 + 0.75 s) of the side, then back up with no smoothing: the blocks are the lost pixels.
      const f = 1 / (1 + 0.75 * s);
      const small = makeCanvas(Math.max(1, Math.round(W * f)), Math.max(1, Math.round(H * f)));
      small.ctx.drawImage(bmp, 0, 0, small.canvas.width, small.canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(small.canvas, 0, 0, W, H);
      ctx.imageSmoothingEnabled = true;
      break;
    }
    case 'distance': {
      // The scene shrinks towards its centre; what surrounds it is the image itself, blurred and dimmed.
      const f = distanceScale(s);
      ctx.filter = `blur(${(0.02 * short).toFixed(1)}px) brightness(0.82) saturate(0.8)`;
      ctx.drawImage(bmp, -W * 0.05, -H * 0.05, W * 1.1, H * 1.1);
      ctx.filter = 'none';
      ctx.drawImage(bmp, (W * (1 - f)) / 2, (H * (1 - f)) / 2, W * f, H * f);
      break;
    }
    case 'occlusion': {
      ctx.drawImage(bmp, 0, 0);
      const r = rng(seed);
      for (const p of occlusionPatches(o.boxes, s, o.seed)) {
        const x = p.x * k;
        const y = p.y * k;
        const w = p.w * k;
        const h = p.h * k;
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.45)';
        ctx.shadowBlur = Math.min(w, h) * 0.25;
        ctx.shadowOffsetY = Math.min(w, h) * 0.06;
        if (p.round) {
          // A leaf.
          const g = ctx.createLinearGradient(x, y, x + w, y + h);
          g.addColorStop(0, 'hsl(118 34% 38%)');
          g.addColorStop(1, 'hsl(128 40% 20%)');
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, (r() - 0.5) * 0.6, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // A plain object in front.
          ctx.fillStyle = `hsl(28 ${12 + r() * 10}% ${26 + r() * 12}%)`;
          ctx.beginPath();
          ctx.roundRect(x, y, w, h, Math.min(w, h) * 0.18);
          ctx.fill();
        }
        ctx.restore();
      }
      break;
    }
    default:
      ctx.drawImage(bmp, 0, 0);
  }
  ctx.filter = 'none';
  bmp.close();
  return encode(canvas, quality);
}
