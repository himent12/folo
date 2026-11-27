// Draws mock dataset images on a canvas: coloured shapes on soft backgrounds, or metal fasteners on a conveyor.
// The objects fill exactly the boxes from ./datasets, so ground truth overlays line up.

import type { Box, DatasetImage } from '../types';
import { seedOf, type MockDatasetSpec } from './datasets';
import { range, rng } from './rng';

type Ctx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement | OffscreenCanvas; ctx: Ctx } {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(w, h);
    return { canvas, ctx: canvas.getContext('2d') as OffscreenCanvasRenderingContext2D };
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  return { canvas, ctx: canvas.getContext('2d') as CanvasRenderingContext2D };
}

/** Renders the image to a JPEG blob; `maxSide` downsizes (thumbnails). */
export async function renderImage(spec: MockDatasetSpec, img: DatasetImage, index: number, maxSide?: number): Promise<Blob> {
  const scale = maxSide ? Math.min(1, maxSide / Math.max(img.width, img.height)) : 1;
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.scale(scale, scale);
  const r = rng(seedOf(spec) ^ Math.imul(index + 7, 40503));
  if (spec.style === 'fasteners') drawConveyor(ctx, img, r);
  else if (spec.style === 'pond') drawPond(ctx, img, r);
  else drawBackdrop(ctx, img, r, spec.id === 'ds_shapes_hard');
  for (const b of img.boxes) {
    const cls = spec.classes[b.class] ?? '';
    if (spec.style === 'fasteners') drawFastener(ctx, cls, b, r);
    else if (spec.style === 'pond') drawCreature(ctx, cls, b, r);
    else drawShape(ctx, spec.renderShapes?.[b.class] ?? cls.toLowerCase(), b, r);
  }
  if (spec.style === 'fasteners') vignette(ctx, img.width, img.height, 0.55);
  if (spec.style === 'pond') {
    // The water over everything: a faint green film and the light from above, so the fish sit in it.
    ctx.fillStyle = 'rgba(40, 92, 70, 0.14)';
    ctx.fillRect(0, 0, img.width, img.height);
    vignette(ctx, img.width, img.height, 0.4);
  }
  if ('convertToBlob' in canvas) return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.86 });
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', 0.86),
  );
}

// ---------- shapes ----------

function drawBackdrop(ctx: Ctx, img: DatasetImage, r: () => number, hard: boolean) {
  const { width: W, height: H } = img;
  const hue = range(r, 0, 360);
  const g = ctx.createLinearGradient(0, 0, W * range(r, 0.3, 1), H);
  g.addColorStop(0, `hsl(${hue} ${range(r, 12, 30)}% ${range(r, 70, 86)}%)`);
  g.addColorStop(1, `hsl(${(hue + range(r, 30, 90)) % 360} ${range(r, 10, 26)}% ${range(r, 46, 66)}%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  // Soft blobs for texture.
  for (let i = 0; i < 6; i++) {
    const x = range(r, 0, W);
    const y = range(r, 0, H);
    const rad = range(r, W * 0.15, W * 0.5);
    const rg = ctx.createRadialGradient(x, y, 0, x, y, rad);
    rg.addColorStop(0, `hsla(${(hue + range(r, -60, 60) + 360) % 360} 40% ${range(r, 50, 90)}% / 0.35)`);
    rg.addColorStop(1, 'hsla(0 0% 50% / 0)');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, W, H);
  }
  if (hard) {
    // Clutter: thin random strokes that look like distractors.
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 40; i++) {
      ctx.strokeStyle = `hsla(${range(r, 0, 360)} 30% ${range(r, 30, 70)}% / 0.35)`;
      ctx.beginPath();
      const x = range(r, 0, W);
      const y = range(r, 0, H);
      ctx.moveTo(x, y);
      ctx.quadraticCurveTo(x + range(r, -80, 80), y + range(r, -80, 80), x + range(r, -120, 120), y + range(r, -120, 120));
      ctx.stroke();
    }
  }
}

function shapePath(ctx: Ctx, cls: string, b: Box) {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const cx = b.x1 + w / 2;
  const cy = b.y1 + h / 2;
  ctx.beginPath();
  switch (cls) {
    case 'circle':
      ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, Math.PI * 2);
      break;
    case 'square': {
      const rr = Math.min(w, h) * 0.08;
      ctx.roundRect(b.x1, b.y1, w, h, rr);
      break;
    }
    case 'triangle':
      ctx.moveTo(cx, b.y1);
      ctx.lineTo(b.x2, b.y2);
      ctx.lineTo(b.x1, b.y2);
      ctx.closePath();
      break;
    case 'star':
      for (let i = 0; i < 10; i++) {
        const a = -Math.PI / 2 + (i * Math.PI) / 5;
        const rad = i % 2 === 0 ? 1 : 0.42;
        const x = cx + Math.cos(a) * (w / 2) * rad;
        const y = cy + (Math.sin(a) * (h / 2) * rad + h * 0.05);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      break;
    case 'ring': {
      const rx = w / 2;
      const ry = h / 2;
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.moveTo(cx + rx * 0.52, cy);
      ctx.ellipse(cx, cy, rx * 0.52, ry * 0.52, 0, 0, Math.PI * 2, true);
      break;
    }
    case 'heart': {
      const topY = b.y1 + h * 0.3;
      ctx.moveTo(cx, b.y2);
      ctx.bezierCurveTo(b.x1 - w * 0.05, b.y1 + h * 0.62, b.x1, b.y1 + h * 0.06, cx, topY);
      ctx.bezierCurveTo(b.x2, b.y1 + h * 0.06, b.x2 + w * 0.05, b.y1 + h * 0.62, cx, b.y2);
      ctx.closePath();
      break;
    }
    case 'hexagon':
      for (let i = 0; i < 6; i++) {
        const a = (i * Math.PI) / 3;
        const x = cx + Math.cos(a) * (w / 2);
        const y = cy + Math.sin(a) * (h / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      break;
    default: {
      // cross
      const t = 0.32;
      ctx.rect(b.x1, cy - (h * t) / 2, w, h * t);
      ctx.rect(cx - (w * t) / 2, b.y1, w * t, h);
    }
  }
}

function drawShape(ctx: Ctx, cls: string, b: Box, r: () => number) {
  const fillRule: CanvasFillRule = cls === 'ring' ? 'evenodd' : 'nonzero';
  const hue = range(r, 0, 360);
  const g = ctx.createLinearGradient(b.x1, b.y1, b.x2, b.y2);
  g.addColorStop(0, `hsl(${hue} ${range(r, 55, 85)}% ${range(r, 55, 66)}%)`);
  g.addColorStop(1, `hsl(${(hue + 20) % 360} ${range(r, 55, 85)}% ${range(r, 32, 44)}%)`);
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.28)';
  ctx.shadowBlur = Math.min(b.x2 - b.x1, 40) * 0.25;
  ctx.shadowOffsetY = Math.min(b.y2 - b.y1, 40) * 0.08;
  ctx.fillStyle = g;
  shapePath(ctx, cls, b);
  ctx.fill(fillRule);
  ctx.restore();
  ctx.save();
  shapePath(ctx, cls, b);
  ctx.clip(fillRule);
  const hl = ctx.createRadialGradient(b.x1 + (b.x2 - b.x1) * 0.3, b.y1 + (b.y2 - b.y1) * 0.25, 0, b.x1, b.y1, (b.x2 - b.x1) * 0.9);
  hl.addColorStop(0, 'rgba(255,255,255,0.35)');
  hl.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = hl;
  ctx.fillRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
  ctx.restore();
}

// ---------- fasteners ----------

function drawConveyor(ctx: Ctx, img: DatasetImage, r: () => number) {
  const { width: W, height: H } = img;
  const base = range(r, 16, 24);
  ctx.fillStyle = `hsl(210 8% ${base}%)`;
  ctx.fillRect(0, 0, W, H);
  // Belt texture: fine diagonal weave.
  ctx.globalAlpha = 0.07;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  for (let x = -H; x < W; x += 6) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + H, H);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // Belt seams.
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  const seam = range(r, 0, W);
  ctx.fillRect(seam, 0, 3, H);
  // Overhead light.
  const lg = ctx.createRadialGradient(W * range(r, 0.3, 0.7), H * 0.35, 0, W * 0.5, H * 0.5, W * 0.75);
  lg.addColorStop(0, 'rgba(255,248,235,0.16)');
  lg.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = lg;
  ctx.fillRect(0, 0, W, H);
}

function vignette(ctx: Ctx, W: number, H: number, k: number) {
  const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(1, `rgba(0,0,0,${k})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

function metal(ctx: Ctx, x1: number, y1: number, x2: number, y2: number, brass: boolean) {
  const g = ctx.createLinearGradient(x1, y1, x2, y2);
  if (brass) {
    g.addColorStop(0, '#f3d98b');
    g.addColorStop(0.45, '#c79a3c');
    g.addColorStop(1, '#6e5320');
  } else {
    g.addColorStop(0, '#eef1f4');
    g.addColorStop(0.4, '#a9b1ba');
    g.addColorStop(0.7, '#7b848e');
    g.addColorStop(1, '#4a5058');
  }
  return g;
}

function drawFastener(ctx: Ctx, cls: string, b: Box, r: () => number) {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const cx = b.x1 + w / 2;
  const cy = b.y1 + h / 2;
  const brass = r() < 0.18;
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.55)';
  ctx.shadowBlur = Math.min(w, h) * 0.18;
  ctx.shadowOffsetX = Math.min(w, h) * 0.06;
  ctx.shadowOffsetY = Math.min(w, h) * 0.1;
  const fill = metal(ctx, b.x1, b.y1, b.x2, b.y2, brass);
  ctx.fillStyle = fill;
  ctx.strokeStyle = 'rgba(0,0,0,0.35)';
  ctx.lineWidth = Math.max(1, Math.min(w, h) * 0.02);
  const horizontal = w >= h;
  const L = horizontal ? w : h;
  const T = horizontal ? h : w;
  // Draw elongated parts along +x, then rotate into place.
  ctx.translate(cx, cy);
  if (!horizontal) ctx.rotate(Math.PI / 2);
  switch (cls) {
    case 'bolt': {
      const head = L * 0.22;
      ctx.beginPath();
      ctx.roundRect(-L / 2, -T / 2, head, T, T * 0.08);
      ctx.fill();
      ctx.beginPath();
      ctx.roundRect(-L / 2 + head, -T * 0.26, L - head, T * 0.52, T * 0.1);
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      for (let x = -L / 2 + head + T * 0.25; x < L / 2 - 2; x += Math.max(2, T * 0.14)) {
        ctx.beginPath();
        ctx.moveTo(x, -T * 0.26);
        ctx.lineTo(x + T * 0.08, T * 0.26);
        ctx.stroke();
      }
      break;
    }
    case 'screw': {
      ctx.beginPath();
      ctx.ellipse(-L / 2 + T / 2, 0, T / 2, T / 2, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(-L / 2 + T / 2, -T * 0.2);
      ctx.lineTo(L / 2 - T * 0.4, -T * 0.16);
      ctx.lineTo(L / 2, 0);
      ctx.lineTo(L / 2 - T * 0.4, T * 0.16);
      ctx.lineTo(-L / 2 + T / 2, T * 0.2);
      ctx.closePath();
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.strokeStyle = 'rgba(0,0,0,0.45)';
      ctx.lineWidth = Math.max(1, T * 0.08);
      ctx.beginPath();
      ctx.moveTo(-L / 2 + T * 0.25, 0);
      ctx.lineTo(-L / 2 + T * 0.75, 0);
      ctx.moveTo(-L / 2 + T / 2, -T * 0.25);
      ctx.lineTo(-L / 2 + T / 2, T * 0.25);
      ctx.stroke();
      break;
    }
    case 'spring': {
      ctx.shadowColor = 'rgba(0,0,0,0.4)';
      ctx.strokeStyle = fill;
      ctx.lineWidth = Math.max(1.5, T * 0.14);
      ctx.lineCap = 'round';
      ctx.beginPath();
      const turns = Math.max(4, Math.round(L / T) * 2);
      for (let i = 0; i <= turns; i++) {
        const x = -L / 2 + T * 0.1 + ((L - T * 0.2) * i) / turns;
        const y = (i % 2 === 0 ? -1 : 1) * (T / 2 - ctx.lineWidth / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      break;
    }
    case 'pin': {
      ctx.beginPath();
      ctx.roundRect(-L / 2, -T * 0.32, L, T * 0.64, T * 0.32);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(L / 2 - T * 0.6, 0, T * 0.12, 0, Math.PI * 2);
      ctx.shadowColor = 'transparent';
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();
      break;
    }
    case 'nut': {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (i * Math.PI) / 3 + Math.PI / 6;
        const x = Math.cos(a) * (L / 2);
        const y = Math.sin(a) * (T / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.arc(0, 0, Math.min(L, T) * 0.24, 0, Math.PI * 2, true);
      ctx.fill('evenodd');
      break;
    }
    case 'washer': {
      ctx.beginPath();
      ctx.ellipse(0, 0, L / 2, T / 2, 0, 0, Math.PI * 2);
      ctx.ellipse(0, 0, L * 0.22, T * 0.22, 0, 0, Math.PI * 2, true);
      ctx.fill('evenodd');
      break;
    }
    case 'rivet': {
      ctx.beginPath();
      ctx.ellipse(0, 0, L / 2, T / 2, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowColor = 'transparent';
      const hl = ctx.createRadialGradient(-L * 0.15, -T * 0.15, 0, 0, 0, L / 2);
      hl.addColorStop(0, 'rgba(255,255,255,0.55)');
      hl.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = hl;
      ctx.fill();
      break;
    }
    default: {
      // bearing
      ctx.beginPath();
      ctx.ellipse(0, 0, L / 2, T / 2, 0, 0, Math.PI * 2);
      ctx.ellipse(0, 0, L * 0.18, T * 0.18, 0, 0, Math.PI * 2, true);
      ctx.fill('evenodd');
      ctx.shadowColor = 'transparent';
      ctx.fillStyle = 'rgba(20,22,26,0.85)';
      ctx.beginPath();
      ctx.ellipse(0, 0, L * 0.38, T * 0.38, 0, 0, Math.PI * 2);
      ctx.ellipse(0, 0, L * 0.26, T * 0.26, 0, 0, Math.PI * 2, true);
      ctx.fill('evenodd');
      ctx.fillStyle = metal(ctx, -L / 2, -T / 2, L / 2, T / 2, false);
      for (let i = 0; i < 9; i++) {
        const a = (i / 9) * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(Math.cos(a) * L * 0.32, Math.sin(a) * T * 0.32, Math.min(L, T) * 0.05, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

// ---------- pond (a benchmark's camera over a garden pond: koi, goldfish and frogs from above) ----------

function drawPond(ctx: Ctx, img: DatasetImage, r: () => number) {
  const { width: W, height: H } = img;
  // Murky green water, lighter where the light comes in.
  const g = ctx.createLinearGradient(0, 0, W * range(r, 0.2, 0.8), H);
  g.addColorStop(0, `hsl(${range(r, 150, 165)} 32% ${range(r, 26, 32)}%)`);
  g.addColorStop(1, `hsl(${range(r, 165, 180)} 38% ${range(r, 12, 17)}%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  // The bottom: dim stones.
  for (let i = 0; i < 14; i++) {
    const x = range(r, 0, W);
    const y = range(r, 0, H);
    const rad = range(r, W * 0.02, W * 0.07);
    ctx.fillStyle = `hsla(${range(r, 30, 60)} 12% ${range(r, 20, 34)}% / 0.55)`;
    ctx.beginPath();
    ctx.ellipse(x, y, rad, rad * range(r, 0.55, 0.9), range(r, 0, Math.PI), 0, Math.PI * 2);
    ctx.fill();
  }
  // Light from above: soft caustic patches.
  for (let i = 0; i < 7; i++) {
    const x = range(r, 0, W);
    const y = range(r, 0, H);
    const rad = range(r, W * 0.08, W * 0.3);
    const rg = ctx.createRadialGradient(x, y, 0, x, y, rad);
    rg.addColorStop(0, `hsla(${range(r, 70, 110)} 40% 70% / 0.16)`);
    rg.addColorStop(1, 'hsla(90 40% 70% / 0)');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, W, H);
  }
  // Lily pads on the surface.
  for (let i = 0; i < 4; i++) {
    const x = range(r, 0, W);
    const y = range(r, 0, H);
    const rad = range(r, W * 0.04, W * 0.08);
    const a = range(r, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${range(r, 95, 120)} 38% ${range(r, 24, 32)}%)`;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, rad, a + 0.35, a + Math.PI * 2 - 0.05);
    ctx.closePath();
    ctx.fill();
  }
}

function drawCreature(ctx: Ctx, cls: string, b: Box, r: () => number) {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const cx = b.x1 + w / 2;
  const cy = b.y1 + h / 2;
  ctx.save();
  ctx.translate(cx, cy);
  if (cls === 'frog') {
    // From above: a body, the hind legs folded beside it, two eyes at the front.
    const flip = r() < 0.5 ? 1 : -1;
    ctx.scale(1, flip);
    const skin = `hsl(${range(r, 85, 110)} ${range(r, 38, 55)}% ${range(r, 32, 42)}%)`;
    ctx.shadowColor = 'rgba(0,0,0,0.4)';
    ctx.shadowBlur = Math.min(w, h) * 0.15;
    ctx.fillStyle = skin;
    for (const sx of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse((sx * w) / 3, h * 0.22, w * 0.2, h * 0.24, sx * 0.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.ellipse(0, 0, w * 0.32, h * 0.42, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.fillStyle = 'hsl(60 30% 70% / 0.5)';
    ctx.beginPath();
    ctx.ellipse(0, h * 0.02, w * 0.16, h * 0.26, 0, 0, Math.PI * 2);
    ctx.fill();
    for (const sx of [-1, 1]) {
      ctx.fillStyle = skin;
      ctx.beginPath();
      ctx.arc(sx * w * 0.15, -h * 0.33, Math.min(w, h) * 0.11, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#14160f';
      ctx.beginPath();
      ctx.arc(sx * w * 0.15, -h * 0.35, Math.min(w, h) * 0.05, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    return;
  }
  // A fish along the box's long side, head to one end, the tail fanned at the other.
  const horizontal = w >= h;
  if (!horizontal) ctx.rotate(Math.PI / 2);
  const L = horizontal ? w : h;
  const T = horizontal ? h : w;
  if (r() < 0.5) ctx.scale(-1, 1);
  const koi = cls === 'koi';
  const base = koi ? `hsl(${range(r, 30, 45)} 30% ${range(r, 86, 92)}%)` : `hsl(${range(r, 26, 38)} 92% ${range(r, 52, 60)}%)`;
  ctx.shadowColor = 'rgba(0,0,0,0.45)';
  ctx.shadowBlur = T * 0.3;
  ctx.shadowOffsetY = T * 0.12;
  ctx.fillStyle = base;
  // Tail.
  ctx.beginPath();
  ctx.moveTo(L * 0.22, 0);
  ctx.quadraticCurveTo(L * 0.42, -T * 0.55, L * 0.5, -T * 0.48);
  ctx.quadraticCurveTo(L * 0.4, 0, L * 0.5, T * 0.48);
  ctx.quadraticCurveTo(L * 0.42, T * 0.55, L * 0.22, 0);
  ctx.fill();
  // Body.
  ctx.beginPath();
  ctx.ellipse(-L * 0.08, 0, L * 0.38, T * 0.3, 0, 0, Math.PI * 2);
  ctx.fill();
  // Pectoral fins.
  for (const sy of [-1, 1]) {
    ctx.beginPath();
    ctx.ellipse(-L * 0.18, sy * T * 0.3, L * 0.08, T * 0.12, sy * 0.7, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowColor = 'transparent';
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(-L * 0.08, 0, L * 0.38, T * 0.3, 0, 0, Math.PI * 2);
  ctx.clip();
  if (koi) {
    // Koi: red-orange patches on white, a black fleck or two.
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = `hsl(${range(r, 4, 18)} 85% ${range(r, 46, 54)}%)`;
      ctx.beginPath();
      ctx.ellipse(-L * range(r, 0.3, -0.1), T * range(r, -0.15, 0.15), L * range(r, 0.08, 0.16), T * range(r, 0.14, 0.24), range(r, 0, Math.PI), 0, Math.PI * 2);
      ctx.fill();
    }
    if (r() < 0.6) {
      ctx.fillStyle = 'rgba(20,20,22,0.8)';
      ctx.beginPath();
      ctx.ellipse(-L * range(r, 0.2, 0), T * range(r, -0.1, 0.1), L * 0.04, T * 0.07, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  // The back's highlight.
  const hl = ctx.createLinearGradient(0, -T * 0.3, 0, T * 0.3);
  hl.addColorStop(0, 'rgba(255,255,255,0.28)');
  hl.addColorStop(0.5, 'rgba(255,255,255,0)');
  hl.addColorStop(1, 'rgba(0,0,0,0.18)');
  ctx.fillStyle = hl;
  ctx.fillRect(-L / 2, -T / 2, L, T);
  ctx.restore();
  // Eyes.
  ctx.fillStyle = '#101214';
  for (const sy of [-1, 1]) {
    ctx.beginPath();
    ctx.arc(-L * 0.38, sy * T * 0.12, Math.max(1, T * 0.05), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
