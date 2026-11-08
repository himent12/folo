// Dataset health checks and histogram labels. Pure functions so the thresholds are easy to test and tune.

import type { Dataset, DatasetStats, Histogram } from '$lib/api/types';

export type Severity = 'danger' | 'warning' | 'info' | 'ok';

export interface HealthItem {
  id: string;
  severity: Severity;
  title: string;
  detail: string;
  /** A concrete next step where the data allows one: a label and an app path (without the leading #). */
  action?: { label: string; path: string };
}

/** Findings that need attention (errors and warnings): the "N to review" count, everywhere it is shown. */
export function reviewCount(items: readonly HealthItem[]): number {
  return items.filter((h) => h.severity === 'danger' || h.severity === 'warning').length;
}

/** Image size suggested for datasets with many tiny boxes (1.5× the 640 default). */
export const TINY_IMGSZ = 960;

/** Imbalance (largest / smallest class) above which we warn. */
export const IMBALANCE_WARN = 5;
/** Boxes whose relative size (sqrt of the area share, as in docs/API.md) is below this count as tiny: 5% of the
 * image side, about 32 px at 640. */
export const TINY_SIZE = 0.05;
/** Share of tiny boxes above which we warn. */
export const TINY_WARN = 0.1;
/** Share of empty (background) images above which we warn. */
export const EMPTY_WARN = 0.1;
/** Boxes per class below which a class is considered under-represented. */
export const FEW_BOXES = 100;

const int = (x: number) => Math.round(x).toLocaleString('en-US');
const share = (x: number) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;

export function imageCount(d: Pick<Dataset, 'splits'>): number {
  return d.splits.train + d.splits.val + d.splits.test;
}

/** Fraction of `h`'s mass in bins that lie entirely below `limit`. */
export function shareBelow(h: Histogram, limit: number): number {
  let below = 0;
  let total = 0;
  h.counts.forEach((c, i) => {
    total += c;
    if ((h.edges[i + 1] ?? Infinity) <= limit + 1e-12) below += c;
  });
  return total ? below / total : 0;
}

/** Health findings, most severe first. Returns one `ok` item when nothing stands out. */
export function healthChecks(d: Dataset, s: DatasetStats): HealthItem[] {
  const out: HealthItem[] = [];
  const images = imageCount(d);
  const id = encodeURIComponent(d.id);
  const showClass = (i: number) => ({ label: `Show ${d.classes[i] ?? `class ${i}`} images`, path: `/datasets/${id}?class=${i}` });

  if (s.broken_labels > 0) {
    out.push({
      id: 'broken',
      severity: 'danger',
      title: `${int(s.broken_labels)} broken label ${s.broken_labels === 1 ? 'file' : 'files'}`,
      detail: 'These labels could not be parsed, so their images are skipped in training. Fix or delete them, then re-register the dataset.',
    });
  }

  const missing = d.classes.filter((_, i) => (s.class_counts[i] ?? 0) === 0);
  if (missing.length) {
    out.push({
      id: 'missing',
      severity: 'danger',
      title: `${missing.length === 1 ? 'Class' : 'Classes'} without boxes: ${missing.join(', ')}`,
      detail: 'The model can never learn a class it has not seen. Label some examples or remove the class.',
    });
  }

  const present = s.class_counts.map((c, i) => ({ c, i })).filter((x) => x.c > 0);
  if (present.length >= 2) {
    const hi = present.reduce((a, b) => (b.c > a.c ? b : a));
    const lo = present.reduce((a, b) => (b.c < a.c ? b : a));
    // Compare the ratio as displayed (one decimal), so "5.0×" never shows up as a warning.
    const ratio = Math.round((hi.c / lo.c) * 10) / 10;
    if (ratio > IMBALANCE_WARN) {
      out.push({
        id: 'imbalance',
        severity: 'warning',
        title: `Class imbalance ${ratio.toFixed(1)}×`,
        detail: `${d.classes[hi.i]} has ${int(hi.c)} boxes, ${d.classes[lo.i]} only ${int(lo.c)}. Rare classes usually score lower; add examples or oversample them.`,
        action: showClass(lo.i),
      });
    }
  }

  const few = present.filter((x) => x.c < FEW_BOXES);
  if (few.length) {
    out.push({
      id: 'few',
      severity: 'warning',
      title: `${few.length === 1 ? 'A class has' : `${few.length} classes have`} fewer than ${FEW_BOXES} boxes`,
      detail: `${few.map((x) => `${d.classes[x.i]} (${int(x.c)})`).join(', ')}. Aim for a few hundred boxes per class or more.`,
      ...(few.length === 1 ? { action: showClass(few[0]!.i) } : {}),
    });
  }

  const tiny = shareBelow(s.box_size_hist, TINY_SIZE);
  if (tiny > TINY_WARN) {
    out.push({
      id: 'tiny',
      severity: 'warning',
      title: `${share(tiny)} of boxes are tiny`,
      detail: 'They are under 5% of the image side (about 32 px at 640). Train at a larger image size or tile large images.',
      action: { label: `Train at ${TINY_IMGSZ} px`, path: `/runs/new?dataset=${id}&imgsz=${TINY_IMGSZ}` },
    });
  }

  if (d.splits.val === 0) {
    out.push({
      id: 'no-val',
      severity: 'warning',
      title: 'No validation split',
      detail: 'mAP cannot be measured during training. Move 10–20% of the images into a val split.',
    });
  }

  if (images > 0 && s.empty_images > 0) {
    const e = s.empty_images / images;
    out.push({
      id: 'empty',
      severity: e > EMPTY_WARN ? 'warning' : 'info',
      title: `${int(s.empty_images)} empty ${s.empty_images === 1 ? 'image' : 'images'} (${share(e)})`,
      detail:
        e > EMPTY_WARN
          ? 'Background images reduce false positives, but above ~10% they slow learning. Check that labels were not lost.'
          : 'Images without boxes act as background examples and reduce false positives. Up to ~10% is healthy.',
    });
  }

  if (images > 0 && images < 200) {
    out.push({
      id: 'small',
      severity: 'info',
      title: `Small dataset (${int(images)} images)`,
      detail: 'Expect overfitting; keep augmentation on and start from pretrained weights.',
    });
  }

  if (!out.some((x) => x.severity === 'danger' || x.severity === 'warning')) {
    out.unshift({
      id: 'ok',
      severity: 'ok',
      title: 'No problems found',
      detail: 'Labels parse, every class is represented and the class balance is within 5×.',
    });
  }

  const rank: Record<Severity, number> = { danger: 0, warning: 1, ok: 2, info: 3 };
  return out.sort((a, b) => rank[a.severity] - rank[b.severity]);
}

/** 0.0025 → "0.25", 0.01 → "1", 0.16 → "16" (percent, no sign). */
function pctNum(x: number): string {
  const p = x * 100;
  return String(Number(p.toFixed(p < 1 ? 2 : p < 10 ? 1 : 0)));
}

/** Box-size bin label in % of the image side: "<2%", "10–15%", "≥50%". */
export function sizeBinLabel(lo: number, hi: number): string {
  if (lo <= 0) return `<${pctNum(hi)}%`;
  if (hi >= 1) return `≥${pctNum(lo)}%`;
  return `${pctNum(lo)}–${pctNum(hi)}%`;
}

/** Boxes-per-image bin label for integer bins [lo, hi): "0", "2–3", "32+" (last bin). */
export function countBinLabel(lo: number, hi: number, last: boolean): string {
  if (last) return `${lo}+`;
  if (hi - lo <= 1) return String(lo);
  return `${lo}–${hi - 1}`;
}
