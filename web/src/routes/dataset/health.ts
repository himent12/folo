// Dataset health checks and histogram labels. Pure functions so the thresholds are easy to test and tune.

import type { Dataset, DatasetStats, Histogram, IssueCounts, IssueSeverity } from '$lib/api/types';

export type Severity = 'danger' | 'warning' | 'info' | 'ok';

export interface HealthItem {
  id: string;
  severity: Severity;
  title: string;
  /** A few words for the folded Health line, with the unit named ("3 check warnings", "35% tiny boxes"). */
  short: string;
  detail: string;
  /** A concrete next step where the data allows one: a label and an app path (without the leading #). */
  action?: { label: string; path: string };
  /** The `folo dataset check` issues behind it, listed from this severity on. */
  issues?: IssueSeverity;
  /**
   * How many file issues it stands for (one per broken label file; the check's count otherwise). Findings without it
   * are dataset-wide observations (class balance, box sizes, splits), counted apart from the files.
   */
  files?: number;
}

/** One line of the health summary: the findings of one severity, most severe first. */
export interface SeverityGroup {
  severity: Exclude<Severity, 'ok'>;
  /** "Errors", "Warnings", "Notes". */
  label: string;
  items: HealthItem[];
}

/** Findings grouped by severity (errors, warnings, notes), leaving out empty groups and the `ok` item. */
export function bySeverity(items: readonly HealthItem[]): SeverityGroup[] {
  const groups: SeverityGroup[] = [
    { severity: 'danger', label: 'Errors', items: [] },
    { severity: 'warning', label: 'Warnings', items: [] },
    { severity: 'info', label: 'Notes', items: [] },
  ];
  for (const h of items) groups.find((g) => g.severity === h.severity)?.items.push(h);
  return groups.filter((g) => g.items.length > 0);
}

/**
 * How many findings there are: errors, warnings and notes alike (not the `ok` item), each one finding whatever it
 * counts inside (files, boxes, classes). The one count, said one way everywhere: the Stats tab, Health's heading, and
 * its folded line, which shows each finding once.
 */
export function findingCount(items: readonly HealthItem[]): number {
  return items.filter((h) => h.severity !== 'ok').length;
}

const plural = (n: number, one: string, many: string) => `${int(n)} ${n === 1 ? one : many}`;

/** "4 findings", or undefined when there are none. */
export function findingLabel(n: number): string | undefined {
  return n > 0 ? plural(n, 'finding', 'findings') : undefined;
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

/**
 * Health findings, most severe first. Returns one `ok` item when nothing stands out. `counts` (the issues endpoint's,
 * from servers that send them) adds what `folo dataset check` finds beyond the stats: unreadable images, and warnings
 * such as clipped boxes or duplicate images.
 */
export function healthChecks(d: Dataset, s: DatasetStats, counts?: IssueCounts): HealthItem[] {
  const out: HealthItem[] = [];
  const images = imageCount(d);
  const id = encodeURIComponent(d.id);
  const showClass = (i: number) => ({ label: `Show ${d.classes[i] ?? `class ${i}`} images`, path: `/datasets/${id}?class=${i}` });

  if (s.broken_labels > 0) {
    out.push({
      id: 'broken',
      severity: 'danger',
      title: `${int(s.broken_labels)} broken label ${s.broken_labels === 1 ? 'file' : 'files'}`,
      short: `${int(s.broken_labels)} broken label ${s.broken_labels === 1 ? 'file' : 'files'}`,
      detail: 'These labels could not be parsed, so their images are skipped in training. Fix or delete them, then re-register the dataset.',
      issues: 'error',
      files: s.broken_labels,
    });
    // The check's image errors count too: unreadable or missing images. Its label errors (broken_label,
    // class_overflow, out_of_range_box) are per line, and already the broken files above, so they are not added again.
    const other = counts ? (counts.by_kind.corrupt_image ?? 0) + (counts.by_kind.missing_image ?? 0) : 0;
    if (other > 0) {
      out.push({
        id: 'check-errors',
        severity: 'danger',
        title: `${int(other)} unreadable or missing ${other === 1 ? 'image' : 'images'}`,
        short: `${int(other)} unreadable ${other === 1 ? 'image' : 'images'}`,
        detail: 'Images that cannot be read are left out of training. Fix or remove them.',
        issues: 'error',
        files: other,
      });
    }
  } else if (counts && counts.error > 0) {
    out.push({
      id: 'check-errors',
      severity: 'danger',
      title: `${int(counts.error)} file ${counts.error === 1 ? 'error' : 'errors'} from the dataset check`,
      short: `${int(counts.error)} file ${counts.error === 1 ? 'error' : 'errors'}`,
      detail: 'Images that cannot be read and labels that cannot be parsed are left out of training. Fix or remove them.',
      issues: 'error',
      files: counts.error,
    });
  }
  if (counts && counts.warning > 0) {
    out.push({
      id: 'check-warnings',
      severity: 'warning',
      title: `${int(counts.warning)} ${counts.warning === 1 ? 'warning' : 'warnings'} from the dataset check`,
      short: `${int(counts.warning)} check ${counts.warning === 1 ? 'warning' : 'warnings'}`,
      detail: 'Training still uses these files: boxes past the image edge, duplicate labels or images. Each says what a cleanup would do.',
      issues: 'warning',
      files: counts.warning,
    });
  }

  const missing = d.classes.filter((_, i) => (s.class_counts[i] ?? 0) === 0);
  if (missing.length) {
    out.push({
      id: 'missing',
      severity: 'danger',
      title: `${missing.length === 1 ? 'Class' : 'Classes'} without boxes: ${missing.join(', ')}`,
      short: `${missing.length} ${missing.length === 1 ? 'class' : 'classes'} without boxes`,
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
        short: `${ratio.toFixed(1)}× class imbalance`,
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
      short: `${few.length} ${few.length === 1 ? 'class' : 'classes'} under ${FEW_BOXES} boxes`,
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
      short: `${share(tiny)} tiny boxes`,
      detail: 'They are under 5% of the image side (about 32 px at 640). Train at a larger image size or tile large images.',
      action: { label: `Train at ${TINY_IMGSZ} px`, path: `/runs/new?dataset=${id}&imgsz=${TINY_IMGSZ}` },
    });
  }

  if (d.splits.val === 0) {
    out.push({
      id: 'no-val',
      severity: 'warning',
      title: 'No validation split',
      short: 'no validation split',
      detail: 'mAP cannot be measured during training. Move 10–20% of the images into a val split.',
    });
  }

  // The check's notes are its empty images: the empty-image finding opens them, or one of their own when the stats count
  // none.
  const notes = counts?.info ?? 0;
  if (images > 0 && s.empty_images > 0) {
    const e = s.empty_images / images;
    out.push({
      id: 'empty',
      severity: e > EMPTY_WARN ? 'warning' : 'info',
      title: `${int(s.empty_images)} empty ${s.empty_images === 1 ? 'image' : 'images'} (${share(e)})`,
      short: `${int(s.empty_images)} empty ${s.empty_images === 1 ? 'image' : 'images'} (${share(e)})`,
      detail:
        e > EMPTY_WARN
          ? 'Background images reduce false positives, but above ~10% they slow learning. Check that labels were not lost.'
          : 'Images without boxes act as background examples and reduce false positives. Up to ~10% is healthy.',
      ...(notes > 0 ? { issues: 'info' as const, files: notes } : {}),
    });
  } else if (notes > 0) {
    out.push({
      id: 'check-notes',
      severity: 'info',
      title: `${int(notes)} ${notes === 1 ? 'note' : 'notes'} from the dataset check`,
      short: `${int(notes)} check ${notes === 1 ? 'note' : 'notes'}`,
      detail: 'Images without boxes. Nothing to fix: they act as background examples.',
      issues: 'info',
      files: notes,
    });
  }

  if (images > 0 && images < 200) {
    out.push({
      id: 'small',
      severity: 'info',
      title: `Small dataset (${int(images)} images)`,
      short: `only ${int(images)} images`,
      detail: 'Expect overfitting; keep augmentation on and start from pretrained weights.',
    });
  }

  if (!out.some((x) => x.severity === 'danger' || x.severity === 'warning')) {
    out.unshift({
      id: 'ok',
      severity: 'ok',
      title: 'No problems found',
      short: 'no problems found',
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
