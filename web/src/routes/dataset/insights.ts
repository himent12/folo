// Pure class-balance, annotation-quality, and split-breakdown metrics derived strictly from real Dataset,
// DatasetStats, and optional IssueCounts payloads (shared thresholds with ./health so both tabs always agree).
import type { Dataset, DatasetStats, IssueCounts, Split } from '$lib/api/types';
import { EMPTY_WARN, FEW_BOXES, IMBALANCE_WARN, imageCount, shareBelow, TINY_SIZE, TINY_WARN } from './health';

export type InsightTone = 'danger' | 'warning' | 'ok';

export interface ClassStat {
  cls: number;
  name: string;
  count: number;
  sharePct: number;
}

export interface SplitShare {
  split: Split;
  count: number;
  pct: number;
}

export interface DatasetInsightsData {
  totalImages: number;
  totalBoxes: number;
  boxesPerImage: number | null;
  presentClasses: ClassStat[];
  missingClasses: { cls: number; name: string }[];
  mostCommon: ClassStat | null;
  rarest: ClassStat | null;
  /** Present classes below `FEW_BOXES` or more than `IMBALANCE_WARN`× rarer than the most common class, rarest first. */
  underrepresented: ClassStat[];
  /** Largest / smallest present class count rounded to 1 decimal (matching `healthChecks`), or null if < 2 present. */
  imbalanceRatio: number | null;
  imbalanced: boolean;
  balanceTone: InsightTone;
  emptyImages: number;
  emptyPct: number;
  emptyWarn: boolean;
  brokenLabels: number;
  tinyBoxPct: number;
  tinyWarn: boolean;
  checkErrors: number;
  checkWarnings: number;
  qualityTone: InsightTone;
  splits: SplitShare[];
  noVal: boolean;
  noTrain: boolean;
}

const SPLITS: readonly Split[] = ['train', 'val', 'test'];

/** Computes structured class-balance, annotation-quality, and split insights strictly from real dataset metadata. */
export function computeDatasetInsights(
  dataset: Dataset,
  stats: DatasetStats,
  counts?: IssueCounts,
): DatasetInsightsData {
  const totalImages = imageCount(dataset);
  const totalBoxes = Math.max(0, stats.boxes);
  const boxesPerImage = totalImages > 0 ? totalBoxes / totalImages : null;

  const classTotal = Math.max(dataset.classes.length, stats.class_counts.length);
  const presentClasses: ClassStat[] = [];
  const missingClasses: { cls: number; name: string }[] = [];

  for (let i = 0; i < classTotal; i++) {
    const c = Math.max(0, stats.class_counts[i] ?? 0);
    const name = dataset.classes[i] ?? `class ${i}`;
    if (c > 0) {
      presentClasses.push({
        cls: i,
        name,
        count: c,
        sharePct: totalBoxes > 0 ? (c / totalBoxes) * 100 : 0,
      });
    } else if (i < dataset.classes.length) {
      missingClasses.push({ cls: i, name });
    }
  }

  presentClasses.sort((a, b) => b.count - a.count || a.cls - b.cls);
  const mostCommon = presentClasses[0] ?? null;
  const rarest = presentClasses.length >= 2 ? presentClasses[presentClasses.length - 1]! : null;

  const imbalanceRatio =
    mostCommon && rarest ? Math.round((mostCommon.count / rarest.count) * 10) / 10 : null;
  const imbalanced = imbalanceRatio !== null && imbalanceRatio > IMBALANCE_WARN;

  const underrepresented = presentClasses
    .filter((c) => c.count < FEW_BOXES || (imbalanced && mostCommon !== null && Math.round((mostCommon.count / c.count) * 10) / 10 > IMBALANCE_WARN))
    .slice()
    .sort((a, b) => a.count - b.count || a.cls - b.cls);

  const balanceTone: InsightTone =
    missingClasses.length > 0
      ? 'danger'
      : imbalanced || underrepresented.length > 0 || (totalBoxes === 0 && dataset.classes.length > 0)
        ? 'warning'
        : 'ok';

  const emptyImages = Math.max(0, stats.empty_images);
  const emptyShare = totalImages > 0 ? emptyImages / totalImages : 0;
  const emptyPct = emptyShare * 100;
  const emptyWarn = totalImages > 0 && emptyShare > EMPTY_WARN;

  const brokenLabels = Math.max(0, stats.broken_labels);
  const tinyShare = shareBelow(stats.box_size_hist, TINY_SIZE);
  const tinyBoxPct = tinyShare * 100;
  const tinyWarn = tinyShare > TINY_WARN;

  const checkErrors = Math.max(0, counts?.error ?? 0);
  const checkWarnings = Math.max(0, counts?.warning ?? 0);

  const qualityTone: InsightTone =
    brokenLabels > 0 || checkErrors > 0
      ? 'danger'
      : emptyWarn || tinyWarn || checkWarnings > 0
        ? 'warning'
        : 'ok';

  const splits: SplitShare[] = SPLITS.map((split) => {
    const n = Math.max(0, dataset.splits[split] ?? 0);
    return {
      split,
      count: n,
      pct: totalImages > 0 ? (n / totalImages) * 100 : 0,
    };
  });

  return {
    totalImages,
    totalBoxes,
    boxesPerImage,
    presentClasses,
    missingClasses,
    mostCommon,
    rarest,
    underrepresented,
    imbalanceRatio,
    imbalanced,
    balanceTone,
    emptyImages,
    emptyPct,
    emptyWarn,
    brokenLabels,
    tinyBoxPct,
    tinyWarn,
    checkErrors,
    checkWarnings,
    qualityTone,
    splits,
    noVal: (dataset.splits.val ?? 0) === 0,
    noTrain: (dataset.splits.train ?? 0) === 0,
  };
}
