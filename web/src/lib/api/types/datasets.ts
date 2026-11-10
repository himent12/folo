// Datasets: their stats, images, boxes and issues (docs/API.md, Datasets).

import type { Time } from './time';

// ---------- Datasets ----------

export type DatasetFormat = 'yolo' | 'coco';
export type Split = 'train' | 'val' | 'test';

export interface Histogram {
  /** Bin edges, length counts.length + 1. */
  edges: number[];
  counts: number[];
}

export interface DatasetStats {
  boxes: number;
  empty_images: number;
  broken_labels: number;
  /** Indexed by class id. */
  class_counts: number[];
  /** Relative box size, `sqrt(box area / image area)`; counts[i] covers [edges[i], edges[i+1]), the last bin is open. */
  box_size_hist: Histogram;
  boxes_per_image_hist: Histogram;
}

export interface Dataset {
  id: string;
  name: string;
  path: string;
  format: DatasetFormat;
  classes: string[];
  created: Time;
  splits: Record<Split, number>;
  /** Present on GET /api/datasets/{id}. */
  stats?: DatasetStats;
}

export interface CreateDataset {
  name: string;
  path: string;
  format: 'auto' | DatasetFormat;
}

export interface CreateSyntheticDataset {
  name: string;
  images: number;
  classes: string[];
  imgsz: number;
  seed: number;
}

/** Pixel xyxy box in the original image. */
export interface Box {
  class: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** A prediction: a box with its score and class name. */
export interface ScoredBox extends Box {
  name: string;
  score: number;
}

export interface DatasetImage {
  id: string;
  file: string;
  split: Split;
  width: number;
  height: number;
  boxes: Box[];
}

export interface DatasetImagesQuery {
  split?: Split;
  offset?: number;
  limit?: number;
  class?: number;
}

export interface DatasetImages {
  total: number;
  images: DatasetImage[];
}

/** `GET /api/datasets/{id}/issues` (docs/API.md): the `folo dataset check` findings. */
export type IssueKind =
  | 'broken_label'
  | 'class_overflow'
  | 'out_of_range_box'
  | 'clipped_box'
  | 'zero_size_box'
  | 'duplicate_label'
  | 'corrupt_image'
  | 'missing_image'
  | 'duplicate_image'
  | 'empty_image'
  | 'other';

export type IssueSeverity = 'error' | 'warning' | 'info';

export interface DatasetIssue {
  kind: IssueKind;
  severity: IssueSeverity;
  /** Relative to the dataset root (absolute outside it). */
  file: string;
  /** 1-based label line, or null. */
  line: number | null;
  message: string;
  /** What a cleanup would do, e.g. `delete line 3` (absent for infos). */
  fix?: string;
}

export interface DatasetIssuesQuery {
  kind?: IssueKind;
  severity?: IssueSeverity;
  offset?: number;
  limit?: number;
}

/** Every issue of the dataset by severity and kind, whatever the filter. */
export interface IssueCounts {
  error: number;
  warning: number;
  info: number;
  by_kind: Partial<Record<IssueKind, number>>;
}

export interface DatasetIssues {
  total: number;
  issues: DatasetIssue[];
  /** Absent from older servers. */
  counts?: IssueCounts;
}
