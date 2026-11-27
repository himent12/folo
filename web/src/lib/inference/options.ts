// Inference enhancement options (SAHI tiled inference and Test-Time Augmentation) shared between
// the browser folo-wasm engine, HTTP /api/predict (`predict_options`), and Playground UI/exports.

export type InferenceMode = 'standard' | 'sahi' | 'tta';

export type TtaScalePreset = 'multi' | 'single';

/** Precision choice exposed by `folo-wasm` (`FoloLoadOptions.precision`); `'auto'` omits the option so `folo-wasm` picks INT8 on CPU for quantized checkpoints and f32 otherwise. */
export type WasmPrecision = 'f32' | 'f16' | 'int8';
export type BrowserPrecision = 'auto' | WasmPrecision;

export const BROWSER_PRECISION_OPTIONS: readonly { value: BrowserPrecision; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'f32', label: 'f32' },
  { value: 'f16', label: 'f16' },
  { value: 'int8', label: 'INT8' },
];

export interface TileOptions {
  /** Square tile size in pixels (`320 | 512 | 640 | 1024`). */
  tileSize: number;
  /** Symmetric overlap ratio in `[0, 1)` (`0 | 0.1 | 0.2 | 0.3`). */
  overlap: number;
  /** Also run one full-image pass alongside the tiles to preserve large boundary-spanning objects. */
  fullImage: boolean;
}

export interface TtaOptions {
  /** Also evaluate a horizontally flipped mirror at each scale. */
  hflip: boolean;
  /** Multi-scale multipliers applied to the model's `imgsz`. */
  scales: readonly number[];
}

export interface InferenceConfig {
  mode: InferenceMode;
  tile: TileOptions;
  tta: TtaOptions;
  /** Optional browser WASM compute precision (`auto` | `f32` | `f16` | `int8`; ignored for server runs). */
  precision?: BrowserPrecision;
}

/** Recorded on a `Prediction` (and exported in JSON) when SAHI or TTA produced the boxes. */
export type InferenceMeta =
  | {
      mode: 'sahi';
      tile_size: number;
      overlap: number;
      full_image: boolean;
    }
  | {
      mode: 'tta';
      hflip: boolean;
      scales: number[];
    };

/** Base per-call options accepted by `FoloDetector.detect` (crates/folo-wasm/src/lib.rs). */
export interface WasmDetectOptions {
  conf?: number;
  iou?: number;
  maxDet?: number;
  /** Optional per-class confidence thresholds (`class_id -> conf`, finite values in `[0, 1]`). */
  perClassConf?: Record<number, number>;
  /** Use inclusive (`score >= conf`) thresholding matching `folo val`'s calibrated F1 operating points. */
  confInclusive?: boolean;
}

/** Per-call options accepted by `FoloDetector.detectTiled` (crates/folo-wasm, docs/WASM.md). */
export interface WasmTileDetectOptions extends WasmDetectOptions {
  tileSize?: number;
  overlap?: number;
  fullImage?: boolean;
}

/** Per-call options accepted by `FoloDetector.detectTta` (crates/folo-wasm, docs/WASM.md). */
export interface WasmTtaDetectOptions extends WasmDetectOptions {
  hflip?: boolean;
  scales?: number[];
  mergeIou?: number;
}

/** Server `POST /api/predict` `tile` payload (`folo_vision::TileConfig`). */
export interface ServerTilePayload {
  tile_width: number;
  tile_height: number;
  overlap_x: number;
  overlap_y: number;
  full_image: boolean;
  max_det: number;
}

/** Server `POST /api/predict` `tta` payload (`folo_vision::TtaConfig`). */
export interface ServerTtaPayload {
  hflip: boolean;
  scales: number[];
  merge_iou: number;
  max_det: number;
}

export const TILE_SIZE_OPTIONS = [320, 512, 640, 1024] as const;
export const TILE_OVERLAP_OPTIONS = [0, 0.1, 0.2, 0.3] as const;
export const TTA_MULTI_SCALES = [0.83, 1.0, 1.17] as const;
export const TTA_SINGLE_SCALES = [1.0] as const;

export const DEFAULT_TILE_OPTIONS: TileOptions = {
  tileSize: 640,
  overlap: 0.2,
  fullImage: true,
};

export const DEFAULT_TTA_OPTIONS: TtaOptions = {
  hflip: true,
  scales: [...TTA_MULTI_SCALES],
};

export const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  mode: 'standard',
  tile: { ...DEFAULT_TILE_OPTIONS },
  tta: { hflip: DEFAULT_TTA_OPTIONS.hflip, scales: [...DEFAULT_TTA_OPTIONS.scales] },
};

/** Default cross-view TTA merge IoU used when Playground fetches loose candidates (`iou = 1`). */
const DEFAULT_TTA_MERGE_IOU = 0.7;

/** Maps `TtaOptions.scales` to the UI preset (`multi` vs `single`). */
export function ttaScalePreset(scales: readonly number[]): TtaScalePreset {
  return scales.length === 1 && scales[0] === 1 ? 'single' : 'multi';
}

/** Resolves a UI scale preset to the scale multiplier list sent to WASM / server. */
export function scalesForPreset(preset: TtaScalePreset): number[] {
  return preset === 'single' ? [...TTA_SINGLE_SCALES] : [...TTA_MULTI_SCALES];
}

/** Stable key identifying an inference configuration for cache invalidation and effect tracking. */
export function inferenceConfigKey(cfg: InferenceConfig | null | undefined): string {
  if (!cfg || cfg.mode === 'standard') return 'standard';
  if (cfg.mode === 'sahi') {
    return `sahi:${cfg.tile.tileSize}:${cfg.tile.overlap}:${cfg.tile.fullImage ? 1 : 0}`;
  }
  return `tta:${cfg.tta.hflip ? 1 : 0}:${cfg.tta.scales.join(',')}`;
}

/** Stable key for `InferenceMeta` attached to a completed `Prediction`. */
export function inferenceMetaKey(meta: InferenceMeta | null | undefined): string {
  if (!meta) return 'standard';
  if (meta.mode === 'sahi') {
    return `sahi:${meta.tile_size}:${meta.overlap}:${meta.full_image ? 1 : 0}`;
  }
  return `tta:${meta.hflip ? 1 : 0}:${meta.scales.join(',')}`;
}

/** Structured metadata attached to a `Prediction` when SAHI or TTA ran (undefined for standard). */
export function inferenceMetaOf(cfg: InferenceConfig | null | undefined): InferenceMeta | undefined {
  if (!cfg || cfg.mode === 'standard') return undefined;
  if (cfg.mode === 'sahi') {
    return {
      mode: 'sahi',
      tile_size: cfg.tile.tileSize,
      overlap: cfg.tile.overlap,
      full_image: cfg.tile.fullImage,
    };
  }
  return {
    mode: 'tta',
    hflip: cfg.tta.hflip,
    scales: [...cfg.tta.scales],
  };
}

/** Compact label for Playground's timing/metadata strip (`null` for standard single-pass). */
export function inferenceMetaLabel(meta: InferenceMeta | null | undefined): string | null {
  if (!meta) return null;
  if (meta.mode === 'sahi') {
    const pct = Math.round(meta.overlap * 100);
    return `SAHI ${meta.tile_size}px · ${pct}%${meta.full_image ? ' + full' : ''}`;
  }
  const n = meta.scales.length;
  return `TTA ${n === 1 ? '1×' : `${n}×`}${meta.hflip ? ' + flip' : ''}`;
}

/** Short chip label for the collapsed disclosure badge or narrow header. */
export function inferenceShortTag(meta: InferenceMeta | null | undefined): string | null {
  if (!meta) return null;
  if (meta.mode === 'sahi') return `SAHI ${meta.tile_size}px`;
  return `TTA ${meta.scales.length}×${meta.hflip ? ' + flip' : ''}`;
}

const CANONICAL_CLASS_ID = /^(?:0|[1-9]\d*)$/;

/** Sanitizes a per-class confidence map to canonical non-negative integer class IDs and finite `[0, 1]` values for `folo-wasm`. */
export function sanitizeWasmPerClassConf(
  map: Readonly<Record<number, number>> | null | undefined,
): Record<number, number> | undefined {
  if (!map) return undefined;
  const out: Record<number, number> = {};
  let count = 0;
  for (const [k, v] of Object.entries(map)) {
    if (!CANONICAL_CLASS_ID.test(k)) continue;
    const id = Number(k);
    if (!Number.isSafeInteger(id) || id > 0xffff_ffff) continue;
    if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) continue;
    out[id] = v;
    count++;
  }
  return count > 0 ? out : undefined;
}

/** Builds `FoloDetector.detect` options from base thresholds plus optional `perClassConf` / `confInclusive`. */
export function toWasmDetectOptions(base: {
  conf: number;
  iou: number;
  maxDet: number;
  perClassConf?: Readonly<Record<number, number>> | null;
  confInclusive?: boolean;
}): WasmDetectOptions {
  const perClassConf = sanitizeWasmPerClassConf(base.perClassConf);
  return {
    conf: base.conf,
    iou: base.iou,
    maxDet: base.maxDet,
    ...(perClassConf ? { perClassConf } : {}),
    ...(typeof base.confInclusive === 'boolean' ? { confInclusive: base.confInclusive } : {}),
  };
}

/** Builds `FoloDetector.detectTiled` options from base thresholds and `TileOptions`. */
export function toWasmTileOptions(
  base: {
    conf: number;
    iou: number;
    maxDet: number;
    perClassConf?: Readonly<Record<number, number>> | null;
    confInclusive?: boolean;
  },
  tile: TileOptions,
): WasmTileDetectOptions {
  return {
    ...toWasmDetectOptions(base),
    tileSize: tile.tileSize,
    overlap: tile.overlap,
    fullImage: tile.fullImage,
  };
}

/** Builds `FoloDetector.detectTta` options; clamps loose `iou = 1` fetches so cross-view fusion still runs. */
export function toWasmTtaOptions(
  base: {
    conf: number;
    iou: number;
    maxDet: number;
    perClassConf?: Readonly<Record<number, number>> | null;
    confInclusive?: boolean;
  },
  tta: TtaOptions,
): WasmTtaDetectOptions {
  return {
    ...toWasmDetectOptions(base),
    hflip: tta.hflip,
    scales: [...tta.scales],
    mergeIou: base.iou >= 0.99 ? DEFAULT_TTA_MERGE_IOU : base.iou,
  };
}

/** Normalizes an optional browser precision into `'auto' | 'f32' | 'f16' | 'int8'`. */
export function normalizeBrowserPrecision(v: unknown): BrowserPrecision {
  return v === 'f32' || v === 'f16' || v === 'int8' ? v : 'auto';
}

/** Builds `POST /api/predict` `tile` payload (`folo_vision::TileConfig`). */
export function toServerTilePayload(tile: TileOptions, maxDet: number): ServerTilePayload {
  return {
    tile_width: tile.tileSize,
    tile_height: tile.tileSize,
    overlap_x: tile.overlap,
    overlap_y: tile.overlap,
    full_image: tile.fullImage,
    max_det: maxDet,
  };
}

/** Builds `POST /api/predict` `tta` payload (`folo_vision::TtaConfig`). */
export function toServerTtaPayload(tta: TtaOptions, maxDet: number, iou: number): ServerTtaPayload {
  return {
    hflip: tta.hflip,
    scales: [...tta.scales],
    merge_iou: iou >= 0.99 ? DEFAULT_TTA_MERGE_IOU : iou,
    max_det: maxDet,
  };
}
