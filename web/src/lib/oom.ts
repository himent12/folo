// Out of GPU memory: the one failure whose fixes are known in advance (docs/API.md `507 gpu_out_of_memory`). A run
// that runs out fails with `Run.error` = "the GPU ran out of memory (<first line of the driver report>)"; older
// servers and `folo serve --mock` word it differently ("gpu:0 out of memory: ..."), so both are recognised.

import { ApiError, type Api } from '$lib/api/client';
import type { Arch, SystemInfo, TrainConfig } from '$lib/api/types';
import { smallerArchInFamily } from '$lib/arch';
import { hasFeature } from '$lib/api/queue';
import { referenceVram } from '$lib/reference';
import { withoutInexactSeed } from '$lib/seed';

export const OOM_CODE = 'gpu_out_of_memory';
/** How lists name the cause instead of the driver text. */
export const OOM_LABEL = 'Out of GPU memory';
/** At or below this batch, halving it again rarely helps: a smaller image or model leads. */
export const SMALL_BATCH = 4;
/** The image sizes New run offers, smallest first. */
export const IMGSZ_STEPS = [320, 480, 640, 960, 1280];

const PREFIX = /^the GPU ran out of memory\s*/i;
/**
 * The GPU wordings, and only those: the server's summary, the older servers' "gpu:0 out of memory: ...", and wgpu's
 * own report. A message that merely mentions memory (a config error, the host running out) is not one.
 */
const GPU_WORDINGS = [PREFIX, /^gpu(:\d+)? out of memory\b/i, /\bwgpu backend: out ?of ?memory\b/i];

/** A failed run's `error`, or an API error, that says the GPU ran out of memory. */
export function isOom(e: unknown): boolean {
  if (e instanceof ApiError && e.code === OOM_CODE) return true;
  const msg = (typeof e === 'string' ? e : e instanceof Error ? e.message : '').trim();
  return GPU_WORDINGS.some((re) => re.test(msg));
}

/** The driver's report inside an OOM message ("wgpu backend: Out of Memory"), or the older message whole. */
export function oomDetail(msg: string | null | undefined): string | null {
  if (!msg) return null;
  const m = msg.trim();
  if (!PREFIX.test(m)) return m || null;
  const rest = m.replace(PREFIX, '');
  const inner = rest.startsWith('(') && rest.endsWith(')') ? rest.slice(1, -1) : rest;
  return inner.trim() || null;
}

export type OomFixKind = 'batch' | 'imgsz' | 'arch' | 'cpu';
export interface OomFix {
  kind: OomFixKind;
  label: string;
  /** The New run query that applies it next to `from=<run id>`. */
  query: Record<string, string>;
  /** The peak GPU memory forecast for the run with this fix (`fitFixes`), where it was forecast. */
  vram?: number;
  /** Whether that forecast fits the GPU with `OOM_HEADROOM` to spare. */
  fits?: boolean;
}

/** A smaller batch: half, rounded down, at least 1. */
export const halfBatch = (b: number) => Math.max(1, Math.floor(b / 2));

/**
 * What to try after a run ran out of memory, the most likely fix first: half the batch while it is above
 * `SMALL_BATCH`, else the next smaller image size, then the next smaller model. The CPU only when nothing smaller is
 * left on the GPU.
 */
export function oomFixes(c: Pick<TrainConfig, 'batch' | 'imgsz' | 'arch' | 'device'>): OomFix[] {
  const batch: OomFix[] = c.batch > 1 ? [{ kind: 'batch', label: `Retry with batch ${halfBatch(c.batch)}`, query: { batch: String(halfBatch(c.batch)) } }] : [];
  const size = [...IMGSZ_STEPS].reverse().find((s) => s < c.imgsz);
  const imgsz: OomFix[] = size ? [{ kind: 'imgsz', label: `Retry at ${size} px`, query: { imgsz: String(size) } }] : [];
  const smaller = smallerArchInFamily(c.arch);
  const arch: OomFix[] = smaller ? [{ kind: 'arch', label: `Retry with ${smaller}`, query: { arch: smaller } }] : [];
  const out = c.batch > SMALL_BATCH ? [...batch, ...imgsz, ...arch] : [...imgsz, ...arch, ...batch];
  if (!out.length && c.device !== 'cpu') out.push({ kind: 'cpu', label: 'Retry on the CPU', query: { device: 'cpu' } });
  return out;
}

// ---------- fixes that fit ----------
//
// A suggested fix must fix it: the retry Studio offers is one the memory estimate New run shows says fits. Both the run
// page's fixes and New run's "Use batch 8" are chosen here, from one forecast (`vramForecaster`) under one rule.

/** The share of the GPU's memory a fix leaves free: a forecast fits at or under 90% of it. */
export const OOM_HEADROOM = 0.1;

/** A forecast of a run's peak GPU memory: the server's measured figure, or the reference model's. */
export interface VramForecast {
  bytes: number;
  measured: boolean;
}
export type VramOf = (c: TrainConfig) => Promise<VramForecast | null>;

/** Whether `bytes` fits in `gpuBytes` with `OOM_HEADROOM` to spare. */
export const fitsGpu = (bytes: number, gpuBytes: number) => bytes <= gpuBytes * (1 - OOM_HEADROOM);

/**
 * The factor that makes the estimate agree with a run that ran out of GPU memory: when the estimate for its settings
 * (`failedBytes`) fit in `gpuBytes`, it was low, so estimates are scaled until those settings exceed the memory by 5%.
 * 1 when the estimate already said they would not fit.
 */
export function oomScale(failedBytes: number, gpuBytes: number): number {
  return failedBytes > 0 && failedBytes < gpuBytes * 1.05 ? (gpuBytes * 1.05) / failedBytes : 1;
}

/** `oomScale` for a run that ran out of memory, from the reference model of its settings (New run's correction). */
export function oomCalibration(failed: Pick<TrainConfig, 'arch' | 'imgsz' | 'batch' | 'amp' | 'device'>, gpuBytes: number | null): number {
  if (!gpuBytes || failed.device === 'cpu') return 1;
  return oomScale(referenceVram(failed.arch, failed.imgsz, failed.batch, failed.amp ?? false), gpuBytes);
}

/** A forecast as shown: a measured peak as it is, the reference model corrected by a failed run (`oomCalibration`). */
export const forecastBytes = (f: VramForecast, scale: number) => (f.measured ? f.bytes : f.bytes * scale);

/**
 * The forecast New run shows: the server's (`POST /api/runs/estimate`, feature `run_estimate`), else, or when it cannot
 * answer, the reference model (`lib/reference`), as New run's local estimate.
 */
export function vramForecaster(api: Pick<Api, 'estimateRun'>, system: SystemInfo | null | undefined, signal?: AbortSignal): VramOf {
  const local = (c: TrainConfig): VramForecast => ({ bytes: referenceVram(c.arch, c.imgsz, c.batch, c.amp ?? false), measured: false });
  const served = hasFeature(system, 'run_estimate');
  return async (c) => {
    if (c.device === 'cpu') return null;
    if (!served) return local(c);
    try {
      // A failed run's seed over 2^53 − 1 arrived rounded: left out (lib/seed, rule B); memory does not depend on it.
      const r = await api.estimateRun(withoutInexactSeed(c), signal);
      return r.est_peak_vram_bytes === null ? local(c) : { bytes: r.est_peak_vram_bytes, measured: r.vram_source === 'measured' };
    } catch {
      if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
      return local(c);
    }
  };
}

/** A batch and its forecast, and whether that fits. */
export interface BatchFit {
  batch: number;
  bytes: number;
  fits: boolean;
}

/**
 * The largest batch in 1..`max` whose forecast fits, by binary search (a forecast grows with the batch): one forecast
 * at a time, at most ⌈log2(max + 1)⌉ of them. When none fits, batch 1 with its forecast and `fits: false`; null where a
 * forecast is unknown, or `max` is below 1.
 */
export async function largestFittingBatch(max: number, bytesAt: (batch: number) => Promise<number | null>, fits: (bytes: number) => boolean): Promise<BatchFit | null> {
  if (max < 1) return null;
  // Every batch up to `lo` fits (none known: 0), every one above `hi` does not.
  let lo = 0;
  let hi = max;
  let fitBytes = 0;
  let missBytes = 0;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    const v = await bytesAt(mid);
    if (v === null) return null;
    if (fits(v)) {
      lo = mid;
      fitBytes = v;
    } else {
      hi = mid - 1;
      missBytes = v;
    }
  }
  // Nothing fits: the last forecast was batch 1's.
  return lo > 0 ? { batch: lo, bytes: fitBytes, fits: true } : { batch: 1, bytes: missBytes, fits: false };
}

/**
 * `oomFixes`, each forecast and made to fit: the batch fix takes the largest smaller batch whose forecast fits the GPU
 * with `OOM_HEADROOM` to spare (any whole batch: the trainer takes them all); a smaller image or model keeps the batch
 * when it fits that way, else takes the largest batch that does ("Retry at 640 px, batch 19"). A fix that fits at no
 * batch is dropped; when none fits, the CPU (which has no GPU memory to run out of) leads and the others follow marked
 * `fits: false`. Without the GPU's size the fixes come back as `oomFixes` gives them. Each fix asks for one forecast at
 * a time (`largestFittingBatch`), so at most one per fix is out at once; an abort stops the search (the forecaster
 * throws).
 */
export async function fitFixes(c: TrainConfig, gpuBytes: number | null, vramOf: VramOf, scale = 1): Promise<OomFix[]> {
  const base = oomFixes(c);
  if (!gpuBytes || c.device === 'cpu') return base;
  const fits = (v: number) => fitsGpu(v, gpuBytes);
  const out = await Promise.all(
    base.map(async (fix): Promise<OomFix> => {
      if (fix.kind === 'cpu') return { ...fix, fits: true };
      const change: Partial<TrainConfig> = fix.kind === 'imgsz' ? { imgsz: Number(fix.query.imgsz) } : fix.kind === 'arch' ? { arch: fix.query.arch as Arch } : {};
      const bytesAt = async (b: number) => {
        const f = await vramOf({ ...c, ...change, batch: b });
        return f ? forecastBytes(f, scale) : null;
      };
      // A smaller image or model tries the run's own batch first: it usually fits as it is.
      if (fix.kind !== 'batch') {
        const v = await bytesAt(c.batch);
        if (v === null) return fix;
        if (fits(v)) return { ...fix, vram: v, fits: true };
        if (c.batch === 1) return { ...fix, vram: v, fits: false };
      }
      const found = await largestFittingBatch(c.batch - 1, bytesAt, fits);
      if (!found) return fix;
      const { batch: b, bytes: v } = found;
      if (!found.fits) return { ...fix, vram: v, fits: false };
      if (fix.kind === 'batch') return { ...fix, label: `Retry with batch ${b}`, query: { batch: String(b) }, vram: v, fits: true };
      return { ...fix, label: `${fix.label}, batch ${b}`, query: { ...fix.query, batch: String(b) }, vram: v, fits: true };
    }),
  );
  const fit = out.filter((f) => f.fits !== false);
  if (fit.length) return fit;
  return [{ kind: 'cpu', label: 'Retry on the CPU', query: { device: 'cpu' }, fits: true }, ...out];
}

/**
 * The one change New run offers when its settings are forecast not to fit: the first of `fitFixes` for them that fits,
 * or null when none does (or the memory is unknown).
 */
export async function fitFix(c: TrainConfig, gpuBytes: number | null, vramOf: VramOf, scale = 1): Promise<OomFix | null> {
  const fixes = await fitFixes(c, gpuBytes, vramOf, scale);
  return fixes.find((f) => f.fits === true && f.kind !== 'cpu') ?? null;
}

/** A fix as New run's action names it: "Use batch 8", "Use 640 px, batch 16", "Use the CPU". */
export const useLabel = (f: OomFix) => f.label.replace(/^Retry (with|at|on) /, 'Use ');

/** `/runs/new?from=<id>&…`: New run prefilled from the run, with the fix applied. */
export function retryPath(runId: string, fix?: OomFix): string {
  const q = new URLSearchParams({ from: runId, ...(fix?.query ?? {}) });
  return `/runs/new?${q}`;
}

/** The short cause a list shows for a failed run, where there is one ("Out of GPU memory"). */
export function failureCause(run: { status: string; error: string | null }): string | null {
  return run.status === 'failed' && isOom(run.error) ? OOM_LABEL : null;
}
