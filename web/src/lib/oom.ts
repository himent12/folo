// Out of GPU memory: the one failure whose fixes are known in advance (docs/API.md `507 gpu_out_of_memory`). A run
// that runs out fails with `Run.error` = "the GPU ran out of memory (<first line of the driver report>)"; older
// servers and `folo serve --mock` word it differently ("gpu:0 out of memory: ..."), so both are recognised.

import { ApiError } from '$lib/api/client';
import type { Arch, TrainConfig } from '$lib/api/types';

export const OOM_CODE = 'gpu_out_of_memory';
/** How lists name the cause instead of the driver text. */
export const OOM_LABEL = 'Out of GPU memory';
/** At or below this batch, halving it again rarely helps: a smaller image or model leads. */
export const SMALL_BATCH = 4;
/** The image sizes New run offers, smallest first. */
export const IMGSZ_STEPS = [320, 480, 640, 960, 1280];
const ARCHES: Arch[] = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];

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
  const smaller = ARCHES[ARCHES.indexOf(c.arch) - 1];
  const arch: OomFix[] = smaller ? [{ kind: 'arch', label: `Retry with ${smaller}`, query: { arch: smaller } }] : [];
  const out = c.batch > SMALL_BATCH ? [...batch, ...imgsz, ...arch] : [...imgsz, ...arch, ...batch];
  if (!out.length && c.device !== 'cpu') out.push({ kind: 'cpu', label: 'Retry on the CPU', query: { device: 'cpu' } });
  return out;
}

/** `/runs/new?from=<id>&…`: New run prefilled from the run, with the fix applied. */
export function retryPath(runId: string, fix?: OomFix): string {
  const q = new URLSearchParams({ from: runId, ...(fix?.query ?? {}) });
  return `/runs/new?${q}`;
}

/** The short cause a list shows for a failed run, where there is one ("Out of GPU memory"). */
export function failureCause(run: { status: string; error: string | null }): string | null {
  return run.status === 'failed' && isOom(run.error) ? OOM_LABEL : null;
}
