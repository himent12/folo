// The live card's device line: the run's own device label, plus live load only from the GPU that run actually uses.

import type { Gpu, Run } from '$lib/api/types';
import { deviceName, pct } from '$lib/format';

/**
 * The GPU a run trains on, or null for CPU runs and when it cannot be told. `gpu:<n>` picks ordinal n; `auto` matches
 * the run's resolved device label (e.g. "NVIDIA GeForce RTX 4080 SUPER (vulkan)") against the adapter names.
 */
export function runGpu(run: Pick<Run, 'device' | 'config'>, gpus: Gpu[]): Gpu | null {
  const want = run.config.device;
  if (want === 'cpu') return null;
  const m = /^gpu:(\d+)$/.exec(want);
  if (m) return gpus.find((g) => g.ordinal === Number(m[1])) ?? null;
  const label = run.device.toLowerCase();
  if (label === 'cpu' || label.startsWith('cpu')) return null;
  const hits = gpus.filter((g) => label.includes(g.name.toLowerCase()));
  return hits.length === 1 ? hits[0]! : null;
}

/** "RTX 4080 SUPER · GPU 93% busy": the run's device, with utilisation (`pct`, no space) when its GPU reports one. */
export function deviceLine(run: Pick<Run, 'device' | 'config'>, gpus: Gpu[]): string {
  const name = deviceName(run.device);
  const g = runGpu(run, gpus);
  return g && g.util !== null ? `${name} · GPU ${pct(g.util, 0)} busy` : name;
}
