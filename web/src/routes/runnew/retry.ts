import type { Arch, Run } from '$lib/api/types';

/** A retry of a run that ran out of GPU memory: each setting changed since, as it was and as it is now. */
export function oomChangesOf(oomRun: Run | null, batchN: number, imgszN: number, arch: Arch, device: string) {
  const c = oomRun?.config;
  if (!c) return [];
  const out: { key: 'batch' | 'imgsz' | 'model' | 'device'; was: string; now: string }[] = [];
  if (c.batch !== batchN) out.push({ key: 'batch', was: `batch ${c.batch}`, now: `batch ${batchN}` });
  if (c.imgsz !== imgszN) out.push({ key: 'imgsz', was: `${c.imgsz} px`, now: `${imgszN} px` });
  if (c.arch !== arch) out.push({ key: 'model', was: c.arch, now: arch });
  if (c.device !== device) out.push({ key: 'device', was: c.device, now: device });
  return out;
}
