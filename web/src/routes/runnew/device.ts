/** How the form names a device: in its errors, in the Device field's message and in the Summary's Device row. */
import type { Resource } from '$lib/api/resource';
import type { SystemInfo } from '$lib/api/types';
import { deviceName } from '$lib/format';

/**
 * A GPU the form holds (from the tab's draft, a copied run or a pick before the list changed) that the server's device
 * list does not name: Start waits for another device rather than letting the server refuse it (stress d2-runs; Desktop's
 * `nr_device_gone`). Unknown until the list has loaded.
 */
export function missingDevice(device: string, system: SystemInfo | undefined): { device: string; gone: boolean; detail: string } | null {
  const d = canonicalDevice(device);
  if (!system || !d.startsWith('gpu:') || system.gpus.some((g) => `gpu:${g.ordinal}` === d)) return null;
  return { device, gone: true, detail: '' };
}

/**
 * A device as the form's select names it, from any spelling the trainer accepts (`resolve_device` in
 * crates/folo-vision/src/train/config.rs): `GPU:00`, `cuda:1`, `gpu` and ` CPU ` become `gpu:0`, `gpu:1`, `gpu:0` and
 * `cpu`; a run created over the API keeps its own spelling, and a copy of it must not read as a GPU that is gone.
 * Anything else is kept as it is (the server says what is wrong with it).
 */
export function canonicalDevice(device: string): string {
  const s = device.trim().toLowerCase();
  if (s === 'cpu') return 'cpu';
  if (s === 'auto' || s === '') return 'auto';
  if (s === 'gpu' || s === 'cuda') return 'gpu:0';
  const m = /^(?:gpu|cuda):\+?(\d+)$/.exec(s);
  return m && Number.isSafeInteger(Number(m[1])) ? `gpu:${Number(m[1])}` : device;
}

/** A device in a sentence, named once with its value: "Auto", "GPU 1 (gpu:1)", "CPU" (values elsewhere: `gpu:1`). */
export const devWord = (d: string) => (d === 'auto' ? 'Auto' : d === 'cpu' ? 'CPU' : `${d.replace(/^gpu:/, 'GPU ')} (${d})`);

/** The Device field's line after Start was refused: the device is unusable here, or gone, and what to pick instead. */
export function deviceMessage(deviceErr: { device: string; gone: boolean; detail: string } | null, system: Resource<SystemInfo>): string | null {
  const e = deviceErr;
  if (!e) return null;
  if (!e.gone) return `This server cannot train on ${devWord(e.device)}. Pick Auto or CPU.`;
  const others = [...(system.data?.gpus ?? []).map((g) => `gpu:${g.ordinal}`).filter((d) => d !== e.device), 'cpu'].map(devWord);
  return `${devWord(e.device)} is no longer available. Pick Auto, ${others.length > 1 ? `${others.slice(0, -1).join(', ')} or ${others.at(-1)}` : others[0]}.`;
}

/** The Summary's Device row: the device and the GPU it names. */
export function deviceLabelOf(device: string, gpu: SystemInfo['gpus'][number] | undefined, system: Resource<SystemInfo>): string {
  const shortGpu = deviceName;
  if (device === 'auto') return `Auto${gpu ? ` · ${shortGpu(gpu.name)}` : ''}`;
  if (device === 'cpu') return 'CPU';
  const g = system.data?.gpus.find((x) => `gpu:${x.ordinal}` === device);
  return g ? `${device} · ${shortGpu(g.name)}` : device;
}
