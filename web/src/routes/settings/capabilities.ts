// Settings' System panel: `/api/system`'s `capabilities` (docs/API.md) read as plain rows. The server's reasons are
// written for a terminal (`off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)`); each row says what it
// means here in one sentence, then at most one thing to do about it, and keeps the server's own words for the driver
// report. Every action is one docs/CLI.md or docs/GPU.md confirms: `folo serve --tensor-cores` turns tensor cores on,
// `folo doctor --tensor-cores` says why they are off, and on DirectX 12 (no `shader-f16`, no cooperative matrices in
// wgpu) `FOLO_GPU_BACKEND=vulkan` picks the Vulkan adapter, which has both on the GPUs docs/GPU.md lists.

import type { Capabilities, Gpu } from '$lib/api/types';
import { sentence } from '$lib/format';

/** `on`: usable now; `off`: the hardware has it but it is not on; `none`: not usable on this server. */
export type CapState = 'on' | 'off' | 'none';

export interface CapRow {
  id: 'f16' | 'tensor_cores' | 'amp';
  label: string;
  state: CapState;
  /** The state in one word, next to its mark (colour is never the only cue). */
  word: string;
  /** What it means, in one sentence. Identifiers in backticks. */
  note: string;
  /** The one thing to do about it, when there is one. Identifiers in backticks. */
  action: string | null;
  /** The server's reason, verbatim, when it gave one. */
  raw: string | null;
}

const BACKENDS: Record<string, string> = { vulkan: 'Vulkan', dx12: 'DirectX 12', metal: 'Metal' };

/** "NVIDIA GeForce RTX 4080 SUPER · Vulkan" for `gpu:0`; the CPU's name for `cpu`; the label itself otherwise. */
export function deviceSummary(c: Pick<Capabilities, 'device'>, gpus: Gpu[], cpuName: string): string {
  if (c.device === 'cpu') return cpuName || 'CPU';
  const g = gpuOf(c, gpus);
  return g ? `${g.name} · ${BACKENDS[g.backend] ?? g.backend}` : c.device;
}

/** The adapter `device: auto` resolves to, if it is a GPU the server lists. */
function gpuOf(c: Pick<Capabilities, 'device'>, gpus: Gpu[]): Gpu | undefined {
  const m = /^gpu:(\d+)$/.exec(c.device);
  return m ? gpus.find((x) => x.ordinal === Number(m[1])) : undefined;
}

// DirectX 12 means a Windows server, so the Vulkan commands are in PowerShell (the default Windows shell).
const VULKAN_TC = "Try Vulkan, in PowerShell: `$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores`.";
const SERVE_TC = 'Start `folo serve --tensor-cores` to use them.';

function tensorRow(c: Capabilities, dx12: boolean): Pick<CapRow, 'note' | 'action'> {
  const t = c.tensor_cores;
  if (t.enabled) return { note: 'Convolutions and matrix products run on the tensor cores (f16 in, f32 out).', action: null };
  const r = (t.reason ?? '').toLowerCase();
  if (c.device === 'cpu') return { note: 'Training and inference run on the CPU, which has none.', action: null };
  if (r.includes('not requested')) return { note: 'This GPU has them, but the server did not turn them on.', action: SERVE_TC };
  if (r.includes('self-test')) return { note: 'Their start-up self-test failed, so the standard kernels run.', action: 'Run `folo doctor --tensor-cores` to see why.' };
  if (!t.available || r.includes('cooperative matri')) {
    return dx12 || r.includes('dx12')
      ? { note: 'DirectX 12 has no cooperative matrices in wgpu, so the standard kernels run.', action: VULKAN_TC }
      : { note: 'This GPU has no f16 cooperative matrices, so the standard kernels run.', action: null };
  }
  return { note: t.reason ? sentence(t.reason) : 'Off on this server.', action: 'Run `folo doctor --tensor-cores` to see why.' };
}

function ampRow(c: Capabilities): Pick<CapRow, 'note' | 'action'> {
  if (c.amp.available) return { note: 'Runs can train in mixed precision (f16 compute, f32 weights): the Mixed precision switch in New run.', action: null };
  if (c.device === 'cpu') return { note: 'Needs a GPU with tensor cores; this server trains on the CPU, in f32.', action: null };
  if (!c.tensor_cores.enabled) {
    // A failed self-test is not fixed by asking for them again (the server already did): say so, as the row above does.
    if (/self-test/i.test(c.tensor_cores.reason ?? '') || /self-test/i.test(c.amp.reason ?? ''))
      return { note: 'Needs the tensor cores, whose self-test failed on this server, so runs train in f32.', action: 'Run `folo doctor --tensor-cores` to see why.' };
    return c.tensor_cores.available
      ? { note: 'Needs the tensor cores, which are off on this server.', action: SERVE_TC.replace('them', 'it') }
      : { note: 'Needs a GPU with tensor cores, so runs train in f32.', action: null };
  }
  return { note: c.amp.reason ? sentence(c.amp.reason) : 'Not available on this server.', action: null };
}

/**
 * The three capability rows, in the order a run uses them: f16 shaders, then tensor cores, then mixed precision.
 * `gpus` (from `/api/system`) names the backend of the adapter in use, which decides whether Vulkan is worth a try.
 */
export function capabilityRows(c: Capabilities, gpus: Gpu[] = []): CapRow[] {
  const t = c.tensor_cores;
  const dx12 = gpuOf(c, gpus)?.backend === 'dx12';
  return [
    {
      id: 'f16',
      label: 'Half-precision shaders',
      state: c.shader_f16 ? 'on' : 'none',
      word: c.shader_f16 ? 'Available' : 'Not available',
      ...(c.shader_f16
        ? { note: 'f16 models run in half precision on the GPU.', action: null }
        : c.device === 'cpu'
          ? { note: 'The CPU computes in f32, so f16 inference runs in f32.', action: null }
          : dx12
            ? { note: 'DirectX 12 computes in f32 only, so f16 inference runs in f32.', action: "Try Vulkan, in PowerShell: `$env:FOLO_GPU_BACKEND='vulkan'; folo serve`." }
            : { note: 'This GPU computes in f32 only, so f16 inference runs in f32.', action: null }),
      raw: null,
    },
    {
      id: 'tensor_cores',
      label: 'Tensor cores',
      state: t.enabled ? 'on' : t.available ? 'off' : 'none',
      word: t.enabled ? 'On' : t.available ? 'Off' : 'Not available',
      ...tensorRow(c, dx12),
      raw: t.enabled ? null : t.reason,
    },
    {
      id: 'amp',
      label: 'Mixed precision training',
      state: c.amp.available ? 'on' : 'none',
      word: c.amp.available ? 'Available' : 'Not available',
      ...ampRow(c),
      raw: c.amp.available ? null : c.amp.reason,
    },
  ];
}

/** The driver report: each row's server reason under its label, as plain text to copy into an issue. */
export function driverReport(rows: CapRow[], device: string, summary: string): string {
  return [`device: ${device} (${summary})`, ...rows.filter((r) => r.raw).map((r) => `${r.label}: ${r.raw}`)].join('\n');
}
