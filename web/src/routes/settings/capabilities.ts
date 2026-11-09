// Settings' System panel: `/api/system`'s `capabilities` (docs/API.md) read as plain rows. The server's reasons are
// written for a terminal (`off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)`); each row says what it
// means here, in a sentence, and keeps the server's own words for the technical details.

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
  /** What it does when usable; why not (and what to do) otherwise. Identifiers in backticks. */
  note: string;
  /** The server's reason, verbatim, when it gave one. */
  raw: string | null;
}

const BACKENDS: Record<string, string> = { vulkan: 'Vulkan', dx12: 'DirectX 12', metal: 'Metal' };

/** "NVIDIA GeForce RTX 4080 SUPER · Vulkan" for `gpu:0`; the CPU's name for `cpu`; the label itself otherwise. */
export function deviceSummary(c: Pick<Capabilities, 'device'>, gpus: Gpu[], cpuName: string): string {
  if (c.device === 'cpu') return cpuName || 'CPU';
  const m = /^gpu:(\d+)$/.exec(c.device);
  const g = m ? gpus.find((x) => x.ordinal === Number(m[1])) : undefined;
  return g ? `${g.name} · ${BACKENDS[g.backend] ?? g.backend}` : c.device;
}

function tensorNote(c: Capabilities): string {
  const t = c.tensor_cores;
  if (t.enabled) return 'Convolutions and matrix products run on the tensor cores (f16 in, f32 out).';
  const r = (t.reason ?? '').toLowerCase();
  if (c.device === 'cpu') return 'Training and inference run on the CPU, which has none.';
  if (r.includes('not requested')) return 'This GPU has them, but the server did not turn them on. Start it with `folo serve --tensor-cores` to use them.';
  if (r.includes('self-test')) return 'The start-up self-test failed, so the server uses its standard kernels.';
  if (!t.available || r.includes('cooperative matri')) return 'This GPU (or its driver backend) has no f16 cooperative matrices, so the standard kernels run.';
  return t.reason ? sentence(t.reason) : 'Off on this server.';
}

function ampNote(c: Capabilities): string {
  if (c.amp.available) return 'Runs can train in mixed precision (f16 compute, f32 weights): the Mixed precision switch in New run.';
  if (c.device === 'cpu') return 'Needs a GPU with tensor cores; this server trains on the CPU.';
  if (!c.tensor_cores.enabled) return c.tensor_cores.available ? 'Needs the tensor cores, which are off on this server.' : 'Needs a GPU with tensor cores.';
  return c.amp.reason ? sentence(c.amp.reason) : 'Not available on this server.';
}

/** The three capability rows, in the order a run uses them: f16 shaders, then tensor cores, then mixed precision. */
export function capabilityRows(c: Capabilities): CapRow[] {
  const t = c.tensor_cores;
  return [
    {
      id: 'f16',
      label: 'Half-precision shaders',
      state: c.shader_f16 ? 'on' : 'none',
      word: c.shader_f16 ? 'Available' : 'Not available',
      note: c.shader_f16
        ? 'f16 models run in half precision on the GPU.'
        : c.device === 'cpu'
          ? 'The CPU computes in f32, so f16 inference runs in f32.'
          : 'This GPU computes in f32 only, so f16 inference runs in f32.',
      raw: null,
    },
    {
      id: 'tensor_cores',
      label: 'Tensor cores',
      state: t.enabled ? 'on' : t.available ? 'off' : 'none',
      word: t.enabled ? 'On' : t.available ? 'Off' : 'Not available',
      note: tensorNote(c),
      raw: t.enabled ? null : t.reason,
    },
    {
      id: 'amp',
      label: 'Mixed precision training',
      state: c.amp.available ? 'on' : 'none',
      word: c.amp.available ? 'Available' : 'Not available',
      note: ampNote(c),
      raw: c.amp.available ? null : c.amp.reason,
    },
  ];
}
