// Settings' System panel: `/api/system`'s `capabilities` (docs/API.md) read as plain rows. The server's reasons are
// written for a terminal (`off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)`); each row says what it
// means here in one sentence, then at most one thing to do about it (a `Fix`: what it does, the shell, the command,
// set by ui/Remedy.svelte as an unwrapped, copyable line), and keeps the server's own words for the driver report.
// Every fix is one docs/CLI.md or docs/GPU.md confirms: `folo serve --tensor-cores` turns tensor cores on,
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
  /** The one thing to do about it, when there is one. */
  fix: Fix | null;
  /** The server's reason, verbatim, when it gave one. */
  raw: string | null;
}

/**
 * A command that fixes something: `say` leads into it in plain words (ending in a colon when no shell is named),
 * `shell` names the shell it is written for ("In PowerShell:"), `command` is the line to run, whole.
 */
export interface Fix {
  say: string;
  shell?: string;
  command: string;
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
export const VULKAN = "$env:FOLO_GPU_BACKEND='vulkan'; folo serve";
const VULKAN_TC: Fix = { say: 'Restart the server on Vulkan to use them.', shell: 'PowerShell', command: `${VULKAN} --tensor-cores` };
const SERVE_TC: Fix = { say: 'Restart the server with them on:', command: 'folo serve --tensor-cores' };
const DOCTOR: Fix = { say: 'To see why, run:', command: 'folo doctor --tensor-cores' };

function tensorRow(c: Capabilities, dx12: boolean): Pick<CapRow, 'note' | 'fix'> {
  const t = c.tensor_cores;
  if (t.enabled) return { note: 'Convolutions and matrix products run on the tensor cores (f16 in, f32 out).', fix: null };
  const r = (t.reason ?? '').toLowerCase();
  if (c.device === 'cpu') return { note: 'Training and inference run on the CPU, which has none.', fix: null };
  if (r.includes('not requested')) return { note: 'This GPU has them, but the server did not turn them on.', fix: SERVE_TC };
  if (r.includes('self-test')) return { note: 'Their start-up self-test failed, so the standard kernels run.', fix: DOCTOR };
  if (!t.available || r.includes('cooperative matri')) {
    return dx12 || r.includes('dx12')
      ? { note: 'Tensor cores need the Vulkan backend; this server runs DirectX 12.', fix: VULKAN_TC }
      : { note: 'This GPU has no tensor cores FOLO can use, so the standard kernels run.', fix: null };
  }
  return { note: t.reason ? sentence(t.reason) : 'Off on this server.', fix: DOCTOR };
}

function ampRow(c: Capabilities): Pick<CapRow, 'note' | 'fix'> {
  if (c.amp.available) return { note: 'Runs can train in mixed precision (f16 compute, f32 weights): the Mixed precision switch in New run.', fix: null };
  if (c.device === 'cpu') return { note: 'Needs a GPU with tensor cores; this server trains on the CPU, in f32.', fix: null };
  if (!c.tensor_cores.enabled) {
    // A failed self-test is not fixed by asking for them again (the server already did): say so, as the row above does.
    if (/self-test/i.test(c.tensor_cores.reason ?? '') || /self-test/i.test(c.amp.reason ?? ''))
      return { note: 'Needs the tensor cores, whose self-test failed on this server, so runs train in f32.', fix: DOCTOR };
    return c.tensor_cores.available
      ? { note: 'Needs the tensor cores, which are off on this server.', fix: SERVE_TC }
      : { note: 'Needs tensor cores this server can use, so runs train in f32.', fix: null };
  }
  return { note: c.amp.reason ? sentence(c.amp.reason) : 'Not available on this server.', fix: null };
}

/**
 * The three capability rows, in the order a run uses them: f16 shaders, then tensor cores, then mixed precision.
 * `gpus` (from `/api/system`) names the backend of the adapter in use, which decides whether Vulkan is worth a try.
 */
export function capabilityRows(c: Capabilities, gpus: Gpu[] = []): CapRow[] {
  const t = c.tensor_cores;
  const dx12 = gpuOf(c, gpus)?.backend === 'dx12';
  return oneCommandEach([
    {
      id: 'f16',
      label: 'Half-precision shaders',
      state: c.shader_f16 ? 'on' : 'none',
      word: c.shader_f16 ? 'Available' : 'Not available',
      ...(c.shader_f16
        ? { note: 'f16 models run in half precision on the GPU.', fix: null }
        : c.device === 'cpu'
          ? { note: 'The CPU computes in f32, so f16 inference runs in f32.', fix: null }
          : dx12
            ? { note: 'DirectX 12 computes in f32 only, so f16 inference runs in f32.', fix: { say: 'Restart the server on Vulkan to compute in f16.', shell: 'PowerShell', command: VULKAN } }
            : { note: 'This GPU computes in f32 only, so f16 inference runs in f32.', fix: null }),
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
  ]);
}

/**
 * Each command shows once in the panel: a row whose fix is a command an earlier row already shows (Mixed precision,
 * when the tensor cores are off) keeps its sentence and points to that row instead of repeating the block.
 */
function oneCommandEach(rows: CapRow[]): CapRow[] {
  const first = new Map<string, string>();
  return rows.map((r) => {
    if (!r.fix) return r;
    const by = first.get(r.fix.command);
    if (!by) {
      first.set(r.fix.command, r.label);
      return r;
    }
    return { ...r, note: `${r.note} See the command under ${by}.`, fix: null };
  });
}

/** The driver report: each row's server reason under its label, as plain text to copy into an issue. */
export function driverReport(rows: CapRow[], device: string, summary: string): string {
  return [`device: ${device} (${summary})`, ...rows.filter((r) => r.raw).map((r) => `${r.label}: ${r.raw}`)].join('\n');
}
