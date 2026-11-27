// Pure CLI command builders for Model Detail ExportDialog: `folo export` (with dynamic ONNX axes,
// decode, and half precision) and `folo predict` (standard, SAHI tiled `--tile`, `--tta`, and ByteTrack `--track`).
// Every flag matches crates/folo-cli/src/cli/infer.rs and docs/CLI.md verbatim.
import type { ExportFormat, Model, OnnxSwitches } from '$lib/api/types';
import { quote, weightsRef } from '$lib/trainCli';

/** ONNX dynamic axis choices matching `folo export` (`--dynamic-batch`, `--dynamic-hw`, `--dynamic`). */
export type OnnxDynamicAxes = 'static' | 'batch' | 'hw' | 'full';

/** Quick `folo predict` CLI snippets shown alongside export commands. */
export type PredictSnippetMode = 'standard' | 'sahi' | 'tta' | 'track';

export interface ModelCliResult {
  line: string;
  /** The placeholder filename when `model.path` is unavailable (published weights need none). */
  placeholder: string | null;
}

export const ONNX_AXES_OPTIONS: readonly { value: OnnxDynamicAxes; label: string; desc: string }[] = [
  { value: 'static', label: 'Static', desc: 'Fixed batch 1 and training H×W.' },
  { value: 'batch', label: 'Batch', desc: 'Symbolic batch, fixed H×W (`--dynamic-batch`).' },
  { value: 'hw', label: 'H×W', desc: 'Fixed batch 1, symbolic H×W in multiples of 32 (`--dynamic-hw`).' },
  { value: 'full', label: 'Full', desc: 'Symbolic batch, height and width (`--dynamic`).' },
];

export const PREDICT_SNIPPET_OPTIONS: readonly { value: PredictSnippetMode; label: string; desc: string }[] = [
  { value: 'standard', label: 'Standard', desc: 'Single-pass inference on images or a folder.' },
  { value: 'sahi', label: 'SAHI tiled', desc: 'Overlapping 640 px crops (`--tile 640 --tile-overlap 0.2`) for small objects.' },
  { value: 'tta', label: 'TTA', desc: 'Multi-scale + horizontal flip ensembling (`--tta`).' },
  { value: 'track', label: 'ByteTrack', desc: 'Multi-object tracking across ordered frames (`--track`).' },
];

function modelFlag(model: Model): { part: string; placeholder: string | null } {
  const ref = weightsRef(model.id, model);
  if (ref.kind === 'placeholder') {
    return { part: `--model ${ref.value}`, placeholder: ref.file };
  }
  if (ref.value.startsWith('-')) {
    return { part: quote(`--model=${ref.value}`), placeholder: null };
  }
  return { part: `--model ${quote(ref.value)}`, placeholder: null };
}

/**
 * Converts UI ONNX export options into the server's `OnnxSwitches` payload.
 * Never sends `dynamic_batch` / `dynamic_hw` unless the server advertises `export_dynamic_axes`
 * and an independent axis mode (`batch` or `hw`) is chosen.
 */
export function onnxSwitchesForRequest(opts: {
  decode: boolean;
  half: boolean;
  axes: OnnxDynamicAxes;
  supportsSplitAxes: boolean;
}): OnnxSwitches {
  if (opts.axes === 'full') {
    return { decode: opts.decode, dynamic: true, half: opts.half };
  }
  if (opts.supportsSplitAxes && opts.axes === 'batch') {
    return { decode: opts.decode, dynamic: false, dynamic_batch: true, dynamic_hw: false, half: opts.half };
  }
  if (opts.supportsSplitAxes && opts.axes === 'hw') {
    return { decode: opts.decode, dynamic: false, dynamic_batch: false, dynamic_hw: true, half: opts.half };
  }
  return { decode: opts.decode, dynamic: false, half: opts.half };
}

/** Builds a copy-ready `folo export` command matching `crates/folo-cli/src/cli/infer.rs`. */
export function buildExportCli(
  model: Model,
  opts: { format: ExportFormat; decode: boolean; half: boolean; axes: OnnxDynamicAxes },
): ModelCliResult {
  const { part, placeholder } = modelFlag(model);
  const parts = ['folo export', part, '--format', opts.format];
  if (opts.format === 'onnx') {
    if (opts.decode) parts.push('--decode');
    if (opts.axes === 'full') parts.push('--dynamic');
    else if (opts.axes === 'batch') parts.push('--dynamic-batch');
    else if (opts.axes === 'hw') parts.push('--dynamic-hw');
    if (opts.half) parts.push('--half');
  }
  return { line: parts.join(' '), placeholder };
}

/** Builds a copy-ready `folo predict` snippet (Standard, SAHI `--tile`, `--tta`, or ByteTrack `--track`). */
export function buildPredictCli(model: Model, mode: PredictSnippetMode): ModelCliResult {
  const { part, placeholder } = modelFlag(model);
  const target = mode === 'track' ? '<frames_dir>' : '<images>';
  const parts = ['folo predict', part, target];
  if (mode === 'sahi') {
    parts.push('--tile', '640', '--tile-overlap', '0.2');
  } else if (mode === 'tta') {
    parts.push('--tta');
  } else if (mode === 'track') {
    parts.push('--track');
  }
  return { line: parts.join(' '), placeholder };
}
