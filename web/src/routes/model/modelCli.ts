// Pure CLI command builders for Model Detail ExportDialog: `folo export` (with dynamic ONNX axes,
// decode, and half precision), `folo val` (with `--save-thresholds`), `folo predict` (standard, SAHI tiled
// `--tile`, `--tta`, ByteTrack `--track`, and optional `--thresholds`), and `folo quantize` (int8 calibration
// with selective-f32 sensitivity policy).
// Every flag matches crates/folo-cli/src/cli/infer.rs and docs/CLI.md verbatim.
import type { ExportFormat, Model, OnnxSwitches } from '$lib/api/types';
import { quote, weightsRef } from '$lib/trainCli';

/** ONNX dynamic axis choices matching `folo export` (`--dynamic-batch`, `--dynamic-hw`, `--dynamic`). */
export type OnnxDynamicAxes = 'static' | 'batch' | 'hw' | 'full';

/** Quick `folo predict` CLI snippets shown alongside export commands. */
export type PredictSnippetMode = 'standard' | 'sahi' | 'tta' | 'track';

/** Selective-f32 calibration presets matching `folo quantize` (`--f32-policy`, `--max-f32-macs`, `--sqnr-threshold`). */
export type QuantizePreset = 'auto' | 'accurate' | 'p5-tail' | 'none';

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

export const QUANTIZE_PRESET_OPTIONS: readonly { value: QuantizePreset; label: string; desc: string }[] = [
  {
    value: 'auto',
    label: 'Auto 5%',
    desc: 'Sensitivity-ranked selective f32 (`--f32-policy auto --max-f32-macs 0.05 --sqnr-threshold 35`). Run with `folo predict --precision int8`.',
  },
  {
    value: 'accurate',
    label: '10% f32',
    desc: 'Higher-sensitivity selective f32 budget (`--max-f32-macs 0.10 --sqnr-threshold 40`) to protect accuracy on hard layers.',
  },
  {
    value: 'p5-tail',
    label: 'P5 tail',
    desc: 'Keep `neck.out5.*` in f32 (`--f32-policy p5-tail`) and quantize all other eligible convs to int8.',
  },
  {
    value: 'none',
    label: 'Full int8',
    desc: 'Quantize every eligible backbone and neck conv to int8 (`--f32-policy none`).',
  },
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

function flagPart(flag: string, value: string): string {
  return value.startsWith('-') ? quote(`${flag}=${value}`) : `${flag} ${quote(value)}`;
}

/** Builds a copy-ready `folo val` command, optionally writing calibrated per-class F1 thresholds (`--save-thresholds`). */
export function buildValCli(
  model: Model,
  opts?: {
    datasetPath?: string | null;
    split?: string | null;
    saveThresholds?: boolean;
    thresholdsPath?: string;
  },
): ModelCliResult {
  const { part, placeholder } = modelFlag(model);
  const dataPart = opts?.datasetPath ? flagPart('--data', opts.datasetPath) : '--data <dataset>';
  const parts = ['folo val', part, dataPart];
  if (opts?.split && opts.split !== 'val') parts.push('--split', opts.split);
  if (opts?.saveThresholds !== false) {
    const file = opts?.thresholdsPath?.trim() || 'thresholds.json';
    parts.push(flagPart('--save-thresholds', file));
  }
  return { line: parts.join(' '), placeholder };
}

/**
 * Builds a copy-ready `folo predict` snippet (Standard, SAHI `--tile`, `--tta`, or ByteTrack `--track`),
 * optionally applying calibrated per-class thresholds (`--thresholds <FILE>`, mutually exclusive with `--conf`).
 */
export function buildPredictCli(
  model: Model,
  mode: PredictSnippetMode,
  opts?: { thresholdsPath?: string | null },
): ModelCliResult {
  const { part, placeholder } = modelFlag(model);
  const target = mode === 'track' ? '<frames_dir>' : '<images>';
  const parts = ['folo predict', part, target];
  if (opts?.thresholdsPath) {
    parts.push(flagPart('--thresholds', opts.thresholdsPath));
  }
  if (mode === 'sahi') {
    parts.push('--tile', '640', '--tile-overlap', '0.2');
  } else if (mode === 'tta') {
    parts.push('--tta');
  } else if (mode === 'track') {
    parts.push('--track');
  }
  return { line: parts.join(' '), placeholder };
}

/**
 * Builds a copy-ready `folo quantize` int8 calibration command matching `crates/folo-cli/src/cli/infer.rs`.
 * Auto-only sensitivity flags (`--max-f32-macs`, `--sqnr-threshold`) are emitted strictly with `--f32-policy auto`.
 */
export function buildQuantizeCli(model: Model, preset: QuantizePreset): ModelCliResult {
  const { part, placeholder } = modelFlag(model);
  const parts = ['folo quantize', part, '--data', '<dataset>'];
  if (preset === 'auto') {
    parts.push('--f32-policy', 'auto', '--max-f32-macs', '0.05', '--sqnr-threshold', '35');
  } else if (preset === 'accurate') {
    parts.push('--f32-policy', 'auto', '--max-f32-macs', '0.10', '--sqnr-threshold', '40');
  } else if (preset === 'p5-tail') {
    parts.push('--f32-policy', 'p5-tail');
  } else {
    parts.push('--f32-policy', 'none');
  }
  return { line: parts.join(' '), placeholder };
}
