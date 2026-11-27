/**
 * Collapsible CLI commands section for Model Detail ExportDialog: shows copy-ready `folo export`
 * (synced with the selected format, decoded/raw output, half precision, and dynamic ONNX axes),
 * `folo val --save-thresholds` calibration, `folo predict` quick snippets (Standard, SAHI `--tile`,
 * `--tta`, ByteTrack `--track`, with optional `--thresholds`), and `folo quantize` selective-f32
 * int8 calibration presets.
 */
import Check from '$icons/check';
import ChevronRight from '$icons/chevron-right';
import { computed, css, effect, For, signal } from 'fw';
import type { ExportFormat, Model } from '$lib/api/types';
import { safeFilename } from '$lib/tableExport';
import CommandLine from '$lib/ui/CommandLine';
import Segmented from '$lib/ui/Segmented';
import {
  buildExportCli,
  buildPredictCli,
  buildQuantizeCli,
  buildValCli,
  ONNX_AXES_OPTIONS,
  PREDICT_SNIPPET_OPTIONS,
  QUANTIZE_PRESET_OPTIONS,
  type OnnxDynamicAxes,
  type PredictSnippetMode,
  type QuantizePreset,
} from './modelCli';

export interface ExportCliPanelProps {
  model: Model;
  format: ExportFormat;
  decode: boolean;
  half: boolean;
  axes: OnnxDynamicAxes;
  /** True on servers without `export_dynamic_axes`: shows a CLI-only axis picker that never alters server export switches. */
  showAxesPicker: boolean;
  /** Optional local dataset path for `folo val --data`. */
  datasetPath?: string | null;
  /** Evaluated split (`train` adds `--split train` to `folo val`). */
  evalSplit?: string | null;
}

const AXES_SEG = ONNX_AXES_OPTIONS.map((o) => ({ value: o.value, label: o.label }));
const PREDICT_SEG = PREDICT_SNIPPET_OPTIONS.map((o) => ({ value: o.value, label: o.label }));
const QUANTIZE_SEG = QUANTIZE_PRESET_OPTIONS.map((o) => ({ value: o.value, label: o.label }));

export default function ExportCliPanel(props: ExportCliPanelProps) {
  const open = signal(false);
  const predictMode = signal<PredictSnippetMode>('sahi');
  const saveThresholds = signal(true);
  const useThresholds = signal(false);
  const quantizePreset = signal<QuantizePreset>('auto');
  /** Local CLI-only dynamic axis override when the server does not expose split dynamic axes. */
  const cliAxes = signal<OnnxDynamicAxes>(props.axes);
  effect(() => {
    cliAxes.set(props.axes);
  });

  const activeAxes = computed<OnnxDynamicAxes>(() => (props.showAxesPicker ? cliAxes.get() : props.axes));
  const thresholdsFile = computed(() => safeFilename(`${props.model.name || 'model'}-thresholds`, 'json'));

  const exportCmd = computed(() =>
    buildExportCli(props.model, {
      format: props.format,
      decode: props.decode,
      half: props.half,
      axes: activeAxes.get(),
    }),
  );
  const valCmd = computed(() =>
    buildValCli(props.model, {
      datasetPath: props.datasetPath,
      split: props.evalSplit,
      saveThresholds: saveThresholds.get(),
      thresholdsPath: thresholdsFile.get(),
    }),
  );
  const predictCmd = computed(() =>
    buildPredictCli(props.model, predictMode.get(), {
      thresholdsPath: useThresholds.get() ? thresholdsFile.get() : null,
    }),
  );
  const predictDesc = computed(
    () => PREDICT_SNIPPET_OPTIONS.find((o) => o.value === predictMode.get())?.desc ?? '',
  );
  const quantizeCmd = computed(() => buildQuantizeCli(props.model, quantizePreset.get()));
  const quantizeDesc = computed(
    () => QUANTIZE_PRESET_OPTIONS.find((o) => o.value === quantizePreset.get())?.desc ?? '',
  );
  const placeholder = computed(
    () =>
      exportCmd.get().placeholder ??
      valCmd.get().placeholder ??
      predictCmd.get().placeholder ??
      quantizeCmd.get().placeholder,
  );

  return (
    <div class="cli-wrap" data-export-cli>
      <button
        type="button"
        class="cli-toggle"
        aria-expanded={open.get()}
        onclick={() => open.set(!open.peek())}
      >
        <ChevronRight size={14} class={open.get() ? 'chev open' : 'chev'} aria-hidden="true" />
        <span class="cli-title">CLI commands</span>
        <span class="cli-meta">export, val, predict &amp; quantize</span>
      </button>
      {open.get() ? (
        <div class="cli-body">
          <div class="cli-block">
            <div class="cli-head">
              <span class="cli-label">Export</span>
              {props.format === 'onnx' && props.showAxesPicker ? (
                <Segmented label="CLI dynamic ONNX axes" size="sm" bind:value={cliAxes} options={AXES_SEG} />
              ) : null}
            </div>
            <CommandLine
              command={exportCmd.get().line}
              label="Export CLI command"
              copied="Run this command to export the model from the CLI."
            />
          </div>

          <div class="cli-block">
            <div class="cli-head">
              <span class="cli-label">Validate &amp; calibrate</span>
              <label class="cli-check">
                <input
                  type="checkbox"
                  checked={saveThresholds.get()}
                  onchange={(e) => saveThresholds.set((e.currentTarget as HTMLInputElement).checked)}
                />
                <span class="cbox" aria-hidden="true">
                  <Check size={11} strokeWidth={2.5} />
                </span>
                <span>Save thresholds</span>
              </label>
            </div>
            <CommandLine
              command={valCmd.get().line}
              label="Validate CLI command"
              copied={
                saveThresholds.get()
                  ? 'Run this command to evaluate and save calibrated per-class thresholds JSON.'
                  : 'Run this command to evaluate the model from the CLI.'
              }
            />
            <p class="cli-hint">
              {saveThresholds.get() ? (
                <>
                  Writes calibrated global and per-class F1 thresholds via <span class="mono">--save-thresholds</span> for{' '}
                  <span class="mono">folo predict --thresholds</span> or Playground import.
                </>
              ) : (
                <>Evaluate mAP, precision, and recall on your dataset.</>
              )}
            </p>
          </div>

          <div class="cli-block">
            <div class="cli-head">
              <span class="cli-label">Predict</span>
              <Segmented label="Predict snippet mode" size="sm" bind:value={predictMode} options={PREDICT_SEG} />
            </div>
            <label class="cli-check">
              <input
                type="checkbox"
                checked={useThresholds.get()}
                onchange={(e) => useThresholds.set((e.currentTarget as HTMLInputElement).checked)}
              />
              <span class="cbox" aria-hidden="true">
                <Check size={11} strokeWidth={2.5} />
              </span>
              <span>
                Use calibrated thresholds (<span class="mono">--thresholds</span>)
              </span>
            </label>
            <CommandLine
              command={predictCmd.get().line}
              label="Predict CLI command"
              copied="Run this command to run inference with this model."
            />
            <p class="cli-hint">
              <For each={predictDesc.get().split('`')} key={(_, i) => i}>
                {(part, i) => <>{i() % 2 ? <span class="mono">{part()}</span> : part()}</>}
              </For>
            </p>
          </div>

          <div class="cli-block">
            <div class="cli-head">
              <span class="cli-label">Quantize (int8)</span>
              <Segmented label="Quantize selective-f32 policy" size="sm" bind:value={quantizePreset} options={QUANTIZE_SEG} />
            </div>
            <CommandLine
              command={quantizeCmd.get().line}
              label="Quantize CLI command"
              copied="Run this command to calibrate int8 weights with selective-f32 sensitivity."
            />
            <p class="cli-hint">
              <For each={quantizeDesc.get().split('`')} key={(_, i) => i}>
                {(part, i) => <>{i() % 2 ? <span class="mono">{part()}</span> : part()}</>}
              </For>
            </p>
          </div>

          {placeholder.get() ? (
            <p class="cli-hint">
              Replace <span class="mono">&lt;path to {placeholder.get()}&gt;</span> with your local weights file path.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

css`
  .cli-wrap {
    margin-top: var(--space-4);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .cli-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: var(--control-sm);
    padding: 0;
    color: var(--text-primary);
    text-align: left;
  }
  .cli-toggle :global(svg.chev) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .cli-toggle :global(svg.chev.open) {
    transform: rotate(90deg);
  }
  .cli-title {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .cli-meta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .cli-body {
    display: grid;
    gap: var(--space-3);
    padding-top: var(--space-1);
    min-width: 0;
  }
  .cli-block {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .cli-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .cli-label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .cli-check {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    cursor: pointer;
    width: fit-content;
  }
  .cli-check input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
  }
  .cbox {
    display: grid;
    place-items: center;
    width: 14px;
    height: 14px;
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-control);
    background: var(--bg-inset);
    color: transparent;
    flex-shrink: 0;
  }
  .cli-check input:checked + .cbox {
    border-color: var(--text-primary);
    background: var(--text-primary);
    color: var(--bg-panel);
  }
  .cli-check input:focus-visible + .cbox {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 2px;
  }
  .cli-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .cli-hint .mono,
  .cli-check .mono {
    color: var(--text-secondary);
  }
`;
