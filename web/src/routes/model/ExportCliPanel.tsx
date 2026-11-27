/**
 * Collapsible CLI commands section for Model Detail ExportDialog: shows copy-ready `folo export`
 * (synced with the selected format, decoded/raw output, half precision, and dynamic ONNX axes) and
 * `folo predict` quick snippets (Standard, SAHI `--tile`, `--tta`, ByteTrack `--track`).
 */
import ChevronRight from '$icons/chevron-right';
import { computed, css, signal } from 'fw';
import type { ExportFormat, Model } from '$lib/api/types';
import CommandLine from '$lib/ui/CommandLine';
import Segmented from '$lib/ui/Segmented';
import {
  buildExportCli,
  buildPredictCli,
  ONNX_AXES_OPTIONS,
  PREDICT_SNIPPET_OPTIONS,
  type OnnxDynamicAxes,
  type PredictSnippetMode,
} from './modelCli';

export interface ExportCliPanelProps {
  model: Model;
  format: ExportFormat;
  decode: boolean;
  half: boolean;
  axes: OnnxDynamicAxes;
  /** When false (older server without `export_dynamic_axes`), the CLI panel still lets users pick any dynamic axis for copy-ready CLI export. */
  showAxesPicker: boolean;
  onAxes: (next: OnnxDynamicAxes) => void;
}

const AXES_SEG = ONNX_AXES_OPTIONS.map((o) => ({ value: o.value, label: o.label }));
const PREDICT_SEG = PREDICT_SNIPPET_OPTIONS.map((o) => ({ value: o.value, label: o.label }));

export default function ExportCliPanel(props: ExportCliPanelProps) {
  const open = signal(false);
  const predictMode = signal<PredictSnippetMode>('sahi');

  const axesValue = {
    get: () => props.axes,
    set: (v: OnnxDynamicAxes) => props.onAxes(v),
  };

  const exportCmd = computed(() =>
    buildExportCli(props.model, {
      format: props.format,
      decode: props.decode,
      half: props.half,
      axes: props.axes,
    }),
  );
  const predictCmd = computed(() => buildPredictCli(props.model, predictMode.get()));
  const predictDesc = computed(
    () => PREDICT_SNIPPET_OPTIONS.find((o) => o.value === predictMode.get())?.desc ?? '',
  );
  const placeholder = computed(() => exportCmd.get().placeholder ?? predictCmd.get().placeholder);

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
        <span class="cli-meta">export &amp; predict</span>
      </button>
      {open.get() ? (
        <div class="cli-body">
          <div class="cli-block">
            <div class="cli-head">
              <span class="cli-label">Export</span>
              {props.format === 'onnx' && props.showAxesPicker ? (
                <Segmented label="CLI dynamic ONNX axes" size="sm" bind:value={axesValue} options={AXES_SEG} />
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
              <span class="cli-label">Predict</span>
              <Segmented label="Predict snippet mode" size="sm" bind:value={predictMode} options={PREDICT_SEG} />
            </div>
            <CommandLine
              command={predictCmd.get().line}
              label="Predict CLI command"
              copied="Run this command to run inference with this model."
            />
            <p class="cli-hint">{predictDesc.get()}</p>
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
  .cli-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .cli-hint .mono {
    color: var(--text-secondary);
  }
`;
