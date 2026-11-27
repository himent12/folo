/**
 * Concise, shell-safe CLI Curation & Export Helper paired with DatasetInsights and StatsTab: lets users copy real
 * `folo dataset` commands (`check --perceptual --fix-dry-run`, `split`, `filter`, `merge`, `convert`) tailored to the
 * loaded dataset.
 */
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, effect, signal } from 'fw';
import type { Dataset, DatasetStats, Split } from '$lib/api/types';
import { SHELL } from '$lib/trainCli';
import CommandLine from '$lib/ui/CommandLine';
import Segmented from '$lib/ui/Segmented';
import {
  buildDatasetCurateCli,
  CONVERT_TARGET_FORMATS,
  CURATE_CLI_MODES,
  defaultConvertTarget,
  type ConvertTargetFormat,
  type CurateCliMode,
} from './curateCli';

export interface DatasetCurateCliProps {
  dataset: Dataset;
  stats: DatasetStats;
  activeClass?: number;
  activeSplit?: 'all' | Split;
}

export default function DatasetCurateCli(props: DatasetCurateCliProps) {
  const mode = signal<CurateCliMode>(props.dataset.splits.val === 0 ? 'split' : 'check');
  const targetFormat = signal<ConvertTargetFormat>(defaultConvertTarget(props.dataset.format));
  let prevDatasetId = props.dataset.id;
  effect(() => {
    const id = props.dataset.id;
    if (id !== prevDatasetId) {
      prevDatasetId = id;
      targetFormat.set(defaultConvertTarget(props.dataset.format));
    }
  });
  const snippet = computed(() =>
    buildDatasetCurateCli(props.dataset, mode.get(), {
      stats: props.stats,
      activeClass: props.activeClass,
      activeSplit: props.activeSplit,
      targetFormat: targetFormat.get(),
    }),
  );

  return (
    <div class="curate" data-dataset-curate-cli>
      <div class="curate-head">
        <span class="curate-label">CLI curation · {SHELL}</span>
        <div class="curate-controls">
          <Segmented label="Dataset curation command" size="sm" bind:value={mode} options={[...CURATE_CLI_MODES]} />
          {mode.get() === 'convert' ? (
            <Segmented
              label="Target format"
              size="sm"
              bind:value={targetFormat}
              options={[...CONVERT_TARGET_FORMATS]}
            />
          ) : null}
        </div>
      </div>
      <CommandLine
        command={snippet.get().command}
        label={`Dataset ${snippet.get().label} CLI command`}
        copied={`Run this ${snippet.get().label.toLowerCase()} command from your terminal.`}
      />
      {snippet.get().warning ? (
        <p class="curate-warn" data-curate-warning>
          <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
          <span>{snippet.get().warning}</span>
        </p>
      ) : null}
      <p class="curate-desc">{snippet.get().description}</p>
    </div>
  );
}

css`
  .curate {
    flex-basis: 100%;
    display: grid;
    gap: var(--space-2);
    margin-top: var(--space-1);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
    min-width: 0;
  }
  .curate-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .curate-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .curate-label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .curate-warn {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--warning-text);
  }
  .curate-warn :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
  }
  .curate-desc {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
`;
