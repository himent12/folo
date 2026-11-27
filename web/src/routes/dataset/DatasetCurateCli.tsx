/**
 * Concise, shell-safe CLI Curation Helper paired with DatasetInsights: lets users copy real `folo dataset`
 * commands (`check --perceptual --fix-dry-run`, `split`, `filter`, `merge`) tailored to the loaded dataset.
 */
import { computed, css, For, signal } from 'fw';
import type { Dataset, DatasetStats } from '$lib/api/types';
import { SHELL } from '$lib/trainCli';
import CommandLine from '$lib/ui/CommandLine';
import Segmented from '$lib/ui/Segmented';
import { buildDatasetCurateCli, CURATE_CLI_MODES, type CurateCliMode } from './curateCli';

export interface DatasetCurateCliProps {
  dataset: Dataset;
  stats: DatasetStats;
  activeClass?: number;
}

export default function DatasetCurateCli(props: DatasetCurateCliProps) {
  const mode = signal<CurateCliMode>(props.dataset.splits.val === 0 ? 'split' : 'check');
  const snippet = computed(() =>
    buildDatasetCurateCli(props.dataset, mode.get(), {
      stats: props.stats,
      activeClass: props.activeClass,
    }),
  );

  return (
    <div class="curate" data-dataset-curate-cli>
      <div class="curate-head">
        <span class="curate-label">CLI curation · {SHELL}</span>
        <Segmented label="Dataset curation command" size="sm" bind:value={mode} options={[...CURATE_CLI_MODES]} />
      </div>
      <CommandLine
        command={snippet.get().command}
        label={`Dataset ${snippet.get().label} CLI command`}
        copied={`Run this ${snippet.get().label.toLowerCase()} command from your terminal.`}
      />
      <p class="curate-desc">
        <For each={snippet.get().description.split('`')} key={(_, i) => i}>
          {(part, i) => <>{i() % 2 ? <span class="mono">{part()}</span> : part()}</>}
        </For>
      </p>
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
  .curate-label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .curate-desc {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .curate-desc .mono {
    color: var(--text-secondary);
  }
`;
