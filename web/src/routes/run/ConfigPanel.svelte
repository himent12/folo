<script lang="ts">
  /**
   * A run's config. With `resolved` (docs/API.md, `Run.resolved`) the settings it trains with are the values shown,
   * the request a note where it differs ("AdamW (auto)", "0.001429 (requested 0.01)"), plus the batch it accumulates,
   * the LR decay, the warmup and the optimizer's momentum and weight decay; without it, the request as asked for.
   * Close mosaic is the trainer's capped count ("last 7 epochs (requested 10)"). Copy JSON copies the request: the
   * TrainConfig `POST /api/runs` and MCP `train` take, which resolves the same way again.
   */
  import Copy from '@lucide/svelte/icons/copy';
  import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
  import type { Origin } from '$lib/lineage';
  import Button from '$lib/ui/Button.svelte';
  import Lineage from '$lib/ui/Lineage.svelte';
  import Note from '$lib/ui/Note.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { batchSetting, closeMosaicSetting, headSetting, lrSetting, optimizerSetting, precisionSetting, resolvedRows, type Setting } from './resolved';

  let {
    config,
    resolved = null,
    datasetName,
    compact = false,
    weights,
  }: {
    config: TrainConfig;
    /** What the trainer resolved `config` to; null until it is built, missing from older servers. */
    resolved?: ResolvedTrain | null;
    datasetName?: string;
    /** One column for a rail, augmentation folded away. */
    compact?: boolean;
    /** Where the run started, read once ($lib/lineage): the checkpoint by name, its id in the tooltip, as the header says it. */
    weights?: Origin | null;
  } = $props();

  const a = $derived(config.augment);
  // Identifiers (model names, weight paths, devices, a named optimizer) are mono; numbers and prose ("auto", "From
  // scratch") are sans with tabular numerals (design/README.md, Type).
  const id = (value: string): Setting => ({ value, ident: true });
  const plain = (value: string): Setting => ({ value });
  const rows = $derived<[string, Setting][]>([
    ['Dataset', plain(datasetName ?? config.dataset_id)],
    ['Model', id(config.arch)],
    ['Epochs', plain(String(config.epochs))],
    [resolved && resolved.accumulate > 1 ? 'Batch × accumulate' : 'Batch', batchSetting(config, resolved)],
    ['Image size', plain(`${config.imgsz} px`)],
    ['Device', config.device === 'auto' ? plain('auto') : id(config.device)],
    ['Optimizer', optimizerSetting(config, resolved)],
    ['Initial LR', lrSetting(config, resolved)],
    ...(resolved ? resolvedRows(resolved) : []),
    ['Precision', precisionSetting(config)],
    ['Head', headSetting(config)],
    ['Weights', config.weights ? id(config.weights) : plain('From scratch')],
    ['Seed', plain(String(config.seed))],
  ]);
  const aug = $derived<[string, Setting][]>([
    ['Mosaic', plain(String(a.mosaic))],
    ['Mixup', plain(String(a.mixup))],
    ['HSV h · s · v', plain(a.hsv.join(' · '))],
    ['Flip LR', plain(String(a.fliplr))],
    ['Scale', plain(`±${a.scale}`)],
    ['Translate', plain(`±${a.translate}`)],
    ['Close mosaic', closeMosaicSetting({ config, resolved })],
  ]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
      toast.success('Config copied', 'The request as POST /api/runs and MCP train take it.');
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }
</script>

{#snippet training(heading: boolean)}
  <dl>
    <!-- A heading only beside Augmentation's; alone in the rail it would name the panel twice. -->
    {#if heading}<p class="group">Training</p>{/if}
    {#each rows as [k, v] (k)}<dt>{k}</dt>{#if k === 'Weights' && weights}<dd>{#if weights.kind === 'scratch'}From scratch{:else}<Lineage origin={weights} />{/if}</dd>{:else}<dd class="num"><span class:mono={v.ident}>{v.value}</span>{#if v.note}<Note text={v.note} />{/if}</dd>{/if}{/each}
  </dl>
{/snippet}
{#snippet augmentation(heading: boolean)}
  <dl>
    {#if heading}<p class="group">Augmentation</p>{/if}
    {#each aug as [k, v] (k)}<dt>{k}</dt><dd class="num">{v.value}{#if v.note}<Note text={v.note} />{/if}</dd>{/each}
  </dl>
{/snippet}

<Panel title="Config" id="config">
  {#snippet actions()}<Button size="sm" variant="ghost" icon={Copy} onclick={copy}>Copy JSON</Button>{/snippet}
  {#if compact}
    {@render training(false)}
    <details>
      <summary>Augmentation</summary>
      {@render augmentation(false)}
    </details>
  {:else}
    <div class="cols">
      {@render training(true)}
      {@render augmentation(true)}
    </div>
  {/if}
</Panel>

<style>
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-6);
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0;
    align-content: start;
  }
  .group {
    grid-column: 1 / -1;
    margin-bottom: var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  dt,
  dd {
    padding: 4px 0;
    border-bottom: 1px solid var(--line-subtle);
  }
  details {
    margin-top: var(--space-3);
  }
  summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    cursor: pointer;
    list-style: none;
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary::before {
    content: '';
    width: 6px;
    height: 6px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  details[open] summary::before {
    transform: rotate(45deg);
  }
  summary:hover {
    color: var(--text-primary);
  }
  dt {
    color: var(--text-tertiary);
  }
  dd {
    padding-left: var(--space-4);
    text-align: right;
    overflow-wrap: anywhere;
  }
  @media (max-width: 767px) {
    .cols {
      grid-template-columns: 1fr;
    }
  }
</style>
