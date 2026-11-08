<script lang="ts">
  import Copy from '@lucide/svelte/icons/copy';
  import type { TrainConfig } from '$lib/api/types';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let {
    config,
    datasetName,
    compact = false,
  }: { config: TrainConfig; datasetName?: string; /** One column for a rail, augmentation folded away. */ compact?: boolean } = $props();

  const a = $derived(config.augment);
  // Identifiers (model names, weight paths, devices) are mono; numbers and prose are sans with tabular numerals.
  const MONO = new Set(['Model', 'Weights', 'Device', 'Optimizer']);
  const rows = $derived<[string, string][]>([
    ['Dataset', datasetName ?? config.dataset_id],
    ['Model', config.arch],
    ['Epochs', String(config.epochs)],
    ['Batch', String(config.batch)],
    ['Image size', `${config.imgsz} px`],
    ['Device', config.device],
    ['Optimizer', config.optimizer],
    ['Initial LR', String(config.lr0)],
    ['Weights', config.weights ?? 'from scratch'],
    ['Seed', String(config.seed)],
  ]);
  const aug = $derived<[string, string][]>([
    ['Mosaic', String(a.mosaic)],
    ['Mixup', String(a.mixup)],
    ['HSV h · s · v', a.hsv.join(' · ')],
    ['Flip LR', String(a.fliplr)],
    ['Scale', `±${a.scale}`],
    ['Translate', `±${a.translate}`],
    ['Close mosaic', `last ${a.close_mosaic} epochs`],
  ]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
      toast.success('Config copied', 'Paste it into folo train --config or a new run.');
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }
</script>

{#snippet training()}
  <dl>
    <p class="group">Training</p>
    {#each rows as [k, v] (k)}<dt>{k}</dt><dd class="num" class:mono={MONO.has(k) && !(k === 'Weights' && !config.weights)}>{v}</dd>{/each}
  </dl>
{/snippet}
{#snippet augmentation(heading: boolean)}
  <dl>
    {#if heading}<p class="group">Augmentation</p>{/if}
    {#each aug as [k, v] (k)}<dt>{k}</dt><dd class="num">{v}</dd>{/each}
  </dl>
{/snippet}

<Panel title="Config" id="config">
  {#snippet actions()}<Button size="sm" variant="ghost" icon={Copy} onclick={copy}>Copy JSON</Button>{/snippet}
  {#if compact}
    {@render training()}
    <details>
      <summary>Augmentation</summary>
      {@render augmentation(false)}
    </details>
  {:else}
    <div class="cols">
      {@render training()}
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
