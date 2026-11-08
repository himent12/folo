<script lang="ts">
  import Copy from '@lucide/svelte/icons/copy';
  import type { TrainConfig } from '$lib/api/types';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let { config, datasetName }: { config: TrainConfig; datasetName?: string } = $props();

  const a = $derived(config.augment);
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
      toast.success('Config copied', 'Paste it into `folo train --config` or a new run.');
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }
</script>

<Panel title="Config" id="config">
  {#snippet actions()}<Button size="sm" variant="ghost" icon={Copy} onclick={copy}>Copy JSON</Button>{/snippet}
  <div class="cols">
    <dl>
      <p class="overline">Training</p>
      {#each rows as [k, v] (k)}<dt>{k}</dt><dd class="mono">{v}</dd>{/each}
    </dl>
    <dl>
      <p class="overline">Augmentation</p>
      {#each aug as [k, v] (k)}<dt>{k}</dt><dd class="mono">{v}</dd>{/each}
    </dl>
  </div>
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
  dl .overline {
    grid-column: 1 / -1;
    margin-bottom: var(--space-1);
  }
  dt,
  dd {
    padding: 5px 0;
    border-bottom: 1px solid var(--line-subtle);
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
