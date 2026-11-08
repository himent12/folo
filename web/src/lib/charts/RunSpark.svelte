<script lang="ts">
  /** mAP50-95 per epoch for one run, as a sparkline. */
  import { hub } from '$lib/api/index.svelte';
  import { mapHistory } from '$lib/api/metricsCache';
  import type { Run } from '$lib/api/types';
  import Sparkline from './Sparkline.svelte';

  let { run, width = 88, height = 22 }: { run: Run; width?: number; height?: number } = $props();

  let values = $state<number[]>([]);

  $effect(() => {
    let live = true;
    mapHistory(hub.api, run).then(
      (v) => {
        if (live) values = v;
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });

  // Neutral ink; only a live run's line takes the accent. Failure is stated by the status dot, not repeated here.
  const color = $derived(run.status === 'running' ? 'var(--accent-line)' : 'var(--text-tertiary)');
</script>

<Sparkline {values} {width} {height} {color} domain={[0, Math.max(0.1, ...values)]} length={run.epochs} fill={run.status === 'running'} />
