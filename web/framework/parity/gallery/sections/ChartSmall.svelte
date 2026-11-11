<!-- Gallery twin of ChartSmall.tsx: ChartSkeleton and RunSpark (charts/), on fixed metrics. -->
<script lang="ts">
  import { hub } from '$lib/api/index.svelte';
  import { galleryApi } from '../chartData';
  import type { Run } from '$lib/api/types';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import RunSpark from '$lib/charts/RunSpark.svelte';

  // Fixed metrics, at once: what the sparklines draw does not depend on the mock's clock.
  hub.api = galleryApi(hub.api);
  const run = (id: string, status: string, epochs: number) => ({ id, status, epochs, epoch: epochs }) as unknown as Run;
</script>

<div class="g-stack">
  <div class="g-row">
    <RunSpark run={run('r', 'running', 30)} />
    <RunSpark run={run('c', 'completed', 20)} width={120} height={28} />
    <RunSpark run={run('f', 'failed', 20)} />
  </div>
  <div class="g-medium"><ChartSkeleton /></div>
  <div class="g-narrow"><ChartSkeleton height={120} legendRows={2} /></div>
  <div class="g-narrow"><ChartSkeleton height={80} legend={false} /></div>
</div>
