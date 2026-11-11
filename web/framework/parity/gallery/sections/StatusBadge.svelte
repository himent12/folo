<!-- Gallery twin of StatusBadge.tsx. -->
<script lang="ts">
  import type { RunCloud, RunStatus } from '$lib/api/types';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';

  const statuses: RunStatus[] = ['queued', 'running', 'stopping', 'stopped', 'completed', 'failed'];
  const run = (status: RunStatus, phase?: string) => ({ status, cloud: phase === undefined ? undefined : ({ phase } as RunCloud) });
</script>

<div class="g-stack">
  <div class="g-row">
    {#each statuses as s (s)}<StatusBadge run={run(s)} />{/each}
  </div>
  <div class="g-row">
    {#each statuses as s (s)}<StatusBadge run={run(s)} size="sm" />{/each}
  </div>
  <div class="g-row">
    {#each statuses as s (s)}<span><StatusBadge run={run(s)} plain /></span>{/each}
  </div>
  <div class="g-row">
    {#each statuses as s (s)}<span><StatusBadge run={run(s)} plain dotOnly /></span>{/each}
  </div>
  <div class="g-row">
    <StatusBadge run={run('running')} lastKnown="Last known: Running · 2m ago" />
    <span><StatusBadge run={run('running')} plain lastKnown="Last known: Running · 2m ago" /></span>
    <StatusBadge run={run('queued', 'provisioning')} />
    <StatusBadge run={run('queued', 'syncing')} />
    <span><StatusBadge run={run('queued', 'booting')} plain /></span>
    <StatusBadge state={{ status: 'running', label: 'Labeling' }} />
    <span><StatusBadge state={{ status: 'stopped', label: 'Idle' }} plain /></span>
  </div>
  <div style="width: 80px">
    <StatusBadge run={run('queued', 'syncing')} plain />
  </div>
</div>
