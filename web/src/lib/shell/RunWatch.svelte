<script lang="ts">
  /**
   * Watches the run list for the whole app (every 2 s) and says when a training run ends: a toast with "Open run" when
   * it finishes, fails or runs out of GPU memory (./runWatch). Not on that run's own page, whose banner already says
   * so. Renders nothing; the sidebar's live card reads the same list (./runList).
   */
  import { untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Run } from '$lib/api/types';
  import { router } from '$lib/router.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { watched } from './runList.svelte';
  import { runEndMessage, runEnds } from './runWatch';

  const runs = new Resource((s) => hub.api.runs(s), { poll: 2000 });
  let prev: Run[] | undefined;
  // A different server (reconnect, mock): its list is not a change from the last one.
  $effect(() => {
    void hub.epoch;
    prev = undefined;
  });
  $effect(() => {
    const next = runs.data;
    if (!next) return;
    untrack(() => {
      for (const e of runEnds(prev, next)) {
        const route = router.route;
        if (route.name === 'run' && route.params.id === e.run.id) continue;
        const id = e.run.id;
        toast.show({ ...runEndMessage(e), action: { label: 'Open run', run: () => router.go(`/runs/${id}`) } });
      }
      prev = next;
      watched.runs = next;
    });
  });
  $effect(() => {
    watched.since = runs.stale !== undefined ? runs.updated : null;
  });
</script>
