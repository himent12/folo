<script lang="ts">
  import Crosshair from '@lucide/svelte/icons/crosshair';
  import { router, href } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';

  $effect(() => setCrumbs({ label: 'Not found' }));
</script>

<EmptyState
  icon={Crosshair}
  title="Nothing in frame"
  description={`There is no page at ${router.route.path}. It may have moved, or the link is wrong.`}
>
  {#snippet actions()}
    <Button href={href('/')}>Go to Overview</Button>
  {/snippet}
</EmptyState>
