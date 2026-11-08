<script lang="ts">
  import Crosshair from '@lucide/svelte/icons/crosshair';
  import { router, href } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';

  $effect(() => setCrumbs({ label: 'Not found' }));
</script>

<div class="nf">
<EmptyState
  icon={Crosshair}
  title="Nothing in frame"
  description={`There is no page at ${router.route.path}. It may have moved, or the link is wrong.`}
>
  {#snippet actions()}
    <Button href={href('/')}>Go to Overview</Button>
  {/snippet}
</EmptyState>
</div>

<style>
  .nf {
    display: grid;
    place-items: center;
    min-height: calc(100dvh - var(--size-topbar) - var(--space-12));
    padding: var(--space-6);
  }
  .nf > :global(.empty) {
    width: min(480px, 100%);
    border: 1px dashed var(--line-strong);
    border-radius: var(--radius-lg);
  }
</style>
