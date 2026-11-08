<script lang="ts">
  import Construction from '@lucide/svelte/icons/construction';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';

  let { title }: { title: string } = $props();
  $effect(() => setCrumbs({ label: title }));
</script>

<div class="page">
  <h1>{title}</h1>
  <EmptyState icon={Construction} title="Coming soon" description="This screen lands in a following milestone." />
</div>

<style>
  .page {
    padding: var(--space-6);
  }
  h1 {
    font-size: var(--type-xl-size);
    line-height: var(--type-xl-line);
    letter-spacing: var(--type-xl-tracking);
  }
</style>
