<!-- Gallery twin of EmptyState.tsx. -->
<script lang="ts">
  import Database from '@lucide/svelte/icons/database';
  import SearchX from '@lucide/svelte/icons/search-x';
  import Plus from '@lucide/svelte/icons/plus';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
</script>

<div class="g-stack g-medium" data-parity-phone>
  <EmptyState icon={Database} title="No datasets yet" description="Add a folder of images, or run `folo serve --demo` to try the sample data.">
    {#snippet actions()}
      <Button variant="primary" icon={Plus}>Add dataset</Button>
      <Button>Learn more</Button>
    {/snippet}
  </EmptyState>
  <EmptyState icon={SearchX} title="Not found" description="This run does not exist, or it was deleted." />
  <EmptyState title="Nothing to compare" />
  <EmptyState title="No runs match" description="Clear the filter to see every run.">
    {#snippet actions()}
      <Button size="sm">Clear filter</Button>
    {/snippet}
  </EmptyState>
</div>
