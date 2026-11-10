<script lang="ts">
  /** The benchmark screens on a server without them (./support): what they need, in the page's own frame. */
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Target from '@lucide/svelte/icons/target';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
</script>

<Panel pad="none">
  <EmptyState
    icon={Target}
    title="This server has no vision benchmarks"
    description="Benchmarks need a current `folo serve` started without `--mock`: it keeps their images, runs the entries and scores them."
  >
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/models')}>All models</Button>{/snippet}
  </EmptyState>
</Panel>
