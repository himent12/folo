<script lang="ts">
  /** The labeling screens on a server that cannot label (support.ts): what it needs, in the page's own frame. */
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Tags from '@lucide/svelte/icons/tags';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
</script>

<Panel pad="none">
  <EmptyState
    icon={Tags}
    title="This server cannot label images"
    description="Labeling asks Claude Code or Codex on the server's machine, so it needs a current `folo serve` started without `--mock`."
  >
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/datasets')}>All datasets</Button>{/snippet}
  </EmptyState>
</Panel>
