<script lang="ts">
  import Crosshair from '@lucide/svelte/icons/crosshair';
  import { router, href } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Panel from '$lib/ui/Panel.svelte';

  $effect(() => setCrumbs({ label: 'Not found' }));
</script>

<!-- The same page frame and inline empty state as every other screen: title, then the message where content would be. -->
<Page title="Not found">
  <Panel pad="none">
    <EmptyState icon={Crosshair} title="Nothing in frame" description={`There is no page at ${router.route.path}. It may have moved, or the link is wrong.`}>
      {#snippet actions()}
        <Button href={href('/')}>Go to Overview</Button>
      {/snippet}
    </EmptyState>
  </Panel>
</Page>
