<script lang="ts">
  /**
   * The checkpoint a run started from, named one way on every screen (header, Config, Model's Details, Compare): the
   * model's name (an identifier, so mono through ui/Ident) linked to it, with its server id in the tooltip; a file by
   * its file name, the full path in the tooltip; "From scratch" in sans. Resolve the origin with `$lib/lineage`.
   */
  import { href } from '$lib/router.svelte';
  import { FROM_SCRATCH, originTip, type Origin } from '$lib/lineage';
  import Ident from './Ident.svelte';

  let { origin, link = true, scratch = FROM_SCRATCH }: { origin: Origin; link?: boolean; scratch?: string } = $props();
</script>

{#if origin.kind === 'scratch'}{scratch}{:else if origin.kind === 'model'}<Ident
    value={origin.label}
    title={originTip(origin)}
    class={link ? 'lineage-link' : ''}
    href={link ? href(`/models/${encodeURIComponent(origin.id)}`) : undefined}
  />{:else}<Ident value={origin.label} title={originTip(origin)} />{/if}

<style>
  :global(a.ident.lineage-link) {
    color: inherit;
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  :global(a.ident.lineage-link:hover) {
    text-decoration-color: var(--text-secondary);
  }
</style>
