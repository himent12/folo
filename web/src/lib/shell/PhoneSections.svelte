<script lang="ts">
  /**
   * Phones: the sections a tab of the tab bar holds, as underline tabs at the top of each one's list (`nav.ts`
   * `tabUnder`). The bar holds five tabs with their names whole at 320 px; a sixth section lives under the tab it
   * belongs with (Benchmarks grade models, so they sit under Models, as labeling sits under Datasets), and this strip is
   * how a phone gets there in one tap, never only through search. Links, not a tab panel: each section is its own page.
   * From 768 px the sidebar lists every section, and the strip is not drawn.
   */
  import { href } from '$lib/router.svelte';

  let { items, label }: { items: { label: string; path: string; current: boolean }[]; label: string } = $props();
</script>

<nav class="sections" aria-label={label}>
  {#each items as it (it.path)}
    <a href={href(it.path)} class:on={it.current} aria-current={it.current ? 'page' : undefined}>{it.label}</a>
  {/each}
</nav>

<style>
  .sections {
    display: none;
  }
  /* ui/Tabs' look: 36 px tabs on a hairline track that spans the column, the current one an ink underline. */
  @media (max-width: 767px) {
    .sections {
      display: flex;
      gap: var(--space-4);
      margin-top: calc(-1 * var(--space-2));
      border-bottom: 1px solid var(--line-subtle);
    }
    a {
      position: relative;
      display: inline-flex;
      align-items: center;
      height: 44px;
      color: var(--text-secondary);
      font-weight: var(--weight-medium);
    }
    a.on {
      color: var(--text-primary);
    }
    a.on::after {
      content: '';
      position: absolute;
      left: 0;
      right: 0;
      bottom: -1px;
      height: 2px;
      border-radius: 1px;
      background: var(--text-primary);
    }
  }
</style>
