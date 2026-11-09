<script lang="ts">
  /** Renders a shortcut like "Mod+K" as key caps; Mod is ⌘ on Apple platforms and Ctrl elsewhere. */
  import { formatKeys } from '$lib/keys';

  let { keys, subtle = false }: { keys: string; subtle?: boolean } = $props();
  const parts = $derived(formatKeys(keys));
</script>

<span class="kbd" class:subtle aria-label={parts.join(' ')}>
  {#each parts as p, i (i)}<kbd>{p}</kbd>{/each}
</span>

<style>
  .kbd {
    display: inline-flex;
    gap: 2px;
  }
  /* 12 px caps (type.sm): the key names are labels, so they meet the 12 px floor (DESIGN_DIRECTION §2.14). */
  kbd {
    display: inline-grid;
    place-items: center;
    min-width: 20px;
    height: 20px;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-default);
    border-bottom-color: var(--line-strong);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-family: var(--font-sans);
    font-size: var(--type-sm-size);
    line-height: 1;
    font-weight: var(--weight-medium);
  }
  .subtle kbd {
    background: transparent;
    border-color: var(--line-subtle);
    color: var(--text-tertiary);
  }
</style>
