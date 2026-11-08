<script lang="ts">
  /**
   * An empty state in place of the content it stands for: the small viewfinder mark inline, a one-line title, one
   * useful sentence and one action, starting at the content's left edge. Every empty state uses it, Not found too.
   */
  import type { Snippet } from 'svelte';
  import type { IconType } from './types';

  let {
    icon: Icon,
    title,
    description,
    actions,
  }: { icon?: IconType; title: string; description?: string; actions?: Snippet } = $props();
</script>

<div class="empty" class:bare={!Icon}>
  {#if Icon}
    <div class="mark" aria-hidden="true">
      <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span
        class="corner br"
      ></span>
      <Icon size={14} strokeWidth={1.75} />
    </div>
  {/if}
  <p class="title">{title}</p>
  {#if description}<p class="desc">{description}</p>{/if}
  {#if actions}<div class="actions">{@render actions()}</div>{/if}
</div>

<style>
  /* Mark in a 24 px column, the text and the action aligned to the title. */
  .empty {
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr);
    column-gap: var(--space-3);
    align-items: center;
    padding: var(--space-5) var(--space-4);
  }
  .empty.bare {
    grid-template-columns: minmax(0, 1fr);
  }
  .empty > :not(.mark) {
    grid-column: -2;
  }
  /* Inside a padded panel the panel's own padding is the edge. */
  :global(.pad-md > .body) > .empty,
  :global(.pad-sm > .body) > .empty {
    padding: var(--space-1) 0;
  }
  .mark {
    position: relative;
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    color: var(--text-tertiary);
  }
  .corner {
    position: absolute;
    width: 6px;
    height: 6px;
    border-color: var(--line-strong);
    border-style: solid;
    border-width: 0;
  }
  .tl {
    top: 0;
    left: 0;
    border-top-width: 1.5px;
    border-left-width: 1.5px;
    border-top-left-radius: 2px;
  }
  .tr {
    top: 0;
    right: 0;
    border-top-width: 1.5px;
    border-right-width: 1.5px;
    border-top-right-radius: 2px;
  }
  .bl {
    bottom: 0;
    left: 0;
    border-bottom-width: 1.5px;
    border-left-width: 1.5px;
    border-bottom-left-radius: 2px;
  }
  .br {
    bottom: 0;
    right: 0;
    border-bottom-width: 1.5px;
    border-right-width: 1.5px;
    border-bottom-right-radius: 2px;
  }
  .title {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .desc {
    max-width: 60ch;
    color: var(--text-secondary);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: var(--space-3);
  }
</style>
