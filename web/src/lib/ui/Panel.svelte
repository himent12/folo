<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title?: string;
    /** Small text after the title (a count, a unit, a hint). */
    meta?: string;
    actions?: Snippet;
    children?: Snippet;
    /** Inner padding of the body. */
    pad?: 'none' | 'sm' | 'md';
    /** Hairline between header and body. */
    divided?: boolean;
    class?: string;
    id?: string;
    /** Element for the title, so pages keep a sane heading outline. */
    as?: 'h2' | 'h3';
  }

  let { title, meta, actions, children, pad = 'md', divided = false, class: klass = '', id, as = 'h2' }: Props = $props();
</script>

<section class="panel pad-{pad} {klass}" aria-labelledby={title && id ? `${id}-title` : undefined} {id}>
  {#if title || actions}
    <header class:divided>
      {#if title}
        <svelte:element this={as} class="title" id={id ? `${id}-title` : undefined}>
          {title}{#if meta}<span class="meta">{meta}</span>{/if}
        </svelte:element>
      {/if}
      {#if actions}<div class="actions">{@render actions()}</div>{/if}
    </header>
  {/if}
  {#if children}<div class="body">{@render children()}</div>{/if}
</section>

<style>
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  :global([data-theme='dark']) .panel {
    border-color: var(--line-subtle);
    box-shadow: inset 0 1px 0 #ffffff06;
  }
  header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 44px;
    padding: var(--space-2) var(--space-3) var(--space-2) var(--space-4);
  }
  header.divided {
    border-bottom: 1px solid var(--line-subtle);
  }
  .title {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .meta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    flex-shrink: 0;
  }
  .body {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .pad-md .body {
    padding: 0 var(--space-4) var(--space-4);
  }
  .pad-md header.divided + .body,
  .pad-md > .body:first-child {
    padding-top: var(--space-4);
  }
  .pad-sm .body {
    padding: 0 var(--space-3) var(--space-3);
  }
  .pad-sm header.divided + .body,
  .pad-sm > .body:first-child {
    padding-top: var(--space-3);
  }
</style>
