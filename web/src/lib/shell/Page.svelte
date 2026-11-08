<script lang="ts">
  /** Standard page frame: title row (title, optional meta line and actions) above the content. */
  import type { Snippet } from 'svelte';

  let {
    title,
    sub,
    actions,
    lead,
    children,
    wide = false,
  }: {
    title: string;
    /** A meta line under the title (snippet for rich content). */
    sub?: Snippet;
    actions?: Snippet;
    /** Something before the title, e.g. a status badge row. */
    lead?: Snippet;
    children: Snippet;
    wide?: boolean;
  } = $props();
</script>

<div class="page" class:wide>
  <header>
    <div class="titles">
      {#if lead}<div class="lead">{@render lead()}</div>{/if}
      <h1>{title}</h1>
      {#if sub}<div class="sub">{@render sub()}</div>{/if}
    </div>
    {#if actions}<div class="actions">{@render actions()}</div>{/if}
  </header>
  {@render children()}
</div>

<style>
  .page {
    width: 100%;
    max-width: var(--size-content-max);
    margin: 0 auto;
    padding: var(--space-6) var(--space-6) var(--space-12);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
    animation: in var(--dur-slow) var(--ease-out);
  }
  .page.wide {
    max-width: none;
  }
  header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--space-4);
    flex-wrap: wrap;
  }
  .titles {
    display: grid;
    gap: var(--space-1);
    min-width: 0;
  }
  .lead {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: 2px;
  }
  h1 {
    font-size: var(--type-xl-size);
    line-height: var(--type-xl-line);
    letter-spacing: var(--type-xl-tracking);
    font-weight: var(--type-xl-weight);
    overflow-wrap: anywhere;
  }
  .sub {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    color: var(--text-secondary);
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  @keyframes in {
    from {
      opacity: 0;
    }
  }
  @media (max-width: 767px) {
    .page {
      padding: var(--space-4) var(--space-4) var(--space-8);
      gap: var(--space-4);
    }
    header {
      align-items: flex-start;
    }
    /* The top bar shows the title on phones; the heading stays for assistive tech and the outline. */
    h1 {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
  }
</style>
