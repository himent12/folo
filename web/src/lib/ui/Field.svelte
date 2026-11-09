<script lang="ts">
  /**
   * Label + control + hint/error, laid out consistently. The control is the child snippet; pass `for` = its id, and
   * mark the control invalid (`invalid` on Input / Select / Segmented) while `error` is set. The error carries a danger
   * icon as well as its colour, so it is never red text alone.
   */
  import type { Snippet } from 'svelte';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';

  let {
    label,
    for: htmlFor,
    hint,
    error,
    children,
    aside,
  }: {
    label: string;
    for?: string;
    /** Helper text; identifiers in backticks are set as code (`folo serve`). */
    hint?: string;
    error?: string | null;
    children: Snippet;
    aside?: Snippet;
  } = $props();
</script>

<div class="field" class:invalid={!!error}>
  <div class="top">
    <label for={htmlFor}>{label}</label>
    {#if aside}<span class="aside">{@render aside()}</span>{/if}
  </div>
  {@render children()}
  {#if error}
    <p class="msg error" id={htmlFor ? `${htmlFor}-error` : undefined} role="alert">
      <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" /><span>{error}</span>
    </p>
  {:else if hint}
    <p class="msg" id={htmlFor ? `${htmlFor}-hint` : undefined}>{#each hint.split('`') as text, i (i)}{#if i % 2}<code>{text}</code>{:else}{text}{/if}{/each}</p>
  {/if}
</div>

<style>
  .field {
    display: grid;
    gap: 6px;
    min-width: 0;
  }
  .top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  label {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .aside {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Helper text keeps a readable measure (DESIGN_DIRECTION §2.3: ≤ 60 characters a line). */
  .msg {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .msg code {
    font-family: var(--font-mono);
  }
  .msg.error {
    display: flex;
    align-items: flex-start;
    gap: var(--space-1);
    color: var(--danger-text);
  }
  /* The icon sits on the first line's centre (16 px line, 14 px icon). */
  .msg.error :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }
</style>
