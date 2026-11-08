<script lang="ts">
  /** Label + control + hint/error, laid out consistently. The control is the child snippet; pass `for` = its id. */
  import type { Snippet } from 'svelte';

  let {
    label,
    for: htmlFor,
    hint,
    error,
    children,
    aside,
  }: { label: string; for?: string; hint?: string; error?: string | null; children: Snippet; aside?: Snippet } = $props();
</script>

<div class="field" class:invalid={!!error}>
  <div class="top">
    <label for={htmlFor}>{label}</label>
    {#if aside}<span class="aside">{@render aside()}</span>{/if}
  </div>
  {@render children()}
  {#if error}
    <p class="msg error" id={htmlFor ? `${htmlFor}-error` : undefined} role="alert">{error}</p>
  {:else if hint}
    <p class="msg" id={htmlFor ? `${htmlFor}-hint` : undefined}>{hint}</p>
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
  .msg {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .msg.error {
    color: var(--danger-text);
  }
</style>
