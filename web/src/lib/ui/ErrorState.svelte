<script lang="ts">
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import { ApiError } from '$lib/api/client';
  import Button from './Button.svelte';

  let {
    error,
    retry,
    title = 'Could not load this',
    compact = false,
  }: { error: unknown; retry?: () => void; title?: string; compact?: boolean } = $props();

  const message = $derived(error instanceof Error ? error.message : String(error));
  const hint = $derived(error instanceof ApiError ? error.hint : undefined);
  const code = $derived(error instanceof ApiError ? error.code : undefined);
</script>

<div class="err" class:compact role="alert">
  <div class="icon"><TriangleAlert size={16} strokeWidth={1.75} /></div>
  <div class="text">
    <p class="title">{title}</p>
    <p class="msg">{message}</p>
    {#if hint}<p class="hint">{hint}</p>{/if}
    {#if code}<p class="code">{code}</p>{/if}
  </div>
  {#if retry}<Button size="sm" icon={RotateCw} onclick={retry}>Retry</Button>{/if}
</div>

<style>
  .err {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-4);
    margin: var(--space-4);
    border-radius: var(--radius-md);
    background: var(--danger-muted);
    border: 1px solid color-mix(in srgb, var(--danger-solid) 22%, transparent);
  }
  .compact {
    margin: 0;
    padding: var(--space-3);
  }
  .icon {
    color: var(--danger-text);
    padding-top: 2px;
  }
  .text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .title {
    font-weight: var(--weight-medium);
  }
  .msg {
    color: var(--text-secondary);
  }
  .hint {
    color: var(--text-tertiary);
  }
  .code {
    font-family: var(--font-mono);
    color: var(--text-tertiary);
    font-size: var(--type-xs-size);
  }
</style>
