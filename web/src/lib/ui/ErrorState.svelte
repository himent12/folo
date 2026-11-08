<script lang="ts">
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Settings from '@lucide/svelte/icons/settings';
  import { ApiError } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import { href } from '$lib/router.svelte';
  import Button from './Button.svelte';

  let {
    error,
    retry,
    title = 'Could not load this',
    compact = false,
  }: { error: unknown; retry?: () => void; title?: string; compact?: boolean } = $props();

  // Connection failures look the same everywhere: what was tried, and both ways out (retry, settings).
  const offline = $derived(error instanceof ApiError && error.code === 'network');
  const heading = $derived(offline ? "Can't reach the FOLO server" : title);
  const message = $derived(
    offline ? `Nothing answered at ${hub.serverUrl}.` : error instanceof Error ? error.message : String(error),
  );
  const hint = $derived(offline ? 'Start folo serve, or check the server URL.' : error instanceof ApiError ? error.hint : undefined);
  const code = $derived(error instanceof ApiError ? `${error.code}${error.status ? ` · HTTP ${error.status}` : ''}` : undefined);
</script>

<div class="err" class:compact role="alert">
  <div class="icon"><TriangleAlert size={16} strokeWidth={1.75} /></div>
  <div class="text">
    <p class="title">{heading}</p>
    <p class="msg">{message}</p>
    {#if hint}<p class="hint">{hint}</p>{/if}
    {#if code}<details class="code"><summary>Technical details</summary>{code}</details>{/if}
  </div>
  <div class="acts">
    {#if retry}<Button size="sm" icon={RotateCw} onclick={retry}>Retry</Button>{/if}
    {#if offline}<Button size="sm" variant="ghost" icon={Settings} href={href('/settings')}>Connection settings</Button>{/if}
  </div>
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
    margin-top: 2px;
    font-family: var(--font-mono);
    color: var(--text-tertiary);
    font-size: var(--type-xs-size);
  }
  .code summary {
    cursor: pointer;
    font-family: var(--font-sans);
    width: max-content;
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1);
    justify-content: flex-end;
  }
  .title {
    color: var(--danger-text);
  }
</style>
