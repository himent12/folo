<script lang="ts">
  /** Streaming training log: level filter, follow mode (auto-scroll unless you scroll up). */
  import ArrowDownToLine from '@lucide/svelte/icons/arrow-down-to-line';
  import type { LogLevel, LogLine } from '$lib/api/types';
  import { clock } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';

  let { logs, version, live }: { logs: LogLine[]; version: number; live: boolean } = $props();

  let level = $state<'all' | 'warn' | 'error'>('all');
  let follow = $state(true);
  let box: HTMLDivElement | undefined = $state();

  const order: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };
  const shown = $derived.by(() => {
    void version;
    const min = level === 'all' ? 0 : order[level];
    return logs.filter((l) => order[l.level] >= min).slice(-500);
  });
  const counts = $derived.by(() => {
    void version;
    return { warn: logs.filter((l) => l.level === 'warn').length, error: logs.filter((l) => l.level === 'error').length };
  });

  $effect(() => {
    void shown;
    if (follow && box) box.scrollTop = box.scrollHeight;
  });

  function onscroll() {
    if (!box) return;
    follow = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  }
</script>

<Panel title="Logs" meta={String(logs.length)} pad="none" id="logs">
  {#snippet actions()}
    <Segmented
      size="sm"
      label="Log level"
      bind:value={level}
      options={[
        { value: 'all', label: 'All' },
        { value: 'warn', label: 'Warn', meta: counts.warn ? String(counts.warn) : undefined },
        { value: 'error', label: 'Error', meta: counts.error ? String(counts.error) : undefined },
      ]}
    />
  {/snippet}
  <div class="wrap">
    <!-- svelte-ignore a11y_no_noninteractive_tabindex (scrollable region must be keyboard reachable) -->
    <div class="log" bind:this={box} {onscroll} role="log" aria-live={live ? 'polite' : 'off'} aria-label="Training log" tabindex="0">
      {#each shown as l, i (i)}
        <p class="line {l.level}"><span class="t">{clock(l.t)}</span><span class="lv">{l.level}</span><span class="m">{l.message}</span></p>
      {:else}
        <p class="empty">No log lines{level !== 'all' ? ` at ${level} level` : ''} yet.</p>
      {/each}
    </div>
    {#if !follow}
      <div class="jump">
        <Button size="sm" icon={ArrowDownToLine} onclick={() => ((follow = true), box && (box.scrollTop = box.scrollHeight))}>Follow</Button>
      </div>
    {/if}
  </div>
</Panel>

<style>
  .wrap {
    position: relative;
  }
  .log {
    height: 300px;
    overflow: auto;
    padding: var(--space-2) 0;
    background: var(--bg-inset);
    border-top: 1px solid var(--line-subtle);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    font-family: var(--font-mono);
    font-size: var(--type-xs-size);
    line-height: 18px;
  }
  .log:focus-visible {
    outline-offset: -2px;
  }
  .line {
    display: flex;
    gap: var(--space-3);
    padding: 0 var(--space-4);
  }
  .m {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .line:hover {
    background: var(--bg-hover);
  }
  .t {
    color: var(--text-disabled);
    flex-shrink: 0;
  }
  .lv {
    width: 40px;
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .m {
    color: var(--text-secondary);
    min-width: 0;
  }
  .debug .m {
    color: var(--text-tertiary);
  }
  .warn .lv,
  .warn .m {
    color: var(--warning-text);
  }
  .error .lv,
  .error .m {
    color: var(--danger-text);
  }
  .empty {
    padding: var(--space-4);
    color: var(--text-tertiary);
    font-family: var(--font-sans);
  }
  .jump {
    position: absolute;
    right: var(--space-4);
    bottom: var(--space-3);
  }
</style>
