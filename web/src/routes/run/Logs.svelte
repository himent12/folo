<script lang="ts">
  /**
   * Streaming training log: level filter, follow mode (auto-scroll unless you scroll up). The view always starts on a
   * whole entry: following, it pins the last entry to the bottom and, when that would cut the top entry (the wrapped
   * tail of a message, without its time and level), pads the end by the cut so the top entry starts on the edge; read
   * by hand, it snaps to entries. The box is at least `--log-h` tall (300 px) and grows with its panel when the page
   * stretches it (Run's columns end together).
   */
  import ArrowDownToLine from '@lucide/svelte/icons/arrow-down-to-line';
  import type { LogLevel, LogLine } from '$lib/api/types';
  import { clock } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { logPieces } from './logTokens';

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
  // `logs` changes in place; `version` is the signal that it did.
  const total = $derived.by(() => {
    void version;
    return logs.length;
  });
  const counts = $derived.by(() => {
    void version;
    return { warn: logs.filter((l) => l.level === 'warn').length, error: logs.filter((l) => l.level === 'error').length };
  });

  /** Space after the last entry, so that, pinned to the bottom, the top entry starts whole on the box's top edge. */
  let pad = 0;
  /** The box's top padding: an entry the view starts on sits this far under the edge, as the first one does. */
  const INSET = 8;

  /** Scrolls to the end, landing on a whole entry at the top (padding the end by what would otherwise be cut). */
  function pin() {
    if (!box) return;
    const end = box.scrollHeight - pad - box.clientHeight;
    let next = 0;
    if (end > 0) {
      const top = [...box.querySelectorAll<HTMLElement>('.line')].find((l) => l.offsetTop - INSET >= end - 0.5);
      next = top ? top.offsetTop - INSET - end : 0;
    }
    // Set on the element, not through state: the scroll below must see the new padding in this frame.
    pad = next;
    box.style.setProperty('--tail', `${next}px`);
    box.scrollTop = box.scrollHeight;
  }

  $effect(() => {
    void shown;
    if (follow) pin();
  });
  // Re-pin when the box's height follows the page (Run stretches it to the rail) or the entries re-wrap (the mono
  // font arriving, a narrower column).
  let inner: HTMLDivElement | undefined = $state();
  $effect(() => {
    const el = box;
    const content = inner;
    if (!el || !content) return;
    const ro = new ResizeObserver(() => follow && pin());
    ro.observe(el);
    ro.observe(content);
    return () => ro.disconnect();
  });

  function onscroll() {
    if (!box) return;
    follow = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  }
</script>

{#snippet levels()}
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

<!-- No lines at all (a queued run): the state-height rule, the header and one compact line; nothing to filter yet. -->
<Panel title="Logs" meta={total ? String(total) : undefined} pad="none" id="logs" actions={total ? levels : undefined}>
  {#if total === 0}
    <p class="none">No log lines yet.</p>
  {:else}
    <div class="wrap">
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (scrollable region must be keyboard reachable) -->
      <div class="log" bind:this={box} {onscroll} role="log" aria-live={live ? 'polite' : 'off'} aria-label="Training log" tabindex="0">
        <div bind:this={inner}>
        {#each shown as l, i (i)}
          <!-- The message wraps only at spaces, each `key value` pair and flag whole (./logTokens). -->
          <p class="line {l.level}"><span class="t">{clock(l.t)}</span><span class="lv">{l.level}</span><span class="m"
              >{#each logPieces(l.message) as p, j (j)}{#if p.space}{p.text}{:else}<span class="w" class:long={p.long}>{p.text}</span>{/if}{/each}</span
            ></p>
        {:else}
          <p class="empty">No log lines{level !== 'all' ? ` at ${level} level` : ''} yet.</p>
        {/each}
        </div>
      </div>
      {#if !follow}
        <div class="jump">
          <Button size="sm" icon={ArrowDownToLine} onclick={() => ((follow = true), pin())}>Follow</Button>
        </div>
      {/if}
    </div>
  {/if}
</Panel>

<style>
  .wrap {
    position: relative;
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .log {
    position: relative;
    flex: 1 1 auto;
    height: var(--log-h, 300px);
    overflow: auto;
    /* The end's padding grows by --tail when pinned, so the top entry is whole (pin()). */
    padding: var(--space-2) 0 calc(var(--space-2) + var(--tail, 0px));
    /* Read by hand, the view comes to rest on an entry's start. */
    scroll-snap-type: y proximity;
    scroll-padding-top: var(--space-2);
    background: var(--bg-inset);
    border-top: 1px solid var(--line-subtle);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: 18px;
  }
  /* The top padding stays clear while scrolled: a band in the log's own fill over the end of the entry above the
     one the view starts on (pin()), so the first line shown is always a whole entry. */
  .log::before {
    content: '';
    position: sticky;
    top: calc(-1 * var(--space-2));
    z-index: 1;
    display: block;
    height: var(--space-2);
    margin: calc(-1 * var(--space-2)) 0 0;
    background: var(--bg-inset);
  }
  .log:focus-visible {
    outline-offset: -2px;
  }
  .line {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    padding: 0 var(--space-4);
    scroll-snap-align: start;
  }
  /* Wrap at spaces only: a piece (a word, a `key value` pair, a flag and its value) never breaks inside, except one
     too long for any line, as a last resort. */
  .m {
    white-space: pre-wrap;
  }
  .w {
    white-space: nowrap;
  }
  .w.long {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .line:hover {
    background: var(--bg-hover);
  }
  .t {
    color: var(--text-tertiary);
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
  .none {
    padding: 0 var(--space-4) var(--space-4);
    color: var(--text-tertiary);
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
  @media (max-width: 767px) {
    /* Phones: time and level on one 12 px tertiary line above the message, which gets the full width. */
    .line {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      align-items: start;
      gap: 2px var(--space-2);
      padding: 6px var(--space-3);
    }
    /* A hairline between entries, so each meta line reads as the head of the message under it. */
    .line + .line {
      border-top: 1px solid var(--line-subtle);
    }
    .t,
    .lv {
      font-family: var(--font-sans);
      font-size: var(--type-sm-size);
      line-height: 16px;
      font-variant-numeric: tabular-nums;
    }
    .lv {
      width: auto;
    }
    .lv::before {
      content: '·';
      margin-right: var(--space-2);
      color: var(--text-tertiary);
    }
    .m {
      grid-column: 1 / -1;
    }
  }
</style>
