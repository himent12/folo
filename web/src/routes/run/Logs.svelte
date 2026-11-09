<script lang="ts">
  /**
   * Streaming training log: level filter, follow mode (auto-scroll unless you scroll up). The view always starts on a
   * whole entry: following, it pins the last entry to the bottom and, when that would cut the top entry (the wrapped
   * tail of a message, without its time and level), pads the end by the cut so the top entry starts on the edge; read
   * by hand, it snaps to entries. The box is at least `--log-h` tall (300 px) and grows with its panel when the page
   * stretches it (Run's columns end together).
   *
   * **Silent by default** (DESIGN_DIRECTION §2.9, accessibility): a log streams several lines a second, and a live
   * region on it would read every one with no way to stop it (follow mode only scrolls). The log is `role="log"` with
   * `aria-live="off"`; "Announce new lines" (a pressed toggle in its header, while the run is live) turns on a polite
   * status line that reads them in batches, at most one announcement every `ANNOUNCE_MS`: the one new line, or how
   * many arrived and the latest. The run's milestones and failures are not the log's to say: its start is the shell's
   * live region (shell/SidebarLive) and its end, failure included, a toast (shell/RunWatch), announced whether or not
   * this is on.
   */
  import ArrowDownToLine from '@lucide/svelte/icons/arrow-down-to-line';
  import Volume2 from '@lucide/svelte/icons/volume-2';
  import VolumeOff from '@lucide/svelte/icons/volume-off';
  import { untrack } from 'svelte';
  import type { LogLevel, LogLine } from '$lib/api/types';
  import { clock } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { tooltip } from '$lib/ui/tooltip';
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
    // Next frame: pinning changes the tail padding, which would resize what is observed inside this callback.
    let frame = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => follow && pin());
    });
    ro.observe(el);
    ro.observe(content);
    return () => (cancelAnimationFrame(frame), ro.disconnect());
  });

  function onscroll() {
    if (!box) return;
    follow = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  }

  /** At most one announcement this often: a batch of the lines that arrived since the last. */
  const ANNOUNCE_MS = 10_000;
  let announce = $state(false);
  let said = $state('');
  /** The last line announced (or the last one there when announcing was turned on): later ones are new. */
  let heard: LogLine | undefined;
  const lineText = (l: LogLine) => (l.level === 'info' ? l.message : `${l.level}: ${l.message}`);
  /** The lines since `heard` that the level filter shows, said as one batch. */
  function flush() {
    const at = heard ? logs.lastIndexOf(heard) : -1;
    const min = level === 'all' ? 0 : order[level];
    // `heard` may have left the capped list (or a history load replaced it): then what came after it in time.
    const since = at >= 0 || !heard ? logs.slice(at + 1) : logs.filter((l) => l.t > heard!.t);
    const fresh = since.filter((l) => order[l.level] >= min);
    heard = logs[logs.length - 1] ?? heard;
    if (!fresh.length) return;
    const last = fresh[fresh.length - 1]!;
    said = fresh.length === 1 ? lineText(last) : `${fresh.length} new log lines. Latest: ${lineText(last)}`;
  }
  $effect(() => {
    if (!announce || !live) return;
    // From now on: what was there when it was turned on is not news.
    untrack(() => {
      heard = logs[logs.length - 1];
      said = '';
    });
    const t = setInterval(flush, ANNOUNCE_MS);
    return () => clearInterval(t);
  });
</script>

{#snippet levels()}
  {#if live}
    <!-- Off by default: the log never speaks unless asked (see the header comment). -->
    <span class="announce" use:tooltip={announce ? 'Stop announcing new lines' : 'Announce new lines'}>
      <Button size="sm" variant="ghost" icon={announce ? Volume2 : VolumeOff} aria-pressed={announce} aria-label="Announce new lines" onclick={() => (announce = !announce)} />
    </span>
  {/if}
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
      <!-- Silent: new lines are read only through the batched status line below, when asked for. -->
      <div class="log" bind:this={box} {onscroll} role="log" aria-live="off" aria-label="Training log" tabindex="0">
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
      <p class="sr-only" role="status">{announce && live ? said : ''}</p>
      {#if !follow}
        <div class="jump">
          <Button size="sm" icon={ArrowDownToLine} onclick={() => ((follow = true), pin())}>Follow</Button>
        </div>
      {/if}
    </div>
  {/if}
</Panel>

<style>
  .announce {
    display: inline-flex;
  }
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
