<script lang="ts">
  /**
   * Pretty, syntax-tinted JSON with a copy button. From 1024 px every line stays whole (one detection per line, as
   * ./json lays them out): the block scrolls sideways (`data-scrolls`) and fades the edge, right or bottom, that has
   * more behind it. Narrower, long lines wrap at a space and hang deeper than their own indent.
   */
  import Check from '@lucide/svelte/icons/check';
  import Copy from '@lucide/svelte/icons/copy';
  import Button from '$lib/ui/Button.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { jsonLines, jsonText, jsonTokens } from './json';

  let { value, label = 'JSON' }: { value: unknown; label?: string } = $props();

  const lines = $derived(jsonLines(jsonTokens(value)));
  /** A key stays with its value when a line wraps: the space after the colon does not break (Copy keeps plain JSON). */
  const COLON = ':\u00a0';
  let copied = $state(false);

  /** Which edges have more behind them: the block fades those (right, bottom). */
  function edges(node: HTMLElement) {
    const update = () => {
      const more = (rest: number) => (rest > 1 ? '' : null);
      const x = more(node.scrollWidth - node.clientWidth - node.scrollLeft);
      const y = more(node.scrollHeight - node.clientHeight - node.scrollTop);
      if (x === null) delete node.dataset.moreX;
      else node.dataset.moreX = x;
      if (y === null) delete node.dataset.moreY;
      else node.dataset.moreY = y;
    };
    update();
    node.addEventListener('scroll', update, { passive: true });
    const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
    ro?.observe(node);
    return {
      destroy() {
        node.removeEventListener('scroll', update);
        ro?.disconnect();
      },
    };
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(jsonText(value));
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {
      toast.info('Copy failed', 'Your browser blocked clipboard access.');
    }
  }
</script>

<div class="json">
  <div class="copy">
    <Button size="sm" icon={copied ? Check : Copy} onclick={copy}>{copied ? 'Copied' : 'Copy'}</Button>
  </div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <!-- From 1024 px a line never wraps (the block scrolls sideways); narrower, long lines wrap at a space and hang 4
       characters deeper than their own indent. -->
  <pre tabindex="0" aria-label={label} data-scrolls use:edges><code
      >{#each lines as ln, i (i)}<span class="ln" style:--ind="{ln.indent}ch"
          >{' '.repeat(ln.indent)}{#each ln.tokens as tk, j (j)}{#if tk.t === 'ws'}{tk.v}{:else}<span class={tk.t}>{tk.v === ': ' ? COLON : tk.v}</span>{/if}{/each}</span
        >{/each}</code
    ></pre>
</div>

<style>
  /* The frame is the wrapper, so the scroller's edge fades never fade the frame's own edge. */
  .json {
    position: relative;
    min-width: 0;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
    overflow: hidden;
  }
  .copy {
    position: absolute;
    top: var(--space-2);
    right: var(--space-2);
    z-index: 1;
  }
  pre {
    margin: 0;
    max-height: 420px;
    overflow: auto;
    padding: var(--space-3) var(--space-4);
    padding-right: 96px; /* grid-ok: room for the copy button */
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    tab-size: 2;
    white-space: pre;
  }
  .ln {
    display: block;
  }
  /* More below, more to the right, or both: the edge with more behind it fades over its last 32 px. */
  pre:global([data-more-y]) {
    mask-image: linear-gradient(#000 calc(100% - 32px), transparent);
  }
  pre:global([data-more-x]) {
    mask-image: linear-gradient(90deg, #000 calc(100% - 32px), transparent);
  }
  pre:global([data-more-x][data-more-y]) {
    mask-image:
      linear-gradient(#000 calc(100% - 32px), transparent),
      linear-gradient(90deg, #000 calc(100% - 32px), transparent);
    mask-composite: intersect;
  }
  /* Under 1024 px: long lines wrap at a space, and a wrapped line's continuation starts two steps (4 ch) right of the
     line's own indent. */
  @media (max-width: 1023px) {
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .ln {
      padding-left: calc(var(--ind) + 4ch);
      text-indent: calc(-1 * (var(--ind) + 4ch));
    }
  }
  pre:focus-visible {
    outline-offset: -2px;
  }
  .key {
    color: var(--text-primary);
  }
  .str {
    color: var(--chart-text-2);
  }
  .num {
    color: var(--chart-text-4);
  }
  .lit {
    color: var(--chart-text-3);
  }
  .punct {
    color: var(--text-tertiary);
  }
</style>
