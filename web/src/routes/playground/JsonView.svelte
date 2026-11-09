<script lang="ts">
  /** Pretty, syntax-tinted JSON with a copy button. */
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
  <!-- Long lines wrap at a space (never cut at the panel's edge) and hang 4 characters deeper than their own indent. -->
  <pre tabindex="0" aria-label={label}><code
      >{#each lines as ln, i (i)}<span class="ln" style:--ind="{ln.indent}ch"
          >{' '.repeat(ln.indent)}{#each ln.tokens as tk, j (j)}{#if tk.t === 'ws'}{tk.v}{:else}<span class={tk.t}>{tk.v === ': ' ? COLON : tk.v}</span>{/if}{/each}</span
        >{/each}</code
    ></pre>
</div>

<style>
  .json {
    position: relative;
    min-width: 0;
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
    padding-right: 96px; /* grid-ok: room for the copy and wrap buttons */
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    tab-size: 2;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  /* One block per line: a wrapped line's continuation starts two steps (4 ch) right of the line's own indent. */
  .ln {
    display: block;
    padding-left: calc(var(--ind) + 4ch);
    text-indent: calc(-1 * (var(--ind) + 4ch));
  }
  pre:focus-visible {
    outline-offset: 2px;
  }
  .key {
    color: var(--text-primary);
  }
  .str {
    color: var(--chart-2);
  }
  .num {
    color: var(--chart-4);
  }
  .lit {
    color: var(--chart-3);
  }
  .punct {
    color: var(--text-tertiary);
  }
</style>
