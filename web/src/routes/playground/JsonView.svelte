<script lang="ts">
  /** Pretty, syntax-tinted JSON with a copy button. */
  import Check from '@lucide/svelte/icons/check';
  import Copy from '@lucide/svelte/icons/copy';
  import Button from '$lib/ui/Button.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { jsonText, jsonTokens } from './json';

  let { value, label = 'JSON' }: { value: unknown; label?: string } = $props();

  const tokens = $derived(jsonTokens(value));
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
  <pre tabindex="0" aria-label={label}><code
      >{#each tokens as tk, i (i)}{#if tk.t === 'ws'}{tk.v}{:else}<span class={tk.t}>{tk.v}</span>{/if}{/each}</code
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
