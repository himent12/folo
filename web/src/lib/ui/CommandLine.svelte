<script lang="ts">
  /**
   * A command to run, as one line that never wraps (a shell command broken mid-token, `'vulkan` / `';`, cannot be
   * read or pasted): full width, it scrolls sideways with a fade on the clipped edge (`fadeScroll`), and Copy copies
   * it whole. Every command the app shows to be run goes through it (or ui/Remedy, which holds one), never a truncated
   * `<code>`.
   */
  import Copy from '@lucide/svelte/icons/copy';
  import Button from './Button.svelte';
  import { fadeScroll } from './fadeScroll';
  import { toast } from './toast.svelte';

  let {
    command,
    label = 'Command',
    copied,
  }: { command: string; label?: string; /** What the copied command does, said in the toast. */ copied?: string } = $props();

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      toast.success('Command copied', copied);
    } catch (e) {
      toast.error(e, 'Could not copy the command');
    }
  }
</script>

<div class="cmd">
  <pre class="line" data-scrolls aria-label={label} use:fadeScroll><code>{command}</code></pre>
  <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" onclick={copy}>Copy</Button>
</div>

<style>
  .cmd {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  .line {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
    white-space: pre;
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--control-sm);
    color: var(--text-primary);
    border-radius: var(--radius-xs);
  }
  .line::-webkit-scrollbar {
    display: none;
  }
  .line:global([data-fade='end']) {
    mask-image: linear-gradient(90deg, #000 calc(100% - 32px), transparent);
  }
  .line:global([data-fade='start']) {
    mask-image: linear-gradient(90deg, transparent, #000 32px);
  }
  .line:global([data-fade='both']) {
    mask-image: linear-gradient(90deg, transparent, #000 32px, #000 calc(100% - 32px), transparent);
  }
  .line:focus-visible {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 2px;
  }
</style>
