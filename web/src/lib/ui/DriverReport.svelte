<script lang="ts">
  /**
   * The server's own words about a capability, verbatim, for an issue or a terminal: folded under "Driver report"
   * (closed until asked for, in every state) and copied whole.
   */
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Copy from '@lucide/svelte/icons/copy';
  import Button from './Button.svelte';
  import { toast } from './toast.svelte';

  let { report }: { report: string } = $props();

  async function copy() {
    try {
      await navigator.clipboard.writeText(report);
      toast.success('Driver report copied');
    } catch (e) {
      toast.error(e, 'Could not copy the driver report');
    }
  }
</script>

<details class="report">
  <summary><ChevronRight size={12} strokeWidth={2} aria-hidden="true" />Driver report</summary>
  <div class="body">
    <pre aria-label="Driver report">{report}</pre>
    <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy driver report" onclick={copy}>Copy</Button>
  </div>
</details>

<style>
  .report {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  summary {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary:hover {
    color: var(--text-secondary);
  }
  summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .report[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: start;
    gap: var(--space-2);
    margin-top: var(--space-2);
    padding: var(--space-2) var(--space-1) var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  /* The report is the server's prose, not a command: it wraps (between words; inside a long one only if it must). */
  pre {
    padding-top: 2px;
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  @media (pointer: coarse) {
    summary {
      min-height: 44px;
    }
  }
</style>
