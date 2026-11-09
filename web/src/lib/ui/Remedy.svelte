<script lang="ts">
  /**
   * One remediation block, the same wherever something is off and a command turns it on (Settings › System, New
   * run's Mixed precision): the cause in plain text (identifiers in backticks are set as code), the shell the command
   * is written for ("In PowerShell:"), the command as one full-width line (`CommandLine`: never wrapped, scrolls
   * sideways, Copy), then the server's own words folded under Driver report. `compact` (inside a form) tightens the
   * spacing; the parts and their order stay.
   */
  import CommandLine from './CommandLine.svelte';
  import DriverReport from './DriverReport.svelte';

  let {
    cause,
    id,
    shell,
    command,
    report,
    compact = false,
  }: {
    cause: string;
    /** The cause's id, for the control it explains (`aria-describedby`). */
    id?: string;
    shell?: string;
    command?: string;
    report?: string | null;
    compact?: boolean;
  } = $props();

  const parts = $derived(cause.split('`').map((text, i) => ({ text, code: i % 2 === 1 })));
</script>

<div class="remedy" class:compact>
  <p class="cause" {id}>{#each parts as p, i (i)}{#if p.code}<code>{p.text}</code>{:else}{p.text}{/if}{/each}</p>
  {#if command}
    <div class="run">
      {#if shell}<p class="shell">In {shell}:</p>{/if}
      <CommandLine {command} label={shell ? `${shell} command` : 'Command'} />
    </div>
  {/if}
  {#if report}<DriverReport {report} />{/if}
</div>

<style>
  .remedy {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .remedy.compact {
    gap: var(--space-2);
  }
  /* Helper text keeps a readable measure (≤ 60 characters a line); the command under it runs the full width. */
  .cause {
    max-inline-size: var(--measure);
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  .cause code {
    font-family: var(--font-mono);
    white-space: nowrap;
  }
  .run {
    display: grid;
    gap: var(--space-1);
    min-width: 0;
  }
  .shell {
    color: var(--text-tertiary);
  }
</style>
