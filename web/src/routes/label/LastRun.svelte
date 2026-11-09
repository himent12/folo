<script lang="ts">
  /**
   * How the job's last run ended, as facts: its outcome and counts, the agent and model, and what the agent reported it
   * used; the job's totals too once it has had more than one run.
   */
  import type { LabelAgent, LabelJob, LabelRun } from '$lib/api/types';
  import { ago, num } from '$lib/format';
  import Meta from '$lib/ui/Meta.svelte';
  import { OUTCOME, runAgent, usageFacts } from './words';

  let { job, run, agents }: { job: LabelJob; run: LabelRun; agents: readonly LabelAgent[] | undefined } = $props();

  const used = $derived(usageFacts(run.usage));
  const total = $derived(job.usage.calls > run.usage.calls ? usageFacts(job.usage) : []);
</script>

<div class="last">
  <p class="group-label">Last run</p>
  <p class="line num">
    <Meta>
      <span>{run.outcome ? OUTCOME[run.outcome] : 'Running'}</span>
      <span>{num(run.done)} labeled</span>
      {#if run.failed}<span>{num(run.failed)} failed</span>{/if}
      {#each runAgent(run, agents) as f (f)}<span>{f}</span>{/each}
      {#if run.finished}<span>{ago(run.finished)}</span>{/if}
    </Meta>
  </p>
  {#if used.length}
    <p class="line num"><Meta>{#each used as f (f)}<span>{f}</span>{/each}</Meta></p>
  {/if}
  {#if total.length}
    <p class="line num"><Meta><span>All runs</span>{#each total as f (f)}<span>{f}</span>{/each}</Meta></p>
  {/if}
  {#if used.length}<p class="note">Tokens and cost as reported by the agent.</p>{/if}
</div>

<style>
  .last {
    display: grid;
    gap: var(--space-1);
  }
  .line {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
</style>
