<script lang="ts">
  /**
   * An entry that has not finished: where it is, live (the benchmark's stream). Queued: its place behind the model
   * entries that run first. Running: the share done, the pass it is on (the unchanged images, or a condition at a
   * severity), and the plan's passes as a checklist (done, now, to come), so "haze 3 of 12 conditions" is something
   * seen. Stopping: that it is ending what is in flight. An open external entry is receiving its predictions in parts
   * from its sender (any order, so no checklist): the records received of all the plan could take, and that it is
   * scored when its sender finishes it. The score lands here when it is done.
   */
  import Check from '@lucide/svelte/icons/check';
  import type { Bench, BenchCatalog, BenchEntry } from '$lib/api/types';
  import { num } from '$lib/format';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import { progressFacts, progressShare, receiving } from '../words';

  let { entry, bench, catalog, lost, ahead }: { entry: BenchEntry; bench: Bench; catalog: BenchCatalog | undefined; lost: boolean; ahead: BenchEntry[] } = $props();

  const nameOf = (id: string) => catalog?.conditions.find((c) => c.id === id)?.name ?? id;
  /** The passes it runs, in order: the unchanged images, then each planned condition at each severity. */
  const passes = $derived([
    { key: 'clean', condition: null as string | null, severities: [] as number[] },
    ...(entry.conditions ? bench.plan.conditions.map((c) => ({ key: c.condition, condition: c.condition as string | null, severities: c.severities })) : []),
  ]);
  /** How many image passes are done: whole passes of the entry's images. */
  const passesDone = $derived(entry.images ? Math.floor(entry.progress.done / entry.images) : 0);
  /** Each (condition, severity) pass's index in the order run. */
  const indexOf = (cond: string | null, sev: number | null) => {
    let k = 0;
    for (const p of passes) {
      if (p.condition === null) {
        if (cond === null) return k;
        k++;
        continue;
      }
      for (const s of p.severities) {
        if (p.condition === cond && s === sev) return k;
        k++;
      }
    }
    return -1;
  };
  const passState = (cond: string | null, sev: number | null) => {
    const i = indexOf(cond, sev);
    if (entry.status === 'queued') return 'todo';
    return i < passesDone ? 'done' : i === passesDone ? 'now' : 'todo';
  };
  const p = $derived(progressFacts(entry, catalog));
  const recv = $derived(receiving(entry));
</script>

<Panel title={entry.status === 'queued' ? 'Waiting to run' : entry.status === 'stopping' ? 'Stopping' : recv ? 'Receiving predictions' : 'Running'} meta={recv ? undefined : (p.pct ?? undefined)} id="progress">
  <div class="run">
    {#if recv}
      <Progress value={progressShare(entry)} size="md" tone={lost ? 'neutral' : 'accent'} label="{entry.name}: records received" />
      <p class="lead num">{p.pct}, sent in parts by whatever runs the model (a script, or Add entry here).</p>
      <p class="cap">It is scored when its sender finishes it: every unchanged image answered, each condition whole or left out. Stopping it discards what was received.</p>
    {:else if entry.status === 'queued'}
      <p class="lead">
        Model entries run one at a time on the server’s GPU.
        {#if ahead.length}{ahead.length === 1 ? 'One entry runs' : `${ahead.length} entries run`} before it: {ahead.map((e) => e.name).join(', ')}.{/if}
      </p>
    {:else}
      <Progress value={progressShare(entry)} size="md" tone={lost || entry.status === 'stopping' ? 'neutral' : 'accent'} label="{entry.name} progress" />
      <p class="lead num">
        {p.where}{#if entry.status === 'running'}{' '}· {num(entry.progress.done)} of {num(entry.progress.total)} image passes{/if}
      </p>
    {/if}
    {#if !recv}
    <ol class="passes" aria-label="Passes it runs">
      {#each passes as ps (ps.key)}
        <li>
          <span class="p-name">{ps.condition ? nameOf(ps.condition) : 'Unchanged images'}</span>
          <span class="p-sev">
            {#if ps.condition === null}
              {@const st = passState(null, null)}
              <span class="pill {st}" aria-label="{st === 'done' ? 'done' : st === 'now' ? 'running now' : 'to come'}">{#if st === 'done'}<Check size={12} strokeWidth={2} aria-hidden="true" />{/if}{num(entry.images)} images</span>
            {:else}
              {#each ps.severities as s (s)}
                {@const st = passState(ps.condition, s)}
                <span class="pill {st}" aria-label="Severity {s}: {st === 'done' ? 'done' : st === 'now' ? 'running now' : 'to come'}">{#if st === 'done'}<Check size={12} strokeWidth={2} aria-hidden="true" />{/if}{s}</span>
              {/each}
            {/if}
          </span>
        </li>
      {/each}
    </ol>
    <p class="cap">The score and the full result appear here when it is done{entry.conditions || !bench.plan.conditions.length ? '' : '; it runs the unchanged images only, so it will have no robustness'}.</p>
    {/if}
  </div>
</Panel>

<style>
  .run {
    display: grid;
    gap: var(--space-3);
  }
  .lead {
    color: var(--text-secondary);
  }
  .passes {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 0 var(--space-6);
    margin-top: var(--space-1);
  }
  .passes li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    min-height: 36px;
    border-top: 1px solid var(--line-subtle);
  }
  .p-name {
    color: var(--text-secondary);
  }
  .p-sev {
    display: flex;
    gap: var(--space-1);
  }
  .pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 2px;
    min-width: 24px;
    height: 24px;
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
  }
  .pill.done {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  /* Running now: the live mark, lime meaning "now" (DESIGN_DIRECTION §2.6). */
  .pill.now {
    box-shadow: inset 0 0 0 1.5px var(--accent-line);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .pill.todo {
    color: var(--text-tertiary);
    box-shadow: inset 0 0 0 1px var(--line-default);
  }
  .cap {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
</style>
