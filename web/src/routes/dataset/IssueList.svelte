<script lang="ts">
  /**
   * The `folo dataset check` issues, as `file:line`, the message and what a cleanup would do, paged 50 at a time. A
   * severity filter (with the server's counts, where it sends them; left out with `severityFilter={false}` where the
   * list sits under one finding and its severity is fixed) narrows it, and inside a severity with several
   * kinds a sub-filter by check type; a caption above the rows says exactly which rows they are. The same check from a terminal is offered as a copyable command.
   */
  import { untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { DatasetIssue, IssueCounts, IssueKind, IssueSeverity } from '$lib/api/types';
  import { issueCaption, KIND_LABEL, KINDS, loadIssues } from './issues';
  import { num } from '$lib/format';
  import { quote } from '$lib/trainCli';
  import CommandLine from '$lib/ui/CommandLine.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';

  let {
    datasetId,
    path,
    severity = $bindable('error'),
    severityFilter = true,
    counts,
  }: { datasetId: string; path: string; severity?: IssueSeverity; severityFilter?: boolean; counts?: IssueCounts } = $props();

  const PAGE = 50;
  /** Most issues read in total; past it the list says how many there are and points to the CLI. */
  const CAP = 2000;
  const id = $derived(datasetId);
  // The filter moves between the three severities (bindable, so a finding elsewhere can move it too); a kind and the
  // paging start over with each.
  const sev = $derived(severity);
  let kind = $state<'all' | IssueKind>('all');
  let shown = $state(PAGE);
  // Every server can filter by kind; a whole severity is one `?severity=` query where the server sends `counts`,
  // else each of its kinds in turn (./issues), so "no errors" is only said when none exist.
  const res = new Resource((s) => loadIssues(hub.api, id, sev, CAP, s, kind === 'all' ? undefined : kind));
  const list = $derived<DatasetIssue[]>(res.data?.list ?? []);
  const total = $derived(res.data?.total ?? 0);
  // Counts from the latest answer that had them (the parent's until then); without them, per-kind totals of the last
  // unfiltered read.
  const known = $derived(res.data?.counts ?? counts);
  let lastByKind = $state<Partial<Record<IssueKind, number>>>({});
  $effect.pre(() => {
    void sev;
    untrack(() => {
      kind = 'all';
      shown = PAGE;
      lastByKind = {};
    });
  });
  $effect(() => {
    const b = res.data?.byKind;
    if (b && (known || kind === 'all')) lastByKind = b;
  });
  const byKind = $derived(known ? Object.fromEntries(KINDS[sev].filter((k) => known.by_kind[k]).map((k) => [k, known.by_kind[k]!])) : lastByKind);
  const kinds = $derived(KINDS[sev].filter((k) => byKind[k]));
  const NAMES: Record<IssueSeverity, [string, string]> = { error: ['error', 'errors'], warning: ['warning', 'warnings'], info: ['note', 'notes'] };
  const sevOptions = $derived(
    (['error', 'warning', 'info'] as const).map((v) => ({
      value: v,
      label: v === 'error' ? 'Errors' : v === 'warning' ? 'Warnings' : 'Notes',
      meta: known ? num(known[v]) : undefined,
    })),
  );
  function setSeverity(v: IssueSeverity) {
    severity = v;
  }
  const where = (i: DatasetIssue) => (i.line !== null ? `${i.file}:${i.line}` : i.file);
  // --json lists every issue (the plain report stops at 200), --fix-dry-run adds each one's fix; nothing is modified.
  // Quoted for PowerShell, as every command Studio shows (lib/trainCli `quote`).
  const cli = $derived(`folo dataset check ${quote(path)} --fix-dry-run --json`);

</script>

<div class="issues">
  {#if severityFilter || kinds.length > 1 || kind !== 'all'}
  <div class="filters">
    {#if severityFilter}
      <Segmented label="Severity" size="sm" value={sev} options={sevOptions} onchange={setSeverity} />
    {/if}
    {#if kinds.length > 1 || kind !== 'all'}
      <span class="kind">
        <Select
          size="sm"
          label="Check type"
          bind:value={kind}
          onchange={() => (shown = PAGE)}
          options={[
            { value: 'all', label: `All ${NAMES[sev][0]} types` },
            ...kinds.map((k) => ({ value: k, label: `${KIND_LABEL[k]} · ${num(byKind[k])}` })),
          ]}
        />
      </span>
    {/if}
  </div>
  {/if}
  {#if res.error && !res.data}
    <ErrorState error={res.error} retry={() => res.refresh()} compact title="Could not load the issues" />
  {:else if !res.data || res.loading}
    <div class="sk">{#each [0, 1, 2] as i (i)}<Skeleton h="16px" />{/each}</div>
  {:else if total === 0}
    <p class="none">No {NAMES[sev][1]} in the latest check.</p>
  {:else}
    <!-- Under its finding (no severity filter), the finding's own paragraph already says what these rows are: a caption
         only names a picked check type, without repeating what it means. -->
    {#if severityFilter || kind !== 'all'}<p class="caption num">{issueCaption(sev, total, kind === 'all' ? undefined : kind, !severityFilter)}</p>{/if}
    <ul>
      {#each list.slice(0, shown) as i, k (k)}
        <li>
          <span class="file mono" title={where(i)}>{where(i)}</span>
          <span class="msg">{i.message}{#if i.fix}<span class="fix">Fix: {i.fix}</span>{/if}</span>
        </li>
      {/each}
    </ul>
    {#if list.length > shown}
      <button type="button" class="more" onclick={() => (shown += PAGE)}>Show {num(Math.min(PAGE, list.length - shown))} more of {num(list.length - shown)}</button>
    {/if}
    {#if total > list.length}
      <p class="none">Showing the first {num(list.length)} of {num(total)}; the command below lists them all.</p>
    {/if}
  {/if}
  <!-- The same check from a terminal: one line that scrolls, never cut with an ellipsis (ui/CommandLine). -->
  <CommandLine command={cli} label="The dataset check command" copied="It prints every issue with its fix as JSON, and changes nothing." />
</div>

<style>
  .issues {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  ul {
    display: grid;
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
  }
  li {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
    gap: var(--space-3);
    align-items: baseline;
    min-height: 28px;
    padding: var(--space-1) var(--space-3);
    border-bottom: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
  }
  li:last-child {
    border-bottom: 0;
  }
  .file {
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .msg {
    color: var(--text-secondary);
  }
  /* What a cleanup would do: a quiet second line under the message. */
  .fix {
    display: block;
    color: var(--text-tertiary);
  }
  .filters {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .kind {
    min-width: 0;
    width: 200px;
    max-width: 100%;
  }
  .more {
    justify-self: start;
    height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .more:hover {
    color: var(--text-primary);
  }
  .caption {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .none {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .sk {
    display: grid;
    gap: var(--space-2);
  }
  @media (max-width: 767px) {
    li {
      grid-template-columns: minmax(0, 1fr);
      gap: 0;
      padding: var(--space-2) var(--space-3);
    }
  }
  @media (pointer: coarse) {
    .more {
      height: 44px;
    }
  }
</style>
