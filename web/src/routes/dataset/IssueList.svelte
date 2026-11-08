<script lang="ts">
  /**
   * The evidence behind a health finding: the `folo dataset check` issues of one severity, as `file:line` and the
   * message, paged 50 at a time. The same check from a terminal is offered as a copyable command.
   */
  import Copy from '@lucide/svelte/icons/copy';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { DatasetIssue, IssueSeverity } from '$lib/api/types';
  import { loadIssues } from './issues';
  import { num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let { datasetId, path, severity = 'error' }: { datasetId: string; path: string; severity?: IssueSeverity } = $props();

  const PAGE = 50;
  /** Most issues read in total; past it the list says how many there are and points to the CLI. */
  const CAP = 2000;
  const id = $derived(datasetId);
  const sev = $derived(severity);
  let shown = $state(PAGE);
  // Each kind of this severity is read page by page (./issues), so "no errors" is only said when none exist.
  const res = new Resource((s) => loadIssues(hub.api, id, sev, CAP, s));
  const list = $derived<DatasetIssue[]>(res.data?.list ?? []);
  const total = $derived(res.data?.total ?? 0);
  const where = (i: DatasetIssue) => (i.line !== null ? `${i.file}:${i.line}` : i.file);
  const cli = $derived(`folo dataset check ${/\s/.test(path) ? `"${path}"` : path}`);

  async function copyCli() {
    try {
      await navigator.clipboard.writeText(cli);
      toast.success('Command copied', 'It lists every issue with a suggested fix.');
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }
</script>

<div class="issues">
  {#if res.error && !res.data}
    <ErrorState error={res.error} retry={() => res.refresh()} compact title="Could not load the issues" />
  {:else if !res.data}
    <div class="sk">{#each [0, 1, 2] as i (i)}<Skeleton h="16px" />{/each}</div>
  {:else if total === 0}
    <p class="none">No {severity === 'error' ? 'errors' : severity === 'warning' ? 'warnings' : 'notes'} in the latest check.</p>
  {:else}
    <ul>
      {#each list.slice(0, shown) as i, k (k)}
        <li>
          <span class="file mono" title={where(i)}>{where(i)}</span>
          <span class="msg">{i.message}</span>
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
  <div class="cli">
    <code class="mono">{cli}</code>
    <Button size="sm" variant="ghost" icon={Copy} onclick={copyCli}>Copy command</Button>
  </div>
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
  .more {
    justify-self: start;
    height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .more:hover {
    color: var(--text-primary);
  }
  .cli {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    min-width: 0;
  }
  .cli code {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
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
