/**
 * The `folo dataset check` issues, as `file:line`, the message and what a cleanup would do, paged 50 at a time. A
 * severity filter (with the server's counts, where it sends them; left out with `severityFilter={false}` where the
 * list sits under one finding and its severity is fixed) narrows it, and inside a severity with several
 * kinds a sub-filter by check type; a caption above the rows says exactly which rows they are. The same check from a terminal is offered as a copyable command.
 */
import { bindable, computed, css, defaults, effect, Index, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { DatasetIssue, IssueCounts, IssueKind, IssueSeverity } from '$lib/api/types';
import { issueCaption, KIND_LABEL, KINDS, loadIssues, loadKinds } from './issues';
import { num } from '$lib/format';
import { quote } from '$lib/trainCli';
import CommandLine from '$lib/ui/CommandLine';
import ErrorState from '$lib/ui/ErrorState';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';

export interface IssueListProps {
  datasetId: string;
  path: string;
  severity?: IssueSeverity;
  severityFilter?: boolean;
  counts?: IssueCounts;
  /** Only these kinds of the severity (a finding's own issues), when given. */
  kinds?: readonly IssueKind[];
}

const PAGE = 50;
/** Most issues read in total; past it the list says how many there are and points to the CLI. */
const CAP = 2000;
const NAMES: Record<IssueSeverity, [string, string]> = { error: ['error', 'errors'], warning: ['warning', 'warnings'], info: ['note', 'notes'] };

export default function IssueList(props: IssueListProps) {
  const p = defaults(props, { severityFilter: true });
  const severity = bindable(props, 'severity', 'error' as IssueSeverity);
  const id = computed(() => p.datasetId);
  // The filter moves between the three severities (bindable, so a finding elsewhere can move it too); a kind and the
  // paging start over with each.
  const sev = computed(() => severity.get());
  const kind = signal<'all' | IssueKind>('all');
  const shown = signal(PAGE);
  // Every server can filter by kind; a whole severity is one `?severity=` query where the server sends `counts`,
  // else each of its kinds in turn (./issues), so "no errors" is only said when none exist.
  const res = new Resource((s) =>
    kind.get() === 'all' && p.kinds
      ? loadKinds(hub.api, id.get(), sev.get(), p.kinds, CAP, s)
      : loadIssues(hub.api, id.get(), sev.get(), CAP, s, kind.get() === 'all' ? undefined : (kind.get() as IssueKind)),
  );
  const list = computed<DatasetIssue[]>(() => res.data?.list ?? []);
  const total = computed(() => res.data?.total ?? 0);
  // Counts from the latest answer that had them (the parent's until then); without them, per-kind totals of the last
  // unfiltered read.
  const known = computed(() => res.data?.counts ?? p.counts);
  const lastByKind = signal<Partial<Record<IssueKind, number>>>({});
  effect.pre(() => {
    void sev.get();
    untrack(() => {
      kind.set('all');
      shown.set(PAGE);
      lastByKind.set({});
    });
  });
  effect(() => {
    const b = res.data?.byKind;
    if (b && (known.get() || kind.get() === 'all')) lastByKind.set(b);
  });
  const byKind = computed<Partial<Record<IssueKind, number>>>(() => {
    const k = known.get();
    return k ? Object.fromEntries(KINDS[sev.get()].filter((x) => k.by_kind[x]).map((x) => [x, k.by_kind[x]!])) : lastByKind.get();
  });
  const kinds = computed(() => KINDS[sev.get()].filter((k) => byKind.get()[k] && (!p.kinds || p.kinds.includes(k))));
  const sevOptions = computed(() =>
    (['error', 'warning', 'info'] as const).map((v) => ({
      value: v,
      label: v === 'error' ? 'Errors' : v === 'warning' ? 'Warnings' : 'Notes',
      meta: known.get() ? num(known.get()![v]) : undefined,
    })),
  );
  function setSeverity(v: IssueSeverity) {
    severity.set(v);
  }
  const where = (i: DatasetIssue) => (i.line !== null ? `${i.file}:${i.line}` : i.file);
  // --json lists every issue (the plain report stops at 200), --fix-dry-run adds each one's fix; nothing is modified.
  // Quoted for PowerShell, as every command Studio shows (lib/trainCli `quote`).
  const cli = computed(() => `folo dataset check ${quote(p.path)} --fix-dry-run --json`);
  const kindOptions = computed(() => [
    { value: 'all' as const, label: `All ${NAMES[sev.get()][0]} types` },
    ...kinds.get().map((k) => ({ value: k, label: `${KIND_LABEL[k]} · ${num(byKind.get()[k])}` })),
  ]);

  return (
    <div class="issues">
      {p.severityFilter || kinds.get().length > 1 || kind.get() !== 'all' ? (
        <div class="filters">
          {p.severityFilter ? <Segmented label="Severity" size="sm" value={sev.get()} options={sevOptions.get()} onchange={setSeverity} /> : null}
          {kinds.get().length > 1 || kind.get() !== 'all' ? (
            <span class="kind">
              <Select size="sm" label="Check type" bind:value={kind} onchange={() => shown.set(PAGE)} options={kindOptions.get()} />
            </span>
          ) : null}
        </div>
      ) : null}
      {res.error && !res.data ? (
        <ErrorState error={res.error} retry={() => res.refresh()} compact title="Could not load the issues" />
      ) : !res.data || res.loading ? (
        <div class="sk">
          <Skeleton h="16px" />
          <Skeleton h="16px" />
          <Skeleton h="16px" />
        </div>
      ) : total.get() === 0 ? (
        <p class="none">No {NAMES[sev.get()][1]} in the latest check.</p>
      ) : (
        <>
          {/* Under its finding (no severity filter), the finding's own paragraph already says what these rows are: a
              caption only names a picked check type, without repeating what it means. */}
          {p.severityFilter || kind.get() !== 'all' ? (
            <p class="caption num">{issueCaption(sev.get(), total.get(), kind.get() === 'all' ? undefined : (kind.get() as IssueKind), !p.severityFilter)}</p>
          ) : null}
          <ul>
            <Index each={list.get().slice(0, shown.get())}>
              {(i) => (
                <li>
                  <span class="file mono" title={where(i())}>
                    {where(i())}
                  </span>
                  <span class="msg">
                    {i().message}
                    {i().fix ? <span class="fix">Fix: {i().fix}</span> : null}
                  </span>
                </li>
              )}
            </Index>
          </ul>
          {list.get().length > shown.get() ? (
            <button type="button" class="more" onclick={() => shown.set((s) => s + PAGE)}>
              Show {num(Math.min(PAGE, list.get().length - shown.get()))} more of {num(list.get().length - shown.get())}
            </button>
          ) : null}
          {total.get() > list.get().length ? (
            <p class="none">
              Showing the first {num(list.get().length)} of {num(total.get())}; the command below lists them all.
            </p>
          ) : null}
        </>
      )}
      {/* The same check from a terminal: one line that scrolls, never cut with an ellipsis (ui/CommandLine). */}
      <CommandLine command={cli.get()} label="The dataset check command" copied="It prints every issue with its fix as JSON, and changes nothing." />
    </div>
  );
}

css`
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
`;
