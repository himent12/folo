/**
 * The other work running on the server, under the Training card: a labeling job whose agent is labeling, a benchmark
 * with an entry queued or running, a loop at work. Otherwise nothing outside their own pages shows them, and Overview
 * would say "Nothing is training" while a paid agent run or a benchmark entry goes on (labelbench F14). Renders nothing
 * when nothing runs, or when the server has neither feature (lib/label `LABEL_FEATURE`, lib/bench `BENCH_FEATURE`). A
 * benchmark says how far its entry is, as its leaderboard does ("folo-n-best · 46%"), like the labeling row's count; a
 * loop is a row of its own, what it does now as its loops tab says it ("Training gen 1"), and opens the loop. A cloud
 * instance whose termination the provider has not confirmed may still be billed: it is here too (Settings › Cloud says
 * the same at length), so a user who never opens Settings learns of it.
 */
import ArrowRight from '$icons/arrow-right';
import { computed, css, For, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { hasFeature } from '$lib/api/queue';
import { Resource } from '$lib/api/resource';
import { abortError } from '$lib/api/share';
import type { Bench, BenchEntry, BenchLoop, CloudInstance, SystemInfo } from '$lib/api/types';
import { BENCH_FEATURE } from '$lib/bench/rules';
import { hasCloud } from '$lib/cloud';
import { count, num } from '$lib/format';
import { LABEL_FEATURE } from '$lib/label';
import { href } from '$lib/router';
import { loopStatus, progressFacts } from '../bench/words';
import StatusBadge, { type OwnStatus } from '$lib/ui/StatusBadge';

/** The badge of something at work that is not a run: a run's running mark, its own word ("Labeling"). */
const atWork = (label: string): OwnStatus => ({ status: 'running', label });

interface Item {
  id: string;
  kind: string;
  /** Its badge; the kind's running mark when absent. */
  state?: OwnStatus;
  name: string;
  detail: string;
  /** Non-truncating refresh failure notice shown beside `detail` when the latest poll failed. */
  refreshNote?: string;
  path: string;
}

interface BenchWork {
  entries: BenchEntry[];
  loops: BenchLoop[];
  entriesFailed: boolean;
  loopsFailed: boolean;
}

const isMissingOrUnsupported = (e: unknown) => e instanceof ApiError && (e.status === 404 || e.status === 501);

export default function AlsoRunning(props: { system: SystemInfo | undefined }) {
  // Polled with the run list (the same period, so one timer): a job or entry that starts shows within seconds.
  const jobs = new Resource((s) => (hasFeature(props.system, LABEL_FEATURE) ? hub.api.labelJobs(s) : Promise.resolve([])), { poll: 5000 });
  const benches = new Resource((s) => (hasFeature(props.system, BENCH_FEATURE) ? hub.api.benches(s) : Promise.resolve([])), { poll: 5000 });
  // Instances the server could not terminate yet (settings/Cloud `unconfirmed`): each may still be billed.
  const instances = new Resource((s) => (hasCloud(props.system) ? hub.api.cloudInstances(s) : Promise.resolve([] as CloudInstance[])), { poll: 5000 });
  // The entries and loops of the benchmarks at work (seldom more than one), polled with them, for how far each is.
  const activeIds = computed(() => (benches.data ?? []).filter((b) => b.active).map((b) => b.id).join(' '));
  const work = new Resource<Map<string, BenchWork>>(
    async (s) => {
      const ids = activeIds.get().split(' ').filter(Boolean);
      if (!ids.length) return new Map();
      const prev = untrack(() => work.data);
      const lists = await Promise.all(
        ids.map((id) => Promise.allSettled([hub.api.benchEntries(id, s), hub.api.benchLoops(id, s)])),
      );
      if (s.aborted) throw abortError();
      const next = new Map<string, BenchWork>();
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i]!;
        const prior = prev?.get(id);
        const [entriesRes, loopsRes] = lists[i]!;
        let entries: BenchEntry[];
        let entriesFailed = false;
        if (entriesRes.status === 'fulfilled') {
          entries = entriesRes.value;
        } else if (isMissingOrUnsupported(entriesRes.reason)) {
          entries = [];
        } else {
          entries = prior?.entries ?? [];
          entriesFailed = true;
        }
        let loops: BenchLoop[];
        let loopsFailed = false;
        if (loopsRes.status === 'fulfilled') {
          loops = loopsRes.value;
        } else if (isMissingOrUnsupported(loopsRes.reason)) {
          loops = [];
        } else {
          loops = prior?.loops ?? [];
          loopsFailed = true;
        }
        next.set(id, { entries, loops, entriesFailed, loopsFailed });
      }
      return next;
    },
    { poll: 5000 },
  );
  const atWorkEntry = (w: BenchWork | undefined) => {
    const list = w?.entries ?? [];
    return list.find((x) => x.status === 'running' || x.status === 'stopping') ?? list.find((x) => x.status === 'queued');
  };
  const loopsAtWork = (w: BenchWork | undefined) => (w?.loops ?? []).filter((l) => l.status === 'running' || l.status === 'stopping');
  /** A benchmark's line: its entry at work and how far it is, preserving last known detail and noting refresh failures. */
  const benchDetail = (w: BenchWork | undefined, e: BenchEntry | undefined, loops: BenchLoop[]): { detail: string; refreshNote?: string } => {
    const listStale = benches.stale !== undefined;
    if (e) {
      const p = progressFacts(e);
      const base = `${e.name} · ${e.status === 'running' && p.pct ? p.pct : p.where}`;
      if (w?.entriesFailed || listStale) return { detail: base, refreshNote: 'Refresh failed' };
      if (w?.loopsFailed && !loops.length) return { detail: base, refreshNote: 'Could not refresh loops' };
      return { detail: base };
    }
    if (w?.entriesFailed && w?.loopsFailed && !loops.length) return { detail: '', refreshNote: 'Could not refresh benchmark activity' };
    if (w?.entriesFailed) return { detail: '', refreshNote: 'Could not refresh benchmark entries' };
    if (w?.loopsFailed && !loops.length) return { detail: '', refreshNote: 'Could not refresh benchmark loops' };
    if (listStale) return { detail: 'An entry or a loop is running', refreshNote: 'Refresh failed' };
    return { detail: 'An entry or a loop is running' };
  };

  const items = computed<Item[]>(() => [
    ...(instances.data ?? [])
      .filter((i) => i.status === 'terminating' && i.error)
      .map((i) => ({
        id: i.id,
        kind: 'Cloud',
        state: { status: 'failed' as const, label: 'Cloud' },
        name: i.name,
        detail: 'Termination not confirmed: it may still be billed',
        refreshNote: instances.stale !== undefined ? 'Refresh failed' : undefined,
        path: '/settings?section=cloud',
      })),
    ...(jobs.data ?? [])
      .filter((j) => j.status !== 'idle')
      .map((j) => {
        const base = j.run && !j.run.finished ? `${num(j.run.done)} of ${count(j.run.total, 'image')}` : `${num(j.counts.labeled)} of ${count(j.counts.images, 'image')} labeled`;
        return {
          id: j.id,
          kind: 'Labeling',
          name: j.name,
          detail: base,
          refreshNote: jobs.stale !== undefined ? 'Refresh failed' : undefined,
          path: `/datasets/label/${j.id}`,
        };
      }),
    ...(benches.data ?? [])
      .filter((b) => b.active)
      .flatMap((b) => {
        const w = work.data?.get(b.id);
        const entry = atWorkEntry(w);
        const loops = loopsAtWork(w);
        const showBenchRow = !!entry || !loops.length || !!w?.entriesFailed;
        const loopFailed = !!w?.loopsFailed || benches.stale !== undefined;
        return [
          // A benchmark whose only work is a loop between its entries (labeling, training) is said by the loop's row,
          // unless refreshing entries failed and needs to be surfaced.
          ...(showBenchRow ? [{ id: b.id, kind: 'Benchmark', name: b.name, ...benchDetail(w, entry, loops), path: `/benchmarks/${b.id}` }] : []),
          ...loops.map((l) => ({
            id: l.id,
            kind: 'Loop',
            name: l.name,
            detail: `${loopStatus(l).label} · ${b.name}`,
            refreshNote: loopFailed ? 'Refresh failed' : undefined,
            path: `/benchmarks/${b.id}/loops/${l.id}`,
          })),
        ];
      }),
  ]);

  // Top-level poll failures whose resource has no visible row to carry the failure note.
  const hiddenStale = computed(() => {
    if (hub.offline) return [] as string[];
    const list = items.get();
    const out: string[] = [];
    if (instances.stale !== undefined && !list.some((i) => i.kind === 'Cloud')) out.push('cloud instances');
    if (jobs.stale !== undefined && !list.some((i) => i.kind === 'Labeling')) out.push('labeling jobs');
    if (benches.stale !== undefined && !list.some((i) => i.kind === 'Benchmark' || i.kind === 'Loop')) out.push('benchmarks');
    return out;
  });

  return (
    <>
      {items.get().length || hiddenStale.get().length ? (
        <section class="activity" aria-label="Also running">
          {items.get().length ? (
            <ul>
              <For each={items.get()} key={(i) => i.id}>
                {(i) => (
                  <li>
                    <a href={href(i().path)}>
                      <StatusBadge plain state={i().state ?? atWork(i().kind)} />{' '}
                      <span class="name truncate">{i().name}</span>{' '}
                      <span class="detail-cell">
                        {i().detail ? (
                          <span class="detail num truncate" title={i().detail}>
                            {i().detail}
                          </span>
                        ) : null}
                        {i().refreshNote ? (
                          <span class="refresh-note" data-fact data-refresh-failed="true">
                            {i().refreshNote}
                          </span>
                        ) : null}
                      </span>{' '}
                      <ArrowRight size={14} strokeWidth={1.75} />
                    </a>
                  </li>
                )}
              </For>
            </ul>
          ) : null}
          {hiddenStale.get().length ? (
            <p class="stale-notice" data-fact data-refresh-failed="true">
              Could not refresh {hiddenStale.get().join(', ')}
            </p>
          ) : null}
        </section>
      ) : null}
    </>
  );
}

css`
  .activity {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    padding: var(--space-1);
  }
  a {
    display: grid;
    grid-template-columns: 112px minmax(0, auto) minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-3);
    min-height: var(--size-row);
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    color: var(--text-tertiary);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  a:hover {
    background: var(--bg-hover);
  }
  .name {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .detail-cell {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    min-width: 0;
  }
  .detail {
    font-size: var(--type-sm-size);
    flex: 0 1 auto;
    min-width: 0;
  }
  .refresh-note {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    flex-shrink: 0;
    white-space: nowrap;
  }
  .stale-notice {
    margin: 0;
    padding: var(--space-2) var(--space-3);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* Phones: the kind and the name on one line, what it is doing under the name; a refresh failure wraps intact below
     the detail rather than clipping behind an ellipsis. */
  @media (max-width: 767px) {
    a {
      grid-template-columns: 104px minmax(0, 1fr) auto;
      row-gap: 0;
      padding: var(--space-2) var(--space-3);
    }
    .detail-cell {
      grid-column: 2;
      grid-row: 2;
      flex-wrap: wrap;
      gap: var(--space-1) var(--space-2);
    }
    .detail {
      max-width: 100%;
    }
    .refresh-note {
      white-space: normal;
      max-width: 100%;
    }
  }
`;
