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
  /** True when the latest refresh for this item's detail failed transiently. */
  refreshFailed?: boolean;
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
  const atWorkEntry = (b: Bench) => {
    const list = work.data?.get(b.id)?.entries ?? [];
    return list.find((x) => x.status === 'running' || x.status === 'stopping') ?? list.find((x) => x.status === 'queued');
  };
  const loopsAtWork = (b: Bench) => (work.data?.get(b.id)?.loops ?? []).filter((l) => l.status === 'running' || l.status === 'stopping');
  /** A benchmark's line: its entry at work and how far it is, preserving last known detail and noting refresh failures. */
  const benchDetail = (b: Bench): { detail: string; failed: boolean } => {
    const w = work.data?.get(b.id);
    const e = atWorkEntry(b);
    const listStale = benches.stale !== undefined;
    if (e) {
      const p = progressFacts(e);
      const base = `${e.name} · ${e.status === 'running' && p.pct ? p.pct : p.where}`;
      if (w?.entriesFailed) return { detail: `${base} · refresh failed`, failed: true };
      if (w?.loopsFailed && !loopsAtWork(b).length) return { detail: `${base} · could not refresh loops`, failed: true };
      if (listStale) return { detail: `${base} · refresh failed`, failed: true };
      return { detail: base, failed: false };
    }
    if (w?.entriesFailed && w?.loopsFailed) return { detail: 'Could not refresh benchmark activity', failed: true };
    if (w?.entriesFailed) return { detail: 'Could not refresh benchmark entries', failed: true };
    if (w?.loopsFailed) return { detail: 'Could not refresh benchmark loops', failed: true };
    if (listStale) return { detail: 'An entry or a loop is running · refresh failed', failed: true };
    return { detail: 'An entry or a loop is running', failed: false };
  };

  const items = computed<Item[]>(() => [
    ...(instances.data ?? [])
      .filter((i) => i.status === 'terminating' && i.error)
      .map((i) => ({
        id: i.id,
        kind: 'Cloud',
        state: { status: 'failed' as const, label: 'Cloud' },
        name: i.name,
        detail: instances.stale !== undefined ? 'Termination not confirmed: it may still be billed · refresh failed' : 'Termination not confirmed: it may still be billed',
        refreshFailed: instances.stale !== undefined,
        path: '/settings?section=cloud',
      })),
    ...(jobs.data ?? [])
      .filter((j) => j.status !== 'idle')
      .map((j) => {
        const base = j.run && !j.run.finished ? `${num(j.run.done)} of ${count(j.run.total, 'image')}` : `${num(j.counts.labeled)} of ${count(j.counts.images, 'image')} labeled`;
        const failed = jobs.stale !== undefined;
        return {
          id: j.id,
          kind: 'Labeling',
          name: j.name,
          detail: failed ? `${base} · refresh failed` : base,
          refreshFailed: failed,
          path: `/datasets/label/${j.id}`,
        };
      }),
    ...(benches.data ?? [])
      .filter((b) => b.active)
      .flatMap((b) => {
        const w = work.data?.get(b.id);
        const loops = loopsAtWork(b);
        const showBenchRow = !!atWorkEntry(b) || !loops.length || !!w?.entriesFailed;
        const bd = benchDetail(b);
        const loopFailed = !!w?.loopsFailed || benches.stale !== undefined;
        return [
          // A benchmark whose only work is a loop between its entries (labeling, training) is said by the loop's row,
          // unless refreshing entries failed and needs to be surfaced.
          ...(showBenchRow ? [{ id: b.id, kind: 'Benchmark', name: b.name, detail: bd.detail, refreshFailed: bd.failed, path: `/benchmarks/${b.id}` }] : []),
          ...loops.map((l) => ({
            id: l.id,
            kind: 'Loop',
            name: l.name,
            detail: loopFailed ? `${loopStatus(l).label} · ${b.name} · refresh failed` : `${loopStatus(l).label} · ${b.name}`,
            refreshFailed: loopFailed,
            path: `/benchmarks/${b.id}/loops/${l.id}`,
          })),
        ];
      }),
  ]);

  return (
    <>
      {items.get().length ? (
        <section class="activity" aria-label="Also running">
          <ul>
            <For each={items.get()} key={(i) => i.id}>
              {(i) => (
                <li>
                  <a href={href(i().path)}>
                    <StatusBadge plain state={i().state ?? atWork(i().kind)} />{' '}
                    <span class="name truncate">{i().name}</span>{' '}
                    <span class="detail num truncate" class:stale={!!i().refreshFailed} data-refresh-failed={i().refreshFailed ? 'true' : undefined}>
                      {i().detail}
                    </span>{' '}
                    <ArrowRight size={14} strokeWidth={1.75} />
                  </a>
                </li>
              )}
            </For>
          </ul>
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
  .detail {
    font-size: var(--type-sm-size);
  }
  .detail.stale {
    color: var(--text-secondary);
  }
  /* Phones: the kind and the name on one line, what it is doing under the name. */
  @media (max-width: 767px) {
    a {
      grid-template-columns: 104px minmax(0, 1fr) auto;
      row-gap: 0;
      padding: var(--space-2) var(--space-3);
    }
    .detail {
      grid-column: 2;
      grid-row: 2;
    }
  }
`;
