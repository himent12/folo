/**
 * The other work running on the server, under the Training card: a labeling job whose agent is labeling, a benchmark
 * with an entry or a loop queued or running. Otherwise nothing outside their own pages shows them, and Overview would
 * say "Nothing is training" while a paid agent run or a benchmark entry goes on (labelbench F14). Renders nothing when
 * nothing runs, or when the server has neither feature (lib/label `LABEL_FEATURE`, lib/bench `BENCH_FEATURE`). A
 * benchmark says how far its entry is, as its leaderboard does ("folo-n-best · 46%"), like the labeling row's count.
 */
import ArrowRight from '$icons/arrow-right';
import { computed, css, For } from 'fw';
import { hub } from '$lib/api/index';
import { hasFeature } from '$lib/api/queue';
import { Resource } from '$lib/api/resource';
import type { Bench, BenchEntry, SystemInfo } from '$lib/api/types';
import { BENCH_FEATURE } from '$lib/bench/rules';
import { num } from '$lib/format';
import { LABEL_FEATURE } from '$lib/label';
import { href } from '$lib/router';
import { progressFacts } from '../bench/words';
import StatusBadge, { type OwnStatus } from '$lib/ui/StatusBadge';

/** The badge of something at work that is not a run: a run's running mark, its own word ("Labeling"). */
const atWork = (label: string): OwnStatus => ({ status: 'running', label });

interface Item {
  id: string;
  kind: string;
  name: string;
  detail: string;
  path: string;
}

export default function AlsoRunning(props: { system: SystemInfo | undefined }) {
  // Polled with the run list (the same period, so one timer): a job or entry that starts shows within seconds.
  const jobs = new Resource((s) => (hasFeature(props.system, LABEL_FEATURE) ? hub.api.labelJobs(s) : Promise.resolve([])), { poll: 5000 });
  const benches = new Resource((s) => (hasFeature(props.system, BENCH_FEATURE) ? hub.api.benches(s) : Promise.resolve([])), { poll: 5000 });
  // The entries of the benchmarks at work (seldom more than one), polled with them, for how far each is.
  const activeIds = computed(() => (benches.data ?? []).filter((b) => b.active).map((b) => b.id).join(' '));
  const entries = new Resource(
    async (s) => {
      const ids = activeIds.get().split(' ').filter(Boolean);
      const lists = await Promise.all(ids.map((id) => hub.api.benchEntries(id, s).catch((): BenchEntry[] => [])));
      return new Map(ids.map((id, i) => [id, lists[i]!]));
    },
    { poll: 5000 },
  );
  /** A benchmark's line: its entry at work and how far it is, else what its flag says. */
  const benchDetail = (b: Bench) => {
    const list = entries.data?.get(b.id) ?? [];
    const e = list.find((x) => x.status === 'running' || x.status === 'stopping') ?? list.find((x) => x.status === 'queued');
    if (!e) return 'An entry or a loop is running';
    const p = progressFacts(e);
    return `${e.name} · ${e.status === 'running' && p.pct ? p.pct : p.where}`;
  };

  const items = computed<Item[]>(() => [
    ...(jobs.data ?? [])
      .filter((j) => j.status !== 'idle')
      .map((j) => ({
        id: j.id,
        kind: 'Labeling',
        name: j.name,
        detail: j.run && !j.run.finished ? `${num(j.run.done)} of ${num(j.run.total)} images` : `${num(j.counts.labeled)} of ${num(j.counts.images)} images labeled`,
        path: `/datasets/label/${j.id}`,
      })),
    ...(benches.data ?? [])
      .filter((b) => b.active)
      .map((b) => ({ id: b.id, kind: 'Benchmark', name: b.name, detail: benchDetail(b), path: `/benchmarks/${b.id}` })),
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
                    <StatusBadge plain state={atWork(i().kind)} />{' '}
                    <span class="name truncate">{i().name}</span>{' '}
                    <span class="detail num truncate">{i().detail}</span>{' '}
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
