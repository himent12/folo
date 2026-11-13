/**
 * A benchmark's self-improvement loops (docs/BENCH.md, Self-improvement loops): each with its teacher and student,
 * where it stands (what it does for which generation while it runs; a failed one's cause under its name, so a loop
 * that starved for labels says so in the list), its generations' scores as a small line from the first teacher to the
 * last student, and its verdict. A row opens the loop's own page (`#/benchmarks/{id}/loops/{loopId}`, ../BenchLoop);
 * "New loop" starts one.
 */
import ChevronRight from '$icons/chevron-right';
import Plus from '$icons/plus';
import Repeat from '$icons/repeat';
import { computed, css, For } from 'fw';
import type { Bench, BenchEntry, BenchLoop, Dataset, LabelJob } from '$lib/api/types';
import Sparkline from '$lib/charts/Sparkline';
import { ago, date, sentence } from '$lib/format';
import CodeText from '$lib/ui/CodeText';
import { href, router } from '$lib/router';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import Gate from '$lib/ui/Gate';
import { OFFLINE_REASON } from '$lib/ui/offline';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import Panel from '$lib/ui/Panel';
import StatusBadge from '$lib/ui/StatusBadge';
import { loopStatus, scoreText, VERDICT } from './words';
import { tickingNow } from '../runs/when';

export interface LoopsTabProps {
  bench: Bench;
  loops: readonly BenchLoop[];
  entries: readonly BenchEntry[];
  datasets: readonly Dataset[] | undefined;
  jobs: readonly LabelJob[] | undefined;
  /** Opens "New loop". */
  onnew: () => void;
  /** The server does not answer: a loop cannot start. */
  offline?: boolean;
}

const scores = (l: BenchLoop) => l.generations.map((g) => g.score).filter((s): s is number => s !== null);

export default function LoopsTab(props: LoopsTabProps) {
  // Relative times ("made 3m ago") tick while the page is open (runs/when).
  const clock = tickingNow();
  const path = (l: BenchLoop) => `/benchmarks/${props.bench.id}/loops/${l.id}`;

  const pool = (l: BenchLoop) => {
    const p = l.config.pool;
    if (p.kind === 'dataset') return `${props.datasets?.find((d) => d.id === p.dataset_id)?.name ?? p.dataset_id} · ${p.split ?? 'train'}`;
    return props.jobs?.find((j) => j.id === p.job_id)?.name ?? 'a labeling job';
  };
  const teacherArch = (l: BenchLoop) => {
    const e = props.entries.find((x) => x.id === l.generations[0]?.entry_id);
    return e?.contestant.kind === 'model' ? e.contestant.arch : null;
  };

  return (
    <Panel
      title="Self-improvement loops"
      meta={props.loops.length ? String(props.loops.length) : undefined}
      pad="none"
      actions={() => (
        <>
          {props.loops.length ? (
            <Gate reason={props.offline ? OFFLINE_REASON : null}>
              {(why) => (
                <Button size="sm" icon={Plus} disabled={!!why()} aria-describedby={why()} onclick={props.onnew}>
                  New loop
                </Button>
              )}
            </Gate>
          ) : null}
        </>
      )}
    >
      {!props.loops.length ? (
        <EmptyState
          icon={Repeat}
          title="No loops yet"
          description="A loop tests whether a model's own labels can train a better one: it labels a pool of images by itself, a student trains on those labels alone, and this benchmark grades the student."
          actions={() => (
            <Gate reason={props.offline ? OFFLINE_REASON : null}>
              {(why) => (
                <Button icon={Plus} disabled={!!why()} aria-describedby={why()} onclick={props.onnew}>
                  New loop
                </Button>
              )}
            </Gate>
          )}
        />
      ) : (
        <>
          <div class="scroll">
            <table class="table tall loops" aria-label={`Self-improvement loops of ${props.bench.name}`}>
              <thead>
                <tr>
                  <th class="lead">Loop</th>
                  <th class="w-st">Status</th>
                  <th class="w-sc">Score by generation</th>
                  <th class="w-v">Verdict</th>
                  <th class="r hide-md" style:width="104px">
                    Started
                  </th>
                  <th style:width="40px">
                    <span class="sr-only">Open</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                <For each={props.loops} key={(l) => l.id}>
                  {(l) => {
                    const s = computed(() => scores(l()));
                    const arch = computed(() => teacherArch(l()));
                    return (
                      <tr class="link" aria-label={l().name} onclick={() => router.go(path(l()))}>
                        <td class="lead">
                          <a class="name" href={href(path(l()))} onclick={(e) => e.stopPropagation()}>
                            {l().name}
                          </a>{' '}
                          <span class="sub">
                            <Meta>
                              {arch.get() ? (
                                <span>
                                  teacher <Ident value={arch.get()!} />
                                </span>
                              ) : null}
                              {l().config.student?.arch ? (
                                <span>
                                  student <Ident value={l().config.student!.arch!} />
                                </span>
                              ) : null}
                              <span>pool {pool(l())}</span>
                            </Meta>
                          </span>{' '}
                          {l().status === 'failed' ? (
                            <span class="cause" data-fact>
                              <CodeText text={sentence(l().error ?? 'The loop failed.')} />
                            </span>
                          ) : null}
                        </td>
                        <td>
                          <StatusBadge plain state={loopStatus(l())} />
                        </td>
                        <td>
                          <span class="gens">
                            {s.get().length >= 2 ? <Sparkline values={s.get()} width={56} height={20} color="var(--text-secondary)" fill={false} /> : null}
                            <span class="num g-text">{s.get().length ? s.get().map(scoreText).join(' → ') : <NoValue label="no score yet" />}</span>
                          </span>
                        </td>
                        <td>
                          {l().verdict ? (
                            <span class="verdict" title={VERDICT[l().verdict!].means}>
                              {VERDICT[l().verdict!].word}
                            </span>
                          ) : (
                            <NoValue label="no verdict yet" />
                          )}
                        </td>
                        <td class="r faint hide-md" title={date(l().created)}>
                          {ago(l().created, clock.get())}
                        </td>
                        <td class="chev-cell">
                          <ChevronRight size={16} strokeWidth={1.75} aria-hidden="true" />
                        </td>
                      </tr>
                    );
                  }}
                </For>
              </tbody>
            </table>
          </div>
          {/* Phones: the name over its status and scores, the verdict on the right; a tap opens the loop. */}
          <ul class="rows" aria-label={`Self-improvement loops of ${props.bench.name}`}>
            <For each={props.loops} key={(l) => l.id}>
              {(l) => {
                const s = computed(() => scores(l()));
                const gens = () => l().config.generations ?? 1;
                return (
                  <li>
                    <a class="m-row" href={href(path(l()))}>
                      <span class="m-main">
                        <span class="name truncate">{l().name}</span>
                        <span class="m-meta num">
                          <Meta>
                            <span>{loopStatus(l()).label}</span>
                            {s.get().length > 1 ? (
                              <span>
                                {scoreText(s.get()[0]!)} → {scoreText(s.get().at(-1)!)}
                              </span>
                            ) : s.get().length ? (
                              <span>{scoreText(s.get()[0]!)}</span>
                            ) : null}
                            <span>
                              {gens()} {gens() === 1 ? 'generation' : 'generations'}
                            </span>
                          </Meta>
                        </span>
                        {l().status === 'failed' ? (
                          <span class="m-cause" data-fact>
                            <CodeText text={sentence(l().error ?? 'The loop failed.')} />
                          </span>
                        ) : null}
                      </span>
                      <span class="m-side">
                        {l().verdict ? <span class="verdict">{VERDICT[l().verdict!].word}</span> : <StatusBadge plain dotOnly state={loopStatus(l())} />}
                      </span>
                    </a>
                  </li>
                );
              }}
            </For>
          </ul>
        </>
      )}
    </Panel>
  );
}

css`
  .scroll {
    overflow-x: auto;
  }
  .name {
    display: block;
    color: var(--text-primary);
  }
  /* The loop's cell takes what the fixed columns leave, so a failed loop's cause wraps there instead of widening it. */
  .loops td.lead {
    max-width: 0;
    white-space: normal;
  }
  .cause {
    display: block;
    margin-top: 2px;
    white-space: normal;
    max-inline-size: var(--measure-dialog);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .w-st {
    width: 176px;
  }
  .w-sc {
    width: 232px;
  }
  .w-v {
    width: 120px;
  }
  /* Tablets: the fixed columns give the loop's own cell the room its facts need. */
  @media (max-width: 1023px) {
    .w-st {
      width: 160px;
    }
    .w-sc {
      width: 200px;
    }
    .w-v {
      width: 96px;
    }
  }
  .gens {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .g-text {
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
  }
  .verdict {
    font-weight: var(--weight-medium);
  }
  .chev-cell {
    padding-right: var(--space-3);
    color: var(--text-tertiary);
  }
  .rows {
    display: none;
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows li + li {
      border-top: 1px solid var(--line-subtle);
    }
    .m-row:active {
      background: var(--bg-hover);
    }
    .m-row {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      width: 100%;
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
      text-align: left;
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
    }
    .rows .name {
      font-weight: var(--weight-medium);
    }
    .m-cause {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .m-meta {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .m-side {
      flex-shrink: 0;
    }
  }
`;
