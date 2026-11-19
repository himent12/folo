/**
 * One self-improvement loop (`#/benchmarks/{id}/loops/{loopId}`; docs/BENCH.md, Self-improvement loops): a page of its
 * own, since a loop has a chart, a verdict, each generation's provenance and actions, and a link to it is shared. It
 * states the question it answers in one line; draws mAP@0.5:0.95 per generation with each one's 95% interval as
 * whiskers, generation 0's interval as a band across (a generation above the band improved on the first teacher);
 * reads the verdict in words, with BENCH.md's caveat one tap away; and lists per generation what happened: the
 * teacher's labels, the dataset written from them, the training run, the model and its entry, each a link where the
 * thing exists. A running loop shows the step its current generation is on, live; a failed or stopped one why.
 */
import ArrowLeft from '$icons/arrow-left';
import Check from '$icons/check';
import CircleSlash from '$icons/circle-slash';
import Repeat from '$icons/repeat';
import Square from '$icons/square';
import Trash from '$icons/trash-2';
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, deep, effect, For, MediaQuery, signal, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { LiveBench } from '$lib/api/liveBench';
import { Resource } from '$lib/api/resource';
import type { BenchGeneration } from '$lib/api/types';
import { isLoopActive } from '$lib/bench/rules';
import { fitFloorNote } from '$lib/charts/domain';
import XYChart from '$lib/charts/XYChart';
import { media } from '$lib/design/tokens';
import { ago, date, metric, num, sentence } from '$lib/format';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import { pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import Button from '$lib/ui/Button';
import CodeText from '$lib/ui/CodeText';
import Dialog from '$lib/ui/Dialog';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import Ident from '$lib/ui/Ident';
import InfoButton from '$lib/ui/InfoButton';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import { toast } from '$lib/ui/toast';
import Callout from './run/Callout';
import { generationDomain, generationPoints } from './bench/charts';
import { cannotBench } from './bench/support';
import Unsupported from './bench/Unsupported';
import { hintText, intervalText, labelsLine, LOOP_CAVEAT, loopQuestion, loopStatus, loopSteps, loopVerdictText, progressFacts, scoreText, VERDICT } from './bench/words';
import { tickingNow } from './runs/when';

const PLOT = 220;
const SPLIT_NAME = { train: 'training', val: 'validation', test: 'test' } as const;
const genWord = (g: number) => (g === 0 ? 'Teacher' : `Gen ${g}`);
const known = <T extends { id: string }>(list: readonly T[] | undefined, x: string | null) => !!x && !!list?.some((y) => y.id === x);
const caveatWord = () => <>What it does not show</>;

export default function BenchLoop() {
  // Relative times ("made 3m ago") tick while the page is open (runs/when).
  const clock = tickingNow();
  const id = computed(() => router.route.params.id ?? '');
  const loopId = computed(() => router.route.params.loopId ?? '');
  const phone = new MediaQuery(media.phone, false);
  const system = new Resource((s) => hub.api.system(s));
  const models = new Resource((s) => hub.api.models(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));
  const runs = new Resource((s) => hub.api.runs(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const live = new LiveBench(() => id.get() || null);
  const bench = () => live.bench;
  const loop = computed(() => live.loops.find((l) => l.id === loopId.get()));
  const unsupported = computed(() => cannotBench(system.data, live.error));

  effect(() => {
    setCrumbs(
      { label: 'Benchmarks', path: '/benchmarks' },
      { label: bench()?.name ?? id.get(), path: `/benchmarks/${id.get()}?tab=loops` },
      { label: loop.get()?.name ?? loopId.get() },
    );
  });

  const planned = computed(() => loop.get()?.config.generations ?? 1);
  const teacherName = computed(() => {
    const l = loop.get();
    return l?.generations[0]?.model_name ?? models.data?.find((m) => m.id === l?.config.teacher)?.name ?? l?.config.teacher ?? '';
  });
  const poolName = computed(() => {
    const p = loop.get()?.config.pool;
    if (!p) return '';
    if (p.kind === 'dataset') {
      const d = datasets.data?.find((x) => x.id === p.dataset_id)?.name ?? p.dataset_id;
      return `the ${SPLIT_NAME[p.split ?? 'train']} split of ${d}`;
    }
    return `the images of ${jobs.data?.find((j) => j.id === p.job_id)?.name ?? 'a labeling job'}`;
  });
  const student = computed(() => loop.get()?.config.student?.arch ?? null);
  // A live loop makes datasets, runs and models the lists loaded at mount do not have yet: a loop event naming an id
  // a list does not know refetches that list (once per id), and only an id still missing from a list fetched after it
  // appeared is called deleted.
  const asked = deep<Record<string, true>>({});
  effect(() => {
    const l = loop.get();
    if (!l) return;
    const want: [Resource<{ id: string }[]>, string | null][] = l.generations.flatMap(
      (g) =>
        [
          [datasets, g.labels?.dataset_id ?? null],
          [runs, g.run_id],
          [models, g.model_id],
        ] as unknown as [Resource<{ id: string }[]>, string | null][],
    );
    for (const [r, x] of want) {
      // `asked` is written here: read without tracking, so the write never re-runs this.
      if (!x || !r.data || untrack(() => asked[x]) || known(r.data, x)) continue;
      asked[x] = true;
      void r.refresh();
    }
  });
  /** Whether an id is gone: asked for after it appeared, and still not in a list that is no longer loading. */
  const gone = <T extends { id: string }>(r: Resource<T[]>, x: string | null) => !!x && !!r.data && !r.loading && !r.refreshing && !!asked[x] && !known(r.data, x);
  const entryKnown = (x: string | null) => !!x && live.entries.some((e) => e.id === x);

  // ---- the chart ----
  const fit = computed(() => {
    const l = loop.get();
    return l ? generationDomain(l.generations, PLOT) : null;
  });
  const first = computed(() => loop.get()?.generations[0]);
  const cursor = signal<number | null>(null);
  const genAt = (g: number) => loop.get()?.generations.find((x) => x.generation === g);

  // ---- running: the steps of the generation it is on, each done only once what it makes is there (./bench/words) ----
  const current = computed(() => {
    const l = loop.get();
    return l && isLoopActive(l) ? l.generations.at(-1) : undefined;
  });
  const now = computed(() => {
    const l = loop.get();
    return l ? loopSteps(l) : null;
  });
  const nowPhase = computed(() => now.get()?.steps.find((s) => s.state === 'now')?.phase ?? null);
  /** Nothing graded yet (generation 0 still benchmarking): the chart says so instead of standing empty. */
  const graded = computed(() => !!loop.get()?.generations.some((g) => g.map50_95 !== null));

  // ---- stop, delete ----
  const stopping = signal(false);
  const stopBusy = signal(false);
  const stopStatus = new DialogStatus();
  async function stop() {
    const l = loop.peek();
    if (!l) return;
    stopBusy.set(true);
    const mark = live.mark();
    try {
      live.putLoop(await hub.api.stopBenchLoop(id.peek(), l.id), mark);
      stopping.set(false);
    } catch (e) {
      stopStatus.error(e, 'Could not stop the loop');
    } finally {
      stopBusy.set(false);
    }
  }
  const removing = signal(false);
  const removeBusy = signal(false);
  const removeStatus = new DialogStatus();
  async function remove() {
    const l = loop.peek();
    if (!l) return;
    const bid = id.peek();
    removeBusy.set(true);
    try {
      await hub.api.deleteBenchLoop(bid, l.id);
      live.dropLoop(l.id);
      removing.set(false);
      // page-level: its confirm dialog has closed on the line above.
      toast.success(`Deleted ${l.name}`, 'Its datasets, runs, models and entries stay.');
      router.go(`/benchmarks/${bid}?tab=loops`, { replace: true });
    } catch (e) {
      removeStatus.error(e, 'Could not delete the loop');
    } finally {
      removeBusy.set(false);
    }
  }
  const menu = computed<MenuItem[]>(() => {
    const l = loop.get();
    return l && !isLoopActive(l) ? [{ label: 'Delete loop', icon: Trash, danger: true, onclick: () => removing.set(true) }] : [];
  });
  pageCommands((): PageCommand[] => {
    const l = loop.get();
    if (!l) return [];
    const bid = id.get();
    const out: PageCommand[] = [{ id: 'bench-loop-back', label: 'Back to the loops', icon: ArrowLeft, run: () => router.go(`/benchmarks/${bid}?tab=loops`) }];
    if (isLoopActive(l) && l.status !== 'stopping') out.push({ id: 'bench-loop-stop', label: `Stop loop ${l.name}`, icon: Square, run: () => stopping.set(true) });
    return out;
  });
  /** The entry of the generation being graded, for where it is ("Waiting for the GPU", "Haze, severity 3 · 54%"). */
  const nowEntry = computed(() => {
    const c = current.get();
    return c?.entry_id ? live.entries.find((e) => e.id === c.entry_id) : undefined;
  });
  /** Why a generation has no labels or model while the loop no longer runs. */
  const ended = computed(() => {
    const l = loop.get();
    return l ? (l.status === 'failed' ? 'failed' : l.status === 'stopped' ? 'stopped' : null) : null;
  });

  const readout = (c: () => number | null, hint: () => string | null) => (
    <>
      {hint() ? (
        <span class="hint">{hint()}</span>
      ) : c() === null && !graded.get() ? (
        <span class="hint">No generation graded yet: the first point is the teacher’s, once its benchmark entry is done.</span>
      ) : c() === null ? (
        <>
          <span class="key">
            <i class="dot" aria-hidden="true"></i>mAP50-95 per generation, with its 95% interval
          </span>{' '}
          {first.get()?.ci95 ? (
            <span class="key gap">
              <i class="bandk" aria-hidden="true"></i>the first teacher’s interval
            </span>
          ) : null}
        </>
      ) : (
        readoutAt(c)
      )}
    </>
  );
  const readoutAt = (c: () => number | null) => {
    const g = () => genAt(c()!);
    return (
      <>
        <span class="t">{c() === 0 ? 'Generation 0, the first teacher' : `Generation ${c()}`}</span>
        <span class="sep">·</span>{' '}
        {g()?.map50_95 != null ? (
          <>
            <span class="v">mAP50-95 {metric(g()!.map50_95!)}</span>
            {g()!.ci95 ? (
              <>
                <span class="sep">·</span>
                <span>{intervalText(g()!.ci95!)}</span>
              </>
            ) : null}
            {g()!.score !== null ? (
              <>
                <span class="sep">·</span>
                <span>score {scoreText(g()!.score!)}</span>
              </>
            ) : null}
          </>
        ) : (
          <span>{g() ? (ended.get() ? 'not benchmarked' : 'not benchmarked yet') : ended.get() ? 'not run' : 'to come'}</span>
        )}
      </>
    );
  };

  const labelsCell = (g: () => BenchGeneration) => (
    <>
      {g().generation === 0 ? (
        <span class="faint">Its own training labels</span>
      ) : g().labels ? (
        labelsLineCell(g)
      ) : ended.get() && g() === loop.get()?.generations.at(-1) ? (
        <span class="faint">None: the loop {ended.get()} here</span>
      ) : current.get() === g() && nowPhase.get() === 'labeling' ? (
        <NoValue label="being labeled now" />
      ) : (
        <NoValue label="not labeled yet" />
      )}
    </>
  );
  // One plain line, so a generation that kept little is seen at a glance; a thin one says so in the warning ink.
  const labelsLineCell = (g: () => BenchGeneration) => {
    const ln = computed(() => labelsLine(g().labels!));
    return (
      <span class="ln" class:thin={ln.get().thin}>
        {ln.get().thin ? <TriangleAlert size={13} strokeWidth={1.75} aria-hidden="true" /> : null}
        <span>
          <span class="cap-first">{ln.get().text}</span>
          {ln.get().thin ? <span class="sr-only">: a thin dataset</span> : null}
        </span>
      </span>
    );
  };

  const nowLinks = (cur: () => BenchGeneration, l: () => NonNullable<ReturnType<typeof loop.get>>) => (
    <p class="now-links">
      {nowPhase.get() === 'training' ? (
        <>
          {cur().run_id ? (
            <>
              Training run:{' '}
              {known(runs.data, cur().run_id) ? (
                <a class="link" href={href(`/runs/${cur().run_id}`)}>
                  {l().name} gen {cur().generation}
                </a>
              ) : (
                <Ident value={cur().run_id!} />
              )}
              .
            </>
          ) : (
            'Its training run is being made.'
          )}
        </>
      ) : null}
      {' '}
      {nowPhase.get() === 'benchmarking' && cur().entry_id && nowEntry.get() ? entryNow(cur) : null}
      {' '}
      {nowPhase.get() === 'labeling' ? <>The teacher labels the pool by itself; nothing it labels is checked against truth.</> : null}
    </p>
  );
  const entryNow = (cur: () => BenchGeneration) => {
    const p = computed(() => progressFacts(nowEntry.get()!, catalog.data));
    return (
      <>
        Its entry:{' '}
        <a class="link" href={href(`/benchmarks/${bench()?.id}/entries/${cur().entry_id}`)}>
          {nowEntry.get()!.name}
        </a>
        , {(p.get().where[0] ?? '').toLowerCase() + p.get().where.slice(1)}
        {p.get().pct && nowEntry.get()!.status === 'running' ? (
          <>
            {' '}· <span class="num">{p.get().pct}</span>
          </>
        ) : null}
        .
      </>
    );
  };

  const genRow = (k: () => number) => {
    const g = () => genAt(k());
    // A missing score or mAP: still to come while the loop runs, never once it has failed or stopped.
    const noResult = () => (ended.get() ? (g() ? 'not benchmarked' : 'not run') : 'not yet');
    return (
      <tr class:future={!g()}>
        <td class="lead num">{k() === 0 ? 'Teacher' : `Gen ${k()}`}</td>
        <td class="mdl">
          {g()?.model_name ? (
            <>
              {g()!.entry_id && entryKnown(g()!.entry_id) ? (
                <a class="link" href={href(`/benchmarks/${bench()?.id}/entries/${g()!.entry_id}`)}>
                  {g()!.model_name}
                </a>
              ) : (
                g()!.model_name
              )}{' '}
              {g()!.model_id && known(models.data, g()!.model_id) ? (
                <span class="sub">
                  <a class="quiet" href={href(`/models/${g()!.model_id}`)}>
                    model
                  </a>
                </span>
              ) : null}
            </>
          ) : g() && ended.get() ? (
            <span class="faint">Not trained</span>
          ) : g() ? (
            <NoValue label={current.get() === g() && nowPhase.get() === 'training' ? 'training now' : 'trains once labeled'} />
          ) : (
            <span class="faint">{ended.get() ? 'Not run' : 'to come'}</span>
          )}{' '}
          {g()?.labels ? phoneLabels(g as () => BenchGeneration) : null}
        </td>
        <td class="hide-sm lbl">{g() ? labelsCell(g as () => BenchGeneration) : <span class="faint">—</span>}</td>
        <td class="hide-md">{g()?.labels ? madeOf(g as () => BenchGeneration) : <span class="faint">{k() === 0 ? 'none' : '—'}</span>}</td>
        <td class="r num hide-sm">{g()?.score != null ? scoreText(g()!.score!) : <NoValue label={noResult()} />}</td>
        <td class="r num">
          {g()?.map50_95 != null ? (
            <>
              <span class="m">{metric(g()!.map50_95!)}</span>
              {g()!.ci95 ? <span class="ci">{intervalText(g()!.ci95!)}</span> : null}
            </>
          ) : (
            <NoValue label={noResult()} />
          )}
        </td>
      </tr>
    );
  };
  // Phones, where the labels column steps out: the same one line under the model.
  const phoneLabels = (g: () => BenchGeneration) => {
    const ln = computed(() => labelsLine(g().labels!));
    return (
      <span class="sub ph-lbl" class:thin={ln.get().thin}>
        <span class="cap-first">{ln.get().text}</span>
      </span>
    );
  };
  const madeOf = (g: () => BenchGeneration) => (
    <Meta>
      <span>
        {known(datasets.data, g().labels!.dataset_id) ? (
          <a class="link" href={href(`/datasets/${g().labels!.dataset_id}`)}>
            dataset
          </a>
        ) : gone(datasets, g().labels!.dataset_id) ? (
          <span class="faint" title="No longer on this server">
            dataset deleted
          </span>
        ) : (
          <span class="faint">dataset</span>
        )}
      </span>{' '}
      {g().run_id ? (
        <span>
          {known(runs.data, g().run_id) ? (
            <a class="link" href={href(`/runs/${g().run_id}`)}>
              run
            </a>
          ) : gone(runs, g().run_id) ? (
            <span class="faint" title="No longer on this server">
              run deleted
            </span>
          ) : (
            <span class="faint">run</span>
          )}
        </span>
      ) : null}
    </Meta>
  );

  const generationsList = computed(() => Array.from({ length: planned.get() + 1 }, (_, g) => g));

  const loaded = (l: () => NonNullable<ReturnType<typeof loop.get>>) => (
    <>
      <Page
        title={l().name}
        inlineActions
        sub={() => (
          <Meta>
            <span>
              <StatusBadge plain state={loopStatus(l())} />
            </span>
            <span>
              teacher{' '}
              {known(models.data, l().config.teacher) ? (
                <a class="link" href={href(`/models/${l().config.teacher}`)}>
                  {teacherName.get()}
                </a>
              ) : (
                teacherName.get()
              )}
            </span>
            {student.get() ? (
              <span>
                students <Ident value={student.get()!} />
              </span>
            ) : null}
            <span class="num">
              {planned.get()} {planned.get() === 1 ? 'generation' : 'generations'} of {num(l().config.student?.epochs ?? 30)} epochs
            </span>
            <span class="faint" title={date(l().created)}>
              started {ago(l().created, clock.get())}
            </span>
          </Meta>
        )}
        subPhone={() => (
          <span>
            {loopStatus(l()).label} · {planned.get()} {planned.get() === 1 ? 'generation' : 'generations'}
          </span>
        )}
        actions={() => (
          <>
            {!phone.current ? (
              <Button icon={ArrowLeft} href={href(`/benchmarks/${bench()?.id}?tab=loops`)}>
                Loops
              </Button>
            ) : null}
            {isLoopActive(l()) ? (
              <>
                {phone.current ? (
                  <Button
                    variant="danger"
                    icon={Square}
                    disabled={l().status === 'stopping'}
                    onclick={() => stopping.set(true)}
                    aria-label={`Stop ${l().name}`}
                  />
                ) : (
                  <Button variant="danger" icon={Square} disabled={l().status === 'stopping'} onclick={() => stopping.set(true)}>
                    {l().status === 'stopping' ? 'Stopping' : 'Stop loop'}
                  </Button>
                )}
              </>
            ) : null}
            {menu.get().length ? <Menu label="More loop actions" items={menu.get()} /> : null}
          </>
        )}
      >
        <p class="question">{loopQuestion(teacherName.get(), poolName.get(), student.get(), planned.get())}</p>{' '}
        {l().status === 'failed' ? (
          <Callout tone="danger" icon={TriangleAlert} title={`It failed at generation ${l().generations.at(-1)?.generation ?? 0}`} role="alert">
            <CodeText text={sentence(l().error ?? 'The loop failed.')} />
            {hintText(l().hint) ? (
              <>
                {' '}
                <CodeText text={hintText(l().hint)!} />
              </>
            ) : null}
          </Callout>
        ) : l().status === 'stopped' ? (
          <Callout tone="neutral" icon={CircleSlash} title={`Stopped at generation ${l().generations.at(-1)?.generation ?? 0}`} role="note">
            The generations it finished are kept below; it has no verdict.
          </Callout>
        ) : null}{' '}
        <div class="top" class:running={!!current.get()}>
          <Panel title="mAP50-95 by generation" meta={fit.get() ? (fitFloorNote(fit.get()!) ?? 'unchanged images') : undefined}>
            {fit.get() ? (
              <XYChart
                label="mAP50-95 of each generation with its 95% interval; the first teacher's interval as a band"
                series={[{ id: 'map', label: 'mAP50-95', color: 'var(--text-primary)', points: generationPoints(l().generations), dots: true }]}
                xDomain={[-0.4, planned.get() + 0.4]}
                yDomain={fit.get()!.domain}
                xTicks={generationsList.get()}
                yTicks={fit.get()!.ticks}
                xFormat={genWord}
                yFormat={(v) => v.toFixed(fit.get()!.decimals)}
                height={PLOT + 40}
                band={first.get()?.ci95 ? { lo: first.get()!.ci95![0], hi: first.get()!.ci95![1], label: 'first teacher’s 95% interval' } : null}
                stops={generationsList.get()}
                bind:cursor={cursor}
                readout={readout}
                table={{
                  caption: 'mAP50-95 and score per generation',
                  head: ['Generation', 'mAP50-95', '95% interval', 'Score'],
                  rows: generationsList.get().map((g) => {
                    const x = genAt(g);
                    return [genWord(g), x?.map50_95 != null ? metric(x.map50_95) : '—', x?.ci95 ? intervalText(x.ci95) : '—', x?.score != null ? scoreText(x.score) : '—'];
                  }),
                }}
              />
            ) : null}
          </Panel>

          <div class="side">
            {current.get() && now.get() ? (
              // Generation 0 is one step (the benchmark grades the first teacher); a student's generation three. A step
              // reads done only once what it makes is in the generation (./bench/words loopSteps).
              <Panel title="Now" meta={`generation ${current.get()!.generation} of ${planned.get()}`}>
                <ol class="steps" aria-label={`The steps of generation ${current.get()!.generation}`}>
                  <For each={now.get()!.steps} key={(s) => s.phase}>
                    {(s) => (
                      <li class={s().state}>
                        <span class="mark" aria-hidden="true">
                          {s().state === 'done' ? <Check size={12} strokeWidth={2} /> : null}
                        </span>
                        <span class="s-name">{s().name}</span>
                        <span class="s-state">
                          {s().state === 'done' ? 'done' : s().state === 'now' ? (l().status === 'stopping' ? 'stopping' : 'now') : 'next'}
                        </span>
                      </li>
                    )}
                  </For>
                </ol>
                {nowLinks(() => current.get()!, l)}
              </Panel>
            ) : null}{' '}
            <Panel title="Verdict" actions={() => <InfoButton label="What a loop does not show" trigger={caveatWord} text={LOOP_CAVEAT} />}>
              {l().verdict ? (
                <>
                  <p class="v-word">{VERDICT[l().verdict!].word}</p>{' '}
                  {/* The word is the heading: the sentence goes on from it. */}
                  <p class="v-text">{(loopVerdictText(l()) ?? '').replace(/^[^:]+: (\w)/, (_, c: string) => c.toUpperCase())}</p>
                </>
              ) : (
                <p class="v-text faint">
                  {isLoopActive(l())
                    ? `Once the last generation is benchmarked: above the first teacher’s interval${first.get()?.ci95 ? ` (${intervalText(first.get()!.ci95!)})` : ''} is improved, inside it held, below it regressed.`
                    : 'It did not finish, so it has no verdict.'}
                </p>
              )}{' '}
              <p class="v-how">Read on mAP50-95 of the unchanged images; a generation inside the band is not clearly different from the first teacher.</p>
            </Panel>
          </div>
        </div>{' '}
        <Panel title="Generations" meta="what each one was made of" pad="none">
          <div class="scroll">
            <table class="table gens" aria-label={`Generations of ${l().name}`}>
              <thead>
                <tr>
                  <th class="lead w-gen">Generation</th>
                  <th>Model</th>
                  <th class="hide-sm">Labels it trained on</th>
                  <th class="hide-md w-ds">Dataset · run</th>
                  <th class="r hide-sm w-score">Score</th>
                  <th class="r w-map">
                    mAP50-95<span class="ci-h">{' · interval'}</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                <For each={generationsList.get()} key={(k) => k}>
                  {(k) => genRow(k)}
                </For>
              </tbody>
            </table>
          </div>
        </Panel>{' '}
        <p class="list-foot">
          Its datasets are named “{l().name} gen 1”, …: they, its runs, models and entries stay when the loop is deleted. The pool’s own labels were never read.
        </p>
      </Page>{' '}
      <Dialog
        open={stopping.get()}
        onclose={() => stopping.set(false)}
        title={`Stop ${l().name}?`}
        description="It stops what the loop is doing now (its entry or its run). The generations it finished, and their datasets, runs and models, stay."
        size="sm"
        status={stopStatus}
        footer={() => (
          <>
            <Button onclick={() => stopping.set(false)}>Keep running</Button>
            <Button variant="destructive" icon={Square} loading={stopBusy.get()} onclick={stop}>
              Stop loop
            </Button>
          </>
        )}
      />{' '}
      <Dialog
        open={removing.get()}
        onclose={() => removing.set(false)}
        title={`Delete ${l().name}?`}
        description="Removes the loop from this benchmark. Its datasets, runs, models and entries stay."
        size="sm"
        status={removeStatus}
        footer={() => (
          <>
            <Button onclick={() => removing.set(false)}>Cancel</Button>
            <Button variant="destructive" icon={Trash} loading={removeBusy.get()} onclick={remove}>
              Delete loop
            </Button>
          </>
        )}
      />
    </>
  );

  return (
    <>
      {unsupported.get() ? (
        <Page title="Benchmarks">
          <Unsupported />
        </Page>
      ) : live.error && !bench() ? (
        <Page
          title="Loop"
          actions={() => (
            <Button icon={ArrowLeft} href={href('/benchmarks')}>
              All benchmarks
            </Button>
          )}
        >
          {live.error instanceof ApiError && live.error.status === 404 ? (
            <ErrorState error={live.error} title="Benchmark not found" />
          ) : (
            <ErrorState error={live.error} title="Could not load this loop" retry={() => live.retry()} />
          )}
        </Page>
      ) : !bench() || live.loading ? (
        <div class="sk-page" aria-busy="true" aria-label="Loading loop">
          <div class="sk-head">
            <Skeleton w="min(240px, 60%)" h="24px" />
            <Skeleton w="min(520px, 90%)" h="14px" />
          </div>
          <div class="sk-grid">
            <Skeleton h="320px" r="var(--radius-lg)" />
            <Skeleton h="320px" r="var(--radius-lg)" />
          </div>
          <Skeleton h="200px" r="var(--radius-lg)" />
        </div>
      ) : !loop.get() ? (
        <Page
          title="Loop"
          actions={() => (
            <Button icon={ArrowLeft} href={href(`/benchmarks/${bench()?.id}?tab=loops`)}>
              Loops
            </Button>
          )}
        >
          <EmptyState icon={Repeat} title="Loop not found" description={`${bench()!.name} has no loop ${loopId.get()}: it may have been deleted.`} />
        </Page>
      ) : (
        loaded(() => loop.get()!)
      )}
    </>
  );
}

css`
  .question {
    margin-top: calc(-1 * var(--space-2));
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure-dialog);
  }
  .link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .quiet {
    color: var(--text-tertiary);
    text-decoration: underline;
    text-decoration-color: var(--line-default);
    text-underline-offset: 3px;
  }
  /* The chart on 8 of the page's 12 columns, what is happening and the verdict on 4. */
  .top {
    display: grid;
    grid-template-columns: minmax(0, 8fr) minmax(0, 4fr);
    gap: var(--space-4);
    align-items: start;
  }
  .side {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .steps {
    display: grid;
  }
  .steps li {
    display: grid;
    grid-template-columns: 20px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    min-height: 36px;
    border-top: 1px solid var(--line-subtle);
  }
  .steps li:first-child {
    border-top: 0;
  }
  .mark {
    display: grid;
    place-items: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    box-shadow: inset 0 0 0 1px var(--line-strong);
    color: var(--text-primary);
  }
  .steps li.done .mark {
    background: var(--bg-active);
    box-shadow: none;
  }
  /* The step running now: the live mark (lime is "now"), its name in ink. */
  .steps li.now .mark {
    box-shadow: inset 0 0 0 4px var(--accent-line);
  }
  .steps li.todo .s-name {
    color: var(--text-tertiary);
  }
  .s-state {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .steps li.now .s-state {
    color: var(--text-secondary);
  }
  .now-links {
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .now-links:empty {
    display: none;
  }
  .v-word {
    font-size: var(--type-xl-size);
    line-height: var(--type-xl-line);
    font-weight: var(--weight-semibold);
  }
  .v-text {
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .v-how {
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    max-inline-size: var(--measure);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .key.gap {
    margin-left: var(--space-3);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-primary);
  }
  .bandk {
    width: 14px;
    height: 10px;
    border-radius: 2px;
    background: var(--bg-active);
    box-shadow: inset 0 0 0 1px var(--line-default);
  }
  .sep {
    margin: 0 var(--space-2);
    color: var(--text-tertiary);
  }
  .t {
    color: var(--text-tertiary);
  }
  .v {
    color: var(--text-primary);
  }
  .hint {
    color: var(--text-secondary);
  }
  .scroll {
    overflow-x: auto;
  }
  .gens .lbl {
    white-space: normal;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .ln {
    display: inline-flex;
    gap: 6px;
    align-items: baseline;
  }
  .ln :global(svg) {
    flex-shrink: 0;
    align-self: center;
    color: var(--warning-solid);
  }
  .ln.thin {
    color: var(--text-primary);
  }
  .cap-first {
    display: inline-block;
  }
  .cap-first::first-letter {
    text-transform: uppercase;
  }
  /* A student's name is long ("shapes-n-best loop gen 1-best"): it wraps, at its spaces and hyphens, never cut. */
  .gens .mdl {
    white-space: normal;
    overflow-wrap: anywhere;
  }
  .gens tr.future td {
    color: var(--text-tertiary);
  }
  .m {
    font-weight: var(--weight-medium);
  }
  .w-gen {
    width: 104px;
  }
  .w-ds {
    width: 200px;
  }
  .w-score {
    width: 72px;
  }
  .w-map {
    width: 184px;
  }
  .ci {
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-5);
  }
  .sk-head {
    display: grid;
    gap: var(--space-2);
  }
  .sk-grid {
    display: grid;
    grid-template-columns: minmax(0, 8fr) minmax(0, 4fr);
    gap: var(--space-4);
  }
  @media (max-width: 1023px) {
    .top,
    .sk-grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .ph-lbl {
    display: none;
  }
  @media (max-width: 767px) {
    .ph-lbl {
      display: block;
      white-space: normal;
      color: var(--text-secondary);
    }
    .ph-lbl.thin {
      color: var(--warning-text);
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .ci,
    .ci-h {
      display: none;
    }
    .w-gen {
      width: 88px;
    }
    .w-map {
      width: 88px;
    }
  }
  /* A running loop on a tablet or a phone: what it does now before the chart. */
  @media (max-width: 1023px) {
    .top.running .side {
      order: -1;
    }
  }
`;
