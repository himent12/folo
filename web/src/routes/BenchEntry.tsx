/**
 * One entry of a benchmark (`#/benchmarks/{id}/entries/{entryId}`): its whole result, read top to bottom as
 * docs/BENCH.md's "Reading it" asks (how good is it, why that number, where does it break, can I trust its confidence,
 * could it label for itself, what does it get wrong). The header says who it is and what it was graded on; the score
 * leads, then a section per part of the result, which a sticky strip under the top bar jumps between (the page is long;
 * on a phone the same strip scrolls sideways). Kept live by the benchmark's stream: a queued or running entry shows
 * where it is and lands on its result when it is done; a failed or stopped one says why and what to do.
 */
import ArrowLeft from '$icons/arrow-left';
import Box from '$icons/box';
import CircleSlash from '$icons/circle-slash';
import Copy from '$icons/copy';
import RotateCcw from '$icons/rotate-ccw';
import Square from '$icons/square';
import Target from '$icons/target';
import Trash from '$icons/trash-2';
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, effect, For, Index, MediaQuery, signal } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { LiveBench } from '$lib/api/liveBench';
import { Resource } from '$lib/api/resource';
import type { BenchEntry, BenchLoop, BenchResult } from '$lib/api/types';
import { isEntryActive } from '$lib/bench/rules';
import { media } from '$lib/design/tokens';
import { ago, date, num, sentence, usd } from '$lib/format';
import { AGENT_NAME } from '$lib/label';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import Button from '$lib/ui/Button';
import CodeText from '$lib/ui/CodeText';
import Dialog from '$lib/ui/Dialog';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import Ident from '$lib/ui/Ident';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import { toast } from '$lib/ui/toast';
import { tooltip } from '$lib/ui/tooltip';
import Callout from './run/Callout';
import AccuracySection from './bench/entry/AccuracySection';
import CalibrationSection from './bench/entry/CalibrationSection';
import ImagesSection from './bench/entry/ImagesSection';
import LocalizationSection from './bench/entry/LocalizationSection';
import RobustnessSection from './bench/entry/RobustnessSection';
import RunningPanel from './bench/entry/RunningPanel';
import ScoreSection from './bench/entry/ScoreSection';
import SelfLabelSection from './bench/entry/SelfLabelSection';
import JumpNav from './bench/JumpNav';
import { ranks, stale } from './bench/leaderboard';
import NewLoopDialog from './bench/NewLoopDialog';
import { modelLabel } from './label/words';
import { cannotBench } from './bench/support';
import Unsupported from './bench/Unsupported';
import { axisWhyNot, costFacts, coverageFacts, entryStatus, scoreText, SPEED_HELP, teacherEligibility } from './bench/words';

const SECTIONS = [
  { id: 'score', label: 'Score' },
  { id: 'accuracy', label: 'Accuracy' },
  { id: 'robustness', label: 'Robustness' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'self-labeling', label: 'Self-labeling' },
  { id: 'localization', label: 'Localization' },
  { id: 'images', label: 'Images' },
];
type ModelContestant = Extract<BenchEntry['contestant'], { kind: 'model' }>;
type AgentContestant = Extract<BenchEntry['contestant'], { kind: 'agent' }>;

const SK_JUMP = [56, 72, 88, 80, 96, 88, 56];
const SK_ROWS = [0, 1, 2, 3, 4];
const SK_SECTIONS = [420, 520, 360];

export default function BenchEntryPage() {
  const id = computed(() => router.route.params.id ?? '');
  const entryId = computed(() => router.route.params.entryId ?? '');
  const phone = new MediaQuery(media.phone, false);
  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const models = new Resource((s) => hub.api.models(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));
  const live = new LiveBench(() => id.get() || null);
  const bench = computed(() => live.bench);
  const entry = computed(() => live.entries.find((e) => e.id === entryId.get()));
  const unsupported = computed(() => cannotBench(system.data, live.error));
  // The result, once the entry is done: asked again when it lands (a finished result never changes).
  const done = computed(() => entry.get()?.status === 'done');
  const result = new Resource((s) => (done.get() ? hub.api.benchResult(id.get(), entryId.get(), s) : Promise.resolve(undefined)));

  effect(() => {
    setCrumbs(
      { label: 'Benchmarks', path: '/benchmarks' },
      { label: bench.get()?.name ?? id.get(), path: `/benchmarks/${id.get()}` },
      { label: entry.get()?.name ?? entryId.get() },
    );
  });

  const model = computed(() => {
    const e = entry.get();
    return e?.contestant.kind === 'model' ? models.data?.find((m) => m.id === (e.contestant as { model_id: string }).model_id) : undefined;
  });
  const rank = computed(() => {
    const e = entry.get();
    if (!e || !done.get()) return null;
    // Ranked among the entries graded on the benchmark as it is; one graded on an earlier version has no rank here.
    const r = ranks(live.entries.filter((x) => !stale(x, bench.get()?.fingerprint ?? x.fingerprint)));
    const at = r.get(e.id);
    return at ? { at, of: r.size } : null;
  });
  /** Model entries ahead of a queued one (they run one at a time, oldest first). */
  const ahead = computed(() => {
    const e = entry.get();
    return e?.status === 'queued'
      ? live.entries.filter((x) => x.contestant.kind === 'model' && x.id !== e.id && isEntryActive(x) && (x.status !== 'queued' || x.created < e.created))
      : [];
  });

  // ---- the next step: a loop with this model as the first teacher ----
  const loopOpen = signal(false);
  const loop = computed(() => {
    const e = entry.get();
    const b = bench.get();
    const m = model.get();
    if (!e || !b || e.contestant.kind !== 'model') return null;
    if (models.data && !m) return { why: 'Its model is no longer on this server, so it cannot teach.', start: () => {} };
    if (!m) return { why: 'Loading models', start: () => {} };
    const el = teacherEligibility(m, b);
    return { why: el.ok ? null : `${el.why}, so it cannot teach on this benchmark.`, start: () => loopOpen.set(true) };
  });
  function loopStarted(l: BenchLoop) {
    live.putLoop(l, live.mark());
    // page-level: the dialog has closed.
    toast.success(`Loop ${l.name} started`, 'Its first student trains on this model’s labels.', {
      label: 'Open loop',
      run: () => router.go(`/benchmarks/${id.peek()}/loops/${l.id}`),
    });
  }

  // ---- stop, delete ----
  const stopping = signal(false);
  const stopBusy = signal(false);
  const stopStatus = new DialogStatus();
  async function stop() {
    const e = entry.peek();
    if (!e) return;
    stopBusy.set(true);
    const mark = live.mark();
    try {
      live.putEntry(await hub.api.stopBenchEntry(id.peek(), e.id), mark);
      stopping.set(false);
    } catch (err) {
      stopStatus.error(err, 'Could not stop the entry');
    } finally {
      stopBusy.set(false);
    }
  }
  const removing = signal(false);
  const removeBusy = signal(false);
  const removeStatus = new DialogStatus();
  async function remove() {
    const e = entry.peek();
    if (!e) return;
    removeBusy.set(true);
    try {
      await hub.api.deleteBenchEntry(id.peek(), e.id);
      live.dropEntry(e.id);
      removing.set(false);
      // page-level: its confirm dialog has closed on the line above.
      toast.success(`Deleted ${e.name}`, 'Its result is gone from the leaderboard. The model and its files were not touched.');
      router.go(`/benchmarks/${id.peek()}`);
    } catch (err) {
      removeStatus.error(err, 'Could not delete the entry');
    } finally {
      removeBusy.set(false);
    }
  }
  const menu = computed((): MenuItem[] => {
    const e = entry.get();
    if (!e) return [];
    const m = model.get();
    return [
      ...(m ? [{ label: 'Open model', icon: Box, href: href(`/models/${m.id}`) }] : []),
      { label: 'Copy entry id', icon: Copy, onclick: () => void copyText(e.id, 'entry id') },
      ...(isEntryActive(e) && e.status !== 'stopping' ? [{ label: 'Stop entry', icon: Square, danger: true, onclick: () => stopping.set(true) }] : []),
      ...(!isEntryActive(e) ? [{ label: 'Delete entry', icon: Trash, danger: true, onclick: () => removing.set(true) }] : []),
    ];
  });
  pageCommands((): PageCommand[] => {
    const e = entry.get();
    if (!e) return [];
    const out: PageCommand[] = [{ id: 'bench-entry-board', label: 'Back to the leaderboard', icon: ArrowLeft, run: () => router.go(`/benchmarks/${id.peek()}`) }];
    const l = loop.get();
    if (l && !l.why) out.push({ id: 'bench-entry-loop', label: `Start a loop with ${e.name} as teacher`, icon: RotateCcw, run: l.start });
    if (isEntryActive(e) && e.status !== 'stopping') out.push({ id: 'bench-entry-stop', label: `Stop entry ${e.name}`, icon: Square, run: () => stopping.set(true) });
    return out;
  });

  /** The entry and the benchmark, where the page's own branch knows they are there. */
  const en = () => entry.get()!;
  const b = () => bench.get()!;
  const facts = computed(() => (entry.get() && bench.get() ? coverageFacts(entry.get()!, bench.get()!) : []));
  const cost = computed(() => (entry.get() ? costFacts(entry.get()!) : null));

  const who = (e: () => BenchEntry) => {
    const c = () => e().contestant;
    const mc = () => c() as ModelContestant;
    const ac = () => c() as AgentContestant;
    return (
      <>
        {c().kind === 'model' ? (
          <>
            {model.get() ? (
              <Ident
                value={mc().arch}
                href={href(`/models/${model.get()!.id}`)}
                title={`Open ${model.get()!.name}`}
              />
            ) : (
              <span title={models.data ? 'The model is no longer on this server' : undefined}>
                <Ident value={mc().arch} />
              </span>
            )}
            <span class="num">{num(mc().params / 1e6, 2)}M params</span>
            <span class="num">{mc().imgsz} px</span>
            {models.data && !model.get() ? <span>model deleted since</span> : null}
          </>
        ) : c().kind === 'agent' ? (
          <>
            <span>{AGENT_NAME[ac().agent]}</span>
            <span>{modelLabel(ac().model, agents.data?.find((a) => a.id === ac().agent))}</span>
            <span>{ac().effort} effort</span>
          </>
        ) : (
          <>
            <span>External</span>
            {/* Without confidences, the coverage facts after it say so. */}
            {e().scored ? <span>with confidences</span> : null}
          </>
        )}
      </>
    );
  };

  const asked = (e: () => BenchEntry) => (
    // What it was asked to run and how far it got: the facts to add it again with.
    <Panel title="What it was asked to run">
      <dl class="facts">
        <div>
          <dt>Images</dt>
          <dd class="num">
            {num(e().images)} of {num(bench.get()?.images ?? e().images)}
          </dd>
        </div>
        <div>
          <dt>Conditions</dt>
          <dd>
            {!bench.get()?.plan.conditions.length
              ? 'None in the plan'
              : e().conditions
                ? `All ${bench.get()!.plan.conditions.length} of the plan`
                : 'Unchanged images only'}
          </dd>
        </div>
        <div>
          <dt>Got through</dt>
          <dd class="num">
            {num(e().progress.done)} of {num(e().progress.total)} {e().contestant.kind === 'external' ? 'records' : 'image passes'}
          </dd>
        </div>
        {e().usage ? (
          <div>
            <dt>Calls</dt>
            <dd class="num">
              {num(e().usage!.calls)}
              {e().usage!.cost_usd !== null ? <>{' '}· {usd(e().usage!.cost_usd!)}</> : null}
            </dd>
          </div>
        ) : null}
        <div>
          <dt>Added</dt>
          <dd title={date(e().created)}>{ago(e().created)}</dd>
        </div>
        {e().finished ? (
          <div>
            <dt>Ended</dt>
            <dd title={date(e().finished!)}>{ago(e().finished!)}</dd>
          </div>
        ) : null}
      </dl>
    </Panel>
  );

  const r = () => result.data!;

  return (
    <>
      {unsupported.get() ? (
        <Page title="Benchmarks">
          <Unsupported />
        </Page>
      ) : live.error && !bench.get() ? (
        <Page
          title="Entry"
          actions={() => (
            <Button icon={ArrowLeft} href={href('/benchmarks')}>
              All benchmarks
            </Button>
          )}
        >
          {live.error instanceof ApiError && live.error.status === 404 ? (
            <ErrorState error={live.error} title="Benchmark not found" />
          ) : (
            <ErrorState error={live.error} title="Could not load this entry" retry={() => live.retry()} />
          )}
        </Page>
      ) : !bench.get() || live.loading ? (
        // The loaded page's frame: its header, the jump strip, the score panel's hero and rows.
        <div class="sk-page" aria-busy="true" aria-label="Loading entry">
          <div class="sk-head">
            <Skeleton w="min(240px, 60%)" h="24px" />
            <Skeleton w="min(560px, 90%)" h="14px" />
          </div>
          <div class="sk-jump">
            <Index each={SK_JUMP}>{(w) => <Skeleton w={`${w()}px`} h="14px" />}</Index>
          </div>
          <div class="sk-panel">
            <div class="sk-title">
              <Skeleton w="64px" h="14px" />
            </div>
            <div class="sk-score">
              <div class="sk-hero">
                <Skeleton w="120px" h="36px" />
                <Skeleton w="160px" h="12px" />
              </div>
              <div class="sk-rows">
                <Index each={SK_ROWS}>{() => <Skeleton h="16px" />}</Index>
              </div>
            </div>
          </div>
        </div>
      ) : !entry.get() ? (
        <Page
          title="Entry"
          actions={() => (
            <Button icon={ArrowLeft} href={href(`/benchmarks/${b().id}`)}>
              Leaderboard
            </Button>
          )}
        >
          <EmptyState icon={Target} title="Entry not found" description={`${b().name} has no entry ${entryId.get()}: it may have been deleted.`} />
        </Page>
      ) : (
        <>
          <Page
            title={en().name}
            inlineActions
            sub={() => (
              <Meta>
                <span>
                  <StatusBadge plain state={entryStatus(en())} />
                </span>
                {who(en)}
                <For each={facts.get()} key={(x) => x}>
                  {(x) => <span>{x()}</span>}
                </For>
                {cost.get() && done.get() ? (
                  // A model's speed is a throughput ("ms per image", in batches), never called latency; its tooltip says so.
                  <>
                    {en().speed ? (
                      <span class="num" use:tooltip={`${en().speed!.device}: ${SPEED_HELP}`}>
                        {cost.get()!.value} per image
                      </span>
                    ) : (
                      <span class="num" title={cost.get()!.sub}>
                        {cost.get()!.value}
                      </span>
                    )}
                  </>
                ) : null}
                <span class="faint" title={date(en().created)}>
                  added {ago(en().created)}
                </span>
              </Meta>
            )}
            subPhone={() => (
              <span class="num">
                {entryStatus(en()).label}
                {en().score ? (
                  <>
                    {' '}· score {scoreText(en().score!.total)}
                    {en().score!.partial ? <>, partial</> : null}
                  </>
                ) : null}
              </span>
            )}
            actions={() => (
              <>
                {!phone.current ? (
                  <Button icon={ArrowLeft} href={href(`/benchmarks/${b().id}`)}>
                    Leaderboard
                  </Button>
                ) : null}
                <Menu label="More entry actions" items={menu.get()} />
              </>
            )}
          >
            {done.get() && bench.get() && stale(en(), b().fingerprint) ? (
              <Callout tone="warning" icon={TriangleAlert} title="Graded on an earlier version of this benchmark" role="note">
                Its fingerprint is not the benchmark’s (the engine that grades has changed since), so its numbers do not compare with the current
                entries. Run it again to compare.
              </Callout>
            ) : null}
            {done.get() && en().score ? (
              <>
                <JumpNav sections={SECTIONS} label={`Sections of ${en().name}'s result`} />
                <div class="sections">
                  <ScoreSection entry={en()} bench={b()} result={result.data} rank={rank.get()} />
                  {result.data ? (
                    <>
                      <AccuracySection result={r()} entry={en()} bench={b()} />
                      {r().robustness ? (
                        <RobustnessSection result={r()} entry={en()} bench={b()} catalog={catalog.data} />
                      ) : (
                        <Panel title="Robustness" meta="accuracy kept under each condition" id="robustness">
                          <div class="none">
                            <p class="n-title">Not measured</p>
                            <p class="n-why">
                              {axisWhyNot('robustness', en(), b())}{' '}
                              {b().plan.conditions.length ? (
                                <>
                                  {en().contestant.kind === 'external'
                                    ? 'Send its predictions under the plan’s conditions as well (every image at every planned severity) to have this axis.'
                                    : `Add the entry again with “Run the plan’s conditions” on: ${b().plan.conditions.length} conditions, each image again at every severity.`}
                                </>
                              ) : null}
                            </p>
                            <Button size="sm" href={href(`/benchmarks/${b().id}?tab=conditions`)}>
                              See the conditions
                            </Button>
                          </div>
                        </Panel>
                      )}
                      <CalibrationSection result={r()} entry={en()} bench={b()} />
                      <SelfLabelSection result={r()} entry={en()} loop={loop.get()} />
                      <LocalizationSection result={r()} />
                      <ImagesSection bench={b()} entry={en()} bestConf={r().accuracy.best_conf} />
                    </>
                  ) : result.error ? (
                    <ErrorState error={result.error} title="Could not load the result" retry={() => result.refresh()} />
                  ) : (
                    <Index each={SK_SECTIONS}>{(h) => <Skeleton h={`${h()}px`} r="var(--radius-lg)" />}</Index>
                  )}
                </div>
              </>
            ) : isEntryActive(en()) ? (
              <RunningPanel entry={en()} bench={b()} catalog={catalog.data} lost={live.lost !== null} ahead={ahead.get()} />
            ) : en().status === 'failed' ? (
              <>
                <Callout
                  tone="danger"
                  icon={TriangleAlert}
                  title="It failed: nothing was scored"
                  role="alert"
                  actions={() => <Button href={href(`/benchmarks/${b().id}?add=${en().contestant.kind}`)}>Add it again</Button>}
                >
                  <CodeText text={sentence(en().error ?? 'The entry failed.')} />
                  {en().hint ? (
                    <>
                      {' '}
                      <CodeText text={en().hint!} />
                    </>
                  ) : null}
                </Callout>
                {asked(en)}
              </>
            ) : (
              <>
                <Callout
                  tone="neutral"
                  icon={CircleSlash}
                  title="Stopped before the end: nothing was scored"
                  role="note"
                  actions={() => <Button href={href(`/benchmarks/${b().id}?add=${en().contestant.kind}`)}>Add it again</Button>}
                >
                  A stopped entry keeps nothing of what it ran. Add it again to grade it.
                </Callout>
                {asked(en)}
              </>
            )}
          </Page>

          <NewLoopDialog
            bind:open={loopOpen}
            bench={b()}
            models={models.data}
            datasets={datasets.data}
            jobs={jobs.data}
            entries={live.entries}
            teacher={model.get()?.id ?? null}
            onstarted={loopStarted}
          />
          <Dialog
            open={stopping.get()}
            onclose={() => stopping.set(false)}
            title={`Stop ${en().name}?`}
            description="It ends what is in flight. A stopped entry has no score: what it ran so far is not kept."
            size="sm"
            status={stopStatus}
            footer={() => (
              <>
                <Button onclick={() => stopping.set(false)}>Keep running</Button>
                <Button variant="destructive" icon={Square} loading={stopBusy.get()} onclick={stop}>
                  Stop entry
                </Button>
              </>
            )}
          />
          <Dialog
            open={removing.get()}
            onclose={() => removing.set(false)}
            title={`Delete ${en().name}?`}
            description="Removes the entry and its result from the leaderboard. The model, its files and the benchmark's images are not touched."
            size="sm"
            status={removeStatus}
            footer={() => (
              <>
                <Button onclick={() => removing.set(false)}>Cancel</Button>
                <Button variant="destructive" icon={Trash} loading={removeBusy.get()} onclick={remove}>
                  Delete entry
                </Button>
              </>
            )}
          />
        </>
      )}
    </>
  );
}


css`
  .sections {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  /* A section jumped to lands clear of the top bar and the strip. */
  .sections :global(:is(#score, #accuracy, #robustness, #calibration, #self-labeling, #localization, #images)) {
    scroll-margin-top: calc(var(--size-topbar) + 44px + var(--space-4));
  }
  @media (pointer: coarse) {
    .sections :global(:is(#score, #accuracy, #robustness, #calibration, #self-labeling, #localization, #images)) {
      scroll-margin-top: calc(var(--size-topbar) + 52px + var(--space-4));
    }
  }
  .facts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: var(--space-3) var(--space-6);
  }
  .facts div {
    display: grid;
    gap: 2px;
  }
  .facts dt {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .none {
    display: grid;
    justify-items: start;
    gap: var(--space-2);
  }
  .n-title {
    font-weight: var(--weight-medium);
  }
  .n-why {
    color: var(--text-secondary);
    max-inline-size: var(--measure-dialog);
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
  .sk-jump {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    height: 44px;
    border-bottom: 1px solid var(--line-subtle);
  }
  .sk-panel {
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .sk-title {
    display: flex;
    align-items: center;
    height: 40px;
    padding: 0 var(--space-4);
  }
  .sk-score {
    display: grid;
    grid-template-columns: minmax(0, 3fr) minmax(0, 9fr);
    gap: var(--space-8);
    padding: 0 var(--space-4) var(--space-4);
  }
  .sk-hero,
  .sk-rows {
    display: grid;
    gap: var(--space-4);
    align-content: start;
  }
  @media (max-width: 1023px) {
    .sk-score {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .sk-jump {
      overflow: hidden;
    }
  }
`;
