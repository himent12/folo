import ArrowRight from '$icons/arrow-right';
import Activity from '$icons/activity';
import Box from '$icons/box';
import Images from '$icons/images';
import Plus from '$icons/plus';
import WifiOff from '$icons/wifi-off';
import RefreshCwOff from '$icons/refresh-cw-off';
import { computed, css, For, Key, signal } from 'fw';
import { runProgress } from '$lib/api/progress';
import { readDatasets } from '$lib/api/datasetReads';
import { hub } from '$lib/api/index';
import { modelsResource } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Dataset, Run } from '$lib/api/types';
import RunSpark from '$lib/charts/RunSpark';
import ClassStrip from '$lib/detect/ClassStrip';
import { ago, bytes, compact, count, metric } from '$lib/format';
import { rankByContext, scoreContext, scoreTag } from '$lib/scoreSummary';
import { balanced } from '$lib/ui/balancedGrid';
import { failureCause } from '$lib/oom';
import { tooltip } from '$lib/ui/tooltip';
import { prefs } from '$lib/prefs';
import { runDatasetLabel } from '$lib/runDataset';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import { watched } from '$lib/shell/runList';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import { PROBLEM_LABEL } from './datasets/problem';
import LiveRunCard from './overview/LiveRunCard';
import { slotRun } from './overview/slot';
import AlsoRunning from './overview/AlsoRunning';
import Callout from './run/Callout';
import { asOf, Now } from '$lib/api/freshness';
import { hasFeature } from '$lib/api/queue';
import { LABEL_FEATURE } from '$lib/label';

const total = (d: Dataset) => d.splits.train + d.splits.val + d.splits.test;

export default function Overview() {
  setCrumbs({ label: 'Overview' });

  const runs = new Resource((s) => hub.api.runs(s), { poll: 5000 });
  // Kept current as runs end or go (lib/api/models), as on every screen that lists models.
  const models = modelsResource();
  // The list endpoint has no stats; each dataset is read for its class balance (a missing folder is marked, and not
  // asked about again on every visit: lib/api/datasetReads).
  const datasets = new Resource(readDatasets);
  // Which optional features the server has (labeling, benchmarks): what else Overview can point to.
  const system = new Resource((s) => hub.api.system(s));
  // The run list the sidebar's live card reads (the shell polls it every 2 s, shell/RunWatch) while it is current, so the
  // Training card and Recent runs never say something the sidebar contradicts; this page's own list when the shell's
  // stopped refreshing.
  const runList = computed(() => (watched.runs && (watched.since === null || !runs.data) ? watched.runs : runs.data));
  const runsFailed = computed(() => !!runs.error && !runList.get());

  // The run training, or starting (./overview/slot): a run just started shows here at once, as it does in the sidebar.
  const live = computed(() => slotRun(runList.get()));
  const recent = computed(() => {
    const id = live.get()?.id;
    return (runList.get() ?? []).filter((r) => r.id !== id).slice(0, 6);
  });
  const all = [runs, models, datasets];
  // Nothing loaded and every request failed: one page-level block says why and how to recover, instead of four panels
  // each saying the same thing (DESIGN_DIRECTION §2.8).
  const down = computed(() => runsFailed.get() && [models, datasets].every((r) => !!r.error && !r.data));
  // Something loaded, then the server stopped answering: keep it, and say once that it is stale. Two clocks, never
  // confused (lib/api/freshness): the lists' snapshots and the live run's stream. While the stream still streams, only
  // the lists are behind, and the notice says exactly that; the live card stays live. When the stream went quiet too,
  // the server is not answering, and the card shows its last known state.
  const stale = computed(() => !down.get() && all.some((r) => r.stale !== undefined));
  const staleSince = computed(() => Math.min(...all.map((r) => r.updated ?? Infinity)));
  const streamLost = signal<number | null>(null);
  const streaming = computed(() => !!live.get() && streamLost.get() === null);
  const now = new Now(() => stale.get());
  const noDatasets = computed(() => datasets.data?.length === 0);
  const fresh = computed(() => noDatasets.get() && runList.get()?.length === 0 && models.data?.length === 0);
  function retryAll() {
    void runs.refresh();
    void models.refresh();
    void datasets.refresh();
  }
  const dsById = computed(() => new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  // A run's dataset by name, a placeholder while the datasets load, the name it recorded once it was removed.
  const dsLabel = (r: Run) => runDatasetLabel(r, datasets.data ? dsById.get() : undefined, !!datasets.error);
  const runById = computed(() => new Map<string, Run>((runList.get() ?? []).map((r) => [r.id, r])));
  // A score ranks only against scores on the same dataset and split (lib/scoreSummary): the best of each comparable
  // group, its dataset named above it and any split but validation tagged beside the score. Imported and published
  // weights have no known split here, so they are counted under the list, not ranked in it.
  const ranking = computed(() => {
    const runMap = runById.get();
    const dsMap = dsById.get();
    return rankByContext(
      models.data ?? [],
      (m) => scoreContext(m, runMap, dsMap),
      (m) => m.metrics?.map50_95,
      (m) => m.created,
      (m) => m.name,
    );
  });
  const topModels = computed(() => ranking.get().groups.slice(0, 6).map((g) => ({ m: g.items[0]!, ctx: g.ctx })));
  const unranked = computed(() => ranking.get().unranked.length + Math.max(0, ranking.get().groups.length - 6));
  // The ranking needs each model's run and dataset: wait for both (a failed one leaves the splits unknown).
  const rankReady = computed(() => !!models.data && (!!runList.get() || !!runs.error) && (!!datasets.data || !!datasets.error));
  // Why: a model's split is known from its run and dataset, so when either list failed to load that is the reason.
  const unrankedWhy = computed(() => {
    const failed = [runsFailed.get() && 'runs', !!datasets.error && !datasets.data && 'datasets'].filter(Boolean);
    return failed.length ? `could not load ${failed.join(' and ')}` : 'evaluation split unknown';
  });
  const firstDatasets = computed(() => (datasets.data ?? []).slice(0, 6));

  const retry = () => (
    <Button size="sm" onclick={retryAll}>
      Retry
    </Button>
  );

  return (
    <Page title="Overview">
      {fresh.get() ? (
        // A fresh install: one path forward instead of a page of empty panels.
        <>
          <section class="onboard" aria-labelledby="onboard-title">
            <div class="ob-head">
              <h2 id="onboard-title">Train your first detector</h2>{' '}
              <p>Three steps from a folder of images to boxes on new ones.</p>
            </div>{' '}
            <ol class="steps">
              <li class="on">
                <span class="n">1</span>{' '}
                <div>
                  <b>Add a dataset</b>
                  <span>Register a YOLO or COCO folder, or generate a sample set of shapes in seconds.</span>
                </div>{' '}
                <div class="ob-actions">
                  <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>
                    Add dataset
                  </Button>{' '}
                  <Button href={href('/datasets?add=synthetic')}>Generate sample set</Button>
                </div>
                {hasFeature(system.data, LABEL_FEATURE) ? (
                  // Label images lives under Datasets: the first step is where people with unlabeled images look for it.
                  <p class="ob-alt">
                    Have images but no labels?{' '}
                    <a class="ob-link" href={href('/datasets/label')}>
                      Label them with an agent
                    </a>
                  </p>
                ) : null}
              </li>{' '}
              <li>
                <span class="n">2</span>{' '}
                <div>
                  <b>Train</b>
                  <span>Pick a model size; loss, mAP and validation samples stream in live.</span>
                </div>
              </li>{' '}
              <li>
                <span class="n">3</span>{' '}
                <div>
                  <b>Inspect predictions</b>
                  <span>Drop any image into the Playground and tune confidence and IoU instantly.</span>
                </div>
              </li>
            </ol>
          </section>{' '}
          {/* A labeling job can run before there is any dataset (it makes one). */}
          <AlsoRunning system={system.data} />
        </>
      ) : down.get() ? (
        // Nothing loaded and every request failed: one recovery block (Retry, Connection settings) in the frame the live
        // card would fill, the page's focal slot, never four panels each saying the server has not answered and never a
        // frameless block in the corner of an empty canvas (DESIGN_DIRECTION §2.8).
        <div class="live">
          <Panel title="Training">
            <ErrorState error={runs.error} retry={retryAll} title="Could not load the overview" compact />
          </Panel>
        </div>
      ) : (
        <>
          {stale.get() && streaming.get() ? (
            <div class="stale">
              <Callout tone="warning" icon={RefreshCwOff} title="Lists not refreshing" actions={retry}>
                The lists below show what loaded {asOf(staleSince.get(), now.current)}; the run training now is still live.
              </Callout>
            </div>
          ) : stale.get() ? (
            <div class="stale">
              <Callout tone="warning" icon={WifiOff} title="Server not answering" actions={retry}>
                Showing what loaded {asOf(staleSince.get(), now.current)}.
              </Callout>
            </div>
          ) : null}{' '}
          <div class="live">
            {runsFailed.get() ? (
              <Panel title="Training">
                <ErrorState error={runs.error} retry={() => runs.refresh()} title="Could not check what is training" compact />
              </Panel>
            ) : !runList.get() ? (
              // The common state's shape (Nothing is training: the mark, a title, two lines and a button), so nothing
              // below moves when it arrives; a run training is the exception, and grows the card once.
              <Panel title="Training">
                <div class="idle-sk" aria-busy="true" aria-label="Loading training">
                  <Skeleton w="24px" h="24px" />
                  <span class="l-20">
                    <Skeleton w="120px" h="12px" />
                  </span>
                  <span class="l-20">
                    <Skeleton w="min(320px, 100%)" h="10px" />
                  </span>
                  <span class="l-20">
                    <Skeleton w="64px" h="10px" />
                  </span>
                  <span class="idle-cta">
                    <Skeleton w="96px" h="var(--control-md)" />
                  </span>
                </div>
              </Panel>
            ) : live.get() ? (
              <Key when={live.get()!.id}>
                <LiveRunCard
                  run={live.get()!}
                  dataset={datasets.data ? (dsById.get().get(live.get()!.config.dataset_id) ?? null) : datasets.error ? null : undefined}
                  datasetName={dsLabel(live.get()!)}
                  bind:lost={streamLost}
                />
              </Key>
            ) : (
              <Panel title="Training">
                <EmptyState
                  icon={Activity}
                  title="Nothing is training"
                  description={
                    noDatasets.get()
                      ? 'Add a dataset to train on, then start a run to watch it here.'
                      : runList.get()!.length
                        ? 'Start a run to watch its loss, mAP and samples update live here.'
                        : 'Train your first detector: pick a dataset, a model size and go.'
                  }
                  actions={() =>
                    // With no dataset, New run could only say "Add a dataset first": open Add dataset instead.
                    noDatasets.get() ? (
                      <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>
                        Add dataset
                      </Button>
                    ) : (
                      <Button variant="primary" icon={Plus} href={href('/runs/new')}>
                        New run
                      </Button>
                    )
                  }
                />
              </Panel>
            )}
          </div>{' '}
          <AlsoRunning system={system.data} />{' '}
          <div class="cols">
            <div class="col-main">
              <Panel
                title="Recent runs"
                meta={runList.get() ? `${recent.get().length} of ${runList.get()!.length}` : undefined}
                pad="none"
                id="recent-runs"
                actions={() => (
                  <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/runs')}>
                    All runs
                  </Button>
                )}
              >
                {/* The table frame in every state: placeholder rows while loading, the message inside it when there is
                    nothing (without column heads over it, as on the list pages). */}
                <div class="scroll">
                  <table class="table tall" aria-busy={!runList.get()} aria-label="Recent runs">
                    {(runList.get() ? recent.get().length > 0 : !runs.error) ? (
                      <thead>
                        {/* Fixed widths on the shared header row: one geometry loading and loaded. */}
                        <tr>
                          <th>Run</th>
                          <th style:width="128px">Status</th>
                          <th class="hide-sm" style:width="136px">
                            Progress
                          </th>
                          <th class="r" style:width="88px">
                            mAP50-95
                          </th>
                          <th class="hide-sm trend" style:width="120px">
                            mAP50-95 trend
                          </th>
                          <th class="r hide-sm" style:width="96px">
                            Created
                          </th>
                        </tr>
                      </thead>
                    ) : null}
                    <tbody>
                      {runsFailed.get() ? (
                        <TableState cols={6}>
                          <ErrorState error={runs.error} retry={() => runs.refresh()} title="Could not load recent runs" />
                        </TableState>
                      ) : !runList.get() ? (
                        <For each={[0, 1, 2, 3, 4, 5]} key={(i) => i}>
                          {() => (
                            <tr aria-hidden="true">
                              <td>
                                <span class="l-20">
                                  <Skeleton w="112px" h="12px" />
                                </span>
                                <span class="l-16">
                                  <Skeleton w="88px" h="10px" />
                                </span>
                              </td>
                              <td>
                                <Skeleton w="72px" h="12px" />
                              </td>
                              <td class="hide-sm">
                                <div class="prog">
                                  <Skeleton h="4px" />
                                  <span>
                                    <Skeleton w="40px" h="10px" />
                                  </span>
                                </div>
                              </td>
                              <td class="r">
                                <span class="l-20 end">
                                  <Skeleton w="40px" h="12px" />
                                </span>
                              </td>
                              <td class="hide-sm trend">
                                <Skeleton w="88px" h="12px" />
                              </td>
                              <td class="r hide-sm">
                                <span class="l-20 end">
                                  <Skeleton w="56px" h="12px" />
                                </span>
                              </td>
                            </tr>
                          )}
                        </For>
                      ) : recent.get().length === 0 ? (
                        <TableState cols={6}>
                          <EmptyState icon={Activity} title="No runs yet" description="Runs you start appear here with their progress and best mAP." />
                        </TableState>
                      ) : (
                        <For each={recent.get()} key={(r) => r.id}>
                          {(r) => (
                            <tr class="link" onclick={() => router.go(`/runs/${r().id}`)}>
                              <td>
                                <a class="run-name" href={href(`/runs/${r().id}`)} onclick={(e) => e.stopPropagation()} use:tooltip={r().name}>
                                  {r().name}
                                </a>{' '}
                                <span class="sub">
                                  <Meta line>
                                    {dsLabel(r()) === undefined ? (
                                      <span class="ds">
                                        &#8203;<Skeleton inline w="72px" h="10px" />
                                      </span>
                                    ) : dsLabel(r()) ? (
                                      <span class="ds" use:tooltip={dsLabel(r())}>
                                        {dsLabel(r())}
                                      </span>
                                    ) : null}
                                    <Ident value={r().config.arch} />
                                  </Meta>
                                </span>
                              </td>
                              <td use:tooltip={failureCause(r())}>
                                <StatusBadge run={r()} plain />
                              </td>
                              <td class="hide-sm">
                                <div class="prog">
                                  <Progress
                                    value={runProgress(r())}
                                    size="xs"
                                    tone={r().status === 'running' ? 'accent' : 'neutral'}
                                    label={`${r().name} progress`}
                                  />{' '}
                                  <span class="num faint">
                                    {r().epoch}/{r().epochs}
                                  </span>
                                </div>
                              </td>
                              <td class="r num">{r().best ? metric(r().best!.map50_95) : <NoValue />}</td>
                              <td class="hide-sm trend">
                                <RunSpark run={r()} />
                              </td>
                              <td class="r faint hide-sm">{ago(r().created)}</td>
                            </tr>
                          )}
                        </For>
                      )}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </div>{' '}
            <div class="col-side">
              {/* The top model of each comparable group (its dataset named above it, any split but validation tagged
                  beside the score); the metric is named once, here, not under every value. */}
              <Panel
                title="Best models"
                meta="mAP50-95"
                pad="none"
                id="best-models"
                actions={() => (
                  <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/models')}>
                    All models
                  </Button>
                )}
              >
                {models.error && !models.data ? (
                  <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" />
                ) : !rankReady.get() ? (
                  // The loaded list's rows (dataset, name, meta; mAP on the right) at their height.
                  <ul class="models" aria-busy="true" aria-label="Loading models" use:balanced={{ n: 4, min: 280 }}>
                    <For each={[0, 1, 2, 3]} key={(i) => i}>
                      {() => (
                        <li aria-hidden="true">
                          <span class="m-row">
                            <span class="m-text">
                              <span class="l-20">
                                <Skeleton w="64px" h="10px" />
                              </span>{' '}
                              <span class="l-20">
                                <Skeleton w="128px" h="12px" />
                              </span>{' '}
                              <span class="l-20">
                                <Skeleton w="152px" h="10px" />
                              </span>
                            </span>{' '}
                            <span class="m-map">
                              <span class="l-20 end">
                                <Skeleton w="40px" h="12px" />
                              </span>
                            </span>
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                ) : models.data?.length === 0 ? (
                  <EmptyState icon={Box} title="No models yet" description="Finished runs save their best weights here." />
                ) : (
                  <>
                    {topModels.get().length ? (
                      <ul class="models" use:balanced={{ n: topModels.get().length, min: 280 }}>
                        <For each={topModels.get()} key={(t) => t.m.id}>
                          {(t) => {
                            const tag = () => scoreTag(t().ctx);
                            return (
                              <li>
                                <a href={href(`/models/${t().m.id}`)}>
                                  <span class="m-text">
                                    <span class="m-ds truncate">{t().ctx.datasetName}</span>{' '}
                                    <span class="m-name truncate">{t().m.name}</span>{' '}
                                    <span class="m-sub num">
                                      <Meta line>
                                        <Ident value={t().m.arch} />
                                        <span>{count(t().m.classes.length, 'class')}</span>
                                        <span>{bytes(t().m.bytes, prefs.units)}</span>
                                      </Meta>
                                    </span>
                                  </span>{' '}
                                  <span class="m-map">
                                    <span class="num">
                                      <span class="sr-only">mAP50-95 </span>
                                      {metric(t().m.metrics?.map50_95)}
                                    </span>{' '}
                                    {tag() ? (
                                      <span class="m-tag" data-fact>
                                        {tag()}
                                      </span>
                                    ) : null}
                                  </span>
                                </a>
                              </li>
                            );
                          }}
                        </For>
                      </ul>
                    ) : null}{' '}
                    {unranked.get() ? (
                      // Imported and published weights (and any model whose split is unknown) are not ranked against these.
                      <p class="unranked" class:alone={!topModels.get().length}>
                        {count(unranked.get(), 'model')} not ranked: {unrankedWhy.get()}.
                      </p>
                    ) : null}
                  </>
                )}
              </Panel>{' '}
              <Panel
                title="Datasets"
                meta={datasets.data ? String(datasets.data.length) : undefined}
                pad="none"
                id="datasets"
                actions={() => (
                  <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/datasets')}>
                    All datasets
                  </Button>
                )}
              >
                {datasets.error && !datasets.data ? (
                  <ErrorState error={datasets.error} retry={() => datasets.refresh()} title="Could not load datasets" compact />
                ) : !datasets.data ? (
                  <ul class="dslist" aria-busy="true" aria-label="Loading datasets" use:balanced={{ n: 3, min: 280 }}>
                    <For each={[0, 1, 2]} key={(i) => i}>
                      {() => (
                        <li aria-hidden="true">
                          <span class="d-sk">
                            <span class="d-row">
                              <span class="l-20">
                                <Skeleton w="96px" h="12px" />
                              </span>
                              <span class="l-20">
                                <Skeleton w="128px" h="10px" />
                              </span>
                            </span>{' '}
                            <Skeleton h="4px" r="2px" />
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                ) : datasets.data.length === 0 ? (
                  <EmptyState
                    icon={Images}
                    title="No datasets registered"
                    description="Register a YOLO or COCO folder, or generate a sample set of shapes."
                    actions={() => (
                      <Button size="sm" icon={Plus} href={href('/datasets?add=folder')}>
                        Add dataset
                      </Button>
                    )}
                  />
                ) : (
                  <ul class="dslist" use:balanced={{ n: Math.min(6, datasets.data.length), min: 280 }}>
                    <For each={firstDatasets.get()} key={(d) => d.id}>
                      {(d) => (
                        // The link covers the card (a stretched link); the class balance, its own disclosure, sits above it.
                        <li class="d-card">
                          <a href={href(`/datasets/${d().id}`)}>
                            <span class="d-row">
                              <span class="d-name truncate">{d().name}</span>{' '}
                              {/* A folder gone since it was registered: the Datasets list's word in place of its counts. */}
                              <span class="d-nums num">
                                {d().problem ? (
                                  <Badge tone="warning">{PROBLEM_LABEL[d().problem!]}</Badge>
                                ) : (
                                  <Meta line>
                                    <span>{`${compact(total(d()))} ${total(d()) === 1 ? 'image' : 'images'}`}</span>
                                    <span>{count(d().classes.length, 'class')}</span>
                                  </Meta>
                                )}
                              </span>
                            </span>
                          </a>{' '}
                          {d().stats ? (
                            <span class="d-strip">
                              <ClassStrip counts={d().stats!.class_counts} classes={d().classes} height={4} named />
                            </span>
                          ) : null}
                        </li>
                      )}
                    </For>
                  </ul>
                )}
              </Panel>
            </div>
          </div>
        </>
      )}
    </Page>
  );
}

css`
  /* The content width, the three steps across it where they fit (a stack below that): each step's actions sit under its
     sentence, never at the far edge of a wide row, and the card never sits alone in a corner of an empty page. */
  .onboard {
    container-type: inline-size;
    display: grid;
    gap: var(--space-5);
    padding: var(--space-4);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
  }
  .ob-head h2 {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
  }
  .ob-head p {
    color: var(--text-secondary);
  }
  .steps {
    display: grid;
    gap: var(--space-2);
  }
  /* The number sits on the title's line, top-aligned however tall the step's sentence and actions run. */
  .steps li {
    display: grid;
    grid-template-columns: 28px 1fr;
    align-items: start;
    align-content: start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4) var(--space-3) var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-subtle);
    color: var(--text-tertiary);
  }
  .steps li.on {
    border-color: var(--line-strong);
    background: var(--bg-raised);
    color: var(--text-secondary);
  }
  /* The title's first line on the number's centre (28 px circle, 20 px line). */
  .steps li div:not(.ob-actions) {
    display: grid;
    gap: 2px;
    padding-top: var(--space-1);
  }
  .steps b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .n {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid var(--line-strong);
    font-variant-numeric: tabular-nums;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .on .n {
    border-color: var(--text-primary);
    background: var(--text-primary);
    color: var(--bg-panel);
  }
  .ob-actions {
    grid-column: 2;
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    margin-top: var(--space-1);
  }
  .ob-alt {
    grid-column: 2;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .ob-link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .ob-link:hover {
    text-decoration-color: var(--text-secondary);
  }
  /* A container query, not a tier: the onboarding card sits in the content column, whose width depends on the sidebar
     as much as on the viewport; three steps need 840 px of it. */
  @container (min-width: 840px) {
    .steps {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--space-3);
    }
    .steps li {
      padding: var(--space-4);
    }
    .ob-actions {
      margin-top: var(--space-2);
    }
  }
  /* Two independent
 stacks, so a short side panel never leaves a hole in a shared row. */
  .cols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: var(--space-4);
    align-items: start;
  }
  .col-main,
  .col-side {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .live {
    min-width: 0;
  }
  /* The table fits its card at every width, never scrolling sideways: the trend goes first when the card is narrower
     than all six columns (a 1280 px laptop puts Best models beside it), and a long run name ends in an ellipsis
     rather than widen its column (its meta line sets the column's least width). */
  .scroll {
    container-type: inline-size;
    overflow-x: auto;
  }
  @container (max-width: 739px) {
    .trend {
      display: none;
    }
  }
  /* Skeleton bars centred in the line box of the text they stand for (13/20 body, 12/16 meta). */
  .l-20,
  .l-16 {
    display: flex;
    align-items: center;
    height: 20px;
  }
  .l-16 {
    height: 16px;
  }
  .l-20.end {
    justify-content: flex-end;
  }
  /* Neither line sets the Run column's least width (the container query above keeps it 170 px or more): the name ends
     in an ellipsis, and on its meta line only the dataset's name does, never the size beside it (a fact). */
  .run-name,
  .sub {
    width: 0;
    min-width: 100%;
  }
  .idle-sk {
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr);
    column-gap: var(--space-3);
    align-items: center;
    padding: var(--space-1) 0;
  }
  .idle-sk > :not(:first-child) {
    grid-column: 2;
  }
  .idle-cta {
    margin-top: var(--space-3);
  }
  .run-name {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  @media (min-width: 768px) {
    .sub :global(.ui-meta.line) {
      display: flex;
      flex-wrap: nowrap;
    }
    .sub :global(.ui-meta.line > *) {
      flex: none;
    }
    .sub :global(.ui-meta.line > .ds) {
      flex: 0 1 auto;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  }
  /* Stale data: one notice over the page, the panels keep what they have. */
  .stale {
    min-width: 0;
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 120px;
  }
  /* The count never shrinks under its text ("150/150"); the bar takes what is left. */
  .prog span {
    flex-shrink: 0;
    font-size: var(--type-sm-size);
    min-width: 46px;
    text-align: right;
  }
  /* Entries side by side wherever two or more fit, in balanced columns (ui/balancedGrid sets --cols): four datasets
     are 2 × 2, never 3 + 1 with one alone on a row. */
  .models,
  .dslist {
    display: grid;
    grid-template-columns: repeat(var(--cols, 1), minmax(0, 1fr));
    column-gap: var(--space-4);
  }
  .models {
    padding: 0 var(--space-2) var(--space-2);
  }
  .models a,
  .m-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 64px;
    padding-top: var(--space-2) !important;
    padding-bottom: var(--space-2) !important;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .models a:hover {
    background: var(--bg-hover);
  }
  .m-text {
    flex: 1;
    min-width: 0;
    display: grid;
  }
  .m-name {
    font-weight: var(--weight-medium);
  }
  .m-ds {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .m-sub {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .m-map {
    display: grid;
    justify-items: end;
  }
  .m-map .num {
    font-weight: var(--weight-medium);
  }
  /* "Training split", "Split unknown": a label beside a score that is not on a validation split. It hangs under the
     score, so every score keeps the same place in its row with or without one. */
  .m-map {
    position: relative;
  }
  .m-tag {
    position: absolute;
    top: 100%;
    right: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  /* The list's foot: on the panel's bottom edge (the body is a flex column, ui/Panel), so a panel stretched to Recent
     runs' height ends on its note, never on 70 px of nothing under it. */
  .unranked {
    margin-top: auto;
    padding: 0 var(--space-4) var(--space-4);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .unranked.alone {
    padding-top: var(--space-1);
  }
  .dslist {
    padding: 0 var(--space-2) var(--space-2);
  }
  .d-card,
  .d-sk {
    position: relative;
    display: grid;
    gap: 6px;
    padding: var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .d-card:has(> a:hover) {
    background: var(--bg-hover);
  }
  .d-card > a {
    display: block;
    min-width: 0;
  }
  /* The ring goes round the card the link covers. */
  .d-card:has(> a:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .d-card > a:focus-visible {
    outline: none;
  }
  .d-card > a::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
  }
  .d-strip {
    position: relative;
    z-index: 1;
  }
  .d-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
    min-width: 0;
  }
  .d-name {
    font-weight: var(--weight-medium);
  }
  .d-nums {
    flex-shrink: 0;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Wide screens: Recent runs and Best model share a row as equal-height panels (they end together in every state:
     loaded, loading, empty, error), and Datasets runs across the full width under both, its datasets side by side,
     instead of hanging below the side column alone. */
  @media (min-width: 1280px) {
    .cols {
      align-items: stretch;
    }
    .col-main,
    .col-side {
      display: contents;
    }
    .cols > .col-side > :global(#datasets) {
      grid-column: 1 / -1;
    }
  }
  @media (max-width: 1279px) {
    .cols {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  /* Phones: every Recent runs row is one height, a name over one meta line. The dataset goes (the run's own page has
     it), so a long one ("shapes-hard") never wraps its row's meta onto a second line (DESIGN_DIRECTION §2.12). */
  @media (max-width: 767px) {
    .sub :global(.ds) {
      display: none;
    }
  }
`;
