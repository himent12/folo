// Overview (`#/`): the run training now, recent runs, best models and datasets. The parts are in ./overview/.
import Activity from '$icons/activity';
import Plus from '$icons/plus';
import WifiOff from '$icons/wifi-off';
import RefreshCwOff from '$icons/refresh-cw-off';
import { computed, css, Key, signal } from 'fw';
import { readDatasets } from '$lib/api/datasetReads';
import { hub } from '$lib/api/index';
import { modelsResource } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Dataset, Run } from '$lib/api/types';
import { count } from '$lib/format';
import { rankByContext, scoreContext } from '$lib/scoreSummary';
import { runDatasetLabel } from '$lib/runDataset';
import { href } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import { watched } from '$lib/shell/runList';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import LiveRunCard from './overview/LiveRunCard';
import BestModels from './overview/BestModels';
import DatasetsPanel from './overview/DatasetsPanel';
import Onboard from './overview/Onboard';
import RecentRuns from './overview/RecentRuns';
import { slotRun } from './overview/slot';
import AlsoRunning from './overview/AlsoRunning';
import { bestNote, unrankedKinds } from './overview/bestNote';
import { followNewDatasets } from './overview/newDatasets';
import Callout from './run/Callout';
import { asOf, Now } from '$lib/api/freshness';

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
  // Published weights (`folo weights pull`, Desktop's download) are not the user's work: day one still shows the way in.
  const fresh = computed(() => noDatasets.get() && runList.get()?.length === 0 && !!models.data?.every((m) => m.source === 'published'));
  function retryAll() {
    void runs.refresh();
    void models.refresh();
    void datasets.refresh();
  }
  const dsById = computed(() => new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  // A run's dataset by name, a placeholder while the datasets load, the name it recorded once it was removed.
  // A run on a dataset made since the list loaded (./overview/newDatasets): the name it recorded while the list is read again.
  const lookingUp = followNewDatasets(datasets, () => runList.get());
  const dsLabel = (r: Run) =>
    lookingUp(r.config.dataset_id) ? (r.dataset?.name ?? undefined) : runDatasetLabel(r, datasets.data ? dsById.get() : undefined, !!datasets.error);
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
  // The ranking needs each model's run and dataset: wait for both (a failed one leaves the splits unknown).
  const rankReady = computed(() => !!models.data && (!!runList.get() || !!runs.error) && (!!datasets.data || !!datasets.error));
  // Why: a model's split is known from its run and dataset, so when either list failed to load that is the reason.
  const unrankedWhy = computed(() => {
    const failed = [runsFailed.get() && 'runs', !!datasets.error && !datasets.data && 'datasets'].filter(Boolean);
    return failed.length ? `could not load ${failed.join(' and ')}` : 'evaluation split unknown';
  });
  const note = computed(() => {
    const runMap = runById.get();
    const dsMap = dsById.get();
    const kinds = unrankedKinds(ranking.get().unranked.map((m) => scoreContext(m, runMap, dsMap)));
    return bestNote({ more: Math.max(0, ranking.get().groups.length - 6), ...kinds, why: unrankedWhy.get() });
  });
  /**
   * The header's facts line, as Desktop's: "4 datasets · 6 models · 1 run training"; where the app could not reach
   * the server, that (each panel below says what went wrong).
   */
  const facts = computed(() => {
    if (runsFailed.get() && !datasets.data && !models.data) return 'Not connected';
    if (!datasets.data || !models.data || !runList.get()) return null;
    const n = runList.get()!.filter((r) => r.status === 'running' || r.status === 'stopping').length;
    const training = n === 0 ? 'nothing training' : n === 1 ? '1 run training' : `${n} runs training`;
    return `${count(datasets.data.length, 'dataset')} · ${count(models.data.length, 'model')} · ${training}`;
  });

  const retry = () => (
    <Button size="sm" onclick={retryAll}>
      Retry
    </Button>
  );

  return (
    <Page
      title="Overview"
      sub={facts.get() ? () => <span class="facts num">{facts.get()}</span> : undefined}
      actions={
        // A fresh install's one way in is the onboarding's Add dataset: no second primary beside it.
        fresh.get()
          ? undefined
          : () => (
              <Button variant="primary" icon={Plus} href={href('/runs/new')}>
                New run
              </Button>
            )
      }
    >
      {fresh.get() ? (
        // A fresh install: one path forward instead of a page of empty panels.
        <>
          <Onboard system={system.data} systemError={system.error} />{' '}
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
                  dataset={
                    datasets.data && !lookingUp(live.get()!.config.dataset_id)
                      ? (dsById.get().get(live.get()!.config.dataset_id) ?? null)
                      : datasets.error
                        ? null
                        : undefined
                  }
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
                  // New run is the header's primary; with no dataset it could only say "Add a dataset first", so the way
                  // in here is Add dataset.
                  actions={
                    noDatasets.get()
                      ? () => (
                          <Button icon={Plus} href={href('/datasets?add=folder')}>
                            Add dataset
                          </Button>
                        )
                      : undefined
                  }
                />
              </Panel>
            )}
          </div>{' '}
          <AlsoRunning system={system.data} />{' '}
          <div class="cols">
            <div class="col-main">
              <RecentRuns all={runList.get()} recent={recent.get()} failed={runsFailed.get()} error={runs.error} retry={() => runs.refresh()} dsLabel={dsLabel} />
            </div>{' '}
            <div class="col-side">
              <BestModels models={models} ready={rankReady.get()} top={topModels.get()} note={note.get()} />{' '}
              <DatasetsPanel datasets={datasets} />
            </div>
          </div>
        </>
      )}
    </Page>
  );
}

css`
  /* Two independent stacks, so a short side panel never leaves a hole in a shared row: Recent runs on two thirds,
     Best models on one. */
  .cols {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
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
  /* Skeleton bars centred in the line box of the text they stand for (13/20 body). */
  .l-20 {
    display: flex;
    align-items: center;
    height: 20px;
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
  /* Stale data: one notice over the page, the panels keep what they have. */
  .stale {
    min-width: 0;
  }
  /* Wide screens: Recent runs and Best model share a row as equal-height panels (they end together in every state:
     loaded, loading, empty, error), and Datasets runs across the full width under both, its datasets side by side,
     instead of hanging below the side column alone. */
  @media (min-width: 1280px) {
    /* Best models takes its third but never less than 380 px (design/UNIFIED.md §4 Overview), so its rows stay one
       line; Recent runs takes the rest and drops whole columns by its own width. */
    .cols {
      grid-template-columns: minmax(0, 2fr) minmax(380px, 1fr);
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
`;
