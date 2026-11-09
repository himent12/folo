<script lang="ts">
  import { runProgress } from '$lib/api/progress';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Activity from '@lucide/svelte/icons/activity';
  import Box from '@lucide/svelte/icons/box';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Run } from '$lib/api/types';
  import RunSpark from '$lib/charts/RunSpark.svelte';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { ago, bytes, compact, metric } from '$lib/format';
  import { rankByContext, scoreContext, scoreTag } from '$lib/scoreSummary';
  import { balanced } from '$lib/ui/balancedGrid';
  import { failureCause } from '$lib/oom';
  import { tooltip } from '$lib/ui/tooltip';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import TableState from '$lib/ui/TableState.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import LiveRunCard from './overview/LiveRunCard.svelte';
  import Callout from './run/Callout.svelte';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import RefreshCwOff from '@lucide/svelte/icons/refresh-cw-off';
  import { asOf, Now } from '$lib/api/freshness.svelte';

  setCrumbs({ label: 'Overview' });

  const runs = new Resource((s) => hub.api.runs(s), { poll: 5000 });
  const models = new Resource((s) => hub.api.models(s));
  // The list endpoint has no stats; fetch each dataset for its class balance.
  const datasets = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    return Promise.all(list.map((d) => hub.api.dataset(d.id, s).catch(() => d)));
  });

  const live = $derived(runs.data?.find((r) => r.status === 'running' || r.status === 'stopping'));
  const recent = $derived((runs.data ?? []).filter((r) => r.id !== live?.id).slice(0, 6));
  const all = [runs, models, datasets];
  // Nothing loaded and every request failed: one page-level block says why and how to recover, instead of four panels
  // each saying the same thing (DESIGN_DIRECTION §2.8).
  const down = $derived(all.every((r) => !!r.error && !r.data));
  // Something loaded, then the server stopped answering: keep it, and say once that it is stale. Two clocks, never
  // confused (lib/api/freshness): the lists' snapshots and the live run's stream. While the stream still streams, only
  // the lists are behind, and the notice says exactly that; the live card stays live. When the stream went quiet too,
  // the server is not answering, and the card shows its last known state.
  const stale = $derived(!down && all.some((r) => r.stale !== undefined));
  const staleSince = $derived(Math.min(...all.map((r) => r.updated ?? Infinity)));
  let streamLost = $state<number | null>(null);
  const streaming = $derived(!!live && streamLost === null);
  const now = new Now(() => stale);
  const fresh = $derived(datasets.data?.length === 0 && runs.data?.length === 0 && models.data?.length === 0);
  function retryAll() {
    void runs.refresh();
    void models.refresh();
    void datasets.refresh();
  }
  const dsById = $derived(new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  const runById = $derived(new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  // A score ranks only against scores on the same dataset and split (lib/scoreSummary): the best of each comparable
  // group, its dataset named above it and any split but validation tagged beside the score. Imported and published
  // weights have no known split here, so they are counted under the list, not ranked in it.
  const ranking = $derived(
    rankByContext(models.data ?? [], (m) => scoreContext(m, runById, dsById), (m) => m.metrics?.map50_95, (m) => m.created),
  );
  const topModels = $derived(ranking.groups.slice(0, 6).map((g) => ({ m: g.items[0]!, ctx: g.ctx })));
  const unranked = $derived(ranking.unranked.length + Math.max(0, ranking.groups.length - 6));
  // The ranking needs each model's run and dataset: wait for both (a failed one leaves the splits unknown).
  const rankReady = $derived(!!models.data && (!!runs.data || !!runs.error) && (!!datasets.data || !!datasets.error));
  const total = (d: Dataset) => d.splits.train + d.splits.val + d.splits.test;
</script>

<Page title="Overview">
  {#if fresh}
    <!-- A fresh install: one path forward instead of a page of empty panels. -->
    <section class="onboard" aria-labelledby="onboard-title">
      <div class="ob-head">
        <h2 id="onboard-title">Train your first detector</h2>
        <p>Three steps from a folder of images to boxes on new ones.</p>
      </div>
      <ol class="steps">
        <li class="on">
          <span class="n">1</span>
          <div><b>Add a dataset</b><span>Register a YOLO or COCO folder, or generate a sample set of shapes in seconds.</span></div>
          <div class="ob-actions">
            <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>Add dataset</Button>
            <Button href={href('/datasets?add=synthetic')}>Generate sample set</Button>
          </div>
        </li>
        <li>
          <span class="n">2</span>
          <div><b>Train</b><span>Pick a model size; loss, mAP and validation samples stream in live.</span></div>
        </li>
        <li>
          <span class="n">3</span>
          <div><b>Inspect predictions</b><span>Drop any image into the Playground and tune confidence and IoU instantly.</span></div>
        </li>
      </ol>
    </section>
  {:else if down}
    <!-- Nothing loaded and every request failed: one recovery block (Retry, Connection settings) in the frame the live
         card would fill, the page's focal slot, never four panels each saying the server has not answered and never a
         frameless block in the corner of an empty canvas (DESIGN_DIRECTION §2.8). -->
    <div class="live">
      <Panel title="Training"><ErrorState error={runs.error} retry={retryAll} title="Could not load the overview" compact /></Panel>
    </div>
  {:else}
    {#if stale && streaming}
      <div class="stale">
        <Callout tone="warning" icon={RefreshCwOff} title="Lists not refreshing">
          The lists below show what loaded {asOf(staleSince, now.current)}; the run training now is still live.
          {#snippet actions()}<Button size="sm" onclick={retryAll}>Retry</Button>{/snippet}
        </Callout>
      </div>
    {:else if stale}
      <div class="stale">
        <Callout tone="warning" icon={WifiOff} title="Server not answering">
          Showing what loaded {asOf(staleSince, now.current)}.
          {#snippet actions()}<Button size="sm" onclick={retryAll}>Retry</Button>{/snippet}
        </Callout>
      </div>
    {/if}
    <div class="live">
      {#if runs.error && !runs.data}
        <Panel title="Training"><ErrorState error={runs.error} retry={() => runs.refresh()} compact /></Panel>
      {:else if !runs.data}
        <!-- The live card itself with placeholder values, so nothing jumps when it arrives. -->
        <LiveRunCard run={null} />
      {:else if live}
        {#key live.id}<LiveRunCard
            run={live}
            dataset={datasets.data ? (dsById.get(live.config.dataset_id) ?? null) : datasets.error ? null : undefined}
            bind:lost={streamLost}
          />{/key}
      {:else}
        <Panel title="Training">
          <EmptyState
            icon={Activity}
            title="Nothing is training"
            description={runs.data.length ? 'Start a run to watch its loss, mAP and samples update live here.' : 'Train your first detector: pick a dataset, a model size and go.'}
          >
            {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
          </EmptyState>
        </Panel>
      {/if}
    </div>
    <div class="cols">
      <div class="col-main">
        <Panel title="Recent runs" meta={runs.data ? `${recent.length} of ${runs.data.length}` : undefined} pad="none" id="recent-runs">
          {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/runs')}>All runs</Button>{/snippet}
          <!-- The table frame in every state: placeholder rows while loading, the message inside it when there is nothing
               (without column heads over it, as on the list pages). -->
          <div class="scroll">
            <table class="table tall" aria-busy={!runs.data} aria-label="Recent runs">
              {#if runs.data ? recent.length > 0 : !runs.error}<thead>
                <!-- Fixed widths on the shared header row: one geometry loading and loaded. -->
                <tr>
                  <th>Run</th>
                  <th style:width="112px">Status</th>
                  <th class="hide-sm" style:width="136px">Progress</th>
                  <th class="r" style:width="88px">mAP50-95</th>
                  <th class="hide-sm" style:width="120px">mAP50-95 trend</th>
                  <th class="r hide-sm" style:width="96px">Created</th>
                </tr>
              </thead>{/if}
              <tbody>
                {#if runs.error && !runs.data}
                  <TableState cols={6}><ErrorState error={runs.error} retry={() => runs.refresh()} /></TableState>
                {:else if !runs.data}
                  {#each [0, 1, 2, 3, 4, 5] as i (i)}
                    <tr aria-hidden="true">
                      <td><span class="l-20"><Skeleton w="112px" h="12px" /></span><span class="l-16"><Skeleton w="88px" h="10px" /></span></td>
                      <td><Skeleton w="72px" h="12px" /></td>
                      <td class="hide-sm"><div class="prog"><Skeleton h="4px" /><span><Skeleton w="40px" h="10px" /></span></div></td>
                      <td class="r"><span class="l-20 end"><Skeleton w="40px" h="12px" /></span></td>
                      <td class="hide-sm"><Skeleton w="88px" h="12px" /></td>
                      <td class="r hide-sm"><span class="l-20 end"><Skeleton w="56px" h="12px" /></span></td>
                    </tr>
                  {/each}
                {:else if recent.length === 0}
                  <TableState cols={6}>
                    <EmptyState icon={Activity} title="No runs yet" description="Runs you start appear here with their progress and best mAP." />
                  </TableState>
                {:else}
                  {#each recent as r (r.id)}
                    <tr class="link" onclick={() => router.go(`/runs/${r.id}`)}>
                      <td>
                        <a class="run-name" href={href(`/runs/${r.id}`)} onclick={(e) => e.stopPropagation()}>{r.name}</a>
                        <span class="sub"><Meta line><span class="ds">{dsById.get(r.config.dataset_id)?.name ?? r.config.dataset_id}</span><Ident value={r.config.arch} /></Meta></span>
                      </td>
                      <td use:tooltip={failureCause(r)}><StatusBadge status={r.status} plain /></td>
                      <td class="hide-sm">
                        <div class="prog">
                          <Progress value={runProgress(r)} size="xs" tone={r.status === 'running' ? 'accent' : 'neutral'} label="{r.name} progress" />
                          <span class="num faint">{r.epoch}/{r.epochs}</span>
                        </div>
                      </td>
                      <td class="r num">{#if r.best}{metric(r.best.map50_95)}{:else}<NoValue />{/if}</td>
                      <td class="hide-sm"><RunSpark run={r} /></td>
                      <td class="r faint hide-sm">{ago(r.created)}</td>
                    </tr>
                  {/each}
                {/if}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
      <div class="col-side">
        <!-- The top model of each comparable group (its dataset named above it, any split but validation tagged beside
             the score); the metric is named once, here, not under every value. -->
        <Panel title="Best models" meta="mAP50-95" pad="none" id="best-models">
          {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/models')}>All models</Button>{/snippet}
          {#if models.error && !models.data}
            <ErrorState error={models.error} retry={() => models.refresh()} />
          {:else if !rankReady}
            <!-- The loaded list's rows (dataset, name, meta; mAP on the right) at their height. -->
            <ul class="models" aria-busy="true" aria-label="Loading models" use:balanced={{ n: 4, min: 280 }}>
              {#each [0, 1, 2, 3] as i (i)}
                <li aria-hidden="true">
                  <span class="m-row">
                    <span class="m-text">
                      <span class="l-20"><Skeleton w="64px" h="10px" /></span>
                      <span class="l-20"><Skeleton w="128px" h="12px" /></span>
                      <span class="l-20"><Skeleton w="152px" h="10px" /></span>
                    </span>
                    <span class="m-map"><span class="l-20 end"><Skeleton w="40px" h="12px" /></span></span>
                  </span>
                </li>
              {/each}
            </ul>
          {:else if models.data?.length === 0}
            <EmptyState icon={Box} title="No models yet" description="Finished runs save their best weights here." />
          {:else}
            {#if topModels.length}
              <ul class="models" use:balanced={{ n: topModels.length, min: 280 }}>
                {#each topModels as { m, ctx } (m.id)}
                  {@const tag = scoreTag(ctx)}
                  <li>
                    <a href={href(`/models/${m.id}`)}>
                      <span class="m-text">
                        <span class="m-ds truncate">{ctx.datasetName}</span>
                        <span class="m-name truncate">{m.name}</span>
                        <span class="m-sub num"><Meta line><Ident value={m.arch} /><span>{m.classes.length} classes</span><span>{bytes(m.bytes, prefs.units)}</span></Meta></span>
                      </span>
                      <span class="m-map">
                        <span class="num"><span class="sr-only">mAP50-95 </span>{metric(m.metrics?.map50_95)}</span>
                        {#if tag}<span class="m-tag" data-fact>{tag}</span>{/if}
                      </span>
                    </a>
                  </li>
                {/each}
              </ul>
            {/if}
            {#if unranked}
              <!-- Imported and published weights (and any model whose split is unknown) are not ranked against these. -->
              <p class="unranked" class:alone={!topModels.length}>
                {unranked} {unranked === 1 ? 'model' : 'models'} not ranked: evaluation split unknown.
              </p>
            {/if}
          {/if}
        </Panel>
        <Panel title="Datasets" meta={datasets.data ? String(datasets.data.length) : undefined} pad="none" id="datasets">
          {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/datasets')}>All datasets</Button>{/snippet}
          {#if datasets.error && !datasets.data}
            <ErrorState error={datasets.error} retry={() => datasets.refresh()} compact />
          {:else if !datasets.data}
            <ul class="dslist" aria-busy="true" aria-label="Loading datasets" use:balanced={{ n: 3, min: 280 }}>
              {#each [0, 1, 2] as i (i)}
                <li aria-hidden="true">
                  <span class="d-sk">
                    <span class="d-row"><span class="l-20"><Skeleton w="96px" h="12px" /></span><span class="l-20"><Skeleton w="128px" h="10px" /></span></span>
                    <Skeleton h="4px" r="2px" />
                  </span>
                </li>
              {/each}
            </ul>
          {:else if datasets.data.length === 0}
            <EmptyState icon={Images} title="No datasets registered" description="Register a YOLO or COCO folder, or generate a sample set of shapes." />
          {:else}
            <ul class="dslist" use:balanced={{ n: Math.min(6, datasets.data.length), min: 280 }}>
              {#each datasets.data.slice(0, 6) as d (d.id)}
                <!-- The link covers the card (a stretched link); the class balance, its own disclosure, sits above it. -->
                <li class="d-card">
                  <a href={href(`/datasets/${d.id}`)}>
                    <span class="d-row">
                      <span class="d-name truncate">{d.name}</span>
                      <span class="d-nums num"><Meta line><span>{compact(total(d))} images</span><span>{d.classes.length} classes</span></Meta></span>
                    </span>
                  </a>
                  {#if d.stats}<span class="d-strip"><ClassStrip counts={d.stats.class_counts} classes={d.classes} height={4} named /></span>{/if}
                </li>
              {/each}
            </ul>
          {/if}
        </Panel>
      </div>
    </div>
  {/if}
</Page>

<style>
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
  .scroll {
    overflow-x: auto;
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
  .run-name {
    display: block;
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
</style>
