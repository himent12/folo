<script lang="ts">
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Activity from '@lucide/svelte/icons/activity';
  import Box from '@lucide/svelte/icons/box';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Model } from '$lib/api/types';
  import RunSpark from '$lib/charts/RunSpark.svelte';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { ago, bytes, compact, metric } from '$lib/format';
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
  import LiveRunCard from './overview/LiveRunCard.svelte';

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
  // One failure behind every request is a connection problem: say it once instead of in every panel.
  const offline = $derived(!!runs.error && !runs.data && !!models.error && !models.data && !!datasets.error && !datasets.data);
  const fresh = $derived(datasets.data?.length === 0 && runs.data?.length === 0 && models.data?.length === 0);
  function retryAll() {
    void runs.refresh();
    void models.refresh();
    void datasets.refresh();
  }
  const dsById = $derived(new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  // mAP is only comparable on the same data, so rank within each dataset and show the best of each.
  const topModels = $derived.by(() => {
    const runDs = new Map((runs.data ?? []).map((r) => [r.id, r.config.dataset_id]));
    const best = new Map<string, { m: Model; ds: string }>();
    for (const m of models.data ?? []) {
      const dsId = m.run_id ? runDs.get(m.run_id) : undefined;
      const key = dsId ?? 'imported';
      const ds = dsId ? (dsById.get(dsId)?.name ?? dsId) : 'Imported';
      const cur = best.get(key);
      // On a tie the earlier checkpoint wins (a run's best weights are saved before its last).
      const d = (m.metrics?.map50_95 ?? -1) - (cur?.m.metrics?.map50_95 ?? -1);
      if (!cur || d > 0 || (d === 0 && m.created < cur.m.created)) best.set(key, { m, ds });
    }
    return [...best.values()].sort((a, b) => (b.m.created > a.m.created ? 1 : -1)).slice(0, 5);
  });
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
  {:else}
    <div class="live">
      {#if offline}
        <!-- One failure behind every request: said once, where the live run would be; the panels below keep their frames. -->
        <Panel title="Training"><ErrorState error={runs.error} retry={retryAll} compact /></Panel>
      {:else if runs.error && !runs.data}
        <Panel title="Training"><ErrorState error={runs.error} retry={() => runs.refresh()} compact /></Panel>
      {:else if !runs.data}
        <!-- The live card itself with placeholder values, so nothing jumps when it arrives. -->
        <LiveRunCard run={null} />
      {:else if live}
        {#key live.id}<LiveRunCard run={live} dataset={dsById.get(live.config.dataset_id)} />{/key}
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
          <!-- The table frame in every state: placeholder rows while loading, the message inside it when there is nothing. -->
          <div class="scroll">
            <table class="table tall" aria-busy={!runs.data} aria-label="Recent runs">
              <thead>
                <!-- Fixed widths on the shared header row: one geometry loading, empty, offline and loaded. -->
                <tr>
                  <th>Run</th>
                  <th style:width="112px">Status</th>
                  <th class="hide-sm" style:width="136px">Progress</th>
                  <th class="r" style:width="88px">mAP50-95</th>
                  <th class="hide-sm" style:width="120px">mAP trend</th>
                  <th class="r hide-sm" style:width="96px">Created</th>
                </tr>

              </thead>
              <tbody>
                {#if offline}
                  <TableState cols={6}><p class="quiet">Recent runs show here once the server answers.</p></TableState>
                {:else if runs.error && !runs.data}
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
                        <span class="sub">{dsById.get(r.config.dataset_id)?.name ?? r.config.dataset_id} · <Ident value={r.config.arch} /></span>
                      </td>
                      <td use:tooltip={failureCause(r)}><StatusBadge status={r.status} plain /></td>
                      <td class="hide-sm">
                        <div class="prog">
                          <Progress value={r.epoch / r.epochs} size="xs" tone={r.status === 'running' ? 'accent' : 'neutral'} label="{r.name} progress" />
                          <span class="num faint">{r.epoch}/{r.epochs}</span>
                        </div>
                      </td>
                      <td class="r num">{metric(r.best?.map50_95)}</td>
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
        <!-- The top model of each dataset (named above each row); the metric is named once, here, not under every value. -->
        <Panel title="Best models" meta="mAP50-95" pad="none" id="best-models">
          {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/models')}>All models</Button>{/snippet}
          {#if offline}
            <p class="quiet pad">The best model of each dataset shows here once the server answers.</p>
          {:else if models.error && !models.data}
            <ErrorState error={models.error} retry={() => models.refresh()} />
          {:else if !models.data}
            <!-- The loaded list's rows (dataset, name, meta; mAP on the right) at their height. -->
            <ul class="models" aria-busy="true" aria-label="Loading models">
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
          {:else if topModels.length === 0}
            <EmptyState icon={Box} title="No models yet" description="Finished runs save their best weights here." />
          {:else}
            <ul class="models">
              {#each topModels as { m, ds } (m.id)}
                <li>
                  <a href={href(`/models/${m.id}`)}>
                    <span class="m-text">
                      <span class="m-ds truncate">{ds}</span>
                      <span class="m-name truncate">{m.name}</span>
                      <span class="m-sub num"><Ident value={m.arch} /> · {m.classes.length} classes · {bytes(m.bytes, prefs.units)}</span>
                    </span>
                    <span class="m-map"><span class="num"><span class="sr-only">mAP50-95 </span>{metric(m.metrics?.map50_95)}</span></span>
                  </a>
                </li>
              {/each}
            </ul>
          {/if}
        </Panel>
        <Panel title="Datasets" meta={datasets.data ? String(datasets.data.length) : undefined} pad="none" id="datasets">
          {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/datasets')}>All datasets</Button>{/snippet}
          {#if offline}
            <p class="quiet pad">Registered datasets show here once the server answers.</p>
          {:else if datasets.error && !datasets.data}
            <ErrorState error={datasets.error} retry={() => datasets.refresh()} compact />
          {:else if !datasets.data}
            <ul class="dslist" aria-busy="true" aria-label="Loading datasets">
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
            <ul class="dslist">
              {#each datasets.data.slice(0, 6) as d (d.id)}
                <li>
                  <a href={href(`/datasets/${d.id}`)}>
                    <span class="d-row">
                      <span class="d-name truncate">{d.name}</span>
                      <span class="d-nums num">{compact(total(d))} images · {d.classes.length} classes</span>
                    </span>
                    {#if d.stats}<ClassStrip counts={d.stats.class_counts} classes={d.classes} height={4} />{/if}
                  </a>
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
  /* In place of a panel's content while the server is unreachable (the error is said once, in Training). */
  .quiet {
    padding: var(--space-5) var(--space-4);
    color: var(--text-tertiary);
  }
  .quiet.pad {
    padding: 0 var(--space-4) var(--space-4);
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
  .dslist {
    padding: 0 var(--space-2) var(--space-2);
  }
  .dslist a,
  .d-sk {
    display: grid;
    gap: 6px;
    padding: var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .dslist a:hover {
    background: var(--bg-hover);
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
    .dslist {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      column-gap: var(--space-4);
    }
  }
  @media (max-width: 1279px) {
    .cols {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  /* Laptop and tablet: the rail's two panels stack full width under Recent runs (as Datasets is on desktop), each
     laying its entries out side by side, so neither panel stops halfway beside a taller one. */
  @media (min-width: 768px) and (max-width: 1279px) {
    .models,
    .dslist {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      column-gap: var(--space-4);
    }
    /* Model entries carry three lines and a score: two columns keep them readable (and pairs even). */
    .models {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
