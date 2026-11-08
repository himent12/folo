<script lang="ts">
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Activity from '@lucide/svelte/icons/activity';
  import Box from '@lucide/svelte/icons/box';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Settings from '@lucide/svelte/icons/settings';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { ApiError } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Model } from '$lib/api/types';
  import RunSpark from '$lib/charts/RunSpark.svelte';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { ago, bytes, compact, metric } from '$lib/format';
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
  // A specific line beats a generic message: code, status and when it happened.
  const errDetail = $derived.by(() => {
    const e = runs.error;
    if (!(e instanceof ApiError)) return e instanceof Error ? e.message : null;
    const at = new Date().toLocaleTimeString('en-US', { hour12: false });
    return `GET /api/runs → ${e.status ? `HTTP ${e.status}` : 'no response'} · ${e.code} · ${at}`;
  });
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
      if (!cur || (m.metrics?.map50_95 ?? -1) > (cur.m.metrics?.map50_95 ?? -1)) best.set(key, { m, ds });
    }
    return [...best.values()].sort((a, b) => (b.m.created > a.m.created ? 1 : -1)).slice(0, 5);
  });
  const total = (d: Dataset) => d.splits.train + d.splits.val + d.splits.test;
</script>

<Page title="Overview">

  {#if offline}
    <section class="offline" role="alert">
      <div class="off-mark" aria-hidden="true"><WifiOff size={20} strokeWidth={1.5} /></div>
      <h2>Studio can't reach the FOLO server</h2>
      <p>
        Nothing answered at <span class="mono">{hub.serverUrl}</span>. Start <span class="mono">folo serve</span> (or check
        the server URL in Settings), then retry.
      </p>
      {#if errDetail}<p class="off-detail mono">{errDetail}</p>{/if}
      <div class="off-actions">
        <Button variant="primary" icon={RotateCw} onclick={retryAll}>Retry</Button>
        <Button icon={Settings} href={href('/settings')}>Connection settings</Button>
      </div>
    </section>
  {:else if fresh}
    <!-- A fresh install: one path forward instead of a page of empty panels. -->
    <section class="onboard" aria-labelledby="onboard-title">
      <div class="ob-head">
        <h2 id="onboard-title">Train your first detector</h2>
        <p>Three steps from a folder of images to boxes on new ones.</p>
      </div>
      <ol class="steps">
        <li class="on">
          <span class="n">1</span>
          <div><b>Add a dataset</b><span>Register a YOLO or COCO folder, or generate a synthetic shapes set in seconds.</span></div>
          <div class="ob-actions">
            <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>Add dataset</Button>
            <Button href={href('/datasets?add=synthetic')}>Generate sample</Button>
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
      {#if runs.error && !runs.data}
        <Panel title="Training"><ErrorState error={runs.error} retry={() => runs.refresh()} compact /></Panel>
      {:else if !runs.data}
        <!-- The live card's frame (header, stat column, two charts), so nothing jumps when it arrives. -->
        <div class="live-sk" aria-busy="true" aria-label="Loading training">
          <div class="lsk-head"><Skeleton w="72px" h="20px" r="var(--radius-full)" /><Skeleton w="180px" h="20px" /><Skeleton w="min(360px, 70%)" h="14px" /></div>
          <div class="lsk-body">
            <div class="lsk-stats"><Skeleton w="96px" h="40px" /><Skeleton h="4px" /><div class="lsk-grid">{#each [0, 1, 2, 3] as i (i)}<Skeleton h="44px" />{/each}</div></div>
            <Skeleton h="232px" r="var(--radius-md)" />
          </div>
        </div>
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
        {#if runs.error && !runs.data}
          <ErrorState error={runs.error} retry={() => runs.refresh()} />
        {:else if !runs.data}
          <!-- Rows with the table's columns: name, status, progress, score. -->
          <div class="list-sk">
            {#each [0, 1, 2, 3] as i (i)}<div class="rsk"><span class="rsk-name"><Skeleton w="60%" h="14px" /><Skeleton w="40%" h="10px" /></span><Skeleton w="64px" h="18px" r="var(--radius-full)" /><Skeleton h="4px" /><Skeleton w="40px" h="14px" /></div>{/each}
          </div>
        {:else if recent.length === 0}
          <EmptyState compact icon={Activity} title="No runs yet" description="Runs you start appear here with their progress and best mAP." />
        {:else}
          <div class="scroll">
            <table class="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th class="hide-sm">Progress</th>
                  <th class="r">mAP50-95</th>
                  <th class="hide-sm">Trend</th>
                  <th class="r hide-sm">Created</th>
                </tr>
              </thead>
              <tbody>
                {#each recent as r (r.id)}
                  <tr class="link" onclick={() => router.go(`/runs/${r.id}`)}>
                    <td>
                      <a class="run-name" href={href(`/runs/${r.id}`)} onclick={(e) => e.stopPropagation()}>{r.name}</a>
                      <span class="run-sub">{dsById.get(r.config.dataset_id)?.name ?? r.config.dataset_id} · {r.config.arch}</span>
                    </td>
                    <td><StatusBadge status={r.status} plain /></td>
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
              </tbody>
            </table>
          </div>
        {/if}
      </Panel>
    </div>
    <div class="col-side">
      <Panel title="Best model per dataset" pad="none" id="best-models">
        {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/models')}>All models</Button>{/snippet}
        {#if models.error && !models.data}
          <ErrorState error={models.error} retry={() => models.refresh()} />
        {:else if !models.data}
          <div class="list-sk">{#each [0, 1, 2] as i (i)}<Skeleton h="20px" />{/each}</div>
        {:else if topModels.length === 0}
          <EmptyState compact icon={Box} title="No models yet" description="Finished runs save their best weights here." />
        {:else}
          <ul class="models">
            {#each topModels as { m, ds } (m.id)}
              <li>
                <a href={href(`/models/${m.id}`)}>
                  <span class="m-text">
                    <span class="m-ds truncate">{ds}</span>
                    <span class="m-name truncate">{m.name}</span>
                    <span class="m-sub num">{m.arch} · {m.classes.length} classes · {bytes(m.bytes, prefs.units)}</span>
                  </span>
                  <span class="m-map">
                    <span class="num">{metric(m.metrics?.map50_95)}</span>
                    <span class="m-cap">mAP50-95</span>
                  </span>
                </a>
              </li>
            {/each}
          </ul>
        {/if}
      </Panel>
      <Panel title="Datasets" meta={datasets.data ? String(datasets.data.length) : undefined} pad="none" id="datasets">
        {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/datasets')}>All datasets</Button>{/snippet}
        {#if datasets.error && !datasets.data}
          <ErrorState error={datasets.error} retry={() => datasets.refresh()} compact />
        {:else if !datasets.data}
          <div class="list-sk">{#each [0, 1, 2] as i (i)}<Skeleton h="20px" />{/each}</div>
        {:else if datasets.data.length === 0}
          <EmptyState compact icon={Images} title="No datasets registered" description="Register a YOLO or COCO folder, or generate a synthetic shapes set." />
        {:else}
          <ul class="dslist">
            {#each datasets.data.slice(0, 4) as d (d.id)}
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
  .onboard {
    display: grid;
    gap: var(--space-5);
    padding: var(--space-6);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
  }
  :global([data-theme='light']) .onboard {
    border-color: var(--line-default);
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
  .steps li {
    display: grid;
    grid-template-columns: 28px 1fr auto;
    align-items: center;
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
  .steps li div:not(.ob-actions) {
    display: grid;
    gap: 2px;
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
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  @media (max-width: 767px) {
    .onboard {
      padding: var(--space-4);
    }
    .steps li {
      grid-template-columns: 28px 1fr;
    }
    .ob-actions {
      grid-column: 2;
    }
  }
  .offline {
    display: grid;
    justify-items: center;
    gap: var(--space-3);
    padding: var(--space-16) var(--space-6);
    text-align: center;
    background: var(--bg-panel);
    border: 1px solid color-mix(in srgb, var(--danger-solid) 35%, transparent);
    border-radius: var(--radius-lg);
  }
  .off-mark {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--danger-muted);
    color: var(--danger-text);
  }
  .offline h2 {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
  }
  .offline p {
    max-width: 440px;
    color: var(--text-secondary);
  }
  .offline .off-detail {
    color: var(--text-tertiary);
    font-size: var(--type-xs-size);
  }
  .off-actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  /* Two independent stacks, so a short side panel never leaves a hole in a shared row. */
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
  .live,
  .live-sk {
    min-width: 0;
  }
  .scroll {
    overflow-x: auto;
  }
  .live-sk {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-4);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .lsk-head {
    display: grid;
    gap: var(--space-2);
  }
  .lsk-body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: var(--space-4);
  }
  .lsk-stats {
    display: grid;
    gap: var(--space-3);
    align-content: start;
  }
  .lsk-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }
  .list-sk {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-4);
  }
  .rsk {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) 72px minmax(0, 1fr) 48px;
    align-items: center;
    gap: var(--space-4);
  }
  .rsk-name {
    display: grid;
    gap: 4px;
  }
  @media (max-width: 1023px) {
    .lsk-body {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .run-name {
    display: block;
    font-weight: var(--weight-medium);
  }
  .run-name:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .run-sub {
    display: block;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    line-height: var(--type-xs-line);
  }
  .table td {
    height: 48px;
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 120px;
  }
  .prog span {
    font-size: var(--type-xs-size);
    min-width: 46px;
    text-align: right;
  }
  .models {
    padding: 0 var(--space-2) var(--space-2);
  }
  .models a {
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
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .m-map {
    display: grid;
    justify-items: end;
  }
  .m-map .num {
    font-weight: var(--weight-medium);
  }
  .m-cap {
    font-size: var(--type-micro-size);
    color: var(--text-tertiary);
  }
  .dslist {
    padding: 0 var(--space-2) var(--space-2);
  }
  .dslist a {
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
  @media (max-width: 1199px) {
    .cols {
      grid-template-columns: minmax(0, 1fr);
    }
    /* Laptop: the rail's two lists side by side under the main column. */
    .col-side {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: start;
    }
  }
  @media (max-width: 767px) {
    .col-side {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .hide-sm {
      display: none;
    }
  }
</style>
