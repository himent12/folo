<script lang="ts">
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Download from '@lucide/svelte/icons/download';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import { hub } from '$lib/api/index.svelte';
  import { synthEvaluation } from '$lib/api/mock/evaluation';
  import { archInfo } from '$lib/api/mock/runs';
  import { Resource } from '$lib/api/resource.svelte';
  import type { ClassMetrics, Dataset, Model, Run } from '$lib/api/types';
  import ConfusionMatrix from '$lib/charts/ConfusionMatrix.svelte';
  import { classColor } from '$lib/design/tokens';
  import { ago, bytes, compact, date, metric, ms, num } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import MetricBar from './models/MetricBar.svelte';
  import ExportDialog from './model/ExportDialog.svelte';
  import PrCurves from './model/PrCurves.svelte';

  const id = $derived(router.route.params.id ?? '');
  const model = new Resource((s) => hub.api.model(id, s));

  $effect(() => {
    setCrumbs({ label: 'Models', path: '/models' }, { label: model.data?.name ?? id });
  });

  // Evaluation context from the source run: per-class AP and the validation class counts of its dataset.
  interface Context {
    run: Run | null;
    dataset: Dataset | null;
    perClass: ClassMetrics[] | null;
    instances: number[] | null;
  }
  const runId = $derived(model.data?.run_id ?? null);
  const target = $derived(model.data?.metrics?.map50_95 ?? null);
  const context = new Resource<Context>(async (s) => {
    const rid = runId;
    const best = target;
    const none: Context = { run: null, dataset: null, perClass: null, instances: null };
    if (!rid) return none;
    try {
      const [run, metrics] = await Promise.all([hub.api.run(rid, s), hub.api.runMetrics(rid, s)]);
      // The epoch these weights came from (matched by mAP), else the last one.
      const ep = metrics.epochs.find((e) => best !== null && Math.abs(e.val.map50_95 - best) < 1e-4) ?? metrics.epochs.at(-1);
      let dataset: Dataset | null = null;
      try {
        dataset = await hub.api.dataset(run.config.dataset_id, s);
      } catch {
        /* the dataset may have been unregistered; fall back to synthetic counts */
      }
      const all = dataset ? dataset.splits.train + dataset.splits.val + dataset.splits.test : 0;
      const share = dataset && all ? dataset.splits.val / all : 0;
      return {
        run,
        dataset,
        perClass: ep?.val.per_class ?? null,
        instances: dataset?.stats && share ? dataset.stats.class_counts.map((c) => c * share) : null,
      };
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      return none;
    }
  });

  const ev = $derived.by(() => {
    const m = model.data;
    const c = context.data;
    if (!m || !c) return null;
    return synthEvaluation({ seed: m.id, classes: m.classes, metrics: m.metrics, perClass: c.perClass, instances: c.instances });
  });

  // Large class sets (COCO) get a readable matrix of the most frequent classes.
  const CM_MAX = 12;
  const cm = $derived.by(() => {
    const m = model.data;
    if (!m || !ev) return null;
    const n = m.classes.length;
    if (n <= CM_MAX) return { classes: m.classes, matrix: ev.confusion, partial: false };
    const keep = ev.perClass
      .map((c) => c)
      .sort((a, b) => b.instances - a.instances)
      .slice(0, CM_MAX)
      .map((c) => c.class)
      .sort((a, b) => a - b);
    const idx = [...keep, n];
    return {
      classes: keep.map((i) => m.classes[i] ?? `class ${i}`),
      matrix: idx.map((r) => idx.map((c) => ev.confusion[r]?.[c] ?? 0)),
      partial: true,
    };
  });

  let exportOpen = $state(false);

  const latency = (m: Model) => archInfo[m.arch].ms * (m.imgsz / 640) ** 2;
  const tiles = (m: Model) => [
    { label: 'mAP50-95', value: m.metrics?.map50_95, sub: 'IoU 0.50:0.95', hero: true },
    { label: 'mAP50', value: m.metrics?.map50, sub: 'IoU 0.50', hero: false },
    { label: 'Precision', value: m.metrics?.precision, sub: 'boxes, all classes', hero: false },
    { label: 'Recall', value: m.metrics?.recall, sub: 'boxes, all classes', hero: false },
  ];
</script>

{#if model.error && !model.data}
  <Page title="Model">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/models')}>All models</Button>{/snippet}
    <Panel><ErrorState error={model.error} retry={() => model.refresh()} title="Could not load this model" compact /></Panel>
  </Page>
{:else if !model.data}
  <Page title="Model">
    {#snippet lead()}<Skeleton w="180px" h="20px" />{/snippet}
    <div class="tiles">{#each [0, 1, 2, 3] as i (i)}<Skeleton h="104px" r="var(--radius-lg)" />{/each}</div>
    <div class="row"><Skeleton h="380px" r="var(--radius-lg)" /><Skeleton h="380px" r="var(--radius-lg)" /></div>
  </Page>
{:else}
  {@const m = model.data}
  {@const run = context.data?.run}
  <Page title={m.name}>
    {#snippet lead()}
      <Badge tone="outline" mono>{m.arch}</Badge>
      {#if !m.run_id}<Badge>imported</Badge>{/if}
    {/snippet}
    {#snippet sub()}
      <span class="mono faint">{m.id}</span>
      <span class="dot" aria-hidden="true"></span>
      <span class="num">{m.classes.length} classes · {m.imgsz}px · {bytes(m.bytes, prefs.units)}</span>
      <span class="dot" aria-hidden="true"></span>
      {#if m.run_id}
        <span>trained by <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a></span>
        <span class="dot" aria-hidden="true"></span>
      {/if}
      <span title={date(m.created)}>created {ago(m.created)}</span>
    {/snippet}
    {#snippet actions()}
      <Button icon={Download} onclick={() => (exportOpen = true)}>Export</Button>
      <Button variant="primary" icon={ScanSearch} href={href(`/playground?model=${encodeURIComponent(m.id)}`)}>Try in Playground</Button>
    {/snippet}

    <div class="tiles">
      {#each tiles(m) as t (t.label)}
        <div class="tile" class:hero={t.hero}>
          <span class="t-label">{t.label}</span>
          <span class="t-value num">{metric(t.value)}</span>
          <div class="t-bar" aria-hidden="true"><span style:transform="scaleX({t.value ?? 0})"></span></div>
          <span class="t-sub">{t.sub}</span>
        </div>
      {/each}
    </div>

    <div class="row">
      <Panel title="Precision–recall" meta="IoU 0.5" id="pr">
        {#if ev}
          <PrCurves classes={m.classes} {ev} map50={m.metrics?.map50 ?? null} />
        {:else}
          <Skeleton h="300px" />
        {/if}
      </Panel>
      <Panel title="Confusion matrix" meta={cm?.partial ? `top ${CM_MAX} of ${m.classes.length} classes` : undefined} id="cm">
        {#if cm}
          <ConfusionMatrix classes={cm.classes} matrix={cm.matrix} label="Confusion matrix of {m.name}" />
        {:else}
          <Skeleton h="300px" />
        {/if}
      </Panel>
    </div>

    <div class="row">
      <Panel title="Per class" meta={String(m.classes.length)} pad="none" id="per-class">
        {#if ev}
          <div class="scroll" class:tall={m.classes.length > 14}>
            <table class="table dense">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col" class="r">AP50</th>
                  <th scope="col" class="r">AP50-95</th>
                  <th scope="col" class="r hide-sm">Precision</th>
                  <th scope="col" class="r hide-sm">Recall</th>
                  <th scope="col" class="r">Instances</th>
                </tr>
              </thead>
              <tbody>
                {#each ev.perClass as c (c.class)}
                  <tr>
                    <td>
                      <span class="cls">
                        <span class="sw" style:background={classColor(c.class)} aria-hidden="true"></span>
                        <span class="truncate">{m.classes[c.class]}</span>
                      </span>
                    </td>
                    <td class="r"><MetricBar value={c.ap50} /></td>
                    <td class="r">{metric(c.ap50_95)}</td>
                    <td class="r hide-sm muted">{metric(c.precision)}</td>
                    <td class="r hide-sm muted">{metric(c.recall)}</td>
                    <td class="r muted">{num(c.instances)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="foot">
            {#if context.data?.perClass}AP from the run's validation epoch.{:else}Per-class AP estimated from the model's summary metrics.{/if}
            Curves, confusions and instances are estimated until the server exposes evaluation data.
          </p>
        {:else}
          <div class="sk">{#each [0, 1, 2, 3] as i (i)}<Skeleton h="18px" />{/each}</div>
        {/if}
      </Panel>

      <Panel title="Details" id="details">
        <dl class="details">
          <dt>Architecture</dt>
          <dd class="mono">{m.arch}</dd>
          <dt>Input size</dt>
          <dd class="num">{m.imgsz} × {m.imgsz}px</dd>
          <dt>Parameters</dt>
          <dd class="num">{compact(m.params)} <span class="faint">({m.params.toLocaleString('en-US')})</span></dd>
          <dt>Compute</dt>
          <dd class="num">{num(m.flops_g, 1)} GFLOPs</dd>
          <dt>File size</dt>
          <dd class="num">{bytes(m.bytes, prefs.units)}</dd>
          <dt>Typical latency</dt>
          <dd class="num">
            <span use:tooltip={'Published hint for this architecture on a desktop GPU; measure yours in the Playground.'}>≈ {ms(latency(m))}</span>
          </dd>
          <dt>Classes</dt>
          <dd class="classes">
            {#each m.classes.slice(0, 10) as c, i (i)}
              <span class="chip"><span class="sw" style:background={classColor(i)} aria-hidden="true"></span>{c}</span>
            {/each}
            {#if m.classes.length > 10}<span class="faint">+{m.classes.length - 10} more</span>{/if}
          </dd>
          <dt>Source</dt>
          <dd>
            {#if m.run_id}
              <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a>
              {#if context.data?.dataset}<span class="faint">{' on '}</span><a class="link" href={href(`/datasets/${context.data.dataset.id}`)}>{context.data.dataset.name}</a>{/if}
            {:else}
              <span class="muted">Imported .folo file</span>
            {/if}
          </dd>
          <dt>Created</dt>
          <dd>{date(m.created)}</dd>
        </dl>
      </Panel>
    </div>
  </Page>
  <ExportDialog model={m} bind:open={exportOpen} />
{/if}

<style>
  .dot {
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: var(--text-tertiary);
  }
  .link {
    color: var(--text-primary);
  }
  .link:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .tiles {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-4);
  }
  .tile {
    display: grid;
    gap: var(--space-1);
    padding: var(--space-4);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    min-width: 0;
  }
  :global([data-theme='light']) .tile {
    border-color: var(--line-default);
  }
  .t-label {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .t-value {
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .hero .t-value {
    color: var(--accent-text);
  }
  .t-bar {
    position: relative;
    height: 2px;
    margin: var(--space-1) 0 2px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .t-bar span {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--text-tertiary);
    transition: transform var(--dur-chart) var(--ease-out);
  }
  .hero .t-bar span {
    background: var(--accent-line);
  }
  .t-sub {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: stretch;
  }
  .scroll {
    overflow-x: auto;
  }
  .scroll.tall {
    max-height: 480px;
    overflow-y: auto;
  }
  .scroll.tall thead th {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
    z-index: 1;
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    max-width: 200px;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .foot {
    margin-top: auto;
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .sk {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  .details {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: var(--space-3) var(--space-4);
    align-items: baseline;
  }
  dt {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  dd {
    min-width: 0;
    text-align: right;
  }
  .classes {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-1);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 6px;
    border-radius: var(--radius-xs);
    background: var(--bg-active);
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
  }
  @media (max-width: 1199px) {
    .row {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .tiles {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-2);
    }
    .tile {
      padding: var(--space-3);
    }
    .t-value {
      font-size: var(--type-2xl-size);
      line-height: var(--type-2xl-line);
    }
    .hide-sm {
      display: none;
    }
  }
</style>
