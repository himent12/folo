<script lang="ts">
  /**
   * One self-improvement loop (`#/benchmarks/{id}/loops/{loopId}`; docs/BENCH.md, Self-improvement loops): a page of its
   * own, since a loop has a chart, a verdict, each generation's provenance and actions, and a link to it is shared. It
   * states the question it answers in one line; draws mAP@0.5:0.95 per generation with each one's 95% interval as
   * whiskers, generation 0's interval as a band across (a generation above the band improved on the first teacher);
   * reads the verdict in words, with BENCH.md's caveat one tap away; and lists per generation what happened: the
   * teacher's labels, the dataset written from them, the training run, the model and its entry, each a link where the
   * thing exists. A running loop shows the step its current generation is on, live; a failed or stopped one why.
   */
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Check from '@lucide/svelte/icons/check';
  import CircleSlash from '@lucide/svelte/icons/circle-slash';
  import Repeat from '@lucide/svelte/icons/repeat';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { LiveBench } from '$lib/api/liveBench.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { BenchGeneration } from '$lib/api/types';
  import { isLoopActive } from '$lib/bench/rules';
  import { fitFloorNote } from '$lib/charts/domain';
  import XYChart from '$lib/charts/XYChart.svelte';
  import { media } from '$lib/design/tokens';
  import { ago, conf, date, metric, num, sentence } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import { pageCommands, type PageCommand } from '$lib/shell/pageCommands.svelte';
  import Button from '$lib/ui/Button.svelte';
  import CodeText from '$lib/ui/CodeText.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import Callout from './run/Callout.svelte';
  import { generationDomain, generationPoints } from './bench/charts';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { intervalText, labelsLine, LOOP_CAVEAT, loopQuestion, loopStatus, loopSteps, loopVerdictText, progressFacts, scoreText, VERDICT } from './bench/words';

  const id = $derived(router.route.params.id ?? '');
  const loopId = $derived(router.route.params.loopId ?? '');
  const phone = new MediaQuery(media.phone, false);
  const system = new Resource((s) => hub.api.system(s));
  const models = new Resource((s) => hub.api.models(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));
  const runs = new Resource((s) => hub.api.runs(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const live = new LiveBench(() => id || null);
  const bench = $derived(live.bench);
  const loop = $derived(live.loops.find((l) => l.id === loopId));
  const unsupported = $derived(cannotBench(system.data, live.error));

  $effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench?.name ?? id, path: `/benchmarks/${id}?tab=loops` }, { label: loop?.name ?? loopId });
  });

  const planned = $derived(loop?.config.generations ?? 1);
  const teacherName = $derived(loop?.generations[0]?.model_name ?? models.data?.find((m) => m.id === loop?.config.teacher)?.name ?? loop?.config.teacher ?? '');
  const poolName = $derived.by(() => {
    const p = loop?.config.pool;
    if (!p) return '';
    if (p.kind === 'dataset') {
      const d = datasets.data?.find((x) => x.id === p.dataset_id)?.name ?? p.dataset_id;
      return `the ${({ train: 'training', val: 'validation', test: 'test' } as const)[p.split ?? 'train']} split of ${d}`;
    }
    return `the images of ${jobs.data?.find((j) => j.id === p.job_id)?.name ?? 'a labeling job'}`;
  });
  const student = $derived(loop?.config.student?.arch ?? null);
  const known = <T extends { id: string }>(list: readonly T[] | undefined, x: string | null) => !!x && !!list?.some((y) => y.id === x);
  const entryKnown = (x: string | null) => !!x && live.entries.some((e) => e.id === x);

  // ---- the chart ----
  const PLOT = 220;
  const fit = $derived(loop ? generationDomain(loop.generations, PLOT) : null);
  const first = $derived(loop?.generations[0]);
  let cursor = $state<number | null>(null);
  const genWord = (g: number) => (g === 0 ? 'Teacher' : `Gen ${g}`);
  const genAt = (g: number) => loop?.generations.find((x) => x.generation === g);

  // ---- running: the steps of the generation it is on, each done only once what it makes is there (./bench/words) ----
  const current = $derived(loop && isLoopActive(loop) ? loop.generations.at(-1) : undefined);
  const now = $derived(loop ? loopSteps(loop) : null);
  const nowPhase = $derived(now?.steps.find((s) => s.state === 'now')?.phase ?? null);
  /** Nothing graded yet (generation 0 still benchmarking): the chart says so instead of standing empty. */
  const graded = $derived(!!loop?.generations.some((g) => g.map50_95 !== null));

  // ---- stop, delete ----
  let stopping = $state(false);
  let stopBusy = $state(false);
  const stopStatus = new DialogStatus();
  async function stop() {
    if (!loop) return;
    stopBusy = true;
    const mark = live.mark();
    try {
      live.putLoop(await hub.api.stopBenchLoop(id, loop.id), mark);
      stopping = false;
    } catch (e) {
      stopStatus.error(e, 'Could not stop the loop');
    } finally {
      stopBusy = false;
    }
  }
  let removing = $state(false);
  let removeBusy = $state(false);
  const removeStatus = new DialogStatus();
  async function remove() {
    if (!loop) return;
    const l = loop;
    removeBusy = true;
    try {
      await hub.api.deleteBenchLoop(id, l.id);
      live.dropLoop(l.id);
      removing = false;
      // page-level: its confirm dialog has closed on the line above.
      toast.success(`Deleted ${l.name}`, 'Its datasets, runs, models and entries stay.');
      router.go(`/benchmarks/${id}?tab=loops`);
    } catch (e) {
      removeStatus.error(e, 'Could not delete the loop');
    } finally {
      removeBusy = false;
    }
  }
  const menu = $derived<MenuItem[]>(loop && !isLoopActive(loop) ? [{ label: 'Delete loop', icon: Trash, danger: true, onclick: () => (removing = true) }] : []);
  pageCommands((): PageCommand[] => {
    const l = loop;
    if (!l) return [];
    const out: PageCommand[] = [{ id: 'bench-loop-back', label: 'Back to the loops', icon: ArrowLeft, run: () => router.go(`/benchmarks/${id}?tab=loops`) }];
    if (isLoopActive(l) && l.status !== 'stopping') out.push({ id: 'bench-loop-stop', label: `Stop loop ${l.name}`, icon: Square, run: () => (stopping = true) });
    return out;
  });
  /** The entry of the generation being graded, for where it is ("Waiting for the GPU", "Haze, severity 3 · 54%"). */
  const nowEntry = $derived(current?.entry_id ? live.entries.find((e) => e.id === current.entry_id) : undefined);
  /** Why a generation has no labels or model while the loop no longer runs. */
  const ended = $derived(loop ? (loop.status === 'failed' ? 'failed' : loop.status === 'stopped' ? 'stopped' : null) : null);
</script>

{#snippet caveatWord()}What it does not show{/snippet}

{#snippet readout(c: number | null, hint: string | null)}
  {#if hint}<span class="hint">{hint}</span>
  {:else if c === null && !graded}
    <span class="hint">No generation graded yet: the first point is the teacher’s, once its benchmark entry is done.</span>
  {:else if c === null}
    <span class="key"><i class="dot" aria-hidden="true"></i>mAP50-95 per generation, with its 95% interval</span>
    {#if first?.ci95}<span class="key gap"><i class="bandk" aria-hidden="true"></i>the first teacher’s interval</span>{/if}
  {:else}
    {@const g = genAt(c)}
    <span class="t">{c === 0 ? 'Generation 0, the first teacher' : `Generation ${c}`}</span><span class="sep">·</span>
    {#if g?.map50_95 != null}
      <span class="v">mAP50-95 {metric(g.map50_95)}</span>{#if g.ci95}<span class="sep">·</span><span>{intervalText(g.ci95)}</span>{/if}{#if g.score !== null}<span class="sep">·</span><span>score {scoreText(g.score)}</span>{/if}
    {:else}
      <span>{g ? 'not benchmarked yet' : ended ? 'not run' : 'to come'}</span>
    {/if}
  {/if}
{/snippet}

{#snippet labelsCell(g: BenchGeneration)}
  {#if g.generation === 0}
    <span class="faint">Its own training labels</span>
  {:else if g.labels}
    <!-- One plain line, so a generation that kept little is seen at a glance; a thin one says so in the warning ink. -->
    {@const ln = labelsLine(g.labels)}
    <span class="ln" class:thin={ln.thin}>{#if ln.thin}<TriangleAlert size={13} strokeWidth={1.75} aria-hidden="true" />{/if}<span><span class="cap-first">{ln.text}</span>{#if ln.thin}<span class="sr-only">: a thin dataset</span>{/if}</span></span>
  {:else if ended && g === loop?.generations.at(-1)}
    <span class="faint">None: the loop {ended} here</span>
  {:else if current === g && nowPhase === 'labeling'}
    <NoValue label="being labeled now" />
  {:else}
    <NoValue label="not labeled yet" />
  {/if}
{/snippet}

{#if unsupported}
  <Page title="Benchmarks"><Unsupported /></Page>
{:else if live.error && !bench}
  <Page title="Loop">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/benchmarks')}>All benchmarks</Button>{/snippet}
    {#if live.error instanceof ApiError && live.error.status === 404}
      <ErrorState error={live.error} title="Benchmark not found" />
    {:else}
      <ErrorState error={live.error} title="Could not load this loop" retry={() => live.retry()} />
    {/if}
  </Page>
{:else if !bench || live.loading}
  <div class="sk-page" aria-busy="true" aria-label="Loading loop">
    <div class="sk-head"><Skeleton w="min(240px, 60%)" h="24px" /><Skeleton w="min(520px, 90%)" h="14px" /></div>
    <div class="sk-grid"><Skeleton h="320px" r="var(--radius-lg)" /><Skeleton h="320px" r="var(--radius-lg)" /></div>
    <Skeleton h="200px" r="var(--radius-lg)" />
  </div>
{:else if !loop}
  <Page title="Loop">
    {#snippet actions()}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}?tab=loops`)}>Loops</Button>{/snippet}
    <EmptyState icon={Repeat} title="Loop not found" description="{bench.name} has no loop {loopId}: it may have been deleted." />
  </Page>
{:else}
  <Page title={loop.name} inlineActions>
    {#snippet sub()}
      <Meta>
        <span><StatusBadge plain state={loopStatus(loop)} /></span>
        <span>teacher {#if known(models.data, loop.config.teacher)}<a class="link" href={href(`/models/${loop.config.teacher}`)}>{teacherName}</a>{:else}{teacherName}{/if}</span>
        {#if student}<span>students <Ident value={student} /></span>{/if}
        <span class="num">{planned} {planned === 1 ? 'generation' : 'generations'} of {num(loop.config.student?.epochs ?? 30)} epochs</span>
        <span class="faint" title={date(loop.created)}>started {ago(loop.created)}</span>
      </Meta>
    {/snippet}
    {#snippet subPhone()}<span>{loopStatus(loop).label} · {planned} {planned === 1 ? 'generation' : 'generations'}</span>{/snippet}
    {#snippet actions()}
      {#if !phone.current}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}?tab=loops`)}>Loops</Button>{/if}
      {#if isLoopActive(loop)}
        {#if phone.current}
          <Button variant="danger" icon={Square} disabled={loop.status === 'stopping'} onclick={() => (stopping = true)} aria-label="Stop {loop.name}" />
        {:else}
          <Button variant="danger" icon={Square} disabled={loop.status === 'stopping'} onclick={() => (stopping = true)}>{loop.status === 'stopping' ? 'Stopping' : 'Stop loop'}</Button>
        {/if}
      {/if}
      {#if menu.length}<Menu label="More loop actions" items={menu} />{/if}
    {/snippet}

    <p class="question">{loopQuestion(teacherName, poolName, student, planned)}</p>

    {#if loop.status === 'failed'}
      <Callout tone="danger" icon={TriangleAlert} title="It failed at generation {loop.generations.at(-1)?.generation ?? 0}" role="alert">
        <CodeText text={sentence(loop.error ?? 'The loop failed.')} />{#if loop.hint}{' '}<CodeText text={sentence(loop.hint)} />{/if}
      </Callout>
    {:else if loop.status === 'stopped'}
      <Callout tone="neutral" icon={CircleSlash} title="Stopped at generation {loop.generations.at(-1)?.generation ?? 0}" role="note">
        The generations it finished are kept below; it has no verdict.
      </Callout>
    {/if}

    <div class="top" class:running={!!current}>
      <Panel title="mAP50-95 by generation" meta={fit ? (fitFloorNote(fit) ?? 'unchanged images') : undefined}>
        {#if fit}
          <XYChart
            label="mAP50-95 of each generation with its 95% interval; the first teacher's interval as a band"
            series={[{ id: 'map', label: 'mAP50-95', color: 'var(--text-primary)', points: generationPoints(loop.generations), dots: true }]}
            xDomain={[-0.4, planned + 0.4]}
            yDomain={fit.domain}
            xTicks={Array.from({ length: planned + 1 }, (_, g) => g)}
            yTicks={fit.ticks}
            xFormat={genWord}
            yFormat={(v) => v.toFixed(fit.decimals)}
            height={PLOT + 40}
            band={first?.ci95 ? { lo: first.ci95[0], hi: first.ci95[1], label: 'first teacher’s 95% interval' } : null}
            stops={Array.from({ length: planned + 1 }, (_, g) => g)}
            bind:cursor
            {readout}
            table={{
              caption: 'mAP50-95 and score per generation',
              head: ['Generation', 'mAP50-95', '95% interval', 'Score'],
              rows: Array.from({ length: planned + 1 }, (_, g) => {
                const x = genAt(g);
                return [genWord(g), x?.map50_95 != null ? metric(x.map50_95) : '—', x?.ci95 ? intervalText(x.ci95) : '—', x?.score != null ? scoreText(x.score) : '—'];
              }),
            }}
          />
        {/if}
      </Panel>

      <div class="side">
        {#if current && now}
          <!-- Generation 0 is one step (the benchmark grades the first teacher); a student's generation three. A step
               reads done only once what it makes is in the generation (./bench/words loopSteps). -->
          <Panel title="Now" meta="generation {current.generation} of {planned}">
            <ol class="steps" aria-label="The steps of generation {current.generation}">
              {#each now.steps as s (s.phase)}
                <li class={s.state}>
                  <span class="mark" aria-hidden="true">{#if s.state === 'done'}<Check size={12} strokeWidth={2} />{/if}</span>
                  <span class="s-name">{s.name}</span>
                  <span class="s-state">{s.state === 'done' ? 'done' : s.state === 'now' ? (loop.status === 'stopping' ? 'stopping' : 'now') : 'next'}</span>
                </li>
              {/each}
            </ol>
            <p class="now-links">
              {#if nowPhase === 'training'}{#if current.run_id}Training run: {#if known(runs.data, current.run_id)}<a class="link" href={href(`/runs/${current.run_id}`)}>{loop.name} gen {current.generation}</a>{:else}<Ident value={current.run_id} />{/if}.{:else}Its training run is being made.{/if}{/if}
              {#if nowPhase === 'benchmarking' && current.entry_id && nowEntry}
                {@const p = progressFacts(nowEntry, catalog.data)}
                Its entry: <a class="link" href={href(`/benchmarks/${bench.id}/entries/${current.entry_id}`)}>{nowEntry.name}</a>, {(p.where[0] ?? '').toLowerCase() + p.where.slice(1)}{#if p.pct && nowEntry.status === 'running'}{' '}· <span class="num">{p.pct}</span>{/if}.
              {/if}
              {#if nowPhase === 'labeling'}The teacher labels the pool by itself; nothing it labels is checked against truth.{/if}
            </p>
          </Panel>
        {/if}
        <Panel title="Verdict">
          {#snippet actions()}
            <InfoButton label="What a loop does not show" trigger={caveatWord} text={LOOP_CAVEAT} />
          {/snippet}
          {#if loop.verdict}
            <p class="v-word">{VERDICT[loop.verdict].word}</p>
            <!-- The word is the heading: the sentence goes on from it. -->
            <p class="v-text">{(loopVerdictText(loop) ?? '').replace(/^[^:]+: (\w)/, (_, c: string) => c.toUpperCase())}</p>
          {:else}
            <p class="v-text faint">{isLoopActive(loop) ? `Once the last generation is benchmarked: above the first teacher’s interval${first?.ci95 ? ` (${intervalText(first.ci95)})` : ''} is improved, inside it held, below it regressed.` : 'It did not finish, so it has no verdict.'}</p>
          {/if}
          <p class="v-how">Read on mAP50-95 of the unchanged images; a generation inside the band is not clearly different from the first teacher.</p>
        </Panel>
      </div>
    </div>

    <Panel title="Generations" meta="what each one was made of" pad="none">
      <div class="scroll">
        <table class="table gens" aria-label="Generations of {loop.name}">
          <thead>
            <tr>
              <th class="lead w-gen">Generation</th>
              <th>Model</th>
              <th class="hide-sm">Labels it trained on</th>
              <th class="hide-md w-ds">Dataset · run</th>
              <th class="r hide-sm w-score">Score</th>
              <th class="r w-map">mAP50-95<span class="ci-h">{' · interval'}</span></th>
            </tr>
          </thead>
          <tbody>
            {#each Array.from({ length: planned + 1 }, (_, g) => g) as k (k)}
              {@const g = genAt(k)}
              <tr class:future={!g}>
                <td class="lead num">{k === 0 ? 'Teacher' : `Gen ${k}`}</td>
                <td class="mdl">
                  {#if g?.model_name}
                    {#if g.entry_id && entryKnown(g.entry_id)}<a class="link" href={href(`/benchmarks/${bench.id}/entries/${g.entry_id}`)}>{g.model_name}</a>{:else}{g.model_name}{/if}
                    {#if g.model_id && known(models.data, g.model_id)}<span class="sub"><a class="quiet" href={href(`/models/${g.model_id}`)}>model</a></span>{/if}
                  {:else if g && ended}
                    <span class="faint">Not trained</span>
                  {:else if g}
                    <NoValue label={current === g && nowPhase === 'training' ? 'training now' : 'trains once labeled'} />
                  {:else}
                    <span class="faint">{ended ? 'Not run' : 'to come'}</span>
                  {/if}
                  {#if g?.labels}
                    <!-- Phones, where the labels column steps out: the same one line under the model. -->
                    {@const ln = labelsLine(g.labels)}
                    <span class="sub ph-lbl" class:thin={ln.thin}><span class="cap-first">{ln.text}</span></span>
                  {/if}
                </td>
                <td class="hide-sm lbl">{#if g}{@render labelsCell(g)}{:else}<span class="faint">—</span>{/if}</td>
                <td class="hide-md">
                  {#if g?.labels}
                    <Meta>
                      <span>{#if known(datasets.data, g.labels.dataset_id)}<a class="link" href={href(`/datasets/${g.labels.dataset_id}`)}>dataset</a>{:else if datasets.data}<span class="faint" title="No longer on this server">dataset deleted</span>{:else}<span class="faint">dataset</span>{/if}</span>
                      {#if g.run_id}<span>{#if known(runs.data, g.run_id)}<a class="link" href={href(`/runs/${g.run_id}`)}>run</a>{:else if runs.data}<span class="faint" title="No longer on this server">run deleted</span>{:else}<span class="faint">run</span>{/if}</span>{/if}
                    </Meta>
                  {:else}<span class="faint">{k === 0 ? 'none' : '—'}</span>{/if}
                </td>
                <td class="r num hide-sm">{#if g?.score != null}{scoreText(g.score)}{:else}<NoValue label="not yet" />{/if}</td>
                <td class="r num">{#if g?.map50_95 != null}<span class="m">{metric(g.map50_95)}</span>{#if g.ci95}<span class="ci">{intervalText(g.ci95)}</span>{/if}{:else}<NoValue label="not yet" />{/if}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </Panel>
    <p class="list-foot">Its datasets are named “{loop.name} gen 1”, …: they, its runs, models and entries stay when the loop is deleted. The pool’s own labels were never read.</p>
  </Page>

  <Dialog open={stopping} onclose={() => (stopping = false)} title="Stop {loop.name}?" description="It stops what the loop is doing now (its entry or its run). The generations it finished, and their datasets, runs and models, stay." size="sm" status={stopStatus}>
    {#snippet footer()}
      <Button onclick={() => (stopping = false)}>Keep running</Button>
      <Button variant="destructive" icon={Square} loading={stopBusy} onclick={stop}>Stop loop</Button>
    {/snippet}
  </Dialog>
  <Dialog open={removing} onclose={() => (removing = false)} title="Delete {loop.name}?" description="Removes the loop from this benchmark. Its datasets, runs, models and entries stay." size="sm" status={removeStatus}>
    {#snippet footer()}
      <Button onclick={() => (removing = false)}>Cancel</Button>
      <Button variant="destructive" icon={Trash} loading={removeBusy} onclick={remove}>Delete loop</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
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
</style>
