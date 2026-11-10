<script lang="ts">
  /**
   * One entry of a benchmark (`#/benchmarks/{id}/entries/{entryId}`): its whole result, read top to bottom as
   * docs/BENCH.md's "Reading it" asks (how good is it, why that number, where does it break, can I trust its confidence,
   * could it label for itself, what does it get wrong). The header says who it is and what it was graded on; the score
   * leads, then a section per part of the result, which a sticky strip under the top bar jumps between (the page is long;
   * on a phone the same strip scrolls sideways). Kept live by the benchmark's stream: a queued or running entry shows
   * where it is and lands on its result when it is done; a failed or stopped one says why and what to do.
   */
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Box from '@lucide/svelte/icons/box';
  import Copy from '@lucide/svelte/icons/copy';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import Square from '@lucide/svelte/icons/square';
  import Target from '@lucide/svelte/icons/target';
  import Trash from '@lucide/svelte/icons/trash-2';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import CircleSlash from '@lucide/svelte/icons/circle-slash';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { LiveBench } from '$lib/api/liveBench.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { BenchEntry, BenchLoop } from '$lib/api/types';
  import { isEntryActive } from '$lib/bench/rules';
  import { media } from '$lib/design/tokens';
  import { ago, date, num, sentence, usd } from '$lib/format';
  import { AGENT_NAME } from '$lib/label';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands.svelte';
  import Button from '$lib/ui/Button.svelte';
  import CodeText from '$lib/ui/CodeText.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import Callout from './run/Callout.svelte';
  import AccuracySection from './bench/entry/AccuracySection.svelte';
  import CalibrationSection from './bench/entry/CalibrationSection.svelte';
  import ImagesSection from './bench/entry/ImagesSection.svelte';
  import LocalizationSection from './bench/entry/LocalizationSection.svelte';
  import RobustnessSection from './bench/entry/RobustnessSection.svelte';
  import RunningPanel from './bench/entry/RunningPanel.svelte';
  import ScoreSection from './bench/entry/ScoreSection.svelte';
  import SelfLabelSection from './bench/entry/SelfLabelSection.svelte';
  import JumpNav from './bench/JumpNav.svelte';
  import { ranks } from './bench/leaderboard';
  import NewLoopDialog from './bench/NewLoopDialog.svelte';
  import { modelLabel } from './label/words';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { axisWhyNot, costFacts, coverageFacts, entryStatus, scoreText, teacherEligibility } from './bench/words';

  const id = $derived(router.route.params.id ?? '');
  const entryId = $derived(router.route.params.entryId ?? '');
  const phone = new MediaQuery(media.phone, false);
  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const models = new Resource((s) => hub.api.models(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));
  const live = new LiveBench(() => id || null);
  const bench = $derived(live.bench);
  const entry = $derived(live.entries.find((e) => e.id === entryId));
  const unsupported = $derived(cannotBench(system.data, live.error));
  // The result, once the entry is done: asked again when it lands (a finished result never changes).
  const done = $derived(entry?.status === 'done');
  const result = new Resource((s) => (done ? hub.api.benchResult(id, entryId, s) : Promise.resolve(undefined)));

  $effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench?.name ?? id, path: `/benchmarks/${id}` }, { label: entry?.name ?? entryId });
  });

  const model = $derived(entry?.contestant.kind === 'model' ? models.data?.find((m) => m.id === (entry.contestant as { model_id: string }).model_id) : undefined);
  const rank = $derived.by(() => {
    if (!entry || !done) return null;
    const r = ranks(live.entries);
    const at = r.get(entry.id);
    return at ? { at, of: r.size } : null;
  });
  /** Model entries ahead of a queued one (they run one at a time, oldest first). */
  const ahead = $derived(
    entry?.status === 'queued' ? live.entries.filter((e) => e.contestant.kind === 'model' && e.id !== entry.id && isEntryActive(e) && (e.status !== 'queued' || e.created < entry.created)) : [],
  );

  // ---- the next step: a loop with this model as the first teacher ----
  let loopOpen = $state(false);
  const loop = $derived.by(() => {
    if (!entry || !bench || entry.contestant.kind !== 'model') return null;
    if (models.data && !model) return { why: 'Its model is no longer on this server, so it cannot teach.', start: () => {} };
    if (!model) return { why: 'Loading models', start: () => {} };
    const e = teacherEligibility(model, bench);
    return { why: e.ok ? null : `${e.why}, so it cannot teach on this benchmark.`, start: () => (loopOpen = true) };
  });
  function loopStarted(l: BenchLoop) {
    live.putLoop(l, live.mark());
    // page-level: the dialog has closed.
    toast.success(`Loop ${l.name} started`, 'Its first student trains on this model’s labels.', { label: 'Open loop', run: () => router.go(`/benchmarks/${id}/loops/${l.id}`) });
  }

  // ---- stop, delete ----
  let stopping = $state(false);
  let stopBusy = $state(false);
  const stopStatus = new DialogStatus();
  async function stop() {
    if (!entry) return;
    stopBusy = true;
    const mark = live.mark();
    try {
      live.putEntry(await hub.api.stopBenchEntry(id, entry.id), mark);
      stopping = false;
    } catch (e) {
      stopStatus.error(e, 'Could not stop the entry');
    } finally {
      stopBusy = false;
    }
  }
  let removing = $state(false);
  let removeBusy = $state(false);
  const removeStatus = new DialogStatus();
  async function remove() {
    if (!entry) return;
    const e = entry;
    removeBusy = true;
    try {
      await hub.api.deleteBenchEntry(id, e.id);
      live.dropEntry(e.id);
      removing = false;
      // page-level: its confirm dialog has closed on the line above.
      toast.success(`Deleted ${e.name}`, 'Its result is gone from the leaderboard. The model and its files were not touched.');
      router.go(`/benchmarks/${id}`);
    } catch (err) {
      removeStatus.error(err, 'Could not delete the entry');
    } finally {
      removeBusy = false;
    }
  }
  const menu = $derived.by((): MenuItem[] => {
    const e = entry;
    if (!e) return [];
    return [
      ...(model ? [{ label: 'Open model', icon: Box, href: href(`/models/${model.id}`) }] : []),
      { label: 'Copy entry id', icon: Copy, onclick: () => void copyText(e.id, 'entry id') },
      ...(isEntryActive(e) && e.status !== 'stopping' ? [{ label: 'Stop entry', icon: Square, danger: true, onclick: () => (stopping = true) }] : []),
      ...(!isEntryActive(e) ? [{ label: 'Delete entry', icon: Trash, danger: true, onclick: () => (removing = true) }] : []),
    ];
  });
  pageCommands((): PageCommand[] => {
    const e = entry;
    if (!e) return [];
    const out: PageCommand[] = [{ id: 'bench-entry-board', label: 'Back to the leaderboard', icon: ArrowLeft, run: () => router.go(`/benchmarks/${id}`) }];
    if (loop && !loop.why) out.push({ id: 'bench-entry-loop', label: `Start a loop with ${e.name} as teacher`, icon: RotateCcw, run: loop.start });
    if (isEntryActive(e) && e.status !== 'stopping') out.push({ id: 'bench-entry-stop', label: `Stop entry ${e.name}`, icon: Square, run: () => (stopping = true) });
    return out;
  });

  const SECTIONS = [
    { id: 'score', label: 'Score' },
    { id: 'accuracy', label: 'Accuracy' },
    { id: 'robustness', label: 'Robustness' },
    { id: 'calibration', label: 'Calibration' },
    { id: 'self-labeling', label: 'Self-labeling' },
    { id: 'localization', label: 'Localization' },
    { id: 'images', label: 'Images' },
  ];
</script>

{#snippet who(e: BenchEntry)}
  {@const c = e.contestant}
  {#if c.kind === 'model'}
    {#if model}<Ident value={c.arch} href={href(`/models/${model.id}`)} title="Open {model.name}" />{:else}<span title={models.data ? 'The model is no longer on this server' : undefined}><Ident value={c.arch} /></span>{/if}
    <span class="num">{num(c.params / 1e6, 2)}M params</span>
    <span class="num">{c.imgsz} px</span>
    {#if models.data && !model}<span>model deleted since</span>{/if}
  {:else if c.kind === 'agent'}
    <span>{AGENT_NAME[c.agent]}</span>
    <span>{modelLabel(c.model, agents.data?.find((a) => a.id === c.agent))}</span>
    <span>{c.effort} effort</span>
  {:else}
    <span>External</span>
    <!-- Without confidences, the coverage facts after it say so. -->
    {#if e.scored}<span>with confidences</span>{/if}
  {/if}
{/snippet}

{#snippet asked(e: BenchEntry)}
  <!-- What it was asked to run and how far it got: the facts to add it again with. -->
  <Panel title="What it was asked to run">
    <dl class="facts">
      <div><dt>Images</dt><dd class="num">{num(e.images)} of {num(bench?.images ?? e.images)}</dd></div>
      <div><dt>Conditions</dt><dd>{!bench?.plan.conditions.length ? 'None in the plan' : e.conditions ? `All ${bench.plan.conditions.length} of the plan` : 'Unchanged images only'}</dd></div>
      <div><dt>Got through</dt><dd class="num">{num(e.progress.done)} of {num(e.progress.total)} image passes</dd></div>
      {#if e.usage}<div><dt>Calls</dt><dd class="num">{num(e.usage.calls)}{#if e.usage.cost_usd !== null}{' '}· {usd(e.usage.cost_usd)}{/if}</dd></div>{/if}
      <div><dt>Added</dt><dd title={date(e.created)}>{ago(e.created)}</dd></div>
      {#if e.finished}<div><dt>Ended</dt><dd title={date(e.finished)}>{ago(e.finished)}</dd></div>{/if}
    </dl>
  </Panel>
{/snippet}

{#if unsupported}
  <Page title="Benchmarks"><Unsupported /></Page>
{:else if live.error && !bench}
  <Page title="Entry">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/benchmarks')}>All benchmarks</Button>{/snippet}
    {#if live.error instanceof ApiError && live.error.status === 404}
      <ErrorState error={live.error} title="Benchmark not found" />
    {:else}
      <ErrorState error={live.error} title="Could not load this entry" retry={() => live.retry()} />
    {/if}
  </Page>
{:else if !bench || live.loading}
  <!-- The loaded page's frame: its header, the jump strip, the score panel's hero and rows. -->
  <div class="sk-page" aria-busy="true" aria-label="Loading entry">
    <div class="sk-head"><Skeleton w="min(240px, 60%)" h="24px" /><Skeleton w="min(560px, 90%)" h="14px" /></div>
    <div class="sk-jump">{#each [56, 72, 88, 80, 96, 88, 56] as w, i (i)}<Skeleton w="{w}px" h="14px" />{/each}</div>
    <div class="sk-panel">
      <div class="sk-title"><Skeleton w="64px" h="14px" /></div>
      <div class="sk-score">
        <div class="sk-hero"><Skeleton w="120px" h="36px" /><Skeleton w="160px" h="12px" /></div>
        <div class="sk-rows">{#each [0, 1, 2, 3, 4] as i (i)}<Skeleton h="16px" />{/each}</div>
      </div>
    </div>
  </div>
{:else if !entry}
  <Page title="Entry">
    {#snippet actions()}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>Leaderboard</Button>{/snippet}
    <EmptyState icon={Target} title="Entry not found" description="{bench.name} has no entry {entryId}: it may have been deleted." />
  </Page>
{:else}
  {@const f = coverageFacts(entry, bench)}
  {@const cost = costFacts(entry)}
  <Page title={entry.name} inlineActions>
    {#snippet sub()}
      <Meta>
        <span><StatusBadge plain state={entryStatus(entry)} /></span>
        {@render who(entry)}
        {#each f as x (x)}<span>{x}</span>{/each}
        {#if cost && done}<span class="num" title={cost.sub}>{cost.value}{entry.speed ? ' per image' : ''}</span>{/if}
        <span class="faint" title={date(entry.created)}>added {ago(entry.created)}</span>
      </Meta>
    {/snippet}
    {#snippet subPhone()}
      <span class="num">{entryStatus(entry).label}{#if entry.score}{' '}· score {scoreText(entry.score.total)}{#if entry.score.partial}, partial{/if}{/if}</span>
    {/snippet}
    {#snippet actions()}
      {#if !phone.current}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>Leaderboard</Button>{/if}
      <Menu label="More entry actions" items={menu} />
    {/snippet}

    {#if done && entry.score}
      <JumpNav sections={SECTIONS} label="Sections of {entry.name}'s result" />
      <div class="sections">
        <ScoreSection {entry} {bench} result={result.data} {rank} />
        {#if result.data}
          {@const r = result.data}
          <AccuracySection result={r} {entry} {bench} />
          {#if r.robustness}
            <RobustnessSection result={r} {entry} {bench} catalog={catalog.data} />
          {:else}
            <Panel title="Robustness" meta="accuracy kept under each condition" id="robustness">
              <div class="none">
                <p class="n-title">Not measured</p>
                <p class="n-why">
                  {axisWhyNot('robustness', entry, bench)}
                  {#if bench.plan.conditions.length}
                    {entry.contestant.kind === 'external'
                      ? 'Send its predictions under the plan’s conditions as well (every image at every planned severity) to have this axis.'
                      : `Add the entry again with “Run the plan’s conditions” on: ${bench.plan.conditions.length} conditions, each image again at every severity.`}
                  {/if}
                </p>
                <Button size="sm" href={href(`/benchmarks/${bench.id}?tab=conditions`)}>See the conditions</Button>
              </div>
            </Panel>
          {/if}
          <CalibrationSection result={r} {entry} {bench} />
          <SelfLabelSection result={r} {entry} {loop} />
          <LocalizationSection result={r} />
          <ImagesSection {bench} {entry} bestConf={r.accuracy.best_conf} />
        {:else if result.error}
          <ErrorState error={result.error} title="Could not load the result" retry={() => result.refresh()} />
        {:else}
          {#each [420, 520, 360] as h, i (i)}<Skeleton h="{h}px" r="var(--radius-lg)" />{/each}
        {/if}
      </div>
    {:else if isEntryActive(entry)}
      <RunningPanel {entry} {bench} catalog={catalog.data} lost={live.lost !== null} {ahead} />
    {:else if entry.status === 'failed'}
      <Callout tone="danger" icon={TriangleAlert} title="It failed: nothing was scored" role="alert">
        <CodeText text={sentence(entry.error ?? 'The entry failed.')} />{#if entry.hint}{' '}<CodeText text={entry.hint} />{/if}
        {#snippet actions()}<Button href={href(`/benchmarks/${bench.id}?add=${entry.contestant.kind}`)}>Add it again</Button>{/snippet}
      </Callout>
      {@render asked(entry)}
    {:else}
      <Callout tone="neutral" icon={CircleSlash} title="Stopped before the end: nothing was scored" role="note">
        A stopped entry keeps nothing of what it ran. Add it again to grade it.
        {#snippet actions()}<Button href={href(`/benchmarks/${bench.id}?add=${entry.contestant.kind}`)}>Add it again</Button>{/snippet}
      </Callout>
      {@render asked(entry)}
    {/if}
  </Page>

  <NewLoopDialog bind:open={loopOpen} {bench} models={models.data} datasets={datasets.data} jobs={jobs.data} entries={live.entries} teacher={model?.id ?? null} onstarted={loopStarted} />
  <Dialog open={stopping} onclose={() => (stopping = false)} title="Stop {entry.name}?" description="It ends what is in flight. A stopped entry has no score: what it ran so far is not kept." size="sm" status={stopStatus}>
    {#snippet footer()}
      <Button onclick={() => (stopping = false)}>Keep running</Button>
      <Button variant="destructive" icon={Square} loading={stopBusy} onclick={stop}>Stop entry</Button>
    {/snippet}
  </Dialog>
  <Dialog open={removing} onclose={() => (removing = false)} title="Delete {entry.name}?" description="Removes the entry and its result from the leaderboard. The model, its files and the benchmark's images are not touched." size="sm" status={removeStatus}>
    {#snippet footer()}
      <Button onclick={() => (removing = false)}>Cancel</Button>
      <Button variant="destructive" icon={Trash} loading={removeBusy} onclick={remove}>Delete entry</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
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
</style>
