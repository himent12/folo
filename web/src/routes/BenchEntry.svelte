<script lang="ts">
  /**
   * SEAM (the entry's result page, built next): one entry of a benchmark. For now: who it is, what it was graded on,
   * where it stands, and its score with the five axes; the full result (accuracy by class and size, each condition's
   * curve, the reliability of its confidences, the label curve, its images) follows. Kept live by the benchmark's
   * stream, so a running entry shows its progress and lands on its score.
   */
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { LiveBench } from '$lib/api/liveBench.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { isEntryActive } from '$lib/bench/rules';
  import { ago, date } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import { media } from '$lib/design/tokens';
  import { MediaQuery } from 'svelte/reactivity';
  import Button from '$lib/ui/Button.svelte';
  import CodeText from '$lib/ui/CodeText.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import ScoreBreakdown from './bench/ScoreBreakdown.svelte';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { contestantFacts, coverageFacts, entryStatus, progressFacts, progressShare, scoreText } from './bench/words';
  import Target from '@lucide/svelte/icons/target';

  const id = $derived(router.route.params.id ?? '');
  const phone = new MediaQuery(media.phone, false);
  const entryId = $derived(router.route.params.entryId ?? '');
  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const live = new LiveBench(() => id || null);
  const bench = $derived(live.bench);
  const entry = $derived(live.entries.find((e) => e.id === entryId));
  const unsupported = $derived(cannotBench(system.data, live.error));

  $effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench?.name ?? id, path: `/benchmarks/${id}` }, { label: entry?.name ?? entryId });
  });
</script>

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
  <div class="sk-page" aria-busy="true" aria-label="Loading entry">
    <Skeleton w="min(240px, 60%)" h="24px" />
    <Skeleton w="min(360px, 90%)" h="14px" />
    <Skeleton h="280px" r="var(--radius-lg)" />
  </div>
{:else if !entry}
  <Page title="Entry">
    {#snippet actions()}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>{bench.name}</Button>{/snippet}
    <EmptyState icon={Target} title="Entry not found" description="{bench.name} has no entry {entryId}: it may have been deleted." />
  </Page>
{:else}
  {@const c = contestantFacts(entry, agents.data)}
  {@const f = coverageFacts(entry, bench)}
  <Page title={entry.name}>
    {#snippet sub()}
      <Meta>
        <span><StatusBadge plain state={entryStatus(entry)} /></span>
        {#if c.arch}<Ident value={c.arch} />{/if}
        {#each c.facts as x (x)}<span>{x}</span>{/each}
        {#each f as x (x)}<span>{x}</span>{/each}
        <span class="faint" title={date(entry.created)}>added {ago(entry.created)}</span>
      </Meta>
    {/snippet}
    <!-- Phones: the top bar's back leads to the benchmark already. -->
    {#snippet actions()}{#if !phone.current}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>Leaderboard</Button>{/if}{/snippet}

    <div class="grid">
      <Panel title="Score" meta={entry.score?.partial ? 'partial' : undefined}>
        {#if entry.score}
          <div class="hero">
            <span class="big num">{scoreText(entry.score.total)}</span>
            <span class="of">of 100{#if entry.score.partial}, over the axes it was measured on: it compares only loosely with complete entries{/if}</span>
          </div>
          <ScoreBreakdown score={entry.score} />
        {:else if isEntryActive(entry)}
          {@const p = progressFacts(entry, catalog.data)}
          <div class="run">
            <Progress value={progressShare(entry)} size="md" tone={live.lost !== null ? 'neutral' : 'accent'} label="{entry.name} progress" />
            <p class="muted num"><Meta><span>{p.where}</span>{#if p.pct}<span>{p.pct}</span>{/if}</Meta></p>
            <p class="faint">The score appears here when the entry is done.</p>
          </div>
        {:else}
          <p class="muted"><CodeText text={entry.error ?? 'Stopped before the end: nothing was scored.'} /></p>
          {#if entry.hint}<p class="faint"><CodeText text={entry.hint} /></p>{/if}
        {/if}
      </Panel>
      <Panel title="Result">
        <p class="muted measure">The full result follows here: accuracy by class and by object size, how much of it each condition takes away, whether its confidences can be trusted, what a confidence threshold costs in labels, and its worst images.</p>
      </Panel>
    </div>
  </Page>
{/if}

<style>
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .hero {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    flex-wrap: wrap;
    margin-bottom: var(--space-3);
  }
  .big {
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--type-3xl-weight);
  }
  .of {
    color: var(--text-secondary);
  }
  .run {
    display: grid;
    gap: var(--space-2);
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-4);
  }
  @media (max-width: 1023px) {
    .grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
  }
</style>
