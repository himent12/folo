<script lang="ts">
  /**
   * SEAM (the compare page, built next): two to four finished entries of one benchmark side by side
   * (`#/benchmarks/{id}/compare?entries=a,b`, the leaderboard's Compare). For now their scores and axes in one table,
   * each axis's best marked with a word; the comparison's charts (per condition, reliability, labels) follow.
   */
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { LiveBench } from '$lib/api/liveBench.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { AXIS_NAME } from '$lib/bench/rules';
  import { metric, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import { media } from '$lib/design/tokens';
  import { MediaQuery } from 'svelte/reactivity';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { compareIds, finished } from './bench/leaderboard';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { scoreText } from './bench/words';

  const id = $derived(router.route.params.id ?? '');
  const phone = new MediaQuery(media.phone, false);
  const ids = $derived(compareIds(router.route.query));
  const system = new Resource((s) => hub.api.system(s));
  const live = new LiveBench(() => id || null);
  const bench = $derived(live.bench);
  const unsupported = $derived(cannotBench(system.data, live.error));
  const shown = $derived(ids.map((x) => live.entries.find((e) => e.id === x)).filter((e) => !!e && finished(e)));

  $effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench?.name ?? id, path: `/benchmarks/${id}` }, { label: 'Compare' });
  });

  /** The best value of a row (ties all marked), or none for a row with fewer than two values. */
  function best(values: (number | null)[]): Set<number> {
    const v = values.filter((x): x is number => x !== null);
    if (v.length < 2) return new Set();
    const top = Math.max(...v);
    return new Set(values.flatMap((x, i) => (x === top ? [i] : [])));
  }
  const rows = $derived.by(() => {
    const score = shown.map((e) => e!.score!.total);
    // The accuracy axis is mAP50-95: one row, read as the leaderboard prints it.
    const axes = (shown[0]?.score?.axes ?? []).map((a, k) => ({
      label: a.axis === 'accuracy' ? 'Accuracy · mAP50-95' : AXIS_NAME[a.axis],
      values: shown.map((e) => e!.score!.axes[k]?.value ?? null),
      fmt: a.axis === 'accuracy' ? metric : (x: number) => num(x, 2),
    }));
    return [{ label: 'Score', values: score, fmt: scoreText }, ...axes];
  });
</script>

{#if unsupported}
  <Page title="Benchmarks"><Unsupported /></Page>
{:else if live.error && !bench}
  <Page title="Compare entries">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/benchmarks')}>All benchmarks</Button>{/snippet}
    {#if live.error instanceof ApiError && live.error.status === 404}
      <ErrorState error={live.error} title="Benchmark not found" />
    {:else}
      <ErrorState error={live.error} title="Could not load the entries" retry={() => live.retry()} />
    {/if}
  </Page>
{:else if !bench || live.loading}
  <div class="sk-page" aria-busy="true" aria-label="Loading comparison"><Skeleton w="min(240px, 60%)" h="24px" /><Skeleton h="280px" r="var(--radius-lg)" /></div>
{:else}
  <Page title="Compare entries">
    {#snippet sub()}<span>{bench.name} · {shown.length} {shown.length === 1 ? 'entry' : 'entries'} on one fingerprint</span>{/snippet}
    <!-- Phones: the top bar's back leads to the benchmark already. -->
    {#snippet actions()}{#if !phone.current}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>Leaderboard</Button>{/if}{/snippet}
    {#if shown.length < 2}
      <EmptyState icon={GitCompare} title="Pick two to four entries" description="Select finished entries on the leaderboard, then Compare." />
    {:else}
      <Panel title="Side by side" pad="none">
        <div class="scroll">
          <table class="table cmp" aria-label="Entries side by side">
            <thead>
              <tr>
                <th class="lead" style:width="200px"><span class="sr-only">Measure</span></th>
                {#each shown as e (e!.id)}<th class="r">{e!.name}</th>{/each}
              </tr>
            </thead>
            <tbody>
              {#each rows as row (row.label)}
                {@const b = best(row.values)}
                <tr>
                  <td class="lead">{row.label}</td>
                  {#each row.values as v, i (i)}
                    <td class="r num" class:top={b.has(i)}>{#if v === null}<NoValue label="not measured" />{:else}{#if b.has(i)}<span class="best">best</span>{/if}{row.fmt(v)}{/if}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </Panel>
      <p class="faint measure">Charts of the comparison follow here: each condition's curve, the reliability of the confidences, and the labels each would make.</p>
    {/if}
  </Page>
{/if}

<style>
  .scroll {
    overflow-x: auto;
  }
  /* The entries share the width equally, the measure's names a fixed column. */
  .cmp {
    table-layout: fixed;
  }
  .top {
    font-weight: var(--weight-medium);
  }
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-4);
  }
</style>
