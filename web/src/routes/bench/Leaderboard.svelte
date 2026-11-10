<script lang="ts">
  /**
   * A benchmark's leaderboard (docs/BENCH.md, The score): its entries ranked by score, each with who it is, the score
   * as the row's figure with its five axes at a glance (an axis not measured is an empty dashed slot, not a zero),
   * mAP@0.5:0.95 with its 95% interval as a dot and whiskers on one scale for every row (the leader's interval a faint
   * band behind them all, so an interval that overlaps it reads as "not clearly different"), what it was graded on
   * where that is less than the whole benchmark, and its speed or what its calls cost. Entries that run show where
   * they are; failed ones their cause and the fix. Finished entries can be picked (2–4) for Compare, as on Runs.
   */
  import Plus from '@lucide/svelte/icons/plus';
  import Square from '@lucide/svelte/icons/square';
  import Target from '@lucide/svelte/icons/target';
  import Trash from '@lucide/svelte/icons/trash-2';
  import type { Bench, BenchCatalog, BenchEntry, LabelAgent } from '$lib/api/types';
  import { AXES, AXIS_NAME, AXIS_WEIGHT, isEntryActive } from '$lib/bench/rules';
  import { metric, pct } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import CodeText from '$lib/ui/CodeText.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import { rowNav } from '$lib/ui/rowNav';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import IntervalPlot from './IntervalPlot.svelte';
  import { compareHint, finished, intervalDomain, overlaps, ranks, toggleSelection } from './leaderboard';
  import type { BenchResults } from './results.svelte';
  import ScoreAxes from './ScoreAxes.svelte';
  import { contestantFacts, costFacts, coverageFacts, entryStatus, partialWhy, progressFacts, progressShare, scoreText } from './words';

  let {
    bench,
    entries,
    results,
    catalog,
    agents,
    selected = $bindable([]),
    lost = false,
    onadd,
    onstop,
    ondelete,
  }: {
    bench: Bench;
    entries: readonly BenchEntry[];
    results: BenchResults;
    catalog?: BenchCatalog;
    agents?: readonly LabelAgent[];
    /** Entries picked for Compare (finished ones, at most four). */
    selected?: string[];
    /** The stream is down: progress is as last heard. */
    lost?: boolean;
    onadd: () => void;
    onstop: (e: BenchEntry) => void;
    ondelete: (e: BenchEntry) => void;
  } = $props();

  const rank = $derived(ranks(entries));
  const ci = (e: BenchEntry) => results.get(e.id)?.accuracy.ci95;
  const domain = $derived(intervalDomain(entries.filter(finished).flatMap((e) => (ci(e) ? [ci(e)!] : []))));
  const leader = $derived(entries.find(finished));
  const band = $derived(leader ? (ci(leader) ?? null) : null);
  const ranRun = (e: BenchEntry) => results.get(e.id)?.robustness?.conditions.length;
  const active = $derived(entries.filter((e) => isEntryActive(e)).length);
  const picking = $derived(selected.length > 0);
  const toggle = (id: string) => (selected = toggleSelection(selected, id, entries));

  function items(e: BenchEntry): MenuItem[] {
    return [
      ...(finished(e) ? [{ label: 'Open result', icon: Target, href: href(`/benchmarks/${bench.id}/entries/${e.id}`) }] : []),
      ...(isEntryActive(e) && e.status !== 'stopping' ? [{ label: 'Stop entry', icon: Square, onclick: () => onstop(e) }] : []),
      ...(!isEntryActive(e) ? [{ label: 'Delete entry', icon: Trash, danger: true, onclick: () => ondelete(e) }] : []),
    ];
  }
  const open = (e: BenchEntry) => router.go(`/benchmarks/${bench.id}/entries/${e.id}`);
  /** Whose interval overlaps the leader's: not clearly different from it on this benchmark. */
  const tied = (e: BenchEntry) => !!band && e.id !== leader?.id && !!ci(e) && overlaps(ci(e)!, band);
  const SCORE_HELP = AXES.map((a) => `${AXIS_NAME[a]} ${pct(AXIS_WEIGHT[a], 0)}`).join(' · ');
</script>

{#snippet howWords()}How the score is made{/snippet}

{#snippet who(e: BenchEntry)}
  {@const c = contestantFacts(e, agents)}
  <Meta>{#if c.arch}<Ident value={c.arch} />{/if}{#each c.facts as f (f)}<span>{f}</span>{/each}</Meta>
{/snippet}

{#snippet coverage(e: BenchEntry)}
  {@const f = coverageFacts(e, bench, ranRun(e))}
  {#if f.length}<Meta>{#each f as x (x)}<span>{x}</span>{/each}</Meta>{:else}<span class="faint">Complete</span>{/if}
{/snippet}

<Panel title="Leaderboard" meta={entries.length ? `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}` : undefined} pad="none">
  {#snippet actions()}
    <InfoButton
      label="How the score is made"
      trigger={howWords}
      text="One number, 0 to 100: the weighted mean of five axes, each 0–1, over the ones that were measured."
      how={`${SCORE_HELP}. An axis that could not be measured is left out and the entry is partial, as is one that ran part of the images.`}
    />
  {/snippet}
  {#if !entries.length}
    <EmptyState
      icon={Target}
      title="No entries yet"
      description="An entry is one contestant's run of this benchmark: a model of this server, Claude Code or Codex looking at the images, or any other model's predictions. The first entry fixes the plan, so look over the conditions first."
    >
      {#snippet actions()}<Button icon={Plus} onclick={onadd}>Add entry</Button>{/snippet}
    </EmptyState>
  {:else}
    <div class="scroll">
      <table class="table tall board" class:picking aria-label="Leaderboard of {bench.name}" use:rowNav={{ toggle: (row) => toggle(row.dataset.id ?? '') }}>
        <thead>
          <tr>
            <th class="check"><span class="sr-only">Select for comparison</span></th>
            <th class="r rank">#</th>
            <th class="lead">Entry</th>
            <th class="r score">Score</th>
            <th class="axes-col" use:tooltip={'Accuracy · robustness · self-labeling · calibration · localization'}>Axes</th>
            <th class="map">
              <span class="map-head">
                <span class="mh-label" title="mAP50-95 on the unchanged images, with its 95% interval">mAP50-95</span>
                {#if domain}
                  <span class="scale" aria-label="95% interval, on a scale from {domain[0].toFixed(2)} to {domain[1].toFixed(2)}">
                    <span class="tick">{domain[0].toFixed(2)}</span>
                    <span class="tick-mid">95% interval</span>
                    <span class="tick">{domain[1].toFixed(2)}</span>
                  </span>
                {/if}
              </span>
            </th>
            <th class="hide-md ran">Graded on</th>
            <th class="r hide-lg cost">Speed · cost</th>
            <th class="act"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#each entries as e (e.id)}
            {@const on = selected.includes(e.id)}
            {@const done = finished(e)}
            {@const st = entryStatus(e)}
            {@const cost = costFacts(e)}
            <tr class="link" class:on class:faded={!done} data-row data-id={e.id} aria-label={e.name} onclick={() => open(e)}>
              <td class="check" onclick={(ev) => ev.stopPropagation()}>
                {#if done}
                  <label class="hit">
                    <input type="checkbox" checked={on} disabled={!on && selected.length >= 4} aria-label="Select {e.name} for comparison" onchange={() => toggle(e.id)} />
                  </label>
                {/if}
              </td>
              <td class="r rank num">{#if rank.has(e.id)}{rank.get(e.id)}{:else}<NoValue label="not ranked" />{/if}</td>
              <td class="lead">
                <a class="name" href={href(`/benchmarks/${bench.id}/entries/${e.id}`)} onclick={(ev) => ev.stopPropagation()} title={e.name}>{e.name}</a>
                <span class="sub">{@render who(e)}</span>
              </td>
              {#if done && e.score}
                <td class="r score">
                  <span class="score-v num">
                    {#if e.score.partial}<span class="partial" use:tooltip={partialWhy(e, bench)}>partial</span>{/if}{scoreText(e.score.total)}
                  </span>
                </td>
                <td class="axes-col"><ScoreAxes axes={e.score.axes} /></td>
                <td class="map">
                  <span class="map-cell">
                    <span class="map-v num">{metric(e.map50_95)}</span>
                    {#if domain && ci(e)}
                      <span class="plot" use:tooltip={`95% interval ${metric(ci(e)![0])}–${metric(ci(e)![1])}${tied(e) ? ': overlaps the leader’s, so not clearly different' : ''}`}>
                        <IntervalPlot value={e.map50_95 ?? 0} ci={ci(e)!} {domain} {band} height={47} />
                      </span>
                    {:else}
                      <span class="plot"></span>
                    {/if}
                  </span>
                </td>
              {:else}
                <td class="r score"><StatusBadge plain state={st} /></td>
                <td class="state-cell" colspan="2">
                  {#if isEntryActive(e)}
                    {@const p = progressFacts(e, catalog)}
                    <span class="prog">
                      {#if e.status === 'running'}<Progress value={progressShare(e)} size="xs" tone={lost ? 'neutral' : 'accent'} label="{e.name} progress" />{/if}
                      <span class="prog-text"><Meta line><span>{p.where}</span>{#if p.pct && e.status === 'running'}<span class="num">{p.pct}</span>{/if}</Meta></span>
                    </span>
                  {:else if e.status === 'failed'}
                    <span class="fail"><span class="fail-why"><CodeText text={e.error ?? 'The entry failed.'} /></span>{#if e.hint}<span class="fail-hint"><CodeText text={e.hint} /></span>{/if}</span>
                  {:else}
                    <span class="faint">Stopped before the end: nothing was scored.</span>
                  {/if}
                </td>
              {/if}
              <td class="hide-md ran">{@render coverage(e)}</td>
              <td class="r hide-lg cost">
                {#if cost && done}<span class="num">{cost.value}</span><span class="sub">{cost.sub}</span>{:else if e.contestant.kind === 'external' && done}<NoValue label="not measured for an external model" />{:else}<NoValue label="not yet" />{/if}
              </td>
              <td class="act" onclick={(ev) => ev.stopPropagation()}>
                <Menu label="Actions for {e.name}" items={items(e)} variant="ghost" fixed />
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    <!-- Phones: rank and name, who it is and what it was graded on; the score (or the status) on the right. -->
    <ul class="rows" aria-label="Leaderboard of {bench.name}">
      {#each entries as e (e.id)}
        {@const done = finished(e)}
        {@const f = coverageFacts(e, bench, ranRun(e))}
        <li class:faded={!done}>
          <a href={href(`/benchmarks/${bench.id}/entries/${e.id}`)}>
            <span class="m-rank num">{rank.get(e.id) ?? ''}</span>
            <span class="m-main">
              <span class="name truncate">{e.name}</span>
              <span class="m-meta">{@render who(e)}</span>
              {#if done && f.length}<span class="m-meta"><Meta>{#each f as x (x)}<span>{x}</span>{/each}</Meta></span>{/if}
              {#if isEntryActive(e)}<span class="m-meta"><Meta><span>{progressFacts(e, catalog).where}</span>{#if e.status === 'running'}<span class="num">{progressFacts(e, catalog).pct}</span>{/if}</Meta></span>{/if}
              {#if e.status === 'failed'}<span class="m-cause" data-fact><CodeText text={e.error ?? 'The entry failed.'} /></span>{/if}
            </span>
              <span class="m-side">
              {#if done && e.score}
                <span class="num m-score">{#if e.score.partial}<span class="m-partial">partial</span>{/if}{scoreText(e.score.total)}</span>
                <ScoreAxes axes={e.score.axes} />
              {:else}
                <StatusBadge plain state={entryStatus(e)} />
              {/if}
            </span>
          </a>
          {#if e.status === 'running'}<span class="m-prog" aria-hidden="true" style:transform="scaleX({progressShare(e)})"></span>{/if}
        </li>
      {/each}
    </ul>
  {/if}
</Panel>
{#if entries.length}
  <p class="list-foot num">
    <Meta>
      <span>{entries.length} {entries.length === 1 ? 'entry' : 'entries'}</span>
      {#if active}<span>{active} running or waiting</span>{/if}
      {#if band}<span class="wide-only">the band is the leader's 95% interval: an interval that overlaps it is not clearly different</span>{/if}
      <!-- Tablets: Compare's prerequisite under the board, since the header keeps one row of actions. -->
      {#if entries.filter(finished).length >= 2}<span class="tablet-only" data-gate-why>{compareHint(selected.length)}</span>{/if}
    </Meta>
  </p>
{/if}

<style>
  .scroll {
    overflow-x: auto;
  }
  /* Fixed columns on the header row; the entry's name takes what is left. */
  .board .check {
    position: relative;
    width: 32px;
    min-width: 32px;
    padding: 0;
  }
  .board :is(th, td).rank {
    width: 32px;
    padding-left: 0;
    padding-right: var(--space-2);
    color: var(--text-tertiary);
  }
  .board th.score {
    width: 104px;
  }
  .board .axes-col {
    width: 80px;
  }
  .board th.map {
    width: 240px;
  }
  .board th.ran {
    width: 224px;
  }
  /* What it was graded on wraps whole facts onto a second line rather than widen the column. */
  .board td.ran {
    white-space: normal;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  /* The interval's cell: the plot runs the row's full height, so the leader's band reads as one column down the board. */
  .board td.map {
    height: 48px;
    padding-top: 0;
    padding-bottom: 0;
  }
  .board th.cost {
    width: 112px;
  }
  .board .act {
    width: 40px;
    padding: 0 var(--space-2) 0 0;
    text-align: right;
  }
  .board td.lead {
    max-width: 0;
  }
  .name {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* A row that has not finished reads one step quieter in its name: it has no place on the board yet. */
  .board tr.faded .name {
    color: var(--text-secondary);
  }
  .hit {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    padding-left: var(--space-3);
    cursor: pointer;
  }
  input[type='checkbox'] {
    width: 14px;
    height: 14px;
    margin: 0;
    accent-color: var(--text-primary);
    cursor: pointer;
  }
  .board .check input {
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  .board tbody tr:is(:hover, :focus-within, .on) .check input,
  .board.picking .check input,
  .board .check input:focus-visible {
    opacity: 1;
  }
  @media (pointer: coarse) {
    .board .check {
      width: 44px;
    }
    .board .check input {
      opacity: 1;
      width: 20px;
      height: 20px;
    }
  }
  /* The score: the row's figure, a step up from the cells; "partial" a tertiary word before it, as "best" is on Models. */
  .score-v {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    font-weight: var(--weight-semibold);
  }
  .partial {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .map-head {
    display: grid;
    grid-template-columns: 64px minmax(0, 1fr);
    align-items: center;
    gap: var(--space-3);
  }
  .mh-label {
    text-align: right;
  }
  .scale {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    color: var(--text-tertiary);
    /* small: chart tick */
    font-size: var(--type-xs-size);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  .tick-mid {
    color: var(--text-tertiary);
  }
  .map-cell {
    display: grid;
    grid-template-columns: 64px minmax(0, 1fr);
    align-items: center;
    gap: var(--space-3);
    height: 100%;
  }
  .map-v {
    text-align: right;
    color: var(--text-secondary);
  }
  .plot {
    display: block;
    align-self: stretch;
    min-width: 0;
  }
  .ran,
  .cost {
    color: var(--text-secondary);
  }
  .state-cell {
    max-width: 0;
  }
  .prog {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 6px;
  }
  .prog-text {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .fail {
    display: grid;
    white-space: normal;
  }
  .fail-why {
    color: var(--text-secondary);
  }
  .fail-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .rows {
    display: none;
  }
  /* Laptops: the plot and the facts columns narrow, so who an entry is stays on one line. */
  @media (max-width: 1279px) {
    .wide-only {
      display: none;
    }
    .board th.score {
      width: 96px;
    }
    .board th.map {
      width: 200px;
    }
    .board th.ran {
      width: 184px;
    }
    /* The scale keeps its two ends; what it is, the column's title says. */
    .tick-mid {
      display: none;
    }
  }
  .tablet-only {
    display: none;
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .tablet-only {
      display: inline;
    }
  }
  /* Tablets: the plot narrows with its column. */
  @media (max-width: 1023px) {
    .board th.map {
      width: 200px;
    }
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows li {
      position: relative;
    }
    .rows li + li {
      border-top: 1px solid var(--line-subtle);
    }
    .rows a {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    .rows a:active {
      background: var(--bg-hover);
    }
    .m-rank {
      flex-shrink: 0;
      width: 16px;
      align-self: flex-start;
      padding-top: 2px;
      color: var(--text-tertiary);
      text-align: right;
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
      gap: 2px;
    }
    .rows .name {
      font-weight: var(--weight-medium);
    }
    .rows li.faded .name {
      color: var(--text-secondary);
    }
    .m-meta,
    .m-cause {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
    .m-cause {
      color: var(--text-secondary);
    }
    .m-side {
      flex-shrink: 0;
      display: grid;
      justify-items: end;
      gap: 4px;
    }
    .m-score {
      font-size: var(--type-lg-size);
      line-height: var(--type-lg-line);
      font-weight: var(--weight-semibold);
    }
    .m-partial {
      margin-right: var(--space-2);
      font-size: var(--type-sm-size);
      font-weight: var(--weight-regular);
      color: var(--text-tertiary);
    }
    .m-prog {
      position: absolute;
      left: 0;
      right: 0;
      bottom: -1px;
      height: 2px;
      transform-origin: left;
      background: var(--accent-line);
    }
  }
</style>
