<script lang="ts">
  /**
   * Two to four finished entries of one benchmark side by side and overlaid (`#/benchmarks/{id}/compare?entries=a,b`):
   * what makes them not like for like, said first (fewer images, conditions left out, no confidences, unknown classes);
   * their scores and five axes against the first (the baseline, chosen in the table's head), the best of each row
   * marked; mAP50-95 with each 95% interval on one scale and a plain sentence on whether they are told apart; the
   * accuracy each keeps per condition as an aligned dot plot (who wins where); their reliability curves and their labels'
   * precision against coverage overlaid; and AP per class as aligned dots. Each entry keeps one hue and one name on
   * every chart (RunCompare's slots: sky, amber, magenta dashed, teal dotted; a dot plot's shapes), and its legend chip
   * keeps its line in front. Entries are added and removed here, the URL following, on phones too.
   */
  import { untrack } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import Scale from '@lucide/svelte/icons/scale';
  import Plus from '@lucide/svelte/icons/plus';
  import X from '@lucide/svelte/icons/x';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { LiveBench } from '$lib/api/liveBench.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { BenchEntry } from '$lib/api/types';
  import { AXES, AXIS_NAME, TARGET_PRECISION } from '$lib/bench/rules';
  import XYChart from '$lib/charts/XYChart.svelte';
  import type { XYSeries } from '$lib/charts/xy';
  import { media } from '$lib/design/tokens';
  import { conf, metric, num, pct } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Callout from './run/Callout.svelte';
  import { assignSlots, bestAsPrinted, classRows, diffAsPrinted, compareShown, conditionRows, dotDomain, labelTradeoff, RELIABILITY_X, SLOT_PATTERN, SLOTS } from './bench/charts';
  import DotRows, { type DotKey } from './bench/compare/DotRows.svelte';
  import IntervalPlot from './bench/IntervalPlot.svelte';
  import { along, COMPARE_MAX, compareIds, finished, intervalDomain, stale } from './bench/leaderboard';
  import { capTicks } from '$lib/charts/domain';
  import { BenchResults } from './bench/results.svelte';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { compareCaveats, intervalsVerdict, intervalText, keptText, scoreText } from './bench/words';

  const id = $derived(router.route.params.id ?? '');
  const phone = new MediaQuery(media.phone, false);
  const ids = $derived(compareIds(router.route.query));
  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const live = new LiveBench(() => id || null);
  const bench = $derived(live.bench);
  const unsupported = $derived(cannotBench(system.data, live.error));
  const picked = $derived(compareShown(ids.slice(0, COMPARE_MAX), live.entries));
  const shown = $derived(picked.shown);
  const results = new BenchResults(
    () => id || null,
    () => shown,
  );

  $effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench?.name ?? id, path: `/benchmarks/${id}` }, { label: 'Compare' });
  });

  function setIds(next: string[]) {
    router.go(`/benchmarks/${encodeURIComponent(id)}/compare${next.length ? `?entries=${next.map(encodeURIComponent).join(',')}` : ''}`, { replace: true });
  }
  const candidates = $derived(live.entries.filter((e) => finished(e) && !ids.includes(e.id)));

  // Colours follow the entry, not its place: removing one never repaints the others (RunCompare).
  let slots = $state<Record<string, number>>({});
  $effect(() => {
    const cur = shown.map((e) => e.id);
    untrack(() => (slots = assignSlots(cur, slots)));
  });
  const slotOf = (eid: string) => slots[eid] ?? Math.max(0, shown.findIndex((e) => e.id === eid));
  const color = (eid: string) => `var(--chart-${SLOTS[slotOf(eid)] ?? 1})`;
  const fill = (eid: string) => `var(--chart-fill-${SLOTS[slotOf(eid)] ?? 1})`;
  const SHAPES = ['circle', 'square', 'diamond', 'triangle'] as const;
  const keys = $derived<DotKey[]>(shown.map((e) => ({ id: e.id, name: e.name, color: color(e.id), shape: SHAPES[slotOf(e.id)] ?? 'circle' })));

  // The legend: hover or focus a chip to bring its entry forward everywhere; a click keeps it there.
  let hovered = $state<string | null>(null);
  let pinned = $state<string | null>(null);
  const highlight = $derived([hovered, pinned].find((x) => x !== null && shown.some((e) => e.id === x)) ?? null);

  const res = $derived(shown.map((e) => results.get(e.id)));
  const loaded = $derived(shown.length > 0 && res.every((r) => !!r));
  /** An entry whose result could not be loaded: said with a retry, never skeletons forever. */
  const failedResult = $derived(shown.map((e) => ({ e, err: results.error(e.id) })).find((x) => x.err !== undefined));
  const caveats = $derived(bench ? compareCaveats(shown, bench) : []);
  const nameOf = (cid: string) => catalog.data?.conditions.find((c) => c.id === cid)?.name ?? cid;

  // ---- side by side: the score and its axes, against the baseline (the first) ----
  const rows = $derived([
    { key: 'score', label: 'Score', values: shown.map((e) => e.score!.total), fmt: (v: number) => scoreText(v), d: 1 },
    ...AXES.map((a, k) => ({
      key: a,
      label: a === 'accuracy' ? 'Accuracy · mAP50-95' : AXIS_NAME[a],
      values: shown.map((e) => e.score!.axes[k]?.value ?? null),
      fmt: (v: number) => (a === 'accuracy' ? metric(v) : num(v, 2)),
      d: a === 'accuracy' ? 3 : 2,
    })),
  ]);
  /** Entries graded on an earlier version of the benchmark: never the best of a row, never in the intervals' verdict. */
  const old = $derived(shown.map((e) => !!bench && stale(e, bench.fingerprint)));
  const best = (key: string, values: (number | null)[], d: number) =>
    bestAsPrinted(values.map((v, i) => (old[i] ? null : v)), d, key === 'accuracy' ? shown.map((e, i) => (old[i] ? null : e.ci95)) : null);
  const diff = diffAsPrinted;
  const makeBaseline = (eid: string) => setIds([eid, ...ids.filter((x) => x !== eid)]);

  // ---- mAP intervals ----
  const ivDomain = $derived(intervalDomain(shown.flatMap((e) => (e.ci95 ? [e.ci95] : []))));
  const verdict = $derived(intervalsVerdict(shown.filter((_, i) => !old[i]).map((e) => ({ name: e.name, map: e.map50_95 ?? 0, ci: e.ci95 }))));

  // ---- robustness per condition ----
  const condRows = $derived(bench && loaded ? conditionRows(bench.plan.conditions.map((c) => c.condition), res) : []);
  const condDomain = $derived(dotDomain(condRows.flatMap((r) => r.values)));
  const noRobust = $derived(shown.filter((_, i) => !res[i]?.robustness));

  // ---- calibration overlaid ----
  const calSeries = $derived<XYSeries[]>(
    shown.flatMap((e, i) => {
      const c = res[i]?.calibration;
      if (!c) return [];
      const n = c.bins.reduce((s, b) => s + b.count, 0);
      return [
        {
          id: e.id,
          label: e.name,
          color: color(e.id),
          pattern: SLOT_PATTERN[slotOf(e.id)],
          points: c.bins.filter((b) => b.count > 0).map((b) => ({ x: b.confidence!, y: b.precision!, r: 3.5, hollow: b.count < 0.05 * n })),
        },
      ];
    }),
  );
  const noCal = $derived(shown.filter((_, i) => res[i] && !res[i]!.calibration));
  let calCursor = $state<number | null>(null);
  const BIN_MIDS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];

  // ---- self-labeling overlaid: precision of the labels kept against the share of objects they find ----
  const slSeries = $derived<XYSeries[]>(
    shown.flatMap((e, i) => {
      const s = res[i]?.self_labeling;
      if (!s) return [];
      const pts = labelTradeoff(s.curve);
      return [{ id: e.id, label: e.name, color: color(e.id), pattern: SLOT_PATTERN[slotOf(e.id)], points: pts, dots: pts.length === 1, line: pts.length > 1 }];
    }),
  );
  const slMarks = $derived(
    shown.flatMap((e, i) => {
      const s = res[i]?.self_labeling;
      return s && s.recall > 0 ? [{ x: s.recall, y: s.precision, label: '' }] : [];
    }),
  );

  // ---- per class ----
  const clsRows = $derived(bench && loaded ? classRows(bench.classes, bench.class_boxes, res) : []);
  const CLS_SHOWN = 12;
  let allClasses = $state(false);
  const clsShown = $derived(allClasses ? clsRows : clsRows.slice(0, CLS_SHOWN));
  const clsDomain = $derived(dotDomain(clsRows.flatMap((r) => r.values)));
</script>

{#snippet key(eid: string)}<i class="key" style:--c={fill(eid)} data-pattern={SLOT_PATTERN[slotOf(eid)]} aria-hidden="true"></i>{/snippet}

{#snippet adder()}
  <label class="add">
    <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
    <span aria-hidden="true">Add entry…</span>
    <select
      aria-label="Add an entry to the comparison"
      onchange={(ev) => {
        const v = (ev.currentTarget as HTMLSelectElement).value;
        if (v) setIds([...ids, v]);
      }}
    >
      <option value="">Add entry…</option>
      {#each candidates as c (c.id)}<option value={c.id}>{c.name} · score {scoreText(c.score!.total)}</option>{/each}
    </select>
  </label>
{/snippet}

{#snippet legendRead(_c: number | null, hint: string | null)}
  {#if hint}<span class="hint">{hint}</span>{:else}<span class="lr">The entries’ hues and names are the chips above.</span>{/if}
{/snippet}

{#snippet calRead(c: number | null, hint: string | null)}
  {#if hint}<span class="hint">{hint}</span>
  {:else if c === null}
    <span class="lr">Each line one entry’s bins: the share right against what it said; the dashed line is said = right.</span>
  {:else}
    <span class="t">Bin {num(c - 0.05, 1)}–{num(c + 0.05, 1)}</span>
    {#each shown as e, i (e.id)}
      {@const b = res[i]?.calibration?.bins.find((x) => Math.abs(x.lo + 0.05 - c) < 1e-6)}
      <span class="rv">{@render key(e.id)}{b && b.count ? `says ${num(b.confidence!, 2)}, right ${pct(b.precision!, 0)}` : 'none'}</span>
    {/each}
  {/if}
{/snippet}

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
  <div class="sk-page" aria-busy="true" aria-label="Loading comparison">
    <div class="sk-head"><Skeleton w="min(240px, 60%)" h="24px" /><Skeleton w="min(420px, 90%)" h="14px" /></div>
    <div class="sk-chips">{#each [0, 1] as i (i)}<Skeleton h="44px" r="var(--radius-md)" />{/each}</div>
    <Skeleton h="320px" r="var(--radius-lg)" />
    <Skeleton h="240px" r="var(--radius-lg)" />
  </div>
{:else}
  <Page title="Compare entries" inlineActions>
    {#snippet sub()}<span>{bench.name} · {shown.length} {shown.length === 1 ? 'entry' : 'entries'}{old.some((x) => x) ? ', not all on its current fingerprint' : ' on one fingerprint'}</span>{/snippet}
    {#snippet subPhone()}<span>{bench.name} · {shown.length} {shown.length === 1 ? 'entry' : 'entries'}</span>{/snippet}
    {#snippet actions()}{#if !phone.current}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>Leaderboard</Button>{/if}{/snippet}

    <!-- The legend: one chip per entry, its key, its score, open and remove; then Add. -->
    <div class="chips" role="group" aria-label="Entries being compared">
      {#each shown as e (e.id)}
        <!-- svelte-ignore a11y_no_static_element_interactions (the chip's key is the toggle button; pointing only previews it) -->
        <span class="chip" class:pinned={pinned === e.id} style:--c={fill(e.id)} onpointerenter={() => (hovered = e.id)} onpointerleave={() => (hovered = null)} onfocusin={() => (hovered = e.id)} onfocusout={() => (hovered = null)}>
          <button type="button" class="ck" aria-pressed={pinned === e.id} title={pinned === e.id ? 'Let it fall back' : 'Keep it in front'} onclick={() => (pinned = pinned === e.id ? null : e.id)}>
            {@render key(e.id)}<span class="nm">{e.name}</span>
          </button>
          <span class="v num">{scoreText(e.score!.total)}{#if e.score!.partial}<span class="st">{' · partial'}</span>{/if}{#if bench && stale(e, bench.fingerprint)}<span class="st">{' · earlier version'}</span>{/if}</span>
          <a class="act" href={href(`/benchmarks/${bench.id}/entries/${e.id}`)} aria-label="Open {e.name}" title="Open its result"><ArrowUpRight size={14} /></a>
          <button type="button" class="act rm" aria-label="Remove {e.name}" title="Remove from the comparison" onclick={() => setIds(ids.filter((x) => x !== e.id))}><X size={14} /></button>
        </span>
      {/each}
      {#if ids.length < COMPARE_MAX && candidates.length}{@render adder()}{/if}
    </div>

    {#if shown.length < 2}
      <section class="state">
        <EmptyState
          icon={GitCompare}
          title={shown.length ? 'Add one more entry to compare' : 'Pick two to four finished entries'}
          description={picked.missing.length
            ? `${picked.missing.length === 1 ? 'One entry in this link is' : `${picked.missing.length} entries in this link are`} not on ${bench.name} or not finished. Add finished entries here, or pick them on the leaderboard.`
            : `Add finished entries of ${bench.name} here, or pick them on the leaderboard.`}
        >
          {#snippet actions()}<Button icon={ArrowLeft} href={href(`/benchmarks/${bench.id}`)}>Back to the leaderboard</Button>{/snippet}
        </EmptyState>
      </section>
    {:else}
      {#if caveats.length}
        <Callout tone="neutral" icon={Scale} title="Not like for like" role="note">
          <ul class="cav">{#each caveats as c, i (i)}<li>{c.text}</li>{/each}</ul>
        </Callout>
      {/if}

      <Panel title="Side by side" meta="against the first, the baseline" pad="none">
        <div class="scroll">
          <table class="table cmp" aria-label="Scores and axes side by side">
            <thead>
              <tr>
                <th class="lead m-col"><span class="sr-only">Measure</span></th>
                {#each shown as e, i (e.id)}
                  <th class="r e-col">
                    <span class="eh">{@render key(e.id)}<span class="eh-n">{e.name}</span></span>
                    <span class="role">{#if i === 0}Baseline{:else}<button type="button" class="swap" onclick={() => makeBaseline(e.id)}>Make baseline</button>{/if}</span>
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each rows as row (row.key)}
                {@const b = best(row.key, row.values, row.d)}
                <tr class:total={row.key === 'score'}>
                  <td class="lead">{row.label}</td>
                  {#each row.values as v, i (i)}
                    <td class="r num" class:top={b.has(i)}>
                      {#if v === null}<NoValue label="not measured" />{:else}
                        {#if b.has(i)}<span class="best">best</span>{/if}{row.fmt(v)}
                        {#if i > 0}{@const d = old[i] || old[0] ? null : diff(v, row.values[0] ?? null, row.d)}{#if d}<span class="dd">{d}</span>{/if}{/if}
                      {/if}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        <!-- Phones: one block per measure, each entry's value (and its difference from the baseline) in a 2 × 2 grid. -->
        <ul class="sbs" aria-label="Scores and axes, entry by entry">
          {#each rows as row (row.key)}
            {@const b = best(row.key, row.values, row.d)}
            <li>
              <span class="sbs-h">{row.label}</span>
              <span class="sbs-g">
                {#each shown as e, i (e.id)}
                  {@const v = row.values[i] ?? null}
                  <span class="sbs-v" class:top={b.has(i)}>
                    <span class="sbs-n">{@render key(e.id)}<span class="truncate">{e.name}</span></span>
                    <span class="num">{#if v === null}<NoValue label="not measured" />{:else}{row.fmt(v)}{#if i > 0}{@const d = old[i] || old[0] ? null : diff(v, row.values[0] ?? null, row.d)}{#if d}<span class="dd">{d}</span>{/if}{:else}<span class="dd">baseline</span>{/if}{#if b.has(i)}<span class="dd">best</span>{/if}{/if}</span>
                  </span>
                {/each}
              </span>
            </li>
          {/each}
        </ul>
      </Panel>

      <Panel title="mAP50-95 with 95% intervals" meta="unchanged images">
        {#if ivDomain}
          <ul class="iv" aria-label="mAP50-95 and its 95% interval per entry, on one scale">
            {#each shown as e (e.id)}
              <li class:dim={highlight !== null && highlight !== e.id}>
                <span class="iv-n">{@render key(e.id)}<span class="truncate">{e.name}</span></span>
                <span class="iv-v num">{metric(e.map50_95)}</span>
                <span class="iv-plot">{#if e.ci95}<IntervalPlot value={e.map50_95 ?? 0} ci={e.ci95} domain={ivDomain} height={28} color={color(e.id)} />{/if}</span>
                <span class="iv-ci num">{e.ci95 ? intervalText(e.ci95) : ''}</span>
              </li>
            {/each}
            <li class="iv-axis" aria-hidden="true">
              <span></span><span></span>
              <span class="iv-ticks">{#each capTicks(ivDomain[0], ivDomain[1], 5) as t (t)}<span class="num" style:left="{(along(ivDomain, t) * 100).toFixed(2)}%">{t.toFixed(2)}</span>{/each}</span>
              <span></span>
            </li>
          </ul>
          <p class="verdict">{verdict}</p>
        {/if}
      </Panel>

      {#if !loaded && failedResult}
        <ErrorState error={failedResult.err} title="Could not load the result of {failedResult.e.name}" retry={() => results.retry()} />
      {:else if !loaded}
        {#each [280, 320, 320] as h, i (i)}<Skeleton h="{h}px" r="var(--radius-lg)" />{/each}
      {:else}
        <Panel title="Robustness by condition" meta="accuracy kept">
          {#if condRows.length}
            <DotRows
              rows={condRows.map((r) => ({ ...r, name: nameOf(r.key) }))}
              {keys}
              domain={condDomain}
              format={(v) => keptText(v)}
              label="Accuracy kept per condition, one dot per entry"
              {highlight}
            />
          {/if}
          {#if noRobust.length}
            <p class="note">{noRobust.map((e) => e.name).join(', ')} {noRobust.length === 1 ? 'ran' : 'ran'} no condition: {noRobust.length === 1 ? 'it has' : 'they have'} no dots here.</p>
          {/if}
        </Panel>

        <div class="pair">
          <Panel title="Calibration" meta="share right by confidence">
            {#if calSeries.length}
              <XYChart
                label="Reliability of each entry's confidences"
                series={calSeries}
                {highlight}
                xDomain={RELIABILITY_X}
                yDomain={[0, 1]}
                xTicks={[0.1, 0.4, 0.7, 1]}
                yTicks={[0, 0.25, 0.5, 0.75, 1]}
                xFormat={(v) => v.toFixed(1)}
                yFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(2))}
                xLabel="Confidence it gave"
                yLabel="Share right"
                height={phone.current ? 240 : 280}
                refs={[{ axis: 'diagonal', label: 'said = right' }]}
                stops={BIN_MIDS}
                bind:cursor={calCursor}
                readout={calRead}
                table={{
                  caption: 'ECE per entry',
                  head: ['Entry', 'ECE', 'Detections'],
                  rows: shown.flatMap((e, i) => (res[i]?.calibration ? [[e.name, num(res[i]!.calibration!.ece, 3), num(res[i]!.calibration!.detections)]] : [])),
                }}
              />
            {/if}
            <p class="note">
              {#each shown as e, i (e.id)}{#if res[i]?.calibration}<span class="ece">{@render key(e.id)}ECE {num(res[i]!.calibration!.ece, 2)}</span>{/if}{/each}
              <!-- Not measured without confidences, or with confidences that never reach 0.1 (docs/BENCH.md, Calibration). -->
              {#each noCal as e (e.id)}<span>{e.name}: {e.scored ? 'no detection reached confidence 0.1' : 'no confidences'}, not measured.</span>{/each}
            </p>
          </Panel>

          <Panel title="Self-labeling" meta="labels right by objects found">
            <XYChart
              label="Precision of each entry's labels against the share of objects they find, as its threshold moves"
              series={slSeries}
              {highlight}
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              xTicks={[0, 0.25, 0.5, 0.75, 1]}
              yTicks={[0, 0.25, 0.5, 0.75, 1]}
              xFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(2))}
              yFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(2))}
              xLabel="Objects found (recall)"
              yLabel="Labels right"
              height={phone.current ? 240 : 280}
              refs={[{ axis: 'y', value: TARGET_PRECISION, label: '90% right' }]}
              marks={slMarks}
              stops={[]}
              readout={legendRead}
              table={{
                caption: 'Self-labeling per entry',
                head: ['Entry', 'Threshold t*', 'Objects at 90%', 'Labels right'],
                rows: shown.flatMap((e, i) => {
                  const s = res[i]?.self_labeling;
                  return s ? [[e.name, s.threshold !== null ? conf(s.threshold) : '—', pct(s.recall, 0), pct(s.precision, 0)]] : [];
                }),
              }}
            />
            <p class="note">
              The furthest right a line gets above 90% is its self-labeling axis (ringed).
              {#each shown as e, i (e.id)}{#if res[i]}<span class="ece">{@render key(e.id)}{pct(res[i]!.self_labeling.recall, 0)}{#if res[i]!.self_labeling.threshold !== null}{' '}from {conf(res[i]!.self_labeling.threshold)}{/if}</span>{/if}{/each}
            </p>
          </Panel>
        </div>

        <Panel title="AP50-95 per class" meta={clsRows.length > CLS_SHOWN && !allClasses ? `the ${CLS_SHOWN} hardest of ${clsRows.length}` : `${clsRows.length} classes, hardest first`}>
          <DotRows
            rows={clsShown.map((r) => ({ key: r.key, name: r.name, values: r.values }))}
            {keys}
            domain={clsDomain}
            format={(v) => metric(v)}
            tickFormat={(v) => v.toFixed(2)}
            label="AP50-95 per class, one dot per entry"
            {highlight}
            sub={(k) => {
              // The benchmark's count; an entry graded on a part of the images says on how many of them.
              const row = clsRows.find((r) => r.key === k);
              const n = row?.truths ?? 0;
              const part = shown.flatMap((e, i) => (bench && e.images < bench.images && row?.graded[i] != null ? [`${num(row.graded[i]!)} for ${e.name}`] : []));
              return `${num(n)} true ${n === 1 ? 'box' : 'boxes'}${part.length ? ` (graded on ${part.join(', ')})` : ''}`;
            }}
          />
          {#if clsRows.length > CLS_SHOWN}
            <Button size="sm" variant="ghost" onclick={() => (allClasses = !allClasses)}>{allClasses ? `Show the ${CLS_SHOWN} hardest` : `Show all ${clsRows.length} classes`}</Button>
          {/if}
        </Panel>
      {/if}
    {/if}
  </Page>
{/if}

<style>
  /* The legend: one card per entry (RunCompare's chips), side by side, a 2 × 2 grid from 768 to 1279 px. */
  /* Chips as wide as what they hold (at least 240 px), so a name like "Claude Code (haiku)" stays on one line beside
     its score instead of breaking inside its parentheses in a cell a quarter of the row wide. */
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: calc(-1 * var(--space-2));
  }
  .chips > :global(*) {
    flex: 0 1 auto;
    min-width: 240px;
    max-width: 100%;
  }
  .chip {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto auto;
    align-items: center;
    column-gap: var(--space-1);
    min-height: 44px;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-2);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
  }
  .chip:hover {
    border-color: var(--line-strong);
  }
  .chip.pinned {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
  }
  .ck {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    min-height: 32px;
    padding: 0 var(--space-1);
    border-radius: var(--radius-sm);
    text-align: left;
  }
  .nm {
    font-weight: var(--weight-medium);
    overflow-wrap: anywhere;
  }
  .v {
    padding: 0 var(--space-1);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .st {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .act {
    position: relative;
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  @media (hover: hover) {
    .ck:hover,
    .act:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
  }
  @media (pointer: coarse) {
    .act {
      width: 40px;
      height: 40px;
    }
  }
  /* A key is a fill: the palette hue with the swatch hairline, its slot's pattern cut along it (RunCompare). */
  .key {
    flex-shrink: 0;
    display: inline-block;
    width: 16px;
    height: 4px;
    border-radius: 2px;
    background: var(--c);
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  .key[data-pattern='dash'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 5px, transparent 5px 7px);
    border-radius: 0;
  }
  .key[data-pattern='dot'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 3px, transparent 3px 5px);
    border-radius: 0;
  }
  /* Add: a secondary button's face, the native picker over all of it. */
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    min-height: 44px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    border: 1px dashed var(--line-strong);
    color: var(--text-secondary);
  }
  .add select {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  @media (pointer: coarse) {
    .add select {
      font-size: var(--type-lg-size);
    }
  }
  .add:has(select:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .state {
    display: grid;
    place-items: center;
    min-height: 320px;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .cav {
    display: grid;
    gap: var(--space-1);
    margin-top: var(--space-1);
  }
  .scroll {
    overflow-x: auto;
  }
  /* The measure a fixed column, the entries sharing the rest equally (DESIGN_DIRECTION §2.7: a comparison). */
  .cmp {
    table-layout: fixed;
  }
  .m-col {
    width: 200px;
  }
  .cmp thead th {
    height: auto;
    padding-top: var(--space-2);
    padding-bottom: var(--space-2);
    vertical-align: bottom;
  }
  .eh {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    max-width: 100%;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    white-space: normal;
  }
  .eh-n {
    overflow-wrap: anywhere;
    text-align: right;
  }
  .role {
    display: block;
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .swap {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .cmp tr.total td {
    font-weight: var(--weight-semibold);
  }
  .top {
    font-weight: var(--weight-semibold);
  }
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .dd {
    display: inline-block;
    min-width: 4.5em;
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    text-align: right;
  }
  .iv {
    display: grid;
  }
  .iv li {
    display: grid;
    grid-template-columns: minmax(120px, 220px) 56px minmax(0, 1fr) 112px;
    align-items: center;
    gap: var(--space-3);
    min-height: 36px;
    border-top: 1px solid var(--line-subtle);
  }
  .iv li.dim {
    opacity: 0.45;
  }
  .iv li.iv-axis {
    min-height: 20px;
    border-top: 0;
  }
  .iv-n {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .iv-v {
    text-align: right;
    font-weight: var(--weight-medium);
  }
  .iv-ci {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .iv-ticks {
    position: relative;
    display: block;
    height: 16px;
    color: var(--chart-label);
    /* small: chart tick */
    font-size: var(--type-xs-size);
  }
  .iv-ticks span {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
  }
  .verdict {
    margin-top: var(--space-3);
    max-inline-size: var(--measure-dialog);
  }
  .sbs {
    display: none;
  }
  .pair {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
    align-items: start;
  }
  .note {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .ece,
  .rv {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .rv {
    margin-left: var(--space-3);
    color: var(--text-primary);
  }
  .t,
  .lr {
    color: var(--text-tertiary);
  }
  .hint {
    color: var(--text-secondary);
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
  .sk-chips {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-2);
  }
  /* Narrower columns: the difference under its value. */
  @media (max-width: 1279px) {
    .cmp .dd {
      display: block;
      min-width: 0;
      margin-left: 0;
    }
  }
  @media (max-width: 1023px) {
    .pair {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .chips > :global(*) {
      flex-basis: 100%;
    }
    .scroll:has(.cmp) {
      display: none;
    }
    .sbs {
      display: grid;
    }
    .sbs li {
      display: grid;
      gap: var(--space-2);
      padding: var(--space-3) var(--space-4);
      border-top: 1px solid var(--line-subtle);
    }
    .sbs li:first-child {
      border-top: 0;
    }
    .sbs-h {
      font-weight: var(--weight-medium);
    }
    .sbs-g {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-2) var(--space-4);
    }
    .sbs-v {
      display: grid;
      min-width: 0;
    }
    .sbs-n {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      min-width: 0;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .sbs-v .dd {
      min-width: 0;
    }
    .iv li {
      grid-template-columns: minmax(0, 1fr) 48px;
      grid-template-areas: 'n v' 'p p';
      row-gap: 0;
      padding: var(--space-1) 0;
    }
    .iv-n {
      grid-area: n;
    }
    .iv-v {
      grid-area: v;
    }
    .iv-plot {
      grid-area: p;
    }
    .iv-ci {
      display: none;
    }
    .iv li.iv-axis {
      grid-template-areas: 'p p';
    }
    .iv li.iv-axis > span:not(.iv-ticks) {
      display: none;
    }
    .iv-ticks {
      grid-area: p;
    }
  }
</style>
