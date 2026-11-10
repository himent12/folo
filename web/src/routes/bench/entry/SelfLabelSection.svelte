<script lang="ts">
  /**
   * Could it label for itself? (docs/BENCH.md, Self-labeling): the precision of the labels it keeps and the share of all
   * objects they find, against the confidence threshold they are kept from (the 101-point curve); the 90% target as a
   * reference level and the threshold t* where precision first reaches it, with what it covers there. The readout in
   * words, the verdict with what it means, and, for a model whose classes are the benchmark's, the next step: a loop
   * that tests whether those labels train a better model. Designed for a curve that never reaches 90% (no t*) and for
   * a contestant without confidences (one operating point, no curve to draw).
   */
  import { MediaQuery } from 'svelte/reactivity';
  import Repeat from '@lucide/svelte/icons/repeat';
  import type { BenchEntry, BenchResult } from '$lib/api/types';
  import { TARGET_PRECISION } from '$lib/bench/rules';
  import XYChart from '$lib/charts/XYChart.svelte';
  import { media } from '$lib/design/tokens';
  import { conf, num, pct } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { labelCurves, labelPointAt } from '../charts';
  import { MIN_LABELS } from '$lib/bench/score';
  import { fine, labelsText, perImageText, READINESS, READINESS_MEANS, selfLabelReading, share, shareApart } from '../words';

  let {
    result,
    entry,
    loop,
  }: {
    result: BenchResult;
    entry: BenchEntry;
    /** The loop it could start: `why` null when it can (a model of this server with the benchmark's classes). */
    loop: { why: string | null; start: () => void } | null;
  } = $props();

  const s = $derived(result.self_labeling);
  const phone = new MediaQuery(media.phone, false);
  const curves = $derived(labelCurves(s.curve));
  const scored = $derived(entry.scored && s.curve.length > 1);
  const at = $derived(s.threshold !== null ? labelPointAt(s.curve, s.threshold) : null);
  let cursor = $state<number | null>(null);
  const stops = $derived(s.curve.filter((p) => p.labels > 0).map((p) => p.confidence));
  const one = $derived(s.curve[0]);
</script>

{#snippet readout(c: number | null, hint: string | null)}
  {@const p = c === null ? (at ?? s.curve[0]) : labelPointAt(s.curve, c)}
  {#if hint}<span class="hint">{hint}</span>
  {:else}
    <span class="t">{c === null ? (at ? `At t* ${conf(s.threshold)}` : 'All its detections') : `From ${conf(c)} up`}</span>
    <span class="key"><i class="ln" aria-hidden="true"></i>Right {p?.precision != null ? fine(p.precision) : '—'}</span>
    <span class="key"><i class="ln dash" aria-hidden="true"></i>Objects found {p ? share(p.recall) : '—'}</span>
    {#if p}<span class="n">{#if c === null && !at}{num(p.labels)} detections, none kept{:else}{labelsText(p.labels)}{#if p.labels > 0 && p.labels < MIN_LABELS}, too few to rest a threshold on{/if}{/if}</span>{/if}
  {/if}
{/snippet}

<Panel title="Self-labeling" meta="the labels it keeps, by confidence threshold" id="self-labeling">
  <div class="sl">
    <div class="chart">
      {#if scored}
        <XYChart
          label="Share of the labels kept that is right, and share of all objects they find, by the confidence they are kept from"
          series={[
            { id: 'p', label: 'Right (precision)', color: 'var(--text-primary)', points: curves.precision },
            { id: 'r', label: 'Objects found (recall)', color: 'var(--text-tertiary)', points: curves.recall, pattern: 'dash', thin: true },
          ]}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xTicks={phone.current ? [0, 0.5, 1] : [0, 0.2, 0.4, 0.6, 0.8, 1]}
          yTicks={[0, 0.25, 0.5, 0.75, 1]}
          xFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(1))}
          yFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(2))}
          xLabel="Kept from confidence"
          yLabel="Share"
          height={phone.current ? 240 : 280}
          refs={[{ axis: 'y', value: TARGET_PRECISION, label: '90% right: enough to train on', below: s.threshold !== null }, ...(s.threshold !== null ? [{ axis: 'x' as const, value: s.threshold, label: `t* ${conf(s.threshold)}` }] : [])]}
          marks={at ? [{ x: at.confidence, y: at.recall, label: `${share(at.recall)} of objects, ${labelsText(at.labels)}`, side: at.confidence > 0.6 ? 'left' : 'right' }] : []}
          {stops}
          bind:cursor
          {readout}
          speak={(c) => {
            const p = labelPointAt(s.curve, c);
            return p ? `From ${conf(c)} up: ${labelsText(p.labels)}, ${p.precision !== null ? pct(p.precision, 0) : 'none'} right, ${share(p.recall)} of all objects` : `From ${conf(c)} up: no labels`;
          }}
          table={{
            caption: 'Labels kept by confidence threshold',
            head: ['Kept from', 'Right', 'Objects found', 'Labels'],
            rows: s.curve.filter((p) => p.labels > 0 && Math.round(p.confidence * 100) % 5 === 0).map((p) => [conf(p.confidence), p.precision !== null ? pct(p.precision, 0) : '—', share(p.recall), num(p.labels)]),
          }}
        />
      {:else if one}
        <!-- No confidences: one operating point, read as two shares against the 90% a label set needs. -->
        <div class="one">
          <p class="one-h">One operating point: it gives no confidences, so no threshold can trade labels for precision.</p>
          {#each [{ k: 'Right (precision)', v: one.precision ?? 0, target: true }, { k: 'Objects found (recall)', v: one.recall, target: false }] as row (row.k)}
            <div class="meter">
              <span class="m-k">{row.k}</span>
              <span class="m-bar" aria-hidden="true">
                <span class="m-fill" style:width="{row.v * 100}%"></span>
                {#if row.target}<span class="m-target" style:left="{TARGET_PRECISION * 100}%"></span>{/if}
              </span>
              <span class="m-v num">{row.target ? (one.labels >= MIN_LABELS ? shareApart(row.v, TARGET_PRECISION) : pct(row.v, 0)) : share(row.v)}</span>
            </div>
          {/each}
          <p class="one-cap">The mark on the first bar is the 90% a label set to train on needs. {labelsText(one.labels)} on {num(entry.images)} images{#if one.labels < MIN_LABELS}: too few to rest a label set on (it takes at least {MIN_LABELS}){/if}.</p>
        </div>
      {/if}
    </div>
    <aside class="read">
      <h3 class="verdict">{READINESS[s.readiness]}</h3>
      <p class="means">{READINESS_MEANS[s.readiness]}</p>
      <p class="sentence">{selfLabelReading(s, scored)}</p>
      <dl class="kv">
        <div><dt>Threshold t*</dt><dd class="num">{s.threshold !== null ? conf(s.threshold) : '—'}</dd></div>
        <div><dt>Axis (objects at 90%)</dt><dd class="num">{share(s.recall)}</dd></div>
        <!-- The labels the threshold rests on (all its detections without one), and per image as a reader can see it. -->
        <div><dt>{s.threshold !== null ? 'Labels at t*' : 'Detections (none kept)'}</dt><dd class="num">{num(s.labels)}<span class="per">{perImageText(s.labels_per_image, s.labels)}</span></dd></div>
        <div><dt>Mean IoU</dt><dd class="num">{s.mean_iou !== null ? num(s.mean_iou, 2) : '—'}</dd></div>
      </dl>
      {#if loop}
        <div class="next">
          <p class="next-h">Does training on these labels help? A loop tests it: this model labels a pool of images, a student trains on those labels alone, and this benchmark grades the student.</p>
          {#if loop.why}
            <p class="next-why" data-gate-why>{loop.why}</p>
          {:else}
            <Button variant="primary" icon={Repeat} onclick={loop.start}>Test it: start a loop</Button>
          {/if}
        </div>
      {/if}
    </aside>
  </div>
</Panel>

<style>
  .sl {
    display: grid;
    grid-template-columns: minmax(0, 8fr) minmax(0, 4fr);
    gap: var(--space-4) var(--space-8);
    align-items: start;
  }
  .chart {
    min-width: 0;
  }
  .read {
    display: grid;
    gap: var(--space-3);
    align-content: start;
  }
  .verdict {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .means {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .sentence {
    max-inline-size: var(--measure);
  }
  .kv {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .kv div {
    display: grid;
    gap: 2px;
  }
  .kv dt {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .kv dd {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    font-weight: var(--weight-semibold);
  }
  .next {
    display: grid;
    justify-items: start;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .next-h,
  .next-why {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: var(--space-3);
    color: var(--text-primary);
  }
  .ln {
    width: 14px;
    height: 0;
    border-top: 2px solid var(--text-primary);
  }
  .ln.dash {
    border-top: 1px dashed var(--text-tertiary);
  }
  .t,
  .n {
    color: var(--text-tertiary);
  }
  .n {
    margin-left: var(--space-3);
  }
  .hint {
    color: var(--text-secondary);
  }
  .one {
    display: grid;
    gap: var(--space-3);
  }
  .one-h {
    color: var(--text-secondary);
    max-inline-size: var(--measure-dialog);
  }
  .meter {
    display: grid;
    grid-template-columns: 176px minmax(0, 1fr) 48px;
    align-items: center;
    gap: var(--space-3);
  }
  .m-bar {
    position: relative;
    height: 8px;
    border-radius: 4px;
    background: var(--bg-active);
  }
  .m-fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 4px;
    background: var(--text-secondary);
  }
  /* The 90% target: an ink tick across the bar, 4 px over each edge. */
  .m-target {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    margin-left: -1px;
    background: var(--text-primary);
  }
  .m-v {
    text-align: right;
  }
  .one-cap {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (max-width: 1023px) {
    .sl {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .meter {
      grid-template-columns: minmax(0, 1fr) 48px;
    }
    .m-k {
      grid-column: 1 / -1;
    }
    .key,
    .n {
      margin-left: var(--space-2);
    }
  }
  /* Per image beside the count, small, on its line: the next value's baseline is not pushed down. */
  .per {
    display: inline;
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
  }
</style>
