<script lang="ts">
  /**
   * Can you trust its confidence? (docs/BENCH.md, Calibration): a reliability diagram. Per bin of confidence, the share
   * of its detections that is right against the bin's mean confidence, the diagonal (said = right) as the reference; a
   * dot's size follows how many detections the bin holds, a bin of under 5% of them is hollow, and their counts stand
   * as bars under the plot, so a thin bin never looks as sure as a full one. The `[0, 0.1)` bin is always empty
   * (detections enter from 0.1), so the axis starts at 0.1. The ECE, and one sentence that reads the chart. Without
   * confidences: what that means and why agents have none.
   */
  import { MediaQuery } from 'svelte/reactivity';
  import type { Bench, BenchEntry, BenchResult } from '$lib/api/types';
  import XYChart from '$lib/charts/XYChart.svelte';
  import { media } from '$lib/design/tokens';
  import { num, pct } from '$lib/format';
  import Panel from '$lib/ui/Panel.svelte';
  import { binAtConfidence, RELIABILITY_X, reliabilityPoints, reliabilityStrip } from '../charts';
  import { axisWhyNot, calibrationReading } from '../words';

  let { result, entry, bench }: { result: BenchResult; entry: BenchEntry; bench: Bench } = $props();

  const cal = $derived(result.calibration);
  const phone = new MediaQuery(media.phone, false);
  const reading = $derived(cal ? calibrationReading(cal) : null);
  const TONE = { close: 'Well calibrated', over: 'Overconfident', under: 'Underconfident' } as const;
  let cursor = $state<number | null>(null);
  const filled = $derived(cal ? cal.bins.filter((b) => b.count > 0) : []);
</script>

{#snippet readout(c: number | null, hint: string | null)}
  {#if hint}<span class="hint">{hint}</span>
  {:else if c === null}
    <span class="key"><i class="dotk" aria-hidden="true"></i>A bin of confidences: share right</span><span class="key gap"><i class="diag" aria-hidden="true"></i>said = right</span>
  {:else}
    {@const b = binAtConfidence(cal!, c)}
    {#if b}
      <span class="t">{num(b.lo, 1)}–{num(b.hi, 1)}</span><span class="sep">·</span><span>says {num(b.confidence!, 2)}</span><span class="sep">·</span><span class="v">right {pct(b.precision!, 0)}</span><span class="sep">·</span><span>{num(b.count)} detections</span>
    {/if}
  {/if}
{/snippet}

<Panel title="Calibration" meta="confidence against how often it is right" id="calibration">
  {#if cal && reading}
    <div class="cal">
      <XYChart
        label="Reliability diagram: share of detections right per confidence bin, against the bin's mean confidence"
        series={[{ id: 'bins', label: 'Share right', color: 'var(--text-primary)', points: reliabilityPoints(cal) }]}
        xDomain={RELIABILITY_X}
        yDomain={[0, 1]}
        xTicks={phone.current ? [0.1, 0.4, 0.7, 1] : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        yTicks={[0, 0.25, 0.5, 0.75, 1]}
        xFormat={(v) => v.toFixed(1)}
        yFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(2))}
        xLabel="Confidence it gave"
        yLabel="Share right"
        height={phone.current ? 260 : 300}
        refs={[{ axis: 'diagonal', label: 'said = right' }]}
        strip={{ items: reliabilityStrip(cal), unit: 'detections' }}
        bind:cursor
        {readout}
        table={{
          caption: 'Reliability per confidence bin',
          head: ['Mean confidence', 'Bin', 'Share right', 'Detections'],
          rows: filled.map((b) => [num(b.confidence!, 2), `${num(b.lo, 1)}–${num(b.hi, 1)}`, pct(b.precision!, 0), num(b.count)]),
        }}
      />
      <aside class="read">
        <h3 class="tone">{TONE[reading.tone]}</h3>
        <!-- The tone is the heading: the sentence goes on from it (its first clause only: never across a full stop, which
             a decimal point is not). -->
        <p class="sentence">{reading.sentence.replace(/^(?:[^:.]|\.\d)+: (\w)/, (_, c: string) => c.toUpperCase())}</p>
        <dl class="kv">
          <div><dt>ECE</dt><dd class="num">{num(cal.ece, 3)}</dd></div>
          <div><dt>Calibration axis</dt><dd class="num">{num(1 - cal.ece, 2)}</dd></div>
          <div><dt>Detections</dt><dd class="num">{num(cal.detections)}</dd></div>
        </dl>
        <p class="how">
          Above the diagonal it is right more often than it says; below, less. Bigger dots stand on more detections, a
          hollow one on under 5% of them; the bars under the plot are the counts. ECE is the gap to the diagonal, weighted by
          those counts. Detections from confidence 0.1 up, on the unchanged images.
        </p>
      </aside>
    </div>
  {:else}
    <div class="none">
      <p class="n-title">Not measured</p>
      <p class="n-why">{axisWhyNot('calibration', entry, bench)} The score leaves this axis out and the entry is partial; compare it with other entries on precision, recall and self-labeling.</p>
    </div>
  {/if}
</Panel>

<style>
  .cal {
    display: grid;
    grid-template-columns: minmax(0, 8fr) minmax(0, 4fr);
    gap: var(--space-4) var(--space-8);
    align-items: start;
  }
  .read {
    display: grid;
    gap: var(--space-3);
    align-content: start;
  }
  .tone {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .sentence {
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .kv {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
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
  .how {
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
  .dotk {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-primary);
  }
  .diag {
    width: 14px;
    height: 0;
    border-top: 1px dashed var(--text-tertiary);
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
  .none {
    display: grid;
    gap: var(--space-1);
  }
  .n-title {
    font-weight: var(--weight-medium);
  }
  .n-why {
    color: var(--text-secondary);
    max-inline-size: var(--measure-dialog);
  }
  @media (max-width: 1023px) {
    .cal {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
