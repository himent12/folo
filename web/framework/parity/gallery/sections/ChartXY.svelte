<!-- Gallery twin of ChartXY.tsx: XYChart with every kind of mark (charts/XYChart). -->
<script lang="ts">
  import XYChart from '$lib/charts/XYChart.svelte';
  import { xyMarks, xyRefs, xySeries, xyStrip, xyTable } from '../chartData';

  const ticks = [0, 0.25, 0.5, 0.75, 1];
</script>

{#snippet readout(c: number | null, hint: string | null)}
  {#if c === null}<span>{hint ?? 'folo-s · yolo26n'}</span>{:else}<span>At {c}</span>{/if}
{/snippet}

<div class="g-stack" data-parity-phone>
  <div data-parity-step="keys" data-parity-do="press:End" data-parity-target="svg[aria-roledescription=chart]">
    <XYChart
      label="Accuracy by stress level"
      series={xySeries}
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      xTicks={ticks}
      yTicks={ticks}
      xFormat={(v) => v.toFixed(2)}
      yFormat={(v) => v.toFixed(2)}
      xLabel="Stress"
      yLabel="Accuracy"
      refs={xyRefs}
      band={{ lo: 0.6, hi: 0.7, label: 'teacher 95%' }}
      marks={xyMarks}
      strip={xyStrip}
      {readout}
      table={xyTable}
    />
  </div>
  <div data-parity-step="hover" data-parity-do="hover" data-parity-target="svg[aria-roledescription=chart]">
    <XYChart label="Highlighted" series={xySeries} xDomain={[0, 1]} yDomain={[0, 1]} xTicks={ticks} yTicks={ticks} highlight="yolo" height={180} {readout} table={xyTable} />
  </div>
</div>
