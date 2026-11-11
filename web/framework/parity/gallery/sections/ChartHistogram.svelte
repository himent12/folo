<!-- Gallery twin of ChartHistogram.tsx: Histogram (charts/Histogram). -->
<script lang="ts">
  import Histogram from '$lib/charts/Histogram.svelte';
  import { histBin, histCounts, histEdges } from '../chartData';
</script>

<div class="g-stack" data-parity-phone>
  <div data-parity-step="keys" data-parity-do="press:ArrowRight" data-parity-target="svg[aria-roledescription=chart]">
    <Histogram edges={histEdges} counts={histCounts} label="Box sizes" yLabel="Boxes" binLabel={histBin} marker={{ x: 0.02, label: 'tiny' }} />
  </div>
  <div data-parity-step="hover" data-parity-do="hover" data-parity-target="svg[aria-roledescription=chart]">
    <Histogram edges={histEdges} counts={histCounts.map((c) => c * 9)} label="Box sizes, all" unit="objects" height={120} hoverColor="var(--chart-1)" tickLabel={(lo) => `${lo * 100}`} />
  </div>
</div>
