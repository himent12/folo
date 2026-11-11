<!-- Gallery twin of ChartLossOptions.tsx: the loss chart's view control (charts/LossOptions). -->
<script lang="ts">
  import LossOptions, { type Scale, type Smoothing } from '$lib/charts/LossOptions.svelte';

  let smoothing = $state<Smoothing>('0.98');
  let scale = $state<Scale>('log');
</script>

<div class="g-row g-end g-medium" data-parity-phone data-parity-step="open" data-parity-target="button.trigger">
  <LossOptions bind:smoothing bind:scale />
</div>
