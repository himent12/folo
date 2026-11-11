// Gallery twin of ChartLossOptions.svelte: the loss chart's view control (charts/LossOptions).
import { signal } from 'fw';
import LossOptions, { type Scale, type Smoothing } from '$lib/charts/LossOptions';

export default function ChartLossOptionsSection() {
  const smoothing = signal<Smoothing>('0.98');
  const scale = signal<Scale>('log');
  return (
    <div class="g-row g-end g-medium" data-parity-phone data-parity-step="open" data-parity-target="button.trigger">
      <LossOptions bind:smoothing={smoothing} bind:scale={scale} />
    </div>
  );
}
