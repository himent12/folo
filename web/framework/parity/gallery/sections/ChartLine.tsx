// Gallery twin of ChartLine.svelte: LineChart in its main shapes (charts/LineChart).
import { signal } from 'fw';
import LineChart from '$lib/charts/LineChart';
import { compare, compareMarkers, loss, mosaicOff, perClass } from '../chartData';

export default function ChartLineSection() {
  // End labels are measured in the chart's font: drawn once it has loaded, so both twins measure the same text.
  const ready = signal(false);
  void document.fonts.ready.then(() => ready.set(true));
  return (
    <>
      {ready.get() ? (
        <div class="g-stack" data-parity-phone>
          <div data-parity-step="hover" data-parity-do="hover" data-parity-target="svg[aria-roledescription=chart]">
            <LineChart
              series={compare}
              label="mAP50-95 by run"
              height={240}
              xDomain={[0, 30]}
              xInteger
              endLabels
              endDots
              markers={compareMarkers}
              xMark={mosaicOff}
              xLabel="epoch"
              yLabel="mAP50-95"
              legendNote="Latest"
              xTitle={(x) => `Epoch ${x}`}
            />
          </div>
          <div data-parity-step="tooltip" data-parity-do="hover" data-parity-target="svg[aria-roledescription=chart]">
            <LineChart series={loss} label="Loss" yLog curve="monotone" tooltip valueFormat={(v) => v.toFixed(3)} />
          </div>
          <div data-parity-step="keys" data-parity-do="press:End" data-parity-target="svg[aria-roledescription=chart]">
            <LineChart series={perClass} label="Precision by class" yDomain={[0, 1]} xDomain={[0, 1]} highlight="mean" recede endDots yLabel="Precision" />
          </div>
          <LineChart series={[]} label="Empty" height={120} />
        </div>
      ) : null}
    </>
  );
}
