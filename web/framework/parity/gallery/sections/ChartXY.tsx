// Gallery twin of ChartXY.svelte: XYChart with every kind of mark (charts/XYChart).
import XYChart from '$lib/charts/XYChart';
import { xyMarks, xyRefs, xySeries, xyStrip, xyTable } from '../chartData';

export default function ChartXYSection() {
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const readout = (c: () => number | null, hint: () => string | null) => <>{c() === null ? <span>{hint() ?? 'folo-s · yolo26n'}</span> : <span>{`At ${c()}`}</span>}</>;
  return (
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
          readout={readout}
          table={xyTable}
        />
      </div>
      <div data-parity-step="hover" data-parity-do="hover" data-parity-target="svg[aria-roledescription=chart]">
        <XYChart label="Highlighted" series={xySeries} xDomain={[0, 1]} yDomain={[0, 1]} xTicks={ticks} yTicks={ticks} highlight="yolo" height={180} readout={readout} table={xyTable} />
      </div>
    </div>
  );
}
