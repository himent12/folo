/**
 * How tight are its correct boxes? (docs/BENCH.md, Localization): the IoU of every correct detection with its true
 * box, as a histogram in tenths from 0.5 (where a detection starts counting as right) to 1, the mean marked; the axis
 * is that mean rescaled from 0.5–1 to 0–1.
 */
import { computed, css } from 'fw';
import type { BenchResult } from '$lib/api/types';
import Histogram from '$lib/charts/Histogram.tsx';
import { localizationAxis } from '$lib/bench/score';
import { num } from '$lib/format';
import Panel from '$lib/ui/Panel';

export interface LocalizationSectionProps {
  result: BenchResult;
}

const edges = Array.from({ length: 11 }, (_, i) => +(0.5 + i * 0.05).toFixed(2));

export default function LocalizationSection(props: LocalizationSectionProps) {
  const loc = computed(() => props.result.localization);

  return (
    <Panel title="Localization" meta="IoU of its correct boxes" id="localization">
      <div class="loc">
        <div class="chart">
          {loc.get().matches > 0 ? (
            <Histogram
              edges={edges}
              counts={loc.get().histogram}
              height={200}
              color="var(--text-tertiary)"
              hoverColor="var(--text-secondary)"
              unit="detections"
              label="IoU of the correct detections with their true boxes, in steps of 0.05 from 0.5 to 1"
              binLabel={(lo, hi) => `IoU ${lo.toFixed(2)}–${hi.toFixed(2)}`}
              tickLabel={(lo, _hi, i) => (i % 2 === 0 ? lo.toFixed(1) : '')}
              marker={loc.get().mean_iou !== null ? { x: loc.get().mean_iou!, label: `mean ${num(loc.get().mean_iou!, 2)}` } : undefined}
              yLabel="Detections"
            />
          ) : (
            <p class="none">Nothing it found was right (IoU 0.5 with a true box of its class), so there is no box to measure. The axis is 0.</p>
          )}
        </div>
        <aside class="read">
          <dl class="kv">
            <div>
              <dt>Mean IoU</dt>
              <dd class="num">{loc.get().mean_iou !== null ? num(loc.get().mean_iou!, 2) : '—'}</dd>
            </div>
            <div>
              <dt>Localization axis</dt>
              <dd class="num">{num(localizationAxis(loc.get().mean_iou), 2)}</dd>
            </div>
            <div>
              <dt>Correct boxes</dt>
              <dd class="num">{num(loc.get().matches)}</dd>
            </div>
          </dl>
          <p class="how">
            A detection counts as right from IoU 0.5; the further right the mass sits, the tighter its boxes. The axis rescales the mean from 0.5–1 to
            0–1, so a mean of 0.75 is 0.5. Correct detections at or above its best-F1 confidence, on the unchanged images.
          </p>
        </aside>
      </div>
    </Panel>
  );
}

css`
  .loc {
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
  }
  .kv {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
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
  .how,
  .none {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    max-inline-size: var(--measure);
  }
  .none {
    color: var(--text-secondary);
  }
  @media (max-width: 1023px) {
    .loc {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
