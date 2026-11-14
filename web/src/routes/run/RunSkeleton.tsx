/** A run while it loads: the real layout's shapes (header, metrics, the detail row), so nothing jumps when data arrives. */
import { css, Index } from 'fw';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import Skeleton from '$lib/ui/Skeleton';
import { SHORT, TALL } from './RunCharts';

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

/** A chart panel's frame: its 40 px header row over the plot, as ui/Panel and run/RunChart lay them out. */
const chart = (height: number, title: string) => (
  <div class="sk-panel">
    <div class="sk-ph">
      <Skeleton w={title} h="14px" />
    </div>
    <div class="sk-pb">
      <ChartSkeleton height={height} />
    </div>
  </div>
);

export default function RunSkeleton() {
  // Mirrors the real layout (header, summary strip, chart rows) so nothing jumps when data arrives.
  return (
    <div class="loading" aria-busy="true" aria-label="Loading run">
      <div class="sk-head">
        <Skeleton w="140px" h="20px" r="var(--radius-full)" />
        <Skeleton w="240px" h="28px" />
        <Skeleton w="min(420px, 100%)" h="16px" />
      </div>
      {/* The KPI strip (five tiles), Validation mAP beside Training loss, Precision / Recall beside Learning rate
          (run/RunCharts), then the samples beside the per-class and config rail. */}
      <div class="sk-strip">
        <Index each={range(5)}>
          {() => (
            <div class="sk-cell">
              <Skeleton w="56%" h="12px" />
              <Skeleton w="48%" h="24px" />
              <Skeleton w="68%" h="12px" />
            </div>
          )}
        </Index>
      </div>
      <div class="sk-row2 wide">
        {chart(TALL, '160px')}
        {chart(TALL, '120px')}
      </div>
      <div class="sk-row2 even">
        {chart(SHORT, '140px')}
        {chart(SHORT, '112px')}
      </div>
      <div class="sk-pick">
        <Skeleton w="280px" h="28px" r="var(--radius-md)" />
      </div>
      <div class="sk-detail">
        <div class="sk-main">
          <div class="sk-panel">
            <div class="sk-ph">
              <Skeleton w="128px" h="14px" />
              <span class="sk-ctl">
                <Skeleton w="160px" h="24px" r="var(--radius-md)" />
                <Skeleton w="120px" h="24px" r="var(--radius-md)" />
              </span>
            </div>
            <div class="sk-pb">
              <div class="sk-bar">
                <Skeleton w="min(300px, 60%)" h="12px" />
                <Skeleton w="200px" h="12px" />
              </div>
              <div class="sk-tiles">
                <Index each={range(9)}>{() => <Skeleton h="auto" class="sk-tile" r="var(--radius-md)" />}</Index>
              </div>
            </div>
          </div>
          {/* Logs under the samples (desktop and laptop), growing to the rail's foot like the loaded log. */}
          <div class="sk-panel sk-logs">
            <div class="sk-ph">
              <Skeleton w="56px" h="14px" />
              <Skeleton w="128px" h="24px" r="var(--radius-md)" />
            </div>
            <div class="sk-lines">
              <Index each={range(9)}>{(_, i) => <Skeleton w={`${46 + ((i * 13) % 24)}%`} h="10px" />}</Index>
            </div>
          </div>
        </div>
        <div class="sk-rail">
          <div class="sk-panel">
            <div class="sk-ph">
              <Skeleton w="72px" h="14px" />
            </div>
            <div class="sk-rows">
              <Index each={range(8)}>
                {(_, i) => (
                  <div class="sk-row">
                    <Skeleton w="8px" h="8px" r="2px" />
                    <Skeleton w={`${56 + ((i * 23) % 40)}px`} h="12px" />
                    <Skeleton w="40px" h="12px" />
                    <Skeleton h="4px" />
                  </div>
                )}
              </Index>
            </div>
          </div>
          <div class="sk-panel">
            <div class="sk-ph">
              <Skeleton w="56px" h="14px" />
            </div>
            <div class="sk-rows">
              <Index each={range(6)}>
                {(_, i) => (
                  <div class="sk-row cfg">
                    <Skeleton w={`${64 + ((i * 17) % 32)}px`} h="12px" />
                    <Skeleton w={`${36 + ((i * 29) % 40)}px`} h="12px" />
                  </div>
                )}
              </Index>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

css`
  .loading {
    display: grid;
    gap: var(--space-4);
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
  }
  .sk-head {
    display: grid;
    gap: var(--space-2);
    max-width: 100%;
  }
  /* One panel cut into five tiles by hairlines, as run/RunKpis. */
  .sk-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .sk-cell {
    display: grid;
    grid-template-rows: 16px 32px 16px;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
  }
  .sk-row2 {
    display: grid;
    gap: var(--space-4);
  }
  .sk-row2.wide {
    grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  }
  .sk-row2.even {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }
  .sk-pick {
    display: none;
  }
  /* The loaded detail row: samples over logs beside per-class over config, the columns ending together. */
  .sk-detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: stretch;
  }
  .sk-main,
  .sk-rail {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
  }
  .sk-lines {
    display: grid;
    align-content: start;
    gap: 8px;
    min-height: 200px;
    padding: var(--space-2) var(--space-4) var(--space-4);
  }
  .sk-panel {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .sk-ph {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    height: 40px;
    padding: 0 var(--space-3) 0 var(--space-4);
  }
  .sk-ctl {
    display: flex;
    gap: var(--space-2);
  }
  .sk-pb {
    display: grid;
    gap: var(--space-3);
    padding: 0 var(--space-4) var(--space-4);
  }
  .sk-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    height: 24px;
  }
  /* The loaded grid's geometry: tiles at least 152 px wide, 4:3, 8 px apart. */
  .sk-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(152px, 1fr));
    gap: var(--space-2);
  }
  .sk-tiles :global(.sk-tile) {
    aspect-ratio: 4 / 3;
  }
  /* Full rows only, as loaded: eight tiles in four or two columns, nine in three (laptops). */
  .sk-tiles :global(.sk-tile:nth-child(9)) {
    display: none;
  }
  .sk-rows {
    display: grid;
    padding-bottom: var(--space-2);
  }
  /* Per-class rows (swatch, name, value, bar) and config rows (label, value), 28 px like the loaded tables. */
  .sk-row {
    display: grid;
    grid-template-columns: 8px minmax(0, 1fr) 40px 30%;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    padding: 0 var(--space-4);
  }
  .sk-row.cfg {
    grid-template-columns: auto auto;
    justify-content: space-between;
  }
  @media (max-width: 1279px) {
    .sk-tiles :global(.sk-tile:nth-child(9)) {
      display: block;
    }
  }
  @media (max-width: 1023px) {
    .sk-strip {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .sk-row2.wide,
    .sk-row2.even {
      grid-template-columns: minmax(0, 1fr);
    }
    .sk-pick {
      display: block;
    }
    .sk-detail {
      grid-template-columns: minmax(0, 1fr);
    }
    .sk-rail,
    .sk-logs {
      display: none;
    }

    .sk-tiles :global(.sk-tile:nth-child(9)) {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .loading {
      padding: var(--space-4);
    }
    .sk-strip {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .sk-cell:last-child {
      grid-column: 1 / -1;
    }
    .sk-row2.even {
      display: none;
    }
    .sk-ctl :global(:first-child) {
      display: none;
    }
    .sk-tiles {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
`;
