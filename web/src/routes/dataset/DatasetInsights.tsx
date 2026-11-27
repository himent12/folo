/**
 * Compact Class Balance & Annotation Quality insights strip inside the Dataset detail Images view.
 * Surfaces real server/mock class imbalance, rarest-vs-most-common contrast, empty/broken label indicators,
 * split proportions, and one-click jump/filter controls for underrepresented classes or full Statistics.
 */
import ArrowRight from '$icons/arrow-right';
import ChevronDown from '$icons/chevron-down';
import CircleAlert from '$icons/circle-alert';
import CircleCheck from '$icons/circle-check';
import TriangleAlert from '$icons/triangle-alert';
import X from '$icons/x';
import { computed, css, Dynamic, effect, For, Show, signal } from 'fw';
import type { Dataset, DatasetStats, IssueCounts, Split } from '$lib/api/types';
import { compact, num } from '$lib/format';
import DatasetCurateCli from './DatasetCurateCli';
import { computeDatasetInsights, type InsightTone } from './insights';
import { FEW_BOXES } from './health';
import { swatchStyle } from './swatch';

export interface DatasetInsightsProps {
  dataset: Dataset;
  stats: DatasetStats;
  counts?: IssueCounts;
  activeClass: number | undefined;
  activeSplit: 'all' | Split;
  onSelectClass: (cls: number | undefined) => void;
  onSelectSplit: (split: 'all' | Split) => void;
  onOpenStats: () => void;
}

const TONE_ICON = {
  danger: CircleAlert,
  warning: TriangleAlert,
  ok: CircleCheck,
} satisfies Record<InsightTone, unknown>;

function fmtPct(pct: number): string {
  return pct > 0 && pct < 1 ? '<1%' : `${Math.round(pct)}%`;
}

export default function DatasetInsights(props: DatasetInsightsProps) {
  const data = computed(() => computeDatasetInsights(props.dataset, props.stats, props.counts));

  const balanceHeadline = computed(() => {
    const d = data.get();
    if (d.missingClasses.length > 0) {
      return `${d.missingClasses.length} ${d.missingClasses.length === 1 ? 'class' : 'classes'} unboxed`;
    }
    if (d.imbalanced && d.imbalanceRatio !== null) {
      return `${d.imbalanceRatio.toFixed(1)}× class imbalance`;
    }
    if (d.underrepresented.length > 0) {
      return `${d.underrepresented.length} rare ${d.underrepresented.length === 1 ? 'class' : 'classes'} (<${FEW_BOXES})`;
    }
    if (d.imbalanceRatio !== null) {
      return `${d.imbalanceRatio.toFixed(1)}× balanced`;
    }
    if (d.presentClasses.length === 1) {
      return `1 class · ${compact(d.presentClasses[0]!.count)} boxes`;
    }
    return 'No annotations';
  });

  // Target class for quick-jump inspection: first unboxed class if any, else rarest underrepresented / rarest class.
  const jumpTarget = computed<{ cls: number; name: string; label: string; absent: boolean; attr: 'missing' | 'rarest' } | null>(() => {
    const d = data.get();
    if (d.missingClasses.length > 0) {
      const m = d.missingClasses[0]!;
      return { cls: m.cls, name: m.name, label: '0 boxes', absent: true, attr: 'missing' };
    }
    const t = d.underrepresented[0] ?? d.rarest ?? null;
    if (!t) return null;
    return { cls: t.cls, name: t.name, label: d.underrepresented.length > 0 ? 'Inspect rarest' : 'Rarest', absent: false, attr: 'rarest' };
  });

  const nonZeroSplits = computed(() => data.get().splits.filter((s) => s.count > 0));

  const isCleanQuality = computed(() => {
    const d = data.get();
    return d.brokenLabels === 0 && d.imageErrors === 0 && d.emptyImages === 0 && !d.tinyWarn && d.checkErrors === 0 && d.checkWarnings === 0;
  });

  const curateOpen = signal(false);
  let prevDatasetId = props.dataset.id;
  effect(() => {
    const nextId = props.dataset.id;
    if (nextId !== prevDatasetId) {
      prevDatasetId = nextId;
      curateOpen.set(false);
    }
  });

  return (
    <section class="insights" aria-label="Dataset class balance and annotation quality" data-dataset-insights>
      <div class="seg balance">
        <span class={`headline ${data.get().balanceTone}`}>
          <Dynamic component={TONE_ICON[data.get().balanceTone]} size={14} strokeWidth={2} aria-hidden="true" />
          <span class="num">{balanceHeadline.get()}</span>
        </span>
        {data.get().mostCommon && data.get().rarest ? (
          <span class="detail num" title={`${data.get().mostCommon!.name}: ${num(data.get().mostCommon!.count)} boxes · ${data.get().rarest!.name}: ${num(data.get().rarest!.count)} boxes`}>
            <span class="truncate">{data.get().mostCommon!.name}</span> {compact(data.get().mostCommon!.count)}
            <span class="sep" aria-hidden="true">
              vs
            </span>
            <span class="truncate">{data.get().rarest!.name}</span> {compact(data.get().rarest!.count)}
          </span>
        ) : null}
        <Show when={jumpTarget.get()}>
          {(t) => {
            const selected = () => props.activeClass === t().cls;
            return (
              <button
                type="button"
                class="jump"
                class:active={selected()}
                aria-pressed={selected()}
                data-inspect-missing={t().attr === 'missing' ? '' : undefined}
                data-inspect-rarest={t().attr === 'rarest' ? '' : undefined}
                onclick={() => props.onSelectClass(selected() ? undefined : t().cls)}
              >
                <span class="sw" style={swatchStyle(t().cls, t().absent)} aria-hidden="true"></span>
                <span>
                  {t().label}: <b>{t().name}</b>
                </span>
                {selected() ? <X size={12} strokeWidth={2} aria-hidden="true" /> : null}
              </button>
            );
          }}
        </Show>
      </div>

      <div class="seg quality">
        <span class={`headline ${data.get().qualityTone}`}>
          <Dynamic component={TONE_ICON[data.get().qualityTone]} size={14} strokeWidth={2} aria-hidden="true" />
          <span>{isCleanQuality.get() ? 'Labels clean' : 'Quality'}</span>
        </span>
        <span class="facts num">
          {data.get().boxesPerImage !== null ? <span class="fact">{data.get().boxesPerImage!.toFixed(1)} boxes/img</span> : null}
          {data.get().brokenLabels > 0 ? <span class="fact danger">{num(data.get().brokenLabels)} broken</span> : null}
          {data.get().brokenLabels > 0 && data.get().imageErrors > 0 ? (
            <span class="fact danger">{num(data.get().imageErrors)} unreadable</span>
          ) : data.get().brokenLabels === 0 && data.get().checkErrors > 0 ? (
            <span class="fact danger">
              {num(data.get().checkErrors)} file {data.get().checkErrors === 1 ? 'error' : 'errors'}
            </span>
          ) : null}
          {data.get().emptyImages > 0 ? (
            <span class="fact" class:warn={data.get().emptyWarn} data-empty-indicator>
              {num(data.get().emptyImages)} empty ({fmtPct(data.get().emptyPct)})
            </span>
          ) : null}
          {data.get().tinyWarn ? <span class="fact warn">{Math.round(data.get().tinyBoxPct)}% tiny</span> : null}
          {data.get().checkWarnings > 0 ? (
            <span class="fact warn">
              {num(data.get().checkWarnings)} {data.get().checkWarnings === 1 ? 'warning' : 'warnings'}
            </span>
          ) : null}
        </span>
        <button type="button" class="stats-link" onclick={props.onOpenStats}>
          Stats
          <ArrowRight size={12} strokeWidth={2} aria-hidden="true" />
        </button>
        <button
          type="button"
          class="stats-link curate-btn"
          class:open={curateOpen.get()}
          aria-expanded={curateOpen.get()}
          data-curate-toggle
          onclick={() => curateOpen.set(!curateOpen.peek())}
        >
          CLI
          <ChevronDown size={12} strokeWidth={2} aria-hidden="true" />
        </button>
      </div>

      {nonZeroSplits.get().length > 0 ? (
        <div class="seg splits">
          <span class="split-bar" aria-hidden="true">
            <For each={nonZeroSplits.get()} key={(s) => s.split}>
              {(s) => <span class={`sbar ${s().split}`} style:width={`${Math.max(4, s().pct)}%`}></span>}
            </For>
          </span>
          <span class="split-pills num" role="group" aria-label="Split breakdown">
            <For each={nonZeroSplits.get()} key={(s) => s.split}>
              {(s) => {
                const active = () => props.activeSplit === s().split;
                return (
                  <button
                    type="button"
                    class="spill"
                    class:active={active()}
                    aria-pressed={active()}
                    onclick={() => props.onSelectSplit(active() ? 'all' : s().split)}
                  >
                    <span class={`sdot ${s().split}`} aria-hidden="true"></span>
                    {s().split} {fmtPct(s().pct)}
                  </button>
                );
              }}
            </For>
            {data.get().noVal && data.get().totalImages > 0 ? <span class="no-val">no val</span> : null}
          </span>
        </div>
      ) : null}

      {curateOpen.get() ? <DatasetCurateCli dataset={props.dataset} stats={props.stats} activeClass={props.activeClass} /> : null}
    </section>
  );
}

css`
  .insights {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-4);
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    min-width: 0;
  }
  .seg {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .seg.splits {
    margin-left: auto;
  }
  .headline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    white-space: nowrap;
  }
  .headline.danger :global(svg) {
    color: var(--danger-text);
  }
  .headline.warning :global(svg) {
    color: var(--warning-text);
  }
  .headline.ok :global(svg) {
    color: var(--success-text);
  }
  .detail {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--text-tertiary);
    min-width: 0;
    max-width: 240px;
  }
  .detail .sep {
    color: var(--text-tertiary);
    padding: 0 2px;
  }
  .jump {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    background: var(--bg-app);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    white-space: nowrap;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .jump b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .jump:hover {
    border-color: var(--line-strong);
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .jump.active {
    border-color: var(--text-tertiary);
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .facts {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .fact {
    white-space: nowrap;
  }
  .fact + .fact::before {
    content: '·';
    margin-right: var(--space-2);
    color: var(--text-tertiary);
  }
  .fact.danger {
    color: var(--danger-text);
    font-weight: var(--weight-medium);
  }
  .fact.warn {
    color: var(--warning-text);
    font-weight: var(--weight-medium);
  }
  .stats-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 24px;
    padding: 0 var(--space-1);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .stats-link:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .curate-btn :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .curate-btn.open {
    color: var(--text-primary);
    background: var(--bg-active);
  }
  .curate-btn.open :global(svg) {
    transform: rotate(180deg);
  }
  .split-bar {
    display: flex;
    width: 56px;
    height: 6px;
    border-radius: var(--radius-xs);
    overflow: hidden;
    background: var(--bg-active);
    gap: 1px;
    flex-shrink: 0;
  }
  .sbar {
    display: block;
    height: 100%;
  }
  .sbar.train,
  .sdot.train {
    background: var(--accent-solid);
  }
  .sbar.val,
  .sdot.val {
    background: var(--text-secondary);
  }
  .sbar.test,
  .sdot.test {
    background: var(--text-tertiary);
  }
  .split-pills {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
  }
  .spill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 24px;
    padding: 0 6px;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    white-space: nowrap;
    transition: background-color var(--dur-fast) var(--ease-standard), color var(--dur-fast) var(--ease-standard);
  }
  .spill:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .spill.active {
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .sdot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-full);
    flex-shrink: 0;
  }
  .no-val {
    color: var(--warning-text);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    padding-left: 4px;
  }
  @media (pointer: coarse) {
    .jump,
    .stats-link,
    .spill {
      min-height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .seg.splits {
      margin-left: 0;
    }
  }
  @media (max-width: 767px) {
    .insights {
      gap: var(--space-2);
      padding: var(--space-2) var(--space-3);
    }
    .detail {
      max-width: 100%;
    }
  }
`;
