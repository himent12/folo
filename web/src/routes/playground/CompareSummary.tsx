/**
 * Instant Model A vs Model B comparison summary: detection count delta, latency A vs B comparison, per-class diff
 * breakdown (`Both`, `Only A`, `Only B`), per-class counts table, one-click side-by-side PNG & Compare JSON export,
 * and a quick side switcher so users can inspect Model A or Model B rows in the detections panel.
 */
import Download from '$icons/download';
import FileJson from '$icons/file-json';
import { computed, css, effect, For, signal } from 'fw';
import { classColor } from '$lib/design/tokens';
import { ms } from '$lib/format';
import Button from '$lib/ui/Button';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import { formatLatencyComparison, formatSignedDelta, type CompareClassPresence, type CompareSummary } from './compare';
import Swatch from './Swatch';

export interface CompareSummaryPanelProps {
  rail?: boolean;
  summary: CompareSummary;
  modelAName: string;
  modelBName: string;
  hasAnyResult: boolean;
  predictingA: boolean;
  predictingB: boolean;
  inspectSide: 'a' | 'b';
  onInspectSide: (side: 'a' | 'b') => void;
  canExport?: boolean;
  exportingPng?: boolean;
  onExportSideBySidePng?: () => void;
  onExportCompareJson?: () => void;
}

export default function CompareSummaryPanel(props: CompareSummaryPanelProps) {
  const deltaTone = (d: number | null) => (d === null || d === 0 ? 'zero' : d > 0 ? 'pos' : 'neg');
  const countText = (n: number | null) => (n === null ? '—' : String(n));
  const diffFilter = signal<'all' | CompareClassPresence>('all');

  effect(() => {
    const diff = props.summary.classDiff;
    const f = diffFilter.peek();
    if (
      !diff ||
      (f === 'both' && !diff.both.length) ||
      (f === 'only-a' && !diff.onlyA.length) ||
      (f === 'only-b' && !diff.onlyB.length)
    ) {
      diffFilter.set('all');
    }
  });

  const filteredClasses = computed(() => {
    const f = diffFilter.get();
    if (f === 'all' || !props.summary.classDiff) return props.summary.classes;
    return props.summary.classes.filter((r) => r.presence === f);
  });

  function toggleFilter(bucket: CompareClassPresence) {
    diffFilter.set(diffFilter.peek() === bucket ? 'all' : bucket);
  }

  const body = () => (
    <div class="ab-sum">
      {!props.hasAnyResult ? (
        <p class="hint">{props.predictingA || props.predictingB ? 'Running both models…' : 'Load an image to compare Model A and Model B.'}</p>
      ) : (
        <>
          <div class="kpis num">
            <div class="kpi">
              <span class="k-label">Detections</span>
              <div class="k-vals">
                <span>
                  <span class="tag-a">A</span> <b>{countText(props.summary.countA)}</b>
                </span>
                <span class="vs">vs</span>
                <span>
                  <span class="tag-b">B</span> <b>{countText(props.summary.countB)}</b>
                </span>
                {props.summary.countDelta !== null ? (
                  <span class={`delta ${deltaTone(props.summary.countDelta)}`} data-fact>
                    Δ {formatSignedDelta(props.summary.countDelta)}
                  </span>
                ) : null}
              </div>
            </div>
            <div class="kpi">
              <span class="k-label">Latency</span>
              <div class="k-vals">
                <span>
                  <span class="tag-a">A</span> <b>{props.summary.latencyA !== null ? ms(props.summary.latencyA) : '—'}</b>
                </span>
                <span class="vs">vs</span>
                <span>
                  <span class="tag-b">B</span> <b>{props.summary.latencyB !== null ? ms(props.summary.latencyB) : '—'}</b>
                </span>
              </div>
              <span class="k-sub">{formatLatencyComparison(props.summary)}</span>
            </div>
          </div>

          {props.summary.classDiff && props.summary.classes.length ? (
            <div class="diff-strip" role="group" aria-label="Filter classes by model overlap" data-class-diff>
              <button
                type="button"
                class="diff-pill"
                class:on={diffFilter.get() === 'both'}
                aria-pressed={diffFilter.get() === 'both'}
                disabled={!props.summary.classDiff.both.length}
                onclick={() => toggleFilter('both')}
              >
                <span>Both</span>{' '}
                <b class="num">{props.summary.classDiff.both.length}</b>
              </button>
              <button
                type="button"
                class="diff-pill only-a"
                class:on={diffFilter.get() === 'only-a'}
                aria-pressed={diffFilter.get() === 'only-a'}
                disabled={!props.summary.classDiff.onlyA.length}
                onclick={() => toggleFilter('only-a')}
              >
                <span>Only A</span>{' '}
                <b class="num">{props.summary.classDiff.onlyA.length}</b>
              </button>
              <button
                type="button"
                class="diff-pill only-b"
                class:on={diffFilter.get() === 'only-b'}
                aria-pressed={diffFilter.get() === 'only-b'}
                disabled={!props.summary.classDiff.onlyB.length}
                onclick={() => toggleFilter('only-b')}
              >
                <span>Only B</span>{' '}
                <b class="num">{props.summary.classDiff.onlyB.length}</b>
              </button>
            </div>
          ) : null}

          {filteredClasses.get().length ? (
            <div class="cls-wrap">
              <table class="table dense cls-table">
                <thead>
                  <tr>
                    <th scope="col">Class</th>
                    <th scope="col" class="r">
                      A
                    </th>
                    <th scope="col" class="r">
                      B
                    </th>
                    <th scope="col" class="r">
                      Δ
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <For each={filteredClasses.get()} key={(r) => r.name}>
                    {(r) => (
                      <tr>
                        <td class="lead">
                          <span class="cls">
                            <Swatch color={classColor(r().colorClass)} />
                            <span class="truncate">{r().name}</span>
                            {r().presence === 'only-a' ? (
                              <span class="only-tag only-a" title="Detected only by Model A">
                                A
                              </span>
                            ) : r().presence === 'only-b' ? (
                              <span class="only-tag only-b" title="Detected only by Model B">
                                B
                              </span>
                            ) : null}
                          </span>
                        </td>
                        <td class="r num">{countText(r().countA)}</td>
                        <td class="r num">{countText(r().countB)}</td>
                        <td class={`r num d-cell ${deltaTone(r().delta)}`}>{formatSignedDelta(r().delta)}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          ) : (
            <p class="hint">
              {props.summary.classes.length
                ? 'No classes in this filter.'
                : props.summary.countA !== null && props.summary.countB !== null
                  ? 'Neither model kept any detections at the current thresholds.'
                  : 'No detections kept yet at the current thresholds.'}
            </p>
          )}

          {props.onExportSideBySidePng || props.onExportCompareJson ? (
            <div class="exp-row" data-compare-export>
              {props.onExportSideBySidePng ? (
                <Button
                  size="sm"
                  icon={Download}
                  disabled={!props.canExport || !!props.exportingPng}
                  loading={!!props.exportingPng}
                  onclick={props.onExportSideBySidePng}
                >
                  Side-by-side PNG
                </Button>
              ) : null}
              {props.onExportCompareJson ? (
                <Button
                  size="sm"
                  icon={FileJson}
                  disabled={!props.canExport}
                  onclick={props.onExportCompareJson}
                >
                  Compare JSON
                </Button>
              ) : null}
            </div>
          ) : null}

          <div class="inspect-row">
            <span class="k-label">Inspect table</span>
            <Segmented
              label="Inspect detections for"
              size="sm"
              value={props.inspectSide}
              onchange={props.onInspectSide}
              options={[
                { value: 'a', label: 'Model A', meta: props.summary.countA !== null ? String(props.summary.countA) : undefined },
                { value: 'b', label: 'Model B', meta: props.summary.countB !== null ? String(props.summary.countB) : undefined },
              ]}
            />
          </div>
        </>
      )}
    </div>
  );

  return props.rail ? (
    body()
  ) : (
    <div class="ab-sum-panel">
      <Panel title="A/B Summary" meta={`${props.modelAName} vs ${props.modelBName}`} divided>
        {body()}
      </Panel>
    </div>
  );
}

css`
  .ab-sum {
    display: grid;
    gap: var(--space-3);
  }
  .ab-sum-panel {
    order: 2;
  }
  .hint {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .kpis {
    display: grid;
    gap: var(--space-2);
  }
  .kpi {
    display: grid;
    gap: 2px;
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  .k-label {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .k-vals {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .k-vals b {
    color: var(--text-primary);
    font-weight: var(--weight-semibold);
  }
  .k-sub {
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
  }
  .vs {
    color: var(--text-tertiary);
  }
  .tag-a,
  .tag-b {
    display: inline-block;
    font-weight: var(--weight-semibold);
    margin-right: 2px;
  }
  .tag-a {
    color: var(--accent-text);
  }
  .tag-b {
    color: var(--info-text);
  }
  .delta {
    margin-left: auto;
    padding: 0 6px;
    border-radius: var(--radius-xs);
    font-weight: var(--weight-medium);
    background: var(--bg-raised);
    color: var(--text-secondary);
  }
  .delta.pos,
  .d-cell.pos {
    color: var(--success-text);
  }
  .delta.neg,
  .d-cell.neg {
    color: var(--warning-text);
  }
  .delta.zero,
  .d-cell.zero {
    color: var(--text-tertiary);
  }
  .cls-wrap {
    margin: 0 calc(-1 * var(--space-2));
    max-height: 200px;
    overflow-y: auto;
  }
  .cls-table .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    max-width: 140px;
  }
  .diff-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-1);
  }
  .diff-pill {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-1);
    min-height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-subtle);
    background: var(--bg-inset);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .diff-pill:hover:not(:disabled) {
    border-color: var(--line-strong);
    color: var(--text-primary);
  }
  .diff-pill.on {
    border-color: var(--line-strong);
    background: var(--bg-raised);
    color: var(--text-primary);
  }
  .diff-pill:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .diff-pill b {
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
  }
  .diff-pill.only-a b {
    color: var(--accent-text);
  }
  .diff-pill.only-b b {
    color: var(--info-text);
  }
  .only-tag {
    flex-shrink: 0;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-semibold);
    background: var(--bg-raised);
  }
  .only-tag.only-a {
    color: var(--accent-text);
  }
  .only-tag.only-b {
    color: var(--info-text);
  }
  .exp-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .inspect-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
`;
