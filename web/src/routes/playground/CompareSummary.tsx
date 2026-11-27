/**
 * Instant Model A vs Model B comparison summary: detection count delta, latency A vs B comparison, per-class counts
 * table, and a quick side switcher so users can inspect Model A or Model B rows in the detections panel.
 */
import { css, For } from 'fw';
import { classColor } from '$lib/design/tokens';
import { ms } from '$lib/format';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import { formatLatencyComparison, formatSignedDelta, type CompareSummary } from './compare';
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
}

export default function CompareSummaryPanel(props: CompareSummaryPanelProps) {
  const deltaTone = (d: number | null) => (d === null || d === 0 ? 'zero' : d > 0 ? 'pos' : 'neg');
  const countText = (n: number | null) => (n === null ? '—' : String(n));

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

          {props.summary.classes.length ? (
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
                  <For each={props.summary.classes} key={(r) => r.name}>
                    {(r) => (
                      <tr>
                        <td class="lead">
                          <span class="cls">
                            <Swatch color={classColor(r().colorClass)} />
                            <span class="truncate">{r().name}</span>
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
            <p class="hint">Neither model kept any detections at the current thresholds.</p>
          )}

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
  .inspect-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
`;
