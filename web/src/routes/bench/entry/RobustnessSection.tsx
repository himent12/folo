/**
 * Where it breaks (docs/BENCH.md, Robustness): the plan's conditions × severities as a heat table of accuracy kept
 * (`kept = min(1, mAP under / mAP unchanged)`), the number in every cell and the tone for what was lost, rows from the
 * worst kept to the best, each with its group, its mean (the condition's value) and its weight; severities the plan
 * does not run are empty; the axis at the foot. Beside it, accuracy against severity for the condition chosen (or
 * pointed at) over every other condition's as context lines, from the unchanged level, so the shape of the fall shows;
 * and the condition at that severity on a benchmark image (./Preview). The table is one keyboard stop: the arrows
 * move the cell, which is the choice.
 */
import { computed, css, effect, For, signal } from 'fw';
import type { Bench, BenchCatalog, BenchEntry, BenchResult } from '$lib/api/types';
import { GROUP_NAME, SEVERITIES } from '$lib/bench/rules';
import XYChart from '$lib/charts/XYChart';
import type { XYSeries } from '$lib/charts/xy';
import { metric, num } from '$lib/format';
import { prefs } from '$lib/prefs';
import Panel from '$lib/ui/Panel';
import { byWorstKept, severityDomain, severityPoints } from '../charts';
import { heatLabelHigh, heatShare } from '../heat';
import Preview from '../Preview';
import { keptText, robustnessReading } from '../words';

export interface RobustnessSectionProps {
  result: BenchResult;
  entry: BenchEntry;
  bench: Bench;
  catalog: BenchCatalog | undefined;
}

const PLOT = 200;
const KEYS: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };


export default function RobustnessSection(props: RobustnessSectionProps) {
  const rob = computed(() => props.result.robustness!);
  const clean = computed(() => props.result.accuracy.map50_95);
  const rows = computed(() => byWorstKept(rob.get().conditions));
  const nameOf = (id: string) => props.catalog?.conditions.find((c) => c.id === id)?.name ?? id;
  const groupOf = (id: string) => {
    const g = props.catalog?.conditions.find((c) => c.id === id)?.group;
    return g ? GROUP_NAME[g] : '';
  };
  const theme = () => prefs.resolvedTheme;

  // The cell chosen: at first the worst cell of the worst condition (where it breaks most).
  const worst = computed(() => {
    const c = rows.get()[0];
    const p = c?.points.reduce((a, b) => (b.retention < a.retention ? b : a), c.points[0]!);
    return { condition: c?.condition ?? 'haze', severity: p?.severity ?? 3 };
  });
  const chosen = signal<{ condition: string; severity: number } | null>(null);
  const sel = computed(() => chosen.get() ?? worst.get());
  // Preview's own pickers move the choice too.
  const previewCond = signal('');
  const previewSev = signal(3);
  effect(() => {
    const s = sel.get();
    previewCond.set(s.condition);
    previewSev.set(s.severity);
  });
  effect(() => {
    const pc = previewCond.get();
    const ps = previewSev.get();
    const s = sel.get();
    if (!pc || (pc === s.condition && ps === s.severity)) return;
    // Only a cell the table has: the preview also shows conditions and severities the plan does not run.
    const c = rob.get().conditions.find((x) => x.condition === pc);
    if (c?.points.some((p) => p.severity === ps)) chosen.set({ condition: pc, severity: ps });
  });
  const pointed = signal<string | null>(null);
  const focusCond = computed(() => pointed.get() ?? sel.get().condition);
  const focused = computed(() => rob.get().conditions.find((c) => c.condition === focusCond.get()) ?? rows.get()[0]);

  // ---- the heat table's keys: arrows move the chosen cell over the cells that ran ----
  let tableEl: HTMLTableElement | undefined;
  function move(dr: number, dc: number) {
    const list = rows.peek();
    const at = sel.peek();
    let r = list.findIndex((c) => c.condition === at.condition);
    let s = at.severity;
    for (let k = 0; k < 6; k++) {
      if (dr) r = Math.max(0, Math.min(list.length - 1, r + dr));
      if (dc) s = Math.max(1, Math.min(5, s + dc));
      const row = list[r]!;
      if (row.points.some((p) => p.severity === s)) break;
      // Down or up into a row without that severity: its nearest one.
      if (dr && !dc) {
        s = row.points.reduce((a, p) => (Math.abs(p.severity - s) < Math.abs(a - s) ? p.severity : a), row.points[0]!.severity);
        break;
      }
    }
    const row = list[r]!;
    if (!row.points.some((p) => p.severity === s)) return;
    chosen.set({ condition: row.condition, severity: s });
    queueMicrotask(() => tableEl?.querySelector<HTMLElement>('button.cell[tabindex="0"]')?.focus());
  }
  function onkeydown(e: KeyboardEvent) {
    const step = KEYS[e.key];
    if (!step) return;
    e.preventDefault();
    move(step[0], step[1]);
  }

  // ---- accuracy by severity ----
  const fit = computed(() => severityDomain(clean.get(), rob.get().conditions, PLOT));
  const series = computed<XYSeries[]>(() =>
    rob.get().conditions.map((c) => ({
      id: c.condition,
      label: nameOf(c.condition),
      color: 'var(--text-primary)',
      points: severityPoints(c, clean.get()),
      dots: c.condition === focusCond.get(),
    })),
  );
  const cursor = signal<number | null>(null);
  const pointAt = (s: number) => focused.get()?.points.find((p) => p.severity === s);
  const reading = computed(() => robustnessReading(rob.get().conditions, nameOf));

  const readout = (c: () => number | null, hint: () => string | null) => {
    const p = computed(() => {
      const v = c();
      return v === null || v === 0 ? undefined : pointAt(v);
    });
    return (
      <>
        {hint() ? (
          <span class="hint">{hint()}</span>
        ) : c() === null || c() === 0 ? (
          <>
            <span class="key">
              <i class="sw" aria-hidden="true"></i>
              {nameOf(focusCond.get())}
            </span>
            <span class="sep">·</span>
            <span>{keptText(focused.get()?.retention ?? 0)} kept on average</span>
          </>
        ) : (
          <>
            <span class="key">
              <i class="sw" aria-hidden="true"></i>
              {nameOf(focusCond.get())} {c()}
            </span>
            <span class="sep">·</span>
            {p.get() ? (
              <>
                <span class="v">mAP50-95 {metric(p.get()!.map50_95)}</span>
                <span class="sep">·</span>
                <span>{keptText(p.get()!.retention)} kept</span>
              </>
            ) : (
              <span>not run</span>
            )}
          </>
        )}
      </>
    );
  };

  return (
    <div class="rob" id="robustness">
      <Panel title="Robustness" meta="accuracy kept under each condition" pad="none">
        <p class="lead">
          {reading.get()} Darker cells lost more; each cell is the accuracy kept at that severity, of {metric(clean.get())} unchanged.
        </p>
        <div class="scroll">
          <table
            class="heat"
            ref={(e) => (tableEl = e)}
            aria-label="Accuracy kept by condition and severity. Arrow keys move between cells."
            onkeydown={onkeydown}
          >
            <thead>
              <tr>
                <th class="h-cond">Condition</th>
                <For each={SEVERITIES} key={(s) => s}>
                  {(s) => <th class="h-sev num">{s()}</th>}
                </For>
                <th class="h-mean num">Mean</th>
                <th class="h-w num">Weight</th>
              </tr>
            </thead>
            <tbody>
              <For each={rows.get()} key={(c) => c.condition}>
                {(c) => (
                  <tr class:on={c().condition === sel.get().condition} onpointerenter={() => pointed.set(c().condition)} onpointerleave={() => pointed.set(null)}>
                    <th scope="row" class="cond">
                      <span class="c-name">{nameOf(c().condition)}</span>
                      <span class="c-group">
                        {groupOf(c().condition)}
                        <span class="w-inline">
                          {' '}· ×{num(c().weight, c().weight % 1 ? 1 : 0)}
                        </span>
                      </span>
                    </th>
                    <For each={SEVERITIES} key={(s) => s}>
                      {(s) => {
                        const p = () => c().points.find((x) => x.severity === s());
                        const on = () => c().condition === sel.get().condition && s() === sel.get().severity;
                        return (
                          <td class="sev">
                            {p() ? (
                              <button
                                type="button"
                                class="cell num"
                                class:on={on()}
                                class:hi={heatLabelHigh(p()!.retention, theme())}
                                style={{ '--s': heatShare(p()!.retention, theme()) }}
                                tabindex={on() ? 0 : -1}
                                aria-pressed={on()}
                                aria-label={`${nameOf(c().condition)}, severity ${s()}: ${keptText(p()!.retention)} kept, mAP50-95 ${metric(p()!.map50_95)}`}
                                onclick={() => chosen.set({ condition: c().condition, severity: s() })}
                                onfocus={() => pointed.set(null)}
                              >
                                {keptText(p()!.retention)}
                              </button>
                            ) : (
                              <span class="nil" aria-label={`Severity ${s()}: not in the plan`}></span>
                            )}
                          </td>
                        );
                      }}
                    </For>
                    <td class="mean num">{keptText(c().retention)}</td>
                    <td class="w num">×{num(c().weight, c().weight % 1 ? 1 : 0)}</td>
                  </tr>
                )}
              </For>
            </tbody>
            <tfoot>
              <tr>
                <th scope="row" class="cond">
                  Robustness
                </th>
                <td colspan={5} class="foot-why">
                  the conditions’ means, weighted
                </td>
                <td class="mean num axis">{keptText(rob.get().retention)}</td>
                <td></td>
              </tr>
            </tfoot>
          </table>
        </div>
        {rob.get().conditions.length < props.bench.plan.conditions.length ? (
          <p class="lead note">
            It ran {rob.get().conditions.length} of the plan’s {props.bench.plan.conditions.length} conditions: the axis is over those.
          </p>
        ) : null}
      </Panel>

      <div class="side">
        <Panel title="Accuracy by severity" meta="mAP50-95, from unchanged">
          <XYChart
            label={`mAP50-95 of ${nameOf(focusCond.get())} by severity, the other conditions as context`}
            series={series.get()}
            highlight={focusCond.get()}
            xDomain={[0, 5]}
            yDomain={fit.get().domain}
            xTicks={[0, 1, 2, 3, 4, 5]}
            yTicks={fit.get().ticks}
            yFormat={(v) => v.toFixed(fit.get().decimals)}
            xFormat={(v) => (v === 0 ? 'none' : String(v))}
            xLabel="Severity"
            height={PLOT + 52}
            refs={[{ axis: 'y', value: clean.get(), label: `unchanged ${metric(clean.get())}` }]}
            stops={[0, ...(focused.get()?.points.map((p) => p.severity) ?? [])]}
            bind:cursor={cursor}
            readout={readout}
            table={{
              caption: `mAP50-95 by severity for ${nameOf(focusCond.get())}`,
              head: ['Severity', 'mAP50-95', 'Kept'],
              rows: [
                ['none', metric(clean.get()), '100%'],
                ...(focused.get()?.points ?? []).map((p) => [String(p.severity), metric(p.map50_95), keptText(p.retention)]),
              ],
            }}
          />
        </Panel>
        {props.catalog ? (
          <Preview bench={props.bench} catalog={props.catalog} bind:condition={previewCond} bind:severity={previewSev} />
        ) : null}
      </div>
    </div>
  );
}

css`
  /* The table on 7 of the page's 12 columns, the chart and the preview on 5; stacked under 1024 px. */
  .rob {
    display: grid;
    grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
    gap: var(--space-4);
    align-items: start;
  }
  .side {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .lead {
    padding: 0 var(--space-4) var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .lead.note {
    padding-top: var(--space-3);
    color: var(--text-tertiary);
  }
  .scroll {
    overflow-x: auto;
  }
  .heat {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
  }
  .heat th,
  .heat td {
    padding: 0;
    height: 36px;
    border-top: 1px solid var(--line-subtle);
  }
  .heat thead th {
    height: 32px;
    border-top: 0;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-tertiary);
  }
  .heat .h-cond,
  .heat .cond {
    padding-left: var(--space-4);
    text-align: left;
  }
  .h-sev {
    width: 60px;
    text-align: center;
  }
  .heat .h-mean,
  .heat .h-w {
    width: 64px;
    text-align: right;
    padding-right: var(--space-4);
  }
  .heat .h-mean {
    width: 72px;
  }
  .cond {
    font-weight: var(--weight-regular);
  }
  .c-name {
    display: block;
    color: var(--text-primary);
    white-space: nowrap;
  }
  .c-group {
    display: block;
    font-size: var(--type-sm-size);
    line-height: 14px;
    color: var(--text-tertiary);
  }
  .heat tbody th,
  .heat tbody td {
    height: 44px;
  }
  /* A selected row: the one selection rule (bg.active and the ink bar at its leading edge). */
  .heat tbody tr.on th {
    box-shadow: var(--select-bar);
  }
  .heat tbody tr.on > * {
    background: var(--bg-active);
  }
  .heat .sev {
    padding: 2px 1px;
  }
  /* A cell: ink mixed into the panel for what was lost (./heat), its number in the label ink that keeps 4.5:1. */
  .cell {
    display: grid;
    place-items: center;
    width: 100%;
    height: 38px;
    border-radius: var(--radius-sm);
    background: color-mix(in oklab, var(--text-primary) calc(var(--s) * 100%), var(--bg-panel));
    box-shadow: inset 0 0 0 1px var(--line-subtle);
    font-size: var(--type-sm-size);
  }
  :global([data-theme='dark']) .cell {
    color: #ffffff;
  }
  :global([data-theme='dark']) .cell.hi,
  :global([data-theme='light']) .cell {
    color: #000000;
  }
  :global([data-theme='light']) .cell.hi {
    color: #ffffff;
  }
  /* The chosen cell: a 2 px ink frame (a selection is ink, never lime), the matrix's pinned cell. */
  .cell.on {
    box-shadow:
      inset 0 0 0 2px var(--text-primary),
      inset 0 0 0 3px var(--bg-panel);
    font-weight: var(--weight-semibold);
  }
  .cell:focus-visible {
    outline-offset: 1px;
  }
  .nil {
    display: block;
    width: 100%;
    height: 38px;
  }
  .heat .mean,
  .heat .w {
    text-align: right;
    padding-right: var(--space-4);
  }
  .mean {
    font-weight: var(--weight-medium);
  }
  .w {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .heat tfoot th,
  .heat tfoot td {
    height: 40px;
    border-top: 1px solid var(--line-default);
  }
  .heat .foot-why {
    padding-left: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .axis {
    font-weight: var(--weight-semibold);
  }
  /* The weight beside the group, where the weight column steps out (phones). */
  .w-inline {
    display: none;
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-primary);
  }
  .sw {
    width: 14px;
    height: 2px;
    border-radius: 1px;
    background: var(--text-primary);
  }
  .sep {
    margin: 0 var(--space-2);
    color: var(--text-tertiary);
  }
  .v {
    color: var(--text-primary);
  }
  .hint {
    color: var(--text-secondary);
  }
  @media (max-width: 1023px) {
    .rob {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  /* Phones: the cells narrow to their numbers; the group under the name. */
  @media (max-width: 767px) {
    .h-sev {
      width: 44px;
    }
    .heat .h-mean {
      width: 48px;
      padding-right: var(--space-3);
    }
    .heat .mean {
      padding-right: var(--space-3);
    }
    .heat .h-w,
    .heat .w {
      display: none;
    }
    .w-inline {
      display: inline;
    }
    .foot-why {
      font-size: var(--type-sm-size);
    }
    .c-name {
      white-space: normal;
    }
  }
`;
