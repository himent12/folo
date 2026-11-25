/**
 * Side-by-side comparison table (desktop/tablet) and stacked entry-by-entry list (phones) for BenchCompare:
 * scores and five axes against the first entry (the baseline), with "Make baseline" swaps and best-in-row highlights.
 */
import { computed, css, For, Index } from 'fw';
import type { BenchEntry } from '$lib/api/types';
import NoValue from '$lib/ui/NoValue';
import Panel from '$lib/ui/Panel';
import { diffAsPrinted } from '../charts';
import type { Row } from './state';

export interface SideBySideProps {
  shown: BenchEntry[];
  rows: Row[];
  /** Entries graded on an earlier benchmark fingerprint (never marked best or diffed). */
  old: boolean[];
  best: (key: string, values: (number | null)[], d: number) => Set<number>;
  makeBaseline: (eid: string) => void;
  /** Renders the slot colour/pattern swatch for an entry id. */
  keyMark: (eid: string) => JSX.Element;
}

export default function SideBySide(props: SideBySideProps) {
  let tableEl: HTMLTableElement | undefined;
  /** A value's difference from the baseline's, unless either is graded on an earlier version. */
  const diffOf = (row: () => Row, v: () => number, i: number) =>
    props.old[i] || props.old[0] ? null : diffAsPrinted(v(), row().values[0] ?? null, row().d);

  function swapBaseline(eid: string) {
    props.makeBaseline(eid);
    // The clicked "Make baseline" button unmounts as this entry moves to column 0: keep keyboard focus inside the table.
    queueMicrotask(() => {
      if (document.activeElement && document.activeElement !== document.body && tableEl?.contains(document.activeElement)) return;
      (tableEl?.querySelector<HTMLButtonElement>('button.swap') ?? tableEl)?.focus();
    });
  }

  const cmpCell = (row: () => Row, b: () => Set<number>, v: () => number | null, i: number) => (
    <td class="r num" class:top={b().has(i)}>
      {v() === null ? (
        <NoValue label="not measured" />
      ) : (
        <>
          {b().has(i) ? <span class="best">best</span> : null}
          {row().fmt(v()!)}{' '}
          {i > 0 && diffOf(row, v as () => number, i) ? <span class="dd">{diffOf(row, v as () => number, i)}</span> : null}
        </>
      )}
    </td>
  );

  const sbsValue = (row: () => Row, b: () => Set<number>, e: () => BenchEntry, i: () => number) => {
    const v = () => row().values[i()] ?? null;
    return (
      <span class="sbs-v" class:top={b().has(i())}>
        <span class="sbs-n">
          {props.keyMark(e().id)}
          <span class="truncate">{e().name}</span>
        </span>{' '}
        <span class="num">
          {v() === null ? (
            <NoValue label="not measured" />
          ) : (
            <>
              {row().fmt(v()!)}
              {i() > 0 ? (
                <>{diffOf(row, v as () => number, i()) ? <span class="dd">{diffOf(row, v as () => number, i())}</span> : null}</>
              ) : (
                <span class="dd">baseline</span>
              )}
              {b().has(i()) ? <span class="dd">best</span> : null}
            </>
          )}
        </span>
      </span>
    );
  };

  return (
    <Panel title="Side by side" meta="against the first, the baseline" pad="none">
      <div class="scroll">
        <table ref={(el) => (tableEl = el)} tabindex={-1} class="table cmp" aria-label="Scores and axes side by side">
          <thead>
            <tr>
              <th scope="col" class="lead m-col">
                <span class="sr-only">Measure</span>
              </th>
              <For each={props.shown} key={(e) => e.id}>
                {(e, i) => (
                  <th scope="col" class="r e-col">
                    <span class="eh">
                      {props.keyMark(e().id)}
                      <span class="eh-n">{e().name}</span>
                    </span>{' '}
                    <span class="role">
                      {i() === 0 ? (
                        'Baseline'
                      ) : (
                        <button type="button" class="swap" aria-label={`Make baseline: ${e().name}`} onclick={() => swapBaseline(e().id)}>
                          Make baseline
                        </button>
                      )}
                    </span>
                  </th>
                )}
              </For>
            </tr>
          </thead>
          <tbody>
            <For each={props.rows} key={(r) => r.key}>
              {(row) => {
                const b = computed(() => props.best(row().key, row().values, row().d));
                return (
                  <tr class:total={row().key === 'score'}>
                    <th scope="row" class="lead">
                      {row().label}
                    </th>
                    <Index each={row().values}>{(v, i) => cmpCell(row, () => b.get(), v, i)}</Index>
                  </tr>
                );
              }}
            </For>
          </tbody>
        </table>
      </div>
      {/* Phones: one block per measure, each entry's value (and its difference from the baseline) in a 2 × 2 grid. */}
      <ul class="sbs" aria-label="Scores and axes, entry by entry">
        <For each={props.rows} key={(r) => r.key}>
          {(row) => {
            const b = computed(() => props.best(row().key, row().values, row().d));
            return (
              <li>
                <span class="sbs-h">{row().label}</span>{' '}
                <span class="sbs-g">
                  <For each={props.shown} key={(e) => e.id}>
                    {(e, i) => sbsValue(row, () => b.get(), e, i)}
                  </For>
                </span>
              </li>
            );
          }}
        </For>
      </ul>
    </Panel>
  );
}

css`
  .scroll {
    overflow-x: auto;
  }
  /* The measure a fixed column, the entries sharing the rest equally (DESIGN_DIRECTION §2.7: a comparison). */
  .cmp {
    table-layout: fixed;
  }
  .m-col {
    width: 200px;
  }
  .cmp thead th {
    height: auto;
    padding-top: var(--space-2);
    padding-bottom: var(--space-2);
    vertical-align: bottom;
  }
  .eh {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    max-width: 100%;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    white-space: normal;
  }
  .eh-n {
    overflow-wrap: anywhere;
    text-align: right;
  }
  .role {
    display: block;
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .swap {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .cmp tbody th.lead {
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
  }
  .cmp tbody tr:last-child th {
    border-bottom: 0;
  }
  .cmp tr.total :is(th, td) {
    font-weight: var(--weight-semibold);
  }
  .top {
    font-weight: var(--weight-semibold);
  }
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .dd {
    display: inline-block;
    min-width: 4.5em;
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    text-align: right;
  }
  .sbs {
    display: none;
  }
  /* Narrower columns: the difference under its value. */
  @media (max-width: 1279px) {
    .cmp .dd {
      display: block;
      min-width: 0;
      margin-left: 0;
    }
  }
  @media (max-width: 767px) {
    .scroll:has(.cmp) {
      display: none;
    }
    .sbs {
      display: grid;
    }
    .sbs li {
      display: grid;
      gap: var(--space-2);
      padding: var(--space-3) var(--space-4);
      border-top: 1px solid var(--line-subtle);
    }
    .sbs li:first-child {
      border-top: 0;
    }
    .sbs-h {
      font-weight: var(--weight-medium);
    }
    .sbs-g {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-2) var(--space-4);
    }
    .sbs-v {
      display: grid;
      min-width: 0;
    }
    .sbs-n {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      min-width: 0;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .sbs-v .dd {
      min-width: 0;
    }
  }
`;
