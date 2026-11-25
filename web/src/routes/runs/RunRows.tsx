/**
 * The Runs list on phones: stacked rows with the essentials (selection, name, status, score, progress), the table's
 * rows in the same order. Select mode: Compare (with its count) the primary once two runs are selected, Cancel a ghost
 * one, both 44 px targets on touch.
 */
import GitCompare from '$icons/git-compare-arrows';
import { computed, css, For } from 'fw';
import { queuePlace } from '$lib/api/queue';
import { runProgress } from '$lib/api/progress';
import type { Run } from '$lib/api/types';
import { ago, metric } from '$lib/format';
import { failureCause } from '$lib/oom';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge, { statusWord } from '$lib/ui/StatusBadge';
import { tooltip } from '$lib/ui/tooltip';
import { COMPARE_CAP_TIP, COMPARE_MAX } from './compareHint';
import { exactTime } from './when';

export interface RunRowsProps {
  /** The rows shown (filtered and sorted); empty with `loading`: the rows' skeleton. */
  runs: Run[];
  loading: boolean;
  selected: string[];
  toggle: (id: string) => void;
  /** Select mode: the checkboxes show (picked, or a run is selected). */
  choosing: boolean;
  onpick: () => void;
  oncancel: () => void;
  /** Compare, gated (the page's Gate around its button). */
  compare: () => JSX.Element;
  /** Compare's prerequisite under the head while choosing (null: none, offline). */
  hint: string | null;
  /** Every row keeps the ⋯ slot while any row shown has a menu, so all rows share one right edge. */
  menuSlot: boolean;
  items: (r: Run) => MenuItem[];
  placeOf: (r: Run) => number | null;
  /** The name of a run's dataset (as recorded with the run once it is removed). */
  datasetName: (r: Run) => string;
  /** The clock relative times read (it ticks). */
  now: number;
}

export default function RunRows(props: RunRowsProps) {
  return (
    <>
      {props.runs.length ? (
        <>
          <div class="rows-head" class:slot={props.menuSlot && !props.choosing}>
            {props.choosing ? (
              <>
                <Button variant="ghost" onclick={props.oncancel}>
                  Cancel
                </Button>{' '}
                {props.compare()}
              </>
            ) : (
              <Button variant="ghost" icon={GitCompare} onclick={props.onpick}>
                Select to compare
              </Button>
            )}{' '}
            <span class="col" aria-hidden="true">
              Best mAP50-95
            </span>
          </div>{' '}
          {props.choosing && props.hint ? (
            <p class="rows-hint" role="status" data-gate-why>
              {props.hint}
            </p>
          ) : null}{' '}
          <ul class="rows" class:choosing={props.choosing}>
            <For each={props.runs} key={(r) => r.id}>
              {(r) => {
                const on = () => props.selected.includes(r().id);
                const capped = () => !on() && props.selected.length >= COMPARE_MAX;
                const cause = () => failureCause(r());
                const items = computed(() => props.items(r()));
                return (
                  <li class:on={on()}>
                    <label class="r-check" use:tooltip={capped() ? COMPARE_CAP_TIP : null}>
                      <input
                        type="checkbox"
                        checked={on()}
                        disabled={capped()}
                        aria-label={`Select ${r().name} for comparison`}
                        onchange={() => props.toggle(r().id)}
                      />
                    </label>{' '}
                    <a class="r-main" href={href(`/runs/${r().id}`)}>
                      <span class="r-text">
                        <span class="r-name">
                          <StatusBadge run={r()} plain dotOnly />
                          <span class="name truncate" title={r().name}>
                            {r().name}
                          </span>
                        </span>{' '}
                        {/* A failure's cause is a line of its own (it is the news), with the epoch and the whole
                            dataset under it; otherwise the status leads that line. Lines wrap whole items, so no
                            fact is cut (DESIGN_DIRECTION §2.3). With the checkboxes in, the dataset steps out. */}
                        {cause() ? (
                          <span class="r-cause" data-fact>
                            {cause()}
                          </span>
                        ) : null}{' '}
                        <span class="r-meta num">
                          <Meta>
                            {!cause() ? <span>{props.placeOf(r()) !== null ? queuePlace(props.placeOf(r()) ?? 0) : statusWord(r())}</span> : null}
                            <span>
                              {r().epoch} / {r().epochs}
                            </span>
                            {/* A free-text name, the one item that may cut itself (ui/Meta), whole in its tooltip. */}
                            {!props.choosing ? (
                              <span class="truncate" title={props.datasetName(r())}>
                                {props.datasetName(r())}
                              </span>
                            ) : null}
                          </Meta>
                        </span>
                      </span>{' '}
                      <span class="r-end">
                        <span class="r-val num">{r().best ? metric(r().best!.map50_95) : <NoValue />}</span>
                        <time class="r-age" datetime={r().created} title={exactTime(r().created)}>
                          {ago(r().created, props.now)}
                        </time>
                      </span>
                    </a>{' '}
                    {!props.choosing && props.menuSlot ? (
                      <span class="r-menu">{items.get().length ? <Menu label={`${props.placeOf(r()) !== null ? 'Queue actions' : 'Actions'} for ${r().name}`} items={items.get()} variant="ghost" fixed /> : null}</span>
                    ) : null}{' '}
                    {r().status === 'running' ? <span class="r-prog" aria-hidden="true" style:transform={`scaleX(${runProgress(r())})`}></span> : null}
                  </li>
                );
              }}
            </For>
          </ul>
        </>
      ) : props.loading ? (
        // Phones, loading: the stacked rows' shape (name over one meta line, the score on the right).
        <ul class="rows sk" aria-hidden="true">
          <For each={[0, 1, 2, 3, 4]} key={(i) => i}>
            {() => (
              <li>
                <span class="r-main">
                  <span class="r-text">
                    <Skeleton w="55%" h="14px" />
                    <Skeleton w="75%" h="10px" />
                  </span>{' '}
                  <span class="r-end">
                    <Skeleton w="40px" h="14px" />
                  </span>
                </span>
              </li>
            )}
          </For>
        </ul>
      ) : null}
    </>
  );
}

css`
  .rows,
  .rows-head,
  .rows-hint {
    display: none;
  }
  .rows-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .rows {
      display: block;
    }
    .rows-head {
      display: flex;
      justify-content: space-between;
      gap: var(--space-2);
      height: 48px;
      align-items: center;
      padding: 0 var(--space-4) 0 var(--space-1);
      font-size: var(--type-sm-size);
      color: var(--text-tertiary);
      border-bottom: 1px solid var(--line-subtle);
    }
    /* Phones: 56 px rows (status and name, one meta line, the score on the right), like the desktop table's essentials. */
    .rows li {
      position: relative;
      display: flex;
      align-items: center;
      min-height: 56px;
      padding-right: var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    /* Not choosing: no checkbox column, the row starts at the gutter. */
    .rows:not(.choosing) .r-check {
      display: none;
    }
    .rows:not(.choosing) li {
      padding-left: var(--space-4);
    }
    /* Phones: under Cancel and Compare, on the rows' edge, before the first row (one rule under both). */
    .rows-head:has(+ .rows-hint) {
      border-bottom: 0;
    }
    .rows-hint {
      display: block;
      padding: 0 var(--space-4) var(--space-3);
      border-bottom: 1px solid var(--line-subtle);
    }
    .col {
      margin-left: auto;
    }
    .rows li:last-child {
      border-bottom: 0;
    }
    .rows li.on {
      background: var(--bg-active);
      box-shadow: var(--select-bar);
    }
    /* The ⋯ slot: on every row while any row has a menu (empty where it has none), so values share one right edge. */
    .r-menu {
      flex-shrink: 0;
      display: flex;
      justify-content: flex-end;
      width: 32px;
      margin-left: var(--space-2);
    }
    .rows-head.slot {
      padding-right: calc(var(--space-4) + 40px);
    }
    .r-check {
      display: grid;
      place-items: center;
      flex-shrink: 0;
      width: 44px;
      height: 56px;
    }
    .r-check:has(> input:disabled) {
      cursor: not-allowed;
    }
    /* The checkbox itself is the 44 px target (DESIGN_DIRECTION §2.10), not only its label: the input is a 44 px box
       whose drawn mark is the 20 px square at its centre, ink when checked as everywhere (§2.6). */
    .r-check input[type='checkbox'] {
      -webkit-appearance: none;
      appearance: none;
      position: relative;
      width: 44px;
      height: 44px;
      margin: 0;
      border-radius: var(--radius-sm);
      cursor: pointer;
    }
    .r-check input[type='checkbox']::before {
      content: '';
      position: absolute;
      inset: 12px;
      border-radius: 4px;
      background: var(--bg-raised);
      box-shadow: inset 0 0 0 1.5px var(--line-control);
    }
    .r-check input[type='checkbox']:checked::before {
      background: var(--text-primary);
      box-shadow: none;
    }
    /* The check: a short L turned 45°, in the inverse ink on the filled box. */
    .r-check input[type='checkbox']:checked::after {
      content: '';
      position: absolute;
      left: 50%;
      top: 50%;
      width: 6px;
      height: 12px;
      border: solid var(--text-inverse);
      border-width: 0 2px 2px 0;
      transform: translate(-50%, -60%) rotate(45deg);
    }
    .r-check input[type='checkbox']:disabled {
      cursor: not-allowed;
      pointer-events: none;
    }
    /* Off (four runs picked, the most Compare takes): an outline only, the control edge at half strength, so it never
       reads as filled, and a filled square means checked (round 26: a dark inset fill read as checked). */
    .r-check input[type='checkbox']:disabled::before {
      background: transparent;
      box-shadow: inset 0 0 0 1.5px color-mix(in srgb, var(--line-control) 50%, transparent);
    }
    /* Focus rings the drawn square, not the invisible target around it. */
    .r-check input[type='checkbox']:focus-visible {
      outline: none;
    }
    .r-check input[type='checkbox']:focus-visible::before {
      outline: var(--size-focus-ring) solid var(--focus);
      outline-offset: var(--size-focus-offset);
    }
    /* 56 px at least; a row that grows a line (a failure's cause) keeps 8 px above and below its text. */
    .r-main {
      flex: 1;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding-block: var(--space-2);
    }
    .r-text {
      flex: 1;
      min-width: 0;
      display: grid;
      /* One column no wider than the row: a 64-character run name truncates instead of widening the page. */
      grid-template-columns: minmax(0, 1fr);
      gap: 2px;
    }
    .r-name .name {
      min-width: 0;
      font-weight: var(--weight-medium);
    }
    .r-name {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      min-width: 0;
    }
    .r-meta,
    .r-cause {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
    /* What went wrong, in words, over the epoch and dataset: secondary ink, the status's dot already says failed. */
    .r-cause {
      color: var(--text-secondary);
    }
    .sk .r-text {
      gap: var(--space-2);
    }
    .r-end {
      flex-shrink: 0;
      display: grid;
      justify-items: end;
      gap: 2px;
    }
    .r-val {
      font-weight: var(--weight-medium);
    }
    .r-age {
      font-size: var(--type-sm-size);
      color: var(--text-tertiary);
    }
    /* A live run shows its progress as a hairline along the row's bottom edge. */
    .r-prog {
      position: absolute;
      left: 0;
      right: 0;
      bottom: -1px;
      height: 2px;
      transform-origin: left;
      background: var(--accent-line);
    }
  }
`;
