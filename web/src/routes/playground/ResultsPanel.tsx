/**
 * The detections kept by the thresholds, as a table (each row a way to inspect its box) or the prediction as JSON.
 * `rail` (from 1024 px) is the rail's Detections section: class, size and confidence in 320 px, the selected row's
 * coordinates on a line under the list, and the classes found (each a show / hide toggle) as a tab between the two;
 * below 1024 px it is a panel under the image with the coordinates as columns.
 */
import Check from '$icons/check';
import Copy from '$icons/copy';
import { bindable, computed, css, Dynamic, effect, For, onCleanup, signal, type Signal } from 'fw';
import type { Prediction } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import Button from '$lib/ui/Button';
import Panel from '$lib/ui/Panel';
import Tabs from '$lib/ui/Tabs';
import { toast } from '$lib/ui/toast';
import { tooltip } from '$lib/ui/tooltip';
import { jsonText } from './json';
import JsonView from './JsonView';
import { FETCH, px, score } from './predict';
import Swatch from './Swatch';
import { confidenceSummary } from './viewState';

type Det = Prediction['boxes'][number];

export interface ResultsPanelProps {
  result: { boxes: Det[]; ms: number } | null;
  tab?: 'detections' | 'classes' | 'json';
  /** The rail's section (from 1024 px), with a Classes tab. */
  rail?: boolean;
  classCount?: number;
  classList?: () => JSX.Element;
  showAllButton?: () => JSX.Element;
  onShowAll?: () => void;
  rows: { b: Det; i: number }[];
  predicting: boolean;
  notice: 'file' | 'error' | null;
  json: unknown;
  pred: Prediction | null;
  conf?: number;
  maxDet?: number;
  hidden: Set<number>;
  tbody?: HTMLTableSectionElement | undefined;
  tabRow: number | null;
  activeIdx: number | null;
  selectedIdx: number | null;
  hover?: Det | null;
  picked?: Det | null;
  onrowkey: (e: KeyboardEvent, k: number) => void;
}

export default function ResultsPanel(props: ResultsPanelProps) {
  const tab = bindable(props, 'tab', 'detections');
  const conf = bindable(props, 'conf', 0);
  const tbody = bindable(props, 'tbody') as Signal<HTMLTableSectionElement | undefined>;
  const hover = bindable(props, 'hover');
  const picked = bindable(props, 'picked');
  /** The view shown: Classes is the rail's tab only (a panel under the image has its own Classes panel). */
  const view = () => (!props.rail && tab.get() === 'classes' ? 'detections' : tab.get());
  // Leaving the rail (the window narrowed below 1024 px) on Classes selects Detections, so the tab list always has its
  // selected, focusable tab.
  effect(() => {
    if (!props.rail && tab.get() === 'classes') tab.set('detections');
  });
  const sel = () => (props.selectedIdx === null ? null : (props.rows.find((r) => r.i === props.selectedIdx)?.b ?? null));
  const summary = computed(() => (props.result ? confidenceSummary(props.rows) : null));
  const copied = signal(false);
  const copyStatus = signal('');
  let copyReset: ReturnType<typeof setTimeout> | undefined;
  let copySeq = 0;
  let alive = true;
  onCleanup(() => {
    alive = false;
    clearTimeout(copyReset);
  });

  // Reset the transient "Copied" check and invalidate any in-flight copy when the exported payload changes.
  effect(() => {
    void props.json;
    copySeq++;
    clearTimeout(copyReset);
    copied.set(false);
    copyStatus.set('');
  });

  async function copyDetectionsJson() {
    const payload = props.json;
    if (!payload) return;
    const seq = ++copySeq;
    const boxes = (payload as { boxes?: unknown }).boxes;
    const exportedCount = Array.isArray(boxes) ? boxes.length : (props.result?.boxes.length ?? 0);
    copyStatus.set('');
    try {
      await navigator.clipboard.writeText(jsonText(payload));
      if (!alive || seq !== copySeq) return;
      copied.set(true);
      copyStatus.set(`Copied ${exportedCount} ${exportedCount === 1 ? 'detection' : 'detections'} as JSON`);
      clearTimeout(copyReset);
      copyReset = setTimeout(() => {
        if (alive && seq === copySeq) {
          copied.set(false);
          copyStatus.set('');
        }
      }, 1600);
    } catch {
      if (!alive || seq !== copySeq) return;
      copyStatus.set('Could not copy detections JSON');
      toast.info('Copy failed', 'Your browser blocked clipboard access.');
    }
  }

  const hiddenAtConf = computed(() => props.result?.boxes.length ?? 0);
  const hasVisibleCandidates = computed(() => !props.hidden.size || !!props.pred?.boxes.some((b) => !props.hidden.has(b.class)));
  const canLower = computed(
    () =>
      conf.get() > 0.05 &&
      hasVisibleCandidates.get() &&
      (props.maxDet === undefined || hiddenAtConf.get() < props.maxDet),
  );
  const canShowAll = computed(() => !!props.onShowAll && (hiddenAtConf.get() > 0 || (props.hidden.size > 0 && !hasVisibleCandidates.get())));
  const emptyMessage = computed(() => {
    const n = hiddenAtConf.get();
    if (n > 0) return `${n === 1 ? '1 detection is' : `${n} detections are`} hidden by the class filter.`;
    if (props.hidden.size > 0 && !hasVisibleCandidates.get()) return `No detections at conf ≥ ${conf.get().toFixed(2)} — every detected class is hidden.`;
    return `No detections at conf ≥ ${conf.get().toFixed(2)}${props.hidden.size ? ' in the visible classes' : ''}.`;
  });

  return (
    <div class="results" class:rail={!!props.rail}>
      <Dynamic component={props.rail ? Bare : Panel} pad="none">
        <div class="tabs-row">
          <Tabs
            label="Result views"
            idPrefix="pg"
            bind:value={tab}
            tabs={[
              { value: 'detections', label: 'Detections', meta: props.result ? String(props.rows.length) : undefined },
              ...(props.rail ? [{ value: 'classes' as const, label: 'Classes', meta: props.result ? String(props.classCount ?? 0) : undefined }] : []),
              { value: 'json', label: 'JSON' },
            ]}
          />
          {props.result && props.json ? (
            <span class="copy-json" use:tooltip={copied.get() ? 'Copied detections JSON' : 'Copy detections JSON'}>
              {props.rail ? (
                <Button
                  size="sm"
                  variant="ghost"
                  icon={copied.get() ? Check : Copy}
                  aria-label={copied.get() ? 'Copied detections JSON' : 'Copy detections JSON'}
                  onclick={copyDetectionsJson}
                />
              ) : (
                <Button size="sm" variant="ghost" icon={copied.get() ? Check : Copy} onclick={copyDetectionsJson}>
                  {copied.get() ? 'Copied' : 'Copy JSON'}
                </Button>
              )}
            </span>
          ) : null}
        </div>
        <span class="sr-only" role="status" aria-live="polite">
          {copyStatus.get()}
        </span>
        <div id="pg-panel" role="tabpanel" aria-labelledby={`pg-${view()}`}>
          {!props.result ? (
            <p class="placeholder">
              {props.predicting
                ? 'Running inference…'
                : props.notice === 'error'
                  ? 'No detections: the last prediction failed.'
                  : 'Detections appear here once you run the model on an image.'}
            </p>
          ) : view() === 'classes' ? (
            <div class="classes-tab">
              {props.showAllButton ? <div class="show-all">{props.showAllButton}</div> : null}
              {props.classList?.()}
            </div>
          ) : view() === 'json' ? (
            <div class="json-wrap">
              <JsonView value={props.json} label="Prediction as JSON" />
            </div>
          ) : !props.pred?.boxes.length ? (
            /* Candidates are fetched down to the lowest threshold (FETCH): none means no threshold would show more. */
            <p class="placeholder">The model found nothing in this image, even at conf ≥ {FETCH.conf}.</p>
          ) : props.rows.length === 0 ? (
            <p class="placeholder">
              {emptyMessage.get()}{' '}
              {canShowAll.get() ? (
                <button type="button" class="inline" onclick={props.onShowAll}>
                  Show all classes
                </button>
              ) : null}
              {canShowAll.get() && canLower.get() ? ' · ' : null}
              {canLower.get() ? (
                <button type="button" class="inline" onclick={() => conf.set(Math.max(0.01, +(conf.get() / 2).toFixed(2)))}>
                  Lower the threshold
                </button>
              ) : null}
            </p>
          ) : (
            <>
              {summary.get() ? (
                <div
                  class="conf-summary num"
                  role="group"
                  aria-label={`Confidence distribution for ${summary.get()!.count} ${summary.get()!.count === 1 ? 'detection' : 'detections'}: median ${score(summary.get()!.median)}, range ${score(summary.get()!.min)} to ${score(summary.get()!.max)}, ${summary.get()!.high} high, ${summary.get()!.mid} medium, ${summary.get()!.low} low`}
                >
                  <div class="conf-meta">
                    <span class="conf-count">
                      <b>{summary.get()!.count}</b> {summary.get()!.count === 1 ? 'detection' : 'detections'}
                    </span>
                    <span class="conf-range">
                      med <b>{score(summary.get()!.median)}</b>
                      <span class="sep" aria-hidden="true">
                        ·
                      </span>
                      {score(summary.get()!.min)}–{score(summary.get()!.max)}
                    </span>
                  </div>
                  <div
                    class="conf-bar"
                    title={`High (≥0.75): ${summary.get()!.high} · Mid (0.50–0.74): ${summary.get()!.mid} · Low (<0.50): ${summary.get()!.low}`}
                    aria-hidden="true"
                  >
                    {summary.get()!.high > 0 ? <span class="seg-high" style={{ flexGrow: summary.get()!.high }}></span> : null}
                    {summary.get()!.mid > 0 ? <span class="seg-mid" style={{ flexGrow: summary.get()!.mid }}></span> : null}
                    {summary.get()!.low > 0 ? <span class="seg-low" style={{ flexGrow: summary.get()!.low }}></span> : null}
                  </div>
                  <div class="conf-buckets">
                    <span class="bucket">
                      <span class="dot high" aria-hidden="true"></span>≥0.75 <b>{summary.get()!.high}</b>
                    </span>
                    <span class="bucket">
                      <span class="dot mid" aria-hidden="true"></span>0.50–0.74 <b>{summary.get()!.mid}</b>
                    </span>
                    <span class="bucket">
                      <span class="dot low" aria-hidden="true"></span>&lt;0.50 <b>{summary.get()!.low}</b>
                    </span>
                  </div>
                </div>
              ) : null}
              <div class="scroll">
              <table class="table dense">
                <thead>
                  <tr>
                    {props.rail ? null : (
                      <th scope="col" class="r idx">
                        #
                      </th>
                    )}
                    <th scope="col">Class</th>
                    {props.rail ? (
                      <th scope="col" class="r">
                        Size (px)
                      </th>
                    ) : null}
                    <th scope="col" class:r={!!props.rail}>
                      {props.rail ? 'Confidence' : 'Score'}
                    </th>
                    {props.rail ? null : (
                      <>
                        <th scope="col" class="r hide-sm coord" title="Box left edge, in image pixels">
                          x1
                        </th>
                        <th scope="col" class="r hide-sm coord" title="Box top edge, in image pixels">
                          y1
                        </th>
                        <th scope="col" class="r hide-sm coord" title="Box right edge, in image pixels">
                          x2
                        </th>
                        <th scope="col" class="r hide-sm coord" title="Box bottom edge, in image pixels">
                          y2
                        </th>
                        <th scope="col" class="r">
                          Size (px)
                        </th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody ref={tbody}>
                  <For each={props.rows} key={(r) => r.i}>
                    {(r, k) => {
                      const b = () => r().b;
                      const i = () => r().i;
                      return (
                        <tr
                          data-det={i()}
                          tabindex={i() === props.tabRow ? 0 : -1}
                          class:hot={props.activeIdx === i()}
                          class:on={props.selectedIdx === i()}
                          aria-selected={props.selectedIdx === i() ? 'true' : undefined}
                          aria-label={`${b().name}, score ${score(b().score)}, ${px(b().x2 - b().x1)} by ${px(b().y2 - b().y1)} pixels`}
                          onpointerenter={(e) => {
                            // A tap is not a hover: on touch the selection alone marks the row.
                            if (e.pointerType !== 'touch') hover.set(b());
                          }}
                          onpointerleave={() => hover.set(null)}
                          onfocus={() => picked.set(b())}
                          onclick={() => picked.set(b())}
                          onkeydown={(e) => props.onrowkey(e, k())}
                        >
                          {props.rail ? null : <td class="r idx faint">{i() + 1}</td>}
                          <td class="lead">
                            <span class="cls">
                              <Swatch color={classColor(b().class)} />
                              <span class="truncate">{b().name}</span>
                            </span>
                          </td>
                          {props.rail ? (
                            <td class="r muted">
                              {px(b().x2 - b().x1)}×{px(b().y2 - b().y1)}
                            </td>
                          ) : null}
                          <td class:r={!!props.rail}>
                            <span class="score">
                              <span class="s-track" aria-hidden="true">
                                <span style:transform={`scaleX(${b().score})`}></span>
                              </span>
                              <span class="num">{score(b().score)}</span>
                            </span>
                          </td>
                          {props.rail ? null : (
                            <>
                              <td class="r muted hide-sm coord">{px(b().x1)}</td>
                              <td class="r muted hide-sm coord">{px(b().y1)}</td>
                              <td class="r muted hide-sm coord">{px(b().x2)}</td>
                              <td class="r muted hide-sm coord">{px(b().y2)}</td>
                              <td class="r muted">
                                {px(b().x2 - b().x1)}×{px(b().y2 - b().y1)}
                              </td>
                            </>
                          )}
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
              {/* The rail has no coordinate columns: the selected box's are here (and in JSON). */}
              {props.rail ? (
                <p class="sel-line num" aria-live="polite">
                  {sel() ? (
                    <>
                      <span class="faint">Box</span> x1 {px(sel()!.x1)} · y1 {px(sel()!.y1)} · x2 {px(sel()!.x2)} · y2 {px(sel()!.y2)}
                    </>
                  ) : (
                    <span class="faint">Select a box for its coordinates</span>
                  )}
                </p>
              ) : null}
            </div>
            </>
          )}
        </div>
      </Dynamic>
    </div>
  );
}

/** The rail's section has no frame of its own: the rail is the panel. */
function Bare(props: { pad?: string; children?: JSX.Element }) {
  return <>{props.children}</>;
}

css`
  .tabs-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 0 var(--space-3) 0 var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .tabs-row :global(.tabs) {
    flex: 1;
    border-bottom: 0;
  }
  .tabs-row :global(.tabs button::after) {
    bottom: 0;
  }
  .copy-json {
    display: inline-flex;
    flex-shrink: 0;
  }
  .conf-summary {
    display: grid;
    gap: 6px;
    padding: var(--space-2) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .conf-meta {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .conf-summary b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .conf-range {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .conf-range .sep {
    color: var(--text-tertiary);
  }
  .conf-bar {
    display: flex;
    gap: 2px;
    height: 4px;
    border-radius: var(--radius-full);
    overflow: hidden;
    background: var(--bg-active);
  }
  .seg-high {
    background: var(--success-solid);
  }
  .seg-mid {
    background: var(--warning-solid);
  }
  .seg-low {
    background: var(--danger-solid);
  }
  .conf-buckets {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-3);
    color: var(--text-tertiary);
  }
  .bucket {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-full);
    flex-shrink: 0;
  }
  .dot.high {
    background: var(--success-solid);
  }
  .dot.mid {
    background: var(--warning-solid);
  }
  .dot.low {
    background: var(--danger-solid);
  }
  /* A state in the frame (DESIGN_DIRECTION 8): one line with the panel's 16 px padding, on its left edge, never a
     centred block held open. */
  .placeholder {
    padding: var(--space-4);
    color: var(--text-tertiary);
  }

  /* An inline action is a link: text.primary, underlined on hover, never lime (DESIGN_DIRECTION §2.6). */
  .inline {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .inline:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .json-wrap {
    padding: var(--space-3);
  }
  .scroll {
    overflow: auto;
    max-height: 420px;
  }
  .scroll thead th {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
    z-index: 1;
  }
  .idx {
    width: 40px;
  }
  /* x1 y1 x2 y2 read as one group: tighter gaps between the four coordinate columns. */
  .table .coord + .coord {
    padding-left: var(--space-2);
  }
  .table .coord {
    font-variant-numeric: tabular-nums;
  }
  tbody tr {
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr {
    cursor: pointer;
  }
  /* Hover (the pointer, or the box under it on the image) is a light step, only where a pointer can hover, and never
     on the selected row, which keeps the selection's fill (base.css .on): one highlight per meaning. */
  @media (hover: hover) {
    tbody tr:hover:not(.on),
    tbody tr.hot:not(.on) {
      background: var(--bg-hover);
    }
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    max-width: 180px;
  }
  .score {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .s-track {
    position: relative;
    width: 72px;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .s-track span {
    position: absolute;
    inset: 0;
    transform-origin: left;
    border-radius: inherit;
  }
  /* The table's one inline bar is neutral ink (the hue is on the swatch); the hovered or selected row's in primary. */
  .s-track span {
    background: var(--text-tertiary);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr:hover .s-track span,
  tbody tr.hot .s-track span,
  tbody tr.on .s-track span {
    background: var(--text-primary);
  }
  /* The rail's section: the list grows with the rail, which scrolls (no scroll box of its own); the tabs sit on the
     section's 16 px inset like the rail's other headings. */
  .rail .scroll {
    max-height: none;
    overflow: visible;
  }
  .rail .scroll thead th {
    position: static;
  }
  .rail .s-track {
    width: 48px;
  }
  .rail .cls {
    max-width: 112px;
  }
  .sel-line {
    padding: var(--space-2) var(--space-4) 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .classes-tab {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4) 0;
  }
  .show-all {
    display: flex;
    justify-content: flex-end;
  }
  .show-all:empty {
    display: none;
  }
  @media (pointer: coarse) {
    .table.dense td {
      height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .results {
      order: 3;
    }
  }
`;
