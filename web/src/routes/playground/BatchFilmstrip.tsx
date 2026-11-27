/**
 * Compact multi-image batch filmstrip and toolbar rendered under Playground's stage when 2+ image files are loaded.
 * Provides Previous/Next stepping, keyboard arrow navigation, QA filtering (All / With detections / Empty / Specific
 * class) with live match counts, thumbnail selection with live per-image detection count badges, sequential Run All
 * with Stop/abort, and direct Batch JSON / Batch CSV / Summary CSV export controls.
 */
import Check from '$icons/check';
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Download from '$icons/download';
import FileJson from '$icons/file-json';
import Play from '$icons/play';
import Square from '$icons/square';
import { computed, css, effect, For, signal } from 'fw';
import Button from '$lib/ui/Button';
import { fadeScroll } from '$lib/ui/fadeScroll';
import Select from '$lib/ui/Select';
import { tooltip } from '$lib/ui/tooltip';
import type { BatchQaFilter } from './batch';
import type { BatchSession } from './useBatchSession';

export interface BatchFilmstripProps {
  batch: BatchSession;
  hasModel: boolean;
  /** True when Compare A/B is active, clarifying that batch execution and export target Model A. */
  compare?: boolean;
}

export default function BatchFilmstrip(props: BatchFilmstripProps) {
  const b = props.batch;
  const barEl = signal<HTMLDivElement | undefined>(undefined);
  const stripEl = signal<HTMLDivElement | undefined>(undefined);
  const actionsEl = signal<HTMLDivElement | undefined>(undefined);
  let runBtnFocused = false;

  // Track only the selected item's identity so per-item status/prediction updates during Run All never yank manual scroll.
  const activeItemId = computed(() => b.items.get()[b.activeIndex.get()]?.id ?? null);

  // Keep the active thumbnail visible inside the horizontal filmstrip without scrolling ancestor containers (`window`).
  effect(() => {
    const idx = b.activeIndex.get();
    void activeItemId.get();
    const root = stripEl.get();
    if (!root) return;
    const btn = root.querySelector<HTMLButtonElement>(`button[data-batch-idx="${idx}"]`);
    if (!btn) return;
    const rr = root.getBoundingClientRect();
    const br = btn.getBoundingClientRect();
    if (rr.width <= 0 || br.width <= 0) return;
    if (br.left < rr.left + 2) root.scrollLeft += br.left - rr.left - 2;
    else if (br.right > rr.right - 2) root.scrollLeft += br.right - rr.right + 2;
  });

  // When Run all completes and disables the run button while it held focus, move focus cleanly to an enabled batch action.
  effect(() => {
    const running = b.runningAll.get();
    const hasModel = props.hasModel;
    const s = b.summary.get();
    const allDone = !running && s.total > 0 && s.done >= s.total;
    if (!running && !hasModel && !allDone) {
      runBtnFocused = false;
      return;
    }
    if (!allDone || !runBtnFocused) return;
    runBtnFocused = false;
    queueMicrotask(() => {
      if (!b.active.peek()) return;
      const active = document.activeElement;
      const runBtn = actionsEl.peek()?.querySelector<HTMLButtonElement>('button[data-batch-run]');
      if (active && active !== document.body && active !== runBtn) return;
      const next =
        actionsEl.peek()?.querySelector<HTMLButtonElement>('button[data-batch-export="json"]:not(:disabled)') ??
        stripEl.peek()?.querySelector<HTMLButtonElement>(`button[data-batch-idx="${b.activeIndex.peek()}"]`);
      next?.focus({ preventScroll: true });
    });
  });

  const isFiltered = computed(() => b.qaFilter.get().kind !== 'all');
  const visiblePos = computed(() => {
    const activeIdx = b.activeIndex.get();
    return b.visibleEntries.get().findIndex((e) => e.index === activeIdx);
  });
  const canStepPrev = computed(() => {
    if (!isFiltered.get()) return b.activeIndex.get() > 0;
    const vis = b.visibleEntries.get();
    if (vis.length === 0) return false;
    const pos = visiblePos.get();
    return pos < 0 || pos > 0;
  });
  const canStepNext = computed(() => {
    if (!isFiltered.get()) return b.activeIndex.get() < b.items.get().length - 1;
    const vis = b.visibleEntries.get();
    if (vis.length === 0) return false;
    const pos = visiblePos.get();
    return pos < 0 || pos < vis.length - 1;
  });
  const classOptions = computed(() => {
    const counts = b.qaCounts.get();
    const qa = b.qaFilter.get();
    const opts = [
      { value: 'all', label: 'All classes' },
      ...counts.byClass.map((c) => ({
        value: String(c.classId),
        label: `${c.name} · ${c.imageCount}`,
      })),
    ];
    if (qa.kind === 'class' && !counts.byClass.some((c) => c.classId === qa.classId)) {
      const name = b.classes()[qa.classId] ?? `class ${qa.classId}`;
      opts.push({ value: String(qa.classId), label: `${name} · 0` });
    }
    return opts;
  });
  const selectedClassValue = computed(() => {
    const qa = b.qaFilter.get();
    return qa.kind === 'class' ? String(qa.classId) : 'all';
  });

  function setFilterKind(kind: 'all' | 'with-detections' | 'empty') {
    b.setQaFilter({ kind } as BatchQaFilter);
  }

  function onClassFilterChange(val: string) {
    if (val === 'all') {
      b.setQaFilter({ kind: 'all' });
    } else {
      const id = Number(val);
      if (Number.isInteger(id) && id >= 0) b.setQaFilter({ kind: 'class', classId: id });
    }
  }

  function exportTooltip(kind: 'JSON' | 'CSV' | 'summary CSV') {
    if (kind === 'summary CSV') {
      const ready = b.summaryExportableCount.get();
      if (ready === 0) {
        return isFiltered.get() ? 'No matching images to summarize' : 'Run inference on at least one image first';
      }
      const scope = isFiltered.get() ? `matching (${ready})` : 'batch';
      return props.compare
        ? `Export ${scope} Model A per-image summary as CSV`
        : `Export ${scope} per-image summary as CSV`;
    }
    const ready = b.exportableCount.get();
    if (ready === 0) {
      return isFiltered.get() ? 'No matching completed images to export' : 'Run inference on at least one image first';
    }
    const scope = isFiltered.get() ? `matching (${ready})` : 'all completed';
    return props.compare
      ? `Export ${scope} Model A batch detections as ${kind}`
      : `Export ${scope} batch detections as ${kind}`;
  }

  function resetFilterAndFocus() {
    b.setQaFilter({ kind: 'all' });
    queueMicrotask(() => {
      const next =
        barEl.peek()?.querySelector<HTMLButtonElement>('button[data-qa-filter="all"]') ??
        stripEl.peek()?.querySelector<HTMLButtonElement>('button[tabindex="0"]');
      next?.focus({ preventScroll: true });
    });
  }

  function stepWithFocus(delta: -1 | 1, e: MouseEvent) {
    const btn = e.currentTarget as HTMLButtonElement;
    const hadFocus = document.activeElement === btn;
    b.step(delta);
    if (!hadFocus) return;
    queueMicrotask(() => {
      if (!btn.isConnected || !btn.disabled) return;
      const active = document.activeElement;
      if (active && active !== document.body && active !== btn) return;
      const opposite = delta > 0 ? 'Previous image' : 'Next image';
      const next =
        stripEl.peek()?.querySelector<HTMLButtonElement>(`button[data-batch-idx="${b.activeIndex.peek()}"]`) ??
        barEl.peek()?.querySelector<HTMLButtonElement>(`button[aria-label="${opposite}"]:not(:disabled)`) ??
        stripEl.peek()?.querySelector<HTMLButtonElement>('button[tabindex="0"]');
      next?.focus({ preventScroll: true });
    });
  }

  function onStripKey(e: KeyboardEvent) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const vis = b.visibleEntries.peek();
    if (vis.length === 0) return;
    const pos = vis.findIndex((entry) => entry.index === b.activeIndex.peek());
    let targetPos: number | null = null;
    if (e.key === 'ArrowLeft') targetPos = pos < 0 ? 0 : Math.max(0, pos - 1);
    else if (e.key === 'ArrowRight') targetPos = pos < 0 ? 0 : Math.min(vis.length - 1, pos + 1);
    else if (e.key === 'Home') targetPos = 0;
    else if (e.key === 'End') targetPos = vis.length - 1;
    if (targetPos === null) return;
    const targetIdx = vis[targetPos]!.index;
    e.preventDefault();
    e.stopPropagation();
    b.selectIndex(targetIdx);
    stripEl.peek()?.querySelector<HTMLButtonElement>(`button[data-batch-idx="${targetIdx}"]`)?.focus({ preventScroll: true });
  }

  return (
    <div class="batch-bar" data-batch-bar ref={barEl}>
      <div class="batch-row">
        <div class="nav">
          <Button
            size="sm"
            variant="ghost"
            icon={ChevronLeft}
            aria-label="Previous image"
            disabled={!canStepPrev.get()}
            onclick={(e) => stepWithFocus(-1, e)}
          />
          <span class="pos num" data-batch-pos>
            {isFiltered.get()
              ? `${visiblePos.get() >= 0 ? visiblePos.get() + 1 : '—'} / ${b.visibleEntries.get().length}`
              : `${b.activeIndex.get() + 1} / ${b.items.get().length}`}
          </span>
          <Button
            size="sm"
            variant="ghost"
            icon={ChevronRight}
            aria-label="Next image"
            disabled={!canStepNext.get()}
            onclick={(e) => stepWithFocus(1, e)}
          />
        </div>

        <span class="meta num" aria-live="polite" data-batch-summary>
          {(() => {
            const s = b.summary.get();
            const prefix = props.compare ? 'Model A · ' : '';
            if (b.runningAll.get()) {
              const cur = (b.runningIndex.get() ?? 0) + 1;
              return `${prefix}Running ${cur} of ${s.total}…`;
            }
            const parts = [`${s.done}/${s.total} run`];
            if (s.done > 0) parts.push(`${s.totalDetections}${s.truncated > 0 ? '+' : ''} ${s.totalDetections === 1 && s.truncated === 0 ? 'det' : 'dets'}`);
            if (s.failed > 0) parts.push(`${s.failed} failed`);
            return `${prefix}${parts.join(' · ')}`;
          })()}
        </span>

        <div class="actions" ref={actionsEl}>
          <Button
            size="sm"
            variant={b.runningAll.get() ? 'secondary' : b.summary.get().done < b.summary.get().total ? 'primary' : 'secondary'}
            icon={b.runningAll.get() ? Square : b.summary.get().done < b.summary.get().total ? Play : Check}
            disabled={!b.runningAll.get() && (!props.hasModel || b.summary.get().done >= b.summary.get().total)}
            data-batch-run
            aria-label={
              b.runningAll.get()
                ? props.compare
                  ? 'Stop Model A batch run'
                  : 'Stop batch run'
                : b.summary.get().done < b.summary.get().total
                  ? props.compare
                    ? 'Run all A (Model A)'
                    : 'Run all images'
                  : props.compare
                    ? 'All A run (Model A)'
                    : 'All run'
            }
            onfocus={() => {
              runBtnFocused = true;
            }}
            onblur={(e) => {
              const btn = e.currentTarget as HTMLButtonElement;
              if (!btn.disabled && e.relatedTarget !== null) runBtnFocused = false;
              else if (!btn.disabled && b.summary.peek().done < b.summary.peek().total) runBtnFocused = false;
            }}
            onclick={() => {
              if (b.runningAll.peek()) b.stopRunAll();
              else void b.runAll();
            }}
          >
            {b.runningAll.get()
              ? 'Stop'
              : b.summary.get().done < b.summary.get().total
                ? props.compare
                  ? 'Run all A'
                  : 'Run all'
                : props.compare
                  ? 'All A run'
                  : 'All run'}
          </Button>
          <span use:tooltip={exportTooltip('JSON')}>
            <Button
              size="sm"
              variant="ghost"
              icon={FileJson}
              data-batch-export="json"
              aria-label={props.compare ? 'A JSON — export Model A batch' : 'Export batch JSON'}
              disabled={b.runningAll.get() || b.exportableCount.get() === 0}
              onclick={() => b.onExportJson()}
            >
              {props.compare ? 'A JSON' : 'Batch JSON'}
            </Button>
          </span>
          <span use:tooltip={exportTooltip('CSV')}>
            <Button
              size="sm"
              variant="ghost"
              icon={Download}
              data-batch-export="csv"
              aria-label={props.compare ? 'A CSV — export Model A batch' : 'Export batch CSV'}
              disabled={b.runningAll.get() || b.exportableCount.get() === 0}
              onclick={() => b.onExportCsv()}
            >
              {props.compare ? 'A CSV' : 'Batch CSV'}
            </Button>
          </span>
          <span use:tooltip={exportTooltip('summary CSV')}>
            <Button
              size="sm"
              variant="ghost"
              icon={Download}
              data-batch-export="summary-csv"
              aria-label={props.compare ? 'A Summary CSV — export Model A per-image summary' : 'Export batch summary CSV'}
              disabled={b.runningAll.get() || b.summaryExportableCount.get() === 0}
              onclick={() => b.onExportSummaryCsv()}
            >
              {props.compare ? 'A Summary' : 'Summary CSV'}
            </Button>
          </span>
        </div>
      </div>

      {b.summary.get().done > 0 || isFiltered.get() ? (
        <div class="qa-row" data-batch-qa-filter>
          <div class="qa-pills" role="group" aria-label="Filter batch images">
            <button
              type="button"
              class="qa-pill"
              class:active={b.qaFilter.get().kind === 'all'}
              aria-pressed={b.qaFilter.get().kind === 'all'}
              data-qa-filter="all"
              onclick={() => setFilterKind('all')}
            >
              <span>All</span>
              <span class="qa-count num">{b.qaCounts.get().all}</span>
            </button>
            <button
              type="button"
              class="qa-pill"
              class:active={b.qaFilter.get().kind === 'with-detections'}
              aria-pressed={b.qaFilter.get().kind === 'with-detections'}
              data-qa-filter="with-detections"
              onclick={() => setFilterKind('with-detections')}
            >
              <span>With detections</span>
              <span class="qa-count num">{b.qaCounts.get().withDetections}</span>
            </button>
            <button
              type="button"
              class="qa-pill"
              class:active={b.qaFilter.get().kind === 'empty'}
              aria-pressed={b.qaFilter.get().kind === 'empty'}
              data-qa-filter="empty"
              onclick={() => setFilterKind('empty')}
            >
              <span>Empty</span>
              <span class="qa-count num">{b.qaCounts.get().empty}</span>
            </button>
          </div>
          {classOptions.get().length > 1 ? (
            <span class="qa-class" data-batch-class-filter>
              <Select
                size="sm"
                label="Filter batch by class"
                value={selectedClassValue.get()}
                options={classOptions.get()}
                onchange={onClassFilterChange}
              />
            </span>
          ) : null}
          {isFiltered.get() ? (
            <span class="qa-match num" data-batch-match-count aria-live="polite">
              {b.visibleEntries.get().length} of {b.items.get().length} shown
            </span>
          ) : null}
        </div>
      ) : null}

      {b.visibleEntries.get().length === 0 ? (
        <div class="qa-empty" data-batch-empty>
          <span>No images match this filter.</span>
          <Button size="sm" variant="ghost" onclick={resetFilterAndFocus}>
            Show all
          </Button>
        </div>
      ) : (
        <div
          class="strip"
          role="listbox"
          aria-label="Batch images"
          data-batch-filmstrip
          data-scrolls
          ref={stripEl}
          use:fadeScroll
          onkeydown={onStripKey}
        >
          <For each={b.visibleEntries.get()} key={(entry) => entry.item.id}>
            {(entry) => {
              const item = () => entry().item;
              const origIdx = () => entry().index;
              const selected = () => b.activeIndex.get() === origIdx();
              const info = () => b.detectionCountFor(item());
              const badgeText = () => {
                const st = item().status;
                if (st === 'running') return '…';
                if (st === 'error') return '!';
                const c = info();
                return c !== null ? `${c.count}${c.cut ? '+' : ''}` : '—';
              };
              const statusDesc = () => {
                const st = item().status;
                if (st === 'running') return 'running inference';
                if (st === 'error') return `failed: ${item().error ?? 'error'}`;
                const c = info();
                return c !== null ? `${c.count}${c.cut ? '+' : ''} ${c.count === 1 && !c.cut ? 'detection' : 'detections'}` : 'not run';
              };
              const tabbable = () => selected() || (visiblePos.get() < 0 && entry().index === b.visibleEntries.get()[0]?.index);
              return (
                <button
                  type="button"
                  role="option"
                  class="thumb"
                  class:active={selected()}
                  class:err={item().status === 'error'}
                  class:busy={item().status === 'running'}
                  aria-selected={selected()}
                  tabindex={tabbable() ? 0 : -1}
                  data-batch-idx={origIdx()}
                  title={`${item().name} · ${statusDesc()}`}
                  aria-label={`${item().name}, ${statusDesc()}`}
                  onclick={() => b.selectIndex(origIdx())}
                >
                  <img src={item().url} alt="" loading="lazy" decoding="async" />
                  <span
                    class="badge num"
                    class:done={item().status === 'done'}
                    class:err={item().status === 'error'}
                    class:busy={item().status === 'running'}
                    data-batch-badge
                    aria-hidden="true"
                  >
                    {badgeText()}
                  </span>
                  <span class="name truncate" aria-hidden="true">
                    {item().name}
                  </span>
                </button>
              );
            }}
          </For>
        </div>
      )}
    </div>
  );
}

css`
  .batch-bar {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-top: 1px solid var(--line-subtle);
    background: var(--bg-panel);
  }
  .batch-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
  }
  .nav,
  .actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .actions {
    margin-left: auto;
  }
  .pos {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    min-width: 40px;
    text-align: center;
    white-space: nowrap;
  }
  .meta {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .qa-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .qa-pills {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .qa-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-subtle);
    background: var(--bg-app);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    white-space: nowrap;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  @media (hover: hover) {
    .qa-pill:hover:not(.active) {
      border-color: var(--line-strong);
      background: var(--bg-hover);
      color: var(--text-primary);
    }
  }
  .qa-pill.active {
    border-color: var(--text-tertiary);
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .qa-count {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  .qa-pill.active .qa-count {
    color: var(--text-secondary);
  }
  .qa-class {
    min-width: 128px;
    max-width: 200px;
  }
  .qa-match {
    margin-left: auto;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .qa-empty {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    min-height: 60px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-sm);
    border: 1px dashed var(--line-default);
    background: var(--bg-inset);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (pointer: coarse) {
    .qa-pill {
      min-height: 44px;
    }
  }
  .strip {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    overflow-x: auto;
    padding: 2px;
    scrollbar-width: thin;
  }
  .strip:global([data-fade='end']) {
    mask-image: linear-gradient(90deg, #000 calc(100% - 36px), transparent);
  }
  .strip:global([data-fade='start']) {
    mask-image: linear-gradient(90deg, transparent, #000 36px);
  }
  .strip:global([data-fade='both']) {
    mask-image: linear-gradient(90deg, transparent, #000 36px, #000 calc(100% - 36px), transparent);
  }
  .thumb {
    position: relative;
    flex: 0 0 84px;
    height: 60px;
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--bg-inset);
    box-shadow: 0 0 0 1px var(--line-default);
    transition:
      box-shadow var(--dur-fast) var(--ease-standard),
      transform var(--dur-fast) var(--ease-standard);
  }
  .thumb:hover {
    box-shadow: 0 0 0 1px var(--text-tertiary);
  }
  .thumb.active {
    box-shadow: 0 0 0 2px var(--accent-line);
  }
  .thumb.err {
    box-shadow: 0 0 0 1px var(--danger-line);
  }
  .thumb.err.active {
    box-shadow: 0 0 0 2px var(--danger-line);
  }
  .thumb:focus-visible {
    outline-offset: 2px;
  }
  .thumb img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .badge {
    position: absolute;
    top: 4px;
    right: 4px;
    min-width: 18px;
    height: 16px;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    color: var(--text-tertiary);
    /* small: image chip */
    font-size: var(--type-micro-size);
    /* small: image chip */
    line-height: 14px;
    font-weight: var(--weight-semibold);
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .badge.done {
    color: var(--text-primary);
    border-color: var(--line-strong);
  }
  .badge.busy {
    color: var(--accent-text);
    border-color: var(--accent-line);
  }
  .badge.err {
    color: var(--danger-text);
    border-color: var(--danger-line);
  }
  .name {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 1px 4px;
    background: color-mix(in srgb, var(--bg-overlay) 86%, transparent);
    color: var(--text-secondary);
    /* small: image chip */
    font-size: var(--type-micro-size);
    /* small: image chip */
    line-height: var(--type-micro-line);
    text-align: left;
  }
  .thumb.active .name {
    color: var(--text-primary);
  }
  @media (max-width: 767px) {
    .batch-bar {
      padding: var(--space-2) var(--space-4);
    }
    .actions {
      margin-left: 0;
    }
  }
`;
