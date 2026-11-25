/**
 * Interactive class-balance filter strip for the Dataset Images tab.
 * Renders classes ranked by box count as compact toggleable chips with inline proportional distribution bars so
 * users can inspect class balance and filter images in one click without opening a menu.
 */
import { computed, css, For, signal } from 'fw';
import { classInk } from '$lib/design/classInk';
import { compact, count as plural } from '$lib/format';
import { swatchStyle } from './swatch';

export interface ClassFilterStripProps {
  classes: readonly string[];
  counts: readonly number[];
  value: number | undefined;
  onchange: (cls: number | undefined) => void;
  /** Maximum chips shown before collapsing behind a "+N more" toggle (default 8). */
  limit?: number;
}

export interface RankedClassChip {
  cls: number;
  name: string;
  count: number;
  pctOfMax: number;
  sharePct: number;
}

/** Sorts classes by descending box count (tie-breaking by class index) and slices for the strip, pinning any active class. */
export function rankedClassChips(
  classes: readonly string[],
  counts: readonly number[],
  active: number | undefined,
  expanded: boolean,
  limit = 8,
): { visible: RankedClassChip[]; hiddenCount: number; totalBoxes: number } {
  const n = Math.min(classes.length, counts.length);
  if (n === 0) return { visible: [], hiddenCount: 0, totalBoxes: 0 };

  let totalBoxes = 0;
  let maxCount = 0;
  const all: { cls: number; name: string; count: number }[] = [];
  for (let i = 0; i < n; i++) {
    const c = Math.max(0, counts[i] ?? 0);
    totalBoxes += c;
    if (c > maxCount) maxCount = c;
    all.push({ cls: i, name: classes[i] ?? `class ${i}`, count: c });
  }
  all.sort((a, b) => b.count - a.count || a.cls - b.cls);

  const denom = Math.max(1, maxCount);
  const mapped: RankedClassChip[] = all.map((item) => ({
    ...item,
    pctOfMax: item.count > 0 ? Math.max(4, (item.count / denom) * 100) : 0,
    sharePct: totalBoxes > 0 ? (item.count / totalBoxes) * 100 : 0,
  }));

  if (expanded || mapped.length <= limit) {
    return { visible: mapped, hiddenCount: 0, totalBoxes };
  }

  const head = mapped.slice(0, limit);
  if (active !== undefined && !head.some((x) => x.cls === active)) {
    const pinned = mapped.find((x) => x.cls === active);
    if (pinned) head.push(pinned);
  }
  return { visible: head, hiddenCount: mapped.length - head.length, totalBoxes };
}

export default function ClassFilterStrip(props: ClassFilterStripProps) {
  const expanded = signal(false);
  const limit = () => props.limit ?? 8;
  const ranked = computed(() => rankedClassChips(props.classes, props.counts, props.value, expanded.get(), limit()));

  return (
    <div class="class-strip" role="group" aria-label="Filter by class distribution">
      <For each={ranked.get().visible} key={(item) => item.cls}>
        {(item) => {
          const active = () => props.value === item().cls;
          const absent = () => item().count === 0;
          return (
            <button
              type="button"
              class="chip"
              class:active={active()}
              class:absent={absent()}
              aria-pressed={active()}
              aria-label={`${item().name}: ${plural(item().count, 'box')}, ${item().sharePct.toFixed(1)}%${active() ? '. Active class filter, click to clear' : ''}`}
              onclick={() => props.onchange(active() ? undefined : item().cls)}
            >
              <span class="chip-top">
                <span class="sw" style={swatchStyle(item().cls, absent())} aria-hidden="true"></span>
                <span class="cname truncate">{item().name}</span>
                <span class="ccount num">{compact(item().count)}</span>
              </span>
              <span class="bar-track" aria-hidden="true">
                <span class="bar-fill" style:width={`${item().pctOfMax}%`} style:background={classInk(item().cls)}></span>
              </span>
            </button>
          );
        }}
      </For>
      {ranked.get().hiddenCount > 0 || (expanded.get() && Math.min(props.classes.length, props.counts.length) > limit()) ? (
        <button
          type="button"
          class="more-btn"
          aria-expanded={expanded.get()}
          onclick={() => expanded.set((v) => !v)}
        >
          {expanded.get() ? 'Show fewer' : `+${ranked.get().hiddenCount} more`}
        </button>
      ) : null}
    </div>
  );
}

css`
  .class-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }
  .chip {
    position: relative;
    display: inline-flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    min-width: 0;
    max-width: 180px;
    padding: 4px var(--space-2) 6px;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-subtle);
    background: var(--bg-panel);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    text-align: left;
    overflow: hidden;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .chip:hover {
    border-color: var(--line-strong);
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .chip.active {
    border-color: var(--accent-line);
    background: var(--bg-selected);
    color: var(--text-primary);
  }
  .chip.absent {
    color: var(--text-tertiary);
  }
  .chip-top {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    width: 100%;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .cname {
    min-width: 0;
    font-weight: var(--weight-medium);
  }
  .ccount {
    flex-shrink: 0;
    margin-left: auto;
    color: var(--text-tertiary);
  }
  .chip.active .ccount {
    color: var(--text-secondary);
  }
  .bar-track {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 2px;
    background: var(--bg-active);
  }
  .bar-fill {
    display: block;
    height: 100%;
    opacity: 0.85;
  }
  .chip.active .bar-fill {
    opacity: 1;
  }
  .more-btn {
    display: inline-flex;
    align-items: center;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    border: 1px dashed var(--line-default);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    white-space: nowrap;
    flex-shrink: 0;
    transition: border-color var(--dur-fast) var(--ease-standard), color var(--dur-fast) var(--ease-standard);
  }
  .more-btn:hover {
    border-color: var(--line-strong);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .chip,
    .more-btn {
      min-height: 44px;
    }
  }
  @media (max-width: 767px) {
    .class-strip {
      flex-wrap: nowrap;
      overflow-x: auto;
      scrollbar-width: none;
      padding-bottom: 2px;
    }
    .chip {
      flex-shrink: 0;
    }
  }
`;
